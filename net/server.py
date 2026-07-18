"""
net/server.py
The authoritative server.

It owns the World. It runs the tick loop. It decides what happened. Clients send
it *intentions* ("I want to break the block at 4,60,7") and it sends back
*facts* ("the block at 4,60,7 is now air"). No client ever mutates its own copy
of the world directly, including the host's - see the module note in net/ for
why the host is not special-cased.

THREADING, AND WHY THIS IS THE SHAPE IT IS
------------------------------------------
The server runs on its own thread so that the host's frame rate and the world's
tick rate stop being the same number. In singleplayer they are coupled by
main.py's accumulator and that is fine; with a guest attached it is not, because
the host alt-tabbing (pygame stops getting frames) would stop time for everyone.

That gives exactly one shared-state problem, and it is solved in exactly one
place: `World` is touched ONLY from the server thread. Sockets are handled by
per-client reader threads that do nothing but parse frames and push them onto a
queue; the server thread drains that queue between ticks. Outgoing packets go
onto per-client send queues drained by per-client writer threads. So:

    reader threads   -> inbound Queue  -> [SERVER THREAD: the only World toucher] -> outbound Queues -> writer threads

No locks around the world. No lock ordering. No "did I remember to hold the
mutex" bugs at 2am. The queues are the only crossing points and Queue is already
thread-safe.

WHY WRITER THREADS AND NOT JUST send()
--------------------------------------
send() blocks when the peer's receive window fills. A guest whose game hitches
for a second would otherwise block the server thread mid-broadcast, stalling the
tick loop and therefore the host's own world. One slow client must never be able
to freeze everybody. Each client gets a queue with a bound; a client that cannot
keep up gets disconnected rather than consuming unbounded memory.
"""

import math
import queue
import socket
import threading
import time
import traceback

import config
from net import protocol
from net.protocol import PacketStream, ProtocolError, encode_packet
from world.blocks import Block, get_item_def
from world.world import World, world_to_chunk_coords
from world.chunk import CX, CZ

# How far around each player the server keeps chunks streamed and sent. Kept
# separate from any client's render distance: the server must have a chunk
# loaded to simulate it regardless of who is looking at it, and a client asking
# for a huge view must not be able to make the server load the whole world.
SERVER_VIEW_DISTANCE = 8

# Guests get their movement trusted, not validated, and that is a deliberate
# scope decision for a LAN game between friends - full server-side movement
# simulation would mean running two physics models and reconciling them, which
# is a project of its own. This cap only catches the accidental: a teleport
# glitch, a stuck key during a lag spike, a client that fell through the world.
MAX_MOVE_PER_PACKET = 12.0

# A client that stops responding for this long is dropped. Radmin VPN links do
# hiccup, so this is generous.
CLIENT_TIMEOUT_SECONDS = 20.0
KEEPALIVE_INTERVAL = 5.0

# Bound on a client's outbound backlog. Roughly a couple of seconds of chunks.
MAX_SEND_QUEUE = 512

# Radmin VPN hands its members addresses out of 26.0.0.0/8, and that is the
# whole reason this build exists - the two players are not on the same physical
# LAN and never will be.
RADMIN_PREFIX = "26."


def find_lan_address():
    """
    The address a friend actually has to type, or None if we cannot tell.

    A guess, and deliberately a narrow one. A machine with Radmin running has
    several IPv4 addresses (real LAN, maybe a VM bridge, maybe Hyper-V) and
    nothing in the OS marks which one a stranger on a VPN can reach. Picking the
    "main" one is a coin flip that lands on 192.168.x.x - an address that is
    correct, routable, and useless to the person on the other end.

    So: look for 26.x.x.x specifically, and return None rather than a guess if
    there isn't one. A screen that says nothing is better than a screen that
    confidently prints an address which cannot work - the player types it in,
    it times out, and now they are debugging the network instead of reading the
    one number that was right.
    """
    try:
        candidates = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return None
    for info in candidates:
        address = info[4][0]
        if address.startswith(RADMIN_PREFIX):
            return address
    return None


class _ClientConnection:
    """
    One connected player, from the server's side.

    Everything here except `sock`/`send_queue` is touched only by the server
    thread. The reader/writer threads touch only the socket and the queues.
    """

    _next_entity_id = 1

    def __init__(self, sock, address, server):
        self.sock = sock
        self.address = address
        self.server = server

        self.entity_id = _ClientConnection._next_entity_id
        _ClientConnection._next_entity_id += 1

        self.username = f"Player{self.entity_id}"
        self.authenticated = False
        self.alive = True

        # Player state the server owns for this client.
        self.x = 0.0
        self.y = 64.0
        self.z = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.on_ground = False
        self.in_water = False
        self.sneaking = False
        self.health = config.MAX_HEALTH
        self.air = config.AIR_MAX_SECONDS
        self.game_mode = "survival"
        self.inventory_slots = None
        self.selected_slot = 0

        # Chunks this client has already been sent, so we never send one twice.
        self.sent_chunks = set()

        self.last_packet_time = time.monotonic()
        self.send_queue = queue.Queue(maxsize=MAX_SEND_QUEUE)
        self._stream = PacketStream()
        self._reader_thread = None
        self._writer_thread = None

    # -- socket plumbing (reader/writer threads) ------------------------------

    def start(self):
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True,
                                                name=f"net-read-{self.entity_id}")
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True,
                                                name=f"net-write-{self.entity_id}")
        self._reader_thread.start()
        self._writer_thread.start()

    def _read_loop(self):
        try:
            while self.alive:
                data = self.sock.recv(65536)
                if not data:
                    break  # peer closed cleanly
                self._stream.feed(data)
                for packet in self._stream.packets():
                    self.server.inbound.put((self, packet))
        except (OSError, ProtocolError):
            pass  # a dead or desynced socket is just a disconnect
        finally:
            self.server.inbound.put((self, None))  # sentinel: "I'm gone"

    def _write_loop(self):
        try:
            while True:
                frame = self.send_queue.get()
                if frame is None:
                    break  # shutdown sentinel
                self.sock.sendall(frame)
        except OSError:
            pass
        finally:
            self.alive = False

    def send(self, packet_type: str, payload: dict = None):
        """Queues a packet. Never blocks the server thread."""
        if not self.alive:
            return
        try:
            self.send_queue.put_nowait(encode_packet(packet_type, payload))
        except queue.Full:
            # This client is not draining its socket. Dropping packets would
            # silently desync its world; disconnecting is honest.
            self.alive = False

    def close(self):
        self.alive = False
        try:
            self.send_queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class GameServer:
    """
    Owns the world and the simulation. Start it, and it listens until stopped.

    `local_client_hook` lets the host's own in-process client skip the TCP loop
    entirely (see net/client.py LoopbackConnection): same packets, same handlers,
    same ordering - just a queue instead of a socket. The host is not a special
    case in the LOGIC, only in the transport.
    """

    def __init__(self, world: World, save_dir: str, game_mode: str,
                 spawn_x: float, spawn_z: float, port: int = protocol.DEFAULT_PORT):
        self.world = world
        self.save_dir = save_dir
        self.game_mode = game_mode
        self.spawn_x = spawn_x
        self.spawn_z = spawn_z
        self.port = port

        self.clients: list[_ClientConnection] = []
        self.inbound = queue.Queue()

        self._listen_sock = None
        self._thread = None
        self._running = False
        self._tick_accumulator = 0.0
        self._last_time = None
        self._keepalive_timer = 0.0
        self._save_timer = 0.0

        # Block changes accumulated during the current tick, flushed as ONE
        # broadcast at the end of it. Water spreading a step can touch dozens of
        # cells in a single tick; a packet each would be dozens of TCP writes for
        # something the client cannot render at finer granularity than a frame
        # anyway.
        self._pending_changes = []

        self.error = None  # set if the listen socket could not be opened

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> bool:
        """Binds and starts the server thread. Returns False (and sets .error)
        if the port could not be opened, which is the one failure the player
        actually needs to be told about.

        A port of 0 means "any free one" and is what opening a world to the
        network uses. The kernel picks it, atomically, from the ephemeral range -
        which is both why there is no retry loop here and why there is no race:
        asking for a specific port and hoping is how you collide with the real
        Minecraft server the player forgot they left running. Read self.port
        AFTER this returns to find out what you got; before it, 0 is a request,
        not an answer."""
        try:
            self._listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 0.0.0.0, not 127.0.0.1: binding to loopback would make the server
            # reachable only from this machine, which is the single most common
            # way "my friend can't connect" happens. Radmin's adapter is just
            # another interface, and this covers it along with real LAN.
            self._listen_sock.bind(("0.0.0.0", self.port))
            self._listen_sock.listen(8)
            self._listen_sock.settimeout(0.5)
            self.port = self._listen_sock.getsockname()[1]
        except OSError as exc:
            requested = self.port if self.port else "a free port"
            self.error = f"Could not open {requested}: {exc.strerror or exc}"
            self._listen_sock = None
            return False

        self.world.change_listener = self._on_block_change
        self._running = True
        self._last_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True, name="game-server")
        self._thread.start()
        threading.Thread(target=self._accept_loop, daemon=True, name="game-accept").start()
        return True

    def stop(self):
        self._running = False
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except OSError:
                pass
        for client in list(self.clients):
            client.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.world.change_listener = None

    def _accept_loop(self):
        while self._running:
            try:
                sock, address = self._listen_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            # Nagle batches small writes to save bandwidth by adding up to 40ms
            # of latency. Movement packets are small and frequent and latency is
            # exactly what we are spending bandwidth to avoid.
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client = _ClientConnection(sock, address, self)
            client.start()
            # Not added to self.clients here: that list belongs to the server
            # thread. It joins on C_HELLO, inside _handle_packet.

    def attach_local_client(self, connection, position=None):
        """Registers the host's own in-process client. `connection` is a
        LoopbackConnection whose send() pushes straight onto the client's inbound
        queue - no socket, no serialisation.

        `position` is where that client should join, and exists because the host
        is the one player whose position is already known before they connect:
        it was loaded off disk with the world. Without it the host joins at world
        spawn like any guest, which for a saved world means being yanked away
        from wherever they logged out - and then yanked BACK by S_TELEPORT the
        moment they walk more than MAX_MOVE_PER_PACKET from it. Guests still get
        spawn, which is right: the server has never heard of them before."""
        client = _ClientConnection(connection, ("local", 0), self)
        client.send = connection.server_to_client
        client.close = lambda: None
        client.preset_position = position
        connection.bind_server(self, client)
        return client

    # -- the tick loop --------------------------------------------------------

    def _run(self):
        try:
            while self._running:
                now = time.monotonic()
                dt = now - self._last_time
                self._last_time = now

                self._drain_inbound()

                self._tick_accumulator += dt
                ticks = 0
                while self._tick_accumulator >= config.TICK_SECONDS and ticks < config.MAX_TICKS_PER_FRAME:
                    self._tick_accumulator -= config.TICK_SECONDS
                    self._tick()
                    ticks += 1
                if ticks >= config.MAX_TICKS_PER_FRAME:
                    self._tick_accumulator = 0.0

                self._keepalive_timer += dt
                if self._keepalive_timer >= KEEPALIVE_INTERVAL:
                    self._keepalive_timer = 0.0
                    self._sweep_timeouts()

                self._save_timer += dt
                if self._save_timer >= 30.0:
                    self._save_timer = 0.0
                    self._save()

                # Sleep the remainder of the tick. Without this the loop spins a
                # core at 100% doing nothing, which on the host's machine is a
                # core the renderer wanted.
                slack = config.TICK_SECONDS - (time.monotonic() - now)
                if slack > 0:
                    time.sleep(slack)
        except Exception:
            traceback.print_exc()
            self._running = False

    def _tick(self):
        self.world.tick()

        for client in self.clients:
            self._update_client_chunks(client)
            self._update_client_vitals(client)

        self._flush_changes()
        self._broadcast_positions()

        self.world.process_generation_budget(gen_budget=8, tree_budget=4)

    # -- world change plumbing ------------------------------------------------

    def _on_block_change(self, wx, wy, wz, block_id, meta_value):
        self._pending_changes.append([wx, wy, wz, int(block_id), int(meta_value)])

    def _flush_changes(self):
        if not self._pending_changes:
            return
        changes = self._pending_changes
        self._pending_changes = []
        for client in self.clients:
            # Only changes inside chunks this client actually holds. Sending a
            # change for a chunk it has never seen would be a write into a
            # chunk it has to invent, and inventing one client-side is exactly
            # the desync this architecture exists to prevent.
            relevant = [c for c in changes
                        if (c[0] // CX, c[2] // CZ) in client.sent_chunks]
            if relevant:
                client.send(protocol.S_BLOCK_CHANGE, {"changes": relevant})

    def _broadcast_positions(self):
        if len(self.clients) < 2:
            return  # nobody to tell
        for client in self.clients:
            others = []
            for other in self.clients:
                if other is client or not other.authenticated:
                    continue
                flags = 0
                if other.on_ground:
                    flags |= protocol.FLAG_ON_GROUND
                if other.in_water:
                    flags |= protocol.FLAG_IN_WATER
                if other.sneaking:
                    flags |= protocol.FLAG_SNEAKING
                others.append([other.entity_id,
                                round(other.x, 3), round(other.y, 3), round(other.z, 3),
                                round(other.yaw, 3), round(other.pitch, 3), flags])
            if others:
                client.send(protocol.S_ENTITY_MOVE, {"entities": others})

    # -- per-client streaming -------------------------------------------------

    def _update_client_chunks(self, client):
        pcx, pcz = world_to_chunk_coords(int(client.x), int(client.z))

        wanted = set()
        for dx in range(-SERVER_VIEW_DISTANCE, SERVER_VIEW_DISTANCE + 1):
            for dz in range(-SERVER_VIEW_DISTANCE, SERVER_VIEW_DISTANCE + 1):
                if dx * dx + dz * dz <= SERVER_VIEW_DISTANCE * SERVER_VIEW_DISTANCE:
                    wanted.add((pcx + dx, pcz + dz))

        # Send a bounded number per tick. A fresh join wants ~200 chunks; all at
        # once is several MB in one burst, which stalls the writer thread and
        # spikes the client's decode. Spread over ~25 ticks it is invisible.
        budget = 6
        for key in sorted(wanted - client.sent_chunks,
                          key=lambda k: (k[0] - pcx) ** 2 + (k[1] - pcz) ** 2):
            if budget <= 0:
                break
            chunk = self.world.ensure_chunk_loaded(*key)
            client.send(protocol.S_CHUNK, protocol.encode_chunk(chunk))
            client.sent_chunks.add(key)
            budget -= 1

        # Tell it to drop what it has wandered away from, so its memory and its
        # renderer don't grow without bound.
        stale = [k for k in client.sent_chunks
                 if (k[0] - pcx) ** 2 + (k[1] - pcz) ** 2 > (SERVER_VIEW_DISTANCE + 3) ** 2]
        for key in stale:
            client.sent_chunks.discard(key)
            client.send(protocol.S_UNLOAD_CHUNK, {"cx": key[0], "cz": key[1]})

    def _update_client_vitals(self, client):
        """
        Drowning, on the server, for everyone.

        Health cannot live on the client: a client that owns its own health has
        no reason ever to report losing any, and more practically the host and
        the guest would each be running their own drowning timer against their
        own copy of the water and drifting apart. The client still runs its own
        physics (that is what makes movement feel instant), but what it reports
        is where it IS, not how it's doing.
        """
        if client.game_mode != "survival" or client.health <= 0:
            return

        from world import fluids
        submerged = fluids.is_head_in_water(
            self.world, client.x, client.y + config.PLAYER_EYE_OFFSET, client.z)

        before_air = client.air
        before_health = client.health

        if submerged:
            if client.air > 0.0:
                client.air = max(0.0, client.air - config.TICK_SECONDS)
                if client.air <= 0.0:
                    client._drown_accum = config.DROWN_DAMAGE_INTERVAL
            else:
                client._drown_accum = getattr(client, "_drown_accum", 0.0) + config.TICK_SECONDS
                while client._drown_accum >= config.DROWN_DAMAGE_INTERVAL:
                    client._drown_accum -= config.DROWN_DAMAGE_INTERVAL
                    client.health = max(0, client.health - config.DROWN_DAMAGE)
        else:
            client.air = config.AIR_MAX_SECONDS
            client._drown_accum = 0.0

        if client.health != before_health or abs(client.air - before_air) > 0.001:
            client.send(protocol.S_PLAYER_STATE, {"health": client.health, "air": client.air})

        if client.health <= 0 and before_health > 0:
            self._broadcast_chat(f"{client.username} drowned")

    def _sweep_timeouts(self):
        now = time.monotonic()
        for client in list(self.clients):
            if not client.alive or now - client.last_packet_time > CLIENT_TIMEOUT_SECONDS:
                self._disconnect(client)
            else:
                client.send(protocol.S_KEEPALIVE)

    # -- inbound packets ------------------------------------------------------

    def _drain_inbound(self):
        while True:
            try:
                client, packet = self.inbound.get_nowait()
            except queue.Empty:
                return
            if packet is None:
                self._disconnect(client)
                continue
            client.last_packet_time = time.monotonic()
            try:
                self._handle_packet(client, packet)
            except Exception:
                traceback.print_exc()  # one bad packet must never kill the tick loop

    def _handle_packet(self, client, packet):
        packet_type = packet.get("t")

        if packet_type == protocol.C_HELLO:
            self._handle_hello(client, packet)
            return
        if not client.authenticated:
            return  # ignore everything until a valid hello

        if packet_type == protocol.C_PLAYER_MOVE:
            self._handle_move(client, packet)
        elif packet_type == protocol.C_BLOCK_BREAK:
            self._handle_break(client, packet)
        elif packet_type == protocol.C_BLOCK_PLACE:
            self._handle_place(client, packet)
        elif packet_type == protocol.C_DOOR_TOGGLE:
            self.world.toggle_door(packet["x"], packet["y"], packet["z"])
        elif packet_type == protocol.C_SWING:
            for other in self.clients:
                if other is not client:
                    other.send(protocol.S_ENTITY_SWING, {"entity_id": client.entity_id})
        elif packet_type == protocol.C_DAMAGE:
            self._handle_damage(client, packet)
        elif packet_type == protocol.C_RESPAWN:
            self._handle_respawn(client)
        elif packet_type == protocol.C_CHAT:
            text = str(packet.get("text", ""))[:200]
            if text:
                self._broadcast_chat(f"<{client.username}> {text}")

    def _handle_damage(self, client, packet):
        """
        Damage the CLIENT worked out and is reporting.

        This trusts the client, and that is deliberate rather than lazy: the
        server does not simulate anyone's movement (see MAX_MOVE_PER_PACKET),
        so it does not know how far anyone fell, and giving it fall damage would
        mean running a second physics model here and reconciling it with the
        one that already exists - the project this file's header explicitly
        declines to start. A LAN game between friends already trusts every
        position packet it receives; trusting "I fell and it hurt" adds no new
        capability to a cheater who could simply not send it.

        What this DOES buy is a single owner for health. Before it, drowning was
        the server's number and falling was the client's, the two overwrote each
        other, and the HUD flickered between them. Now the client reports events
        and the server owns the total - which is also why the client must not
        subtract anything locally, and doesn't.
        """
        if client.health <= 0:
            return  # already dead; a corpse cannot take more damage
        try:
            amount = int(packet.get("amount", 0))
        except (TypeError, ValueError):
            return
        if amount <= 0:
            return
        # Ceiling, not validation: a bad packet cannot take more than a full
        # bar, so the worst a broken/hostile client achieves is killing itself.
        amount = min(amount, config.MAX_HEALTH)

        client.health = max(0, client.health - amount)
        client.send(protocol.S_PLAYER_STATE, {"health": client.health, "air": client.air})
        if client.health <= 0:
            cause = str(packet.get("cause", ""))[:32]
            if cause == "fall":
                self._broadcast_chat(f"{client.username} fell from a high place")
            elif cause == "void":
                self._broadcast_chat(f"{client.username} fell out of the world")
            else:
                self._broadcast_chat(f"{client.username} died")

    def _handle_respawn(self, client):
        """
        Puts a dead client back at world spawn.

        The position has to come from here rather than being decided locally,
        for a reason that is easy to miss: a client's ReplicaWorld only holds
        the chunks the server sent it, and a player who died 500 blocks out does
        not have the spawn chunk. Asking its own world how tall the ground is at
        spawn gets an empty chunk back, height 0 - so the client would place
        itself at y=1, under the terrain, and fall to its death again. And
        again.
        """
        if client.health > 0:
            return  # not dead; nothing to respawn from
        ground_y = self.world.get_ground_height_at(int(self.spawn_x), int(self.spawn_z))
        client.x, client.y, client.z = self.spawn_x, ground_y + 1.0, self.spawn_z
        client.health = config.MAX_HEALTH
        client.air = config.AIR_MAX_SECONDS
        client._drown_accum = 0.0
        client.send(protocol.S_TELEPORT, {"x": client.x, "y": client.y, "z": client.z})
        client.send(protocol.S_PLAYER_STATE, {"health": client.health, "air": client.air})

    def _handle_hello(self, client, packet):
        if client.authenticated:
            return
        if packet.get("protocol") != protocol.PROTOCOL_VERSION:
            client.send(protocol.S_REJECT, {
                "reason": f"Version mismatch - server is on protocol "
                          f"{protocol.PROTOCOL_VERSION}, you are on {packet.get('protocol')}. "
                          f"Both players need the same build."})
            client.close()
            return

        name = str(packet.get("username", "")).strip()[:16] or f"Player{client.entity_id}"
        client.username = name
        client.game_mode = self.game_mode
        client.authenticated = True

        # Place them on solid ground at world spawn. Per-player saved positions
        # are a later feature; joining at spawn is what vanilla does for a new
        # player anyway and it cannot drop anyone inside terrain.
        #
        # The exception is a client that arrived with a position already known -
        # only ever the host's own local client, whose position came off disk
        # with the world. See attach_local_client.
        preset = getattr(client, "preset_position", None)
        if preset is not None:
            client.x, client.y, client.z = preset
        else:
            ground_y = self.world.get_ground_height_at(int(self.spawn_x), int(self.spawn_z))
            client.x, client.y, client.z = self.spawn_x, ground_y + 1.0, self.spawn_z

        self.clients.append(client)

        client.send(protocol.S_WELCOME, {
            "entity_id": client.entity_id,
            "seed": self.world.seed,
            "game_mode": client.game_mode,
            "spawn_x": self.spawn_x,
            "spawn_z": self.spawn_z,
            "x": client.x, "y": client.y, "z": client.z,
            "yaw": client.yaw, "pitch": client.pitch,
            "health": client.health, "air": client.air,
        })

        # Introduce everyone to everyone, both directions.
        for other in self.clients:
            if other is client or not other.authenticated:
                continue
            client.send(protocol.S_ENTITY_SPAWN, {
                "entity_id": other.entity_id, "username": other.username,
                "x": other.x, "y": other.y, "z": other.z,
                "yaw": other.yaw, "pitch": other.pitch})
            other.send(protocol.S_ENTITY_SPAWN, {
                "entity_id": client.entity_id, "username": client.username,
                "x": client.x, "y": client.y, "z": client.z,
                "yaw": client.yaw, "pitch": client.pitch})

        self._broadcast_chat(f"{client.username} joined the game")

    def _handle_move(self, client, packet):
        try:
            new_x = float(packet["x"])
            new_y = float(packet["y"])
            new_z = float(packet["z"])
        except (KeyError, TypeError, ValueError):
            return
        if not all(math.isfinite(v) for v in (new_x, new_y, new_z)):
            return  # a NaN position would poison every distance test downstream

        moved = math.dist((new_x, new_y, new_z), (client.x, client.y, client.z))
        if moved > MAX_MOVE_PER_PACKET:
            client.send(protocol.S_TELEPORT, {"x": client.x, "y": client.y, "z": client.z})
            return

        client.x, client.y, client.z = new_x, new_y, new_z
        client.yaw = float(packet.get("yaw", client.yaw))
        client.pitch = float(packet.get("pitch", client.pitch))
        client.on_ground = bool(packet.get("on_ground", False))
        client.in_water = bool(packet.get("in_water", False))
        client.sneaking = bool(packet.get("sneaking", False))

    def _handle_break(self, client, packet):
        x, y, z = int(packet["x"]), int(packet["y"]), int(packet["z"])
        if math.dist((x + 0.5, y + 0.5, z + 0.5),
                     (client.x, client.y + config.PLAYER_EYE_OFFSET, client.z)) > config.REACH_DISTANCE + 2.0:
            return  # outside any plausible reach; ignore rather than trust
        self.world.break_block(x, y, z)

    def _handle_place(self, client, packet):
        x, y, z = int(packet["x"]), int(packet["y"]), int(packet["z"])
        block_id = int(packet.get("id", 0))
        if math.dist((x + 0.5, y + 0.5, z + 0.5),
                     (client.x, client.y + config.PLAYER_EYE_OFFSET, client.z)) > config.REACH_DISTANCE + 2.0:
            return
        item_def = get_item_def(block_id)
        if item_def is None or not item_def.is_block:
            return

        kind = packet.get("kind", "block")
        if kind == "door":
            self.world.place_door(x, y, z, int(packet.get("facing", 0)),
                                   float(packet.get("yaw", 0.0)))
        elif kind == "stairs":
            self.world.place_stairs(x, y, z, block_id, int(packet.get("facing", 0)),
                                     bool(packet.get("is_top", False)))
        else:
            self.world.place_block(x, y, z, block_id)

    # -- misc -----------------------------------------------------------------

    def _broadcast_chat(self, text: str):
        for client in self.clients:
            client.send(protocol.S_CHAT, {"text": text})

    def _disconnect(self, client):
        if client in self.clients:
            self.clients.remove(client)
            self._broadcast_chat(f"{client.username} left the game")
            for other in self.clients:
                other.send(protocol.S_ENTITY_DESPAWN, {"entity_id": client.entity_id})
        client.close()

    def _save(self):
        if self.save_dir is None:
            return
        from save import world_save
        try:
            world_save.save_all_loaded_chunks(self.save_dir, self.world)
        except OSError:
            traceback.print_exc()

    @property
    def player_count(self):
        return len(self.clients)
