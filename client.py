"""
net/client.py
The client half: a replica of the world, and the players in it.

WHAT A CLIENT IS
----------------
A picture of the server's world, plus the local player's physics. It NEVER
decides anything about the world - it does not generate terrain, does not tick
fluids, does not grow trees, does not save. It receives facts (S_CHUNK,
S_BLOCK_CHANGE) and draws them. The one thing it owns is where the local player
is standing, because that is the only way movement can feel instant; even that
is a claim the server can overrule with S_TELEPORT.

THREADING
---------
Same rule as the server, mirrored: the replica World is touched ONLY from the
main (render) thread. The socket reader thread does nothing but parse frames and
push them onto `inbound`. `poll()` drains that queue once per frame from the main
thread and is the only thing that applies anything.

This is why LoopbackConnection.server_to_client only enqueues. The host's server
thread calls client.send(...) directly, so if that ran a packet handler inline it
would be mutating the replica world - and its numpy arrays, and its dirty queues -
from the server thread while the renderer walked them. Queue in, drain on the main
thread: the loopback path and the socket path then have identical semantics, and
the host stops being a special case in anything but transport.

WHY A REPLICA World SUBCLASS
----------------------------
Everything downstream - physics, raycast, the six renderers, the mesh builders -
already speaks World. Handing them a different object would mean touching all of
it. So ReplicaWorld IS a World with the authoritative half removed: generation is
disabled (chunks arrive over the wire instead), saving is disabled (the server
owns the disk), ticking is disabled (the server owns time). What is left - the
chunk dict, get_block/set_block, the dirty-mesh queues, padding for the meshers -
is exactly the part a client needs, unchanged.
"""

import math
import queue
import socket
import threading
import time
import traceback

from net import protocol
from net.protocol import PacketStream, ProtocolError, encode_packet
from world.chunk import Chunk
from world.world import World, world_to_chunk_coords

# How far behind the newest received state other players are drawn. Entity
# updates arrive 20x/second (one per server tick); frames run at 60+. Drawing
# the newest position directly means every third frame or so has nothing new to
# show and the player visibly stutters. Rendering ~100ms in the past means there
# is almost always a state on either side of "now" to interpolate between, at the
# cost of other players being drawn a tenth of a second stale - which nobody can
# perceive, whereas the stutter is obvious.
INTERPOLATION_DELAY = 0.1

# Movement packets per second. Matches the server's tick rate: sending faster
# just gives the server several positions to overwrite before it looks at one.
MOVE_SEND_HZ = 20.0
MOVE_SEND_INTERVAL = 1.0 / MOVE_SEND_HZ

# Give up on a connect attempt that has not completed in this long. The default
# OS timeout is ~20s of a frozen UI, which reads as a hang rather than a failure.
CONNECT_TIMEOUT = 8.0

# No traffic for this long and we consider the link dead. The server sends
# keepalives every 5s, so this is generous enough to survive a Radmin hiccup.
SERVER_TIMEOUT = 20.0


class ReplicaWorld(World):
    """
    A World that never invents anything.

    Overrides exactly the three authoritative behaviours and nothing else:
      - chunk creation: empty, awaiting S_CHUNK, never worldgen
      - tick: nothing; the server ticks
      - generation budget: nothing; there is nothing to generate

    set_block still works, and MUST - it is how S_BLOCK_CHANGE lands, and it is
    what keeps height_map and the dirty-mesh queues correct for free. The
    difference from a real World is only in who is allowed to call it: on a
    client, only apply_block_change.
    """

    def __init__(self, seed: int):
        super().__init__(seed=seed, save_dir=None)

    def _get_or_create_chunk(self, cx: int, cz: int) -> Chunk:
        """
        Returns an EMPTY chunk for coordinates the server has not sent yet,
        rather than generating terrain for them.

        The alternative - running worldgen client-side off the shared seed - is
        the classic trap. It looks like it works (same seed, same noise, same
        terrain) right up until anyone changes a block: worldgen reproduces the
        ORIGINAL terrain, not the terrain as edited, so every chunk the client
        invented is subtly wrong forever and there is nothing to correct it.
        Empty-until-told is honest: an air chunk you can see through is an
        obvious "not loaded yet", not a silent lie.
        """
        key = (cx, cz)
        chunk = self.chunks.get(key)
        if chunk is None:
            chunk = Chunk(cx, cz)
            chunk.generated = False      # no terrain here until S_CHUNK says so
            chunk.trees_generated = True  # never grow trees client-side
            chunk.needs_save = False      # the client owns no disk
            self.chunks[key] = chunk
        return chunk

    def tick(self):
        """The server owns simulated time. Fluids, falling blocks and scheduled
        wakeups all arrive as S_BLOCK_CHANGE."""
        return

    def process_generation_budget(self, gen_budget: int = None, tree_budget: int = None):
        return

    def update_streaming(self, player_wx: float, player_wz: float, render_distance: int):
        """
        Unloading is driven by the SERVER's S_UNLOAD_CHUNK, not by local
        distance, so this does nothing.

        A client that unloaded on its own would drop chunks the server still
        thinks it holds - and the server only sends block changes for chunks in
        client.sent_chunks (see GameServer._flush_changes). Every edit in a
        dropped-but-still-tracked chunk would be missed, and since nothing would
        ever re-send it, the desync would be permanent and invisible.
        """
        return set()


class LoopbackConnection:
    """
    The host's own client, wired to the host's own server without a socket.

    Quacks like a socket to the server (`send`/`close` are swapped in by
    GameServer.attach_local_client) and like a connection to the client. Same
    packets, same handlers, same ordering as a real guest - the only thing
    skipped is serialisation and the kernel.

    Both directions are queues, deliberately. c2s so the server thread drains it
    exactly like a reader thread's output; s2c so the client applies it on the
    main thread exactly like a socket reader's output. Neither side ever runs the
    other's code on its own thread.
    """

    def __init__(self):
        self.inbound = queue.Queue()   # server -> client, drained by GameClient.poll
        self._server = None
        self._server_client = None
        self.closed = False

    def bind_server(self, server, server_client):
        """Called by GameServer.attach_local_client once it has made the
        _ClientConnection that represents us."""
        self._server = server
        self._server_client = server_client

    def server_to_client(self, packet_type: str, payload: dict = None):
        """
        Installed as _ClientConnection.send, so this runs ON THE SERVER THREAD.

        It therefore does exactly one thing: enqueue. No handler call, no world
        touch, no rendering state - see the module docstring. The dict is copied
        because the server builds payloads from live state and reuses/mutates
        some of them (the entity list in _broadcast_positions is rebuilt per
        client, but chunk payloads and change lists are shared); without a copy
        the client could observe a payload that changed under it between enqueue
        and poll.
        """
        if self.closed:
            return
        body = {"t": packet_type}
        if payload:
            body.update(payload)
        self.inbound.put(body)

    def client_to_server(self, packet_type: str, payload: dict = None):
        """Called on the CLIENT's main thread; hands the packet to the server's
        inbound queue, which its own thread drains. Mirrors what a reader thread
        does with a parsed frame."""
        if self.closed or self._server is None:
            return
        body = {"t": packet_type}
        if payload:
            body.update(payload)
        self._server.inbound.put((self._server_client, body))

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Tell the server we're gone the same way a dropped socket does, so the
        # host leaving runs the identical disconnect path a guest leaving does.
        if self._server is not None and self._server_client is not None:
            self._server.inbound.put((self._server_client, None))


class RemotePlayer:
    """
    Another player, as this client sees them.

    Holds two timestamped states and interpolates between them. Two is enough:
    with a 100ms delay and 50ms between updates there is always a pair bracketing
    the render time, and keeping a longer history would only matter for
    extrapolation, which is not wanted here - guessing ahead makes players
    visibly rubber-band when the guess is wrong, and standing still is the
    common case.
    """

    __slots__ = ("entity_id", "username", "x", "y", "z", "yaw", "pitch",
                 "on_ground", "in_water", "sneaking", "distance_walked",
                 "swing_timer", "_prev", "_next")

    def __init__(self, entity_id, username, x, y, z, yaw, pitch):
        self.entity_id = entity_id
        self.username = username
        self.x, self.y, self.z = x, y, z
        self.yaw, self.pitch = yaw, pitch
        self.on_ground = True
        self.in_water = False
        self.sneaking = False

        # Total horizontal ground covered, in blocks. The walk animation is a
        # function of DISTANCE, not of time: legs that swing on a timer keep
        # walking when the player stops, and a player being pushed along by
        # water would moonwalk. sin(distance) is also automatically in phase
        # across every client without syncing anything.
        self.distance_walked = 0.0
        self.swing_timer = 0.0

        now = time.monotonic()
        state = (now, x, y, z, yaw, pitch)
        self._prev = state
        self._next = state

    def push_state(self, x, y, z, yaw, pitch, flags):
        self.on_ground = bool(flags & protocol.FLAG_ON_GROUND)
        self.in_water = bool(flags & protocol.FLAG_IN_WATER)
        self.sneaking = bool(flags & protocol.FLAG_SNEAKING)
        self._prev = self._next
        self._next = (time.monotonic(), x, y, z, yaw, pitch)

    def update(self, dt: float):
        """Advances the interpolated pose. Call once per frame."""
        render_time = time.monotonic() - INTERPOLATION_DELAY
        t_prev, px, py, pz, pyaw, ppitch = self._prev
        t_next, nx, ny, nz, nyaw, npitch = self._next

        span = t_next - t_prev
        if span <= 1e-6:
            alpha = 1.0
        else:
            alpha = (render_time - t_prev) / span
            # Clamped, never extrapolated: past the newest state we hold the
            # newest state. A player who stopped sending (lag spike, or they
            # alt-tabbed) freezes in place rather than sliding off through a
            # wall on a stale velocity and snapping back.
            alpha = max(0.0, min(1.0, alpha))

        old_x, old_z = self.x, self.z
        self.x = px + (nx - px) * alpha
        self.y = py + (ny - py) * alpha
        self.z = pz + (nz - pz) * alpha
        self.yaw = _lerp_angle(pyaw, nyaw, alpha)
        self.pitch = ppitch + (npitch - ppitch) * alpha

        self.distance_walked += math.hypot(self.x - old_x, self.z - old_z)
        if self.swing_timer > 0.0:
            self.swing_timer = max(0.0, self.swing_timer - dt)

    def trigger_swing(self):
        self.swing_timer = 0.25


def _lerp_angle(a: float, b: float, t: float) -> float:
    """
    Interpolates yaw the short way around.

    RADIANS, not degrees - Player.yaw is radians everywhere in this project
    (player/player.py feeds it straight to math.sin, and controls.py advances it
    by mouse_dx * MOUSE_SENSITIVITY_BASE). The degree version of this function
    happened to look harmless, because wrapping at 360 RADIANS is ~57 full turns
    and no real diff ever gets near it - so it silently degraded into a plain
    lerp that could never actually do its one job.

    The job: a player turning through 2pi -> 0 must spin a hair, not all the way
    back around. Player.set_look does not currently normalise yaw (it just
    accumulates), so today nothing wraps and both versions produce the same
    numbers - which is exactly why this needs fixing NOW rather than after
    someone adds a sensible `yaw %= 2*pi` and every remote player starts
    spinning like a top once per turn, from a bug three modules away.
    """
    two_pi = 2.0 * math.pi
    diff = (b - a + math.pi) % two_pi - math.pi
    return a + diff * t


class GameClient:
    """
    Owns the replica world, the remote players, and the link to the server.

    Used identically whether the server is across the network or in this very
    process - construct with connect() for the former, attach_local() for the
    latter, then poll() every frame either way.
    """

    def __init__(self, username: str):
        self.username = username

        self.world = None            # ReplicaWorld, created on S_WELCOME
        self.entity_id = None
        self.game_mode = "survival"
        self.spawn_x = 0.0
        self.spawn_z = 0.0
        self.remote_players: dict[int, RemotePlayer] = {}
        self.chat_log = []

        # Server-authoritative vitals for THIS client, applied to the local
        # Player by main.py. Kept here rather than written straight into the
        # Player because packets arrive before there is necessarily a Player to
        # write into (S_WELCOME lands during the loading screen).
        self.health = None
        self.air = None
        self.pending_teleport = None  # (x, y, z) the server insists on

        self.connected = False
        self.welcomed = False
        self.error = None            # set on any fatal condition; main.py shows it
        self.disconnect_reason = None

        self.inbound = queue.Queue()
        self._sock = None
        self._loopback = None
        self._stream = PacketStream()
        self._reader_thread = None
        self._writer_thread = None
        self._send_queue = queue.Queue()
        self._last_packet_time = time.monotonic()
        self._move_timer = 0.0
        self._last_sent_move = None

        # Chunks whose S_CHUNK arrived before the mesh could be built. main.py
        # drains this with a per-frame budget - a fresh join lands ~200 chunks
        # and meshing them all in one frame is a multi-second freeze, the exact
        # thing the loading screen's mesh phase exists to avoid.
        self.pending_meshes = []

    # -- connecting -----------------------------------------------------------

    def connect(self, host: str, port: int) -> bool:
        """Opens the socket and starts the reader/writer threads. Returns False
        and sets .error on failure - the only thing the player needs told."""
        try:
            self._sock = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT)
            # Same reasoning as the server: movement packets are small and
            # frequent, and Nagle would add up to 40ms to each one to save bytes
            # we are deliberately spending.
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Back to blocking for the reader thread: the connect timeout must
            # not become a recv timeout, or an idle moment reads as a drop.
            self._sock.settimeout(None)
        except OSError as exc:
            self.error = _friendly_connect_error(exc, host, port)
            self._sock = None
            return False

        self.connected = True
        self._last_packet_time = time.monotonic()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True,
                                                name="client-read")
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True,
                                                name="client-write")
        self._reader_thread.start()
        self._writer_thread.start()
        self.send(protocol.C_HELLO, {"username": self.username,
                                      "protocol": protocol.PROTOCOL_VERSION})
        return True

    def attach_local(self, server, position=None) -> bool:
        """Wires this client straight into an in-process server. No socket, no
        threads - the queues are the transport.

        `position` is where to join; see GameServer.attach_local_client. Only the
        host ever passes one, because the host is the only player whose position
        the server already knows before hello."""
        self._loopback = LoopbackConnection()
        server.attach_local_client(self._loopback, position=position)
        self.inbound = self._loopback.inbound
        self.connected = True
        self._last_packet_time = time.monotonic()
        self.send(protocol.C_HELLO, {"username": self.username,
                                      "protocol": protocol.PROTOCOL_VERSION})
        return True

    def _read_loop(self):
        try:
            while self.connected:
                data = self._sock.recv(65536)
                if not data:
                    break
                self._stream.feed(data)
                for packet in self._stream.packets():
                    self.inbound.put(packet)
        except (OSError, ProtocolError):
            pass
        finally:
            self.inbound.put(None)  # sentinel: the link is gone

    def _write_loop(self):
        try:
            while True:
                frame = self._send_queue.get()
                if frame is None:
                    break
                self._sock.sendall(frame)
        except OSError:
            pass

    def send(self, packet_type: str, payload: dict = None):
        """Queues a packet to the server. Never blocks the frame loop."""
        if not self.connected:
            return
        if self._loopback is not None:
            self._loopback.client_to_server(packet_type, payload)
            return
        try:
            self._send_queue.put_nowait(encode_packet(packet_type, payload))
        except queue.Full:
            pass

    def disconnect(self):
        self.connected = False
        if self._loopback is not None:
            self._loopback.close()
            self._loopback = None
        if self._sock is not None:
            try:
                self._send_queue.put_nowait(None)
            except queue.Full:
                pass
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # -- the per-frame drain --------------------------------------------------

    def poll(self, dt: float):
        """
        Applies every packet that has arrived, then advances interpolation.
        Call once per frame from the main thread, and from nowhere else - this
        is the only thing permitted to touch the replica world.
        """
        while True:
            try:
                packet = self.inbound.get_nowait()
            except queue.Empty:
                break
            if packet is None:
                self._on_link_lost()
                break
            self._last_packet_time = time.monotonic()
            try:
                self._handle_packet(packet)
            except Exception:
                # One malformed packet must not take the frame loop down with
                # it, exactly as on the server.
                traceback.print_exc()

        if self.connected and time.monotonic() - self._last_packet_time > SERVER_TIMEOUT:
            self.disconnect_reason = "Connection timed out"
            self._on_link_lost()

        for player in self.remote_players.values():
            player.update(dt)

    def _on_link_lost(self):
        if self.connected:
            self.connected = False
            if self.disconnect_reason is None:
                self.disconnect_reason = "Lost connection to server"

    def _handle_packet(self, packet):
        packet_type = packet.get("t")

        if packet_type == protocol.S_WELCOME:
            self._handle_welcome(packet)
        elif packet_type == protocol.S_REJECT:
            self.error = packet.get("reason", "Rejected by server")
            self.disconnect_reason = self.error
            self.disconnect()
        elif packet_type == protocol.S_CHUNK:
            self._handle_chunk(packet)
        elif packet_type == protocol.S_UNLOAD_CHUNK:
            self._handle_unload(packet)
        elif packet_type == protocol.S_BLOCK_CHANGE:
            self._handle_block_change(packet)
        elif packet_type == protocol.S_ENTITY_SPAWN:
            self._handle_spawn(packet)
        elif packet_type == protocol.S_ENTITY_MOVE:
            self._handle_entity_move(packet)
        elif packet_type == protocol.S_ENTITY_DESPAWN:
            self.remote_players.pop(packet.get("entity_id"), None)
        elif packet_type == protocol.S_ENTITY_SWING:
            player = self.remote_players.get(packet.get("entity_id"))
            if player is not None:
                player.trigger_swing()
        elif packet_type == protocol.S_PLAYER_STATE:
            self.health = packet.get("health", self.health)
            self.air = packet.get("air", self.air)
        elif packet_type == protocol.S_TELEPORT:
            self.pending_teleport = (packet["x"], packet["y"], packet["z"])
        elif packet_type == protocol.S_CHAT:
            text = packet.get("text", "")
            if text:
                self.chat_log.append(text)
                del self.chat_log[:-50]
        elif packet_type == protocol.S_KEEPALIVE:
            self.send(protocol.C_KEEPALIVE)

    def _handle_welcome(self, packet):
        self.entity_id = packet["entity_id"]
        self.game_mode = packet.get("game_mode", "survival")
        self.spawn_x = packet.get("spawn_x", 0.0)
        self.spawn_z = packet.get("spawn_z", 0.0)
        self.health = packet.get("health")
        self.air = packet.get("air")
        self.world = ReplicaWorld(seed=packet["seed"])
        # The server placed us; treat it as a teleport so main.py positions the
        # Player through the one path that handles this, rather than two.
        self.pending_teleport = (packet["x"], packet["y"], packet["z"])
        self.welcomed = True

    def _handle_chunk(self, packet):
        if self.world is None:
            return  # a chunk before welcome would have no world to land in
        cx, cz = packet["cx"], packet["cz"]
        chunk = self.world.ensure_chunk_loaded(cx, cz)
        protocol.decode_chunk_into(chunk, packet)
        # decode_chunk_into sets chunk.dirty itself, but the world's dirty QUEUE
        # is a separate thing and is what main.py actually drains - a chunk that
        # only set its own flag would never be handed to a renderer.
        self.world._mark_dirty(cx, cz, urgent=False)
        self.world._mark_neighbors_dirty(cx, cz)
        self.pending_meshes.append((cx, cz))

    def _handle_unload(self, packet):
        if self.world is None:
            return
        cx, cz = packet["cx"], packet["cz"]
        # World.unload_chunk would try to save (save_dir is None, so it won't)
        # and cancel scheduled ticks (there are none). It also does the dirty-set
        # cleanup we want, so it is still the right call - the replica just has
        # less for it to do.
        self.world.unload_chunk(cx, cz)
        self.world.recently_unloaded.append((cx, cz))

    def _handle_block_change(self, packet):
        if self.world is None:
            return
        for change in packet.get("changes", []):
            wx, wy, wz, block_id, meta_value = change
            cx, cz = world_to_chunk_coords(wx, wz)
            if (cx, cz) not in self.world.chunks:
                # A change for a chunk we don't hold. The server filters by its
                # own sent_chunks, so this is a race around an unload, not a bug:
                # dropping it is correct, since if the chunk comes back it comes
                # back as a fresh S_CHUNK with this change already baked in.
                continue
            # update_neighbors=False is not an optimisation - it is the
            # client/server split restated at block level. The reactive layer's
            # whole job is to SCHEDULE wakeups (water spreading a step, sand
            # deciding to fall - see world/block_behavior.py), and every one of
            # those is a decision the server has already made and will send here
            # as its own S_BLOCK_CHANGE. Letting them fire would queue wakeups
            # into ReplicaWorld.ticks that tick() is overridden never to run: a
            # scheduler that only ever grows, in a world where water is the most
            # common thing there is to change.
            self.world.set_block(wx, wy, wz, block_id, meta_value=meta_value,
                                 update_neighbors=False)

    def _handle_spawn(self, packet):
        entity_id = packet["entity_id"]
        self.remote_players[entity_id] = RemotePlayer(
            entity_id, packet.get("username", "Player"),
            packet["x"], packet["y"], packet["z"],
            packet.get("yaw", 0.0), packet.get("pitch", 0.0))

    def _handle_entity_move(self, packet):
        for entry in packet.get("entities", []):
            entity_id, x, y, z, yaw, pitch, flags = entry
            player = self.remote_players.get(entity_id)
            if player is None:
                # Movement for someone we were never introduced to. Can happen
                # if S_ENTITY_MOVE overtakes S_ENTITY_SPAWN's ordering across a
                # reconnect; spawning them here is better than dropping them and
                # having an invisible player.
                self.remote_players[entity_id] = RemotePlayer(
                    entity_id, f"Player{entity_id}", x, y, z, yaw, pitch)
                continue
            player.push_state(x, y, z, yaw, pitch, flags)

    # -- outbound -------------------------------------------------------------

    def send_move(self, player, dt: float):
        """
        Reports where the local player is, at MOVE_SEND_HZ.

        Skipped entirely when nothing has changed since the last report: a
        player standing still is the common case (in a menu, building, idle) and
        there is no reason to spend 20 packets a second saying so. The server's
        timeout is driven by keepalives, not movement, so silence is safe.
        """
        self._move_timer += dt
        if self._move_timer < MOVE_SEND_INTERVAL:
            return
        self._move_timer = 0.0

        physics = player.physics
        # PlayerPhysics has no sneak state: Shift is dive/descend here, not
        # crouch (see player/physics.py). Read defensively rather than drop the
        # field - the protocol already carries FLAG_SNEAKING, so crouching starts
        # working across the wire for free the day it lands, whereas a hard
        # attribute access raises inside the frame loop 20 times a second.
        state = (round(physics.x, 3), round(physics.y, 3), round(physics.z, 3),
                 round(player.yaw, 2), round(player.pitch, 2),
                 physics.on_ground, physics.head_in_water,
                 getattr(physics, "is_sneaking", False))
        if state == self._last_sent_move:
            return
        self._last_sent_move = state

        self.send(protocol.C_PLAYER_MOVE, {
            "x": state[0], "y": state[1], "z": state[2],
            "yaw": state[3], "pitch": state[4],
            "on_ground": bool(state[5]),
            "in_water": bool(state[6]),
            "sneaking": bool(state[7]),
        })

    def send_break(self, x: int, y: int, z: int):
        self.send(protocol.C_BLOCK_BREAK, {"x": x, "y": y, "z": z})

    def send_place(self, x: int, y: int, z: int, block_id: int, kind: str = "block",
                   facing: int = 0, is_top: bool = False, yaw: float = 0.0, meta: int = 0):
        self.send(protocol.C_BLOCK_PLACE, {
            "x": x, "y": y, "z": z, "id": block_id, "meta": meta,
            "kind": kind, "facing": facing, "is_top": is_top, "yaw": yaw,
        })

    def send_door_toggle(self, x: int, y: int, z: int):
        self.send(protocol.C_DOOR_TOGGLE, {"x": x, "y": y, "z": z})

    def send_swing(self):
        self.send(protocol.C_SWING)

    def send_damage(self, amount: int, cause: str = ""):
        """
        Reports damage this client worked out for itself - falling, or the void.

        Reports, not applies. The local Player's health is not touched here and
        must not be: the server owns the number and will send it back as
        S_PLAYER_STATE. Subtracting it here as well would double the hit on a
        LAN where the round trip is a frame, and would leave the two copies
        permanently disagreeing whenever a packet was ever dropped.
        """
        self.send(protocol.C_DAMAGE, {"amount": int(amount), "cause": cause})

    def send_respawn(self):
        """Asks to be put back at spawn. Where that IS comes back as S_TELEPORT -
        this client cannot work it out, because it has never been sent the spawn
        chunk. See GameServer._handle_respawn."""
        self.send(protocol.C_RESPAWN)

    def send_chat(self, text: str):
        self.send(protocol.C_CHAT, {"text": text})

    def pop_pending_meshes(self, max_count: int):
        """Returns up to max_count chunk coords that have received data and need
        a mesh built. main.py budgets these per frame."""
        if not self.pending_meshes:
            return []
        result = self.pending_meshes[:max_count]
        del self.pending_meshes[:max_count]
        return result


def _friendly_connect_error(exc: OSError, host: str, port: int) -> str:
    """
    Turns a socket errno into something worth showing a player.

    "[WinError 10061] No connection could be made because the target machine
    actively refused it" is accurate and useless. The three cases below are what
    actually happens on a Radmin LAN, and each has a different fix - which is the
    whole point of telling them apart.
    """
    import errno
    code = getattr(exc, "errno", None)
    if isinstance(exc, socket.timeout):
        return (f"No answer from {host}:{port}.\n"
                f"Check the address, and that the host has opened their world.")
    if code in (errno.ECONNREFUSED, 10061):
        return (f"{host}:{port} refused the connection.\n"
                f"The host's world isn't open to the network, or the port is wrong.")
    if code in (errno.EHOSTUNREACH, errno.ENETUNREACH, 10065, 10051):
        return (f"Can't reach {host}.\n"
                f"Check your VPN (Radmin) is connected on both machines.")
    return f"Could not connect to {host}:{port}: {exc.strerror or exc}"
