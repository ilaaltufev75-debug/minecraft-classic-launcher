"""
_net_test.py
Headless end-to-end test of the multiplayer stack. Run it:

    .venv\\Scripts\\python.exe _net_test.py

Touches no OpenGL and no window, so it runs on one machine in a few seconds
and tells you which of these actually works rather than which of them compiles:

    a real server binding a real port
    a host joining through the loopback, at the position it was standing at
    a guest joining over a real TCP socket to 127.0.0.1
    the two of them seeing each other spawn, move and swing
    a block broken by one appearing broken to the other
    a block placed by one appearing placed to the other
    fall damage reaching the server and coming back as health
    death and respawn, including that respawn puts you on solid ground
    a version mismatch being refused with a readable message

What it does NOT prove: that it works over Radmin, that the renderer draws any
of it, or that the frame loop drives it correctly. Loopback and 127.0.0.1 have
no latency, no packet loss and no MTU. This is the floor, not the ceiling - if
this fails, nothing else can work; if it passes, you still have to actually
play it with someone.
"""

import shutil
import sys
import tempfile
import time
import traceback

import config
from net import protocol
from net.client import GameClient
from net.server import GameServer
from world.blocks import Block
from world.world import World

TEST_PORT = 0  # 0 = let the kernel pick a free one, exactly like Open to LAN does
TIMEOUT = 15.0

_failures = []
_passes = 0


def check(name, condition, detail=""):
    global _passes
    if condition:
        _passes += 1
        print(f"  PASS  {name}")
    else:
        _failures.append(name)
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def pump(clients, seconds=0.5, until=None):
    """
    Drives the clients the way main.py's frame loop does - poll() on this
    thread, ~60 times a second - until `until` returns True or the time runs
    out. Returns whether the condition was met.

    Nothing here may call poll() from anywhere but this thread, for the same
    reason main.py may not: poll is the only thing allowed to touch a replica
    world.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for client in clients:
            client.poll(1 / 60.0)
        if until is not None and until():
            return True
        time.sleep(1 / 60.0)
    return until is None or until()


def wait_for(clients, condition, seconds=TIMEOUT):
    return pump(clients, seconds, condition)


def main():
    save_dir = tempfile.mkdtemp(prefix="nettest_")
    server = None
    host = None
    guest = None
    try:
        print("\n[setup] world + server")
        world = World(seed=1337, save_dir=save_dir)
        # Resolve the host's standing position BEFORE the server thread starts.
        # After start() the world belongs to that thread and this one must not
        # read it - exactly what main.py._open_to_lan has to respect.
        ground_y = world.get_ground_height_at(0, 0)
        host_position = (0.5, ground_y + 1.0, 0.5)

        server = GameServer(world, save_dir, "survival", spawn_x=0.0, spawn_z=0.0, port=TEST_PORT)
        started = server.start()
        check("server binds a free port", started, server.error or "")
        if not started:
            return
        check("server reports the port the kernel gave it", server.port > 0, str(server.port))
        port = server.port

        print("\n[host] loopback join")
        host = GameClient("Host")
        host.attach_local(server, position=host_position)
        check("host welcomed", wait_for([host], lambda: host.welcomed))
        check("host got a replica world", host.world is not None)
        check("host joined where it was standing, not at spawn",
              host.pending_teleport is not None
              and abs(host.pending_teleport[1] - host_position[1]) < 0.01,
              f"got {host.pending_teleport}")
        host_pos = host.pending_teleport
        host.pending_teleport = None

        print("\n[host] chunks arrive")
        def host_has_chunk():
            chunk = host.world.chunks.get((0, 0))
            return chunk is not None and chunk.generated
        check("host receives chunk (0,0)", wait_for([host], host_has_chunk))

        print("\n[guest] TCP join")
        guest = GameClient("Guest")
        connected = guest.connect("127.0.0.1", port)
        check("guest connects over TCP", connected, guest.error or "")
        if not connected:
            return
        check("guest welcomed", wait_for([host, guest], lambda: guest.welcomed))
        check("guest joined at world spawn", guest.pending_teleport is not None)
        guest.pending_teleport = None

        print("\n[both] see each other")
        check("host sees guest", wait_for([host, guest], lambda: len(host.remote_players) == 1))
        check("guest sees host", wait_for([host, guest], lambda: len(guest.remote_players) == 1))
        if host.remote_players:
            other = next(iter(host.remote_players.values()))
            check("host sees the guest's name", other.username == "Guest", other.username)

        print("\n[guest] movement reaches the host")
        target_x = host_pos[0] + 4.0
        guest.send(protocol.C_PLAYER_MOVE, {
            "x": target_x, "y": host_pos[1], "z": host_pos[2],
            "yaw": 1.0, "pitch": 0.0,
            "on_ground": True, "in_water": False, "sneaking": False,
        })
        def host_sees_move():
            if not host.remote_players:
                return False
            other = next(iter(host.remote_players.values()))
            # push_state feeds the interpolator, so read the raw target rather
            # than the interpolated pose - the pose is deliberately ~100ms behind.
            return abs(other._next[1] - target_x) < 0.01
        check("host receives the guest's position", wait_for([host, guest], host_sees_move))

        print("\n[guest] swing reaches the host")
        guest.send_swing()
        def host_sees_swing():
            if not host.remote_players:
                return False
            return next(iter(host.remote_players.values())).swing_timer > 0.0
        check("host receives the guest's swing", wait_for([host, guest], host_sees_swing, 3.0))

        print("\n[world] block edits propagate")
        bx, by, bz = 0, int(ground_y), 0
        before = host.world.get_block(bx, by, bz)
        check("there is a block to break", before != Block.AIR, f"id {before}")
        guest.send_break(bx, by, bz)
        check("break by guest reaches the host",
              wait_for([host, guest], lambda: host.world.get_block(bx, by, bz) == Block.AIR))

        host.send_place(bx, by, bz, Block.PLANKS)
        check("place by host reaches the guest",
              wait_for([host, guest], lambda: guest.world.get_block(bx, by, bz) == Block.PLANKS))

        print("\n[health] fall damage")
        start_health = guest.health
        check("guest starts at full health", start_health == config.MAX_HEALTH, str(start_health))
        guest.send_damage(6, cause="fall")
        check("server applies reported damage",
              wait_for([host, guest], lambda: guest.health == config.MAX_HEALTH - 6),
              f"health {guest.health}")

        print("\n[health] death and respawn")
        guest.send_damage(config.MAX_HEALTH, cause="fall")
        check("guest dies", wait_for([host, guest], lambda: guest.health == 0), f"health {guest.health}")
        guest.send_respawn()
        check("respawn restores health",
              wait_for([host, guest], lambda: guest.health == config.MAX_HEALTH),
              f"health {guest.health}")
        check("respawn teleports the guest", guest.pending_teleport is not None)
        if guest.pending_teleport is not None:
            _x, y, _z = guest.pending_teleport
            # The bug this exists to catch: a client resolving its own respawn
            # height against a replica that has no spawn chunk gets y=1 and
            # falls out of the world, forever.
            check("respawn lands on solid ground, not y=1", y > 2.0, f"y={y}")

        print("\n[protocol] version mismatch is refused")
        stranger = GameClient("Stranger")
        if stranger.connect("127.0.0.1", port):
            stranger.send(protocol.C_HELLO, {"username": "Stranger",
                                             "protocol": protocol.PROTOCOL_VERSION + 99})
            check("mismatched version is rejected with a message",
                  wait_for([stranger], lambda: stranger.error is not None, 5.0),
                  str(stranger.error))
            stranger.disconnect()

        print("\n[teardown] guest leaves")
        guest.disconnect()
        guest = None
        check("host sees the guest despawn",
              wait_for([host], lambda: len(host.remote_players) == 0, 5.0))

    except Exception:
        traceback.print_exc()
        _failures.append("unhandled exception")
    finally:
        for client in (guest, host):
            if client is not None:
                try:
                    client.disconnect()
                except Exception:
                    pass
        if server is not None:
            server.stop()
        shutil.rmtree(save_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    if _failures:
        print(f"{_passes} passed, {len(_failures)} FAILED:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print(f"all {_passes} checks passed")
    print("This means the stack works on one machine over loopback and")
    print("127.0.0.1. It does not mean it works over Radmin with a real")
    print("person - go and try that next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
