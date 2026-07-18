"""
net/protocol.py
The wire format. Both server and client import ONLY this module to agree on
what a message looks like - neither imports the other.

FRAMING
-------
TCP is a byte stream, not a message stream. `recv()` hands back whatever
happened to arrive: half a packet, three packets, a packet split across two
calls. Every real bug in a first networking layer traces back to pretending
otherwise. So every message goes out as:

    [4 bytes: big-endian uint32 payload length][payload]

and the reader (see PacketStream) refuses to hand anything upward until it holds
the whole payload. Nothing else in the codebase is allowed to touch the socket.

ENCODING
--------
JSON + zlib, not pickle. Two reasons, and the first one is not negotiable:

  - pickle executes arbitrary code on load. A server that unpickles whatever a
    socket sends it is a remote shell, not a game. This is a LAN game between
    friends, but "it's only my friend" stops being true the moment the port is
    open on a VPN with strangers on it.
  - JSON is inspectable. When a desync happens - and it will - being able to
    print the exact bytes that caused it is worth more than the microseconds a
    binary format would save.

Chunk payloads are the one place where JSON's cost would actually bite (a chunk
is ~50k block ids), so those ship as base64'd raw numpy bytes inside the JSON
envelope, compressed by the same zlib pass. See encode_chunk/decode_chunk.

WHY THERE IS A PROTOCOL_VERSION
-------------------------------
The host and the friend WILL end up on different builds - that is the normal
case for two people passing an .exe around, not an edge case. Without a version
check the symptom is a KeyError deep inside a packet handler ten seconds after
joining, which reads as "the game is broken". With one, it is a clear message on
the connect screen.
"""

import base64
import json
import struct
import zlib

import numpy as np

# Bump on ANY change to a packet's fields. Cheap to bump, miserable to debug
# when you didn't.
PROTOCOL_VERSION = 2

DEFAULT_PORT = 25565

# Header is a fixed 4-byte length prefix.
_HEADER = struct.Struct(">I")
_HEADER_SIZE = _HEADER.size

# A single message may not exceed this. Without a ceiling, a hostile (or merely
# confused) peer sends a 4-byte header claiming 4 GB and the receiver
# obediently tries to buffer it. Chunks are the largest legitimate payload at
# well under 1 MB compressed.
MAX_PACKET_BYTES = 8 * 1024 * 1024


# --- packet type ids ---------------------------------------------------------
# Strings rather than ints: the readability is worth the handful of bytes, and
# zlib collapses the repetition anyway.

# client -> server
C_HELLO = "hello"                 # {username, protocol}
C_PLAYER_MOVE = "move"            # {x, y, z, yaw, pitch, on_ground, in_water, sneaking}
C_BLOCK_BREAK = "break"           # {x, y, z}
C_BLOCK_PLACE = "place"           # {x, y, z, id, meta, kind, facing, is_top, yaw}
C_DOOR_TOGGLE = "door"            # {x, y, z}
C_SWING = "swing"                 # {} - cosmetic arm swing, broadcast to others
C_DAMAGE = "damage"               # {amount, cause}
C_RESPAWN = "respawn"             # {}
C_CHAT = "chat"                   # {text}
C_REQUEST_CHUNK = "req_chunk"     # {cx, cz}
C_KEEPALIVE = "ka"                # {}

# server -> client
S_WELCOME = "welcome"             # {entity_id, seed, game_mode, spawn_x, spawn_z, x, y, z, yaw, pitch,
                                  #  health, air, inventory_slots, selected_slot, tick}
S_REJECT = "reject"               # {reason}
S_CHUNK = "chunk"                 # {cx, cz, blocks, meta, height_map, terrain_height}
S_UNLOAD_CHUNK = "unload_chunk"   # {cx, cz}
S_BLOCK_CHANGE = "block"          # {changes: [[x, y, z, id, meta], ...]}
S_ENTITY_SPAWN = "spawn"          # {entity_id, username, x, y, z, yaw, pitch}
S_ENTITY_MOVE = "emove"           # {entities: [[id, x, y, z, yaw, pitch, flags], ...]}
S_ENTITY_DESPAWN = "despawn"      # {entity_id}
S_ENTITY_SWING = "eswing"         # {entity_id}
S_PLAYER_STATE = "pstate"         # {health, air} - this client's own vitals, server-authoritative
S_TELEPORT = "tp"                 # {x, y, z} - server correcting/respawning this client
S_CHAT = "chat"                   # {text}
S_KEEPALIVE = "ka"

# Bit flags packed into S_ENTITY_MOVE's per-entity `flags` field. Packed rather
# than sent as named booleans because this packet goes out 20x/second per player
# and is the only thing here whose size actually matters.
FLAG_ON_GROUND = 1
FLAG_IN_WATER = 2
FLAG_SNEAKING = 4


class ProtocolError(Exception):
    """Raised on a malformed or oversized frame. Always fatal for that
    connection - a stream that has lost framing cannot be resynchronised."""


def encode_packet(packet_type: str, payload: dict = None) -> bytes:
    """Serialises one message into a length-prefixed, compressed frame."""
    body = {"t": packet_type}
    if payload:
        body.update(payload)
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    # level 1: this runs on the server's hot path for every entity update. The
    # difference between level 1 and level 9 on a 200-byte movement packet is
    # a few bytes and an order of magnitude of CPU.
    compressed = zlib.compress(raw, 1)
    return _HEADER.pack(len(compressed)) + compressed


def decode_packet(frame: bytes) -> dict:
    try:
        return json.loads(zlib.decompress(frame).decode("utf-8"))
    except (zlib.error, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"undecodable frame: {exc}") from exc


class PacketStream:
    """
    Turns a TCP byte stream into whole messages.

    Feed it whatever recv() gave you; take back however many complete messages
    that completed - which may be none, or may be five. This class exists so
    that no caller anywhere is ever tempted to assume one recv() equals one
    packet, which is the single most common way a hand-rolled protocol breaks
    (and it breaks under load, on the other person's machine, not yours).
    """

    __slots__ = ("_buffer",)

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data: bytes):
        self._buffer.extend(data)

    def packets(self):
        """Yields every complete message currently buffered, then stops."""
        while True:
            if len(self._buffer) < _HEADER_SIZE:
                return
            (length,) = _HEADER.unpack_from(self._buffer, 0)
            if length > MAX_PACKET_BYTES:
                raise ProtocolError(f"frame of {length} bytes exceeds the {MAX_PACKET_BYTES} limit")
            if len(self._buffer) < _HEADER_SIZE + length:
                return  # tail of a packet still in flight
            frame = bytes(self._buffer[_HEADER_SIZE:_HEADER_SIZE + length])
            del self._buffer[:_HEADER_SIZE + length]
            yield decode_packet(frame)


# --- chunk payloads ----------------------------------------------------------

def encode_array(array: np.ndarray) -> dict:
    """
    Packs a numpy array for the wire: raw bytes, base64'd so JSON can carry
    them, plus the dtype/shape needed to rebuild it.

    Base64 costs 33% overhead, which sounds bad until you notice zlib is
    applied to the whole envelope afterwards and a chunk's block array is
    overwhelmingly runs of one value. A real chunk lands around 3-8 KB.
    """
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def decode_array(encoded: dict) -> np.ndarray:
    raw = base64.b64decode(encoded["data"])
    array = np.frombuffer(raw, dtype=np.dtype(encoded["dtype"]))
    return array.reshape(encoded["shape"]).copy()  # copy: frombuffer is read-only


def encode_chunk(chunk) -> dict:
    """
    Everything a client needs to render and collide against a chunk.

    Deliberately NOT sent: `trees_generated`, `needs_save`, `dirty`. Those are
    generator/persistence bookkeeping and the client has no business having an
    opinion about any of them - it never generates or saves anything. The
    server is the only thing that owns world state; the client owns a picture
    of it.
    """
    return {
        "cx": chunk.cx,
        "cz": chunk.cz,
        "blocks": encode_array(chunk.blocks),
        "meta": encode_array(chunk.meta),
        "height_map": encode_array(chunk.height_map),
        "terrain_height": encode_array(chunk.terrain_height),
    }


def decode_chunk_into(chunk, payload: dict):
    """Fills an existing Chunk object from a S_CHUNK payload."""
    chunk.blocks[:] = decode_array(payload["blocks"])
    chunk.meta[:] = decode_array(payload["meta"])
    chunk.height_map[:, :] = decode_array(payload["height_map"])
    chunk.terrain_height[:, :] = decode_array(payload["terrain_height"])
    chunk.generated = True
    chunk.trees_generated = True   # the server already grew them; never do it again client-side
    chunk.needs_save = False
    chunk.dirty = True


# --- address parsing ---------------------------------------------------------

def parse_address(text: str):
    """
    Parses "26.31.44.10" or "26.31.44.10:25565" into (host, port), the way
    Minecraft's own Direct Connect box does.

    Returns (host, port) or raises ValueError with a message fit to show the
    player. Radmin VPN hands out 26.x.x.x addresses, which is what goes in here.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Enter an address")

    if text.count(":") > 1:
        raise ValueError("Bad address")

    if ":" in text:
        host, _, port_text = text.partition(":")
        host = host.strip()
        try:
            port = int(port_text.strip())
        except ValueError:
            raise ValueError("Port must be a number") from None
        if not (1 <= port <= 65535):
            raise ValueError("Port must be 1-65535")
    else:
        host = text
        port = DEFAULT_PORT

    if not host:
        raise ValueError("Enter an address")
    return host, port
