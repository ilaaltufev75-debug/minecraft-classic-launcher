"""
world/fences.py
Fence connection state: 4 bits (north/south/east/west) recomputed from
neighbors whenever the fence or an adjacent block changes. A fence connects
to another fence, or to any solid opaque full-cube neighbor - not to glass,
stairs, doors, or other custom-shaped blocks.
"""

from world.blocks import Block, OPAQUE_BLOCKS, CUSTOM_RENDER_BLOCKS

_NORTH_BIT = 0b0001
_SOUTH_BIT = 0b0010
_EAST_BIT = 0b0100
_WEST_BIT = 0b1000

_DELTAS = {
    "north": (0, -1, _NORTH_BIT),
    "south": (0, 1, _SOUTH_BIT),
    "east": (1, 0, _EAST_BIT),
    "west": (-1, 0, _WEST_BIT),
}


def can_connect_to(block_id: int) -> bool:
    if block_id == Block.FENCE:
        return True
    if block_id in CUSTOM_RENDER_BLOCKS:
        return False  # stairs/doors - not full solid cubes
    return block_id in OPAQUE_BLOCKS


def compute_connections(world, wx: int, wy: int, wz: int) -> int:
    value = 0
    for _name, (dx, dz, bit) in _DELTAS.items():
        neighbor = world.get_block(wx + dx, wy, wz + dz)
        if can_connect_to(neighbor):
            value |= bit
    return value


def unpack_connections(meta_value: int):
    """Returns (north, south, east, west) booleans."""
    return (bool(meta_value & _NORTH_BIT), bool(meta_value & _SOUTH_BIT),
            bool(meta_value & _EAST_BIT), bool(meta_value & _WEST_BIT))


def neighbors_to_update(wx, wy, wz):
    return [(wx + 1, wy, wz), (wx - 1, wy, wz), (wx, wy, wz + 1), (wx, wy, wz - 1)]


# -- collision --------------------------------------------------------------
# Post is always solid 3/8..5/8 in x/z, full height. Each connected side
# extends a thinner rail box out to that edge, at post height (so mobs/
# players can't walk through even where the rail sits, matching vanilla).
_POST_MIN, _POST_MAX = 0.375, 0.625
# Vanilla fence collision extends to 1.5 blocks tall (taller than its visible
# post/rails), specifically so a normal jump can't clear it - matches real
# Minecraft's anti-jump behavior. Physics does a full 3D box-vs-box overlap
# test (see player/physics.py is_solid_at), so a collision box taller than
# the block cell itself still works correctly: it's checked against every
# block cell the player's AABB currently overlaps, including the one above
# the fence post when the player is mid-jump.
FENCE_HEIGHT = 1.5


def collision_boxes(north: bool, south: bool, east: bool, west: bool):
    boxes = [(_POST_MIN, 0.0, _POST_MIN, _POST_MAX, FENCE_HEIGHT, _POST_MAX)]
    if north:
        boxes.append((_POST_MIN, 0.0, 0.0, _POST_MAX, FENCE_HEIGHT, _POST_MIN))
    if south:
        boxes.append((_POST_MIN, 0.0, _POST_MAX, _POST_MAX, FENCE_HEIGHT, 1.0))
    if west:
        boxes.append((0.0, 0.0, _POST_MIN, _POST_MIN, FENCE_HEIGHT, _POST_MAX))
    if east:
        boxes.append((_POST_MAX, 0.0, _POST_MIN, 1.0, FENCE_HEIGHT, _POST_MAX))
    return boxes
