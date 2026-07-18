"""
world/doors.py
Door state encoding/decoding and door-specific collision/interaction logic.
Doors are stored as a normal Block.DOOR id in the chunk's block array (so
saving, breaking, get_block, etc. all keep working unmodified) plus one
metadata byte per block position (see Chunk.meta) that packs facing,
open/closed state, which vertical half a block is, and (new) which side
the HINGE is on - real Minecraft doors are two blocks tall (a bottom half
and a top half sharing one open/closed state) AND track hinge side
separately from facing (see door_collision_bounds), which is what
determines which way the door actually swings when opened. Without a
tracked hinge, every door with the same facing swings the same fixed
direction regardless of neighbors, which is exactly what caused two
adjacent doors to collide with each other on open - matching vanilla
requires hinge auto-detection at placement time (see
choose_hinge_for_placement) so adjacent doors swing AWAY from each other
into a proper double door, and so a door against a wall swings away from
the wall instead of into it.
"""

import math

# Facing: which horizontal direction the door's face (the flat side you see
# when it's shut) points towards.
FACING_NORTH = 0   # slab spans along X, sits at the -Z edge of the block, faces -Z
FACING_EAST = 1     # slab spans along Z, sits at the +X edge of the block, faces +X
FACING_SOUTH = 2    # slab spans along X, sits at the +Z edge of the block, faces +Z
FACING_WEST = 3     # slab spans along Z, sits at the -X edge of the block, faces -X

HINGE_LEFT = 0
HINGE_RIGHT = 1

_OPEN_BIT = 0b100         # bit 2
_TOP_HALF_BIT = 0b1000    # bit 3: 1 = this block is the door's top half, 0 = bottom half
_HINGE_BIT = 0b10000      # bit 4: 0 = left hinge, 1 = right hinge
_FACING_MASK = 0b011      # bits 0-1


def pack_door_meta(facing: int, is_open: bool, is_top: bool = False, hinge: int = HINGE_LEFT) -> int:
    value = (facing & _FACING_MASK) | (_OPEN_BIT if is_open else 0)
    if is_top:
        value |= _TOP_HALF_BIT
    if hinge == HINGE_RIGHT:
        value |= _HINGE_BIT
    return value


def unpack_door_meta(meta_value: int):
    """Returns (facing, is_open, is_top, hinge)."""
    facing = meta_value & _FACING_MASK
    is_open = bool(meta_value & _OPEN_BIT)
    is_top = bool(meta_value & _TOP_HALF_BIT)
    hinge = HINGE_RIGHT if (meta_value & _HINGE_BIT) else HINGE_LEFT
    return facing, is_open, is_top, hinge


def facing_from_player_yaw(yaw: float) -> int:
    """
    Picks the door's facing based on which way the player is looking when
    placing it, matching Minecraft's placement behavior: the door is
    oriented so its face points toward the player (i.e. away from the
    direction the player is looking), so it visually "faces" whoever placed
    it head-on rather than edge-on.
    """
    # normalize yaw into [0, 2*pi)
    two_pi = math.pi * 2
    y = yaw % two_pi
    if y < 0:
        y += two_pi
    # yaw=0 looks toward -Z (see Player.forward_vector); divide the circle
    # into four 90-degree sectors centered on each cardinal direction
    if y < math.pi / 4 or y >= 7 * math.pi / 4:
        return FACING_SOUTH   # player looking -Z -> door faces +Z (toward player's back approach)
    elif y < 3 * math.pi / 4:
        return FACING_WEST
    elif y < 5 * math.pi / 4:
        return FACING_NORTH
    else:
        return FACING_EAST


# "Left"/"right" of a facing, as seen by someone standing on the outside
# looking at the door's face (i.e. looking in the direction the door
# faces) - this matches how Minecraft players read "hinge on the left".
# Rotating `facing` 90 degrees counter-clockwise (in top-down X/Z) gives
# the direction that is "to the left" of someone facing that way.
_LEFT_OF = {FACING_NORTH: FACING_WEST, FACING_WEST: FACING_SOUTH,
            FACING_SOUTH: FACING_EAST, FACING_EAST: FACING_NORTH}
_RIGHT_OF = {v: k for k, v in _LEFT_OF.items()}
_FACING_DELTA = {FACING_NORTH: (0, -1), FACING_SOUTH: (0, 1),
                  FACING_EAST: (1, 0), FACING_WEST: (-1, 0)}


def choose_hinge_for_placement(world, wx: int, wy: int, wz: int, facing: int, player_yaw: float) -> int:
    """
    Auto-picks HINGE_LEFT or HINGE_RIGHT for a newly-placed door at
    (wx,wy,wz) with the given facing, following vanilla Minecraft's actual
    rule order (see the Door wiki page):
      1. If an adjacent matching door (same facing) is immediately to the
         left or right, form a proper double door: hinges end up OPPOSITE
         each other with handles touching, so the two doors swing apart
         instead of into each other.
      2. Otherwise, put the hinge on whichever side has more adjacent
         solid block faces (so the door swings away from a wall corner
         instead of into it).
      3. Otherwise (a tie, or nothing adjacent), fall back to whichever
         side is closer to the player's aim.
    `world` must expose get_block(x,y,z) and get_block_meta(x,y,z);
    is_solid must be importable from world.blocks.
    """
    from world.blocks import Block, is_solid

    left_dx, left_dz = _FACING_DELTA[_LEFT_OF[facing]]
    right_dx, right_dz = _FACING_DELTA[_RIGHT_OF[facing]]
    left_x, left_z = wx + left_dx, wz + left_dz
    right_x, right_z = wx + right_dx, wz + right_dz

    left_block = world.get_block(left_x, wy, left_z)
    right_block = world.get_block(right_x, wy, right_z)

    # Rule 1: adjacent matching-facing door -> opposite hinges, handles touching
    if left_block == Block.DOOR:
        lf, _, _, l_hinge = unpack_door_meta(world.get_block_meta(left_x, wy, left_z))
        if lf == facing:
            # neighbor's handle is on whichever side is NOT its hinge; put
            # our hinge on the side away from that door (i.e. our handle
            # touches its handle) -> our hinge is on the side facing away
            # from the neighbor, i.e. HINGE_RIGHT if neighbor is HINGE_LEFT
            return HINGE_RIGHT if l_hinge == HINGE_LEFT else HINGE_LEFT
    if right_block == Block.DOOR:
        rf, _, _, r_hinge = unpack_door_meta(world.get_block_meta(right_x, wy, right_z))
        if rf == facing:
            return HINGE_LEFT if r_hinge == HINGE_RIGHT else HINGE_RIGHT

    # Rule 2: more adjacent solid faces on one side -> hinge there. Checks
    # both the block directly beside this door AND the block beside the
    # cell it swings into (the "front" corner), matching vanilla's actual
    # corner-detection so a door in a corner swings away from the wall.
    front_dx, front_dz = -_FACING_DELTA[facing][0], -_FACING_DELTA[facing][1]

    def solid_count(dx, dz):
        count = 0
        if is_solid(world.get_block(wx + dx, wy, wz + dz)):
            count += 1
        if is_solid(world.get_block(wx + dx + front_dx, wy, wz + dz + front_dz)):
            count += 1
        return count

    left_solid = solid_count(left_dx, left_dz)
    right_solid = solid_count(right_dx, right_dz)
    if left_solid != right_solid:
        return HINGE_LEFT if left_solid > right_solid else HINGE_RIGHT

    # Rule 3: tie or nothing adjacent -> whichever side is closer to the
    # player's aim (approximate: compare player_yaw to the "left" facing's
    # direction vs the door's own facing).
    import math as _math
    yaw_diff = (player_yaw - _math.atan2(-_FACING_DELTA[facing][0], -_FACING_DELTA[facing][1])) % (_math.pi * 2)
    if yaw_diff > _math.pi:
        yaw_diff -= _math.pi * 2
    return HINGE_LEFT if yaw_diff < 0 else HINGE_RIGHT


# Collision AABB (in local 0..1 block-space) for each facing when CLOSED.
# A thin slab flush against one edge of the block, matching where it's drawn.
# The closed position does NOT depend on hinge - both hinge variants of the
# same facing look identical when shut (the slab fills the same edge);
# hinge only changes which corner the door swings AROUND when opened.
_SLAB_THICKNESS = 0.1875  # 3/16, matches classic door texture thickness

CLOSED_COLLISION_BOUNDS = {
    # (min_x, min_z, max_x, max_z) in local block-space 0..1; full height (0..1 in y)
    FACING_NORTH: (0.0, 0.0, 1.0, _SLAB_THICKNESS),
    FACING_SOUTH: (0.0, 1.0 - _SLAB_THICKNESS, 1.0, 1.0),
    FACING_WEST: (0.0, 0.0, _SLAB_THICKNESS, 1.0),
    FACING_EAST: (1.0 - _SLAB_THICKNESS, 0.0, 1.0, 1.0),
}

# When OPEN, the door swings 90 degrees AROUND ITS HINGE to lie flush
# against the perpendicular wall on the hinge side - so which of the two
# possible open positions it ends up in depends on BOTH facing and hinge,
# not facing alone. A previous revision only kept one fixed open position
# per facing (effectively always HINGE_LEFT''s swing direction), which is
# exactly why two doors placed as a matching pair (which vanilla would
# give opposite hinges, swinging apart) instead swung the same way and
# collided with each other.
#
# Each entry: (facing, hinge) -> (min_x, min_z, max_x, max_z). The door
# pivots around the hinge corner (a fixed point shared with its closed
# position''s corresponding corner) and sweeps into the quarter of the
# block on the hinge side.
OPEN_COLLISION_BOUNDS = {
    (FACING_NORTH, HINGE_LEFT): (0.0, 0.0, _SLAB_THICKNESS, 1.0),
    (FACING_NORTH, HINGE_RIGHT): (1.0 - _SLAB_THICKNESS, 0.0, 1.0, 1.0),
    (FACING_SOUTH, HINGE_LEFT): (1.0 - _SLAB_THICKNESS, 0.0, 1.0, 1.0),
    (FACING_SOUTH, HINGE_RIGHT): (0.0, 0.0, _SLAB_THICKNESS, 1.0),
    (FACING_WEST, HINGE_LEFT): (0.0, 1.0 - _SLAB_THICKNESS, 1.0, 1.0),
    (FACING_WEST, HINGE_RIGHT): (0.0, 0.0, 1.0, _SLAB_THICKNESS),
    (FACING_EAST, HINGE_LEFT): (0.0, 0.0, 1.0, _SLAB_THICKNESS),
    (FACING_EAST, HINGE_RIGHT): (0.0, 1.0 - _SLAB_THICKNESS, 1.0, 1.0),
}


def door_collision_bounds(facing: int, is_open: bool, hinge: int = HINGE_LEFT):
    """Returns (min_x, min_z, max_x, max_z) in LOCAL block-space (0..1)."""
    if not is_open:
        return CLOSED_COLLISION_BOUNDS.get(facing, CLOSED_COLLISION_BOUNDS[FACING_NORTH])
    return OPEN_COLLISION_BOUNDS.get((facing, hinge), OPEN_COLLISION_BOUNDS[(FACING_NORTH, HINGE_LEFT)])
