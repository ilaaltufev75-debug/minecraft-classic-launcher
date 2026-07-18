"""
world/stairs.py
Stairs: facing (N/E/S/W) + half (bottom/top) + shape (straight/inner/outer,
each left or right) - direct port of vanilla Minecraft Java Edition''s
StairBlock#getStairsShape/canTakeShape algorithm (net.minecraft.world.
level.block.StairBlock), plus a geometry system built the same way the
real game builds it: 8 fixed octet sub-cubes (0.5x0.5x0.5 each, one per
XYZ +/- combination - this mirrors StairBlock''s own OCTET_NNN..OCTET_PPP
constants) toggled on/off per (facing, half, shape) instead of an
approximated 2-3-box silhouette. This removes any ambiguity in exactly
where the riser/tread boundary sits, since every octet is independently
either present or absent - there is no "half box spanning a computed
range" for a shape to get subtly wrong.

FACING CONVENTION (matches real Minecraft''s stair block state exactly):
`facing` is the direction the stair''s FLAT/TALL riser face points toward.
Standing on the OPPOSITE side from `facing` is where the open step/cutout
faces you - this is vanilla''s actual block state semantics.

Meta byte layout (fits in one byte, mirrors doors.py''s approach):
  bits 0-1: facing (0=N,1=E,2=S,3=W)
  bit 2:    half (0=bottom, 1=top)
  bits 3-5: shape (0=straight,1=inner_left,2=inner_right,3=outer_left,4=outer_right)
"""

import math

FACING_NORTH = 0
FACING_EAST = 1
FACING_SOUTH = 2
FACING_WEST = 3

SHAPE_STRAIGHT = 0
SHAPE_INNER_LEFT = 1
SHAPE_INNER_RIGHT = 2
SHAPE_OUTER_LEFT = 3
SHAPE_OUTER_RIGHT = 4

_FACING_MASK = 0b011
_HALF_BIT = 0b100
_SHAPE_SHIFT = 3
_SHAPE_MASK = 0b111 << _SHAPE_SHIFT


# ---------------------------------------------------------------------------
# Direction helpers - port of net.minecraft.core.Direction, restricted to
# the 4 horizontal directions (stairs never use UP/DOWN for `facing`).
# ---------------------------------------------------------------------------

_DIRECTION_VECTORS = {
    FACING_NORTH: (0, -1),
    FACING_SOUTH: (0, 1),
    FACING_EAST: (1, 0),
    FACING_WEST: (-1, 0),
}

_OPPOSITE = {
    FACING_NORTH: FACING_SOUTH, FACING_SOUTH: FACING_NORTH,
    FACING_EAST: FACING_WEST, FACING_WEST: FACING_EAST,
}

# getClockWise(): NORTH->EAST->SOUTH->WEST->NORTH (viewed from above, +Y looking down)
_CLOCKWISE = {
    FACING_NORTH: FACING_EAST, FACING_EAST: FACING_SOUTH,
    FACING_SOUTH: FACING_WEST, FACING_WEST: FACING_NORTH,
}
_COUNTER_CLOCKWISE = {v: k for k, v in _CLOCKWISE.items()}

_AXIS_X = "x"
_AXIS_Z = "z"
_AXIS = {
    FACING_NORTH: _AXIS_Z, FACING_SOUTH: _AXIS_Z,
    FACING_EAST: _AXIS_X, FACING_WEST: _AXIS_X,
}


def _direction_vector(direction):
    return _DIRECTION_VECTORS[direction]


def _opposite(direction):
    return _OPPOSITE[direction]


def _counter_clockwise(direction):
    return _COUNTER_CLOCKWISE[direction]


def _axis(direction):
    return _AXIS[direction]


def _relative(wx, wy, wz, direction):
    dx, dz = _direction_vector(direction)
    return wx + dx, wy, wz + dz


def pack_stair_meta(facing: int, is_top: bool, shape: int = SHAPE_STRAIGHT) -> int:
    value = facing & _FACING_MASK
    if is_top:
        value |= _HALF_BIT
    value |= (shape << _SHAPE_SHIFT) & _SHAPE_MASK
    return value


def unpack_stair_meta(meta_value: int):
    """Returns (facing, is_top, shape)."""
    facing = meta_value & _FACING_MASK
    is_top = bool(meta_value & _HALF_BIT)
    shape = (meta_value & _SHAPE_MASK) >> _SHAPE_SHIFT
    return facing, is_top, shape


def facing_from_player_yaw(yaw: float) -> int:
    """
    Picks the stair's facing (the direction its solid riser face points)
    so the open cutout ends up facing the player, per vanilla's own
    placement rule ("a stair orients itself with the half-block side
    closest to the player"). yaw=0 looks toward -Z (Player.forward_vector),
    so a stair placed straight ahead sits at -Z relative to the player -
    the player is on the block''s +Z side, so the open cutout must be at
    +Z, meaning the solid riser (and therefore `facing`) points -Z, i.e.
    FACING_NORTH for yaw=0.
    """
    two_pi = math.pi * 2
    y = yaw % two_pi
    if y < 0:
        y += two_pi
    if y < math.pi / 4 or y >= 7 * math.pi / 4:
        return FACING_NORTH
    elif y < 3 * math.pi / 4:
        return FACING_EAST
    elif y < 5 * math.pi / 4:
        return FACING_SOUTH
    else:
        return FACING_WEST


# ---------------------------------------------------------------------------
# getStairsShape - direct port of StairBlock#getStairsShape.
# ---------------------------------------------------------------------------

def _is_stairs_at(world, wx, wy, wz):
    from world.blocks import Block
    return world.get_block(wx, wy, wz) in (Block.STAIRS_WOOD, Block.STAIRS_STONE)


def _can_take_shape(facing, is_top, world, wx, wy, wz, direction):
    """
    Port of StairBlock#canTakeShape: true unless the neighbor in
    `direction` is another stair with the SAME facing and SAME half as
    this one (a continuation of the same straight run that must not be
    pulled into a corner).
    """
    nx, ny, nz = _relative(wx, wy, wz, direction)
    if not _is_stairs_at(world, nx, ny, nz):
        return True
    n_facing, n_is_top, _ = unpack_stair_meta(world.get_block_meta(nx, ny, nz))
    if n_facing != facing:
        return True
    if n_is_top != is_top:
        return True
    return False


def compute_shape(world, wx, wy, wz, facing, is_top):
    """
    Direct port of StairBlock#getStairsShape:

        Direction facing = state.getValue(FACING);
        BlockState behindState = level.getBlockState(pos.relative(facing));
        if (isStairs(behindState) && state.getValue(HALF) == behindState.getValue(HALF)) {
            Direction behindFacing = behindState.getValue(FACING);
            if (behindFacing.getAxis() != facing.getAxis()
                    && canTakeShape(state, level, pos, behindFacing.getOpposite())) {
                if (behindFacing == facing.getCounterClockWise()) return StairsShape.OUTER_LEFT;
                return StairsShape.OUTER_RIGHT;
            }
        }
        BlockState frontState = level.getBlockState(pos.relative(facing.getOpposite()));
        if (isStairs(frontState) && state.getValue(HALF) == frontState.getValue(HALF)) {
            Direction frontFacing = frontState.getValue(FACING);
            if (frontFacing.getAxis() != facing.getAxis()
                    && canTakeShape(state, level, pos, frontFacing)) {
                if (frontFacing == facing.getCounterClockWise()) return StairsShape.INNER_LEFT;
                return StairsShape.INNER_RIGHT;
            }
        }
        return StairsShape.STRAIGHT;
    """
    behind_x, behind_y, behind_z = _relative(wx, wy, wz, facing)
    if _is_stairs_at(world, behind_x, behind_y, behind_z):
        behind_facing, behind_is_top, _ = unpack_stair_meta(world.get_block_meta(behind_x, behind_y, behind_z))
        if behind_is_top == is_top:
            if _axis(behind_facing) != _axis(facing) and _can_take_shape(
                    facing, is_top, world, wx, wy, wz, _opposite(behind_facing)):
                if behind_facing == _counter_clockwise(facing):
                    return SHAPE_OUTER_LEFT
                return SHAPE_OUTER_RIGHT

    front_x, front_y, front_z = _relative(wx, wy, wz, _opposite(facing))
    if _is_stairs_at(world, front_x, front_y, front_z):
        front_facing, front_is_top, _ = unpack_stair_meta(world.get_block_meta(front_x, front_y, front_z))
        if front_is_top == is_top:
            if _axis(front_facing) != _axis(facing) and _can_take_shape(
                    facing, is_top, world, wx, wy, wz, front_facing):
                if front_facing == _counter_clockwise(facing):
                    return SHAPE_INNER_LEFT
                return SHAPE_INNER_RIGHT

    return SHAPE_STRAIGHT


def neighbors_to_update(wx, wy, wz):
    """4 horizontal neighbor coords whose shape may need recomputing when
    this stair changes."""
    return [(wx + 1, wy, wz), (wx - 1, wy, wz), (wx, wy, wz + 1), (wx, wy, wz - 1)]


# ---------------------------------------------------------------------------
# Geometry: 8 fixed octet sub-cubes (mirrors StairBlock''s own OCTET_NNN,
# OCTET_NNP, OCTET_NPN, OCTET_NPP, OCTET_PNN, OCTET_PNP, OCTET_PPN,
# OCTET_PPP constants - each name encodes the sign of X, Y, Z: N=negative
# half (0..0.5), P=positive half (0.5..1)). Every octet is INDEPENDENTLY
# on or off, so there is no shared "computed range" for the tread and
# riser to disagree about - this is what removes the ambiguity that
# produced a visually "inverted" stair in the previous 2-3-box
# approximation.
# ---------------------------------------------------------------------------

# Octet key: (x_sign, y_sign, z_sign), each -1 (lower half) or +1 (upper half)
_ALL_OCTETS = [
    (x, y, z)
    for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)
]


def _octet_box(octet):
    x, y, z = octet
    x0 = 0.0 if x < 0 else 0.5
    y0 = 0.0 if y < 0 else 0.5
    z0 = 0.0 if z < 0 else 0.5
    return (x0, y0, z0, x0 + 0.5, y0 + 0.5, z0 + 0.5)


def _riser_octet_signs(facing: int):
    """The (x_sign, z_sign) of the octet column that sits on the SOLID
    riser side for this facing (the side `facing` points toward) - e.g.
    facing EAST (+X) means the riser is on the +X half, so x_sign=+1; the
    riser spans the FULL Z width at that X, so this only fixes X (or Z for
    N/S facings) and leaves the other axis unconstrained (both signs used)."""
    dx, dz = _direction_vector(facing)
    if dx != 0:
        return ("x", 1 if dx > 0 else -1)
    else:
        return ("z", 1 if dz > 0 else -1)


def _corner_octet_signs(facing: int, shape: int):
    """For INNER/OUTER shapes: the (x_sign, z_sign) of the SPECIFIC
    quarter-column (both axes fixed) that the corner box occupies,
    derived from facing + which way the corner turns (left/right)."""
    cdx, cdz = _direction_vector(facing)
    is_left = shape in (SHAPE_INNER_LEFT, SHAPE_OUTER_LEFT)
    perp = (-cdz, cdx) if is_left else (cdz, -cdx)
    x_sign = 1 if perp[0] > 0 else -1
    z_sign = 1 if perp[1] > 0 else -1
    return x_sign, z_sign


def _active_octets(facing: int, shape: int):
    """
    Returns the set of (x_sign, y_sign, z_sign) octets that are SOLID for
    a bottom-half stair of this facing/shape (y_sign=-1 is the tread half,
    y_sign=+1 is the riser half - top-half stairs mirror this in
    collision_boxes() below, not here, keeping this function half-agnostic).

    - TREAD (y_sign=-1): all 4 octets always present - full-footprint slab.
    - RISER (y_sign=+1): starts as the 2 octets on the riser side (see
      _riser_octet_signs), matching the half-footprint riser wall.
      - STRAIGHT: exactly those 2 octets, nothing more/less.
      - OUTER: the riser SHRINKS to just 1 octet (the corner one from
        _corner_octet_signs) - the far octet on the riser side is removed.
      - INNER: the riser KEEPS its normal 2 octets AND gains 1 extra octet
        on the adjoining perpendicular side (also from
        _corner_octet_signs), extending the riser into an L-shape.
    """
    riser_axis, riser_sign = _riser_octet_signs(facing)
    active = set()

    # tread: full 2x2 footprint, always on
    for x in (-1, 1):
        for z in (-1, 1):
            active.add((x, -1, z))

    if shape in (SHAPE_OUTER_LEFT, SHAPE_OUTER_RIGHT):
        cx, cz = _corner_octet_signs(facing, shape)
        active.add((cx, 1, cz))
    else:
        # STRAIGHT or INNER: normal half-footprint riser (2 octets)
        for other in (-1, 1):
            if riser_axis == "x":
                active.add((riser_sign, 1, other))
            else:
                active.add((other, 1, riser_sign))
        if shape in (SHAPE_INNER_LEFT, SHAPE_INNER_RIGHT):
            cx, cz = _corner_octet_signs(facing, shape)
            active.add((cx, 1, cz))

    return active


def collision_boxes(facing: int, is_top: bool, shape: int):
    """
    Builds the final box list from the fixed octet grid. For a bottom-half
    stair, y_sign=-1 octets ARE the tread (y 0..0.5) and y_sign=+1 octets
    ARE the riser (y 0.5..1.0) - exactly matching the geometry a stair
    resting on the floor should have (solid slab on the bottom, raised
    step on top). For a TOP-half (upside-down) stair, this whole
    arrangement mirrors vertically: y_sign=-1 becomes the riser (y
    0..0.5, hanging down from the ceiling) and y_sign=+1 becomes the
    solid slab flush against the ceiling (y 0.5..1.0) - which is the
    correct, and ONLY correct, vanilla upside-down-stair silhouette.
    Getting this mirroring backwards (using the same y-mapping regardless
    of is_top) is what produced a visually "inverted" stair report.
    """
    octets = _active_octets(facing, shape)
    boxes = []
    for (xs, ys, zs) in octets:
        # mirror the Y half for a top-half (upside-down) stair
        effective_ys = ys if not is_top else -ys
        boxes.append(_octet_box((xs, effective_ys, zs)))
    return _merge_boxes(boxes)


def _merge_boxes(boxes):
    """
    Merges adjacent same-Y-range octet boxes that share a full edge into
    fewer, larger boxes purely as a minor optimization (fewer draw calls/
    collision checks) - does NOT change the resulting shape, only how many
    boxes represent it. Two boxes merge along X if they have identical Y
    and Z ranges and are adjacent in X (and symmetrically for Z).
    """
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ax0, ay0, az0, ax1, ay1, az1 = a
                bx0, by0, bz0, bx1, by1, bz1 = b
                if ay0 == by0 and ay1 == by1 and az0 == bz0 and az1 == bz1 and abs(ax1 - bx0) < 1e-9:
                    boxes[i] = (ax0, ay0, az0, bx1, ay1, az1)
                    boxes.pop(j)
                    changed = True
                    break
                if ay0 == by0 and ay1 == by1 and az0 == bz0 and az1 == bz1 and abs(bx1 - ax0) < 1e-9:
                    boxes[i] = (bx0, ay0, bz0, ax1, ay1, az1)
                    boxes.pop(j)
                    changed = True
                    break
                if ay0 == by0 and ay1 == by1 and ax0 == bx0 and ax1 == bx1 and abs(az1 - bz0) < 1e-9:
                    boxes[i] = (ax0, ay0, az0, ax1, ay1, bz1)
                    boxes.pop(j)
                    changed = True
                    break
                if ay0 == by0 and ay1 == by1 and ax0 == bx0 and ax1 == bx1 and abs(bz1 - az0) < 1e-9:
                    boxes[i] = (bx0, ay0, bz0, ax1, ay1, az1)
                    boxes.pop(j)
                    changed = True
                    break
            else:
                continue
            break
    return boxes
