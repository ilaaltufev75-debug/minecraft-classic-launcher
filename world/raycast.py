"""
world/raycast.py
Voxel raycasting for block selection/breaking/placing: walks a ray from the
camera forward a fixed reach distance, stepping in small increments (simple
and robust, though not as elegant as a true DDA/Amanatides-Woo algorithm -
fine at this reach distance and world scale, and easy to reason about).
Returns both the hit block coordinate and the "placement" coordinate (the
last empty cell before the hit), matching the interaction model used
throughout the project: left-click breaks the hit block, right-click places
into the empty cell just in front of it.
"""

import config
from world.blocks import Block

# Blocks a block-selection ray passes straight through, exactly as if they were
# air.
#
# Fluids belong here because Minecraft's targeting ray runs with
# FluidMode.NONE: water is invisible to it. You cannot outline, break or
# right-click against a water block, and standing chest-deep in the sea you
# still target the seabed under your feet rather than the water in front of
# your face. Only a bucket switches the ray to a fluid-aware mode.
#
# Without this, water was simply a normal block: it drew a selection box when
# looked at, mined away like dirt, and left a hole in the ocean.
RAY_TRANSPARENT_BLOCKS = frozenset((Block.AIR, Block.WATER))


def raycast(world, origin, direction, max_distance=None, step=0.05):
    """
    origin: (x,y,z) float tuple, ray start (camera position)
    direction: (x,y,z) float tuple, normalized
    Returns dict {"x","y","z","place": (x,y,z)|None} or None if nothing hit.

    Passes through RAY_TRANSPARENT_BLOCKS. Note that "place" can therefore be a
    water cell, which is intended: a block placed into water replaces it, same
    as vanilla.
    """
    max_distance = config.REACH_DISTANCE if max_distance is None else max_distance
    ox, oy, oz = origin
    dx, dy, dz = direction

    prev_block = None
    t = 0.0
    while t < max_distance:
        px, py, pz = ox + dx * t, oy + dy * t, oz + dz * t
        bx, by, bz = int(px // 1), int(py // 1), int(pz // 1)

        if world.get_block(bx, by, bz) not in RAY_TRANSPARENT_BLOCKS:
            hit_frac_y = py - by  # 0..1, where within the block's vertical extent the ray hit
            # face_normal: which face of the hit block the ray actually
            # crossed into it through, derived from which cell the ray was
            # in just before entering the hit block. None if the ray
            # started inside a solid block (shouldn't normally happen).
            face_normal = None
            if prev_block is not None:
                pbx, pby, pbz = prev_block
                face_normal = (pbx - bx, pby - by, pbz - bz)
            return {"x": bx, "y": by, "z": bz, "place": prev_block,
                    "hit_frac_y": hit_frac_y, "face_normal": face_normal}
        prev_block = (bx, by, bz)
        t += step
    return None
