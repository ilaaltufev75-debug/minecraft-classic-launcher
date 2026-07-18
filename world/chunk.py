"""
world/chunk.py
A single chunk: a flat array of block IDs for a CHUNK_SIZE_X x CHUNK_HEIGHT x
CHUNK_SIZE_Z column of the world, plus mesh-building logic that turns block
data into vertex buffers ready for OpenGL upload.

Face winding and UV mapping here were verified analytically (cross-product
winding check + top/bottom UV alignment) during the original JS build, where
a real bug was found and fixed: two faces had reversed winding, and UVs were
copy-pasted without accounting for per-face vertex order, causing mirrored/
rotated textures. The same verified-correct FACES table is reused here.

Mesh building is fully vectorized with numpy (profiling showed the original
per-block Python loop - calling back into World.get_block for every single
face of every block - took over 40ms per chunk, dominated by function-call
and dict-lookup overhead; the vectorized version does face-visibility
checks and vertex generation over whole arrays in well under 1ms/chunk).
"""

import numpy as np

from world.blocks import Block, OPAQUE_BLOCKS, CUSTOM_RENDER_BLOCKS, is_solid

# Blocks that are drawn as part of the regular cube mesh but need alpha
# blending (glass) rather than a hard opaque draw. Kept in their own mesh
# group, built and uploaded separately from the opaque geometry, and drawn
# in a second pass AFTER all opaque geometry with depth writes turned off
# (see render/chunk_renderer.py) - a transparent fragment that still WRITES
# depth blocks anything drawn after it at that pixel from passing the depth
# test at all, even though the transparent fragment itself only partially
# covers it. That was the actual cause of "everything behind glass
# disappears" - glass, drawn in the same single opaque pass/buffer as solid
# blocks in whatever order np.unique happened to produce, would write its
# own (nearer) depth first, and then blocks materially BEHIND it in world
# space would silently fail the depth test and never get drawn at all,
# leaving bare sky/clear-color showing through instead of solid ground.
#
# Water is NOT here: it left the cube mesher entirely (see
# blocks.CUSTOM_RENDER_BLOCKS) and is built and drawn by
# render/water_renderer.py, which needs per-corner heights and its own shader.
ALPHA_BLEND_BLOCKS = frozenset((Block.GLASS,))
import config

CX = config.CHUNK_SIZE_X
CZ = config.CHUNK_SIZE_Z
CH = config.CHUNK_HEIGHT

# Precompute a boolean lookup table so "is this block id opaque" is an O(1)
# numpy fancy-index instead of a Python-level set membership check per block.
_MAX_BLOCK_ID = 255
_OPAQUE_LOOKUP = np.zeros(_MAX_BLOCK_ID + 1, dtype=bool)
for _bid in OPAQUE_BLOCKS:
    _OPAQUE_LOOKUP[_bid] = True

# Custom-render blocks (doors) are stored in the block array like any other
# block ID (for save/load, get_block, breaking, etc. to all keep working
# unmodified) but must be excluded from the regular cube mesher entirely -
# otherwise they'd render as an ordinary full cube with the door texture
# instead of the thin rotated slab their own dedicated renderer draws.
_CUBE_MESH_LOOKUP = np.ones(_MAX_BLOCK_ID + 1, dtype=bool)
_CUBE_MESH_LOOKUP[Block.AIR] = False
for _bid in CUSTOM_RENDER_BLOCKS:
    _CUBE_MESH_LOOKUP[_bid] = False

# Blocks that cast a ground shadow decal. Anything drawn as a normal cube
# qualifies - the old rule (logs and leaves only) is why a placed plank sat
# on thin air with nothing under it while the leaf block next to it had a
# shadow. Glass is excluded for the obvious reason, and the custom-render
# blocks (doors/fences/stairs) inherit their exclusion from _CUBE_MESH_LOOKUP:
# they are thin, mostly-empty shapes that a full-block decal would badly
# misrepresent.
_SHADOW_CASTER_LOOKUP = _CUBE_MESH_LOOKUP.copy()
_SHADOW_CASTER_LOOKUP[Block.GLASS] = False
_SHADOW_CASTER_LOOKUP[Block.WATER] = False

# Blocks whose faces are hidden by their OWN kind, even though they are
# transparent and so do not appear in OPAQUE_BLOCKS.
#
# Currently empty: this existed solely for water, which has since moved out of
# the cube mesher altogether (see blocks.CUSTOM_RENDER_BLOCKS) and does its own
# fluid-against-same-fluid culling in render/water_renderer.py, where it also
# has to reason about neighbouring water's HEIGHT rather than merely its
# presence.
#
# Kept rather than deleted because the mechanism is the right one for any future
# transparent block that occurs in large contiguous volumes: without it such a
# block draws all six faces against its own kind, and a 16x16x16 volume of it is
# ~24k faces of invisible internal surface per chunk. Deliberately NOT for glass
# or leaves - both are SUPPOSED to show their internal faces (that is what makes
# a stack of glass read as panes and a canopy read as individual leaves).
_SELF_CULLING_BLOCKS = frozenset()
_SELF_CULL_LOOKUP = np.zeros(_MAX_BLOCK_ID + 1, dtype=bool)
for _bid in _SELF_CULLING_BLOCKS:
    _SELF_CULL_LOOKUP[_bid] = True


def _idx(x: int, y: int, z: int) -> int:
    return x + z * CX + y * CX * CZ


# Each face: outward normal, 4 corners (CCW as seen from outside the cube -
# required for correct backface culling with GL_CCW front-face winding),
# and matching per-vertex UVs so the texture reads right-side-up and
# unmirrored on every face regardless of winding order.
FACES = [
    {  # +x (east)
        "dir": (1, 0, 0), "tex": "side",
        "corners": ((1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
        "uvs": ((1, 0), (0, 0), (0, 1), (1, 1)),
    },
    {  # -x (west)
        "dir": (-1, 0, 0), "tex": "side",
        "corners": ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
        "uvs": ((0, 0), (1, 0), (1, 1), (0, 1)),
    },
    {  # +y (top)
        "dir": (0, 1, 0), "tex": "up",
        "corners": ((0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)),
        "uvs": ((0, 1), (1, 1), (1, 0), (0, 0)),
    },
    {  # -y (bottom)
        "dir": (0, -1, 0), "tex": "down",
        "corners": ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
        "uvs": ((0, 0), (1, 0), (1, 1), (0, 1)),
    },
    {  # +z (south)
        "dir": (0, 0, 1), "tex": "side",
        "corners": ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
        "uvs": ((0, 0), (1, 0), (1, 1), (0, 1)),
    },
    {  # -z (north)
        "dir": (0, 0, -1), "tex": "side",
        "corners": ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
        "uvs": ((0, 0), (0, 1), (1, 1), (1, 0)),
    },
]


class Chunk:
    """
    Holds block data for one (chunk_x, chunk_z) column of the infinite world.
    World-space block coordinates map to chunk-local coordinates as:
        local_x = world_x - chunk_x * CHUNK_SIZE_X
        local_z = world_z - chunk_z * CHUNK_SIZE_Z
    """

    __slots__ = ("cx", "cz", "blocks", "meta", "height_map", "terrain_height", "dirty",
                 "has_mesh", "trees_generated", "generated", "needs_save")

    def __init__(self, cx: int, cz: int):
        self.cx = cx
        self.cz = cz
        self.blocks = np.zeros(CX * CH * CZ, dtype=np.uint8)
        # Per-block metadata byte, used only by custom-render blocks (doors)
        # that need extra state beyond "which block id is here" - which way
        # they face, and whether they're currently open. Kept as a separate
        # sparse-in-practice array rather than widening `blocks` to 16-bit,
        # since only a tiny fraction of blocks in a world are ever doors.
        self.meta = np.zeros(CX * CH * CZ, dtype=np.uint8)
        self.height_map = np.zeros((CX, CZ), dtype=np.int16)
        # The NATURAL surface height of each column, as worldgen produced it.
        # Unlike height_map (the topmost solid block, which rises the moment
        # the player stacks a block and falls when they dig) this never moves
        # after generation, so "is this block part of the ground, or is it
        # standing on top of the ground?" stays answerable for the lifetime of
        # the chunk. build_shadow_spots is the consumer; see it for why the
        # distinction cannot be recovered from the block array alone.
        self.terrain_height = np.zeros((CX, CZ), dtype=np.int16)
        self.dirty = True          # needs its GPU mesh (re)built
        self.has_mesh = False
        self.trees_generated = False
        self.generated = False     # terrain/caves/ores pass has run
        # Whether this chunk differs from what is on disk. Starts True so a
        # brand new chunk is written out once; cleared by world_save on write
        # and on load, set again by any block/meta change. `dirty` above is a
        # DIFFERENT thing - it means "GPU mesh is stale" and gets set by
        # cosmetic rebuilds that don't imply a re-save, so the two must not be
        # merged. Without this flag the periodic autosave rewrote every loaded
        # chunk every time, which at render distance 32 meant pushing ~3200
        # compressed files through zlib on the main thread - the multi-second
        # hitch that appeared every 20-40 seconds.
        self.needs_save = True

    def get_local(self, lx: int, y: int, lz: int) -> int:
        if lx < 0 or lx >= CX or lz < 0 or lz >= CZ or y < 0 or y >= CH:
            return Block.AIR
        return int(self.blocks[_idx(lx, y, lz)])

    def get_local_meta(self, lx: int, y: int, lz: int) -> int:
        if lx < 0 or lx >= CX or lz < 0 or lz >= CZ or y < 0 or y >= CH:
            return 0
        return int(self.meta[_idx(lx, y, lz)])

    def set_local_meta(self, lx: int, y: int, lz: int, meta_value: int):
        if lx < 0 or lx >= CX or lz < 0 or lz >= CZ or y < 0 or y >= CH:
            return
        self.meta[_idx(lx, y, lz)] = meta_value
        self.dirty = True
        self.needs_save = True

    def set_local(self, lx: int, y: int, lz: int, block_id: int):
        if lx < 0 or lx >= CX or lz < 0 or lz >= CZ or y < 0 or y >= CH:
            return
        idx = _idx(lx, y, lz)
        # Metadata belongs to the block that was here, not to the cell. Clearing
        # it whenever the ID actually changes is what stops one block's state
        # bits from being read as another's: planks placed into a water cell used
        # to inherit that water's flow level, and a door built where flowing
        # water had been came up already facing/hinged from those same bits.
        # Callers that DO want metadata (World.set_block's meta_value, doors,
        # stairs, fences) write it immediately after this, so nothing legitimate
        # is lost. Previously this only fired for AIR, which happened to cover
        # break-then-place but nothing else.
        if int(self.blocks[idx]) != block_id:
            self.meta[idx] = 0
        self.blocks[idx] = block_id
        self.dirty = True
        self.needs_save = True
        if block_id == Block.AIR:
            if self.height_map[lx, lz] == y:
                h = y - 1
                while h >= 0 and self.blocks[_idx(lx, h, lz)] == Block.AIR:
                    h -= 1
                self.height_map[lx, lz] = h
        elif y > self.height_map[lx, lz]:
            self.height_map[lx, lz] = y

    def world_origin(self):
        return self.cx * CX, self.cz * CZ

    def blocks_yzx(self) -> np.ndarray:
        """Returns this chunk's blocks reshaped as (CH, CZ, CX) - verified to satisfy
        reshaped[y, z, x] == blocks[_idx(x, y, z)]."""
        return self.blocks.reshape(CH, CZ, CX)

    def meta_yzx(self) -> np.ndarray:
        """Same view as blocks_yzx(), over the metadata array. Needed by the
        water mesher, which is the first thing to read metadata in BULK - doors
        and stairs are sparse enough to look up one at a time, whereas an ocean
        chunk holds thousands of water cells whose flow level every one of them
        needs."""
        return self.meta.reshape(CH, CZ, CX)

    # Simple flat "blob" shadow decals drawn on the ground under anything
    # standing above it (trees, and anything the player builds) - matches
    # classic Minecraft's cheap ground shadows rather than real shadow-mapping.
    SHADOW_MAX_GAP = 7          # don't project a shadow from further than this many blocks up

    def build_shadow_spots(self):
        """
        Returns a list of (world_x, ground_surface_y, world_z, strength) spots
        to draw a ground-shadow decal at: one per solid run that stands clear
        of whatever is below it. A column can produce several - a block
        floating high and another one lower down each cast their own. Any cube
        block casts - logs, leaves, planks, cobblestone, whatever the player
        put there. ground_surface_y is the Y of the catching block's TOP FACE
        (one above its own Y), i.e. the walkable surface the decal sits flush
        against. strength fades with the gap, so a shadow reads strongest right
        under the thing casting it.

        A caster must sit ABOVE terrain_height - that single test is doing
        two jobs.

        Correctness: it is the only thing separating "a plank floating over
        grass" from "a cave roof over a cave floor". Those two are structurally
        identical in the block array - solid run, air gap, solid ground - so a
        naive "any solid block with air beneath it casts" rule stamps a decal
        onto the floor of every cave in the world. Invisible from the surface,
        but they still fill up ShadowRenderer's instance cap and start starving
        the real shadows on screen. Caves live below the natural surface by
        construction (worldgen's cave_top is heights-3 on flat ground), so
        comparing against terrain_height rejects every one of them for free.
        The old code dodged this only by accident, by hardcoding the caster set
        to logs and leaves - which is exactly the bug being fixed here.

        Speed: the previous version scanned all 256 columns from the top down
        hunting for a log or leaf and, on the ~99% of columns that have no
        tree, walked the entire column to y=0 before giving up. Now one integer
        compare per column rejects bare ground outright and only the handful of
        columns with something built or grown on them get scanned at all.
        """
        ox, oz = self.world_origin()
        blocks = self.blocks_yzx()  # (CH, CZ, CX)

        candidates = np.argwhere(self.height_map > self.terrain_height)
        if len(candidates) == 0:
            return []

        spots = []
        for cand in candidates:
            lx = int(cand[0])
            lz = int(cand[1])
            column = blocks[:, lz, lx]
            ground_limit = int(self.terrain_height[lx, lz])
            y = min(int(self.height_map[lx, lz]), CH - 1)

            # Walk the column down to the natural surface and cast from EVERY
            # solid run on the way, not just the topmost one.
            #
            # Only casting from the top run is wrong whenever a column holds
            # two separate things - say a block floating high up and another
            # one lower down. The high block would cast (onto the lower block's
            # top face, or nowhere at all if the two are more than
            # SHADOW_MAX_GAP apart) and the lower block, which is the one
            # actually hanging over the grass, would never be considered at
            # all: the column was already spent. The result is a shadow missing
            # from the ground under exactly those blocks that have something
            # above them, punching a hole in an otherwise continuous row.
            #
            # The old log/leaves-only version had the same flaw and got away
            # with it because nothing is ever stacked above a tree's canopy.
            while y > ground_limit:
                if not _SHADOW_CASTER_LOOKUP[column[y]]:
                    y -= 1
                    continue

                # Bottom of this run of caster blocks. A run never descends into
                # the terrain itself, so a tree trunk stops at the grass block
                # it grows out of rather than continuing down through the dirt
                # and stone below it.
                run_bottom = y
                while y > ground_limit and _SHADOW_CASTER_LOOKUP[column[y]]:
                    run_bottom = y
                    y -= 1

                # What catches this run's shadow: the first solid block below
                # it, within SHADOW_MAX_GAP. Deliberately any solid block
                # rather than the terrain specifically, so a block hanging over
                # another block shades that block instead of punching through
                # to the ground. y is already positioned below the run, so the
                # `continue`s here resume the scan at the next run down.
                gy = run_bottom - 1
                gap = 0
                while gy >= 0 and int(column[gy]) == Block.AIR and gap < self.SHADOW_MAX_GAP:
                    gy -= 1
                    gap += 1
                if gy < 0 or int(column[gy]) == Block.AIR:
                    continue  # nothing within range to catch it
                if not is_solid(int(column[gy])):
                    continue
                if gap < 1:
                    # Caster resting directly on whatever is below it.
                    # _SHADOW_SIZE is exactly one block, so the decal would land
                    # on the caster's own underside and be completely covered by
                    # it - an invisible spot that still costs an instance slot.
                    # Vanilla doesn't shade under a block flush on the ground
                    # either.
                    continue

                strength = max(0.15, 0.55 - gap * 0.06)
                spots.append((ox + lx, gy + 1, oz + lz, strength))
        return spots

    def build_door_instances(self):
        """
        Scans this chunk's blocks for DOOR blocks and returns a list of
        (world_x, world_y, world_z, facing, is_open, is_top, hinge) tuples
        for the door renderer to draw thin rotated slabs at. Each vertical
        half (bottom/top) is its own block entry with its own Y, so a
        two-block door naturally produces two instances here. hinge
        determines which corner the open animation swings AROUND (see
        world/doors.py OPEN_COLLISION_BOUNDS) - without it the renderer
        can't tell which of the two valid open positions to draw.
        """
        from world.blocks import Block as _B
        from world.doors import unpack_door_meta
        ox, oz = self.world_origin()
        blocks = self.blocks_yzx()  # (CH, CZ, CX)
        door_mask = blocks == _B.DOOR
        if not door_mask.any():
            return []
        ys, zs, xs = np.nonzero(door_mask)
        instances = []
        for y, z, x in zip(ys.tolist(), zs.tolist(), xs.tolist()):
            meta_val = self.get_local_meta(x, y, z)
            facing, is_open, is_top, hinge = unpack_door_meta(meta_val)
            instances.append((ox + x, y, oz + z, facing, is_open, is_top, hinge))
        return instances

    def build_stair_instances(self):
        """Returns (world_x, world_y, world_z, block_id, facing, is_top, shape) per stair block."""
        from world.blocks import Block as _B
        from world.stairs import unpack_stair_meta
        ox, oz = self.world_origin()
        blocks = self.blocks_yzx()
        mask = (blocks == _B.STAIRS_WOOD) | (blocks == _B.STAIRS_STONE)
        if not mask.any():
            return []
        ys, zs, xs = np.nonzero(mask)
        instances = []
        for y, z, x in zip(ys.tolist(), zs.tolist(), xs.tolist()):
            block_id = int(blocks[y, z, x])
            facing, is_top, shape = unpack_stair_meta(self.get_local_meta(x, y, z))
            instances.append((ox + x, y, oz + z, block_id, facing, is_top, shape))
        return instances

    def build_fence_instances(self):
        """Returns (world_x, world_y, world_z, north, south, east, west) per fence block."""
        from world.blocks import Block as _B
        from world.fences import unpack_connections
        ox, oz = self.world_origin()
        blocks = self.blocks_yzx()
        mask = blocks == _B.FENCE
        if not mask.any():
            return []
        ys, zs, xs = np.nonzero(mask)
        instances = []
        for y, z, x in zip(ys.tolist(), zs.tolist(), xs.tolist()):
            north, south, east, west = unpack_connections(self.get_local_meta(x, y, z))
            instances.append((ox + x, y, oz + z, north, south, east, west))
        return instances

    def build_mesh_data(self, padded_blocks: np.ndarray):
        """
        Builds interleaved vertex data grouped by (block_id, tex_face) so the
        renderer can batch draw calls per texture, using a fully vectorized
        approach: face visibility for every block and every direction is
        computed via shifted-array comparison against padded_blocks (shape
        (CH, CZ+2, CX+2): this chunk's blocks plus a 1-block border pulled
        from neighboring chunks by the caller), then vertex positions/UVs
        for all visible faces of a given (block_id, face) are generated in
        one batch via broadcasting instead of a per-block Python loop.

        Returns: dict[(block_id, tex_face)] -> {
            "positions": np.ndarray (N,3) float32 in world space,
            "normals":   np.ndarray (N,3) float32,
            "uvs":       np.ndarray (N,2) float32,
            "indices":   np.ndarray (M,)  uint32,
        }
        """
        ox, oz = self.world_origin()
        own = padded_blocks[:, 1:-1, 1:-1]  # (CH, CZ, CX), this chunk's own blocks

        # Y has no real neighbor chunks (there's only one world column of chunks
        # vertically), so pad it with air top/bottom: nothing exists below y=0,
        # and the block above the build ceiling is always open air.
        y_padded = np.zeros((CH + 2, CZ + 2, CX + 2), dtype=np.uint8)
        y_padded[1:-1, :, :] = padded_blocks
        opaque_padded = _OPAQUE_LOOKUP[y_padded]

        result = {}

        for face in FACES:
            dx, dy, dz = face["dir"]
            # shift the padded opaque mask by (dx,dy,dz) to get "is the neighbor in
            # this direction opaque" for every block in the chunk at once
            neighbor_opaque = opaque_padded[
                1 + dy: 1 + dy + CH,
                1 + dz: 1 + dz + CZ,
                1 + dx: 1 + dx + CX,
            ]
            # Same shift again, but carrying the neighbour's block ID rather
            # than just "is it opaque" - needed for the self-culling test
            # below, which has to know WHICH block is next door, not merely
            # whether it blocks light.
            neighbor_ids = y_padded[
                1 + dy: 1 + dy + CH,
                1 + dz: 1 + dz + CZ,
                1 + dx: 1 + dx + CX,
            ]
            own_opaque = _CUBE_MESH_LOOKUP[own]
            hidden = neighbor_opaque | (_SELF_CULL_LOOKUP[own] & (neighbor_ids == own))
            visible = own_opaque & ~hidden
            if not visible.any():
                continue

            ys, zs, xs = np.nonzero(visible)
            block_ids_here = own[ys, zs, xs]

            corners = np.array(face["corners"], dtype=np.float32)  # (4,3) local offsets
            uv_local = np.array(face["uvs"], dtype=np.float32)      # (4,2)
            normal = np.array(face["dir"], dtype=np.float32)

            # group by block_id so different block types end up in separate draw groups
            unique_ids = np.unique(block_ids_here)
            for block_id in unique_ids:
                block_id_int = int(block_id)
                sel = block_ids_here == block_id
                n = int(sel.sum())
                if n == 0:
                    continue

                base = np.stack([
                    xs[sel] + ox, ys[sel], zs[sel] + oz
                ], axis=1).astype(np.float32)  # (n,3) world-space base position

                positions = (base[:, None, :] + corners[None, :, :]).reshape(-1, 3)
                uvs = np.tile(uv_local, (n, 1))
                # BUG FIX: normal is a single (3,) vector (one direction per face),
                # but each face contributes 4 VERTICES, matching positions/uvs
                # (which are correctly (4n, ...)). The previous np.tile(normal, (n,1))
                # produced only (n, 3) - one normal per face instead of one per
                # vertex - silently misaligning the normals array against
                # positions/uvs by a factor of 4 once multiple face groups were
                # concatenated together in chunk_renderer.py. That misalignment
                # fed garbage/out-of-range normal data into the lighting shader,
                # which is what produced the near-total-black rendering reported
                # on real hardware (some vertices coincidentally landed on valid
                # normal values from unrelated faces, most didn't).
                normals = np.tile(normal, (n * 4, 1))

                base_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
                vertex_offsets = (np.arange(n, dtype=np.uint32) * 4).reshape(-1, 1)
                indices = (base_indices[None, :] + vertex_offsets).reshape(-1)

                key = (block_id_int, face["tex"])
                if key not in result:
                    result[key] = {
                        "positions": [positions], "normals": [normals],
                        "uvs": [uvs], "indices": [indices], "vert_count": n * 4,
                    }
                else:
                    g = result[key]
                    # subsequent face directions contributing to the same (block_id, tex)
                    # group need their indices offset by the vertex count seen so far
                    indices = indices + g["vert_count"]
                    g["positions"].append(positions)
                    g["normals"].append(normals)
                    g["uvs"].append(uvs)
                    g["indices"].append(indices)
                    g["vert_count"] += n * 4

        # concatenate each group's per-face-direction chunks into final flat arrays
        final = {}
        for key, g in result.items():
            if g["vert_count"] == 0:
                continue
            final[key] = {
                "positions": np.concatenate(g["positions"]).astype(np.float32),
                "normals": np.concatenate(g["normals"]).astype(np.float32),
                "uvs": np.concatenate(g["uvs"]).astype(np.float32),
                "indices": np.concatenate(g["indices"]).astype(np.uint32),
            }
        return final
