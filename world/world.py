"""
world/world.py
The infinite world manager. Owns a dict of loaded Chunk objects keyed by
(cx, cz) and streams chunks in/out around the player based on render
distance - this is the core difference from the old fixed 128x128 JS
build, which pre-generated everything up front. Here, chunks are generated
lazily the first time they're needed and unloaded once the player wanders
far enough away (keeping memory bounded no matter how far the player
travels).

Block access always goes through get_block/set_block using WORLD
coordinates; callers never need to know about chunk boundaries.
"""

from collections import deque

from world.chunk import Chunk, CX, CZ, CH
from world.noise import WorldNoise
from world.blocks import Block
from world import worldgen
from world import block_behavior
from world.ticks import TickScheduler
# Imported for their registration side effect: these fill block_behavior's
# handler registries at import time, and nothing else references them by name.
from world import falling  # noqa: F401
from world import fluids  # noqa: F401
import config
import numpy as np


# The six orthogonal neighbours a block change notifies. Matches Minecraft's
# updateNeighbors: diagonals are deliberately NOT included - a block diagonally
# adjacent to a change is not considered disturbed by it.
_NEIGHBOR_OFFSETS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

# Cells a newly placed block is allowed to overwrite.
#
# Water is in here for the same reason it is in raycast's pass-through set, and
# the two have to agree: the targeting ray now goes straight through water, so
# aiming at the seabed from the shore hands back a WATER cell as the placement
# spot. If placing could only overwrite air, that click would silently do
# nothing and building at the water's edge would break. Minecraft calls this
# property "replaceable" and water has it - a block placed into water displaces
# it rather than being refused.
REPLACEABLE_BLOCKS = frozenset((Block.AIR, Block.WATER))


def world_to_chunk_coords(wx: int, wz: int):
    return wx // CX, wz // CZ


def world_to_local(wx: int, wz: int):
    return wx % CX, wz % CZ


class World:
    def __init__(self, seed: int, save_dir: str = None):
        self.seed = seed
        self.noise = WorldNoise(seed)
        self.save_dir = save_dir  # if set, chunks are loaded from / saved to disk
        self.chunks: dict[tuple[int, int], Chunk] = {}

        # Scheduled block wakeups (fluids, falling blocks). See world/ticks.py.
        self.ticks = TickScheduler()

        # Chunks whose mesh needs (re)building, split into TWO queues by
        # priority rather than kept in one FIFO.
        #
        # `_dirty_bulk_*` holds chunks marked dirty by streaming/generation.
        # It comes out roughly closest-first for free, because
        # update_streaming() sorts the generation queue by distance to the
        # player and chunks are marked dirty in the order they generate.
        #
        # `_dirty_urgent_*` holds chunks a PLAYER EDIT touched, and is always
        # drained first.
        #
        # Why the split: raising the render distance from 2 to 32 marks ~3200
        # freshly generated chunks dirty at once. At
        # CHUNK_BUILD_BUDGET_PER_FRAME = 2 that queue needs ~1600 frames
        # (~27 seconds) to drain. With a single FIFO, a block the player broke
        # during that window was appended to the BACK of those 3200 entries,
        # so the mesh containing it did not rebuild for half a minute: the
        # block was already gone from `blocks` (physics and raycasting walked
        # straight through it) but was still visibly on screen. That was never
        # a renderer bug - the edit was correct and simply never got a build
        # slot.
        #
        # Bulk entries are not removed from the queue when they get promoted
        # to urgent or unloaded (that would be an O(n) list scan on every
        # edit); the membership set is the authority, and queue entries whose
        # key is no longer in the set are skipped on pop.
        self._dirty_urgent_queue = deque()
        self._dirty_urgent_set: set[tuple[int, int]] = set()
        self._dirty_bulk_queue = deque()
        self._dirty_bulk_set: set[tuple[int, int]] = set()

        # chunk coordinates that are "wanted" (in range) but not yet generated,
        # queued so generation spreads across frames instead of spiking when
        # the player crosses a chunk boundary and a whole new ring is needed
        self._pending_gen_queue: list[tuple[int, int]] = []
        self._pending_gen_set: set[tuple[int, int]] = set()

        # chunk coords that have terrain but are waiting for their full 3x3
        # neighborhood before trees can be grown (avoids leaf spill onto
        # not-yet-generated neighbors)
        self._pending_trees_queue: list[tuple[int, int]] = []
        self._pending_trees_set: set[tuple[int, int]] = set()

        # chunk coords unloaded since the last pop_recently_unloaded() call
        self.recently_unloaded: list[tuple[int, int]] = []

        # Optional fn(wx, wy, wz, block_id, meta_value) called after EVERY block
        # write. The multiplayer server installs one here and turns the calls
        # into S_BLOCK_CHANGE broadcasts.
        #
        # A hook rather than the server diffing chunks: set_block is already the
        # single funnel every mutation in the game goes through - player edits,
        # falling sand, water spreading, tree growth - so one listener here
        # cannot miss a change. Diffing would have to run every frame over every
        # loaded chunk to find the two blocks that actually moved, and would
        # still lose the ORDER they moved in, which for water is the difference
        # between a waterfall and a puddle appearing from nowhere.
        #
        # None in singleplayer: the attribute check costs a load and a branch per
        # set_block, which is noise next to the numpy write it guards.
        self.change_listener = None

    # -- chunk lifecycle -----------------------------------------------------

    def _get_or_create_chunk(self, cx: int, cz: int) -> Chunk:
        """Generates immediately, bypassing the per-frame budget. Use for
        direct block access (get_block/set_block) where a synchronous
        answer is required; use request_chunk() for streaming-driven loads.
        If this world is backed by a save directory and a saved version of
        this chunk exists on disk, it's loaded instead of regenerated -
        this is what makes player edits (breaking/placing blocks) persist
        across sessions instead of being overwritten by fresh worldgen."""
        key = (cx, cz)
        chunk = self.chunks.get(key)
        if chunk is None:
            if self.save_dir is not None:
                from save import world_save
                if world_save.chunk_save_exists(self.save_dir, cx, cz):
                    chunk = world_save.load_chunk(self.save_dir, cx, cz)
                    self.chunks[key] = chunk
                    self._mark_dirty(cx, cz, urgent=False)
                    self._mark_neighbors_dirty(cx, cz)
                    return chunk
            chunk = Chunk(cx, cz)
            self.chunks[key] = chunk
            worldgen.generate_chunk_terrain(chunk, self.noise)
            self._mark_dirty(cx, cz, urgent=False)
            self._mark_neighbors_dirty(cx, cz)
        return chunk

    def _mark_neighbors_dirty(self, cx: int, cz: int):
        """
        A chunk arriving changes what its already-built neighbours should look
        like, so their meshes are stale the moment it lands.

        This never used to matter enough to notice: the cube mesher only culls
        against ORTHOGONAL neighbours, and a chunk generated later simply meant
        one column of faces at the seam was drawn that could have been culled -
        invisible, merely wasteful. The water mesher changed that. A water
        corner's height is averaged from the four cells DIAGONALLY around it,
        so a chunk whose diagonal neighbour was missing at mesh time read those
        cells as air, and air drags a corner's height to zero. The result was a
        visible notch dimpled into the sea at every chunk corner.

        Marked bulk, never urgent: this is streaming, not a player edit, and the
        dirty sets dedupe, so during a bulk load - where neighbours are queued
        but unbuilt anyway - it costs nothing at all.
        """
        for (dx, dz) in ((-1, 0), (1, 0), (0, -1), (0, 1),
                         (-1, -1), (1, -1), (-1, 1), (1, 1)):
            if (cx + dx, cz + dz) in self.chunks:
                self._mark_dirty(cx + dx, cz + dz, urgent=False)

    def get_chunk(self, cx: int, cz: int):
        """Read-only lookup; does NOT generate a missing chunk (use ensure_chunk for that)."""
        return self.chunks.get((cx, cz))

    def ensure_chunk_loaded(self, cx: int, cz: int) -> Chunk:
        return self._get_or_create_chunk(cx, cz)

    def unload_chunk(self, cx: int, cz: int):
        key = (cx, cz)
        if key in self.chunks:
            chunk = self.chunks[key]
            # Only touch the disk if this chunk actually differs from what is
            # already there. Unload used to write unconditionally, which made
            # lowering the render distance a disaster: dropping 32 -> 6 unloads
            # ~3000 chunks and so fired ~3000 synchronous savez_compressed
            # calls, nearly all of them rewriting bytes identical to the ones
            # already on disk. Same reasoning as save_all_loaded_chunks.
            if self.save_dir is not None and chunk.needs_save:
                from save import world_save
                world_save.save_chunk(self.save_dir, chunk)
            del self.chunks[key]
        # Pending wakeups inside a chunk we no longer have block data for would
        # fire against nothing. Drop them here rather than filtering on pop, so
        # the same position can be scheduled again cleanly if the chunk comes
        # back.
        self.ticks.cancel_in_chunk(cx, cz, CX, CZ)
        self._dirty_urgent_set.discard(key)
        self._dirty_bulk_set.discard(key)
        self._pending_gen_set.discard(key)
        self._pending_trees_set.discard(key)

    def _mark_dirty(self, cx: int, cz: int, urgent: bool = True):
        """
        Queues a chunk's mesh for rebuilding.

        `urgent` defaults to True so every player-facing edit path
        (break/place/door/stairs/fence) gets priority without each one having
        to remember to ask for it. The only bulk callers - chunk generation
        and tree growth - opt out explicitly with urgent=False. The default
        deliberately fails safe toward "responsive": a bulk caller that
        forgets to opt out merely costs a few wasted priority slots, whereas
        a player edit that forgets to opt IN is invisible for half a minute,
        which is exactly the bug this split exists to kill.
        """
        key = (cx, cz)
        if urgent:
            if key in self._dirty_urgent_set:
                return
            self._dirty_urgent_set.add(key)
            self._dirty_urgent_queue.append(key)
            # any bulk entry for this chunk is now redundant - dropping it
            # from the set leaves the queued copy stale, to be skipped on pop
            self._dirty_bulk_set.discard(key)
        else:
            if key in self._dirty_bulk_set or key in self._dirty_urgent_set:
                return
            self._dirty_bulk_set.add(key)
            self._dirty_bulk_queue.append(key)

    def pop_dirty_chunks(self, max_count: int):
        """Returns up to max_count chunk coords whose mesh needs rebuilding,
        removing them from the queue. Player edits come out first, then
        streaming chunks (closest-first, see __init__)."""
        result = []
        while self._dirty_urgent_queue and len(result) < max_count:
            key = self._dirty_urgent_queue.popleft()
            self._dirty_urgent_set.discard(key)
            if key in self.chunks:  # could have been unloaded since queued
                result.append(key)
        while self._dirty_bulk_queue and len(result) < max_count:
            key = self._dirty_bulk_queue.popleft()
            if key not in self._dirty_bulk_set:
                continue  # stale: promoted to urgent, or unloaded
            self._dirty_bulk_set.discard(key)
            if key in self.chunks:
                result.append(key)
        return result

    def _queue_generation(self, cx: int, cz: int):
        key = (cx, cz)
        if key in self.chunks or key in self._pending_gen_set:
            return
        self._pending_gen_set.add(key)
        self._pending_gen_queue.append(key)

    def _queue_tree_pass(self, cx: int, cz: int):
        key = (cx, cz)
        chunk = self.chunks.get(key)
        if chunk is None or chunk.trees_generated or key in self._pending_trees_set:
            return
        self._pending_trees_set.add(key)
        self._pending_trees_queue.append(key)

    def process_generation_budget(self, gen_budget: int = None, tree_budget: int = None):
        """
        Generates a limited number of newly-queued chunks this frame, and
        grows trees for a limited number of chunks whose full 3x3
        neighborhood is now ready. Call this once per frame; spreads the
        cost of crossing a chunk boundary across several frames instead of
        one large stall. Profiling showed tree generation (many individual
        set_block calls per tree) was actually the dominant cost here, not
        terrain generation itself - so it gets its own budget rather than
        running unbounded over the whole pending-trees queue.
        """
        gen_budget = config.CHUNK_GEN_BUDGET_PER_FRAME if gen_budget is None else gen_budget
        tree_budget = config.CHUNK_TREE_BUDGET_PER_FRAME if tree_budget is None else tree_budget

        generated = 0
        newly_generated = []
        while self._pending_gen_queue and generated < gen_budget:
            key = self._pending_gen_queue.pop(0)
            self._pending_gen_set.discard(key)
            if key in self.chunks:
                continue
            cx, cz = key
            self._get_or_create_chunk(cx, cz)
            newly_generated.append(key)
            generated += 1

        # Bug fix: _queue_tree_pass() only enqueues a chunk once it already
        # exists in self.chunks. update_streaming() calls _queue_tree_pass()
        # for every "needed" chunk, but on the very first call none of those
        # chunks exist yet (they've only just been queued for generation
        # above/on future frames) - so every one of those requests was
        # silently dropped and chunks could sit fully terrain-generated but
        # tree-less until some LATER update_streaming() call happened to
        # re-request them. In normal gameplay that "later call" arrives next
        # frame (streaming runs every frame), which is what caused trees to
        # visibly pop in a moment after chunks appeared; during a one-shot
        # bulk load it could mean trees never grow at all. Re-queuing tree
        # passes for chunks that just finished generating - and for their
        # already-loaded neighbors, since a chunk finishing generation may
        # be the missing piece of a neighbor's 3x3 neighborhood - closes
        # that gap without relying on the caller to re-invoke update_streaming.
        for (cx, cz) in newly_generated:
            for ddx in (-1, 0, 1):
                for ddz in (-1, 0, 1):
                    ncx, ncz = cx + ddx, cz + ddz
                    if (ncx, ncz) in self.chunks:
                        self._queue_tree_pass(ncx, ncz)

        # tree growth: only for chunks whose full 3x3 neighborhood has terrain,
        # capped at tree_budget chunks processed per call
        trees_done = 0
        still_pending = []
        for key in self._pending_trees_queue:
            cx, cz = key
            if key not in self._pending_trees_set:
                continue  # already handled or unloaded
            if trees_done >= tree_budget:
                still_pending.append(key)
                continue
            neighborhood_ready = all(
                (cx + ddx, cz + ddz) in self.chunks
                for ddx in (-1, 0, 1) for ddz in (-1, 0, 1)
            )
            if neighborhood_ready:
                self._pending_trees_set.discard(key)
                self.generate_trees_for_chunk(cx, cz)
                trees_done += 1
            else:
                still_pending.append(key)
        self._pending_trees_queue = still_pending

    # -- tree placement (needs neighbor awareness for leaf spill) ------------

    def generate_trees_for_chunk(self, cx: int, cz: int):
        """
        Grows trees for a chunk once its immediate neighbors' terrain exists,
        so leaf blocks that spill into a neighboring chunk land on already-
        generated terrain rather than on empty/ungenerated blocks.
        """
        chunk = self.chunks.get((cx, cz))
        if chunk is None or chunk.trees_generated:
            return
        chunk.trees_generated = True

        spots = worldgen.tree_columns_for_chunk(chunk, self.noise)
        for lx, base_y, lz, trunk_height in spots:
            ox, oz = chunk.world_origin()
            wx, wz = ox + lx, oz + lz
            for dx, dy, dz, block_id in worldgen.tree_block_offsets(trunk_height):
                # urgent=False: this is bulk world generation, not a player
                # edit. Marking it urgent would put thousands of streaming
                # chunks into the priority queue and re-create the starvation
                # the split was built to prevent (see _mark_dirty).
                #
                # update_neighbors=False for the same reason, and it matters
                # more: worldgen places millions of blocks, and notifying six
                # neighbours per block would fire the reactive layer across the
                # whole world as it generates. Minecraft suppresses neighbour
                # updates during generation identically (its setBlock flag 2).
                # Water and sand placed by the generator are expected to be
                # laid down already settled, not to flow into place.
                self.set_block(wx + dx, base_y + dy, wz + dz, block_id, mark_dirty=True,
                               overwrite_leaves_only=(block_id == Block.LEAVES),
                               urgent=False, update_neighbors=False)

    # -- ticking -------------------------------------------------------------

    def tick(self):
        """
        Advances simulated time by one 1/20 s game tick and runs every block
        wakeup that has come due.

        Must be called on a fixed timestep, never once per frame - see
        world/ticks.py. Blocks at rest cost nothing here; the queue only holds
        blocks mid-change.
        """
        self.ticks.advance()
        for (pos, block_id) in self.ticks.pop_due(config.MAX_BLOCK_TICKS_PER_TICK):
            wx, wy, wz = pos
            cx, cz = world_to_chunk_coords(wx, wz)
            chunk = self.chunks.get((cx, cz))
            if chunk is None:
                continue  # unloaded between scheduling and firing
            # The world moved on since this was scheduled: the player broke the
            # block, or a fluid already reclaimed the cell. Minecraft re-checks
            # the same way rather than trusting the queue.
            if chunk.get_local(wx % CX, wy, wz % CZ) != block_id:
                continue
            handler = block_behavior.ON_TICK.get(block_id)
            if handler is not None:
                handler(self, wx, wy, wz)

    def _notify_neighbors(self, wx: int, wy: int, wz: int):
        """
        Tells the six orthogonal neighbours of a changed block that something
        next to them moved, so they can decide whether to schedule themselves a
        wakeup. This is Minecraft's updateNeighbors.

        Deliberately does NOT generate missing chunks: a neighbour whose chunk
        is not resident cannot react anyway, and calling through get_block here
        would let a single block edit at a chunk border pull a chunk off disk
        mid-frame.
        """
        for (dx, dy, dz) in _NEIGHBOR_OFFSETS:
            nx, ny, nz = wx + dx, wy + dy, wz + dz
            if ny < 0 or ny >= CH:
                continue
            chunk = self.chunks.get(world_to_chunk_coords(nx, nz))
            if chunk is None:
                continue
            block_id = chunk.get_local(nx % CX, ny, nz % CZ)
            handler = block_behavior.ON_NEIGHBOR_UPDATE.get(block_id)
            if handler is not None:
                handler(self, nx, ny, nz)

    # -- block access (world-space coordinates) ------------------------------

    def get_block(self, wx: int, wy: int, wz: int) -> int:
        if wy < 0 or wy >= CH:
            return Block.AIR
        cx, cz = world_to_chunk_coords(wx, wz)
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            # Not loaded in memory. If this world is save-backed and the chunk
            # was previously generated/edited, load it now rather than
            # silently reporting AIR - otherwise a block query issued before
            # update_streaming() has run for this area (e.g. right after
            # loading a saved world) would incorrectly see empty space where
            # persisted terrain/edits actually exist.
            if self.save_dir is not None:
                from save import world_save
                if world_save.chunk_save_exists(self.save_dir, cx, cz):
                    chunk = self._get_or_create_chunk(cx, cz)
                else:
                    return Block.AIR
            else:
                return Block.AIR
        lx, lz = world_to_local(wx, wz)
        return chunk.get_local(lx, wy, lz)

    def get_block_meta(self, wx: int, wy: int, wz: int) -> int:
        """Returns the per-block metadata byte (see Chunk.meta) - currently
        only meaningful for doors (facing + open/closed bits, see
        world/doors.py). Returns 0 for anything else / unloaded chunks."""
        if wy < 0 or wy >= CH:
            return 0
        cx, cz = world_to_chunk_coords(wx, wz)
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            return 0
        lx, lz = world_to_local(wx, wz)
        return chunk.get_local_meta(lx, wy, lz)

    def set_block_meta(self, wx: int, wy: int, wz: int, meta_value: int):
        if wy < 0 or wy >= CH:
            return
        cx, cz = world_to_chunk_coords(wx, wz)
        chunk = self._get_or_create_chunk(cx, cz)
        lx, lz = world_to_local(wx, wz)
        chunk.set_local_meta(lx, wy, lz, meta_value)
        # Metadata-only changes are real changes: a door opening, a stair
        # re-shaping against a new neighbour, a fence connecting. None of those
        # touch the block id, so without this line they would be invisible to
        # everyone but the player who caused them.
        if self.change_listener is not None:
            self.change_listener(wx, wy, wz, chunk.get_local(lx, wy, lz), meta_value)
        self._mark_dirty(cx, cz)

    def set_block(self, wx: int, wy: int, wz: int, block_id: int, mark_dirty: bool = True,
                  overwrite_leaves_only: bool = False, meta_value: int = None,
                  urgent: bool = True, update_neighbors: bool = True):
        if wy < 0 or wy >= CH:
            return
        cx, cz = world_to_chunk_coords(wx, wz)
        chunk = self._get_or_create_chunk(cx, cz)
        lx, lz = world_to_local(wx, wz)

        if overwrite_leaves_only:
            # trees placing leaves shouldn't punch through solid terrain/other logs
            existing = chunk.get_local(lx, wy, lz)
            if existing != Block.AIR:
                return

        chunk.set_local(lx, wy, lz, block_id)
        if meta_value is not None:
            chunk.set_local_meta(lx, wy, lz, meta_value)

        if self.change_listener is not None:
            # Reads the meta back out rather than passing meta_value through:
            # meta_value is None for most callers, and set_local has just zeroed
            # the byte if the id changed. What the cell actually holds now is the
            # only thing a client can be told without guessing.
            self.change_listener(wx, wy, wz, block_id, chunk.get_local_meta(lx, wy, lz))

        if mark_dirty:
            self._mark_dirty(cx, cz, urgent=urgent)
            # also dirty neighbors if we're on a chunk edge, so seam faces update
            if lx == 0:
                self._mark_dirty(cx - 1, cz, urgent=urgent)
            elif lx == CX - 1:
                self._mark_dirty(cx + 1, cz, urgent=urgent)
            if lz == 0:
                self._mark_dirty(cx, cz - 1, urgent=urgent)
            elif lz == CZ - 1:
                self._mark_dirty(cx, cz + 1, urgent=urgent)

        if update_neighbors:
            # The block that just landed reacts to its own placement first
            # (Minecraft's onBlockAdded: sand placed in midair schedules its own
            # fall), then the six around it are told something moved.
            own = block_behavior.ON_NEIGHBOR_UPDATE.get(block_id)
            if own is not None:
                own(self, wx, wy, wz)
            self._notify_neighbors(wx, wy, wz)

    def toggle_door(self, wx: int, wy: int, wz: int) -> bool:
        """Flips a door between open/closed, updating BOTH halves (top and
        bottom) together so they always stay in sync - real Minecraft doors
        are two blocks tall sharing one open/closed state. Returns True if
        a door was actually there and got toggled."""
        if self.get_block(wx, wy, wz) != Block.DOOR:
            return False
        from world.doors import unpack_door_meta, pack_door_meta
        facing, is_open, is_top, hinge = unpack_door_meta(self.get_block_meta(wx, wy, wz))
        new_open = not is_open

        this_y = wy
        other_y = wy - 1 if is_top else wy + 1

        self.set_block_meta(wx, this_y, wz, pack_door_meta(facing, new_open, is_top, hinge))
        if self.get_block(wx, other_y, wz) == Block.DOOR:
            other_facing, _, other_is_top, other_hinge = unpack_door_meta(self.get_block_meta(wx, other_y, wz))
            self.set_block_meta(wx, other_y, wz, pack_door_meta(other_facing, new_open, other_is_top, other_hinge))
        return True

    def place_door(self, wx: int, wy: int, wz: int, facing: int, player_yaw: float = 0.0) -> bool:
        """
        Places a full two-block-tall door: a bottom half at (wx,wy,wz) and a
        top half directly above it at (wx,wy+1,wz), both starting closed and
        sharing the given facing. Requires BOTH cells to be empty air first
        (matching vanilla Minecraft, which refuses to place a door if there
        isn't a full 2-block-tall gap) - otherwise nothing is placed and
        this returns False, so the caller (PlayerControls.place_block)
        knows not to consume the item from the player's inventory.

        Hinge side is auto-selected per vanilla's actual placement rules
        (see world.doors.choose_hinge_for_placement) - checking adjacent
        doors/walls BEFORE writing this door's own blocks, so it correctly
        sees its neighbors as they currently are.
        """
        from world.doors import pack_door_meta, choose_hinge_for_placement
        if self.get_block(wx, wy, wz) != Block.AIR or self.get_block(wx, wy + 1, wz) != Block.AIR:
            return False
        hinge = choose_hinge_for_placement(self, wx, wy, wz, facing, player_yaw)
        self.set_block(wx, wy, wz, Block.DOOR, meta_value=pack_door_meta(facing, is_open=False, is_top=False, hinge=hinge))
        self.set_block(wx, wy + 1, wz, Block.DOOR, meta_value=pack_door_meta(facing, is_open=False, is_top=True, hinge=hinge))
        return True

    def break_door(self, wx: int, wy: int, wz: int):
        """
        Breaks a door, removing BOTH halves together regardless of which
        half was actually targeted - a lone door half floating in midair
        would be a broken, un-toggleable state (no partner block to keep
        it in sync with), so vanilla Minecraft always removes the whole
        door as one unit too. Returns Block.DOOR if a door was there (for
        the caller's drop-an-item logic), or None if there was nothing to
        break.
        """
        if self.get_block(wx, wy, wz) != Block.DOOR:
            return None
        from world.doors import unpack_door_meta
        _, _, is_top, _ = unpack_door_meta(self.get_block_meta(wx, wy, wz))
        bottom_y = wy - 1 if is_top else wy
        top_y = bottom_y + 1
        self.set_block(wx, bottom_y, wz, Block.AIR)
        if self.get_block(wx, top_y, wz) == Block.DOOR:
            self.set_block(wx, top_y, wz, Block.AIR)
        return Block.DOOR

    def break_block(self, wx: int, wy: int, wz: int):
        block_id = self.get_block(wx, wy, wz)
        if block_id == Block.AIR:
            return None
        if block_id == Block.DOOR:
            result = self.break_door(wx, wy, wz)
        else:
            self.set_block(wx, wy, wz, Block.AIR)
            result = block_id
        self.update_stair_neighbors(wx, wy, wz)
        self.update_fence_connections(wx, wy, wz)
        return result

    def place_block(self, wx: int, wy: int, wz: int, block_id: int) -> bool:
        if self.get_block(wx, wy, wz) not in REPLACEABLE_BLOCKS:
            return False
        self.set_block(wx, wy, wz, block_id)
        if block_id == Block.FENCE:
            self.update_fence_connections(wx, wy, wz)
        return True

    def place_stairs(self, wx: int, wy: int, wz: int, block_id: int, facing: int, is_top: bool) -> bool:
        if self.get_block(wx, wy, wz) != Block.AIR:
            return False
        from world.stairs import pack_stair_meta, compute_shape
        self.set_block(wx, wy, wz, block_id, meta_value=pack_stair_meta(facing, is_top))
        shape = compute_shape(self, wx, wy, wz, facing, is_top)
        self.set_block_meta(wx, wy, wz, pack_stair_meta(facing, is_top, shape))
        self.update_stair_neighbors(wx, wy, wz)
        return True

    def update_stair_neighbors(self, wx, wy, wz):
        from world.stairs import unpack_stair_meta, compute_shape, pack_stair_meta, neighbors_to_update
        for (nx, ny, nz) in neighbors_to_update(wx, wy, wz):
            if self.get_block(nx, ny, nz) not in (Block.STAIRS_WOOD, Block.STAIRS_STONE):
                continue
            facing, is_top, old_shape = unpack_stair_meta(self.get_block_meta(nx, ny, nz))
            new_shape = compute_shape(self, nx, ny, nz, facing, is_top)
            if new_shape != old_shape:
                self.set_block_meta(nx, ny, nz, pack_stair_meta(facing, is_top, new_shape))

    def update_fence_connections(self, wx, wy, wz):
        from world.fences import compute_connections, neighbors_to_update
        if self.get_block(wx, wy, wz) == Block.FENCE:
            self.set_block_meta(wx, wy, wz, compute_connections(self, wx, wy, wz))
        for (nx, ny, nz) in neighbors_to_update(wx, wy, wz):
            if self.get_block(nx, ny, nz) == Block.FENCE:
                self.set_block_meta(nx, ny, nz, compute_connections(self, nx, ny, nz))

    def get_height_at(self, wx: int, wz: int) -> int:
        cx, cz = world_to_chunk_coords(wx, wz)
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            chunk = self._get_or_create_chunk(cx, cz)
        lx, lz = world_to_local(wx, wz)
        return int(chunk.height_map[lx, lz])

    def get_ground_height_at(self, wx: int, wz: int) -> int:
        """
        Like get_height_at, but skips over tree blocks (log/leaves) to find
        the actual terrain surface underneath. height_map tracks the
        topmost solid block in the column, which becomes the top of a
        tree's canopy once one grows there - using that directly for player
        spawn placement put the player on/above a tree instead of on the
        ground (or, if a stale/mid-growth height_map value briefly pointed
        below the real surface, dropped them through open air next to it).
        This scans down from the column top past any WOOD_LOG/LEAVES until
        it hits solid ground, matching what "spawn on the surface" actually
        means regardless of whether a tree happens to be standing there.
        """
        cx, cz = world_to_chunk_coords(wx, wz)
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            chunk = self._get_or_create_chunk(cx, cz)
        lx, lz = world_to_local(wx, wz)

        y = int(chunk.height_map[lx, lz])
        y = min(y, CH - 1)
        while y >= 0:
            block_id = chunk.get_local(lx, y, lz)
            if block_id not in (Block.AIR, Block.WOOD_LOG, Block.LEAVES):
                return y
            y -= 1
        return 0  # fallback: no solid ground found (shouldn't normally happen)

    def _padded_from(self, cx: int, cz: int, view_name: str):
        """
        Shared implementation behind get_padded_blocks_for_chunk and
        get_padded_meta_for_chunk: returns a (CH, CZ+2, CX+2) array holding this
        chunk's own data in the interior and a 1-cell border pulled from the
        eight surrounding chunks (left as 0 where a neighbor isn't resident).

        `view_name` is the Chunk method giving the (CH, CZ, CX) view to copy -
        "blocks_yzx" or "meta_yzx". Blocks and metadata MUST be padded by the
        same code: the water mesher reads a cell's id and its level together and
        would produce nonsense from a border where one was pulled from the
        neighbour and the other wasn't.

        The four DIAGONAL corners matter even though the cube mesher never looks
        at them. A water corner's height averages the four cells around it, so a
        chunk corner whose diagonal neighbour is missing reads that cell as air
        and the sea gets a dimple. One cell each, so they are free.
        """
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            return None

        def view(c):
            return getattr(c, view_name)()

        padded = np.zeros((CH, CZ + 2, CX + 2), dtype=np.uint8)
        padded[:, 1:-1, 1:-1] = view(chunk)

        west = self.chunks.get((cx - 1, cz))
        if west is not None:
            padded[:, 1:-1, 0] = view(west)[:, :, CX - 1]
        east = self.chunks.get((cx + 1, cz))
        if east is not None:
            padded[:, 1:-1, -1] = view(east)[:, :, 0]
        north = self.chunks.get((cx, cz - 1))
        if north is not None:
            padded[:, 0, 1:-1] = view(north)[:, CZ - 1, :]
        south = self.chunks.get((cx, cz + 1))
        if south is not None:
            padded[:, -1, 1:-1] = view(south)[:, 0, :]

        north_west = self.chunks.get((cx - 1, cz - 1))
        if north_west is not None:
            padded[:, 0, 0] = view(north_west)[:, CZ - 1, CX - 1]
        north_east = self.chunks.get((cx + 1, cz - 1))
        if north_east is not None:
            padded[:, 0, -1] = view(north_east)[:, CZ - 1, 0]
        south_west = self.chunks.get((cx - 1, cz + 1))
        if south_west is not None:
            padded[:, -1, 0] = view(south_west)[:, 0, CX - 1]
        south_east = self.chunks.get((cx + 1, cz + 1))
        if south_east is not None:
            padded[:, -1, -1] = view(south_east)[:, 0, 0]

        return padded

    def get_padded_blocks_for_chunk(self, cx: int, cz: int):
        """
        Returns a (CH, CZ+2, CX+2) array: this chunk's own blocks in the
        interior, with a 1-block-wide border pulled from the adjacent chunks
        (or left as air/0 if a neighbor isn't loaded). Used by the vectorized
        mesh builders so faces at chunk borders correctly cull against real
        neighbor data instead of guessing.
        """
        return self._padded_from(cx, cz, "blocks_yzx")

    def get_padded_meta_for_chunk(self, cx: int, cz: int):
        """Same shape/layout as get_padded_blocks_for_chunk, over the metadata
        array - the water mesher needs each cell's flow level, including the
        border cells, to compute corner heights that agree across a chunk seam."""
        return self._padded_from(cx, cz, "meta_yzx")

    # -- streaming: load/unload chunks around the player ---------------------

    def update_streaming(self, player_wx: float, player_wz: float, render_distance: int):
        """
        Queues every chunk within render_distance (in chunk units) of the
        player for generation (closest first) and unloads chunks that have
        drifted too far away. Does NOT generate chunks itself - actual
        terrain generation happens incrementally in process_generation_budget(),
        called once per frame with a small budget, so crossing a chunk
        boundary spreads its cost across several frames instead of causing
        a single large stall.
        """
        pcx, pcz = world_to_chunk_coords(int(player_wx), int(player_wz))

        needed = set()
        for dx in range(-render_distance, render_distance + 1):
            for dz in range(-render_distance, render_distance + 1):
                if dx * dx + dz * dz <= render_distance * render_distance:
                    needed.add((pcx + dx, pcz + dz))

        # halo so tree generation always has full neighbor data at the border
        halo_needed = set()
        for (cx, cz) in needed:
            for ddx in (-1, 0, 1):
                for ddz in (-1, 0, 1):
                    halo_needed.add((cx + ddx, cz + ddz))

        # queue by distance to player so the closest missing chunks generate first
        missing = [key for key in halo_needed if key not in self.chunks and key not in self._pending_gen_set]
        missing.sort(key=lambda k: (k[0] - pcx) ** 2 + (k[1] - pcz) ** 2)
        for (cx, cz) in missing:
            self._queue_generation(cx, cz)

        for (cx, cz) in needed:
            self._queue_tree_pass(cx, cz)

        # unload chunks well outside render distance + margin
        unload_radius = render_distance + config.CHUNK_UNLOAD_MARGIN
        to_unload = []
        for (cx, cz) in self.chunks.keys():
            if (cx - pcx) ** 2 + (cz - pcz) ** 2 > unload_radius ** 2:
                to_unload.append((cx, cz))
        for key in to_unload:
            self.unload_chunk(*key)
        self.recently_unloaded.extend(to_unload)

        return needed

    def pop_recently_unloaded(self):
        """Returns and clears the list of chunk coords unloaded since the last
        call - callers (e.g. the renderer) use this to free per-chunk GPU
        resources (meshes) that would otherwise leak and keep rendering
        chunks that are no longer supposed to be within render distance."""
        result = self.recently_unloaded
        self.recently_unloaded = []
        return result
