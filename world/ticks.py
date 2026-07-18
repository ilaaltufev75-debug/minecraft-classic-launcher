"""
world/ticks.py
Scheduled block ticks - what Minecraft's code calls "TileTicks".

This is the clock everything reactive in the world hangs off. Fluids and
falling blocks do not act the moment they are placed; they ask to be woken up
a fixed number of GAME TICKS later, and act then. Water in the Overworld
spreads one step every 5 ticks (0.25 s), sand starts falling 2 ticks after it
loses its support. Those numbers are the behaviour - get the clock wrong and
nothing downstream can be right no matter how faithful the spreading rule is.

Two properties matter and both are easy to get wrong:

1. A tick is 1/20 s of SIMULATED time, not one frame. Driving fluids from the
   frame loop would make water flow at a speed that depends on the player's
   frame rate - faster on a good GPU, slower during a chunk load spike. The
   caller is responsible for feeding this a fixed timestep (see
   config.TICK_SECONDS and Game._update_in_game's accumulator).

2. Ordering must be deterministic. Two blocks scheduled for the same tick have
   to run in a defined order or the same world state can evolve differently
   between runs. Ties break on priority, then on insertion order - the same
   rule Minecraft uses.

Nothing polls here. A block that is at rest schedules nothing and costs
nothing, which is why this stays cheap even with ~3200 chunks resident: the
queue only ever holds blocks that are actively mid-change. That is also why
there is no "simulation distance" cutoff - there is nothing to cut off.
"""

import heapq


class TickScheduler:
    """
    A priority queue of (position, block_id) wakeups keyed by game tick.

    Deduplicated: asking twice for the same (position, block_id) before it
    fires is a no-op, exactly like Minecraft's isTickScheduled check. Without
    that, a block surrounded by six changing neighbours would queue six
    identical wakeups and run its logic six times in one tick.
    """

    __slots__ = ("current_tick", "_heap", "_scheduled", "_seq")

    def __init__(self):
        self.current_tick = 0
        # (due_tick, priority, seq, pos, block_id); seq keeps ties FIFO and
        # also stops heapq from ever comparing the tuples' payloads
        self._heap = []
        # (pos, block_id) -> due_tick. This dict, not the heap, is the
        # authority on what is actually pending: cancelling walks it instead of
        # rebuilding the heap, and stale heap entries are recognised on pop by
        # their due_tick no longer matching.
        self._scheduled = {}
        self._seq = 0

    def __len__(self):
        return len(self._scheduled)

    def schedule(self, pos, block_id: int, delay: int, priority: int = 0) -> bool:
        """
        Wakes (pos, block_id) up `delay` ticks from now. Returns False if it was
        already pending - the existing wakeup stands rather than being pushed
        back, so a block under continuous disturbance still fires on time
        instead of being starved by its own neighbours.
        """
        key = (pos, block_id)
        if key in self._scheduled:
            return False
        due = self.current_tick + max(1, int(delay))
        self._scheduled[key] = due
        self._seq += 1
        heapq.heappush(self._heap, (due, priority, self._seq, pos, block_id))
        return True

    def is_scheduled(self, pos, block_id: int) -> bool:
        return (pos, block_id) in self._scheduled

    def advance(self):
        self.current_tick += 1

    def pop_due(self, max_count: int):
        """Returns the (pos, block_id) wakeups due at or before the current
        tick, removing them. Caller must re-verify the block is still what was
        scheduled - the world may have moved on since."""
        due = []
        while self._heap and len(due) < max_count and self._heap[0][0] <= self.current_tick:
            due_tick, _priority, _seq, pos, block_id = heapq.heappop(self._heap)
            key = (pos, block_id)
            want = self._scheduled.get(key)
            if want is None or want != due_tick:
                # cancelled (its chunk unloaded), or superseded by a later
                # schedule after a cancel - either way this entry is stale
                continue
            del self._scheduled[key]
            due.append((pos, block_id))
        return due

    def cancel_in_chunk(self, cx: int, cz: int, chunk_size_x: int, chunk_size_z: int):
        """
        Drops every pending wakeup inside a chunk that is being unloaded.

        Only the dict is walked; the matching heap entries are left to decay
        and are skipped on pop. Rebuilding the heap here would be O(n log n) on
        every unload, and lowering the render distance unloads ~3000 chunks in
        one frame.

        NOTE (known gap vs Minecraft): Minecraft persists pending ticks into
        the chunk's save data, so a river frozen mid-flow resumes flowing when
        you walk back. Here they are dropped, so fluid that was still spreading
        when its chunk unloaded stays where it stopped until something
        disturbs it again. Fine while nothing generates flowing water far from
        the player; revisit when rivers land.
        """
        if not self._scheduled:
            return
        doomed = [
            key for key in self._scheduled
            if key[0][0] // chunk_size_x == cx and key[0][2] // chunk_size_z == cz
        ]
        for key in doomed:
            del self._scheduled[key]
