"""
world/block_behavior.py
Registry of per-block reactive behaviour: what a block does when it is woken
by a scheduled tick, and what it does when one of its six neighbours changes.

This mirrors the two halves of Minecraft's Block API that matter for fluids
and falling blocks:

  scheduledTick(pos)     -> ON_TICK here
  neighborChanged(pos)   -> ON_NEIGHBOR_UPDATE here

The split is the whole point and is worth being precise about. A neighbour
change never performs the action itself - it only asks the scheduler for a
wakeup some number of ticks later. That delay IS the observable behaviour:
water crawling outward a block at a time rather than snapping to its final
shape, sand hanging for a moment before it drops. Doing the work directly in
the neighbour callback would produce a world that settles instantly and looks
nothing like Minecraft, however correct the end state.

Kept as a plain registry rather than methods on a Block class so that
world/world.py can import it without importing the fluid or falling-block
modules, which themselves need World. The behaviour modules register
themselves on import; world.py imports them at the bottom of the file, after
World exists.
"""

# block_id -> fn(world, wx, wy, wz)
# Called when a scheduled tick for this block comes due. The block is
# guaranteed to still be this id and its chunk guaranteed loaded.
ON_TICK = {}

# block_id -> fn(world, wx, wy, wz)
# Called when any of the six orthogonal neighbours of this block changes.
# Should be cheap and should normally do nothing but schedule a tick.
ON_NEIGHBOR_UPDATE = {}


def on_tick(block_id: int):
    def _register(fn):
        ON_TICK[block_id] = fn
        return fn
    return _register


def on_neighbor_update(block_id: int):
    def _register(fn):
        ON_NEIGHBOR_UPDATE[block_id] = fn
        return fn
    return _register
