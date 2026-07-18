"""
world/falling.py
Gravity-affected blocks. Sand, for now.

Minecraft's rule, reproduced exactly:

  - A falling block reacts to a NEIGHBOUR UPDATE, not to a poll. Nothing scans
    the world looking for unsupported sand; sand only ever checks itself when
    something next to it changed, or when it was just placed.
  - It does not fall immediately. It schedules itself 2 ticks out
    (BlockFalling.getFallDelay) and falls when that fires. This delay is why
    you can watch a sand column hang for an instant after you mine its base,
    and it is why breaking the bottom of a tall column collapses it in a visible
    ripple from the bottom up rather than all at once.
  - It falls into air OR fluid, replacing the fluid. Sand dropped into the sea
    sinks to the seabed and takes the water block's place, exactly as in MC.
  - Sand that loses support while its chunk is unloaded stays hanging until
    something disturbs it again. Also vanilla behaviour, and the reason floating
    sand exists in every Minecraft world ever generated.

KNOWN GAP vs Minecraft, and it is a real one: vanilla converts the block into a
FallingBlock ENTITY that accelerates downward (0.04 b/t^2, terminal 3.92 b/t)
and is drawn mid-air, so a long drop is smooth and speeds up. This moves the
block one cell per FALL_DELAY_TICKS instead - a constant 10 blocks/second, with
no airborne block to look at. Over the two or three blocks a normal cave-in
covers, the difference is hard to see; down a 40-block shaft it is obvious, both
because it is slower and because it steps. Closing it needs an entity list and
something that can draw a single textured cube at an arbitrary float position,
neither of which exists yet - so it is flagged here rather than faked.
"""

from world import block_behavior
from world.blocks import Block

# BlockFalling.getFallDelay() in vanilla. Not a tuning knob: it is what makes a
# collapse read as a collapse instead of a teleport.
FALL_DELAY_TICKS = 2

# Blocks that obey gravity. Gravel/concrete powder would just be added here.
FALLING_BLOCKS = (Block.SAND,)

# What a falling block may move into. Air, and fluids - which it displaces.
_PASSABLE = frozenset((Block.AIR, Block.WATER))


def _register(block_id):
    @block_behavior.on_neighbor_update(block_id)
    def _neighbor_update(world, wx, wy, wz, _bid=block_id):
        # Only ever asks for a wakeup - never falls here. Doing the move
        # directly in the neighbour callback would make a column collapse
        # instantly and recursively in a single frame, in the middle of whatever
        # set_block started it, and would look nothing like Minecraft even
        # though it would end in the same place.
        world.ticks.schedule((wx, wy, wz), _bid, FALL_DELAY_TICKS)

    @block_behavior.on_tick(block_id)
    def _tick(world, wx, wy, wz, _bid=block_id):
        if wy <= 0:
            return
        if world.get_block(wx, wy - 1, wz) not in _PASSABLE:
            return
        # Order matters: clear the old cell first, then write the new one. Each
        # set_block notifies its own neighbours, which is what wakes the sand
        # ABOVE this one (its floor just vanished) and lets a column come down
        # one block at a time from the bottom - the vanilla ripple. Writing the
        # new cell first would let the block above see solid ground underneath
        # and settle.
        world.set_block(wx, wy, wz, Block.AIR)
        world.set_block(wx, wy - 1, wz, _bid)


for _bid in FALLING_BLOCKS:
    _register(_bid)
