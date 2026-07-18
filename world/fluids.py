"""
world/fluids.py
Flowing water.

This is a direct port of Minecraft's BlockFluid/BlockFlowing rules, not an
approximation of them, because water's whole character lives in the exact
numbers: it spreads seven blocks and stops, it falls before it spreads, it
prefers the direction that leads to a hole even when the hole is four blocks
away and out of sight, and two source blocks either side of a gap fill that gap
with a third source. Get any one of those wrong and it stops reading as
Minecraft water however plausible the rest looks.

STATE
-----
A water cell's metadata byte holds:

    bits 0-2   level, 0..7. 0 is a SOURCE (full block). 7 is the thinnest
               film, one step from drying up.
    bit  3     the FALLING flag (value 8). Water that is falling from water
               directly above it. Falling water behaves as level 0 for
               height/spreading purposes but is not a source: cut off its
               supply and it vanishes rather than persisting.

That is vanilla's encoding, including the quirk that metadata >= 8 means
"falling, and treat the level as 0" rather than being a level of its own.

WHY IT IS EVENT DRIVEN
----------------------
Nothing polls. Water only acts when it is woken - either because a neighbour
changed (World._notify_neighbors -> ON_NEIGHBOR_UPDATE here) or because it
asked to be woken (TickScheduler). Water at rest costs exactly zero, which is
what makes an ocean of a few million cells free: the queue only ever holds the
handful of cells actually mid-flow. It also means worldgen can lay an entire
sea down instantly by writing the array directly, with no neighbour updates
(see World.set_block's update_neighbors flag) - the sea starts settled instead
of spending the first minute of a new world flowing into place.

THE FIVE-TICK DELAY IS THE BEHAVIOUR
------------------------------------
Water spreads one step every 5 ticks (0.25 s). Doing the work directly in the
neighbour callback would land in exactly the same final state, instantly, and
would look nothing like water - see the note at the top of block_behavior.py.

KNOWN GAPS vs Minecraft, both deliberate:
  - No separate "flowing" vs "stationary" block IDs. Vanilla swaps between them
    purely as an optimisation, to keep settled water out of the random-tick
    path. This engine has no random ticking to keep it out of, and the
    scheduler is already event-driven, so the distinction would buy nothing and
    would double the block IDs, the texture entries and the save format.
  - Water does not flow into unloaded chunks (see _block). Vanilla does the
    same; without the check a single flow at a render-distance edge would pull
    a chunk off disk mid-tick and could write water into terrain that was
    generated a microsecond earlier.
"""

import math

from world import block_behavior
from world.blocks import Block
from world.chunk import CX, CZ, CH

# BlockLiquid.tickRate() for water in the Overworld. Not a tuning knob.
WATER_TICK_RATE = 5

# Metadata layout.
LEVEL_MASK = 7
FALLING = 8
MAX_LEVEL = 7

# Cells water is allowed to move into. Vanilla asks "does the block's material
# block movement", and additionally hardcodes a few non-solid blocks (doors,
# signs, ladders) that DO stop water despite not blocking movement. Since
# everything in this world other than air and water stops water, asking the
# question the other way round is both simpler and safer: a block added later
# blocks water by default rather than being silently washed away because
# somebody set solid=False on it for the mesher's benefit (which is exactly what
# doors, fences and stairs all do).
_PASSABLE = (Block.AIR, Block.WATER)


# --- raw cell access ---------------------------------------------------------
#
# These deliberately do NOT go through World.get_block/set_block's chunk
# autoloading. A fluid tick touches up to a few hundred cells (see
# _optimal_flow_directions), and any one of them straying outside the resident
# set must read as "wall", not as an invitation to generate terrain.

def _block(world, x: int, y: int, z: int):
    """Block id at this cell, or None if it is outside the world vertically or
    inside a chunk that is not currently resident."""
    if y < 0 or y >= CH:
        return None
    chunk = world.chunks.get((x // CX, z // CZ))
    if chunk is None:
        return None
    return int(chunk.blocks[(x % CX) + (z % CZ) * CX + y * CX * CZ])


def _meta(world, x: int, y: int, z: int) -> int:
    chunk = world.chunks.get((x // CX, z // CZ))
    if chunk is None or y < 0 or y >= CH:
        return 0
    return int(chunk.meta[(x % CX) + (z % CZ) * CX + y * CX * CZ]) & 15


# --- fluid geometry ----------------------------------------------------------
#
# Vanilla answers "how tall is this water" THREE different ways depending on who
# is asking, and the differences are not rounding noise - they are load-bearing.
# All three are reproduced here rather than unified, because unifying them
# changes observable behaviour.

def render_height(meta: int) -> float:
    """
    Visual height, 0..1. RenderBlocks.getLiquidHeight -> a source renders at
    8/9 (~0.889), not 1.0, which is why you can see the horizon line across an
    ocean surface and why a shoreline's water visibly sits below the sand next
    to it.
    """
    level = 0 if (meta & FALLING) else (meta & LEVEL_MASK)
    return 1.0 - (level + 1) / 9.0


def collision_top(meta: int) -> float:
    """
    Top of the fluid for MOVEMENT/material tests, 0..1 within the cell.
    World.isAABBInMaterial: a source is a FULL cube here even though it RENDERS
    at 8/9.

    The disagreement is vanilla's and it is what produces the feel of floating:
    the water you are buoyed up by reaches a ninth of a block higher than the
    water you can see, so you bob with your eyes clear of a surface that is
    drawn below them.
    """
    if meta & FALLING:
        return 1.0
    return 1.0 - (meta & LEVEL_MASK) / 8.0


def submersion_top(meta: int) -> float:
    """
    Top of the fluid for Entity.isInsideOfMaterial, 0..1 - the test vanilla uses
    for "is this entity's head underwater", i.e. for drowning and for the
    underwater view. Its own third answer again: render height shifted back up
    by exactly the one ninth render_height took off, so a source reads as a full
    cube (1.0) and a level-1 flow as 8/9.
    """
    level = 0 if (meta & FALLING) else (meta & LEVEL_MASK)
    return 1.0 - ((level + 1) / 9.0 - 1.0 / 9.0)


# --- queries used by the player ---------------------------------------------

def is_head_in_water(world, x: float, eye_y: float, z: float) -> bool:
    """Entity.isInsideOfMaterial(Material.water): drives drowning and the
    underwater screen tint/fog."""
    bx, by, bz = math.floor(x), math.floor(eye_y), math.floor(z)
    if _block(world, bx, by, bz) != Block.WATER:
        return False
    return eye_y < by + submersion_top(_meta(world, bx, by, bz))


def is_body_in_water(world, x: float, feet_y: float, z: float,
                     half_width: float, height: float) -> bool:
    """
    Entity.isInWater(): drives the swimming physics.

    Vanilla tests the entity's box shrunk by 0.4 at BOTH ends of the Y axis
    (boundingBox.expand(0, -0.4, 0)), and that shrink is not incidental - it is
    what makes you stop counting as "in water" once your feet clear the surface,
    which is what makes you settle AT the surface rather than either sinking or
    launching out of the sea.
    """
    box_min_y = feet_y + 0.4
    box_max_y = feet_y + height - 0.4
    if box_max_y < box_min_y:
        box_max_y = box_min_y

    x0 = math.floor(x - half_width + 0.001)
    x1 = math.floor(x + half_width - 0.001)
    z0 = math.floor(z - half_width + 0.001)
    z1 = math.floor(z + half_width - 0.001)
    y0 = math.floor(box_min_y)
    y1 = math.floor(box_max_y)

    for by in range(y0, y1 + 1):
        for bx in range(x0, x1 + 1):
            for bz in range(z0, z1 + 1):
                if _block(world, bx, by, bz) != Block.WATER:
                    continue
                if by + collision_top(_meta(world, bx, by, bz)) >= box_min_y:
                    return True
    return False


# --- flow rules --------------------------------------------------------------

def _flow_decay(world, x: int, y: int, z: int) -> int:
    """Raw metadata of the water here, or -1 if this cell is not water."""
    if _block(world, x, y, z) != Block.WATER:
        return -1
    return _meta(world, x, y, z)


def _effective_decay(world, x: int, y: int, z: int) -> int:
    """Like _flow_decay but collapsing the falling flag: falling water counts as
    level 0."""
    decay = _flow_decay(world, x, y, z)
    if decay < 0:
        return -1
    if decay >= 8:
        return 0
    return decay


def _blocks_flow(world, x: int, y: int, z: int) -> bool:
    """Whether this cell stops water dead. Unloaded/out-of-world reads as a wall."""
    block_id = _block(world, x, y, z)
    if block_id is None:
        return True
    return block_id not in _PASSABLE


def _can_flow_into(world, x: int, y: int, z: int) -> bool:
    """Whether water may claim this cell. Water never displaces water - a cell
    that already holds any water is settled by the level rules instead."""
    return _block(world, x, y, z) == Block.AIR


def _flow_into(world, x: int, y: int, z: int, new_meta: int):
    if not _can_flow_into(world, x, y, z):
        return
    world.set_block(x, y, z, Block.WATER, meta_value=new_meta)


def _smallest_decay(world, x: int, y: int, z: int, current: int, sources: list) -> int:
    """
    BlockFluid.getSmallestFlowDecay. Folds one horizontal neighbour into the
    running minimum, and counts it if it is a source.

    Note it counts FALLING water as a source too (_effective_decay collapses it
    to 0). That is vanilla, and it is not a rounding error to tidy up: it is
    precisely why a two-block hole under a waterfall fills with real source
    water instead of a permanent trickle.
    """
    decay = _effective_decay(world, x, y, z)
    if decay < 0:
        return current
    if decay == 0:
        sources[0] += 1
    if current >= 0 and decay >= current:
        return current
    return decay


def _calculate_flow_cost(world, x: int, y: int, z: int, distance: int, from_dir: int) -> int:
    """
    BlockFluid.calculateFlowCost: how many steps from here to somewhere water
    could fall, searching up to 4 blocks out and never doubling back the way it
    came.

    This recursion is the reason Minecraft water looks intelligent. A pool that
    has one hole in it does not spread evenly and happen to find the hole - it
    heads STRAIGHT for the hole, because every cell on the way scores that
    direction cheapest. Replace this with "spread in all four directions
    equally" and water still eventually drains, but the thing everybody
    recognises as Minecraft water is gone.
    """
    cost = 1000
    for direction in range(4):
        # Never look back along the direction we arrived from.
        if ((direction == 0 and from_dir == 1) or (direction == 1 and from_dir == 0)
                or (direction == 2 and from_dir == 3) or (direction == 3 and from_dir == 2)):
            continue

        nx, ny, nz = x, y, z
        if direction == 0:
            nx -= 1
        elif direction == 1:
            nx += 1
        elif direction == 2:
            nz -= 1
        else:
            nz += 1

        if _blocks_flow(world, nx, ny, nz):
            continue
        if _block(world, nx, ny, nz) == Block.WATER and _meta(world, nx, ny, nz) == 0:
            continue  # a source already sits there; nothing to route through

        if not _blocks_flow(world, nx, ny - 1, nz):
            return distance  # found somewhere to fall

        if distance < 4:
            sub = _calculate_flow_cost(world, nx, ny, nz, distance + 1, direction)
            if sub < cost:
                cost = sub
    return cost


def _optimal_flow_directions(world, x: int, y: int, z: int):
    """
    BlockFluid.getOptimalFlowDirections. Returns [west, east, north, south]
    booleans: which of the four horizontal directions are tied for cheapest
    route to a drop. Ties spread to all of them, which is what makes water fan
    out symmetrically on flat ground.
    """
    costs = [1000, 1000, 1000, 1000]
    for direction in range(4):
        nx, ny, nz = x, y, z
        if direction == 0:
            nx -= 1
        elif direction == 1:
            nx += 1
        elif direction == 2:
            nz -= 1
        else:
            nz += 1

        if _blocks_flow(world, nx, ny, nz):
            continue
        if _block(world, nx, ny, nz) == Block.WATER and _meta(world, nx, ny, nz) == 0:
            continue

        if not _blocks_flow(world, nx, ny - 1, nz):
            costs[direction] = 0
        else:
            costs[direction] = _calculate_flow_cost(world, nx, ny, nz, 1, direction)

    best = min(costs)
    return [c == best for c in costs]


# --- registration ------------------------------------------------------------

@block_behavior.on_neighbor_update(Block.WATER)
def _water_neighbor_update(world, wx, wy, wz):
    """
    Something next to this water moved. Only ever asks for a wakeup - never
    flows here.

    This is also the entire fix for "break a block underwater and the hole stays
    empty". That hole is a neighbour change; the water around it is woken by it,
    and 5 ticks later the level rules refill the cell (and, with two or more
    sources adjacent and something solid underneath, promote it straight back to
    a source - vanilla's infinite water, which is what an ocean is made of).
    Before this existed water was an inert decorative block and nothing was
    listening at all.
    """
    world.ticks.schedule((wx, wy, wz), Block.WATER, WATER_TICK_RATE)


@block_behavior.on_tick(Block.WATER)
def _water_tick(world, wx, wy, wz):
    """BlockFluid.updateTick, water branch."""
    level = _meta(world, wx, wy, wz)

    if level > 0:
        # --- settle this cell's own level against its surroundings ---
        sources = [0]
        smallest = -100
        smallest = _smallest_decay(world, wx - 1, wy, wz, smallest, sources)
        smallest = _smallest_decay(world, wx + 1, wy, wz, smallest, sources)
        smallest = _smallest_decay(world, wx, wy, wz - 1, smallest, sources)
        smallest = _smallest_decay(world, wx, wy, wz + 1, smallest, sources)

        new_level = smallest + 1
        if new_level >= 8 or smallest < 0:
            new_level = -1  # nothing feeding this cell any more: dry up

        # Water directly above overrides everything: this cell is part of a
        # falling column, not a horizontal spread, and inherits the column's
        # falling state rather than decaying with distance.
        above = _flow_decay(world, wx, wy + 1, wz)
        if above >= 0:
            new_level = above if above >= 8 else above + 8

        # Infinite water. Two or more adjacent sources over something that can
        # hold them makes a third source. This one rule is why breaking a block
        # in the sea refills with real ocean rather than leaving a flowing scar.
        if sources[0] >= 2:
            below = _block(world, wx, wy - 1, wz)
            if below is not None and below not in _PASSABLE:
                new_level = 0
            elif below == Block.WATER and _meta(world, wx, wy - 1, wz) == 0:
                new_level = 0

        if new_level != level:
            level = new_level
            if level < 0:
                # Vanilla falls through to the flow section here with level -1,
                # which then writes water at level 7 into the cell below - a
                # transient that its own next tick immediately undoes. Returning
                # instead skips a wrong intermediate state nobody can see and
                # saves the wasted tick.
                world.set_block(wx, wy, wz, Block.AIR)
                return
            world.set_block(wx, wy, wz, Block.WATER, meta_value=level)
            world.ticks.schedule((wx, wy, wz), Block.WATER, WATER_TICK_RATE)

    # --- now move: down first, sideways only if down is refused ---
    if _can_flow_into(world, wx, wy - 1, wz):
        # Falling water keeps its falling meta all the way down the column, so a
        # 40-block waterfall is 40 identical cells rather than a gradient.
        _flow_into(world, wx, wy - 1, wz, level if level >= 8 else level + 8)
        return

    if level < 0:
        return
    if not (level == 0 or _blocks_flow(world, wx, wy - 1, wz)):
        return

    # Sideways. A falling column that lands on the floor restarts the horizontal
    # spread at level 1 rather than continuing from 8+, which is what gives a
    # waterfall its full seven-block pool at the bottom.
    spread_level = 1 if level >= 8 else level + 1
    if spread_level >= 8:
        return  # level 7 is the last step: water reaches exactly 7 blocks

    directions = _optimal_flow_directions(world, wx, wy, wz)
    if directions[0]:
        _flow_into(world, wx - 1, wy, wz, spread_level)
    if directions[1]:
        _flow_into(world, wx + 1, wy, wz, spread_level)
    if directions[2]:
        _flow_into(world, wx, wy, wz - 1, spread_level)
    if directions[3]:
        _flow_into(world, wx, wy, wz + 1, spread_level)
