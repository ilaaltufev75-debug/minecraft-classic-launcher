"""
_water_test.py
Throwaway harness for the fluid rules in world/fluids.py. Runs headless - no
window, no GL, no worldgen - by building a synthetic world out of a blank chunk
and stepping World.tick() by hand.

Scratch file, same as _wg_calib.py / _wg_check.py. Delete freely.
"""

import config
from world.blocks import Block
from world.chunk import Chunk, CX, CZ
from world.world import World

FAILS = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILS.append(name)


def blank_world():
    """A world of empty (all-air) chunks - no worldgen, so every test starts
    from a shape we chose rather than from terrain."""
    world = World(seed=1, save_dir=None)
    for cx in range(-1, 2):
        for cz in range(-1, 2):
            world.chunks[(cx, cz)] = Chunk(cx, cz)
    return world


def fill(world, x0, x1, y0, y1, z0, z1, block_id):
    """Writes blocks straight into the arrays - no neighbour updates, exactly as
    worldgen does, so the setup itself never triggers a flow."""
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                chunk = world.chunks[(x // CX, z // CZ)]
                chunk.set_local(x % CX, y, z % CZ, block_id)


def run(world, ticks):
    for _ in range(ticks):
        world.tick()


def meta(world, x, y, z):
    return world.get_block_meta(x, y, z)


def dump_row(world, y, z, x0, x1):
    out = []
    for x in range(x0, x1 + 1):
        b = world.get_block(x, y, z)
        if b == Block.AIR:
            out.append(" . ")
        elif b == Block.WATER:
            out.append(f"{meta(world, x, y, z):2d} ")
        else:
            out.append(" # ")
    return "".join(out)


# ---------------------------------------------------------------------------
print("\n[1] worldgen water must sit perfectly still (no neighbour updates)")
w = blank_world()
fill(w, 0, 15, 0, 4, 0, 15, Block.STONE)
fill(w, 0, 15, 5, 8, 0, 15, Block.WATER)   # a 4-deep 'sea' over stone
run(w, 40)
check("sea unchanged after 40 ticks",
      all(w.get_block(x, 5, z) == Block.WATER for x in range(16) for z in range(16)))
check("no ticks were ever scheduled", len(w.ticks) == 0, f"queue={len(w.ticks)}")

# ---------------------------------------------------------------------------
print("\n[2] THE REPORTED BUG: break a block under the sea -> water must refill")
w = blank_world()
fill(w, 0, 15, 0, 4, 0, 15, Block.STONE)
fill(w, 0, 15, 5, 8, 0, 15, Block.WATER)
# player places a block at y=6 in the middle of the sea, then mines it away
w.set_block(8, 6, 8, Block.PLANKS)
run(w, 30)
check("placed block survives in the sea", w.get_block(8, 6, 8) == Block.PLANKS)

w.break_block(8, 6, 8)
check("cell is air the instant it's broken", w.get_block(8, 6, 8) == Block.AIR)
run(w, 60)
check("hole refilled with water", w.get_block(8, 6, 8) == Block.WATER,
      f"got block {w.get_block(8, 6, 8)}")
check("refill is a real SOURCE, not a flowing scar", meta(w, 8, 6, 8) == 0,
      f"meta={meta(w, 8, 6, 8)}")

# ---------------------------------------------------------------------------
print("\n[3] same, but on the seabed (block placed then mined at the bottom)")
w = blank_world()
fill(w, 0, 15, 0, 4, 0, 15, Block.STONE)
fill(w, 0, 15, 5, 8, 0, 15, Block.WATER)
w.break_block(8, 4, 8)          # dig a pit INTO the seabed
run(w, 80)
check("pit floods", w.get_block(8, 4, 8) == Block.WATER,
      f"got {w.get_block(8, 4, 8)}")
check("pit is a source", meta(w, 8, 4, 8) == 0, f"meta={meta(w, 8, 4, 8)}")
check("sea above is intact", w.get_block(8, 5, 8) == Block.WATER)

# ---------------------------------------------------------------------------
print("\n[4] a single source on flat ground spreads exactly 7 blocks and stops")
w = blank_world()
fill(w, 0, 15, 0, 4, 0, 15, Block.STONE)
w.set_block(0, 5, 8, Block.WATER, meta_value=0)
run(w, 200)
print("      row y=5 z=8:", dump_row(w, 5, 8, 0, 15))
reach = max((x for x in range(16) if w.get_block(x, 5, 8) == Block.WATER), default=-1)
check("water reaches exactly x=7 (source + 7 steps)", reach == 7, f"reach={reach}")
check("x=8 stays dry", w.get_block(8, 5, 8) == Block.AIR)
check("levels step 1..7 outward",
      [meta(w, x, 5, 8) for x in range(1, 8)] == [1, 2, 3, 4, 5, 6, 7],
      f"{[meta(w, x, 5, 8) for x in range(1, 8)]}")
check("everything settled - nothing still ticking", len(w.ticks) == 0, f"queue={len(w.ticks)}")

# ---------------------------------------------------------------------------
print("\n[5] infinite source: two sources either side of a gap fill it")
w = blank_world()
fill(w, 0, 15, 0, 4, 0, 15, Block.STONE)
w.set_block(7, 5, 8, Block.WATER, meta_value=0)
w.set_block(9, 5, 8, Block.WATER, meta_value=0)
run(w, 60)
check("gap became water", w.get_block(8, 5, 8) == Block.WATER)
check("gap became a SOURCE", meta(w, 8, 5, 8) == 0, f"meta={meta(w, 8, 5, 8)}")

# ---------------------------------------------------------------------------
print("\n[6] waterfall: source falls before it spreads, pools at the bottom")
w = blank_world()
fill(w, 0, 15, 0, 0, 0, 15, Block.STONE)     # floor at y=0
w.set_block(8, 10, 8, Block.WATER, meta_value=0)
run(w, 200)
column = [w.get_block(8, y, 8) == Block.WATER for y in range(1, 11)]
check("full column of falling water y=1..10", all(column), f"{column}")
check("falling cells carry the falling bit",
      all(meta(w, 8, y, 8) & 8 for y in range(1, 10)),
      f"{[meta(w, 8, y, 8) for y in range(1, 10)]}")
print("      pool y=1 z=8:", dump_row(w, 1, 8, 1, 15))
pool_reach = max((x for x in range(8, 16) if w.get_block(x, 1, 8) == Block.WATER), default=-1)
check("pool spreads 7 from the base of the fall", pool_reach == 15, f"reach={pool_reach}")

# ---------------------------------------------------------------------------
print("\n[7] water finds the hole: it heads for a drop instead of spreading evenly")
w = blank_world()
fill(w, 0, 15, 0, 4, 0, 15, Block.STONE)
fill(w, 0, 15, 5, 5, 0, 15, Block.AIR)
w.set_block(11, 4, 8, Block.AIR)             # a hole 3 blocks east of the source
w.set_block(8, 5, 8, Block.WATER, meta_value=0)
run(w, 12)                                    # only a couple of spread steps in
east = w.get_block(9, 5, 8) == Block.WATER
west = w.get_block(7, 5, 8) == Block.WATER
check("flows EAST toward the hole", east)
check("does NOT flow west (away from it)", not west,
      "spread evenly - the flow-cost search isn't working")

# ---------------------------------------------------------------------------
print("\n[8] cut the supply and flowing water dries up again")
w = blank_world()
fill(w, 0, 15, 0, 4, 0, 15, Block.STONE)
w.set_block(0, 5, 8, Block.WATER, meta_value=0)
run(w, 200)
w.set_block(0, 5, 8, Block.AIR)               # remove the source
run(w, 300)
leftovers = [x for x in range(16) if w.get_block(x, 5, 8) == Block.WATER]
check("every trace of water is gone", leftovers == [], f"left at x={leftovers}")
check("scheduler drained", len(w.ticks) == 0, f"queue={len(w.ticks)}")

# ---------------------------------------------------------------------------
print("\n[9] metadata does not leak between block types")
w = blank_world()
fill(w, 0, 15, 0, 4, 0, 15, Block.STONE)
w.set_block(5, 5, 8, Block.WATER, meta_value=5)
w.set_block(5, 5, 8, Block.PLANKS)
check("planks placed over level-5 water have meta 0", meta(w, 5, 5, 8) == 0,
      f"meta={meta(w, 5, 5, 8)}")

# ---------------------------------------------------------------------------
print("\n[10] fluid heights: vanilla's three different answers")
from world import fluids
check("source renders at 8/9", abs(fluids.render_height(0) - 8 / 9) < 1e-6)
check("source is a FULL cube for movement", fluids.collision_top(0) == 1.0)
check("source is a full cube for submersion", fluids.submersion_top(0) == 1.0)
check("level 7 renders at 1/9", abs(fluids.render_height(7) - 1 / 9) < 1e-6)
check("level 7 collides at 1/8", abs(fluids.collision_top(7) - 1 / 8) < 1e-6)
check("falling water renders full-ish (8/9)", abs(fluids.render_height(8) - 8 / 9) < 1e-6)

# ---------------------------------------------------------------------------
print("\n[11] player floats instead of sinking or launching")
w = blank_world()
fill(w, 0, 15, 0, 4, 0, 15, Block.STONE)
fill(w, 0, 15, 5, 12, 0, 15, Block.WATER)     # 8 blocks of sea, surface at y=13

half = config.PLAYER_WIDTH / 2
# feet on the seabed, deep under: body box 5.4..6.4 -> water
check("submerged player is in water",
      fluids.is_body_in_water(w, 8.5, 5.0, 8.5, half, config.PLAYER_HEIGHT))
# feet at the water line: box 13.4..14.4, water tops out at 13.0 -> NOT in water
check("player with feet clear of the surface is NOT in water",
      not fluids.is_body_in_water(w, 8.5, 13.0, 8.5, half, config.PLAYER_HEIGHT),
      "this gap is what makes the player settle at the surface instead of sinking")
check("eye under the surface counts as submerged",
      fluids.is_head_in_water(w, 8.5, 12.5, 8.5))
check("eye above the surface does not",
      not fluids.is_head_in_water(w, 8.5, 13.1, 8.5))

# ---------------------------------------------------------------------------
print("\n[12] drowning: 2 damage every 2 seconds once the bubbles run out")
from player.player import Player


class _FakePhysics:
    head_in_water = True


p = Player(game_mode="survival")
p.physics = _FakePhysics()
check("starts with full air", p.air == config.AIR_MAX_SECONDS)
for _ in range(int(config.AIR_MAX_SECONDS / 0.1)):
    p.update_breathing(0.1)
check("air is spent after AIR_MAX_SECONDS", p.air <= 0.0, f"air={p.air}")
check("no damage taken yet", p.health == config.MAX_HEALTH, f"hp={p.health}")

p.update_breathing(0.1)
check("first hit lands as the last bubble pops",
      p.health == config.MAX_HEALTH - config.DROWN_DAMAGE, f"hp={p.health}")

for _ in range(20):          # +2.0 s
    p.update_breathing(0.1)
check("second hit exactly 2s later",
      p.health == config.MAX_HEALTH - 2 * config.DROWN_DAMAGE, f"hp={p.health}")

hp_before = p.health
for _ in range(19):          # +1.9 s - not enough for a third
    p.update_breathing(0.1)
check("no hit before the interval elapses", p.health == hp_before, f"hp={p.health}")

while p.alive:
    p.update_breathing(0.1)
check("player eventually drowns to death", not p.alive and p.health == 0)

p2 = Player(game_mode="survival")
p2.physics = _FakePhysics()
p2.update_breathing(5.0)
p2.physics.head_in_water = False
p2.update_breathing(0.1)
check("surfacing refills air instantly", p2.air == config.AIR_MAX_SECONDS)

# ---------------------------------------------------------------------------
print("\n[13] water mesh geometry (CPU side - no GL context needed)")
from render.water_renderer import _corner_heights, build_water_mesh_arrays
import numpy as np

w = blank_world()
fill(w, 0, 15, 0, 4, 0, 15, Block.STONE)
fill(w, 0, 15, 5, 5, 0, 15, Block.WATER)
chunk = w.chunks[(0, 0)]
blocks = w.get_padded_blocks_for_chunk(0, 0)
metas = w.get_padded_meta_for_chunk(0, 0)
heights = _corner_heights(blocks, metas)
inner = heights[5, 2:CZ - 1, 2:CX - 1]
check("a flat sheet of sources sits at 8/9 everywhere",
      np.allclose(inner, 8 / 9, atol=1e-5), f"min={inner.min()} max={inner.max()}")

fill(w, 0, 15, 6, 6, 0, 15, Block.WATER)
blocks = w.get_padded_blocks_for_chunk(0, 0)
metas = w.get_padded_meta_for_chunk(0, 0)
heights = _corner_heights(blocks, metas)
check("water with water above it is welded to full height (1.0)",
      np.allclose(heights[5, 2:CZ - 1, 2:CX - 1], 1.0), f"{heights[5, 5, 5]}")

arrays = build_water_mesh_arrays(chunk, w)
check("mesh built", arrays is not None)
positions, normals, depths, indices = arrays
check("positions/normals/depths all agree in length",
      len(positions) == len(normals) == len(depths),
      f"{len(positions)} {len(normals)} {len(depths)}")
check("indices are in range", indices.max() < len(positions))
check("no internal water-vs-water faces: only the top layer's lid shows",
      len(indices) // 6 <= CX * CZ + 4 * CX, f"{len(indices) // 6} faces")
check("depth attribute reports the 2-block column", depths.max() == 2.0, f"{depths.max()}")

dry = w.chunks[(1, 1)]
check("a chunk with no water builds no mesh at all",
      build_water_mesh_arrays(dry, w) is None)

# ---------------------------------------------------------------------------
print()
if FAILS:
    print(f"=== {len(FAILS)} FAILED: {FAILS}")
    raise SystemExit(1)
print("=== all water checks passed")
