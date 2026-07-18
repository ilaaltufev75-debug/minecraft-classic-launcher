"""
Throwaway verification script for the new worldgen. Not part of the game -
delete when done. Run:  .venv\\Scripts\\python.exe _wg_check.py
"""
import time
import numpy as np

from world.chunk import Chunk, CX, CZ, CH
from world.noise import WorldNoise
from world import worldgen
from world.blocks import Block
import config

SEED = 1337
noise = WorldNoise(SEED)


def gen(cx, cz):
    c = Chunk(cx, cz)
    worldgen.generate_chunk_terrain(c, noise)
    return c


print("=" * 60)
print("1. DETERMINISM (same chunk twice must be identical)")
a = gen(3, -7)
b = gen(3, -7)
print("   blocks identical    :", np.array_equal(a.blocks, b.blocks))
print("   height_map identical:", np.array_equal(a.height_map, b.height_map))

print("=" * 60)
print("2. ORDER INDEPENDENCE (neighbour generated first must not change us)")
_ = gen(4, -7)
_ = gen(2, -7)
c = gen(3, -7)
print("   still identical     :", np.array_equal(a.blocks, c.blocks))

print("=" * 60)
print("3. SCALAR vs VECTORIZED terrain_height agreement")
ok = True
for (wx, wz) in [(0, 0), (37, -91), (410, 355), (-1200, 830), (99, 99)]:
    h_scalar, _ = worldgen.terrain_height(noise, wx, wz)
    ch = gen(wx // CX, wz // CZ)
    h_grid = int(ch.height_map[wx % CX, wz % CZ])
    if h_scalar != h_grid:
        ok = False
        print(f"   MISMATCH at {wx},{wz}: scalar={h_scalar} grid={h_grid}")
print("   agree on all samples:", ok)

print("=" * 60)
print("4. TERRAIN STATS over 24x24 chunks (384x384 blocks)")
t0 = time.perf_counter()
R = 12
heights = np.zeros((R * 2 * CX, R * 2 * CZ), dtype=np.int32)
air_below_surface = 0
solid_below_surface = 0
chunks = []
for cx in range(-R, R):
    for cz in range(-R, R):
        ch = gen(cx, cz)
        chunks.append(ch)
        heights[(cx + R) * CX:(cx + R + 1) * CX, (cz + R) * CZ:(cz + R + 1) * CZ] = ch.height_map
elapsed = time.perf_counter() - t0
n = (R * 2) ** 2
print(f"   chunks generated    : {n}")
print(f"   total time          : {elapsed:.2f}s  ({elapsed / n * 1000:.2f} ms/chunk)")
print(f"   height min/max      : {heights.min()} / {heights.max()}")
print(f"   base height         : {config.BASE_TERRAIN_HEIGHT}")
print(f"   tallest rise        : {heights.max() - config.BASE_TERRAIN_HEIGHT} blocks above base")
print(f"   mean height         : {heights.mean():.1f}")

print("=" * 60)
print("5. MOUNTAINS present and sized correctly")
mountain_cols = (heights >= config.BASE_TERRAIN_HEIGHT + 25)
print(f"   cols >=25 above base: {mountain_cols.sum()} ({100.0 * mountain_cols.mean():.2f}% of area)")
print(f"   -> expect a few %, not 0 and not most of the map")

print("=" * 60)
print("6. PLAINS fraction (target ~10-15% of area)")
# sample far wider than the generated patch: the plains mask has a ~690 block
# period, so a 384-block window says nothing about the global fraction
for span in (384, 2000):
    lx = np.arange(-span // 2, span // 2, 2)
    lz = np.arange(-span // 2, span // 2, 2)
    GX, GZ = np.meshgrid(lx, lz, indexing="ij")
    pm = noise.plains_mask_grid(GX.astype(np.float64), GZ.astype(np.float64))
    print(f"   over {span:>5}x{span:<5} blocks: {100.0 * (pm > config.PLAINS_THRESHOLD).mean():.2f}%")

print("=" * 60)
print("7. CAVES: air volume in the stone band")
air = 0
stone_band = 0
for ch in chunks[:80]:
    blocks = ch.blocks.reshape(CH, CZ, CX)
    hm = ch.height_map.T  # (CZ, CX)
    for y in range(1, 40):
        band = (y < hm - 3)
        air += int(((blocks[y] == Block.AIR) & band).sum())
        stone_band += int(band.sum())
print(f"   air in cave band    : {100.0 * air / max(1, stone_band):.2f}%  (measured old baseline: ~7.3%)")

print("=" * 60)
print("8. SLOPE DISTRIBUTION + cave entrances (scanning ALL chunks)")
# Build a slope map over the whole generated area. heights is a pure
# function of position, so slopes at chunk seams are valid too.
hs = heights.astype(np.int32)
sl = np.maximum.reduce([
    np.abs(hs[1:-1, 1:-1] - hs[0:-2, 1:-1]),
    np.abs(hs[1:-1, 1:-1] - hs[2:, 1:-1]),
    np.abs(hs[1:-1, 1:-1] - hs[1:-1, 0:-2]),
    np.abs(hs[1:-1, 1:-1] - hs[1:-1, 2:]),
])
print(f"   slope max           : {sl.max()}")
for thr in (1, 2, 3, 4, 5, 6, 8):
    print(f"   columns slope >= {thr:>2}  : {100.0 * (sl >= thr).mean():7.4f}% of area")

entrances = 0
steep_cols = 0
for ch in chunks:
    blocks = ch.blocks.reshape(CH, CZ, CX)
    hm = ch.height_map
    for x in range(1, CX - 1):
        for z in range(1, CZ - 1):
            h = int(hm[x, z])
            slope = max(abs(h - int(hm[x - 1, z])), abs(h - int(hm[x + 1, z])),
                        abs(h - int(hm[x, z - 1])), abs(h - int(hm[x, z + 1])))
            if slope >= config.CAVE_ENTRANCE_SLOPE:
                steep_cols += 1
                if h - 1 > 0 and blocks[h - 1, z, x] == Block.AIR:
                    entrances += 1
print(f"   CAVE_ENTRANCE_SLOPE : {config.CAVE_ENTRANCE_SLOPE}")
print(f"   qualifying columns  : {steep_cols}")
print(f"   of those, breached  : {entrances}")

print("=" * 60)
print("9. TREES: rules respected")
tot = 0
on_plains = 0
on_steep = 0
on_stone = 0
for ch in chunks[:120]:
    spots = worldgen.tree_columns_for_chunk(ch, noise)
    ox, oz = ch.world_origin()
    hm = ch.height_map
    for (x, top, z, th) in spots:
        tot += 1
        p = noise.plains_mask_grid(np.array([[float(ox + x)]]), np.array([[float(oz + z)]]))[0, 0]
        if p > config.PLAINS_THRESHOLD:
            on_plains += 1
        slope = max(abs(top - int(hm[x - 1, z])), abs(top - int(hm[x + 1, z])),
                    abs(top - int(hm[x, z - 1])), abs(top - int(hm[x, z + 1])))
        if slope > config.TREE_MAX_SLOPE:
            on_steep += 1
        if ch.get_local(x, top, z) != Block.GRASS:
            on_stone += 1
print(f"   trees placed        : {tot}")
print(f"   on plains (want 0)  : {on_plains}")
print(f"   on steep  (want 0)  : {on_steep}")
print(f"   on non-grass (want 0): {on_stone}")

print("=" * 60)
print("done")
