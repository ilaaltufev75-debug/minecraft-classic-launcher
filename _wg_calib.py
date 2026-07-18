"""
Throwaway threshold calibration. Measures the ACTUAL distribution of each
noise field over a large sample and reports the cutoff needed to hit a
target coverage, instead of guessing from the theoretical range (the
theoretical range is nearly useless here: a sum of sines is bell-shaped,
not uniform, so a threshold at 80% of the range covers far more than 20%
of the area). Delete when done.
"""
import numpy as np
from world.noise import WorldNoise

SEEDS = [1337, 4242, 99, 2024, 7]
TARGET_PLAINS = 0.12          # 12% of area -> spec asks 10-15%
TARGET_CAVE_AIR = 0.130       # measured: the OLD generator already ran ~7.3% air.
                              # ~13% is a clear step up without turning stone
                              # into sponge (26% was unplayable).
CAVERN_SHARE = 0.45           # of that air, the fraction the coarse pass owns.
                              # High on purpose: the coarse pass is what makes
                              # caves feel BIGGER (open rooms) rather than just
                              # more numerous.


def pct(samples, frac):
    """Threshold above which `frac` of samples lie."""
    return float(np.quantile(samples, 1.0 - frac))


print("=" * 64)
print("PLAINS MASK - sampling 2000x2000 blocks per seed")
plain_cuts = []
for s in SEEDS:
    n = WorldNoise(s)
    xs = np.arange(-1000, 1000, 4).astype(np.float64)
    zs = np.arange(-1000, 1000, 4).astype(np.float64)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    v = n.plains_mask_grid(X, Z)
    cut = pct(v.ravel(), TARGET_PLAINS)
    plain_cuts.append(cut)
    print(f"  seed {s:>5}: min={v.min():.3f} max={v.max():.3f} mean={v.mean():.3f} "
          f"| cut for {TARGET_PLAINS:.0%} = {cut:.4f}")
print(f"  --> PLAINS_THRESHOLD = {np.mean(plain_cuts):.3f}")

print("=" * 64)
print("CAVE NOISE - sampling a 3D volume per seed (y 1..40)")
tun_cuts, cav_cuts = [], []
for s in SEEDS:
    n = WorldNoise(s)
    xs = np.arange(-160, 160, 2).astype(np.float64)
    zs = np.arange(-160, 160, 2).astype(np.float64)
    ys = np.arange(1, 40, 1).astype(np.float64)
    X, Z, Y = np.meshgrid(xs, zs, ys, indexing="ij")
    tun = n.cave3d_grid(X, Y, Z).ravel()
    cav = n.cavern3d_grid(X, Y, Z).ravel()

    # Union of two passes overshoots if each is set to its own share, so aim
    # each a bit below and verify the union afterwards.
    tun_target = TARGET_CAVE_AIR * (1.0 - CAVERN_SHARE)
    cav_target = TARGET_CAVE_AIR * CAVERN_SHARE
    tc = pct(tun, tun_target)
    cc = pct(cav, cav_target)
    tun_cuts.append(tc)
    cav_cuts.append(cc)
    union = ((tun > tc) | (cav > cc)).mean()
    print(f"  seed {s:>5}: tunnel cut={tc:.4f}  cavern cut={cc:.4f}  -> union air {union:.2%}")

t_final = float(np.mean(tun_cuts))
c_final = float(np.mean(cav_cuts))
print(f"  --> CAVE_THRESHOLD    = {t_final:.3f}")
print(f"  --> CAVERN_THRESHOLD  = {c_final:.3f}")

print("=" * 64)
print("VERIFY averaged cuts across all seeds")
tot = []
for s in SEEDS:
    n = WorldNoise(s)
    xs = np.arange(-160, 160, 2).astype(np.float64)
    zs = np.arange(-160, 160, 2).astype(np.float64)
    ys = np.arange(1, 40, 1).astype(np.float64)
    X, Z, Y = np.meshgrid(xs, zs, ys, indexing="ij")
    u = ((n.cave3d_grid(X, Y, Z) > t_final) | (n.cavern3d_grid(X, Y, Z) > c_final)).mean()
    tot.append(u)
    print(f"  seed {s:>5}: union air {u:.2%}")
print(f"  mean union air: {np.mean(tot):.2%}  (target {TARGET_CAVE_AIR:.1%})")

print("=" * 64)
print("OLD baseline for reference: cave3d > 0.76 only")
for s in SEEDS[:2]:
    n = WorldNoise(s)
    xs = np.arange(-160, 160, 2).astype(np.float64)
    zs = np.arange(-160, 160, 2).astype(np.float64)
    ys = np.arange(1, 40, 1).astype(np.float64)
    X, Z, Y = np.meshgrid(xs, zs, ys, indexing="ij")
    print(f"  seed {s:>5}: {(n.cave3d_grid(X, Y, Z) > 0.76).mean():.2%} air")

# ---------------------------------------------------------------------------
# CONTINENTS / OCEANS
# ---------------------------------------------------------------------------
# The threshold in config was derived from the RAW continent_grid distribution.
# That is only half the answer: what the player experiences is where the
# finished column height crosses SEA_LEVEL, after hills, detail, the plains
# flattening and any mountain have been added on top of the baseline. Those
# terms are worth +-8 blocks against a 16-block ramp, so they move the real
# coastline relative to the mask's own contour - which is exactly what makes
# the shore ragged, and exactly why the raw figure cannot be trusted on its
# own. This section runs the REAL _column_fields and measures the result.
import config as _cfg
from world import worldgen

print("=" * 64)
print("CONTINENTS - land fraction via the real generator")
print(f"  SEA_LEVEL={_cfg.SEA_LEVEL}  OCEAN_FLOOR_HEIGHT={_cfg.OCEAN_FLOOR_HEIGHT}  "
      f"BASE_TERRAIN_HEIGHT={_cfg.BASE_TERRAIN_HEIGHT}")
print(f"  OCEAN_THRESHOLD={_cfg.OCEAN_THRESHOLD}  OCEAN_FALLOFF={_cfg.OCEAN_FALLOFF}")
land_fracs, beach_fracs = [], []
for s in SEEDS:
    n = WorldNoise(s)
    # step 8 over +-8000 blocks: fine enough that a beach strip is not missed,
    # coarse enough that the mountain field's per-cell loop stays affordable
    axis = np.arange(-8000, 8000, 8).astype(np.float64)
    X, Z = np.meshgrid(axis, axis, indexing="ij")
    heights, _a, _p, _i, _pl, land_w = worldgen._column_fields(n, X, Z)

    land = (heights > _cfg.SEA_LEVEL).mean()
    beach = ((heights > _cfg.SEA_LEVEL) &
             (heights <= _cfg.SEA_LEVEL + _cfg.BEACH_RISE)).mean()
    depth = _cfg.SEA_LEVEL - heights[heights <= _cfg.SEA_LEVEL]
    land_fracs.append(land)
    beach_fracs.append(beach)
    print(f"  seed {s:>5}: land {land:.1%} | sand strip above water {beach:.2%} | "
          f"sea depth mean {depth.mean():.1f} max {depth.max():.0f}")

print(f"  --> mean land {np.mean(land_fracs):.1%}  (config aims for 65%)")
print(f"      spread across seeds {np.std(land_fracs):.2%}")
print("      If this is far off 65%, the raw-field cut in config.OCEAN_THRESHOLD")
print("      is being dragged by the relief terms - re-derive it from THIS number,")
print("      not from the raw quantile table.")
