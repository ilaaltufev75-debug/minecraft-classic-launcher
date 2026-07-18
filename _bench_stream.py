"""
_bench_stream.py
Headless per-stage profiler for the chunk streaming pipeline (Part 18 prep).

Runs with no GL context and no window: every stage measured here is pure
CPU/numpy work that happens BEFORE anything is handed to OpenGL, which is
exactly the work Part 18 wants to move off the main thread. Run it with:

    .venv\\Scripts\\python.exe _bench_stream.py

It answers two questions that decide the whole plan:

1. WHERE does the per-chunk time actually go? The budget in
   process_generation_budget() is split between terrain generation and tree
   growth, and world.py's own comment claims trees dominate - but mesh
   building and build_shadow_spots() were never measured against them at
   all. If shadow spots or trees dominate, a worker pool is the wrong fix
   (see #2).

2. How much of that work ACTUALLY releases the GIL? "numpy releases the
   GIL" is only true for ufunc inner loops on arrays big enough for the
   inner loop to dominate the surrounding Python bytecode. _mountain_field
   runs on 16x16 = 256-element grids; build_mesh_data runs on ~64k-element
   masks. Those two are not remotely the same thing, and tree growth /
   build_shadow_spots are scalar Python loops that release nothing.
   Section 2 below measures the real speedup at 1/2/4/8 threads instead of
   assuming one, because a thread that holds the GIL doesn't just fail to
   help - it preempts the main thread every sys.setswitchinterval seconds
   and makes frame pacing worse than doing the work inline.

Nothing here mutates the project state or touches saves/ - World is built
with save_dir=None so no disk I/O is involved and no save file is read or
written.
"""

import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config
from world import worldgen
from world.chunk import Chunk, CX, CZ, CH
from world.noise import WorldNoise
from world.world import World

SEED = 1337
GRID_R = 5          # bench grid radius in chunks -> (2R+1)^2 chunks generated
THREAD_JOBS = 64    # chunks per thread-scaling run
THREAD_COUNTS = (1, 2, 4, 8)

_perf = time.perf_counter


def _fmt(times):
    """mean / median / p95 in ms for a list of per-chunk durations."""
    ms = sorted(t * 1000.0 for t in times)
    if not ms:
        return "        -"
    mean = statistics.fmean(ms)
    med = statistics.median(ms)
    p95 = ms[min(len(ms) - 1, int(len(ms) * 0.95))]
    return f"{mean:7.3f} {med:7.3f} {p95:7.3f}"


def _chunks_in_radius(r):
    """Same disc test update_streaming() uses, so the extrapolation below
    counts exactly the chunks the real streaming path would want."""
    return sum(
        1
        for dx in range(-r, r + 1)
        for dz in range(-r, r + 1)
        if dx * dx + dz * dz <= r * r
    )


# ---------------------------------------------------------------------------
# Section 1: per-stage cost breakdown
# ---------------------------------------------------------------------------

def bench_stages():
    noise = WorldNoise(SEED)

    # Warm up: first numpy call in a process pays allocator/dispatch costs
    # that would otherwise land entirely on the first measured chunk and
    # skew the p95.
    for i in range(3):
        worldgen.generate_chunk_terrain(Chunk(900 + i, 900), noise)

    keys = [(cx, cz) for cx in range(-GRID_R, GRID_R + 1)
            for cz in range(-GRID_R, GRID_R + 1)]

    # -- terrain: fresh Chunk objects, no World bookkeeping in the timing
    t_terrain = []
    for (cx, cz) in keys:
        ch = Chunk(cx, cz)
        t0 = _perf()
        worldgen.generate_chunk_terrain(ch, noise)
        t_terrain.append(_perf() - t0)

    # -- everything else needs a real World (trees cross chunk borders via
    #    set_block, meshes need neighbor data for the padded border)
    world = World(seed=SEED, save_dir=None)
    for (cx, cz) in keys:
        world._get_or_create_chunk(cx, cz)

    # only interior chunks have the full 3x3 neighborhood tree growth requires
    inner = [(cx, cz) for (cx, cz) in keys if abs(cx) < GRID_R and abs(cz) < GRID_R]

    t_trees = []
    for (cx, cz) in inner:
        t0 = _perf()
        world.generate_trees_for_chunk(cx, cz)
        t_trees.append(_perf() - t0)

    t_pad, t_mesh, t_merge, t_shadow, t_inst = [], [], [], [], []
    total_vertex_bytes = 0
    total_faces = 0

    for (cx, cz) in inner:
        chunk = world.get_chunk(cx, cz)

        t0 = _perf()
        padded = world.get_padded_blocks_for_chunk(cx, cz)
        t_pad.append(_perf() - t0)

        t0 = _perf()
        groups = chunk.build_mesh_data(padded)
        t_mesh.append(_perf() - t0)

        # Approximates chunk_renderer._merge: the UV remap plus the big
        # concatenate. The glBufferData upload is excluded on purpose -
        # that is the part that must stay on the main thread no matter what
        # the rest of this bench says.
        t0 = _perf()
        if groups:
            pos = np.concatenate([g["positions"] for g in groups.values()])
            nrm = np.concatenate([g["normals"] for g in groups.values()])
            uvs = np.concatenate([g["uvs"] for g in groups.values()])
            uvs = (uvs * 0.0625 + 0.5).astype(np.float32)  # stand-in for atlas uv_for() remap
            idx = np.concatenate([g["indices"] for g in groups.values()])
            total_vertex_bytes += pos.nbytes + nrm.nbytes + uvs.nbytes + idx.nbytes
            total_faces += len(pos) // 4
        t_merge.append(_perf() - t0)

        t0 = _perf()
        chunk.build_shadow_spots()
        t_shadow.append(_perf() - t0)

        t0 = _perf()
        chunk.build_door_instances()
        chunk.build_stair_instances()
        chunk.build_fence_instances()
        t_inst.append(_perf() - t0)

    n = len(inner)
    print("=" * 74)
    print(f"SECTION 1  per-stage cost   (grid radius {GRID_R} chunks, "
          f"{len(keys)} generated, {n} measured)")
    print("=" * 74)
    print(f"{'stage':<34}{'mean':>8}{'med':>8}{'p95':>8}   (ms/chunk)")
    print("-" * 74)
    print(f"{'terrain (gen+caves+ores)':<34}{_fmt(t_terrain)}")
    print(f"{'trees (set_block loop)':<34}{_fmt(t_trees)}")
    print(f"{'get_padded_blocks_for_chunk':<34}{_fmt(t_pad)}")
    print(f"{'build_mesh_data':<34}{_fmt(t_mesh)}")
    print(f"{'merge + uv remap':<34}{_fmt(t_merge)}")
    print(f"{'build_shadow_spots':<34}{_fmt(t_shadow)}")
    print(f"{'door/stair/fence instances':<34}{_fmt(t_inst)}")
    print("-" * 74)

    per_chunk_total = (
        statistics.fmean(t_terrain) + statistics.fmean(t_trees)
        + statistics.fmean(t_pad) + statistics.fmean(t_mesh)
        + statistics.fmean(t_merge) + statistics.fmean(t_shadow)
        + statistics.fmean(t_inst)
    )
    print(f"{'TOTAL per chunk':<34}{per_chunk_total * 1000.0:7.3f} ms")

    n32 = _chunks_in_radius(config.MAX_RENDER_DISTANCE)
    n_def = _chunks_in_radius(config.DEFAULT_RENDER_DISTANCE)
    print()
    print(f"chunks in radius {config.DEFAULT_RENDER_DISTANCE:>2} : {n_def:>5}"
          f"   -> {per_chunk_total * n_def:6.2f} s of CPU work to fill the disc")
    print(f"chunks in radius {config.MAX_RENDER_DISTANCE:>2} : {n32:>5}"
          f"   -> {per_chunk_total * n32:6.2f} s of CPU work to fill the disc")
    print()
    print(f"  (that is the WALL CLOCK floor for entering a world at that")
    print(f"   distance if all of it stays on the main thread - budgets only")
    print(f"   decide how it is spread, not how much there is)")

    # -- memory
    sample = world.get_chunk(0, 0)
    blocks_b = sample.blocks.nbytes
    meta_b = sample.meta.nbytes
    hm_b = sample.height_map.nbytes
    per_chunk_ram = blocks_b + meta_b + hm_b
    print()
    print(f"RAM per chunk : blocks {blocks_b / 1024:6.1f} KB | meta {meta_b / 1024:6.1f} KB"
          f" | height_map {hm_b / 1024:5.1f} KB  = {per_chunk_ram / 1024:6.1f} KB")
    print(f"  at radius {config.MAX_RENDER_DISTANCE}: {per_chunk_ram * n32 / 1e6:7.1f} MB block data"
          f"   (meta alone: {meta_b * n32 / 1e6:6.1f} MB  <- item 4, lazy meta)")

    if n:
        avg_vb = total_vertex_bytes / n
        print()
        print(f"VBO bytes per chunk (32B/vertex) : {avg_vb / 1024:8.1f} KB"
              f"   ({total_faces / n:6.0f} faces/chunk)")
        print(f"  at radius {config.MAX_RENDER_DISTANCE}: {avg_vb * n32 / 1e9:6.2f} GB"
              f"   -> item 3 (byte normal index) would cut ~{avg_vb * n32 * 10 / 32 / 1e9:.2f} GB")

    return world, inner


# ---------------------------------------------------------------------------
# Section 2: does this work actually release the GIL?
# ---------------------------------------------------------------------------

def _scale(label, make_jobs, fn, note):
    """Runs fn over THREAD_JOBS independent jobs at each thread count and
    reports wall time + speedup vs 1 thread. Speedup ~1.0 means the stage
    holds the GIL and a worker pool cannot help it."""
    base = None
    row = []
    for w in THREAD_COUNTS:
        jobs = make_jobs()
        t0 = _perf()
        with ThreadPoolExecutor(max_workers=w) as ex:
            list(ex.map(fn, jobs))
        dt = _perf() - t0
        if base is None:
            base = dt
        row.append((w, dt, base / dt if dt > 0 else 0.0))
    print(f"\n{label}")
    for (w, dt, sp) in row:
        bar = "#" * int(sp * 12)
        print(f"   {w} thread{'s' if w > 1 else ' '} : {dt * 1000:8.1f} ms   "
              f"{sp:5.2f}x  {bar}")
    print(f"   -> {note}")
    return row[-1][2]


def bench_threads(world, inner):
    print()
    print("=" * 74)
    print(f"SECTION 2  real GIL release   ({THREAD_JOBS} chunks per run, "
          f"switchinterval={sys.getswitchinterval() * 1000:.1f} ms)")
    print("=" * 74)
    print("  speedup ~Nx  = work genuinely runs in parallel, worker pool pays off")
    print("  speedup ~1x  = GIL is held; a worker only steals time from the")
    print("                 main thread and makes frame pacing WORSE")

    noise = WorldNoise(SEED)

    # -- terrain: each job builds its own Chunk, WorldNoise is read-only
    #    (pure functions of seed + coords, no shared mutable state)
    def terrain_jobs():
        return [(2000 + i % 8, 2000 + i // 8) for i in range(THREAD_JOBS)]

    def terrain_fn(key):
        worldgen.generate_chunk_terrain(Chunk(key[0], key[1]), noise)

    sp_terrain = _scale(
        "terrain generation (mixed: 256-elem 2D mountain field + 46k-elem 3D caves)",
        terrain_jobs, terrain_fn,
        "the 3D pass releases; _mountain_field's tiny grids mostly do not",
    )

    # -- mesh building: read-only over chunk.blocks via the padded copy
    mesh_inputs = []
    for (cx, cz) in inner[:THREAD_JOBS]:
        chunk = world.get_chunk(cx, cz)
        mesh_inputs.append((chunk, world.get_padded_blocks_for_chunk(cx, cz)))
    while len(mesh_inputs) < THREAD_JOBS and mesh_inputs:
        mesh_inputs.append(mesh_inputs[len(mesh_inputs) % len(inner)])

    sp_mesh = _scale(
        "build_mesh_data (64k-elem boolean masks, np.nonzero, broadcasting)",
        lambda: list(mesh_inputs),
        lambda job: job[0].build_mesh_data(job[1]),
        "the one stage with a big parallel fraction - this is where workers win",
    )

    # -- shadow spots: pure Python double loop, read-only on the chunk
    shadow_inputs = [world.get_chunk(cx, cz) for (cx, cz) in inner[:THREAD_JOBS]]
    while len(shadow_inputs) < THREAD_JOBS and shadow_inputs:
        shadow_inputs.append(shadow_inputs[len(shadow_inputs) % len(inner)])

    sp_shadow = _scale(
        "build_shadow_spots (scalar Python loop over 256 columns)",
        lambda: list(shadow_inputs),
        lambda chunk: chunk.build_shadow_spots(),
        "expect ~1x. If this stage is expensive it must be VECTORIZED, not threaded",
    )

    # Tree growth is deliberately absent: generate_trees_for_chunk() mutates
    # neighboring chunks through set_block, so it is not thread-safe as
    # written and cannot be benchmarked this way. It is also a pure scalar
    # Python loop, so its speedup would be the same ~1x build_shadow_spots
    # shows above. Fixing it means batching the writes into one vectorized
    # numpy index-assign per target chunk, not moving it to a thread.

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  terrain            {sp_terrain:5.2f}x at {THREAD_COUNTS[-1]} threads")
    print(f"  build_mesh_data    {sp_mesh:5.2f}x at {THREAD_COUNTS[-1]} threads")
    print(f"  build_shadow_spots {sp_shadow:5.2f}x at {THREAD_COUNTS[-1]} threads")
    print()
    print("  Any stage under ~1.5x does not belong in a ThreadPoolExecutor.")
    print("  Either vectorize it, or move it to a ProcessPoolExecutor and pay")
    print("  the pickling cost (terrain is a pure function of seed+coords, so")
    print("  it survives that trip; mesh data is far too big to ship back).")


if __name__ == "__main__":
    world, inner = bench_stages()
    bench_threads(world, inner)
