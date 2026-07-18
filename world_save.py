"""
save/world_save.py
Persists worlds to disk under saves/<world_name>/. Each world gets a
metadata JSON (name, seed, game mode, spawn position, timestamps) plus one
compressed NPZ file per generated chunk (numpy's native compressed array
format - fast to read/write and handles our block arrays natively without
needing a custom binary format). This is what backs the "select an existing
world or create a new one" flow: the world list screen just lists
directories under saves/ and reads their metadata.json.
"""

import json
import os
import time
import numpy as np

import config
import paths
from world.blocks import Block
from world.chunk import Chunk, CX, CZ, CH


def _saves_root():
    """
    Resolved through paths.data_path, which is the opposite root from the
    assets: this one has to be WRITABLE and has to SURVIVE.

    A onefile PyInstaller build unpacks itself into a temp folder and deletes it
    on exit. `config.SAVES_DIR` being the bare relative "saves" meant that in a
    build, worlds landed wherever the exe was launched from - or, worse, if
    anything ever chdir'd into the bundle, inside the folder Windows wipes when
    the game closes. Either way the player finds out an hour into a build.
    """
    return paths.data_dir(config.SAVES_DIR)


def sanitize_world_name(name: str) -> str:
    """Turns an arbitrary display name into a filesystem-safe directory name."""
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name).strip()
    return safe or "World"


def list_worlds():
    """Returns a list of dicts (one per saved world) with metadata, newest first."""
    root = _saves_root()
    worlds = []
    for entry in os.listdir(root):
        world_dir = os.path.join(root, entry)
        meta_path = os.path.join(world_dir, "metadata.json")
        if os.path.isdir(world_dir) and os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["_dir_name"] = entry
                worlds.append(meta)
            except (json.JSONDecodeError, OSError):
                continue  # skip corrupted saves rather than crashing the menu
    worlds.sort(key=lambda m: m.get("last_played", 0), reverse=True)
    return worlds


def world_exists(dir_name: str) -> bool:
    return os.path.isdir(os.path.join(_saves_root(), dir_name))


def find_land_spawn(seed: int, search_radius: int = 3000, step: int = 24):
    """
    Picks a spawn point on dry land, nearest to the origin.

    Necessary since oceans exist: continentalness is calibrated to ~35% water,
    so hardcoding (0, 0) drops roughly one new world in three onto the seabed
    under 16 blocks of sea. Minecraft runs the same kind of search for the same
    reason rather than trusting the origin.

    Evaluates the real generator's column heights on a coarse grid - the same
    _column_fields every chunk goes through, so the answer cannot disagree with
    the terrain that actually gets built - and takes the closest candidate
    standing clear of the water line. Coarse on purpose: this runs once at
    world creation, and a 24-block step over a 3000-block radius is 62k columns
    in a couple of vectorized calls.

    Falls back to the origin if nothing qualifies, which in practice means a
    seed whose entire 6000-block neighbourhood is ocean. Extremely unlikely
    given the field's wavelength, but returning something beats raising.
    """
    from world.noise import WorldNoise
    from world import worldgen

    noise = WorldNoise(seed)
    axis = np.arange(-search_radius, search_radius + 1, step, dtype=np.float64)
    X, Z = np.meshgrid(axis, axis, indexing="ij")
    heights, _m_add, _m_peak, _m_infl, _plains, _land = worldgen._column_fields(noise, X, Z)

    # BEACH_RISE + 1 clears the sand strip too: spawning on a beach is fine,
    # spawning ankle-deep in the surf is not.
    dry = heights > config.SEA_LEVEL + config.BEACH_RISE + 1
    if not dry.any():
        return 0.0, 0.0

    dist2 = X * X + Z * Z
    dist2 = np.where(dry, dist2, np.inf)
    flat = int(np.argmin(dist2))
    ix, iz = np.unravel_index(flat, dist2.shape)
    return float(X[ix, iz]), float(Z[ix, iz])


def create_world_metadata(display_name: str, seed: int, game_mode: str) -> dict:
    dir_name = sanitize_world_name(display_name)
    base_dir_name = dir_name
    counter = 2
    while world_exists(dir_name):
        dir_name = f"{base_dir_name}_{counter}"
        counter += 1

    world_dir = os.path.join(_saves_root(), dir_name)
    os.makedirs(os.path.join(world_dir, "chunks"), exist_ok=True)

    spawn_x, spawn_z = find_land_spawn(seed)

    now = time.time()
    meta = {
        "display_name": display_name,
        "dir_name": dir_name,
        "seed": seed,
        "game_mode": game_mode,
        "created": now,
        "last_played": now,
        "save_format_version": config.SAVE_FORMAT_VERSION,
        "spawn_x": spawn_x,
        "spawn_z": spawn_z,
    }
    _write_metadata(dir_name, meta)
    return meta


def _write_metadata(dir_name: str, meta: dict):
    world_dir = os.path.join(_saves_root(), dir_name)
    with open(os.path.join(world_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_metadata(dir_name: str) -> dict:
    world_dir = os.path.join(_saves_root(), dir_name)
    with open(os.path.join(world_dir, "metadata.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def touch_last_played(dir_name: str):
    meta = load_metadata(dir_name)
    meta["last_played"] = time.time()
    _write_metadata(dir_name, meta)


def update_spawn(dir_name: str, x: float, z: float):
    """No-op kept for compatibility - world spawn is fixed at creation time
    (see create_world_metadata) so death respawns always return to the
    original first-spawn point, not wherever the player last logged out."""
    return


def delete_world(dir_name: str):
    import shutil
    world_dir = os.path.join(_saves_root(), dir_name)
    if os.path.isdir(world_dir):
        shutil.rmtree(world_dir)


# --- chunk persistence -------------------------------------------------------

def _chunk_path(dir_name: str, cx: int, cz: int) -> str:
    return os.path.join(_saves_root(), dir_name, "chunks", f"{cx}_{cz}.npz")


def save_chunk(dir_name: str, chunk: Chunk):
    path = _chunk_path(dir_name, chunk.cx, chunk.cz)
    np.savez_compressed(
        path,
        blocks=chunk.blocks,
        meta=chunk.meta,
        height_map=chunk.height_map,
        terrain_height=chunk.terrain_height,
        trees_generated=np.array([chunk.trees_generated]),
    )
    # This chunk now matches what is on disk, so the next autosave can skip it
    # entirely until something actually changes it again.
    chunk.needs_save = False


def chunk_save_exists(dir_name: str, cx: int, cz: int) -> bool:
    return os.path.isfile(_chunk_path(dir_name, cx, cz))


def _restore_column_array(dest: np.ndarray, saved: np.ndarray):
    """
    Copies a saved flat per-block array (blocks or meta) into a freshly
    allocated one, tolerating a change in CHUNK_HEIGHT between the build that
    wrote the save and the build reading it.

    This works because the flat layout is `x + z*CX + y*CX*CZ` (see
    world.chunk._idx): Y is the SLOWEST-varying axis, so the first
    saved_height*CX*CZ entries are exactly the columns y=0..saved_height-1 in
    order. A straight prefix copy therefore puts every stored block back at
    the identical (x, y, z) it occupied before - player builds included - and
    the freshly zeroed remainder above it reads as Block.AIR, which is what
    empty sky should be anyway.

    Raising CHUNK_HEIGHT is thus lossless. Lowering it truncates whatever sat
    above the new ceiling, which is the only available option (there is
    nowhere to put those blocks); it is not a supported downgrade path.

    Without this, bumping CHUNK_HEIGHT 128 -> 199 made `chunk.blocks[:] = data["blocks"]`
    raise ValueError on every chunk written by the older build, i.e. every
    existing world became impossible to open.
    """
    saved = np.asarray(saved).reshape(-1)
    layer = CX * CZ
    saved_height = len(saved) // layer
    if saved_height == CH:
        dest[:] = saved
        return
    keep = min(saved_height, CH) * layer
    dest[:keep] = saved[:keep]


def _derive_terrain_height(chunk: Chunk):
    """
    Reconstructs a plausible terrain_height for a chunk saved before that
    field existed, using the topmost NATURALLY GENERATED block in each column.

    worldgen only ever lays down grass/dirt/stone/ore, so the highest one of
    those is the ground; logs, leaves, planks and everything else above it got
    there by tree growth or by the player and must not count. Deriving it from
    the blocks rather than re-running the noise is deliberate: the noise would
    give the height the CURRENT generator would produce, which for a world
    written by an older build is not the terrain actually stored in the file,
    and it costs a full _column_fields evaluation per chunk on load.

    The one thing this cannot recover is dirt/stone the PLAYER stacked on the
    surface in a legacy world - that reads as ground and casts no shadow.
    Chunks generated from now on carry the real value and are unaffected.
    """
    blocks = chunk.blocks_yzx()  # (CH, CZ, CX)
    natural = (
        (blocks == Block.GRASS) | (blocks == Block.DIRT) | (blocks == Block.STONE)
        | (blocks == Block.COAL_ORE) | (blocks == Block.IRON_ORE)
    )
    # argmax on the Y-reversed mask finds the first natural block from the top
    reversed_y = natural[::-1, :, :]
    first_from_top = np.argmax(reversed_y, axis=0)          # (CZ, CX)
    found = reversed_y.any(axis=0)
    heights = np.where(found, CH - 1 - first_from_top, 0)   # (CZ, CX)
    chunk.terrain_height[:, :] = heights.T.astype(np.int16)  # -> (CX, CZ)


def load_chunk(dir_name: str, cx: int, cz: int) -> Chunk:
    path = _chunk_path(dir_name, cx, cz)
    data = np.load(path)
    chunk = Chunk(cx, cz)
    _restore_column_array(chunk.blocks, data["blocks"])
    # "meta" is a newer field - saves made before doors existed won't have
    # it, so fall back to all-zeros (no door state) rather than crashing on
    # KeyError when loading an older world.
    if "meta" in data:
        _restore_column_array(chunk.meta, data["meta"])
    chunk.height_map[:, :] = data["height_map"]
    # "terrain_height" is newer still (added for ground shadows). Older saves
    # get it reconstructed from their own block data; the derivation is cheap
    # and fully vectorized, so it just runs on every load of a legacy chunk
    # rather than forcing a rewrite of every file in the world - which is
    # exactly the multi-second autosave stall needs_save exists to avoid.
    if "terrain_height" in data:
        chunk.terrain_height[:, :] = data["terrain_height"]
    else:
        _derive_terrain_height(chunk)
    chunk.trees_generated = bool(data["trees_generated"][0])
    chunk.generated = True
    # Freshly read from disk, so by definition identical to it. Chunk.__init__
    # defaults this to True (correct for a newly generated chunk, which must be
    # written out once); a loaded chunk must clear it or every autosave would
    # rewrite an unchanged file.
    chunk.needs_save = False
    return chunk


def save_all_loaded_chunks(dir_name: str, world):
    """
    Saves every loaded chunk that has actually changed since it was last
    written. Called periodically and on quit/world-exit.

    The `needs_save` filter is what keeps this cheap. It used to write every
    loaded chunk unconditionally: harmless at render distance 6 (~110 chunks),
    but at 32 that is ~3200 chunks of ~100KB arrays pushed through zlib
    synchronously, i.e. a multi-second stall every autosave interval. In
    practice almost nothing changes between two autosaves - a handful of blocks
    the player placed, plus whatever chunks were newly generated - so this
    typically drops the work by three orders of magnitude.

    Note this stays CORRECT rather than merely fast: a chunk is written once on
    creation (so procedural content, including leaves that spilled across a
    border from a neighbour's tree, is captured), and rewritten whenever any
    block or metadata byte in it changes. Nothing is skipped that a reload
    couldn't reproduce byte-for-byte.
    """
    for (cx, cz), chunk in world.chunks.items():
        if chunk.needs_save:
            save_chunk(dir_name, chunk)


# --- player state persistence (inventory, health, position) -----------------

def _player_state_path(dir_name: str) -> str:
    return os.path.join(_saves_root(), dir_name, "player.json")


def save_player_state(dir_name: str, player, inventory):
    """
    Persists the player's inventory (hotbar + main grid), selected hotbar
    slot, health, and exact position/look direction. Without this, every
    session started with an empty inventory and full health regardless of
    what the player had collected, since only chunk block data and the
    coarse spawn_x/spawn_z were ever saved.
    """
    state = {
        "inventory_slots": inventory.slots,
        "selected_slot": inventory.selected_slot,
        "health": player.health,
        "air": player.air,
        "x": player.physics.x,
        "y": player.physics.y,
        "z": player.physics.z,
        "yaw": player.yaw,
        "pitch": player.pitch,
    }
    with open(_player_state_path(dir_name), "w", encoding="utf-8") as f:
        json.dump(state, f)


def load_player_state(dir_name: str):
    """Returns the saved player state dict, or None if no save exists yet
    (e.g. a brand new world, or a world saved before this feature existed)."""
    path = _player_state_path(dir_name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
