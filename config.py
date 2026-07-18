"""
config.py
Central place for every tunable constant in the game: window/render settings,
physics, world generation, and UI sizing. Keeping these in one module means
gameplay tuning never requires touching engine code.
"""

# ---------------------------------------------------------------------------
# WINDOW / RENDER
# ---------------------------------------------------------------------------
WINDOW_TITLE = "Minecraft Classic"
DEFAULT_WINDOW_SIZE = (1280, 720)
FOV_DEFAULT = 70.0
NEAR_PLANE = 0.1
FAR_PLANE = 600.0                     # must exceed MAX_RENDER_DISTANCE * CHUNK_SIZE_X (32*16=512),
                                       # otherwise the far chunks are clipped away before the fog
                                       # has finished hiding them and the world ends in a hard edge.
                                       # NEAR_PLANE was raised 0.05 -> 0.1 to buy back the depth
                                       # precision this costs (the near plane dominates that ratio).
FOG_COLOR = (0.53, 0.81, 0.92)       # sky blue, also used for GL clear color
FOG_START_CHUNKS = 3.0                # fog begins this many chunks from the camera
FOG_END_CHUNKS = 6.0                  # fully fogged out by this many chunks

VSYNC = True
DEFAULT_FPS_LIMIT = 0                # 0 = unlimited
MOUSE_SENSITIVITY_BASE = 0.0022
MAX_FRAME_DT = 0.1                    # clamp ceiling for per-frame dt (seconds); prevents physics
                                       # tunneling through blocks after a slow/stalled frame

# ---------------------------------------------------------------------------
# WORLD / CHUNKS
# ---------------------------------------------------------------------------
CHUNK_SIZE_X = 16
CHUNK_SIZE_Z = 16
CHUNK_HEIGHT = 199                    # vertical build limit. Raised from 128 to fit 120-block peaks
                                       # (BASE_TERRAIN_HEIGHT 40 + MOUNTAIN_HEIGHT_MAX 120 = 160, and the
                                       # generator clamps to CH-20 = 179, so summits are never clipped).
                                       # Chunks saved by an older build store a shorter column and are
                                       # migrated on load - see save/world_save._restore_column_array.
                                       # Costs ~56% more RAM per chunk (32KB -> 50KB blocks, same again
                                       # for meta), which is why render distance can't also be raised
                                       # until chunk streaming moves off the main thread.
SEA_LEVEL = 30                        # water fills every column from its surface up to this Y.
                                       # Chosen BELOW the land baseline rather than at it: with
                                       # BASE_TERRAIN_HEIGHT 40 and hills+detail reaching +-8, land
                                       # spans 32..48, so a sea at 30 never floods it and the ocean
                                       # lives entirely inside the continentalness ramp. Putting the
                                       # sea at 40 instead would have drowned half of every plain,
                                       # and raising the land to clear it would have moved the cave
                                       # band, the ore bands and the mountain ceiling with it -
                                       # invalidating every measured threshold in this file.
BASE_TERRAIN_HEIGHT = 40              # baseline surface height inland, around which hills vary
HILL_AMPLITUDE = 6                    # rolling hills height variance
DETAIL_AMPLITUDE = 2.0                # high-frequency roughness on top of the base hills
TREE_CHANCE = 0.015                   # probability per valid grass column that a tree spawns
TREE_CHANCE_PLAINS = 0.0016           # same, inside a plain: sparse scatter, not bare ground
ORE_CHANCE_COAL = 0.011
ORE_CHANCE_COAL_MOUNTAIN = 0.020      # coal is noticeably more common inside mountains
ORE_CHANCE_IRON = 0.005

# LEGACY - superseded by the cell-based mountain system below (see
# worldgen._mountain_field). Kept only so any older tool/script still
# importing them doesn't break; nothing in the generator reads these.
MOUNTAIN_AMPLITUDE = 28
MOUNTAIN_THRESHOLD = 1.25

# ---------------------------------------------------------------------------
# WORLDGEN: OCEANS / CONTINENTS
# ---------------------------------------------------------------------------
# Continentalness (noise.continent_grid) ramps the terrain baseline between
# OCEAN_FLOOR_HEIGHT and BASE_TERRAIN_HEIGHT. Everything at or below SEA_LEVEL
# is then filled with water. Mountains are gated on this too - a massif only
# exists if its CELL CENTRE is on land (see worldgen._mountain_field), which is
# the same trick the alpine test uses and for the same reason: sampling per
# column would let one mountain fade out across its own slope.
OCEAN_FLOOR_HEIGHT = 14               # baseline seabed far from any coast -> max depth 16, which is
                                       # about what a normal (non-deep) Minecraft ocean runs
OCEAN_THRESHOLD = 0.389               # MEASURED, not guessed - same warning as PLAINS_THRESHOLD
                                       # below. The field is bell-shaped, so eyeballing a cut off the
                                       # 0..1 range is badly misleading. Sampled over 2.6M points per
                                       # seed across 5 seeds on a 40000-block span:
                                       #    cut 0.3533 -> 75% land    cut 0.4133 -> 65% land
                                       #    cut 0.3848 -> 70% land    cut 0.4416 -> 60% land
                                       # spread across seeds was +-0.003. Those cuts place the
                                       # SHORELINE, which sits at land_w = (SEA_LEVEL -
                                       # OCEAN_FLOOR_HEIGHT) / (BASE_TERRAIN_HEIGHT -
                                       # OCEAN_FLOOR_HEIGHT) = 0.615, not at land_w = 0 - hence
                                       # 0.4133 - 0.615 * OCEAN_FALLOFF = 0.389 for 65% land.
                                       # Re-run _wg_calib.py if you touch either constant.
OCEAN_FALLOFF = 0.04                  # width of the continental shelf, in mask units. Sized from the
                                       # measured field gradient (median 0.000285 per block), not by
                                       # eye: this puts ~87 blocks between the shoreline and the open
                                       # seabed, i.e. a 1:5.5 slope. At 0.12 the shelf ran 260 blocks
                                       # and you could wade halfway to the horizon.
OCEAN_RELIEF_KEEP = 0.35              # fraction of hill/detail relief surviving on the abyssal plain.
                                       # Not zero: a mathematically flat seabed reads as a bug. The
                                       # relief that DOES survive near the shore is what makes the
                                       # coastline ragged - bays, headlands and the odd offshore
                                       # island - instead of a smooth contour line of the mask.
BEACH_RISE = 2                        # how many blocks ABOVE the water line still count as beach.
                                       # Small on purpose: this strip is what reads as a beach, and
                                       # every extra block of it turns the coast into a desert.
                                       # Below the water line the sand rule simply continues, which
                                       # is correct - MC's seabed is sand, not drowned grass.
BEACH_BASELINE_RISE = 4               # how far ABOVE the water line the continental BASELINE may sit
                                       # and still grow sand. This gate is not optional decoration -
                                       # without it, sand appeared in the bottom of every inland
                                       # valley in the world. Inland the baseline is
                                       # BASE_TERRAIN_HEIGHT (40) and hills+detail reach -8, so a
                                       # valley floor lands on exactly 32 = SEA_LEVEL + BEACH_RISE
                                       # and passed the height test - while the water fill needs
                                       # heights < 30 and never fired. Result: dry sand discs in the
                                       # middle of grassland, hundreds of blocks from any sea.
                                       # Testing the baseline instead asks the right question ("could
                                       # the sea plausibly reach this column?") while the height test
                                       # keeps doing what it is good at (following the real,
                                       # relief-chewed shoreline rather than a contour of the mask).
BEACH_MAX_LAND_W = (SEA_LEVEL + BEACH_BASELINE_RISE - OCEAN_FLOOR_HEIGHT) / \
                   (BASE_TERRAIN_HEIGHT - OCEAN_FLOOR_HEIGHT)
                                       # DERIVED, not tuned - it tracks the four constants above so
                                       # that retuning any of them cannot silently resurrect the
                                       # dry-sand-disc bug. Currently ~0.77.

# ---------------------------------------------------------------------------
# WORLDGEN: MOUNTAINS
# ---------------------------------------------------------------------------
# Mountains live on a jittered grid of cells: at most one massif per cell,
# with its centre, radius, height and profile all hashed from the cell
# coordinate so no two come out alike. Cell size sets the spacing between
# massifs; radius sets how wide one is.
MOUNTAIN_CELL_SIZE = 340              # blocks per mountain cell (~21 chunks between centres)
MOUNTAIN_CELL_FILL = 0.70             # fraction of cells that actually hold a mountain
MOUNTAIN_RADIUS_MIN = 32              # core radius in blocks -> 4 chunks across
MOUNTAIN_RADIUS_MAX = 48              # -> 6 chunks across
MOUNTAIN_HEIGHT_MIN = 30              # peak rise above surrounding ground, in blocks
MOUNTAIN_HEIGHT_MAX = 40
MOUNTAIN_FOOTHILL_FACTOR = 1.6        # influence radius = core radius * this; the gap between
                                       # the two is the gentle foothill approach (~2 chunks).
                                       # Measured: at 2.0 the massif spread over 8-12 chunks
                                       # instead of the intended 4-6 and NOTHING in the world
                                       # ever reached a slope above 2 - no cliffs, no cave
                                       # entrances, and the "no trees on steep ground" rule
                                       # silently never fired. Keep this tight.
MOUNTAIN_PROFILE_MIN = 1.7            # low = broad-shouldered massif, high = sharp spire
MOUNTAIN_PROFILE_MAX = 3.2
MOUNTAIN_PLATEAU_CHANCE = 0.30        # fraction of massifs whose summit is sheared flat
MOUNTAIN_PLATEAU_LEVEL = 0.82         # plateau tops out at this fraction of its peak height
MOUNTAIN_WARP_STRENGTH = 15.0         # domain warp, in blocks, so footprints aren't circular
MOUNTAIN_ORE_INFLUENCE = 0.25         # influence above which a column counts as "inside a mountain"

# Per-massif footprint variation. Radius/height/profile alone still leave
# every massif a scaled copy of the same circular bump, which is what made
# them read as "the same mountain over and over". These three break the
# radial symmetry itself, and because every one of them is hashed from the
# cell coordinate the combination is unique per massif: an ellipse at a
# random rotation, with a random number of radial spurs at a random phase.
MOUNTAIN_ASPECT_MIN = 0.70            # footprint ellipse aspect (1.0 = circular)
MOUNTAIN_ASPECT_MAX = 1.42
MOUNTAIN_RIDGE_COUNT_MIN = 2          # radial ridges/spurs running down the flanks
MOUNTAIN_RIDGE_COUNT_MAX = 5
MOUNTAIN_RIDGE_AMPLITUDE = 0.24       # how deeply those ridges bite into the outline

# ---------------------------------------------------------------------------
# WORLDGEN: ALPINE BIOME (rare, tall, steep)
# ---------------------------------------------------------------------------
# A rare, very large region where massifs switch to a tall, sharp profile
# instead of the rounded ones found elsewhere. Whether a massif is alpine is
# decided by sampling the mask AT ITS CELL CENTRE, not per column - sampling
# per column would let one mountain morph from round to sharp across its own
# slope. Outside these regions the familiar rounded massifs are untouched.
ALPINE_THRESHOLD = 0.78               # mask value above which the region is alpine (~rare)
ALPINE_FALLOFF = 0.10
ALPINE_CELL_FILL = 0.82               # inside a range most cells hold a peak; the empty ~18% are
                                       # the flat pockets between them that make it navigable
ALPINE_RADIUS_MIN = 26                # each peak is narrow, so a range reads as many distinct
ALPINE_RADIUS_MAX = 40                #    summits rather than one giant dome
ALPINE_HEIGHT_MIN = 80                # rise above surrounding ground, in blocks
ALPINE_HEIGHT_MAX = 120               # matches CHUNK_HEIGHT 199 (40 base + 120 = 160, ceiling 179)
ALPINE_PROFILE_MIN = 4.5              # much sharper than the rounded MOUNTAIN_PROFILE 1.7-3.2:
ALPINE_PROFILE_MAX = 6.0              #    the flanks stay low then rear up near the summit
ALPINE_FOOTHILL_FACTOR = 1.35         # tighter than rounded massifs = steeper approach = real cliffs,
                                       # which is also what finally lets CAVE_ENTRANCE_SLOPE fire
ALPINE_STONE_DEPTH = 26               # blocks below the summit that turn to bare stone
MOUNTAIN_ROCKY_CHANCE = 0.55          # fraction of massifs that are bare rock near the top. The rest
                                       # stay soil to the summit and grow trees up there - "every
                                       # mountain is a grey cone" was the thing that read as fake.

TERRACE_STEP = 2                      # mountain slopes are quantized to this many blocks
TERRACE_MIN_INFLUENCE = 0.15          # terracing starts fading in at this mountain influence
TERRACE_FALLOFF = 0.25                # ...and is fully applied this much further in

STONE_CAP_DEPTH = 8                   # blocks below a summit that are bare stone
STONE_MIX_DEPTH = 20                  # blocks below a summit where grass and stone mix
TREE_MAX_SLOPE = 2                    # columns steeper than this grow no trees (terraces only)

# ---------------------------------------------------------------------------
# WORLDGEN: PLAINS
# ---------------------------------------------------------------------------
# All three thresholds below are MEASURED, not guessed. A sum of sines is
# bell-shaped rather than uniform, so picking a cutoff from the theoretical
# range is badly misleading - 0.66 here (which "looks like" a high cutoff on
# a 0..1 field) actually flattened 94% of the world into plains. These come
# from sampling the real fields over millions of points across several seeds;
# see the calibration notes in each comment. Re-measure if you retune the
# noise frequencies, don't nudge by eye.
PLAINS_THRESHOLD = 0.764              # measured: covers ~12% of area (spec asks 10-15%)
PLAINS_FALLOFF = 0.15                 # mask range over which the flattening fades in
PLAINS_FLATNESS = 0.15                # fraction of hill amplitude surviving on a full plain

# ---------------------------------------------------------------------------
# WORLDGEN: CAVES
# ---------------------------------------------------------------------------
# Measured baseline: the previous generator (cave3d > 0.76, no second pass)
# carved ~7.3% air through the cave band. The tunnel cutoff below is
# deliberately left at essentially that value so the familiar winding
# passages are unchanged; the extra volume comes from the coarse cavern
# pass, which is what actually makes caves feel bigger - open rooms instead
# of merely more corridors. Together they measure ~12.5% air, ~1.7x the old.
CAVE_THRESHOLD = 0.762                # fine 3D noise: narrow winding tunnels
CAVERN_THRESHOLD = 0.831              # coarse 3D noise: large open chambers
CAVE_ENTRANCE_SLOPE = 4               # caves only breach the surface on columns at least this steep,
                                       # so entrances appear in cliffs and never as holes in flat ground

DEFAULT_RENDER_DISTANCE = 6           # in chunks (radius around the player)
MAX_RENDER_DISTANCE = 32
MIN_RENDER_DISTANCE = 2
CHUNK_BUILD_BUDGET_PER_FRAME = 2      # max chunk meshes (re)built per frame, keeps FPS smooth
LOADING_MESH_MS_PER_FRAME = 10.0      # while the loading screen is up nothing else needs the frame,
                                       # so meshes are built to a TIME budget instead of a chunk count:
                                       # per-chunk mesh cost varies hugely (a solid stone chunk has
                                       # almost no visible faces, an alpine summit has tens of thousands),
                                       # so a fixed count either wastes the frame or blows through it.
                                       # Kept well under the frame time so the progress bar keeps
                                       # animating and Windows never marks the window unresponsive.
CHUNK_GEN_BUDGET_PER_FRAME = 4         # max NEW chunks (terrain generation) started per frame
CHUNK_TREE_BUDGET_PER_FRAME = 3        # max chunks that grow trees per frame (many set_block calls each)
CHUNK_UNLOAD_MARGIN = 2               # chunks beyond render_distance + margin get unloaded

# ---------------------------------------------------------------------------
# TICKS
# ---------------------------------------------------------------------------
# Simulated time runs at a fixed 20 ticks per second, exactly like Minecraft.
# Every reactive delay in the game is quoted in ticks (water spreads a step
# every 5, sand drops 2 after losing support), so this rate is not a tuning
# knob - changing it changes how fast fluids flow.
TICKS_PER_SECOND = 20
TICK_SECONDS = 1.0 / TICKS_PER_SECOND
MAX_TICKS_PER_FRAME = 10              # after a stall (chunk load spike, alt-tab) the accumulator
                                       # can hold seconds of owed ticks. Running all of them in one
                                       # frame makes the NEXT frame slower still, which owes even
                                       # more - the classic death spiral. Past this the backlog is
                                       # dropped: simulated time skips, which is survivable, whereas
                                       # a lock-up is not.
MAX_BLOCK_TICKS_PER_TICK = 65536      # hard ceiling on scheduled block ticks executed in one game
                                       # tick, same value Minecraft uses. Never reached in practice -
                                       # blocks at rest schedule nothing.

# ---------------------------------------------------------------------------
# PHYSICS
# ---------------------------------------------------------------------------
GRAVITY = 31.5
JUMP_VELOCITY = 8.4
FLY_SPEED = 12.0
FLY_VERTICAL_SPEED = 9.0
WALK_SPEED = 4.4
SPRINT_MULTIPLIER = 1.3
WALK_ACCEL = 45.0
GROUND_FRICTION = 11.0
AIR_CONTROL = 0.20
DOUBLE_TAP_WINDOW = 0.30              # seconds allowed between two Space presses to trigger flight

PLAYER_HEIGHT = 1.8
PLAYER_WIDTH = 0.6
PLAYER_EYE_OFFSET = 1.62
STEP_HEIGHT = 0.5001                  # auto-step: max obstruction height climbed without jumping (stairs, slabs)
STEP_SMOOTH_DURATION = 0.12           # seconds the CAMERA takes to visually catch up after an instant step-up (physics Y itself jumps immediately; only the camera eases, matching vanilla Minecraft's smooth stair-climb feel)

FALL_DAMAGE_MIN_DISTANCE = 3.5
FALL_DAMAGE_PER_BLOCK = 1             # half-heart units per extra block fallen
VOID_DEATH_Y = -60.0                  # falling below this Y kills the player instantly (out of world bounds)

REACH_DISTANCE = 5.0

# ---------------------------------------------------------------------------
# SWIMMING
# ---------------------------------------------------------------------------
# Minecraft's water movement is written per-tick: each tick the entity gets
# +0.02 of acceleration, then ALL THREE velocity components are multiplied by
# 0.8, then motionY -= 0.02. Holding jump adds +0.04 to motionY. Everything
# below is that same model converted to per-second units, because this engine
# integrates physics on frame dt rather than on ticks:
#
#   accel   0.02 blocks/tick^2  ->  0.02 * 20 * 20 =  8.0 blocks/s^2
#   drag    x0.8 per tick       ->  v * 0.8^(20t) = v * e^(-4.46t)
#
# Doing it as an exponential rather than a per-frame multiply is what keeps
# swimming speed identical at 30 and at 300 FPS. Terminal speeds fall out of
# accel/drag and match vanilla: ~1.8 b/s sinking, ~1.8 b/s rising, ~2 b/s
# swimming forward (vs 4.4 walking).
WATER_GRAVITY = 8.0                   # downward accel while in water (vs GRAVITY 31.5 in air)
WATER_DRAG = 4.5                      # exponential velocity decay per second, all three axes
WATER_SWIM_UP_ACCEL = 16.0            # holding Space: net +8 up, so the player rises and then
                                       # floats at the surface instead of sinking - this is what
                                       # "держаться на воде" actually is in vanilla, not a special case
WATER_SINK_ACCEL = 12.0               # holding Shift: dive
WATER_MOVE_ACCEL = 9.0                # horizontal accel -> 9.0/4.5 = 2.0 b/s terminal swim speed
WATER_LEDGE_BOOST = 6.0               # vanilla's motionY = 0.3 when you swim into a wall you could
                                       # climb: this is what lets you get OUT of water onto a shore
WATER_MAX_SINK_SPEED = 8.0            # safety clamp; drag normally holds this far lower

# ---------------------------------------------------------------------------
# DROWNING
# ---------------------------------------------------------------------------
AIR_MAX_SECONDS = 15.0                # vanilla's 300 ticks of air
DROWN_DAMAGE = 2                      # half-heart units per drowning hit (= 1 heart)
DROWN_DAMAGE_INTERVAL = 2.0           # seconds between hits once the bubbles are gone

# ---------------------------------------------------------------------------
# WATER RENDERING
# ---------------------------------------------------------------------------
WATER_SHALLOW_COLOR = (0.29, 0.58, 0.80)   # colour where the water is only a block or two deep
WATER_DEEP_COLOR = (0.05, 0.19, 0.47)      # colour out in the open ocean
WATER_DEPTH_FULL = 7.0                # blocks of depth at which water reaches WATER_DEEP_COLOR/max opacity.
                                       # Depth-driven colour+alpha is the single thing that makes an ocean
                                       # read as an ocean: at a constant alpha the seabed shows through
                                       # everywhere equally, so every contour of the sea floor is painted
                                       # onto the surface as a ring and the whole sea looks like plaid.
WATER_ALPHA_SHALLOW = 0.40            # shallows stay see-through, so beaches read as beaches
WATER_ALPHA_DEEP = 0.90
WATER_RIPPLE_FADE_START = 8.0         # blocks from camera where the animated ripple starts fading out...
WATER_RIPPLE_FADE_END = 48.0          # ...and where it is gone entirely. NOT a look tweak: a high-contrast
                                       # pattern viewed at a grazing angle is the textbook moire case, and
                                       # fading detail with distance is the only real fix without
                                       # anisotropic filtering.
UNDERWATER_FOG_COLOR = (0.05, 0.19, 0.42)
UNDERWATER_FOG_START = 0.4
UNDERWATER_FOG_END = 14.0
UNDERWATER_TINT = (26, 82, 156, 85)   # RGBA screen tint drawn over the frame while submerged

# ---------------------------------------------------------------------------
# PLAYER / GAMEPLAY
# ---------------------------------------------------------------------------
MAX_HEALTH = 20                       # half-heart units (10 hearts)
STACK_SIZE = 64
TOOL_STACK_SIZE = 1
HOTBAR_SIZE = 9
MAIN_INVENTORY_ROWS = 3
MAIN_INVENTORY_COLS = 9
MAIN_INVENTORY_SIZE = MAIN_INVENTORY_ROWS * MAIN_INVENTORY_COLS

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
UI_REFERENCE_HEIGHT = 720             # UI is laid out for this height, then scaled to the window
FONT_GLYPH_SIZE = 8                   # base pixel size of each glyph in the bitmap font atlas
UI_SCALE_DEFAULT = 2                  # integer scale factor applied to UI pixels (Minecraft-like)

# Minecraft-ish palette (approximate, procedurally recreated, not ripped assets)
COLOR_UI_STONE_LIGHT = (198, 198, 198)
COLOR_UI_STONE_DARK = (139, 139, 139)
COLOR_UI_STONE_SHADOW = (55, 55, 55)
COLOR_UI_STONE_HIGHLIGHT = (255, 255, 255)
COLOR_UI_TEXT = (255, 255, 255)
COLOR_UI_TEXT_SHADOW = (63, 63, 63)
COLOR_UI_TEXT_HOVER = (255, 255, 160)

# ---------------------------------------------------------------------------
# SAVES
# ---------------------------------------------------------------------------
SAVES_DIR = "saves"
SAVE_FORMAT_VERSION = 1
