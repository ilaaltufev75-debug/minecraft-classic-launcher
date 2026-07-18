"""
render/texture_atlas.py
Procedurally generates all block textures using Pillow, then packs them
into a single OpenGL texture atlas (a grid of 16x16 tiles) so the whole
world can be drawn with one bound texture instead of switching textures
per block type mid-draw.

Textures aim for a more "alive" Minecraft look than pure flat-noise tiles:
each block gets a base color, then layered noise, subtle gradient shading,
and (for grass/stone/wood) some larger structural details, closer to how
real Minecraft tiles read at a glance even though every pixel here is
generated, not ripped from any existing asset.
"""

import os
import numpy as np
from PIL import Image
from OpenGL.GL import (
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, glGenerateMipmap,
    GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST,
    GL_NEAREST_MIPMAP_LINEAR, GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T, GL_REPEAT,
)

import paths
from world.blocks import Block
from world.block_texture_map import BLOCK_TEXTURE_FOLDERS

TILE = 16          # each tile is 16x16 pixels, matching classic Minecraft resolution
TILE_PADDING = 2    # extra border pixels duplicated around each tile before packing (see _pack)
PADDED_TILE = TILE + TILE_PADDING * 2
ATLAS_COLS = 8      # atlas grid width in tiles
ATLAS_ROWS = 8      # atlas grid height in tiles (64 tile slots total, plenty of headroom)

# Resolved through paths.asset_path, never as a bare relative path: in a
# PyInstaller build the working directory is wherever the player double-clicked
# the exe, so "assets/textures/blocks" points at their Desktop. The failure is
# silent - _load_block_face_png just returns None and every tile quietly falls
# back to its procedural version - which is exactly why it shipped once already.
BLOCK_TEXTURES_DIR = paths.asset_path("assets", "textures", "blocks")


def _load_block_face_png(block_id: int, face: str):
    """
    Tries to load assets/textures/blocks/<folder>/<face>.png for this
    block/face. Falls back to <folder>/all.png if a face-specific file
    isn't there, and returns None if neither exists (caller then falls
    back to the procedural generator) - so a block never breaks just
    because only some of its faces have art yet.
    """
    folder = BLOCK_TEXTURE_FOLDERS.get(block_id)
    if folder is None:
        return None

    for candidate_face in (face, "all"):
        path = os.path.join(BLOCK_TEXTURES_DIR, folder, f"{candidate_face}.png")
        if os.path.isfile(path):
            try:
                img = Image.open(path).convert("RGBA")
            except Exception:
                continue  # corrupt/unreadable file - keep trying other candidates
            if img.size != (TILE, TILE):
                img = img.resize((TILE, TILE), Image.NEAREST)
            return img
    return None


def _extrude_border(tile_img, padding: int):
    """
    Returns a (TILE+2*padding, TILE+2*padding) RGBA image with `tile_img`
    centered and its OWN edge pixels replicated outward into the padding
    border (not mirrored, not wrapped - a straight nearest-edge copy),
    exactly like standard texture-atlas border-extrusion. This is what
    keeps a mipmapped atlas from blending a tile with its neighbor: every
    padding pixel already matches its nearest real tile pixel, so a
    downsample landing partly in the padding still samples "more of the
    same tile" instead of a different one.
    """
    w, h = tile_img.size
    padded = Image.new("RGBA", (w + padding * 2, h + padding * 2))
    padded.paste(tile_img, (padding, padding))
    px = tile_img.load()
    ppx = padded.load()

    for p in range(1, padding + 1):
        # top and bottom edge rows extended upward/downward
        for x in range(w):
            ppx[padding + x, padding - p] = px[x, 0]
            ppx[padding + x, padding + h - 1 + p] = px[x, h - 1]
        # left and right edge columns extended outward
        for y in range(h):
            ppx[padding - p, padding + y] = px[0, y]
            ppx[padding + w - 1 + p, padding + y] = px[w - 1, y]
        # 4 corners
        ppx[padding - p, padding - p] = px[0, 0]
        ppx[padding + w - 1 + p, padding - p] = px[w - 1, 0]
        ppx[padding - p, padding + h - 1 + p] = px[0, h - 1]
        ppx[padding + w - 1 + p, padding + h - 1 + p] = px[w - 1, h - 1]

    return padded


def _rng(seed_tuple):
    """A small local RNG seeded deterministically per-tile, so texture generation
    is reproducible across runs (useful for tests) without affecting world seeding."""
    seed = abs(hash(seed_tuple)) % (2 ** 32)
    return np.random.RandomState(seed)


def _make_tile(size=TILE):
    return Image.new("RGBA", (size, size), (0, 0, 0, 0))


def _add_noise(arr, rng, base, variance, channel_variance=None):
    """arr: (H,W,3) uint8 array, filled in-place with base color + per-pixel noise."""
    h, w = arr.shape[:2]
    cv = channel_variance or (variance, variance, variance)
    for c in range(3):
        noise = rng.randint(-cv[c] // 2, cv[c] // 2 + 1, size=(h, w))
        channel = np.clip(base[c] + noise, 0, 255)
        arr[:, :, c] = channel


def _apply_vertical_shade(arr, strength=18):
    """Subtle top-lit gradient so tiles don't look perfectly flat."""
    h = arr.shape[0]
    gradient = np.linspace(strength, -strength, h).reshape(h, 1, 1)
    arr[:, :, :3] = np.clip(arr[:, :, :3].astype(np.int16) + gradient, 0, 255).astype(np.uint8)


def gen_dirt(rng=None):
    rng = rng or _rng("dirt")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    _add_noise(arr, rng, base=(134, 96, 67), variance=44)
    # scattered darker pebbles/speckles for texture detail
    speck_mask = rng.random((TILE, TILE)) < 0.045
    arr[speck_mask, 0:3] = (78, 55, 38)
    _apply_vertical_shade(arr, strength=10)
    return Image.fromarray(arr, "RGBA")


def gen_grass_top(rng=None):
    rng = rng or _rng("grass_top")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    _add_noise(arr, rng, base=(94, 156, 51), variance=46, channel_variance=(34, 50, 28))
    # lighter and darker blade clumps for a livelier, less uniform green
    light_mask = rng.random((TILE, TILE)) < 0.08
    arr[light_mask, 0:3] = (146, 198, 84)
    dark_mask = rng.random((TILE, TILE)) < 0.05
    arr[dark_mask, 0:3] = (66, 118, 38)
    _apply_vertical_shade(arr, strength=8)
    return Image.fromarray(arr, "RGBA")


def gen_grass_side(dirt_img, rng=None):
    rng = rng or _rng("grass_side")
    arr = np.array(dirt_img).copy()
    grass_h = 5
    grass_base = (94, 156, 51)
    # jagged edge varies PER COLUMN (x), not per row, so the grass/dirt boundary
    # reads as an uneven fringe across the tile rather than one flat horizontal cut
    for x in range(TILE):
        edge = grass_h + rng.randint(-1, 2)
        edge = max(2, min(TILE - 1, edge))
        for y in range(edge):
            variance = 30
            color = tuple(int(np.clip(grass_base[c] + rng.randint(-variance, variance), 0, 255)) for c in range(3))
            arr[y, x, 0:3] = color
        # a couple of stray blades reaching one pixel further down
        if rng.random() < 0.3 and edge < TILE:
            arr[edge, x, 0:3] = grass_base
    _apply_vertical_shade(arr, strength=10)
    return Image.fromarray(arr, "RGBA")


def gen_stone(rng=None):
    rng = rng or _rng("stone")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    _add_noise(arr, rng, base=(128, 128, 130), variance=34)
    # a few crack-like darker streaks
    for _ in range(3):
        x0, y0 = rng.randint(0, TILE), rng.randint(0, TILE)
        length = rng.randint(4, 9)
        x, y = x0, y0
        for _ in range(length):
            if 0 <= x < TILE and 0 <= y < TILE:
                arr[y, x, 0:3] = (70, 70, 72)
            x += rng.randint(-1, 2)
            y += rng.randint(-1, 2)
    _apply_vertical_shade(arr, strength=12)
    return Image.fromarray(arr, "RGBA")


def gen_cobblestone(rng=None):
    rng = rng or _rng("cobblestone")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    _add_noise(arr, rng, base=(120, 120, 122), variance=40)
    # blocky mortar seams
    for _ in range(5):
        x, y = rng.randint(0, TILE - 4), rng.randint(0, TILE - 4)
        w, h = rng.randint(3, 6), rng.randint(3, 6)
        arr[y:y + 1, x:x + w, 0:3] = (55, 55, 57)
        arr[y:y + h, x:x + 1, 0:3] = (55, 55, 57)
    _apply_vertical_shade(arr, strength=14)
    return Image.fromarray(arr, "RGBA")


def gen_wood_log_side(rng=None):
    rng = rng or _rng("wood_log_side")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    _add_noise(arr, rng, base=(108, 80, 48), variance=22)
    for x in range(0, TILE, 3):
        arr[:, x, 0:3] = (74, 53, 30)
    _apply_vertical_shade(arr, strength=10)
    return Image.fromarray(arr, "RGBA")


def gen_wood_log_top(rng=None):
    rng = rng or _rng("wood_log_top")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    _add_noise(arr, rng, base=(178, 142, 94), variance=16)
    cx, cy = TILE / 2, TILE / 2
    for y in range(TILE):
        for x in range(TILE):
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if int(dist) % 2 == 0 and dist > 1:
                arr[y, x, 0:3] = (128, 98, 62)
    return Image.fromarray(arr, "RGBA")


def gen_planks(rng=None):
    rng = rng or _rng("planks")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    _add_noise(arr, rng, base=(186, 148, 96), variance=20)
    for y in range(0, TILE, 4):
        arr[y, :, 0:3] = (132, 100, 62)
    _apply_vertical_shade(arr, strength=8)
    return Image.fromarray(arr, "RGBA")


def gen_leaves(rng=None):
    rng = rng or _rng("leaves")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    _add_noise(arr, rng, base=(52, 110, 40), variance=40, channel_variance=(30, 46, 26))
    # leaves have small transparent gaps so light "pokes through" a little
    gap_mask = rng.random((TILE, TILE)) < 0.05
    arr[:, :, 3] = 255
    arr[gap_mask, 3] = 0
    return Image.fromarray(arr, "RGBA")


def gen_coal_ore(stone_img, rng=None):
    rng = rng or _rng("coal_ore")
    arr = np.array(stone_img).copy()
    for _ in range(5):
        cx, cy = rng.randint(2, TILE - 2), rng.randint(2, TILE - 2)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if rng.random() < 0.6:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < TILE and 0 <= y < TILE:
                        arr[y, x, 0:3] = (24, 24, 26)
    return Image.fromarray(arr, "RGBA")


def gen_iron_ore(stone_img, rng=None):
    rng = rng or _rng("iron_ore")
    arr = np.array(stone_img).copy()
    for _ in range(5):
        cx, cy = rng.randint(2, TILE - 2), rng.randint(2, TILE - 2)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if rng.random() < 0.6:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < TILE and 0 <= y < TILE:
                        arr[y, x, 0:3] = (216, 178, 142)
    return Image.fromarray(arr, "RGBA")


def gen_crafting_table_top(rng=None):
    rng = rng or _rng("crafting_table_top")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    _add_noise(arr, rng, base=(150, 112, 66), variance=18)
    # a simple grid-square "saw pattern" motif to read as a workbench surface
    for i in (5, 10):
        arr[i, :, 0:3] = (96, 68, 40)
        arr[:, i, 0:3] = (96, 68, 40)
    # small corner "tool" accents
    arr[2:4, 2:4, 0:3] = (180, 180, 184)   # a little metal-looking square (saw blade hint)
    _apply_vertical_shade(arr, strength=8)
    return Image.fromarray(arr, "RGBA")


def gen_crafting_table_side(planks_img, rng=None):
    rng = rng or _rng("crafting_table_side")
    arr = np.array(planks_img).copy()
    # a darker horizontal band partway down to suggest a table edge/drawer line
    arr[10:12, :, 0:3] = (90, 65, 38)
    _apply_vertical_shade(arr, strength=6)
    return Image.fromarray(arr, "RGBA")


def gen_door_bottom(rng=None):
    rng = rng or _rng("door_bottom")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    _add_noise(arr, rng, base=(158, 118, 70), variance=16)
    # one recessed panel plus a doorknob, matching a door's BOTTOM half
    panel_y0 = 3
    panel_y1 = panel_y0 + 9
    arr[panel_y0:panel_y1, 2:14, 0:3] = (128, 92, 52)
    arr[panel_y0, 2:14, 0:3] = (170, 130, 80)
    arr[panel_y1 - 1, 2:14, 0:3] = (100, 70, 38)
    arr[8, 12, 0:3] = (222, 196, 90)  # doorknob
    _apply_vertical_shade(arr, strength=6)
    return Image.fromarray(arr, "RGBA")


def gen_door_top(rng=None):
    rng = rng or _rng("door_top")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    _add_noise(arr, rng, base=(158, 118, 70), variance=16)
    panel_y0 = 4
    panel_y1 = panel_y0 + 9
    arr[panel_y0:panel_y1, 2:14, 0:3] = (128, 92, 52)
    arr[panel_y0, 2:14, 0:3] = (170, 130, 80)
    arr[panel_y1 - 1, 2:14, 0:3] = (100, 70, 38)
    _apply_vertical_shade(arr, strength=6)
    return Image.fromarray(arr, "RGBA")


def gen_glass(rng=None):
    rng = rng or _rng("glass")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    # pale, mostly-transparent pane with a subtle pane-frame border and a
    # soft diagonal highlight streak, matching vanilla's clear-glass look
    arr[:, :, 0:3] = (225, 240, 245)
    arr[:, :, 3] = 60
    arr[0, :, 3] = 200
    arr[-1, :, 3] = 200
    arr[:, 0, 3] = 200
    arr[:, -1, 3] = 200
    arr[0, :, 0:3] = (200, 220, 225)
    arr[-1, :, 0:3] = (200, 220, 225)
    arr[:, 0, 0:3] = (200, 220, 225)
    arr[:, -1, 0:3] = (200, 220, 225)
    for i in range(TILE):
        x = i
        y = TILE - 1 - i
        if 0 <= x < TILE and 0 <= y < TILE:
            arr[y, x, 3] = min(255, int(arr[y, x, 3]) + 60)
            arr[y, x, 0:3] = (245, 250, 252)
    return Image.fromarray(arr, "RGBA")


def gen_sand(rng=None):
    rng = rng or _rng("sand")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    # Fine, high-frequency grain and a tight variance: sand's whole visual
    # identity is that it has no structure. Giving it the cracks or seams the
    # stone/cobble generators use would read as sandstone.
    _add_noise(arr, rng, base=(219, 207, 163), variance=22, channel_variance=(20, 20, 26))
    # a sparse scatter of slightly darker grains so large beaches don't band
    # into a flat colour when mipmapped down
    grain_mask = rng.random((TILE, TILE)) < 0.10
    arr[grain_mask, 0:3] = (198, 184, 140)
    _apply_vertical_shade(arr, strength=6)
    return Image.fromarray(arr, "RGBA")


def gen_water(rng=None):
    rng = rng or _rng("water")
    arr = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    # Deliberately almost featureless, and that is the whole point.
    #
    # The first version of this tile had bright ripple bands every 4 pixels. Up
    # close it looked fine; across an ocean it was a disaster - a high-contrast
    # pattern with a 4-pixel period, viewed at a grazing angle, is the textbook
    # worst case for moire. Each screen pixel covers many texels, the minifying
    # mip chain has nothing to fall back on but averages of alternating light
    # and dark rows, and the whole sea turns into a shimmering grey plaid. The
    # atlas uses GL_NEAREST_MIPMAP_LINEAR with no anisotropic filtering, so
    # there is nothing downstream to rescue it either.
    #
    # Real Minecraft water is a near-flat colour whose life comes from ANIMATION
    # and from the sky reflecting in it, not from detail baked into the tile.
    # Very low variance is what survives minification: as the mips average it
    # down, it converges on its own base colour instead of on grey mush.
    _add_noise(arr, rng, base=(50, 96, 178), variance=8, channel_variance=(6, 8, 10))
    arr[:, :, 3] = 190
    # No vertical shade either: _apply_vertical_shade would put a light row at
    # the top of every tile and a dark one at the bottom, which tiles into
    # exactly the horizontal banding described above.
    return Image.fromarray(arr, "RGBA")


# Face key -> generator (some depend on other tiles, e.g. grass_side needs dirt)
def build_all_tiles():
    """Returns dict[(block_id, face)] -> PIL.Image, one 16x16 tile each.

    Each entry tries assets/textures/blocks/<folder>/<face>.png first (see
    _load_block_face_png), and only falls back to the procedural generator
    below if no PNG has been drawn yet for that block/face - so tiles can
    be replaced with hand-drawn art one block at a time without breaking
    anything still using the placeholder generators.
    """
    dirt = gen_dirt()
    stone = gen_stone()
    planks = gen_planks()
    cobblestone = gen_cobblestone()

    procedural_tiles = {
        (Block.DIRT, "all"): dirt,
        (Block.GRASS, "up"): gen_grass_top(),
        (Block.GRASS, "side"): gen_grass_side(dirt),
        (Block.GRASS, "down"): dirt,
        (Block.STONE, "all"): stone,
        (Block.COBBLESTONE, "all"): cobblestone,
        (Block.WOOD_LOG, "side"): gen_wood_log_side(),
        (Block.WOOD_LOG, "up"): gen_wood_log_top(),
        (Block.WOOD_LOG, "down"): gen_wood_log_top(),
        (Block.PLANKS, "all"): planks,
        (Block.LEAVES, "all"): gen_leaves(),
        (Block.COAL_ORE, "all"): gen_coal_ore(stone),
        (Block.IRON_ORE, "all"): gen_iron_ore(stone),
        (Block.CRAFTING_TABLE, "up"): gen_crafting_table_top(),
        (Block.CRAFTING_TABLE, "down"): planks,
        (Block.CRAFTING_TABLE, "side"): gen_crafting_table_side(planks),
        (Block.DOOR, "all"): gen_door_bottom(),        # inventory/hotbar icon uses the bottom-half look
        (Block.DOOR, "bottom_half"): gen_door_bottom(),
        (Block.DOOR, "top_half"): gen_door_top(),
        (Block.FENCE, "all"): planks,
        (Block.STAIRS_WOOD, "all"): planks,
        (Block.STAIRS_STONE, "all"): cobblestone,
        (Block.GLASS, "all"): gen_glass(),
        (Block.WATER, "all"): gen_water(),
        (Block.SAND, "all"): gen_sand(),
    }

    tiles = {}
    for key, procedural_img in procedural_tiles.items():
        block_id, face = key
        disk_img = _load_block_face_png(block_id, face)
        tiles[key] = disk_img if disk_img is not None else procedural_img
    return tiles


class TextureAtlas:
    """
    Packs generated tiles into one big RGBA image, uploads it as a single
    OpenGL texture, and exposes UV rect lookups per (block_id, face).
    """

    def __init__(self):
        self.tiles = build_all_tiles()
        self.atlas_image = Image.new("RGBA", (ATLAS_COLS * PADDED_TILE, ATLAS_ROWS * PADDED_TILE), (0, 0, 0, 0))
        self.uv_rects = {}  # (block_id, face) -> (u0, v0, u1, v1)
        self.gl_texture_id = None
        self._pack()

    def _pack(self):
        """
        Packs tiles into the atlas with TILE_PADDING pixels of border-
        extruded padding around each one (the tile's own edge pixels
        duplicated outward, via PIL's ImageOps.expand + edge-replicate
        cropping) before pasting. This is what stops mipmap generation
        from blending a tile's edge with its NEIGHBOR in the atlas: mipmaps
        are built by downsampling the WHOLE packed atlas image, so a tile
        sitting directly against its neighbor with zero padding gets its
        lower mip levels' edge pixels averaged with whatever texture
        happens to be packed next to it - invisible up close (mip 0 is used
        at short view distances) but showing up as faint color bleeding/
        "shimmer" right at a block's edges as the camera moves away and a
        blended mip level kicks in, which is what was reported as
        conflicting/bleeding-together door textures. UV rects still map
        only to the REAL (non-padding) interior of each tile, so texture
        coordinates never change - only the atlas layout gains a safety
        margin around each tile.
        """
        slot_index = 0
        atlas_w, atlas_h = self.atlas_image.size
        for key, tile_img in self.tiles.items():
            col = slot_index % ATLAS_COLS
            row = slot_index // ATLAS_COLS
            slot_x0, slot_y0 = col * PADDED_TILE, row * PADDED_TILE

            padded_tile = _extrude_border(tile_img, TILE_PADDING)
            self.atlas_image.paste(padded_tile, (slot_x0, slot_y0))

            # The REAL tile content sits TILE_PADDING pixels in from the
            # slot origin - UV rects point at that interior region only.
            x0 = slot_x0 + TILE_PADDING
            y0 = slot_y0 + TILE_PADDING

            # UV v-coordinates are inverted here (1 - y/h) rather than y/h directly.
            # upload_to_gpu() flips the atlas image vertically (np.flipud) before
            # uploading, since PIL images are stored top-down but OpenGL texture
            # v=0 is conventionally the bottom row. Without this inversion, UV
            # rects computed from PIL's top-down pixel coordinates would sample
            # the WRONG part of the flipped GPU texture - concretely, tiles
            # packed into the top rows of the PIL image (where all real tile
            # data lives, since packing fills top-to-bottom) would end up
            # sampling the bottom rows of the GPU texture, which are still
            # empty/transparent since the atlas has far more slots than tiles.
            # This exact bug caused the hand viewmodel to render solid black
            # despite the texture atlas itself containing valid tile data.
            u0 = x0 / atlas_w
            u1 = (x0 + TILE) / atlas_w
            v0 = 1.0 - (y0 + TILE) / atlas_h
            v1 = 1.0 - y0 / atlas_h
            self.uv_rects[key] = (u0, v0, u1, v1)
            slot_index += 1

    def uv_for(self, block_id: int, face: str):
        """face is 'up' | 'side' | 'down'. Falls back to 'all' if a block has one shared texture."""
        if (block_id, face) in self.uv_rects:
            return self.uv_rects[(block_id, face)]
        return self.uv_rects.get((block_id, "all"))

    def upload_to_gpu(self):
        """Creates the OpenGL texture object from the packed atlas image. Call once a GL context exists."""
        data = np.array(self.atlas_image.convert("RGBA"))
        data = np.flipud(data)  # PIL is top-down, OpenGL texture origin is bottom-left
        data = np.ascontiguousarray(data)

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, data.shape[1], data.shape[0], 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data)
        glGenerateMipmap(GL_TEXTURE_2D)

        self.gl_texture_id = tex_id
        return tex_id

    def bind(self, unit: int = 0):
        from OpenGL.GL import glActiveTexture, GL_TEXTURE0
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.gl_texture_id)
