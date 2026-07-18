"""
ui/icon_cache.py
Resolves inventory/hotbar/held-item icons in this priority order:
  1. A hand-drawn PNG under assets/textures/items/ (see item_texture_map.py
     for the item_id -> filename mapping) - this is where you drop your own
     art for Infdev 2.0.
  2. The old procedural icon (simple Pillow-drawn shapes for tools/door/
     fence/stairs/glass) - used automatically for any item that doesn't
     have a PNG yet, so nothing ever renders blank while you're still
     drawing textures.
  3. The block's in-world texture-atlas tile ("up" or "all" face) - used
     for ordinary full-cube blocks that have neither a custom PNG nor a
     procedural icon.
  4. A magenta placeholder square, only if all of the above somehow fail.

PNG files are loaded once and cached alongside the final pygame.Surface,
so drawing a slot never re-reads disk or re-converts Pillow->pygame.
"""

import os
import math
import pygame
from PIL import Image, ImageDraw

import paths
from world.blocks import Item, Block
from ui.item_texture_map import ITEM_TEXTURE_FILENAMES

_icon_cache = {}
_pil_file_cache = {}  # item_id -> PIL.Image, loaded from disk once

# See render/texture_atlas.py's BLOCK_TEXTURES_DIR: a relative path here reads
# from wherever the exe was launched, finds nothing, and silently falls back to
# the procedural icons.
ITEM_TEXTURES_DIR = paths.asset_path("assets", "textures", "items")


def _load_png_icon(item_id: int):
    """Returns a PIL.Image loaded from assets/textures/items/, or None if
    no filename is mapped for this item or the file doesn't exist yet."""
    if item_id in _pil_file_cache:
        return _pil_file_cache[item_id]

    filename = ITEM_TEXTURE_FILENAMES.get(item_id)
    if filename is None:
        return None

    path = os.path.join(ITEM_TEXTURES_DIR, filename)
    if not os.path.isfile(path):
        return None

    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return None  # corrupt/unreadable file - fall back rather than crash

    _pil_file_cache[item_id] = img
    return img


def _get_face_tile(texture_atlas, block_id, tex_face):
    """Returns the block''s actual world/atlas tile for a given face key
    ("up"/"side"/"down"), falling back to "all" - the SAME lookup priority
    render/hand.py uses for the held-item mesh, so an isometric inventory
    icon uses pixel-identical source art to the hand/world."""
    return (texture_atlas.tiles.get((block_id, tex_face))
            or texture_atlas.tiles.get((block_id, "all")))


def _find_perspective_coeffs(src_quad, dst_quad):
    """Solves for the 8 coefficients of a perspective transform mapping
    each point in src_quad to the corresponding point in dst_quad - used to
    build the PIL PERSPECTIVE transform''s `data`, which (unlike QUAD)
    takes a source-to-destination-preimage mapping we can construct
    directly with this, rather than QUAD''s harder-to-reason-about implicit
    inversion."""
    matrix = []
    for (sx, sy), (dx, dy) in zip(dst_quad, src_quad):
        matrix.append([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
        matrix.append([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
    A = matrix
    B = [c for corner in src_quad for c in corner]

    # Solve the 8x8 linear system A @ coeffs = B via Gaussian elimination
    # (no numpy dependency needed for an 8x8 solve).
    n = 8
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[pivot_row][col]) < 1e-12:
            return None
        A[col], A[pivot_row] = A[pivot_row], A[col]
        B[col], B[pivot_row] = B[pivot_row], B[col]
        pivot = A[col][col]
        A[col] = [v / pivot for v in A[col]]
        B[col] /= pivot
        for r in range(n):
            if r == col:
                continue
            factor = A[r][col]
            if factor == 0:
                continue
            A[r] = [A[r][k] - factor * A[col][k] for k in range(n)]
            B[r] -= factor * B[col]
    return B


def _warp_face(img, quad):
    """Perspective-warps `img` (a square tile, corners in order top-left,
    top-right, bottom-right, bottom-left) onto an arbitrary quad in 2D
    space, returning (warped_image, paste_position)."""
    w, h = img.size
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    bbox_w = max(1, int(round(bbox[2] - bbox[0])))
    bbox_h = max(1, int(round(bbox[3] - bbox[1])))
    local_quad = [(p[0] - bbox[0], p[1] - bbox[1]) for p in quad]

    src_corners = [(0, 0), (w, 0), (w, h), (0, h)]
    coeffs = _find_perspective_coeffs(src_corners, local_quad)
    if coeffs is None:
        return img.resize((bbox_w, bbox_h), Image.NEAREST), (int(round(bbox[0])), int(round(bbox[1])))

    warped = img.transform((bbox_w, bbox_h), Image.PERSPECTIVE, coeffs, resample=Image.NEAREST)
    return warped, (int(round(bbox[0])), int(round(bbox[1])))


def _shade_tile(img, factor):
    """Applies brightness shading AND forces full opacity - side/top tiles
    occasionally carry partial alpha in their source pixels (anti-aliased
    edges, etc.) which is fine for a texture sampled across a whole cube
    face in 3D, but at icon scale it read as the icon itself being
    translucent. Solid block faces should always be fully opaque."""
    arr = img.convert("RGBA")
    px = arr.load()
    w, h = arr.size
    out = Image.new("RGBA", (w, h))
    opx = out.load()
    for yy in range(h):
        for xx in range(w):
            r, g, b, a = px[xx, yy]
            opx[xx, yy] = (min(255, int(r * factor)), min(255, int(g * factor)), min(255, int(b * factor)), 255)
    return out


def render_isometric_block_icon(texture_atlas, block_id, boxes, size=64):
    """
    Renders a true isometric 3-face (top/left/right) icon of `boxes` (the
    same (min_x,min_y,min_z,max_x,max_y,max_z) box list format
    render/hand.py''s _held_boxes_for_block uses) using the block''s actual
    world textures, via a simple fixed isometric projection. This is what
    makes the inventory icon for door/fence/stairs genuinely match the
    real 3D geometry shown in the hand/world, instead of a hand-drawn flat
    placeholder icon with only a passing resemblance.

    Thin objects (a door''s 0.1875-thick slab, a fence''s 0.25-thick post)
    would otherwise render as a tiny sliver if projected at a fixed scale
    tuned for a full 1x1x1 cube - so this renders at a large WORKING
    resolution first, measures the actual painted bounding box, then crops
    and rescales that content to fill the final `size`x`size` canvas with a
    small margin, regardless of how thin/oddly-proportioned the boxes are.
    """
    work_size = size * 3  # generous working canvas so thin geometry still has enough source pixels to crop from cleanly
    canvas = Image.new("RGBA", (work_size, work_size), (0, 0, 0, 0))

    cx, cy = work_size * 0.5, work_size * 0.28
    scale = work_size * 0.8

    def project(x, y, z):
        sx = cx + (x - z) * scale * 0.5
        sy = cy + (x + z) * scale * 0.25 - y * scale * 0.5
        return sx, sy

    # Draw boxes back-to-front so nearer geometry correctly paints over
    # farther geometry - depth in THIS fixed isometric camera (looking
    # from +x,+y,+z toward the origin) is proportional to (x+y+z) of each
    # box's own far corner, so boxes with a SMALLER max-corner sum are
    # further away and must be drawn first. The previous sort key ignored
    # the Y axis entirely, which correctly ordered full-height blocks but
    # silently misordered anything using Y-split sub-boxes (stairs' tread/
    # riser split) - producing a jumbled, broken-looking silhouette instead
    # of a clean stepped shape.
    boxes_sorted = sorted(boxes, key=lambda b: (b[3] + b[4] + b[5]))

    for (bx0, by0, bz0, bx1, by1, bz1) in boxes_sorted:
        top_tile = _get_face_tile(texture_atlas, block_id, "up")
        side_tile = _get_face_tile(texture_atlas, block_id, "side")
        if top_tile is None or side_tile is None:
            continue

        top_quad = [
            project(bx0, by1, bz0), project(bx1, by1, bz0),
            project(bx1, by1, bz1), project(bx0, by1, bz1),
        ]
        warped, pos = _warp_face(_shade_tile(top_tile, 1.0), top_quad)
        canvas.alpha_composite(warped, pos)

        right_quad = [
            project(bx1, by1, bz0), project(bx1, by1, bz1),
            project(bx1, by0, bz1), project(bx1, by0, bz0),
        ]
        warped, pos = _warp_face(_shade_tile(side_tile, 0.75), right_quad)
        canvas.alpha_composite(warped, pos)

        left_quad = [
            project(bx0, by1, bz1), project(bx1, by1, bz1),
            project(bx1, by0, bz1), project(bx0, by0, bz1),
        ]
        warped, pos = _warp_face(_shade_tile(side_tile, 0.55), left_quad)
        canvas.alpha_composite(warped, pos)

    bbox = canvas.getbbox()
    if bbox is None:
        return Image.new("RGBA", (size, size), (0, 0, 0, 0))

    content = canvas.crop(bbox)
    content_w, content_h = content.size
    margin = 0.03  # fraction of final size left as empty border, like standard inventory icons
    target_dim = size * (1.0 - 2 * margin)
    fit_scale = min(target_dim / content_w, target_dim / content_h)
    new_w = max(1, int(round(content_w * fit_scale)))
    new_h = max(1, int(round(content_h * fit_scale)))
    resized = content.resize((new_w, new_h), Image.NEAREST)

    final = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    final.alpha_composite(resized, (paste_x, paste_y))
    return final


_ISOMETRIC_ICON_CACHE = {}  # block_id -> PIL.Image, built once per texture_atlas


def _get_isometric_icon(texture_atlas, block_id):
    """Builds (once, cached) a true 3D-matching isometric icon for
    door/fence/stairs, reusing the exact same box geometry render/hand.py
    uses for the held item - so the inventory slot icon and the in-hand
    model are the same shape, not a flat drawn approximation."""
    key = (id(texture_atlas), block_id)
    if key in _ISOMETRIC_ICON_CACHE:
        return _ISOMETRIC_ICON_CACHE[key]

    from render.hand import _held_boxes_for_block
    boxes = _held_boxes_for_block(block_id)
    icon = render_isometric_block_icon(texture_atlas, block_id, boxes, size=64)
    _ISOMETRIC_ICON_CACHE[key] = icon
    return icon


_ISOMETRIC_ICON_BLOCKS = frozenset((Block.DOOR, Block.FENCE, Block.STAIRS_WOOD, Block.STAIRS_STONE))


def _draw_stick_icon(size=16):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.line([(3, 13), (13, 3)], fill=(122, 90, 52, 255), width=2)
    draw.line([(4, 12), (12, 4)], fill=(92, 64, 35, 255), width=1)
    return img


def _draw_coal_icon(size=16):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, 11, 11], fill=(28, 28, 28, 255))
    draw.rectangle([3, 6, 4, 9], fill=(28, 28, 28, 255))
    draw.rectangle([11, 6, 12, 9], fill=(28, 28, 28, 255))
    draw.rectangle([5, 5, 7, 7], fill=(58, 58, 58, 255))
    return img


def _draw_tool_icon(head_color, handle_color, shape, size=16):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.line([(3, 13), (9, 7)], fill=handle_color, width=2)
    if shape == "pickaxe":
        draw.rectangle([7, 2, 13, 4], fill=head_color)
        draw.rectangle([9, 4, 11, 6], fill=head_color)
    elif shape == "axe":
        draw.rectangle([8, 2, 13, 6], fill=head_color)
    elif shape == "shovel":
        draw.rectangle([9, 2, 12, 5], fill=head_color)
    elif shape == "sword":
        draw.rectangle([7, 2, 9, 9], fill=head_color)
        draw.rectangle([5, 8, 11, 10], fill=handle_color)  # crossguard
    return img


def _draw_door_icon(size=16):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 1, 11, 14], fill=(168, 128, 78, 255), outline=(110, 80, 46, 255))
    draw.rectangle([5, 3, 10, 7], fill=(140, 104, 60, 255))
    draw.rectangle([5, 8, 10, 12], fill=(140, 104, 60, 255))
    draw.point((9, 8), fill=(230, 200, 100, 255))
    return img


def _draw_fence_icon(size=16):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    post_color = (150, 112, 66, 255)
    rail_color = (120, 88, 50, 255)
    draw.rectangle([6, 1, 9, 14], fill=post_color)
    draw.rectangle([1, 4, 14, 6], fill=rail_color)
    draw.rectangle([1, 9, 14, 11], fill=rail_color)
    return img


def _draw_stairs_icon(base_color, size=16):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    dark = tuple(max(0, c - 40) for c in base_color[:3]) + (255,)
    draw.rectangle([1, 10, 14, 14], fill=base_color)
    draw.rectangle([6, 5, 14, 9], fill=base_color)
    draw.rectangle([1, 10, 14, 11], fill=dark)
    draw.rectangle([6, 5, 14, 6], fill=dark)
    return img


def _draw_glass_icon(size=16):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, 13, 13], fill=(225, 240, 245, 90), outline=(190, 210, 218, 255))
    draw.line([(2, 7), (13, 7)], fill=(190, 210, 218, 255), width=1)
    draw.line([(7, 2), (7, 13)], fill=(190, 210, 218, 255), width=1)
    draw.line([(3, 12), (11, 4)], fill=(255, 255, 255, 140), width=1)
    return img


_WOOD_TOOL_COLOR = (169, 124, 80, 255)
_STONE_TOOL_COLOR = (154, 154, 154, 255)
_HANDLE_COLOR = (122, 90, 52, 255)
_PLANKS_ICON_COLOR = (186, 148, 96)
_COBBLESTONE_ICON_COLOR = (120, 120, 122)

_PROCEDURAL_ICON_GENERATORS = {
    Item.STICK: lambda size: _draw_stick_icon(size),
    Item.COAL: lambda size: _draw_coal_icon(size),
    Item.WOODEN_PICKAXE: lambda size: _draw_tool_icon(_WOOD_TOOL_COLOR, _HANDLE_COLOR, "pickaxe", size),
    Item.STONE_PICKAXE: lambda size: _draw_tool_icon(_STONE_TOOL_COLOR, _HANDLE_COLOR, "pickaxe", size),
    Item.WOODEN_AXE: lambda size: _draw_tool_icon(_WOOD_TOOL_COLOR, _HANDLE_COLOR, "axe", size),
    Item.WOODEN_SHOVEL: lambda size: _draw_tool_icon(_WOOD_TOOL_COLOR, _HANDLE_COLOR, "shovel", size),
    Item.STONE_AXE: lambda size: _draw_tool_icon(_STONE_TOOL_COLOR, _HANDLE_COLOR, "axe", size),
    Item.STONE_SHOVEL: lambda size: _draw_tool_icon(_STONE_TOOL_COLOR, _HANDLE_COLOR, "shovel", size),
    Item.WOODEN_SWORD: lambda size: _draw_tool_icon(_WOOD_TOOL_COLOR, _HANDLE_COLOR, "sword", size),
    Item.STONE_SWORD: lambda size: _draw_tool_icon(_STONE_TOOL_COLOR, _HANDLE_COLOR, "sword", size),
}


def _resolve_source_image(texture_atlas, item_id: int) -> Image.Image:
    """Picks the base image for this item, following the priority order
    documented at the top of this file. door/fence/stairs specifically get
    a real isometric render of their actual 3D geometry (matching the hand/
    world) rather than falling through to a flat placeholder, UNLESS a
    hand-drawn PNG has been provided (which always wins - an artist's own
    icon should never be silently replaced)."""
    png_icon = _load_png_icon(item_id)
    if png_icon is not None:
        return png_icon

    if item_id in _ISOMETRIC_ICON_BLOCKS:
        return _get_isometric_icon(texture_atlas, item_id)

    if item_id in _PROCEDURAL_ICON_GENERATORS:
        return _PROCEDURAL_ICON_GENERATORS[item_id](16)

    tile_img = (texture_atlas.tiles.get((item_id, "up"))
                or texture_atlas.tiles.get((item_id, "all"))
                or texture_atlas.tiles.get((item_id, "side")))
    if tile_img is not None:
        return tile_img

    candidates = [img for (bid, face), img in texture_atlas.tiles.items() if bid == item_id]
    if candidates:
        return candidates[0]

    return Image.new("RGBA", (16, 16), (255, 0, 255, 255))  # unmistakable "missing icon" placeholder


def get_item_icon_pil(texture_atlas, item_id: int, size: int = None) -> Image.Image:
    """
    Returns a raw PIL Image for the given item/block id. If `size` is None
    (the default), returns the image at its NATIVE resolution - important
    for the held-item viewmodel (render/hand.py), which used to always
    request a forced 16x16 copy and silently downscale any higher-res
    artwork (32x32, 64x64, etc.) before it ever reached the GPU, making
    detailed hand-drawn textures look soft/blurry in-hand even though the
    same file looked crisp in the inventory. Only pass an explicit `size`
    when you actually need a specific resolution (e.g. matching a fixed UI
    slot size); resizing still uses nearest-neighbor so pixel art stays
    crisp when a resize IS requested.
    """
    tile_img = _resolve_source_image(texture_atlas, item_id)
    if size is not None and size != tile_img.size[0]:
        tile_img = tile_img.resize((size, size), Image.NEAREST)
    return tile_img


def get_item_icon(texture_atlas, item_id: int, size: int = 28):
    key = (id(texture_atlas), item_id, size)
    if key in _icon_cache:
        return _icon_cache[key]

    tile_img = _resolve_source_image(texture_atlas, item_id)
    resized = tile_img.resize((size, size), Image.NEAREST)
    mode = resized.mode
    raw = resized.tobytes()
    surf = pygame.image.fromstring(raw, resized.size, mode).convert_alpha()

    _icon_cache[key] = surf
    return surf


def clear_icon_cache():
    """Clears both the pygame.Surface cache and the on-disk PNG cache - call
    this if you edit a texture PNG and want to see the change without
    restarting the game (e.g. wire it to a debug/reload hotkey)."""
    _icon_cache.clear()
    _pil_file_cache.clear()

