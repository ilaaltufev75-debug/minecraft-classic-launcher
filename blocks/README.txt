Drop PNG files here to replace the in-world block textures (walls, ground,
placed blocks — everything you see while walking around). Each block gets
its own subfolder; the inventory/hand icons are a SEPARATE system, see
assets/textures/items/README.txt for those.

RULES
- Size: 16x16 px (anything else gets auto-resized on load, but draw at
  16x16 for full pixel control).
- Format: PNG, RGBA. Fully opaque unless the block is meant to be see-
  through (glass, leaves).
- Style: flat pixel art, no anti-aliasing (the game upscales with nearest-
  neighbor sampling, so soft/blurred edges will look wrong in-game).

FILES PER FOLDER
Most blocks only need ONE file:
  all.png   - used on every face (top, bottom, all 4 sides)

Some blocks look different from different angles and need up to 3 files
instead of all.png:
  up.png    - top face
  side.png  - the 4 side faces
  down.png  - bottom face
If a face-specific file is missing, that face falls back to all.png, so
you can add these one at a time.

WHICH BLOCKS NEED WHICH FILES
  grass/           up.png, side.png, down.png   (green top, dirt+grass-fringe side, dirt bottom)
  dirt/            all.png
  stone/           all.png
  cobblestone/     all.png
  wood_log/        up.png (rings, also used for bottom), side.png (bark)
  planks/          all.png
  coal_ore/        all.png
  iron_ore/        all.png
  leaves/          all.png   (needs transparency in the gaps if you want light to poke through)
  crafting_table/  up.png (saw pattern), side.png, down.png (or just all.png = planks look)
  door/            side.png  (this block is rendered as a thin slab, not a
                    cube — one texture covers it; see NOTE below)
  fence/           all.png   (rendered as post+rails, not a cube — texture
                    is applied like a wood material, same idea as door)
  stairs_wood/     all.png
  stairs_stone/    all.png
  glass/           all.png   (must use partial alpha to actually look
                    like glass — fully opaque pixels will look like a
                    solid colored block instead)

NOTE ON DOOR: in-world, the door is drawn with a SEPARATE two-texture
system (bottom half has a doorknob, top half doesn't) that is independent
of this folder — that one is not wired to file loading yet. side.png here
currently only feeds the door's INVENTORY icon fallback. Ask before
drawing door textures if you want them to show correctly in-world too;
that needs a small code change on my end first.

To add a texture for a NEW block later: add it to
world/block_texture_map.py (BLOCK_TEXTURE_FOLDERS), create the matching
folder here, then drop the PNG(s) in.
