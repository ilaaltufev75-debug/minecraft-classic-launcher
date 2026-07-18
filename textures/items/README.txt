Drop PNG files here to replace the built-in icons - inventory, hotbar,
and the held-item view all update automatically, no code changes needed.

RULES
- Size: 16x16 px (32x32 also works, just keep it square and a power of 2).
- Format: PNG, RGBA (transparent background outside the item's silhouette).
- Style: flat pixel art, no anti-aliasing / no soft edges (matches the
  game's nearest-neighbor upscaling - smooth edges will look blurry).
- Exact filename required - see ui/item_texture_map.py for the full list.

CURRENTLY MISSING (falls back to a simple built-in icon until you add these):
  stick.png
  coal.png
  wooden_pickaxe.png
  stone_pickaxe.png
  wooden_axe.png
  wooden_shovel.png
  stone_axe.png
  stone_shovel.png
  wooden_sword.png
  stone_sword.png
  door.png
  fence.png
  stairs_wood.png
  stairs_stone.png
  glass.png

OPTIONAL (these already look fine using the in-world block texture; only
add a PNG here if you want the inventory icon to look different from the
block placed in the world):
  grass.png
  dirt.png
  stone.png
  cobblestone.png
  wood_log.png
  planks.png
  coal_ore.png
  iron_ore.png
  leaves.png
  crafting_table.png

To add a NEW item later: add its filename to ITEM_TEXTURE_FILENAMES in
ui/item_texture_map.py, then drop the PNG here.
