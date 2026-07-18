"""
ui/item_texture_map.py
Maps every item/block id to the PNG filename it should load from
assets/textures/items/ for inventory-slot and held-item (hand) rendering.

To add or replace an icon:
  1. Draw/save a 16x16 (or 32x32) PNG with a transparent background under
     assets/textures/items/, named exactly as listed below.
  2. That's it - ui/icon_cache.py picks it up automatically next launch.
     No code changes needed.

If a file is missing, ui/icon_cache.py silently falls back to the old
procedural icon (or the block''s world texture) so the game never crashes
or shows a blank/magenta icon just because a texture has not been drawn yet.
"""

from world.blocks import Block, Item

ITEM_TEXTURE_FILENAMES = {
    # tools
    Item.STICK: "stick.png",
    Item.COAL: "coal.png",
    Item.WOODEN_PICKAXE: "wooden_pickaxe.png",
    Item.STONE_PICKAXE: "stone_pickaxe.png",
    Item.WOODEN_AXE: "wooden_axe.png",
    Item.WOODEN_SHOVEL: "wooden_shovel.png",
    Item.STONE_AXE: "stone_axe.png",
    Item.STONE_SHOVEL: "stone_shovel.png",
    Item.WOODEN_SWORD: "wooden_sword.png",
    Item.STONE_SWORD: "stone_sword.png",

    # non-cube blocks (the ones that currently use a procedural icon
    # instead of a plain world-texture swatch)
    Block.DOOR: "door.png",
    Block.FENCE: "fence.png",
    Block.STAIRS_WOOD: "stairs_wood.png",
    Block.STAIRS_STONE: "stairs_stone.png",
    Block.GLASS: "glass.png",

    # ordinary full-cube blocks - optional. These already get a fine
    # default icon from their world texture (texture_atlas tiles), but if
    # you want a hand-drawn inventory icon that differs from the in-world
    # block texture, add a PNG here and it will be used instead.
    Block.GRASS: "grass.png",
    Block.DIRT: "dirt.png",
    Block.STONE: "stone.png",
    Block.COBBLESTONE: "cobblestone.png",
    Block.WOOD_LOG: "wood_log.png",
    Block.PLANKS: "planks.png",
    Block.COAL_ORE: "coal_ore.png",
    Block.IRON_ORE: "iron_ore.png",
    Block.LEAVES: "leaves.png",
    Block.CRAFTING_TABLE: "crafting_table.png",
}
