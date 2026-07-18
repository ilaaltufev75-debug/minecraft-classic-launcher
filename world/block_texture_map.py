"""
world/block_texture_map.py
Maps every block id + face ("up"/"side"/"down"/"all") to the PNG file it
should load from assets/textures/blocks/<block_folder>/<face>.png.

Each block folder can contain up to 3 files:
  up.png    - top face (e.g. grass top, log rings, crafting table saw top)
  side.png  - the 4 side faces (most blocks only need this one)
  down.png  - bottom face (rarely different from the side; grass uses dirt)
  all.png   - single fallback used for any face without its own file
              (most blocks - stone, cobblestone, planks, glass, etc. -
              only need this one file since all 6 faces look the same)

You do not need all 4 files for every block. render/texture_atlas.py tries,
per face: up/side/down -> all -> the old procedural generator, in that
order, so a block with only all.png still renders correctly on every face,
and a block with no PNG at all keeps using its current placeholder texture.
"""

from world.blocks import Block

# block_id -> folder name under assets/textures/blocks/
BLOCK_TEXTURE_FOLDERS = {
    Block.GRASS: "grass",
    Block.DIRT: "dirt",
    Block.STONE: "stone",
    Block.COBBLESTONE: "cobblestone",
    Block.WOOD_LOG: "wood_log",
    Block.PLANKS: "planks",
    Block.COAL_ORE: "coal_ore",
    Block.IRON_ORE: "iron_ore",
    Block.LEAVES: "leaves",
    Block.CRAFTING_TABLE: "crafting_table",
    Block.DOOR: "door",
    Block.FENCE: "fence",
    Block.STAIRS_WOOD: "stairs_wood",
    Block.STAIRS_STONE: "stairs_stone",
    Block.GLASS: "glass",
    Block.WATER: "water",
    Block.SAND: "sand",
}

# Which faces each block actually needs art for (informational - used by
# tooling/README generation, not required at runtime). Blocks not listed
# here just need "all".
BLOCK_FACES_NEEDED = {
    Block.GRASS: ["up", "side", "down"],       # 3 different textures: green top, dirt+green fringe side, dirt bottom
    Block.WOOD_LOG: ["up", "side"],            # rings on top/bottom (same image), bark on the sides
    Block.CRAFTING_TABLE: ["up", "side", "down"],
    Block.DOOR: ["side"],                      # rendered as a custom thin slab, only needs one face image
}
