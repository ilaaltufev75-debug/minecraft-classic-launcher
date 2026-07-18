"""
world/blocks.py
Block and item ID registry, ported from the original JS build's Game.BLOCK /
Game.ITEM / Game.ITEMS tables. Block IDs double as item IDs (an inventory
slot just stores an int id + count), matching how Minecraft Alpha/Infdev
unified blocks and items under one numeric space.

Only materials that existed in the Alpha/Infdev era are included: dirt,
grass, stone, cobblestone, wood log, planks, leaves (new: needed for trees),
coal ore, iron ore, sticks, coal, and basic wood/stone tools.
"""

from dataclasses import dataclass
from typing import Optional


class Block:
    AIR = 0
    GRASS = 1
    DIRT = 2
    STONE = 3
    COBBLESTONE = 4
    WOOD_LOG = 5
    PLANKS = 6
    COAL_ORE = 7
    IRON_ORE = 8
    LEAVES = 9
    CRAFTING_TABLE = 10
    DOOR = 11
    FENCE = 12
    STAIRS_WOOD = 13
    STAIRS_STONE = 14
    GLASS = 15
    WATER = 16
    SAND = 17


class Item:
    STICK = 100
    COAL = 101
    WOODEN_PICKAXE = 102
    STONE_PICKAXE = 103
    WOODEN_AXE = 104
    WOODEN_SHOVEL = 105
    STONE_AXE = 106
    STONE_SHOVEL = 107
    WOODEN_SWORD = 108
    STONE_SWORD = 109


@dataclass(frozen=True)
class ItemDef:
    name: str
    is_block: bool = False
    stack: int = 64
    hardness: Optional[float] = None       # None = block has no meaningful break time entry
    tool: Optional[str] = None              # required tool type ('pickaxe' | 'axe' | 'shovel') or None
    drops: Optional[int] = None             # id dropped on break, if different from the block itself
    tool_type: Optional[str] = None         # for tool items: what kind of tool this is
    tool_tier: int = 0                      # for tool items: higher = faster mining
    solid: bool = True                      # whether the block blocks player movement / occludes neighbors
    transparent: bool = False               # whether neighbors should still render a face against this block
                                             # (used for leaves so light/other leaves show detail; kept simple:
                                             # transparent blocks are rendered but don't cull neighbor faces)


ITEMS = {
    Block.GRASS: ItemDef("Grass Block", is_block=True, stack=64, hardness=0.6, tool="shovel"),
    Block.DIRT: ItemDef("Dirt", is_block=True, stack=64, hardness=0.5, tool="shovel"),
    Block.STONE: ItemDef("Stone", is_block=True, stack=64, hardness=1.5, tool="pickaxe", drops=Block.COBBLESTONE),
    Block.COBBLESTONE: ItemDef("Cobblestone", is_block=True, stack=64, hardness=2.0, tool="pickaxe"),
    Block.WOOD_LOG: ItemDef("Wood Log", is_block=True, stack=64, hardness=2.0, tool="axe"),
    Block.PLANKS: ItemDef("Wooden Planks", is_block=True, stack=64, hardness=2.0, tool="axe"),
    Block.COAL_ORE: ItemDef("Coal Ore", is_block=True, stack=64, hardness=3.0, tool="pickaxe", drops=Item.COAL),
    Block.IRON_ORE: ItemDef("Iron Ore", is_block=True, stack=64, hardness=3.0, tool="pickaxe"),
    Block.LEAVES: ItemDef("Leaves", is_block=True, stack=64, hardness=0.2, tool=None,
                           solid=True, transparent=True),
    Block.CRAFTING_TABLE: ItemDef("Crafting Table", is_block=True, stack=64, hardness=2.5, tool="axe"),
    # Door: solid=False here is the "default"/item-palette value only. Real
    # in-world collision is state-dependent (open vs closed) and is resolved
    # dynamically by world.is_block_passable() / physics, not this static
    # flag - see world/doors.py. transparent=True keeps it out of normal
    # opaque-face-culling math; it never enters the regular chunk mesh at
    # all (see world/chunk.py CUSTOM_RENDER_BLOCKS) since it needs a thin,
    # rotated slab shape a full-cube mesher can't produce.
    Block.DOOR: ItemDef("Wooden Door", is_block=True, stack=64, hardness=3.0, tool="axe",
                         solid=False, transparent=True),
    # Fence: solid=False by default flag - real collision is computed from
    # its connection state (see world/fences.py), same pattern as doors.
    Block.FENCE: ItemDef("Fence", is_block=True, stack=64, hardness=2.0, tool="axe",
                          solid=False, transparent=True),
    # Stairs: solid=False by default flag - real collision uses the shape's
    # sub-cell boxes (see world/stairs.py).
    Block.STAIRS_WOOD: ItemDef("Wooden Stairs", is_block=True, stack=64, hardness=2.0, tool="axe",
                                solid=False, transparent=True),
    Block.STAIRS_STONE: ItemDef("Stone Stairs", is_block=True, stack=64, hardness=2.0, tool="pickaxe",
                                 solid=False, transparent=True),
    Block.GLASS: ItemDef("Glass", is_block=True, stack=64, hardness=0.3, tool=None, transparent=True),
    # Water: solid=False - it never enters the AABB collision path at all;
    # swimming is handled separately in player/physics.py against the fluid's
    # own height (see world/fluids.py collision_top). transparent=True keeps it
    # out of OPAQUE_BLOCKS, so the seabed underneath it still renders its top
    # faces instead of being culled away. hardness=None: fluids are not
    # breakable, and with no bucket yet there is nothing that legitimately
    # removes one.
    Block.WATER: ItemDef("Water", is_block=True, stack=64, hardness=None, tool=None,
                          solid=False, transparent=True),
    Block.SAND: ItemDef("Sand", is_block=True, stack=64, hardness=0.5, tool="shovel"),

    Item.STICK: ItemDef("Stick", is_block=False, stack=64),
    Item.COAL: ItemDef("Coal", is_block=False, stack=64),
    Item.WOODEN_PICKAXE: ItemDef("Wooden Pickaxe", is_block=False, stack=1, tool_type="pickaxe", tool_tier=1),
    Item.STONE_PICKAXE: ItemDef("Stone Pickaxe", is_block=False, stack=1, tool_type="pickaxe", tool_tier=2),
    Item.WOODEN_AXE: ItemDef("Wooden Axe", is_block=False, stack=1, tool_type="axe", tool_tier=1),
    Item.WOODEN_SHOVEL: ItemDef("Wooden Shovel", is_block=False, stack=1, tool_type="shovel", tool_tier=1),
    Item.STONE_AXE: ItemDef("Stone Axe", is_block=False, stack=1, tool_type="axe", tool_tier=2),
    Item.STONE_SHOVEL: ItemDef("Stone Shovel", is_block=False, stack=1, tool_type="shovel", tool_tier=2),
    Item.WOODEN_SWORD: ItemDef("Wooden Sword", is_block=False, stack=1, tool_type="sword", tool_tier=1),
    Item.STONE_SWORD: ItemDef("Stone Sword", is_block=False, stack=1, tool_type="sword", tool_tier=2),
}

# Blocks that need custom (non-full-cube) rendering/collision and must NOT
# be handed to the regular vectorized chunk mesher or the generic is_solid()
# static-flag path. Each such block has its own dedicated renderer + state
# system (see world/doors.py, render/door_renderer.py for doors).
#
# WATER is in here for the same structural reason the others are: it is not a
# cube. A water cell's top sits at a height derived from its own level and from
# its four diagonal neighbours' levels (see render/water_renderer.py), so its
# surface slopes down a hillside instead of stepping. A full-cube mesher cannot
# express that shape at all, and just as importantly the water pass needs its
# own shader - depth-driven colour, a distance-faded ripple and a sky term are
# what separate an ocean from a flat blue lid, and none of them belong in the
# shared block shader.
CUSTOM_RENDER_BLOCKS = frozenset((Block.DOOR, Block.FENCE, Block.STAIRS_WOOD, Block.STAIRS_STONE,
                                   Block.WATER))

# Blocks that occlude neighbor faces during meshing (i.e. are opaque and solid).
# Leaves are intentionally excluded so they render properly against each other/air.
# Custom-render blocks (doors) are always excluded too - they're drawn by
# their own renderer, never as part of the chunk's cube mesh.
OPAQUE_BLOCKS = frozenset(
    block_id for block_id, item_def in ITEMS.items()
    if item_def.is_block and not item_def.transparent and block_id not in CUSTOM_RENDER_BLOCKS
)


def get_item_def(item_id: int) -> Optional[ItemDef]:
    return ITEMS.get(item_id)


def get_stack_size(item_id: int) -> int:
    item_def = ITEMS.get(item_id)
    return item_def.stack if item_def else 64


def is_solid(block_id: int) -> bool:
    """Whether this block blocks player movement (collidable)."""
    if block_id == Block.AIR:
        return False
    item_def = ITEMS.get(block_id)
    return item_def.solid if item_def else True


def is_opaque(block_id: int) -> bool:
    """Whether this block should hide a neighboring face touching it (mesh culling)."""
    return block_id != Block.AIR and block_id in OPAQUE_BLOCKS
