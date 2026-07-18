"""
inventory/inventory.py
Player inventory: hotbar (9 slots) + main grid (27 slots), stored as a flat
list of {"id","count"}|None. Survival starts empty; Creative gets a fixed
palette of every placeable block/item for the palette screen, plus a
pre-filled hotbar for convenience while building. Ported from the original
JS build's Game.INVENTORY module.
"""

import config
from world.blocks import Block, Item, get_stack_size


def empty_slots():
    return [None] * (config.HOTBAR_SIZE + config.MAIN_INVENTORY_SIZE)


CREATIVE_PALETTE = [
    Block.GRASS, Block.DIRT, Block.STONE, Block.COBBLESTONE, Block.SAND,
    Block.WOOD_LOG, Block.PLANKS, Block.LEAVES, Block.COAL_ORE, Block.IRON_ORE,
    Block.CRAFTING_TABLE, Block.DOOR, Block.FENCE,
    Block.STAIRS_WOOD, Block.STAIRS_STONE, Block.GLASS,
    Item.STICK, Item.COAL,
    Item.WOODEN_PICKAXE, Item.STONE_PICKAXE, Item.WOODEN_AXE, Item.WOODEN_SHOVEL,
    Item.STONE_AXE, Item.STONE_SHOVEL, Item.WOODEN_SWORD, Item.STONE_SWORD,
]
# Block.WATER is deliberately absent. It is a fluid, not a placeable block:
# the targeting ray passes straight through it, it cannot be selected or
# broken, and it has no flow logic yet - a water block handed to the player
# would just be a solid blue cube that behaves like nothing in Minecraft.
# It belongs in the palette when there is a bucket to carry it in.


class Inventory:
    def __init__(self, game_mode: str = "survival"):
        self.slots = empty_slots()
        self.selected_slot = 0
        if game_mode == "creative":
            self.slots[0] = {"id": Block.GRASS, "count": 1}
            self.slots[1] = {"id": Block.DIRT, "count": 1}
            self.slots[2] = {"id": Block.STONE, "count": 1}
            self.slots[3] = {"id": Block.COBBLESTONE, "count": 1}
            self.slots[4] = {"id": Block.SAND, "count": 1}
            self.slots[5] = {"id": Block.WOOD_LOG, "count": 1}
            self.slots[6] = {"id": Block.PLANKS, "count": 1}
            self.slots[7] = {"id": Block.GLASS, "count": 1}

    def get_slot(self, i: int):
        return self.slots[i]

    def set_slot(self, i: int, stack):
        self.slots[i] = stack

    def selected_stack(self):
        return self.slots[self.selected_slot]

    def add_item(self, item_id: int, count: int) -> int:
        """Adds count of item_id, filling partial stacks first then empty slots.
        Returns the leftover amount that didn't fit."""
        max_stack = get_stack_size(item_id)
        remaining = count

        for i, s in enumerate(self.slots):
            if remaining <= 0:
                break
            if s is not None and s["id"] == item_id and s["count"] < max_stack:
                room = max_stack - s["count"]
                add = min(room, remaining)
                s["count"] += add
                remaining -= add

        for i, s in enumerate(self.slots):
            if remaining <= 0:
                break
            if s is None:
                add = min(max_stack, remaining)
                self.slots[i] = {"id": item_id, "count": add}
                remaining -= add

        return remaining

    def remove_from_slot(self, i: int, count: int) -> int:
        s = self.slots[i]
        if s is None:
            return 0
        removed = min(s["count"], count)
        s["count"] -= removed
        if s["count"] <= 0:
            self.slots[i] = None
        return removed

    def count_item(self, item_id: int) -> int:
        return sum(s["count"] for s in self.slots if s is not None and s["id"] == item_id)

    def remove_item(self, item_id: int, count: int) -> int:
        """Removes up to count total of item_id across any slots. Returns amount actually removed."""
        remaining = count
        for i, s in enumerate(self.slots):
            if remaining <= 0:
                break
            if s is not None and s["id"] == item_id:
                take = min(s["count"], remaining)
                s["count"] -= take
                remaining -= take
                if s["count"] <= 0:
                    self.slots[i] = None
        return count - remaining
