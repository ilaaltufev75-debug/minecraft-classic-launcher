"""
ui/screen_inventory.py
Full inventory + 2x2 crafting screen, opened with E. Survival shows the
player's own hotbar+main grid plus a personal crafting grid; Creative shows
an infinite item palette instead of crafting (matching real Minecraft's
creative inventory). Items are picked up/placed with the mouse: left-click
picks up or places a whole stack, right-click on a held stack places a
single item at a time - the same interaction model as the original JS
build, ported to pygame mouse events instead of DOM clicks.
"""

import pygame
import config
from ui.widgets import ItemSlot, draw_text, draw_beveled_panel
from ui.icon_cache import get_item_icon
from world.blocks import get_stack_size
from inventory.inventory import CREATIVE_PALETTE
from inventory.crafting import match_recipe

SLOT_SIZE = 36
SLOT_GAP = 4


class InventoryScreen:
    def __init__(self, texture_atlas):
        self.texture_atlas = texture_atlas
        self.drag_stack = None  # {"id","count"} currently held by the cursor
        self.crafting_grid = [None, None, None, None]
        self.crafting_output = None

        self.hotbar_slots = []       # list of (ItemSlot, global_index)
        self.main_grid_slots = []    # list of (ItemSlot, global_index)
        self.crafting_slots = []     # list of (ItemSlot, local_index) - survival only
        self.output_slot = None
        self.palette_slots = []      # list of (ItemSlot, item_id) - creative only

    def layout(self, width, height, game_mode: str):
        panel_w = config.HOTBAR_SIZE * (SLOT_SIZE + SLOT_GAP) + SLOT_GAP + 20
        panel_h = 320 if game_mode == "survival" else 360
        self.panel_rect = pygame.Rect(width // 2 - panel_w // 2, height // 2 - panel_h // 2, panel_w, panel_h)

        x0 = self.panel_rect.x + 10
        y0 = self.panel_rect.y + 50

        if game_mode == "survival":
            self.crafting_slots = []
            for i in range(4):
                cx = x0 + (i % 2) * (SLOT_SIZE + SLOT_GAP)
                cy = y0 + (i // 2) * (SLOT_SIZE + SLOT_GAP)
                self.crafting_slots.append((ItemSlot((cx, cy, SLOT_SIZE, SLOT_SIZE)), i))
            output_x = x0 + 2 * (SLOT_SIZE + SLOT_GAP) + 40
            output_y = y0 + (SLOT_SIZE + SLOT_GAP) // 2
            self.output_slot = ItemSlot((output_x, output_y, SLOT_SIZE, SLOT_SIZE))
            self.palette_slots = []
            grid_bottom = y0 + 2 * (SLOT_SIZE + SLOT_GAP) + 14
        else:
            self.palette_slots = []
            cols = 7
            for i, item_id in enumerate(CREATIVE_PALETTE):
                px = x0 + (i % cols) * (SLOT_SIZE + SLOT_GAP)
                py = y0 + (i // cols) * (SLOT_SIZE + SLOT_GAP)
                self.palette_slots.append((ItemSlot((px, py, SLOT_SIZE, SLOT_SIZE)), item_id))
            rows = (len(CREATIVE_PALETTE) + cols - 1) // cols
            grid_bottom = y0 + rows * (SLOT_SIZE + SLOT_GAP) + 14
            self.crafting_slots = []
            self.output_slot = None

        main_y = grid_bottom
        self.main_grid_slots = []
        for i in range(config.MAIN_INVENTORY_SIZE):
            row, col = divmod(i, config.MAIN_INVENTORY_COLS)
            sx = x0 + col * (SLOT_SIZE + SLOT_GAP)
            sy = main_y + row * (SLOT_SIZE + SLOT_GAP)
            global_index = config.HOTBAR_SIZE + i
            self.main_grid_slots.append((ItemSlot((sx, sy, SLOT_SIZE, SLOT_SIZE)), global_index))

        hotbar_y = main_y + config.MAIN_INVENTORY_ROWS * (SLOT_SIZE + SLOT_GAP) + 10
        self.hotbar_slots = []
        for i in range(config.HOTBAR_SIZE):
            sx = x0 + i * (SLOT_SIZE + SLOT_GAP)
            self.hotbar_slots.append((ItemSlot((sx, hotbar_y, SLOT_SIZE, SLOT_SIZE)), i))

    def _try_craft(self):
        output = match_recipe(self.crafting_grid)
        self.crafting_output = dict(output) if output else None

    def _consume_craft_ingredients(self):
        for i in range(4):
            if self.crafting_grid[i]:
                self.crafting_grid[i]["count"] -= 1
                if self.crafting_grid[i]["count"] <= 0:
                    self.crafting_grid[i] = None
        self._try_craft()

    def _click_slot(self, get_fn, set_fn, right_click: bool):
        current = get_fn()
        if right_click:
            if self.drag_stack is None:
                return
            if current is None:
                set_fn({"id": self.drag_stack["id"], "count": 1})
                self.drag_stack["count"] -= 1
            elif current["id"] == self.drag_stack["id"] and current["count"] < get_stack_size(current["id"]):
                current["count"] += 1
                self.drag_stack["count"] -= 1
            if self.drag_stack["count"] <= 0:
                self.drag_stack = None
            return

        if self.drag_stack is None:
            if current is not None:
                self.drag_stack = current
                set_fn(None)
        else:
            if current is None:
                set_fn(self.drag_stack)
                self.drag_stack = None
            elif current["id"] == self.drag_stack["id"]:
                max_stack = get_stack_size(current["id"])
                room = max_stack - current["count"]
                move = min(room, self.drag_stack["count"])
                current["count"] += move
                self.drag_stack["count"] -= move
                if self.drag_stack["count"] <= 0:
                    self.drag_stack = None
            else:
                set_fn(self.drag_stack)
                self.drag_stack = current

    def handle_click(self, mouse_pos, right_click: bool, inventory, game_mode: str):
        for slot, idx in self.hotbar_slots:
            if slot.rect.collidepoint(mouse_pos):
                self._click_slot(lambda i=idx: inventory.get_slot(i),
                                  lambda v, i=idx: inventory.set_slot(i, v), right_click)
                return
        for slot, idx in self.main_grid_slots:
            if slot.rect.collidepoint(mouse_pos):
                self._click_slot(lambda i=idx: inventory.get_slot(i),
                                  lambda v, i=idx: inventory.set_slot(i, v), right_click)
                return

        if game_mode == "survival":
            for slot, idx in self.crafting_slots:
                if slot.rect.collidepoint(mouse_pos):
                    self._click_slot(lambda i=idx: self.crafting_grid[i],
                                      lambda v, i=idx: self.crafting_grid.__setitem__(i, v), right_click)
                    self._try_craft()
                    return
            if self.output_slot and self.output_slot.rect.collidepoint(mouse_pos) and not right_click:
                if self.crafting_output:
                    if self.drag_stack is None:
                        self.drag_stack = dict(self.crafting_output)
                        self._consume_craft_ingredients()
                    elif self.drag_stack["id"] == self.crafting_output["id"]:
                        max_stack = get_stack_size(self.drag_stack["id"])
                        if self.drag_stack["count"] + self.crafting_output["count"] <= max_stack:
                            self.drag_stack["count"] += self.crafting_output["count"]
                            self._consume_craft_ingredients()
                return
        else:
            for slot, item_id in self.palette_slots:
                if slot.rect.collidepoint(mouse_pos) and not right_click:
                    max_stack = get_stack_size(item_id)
                    if self.drag_stack is None or self.drag_stack["id"] != item_id:
                        self.drag_stack = {"id": item_id, "count": max_stack}
                    else:
                        self.drag_stack["count"] = min(max_stack, self.drag_stack["count"] + max_stack)
                    return

    def close_and_return_items(self, inventory):
        """Returns the held drag stack and any crafting-grid contents back to
        the inventory when the screen closes, so nothing is silently lost."""
        if self.drag_stack is not None:
            leftover = inventory.add_item(self.drag_stack["id"], self.drag_stack["count"])
            self.drag_stack = {"id": self.drag_stack["id"], "count": leftover} if leftover > 0 else None
        for i in range(4):
            if self.crafting_grid[i]:
                inventory.add_item(self.crafting_grid[i]["id"], self.crafting_grid[i]["count"])
                self.crafting_grid[i] = None
        self.crafting_output = None

    def draw(self, surface, width, height, inventory, game_mode: str):
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        surface.blit(overlay, (0, 0))

        draw_beveled_panel(surface, self.panel_rect, base_color=(198, 198, 198),
                            light=(230, 230, 230), dark=(90, 90, 90))
        title = "Creative Inventory" if game_mode == "creative" else "Inventory"
        draw_text(surface, title, (self.panel_rect.x + 14, self.panel_rect.y + 12),
                  size=15, color=(50, 50, 50), shadow=False)

        def draw_stack_slot(slot, stack):
            icon = get_item_icon(self.texture_atlas, stack["id"], size=28) if stack else None
            count = stack["count"] if stack else None
            slot.draw(surface, icon_surface=icon, count=count)

        for slot, idx in self.hotbar_slots:
            draw_stack_slot(slot, inventory.get_slot(idx))
        for slot, idx in self.main_grid_slots:
            draw_stack_slot(slot, inventory.get_slot(idx))

        if game_mode == "survival":
            for slot, idx in self.crafting_slots:
                draw_stack_slot(slot, self.crafting_grid[idx])
            if self.output_slot:
                draw_stack_slot(self.output_slot, self.crafting_output)
        else:
            for slot, item_id in self.palette_slots:
                icon = get_item_icon(self.texture_atlas, item_id, size=28)
                slot.draw(surface, icon_surface=icon, count=None)

        if self.drag_stack is not None:
            mouse_pos = pygame.mouse.get_pos()
            icon = get_item_icon(self.texture_atlas, self.drag_stack["id"], size=28)
            rect = icon.get_rect(center=mouse_pos)
            surface.blit(icon, rect)
            if self.drag_stack["count"] > 1:
                draw_text(surface, str(self.drag_stack["count"]),
                          (mouse_pos[0] + 10, mouse_pos[1] + 10), size=13)
