"""
ui/screen_world_select.py
The saved-world list: play one, delete one, or create a new one.

This used to BE the main menu. It moved out when the title screen grew a
"Multiplayer" button, because the two screens now answer different questions -
the title asks "what kind of game", this one asks "which world" - and the old
combined screen had no room to ask the first one. The code is unchanged from
its previous life beyond the title, the Back button, and the callback names;
the delete-confirmation dance in particular is exactly as it was.
"""

import pygame
import config
from ui.widgets import Button, draw_text, draw_beveled_panel
from save import world_save


class WorldSelectScreen:
    def __init__(self, on_play_world, on_create_world, on_back):
        self.on_play_world = on_play_world
        self.on_create_world = on_create_world
        self.on_back = on_back
        self.worlds = []
        self.world_buttons = []  # list of (play_button, delete_button, meta)
        self.create_button = None
        self.back_button = None
        self.scroll_offset = 0
        self.pending_delete = None  # meta awaiting confirmation
        self.refresh_worlds()

    def refresh_worlds(self):
        self.worlds = world_save.list_worlds()

    def layout(self, width, height):
        self.create_button = Button((width // 2 - 140, height - 70, 280, 44), "Create New World")
        self.back_button = Button((20, 20, 100, 36), "Back", font_size=14)

        list_top = 120
        row_h = 52
        self.world_buttons = []
        for i, meta in enumerate(self.worlds):
            y = list_top + i * row_h - self.scroll_offset
            play_btn = Button((width // 2 - 220, y, 340, 44), meta.get("display_name", "World"))
            delete_btn = Button((width // 2 + 130, y, 90, 44), "Delete", font_size=14)
            self.world_buttons.append((play_btn, delete_btn, meta))

    def handle_event(self, event, width, height):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            if self.pending_delete is not None:
                return  # confirmation dialog handles its own clicks (see handle_confirmation_click)
            if self.create_button.rect.collidepoint(mouse_pos):
                self.on_create_world()
                return
            if self.back_button.rect.collidepoint(mouse_pos):
                self.on_back()
                return
            for play_btn, delete_btn, meta in self.world_buttons:
                if play_btn.rect.collidepoint(mouse_pos):
                    self.on_play_world(meta)
                    return
                if delete_btn.rect.collidepoint(mouse_pos):
                    self.pending_delete = meta
                    return
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.pending_delete is not None:
                self.pending_delete = None
            else:
                self.on_back()
        elif event.type == pygame.MOUSEWHEEL:
            self.scroll_offset = max(0, self.scroll_offset - event.y * 20)

    def confirm_delete(self, confirmed: bool):
        if confirmed and self.pending_delete is not None:
            world_save.delete_world(self.pending_delete.get("dir_name") or self.pending_delete.get("_dir_name"))
            self.refresh_worlds()
        self.pending_delete = None

    def update_hover(self, mouse_pos):
        if self.create_button:
            self.create_button.update_hover(mouse_pos)
        if self.back_button:
            self.back_button.update_hover(mouse_pos)
        for play_btn, delete_btn, _ in self.world_buttons:
            play_btn.update_hover(mouse_pos)
            delete_btn.update_hover(mouse_pos)

    def draw(self, surface, width, height):
        surface.fill((50, 40, 30))

        draw_text(surface, "Select World", (width // 2, 60), size=30, center=True)

        if not self.worlds:
            draw_text(surface, "No worlds yet - create one below!", (width // 2, 200), size=18, center=True)

        for play_btn, delete_btn, meta in self.world_buttons:
            if play_btn.rect.bottom < 100 or play_btn.rect.top > height - 90:
                continue  # clipped out of the scrollable list area
            play_btn.draw(surface)
            delete_btn.draw(surface)
            subtitle = f"{meta.get('game_mode', '?').capitalize()} - seed {meta.get('seed', '?')}"
            draw_text(surface, subtitle, (play_btn.rect.x + 10, play_btn.rect.bottom + 2),
                      size=11, color=(180, 180, 180), shadow=False)

        self.create_button.draw(surface)
        self.back_button.draw(surface)

        if self.pending_delete is not None:
            self._draw_delete_confirmation(surface, width, height)

    def _draw_delete_confirmation(self, surface, width, height):
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surface.blit(overlay, (0, 0))

        panel_rect = (width // 2 - 200, height // 2 - 70, 400, 140)
        draw_beveled_panel(surface, panel_rect, base_color=(60, 60, 60))
        name = self.pending_delete.get("display_name", "this world")
        draw_text(surface, f"Delete '{name}'?", (width // 2, height // 2 - 30), size=18, center=True)
        draw_text(surface, "This cannot be undone.", (width // 2, height // 2 - 5), size=13, center=True,
                  color=(220, 120, 120))

        self.confirm_yes_rect = pygame.Rect(width // 2 - 170, height // 2 + 20, 150, 36)
        self.confirm_no_rect = pygame.Rect(width // 2 + 20, height // 2 + 20, 150, 36)
        draw_beveled_panel(surface, self.confirm_yes_rect, base_color=(120, 50, 50))
        draw_text(surface, "Delete", self.confirm_yes_rect.center, size=15, center=True)
        draw_beveled_panel(surface, self.confirm_no_rect, base_color=config.COLOR_UI_STONE_DARK)
        draw_text(surface, "Cancel", self.confirm_no_rect.center, size=15, center=True)

    def handle_confirmation_click(self, mouse_pos) -> bool:
        """Call this BEFORE handle_event when pending_delete is set, since the
        confirmation dialog intercepts clicks. Returns True if it consumed the click."""
        if self.pending_delete is None:
            return False
        if hasattr(self, "confirm_yes_rect") and self.confirm_yes_rect.collidepoint(mouse_pos):
            self.confirm_delete(True)
            return True
        if hasattr(self, "confirm_no_rect") and self.confirm_no_rect.collidepoint(mouse_pos):
            self.confirm_delete(False)
            return True
        return True  # swallow clicks anywhere else while the dialog is open
