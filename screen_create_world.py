"""
ui/screen_create_world.py
"Create World" screen: world name text field, seed text field (numeric or
arbitrary string - hashed into an integer seed if not purely numeric so
players can type anything memorable), game mode toggle (Survival/Creative),
and a Create button at the bottom, matching the requested layout.
"""

import random
import pygame
from ui.widgets import Button, TextField, draw_text, draw_beveled_panel
import config


def seed_text_to_int(seed_text: str) -> int:
    seed_text = seed_text.strip()
    if not seed_text:
        return random.randint(0, 2 ** 31 - 1)
    if seed_text.lstrip("-").isdigit():
        return int(seed_text) % (2 ** 31)
    # arbitrary string seed: hash it deterministically into an int
    return abs(hash(seed_text)) % (2 ** 31)


class CreateWorldScreen:
    def __init__(self, on_create, on_back):
        self.on_create = on_create
        self.on_back = on_back
        self.name_field = None
        self.seed_field = None
        self.mode_button = None
        self.create_button = None
        self.back_button = None
        self.game_mode = "survival"

    def layout(self, width, height):
        cx = width // 2
        self.name_field = TextField((cx - 180, 150, 360, 36), initial_text="New World")
        self.seed_field = TextField((cx - 180, 220, 360, 36), placeholder="Leave blank for random seed")
        self.mode_button = Button((cx - 180, 290, 360, 40), self._mode_label())
        self.create_button = Button((cx - 140, height - 80, 280, 46), "Create")
        self.back_button = Button((20, 20, 100, 36), "Back", font_size=14)

    def _mode_label(self):
        return f"Game Mode: {self.game_mode.capitalize()}"

    def handle_event(self, event, width, height):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            self.name_field.handle_click(mouse_pos)
            self.seed_field.handle_click(mouse_pos)
            if self.mode_button.rect.collidepoint(mouse_pos):
                self.game_mode = "creative" if self.game_mode == "survival" else "survival"
                self.mode_button.label = self._mode_label()
            if self.create_button.rect.collidepoint(mouse_pos):
                self._submit()
            if self.back_button.rect.collidepoint(mouse_pos):
                self.on_back()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.name_field.handle_backspace()
                self.seed_field.handle_backspace()
            elif event.key == pygame.K_RETURN:
                self._submit()
            elif event.key == pygame.K_TAB:
                if self.name_field.focused:
                    self.name_field.focused = False
                    self.seed_field.focused = True
                else:
                    self.seed_field.focused = False
                    self.name_field.focused = True
        elif event.type == pygame.TEXTINPUT:
            self.name_field.handle_text_input(event.text)
            self.seed_field.handle_text_input(event.text)

    def _submit(self):
        display_name = self.name_field.text.strip() or "New World"
        seed = seed_text_to_int(self.seed_field.text)
        self.on_create(display_name, seed, self.game_mode)

    def update_hover(self, mouse_pos):
        self.mode_button.update_hover(mouse_pos)
        self.create_button.update_hover(mouse_pos)
        self.back_button.update_hover(mouse_pos)

    def draw(self, surface, width, height):
        surface.fill((50, 40, 30))
        draw_text(surface, "Create New World", (width // 2, 70), size=30, center=True)

        draw_text(surface, "World Name", (width // 2 - 180, 128), size=13, color=(200, 200, 200))
        self.name_field.draw(surface)

        draw_text(surface, "World Seed (key)", (width // 2 - 180, 198), size=13, color=(200, 200, 200))
        self.seed_field.draw(surface)

        self.mode_button.draw(surface)
        self.create_button.draw(surface)
        self.back_button.draw(surface)
