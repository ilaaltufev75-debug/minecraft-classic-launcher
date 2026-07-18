"""
ui/screen_main_menu.py
The title screen: a logo and four buttons, and nothing else.

It used to be the saved-world list as well, which worked right up until there
was a second kind of game to start. "Which world" and "what kind of game" are
different questions and the list had nowhere to ask the second one, so the list
moved to ui/screen_world_select.py and this screen went back to being a title.
That is also the shape vanilla has, and for the same reason.

The labels are English because every other screen in this game already is
(Create New World / Options... / Back), and because they are vanilla's own.
Swapping them for Russian is four string literals below - the font is a
SysFont and renders Cyrillic fine - but half a menu in each language would be
worse than either.
"""

import pygame
import config
from ui.widgets import Button, draw_text


class MainMenuScreen:
    def __init__(self, on_singleplayer, on_multiplayer, on_settings, on_quit):
        self.on_singleplayer = on_singleplayer
        self.on_multiplayer = on_multiplayer
        self.on_settings = on_settings
        self.on_quit = on_quit
        self.buttons = []

    def layout(self, width, height):
        button_width = 320
        button_height = 44
        gap = 10
        x = width // 2 - button_width // 2
        top = int(height * 0.42)

        # A quit button the same size and in the same column as the rest, rather
        # than tucked in a corner: this is the one screen where leaving is a
        # normal thing to want, not an escape hatch.
        specs = (
            ("Singleplayer", self.on_singleplayer),
            ("Multiplayer", self.on_multiplayer),
            ("Options...", self.on_settings),
            ("Quit Game", self.on_quit),
        )
        self.buttons = [
            Button((x, top + i * (button_height + gap), button_width, button_height),
                   label, on_click=callback)
            for i, (label, callback) in enumerate(specs)
        ]

    def handle_event(self, event, width, height):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.buttons:
                if button.handle_click(event.pos):
                    return

    def update_hover(self, mouse_pos):
        for button in self.buttons:
            button.update_hover(mouse_pos)

    def draw(self, surface, width, height):
        surface.fill((50, 40, 30))

        draw_text(surface, "MINECRAFT", (width // 2, int(height * 0.20)), size=52, center=True)
        draw_text(surface, "Classic", (width // 2, int(height * 0.20) + 40), size=18,
                  center=True, color=(200, 200, 120))

        for button in self.buttons:
            button.draw(surface)

        draw_text(surface, "Not affiliated with Mojang", (8, height - 18), size=11,
                  color=(170, 170, 170), shadow=False)
