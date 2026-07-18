"""
ui/screen_death.py
Death screen: dark red-tinted overlay, "You Died" title, Respawn and Back
to Title buttons - matches vanilla Minecraft's death screen.
"""

import pygame
from ui.widgets import Button, draw_text


class DeathScreen:
    def __init__(self, on_respawn, on_quit_to_title):
        self.on_respawn = on_respawn
        self.on_quit_to_title = on_quit_to_title
        self.respawn_button = None
        self.quit_button = None

    def layout(self, width, height):
        cx = width // 2
        y = height // 2 - 10
        self.respawn_button = Button((cx - 140, y, 280, 44), "Respawn")
        self.quit_button = Button((cx - 140, y + 56, 280, 44), "Main Menu")

    def handle_event(self, event, width, height):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            if self.respawn_button.rect.collidepoint(mouse_pos):
                self.on_respawn()
            elif self.quit_button.rect.collidepoint(mouse_pos):
                self.on_quit_to_title()

    def update_hover(self, mouse_pos):
        self.respawn_button.update_hover(mouse_pos)
        self.quit_button.update_hover(mouse_pos)

    def draw(self, surface, width, height):
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((60, 0, 0, 160))
        surface.blit(overlay, (0, 0))

        draw_text(surface, "You Died", (width // 2, height // 2 - 90), size=32,
                  color=(255, 255, 255), center=True)
        self.respawn_button.draw(surface)
        self.quit_button.draw(surface)
