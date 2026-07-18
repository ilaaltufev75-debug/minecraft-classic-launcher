"""
ui/screen_settings.py
Options screen: mouse sensitivity, render distance, FPS display toggle,
FPS limit. Reads/writes directly into a shared `settings` dict object
passed in by main.py, so changes take effect immediately without needing
an explicit "Apply" step.
"""

import pygame
from ui.widgets import Button, Slider, draw_text
import config


class SettingsScreen:
    def __init__(self, settings: dict, on_back):
        self.settings = settings
        self.on_back = on_back
        self.sensitivity_slider = None
        self.render_distance_slider = None
        self.fps_toggle_button = None
        self.fps_limit_button = None
        self.back_button = None
        self._fps_limit_options = [0, 30, 60, 120]

    def layout(self, width, height):
        cx = width // 2
        y = 130
        gap = 70

        self.sensitivity_slider = Slider(
            (cx - 180, y, 360, 20),
            lambda v: f"Mouse Sensitivity: {int(v * 100)}%",
            self.settings.get("mouse_sensitivity", 1.0), 0.2, 3.0, 0.05,
            on_change=lambda v: self.settings.__setitem__("mouse_sensitivity", v)
        )
        y += gap
        self.render_distance_slider = Slider(
            (cx - 180, y, 360, 20),
            lambda v: f"Render Distance: {int(v)} chunks",
            self.settings.get("render_distance", config.DEFAULT_RENDER_DISTANCE),
            config.MIN_RENDER_DISTANCE, config.MAX_RENDER_DISTANCE, 1,
            on_change=lambda v: self.settings.__setitem__("render_distance", int(v))
        )
        y += gap
        self.fps_toggle_button = Button((cx - 180, y, 360, 40), self._fps_toggle_label())
        y += gap
        self.fps_limit_button = Button((cx - 180, y, 360, 40), self._fps_limit_label())

        self.back_button = Button((cx - 140, height - 80, 280, 46), "Done")

    def _fps_toggle_label(self):
        return f"Show FPS: {'On' if self.settings.get('show_fps', True) else 'Off'}"

    def _fps_limit_label(self):
        limit = self.settings.get("fps_limit", config.DEFAULT_FPS_LIMIT)
        return f"FPS Limit: {'Unlimited' if limit == 0 else limit}"

    def handle_event(self, event, width, height):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            self.sensitivity_slider.handle_mouse_down(mouse_pos)
            self.render_distance_slider.handle_mouse_down(mouse_pos)
            if self.fps_toggle_button.rect.collidepoint(mouse_pos):
                self.settings["show_fps"] = not self.settings.get("show_fps", True)
                self.fps_toggle_button.label = self._fps_toggle_label()
            if self.fps_limit_button.rect.collidepoint(mouse_pos):
                current = self.settings.get("fps_limit", config.DEFAULT_FPS_LIMIT)
                idx = self._fps_limit_options.index(current) if current in self._fps_limit_options else 0
                self.settings["fps_limit"] = self._fps_limit_options[(idx + 1) % len(self._fps_limit_options)]
                self.fps_limit_button.label = self._fps_limit_label()
            if self.back_button.rect.collidepoint(mouse_pos):
                self.on_back()
        elif event.type == pygame.MOUSEBUTTONUP:
            self.sensitivity_slider.handle_mouse_up()
            self.render_distance_slider.handle_mouse_up()
        elif event.type == pygame.MOUSEMOTION:
            self.sensitivity_slider.handle_mouse_motion(event.pos)
            self.render_distance_slider.handle_mouse_motion(event.pos)

    def update_hover(self, mouse_pos):
        self.fps_toggle_button.update_hover(mouse_pos)
        self.fps_limit_button.update_hover(mouse_pos)
        self.back_button.update_hover(mouse_pos)

    def draw(self, surface, width, height):
        surface.fill((40, 40, 45))
        draw_text(surface, "Options", (width // 2, 60), size=30, center=True)

        self.sensitivity_slider.draw(surface)
        self.render_distance_slider.draw(surface)
        self.fps_toggle_button.draw(surface)
        self.fps_limit_button.draw(surface)
        self.back_button.draw(surface)
