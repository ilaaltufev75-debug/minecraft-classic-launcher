"""
core/input.py
Polls pygame events once per frame and exposes a simple, query-able snapshot
of input state (keys held, mouse delta this frame, buttons pressed/released
this frame, wheel delta). Game/UI code reads from an InputState instance
instead of touching pygame.event directly, which keeps every other module
backend-agnostic and easy to unit test with a fake InputState.
"""

import pygame


class InputState:
    def __init__(self):
        self.keys_down = set()          # keys currently held
        self.keys_pressed = set()       # keys that transitioned down->up->down this frame (just-pressed)
        self.keys_released = set()      # keys that were released this frame
        self.mouse_dx = 0
        self.mouse_dy = 0
        self.mouse_buttons_pressed = set()   # 1=left, 2=middle, 3=right (pygame convention)
        self.mouse_buttons_released = set()
        self.mouse_buttons_down = set()
        self.wheel_delta = 0
        self.text_input = ""            # accumulated text typed this frame (for UI text fields)
        self.quit_requested = False
        self.window_resized = None      # (w, h) if a resize event happened this frame, else None

    def begin_frame(self):
        """Call once at the start of each frame, before poll_events()."""
        self.keys_pressed.clear()
        self.keys_released.clear()
        self.mouse_dx = 0
        self.mouse_dy = 0
        self.mouse_buttons_pressed.clear()
        self.mouse_buttons_released.clear()
        self.wheel_delta = 0
        self.text_input = ""
        self.window_resized = None

    def poll_events(self, event_list=None):
        """Processes a list of pygame events (or pygame.event.get() if not provided)."""
        events = event_list if event_list is not None else pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                self.keys_down.add(event.key)
                self.keys_pressed.add(event.key)
            elif event.type == pygame.KEYUP:
                self.keys_down.discard(event.key)
                self.keys_released.add(event.key)
            elif event.type == pygame.MOUSEMOTION:
                dx, dy = event.rel
                self.mouse_dx += dx
                self.mouse_dy += dy
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_buttons_down.add(event.button)
                self.mouse_buttons_pressed.add(event.button)
                if event.button == 4:
                    self.wheel_delta -= 1  # scroll up (legacy button-based wheel events)
                elif event.button == 5:
                    self.wheel_delta += 1
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_buttons_down.discard(event.button)
                self.mouse_buttons_released.add(event.button)
            elif event.type == pygame.MOUSEWHEEL:
                self.wheel_delta -= event.y
            elif event.type == pygame.TEXTINPUT:
                self.text_input += event.text
            elif event.type == pygame.VIDEORESIZE:
                self.window_resized = (event.w, event.h)

    def is_key_down(self, key: int) -> bool:
        return key in self.keys_down

    def was_key_pressed(self, key: int) -> bool:
        return key in self.keys_pressed

    def was_key_released(self, key: int) -> bool:
        return key in self.keys_released

    def is_mouse_down(self, button: int) -> bool:
        return button in self.mouse_buttons_down

    def was_mouse_pressed(self, button: int) -> bool:
        return button in self.mouse_buttons_pressed

    def was_mouse_released(self, button: int) -> bool:
        return button in self.mouse_buttons_released
