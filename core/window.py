"""
core/window.py
Owns the Pygame window and the OpenGL context it creates. Responsible for
window creation, resizing, buffer swapping, and mouse-grab (the "pointer
lock" equivalent) toggling. Game code should not touch pygame.display
directly outside of this module, so the windowing backend stays swappable.
"""

import pygame
from OpenGL.GL import glViewport, glEnable, glDepthFunc, GL_DEPTH_TEST, GL_LEQUAL, \
    glClearColor, GL_CULL_FACE, glCullFace, GL_BACK, glFrontFace, GL_CCW

import config


class Window:
    def __init__(self, size=None, title=None):
        size = size or config.DEFAULT_WINDOW_SIZE
        title = title or config.WINDOW_TITLE

        pygame.init()
        pygame.display.set_caption(title)

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        self.surface = pygame.display.set_mode(size, flags)
        self.width, self.height = size

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        glClearColor(*config.FOG_COLOR, 1.0)
        glViewport(0, 0, *size)

        self.clock = pygame.time.Clock()
        self._mouse_grabbed = False
        self.set_mouse_grab(False)

    def handle_resize(self, width: int, height: int):
        self.width, self.height = max(1, width), max(1, height)
        glViewport(0, 0, self.width, self.height)

    def set_mouse_grab(self, grabbed: bool):
        """Equivalent of browser Pointer Lock: hides the cursor and confines/relative-mouses it."""
        self._mouse_grabbed = grabbed
        pygame.mouse.set_visible(not grabbed)
        pygame.event.set_grab(grabbed)
        try:
            pygame.mouse.set_relative_mode(grabbed)
        except Exception:
            pass  # older pygame/SDL builds may not support relative mode; grab still helps

    @property
    def mouse_grabbed(self) -> bool:
        return self._mouse_grabbed

    def swap_buffers(self):
        pygame.display.flip()

    def tick(self, fps_limit: int) -> float:
        """Advances the clock, optionally capping framerate. Returns delta time in seconds."""
        if fps_limit and fps_limit > 0:
            dt_ms = self.clock.tick(fps_limit)
        else:
            dt_ms = self.clock.tick()
        dt = dt_ms / 1000.0
        # Clamp dt to avoid a physics "tunneling" bug: if a frame takes
        # unusually long in real time (e.g. the frame right after a heavy
        # chunk-generation burst, or the window losing focus for a moment),
        # a large dt makes vy*dt in PlayerPhysics jump the player's Y by
        # more than a block in one step. Collision is only checked at the
        # destination position, not swept along the path, so a big enough
        # jump can skip clean over the ground block into open air below -
        # this is what caused players to fall through the world/end up
        # under bedrock immediately after a world finished loading.
        return min(dt, config.MAX_FRAME_DT)

    def get_fps(self) -> float:
        return self.clock.get_fps()

    def quit(self):
        pygame.quit()
