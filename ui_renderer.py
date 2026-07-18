"""
ui/ui_renderer.py
Bridges Pygame's 2D drawing (surfaces, fonts, images) into the OpenGL frame:
all UI is drawn onto an off-screen pygame.Surface using normal pygame draw
calls and pygame.font text rendering, then that surface is uploaded as a
texture and blitted over the 3D scene with a simple fullscreen-quad shader.
This is dramatically simpler and more reliable than hand-rolling a bitmap
font renderer in raw OpenGL, and pygame.font gives proper anti-aliased text
for free. The Minecraft-y "look" comes from the widget drawing code in
widgets.py (chunky borders, drop shadows, pixel-style palette), not from
faking a custom font.
"""

import numpy as np
import pygame
from OpenGL.GL import (
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    glTexSubImage2D, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    GL_LINEAR, GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_CLAMP_TO_EDGE, glGenVertexArrays, glBindVertexArray, glGenBuffers,
    glBindBuffer, glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawArrays, GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_FALSE,
    GL_TRIANGLE_STRIP, glActiveTexture, GL_TEXTURE0, glEnable, glDisable,
    GL_BLEND, glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_DEPTH_TEST,
)

from core.shader import Shader

VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec2 in_position;
layout (location = 1) in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_uv;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_ui_tex;
out vec4 frag_color;
void main() {
    frag_color = texture(u_ui_tex, v_uv);
}
"""


class UIRenderer:
    """
    Owns a pygame.Surface the size of the window, a matching GL texture, and
    the fullscreen-quad shader/geometry to blit it. Game code draws UI each
    frame onto self.surface using ordinary pygame calls, then calls
    upload_and_draw() once after all UI drawing is done for the frame.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)

        self.shader = Shader(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, name="ui")

        # fullscreen quad: two triangles as a strip, position + uv interleaved.
        # V coordinates are flipped here (0 at top, 1 at bottom) to match
        # pygame.image.tostring's top-down row order: without this flip, the
        # whole UI renders upside-down, since NDC y=-1 (bottom of screen) was
        # being paired with uv v=0, which is the FIRST row of the pygame
        # surface data - i.e. the TOP of what was actually drawn.
        quad = np.array([
            -1, -1, 0, 1,
             1, -1, 1, 1,
            -1,  1, 0, 0,
             1,  1, 1, 0,
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        stride = 4 * 4
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, None)
        glEnableVertexAttribArray(0)
        from ctypes import c_void_p
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, c_void_p(2 * 4))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        self.tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

    def resize(self, width: int, height: int):
        self.width, self.height = width, height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

    def clear(self):
        self.surface.fill((0, 0, 0, 0))

    def upload_and_draw(self):
        """Uploads the current surface contents to the GPU and draws it as a
        fullscreen overlay on top of whatever was rendered before this call."""
        raw = pygame.image.tostring(self.surface, "RGBA", False)
        data = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)

        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, data)

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.shader.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        self.shader.set_int("u_ui_tex", 0)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
