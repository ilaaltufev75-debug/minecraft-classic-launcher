"""
render/outline_renderer.py
Draws the thin black wireframe box around whichever block the player is
currently looking at (crosshair-targeted), matching vanilla Minecraft's
block-selection outline. Pure line geometry - 12 edges of a unit cube,
offset slightly outward from the block's actual faces so the lines don't
z-fight with the block's own mesh.
"""

import numpy as np
from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
    glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawArrays, glDeleteVertexArrays, glDeleteBuffers,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_FALSE, GL_LINES,
    glLineWidth, glEnable, glDisable, GL_BLEND, glBlendFunc,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_DEPTH_TEST,
)

from core.shader import Shader

VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 in_position;  // unit cube corner, 0..1

uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_block_pos;
uniform float u_pad;

void main() {
    vec3 p = in_position * (1.0 + 2.0 * u_pad) - u_pad;
    vec3 world_pos = u_block_pos + p;
    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
out vec4 frag_color;
void main() {
    frag_color = vec4(0.0, 0.0, 0.0, 0.6);
}
"""

_PAD = 0.002  # small outward offset so edges don't z-fight the block's own faces

# 12 edges of a unit cube, each as a pair of corners (0/1 per axis)
_CORNERS = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]
_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),   # top face
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]


def _edge_line_vertices():
    verts = []
    for a, b in _EDGES:
        verts.append(_CORNERS[a])
        verts.append(_CORNERS[b])
    return np.array(verts, dtype=np.float32)


class OutlineRenderer:
    def __init__(self):
        self.shader = Shader(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, name="outline")
        verts = _edge_line_vertices()
        self.vertex_count = len(verts)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def render(self, camera, block_x, block_y, block_z):
        self.shader.use()
        self.shader.set_mat4("u_view", camera.view_matrix())
        self.shader.set_mat4("u_projection", camera.projection_matrix())
        self.shader.set_vec3("u_block_pos", float(block_x), float(block_y), float(block_z))
        self.shader.set_float("u_pad", _PAD)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(2.0)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINES, 0, self.vertex_count)
        glBindVertexArray(0)

        glDisable(GL_BLEND)

    def destroy(self):
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = None
        if self.vbo is not None:
            glDeleteBuffers(1, [self.vbo])
            self.vbo = None
