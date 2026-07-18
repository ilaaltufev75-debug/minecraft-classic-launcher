"""
render/particles.py
Small cube particles that burst outward when a block is broken, matching
the classic Minecraft "block breaking" feedback. Particles are simulated
with simple gravity and fade out (shrink to nothing) over their lifetime,
then get cleaned up. Rendered as tiny textured cubes using the same shader
approach as the hand (own small VAO, reused every particle via instancing-
lite: one draw call per particle, which is fine since there are only ever
a handful active at once).
"""

import math
import random
import numpy as np
from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
    glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawElements, GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW,
    GL_FLOAT, GL_FALSE, GL_TRIANGLES, GL_UNSIGNED_INT,
)

from core.shader import Shader
import config

VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_uv;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_center;
uniform float u_scale;
uniform vec4 u_uv_rect;  // (u0, v0, u1, v1) - remaps the default [0,1] mesh UVs
                          // to this particle's actual block texture region,
                          // so break particles show the broken block's texture
                          // instead of an arbitrary/default atlas tile.

out vec2 v_uv;

void main() {
    vec3 world_pos = u_center + in_position * u_scale;
    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
    v_uv = mix(u_uv_rect.xy, u_uv_rect.zw, in_uv);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_atlas;
out vec4 frag_color;

void main() {
    vec4 tex_color = texture(u_atlas, v_uv);
    if (tex_color.a < 0.1) discard;
    frag_color = tex_color;
}
"""


class Particle:
    __slots__ = ("x", "y", "z", "vx", "vy", "vz", "life", "max_life", "uv_rect")

    def __init__(self, x, y, z, vx, vy, vz, life, uv_rect):
        self.x, self.y, self.z = x, y, z
        self.vx, self.vy, self.vz = vx, vy, vz
        self.life = life
        self.max_life = life
        self.uv_rect = uv_rect


def _tiny_cube_mesh(size=0.12):
    h = size / 2
    faces = [
        ((1, 0, 0), [(h, -h, h), (h, -h, -h), (h, h, -h), (h, h, h)]),
        ((-1, 0, 0), [(-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, h, -h)]),
        ((0, 1, 0), [(-h, h, h), (h, h, h), (h, h, -h), (-h, h, -h)]),
        ((0, -1, 0), [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)]),
        ((0, 0, 1), [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]),
        ((0, 0, -1), [(-h, -h, -h), (-h, h, -h), (h, h, -h), (h, -h, -h)]),
    ]
    positions, indices = [], []
    vidx = 0
    for _, corners in faces:
        for c in corners:
            positions.append(c)
        indices.extend([vidx, vidx + 1, vidx + 2, vidx, vidx + 2, vidx + 3])
        vidx += 4
    return np.array(positions, dtype=np.float32), np.array(indices, dtype=np.uint32)


class ParticleSystem:
    def __init__(self, texture_atlas, max_particles=200):
        self.texture_atlas = texture_atlas
        self.particles: list[Particle] = []
        self.max_particles = max_particles

        self.shader = Shader(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, name="particles")

        positions, indices = _tiny_cube_mesh()
        # UVs default to [0,1] on the tiny cube; remapped per-particle at draw time
        uvs = np.tile(np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32), (6, 1))

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        self.vbo_uv = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_uv)
        glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        self.index_count = len(indices)
        glBindVertexArray(0)

    def spawn_break_particles(self, block_id, bx, by, bz, count=8):
        uv_rect = self.texture_atlas.uv_for(block_id, "side") or self.texture_atlas.uv_for(block_id, "all")
        if uv_rect is None:
            return
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                self.particles.pop(0)
            vx = (random.random() - 0.5) * 3
            vy = random.random() * 3 + 1
            vz = (random.random() - 0.5) * 3
            self.particles.append(Particle(
                bx + 0.5, by + 0.5, bz + 0.5, vx, vy, vz, life=0.6, uv_rect=uv_rect
            ))

    def update(self, dt: float):
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.vy -= config.GRAVITY * 0.6 * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.z += p.vz * dt
            alive.append(p)
        self.particles = alive

    def render(self, camera):
        if not self.particles:
            return
        self.shader.use()
        self.shader.set_mat4("u_view", camera.view_matrix())
        self.shader.set_mat4("u_projection", camera.projection_matrix())
        self.shader.set_int("u_atlas", 0)
        self.texture_atlas.bind(0)

        glBindVertexArray(self.vao)
        for p in self.particles:
            self.shader.set_vec3("u_center", p.x, p.y, p.z)
            scale = max(0.2, p.life / p.max_life)
            self.shader.set_float("u_scale", scale)
            u0, v0, u1, v1 = p.uv_rect
            self.shader.set_vec4("u_uv_rect", u0, v0, u1, v1)
            glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
