"""
render/fence_renderer.py
Bakes vertex data per chunk for fence blocks: a central post box always
present, plus a thin rail box per connected side (north/south/east/west),
matching world/fences.py's connection state. Rebuilt alongside the regular
chunk mesh - same pattern as stairs_renderer.py.
"""

import numpy as np
from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
    glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawElements, glDeleteVertexArrays, glDeleteBuffers,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FLOAT,
    GL_FALSE, GL_TRIANGLES, GL_UNSIGNED_INT,
)

from core.shader import Shader
from world.blocks import Block

VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec2 v_uv;
out float v_shade;

void main() {
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
    v_uv = in_uv;
    float axis_y = abs(in_normal.y);
    float axis_x = abs(in_normal.x);
    float axis_z = abs(in_normal.z);
    float is_top = step(0.0, in_normal.y);
    float y_component = mix(0.5, 1.0, is_top);
    v_shade = axis_y * y_component + axis_x * 0.6 + axis_z * 0.8;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 v_uv;
in float v_shade;
uniform sampler2D u_atlas;
out vec4 frag_color;
void main() {
    vec4 tex_color = texture(u_atlas, v_uv);
    if (tex_color.a < 0.1) discard;
    frag_color = vec4(tex_color.rgb * v_shade, tex_color.a);
}
"""

_BOX_FACES = [
    {"dir": (1, 0, 0), "corners": ((1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1))},
    {"dir": (-1, 0, 0), "corners": ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))},
    {"dir": (0, 1, 0), "corners": ((0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0))},
    {"dir": (0, -1, 0), "corners": ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))},
    {"dir": (0, 0, 1), "corners": ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))},
    {"dir": (0, 0, -1), "corners": ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))},
]
_FACE_UVS = ((0, 0), (1, 0), (1, 1), (0, 1))

_POST_MIN, _POST_MAX = 0.375, 0.625
_RAIL_Y0, _RAIL_Y1 = 0.375, 0.875   # rails sit at mid-height, like vanilla fence crossbars


def _fence_boxes(north, south, east, west):
    boxes = [(_POST_MIN, 0.0, _POST_MIN, _POST_MAX, 1.0, _POST_MAX)]
    if north:
        boxes.append((_POST_MIN, _RAIL_Y0, 0.0, _POST_MAX, _RAIL_Y1, _POST_MIN))
    if south:
        boxes.append((_POST_MIN, _RAIL_Y0, _POST_MAX, _POST_MAX, _RAIL_Y1, 1.0))
    if west:
        boxes.append((0.0, _RAIL_Y0, _POST_MIN, _POST_MIN, _RAIL_Y1, _POST_MAX))
    if east:
        boxes.append((_POST_MAX, _RAIL_Y0, _POST_MIN, 1.0, _RAIL_Y1, _POST_MAX))
    return boxes


class FenceRenderer:
    def __init__(self, texture_atlas):
        self.texture_atlas = texture_atlas
        self.shader = Shader(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, name="fence")
        self.chunk_meshes: dict[tuple[int, int], tuple] = {}

    def _uv_rect(self):
        rect = self.texture_atlas.uv_for(Block.FENCE, "all")
        return rect if rect is not None else (0.0, 0.0, 0.0, 0.0)

    def _build_chunk_geometry(self, instances):
        u0, v0, u1, v1 = self._uv_rect()
        positions, normals, uvs, indices = [], [], [], []
        for (wx, wy, wz, north, south, east, west) in instances:
            for box in _fence_boxes(north, south, east, west):
                bx0, by0, bz0, bx1, by1, bz1 = box
                for face in _BOX_FACES:
                    base_idx = len(positions)
                    for (cx, cy, cz), (uu, vv) in zip(face["corners"], _FACE_UVS):
                        px = wx + (bx1 if cx else bx0)
                        py = wy + (by1 if cy else by0)
                        pz = wz + (bz1 if cz else bz0)
                        positions.append((px, py, pz))
                        normals.append(face["dir"])
                        uvs.append((u0 + uu * (u1 - u0), v0 + vv * (v1 - v0)))
                    indices.extend([base_idx, base_idx + 1, base_idx + 2,
                                     base_idx, base_idx + 2, base_idx + 3])
        if not positions:
            return None
        return (np.array(positions, dtype=np.float32),
                np.array(normals, dtype=np.float32),
                np.array(uvs, dtype=np.float32),
                np.array(indices, dtype=np.uint32))

    def rebuild_chunk(self, chunk, world):
        key = (chunk.cx, chunk.cz)
        instances = chunk.build_fence_instances()
        self.remove_chunk(*key)
        if not instances:
            return
        geom = self._build_chunk_geometry(instances)
        if geom is None:
            return
        positions, normals, uvs, indices = geom

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        vbo_norm = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_norm)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        vbo_uv = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_uv)
        glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_DYNAMIC_DRAW)

        glBindVertexArray(0)
        self.chunk_meshes[key] = (vao, vbo_pos, vbo_norm, vbo_uv, ebo, len(indices))

    def remove_chunk(self, cx: int, cz: int):
        key = (cx, cz)
        mesh = self.chunk_meshes.pop(key, None)
        if mesh is None:
            return
        vao, vbo_pos, vbo_norm, vbo_uv, ebo, _count = mesh
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo_pos])
        glDeleteBuffers(1, [vbo_norm])
        glDeleteBuffers(1, [vbo_uv])
        glDeleteBuffers(1, [ebo])

    def render(self, camera, chunk_keys_to_draw):
        if not self.chunk_meshes:
            return
        self.shader.use()
        self.shader.set_mat4("u_view", camera.view_matrix())
        self.shader.set_mat4("u_projection", camera.projection_matrix())
        self.shader.set_int("u_atlas", 0)
        self.texture_atlas.bind(0)

        for key in chunk_keys_to_draw:
            mesh = self.chunk_meshes.get(key)
            if mesh is None:
                continue
            vao, _vp, _vn, _vu, _ebo, count = mesh
            glBindVertexArray(vao)
            glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def destroy(self):
        for key in list(self.chunk_meshes.keys()):
            self.remove_chunk(*key)
