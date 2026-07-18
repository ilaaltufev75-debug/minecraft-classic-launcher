"""
render/door_renderer.py
Draws doors as a thin textured slab rather than a full cube - the regular
chunk mesher only produces full cubes, so doors (Block.DOOR) are excluded
from it entirely (see world/chunk.py CUSTOM_RENDER_BLOCKS) and drawn here
instead, using the door's per-block metadata (facing + open/closed + which
vertical half, see world/doors.py) to position/rotate a flat box per door
block. A door is two blocks tall (bottom half + top half), each stored as
its own Block.DOOR entry in the chunk with its own metadata - this renderer
just draws whatever door blocks exist, one instance per block, so a
two-block door naturally produces two draw instances without any special
casing here.

Geometry: a thin box (like a squashed cube) so the door reads as having
real thickness rather than a flat unlit paper cutout, textured on all
faces - bottom half gets the doorknob-bearing tile, top half a plain
panel tile, matching real Minecraft's two-texture door. One instance per
door block; door counts are always small so a straightforward per-instance
uniform draw call (not a shared instanced buffer like shadows) keeps this
simple without any real performance cost.
"""

import numpy as np
from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
    glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawElements, glDeleteVertexArrays, glDeleteBuffers,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT,
    GL_FALSE, GL_TRIANGLES, GL_UNSIGNED_INT,
)

from core.shader import Shader
from world.doors import door_collision_bounds
from world.blocks import Block

VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 in_position;   // unit cube corner, 0..1 (local)
layout (location = 1) in vec3 in_normal;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_block_pos;    // world-space origin of the door's block cell
uniform vec4 u_bounds;       // min_x, min_z, max_x, max_z in local 0..1 space
uniform float u_wide_axis;   // 0.0 = the door's wide face normal is +-X, 1.0 = it's +-Z
uniform vec4 u_uv_rect;      // u0, v0, u1, v1 - this door half's tile rect in the shared atlas

out vec2 v_uv;
out float v_shade;

void main() {
    // remap the unit cube's local x/z into the door's actual thin-slab
    // footprint for this facing/open-state, while y stays full-height
    float local_x = mix(u_bounds.x, u_bounds.z, in_position.x);
    float local_z = mix(u_bounds.y, u_bounds.w, in_position.z);
    vec3 world_pos = u_block_pos + vec3(local_x, in_position.y, local_z);

    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);

    // BUG FIX: UVs used to come from a fixed per-vertex attribute baked
    // into the shared unit-cube mesh, tied to which of the 6 cube FACES a
    // vertex belonged to in local model space. But which face is the
    // door's wide "panel" face (needs the door texture right-side-up
    // across its full width) versus which is the thin "edge" face swaps
    // depending on facing/open state - opening a door rotates it 90
    // degrees, so the face that was wide becomes the thin edge and vice
    // versa. With fixed per-vertex UVs, the door texture kept being
    // painted onto whatever face happened to own those UV values
    // regardless of whether that face was currently wide or thin, so the
    // panel artwork visibly "jumped" to a different, wrongly-shaped face
    // every time the door opened/closed. Instead, UVs are now computed
    // HERE from the vertex's own local position along whichever axis is
    // actually the wide one for this door's current orientation (u_wide_axis),
    // then remapped into the shared atlas tile rect - so the texture
    // always maps onto the true panel face, at a stable full 0..1 span,
    // no matter how the door is currently rotated.
    float along_wide_axis = mix(in_position.x, in_position.z, u_wide_axis);
    vec2 local_uv = vec2(along_wide_axis, in_position.y);
    v_uv = mix(u_uv_rect.xy, u_uv_rect.zw, local_uv);

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

# Unit cube (0..1) - same corner/face layout style as chunk.py's FACES table,
# reused here since a door is just a squashed cube.
_CUBE_FACES = [
    {"dir": (1, 0, 0), "corners": ((1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1))},
    {"dir": (-1, 0, 0), "corners": ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))},
    {"dir": (0, 1, 0), "corners": ((0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0))},
    {"dir": (0, -1, 0), "corners": ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))},
    {"dir": (0, 0, 1), "corners": ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))},
    {"dir": (0, 0, -1), "corners": ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))},
]


def _build_cube_mesh():
    positions, normals, indices = [], [], []
    for face in _CUBE_FACES:
        base = len(positions)
        for corner in face["corners"]:
            positions.append(corner)
            normals.append(face["dir"])
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return (np.array(positions, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(indices, dtype=np.uint32))


class DoorRenderer:
    def __init__(self, texture_atlas):
        self.texture_atlas = texture_atlas
        self.shader = Shader(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, name="door")
        self.chunk_doors: dict[tuple[int, int], list] = {}  # key -> list of (x,y,z,facing,is_open,is_top)

        positions, normals, indices = _build_cube_mesh()
        self.index_count = len(indices)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        self.vbo_normal = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normal)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

        # UV rects for the door's two textures (bottom half has the
        # doorknob, top half is a plain panel) - looked up once here and
        # passed as a uniform per-draw, since the shader now computes each
        # vertex's UV position dynamically (see VERTEX_SHADER_SRC) rather
        # than reading it from a static per-vertex attribute.
        self._bottom_uv_rect = texture_atlas.uv_for(Block.DOOR, "bottom_half") or (0.0, 0.0, 1.0, 1.0)
        self._top_uv_rect = texture_atlas.uv_for(Block.DOOR, "top_half") or (0.0, 0.0, 1.0, 1.0)

    def rebuild_chunk(self, chunk, world):
        key = (chunk.cx, chunk.cz)
        instances = chunk.build_door_instances()
        if instances:
            self.chunk_doors[key] = instances
        else:
            self.chunk_doors.pop(key, None)

    def remove_chunk(self, cx: int, cz: int):
        self.chunk_doors.pop((cx, cz), None)

    def render(self, camera, chunk_keys_to_draw):
        all_instances = []
        for key in chunk_keys_to_draw:
            instances = self.chunk_doors.get(key)
            if instances:
                all_instances.extend(instances)
        if not all_instances:
            return

        self.shader.use()
        self.shader.set_mat4("u_view", camera.view_matrix())
        self.shader.set_mat4("u_projection", camera.projection_matrix())
        self.shader.set_int("u_atlas", 0)
        self.texture_atlas.bind(0)

        glBindVertexArray(self.vao)
        for (wx, wy, wz, facing, is_open, is_top, hinge) in all_instances:
            lmin_x, lmin_z, lmax_x, lmax_z = door_collision_bounds(facing, is_open, hinge)
            self.shader.set_vec3("u_block_pos", float(wx), float(wy), float(wz))
            self.shader.set_vec4("u_bounds", lmin_x, lmin_z, lmax_x, lmax_z)

            # Which local axis is currently the door's WIDE (panel) axis:
            # NORTH/SOUTH facings span the full width along X when closed
            # and along Z when open (and vice versa for EAST/WEST) - see
            # world/doors.py's CLOSED_COLLISION_BOUNDS/OPEN_COLLISION_BOUNDS.
            # This tells the vertex shader which axis to read the texture's
            # horizontal coordinate from, so the panel artwork always lands
            # on the actually-wide face instead of "jumping" to whichever
            # face used to be wide before the door moved. This holds
            # regardless of hinge - open bounds always rotate the door's
            # footprint onto the perpendicular axis relative to its closed
            # orientation, whichever corner it pivots around.
            is_ns_facing = facing in (0, 2)  # FACING_NORTH, FACING_SOUTH
            wide_axis_is_z = is_ns_facing == is_open
            self.shader.set_float("u_wide_axis", 1.0 if wide_axis_is_z else 0.0)

            uv_rect = self._top_uv_rect if is_top else self._bottom_uv_rect
            self.shader.set_vec4("u_uv_rect", *uv_rect)

            glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def destroy(self):
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = None
        for vbo in (self.vbo_pos, self.vbo_normal, self.ebo):
            if vbo is not None:
                glDeleteBuffers(1, [vbo])
        self.vbo_pos = self.vbo_normal = self.ebo = None
