"""
render/shadow_renderer.py
Cheap flat ground-shadow decals drawn under floating solid blocks (trees,
mainly) - the classic Minecraft-style soft dark blob on the ground rather
than real shadow mapping. All shadow spots across every visible chunk are
drawn in a SINGLE instanced draw call (one shared unit-quad mesh, one
per-instance buffer of center/size/strength) rather than one draw call per
spot - the original per-spot glDrawArrays + per-spot uniform updates showed
up as a major frame-time cost once a render distance had more than a
handful of trees in view (each Python-side PyOpenGL call carries real
overhead, and a forest can easily mean hundreds of shadow spots).

Shadow spot data comes from Chunk.build_shadow_spots() and is rebuilt
alongside the chunk's regular mesh (see chunk_renderer.rebuild_chunk /
main.py), so shadows never need their own separate dirty-tracking pass.
"""

import numpy as np
from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
    glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
    glVertexAttribDivisor, glDrawArraysInstanced,
    glDeleteVertexArrays, glDeleteBuffers,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_DYNAMIC_DRAW, GL_FLOAT, GL_FALSE,
    GL_TRIANGLE_FAN, glEnable, glDisable, GL_BLEND, glBlendFunc,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, glDepthMask, GL_TRUE,
    GL_FALSE as GL_FALSE_BOOL, GL_CULL_FACE,
)

from core.shader import Shader

VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec2 in_offset;    // unit quad corner, -0.5..0.5
layout (location = 1) in vec3 in_center;    // per-instance world-space center
layout (location = 2) in float in_size;     // per-instance quad size
layout (location = 3) in float in_strength; // per-instance opacity

uniform mat4 u_view;
uniform mat4 u_projection;

out float v_edge;
out float v_strength;

void main() {
    vec3 world_pos = in_center + vec3(in_offset.x * in_size, 0.0, in_offset.y * in_size);
    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
    // Chebyshev distance (max of |x|,|y|) rather than Euclidean length, so
    // the falloff band in the fragment shader traces the SQUARE edge of
    // the quad evenly on all four sides instead of rounding it into a
    // circle inscribed within the quad.
    v_edge = max(abs(in_offset.x), abs(in_offset.y)) * 2.0;  // 0 at center, 1 at the square's edge
    v_strength = in_strength;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in float v_edge;
in float v_strength;
out vec4 frag_color;

void main() {
    // Square decal covering the full block footprint, with just the
    // outermost sliver softened so the edge doesn't read as a hard-cut
    // rectangle - smoothstep starts fading only in the last ~12% near the
    // border, so the shadow still looks like a solid square block-sized
    // patch (not a circle) rather than a sharp graphic edge.
    float falloff = 1.0 - smoothstep(0.78, 1.0, v_edge);
    float alpha = falloff * v_strength;
    if (alpha < 0.01) discard;
    frag_color = vec4(0.0, 0.0, 0.0, alpha);
}
"""

_QUAD_CORNERS = np.array([
    (0.0, 0.0),
    (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5),
], dtype=np.float32)  # triangle-fan: center then the 4 corners, closed

_SHADOW_SIZE = 1.0  # full block footprint, matching the requested "square, full block" look
_MAX_INSTANCES = 4096  # generous cap; instance buffer is resized up if ever exceeded


class ChunkShadowMesh:
    """Holds the shadow-spot list for one chunk (position/size/strength per
    spot). Geometry itself is the single shared unit-quad VAO owned by
    ShadowRenderer; this just holds the per-instance data contributed by
    this chunk, concatenated with every other visible chunk's spots into
    one instance buffer at render time."""

    __slots__ = ("spots",)

    def __init__(self, spots):
        self.spots = spots  # list of (world_x+0.5, ground_y, world_z+0.5, strength)


class ShadowRenderer:
    def __init__(self):
        self.shader = Shader(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, name="shadow")
        self.chunk_shadows: dict[tuple[int, int], ChunkShadowMesh] = {}

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo_quad = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_quad)
        glBufferData(GL_ARRAY_BUFFER, _QUAD_CORNERS.nbytes, _QUAD_CORNERS, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # per-instance buffer: (center.xyz, size, strength) = 5 floats/instance,
        # reuploaded each frame with whatever spots are currently visible
        self._instance_capacity = _MAX_INSTANCES
        self.vbo_instances = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_instances)
        glBufferData(GL_ARRAY_BUFFER, self._instance_capacity * 5 * 4, None, GL_DYNAMIC_DRAW)

        stride = 5 * 4
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, None)      # center
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        from ctypes import c_void_p
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, c_void_p(3 * 4))  # size
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, c_void_p(4 * 4))  # strength
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)

        glBindVertexArray(0)

    def rebuild_chunk(self, chunk, world):
        key = (chunk.cx, chunk.cz)
        raw_spots = chunk.build_shadow_spots()
        # raw ground_surface_y from Chunk.build_shadow_spots() is already the
        # Y of the ground's TOP FACE - only a tiny epsilon is added here to
        # avoid z-fighting with the terrain mesh directly beneath it. (A
        # previous version added a full +1 here on top of an already-correct
        # value from the chunk, which floated every shadow a whole block
        # above the actual ground - fixed by not double-offsetting.)
        spots = [(wx + 0.5, gy + 0.01, wz + 0.5, strength) for (wx, gy, wz, strength) in raw_spots]
        self.chunk_shadows[key] = ChunkShadowMesh(spots)

    def remove_chunk(self, cx: int, cz: int):
        self.chunk_shadows.pop((cx, cz), None)

    def render(self, camera, chunk_keys_to_draw):
        all_spots = []
        for key in chunk_keys_to_draw:
            mesh = self.chunk_shadows.get(key)
            if mesh is not None and mesh.spots:
                all_spots.extend(mesh.spots)
        if not all_spots:
            return

        if len(all_spots) > self._instance_capacity:
            all_spots = all_spots[:self._instance_capacity]  # hard cap - avoids unbounded per-frame cost

        instance_data = np.array(all_spots, dtype=np.float32)  # (N, 4): x,y,z,strength
        n = instance_data.shape[0]
        # expand to (N, 5): x,y,z,size,strength to match the vertex layout above
        buf = np.empty((n, 5), dtype=np.float32)
        buf[:, 0:3] = instance_data[:, 0:3]
        buf[:, 3] = _SHADOW_SIZE
        buf[:, 4] = instance_data[:, 3]

        self.shader.use()
        self.shader.set_mat4("u_view", camera.view_matrix())
        self.shader.set_mat4("u_projection", camera.projection_matrix())

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE_BOOL)  # shadows shouldn't occlude/z-fight geometry drawn after them
        # Flat ground-facing decal: whether the quad's winding reads as
        # front- or back-facing depends on the camera looking down at it
        # from above vs. from other angles, and backface culling (enabled
        # globally for normal terrain rendering) would otherwise make the
        # decal invisible from straight overhead - exactly the "no shadows
        # visible at all" symptom. Culling doesn't matter for a flat double-
        # sided decal, so just disable it for this draw call.
        glDisable(GL_CULL_FACE)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_instances)
        glBufferData(GL_ARRAY_BUFFER, buf.nbytes, buf, GL_DYNAMIC_DRAW)  # orphan + refill, avoids sync stalls
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 6, n)
        glBindVertexArray(0)

        glEnable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

    def destroy(self):
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = None
        for vbo in (self.vbo_quad, self.vbo_instances):
            if vbo is not None:
                glDeleteBuffers(1, [vbo])
        self.vbo_quad = self.vbo_instances = None

