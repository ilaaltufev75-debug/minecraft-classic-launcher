"""
render/chunk_renderer.py
Turns a Chunk's CPU-side mesh data (from Chunk.build_mesh_data) into GPU
buffers and draws them. Each chunk gets one VAO per (block_id, tex_face)
group merged into a single draw call per chunk by combining all groups into
one big interleaved vertex buffer with UVs already resolved against the
shared texture atlas - this keeps draw calls low (one per chunk, not one
per block-type-per-chunk) which matters a lot once dozens of chunks are
visible under an infinite-world render distance.

Also owns the world shader (position/normal/uv attributes, MVP + fog
uniforms) since the shader's vertex layout is tightly coupled to how this
module packs chunk mesh data.
"""

import numpy as np
from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
    glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawElements, glDeleteVertexArrays, glDeleteBuffers,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT,
    GL_FALSE, GL_TRIANGLES, GL_UNSIGNED_INT,
    glEnable, glDisable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    glDepthMask, GL_TRUE, GL_FALSE as GL_FALSE_BOOL,
)

from core.shader import Shader
from world.chunk import ALPHA_BLEND_BLOCKS, CX, CZ, CH


def _frustum_planes(camera) -> np.ndarray:
    """
    Extracts the six frustum planes from the camera, as a (6, 4) array of
    (a, b, c, d) with a*x + b*y + c*z + d >= 0 meaning "inside".

    Camera.view_matrix()/projection_matrix() hand back matrices ALREADY
    transposed into OpenGL's column-major layout (see core/camera.py - they
    end in `return m.T`), so they must be transposed BACK before this math:
    the Gribb-Hartmann extraction below reads rows of the row-major clip
    matrix. Feeding it the column-major arrays directly yields a frustum
    that is effectively rotated, which culls chunks the player is looking
    straight at while keeping ones behind them.
    """
    proj = np.asarray(camera.projection_matrix(), dtype=np.float64).T
    view = np.asarray(camera.view_matrix(), dtype=np.float64).T
    clip = proj @ view

    r0, r1, r2, r3 = clip[0], clip[1], clip[2], clip[3]
    planes = np.stack([
        r3 + r0,   # left
        r3 - r0,   # right
        r3 + r1,   # bottom
        r3 - r1,   # top
        r3 + r2,   # near
        r3 - r2,   # far
    ])
    # Normalize so `dist` below is a true signed distance. Not strictly
    # required for a pure inside/outside test (only the sign matters), but it
    # keeps the numbers meaningful if this is ever reused for LOD selection.
    norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return planes / norms


def _cull_chunks(planes: np.ndarray, keys):
    """
    Filters chunk (cx, cz) keys down to those whose column AABB intersects
    the frustum, testing every chunk against every plane in one vectorized
    shot rather than looping in Python (the loop would hand back a chunk of
    the savings it just made once a few thousand chunks are loaded).

    Uses the standard conservative "positive vertex" test: for each plane,
    take the AABB corner furthest along the plane normal; if even that corner
    is behind the plane, the whole box is, and the chunk can be skipped. This
    can pass a box that is technically outside near frustum corners, which is
    fine - it only ever costs a redundant draw, never a missing chunk.
    """
    if len(keys) == 0:
        return []
    arr = np.asarray(keys, dtype=np.float64)
    min_x = arr[:, 0] * CX
    max_x = min_x + CX
    min_z = arr[:, 1] * CZ
    max_z = min_z + CZ

    a = planes[:, 0][None, :]
    b = planes[:, 1][None, :]
    c = planes[:, 2][None, :]
    d = planes[:, 3][None, :]

    px = np.where(a >= 0.0, max_x[:, None], min_x[:, None])
    py = np.where(b >= 0.0, float(CH), 0.0)
    pz = np.where(c >= 0.0, max_z[:, None], min_z[:, None])

    outside = (a * px + b * py + c * pz + d) < 0.0
    visible = ~outside.any(axis=1)
    return [keys[i] for i in np.nonzero(visible)[0]]

VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec2 v_uv;
out float v_fog_dist;
out float v_shade;

void main() {
    vec4 view_pos = u_view * vec4(in_position, 1.0);
    gl_Position = u_projection * view_pos;
    v_uv = in_uv;
    v_fog_dist = -view_pos.z;

    // Minecraft-style "fixed per-face" shading rather than a continuous
    // dot-product-with-the-sun gradient: each face direction gets a constant
    // brightness regardless of camera/sun angle - top brightest, the two
    // horizontal axes at two different mid brightnesses (so a block doesn't
    // look flat/uniform from the side), and bottom darkest. This is what
    // actually produces Minecraft's recognizable flat, graphic-novel-like
    // lighting; a smooth directional light (an earlier approach here) reads
    // as much softer/more realistic and doesn't match the reference look.
    //
    // Uses mix()/step() instead of a ternary (?:) to pick the top-vs-bottom
    // brightness: ternary operators in GLSL have historically been a source
    // of driver-specific miscompilation on some GPUs (notably older/weaker
    // integrated drivers), which can silently evaluate to 0 and produce an
    // all-black render - exactly the symptom reported on real hardware even
    // though the equivalent Python-side math was verified correct.
    float axis_y = abs(in_normal.y);
    float axis_x = abs(in_normal.x);
    float axis_z = abs(in_normal.z);

    float top_shade = 1.0;
    float bottom_shade = 0.5;
    float ns_shade = 0.8;   // north/south faces (+-z)
    float ew_shade = 0.6;   // east/west faces (+-x)

    float is_top = step(0.0, in_normal.y);  // 1.0 if normal.y > 0 (top), else 0.0 (bottom or side)
    float y_component = mix(bottom_shade, top_shade, is_top);
    v_shade = axis_y * y_component + axis_x * ew_shade + axis_z * ns_shade;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 v_uv;
in float v_fog_dist;
in float v_shade;

uniform sampler2D u_atlas;
uniform vec3 u_fog_color;
uniform float u_fog_start;
uniform float u_fog_end;

out vec4 frag_color;

void main() {
    vec4 tex_color = texture(u_atlas, v_uv);
    if (tex_color.a < 0.1) discard;  // leaves' transparent gaps

    vec3 shaded = tex_color.rgb * v_shade;

    float fog_factor = clamp((v_fog_dist - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);
    vec3 final_color = mix(shaded, u_fog_color, fog_factor);

    frag_color = vec4(final_color, tex_color.a);
}
"""


class ChunkMesh:
    """GPU-side buffers for one chunk's combined mesh."""

    def __init__(self):
        self.vao = None
        self.vbo_position = None
        self.vbo_normal = None
        self.vbo_uv = None
        self.ebo = None
        self.index_count = 0

    def upload(self, positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray, indices: np.ndarray):
        self.destroy()  # in case this mesh is being rebuilt in place

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo_position = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_position)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        self.vbo_normal = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normal)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        self.vbo_uv = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_uv)
        glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        self.index_count = len(indices)
        glBindVertexArray(0)

    def draw(self):
        if self.vao is None or self.index_count == 0:
            return
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def destroy(self):
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = None
        for vbo in (self.vbo_position, self.vbo_normal, self.vbo_uv, self.ebo):
            if vbo is not None:
                glDeleteBuffers(1, [vbo])
        self.vbo_position = self.vbo_normal = self.vbo_uv = self.ebo = None
        self.index_count = 0


def build_chunk_mesh(chunk, world, texture_atlas):
    """
    Combines all (block_id, tex_face) groups from Chunk.build_mesh_data into
    TWO merged vertex/index buffers per chunk: one for ordinary opaque
    blocks and one for alpha-blended blocks (glass) - remapping each
    group's UVs from its own [0,1] tile-local range into the shared atlas
    UV rect. Kept as two separate meshes (not one) so the renderer can draw
    opaque geometry first with normal depth writes, then blended geometry
    afterward with depth writes disabled - see WorldRenderer.render for why
    that separation matters (a depth-writing transparent block otherwise
    hides anything behind it that hasn't been drawn yet).

    Returns (opaque_mesh, transparent_mesh).
    """
    padded = world.get_padded_blocks_for_chunk(chunk.cx, chunk.cz)
    if padded is None:
        return ChunkMesh(), ChunkMesh()  # chunk was unloaded between being queued and built
    mesh_groups = chunk.build_mesh_data(padded)

    def _merge(groups):
        all_positions, all_normals, all_uvs, all_indices = [], [], [], []
        vertex_offset = 0
        for (block_id, tex_face), group in groups:
            uv_rect = texture_atlas.uv_for(block_id, tex_face)
            if uv_rect is None:
                continue
            u0, v0, u1, v1 = uv_rect
            local_uvs = group["uvs"]
            remapped_u = u0 + local_uvs[:, 0] * (u1 - u0)
            remapped_v = v0 + local_uvs[:, 1] * (v1 - v0)
            remapped_uv = np.stack([remapped_u, remapped_v], axis=1).astype(np.float32)

            all_positions.append(group["positions"])
            all_normals.append(group["normals"])
            all_uvs.append(remapped_uv)
            all_indices.append(group["indices"] + vertex_offset)
            vertex_offset += len(group["positions"])

        mesh = ChunkMesh()
        if not all_positions:
            return mesh
        mesh.upload(
            np.concatenate(all_positions).astype(np.float32),
            np.concatenate(all_normals).astype(np.float32),
            np.concatenate(all_uvs).astype(np.float32),
            np.concatenate(all_indices).astype(np.uint32),
        )
        return mesh

    opaque_groups = [(k, g) for k, g in mesh_groups.items() if k[0] not in ALPHA_BLEND_BLOCKS]
    transparent_groups = [(k, g) for k, g in mesh_groups.items() if k[0] in ALPHA_BLEND_BLOCKS]

    return _merge(opaque_groups), _merge(transparent_groups)


class WorldRenderer:
    """Owns the world shader and the collection of per-chunk GPU meshes."""

    def __init__(self, texture_atlas):
        self.shader = Shader(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, name="world")
        self.texture_atlas = texture_atlas
        self.chunk_meshes: dict[tuple[int, int], ChunkMesh] = {}
        self.chunk_meshes_transparent: dict[tuple[int, int], ChunkMesh] = {}

    def rebuild_chunk(self, chunk, world):
        key = (chunk.cx, chunk.cz)
        opaque_mesh, transparent_mesh = build_chunk_mesh(chunk, world, self.texture_atlas)

        old = self.chunk_meshes.get(key)
        if old is not None:
            old.destroy()
        self.chunk_meshes[key] = opaque_mesh

        old_t = self.chunk_meshes_transparent.get(key)
        if old_t is not None:
            old_t.destroy()
        self.chunk_meshes_transparent[key] = transparent_mesh

        chunk.dirty = False
        chunk.has_mesh = True

    def remove_chunk(self, cx: int, cz: int):
        key = (cx, cz)
        mesh = self.chunk_meshes.pop(key, None)
        if mesh is not None:
            mesh.destroy()
        mesh_t = self.chunk_meshes_transparent.pop(key, None)
        if mesh_t is not None:
            mesh_t.destroy()

    def render(self, camera, chunk_positions_to_draw, render_distance_chunks=None,
               fog_color=None, fog_start=None, fog_end=None):
        self.shader.use()
        self.shader.set_mat4("u_view", camera.view_matrix())
        self.shader.set_mat4("u_projection", camera.projection_matrix())
        self.shader.set_int("u_atlas", 0)
        self.texture_atlas.bind(0)

        import config
        # Fog scales with the actual render distance setting rather than a
        # fixed constant: fog_end lands right at the edge of what's actually
        # drawn (render_distance_chunks * chunk size), so the world edge/pop-in
        # is always hidden regardless of how far the player has set the view
        # distance, and fog_start leaves a comfortable clear zone in front of
        # that (roughly 55% of the way out) so fog reads as gradual depth haze
        # rather than a sudden wall - closer to how Minecraft's own fog eases
        # in before the render distance cutoff instead of appearing abruptly.
        #
        # The caller can override all three, and does while the camera is
        # underwater: down there fog is not distant haze, it is the water
        # itself, and it has to close to a handful of blocks.
        render_distance_chunks = render_distance_chunks or config.DEFAULT_RENDER_DISTANCE
        if fog_end is None:
            fog_end = render_distance_chunks * config.CHUNK_SIZE_X
        if fog_start is None:
            fog_start = fog_end * 0.55
        if fog_color is None:
            fog_color = config.FOG_COLOR

        self.shader.set_vec3("u_fog_color", *fog_color)
        self.shader.set_float("u_fog_start", fog_start)
        self.shader.set_float("u_fog_end", fog_end)

        # Frustum culling. Previously every loaded chunk was submitted every
        # frame, including the roughly three quarters of them sitting behind
        # the camera at FOV 70. The GPU did throw them away - but only after
        # the driver had paid for the draw call, the VAO bind and the vertex
        # fetch. Rejecting them CPU-side with four multiply-adds per chunk is
        # the single biggest win available here, and the one thing that makes
        # a large render distance viable at all.
        visible_keys = _cull_chunks(_frustum_planes(camera), list(chunk_positions_to_draw))

        # Pass 1: opaque geometry, normal depth test + depth write, no blending.
        # This must happen BEFORE any transparent geometry so glass (etc.)
        # correctly tests against a depth buffer that already has all solid
        # blocks in it.
        for key in visible_keys:
            mesh = self.chunk_meshes.get(key)
            if mesh is not None:
                mesh.draw()

        # Pass 2: transparent geometry (glass). Depth test stays ON (so
        # glass is still correctly hidden behind solid blocks in front of
        # it) but depth WRITE is turned off, so glass never blocks anything
        # drawn/tested afterward - this is what fixes "everything behind
        # glass disappears": previously glass wrote its own depth in the
        # same opaque pass, and blocks materially behind it (drawn later,
        # or never re-drawn since chunk meshes are static) would fail the
        # depth test against glass's nearer depth value and never appear.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE_BOOL)

        for key in visible_keys:
            mesh = self.chunk_meshes_transparent.get(key)
            if mesh is not None:
                mesh.draw()

        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
