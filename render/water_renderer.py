"""
render/water_renderer.py
Draws water. Water is excluded from the regular cube mesher entirely (see
world/blocks.py CUSTOM_RENDER_BLOCKS) because it is not a cube and does not want
the shared block shader.

WHY IT IS NOT A CUBE
--------------------
A water cell's surface height is not a property of the cell. It is averaged, per
CORNER, from the flow levels of the four cells around that corner - vanilla's
RenderBlocks.getLiquidHeight. That is what makes a stream run visibly downhill
instead of descending in 1/9-block stairsteps, and it is why the mesher below
computes a (CH, CZ+1, CX+1) grid of corner heights rather than a per-block
height. A source is 8/9 of a block tall, not 1.0, which is also where the
horizon line across an ocean surface comes from.

WHY THE OCEAN USED TO LOOK LIKE PLAID
-------------------------------------
Three separate causes, all fixed here, and it is worth being explicit about
which does what because two of them look like polish and are not:

1. CONSTANT ALPHA. Water drawn at one fixed transparency shows the seabed
   through it equally at 1 block deep and at 16. The sea floor's noise is a sum
   of sines, so every one of its contours got painted onto the sea surface as a
   concentric ring. Alpha and colour now ramp with how much water is actually
   in the column (see the depth attribute), so the shallows stay clear over the
   sand - which is what makes a beach read as a beach - and the deep goes
   properly opaque. This is the big one.

2. DETAIL AT DISTANCE. Any high-contrast texture viewed at a grazing angle
   across an ocean is the textbook worst case for moire: one screen pixel covers
   hundreds of texels and the mip chain has nothing to converge on but grey
   mush. The atlas is GL_NEAREST_MIPMAP_LINEAR with no anisotropic filtering, so
   nothing downstream can rescue it. The ripple here is procedural and FADES OUT
   with distance, so far water converges on flat colour by construction. That is
   also why this shader samples no texture at all.

3. NO SKY. A real ocean is bright at the horizon because at a grazing angle you
   see the sky in it, not the water. The Fresnel term below is what makes the
   far sea lift into the fog instead of staying a flat blue slab underneath it.

Drawn after all opaque geometry, blended, with depth writes OFF (a
depth-writing transparent fragment hides everything drawn behind it afterwards)
and with face culling OFF, so the surface is still there when you look up at it
from underwater.
"""

import numpy as np
from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
    glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawElements, glDeleteVertexArrays, glDeleteBuffers,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT,
    GL_FALSE, GL_TRIANGLES, GL_UNSIGNED_INT,
    glEnable, glDisable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA, glDepthMask, GL_TRUE, GL_CULL_FACE,
)

import config
from core.shader import Shader
from render.chunk_renderer import _cull_chunks, _frustum_planes
from world.blocks import Block
from world.chunk import CX, CZ, CH, _OPAQUE_LOOKUP

VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in float in_depth;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_world;
out vec3 v_normal;
out float v_depth;
out float v_fog_dist;
out float v_shade;

void main() {
    vec4 view_pos = u_view * vec4(in_position, 1.0);
    gl_Position = u_projection * view_pos;

    v_world = in_position;
    v_normal = in_normal;
    v_depth = in_depth;
    v_fog_dist = -view_pos.z;

    // Same fixed-per-face shading model as the block shader, but compressed
    // toward 1.0. Water has no self-shadowing to describe: a stream's side face
    // at 0.6 next to its own top face at 1.0 reads as two different liquids
    // meeting at a seam rather than as one body of water.
    float axis_y = abs(in_normal.y);
    float axis_x = abs(in_normal.x);
    float axis_z = abs(in_normal.z);
    float is_top = step(0.0, in_normal.y);
    float y_component = mix(0.62, 1.0, is_top);
    v_shade = axis_y * y_component + axis_x * 0.78 + axis_z * 0.9;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 v_world;
in vec3 v_normal;
in float v_depth;
in float v_fog_dist;
in float v_shade;

uniform vec3 u_fog_color;
uniform float u_fog_start;
uniform float u_fog_end;
uniform vec3 u_camera_pos;
uniform float u_time;
uniform float u_underwater;      // 1.0 while the camera itself is submerged

uniform vec3 u_shallow_color;
uniform vec3 u_deep_color;
uniform float u_depth_full;
uniform float u_alpha_shallow;
uniform float u_alpha_deep;
uniform float u_ripple_fade_start;
uniform float u_ripple_fade_end;

out vec4 frag_color;

void main() {
    vec3 to_frag = v_world - u_camera_pos;
    float dist = length(to_frag);

    // Everything high-frequency is multiplied by this. Detail near the camera,
    // nothing at all past the fade - see the moire note in the module docstring.
    float detail = clamp(1.0 - (dist - u_ripple_fade_start)
                             / max(1.0, u_ripple_fade_end - u_ripple_fade_start), 0.0, 1.0);

    // Two sine fields at different frequencies/speeds so the surface never
    // reads as one travelling wave, plus a cheap hash for per-texel sparkle.
    float w = sin(v_world.x * 0.9 + u_time * 1.6) * sin(v_world.z * 0.75 - u_time * 1.2)
            + 0.5 * sin((v_world.x + v_world.z) * 1.7 - u_time * 2.4);
    float sparkle = fract(sin(dot(floor(v_world.xz * 5.0), vec2(12.9898, 78.233))) * 43758.5453);
    float ripple = (w * 0.055 + (sparkle - 0.5) * 0.05) * detail;

    // Depth drives BOTH colour and opacity, which is the whole trick: shallow
    // water lets the sand through and the deep does not.
    float depth_t = clamp(v_depth / max(1.0, u_depth_full), 0.0, 1.0);
    vec3 base = mix(u_shallow_color, u_deep_color, depth_t);
    base *= (1.0 + ripple);

    // Sky at grazing angles. Killed while submerged - there is no sky down
    // there, and leaving it on paints a bright band across everything you look
    // at sideways underwater.
    vec3 view_dir = normalize(to_frag);
    vec3 n = normalize(v_normal);
    float fresnel = pow(1.0 - clamp(abs(dot(view_dir, n)), 0.0, 1.0), 4.0);
    fresnel *= step(0.5, n.y) * (1.0 - u_underwater);
    base = mix(base, u_fog_color * 1.05, fresnel * 0.6);

    float alpha = mix(u_alpha_shallow, u_alpha_deep, depth_t);
    alpha = clamp(alpha + fresnel * 0.45, 0.05, 1.0);
    // Seen from below, the surface is the only thing between the player and the
    // sky. At the opacity that makes the open sea read correctly from above it
    // would be a solid blue ceiling, so it thins out while submerged.
    alpha = mix(alpha, 0.45, u_underwater);

    vec3 shaded = base * v_shade;
    float fog_factor = clamp((v_fog_dist - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);
    frag_color = vec4(mix(shaded, u_fog_color, fog_factor), alpha);
}
"""

# Same face table/winding as world/chunk.py FACES (CCW seen from outside).
# ly == 1 vertices do not sit at y+1: they are lifted to the fluid's corner
# height at that corner instead, which is what slopes the surface.
_FACES = (
    {"dir": (1, 0, 0), "corners": ((1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1))},    # +x
    {"dir": (-1, 0, 0), "corners": ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))},   # -x
    {"dir": (0, 1, 0), "corners": ((0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0))},    # +y
    {"dir": (0, -1, 0), "corners": ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))},   # -y
    {"dir": (0, 0, 1), "corners": ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))},    # +z
    {"dir": (0, 0, -1), "corners": ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))},   # -z
)


def _corner_sum(arr):
    """
    Sums each of the four cells surrounding every corner of the chunk's
    (CX+1) x (CZ+1) corner grid, from a (CH, CZ+2, CX+2) padded cell array.

    Corner (i, j) is the meeting point of padded cells x in {i, i+1} and
    z in {j, j+1} - i.e. local cells (i-1..i, j-1..j) - which is exactly the
    2x2 neighbourhood vanilla's getLiquidHeight averages.
    """
    return (arr[:, 0:CZ + 1, 0:CX + 1] + arr[:, 0:CZ + 1, 1:CX + 2]
            + arr[:, 1:CZ + 2, 0:CX + 1] + arr[:, 1:CZ + 2, 1:CX + 2])


def _corner_any(arr):
    """Boolean version of _corner_sum: is any of the four cells around this corner set."""
    return (arr[:, 0:CZ + 1, 0:CX + 1] | arr[:, 0:CZ + 1, 1:CX + 2]
            | arr[:, 1:CZ + 2, 0:CX + 1] | arr[:, 1:CZ + 2, 1:CX + 2])


def _corner_heights(blocks, meta):
    """
    Vanilla RenderBlocks.getLiquidHeight, vectorized over a whole chunk at once.
    Returns (CH, CZ+1, CX+1) float32 corner heights in 0..1.

    Per contributing cell:
      water, source or falling -> its fluid percentage, weighted 11
      water, mid-flow          -> its fluid percentage, weighted 1
      air                      -> percentage 1.0 (i.e. height 0), weighted 1
      anything solid           -> ignored entirely, weight 0

    The lopsided 11:1 weighting is vanilla's and it is deliberate: it means a
    corner touching even one source sits essentially at source height, so a pool
    has a flat surface right up to its edge and only the outermost film of
    flowing water slopes away. Averaging evenly instead visibly dishes the whole
    pool inward.

    Water directly above any of the four contributing cells forces the corner to
    1.0 - that is what welds a submerged column together into one solid body
    with no internal surfaces.
    """
    is_water = blocks == Block.WATER
    m = meta & 15
    falling = (m & 8) != 0
    level = np.where(falling, 0, m & 7)
    pct = (level.astype(np.float32) + 1.0) / 9.0
    source_like = falling | (level == 0)

    val = np.zeros(blocks.shape, dtype=np.float32)
    wgt = np.zeros(blocks.shape, dtype=np.float32)

    np.copyto(val, np.where(source_like, pct * 11.0, pct), where=is_water)
    np.copyto(wgt, np.where(source_like, np.float32(11.0), np.float32(1.0)), where=is_water)

    air = blocks == Block.AIR
    np.copyto(val, np.float32(1.0), where=air)
    np.copyto(wgt, np.float32(1.0), where=air)

    above_water = np.zeros(is_water.shape, dtype=bool)
    above_water[:-1] = is_water[1:]

    sum_val = _corner_sum(val)
    sum_wgt = _corner_sum(wgt)
    any_above = _corner_any(above_water)

    heights = 1.0 - sum_val / np.maximum(sum_wgt, 1e-6)
    heights = np.where(sum_wgt > 0.0, heights, 0.0)
    return np.where(any_above, np.float32(1.0), heights).astype(np.float32)


def build_water_mesh_arrays(chunk, world):
    """
    Returns (positions, normals, depths, indices) for every visible water face in
    this chunk, or None if the chunk holds no water at all (the overwhelmingly
    common case inland - one boolean check and we're out).
    """
    blocks = world.get_padded_blocks_for_chunk(chunk.cx, chunk.cz)
    if blocks is None:
        return None
    own_water = blocks[:, 1:-1, 1:-1] == Block.WATER
    if not own_water.any():
        return None

    meta = world.get_padded_meta_for_chunk(chunk.cx, chunk.cz)
    corner_h = _corner_heights(blocks, meta)

    # Pad the Y axis so the top/bottom of the world read as air, exactly as the
    # cube mesher does - there is no chunk above or below to pull from.
    y_padded = np.zeros((CH + 2, CZ + 2, CX + 2), dtype=np.uint8)
    y_padded[1:-1] = blocks

    # How much water stands at or below each cell in its own column. Used for the
    # depth-driven colour/alpha. cumsum rather than a downward scan because ocean
    # columns are contiguous, so the running count IS the depth; a flooded cave
    # elsewhere in the same column would nudge this, which costs nothing anyone
    # can see and saves a per-cell loop.
    depth_map = np.cumsum(own_water.astype(np.float32), axis=0)

    origin_x, origin_z = chunk.world_origin()
    all_positions, all_normals, all_depths, all_indices = [], [], [], []
    vertex_offset = 0

    for face in _FACES:
        dx, dy, dz = face["dir"]
        neighbor_ids = y_padded[1 + dy: 1 + dy + CH,
                                1 + dz: 1 + dz + CZ,
                                1 + dx: 1 + dx + CX]
        # Water hides its own faces against water (an ocean is otherwise ~24k
        # invisible internal faces per chunk, alpha-blended) and against
        # anything opaque, which is what keeps the seabed from being drawn over.
        visible = own_water & (neighbor_ids != Block.WATER) & ~_OPAQUE_LOOKUP[neighbor_ids]
        if not visible.any():
            continue

        ys, zs, xs = np.nonzero(visible)
        n = len(ys)
        cell_depth = depth_map[ys, zs, xs]

        face_positions = np.empty((n, 4, 3), dtype=np.float32)
        for vertex_index, (lx, ly, lz) in enumerate(face["corners"]):
            face_positions[:, vertex_index, 0] = xs + origin_x + lx
            face_positions[:, vertex_index, 2] = zs + origin_z + lz
            if ly == 1:
                # The one line that makes water water: an upper vertex sits at
                # its CORNER's height, not at the top of its cell.
                face_positions[:, vertex_index, 1] = ys + corner_h[ys, zs + lz, xs + lx]
            else:
                face_positions[:, vertex_index, 1] = ys

        all_positions.append(face_positions.reshape(-1, 3))
        all_normals.append(np.tile(np.array(face["dir"], dtype=np.float32), (n * 4, 1)))
        all_depths.append(np.repeat(cell_depth, 4))

        base = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        offsets = (np.arange(n, dtype=np.uint32) * 4 + vertex_offset).reshape(-1, 1)
        all_indices.append((base[None, :] + offsets).reshape(-1))
        vertex_offset += n * 4

    if not all_positions:
        return None

    return (np.concatenate(all_positions).astype(np.float32),
            np.concatenate(all_normals).astype(np.float32),
            np.concatenate(all_depths).astype(np.float32),
            np.concatenate(all_indices).astype(np.uint32))


class WaterMesh:
    """GPU buffers for one chunk's water."""

    def __init__(self):
        self.vao = None
        self.vbo_position = None
        self.vbo_normal = None
        self.vbo_depth = None
        self.ebo = None
        self.index_count = 0

    def upload(self, positions, normals, depths, indices):
        self.destroy()

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

        self.vbo_depth = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_depth)
        glBufferData(GL_ARRAY_BUFFER, depths.nbytes, depths, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)
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
        for buffer_id in (self.vbo_position, self.vbo_normal, self.vbo_depth, self.ebo):
            if buffer_id is not None:
                glDeleteBuffers(1, [buffer_id])
        self.vbo_position = self.vbo_normal = self.vbo_depth = self.ebo = None
        self.index_count = 0


class WaterRenderer:
    def __init__(self):
        self.shader = Shader(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, name="water")
        self.chunk_meshes: dict[tuple[int, int], WaterMesh] = {}
        self.time = 0.0

    def update(self, dt: float):
        self.time += dt

    def rebuild_chunk(self, chunk, world):
        key = (chunk.cx, chunk.cz)
        arrays = build_water_mesh_arrays(chunk, world)

        old = self.chunk_meshes.pop(key, None)
        if old is not None:
            old.destroy()
        if arrays is None:
            return  # dry chunk: hold no GPU resources for it at all

        mesh = WaterMesh()
        mesh.upload(*arrays)
        self.chunk_meshes[key] = mesh

    def remove_chunk(self, cx: int, cz: int):
        mesh = self.chunk_meshes.pop((cx, cz), None)
        if mesh is not None:
            mesh.destroy()

    def render(self, camera, chunk_keys_to_draw, fog_color, fog_start, fog_end, underwater: bool):
        keys = [key for key in chunk_keys_to_draw if key in self.chunk_meshes]
        if not keys:
            return
        # Same conservative AABB-vs-frustum reject the block renderer uses. It
        # matters more here, not less: water chunks are almost all ocean, so the
        # ones behind the camera are exactly the large ones.
        keys = _cull_chunks(_frustum_planes(camera), keys)
        if not keys:
            return

        self.shader.use()
        self.shader.set_mat4("u_view", camera.view_matrix())
        self.shader.set_mat4("u_projection", camera.projection_matrix())
        self.shader.set_vec3("u_fog_color", *fog_color)
        self.shader.set_float("u_fog_start", fog_start)
        self.shader.set_float("u_fog_end", fog_end)
        self.shader.set_vec3("u_camera_pos", *camera.position)
        self.shader.set_float("u_time", self.time)
        self.shader.set_float("u_underwater", 1.0 if underwater else 0.0)
        self.shader.set_vec3("u_shallow_color", *config.WATER_SHALLOW_COLOR)
        self.shader.set_vec3("u_deep_color", *config.WATER_DEEP_COLOR)
        self.shader.set_float("u_depth_full", config.WATER_DEPTH_FULL)
        self.shader.set_float("u_alpha_shallow", config.WATER_ALPHA_SHALLOW)
        self.shader.set_float("u_alpha_deep", config.WATER_ALPHA_DEEP)
        self.shader.set_float("u_ripple_fade_start", config.WATER_RIPPLE_FADE_START)
        self.shader.set_float("u_ripple_fade_end", config.WATER_RIPPLE_FADE_END)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Depth WRITE off: a blended fragment that writes depth stops everything
        # behind it from ever being drawn (this is the same bug glass had).
        # Depth TEST stays on, so water is still hidden behind terrain in front
        # of it.
        glDepthMask(GL_FALSE)
        # Culling off so the surface exists when looked at from underneath.
        glDisable(GL_CULL_FACE)

        for key in keys:
            self.chunk_meshes[key].draw()

        glEnable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

    def destroy(self):
        for mesh in self.chunk_meshes.values():
            mesh.destroy()
        self.chunk_meshes.clear()
