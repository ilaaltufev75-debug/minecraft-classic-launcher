"""
render/hand.py
Draws the player's first-person viewmodel: the HAND (a simple skin-toned
box, roughly cubic like Minecraft's actual arm proportions) and the HELD
ITEM as a separate object sharing the hand's animated "wrist" transform
(see AnimationState/_wrist_transform), so they always move together:
  - IDLE BOB: a small sway synced to the player's horizontal movement speed
    (only while grounded and not flying), like footsteps nudging the arm.
  - SWING: a quick forward-dip-and-return arc triggered once per attack
    (breaking) or per placement.

HELD ITEM rendering depends on what''s in hand, matching vanilla Minecraft:
  - Ordinary full-cube blocks (dirt, stone, planks, ...): a real 3D cube
    with the SAME per-face textures as the world (top/bottom/side each
    mapped to their own atlas tile, not one texture stretched over all six
    faces) - built from world/chunk.py''s FACES table, the exact same
    source of truth the world renderer uses, so "held in hand" and
    "placed in the world" always look identical.
  - Custom-shaped blocks (door, fence, stairs): their REAL geometry (the
    same box lists world/doors.py, world/fences.py, world/stairs.py use
    for collision/rendering), not a flat icon - so a door looks like a
    door in your hand, a fence looks like a fence, etc.
  - Tools and other non-block items: a cross-sprite (two quads crossed at
    90 degrees, like Minecraft''s classic item/plant billboard) showing the
    item''s own icon texture at its native resolution, so high-res hand-
    drawn art stays crisp instead of being force-downscaled.
"""

import math
import numpy as np
from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
    glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawElements, glClear, GL_DEPTH_BUFFER_BIT, glGenTextures, glBindTexture,
    glTexImage2D, glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER, GL_NEAREST, GL_RGBA, GL_UNSIGNED_BYTE,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, glActiveTexture,
    GL_TEXTURE0, GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW,
    GL_FLOAT, GL_FALSE, GL_TRIANGLES, GL_UNSIGNED_INT,
    glDisable, glEnable, GL_CULL_FACE,
)

from core.shader import Shader
from world.blocks import get_item_def, Block, CUSTOM_RENDER_BLOCKS
from world.chunk import FACES as CHUNK_FACES

# ---------------------------------------------------------------------------
# Shared shader: draws a textured (or solid-color-fallback) mesh with the
# same fixed per-axis "Minecraft-style" shading as the world/chunk shader,
# so hand, held block, and world blocks all read as consistently lit.
# ---------------------------------------------------------------------------
VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

uniform mat4 u_model;
uniform mat4 u_projection;

out vec2 v_uv;
out float v_shade;

void main() {
    gl_Position = u_projection * u_model * vec4(in_position, 1.0);
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
uniform sampler2D u_tex;
uniform vec3 u_solid_color;
uniform int u_use_texture;
out vec4 frag_color;

void main() {
    vec4 tex_color = u_use_texture == 1 ? texture(u_tex, v_uv) : vec4(u_solid_color, 1.0);
    if (u_use_texture == 1 && tex_color.a < 0.1) discard;
    frag_color = vec4(tex_color.rgb * v_shade, tex_color.a);
}
"""


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def _box_mesh_simple(scale=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)):
    """Unit box (all 6 faces share one [0,1] UV square) - used for the plain
    skin-toned hand, which has no texture."""
    sx, sy, sz = scale
    cx, cy, cz = center
    faces = [
        ((1, 0, 0), [(0.5, -0.5, 0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]),
        ((-1, 0, 0), [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)]),
        ((0, 1, 0), [(-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5)]),
        ((0, -1, 0), [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)]),
        ((0, 0, 1), [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
        ((0, 0, -1), [(-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5)]),
    ]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    positions, normals, uv_out, indices = [], [], [], []
    vidx = 0
    for normal, corners in faces:
        for i, c in enumerate(corners):
            positions.append((c[0] * sx + cx, c[1] * sy + cy, c[2] * sz + cz))
            normals.append(normal)
            uv_out.append(uvs[i])
        indices.extend([vidx, vidx + 1, vidx + 2, vidx, vidx + 2, vidx + 3])
        vidx += 4
    return (np.array(positions, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(uv_out, dtype=np.float32),
            np.array(indices, dtype=np.uint32))


def _textured_box_geometry(texture_atlas, block_id, box=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)):
    """
    Builds ONE box''s worth of vertex data using world/chunk.py''s real FACES
    table (correct per-face UV winding) and the block''s ACTUAL per-face
    world textures (top/bottom/side each looked up separately, falling back
    to "all" for blocks that only have one texture) - this is what makes a
    held block look pixel-identical to the same block placed in the world,
    instead of one texture smeared across every face.
    box: (min_x, min_y, min_z, max_x, max_y, max_z) in LOCAL 0..1 space,
    letting the same function build both a full unit cube (ordinary blocks)
    and smaller sub-boxes (door slabs, fence posts/rails, stair treads).
    """
    bx0, by0, bz0, bx1, by1, bz1 = box
    positions, normals, uvs, indices = [], [], [], []
    for face in CHUNK_FACES:
        uv_rect = texture_atlas.uv_for(block_id, face["tex"])
        if uv_rect is None:
            continue
        u0, v0, u1, v1 = uv_rect
        base_idx = len(positions)
        for (cx, cy, cz), (uu, vv) in zip(face["corners"], face["uvs"]):
            px = bx1 if cx else bx0
            py = by1 if cy else by0
            pz = bz1 if cz else bz0
            positions.append((px, py, pz))
            normals.append(face["dir"])
            uvs.append((u0 + uu * (u1 - u0), v0 + vv * (v1 - v0)))
        indices.extend([base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3])
    return positions, normals, uvs, indices


def _multi_box_geometry(texture_atlas, block_id, boxes, pivot_y=0.5):
    """Same as _textured_box_geometry but concatenates several boxes into
    one mesh (index-offset aware), and re-centers the whole result so that
    local (0,0,0) is the item''s GRIP point: X/Z centered at 0.5 (the
    block''s horizontal middle) and Y centered at `pivot_y` (0.5 for a
    normal 1-block item, 1.0 for a 2-block-tall door''s half-way seam) -
    this is what lets _item_model_matrix place the item by a single grip
    anchor shared with the hand, instead of the mesh''s corner."""
    all_pos, all_norm, all_uv, all_idx = [], [], [], []
    for box in boxes:
        p, n, u, i = _textured_box_geometry(texture_atlas, block_id, box)
        offset = len(all_pos)
        all_pos.extend(p)
        all_norm.extend(n)
        all_uv.extend(u)
        all_idx.extend([idx + offset for idx in i])
    positions = np.array(all_pos, dtype=np.float32)
    if len(positions) > 0:
        positions[:, 0] -= 0.5
        positions[:, 1] -= pivot_y
        positions[:, 2] -= 0.5
    return (positions,
            np.array(all_norm, dtype=np.float32),
            np.array(all_uv, dtype=np.float32),
            np.array(all_idx, dtype=np.uint32))


def _cross_sprite_mesh():
    """A single flat double-sided quad for the item icon, fixed at a
    constant angle relative to the camera. Only ONE plane is ever rendered,
    so there is no second plane to visually cross/double with."""
    positions = [
        (-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0),
    ]
    normals = [(0, 0, 1)] * 4
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    indices = [0, 1, 2, 0, 2, 3]
    return (np.array(positions, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(uvs, dtype=np.float32),
            np.array(indices, dtype=np.uint32))


class _Mesh:
    """Minimal static GPU mesh wrapper: position/normal/uv VBOs + index buffer."""

    def __init__(self, positions, normals, uvs, indices):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        self.vbo_norm = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_norm)
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
        if self.index_count == 0:
            return
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def destroy(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo_pos])
        glDeleteBuffers(1, [self.vbo_norm])
        glDeleteBuffers(1, [self.vbo_uv])
        glDeleteBuffers(1, [self.ebo])


def _upload_pil_texture(pil_image):
    """Uploads a small PIL RGBA image (an item icon) as a standalone OpenGL
    texture at its NATIVE resolution - so high-resolution hand-drawn art
    (32x32, 64x64, ...) stays crisp when held instead of being downscaled."""
    data = np.array(pil_image.convert("RGBA"))
    data = np.flipud(data)
    data = np.ascontiguousarray(data)

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, data.shape[1], data.shape[0], 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data)
    return tex_id


# ---------------------------------------------------------------------------
# Custom-shaped block geometry (door/fence/stairs), reusing the SAME box
# lists the world renderer/physics use, so "in hand" always matches
# "placed in the world".
# ---------------------------------------------------------------------------

def _held_boxes_for_block(block_id):
    """Returns a list of (min_x,min_y,min_z,max_x,max_y,max_z) boxes
    representing this block in a neutral/default state, for held-item
    display. Ordinary full-cube blocks get a single full-unit box. Doors
    keep their real 2-block-tall proportions here (both halves included) -
    the overall held-item SCALE is what makes it read as a small handheld
    object, not by artificially squashing the door''s own shape."""
    if block_id == Block.DOOR:
        from world.doors import door_collision_bounds, FACING_SOUTH
        lmin_x, lmin_z, lmax_x, lmax_z = door_collision_bounds(FACING_SOUTH, is_open=False)
        return [
            (lmin_x, 0.0, lmin_z, lmax_x, 1.0, lmax_z),
            (lmin_x, 1.0, lmin_z, lmax_x, 2.0, lmax_z),
        ]
    if block_id == Block.FENCE:
        _POST_MIN, _POST_MAX = 0.375, 0.625
        _RAIL_Y0, _RAIL_Y1 = 0.375, 0.875
        return [
            (_POST_MIN, 0.0, _POST_MIN, _POST_MAX, 1.0, _POST_MAX),
            (_POST_MIN, _RAIL_Y0, _POST_MAX, _POST_MAX, _RAIL_Y1, 1.0),
            (_POST_MAX, _RAIL_Y0, _POST_MIN, 1.0, _RAIL_Y1, _POST_MAX),
        ]
    if block_id in (Block.STAIRS_WOOD, Block.STAIRS_STONE):
        from world.stairs import collision_boxes, FACING_NORTH, SHAPE_STRAIGHT
        # FACING_NORTH puts the riser (solid tall face) away from the
        # camera and the open step/cut edge toward it, so the held stair
        # visibly shows its stepped profile instead of a flat solid face.
        return collision_boxes(FACING_NORTH, is_top=False, shape=SHAPE_STRAIGHT)
    return [(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)]


def _held_item_pivot_y(block_id):
    """Y (in the box''s own 0..1-per-block local space) that should sit at
    the item anchor point - center for a normal 1-block item, but the
    VERTICAL CENTER of a 2-block-tall door (i.e. 1.0, the seam between its
    two halves) so a door held in the hand is centered on the grip instead
    of hanging mostly below it."""
    if block_id == Block.DOOR:
        return 1.0
    return 0.5


# ---------------------------------------------------------------------------
# Animation state: bob (walking sway) + swing (attack/place arc)
# ---------------------------------------------------------------------------

SWING_DURATION = 0.28
BOB_CYCLE_LENGTH = 0.72
BOB_AMOUNT = 0.045


class AnimationState:
    def __init__(self):
        self.swing_timer = SWING_DURATION
        self.bob_phase = 0.0

    def trigger_swing(self):
        self.swing_timer = 0.0

    def update(self, dt: float, horizontal_speed: float, on_ground: bool, flying: bool):
        if self.swing_timer < SWING_DURATION:
            self.swing_timer = min(SWING_DURATION, self.swing_timer + dt)

        is_walking = on_ground and not flying and horizontal_speed > 0.5
        if is_walking:
            cycles_per_second = 1.0 / BOB_CYCLE_LENGTH
            speed_scale = min(1.6, horizontal_speed / 4.3)
            self.bob_phase += dt * cycles_per_second * speed_scale
            self.bob_phase %= 1.0
        else:
            target = 0.0
            diff = (target - self.bob_phase + 0.5) % 1.0 - 0.5
            ease = min(1.0, dt * 6.0)
            self.bob_phase = (self.bob_phase + diff * ease) % 1.0

    def swing_progress(self) -> float:
        return self.swing_timer / SWING_DURATION

    def is_swinging(self) -> bool:
        return self.swing_timer < SWING_DURATION

    def bob_offset(self):
        angle = self.bob_phase * 2.0 * math.pi
        bob_x = math.sin(angle) * BOB_AMOUNT
        bob_y = (1.0 - math.cos(angle * 2.0)) * -0.5 * BOB_AMOUNT * 0.35
        return bob_x, bob_y

    def swing_offset_and_rotation(self):
        if not self.is_swinging():
            return 0.0, 0.0, 0.0, 0.0
        t = self.swing_progress()
        arc = math.sin(t * math.pi)
        x_offset = -arc * 0.12
        y_offset = -arc * 0.09
        z_offset = arc * 0.22
        extra_pitch = arc * 0.9
        return x_offset, y_offset, z_offset, extra_pitch


class HandRenderer:
    def __init__(self, texture_atlas):
        self.shader = Shader(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, name="hand")
        self.texture_atlas = texture_atlas
        self.animation = AnimationState()

        # Hand: a simple skin-toned box with roughly the classic viewmodel
        # arm proportions (a bit taller than wide, not the exaggerated
        # elongated pole from an earlier revision), pivoted at its top/
        # wrist end.
        hand_pos, hand_norm, hand_uv, hand_idx = _box_mesh_simple(
            scale=(0.32, 0.85, 0.32), center=(0.0, -0.425, 0.0)
        )
        self.hand_mesh = _Mesh(hand_pos, hand_norm, hand_uv, hand_idx)

        # Held cross-sprite item (tools, sticks, coal, etc).
        quad_pos, quad_norm, quad_uv, quad_idx = _cross_sprite_mesh()
        self.item_quad_mesh = _Mesh(quad_pos, quad_norm, quad_uv, quad_idx)

        self._item_icon_textures = {}   # item_id -> GL texture id
        self._block_meshes = {}         # block_id -> _Mesh, built lazily from real world textures

    def trigger_swing(self):
        self.animation.trigger_swing()

    def update(self, dt: float, player):
        physics = player.physics
        horizontal_speed = math.hypot(physics.vx, physics.vz)
        self.animation.update(dt, horizontal_speed, physics.on_ground, physics.flying)

    def _get_item_icon_texture(self, item_id):
        if item_id not in self._item_icon_textures:
            from ui.icon_cache import get_item_icon_pil
            pil_img = get_item_icon_pil(self.texture_atlas, item_id)
            self._item_icon_textures[item_id] = _upload_pil_texture(pil_img)
        return self._item_icon_textures[item_id]

    def _get_block_mesh(self, block_id):
        """Builds (once, cached) a real textured 3D mesh for this block id,
        using its actual per-face world textures and, for custom-shaped
        blocks (door/fence/stairs), their real box geometry - not a
        one-texture-on-everything cube and not a flat icon."""
        if block_id not in self._block_meshes:
            boxes = _held_boxes_for_block(block_id)
            pivot_y = _held_item_pivot_y(block_id)
            positions, normals, uvs, indices = _multi_box_geometry(self.texture_atlas, block_id, boxes, pivot_y)
            self._block_meshes[block_id] = _Mesh(positions, normals, uvs, indices)
        return self._block_meshes[block_id]

    def render(self, aspect_ratio: float, held_item_id):
        glClear(GL_DEPTH_BUFFER_BIT)
        glDisable(GL_CULL_FACE)
        self.shader.use()

        proj = _hand_projection(aspect_ratio)
        self.shader.set_mat4("u_projection", proj)
        self.shader.set_int("u_tex", 0)

        item_def = get_item_def(held_item_id) if held_item_id is not None else None

        self._draw_hand()

        if item_def is not None and item_def.is_block:
            model = _item_model_matrix(self.animation)
            self.shader.set_mat4("u_model", model)
            mesh = self._get_block_mesh(held_item_id)
            self.shader.set_int("u_use_texture", 1)
            glActiveTexture(GL_TEXTURE0)
            self.texture_atlas.bind(0)
            mesh.draw()

        elif item_def is not None:
            model = _item_model_matrix(self.animation, is_flat=True)
            self.shader.set_mat4("u_model", model)
            tex_id = self._get_item_icon_texture(held_item_id)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            self.shader.set_int("u_use_texture", 1)
            self.item_quad_mesh.draw()

        glEnable(GL_CULL_FACE)

    def _draw_hand(self):
        model = _hand_model_matrix(self.animation)
        self.shader.set_mat4("u_model", model)
        self.shader.set_int("u_use_texture", 0)
        self.shader.set_vec3("u_solid_color", 0.85, 0.65, 0.5)
        self.hand_mesh.draw()


def _hand_projection(aspect):
    fov = 45.0
    near, far = 0.01, 10.0
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m.T.astype(np.float32)


def _rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float32)


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)


def _wrist_transform(animation: "AnimationState", x_offset, y_offset, z_offset):
    bob_x, bob_y = animation.bob_offset()
    swing_x, swing_y, swing_z, swing_pitch = animation.swing_offset_and_rotation()

    angle_x = -0.25 - swing_pitch
    angle_z = -0.15

    translate = np.identity(4, dtype=np.float32)
    translate[0, 3] = x_offset + bob_x + swing_x
    translate[1, 3] = y_offset + bob_y + swing_y
    translate[2, 3] = z_offset + swing_z

    rot = _rot_z(angle_z) @ _rot_x(angle_x)
    return translate, rot


def _hand_model_matrix(animation: "AnimationState"):
    """Positions the hand so its bottom edge extends below the visible
    frame (like vanilla Minecraft, where the arm appears to emerge from
    off-screen rather than floating with a visible cut-off base), anchored
    toward the RIGHT side of screen with a slight diagonal tilt. y_offset
    has extra downward margin beyond the resting position specifically so
    the swing animation's forward/up thrust (which pushes the whole arm
    toward the camera) never pulls the bottom cut-off edge into view."""
    translate, rot = _wrist_transform(animation, x_offset=0.50, y_offset=-0.20, z_offset=-2.0)
    scale = 0.75
    s = np.diag([scale, scale, scale, 1.0]).astype(np.float32)
    model = translate @ rot @ s
    return model.T.astype(np.float32)


# The hand mesh is a box scaled (0.32, 0.85, 0.32) and pivoted at
# center=(0, -0.425, 0) (see HandRenderer.__init__), so LOCALLY its palm/
# grip end (where an item would rest ON TOP of the fist) sits at
# y=-0.85+0.16=-0.69 roughly the top face of where fingers would close
# around an item - not the very bottom tip of the arm. The item anchor
# below reuses the EXACT SAME translate/rot/scale as the hand
# (_hand_model_matrix), offset up-and-forward (in the hand''s own rotated
# local space) to that grip point, so the item always sits gripped ON the
# fist instead of hanging off its side.
_HAND_GRIP_LOCAL = np.array([0.0, -0.12, 0.24, 1.0], dtype=np.float32)
_ITEM_GRIP_SCALE = 0.75       # matches the hand's own scale, so item size is consistent with the hand
_ITEM_SIZE_MULTIPLIER = 0.55  # the item itself renders smaller than the hand's bounding box


def _item_model_matrix(animation: "AnimationState", is_flat: bool = False):
    """Positions a held item gripped on top of the hand: same base
    transform as the hand, translated (in the hand''s own rotated local
    space) to its grip point, then scaled down for the item''s own visible
    size."""
    translate, rot = _wrist_transform(animation, x_offset=0.50, y_offset=-0.20, z_offset=-2.0)
    grip_local = _HAND_GRIP_LOCAL.copy()
    grip_local[:3] *= _ITEM_GRIP_SCALE
    grip_offset = rot @ grip_local
    grip_translate = np.identity(4, dtype=np.float32)
    grip_translate[0, 3] = grip_offset[0]
    grip_translate[1, 3] = grip_offset[1]
    grip_translate[2, 3] = grip_offset[2]

    item_scale = _ITEM_GRIP_SCALE * _ITEM_SIZE_MULTIPLIER
    s = np.diag([item_scale, item_scale, item_scale, 1.0]).astype(np.float32)
    model = translate @ grip_translate @ rot @ s
    return model.T.astype(np.float32)
