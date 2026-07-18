"""
render/player_renderer.py
Draws the other players: six textured boxes each, wearing a procedurally
generated 64x32 skin.

The model itself - the boxes, their pivots, the skin's UV layout, and every
angle they are held at - lives in entity/player_entity.py. This file owns
buffers, a texture, and a shader, and nothing else. The split is the same one
world/chunk.py and render/chunk_renderer.py already draw: what a thing IS is
answerable without a GL context, and stays testable because of it.

THE SKIN IS GENERATED, NOT SHIPPED
----------------------------------
Same rule as the block atlas (render/texture_atlas.py): every pixel here comes
out of numpy, none of it out of Mojang's assets. It is also why the skin can be
built before there is a window and uploaded after - build_skin() is pure PIL.

Unlike the block atlas, though, there is deliberately no assets/ override path
for this one. The atlas has one because hand-drawn block art is a thing someone
might reasonably add; a skin loaded from disk would need a whole "who is wearing
what, and how does it get to the other player" conversation that this LAN build
has no answer to. Everyone is Steve, and everyone agrees on it without sending
a byte.

ONE TEXTURE, SIX DRAW CALLS PER PLAYER
--------------------------------------
All six parts live in one VBO; each is drawn from its own slice of the shared
index buffer with its own model matrix. Six draw calls per player sounds
wasteful next to the chunk renderer's one-per-chunk, but the parts genuinely
have different transforms and there are at most a handful of players on a LAN
game - instancing this would be complexity bought with nothing.
"""

import ctypes
import math

import numpy as np
from PIL import Image
from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
    glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawElements, glDeleteVertexArrays, glDeleteBuffers,
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, glDeleteTextures,
    glActiveTexture,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT,
    GL_FALSE, GL_TRIANGLES, GL_UNSIGNED_INT, GL_TEXTURE0, GL_TEXTURE_2D,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T, GL_NEAREST, GL_CLAMP_TO_EDGE, GL_RGBA, GL_UNSIGNED_BYTE,
)

import config
from core.shader import Shader
from entity.player_entity import (
    PARTS, PIXEL, SKIN_WIDTH, SKIN_HEIGHT, PlayerEntity, face_rects, face_st,
)

# --- skin palette -----------------------------------------------------------
# Steve, near enough, mixed by eye rather than eyedropped off the original.
SKIN_TONE = (198, 141, 100)
SKIN_SHADOW = (166, 114, 79)
HAIR = (60, 40, 26)
SHIRT = (0, 152, 152)
TROUSERS = (60, 60, 152)
SHOE = (78, 78, 78)
EYE_WHITE = (232, 232, 232)
EYE_IRIS = (58, 88, 172)
MOUTH = (118, 68, 52)

# How many rows of a limb are sleeve/trouser rather than bare skin.
SLEEVE_ROWS = 4
SHOE_ROWS = 3


def _rng(seed_tuple):
    """Deterministic per-region RNG, same idea as texture_atlas._rng: the skin
    must come out identical on every machine, because two players looking at
    each other and seeing different faces is a bug nobody would ever think to
    look for."""
    seed = abs(hash(seed_tuple)) % (2 ** 32)
    return np.random.RandomState(seed)


def _fill(arr, rect, color, rng, variance=12, rows=None):
    """
    Paints one face rect (x, y, w, h in skin pixels) with a colour plus a little
    per-pixel noise, optionally only a row band of it (`rows` is a half-open
    (first, last) in rect-local rows, which is how a sleeve is "the top 4 rows
    of every side face").

    The noise is one value per pixel applied to all three channels rather than
    three independent ones: that shifts brightness while leaving hue alone, so
    the shirt reads as cloth rather than as a shirt with confetti on it.
    """
    x0, y0, w, h = rect
    if rows is not None:
        first, last = rows
        y0 += first
        h = last - first
    if w <= 0 or h <= 0:
        return
    noise = rng.randint(-variance // 2, variance // 2 + 1, size=(h, w, 1))
    base = np.array(color, dtype=np.int16).reshape(1, 1, 3)
    arr[y0:y0 + h, x0:x0 + w, :3] = np.clip(base + noise, 0, 255).astype(np.uint8)
    arr[y0:y0 + h, x0:x0 + w, 3] = 255


def _paint_face(arr, rect):
    """
    Eyes, nose and mouth onto the head's front rect.

    No noise here, and no rng: these are single pixels, and a single pixel with
    noise on it is just a pixel of the wrong colour. Coordinates are rect-local,
    so this does not care where in the skin the head happens to be unwrapped -
    it reads its rect from face_rects() like the mesh builder does. Rows: 0-2
    are under the hair fringe, 3 is the brow, 4 the eyes, 5 the nose, 6 the
    mouth.
    """
    x0, y0, _w, _h = rect

    def put(at_col, at_row, color):
        arr[y0 + at_row, x0 + at_col, :3] = color
        arr[y0 + at_row, x0 + at_col, 3] = 255

    for col in range(1, 7):
        put(col, 3, SKIN_SHADOW)  # brow line, the fringe's shadow on the face

    # Two 2px eyes, whites outward. The face rect is painted as the viewer sees
    # it, so "outward" is fx 1 and fx 6 - the head's own left and right.
    put(1, 4, EYE_WHITE)
    put(2, 4, EYE_IRIS)
    put(5, 4, EYE_IRIS)
    put(6, 4, EYE_WHITE)

    put(3, 5, SKIN_SHADOW)
    put(4, 5, SKIN_SHADOW)

    for col in range(2, 6):
        put(col, 6, MOUTH)


def build_skin() -> Image.Image:
    """
    Generates the 64x32 skin.

    Every rect comes from entity.player_entity.face_rects - the same call the
    mesh builder makes. That is not tidiness, it is the only way the two can be
    trusted to agree: a skin painted from one set of coordinates and sampled
    with another produces a face on the back of someone's head, and both halves
    look perfectly correct in isolation.
    """
    rng = _rng("player_skin")
    arr = np.zeros((SKIN_HEIGHT, SKIN_WIDTH, 4), dtype=np.uint8)

    head = face_rects((0, 0), (8, 8, 8))
    body = face_rects((16, 16), (8, 12, 4))
    arm = face_rects((40, 16), (4, 12, 4))
    leg = face_rects((0, 16), (4, 12, 4))

    # Head: skin all over, then hair over the top, the back, and the top three
    # rows of everything else - which is what a bowl cut is.
    for rect in head.values():
        _fill(arr, rect, SKIN_TONE, rng)
    _fill(arr, head["+y"], HAIR, rng)
    _fill(arr, head["+z"], HAIR, rng)
    for key in ("+x", "-x", "-z"):
        _fill(arr, head[key], HAIR, rng, rows=(0, 3))
    _paint_face(arr, head["-z"])

    for rect in body.values():
        _fill(arr, rect, SHIRT, rng)

    # Arms: sleeve down to SLEEVE_ROWS, bare skin below it, and the cap at the
    # bottom is the palm - the only face of the arm that is never sleeve.
    _fill(arr, arm["+y"], SHIRT, rng)
    _fill(arr, arm["-y"], SKIN_TONE, rng)
    for key in ("+x", "-x", "+z", "-z"):
        _fill(arr, arm[key], SHIRT, rng, rows=(0, SLEEVE_ROWS))
        _fill(arr, arm[key], SKIN_TONE, rng, rows=(SLEEVE_ROWS, 12))

    # Legs: trousers, and the bottom rows plus the sole are the shoe.
    _fill(arr, leg["+y"], TROUSERS, rng)
    _fill(arr, leg["-y"], SHOE, rng)
    for key in ("+x", "-x", "+z", "-z"):
        _fill(arr, leg[key], TROUSERS, rng, rows=(0, 12 - SHOE_ROWS))
        _fill(arr, leg[key], SHOE, rng, rows=(12 - SHOE_ROWS, 12))

    return Image.fromarray(arr, "RGBA")


def _upload_skin(image: Image.Image) -> int:
    data = np.array(image.convert("RGBA"))
    # Same flip as TextureAtlas.upload_to_gpu, and the UVs below invert v to
    # match: PIL is top-down, GL's texture origin is bottom-left.
    data = np.ascontiguousarray(np.flipud(data))

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    # NEAREST both ways and NO mipmaps. Not a style choice on the min filter:
    # the atlas can afford mipmaps because every tile is border-extruded into a
    # padded slot, whereas a skin's rects sit flush against each other by
    # definition of the format - a mip level would average the back of the head
    # into the face. A 64x32 texture on a player-sized quad is never minified
    # far enough for it to matter anyway.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # CLAMP over REPEAT: the outer edges of a skin are real content, and a
    # sampler that wraps turns a rounding error at the last texel column into
    # a stripe of whatever is on the far side of the texture.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, data.shape[1], data.shape[0], 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data)
    return tex_id


# --- geometry ---------------------------------------------------------------
# Unit cube, corners CCW seen from outside. Same table as world/chunk.py's
# FACES and render/door_renderer.py's - the winding there was verified
# analytically once and there is no reason to re-derive it a third time.
_CUBE_FACES = (
    {"key": "+x", "dir": (1, 0, 0), "corners": ((1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1))},
    {"key": "-x", "dir": (-1, 0, 0), "corners": ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))},
    {"key": "+y", "dir": (0, 1, 0), "corners": ((0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0))},
    {"key": "-y", "dir": (0, -1, 0), "corners": ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))},
    {"key": "+z", "dir": (0, 0, 1), "corners": ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))},
    {"key": "-z", "dir": (0, 0, -1), "corners": ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))},
)


def _build_mesh():
    """
    Builds all six parts into one set of arrays, plus the index slice each part
    occupies. Positions are in BLOCKS, relative to the part's own pivot, so the
    per-frame model matrix is a translate-to-pivot and a rotate and nothing else.
    """
    positions, normals, uvs, indices = [], [], [], []
    part_ranges = {}

    for part in PARTS:
        rects = face_rects(part.uv_origin, part.size, part.mirror)
        ox, oy, oz = part.offset
        width, height, depth = part.size
        first_index = len(indices)

        for face in _CUBE_FACES:
            rect_x, rect_y, rect_w, rect_h = rects[face["key"]]
            base = len(positions)
            for (fx, fy, fz) in face["corners"]:
                positions.append((
                    (ox + fx * width) * PIXEL,
                    (oy + fy * height) * PIXEL,
                    (oz + fz * depth) * PIXEL,
                ))
                normals.append(face["dir"])

                s, t = face_st(face["key"], fx, fy, fz)
                if part.mirror:
                    # Reflecting the box through the YZ plane swaps its two side
                    # faces (done in face_rects) and flips every face's U. That
                    # pair is what lets a 64x32 skin paint one arm and get the
                    # other one free.
                    s = 1.0 - s
                # No half-texel inset. Fragment centres are strictly inside the
                # primitive, so an interpolated u never actually reaches the
                # rect's far edge and NEAREST never picks up the neighbouring
                # rect's first texel. Vanilla relies on exactly this.
                uvs.append((
                    (rect_x + s * rect_w) / SKIN_WIDTH,
                    1.0 - (rect_y + t * rect_h) / SKIN_HEIGHT,
                ))
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

        part_ranges[part.name] = (first_index, len(indices) - first_index)

    return (np.array(positions, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(uvs, dtype=np.float32),
            np.array(indices, dtype=np.uint32),
            part_ranges)


# --- matrices ---------------------------------------------------------------
# Built row-major ("maths layout") and transposed on upload, exactly as
# core/camera.py does. Keeping the same convention as the camera is the only
# reason the two can be multiplied together in the shader without anyone having
# to remember which one is which.

def _translate(x, y, z):
    m = np.identity(4, dtype=np.float32)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def _rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    m = np.identity(4, dtype=np.float32)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def _rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    m = np.identity(4, dtype=np.float32)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    m = np.identity(4, dtype=np.float32)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

out vec2 v_uv;
out float v_fog_dist;
out float v_shade;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    vec4 view_pos = u_view * world_pos;
    gl_Position = u_projection * view_pos;
    v_uv = in_uv;
    v_fog_dist = -view_pos.z;

    // The block shader's fixed-per-face brightness (top 1.0, bottom 0.5, x
    // 0.6, z 0.8), but computed from the ROTATED normal - a swinging arm's
    // faces point somewhere new every frame, and a player lit by the model-
    // space normal would keep their lighting bolted to their body while they
    // turned, which reads as a cardboard cutout the moment they walk past.
    //
    // u_model carries rotation and translation only (never scale), so mat3 of
    // it is already orthonormal and the normalize is belt and braces.
    vec3 n = normalize(mat3(u_model) * in_normal);
    float is_top = step(0.0, n.y);
    float y_component = mix(0.5, 1.0, is_top);
    v_shade = abs(n.y) * y_component + abs(n.x) * 0.6 + abs(n.z) * 0.8;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 v_uv;
in float v_fog_dist;
in float v_shade;

uniform sampler2D u_skin;
uniform vec3 u_fog_color;
uniform float u_fog_start;
uniform float u_fog_end;

out vec4 frag_color;

void main() {
    vec4 tex_color = texture(u_skin, v_uv);
    if (tex_color.a < 0.1) discard;
    vec3 shaded = tex_color.rgb * v_shade;
    float fog_factor = clamp((v_fog_dist - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);
    frag_color = vec4(mix(shaded, u_fog_color, fog_factor), 1.0);
}
"""


class PlayerRenderer:
    """
    Draws every remote player. Owns one PlayerEntity per player, created and
    dropped to follow whoever the client currently knows about.

    Why the entities live here rather than in GameClient: they are animation,
    and animation is a frame-rate thing. GameClient is drained once per frame
    today, but nothing in its contract says it has to be - it is fed by a socket
    and could reasonably be polled twice, or not at all during a pause. Limb
    phase advanced from inside a packet handler would speed up or stop with the
    network. Here it advances with the renderer, which is the thing it is for.
    """

    def __init__(self):
        self.shader = Shader(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, name="player")
        self.skin_image = build_skin()
        self.skin_texture_id = _upload_skin(self.skin_image)

        positions, normals, uvs, indices, self.part_ranges = _build_mesh()

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

        glBindVertexArray(0)

        self.entities: dict[int, PlayerEntity] = {}

    def update(self, remote_players: dict, dt: float):
        """Syncs one PlayerEntity per known player. Call once a frame, before
        render()."""
        for entity_id, remote in remote_players.items():
            entity = self.entities.get(entity_id)
            if entity is None:
                entity = PlayerEntity(entity_id, remote.username)
                self.entities[entity_id] = entity
            entity.sync(remote, dt)

        # Whoever left. Dropping the entity drops their limb phase with it, so a
        # player who disconnects and rejoins does not resume mid-stride.
        for entity_id in [k for k in self.entities if k not in remote_players]:
            del self.entities[entity_id]

    def render(self, camera, render_distance_chunks=None,
               fog_color=None, fog_start=None, fog_end=None):
        """
        The fog arguments mirror WorldRenderer.render's exactly, defaults
        included, and main.py must pass it the same values it passes there -
        including the underwater overrides. A player who does not fog with the
        terrain they are standing on is visible as a sharp silhouette through
        haze that has already swallowed the ground under their feet.
        """
        if not self.entities:
            return

        self.shader.use()
        self.shader.set_mat4("u_view", camera.view_matrix())
        self.shader.set_mat4("u_projection", camera.projection_matrix())
        self.shader.set_int("u_skin", 0)

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

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.skin_texture_id)
        glBindVertexArray(self.vao)

        for entity in self.entities.values():
            pose = entity.pose()
            # Where the player is, and which way they are facing. Every part
            # hangs off this.
            entity_matrix = _translate(entity.x, entity.y, entity.z) @ _rot_y(entity.yaw)

            for part in PARTS:
                part_pose = pose[part.name]
                pivot_x = (part.pivot[0] + part_pose.pivot_dx) * PIXEL
                pivot_y = (part.pivot[1] + part_pose.pivot_dy) * PIXEL
                pivot_z = (part.pivot[2] + part_pose.pivot_dz) * PIXEL

                # Z then Y then X, matching vanilla's ModelRenderer call order.
                # Rotation order is not a detail: the arm swing writes X while
                # the idle sway writes Z, and swapping them makes a punching
                # player's elbow orbit their shoulder.
                model = (entity_matrix
                         @ _translate(pivot_x, pivot_y, pivot_z)
                         @ _rot_z(part_pose.angle_z)
                         @ _rot_y(part_pose.angle_y)
                         @ _rot_x(part_pose.angle_x))
                self.shader.set_mat4("u_model", model.T.astype(np.float32))

                first_index, index_count = self.part_ranges[part.name]
                glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT,
                               ctypes.c_void_p(first_index * 4))

        glBindVertexArray(0)

    def destroy(self):
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = None
        for vbo in (self.vbo_position, self.vbo_normal, self.vbo_uv, self.ebo):
            if vbo is not None:
                glDeleteBuffers(1, [vbo])
        self.vbo_position = self.vbo_normal = self.vbo_uv = self.ebo = None
        if self.skin_texture_id is not None:
            glDeleteTextures(1, [self.skin_texture_id])
            self.skin_texture_id = None
        self.entities.clear()
