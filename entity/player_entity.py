"""
entity/player_entity.py
The Minecraft player model: six boxes, and the angles they are held at.

No OpenGL in this file, on purpose. Everything here is geometry and
trigonometry against numbers that arrive over the wire, so it can be reasoned
about - and tested - without a window. render/player_renderer.py turns what
this module says into triangles.

UNITS
-----
The model is quoted in Minecraft's own units: 1 block = 16 pixels, feet at
y = 0, top of the head at y = 32. Every dimension below is therefore the same
integer you would read off a skin file or a vanilla model dump, which is the
entire reason for not converting them to blocks here: the UV layout of a 64x32
skin is defined in those pixels, and a model whose numbers no longer match the
texture's numbers is a model nobody can check.

The one place the two systems meet is PIXEL, and it is deliberately derived
from config.PLAYER_HEIGHT rather than hardcoded at 1/16. Vanilla's model is 32
px = 2.0 blocks tall against a 1.8-block hitbox and papers over the gap with a
0.9375 render scale; doing the same here would mean a head that pokes through
any ceiling the player can legally stand under. Dividing our own hitbox height
by the model height instead makes the model exactly fill the box it collides
with, and keeps doing so if PLAYER_HEIGHT is ever retuned.

ORIENTATION
-----------
Model space matches the project's camera convention (see core/camera.py), NOT
vanilla's internal one:

    -Z is forward (the face)      +Y is up      +X is the player's RIGHT

That is why the right arm sits at positive X here while vanilla's sits at
negative X, and why several rotation signs below are flipped relative to the
values you would find in ModelBiped. Vanilla's model space has Y pointing DOWN
and the limbs extending along +Y; ours has them hanging along -Y. Porting the
angles without porting the sign is the classic way to end up with a player who
moonwalks with their knees on backwards.
"""

import math

import config

# --- the skin ---------------------------------------------------------------
# 64x32 is the classic layout: no second (hat/jacket) layer past x=32 on the
# body rows, and the left arm/leg have no art of their own - they are drawn as
# mirror images of the right ones. See face_rects().
SKIN_WIDTH = 64
SKIN_HEIGHT = 32

# --- the model --------------------------------------------------------------
MODEL_HEIGHT_PIXELS = 32
PIXEL = config.PLAYER_HEIGHT / MODEL_HEIGHT_PIXELS  # blocks per model pixel


class BodyPart:
    """
    One box.

    `pivot` is where it rotates about, in model pixels. `offset` is the box's
    minimum corner RELATIVE TO THE PIVOT - so a leg, which swings from the hip,
    has offset y = -12 (it hangs below its pivot) while the head, which swings
    from the neck, has offset y = 0 (it sits above its pivot). Rotating a part
    is therefore always just "rotate about the origin, after translating to the
    pivot", with no per-part special cases.

    `uv_origin` is the top-left corner of this box's unwrap in the skin, in skin
    pixels. `mirror` marks the parts that have no art of their own (see above).
    """

    __slots__ = ("name", "size", "pivot", "offset", "uv_origin", "mirror")

    def __init__(self, name, size, pivot, offset, uv_origin, mirror=False):
        self.name = name
        self.size = size            # (width, height, depth) in px
        self.pivot = pivot          # (x, y, z) in px, feet-up model space
        self.offset = offset        # box min corner relative to pivot, in px
        self.uv_origin = uv_origin  # (u, v) in skin px
        self.mirror = mirror


# The numbers are vanilla's, rebased into our orientation. Read them as:
# head 24..32, body and arms 12..24, legs 0..12 - so the arms end exactly where
# the torso does, the legs carry the bottom half, and nothing overlaps.
HEAD = BodyPart("head", (8, 8, 8), (0, 24, 0), (-4, 0, -4), (0, 0))
BODY = BodyPart("body", (8, 12, 4), (0, 24, 0), (-4, -12, -2), (16, 16))
ARM_RIGHT = BodyPart("arm_right", (4, 12, 4), (5, 22, 0), (-1, -10, -2), (40, 16))
ARM_LEFT = BodyPart("arm_left", (4, 12, 4), (-5, 22, 0), (-3, -10, -2), (40, 16), mirror=True)
LEG_RIGHT = BodyPart("leg_right", (4, 12, 4), (2, 12, 0), (-2, -12, -2), (0, 16))
LEG_LEFT = BodyPart("leg_left", (4, 12, 4), (-2, 12, 0), (-2, -12, -2), (0, 16), mirror=True)

PARTS = (HEAD, BODY, ARM_RIGHT, ARM_LEFT, LEG_RIGHT, LEG_LEFT)

# The shoulders sit 2 px below the top of the torso, so an arm's box spans
# 12..24 like the body while its pivot is at 22. Kept as a named constant only
# because the arm pivots above encode it and a reader is entitled to know it was
# a decision rather than a typo.
SHOULDER_INSET = 2


def face_rects(uv_origin, size, mirror=False):
    """
    The six texture rectangles for one box, in skin pixels, as (x, y, w, h).

    This is the standard box unwrap - a cross laid out flat - and every 64x32
    skin in existence is drawn against it:

        [   ][top ][bot ][   ]
        [rgt ][frnt][left][back]

    Both this module's mesh builder and the skin generator read the table from
    here rather than each hardcoding coordinates. That is the point: a skin
    painted from one set of numbers and sampled with another is a bug that shows
    up as a face on the back of someone's head, and it is invisible in code
    review because both sets of numbers look right on their own.

    Mirroring swaps the two side rects and (in the mesh builder) flips U within
    every face - which is exactly what reflecting the box through the YZ plane
    does to its texture, and why classic skins can get away with painting only
    one arm.
    """
    u, v = uv_origin
    w, h, d = size
    rects = {
        "+y": (u + d, v, w, d),                  # top
        "-y": (u + d + w, v, w, d),              # bottom
        "+x": (u, v + d, d, h),                  # player's right side
        "-z": (u + d, v + d, w, h),              # front - the face
        "-x": (u + d + w, v + d, d, h),          # player's left side
        "+z": (u + d + w + d, v + d, w, h),      # back
    }
    if mirror:
        rects["+x"], rects["-x"] = rects["-x"], rects["+x"]
    return rects


def face_st(face_key, fx, fy, fz):
    """
    Where a box corner lands inside its face's texture rect.

    (fx, fy, fz) are the corner's 0..1 coordinates within the box; the result is
    (s, t) in 0..1 within the rect, s running left-to-right and t running
    top-to-bottom in SKIN PIXEL space (which is top-down, like the PIL image the
    skin is painted into - the flip to GL's bottom-up v happens once, at upload).

    Each line is "stand outside this face and look at it": s follows the
    viewer's right, t follows the viewer's down. The front face is the one worth
    checking by hand - a viewer looking at the player's face has the player's
    right hand (+X) on their own left, so s = 1 - fx, and getting that backwards
    swaps the eyes on an asymmetric skin without changing anything else.
    """
    if face_key == "-z":
        return 1.0 - fx, 1.0 - fy
    if face_key == "+z":
        return fx, 1.0 - fy
    if face_key == "+x":
        return 1.0 - fz, 1.0 - fy
    if face_key == "-x":
        return fz, 1.0 - fy
    if face_key == "+y":
        return 1.0 - fx, 1.0 - fz
    # "-y": the sole, unfolded the other way round from the top
    return 1.0 - fx, fz


# --- animation --------------------------------------------------------------
# Vanilla drives the walk cycle off limbSwing, which grows by
# horizontal_distance * 4 per tick and is then read as cos(limbSwing * 0.6662).
# Folded together that is 2.665 radians of phase per block walked, i.e. one full
# stride every ~2.36 blocks. Quoting it per block rather than per second is the
# whole trick and the reason the brief asks for sin(distance): legs driven by a
# TIMER keep walking after the player stops, and a player carried along by water
# moonwalks. Distance also puts every client in phase with every other one
# without a byte of synchronisation.
LIMB_SWING_PER_BLOCK = 4.0 * 0.6662

# How far the limbs swing at full speed, in radians.
LEG_SWING_ANGLE = 1.4
ARM_SWING_ANGLE = 1.0

# Vanilla's limbSwingAmount is min(1, blocks_per_tick * 4), which at 20 TPS is
# min(1, blocks_per_second * 0.2) - so WALK_SPEED 4.4 lands at 0.88 and only
# sprinting reaches the full 1.0. That headroom is not an accident: it is what
# makes a sprint visibly different from a walk rather than merely faster.
SPEED_TO_SWING_AMOUNT = 0.2

# Vanilla eases limbSwingAmount by 40% per tick. Written as an exponential
# because this game integrates on frame dt, not ticks: `amount += (target -
# amount) * 0.4` per frame would make legs snap to full swing faster on a better
# GPU. -20 * ln(0.6) reproduces the same curve at any frame rate.
LIMB_SWING_SMOOTHING = -20.0 * math.log(0.6)

# Idle arm sway - vanilla's cos(ticks * 0.09) * 0.05 + 0.05 and
# sin(ticks * 0.067) * 0.05, converted from ticks to seconds. Tiny, and the
# thing that stops a standing player reading as a mannequin.
IDLE_SWAY_Z_HZ = 0.09 * 20.0
IDLE_SWAY_X_HZ = 0.067 * 20.0
IDLE_SWAY_Z_AMOUNT = 0.05
IDLE_SWAY_X_AMOUNT = 0.05

# How far the swinging arm travels, in radians, at the peak of a swing.
SWING_ARM_LIFT = 1.9
SWING_ARM_OUT = 0.35

# Must match net.client.RemotePlayer.trigger_swing. Read defensively below
# (the progress is clamped), so if the two ever drift the arm plays a partial
# swing rather than an angle nobody bounded.
SWING_DURATION = 0.25

# Sneaking. Reachable over the wire (protocol.FLAG_SNEAKING) but never set by
# this build - PlayerPhysics has no crouch, Shift is dive/descend. Implemented
# because the flag arrives and something has to be drawn for it; deliberately
# limited to the parts whose vanilla values port cleanly into our orientation
# (the torso lean and the arms following it). Vanilla also shifts the legs' and
# head's pivots, and those numbers are quoted in its Y-down space where they
# cannot be transcribed without a crouch to look at.
SNEAK_BODY_LEAN = -0.5
SNEAK_ARM_ADJUST = -0.4
SNEAK_HEAD_RAISE = 1.0  # px


class PartPose:
    """Where one part is this frame: its rotation, and any shift of its pivot."""

    __slots__ = ("angle_x", "angle_y", "angle_z", "pivot_dx", "pivot_dy", "pivot_dz")

    def __init__(self):
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0
        self.pivot_dx = 0.0  # px
        self.pivot_dy = 0.0
        self.pivot_dz = 0.0


class PlayerEntity:
    """
    The animation state of one drawn player.

    Kept separate from net.client.RemotePlayer, which owns the INTERPOLATION -
    where the player is right now, given two timestamped states from the server.
    This owns everything derived from that: how fast the legs are swinging, how
    far through a punch the arm is, how long they have been standing there. The
    line between them is "does the server know about it": it knows a position,
    it has never heard of a limb.

    sync() is fed from a RemotePlayer once per frame. It takes anything with the
    right attributes, so nothing here imports net.
    """

    def __init__(self, entity_id, username=""):
        self.entity_id = entity_id
        self.username = username

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.sneaking = False

        self.limb_swing = 0.0         # radians of walk-cycle phase, accumulated
        self.limb_swing_amount = 0.0  # 0..1, how wide the swing currently is
        self.idle_time = 0.0
        self.swing_progress = 0.0     # 0..1 through an arm swing, 0 = not swinging

        # None rather than 0.0: the first sync must not read a player who joined
        # 400 blocks from spawn as having just sprinted there.
        self._last_distance = None

    def sync(self, remote, dt: float):
        """Pulls this frame's state from a RemotePlayer (or anything shaped like one)."""
        self.username = remote.username
        self.x, self.y, self.z = remote.x, remote.y, remote.z
        self.yaw, self.pitch = remote.yaw, remote.pitch
        self.sneaking = bool(remote.sneaking)

        if self._last_distance is None:
            self._last_distance = remote.distance_walked
        moved = remote.distance_walked - self._last_distance
        self._last_distance = remote.distance_walked
        if moved < 0.0:
            moved = 0.0  # distance_walked only grows; a reset means a respawn

        self.limb_swing += moved * LIMB_SWING_PER_BLOCK

        speed = moved / dt if dt > 1e-6 else 0.0
        target = min(1.0, speed * SPEED_TO_SWING_AMOUNT)
        blend = 1.0 - math.exp(-LIMB_SWING_SMOOTHING * dt)
        self.limb_swing_amount += (target - self.limb_swing_amount) * blend

        self.idle_time += dt

        swing_timer = getattr(remote, "swing_timer", 0.0)
        if swing_timer <= 0.0:
            self.swing_progress = 0.0
        else:
            self.swing_progress = min(1.0, max(0.0, 1.0 - swing_timer / SWING_DURATION))

    def pose(self):
        """
        Returns {part_name: PartPose} for this frame.

        Every part is rooted directly at the entity - there is no hierarchy, and
        vanilla has none either. It matters: when the torso leans, the head does
        NOT lean with it, which is the only reason a sneaking player can still
        look straight ahead.
        """
        phase = self.limb_swing
        amount = self.limb_swing_amount

        head = PartPose()
        # pitch > 0 is looking up in this project (see Player.forward_vector),
        # and a positive rotation about +X lifts the nose - so this is a
        # straight pass-through rather than the negation vanilla needs.
        head.angle_x = self.pitch

        body = PartPose()

        arm_right = PartPose()
        arm_left = PartPose()
        leg_right = PartPose()
        leg_left = PartPose()

        # A positive X rotation swings a hanging limb FORWARD (its end is at
        # -Y, and R_x takes -Y toward -Z, which is the way the player faces).
        # Arms are half a cycle out of phase with the leg on the same side,
        # which is what walking is.
        leg_right.angle_x = math.cos(phase) * LEG_SWING_ANGLE * amount
        leg_left.angle_x = math.cos(phase + math.pi) * LEG_SWING_ANGLE * amount
        arm_right.angle_x = math.cos(phase + math.pi) * ARM_SWING_ANGLE * amount
        arm_left.angle_x = math.cos(phase) * ARM_SWING_ANGLE * amount

        # Idle sway, pushing both arms very slightly out from the torso and
        # breathing them back and forth. Signed per side: +Z rotation swings the
        # right arm (at +X) away from the body, -Z does the same for the left.
        sway_z = math.cos(self.idle_time * IDLE_SWAY_Z_HZ) * IDLE_SWAY_Z_AMOUNT + IDLE_SWAY_Z_AMOUNT
        sway_x = math.sin(self.idle_time * IDLE_SWAY_X_HZ) * IDLE_SWAY_X_AMOUNT
        arm_right.angle_z += sway_z
        arm_left.angle_z -= sway_z
        arm_right.angle_x += sway_x
        arm_left.angle_x -= sway_x

        if self.sneaking:
            body.angle_x = SNEAK_BODY_LEAN
            arm_right.angle_x += SNEAK_ARM_ADJUST
            arm_left.angle_x += SNEAK_ARM_ADJUST
            head.pivot_dy = SNEAK_HEAD_RAISE

        if self.swing_progress > 0.0:
            # A sine envelope rather than vanilla's two-piece curve. Vanilla
            # snaps the arm to -pi/2 the instant a swing starts and interpolates
            # from there, which works because it drives the swing from a tick
            # counter it also owns; here a swing arrives as a packet, mid-frame,
            # possibly while the arm is already mid-stride. Starting and ending
            # the envelope at exactly zero means the punch ADDS to whatever the
            # walk cycle was already doing instead of fighting it, and cannot
            # produce a visible snap however it lands.
            envelope = math.sin(self.swing_progress * math.pi)
            arm_right.angle_x += envelope * SWING_ARM_LIFT
            arm_right.angle_z += envelope * SWING_ARM_OUT

        return {
            "head": head,
            "body": body,
            "arm_right": arm_right,
            "arm_left": arm_left,
            "leg_right": leg_right,
            "leg_left": leg_left,
        }
