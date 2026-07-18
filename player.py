"""
player/player.py
Ties together physics, health, and game-mode state for the player. This is
the single object that GAME_LOOP/UI code reaches into for "what is the
player doing right now" - position, health, whether they're flying, etc.
Camera-facing (yaw/pitch) lives here too since it's conceptually part of
"where the player is looking", even though only PlayerPhysics needs yaw for
movement math.
"""

import math

import config
from player.physics import PlayerPhysics


class Player:
    def __init__(self, game_mode: str = "survival"):
        self.game_mode = game_mode
        self.physics = PlayerPhysics()
        self.yaw = 0.0
        self.pitch = 0.0
        self.health = config.MAX_HEALTH
        self.alive = True
        self._pitch_limit = math.pi / 2 - 0.001

        # Breath, in seconds (vanilla's 300 ticks). Drains only while the EYE
        # point is inside water - the same test vanilla uses - so wading
        # chest-deep costs nothing and only actually going under does.
        self.air = config.AIR_MAX_SECONDS
        self._drown_timer = 0.0

    def spawn_at(self, world, wx: float, wz: float):
        h = world.get_ground_height_at(int(wx), int(wz))
        # h is the Y of the topmost solid ground block, which occupies the
        # space from y=h to y=h+1. physics.y is the FEET position, so
        # standing directly on that surface means feet at y=h+1 - using
        # h+2 (the old value) left the player hovering a full block above
        # the ground on every spawn, which is what read as "falling
        # through"/landing awkwardly the moment gravity kicked in.
        self.physics.teleport(wx, h + 1, wz)
        self.health = config.MAX_HEALTH
        self.alive = True
        self.yaw = 0.0
        self.pitch = 0.0
        self.air = config.AIR_MAX_SECONDS
        self._drown_timer = 0.0

    def respawn_at_world_spawn(self, world, spawn_x: float, spawn_z: float):
        """Respawns after death: same behavior as spawn_at (surface
        placement, full health, alive again) but explicitly named for the
        death flow since it's always the world's original spawn point,
        never the player's last logged-out position."""
        self.spawn_at(world, spawn_x, spawn_z)

    def set_look(self, yaw: float, pitch: float):
        self.yaw = yaw
        self.pitch = max(-self._pitch_limit, min(self._pitch_limit, pitch))

    def eye_position(self):
        """Physics-accurate eye position, used for raycasting/targeting -
        MUST use the true physics Y (never the smoothed camera offset),
        otherwise block targeting would visibly lag behind where the
        camera is actually pointing during a step-up."""
        return (self.physics.x, self.physics.y + config.PLAYER_EYE_OFFSET, self.physics.z)

    def camera_eye_position(self):
        """Smoothed eye position for RENDERING the camera - subtracts any
        still-catching-up step-smoothing offset (see
        PlayerPhysics.camera_y_offset) so climbing a stair/slab-height bump
        eases the view up over a few frames instead of snapping instantly,
        while collision/targeting (eye_position()) stays perfectly in sync
        with the real physics position."""
        return (self.physics.x,
                self.physics.y + config.PLAYER_EYE_OFFSET - self.physics.camera_y_offset(),
                self.physics.z)

    def forward_vector(self):
        x = -math.sin(self.yaw) * math.cos(self.pitch)
        y = math.sin(self.pitch)
        z = -math.cos(self.yaw) * math.cos(self.pitch)
        return (x, y, z)

    def fall_damage_amount(self, fall_distance: float) -> int:
        """
        How much a fall of this height hurts, in half-heart units. 0 for a fall
        that doesn't.

        Split out from apply_fall_damage because a networked client has to
        REPORT this number rather than act on it - the server owns health there
        (see net/server.py's _handle_damage). Both callers get the same
        arithmetic from the same place, which matters: two copies of a damage
        formula drifting apart is a bug that only ever shows up as "it hurt more
        on his machine".
        """
        if self.game_mode != "survival":
            return 0
        excess = fall_distance - config.FALL_DAMAGE_MIN_DISTANCE
        if excess <= 0:
            return 0
        return math.ceil(excess * config.FALL_DAMAGE_PER_BLOCK) * 2  # half-heart units

    def apply_fall_damage(self, fall_distance: float):
        self.damage(self.fall_damage_amount(fall_distance))

    def air_fraction(self) -> float:
        """0..1 of breath remaining - what the HUD's bubble row is drawn from."""
        return max(0.0, min(1.0, self.air / config.AIR_MAX_SECONDS))

    def update_breathing(self, dt: float):
        """
        Drains breath while submerged and drowns the player once it's gone.

        Reads physics.head_in_water, which PlayerPhysics.update() already
        sampled this frame - re-testing the world here would mean walking the
        block grid twice per frame for the same answer, and the two could
        disagree by a frame.

        Refills instantly on surfacing, which is vanilla (Entity.onEntityUpdate
        calls setAir(300) outright rather than easing it back).
        """
        if not self.alive:
            return
        if self.game_mode != "survival":
            # Creative players don't drown, and their bubble row must not be
            # left mid-drain from a previous survival session either.
            self.air = config.AIR_MAX_SECONDS
            self._drown_timer = 0.0
            return

        if not self.physics.head_in_water:
            self.air = config.AIR_MAX_SECONDS
            self._drown_timer = 0.0
            return

        if self.air > 0.0:
            self.air = max(0.0, self.air - dt)
            if self.air <= 0.0:
                # Arm the timer full, so the first hit lands the moment the last
                # bubble pops rather than two seconds of nothing happening after
                # it - the bubbles running out has to read as the cause.
                self._drown_timer = config.DROWN_DAMAGE_INTERVAL
            return

        self._drown_timer += dt
        while self._drown_timer >= config.DROWN_DAMAGE_INTERVAL and self.alive:
            self._drown_timer -= config.DROWN_DAMAGE_INTERVAL
            self.damage(config.DROWN_DAMAGE)

    def damage(self, amount: int):
        if not self.alive:
            return
        self.health = max(0, self.health - amount)
        if self.health <= 0:
            self.alive = False

    def kill(self):
        """Instant death regardless of remaining health - used for the void
        death boundary, which in vanilla Minecraft kills unconditionally
        (even in Creative) rather than dealing ordinary damage."""
        if not self.alive:
            return
        self.health = 0
        self.alive = False

    def set_game_mode(self, mode: str):
        self.game_mode = mode
        if mode != "creative":
            self.physics.flying = False
