"""
player/physics.py
AABB collision against the voxel world, with axis-separated resolution so
the player slides along walls instead of getting stuck. Three movement modes:
  - Survival/walking: gravity, jump, ground friction/acceleration (ported
    from the original JS build's tuned feel).
  - Swimming: low gravity plus heavy exponential drag, taken from vanilla's
    per-tick water model (see config's SWIMMING block). Water is never part of
    the AABB collision path - it has no collision box at all - so buoyancy is
    handled here as a force, exactly as Minecraft does it.
  - Creative flight: double-tap Space toggles flight on; while flying,
    gravity is disabled and Space/Left-Shift move straight up/down at a
    constant speed, matching Minecraft's creative-mode controls. Flight beats
    water, same as vanilla.
"""

import math

import config
from world import fluids


def _aabb_overlap_1d(a_min, a_max, b_min, b_max):
    return a_min < b_max and a_max > b_min


def _block_collision_boxes(world, bx, by, bz, block_id):
    """
    Returns a list of (min_x,min_y,min_z,max_x,max_y,max_z) LOCAL (0..1)
    boxes for this block's actual collision shape. Normal solid blocks are
    a single full-cube box; doors/stairs/fences have state-dependent
    sub-cell shapes resolved from their metadata. Returns [] for anything
    non-solid (air, open space, etc.).
    """
    from world.blocks import Block, is_solid

    if block_id == Block.DOOR:
        from world.doors import unpack_door_meta, door_collision_bounds
        facing, is_open, _is_top, hinge = unpack_door_meta(world.get_block_meta(bx, by, bz))
        lmin_x, lmin_z, lmax_x, lmax_z = door_collision_bounds(facing, is_open, hinge)
        return [(lmin_x, 0.0, lmin_z, lmax_x, 1.0, lmax_z)]

    if block_id in (Block.STAIRS_WOOD, Block.STAIRS_STONE):
        from world.stairs import unpack_stair_meta, collision_boxes
        facing, is_top, shape = unpack_stair_meta(world.get_block_meta(bx, by, bz))
        return collision_boxes(facing, is_top, shape)

    if block_id == Block.FENCE:
        from world.fences import unpack_connections, collision_boxes
        north, south, east, west = unpack_connections(world.get_block_meta(bx, by, bz))
        return collision_boxes(north, south, east, west)

    if is_solid(block_id):
        return [(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)]

    return []


def _box_overlaps(bx, by, bz, local_box, box_min_x, box_max_x, box_min_y, box_max_y, box_min_z, box_max_z):
    lmin_x, lmin_y, lmin_z, lmax_x, lmax_y, lmax_z = local_box
    wmin_x, wmax_x = bx + lmin_x, bx + lmax_x
    wmin_y, wmax_y = by + lmin_y, by + lmax_y
    wmin_z, wmax_z = bz + lmin_z, bz + lmax_z
    return (_aabb_overlap_1d(box_min_x, box_max_x, wmin_x, wmax_x)
            and _aabb_overlap_1d(box_min_y, box_max_y, wmin_y, wmax_y)
            and _aabb_overlap_1d(box_min_z, box_max_z, wmin_z, wmax_z))


def is_solid_at(world, x: float, y: float, z: float, half_width: float, height: float) -> bool:
    """AABB-vs-voxel-grid overlap test for an axis-aligned box centered at
    (x, *, z) horizontally, with feet at y and the given height. Checks
    real 3D box overlap (not just "does any solid block occupy this XZ
    column") so state-dependent shapes like stairs and door slabs collide
    correctly on all three axes."""
    min_x = int((x - half_width) // 1)
    max_x = int((x + half_width) // 1)
    min_y = int(y // 1)
    max_y = int((y + height - 0.01) // 1)
    min_z = int((z - half_width) // 1)
    max_z = int((z + half_width) // 1)

    box_min_x, box_max_x = x - half_width, x + half_width
    box_min_y, box_max_y = y, y + height
    box_min_z, box_max_z = z - half_width, z + half_width

    # Some blocks (fences: FENCE_HEIGHT=1.5) have a collision shape TALLER
    # than their own 1x1x1 cell, extending up into the block above them -
    # specifically so a normal jump can't clear them. That means a block
    # whose OWN cell sits one below the player's scanned range can still
    # need to be checked. Scanning one extra row below min_y catches that
    # without falsely detecting anything for ordinary (<=1 block tall)
    # collision shapes, since their boxes never reach past their own cell.
    for bx in range(min_x, max_x + 1):
        for by in range(min_y - 1, max_y + 1):
            for bz in range(min_z, max_z + 1):
                block_id = world.get_block(bx, by, bz)
                if block_id == 0:
                    continue
                for local_box in _block_collision_boxes(world, bx, by, bz, block_id):
                    if _box_overlaps(bx, by, bz, local_box, box_min_x, box_max_x,
                                      box_min_y, box_max_y, box_min_z, box_max_z):
                        return True
    return False


class PlayerPhysics:
    """
    Holds the mutable physics state for a player: position, velocity,
    on-ground flag, flight state, and fall-tracking for fall damage.
    Position is the FEET position (not eye height - the camera adds
    PLAYER_EYE_OFFSET on top when rendering).
    """

    def __init__(self, position=(0.0, 0.0, 0.0)):
        self.x, self.y, self.z = position
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.on_ground = False
        self.flying = False
        self.fall_start_y = None
        self.is_falling = False

        # Fluid state, refreshed at the top of every update() and read by the
        # renderer (underwater fog/tint) and by Player (drowning). Two separate
        # flags because vanilla asks two different questions with two different
        # answers: `in_water` is Entity.isInWater (body box shrunk 0.4 on Y) and
        # drives movement; `head_in_water` is Entity.isInsideOfMaterial (the eye
        # point only) and drives breathing and the view. At the surface the
        # first is true and the second is false, and that gap IS floating.
        self.in_water = False
        self.head_in_water = False

        # Whether the last horizontal move was refused by a wall. Only used to
        # reproduce vanilla's "swim up over the ledge you just bumped into",
        # which is the only way out of a body of water onto land.
        self._blocked_horizontally = False

        # double-tap-space-to-fly detection (creative mode only)
        self._last_space_press_time = -999.0
        self._space_was_down_last_frame = False

        # Smooth step-up: when auto-step (see _move_and_collide) instantly
        # raises `self.y` by STEP_HEIGHT to climb a stair/slab-height bump,
        # that raw Y jump happens in a single physics tick and is what made
        # the camera visibly JERK upward each time a stair was climbed -
        # vanilla Minecraft smooths this over several frames instead of
        # snapping. `_step_visual_lag` tracks how far the CAMERA still owes
        # catching up to the true physics Y after a step-up event; it
        # starts equal to the step height and eases toward 0 each frame
        # (see camera_y_offset()), while `self.y` itself jumps immediately
        # so collision/physics logic is never delayed or approximate.
        self._step_visual_lag = 0.0

    def camera_y_offset(self) -> float:
        """Returns how far BELOW the true physics Y the camera should
        currently be drawn, to smooth out an in-progress step-up. Callers
        computing eye/camera position should subtract this from self.y
        (see Player.eye_position usage in main.py) rather than using
        self.y directly for the camera."""
        return self._step_visual_lag

    def teleport(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z
        self.vx = self.vy = self.vz = 0.0
        self.on_ground = False
        self.flying = False
        self.fall_start_y = None
        self.is_falling = False
        self.in_water = False
        self.head_in_water = False
        self._blocked_horizontally = False
        self._step_visual_lag = 0.0

    def _collides(self, world, x, y, z):
        return is_solid_at(world, x, y, z, config.PLAYER_WIDTH / 2, config.PLAYER_HEIGHT)

    def _move_and_collide(self, world, dx, dy, dz):
        if dx != 0:
            nx = self.x + dx
            if not self._collides(world, nx, self.y, self.z):
                self.x = nx
            elif self.on_ground and self._collides(world, nx, self.y, self.z) \
                    and not self._collides(world, nx, self.y + config.STEP_HEIGHT, self.z):
                # auto-step: horizontal move is blocked at current height but
                # clear half a block up (e.g. a single stair/slab-height
                # obstruction) - matches vanilla Minecraft's step-up-without-
                # jumping behavior for stairs and other <=0.5-block bumps.
                # self.y (the real physics position) jumps immediately, but
                # _step_visual_lag banks the same amount so the CAMERA eases
                # up to it smoothly over the next few frames instead of
                # snapping - see camera_y_offset()/update_step_smoothing().
                self.x = nx
                self.y += config.STEP_HEIGHT
                self._step_visual_lag += config.STEP_HEIGHT
            else:
                self.vx = 0.0
                self._blocked_horizontally = True
        if dz != 0:
            nz = self.z + dz
            if not self._collides(world, self.x, self.y, nz):
                self.z = nz
            elif self.on_ground and self._collides(world, self.x, self.y, nz) \
                    and not self._collides(world, self.x, self.y + config.STEP_HEIGHT, nz):
                self.z = nz
                self.y += config.STEP_HEIGHT
                self._step_visual_lag += config.STEP_HEIGHT
            else:
                self.vz = 0.0
                self._blocked_horizontally = True
        if dy != 0:
            ny = self.y + dy
            if not self._collides(world, self.x, ny, self.z):
                self.y = ny
                self.on_ground = False
            else:
                if dy < 0:
                    self.on_ground = True
                self.vy = 0.0

    def update_step_smoothing(self, dt: float):
        """
        Eases `_step_visual_lag` back toward 0 over a short, fixed window
        (STEP_SMOOTH_DURATION seconds) regardless of frame rate, so the
        camera visibly rises smoothly over the step instead of both (a)
        jumping instantly, or (b) taking a variably-long time to catch up
        on a slow frame. Called once per frame from the main game loop,
        separately from physics.update() so it still runs (and finishes
        smoothing out) even on frames where physics.update() itself isn't
        stepping (e.g. while paused-but-still-rendering, if that''s ever a
        thing) - keeping it decoupled avoids the smoothing getting stuck
        mid-step if physics update ordering ever changes.
        """
        if self._step_visual_lag <= 0.0:
            self._step_visual_lag = 0.0
            return
        rate = config.STEP_HEIGHT / config.STEP_SMOOTH_DURATION
        self._step_visual_lag = max(0.0, self._step_visual_lag - rate * dt)

    def _handle_flight_toggle(self, input_state, now: float, game_mode: str, space_key):
        """
        Double-tap Space (within DOUBLE_TAP_WINDOW seconds) toggles flight,
        creative mode only - matches Minecraft's real control scheme.
        """
        if game_mode != "creative":
            self.flying = False
            return

        space_down = input_state.is_key_down(space_key)
        just_pressed = space_down and not self._space_was_down_last_frame

        if just_pressed:
            if now - self._last_space_press_time <= config.DOUBLE_TAP_WINDOW:
                self.flying = not self.flying
                if self.flying:
                    self.vy = 0.0
                self._last_space_press_time = -999.0  # consume, don't chain a third tap into re-toggling
            else:
                self._last_space_press_time = now

        self._space_was_down_last_frame = space_down

    def _refresh_fluid_state(self, world):
        self.in_water = fluids.is_body_in_water(
            world, self.x, self.y, self.z, config.PLAYER_WIDTH / 2, config.PLAYER_HEIGHT
        )
        self.head_in_water = fluids.is_head_in_water(
            world, self.x, self.y + config.PLAYER_EYE_OFFSET, self.z
        )

    def update(self, world, input_state, dt: float, game_mode: str, yaw: float,
               space_key, shift_key, forward_key, backward_key, left_key, right_key,
               jump_pressed_this_frame: bool, now: float, damage_callback=None):
        """
        Advances physics by dt seconds. `space_key`/`shift_key`/direction keys
        are pygame key constants, passed in rather than imported here to keep
        this module import-light and testable with fake key codes.
        """
        self._handle_flight_toggle(input_state, now, game_mode, space_key)
        # Sampled once, before anything moves, and read by the renderer and by
        # Player.update_breathing afterwards - so the tint, the drowning timer
        # and the physics all agree about the same instant.
        self._refresh_fluid_state(world)

        forward_x = -math.sin(yaw)
        forward_z = -math.cos(yaw)
        right_x = math.sin(yaw + math.pi / 2)
        right_z = math.cos(yaw + math.pi / 2)

        wish_x = wish_z = 0.0
        if input_state.is_key_down(forward_key):
            wish_x += forward_x
            wish_z += forward_z
        if input_state.is_key_down(backward_key):
            wish_x -= forward_x
            wish_z -= forward_z
        if input_state.is_key_down(right_key):
            wish_x += right_x
            wish_z += right_z
        if input_state.is_key_down(left_key):
            wish_x -= right_x
            wish_z -= right_z

        wish_len = math.hypot(wish_x, wish_z)
        if wish_len > 0:
            wish_x /= wish_len
            wish_z /= wish_len

        swimming = self.in_water and not self.flying

        if self.flying:
            speed = config.FLY_SPEED
            target_vx = wish_x * speed
            target_vz = wish_z * speed
            accel = config.WALK_ACCEL
            self.vx += (target_vx - self.vx) * min(1.0, accel * dt)
            self.vz += (target_vz - self.vz) * min(1.0, accel * dt)

            vertical = 0.0
            if input_state.is_key_down(space_key):
                vertical += 1.0
            if input_state.is_key_down(shift_key):
                vertical -= 1.0
            self.vy = vertical * config.FLY_VERTICAL_SPEED

            self.on_ground = False
            self.is_falling = False
            self.fall_start_y = None
        elif swimming:
            # Vanilla's water model: accelerate, then decay ALL THREE axes
            # toward zero, then apply a much weaker gravity. Written as a single
            # exponential rather than a per-frame multiply so the result is
            # identical at any frame rate - a per-frame `v *= 0.8` would make the
            # player swim measurably faster on a worse GPU.
            self.vx += wish_x * config.WATER_MOVE_ACCEL * dt
            self.vz += wish_z * config.WATER_MOVE_ACCEL * dt

            self.vy -= config.WATER_GRAVITY * dt
            if input_state.is_key_down(space_key):
                # Note this is NOT a jump and does not require ground: it is a
                # steady upward push. Held down it beats gravity and the player
                # rises; released, they sink slowly. Let go at the surface and
                # `in_water` flips off the moment the feet clear the water line,
                # normal gravity pulls them back in, and the player settles into
                # vanilla's bob - which is all "floating on water" ever was.
                self.vy += config.WATER_SWIM_UP_ACCEL * dt
            elif input_state.is_key_down(shift_key):
                self.vy -= config.WATER_SINK_ACCEL * dt

            decay = math.exp(-config.WATER_DRAG * dt)
            self.vx *= decay
            self.vy *= decay
            self.vz *= decay

            if self.vy < -config.WATER_MAX_SINK_SPEED:
                self.vy = -config.WATER_MAX_SINK_SPEED

            # Water cancels a fall. Vanilla zeroes fallDistance every tick an
            # entity is in water, so a 60-block drop into a lake costs nothing -
            # without this, hitting the sea would still kill the player outright
            # because the landing check only fires once they reached the seabed.
            self.is_falling = False
            self.fall_start_y = None
        else:
            accel = config.WALK_ACCEL * (1.0 if self.on_ground else config.AIR_CONTROL)
            target_vx = wish_x * config.WALK_SPEED
            target_vz = wish_z * config.WALK_SPEED

            if wish_len > 0:
                self.vx += (target_vx - self.vx) * min(1.0, accel * dt)
                self.vz += (target_vz - self.vz) * min(1.0, accel * dt)
            elif self.on_ground:
                friction = min(1.0, config.GROUND_FRICTION * dt)
                self.vx -= self.vx * friction
                self.vz -= self.vz * friction

            self.vy -= config.GRAVITY * dt
            if jump_pressed_this_frame and self.on_ground:
                self.vy = config.JUMP_VELOCITY
                self.on_ground = False

            max_fall = -50.0
            if self.vy < max_fall:
                self.vy = max_fall

            if not self.on_ground and self.vy < -0.001:
                if not self.is_falling:
                    self.is_falling = True
                    self.fall_start_y = self.y
            elif self.on_ground:
                self.is_falling = False

        self._blocked_horizontally = False
        self._move_and_collide(world, self.vx * dt, 0, 0)
        self._move_and_collide(world, 0, 0, self.vz * dt)

        prev_on_ground = self.on_ground
        self._move_and_collide(world, 0, self.vy * dt, 0)

        # Vanilla: swimming into a wall you could climb sets motionY = 0.3. It
        # is the only way out of water onto land - the swim-up push alone tops
        # out level with the surface, so without this the player treads water
        # against the shore forever.
        if swimming and self._blocked_horizontally and wish_len > 0:
            probe_x = self.x + wish_x * 0.1
            probe_z = self.z + wish_z * 0.1
            if not self._collides(world, probe_x, self.y + 0.6, probe_z):
                self.vy = config.WATER_LEDGE_BOOST

        # landed this frame: apply fall damage if falling was tracked and we're not flying
        if (not self.flying and not swimming and not prev_on_ground and self.on_ground
                and self.fall_start_y is not None):
            fall_distance = self.fall_start_y - self.y
            if damage_callback is not None:
                damage_callback(fall_distance)
            self.is_falling = False
            self.fall_start_y = None
