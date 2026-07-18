"""
player/controls.py
Translates raw input (mouse look, key presses) into player actions: camera
rotation, block breaking (with survival break-time simulation, creative
instant-break), block placing, hotbar scrolling/selection, and inventory
toggling. This is the glue layer between core.input.InputState and the
Player/World/Inventory objects - no rendering here, just game logic.
"""

import pygame
import config
from world.blocks import Block, get_item_def
from world.raycast import raycast
from world.world import REPLACEABLE_BLOCKS
from world.doors import facing_from_player_yaw
from world.stairs import facing_from_player_yaw as stair_facing_from_player_yaw

INTERACTABLE_BLOCKS = frozenset((Block.DOOR, Block.CRAFTING_TABLE))
STAIR_BLOCKS = frozenset((Block.STAIRS_WOOD, Block.STAIRS_STONE))


class PlayerControls:
    def __init__(self):
        self.is_breaking = False
        self.break_progress = 0.0
        self.break_target_key = None
        self.targeted_block = None  # last raycast hit, refreshed each frame

    def update_look(self, player, input_state, sensitivity_multiplier: float):
        if input_state.mouse_dx == 0 and input_state.mouse_dy == 0:
            return
        sens = config.MOUSE_SENSITIVITY_BASE * sensitivity_multiplier
        new_yaw = player.yaw - input_state.mouse_dx * sens
        new_pitch = player.pitch - input_state.mouse_dy * sens
        player.set_look(new_yaw, new_pitch)

    def update_targeting(self, player, world):
        origin = player.eye_position()
        direction = player.forward_vector()
        self.targeted_block = raycast(world, origin, direction)
        if self.targeted_block is None and self.is_breaking:
            self._cancel_break()

    def handle_hotbar_scroll(self, inventory, input_state):
        if input_state.wheel_delta == 0:
            return
        direction = 1 if input_state.wheel_delta > 0 else -1
        inventory.selected_slot = (inventory.selected_slot + direction) % config.HOTBAR_SIZE

    def handle_hotbar_number_keys(self, inventory, input_state):
        digit_keys = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,
                      pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]
        for i, key in enumerate(digit_keys):
            if input_state.was_key_pressed(key):
                inventory.selected_slot = i

    def _get_break_seconds(self, block_id, held_item_id):
        item_def = get_item_def(block_id)
        if item_def is None:
            return 0.3
        hardness = item_def.hardness if item_def.hardness is not None else 0.5

        held_def = get_item_def(held_item_id) if held_item_id else None
        correct_tool = held_def is not None and held_def.tool_type == item_def.tool

        if not item_def.tool:
            multiplier = 1.0
        elif correct_tool:
            tier = held_def.tool_tier or 1
            multiplier = 1.0 / (tier + 0.5)
        else:
            multiplier = 3.0
        return max(0.15, hardness * multiplier)

    def _cancel_break(self):
        self.is_breaking = False
        self.break_progress = 0.0
        self.break_target_key = None

    def _spawn_break_particles(self, particle_system, block_id, x, y, z):
        if particle_system is not None:
            particle_system.spawn_break_particles(block_id, x, y, z)

    def _complete_break(self, world, inventory, player, particle_system, client=None):
        t = self.targeted_block
        block_id = world.get_block(t["x"], t["y"], t["z"])
        if block_id == Block.AIR:
            return
        self._break_at(world, t["x"], t["y"], t["z"], client)
        self._spawn_break_particles(particle_system, block_id, t["x"], t["y"], t["z"])

        if player.game_mode == "survival":
            item_def = get_item_def(block_id)
            drop_id = item_def.drops if (item_def and item_def.drops is not None) else block_id
            inventory.add_item(drop_id, 1)

    def _break_at(self, world, x, y, z, client):
        """
        The one place a break either happens or is asked for.

        With a client, the block is NOT removed here. That is not caution, it is
        the rule the whole net/ package is built on: the server owns the world,
        and this side owns a picture of it. Removing it locally would look
        identical for about 30 milliseconds and then diverge the moment the
        server disagreed - and a client that has already deleted the block has
        nothing left to correct itself against. The block vanishes when
        S_BLOCK_CHANGE says it did, one round trip later, which over a loopback
        is a frame and over Radmin is not much worse.

        Particles and the inventory drop still fire immediately, because those
        are this player's own feedback rather than world state: a swing that
        produced nothing for a frame reads as a dropped input.
        """
        if client is not None:
            client.send_break(x, y, z)
        else:
            world.break_block(x, y, z)

    def handle_break_instant_creative(self, world, player, particle_system, client=None):
        if self.targeted_block is None:
            return
        t = self.targeted_block
        block_id = world.get_block(t["x"], t["y"], t["z"])
        if block_id == Block.AIR:
            return
        self._break_at(world, t["x"], t["y"], t["z"], client)
        self._spawn_break_particles(particle_system, block_id, t["x"], t["y"], t["z"])

    def start_breaking(self):
        self.is_breaking = True
        self.break_progress = 0.0
        self.break_target_key = None

    def stop_breaking(self):
        self._cancel_break()

    def update_breaking(self, world, inventory, player, particle_system, dt: float, client=None):
        if player.game_mode == "creative":
            return  # instant break handled directly on mouse-down
        if not self.is_breaking or self.targeted_block is None:
            return

        t = self.targeted_block
        key = (t["x"], t["y"], t["z"])
        if key != self.break_target_key:
            self.break_target_key = key
            self.break_progress = 0.0

        block_id = world.get_block(t["x"], t["y"], t["z"])
        if block_id == Block.AIR:
            self.is_breaking = False
            return

        held = inventory.selected_stack()
        held_id = held["id"] if held else None
        total_seconds = self._get_break_seconds(block_id, held_id)
        self.break_progress += dt / total_seconds

        if self.break_progress >= 1.0:
            self._complete_break(world, inventory, player, particle_system, client)
            self._cancel_break()

    def try_interact(self, world, player, client=None) -> bool:
        """
        Right-click interaction with whatever block is directly targeted
        (not the placement cell - the block itself). Returns True if the
        click was consumed by an interaction (door toggled, crafting table
        should open, etc.) so callers know NOT to also fall through to
        placing a block from the hotbar this click.

        Doors: clicking anywhere on a door toggles it open/closed, exactly
        like vanilla Minecraft - works regardless of what's in the player's
        hand, and does not consume/place any item. With a client, this asks
        rather than toggles: a door is world state, and the metadata write
        that opens it has to reach everyone in the same order it happened, or
        two players standing at the same door disagree about which way it is
        facing.

        Crafting table: returns True (click consumed) but the actual UI
        opening is handled by the caller (main.py owns screen state) -
        this just identifies that an interactable was clicked. Callers
        should check world.get_block(...) at the targeted cell themselves
        if they need to know WHICH interactable it was. Nothing is sent for
        it: a crafting grid is a menu, not a thing in the world.
        """
        t = self.targeted_block
        if t is None:
            return False
        block_id = world.get_block(t["x"], t["y"], t["z"])
        if block_id == Block.DOOR:
            if client is not None:
                client.send_door_toggle(t["x"], t["y"], t["z"])
            else:
                world.toggle_door(t["x"], t["y"], t["z"])
            return True
        if block_id in INTERACTABLE_BLOCKS:
            return True  # crafting table etc. - caller opens the relevant UI
        return False

    def place_block(self, world, inventory, player, client=None):
        """
        Every check below runs identically with or without a client, and that is
        the point. They are all questions about blocks this side already holds
        (is the cell free, does the door's top half fit, am I standing in it), so
        the replica answers them exactly as the real world would - which is what
        lets the client refuse a placement locally instead of consuming an item
        and watching the server silently drop it.

        What changes is only the last step: with a client the block is requested
        rather than written, and the server's S_BLOCK_CHANGE is what actually
        puts it there. The item still leaves the hotbar immediately - that is
        this player's own inventory, which no packet in this protocol carries,
        and a hotbar that lagged a round trip behind the click would feel broken
        in a way the block appearing a frame late does not.
        """
        t = self.targeted_block
        if t is None or t["place"] is None:
            return
        px, py, pz = t["place"]

        stack = inventory.selected_stack()
        if stack is None:
            return
        item_def = get_item_def(stack["id"])
        if item_def is None or not item_def.is_block:
            return

        half_w = config.PLAYER_WIDTH / 2
        p = player.physics

        def _overlaps_player_at(check_y):
            return (
                int((p.x - half_w) // 1) <= px <= int((p.x + half_w) // 1)
                and int((p.z - half_w) // 1) <= pz <= int((p.z + half_w) // 1)
                and int(p.y // 1) <= check_y <= int((p.y + config.PLAYER_HEIGHT) // 1)
            )

        if stack["id"] == Block.DOOR:
            # Doors occupy TWO vertical cells (bottom half at py, top half
            # at py+1) - check both against the player's own AABB, not just
            # the bottom one, otherwise a door could be placed clipping
            # straight through the player's head.
            if _overlaps_player_at(py) or _overlaps_player_at(py + 1):
                return
            facing = facing_from_player_yaw(player.yaw)
            if client is not None:
                # Mirrors World.place_door's own precondition, because the
                # server will apply exactly that one and say nothing if it
                # fails.
                if world.get_block(px, py, pz) != Block.AIR or world.get_block(px, py + 1, pz) != Block.AIR:
                    return
                client.send_place(px, py, pz, stack["id"], kind="door",
                                  facing=facing, yaw=player.yaw)
            else:
                placed = world.place_door(px, py, pz, facing, player.yaw)
                if not placed:
                    return
        elif stack["id"] in STAIR_BLOCKS:
            if _overlaps_player_at(py):
                return
            facing = stair_facing_from_player_yaw(player.yaw)
            # Matches vanilla Minecraft's stair placement rule: pointing at
            # a block's TOP face always places a right-side-up stair
            # (resting on that surface); pointing at a block's BOTTOM face
            # always places an upside-down stair (hanging from that
            # surface); pointing at a SIDE face places right-side-up if you
            # clicked the lower half of that face, upside-down if the upper
            # half. Using hit_frac_y alone (ignoring which face was hit)
            # got this backwards for the most common case - placing on top
            # of a floor block - since the ray enters the floor block very
            # close to its top (hit_frac_y near 1.0), which read as "upper
            # half" and produced an upside-down stair instead of a normal one.
            #
            # This is resolved HERE and sent as a plain is_top, never re-derived
            # server-side: the ray that decided it started at this player's eye,
            # and the server has neither the ray nor a reason to guess at one.
            face_normal = t.get("face_normal")
            if face_normal == (0, 1, 0):
                is_top = False   # clicked the block's top face -> stair sits on top of it
            elif face_normal == (0, -1, 0):
                is_top = True    # clicked the block's bottom face -> stair hangs upside-down from it
            else:
                is_top = t.get("hit_frac_y", 0.0) >= 0.5
            if client is not None:
                if world.get_block(px, py, pz) != Block.AIR:
                    return
                client.send_place(px, py, pz, stack["id"], kind="stairs",
                                  facing=facing, is_top=is_top)
            else:
                placed = world.place_stairs(px, py, pz, stack["id"], facing, is_top)
                if not placed:
                    return
        else:
            if _overlaps_player_at(py):
                return
            if client is not None:
                if world.get_block(px, py, pz) not in REPLACEABLE_BLOCKS:
                    return
                client.send_place(px, py, pz, stack["id"])
            else:
                placed = world.place_block(px, py, pz, stack["id"])
                if not placed:
                    return

        if player.game_mode == "survival":
            inventory.remove_from_slot(inventory.selected_slot, 1)
