"""
main.py
Entry point: owns the top-level application state machine (title -> world list
/ multiplayer -> in-game, with pause/settings/inventory as overlays on the
in-game state) and the single frame loop that drives input polling, per-state
updates, and rendering. Run this file to start the game.

THE THREE SESSIONS
------------------
Exactly one thing separates them, and it is this attribute:

    self.client is None  ->  singleplayer: this thread owns the World
    self.client is not None -> networked: the World here is a picture

Everything else follows from it. Ticking, streaming, saving, and block edits
all ask that one question, because in a networked session the World is owned by
a server thread (net/server.py) and this one is forbidden to touch it - it holds
a ReplicaWorld instead and asks for changes by packet.

The host runs BOTH: a GameServer over the real World, and a loopback GameClient
that talks to it exactly like a guest's would. So the host is `self.client is
not None` too, and takes the identical path a guest does. That is the point of
the whole arrangement (see net/__init__.py) - the alternative is a host-only
code path, i.e. bugs that only ever reproduce on one of the two machines.

Singleplayer deliberately does NOT run a server. It could - it would be one
fewer path - but the server streams SERVER_VIEW_DISTANCE chunks per client and
no more, and the replica would be a second full copy of every chunk in memory.
That would cap offline render distance at 8 and double its RAM to buy
uniformity in a session that has nobody to be uniform with.
"""

import sys
import threading
import time
from collections import deque

import pygame

# PyOpenGL wraps every single gl* call in a glGetError() check plus array-size
# validation by default. glGetError forces the driver to synchronously drain its
# command queue, so each of the ~1600 GL calls a frame makes (two per visible
# chunk, plus the other renderers) stalls the GPU pipeline dead. That is what
# pins the frame rate at ~30 on hardware that should be bored by this workload:
# the bottleneck is the wrapper, not the card.
#
# These MUST be set before anything imports OpenGL.GL - PyOpenGL bakes them into
# its wrappers at import time, so setting them afterwards silently does nothing.
# That is also why they live here at the entry point instead of in config.py:
# config is imported by modules that themselves import OpenGL.GL, so there is no
# ordering config could guarantee.
#
# Error checking stays available for debugging by flipping these back to True.
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ARRAY_SIZE_CHECKING = False

import config
from core.window import Window
from core.input import InputState
from core.camera import Camera
from net import MODE_CLIENT, MODE_HOST, MODE_SINGLEPLAYER
from net.client import GameClient
from net.protocol import DEFAULT_PORT
from net.server import GameServer, SERVER_VIEW_DISTANCE, find_lan_address
from render.texture_atlas import TextureAtlas
from render.chunk_renderer import WorldRenderer
from render.water_renderer import WaterRenderer
from render.shadow_renderer import ShadowRenderer
from render.outline_renderer import OutlineRenderer
from render.door_renderer import DoorRenderer
from render.stairs_renderer import StairsRenderer
from render.fence_renderer import FenceRenderer
from render.player_renderer import PlayerRenderer
from render.hand import HandRenderer
from render.particles import ParticleSystem
from world.world import World, world_to_chunk_coords
from player.player import Player
from player.controls import PlayerControls
from inventory.inventory import Inventory
from ui.ui_renderer import UIRenderer
from ui.hud import draw_hud
from ui.screen_main_menu import MainMenuScreen
from ui.screen_world_select import WorldSelectScreen
from ui.screen_multiplayer import MultiplayerScreen
from ui.screen_create_world import CreateWorldScreen
from ui.screen_settings import SettingsScreen
from ui.screen_pause import PauseScreen
from ui.screen_death import DeathScreen
from ui.screen_inventory import InventoryScreen
from ui.screen_crafting_table import CraftingTableScreen
from ui.screen_loading import LoadingScreen
from world.blocks import Block
from save import world_save
from save import settings_save


STATE_MAIN_MENU = "main_menu"
STATE_WORLD_SELECT = "world_select"
STATE_CREATE_WORLD = "create_world"
STATE_MULTIPLAYER = "multiplayer"
STATE_SETTINGS_FROM_MENU = "settings_from_menu"
STATE_IN_GAME = "in_game"
STATE_LOADING_WORLD = "loading_world"

# How much world has to have arrived before a networked session is playable, in
# chunks around the player. The server sends 6 per tick closest-first, so this
# is about half a second - long enough that the player does not land on an air
# pocket and fall, short enough that nobody reads it as a hang.
NETWORK_LOADING_RADIUS = 2


class Game:
    def __init__(self):
        self.window = Window()
        pygame.key.set_repeat(0)

        self.input = InputState()
        self.settings = settings_save.load_settings()

        self.texture_atlas = TextureAtlas()
        self.texture_atlas.upload_to_gpu()

        self.ui = UIRenderer(self.window.width, self.window.height)

        self.state = STATE_MAIN_MENU
        self.state_before_settings = STATE_MAIN_MENU

        self.main_menu = MainMenuScreen(self._go_to_world_select, self._go_to_multiplayer,
                                        self._go_to_settings_from_menu, self._quit)
        self.world_select_screen = WorldSelectScreen(self._start_playing_world, self._go_to_create_world,
                                                     self._back_to_main_menu)
        self.multiplayer_screen = MultiplayerScreen(self.settings, self._start_connecting,
                                                    self._back_to_main_menu)
        self.create_world_screen = CreateWorldScreen(self._create_and_play_world, self._go_to_world_select)
        self.settings_screen = SettingsScreen(self.settings, self._back_from_settings)

        for screen in (self.main_menu, self.world_select_screen, self.multiplayer_screen,
                       self.create_world_screen, self.settings_screen):
            screen.layout(self.window.width, self.window.height)

        # in-game objects, created fresh each time a world is entered
        self.world = None
        self.player = None
        self.controls = None
        self.inventory = None
        self.world_renderer = None
        self.water_renderer = None
        self.shadow_renderer = None
        self.outline_renderer = None
        self.door_renderer = None
        self.stairs_renderer = None
        self.fence_renderer = None
        self.player_renderer = None
        self.hand_renderer = None
        self.particles = None
        self.camera = None
        self.current_world_dir = None

        # -- the session ------------------------------------------------------
        # session_mode is for the UI's benefit (the pause menu offers "Open to
        # LAN" only where it means something). The code below asks
        # `self.client is None` instead, because that is the question that
        # actually has consequences: who is allowed to touch the World.
        self.session_mode = MODE_SINGLEPLAYER
        self.server = None
        self.client = None

        # A connect attempt in flight. GameClient.connect() blocks for up to
        # CONNECT_TIMEOUT seconds, which on the frame loop is a frozen window -
        # exactly the "reads as a hang rather than a failure" its own comment
        # warns about. So it runs on a thread and the frame loop watches.
        self._connect_thread = None
        self._connect_client = None

        self.paused = False
        self.inventory_open = False
        self.pause_screen = None
        self.inventory_screen = None
        self.in_game_settings_screen = None
        self.showing_settings_in_game = False
        self.crafting_table_screen = None
        self.crafting_table_open = False
        self._crafting_table_pos = None  # (x,y,z) of the table currently open, so we know where to re-check it's still there
        self.death_screen = None
        self.player_dead = False
        self._world_spawn_x = 0.0
        self._world_spawn_z = 0.0

        self._save_timer = 0.0

        # Leftover fraction of a game tick carried between frames. Simulated
        # time must advance in fixed 1/20 s steps regardless of frame rate -
        # see world/ticks.py.
        self._tick_accumulator = 0.0

        # world-generation loading overlay state (see _enter_world / _update_loading_world)
        self.loading_screen = LoadingScreen()
        self._pending_enter = None       # dict with target_x/target_z/saved_state/meta while loading
        self._loading_total_units = 1
        self._loading_done_units = 0
        self._loading_stall_frames = 0

    # -- menu navigation ------------------------------------------------------

    def _go_to_world_select(self):
        self.world_select_screen.refresh_worlds()
        self.world_select_screen.layout(self.window.width, self.window.height)
        self.state = STATE_WORLD_SELECT

    def _go_to_multiplayer(self):
        # layout() rebuilds the text fields from settings, so re-running it is
        # what re-fills the address and name with whatever was last used.
        self.multiplayer_screen.error = None
        self.multiplayer_screen.connecting = False
        self.multiplayer_screen.status = None
        self.multiplayer_screen.layout(self.window.width, self.window.height)
        self.state = STATE_MULTIPLAYER

    def _go_to_create_world(self):
        self.state = STATE_CREATE_WORLD
        self.create_world_screen.layout(self.window.width, self.window.height)

    def _go_to_settings_from_menu(self):
        self.state_before_settings = STATE_MAIN_MENU
        self.state = STATE_SETTINGS_FROM_MENU
        self.settings_screen.layout(self.window.width, self.window.height)

    def _back_to_main_menu(self):
        self.main_menu.layout(self.window.width, self.window.height)
        self.state = STATE_MAIN_MENU

    def _create_and_play_world(self, display_name, seed, game_mode):
        meta = world_save.create_world_metadata(display_name, seed, game_mode)
        self._enter_world(meta)

    def _start_playing_world(self, meta):
        world_save.touch_last_played(meta.get("dir_name") or meta.get("_dir_name"))
        self._enter_world(meta)

    # -- joining someone else's world -----------------------------------------

    def _start_connecting(self, host, port, username):
        """Called by MultiplayerScreen with an address that already parsed."""
        settings_save.save_settings(self.settings)  # the screen wrote the name/address in
        client = GameClient(username)
        self._connect_client = client
        self._connect_thread = threading.Thread(target=client.connect, args=(host, port),
                                                daemon=True, name="connect")
        self._connect_thread.start()

    def _poll_connect(self):
        """Watches the connect thread. The screen is already drawing
        "Connecting to ..." and has disabled its own button."""
        if self._connect_thread is None or self._connect_thread.is_alive():
            return
        self._connect_thread = None
        client = self._connect_client
        self._connect_client = None

        if not client.connected:
            # client.error is already a sentence written for a player - see
            # net.client._friendly_connect_error. Showing it verbatim is the
            # whole reason that function exists.
            self.multiplayer_screen.set_error(client.error or "Could not connect")
            return

        self.client = client
        self.session_mode = MODE_CLIENT
        self._begin_network_loading(fresh_session=True)

    def _open_to_lan(self):
        """
        Hands this world over to a server and rejoins it through the loopback.

        Everything after this point runs the guest's code path, which is why
        there is a brief loading screen: the World the player has been standing
        in is now the SERVER's, and what they get back is a replica of it,
        rebuilt from chunks over a queue. It is the same world, and they are
        standing in the same spot - but the objects are new, so the meshes have
        to be.
        """
        if self.client is not None or self.world is None:
            return

        # Last chance to write the world from this thread: after start() the
        # server's tick thread owns it, and a save from here would be walking
        # arrays it is mutating.
        self._save_current_world()

        server = GameServer(self.world, self.current_world_dir, self.player.game_mode,
                            self._world_spawn_x, self._world_spawn_z, port=0)
        if not server.start():
            # port=0 practically cannot fail to bind, so if this fires it is
            # something structural rather than "25565 was taken" - server.error
            # carries the OS's own reason.
            self.pause_screen.set_network_state(MODE_SINGLEPLAYER, error=server.error)
            return

        self.server = server
        self.session_mode = MODE_HOST

        physics = self.player.physics
        client = GameClient(self.settings.get("username") or "Player")
        # The host is the one player whose position is known before they
        # connect. Without this they would join at world spawn - see
        # GameServer.attach_local_client.
        client.attach_local(server, position=(physics.x, physics.y, physics.z))
        self.client = client

        self.controls.stop_breaking()  # its target refers to a world we no longer own
        self._begin_network_loading(fresh_session=False)

    def _begin_network_loading(self, fresh_session: bool):
        self._pending_enter = {
            "network": True,
            "fresh_session": fresh_session,
            "phase": "welcome",
            "mesh_queue": None,
            "mesh_total": 1,
        }
        self.loading_screen = LoadingScreen()
        self.loading_screen.set_progress(0.0, "Joining world")
        self.paused = False
        self.inventory_open = False
        self.showing_settings_in_game = False
        self.crafting_table_open = False
        self._save_timer = 0.0
        self._tick_accumulator = 0.0
        self.state = STATE_LOADING_WORLD
        self.window.set_mouse_grab(False)

    # -- entering a world -----------------------------------------------------

    def _build_session_objects(self, game_mode):
        """Everything an in-world session needs that is not the World itself.
        Shared by the offline path and the join path, which differ only in where
        the World came from."""
        self.player = Player(game_mode=game_mode)
        self.controls = PlayerControls()
        self.inventory = Inventory(game_mode=game_mode)
        self.world_renderer = WorldRenderer(self.texture_atlas)
        self.water_renderer = WaterRenderer()
        self.shadow_renderer = ShadowRenderer()
        self.outline_renderer = OutlineRenderer()
        self.door_renderer = DoorRenderer(self.texture_atlas)
        self.stairs_renderer = StairsRenderer(self.texture_atlas)
        self.fence_renderer = FenceRenderer(self.texture_atlas)
        self.player_renderer = PlayerRenderer()
        self.hand_renderer = HandRenderer(self.texture_atlas)
        self.particles = ParticleSystem(self.texture_atlas)
        far_plane = self.settings["render_distance"] * config.CHUNK_SIZE_X * 1.3 + 32
        self.camera = Camera(aspect=self.window.width / self.window.height,
                             fov=config.FOV_DEFAULT, far=far_plane)

        w, h = self.window.width, self.window.height
        self.pause_screen = PauseScreen(self._resume, self._open_settings_in_game,
                                        self._quit_to_title, on_open_to_lan=self._open_to_lan)
        self.pause_screen.layout(w, h)
        self.inventory_screen = InventoryScreen(self.texture_atlas)
        self.inventory_screen.layout(w, h, game_mode)
        self.crafting_table_screen = CraftingTableScreen(self.texture_atlas)
        self.crafting_table_screen.layout(w, h)
        self.crafting_table_open = False
        self._crafting_table_pos = None
        self.death_screen = DeathScreen(self._respawn_player, self._quit_to_title)
        self.death_screen.layout(w, h)
        self.player_dead = False
        self.in_game_settings_screen = SettingsScreen(self.settings, self._close_settings_in_game)
        self.in_game_settings_screen.layout(w, h)

    def _enter_world(self, meta):
        """
        Phase 1 (synchronous, fast): create the world/player/renderer objects
        and figure out where the player needs to appear. Actual chunk/tree
        generation for that area is then spread across frames in
        _update_loading_world() while a "Generating level" overlay is shown -
        see that method for why. This also naturally fixes trees "popping in
        on top of the player": the world isn't revealed until tree growth
        for the whole spawn neighborhood has finished, not just terrain.

        Always offline. A world is opened to the network from the pause menu
        (see _open_to_lan), never from the world list - the list would have to
        ask which port and which name before it could offer it, and that is a
        second, worse copy of the multiplayer screen.
        """
        dir_name = meta.get("dir_name") or meta.get("_dir_name")
        seed = meta["seed"]
        game_mode = meta["game_mode"]

        self.current_world_dir = dir_name
        self.session_mode = MODE_SINGLEPLAYER
        self.server = None
        self.client = None
        self.world = World(seed=seed, save_dir=dir_name)
        self._build_session_objects(game_mode)

        spawn_x = meta.get("spawn_x", 0.0)
        spawn_z = meta.get("spawn_z", 0.0)
        self._world_spawn_x = spawn_x
        self._world_spawn_z = spawn_z

        # Decide WHERE the player needs to appear before generating anything.
        # Bug fix: this used to always stream/generate chunks around the
        # world's spawn_x/spawn_z, then teleport the player to their saved
        # x/y/z afterwards. On a second session the player is almost never
        # standing at spawn, so their saved position landed in a chunk that
        # was never generated (still all-air) - dropping them into the void
        # below bedrock every single time. Now we resolve the real target
        # position first and generate THAT area.
        saved_state = world_save.load_player_state(dir_name)
        if saved_state is not None:
            target_x = saved_state.get("x", spawn_x)
            target_z = saved_state.get("z", spawn_z)
        else:
            target_x = spawn_x
            target_z = spawn_z

        self.world.update_streaming(target_x, target_z, self.settings["render_distance"])
        self._loading_total_units = max(
            1, len(self.world._pending_gen_queue) + len(self.world._pending_trees_queue)
        )
        self._loading_done_units = 0

        self._pending_enter = {
            "network": False,
            "dir_name": dir_name,
            "game_mode": game_mode,
            "target_x": target_x,
            "target_z": target_z,
            "saved_state": saved_state,
            # Loading runs in two phases: "terrain" (generate blocks + grow
            # trees) then "meshes" (turn those blocks into GPU buffers). The
            # second phase used to happen in one unbudgeted burst AFTER the
            # progress bar had already reached 100% - see
            # _update_loading_meshes.
            "phase": "terrain",
            "mesh_queue": None,
            "mesh_total": 1,
        }
        self.loading_screen = LoadingScreen()
        self.loading_screen.set_progress(0.0, "Building terrain")
        self._loading_stall_frames = 0
        self.paused = False
        self.inventory_open = False
        self.showing_settings_in_game = False
        self.state = STATE_LOADING_WORLD
        self.window.set_mouse_grab(False)
        self._save_timer = 0.0
        self._tick_accumulator = 0.0

    # -- loading --------------------------------------------------------------

    def _update_loading(self, dt):
        if self._pending_enter["network"]:
            self._update_loading_network(dt)
        else:
            self._update_loading_world(dt)

    def _update_loading_network(self, dt):
        """
        Joining, offline-generation's counterpart: instead of making the world,
        wait for it to arrive.

        The phases mirror the offline ones on purpose - "the world exists" then
        "the world is meshed" - because the second half is identical work and
        needs the identical time-boxing. What differs is only that terrain is
        produced by a server rather than by worldgen, and that it can fail
        halfway (a guest's server can hang up; worldgen cannot).
        """
        info = self._pending_enter
        client = self.client
        client.poll(dt)

        if not client.connected:
            self._abort_network_loading(
                client.disconnect_reason or client.error or "Lost connection to server")
            return

        if info["phase"] == "welcome":
            if not client.welcomed:
                self.loading_screen.set_progress(0.05, "Joining world")
                self.loading_screen.update(dt)
                return
            self._on_welcomed()
            info["phase"] = "chunks"

        if info["phase"] == "chunks":
            arrived, wanted = self._count_arrived_chunks()
            if arrived < wanted:
                self.loading_screen.set_progress(0.05 + 0.45 * arrived / wanted, "Downloading terrain")
                self.loading_screen.update(dt)
                return
            # Whatever else is still in flight keeps arriving during play, the
            # same way generated chunks do offline.
            self.client.pop_pending_meshes(len(self.client.pending_meshes))
            info["phase"] = "meshes"
            info["mesh_queue"] = deque(self.world.pop_dirty_chunks(len(self.world.chunks)))
            info["mesh_total"] = max(1, len(info["mesh_queue"]))
            self.loading_screen.set_progress(0.5, "Building meshes")
            self.loading_screen.update(dt)
            return

        self._update_loading_meshes(dt)

    def _count_arrived_chunks(self):
        physics_x, physics_z = self._network_spawn_xz()
        pcx, pcz = world_to_chunk_coords(int(physics_x), int(physics_z))
        wanted = 0
        arrived = 0
        for dx in range(-NETWORK_LOADING_RADIUS, NETWORK_LOADING_RADIUS + 1):
            for dz in range(-NETWORK_LOADING_RADIUS, NETWORK_LOADING_RADIUS + 1):
                wanted += 1
                chunk = self.world.chunks.get((pcx + dx, pcz + dz))
                # `generated` is the flag decode_chunk_into sets. Mere presence
                # proves nothing: ReplicaWorld makes an empty chunk for any
                # coordinate anyone asks about, and one of those is exactly what
                # a player would fall through.
                if chunk is not None and chunk.generated:
                    arrived += 1
        return arrived, max(1, wanted)

    def _network_spawn_xz(self):
        return self.player.physics.x, self.player.physics.z

    def _on_welcomed(self):
        """S_WELCOME has landed, so there is a ReplicaWorld to move into."""
        client = self.client
        info = self._pending_enter

        # Drop every mesh built against the world being left behind. For a host
        # that world still exists - it is the one the server now owns - and the
        # replica is about to re-send the same chunks under the same keys. Any
        # mesh not dropped here that falls outside SERVER_VIEW_DISTANCE would
        # keep being drawn forever from a buffer nothing will ever update again.
        if self.world_renderer is not None:
            for key in list(self.world_renderer.chunk_meshes.keys()):
                self._remove_chunk_meshes(*key)

        if info["fresh_session"]:
            self._build_session_objects(client.game_mode)

        self.world = client.world
        self._world_spawn_x = client.spawn_x
        self._world_spawn_z = client.spawn_z
        self.player.set_game_mode(client.game_mode)

        # Never Player.spawn_at here: it resolves a ground height, and on a
        # replica whose chunks have not arrived that reads air all the way down
        # and puts the player at y=1, under the world. The server already said
        # where we are, and said it as a teleport precisely so this has one path.
        if client.pending_teleport is not None:
            x, y, z = client.pending_teleport
            client.pending_teleport = None
            self.player.physics.teleport(x, y, z)
        if client.health is not None:
            self.player.health = client.health
            self.player.alive = client.health > 0
        if client.air is not None:
            self.player.air = client.air
        self.player_dead = not self.player.alive

    def _abort_network_loading(self, reason):
        self._pending_enter = None
        self._teardown_session()
        self._go_to_multiplayer()
        self.multiplayer_screen.set_error(reason)

    def _update_loading_world(self, dt):
        """
        Runs once per frame while STATE_LOADING_WORLD is active for an offline
        world. Generates a budgeted slice of chunks/trees per frame (same
        budgeted approach as normal gameplay streaming, just with a bigger
        budget since nothing else is happening this frame) and advances the
        progress bar, so a big render distance doesn't freeze the game for
        several seconds like the old blocking loop in _enter_world used to.

        Bug fix: World._queue_tree_pass() only enqueues a chunk for tree
        growth if that chunk already exists in world.chunks - but the very
        first update_streaming() call (in _enter_world) runs BEFORE any
        chunk has been generated, so every tree pass request was silently
        dropped and trees_generated stayed False for the whole neighborhood.
        Previously this went unnoticed because normal gameplay calls
        update_streaming() again every single frame once in STATE_IN_GAME,
        so trees caught up a moment later once chunks existed - which is
        exactly the "trees pop in on top of the player" bug being fixed
        here. Re-calling update_streaming() periodically while loading
        re-queues tree passes now that chunks exist, so trees are fully
        grown before the loading screen ever dismisses.

        Note: a thin ring of chunks right at the edge of render_distance can
        never grow trees in place - their 3x3 neighborhood pokes just
        outside the generated halo, by design (avoids leaf spill onto
        ungenerated terrain). Those finish naturally the moment the player
        moves and the streaming window shifts, exactly like normal
        walking-triggered generation. So "done loading" means the tree
        queue has stopped shrinking (only edge-ring stragglers left), not
        that it's fully empty - otherwise the loading screen would hang
        forever waiting for chunks that structurally can never be ready.
        """
        world = self.world
        info = self._pending_enter

        if info["phase"] == "meshes":
            self._update_loading_meshes(dt)
            return

        world.update_streaming(info["target_x"], info["target_z"], self.settings["render_distance"])

        gen_left = len(world._pending_gen_queue)
        trees_left = len(world._pending_trees_queue)
        still_pending = gen_left > 0 or trees_left > 0

        if still_pending:
            before = gen_left + trees_left
            world.process_generation_budget(gen_budget=48, tree_budget=24)
            after = len(world._pending_gen_queue) + len(world._pending_trees_queue)
            progressed = before - after

            self._loading_done_units += max(0, progressed)
            self._loading_total_units = max(self._loading_total_units, self._loading_done_units + after)

            if progressed <= 0:
                # nothing left that can be generated in place this frame -
                # only edge-ring stragglers remain (see docstring); stop
                # waiting on them instead of stalling the loading screen
                self._loading_stall_frames += 1
            else:
                self._loading_stall_frames = 0

            if self._loading_stall_frames < 5:
                status = "Growing trees" if gen_left == 0 and trees_left > 0 else "Building terrain"
                fraction = self._loading_done_units / self._loading_total_units
                # Terrain owns the first HALF of the bar, meshing the second
                # (see _update_loading_meshes). The bar used to run to ~100%
                # on terrain alone and then sit there through several seconds
                # of unbudgeted mesh building, which is precisely why the
                # freeze looked like it happened "after loading finished".
                self.loading_screen.set_progress(min(0.5, fraction * 0.5), status)
                self.loading_screen.update(dt)
                return
            # else: fall through and finish - remaining entries are
            # structurally stuck edge chunks, not real pending work

        # Terrain and trees are done. Hand the whole dirty backlog to the
        # meshing phase rather than letting _finish_entering_world blow
        # through it in a single frame.
        info["phase"] = "meshes"
        info["mesh_queue"] = deque(world.pop_dirty_chunks(len(world.chunks)))
        info["mesh_total"] = max(1, len(info["mesh_queue"]))
        self.loading_screen.set_progress(0.5, "Building meshes")
        self.loading_screen.update(dt)

    def _update_loading_meshes(self, dt):
        """
        Phase 2 of loading: turn blocks into GPU meshes, time-boxed to
        config.LOADING_MESH_MS_PER_FRAME per frame so the frame loop keeps
        returning to pygame.

        This work used to live in _finish_entering_world as one synchronous
        burst: pop_dirty_chunks(len(self.world.chunks)) followed by a loop
        calling five renderers' rebuild_chunk() for every chunk. At render
        distance 32 that is ~3200 chunks x (build_mesh_data + merge +
        glBufferData + build_shadow_spots + three instance scans) inside a
        SINGLE frame - several seconds during which pygame.event.get() is
        never reached, so Windows paints the window "not responding". The
        progress bar could not show any of it, because the bar had already
        been set to 1.0 before this loop even started. That is the hang that
        appears right as the bar fills.

        The chunks arrive closest-first (pop_dirty_chunks preserves the
        distance ordering update_streaming gives the generation queue, and the
        server streams closest-first too), so a popleft() here builds the ground
        under the player before the horizon.
        """
        info = self._pending_enter
        queue = info["mesh_queue"]
        deadline = time.perf_counter() + config.LOADING_MESH_MS_PER_FRAME / 1000.0

        while queue and time.perf_counter() < deadline:
            cx, cz = queue.popleft()
            chunk = self.world.get_chunk(cx, cz)
            if chunk is None:
                continue  # unloaded between being queued and being built
            self._rebuild_chunk_meshes(chunk)

        if queue:
            built = info["mesh_total"] - len(queue)
            self.loading_screen.set_progress(0.5 + 0.5 * built / info["mesh_total"], "Building meshes")
            self.loading_screen.update(dt)
            return

        self.loading_screen.set_progress(1.0, "Done")
        self.loading_screen.update(dt)
        if not self.loading_screen.is_done():
            return

        if info["network"]:
            self._finish_network_entering()
        else:
            self._finish_entering_world()

    def _finish_network_entering(self):
        self._pending_enter = None
        self.state = STATE_IN_GAME
        self.window.set_mouse_grab(not self.player_dead)
        self.window.clock.tick()

    def _finish_entering_world(self):
        """Chunk/tree generation for the spawn area is complete. Places the
        player, restores saved state, and switches into normal gameplay."""
        info = self._pending_enter
        self._pending_enter = None
        target_x = info["target_x"]
        target_z = info["target_z"]
        saved_state = info["saved_state"]

        # Always resolve a fresh surface height first. This guarantees the
        # player is standing on solid ground even if a saved y was stale/
        # invalid, and is also the ONLY placement for a brand new world.
        self.player.spawn_at(self.world, target_x, target_z)

        # Restore saved inventory/health/look direction if this world was
        # played before. Position is intentionally re-derived from the
        # surface height above rather than trusting saved_state["y"]
        # directly - the saved y is only used as a sanity cross-check.
        if saved_state is not None:
            self.inventory.slots = saved_state.get("inventory_slots", self.inventory.slots)
            self.inventory.selected_slot = saved_state.get("selected_slot", 0)
            self.player.health = saved_state.get("health", self.player.health)
            # Saves written before drowning existed have no "air" key, so this
            # falls back to a full bar rather than KeyError-ing an old world open.
            self.player.air = saved_state.get("air", config.AIR_MAX_SECONDS)
            self.player.alive = self.player.health > 0

            saved_y = saved_state.get("y")
            surface_y = self.player.physics.y
            # Trust the saved y only if it's above the freshly-computed
            # surface (i.e. player was airborne/jumping/on a tree, not
            # underground) - never below it, which is exactly the "under
            # bedrock" failure mode this fix targets.
            if saved_y is not None and saved_y >= surface_y - 0.5:
                self.player.physics.teleport(target_x, saved_y, target_z)
            # else: keep the safe surface position spawn_at already set

            self.player.set_look(saved_state.get("yaw", 0.0), saved_state.get("pitch", 0.0))

        self.player_dead = not self.player.alive

        # Meshes for the spawn area were already built during the loading
        # screen's second phase (see _update_loading_meshes), so the player
        # still spawns into a fully-built world rather than watching it pop
        # in chunk by chunk - but the building happened across many
        # time-boxed frames with a live progress bar instead of one frozen
        # multi-second frame right after the bar hit 100%.
        #
        # Anything marked dirty since then (e.g. a chunk spawn_at had to
        # generate to resolve the surface height) is a handful at most, and
        # normal streaming picks it up at CHUNK_BUILD_BUDGET_PER_FRAME.
        self.state = STATE_IN_GAME
        self.window.set_mouse_grab(not self.player_dead)
        # Reset the frame clock right as gameplay starts, so the first real
        # physics update gets a normal dt instead of however long the final
        # (possibly heavy) loading frame actually took in wall-clock time -
        # belt-and-suspenders alongside Window.tick()'s dt clamp against the
        # same fall-through-the-world tunneling failure mode.
        self.window.clock.tick()

    # -- per-chunk GPU meshes -------------------------------------------------

    def _rebuild_chunk_meshes(self, chunk):
        """
        Rebuilds every renderer's GPU data for one chunk.

        Centralised because the list has to be identical in all three places
        that touch it (the loading screen's mesh phase, the per-frame dirty
        drain, and unloading) - it was already duplicated three times before
        water made it six, and a renderer missing from one copy is a silent bug:
        the chunk simply keeps drawing its previous contents.
        """
        self.world_renderer.rebuild_chunk(chunk, self.world)
        self.water_renderer.rebuild_chunk(chunk, self.world)
        self.shadow_renderer.rebuild_chunk(chunk, self.world)
        self.door_renderer.rebuild_chunk(chunk, self.world)
        self.stairs_renderer.rebuild_chunk(chunk, self.world)
        self.fence_renderer.rebuild_chunk(chunk, self.world)

    def _remove_chunk_meshes(self, cx: int, cz: int):
        """Frees every renderer's GPU data for a chunk the world just unloaded.
        A renderer left out here leaks its buffers AND keeps drawing the chunk
        regardless of render distance."""
        self.world_renderer.remove_chunk(cx, cz)
        self.water_renderer.remove_chunk(cx, cz)
        self.shadow_renderer.remove_chunk(cx, cz)
        self.door_renderer.remove_chunk(cx, cz)
        self.stairs_renderer.remove_chunk(cx, cz)
        self.fence_renderer.remove_chunk(cx, cz)

    # -- in-game overlays -----------------------------------------------------

    def _on_fall(self, fall_distance):
        """PlayerPhysics' landing callback. Offline this hurts; on the network
        it reports, and the hurting comes back as S_PLAYER_STATE a round trip
        later. Doing both would take the hit twice."""
        amount = self.player.fall_damage_amount(fall_distance)
        if amount <= 0:
            return
        if self.client is None:
            self.player.damage(amount)
        else:
            self.client.send_damage(amount, cause="fall")

    def _kill_player(self, cause):
        if self.client is None:
            self.player.kill()
        else:
            # config.MAX_HEALTH is the server's own ceiling on a damage packet
            # (see _handle_damage), so this is "everything you have" rather than
            # a magic number that happens to be large.
            self.client.send_damage(config.MAX_HEALTH, cause=cause)
            # The void is the one death the local side cannot afford to wait a
            # round trip for: the player is still falling, still calling this
            # every frame, and every one of those frames is another damage
            # packet. Going dead locally stops the fall being simulated at all.
            self.player.kill()

    def _resume(self):
        self.paused = False
        self.window.set_mouse_grab(True)

    def _respawn_player(self):
        if self.client is not None:
            # Ask, don't act. Where spawn IS comes back as S_TELEPORT, because
            # this side has never been sent the spawn chunk and would place
            # itself under the terrain - see GameServer._handle_respawn. The
            # death screen stays up until it answers.
            self.client.send_respawn()
            return
        self.player.respawn_at_world_spawn(self.world, self._world_spawn_x, self._world_spawn_z)
        self.player_dead = False
        self.window.set_mouse_grab(True)

    def _open_settings_in_game(self):
        self.showing_settings_in_game = True

    def _close_settings_in_game(self):
        self.showing_settings_in_game = False
        settings_save.save_settings(self.settings)

    def _back_from_settings(self):
        self.state = self.state_before_settings
        settings_save.save_settings(self.settings)

    # -- leaving --------------------------------------------------------------

    def _quit_to_title(self):
        self._shutdown_session()
        self._back_to_main_menu()

    def _quit(self):
        self._shutdown_session()
        settings_save.save_settings(self.settings)
        pygame.quit()
        sys.exit(0)

    def _shutdown_session(self):
        """Stops whatever this session was, saving whatever it owns."""
        if self.session_mode == MODE_SINGLEPLAYER:
            self._save_current_world()
        elif self.session_mode == MODE_HOST:
            # Order matters. stop() joins the tick thread, and only once it has
            # returned does this thread own the World again - saving before that
            # would be reading arrays mid-mutation. GameServer.stop deliberately
            # does not save (it cannot know whether the caller wants it to), so
            # the final write is ours.
            server = self.server
            server.stop()
            if self.client is not None:
                self.client.disconnect()
            if self.current_world_dir is not None and self._pending_enter is None:
                world_save.save_all_loaded_chunks(self.current_world_dir, server.world)
                world_save.update_spawn(self.current_world_dir,
                                        self.player.physics.x, self.player.physics.z)
                world_save.save_player_state(self.current_world_dir, self.player, self.inventory)
        elif self.session_mode == MODE_CLIENT:
            # A guest owns no disk. Their inventory and position live on the
            # host's machine or nowhere, and writing them into a local save
            # folder would invent a world that does not exist.
            if self.client is not None:
                self.client.disconnect()

        self._teardown_session()

    def _teardown_session(self):
        self.server = None
        self.client = None
        self.session_mode = MODE_SINGLEPLAYER
        self.world = None
        self.current_world_dir = None
        self._pending_enter = None

    def _save_current_world(self):
        """Offline only: in every other session the World belongs to a thread
        that is not this one."""
        if self._pending_enter is not None:
            return  # world hasn't finished generating yet - nothing valid to save
        if self.client is not None:
            return
        if self.world is not None and self.current_world_dir is not None:
            world_save.save_all_loaded_chunks(self.current_world_dir, self.world)
            world_save.update_spawn(self.current_world_dir, self.player.physics.x, self.player.physics.z)
            world_save.save_player_state(self.current_world_dir, self.player, self.inventory)

    def _save_host_player_state(self):
        """The host's periodic save. Chunks are the server's job (it autosaves
        on its own thread); the player's inventory and position are ours, and
        touch no World."""
        if self.current_world_dir is None or self.player is None:
            return
        world_save.save_player_state(self.current_world_dir, self.player, self.inventory)

    # -- main loop --------------------------------------------------------------

    def run(self):
        while True:
            self.input.begin_frame()
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self._quit()
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                self._route_event(event)
            self.input.poll_events(events)

            if self.input.quit_requested:
                self._quit()

            dt = self.window.tick(self.settings["fps_limit"])
            self._update(dt)
            self._render()

    def _handle_resize(self, w, h):
        self.window.handle_resize(w, h)
        self.ui.resize(w, h)
        if self.camera:
            self.camera.set_aspect(w, h)
        self.main_menu.layout(w, h)
        self.world_select_screen.layout(w, h)
        self.multiplayer_screen.layout(w, h)
        self.create_world_screen.layout(w, h)
        self.settings_screen.layout(w, h)
        if self.pause_screen:
            self.pause_screen.layout(w, h)
        if self.inventory_screen:
            self.inventory_screen.layout(w, h, self.player.game_mode)
        if self.crafting_table_screen:
            self.crafting_table_screen.layout(w, h)
        if self.death_screen:
            self.death_screen.layout(w, h)
        if self.in_game_settings_screen:
            self.in_game_settings_screen.layout(w, h)

    def _route_event(self, event):
        w, h = self.window.width, self.window.height

        if self.state == STATE_MAIN_MENU:
            self.main_menu.handle_event(event, w, h)
        elif self.state == STATE_WORLD_SELECT:
            if self.world_select_screen.pending_delete is not None and event.type == pygame.MOUSEBUTTONDOWN:
                self.world_select_screen.handle_confirmation_click(event.pos)
            else:
                self.world_select_screen.handle_event(event, w, h)
        elif self.state == STATE_MULTIPLAYER:
            self.multiplayer_screen.handle_event(event, w, h)
        elif self.state == STATE_CREATE_WORLD:
            self.create_world_screen.handle_event(event, w, h)
        elif self.state == STATE_SETTINGS_FROM_MENU:
            self.settings_screen.handle_event(event, w, h)
        elif self.state == STATE_IN_GAME:
            self._route_in_game_event(event, w, h)
        # STATE_LOADING_WORLD intentionally ignores input - the player can't
        # act until the world has finished arriving.

    def _route_in_game_event(self, event, w, h):
        if self.player_dead:
            self.death_screen.handle_event(event, w, h)
            return
        if self.showing_settings_in_game:
            self.in_game_settings_screen.handle_event(event, w, h)
            return
        if self.crafting_table_open:
            if event.type == pygame.MOUSEBUTTONDOWN:
                right = event.button == 3
                if event.button in (1, 3):
                    self.crafting_table_screen.handle_click(event.pos, right, self.inventory)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_e, pygame.K_ESCAPE):
                    self._close_crafting_table()
            return
        if self.inventory_open:
            if event.type == pygame.MOUSEBUTTONDOWN:
                right = event.button == 3
                if event.button in (1, 3):
                    self.inventory_screen.handle_click(event.pos, right, self.inventory, self.player.game_mode)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_e, pygame.K_ESCAPE):
                    self._toggle_inventory()
            return
        if self.paused:
            self.pause_screen.handle_event(event, w, h)
            return

        # normal gameplay input
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._pause()
            elif event.key == pygame.K_e:
                self._toggle_inventory()
        elif event.type == pygame.MOUSEBUTTONDOWN and self.window.mouse_grabbed:
            if event.button == 1:
                self.hand_renderer.trigger_swing()
                if self.client is not None:
                    self.client.send_swing()  # cosmetic; so the others see the arm move
                if self.player.game_mode == "creative":
                    self.controls.handle_break_instant_creative(
                        self.world, self.player, self.particles, self.client)
                else:
                    self.controls.start_breaking()
            elif event.button == 3:
                self._handle_right_click()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.controls.stop_breaking()

    def _pause(self):
        self.paused = True
        self.window.set_mouse_grab(False)
        # Refreshed on every open rather than pushed on change: the player count
        # moves on the server's thread, and this is the only moment anyone looks
        # at it.
        self.pause_screen.set_network_state(
            self.session_mode,
            port=self.server.port if self.server is not None else DEFAULT_PORT,
            player_count=self.server.player_count if self.server is not None else 0,
            address=find_lan_address() if self.server is not None else None,
        )

    def _handle_right_click(self):
        """
        Right-click first tries to interact with whatever block is directly
        targeted (open a door, or open the crafting table UI) - only if
        that doesn't consume the click does it fall through to placing a
        block from the hotbar, matching vanilla Minecraft's right-click
        priority (interact-over-place).
        """
        t = self.controls.targeted_block
        if t is not None:
            block_id = self.world.get_block(t["x"], t["y"], t["z"])
            if block_id == Block.CRAFTING_TABLE:
                self._open_crafting_table((t["x"], t["y"], t["z"]))
                return
        if self.controls.try_interact(self.world, self.player, self.client):
            self.hand_renderer.trigger_swing()
            return  # door toggled - don't also place a block this click
        slots_before = self.inventory.selected_stack()
        count_before = slots_before["count"] if slots_before else 0
        self.controls.place_block(self.world, self.inventory, self.player, self.client)
        slots_after = self.inventory.selected_stack()
        count_after = slots_after["count"] if slots_after else 0
        if count_after != count_before:
            # something was actually consumed/placed this click - swing to
            # match, rather than swinging on every right-click regardless
            # of whether it did anything (e.g. clicking with an empty hand
            # or into an already-occupied cell). This reads the INVENTORY
            # rather than the world, which is what makes it work unchanged
            # over the network: the block is still a round trip away, but the
            # item left the hotbar the instant the click was accepted.
            self.hand_renderer.trigger_swing()
            if self.client is not None:
                self.client.send_swing()

    def _open_crafting_table(self, table_pos):
        self._crafting_table_pos = table_pos
        self.crafting_table_open = True
        self.crafting_table_screen.layout(self.window.width, self.window.height)
        self.window.set_mouse_grab(False)

    def _close_crafting_table(self):
        self.crafting_table_screen.close_and_return_items(self.inventory)
        self.crafting_table_open = False
        self._crafting_table_pos = None
        self.window.set_mouse_grab(True)

    def _toggle_inventory(self):
        self.inventory_open = not self.inventory_open
        if self.inventory_open:
            self.window.set_mouse_grab(False)
            self.inventory_screen.layout(self.window.width, self.window.height, self.player.game_mode)
        else:
            self.inventory_screen.close_and_return_items(self.inventory)
            self.window.set_mouse_grab(True)

    # -- per-state update ---------------------------------------------------

    def _update(self, dt):
        mouse_pos = pygame.mouse.get_pos()

        if self.state == STATE_MAIN_MENU:
            self.main_menu.update_hover(mouse_pos)
        elif self.state == STATE_WORLD_SELECT:
            self.world_select_screen.update_hover(mouse_pos)
        elif self.state == STATE_MULTIPLAYER:
            self.multiplayer_screen.update_hover(mouse_pos)
            self._poll_connect()
        elif self.state == STATE_CREATE_WORLD:
            self.create_world_screen.update_hover(mouse_pos)
        elif self.state == STATE_SETTINGS_FROM_MENU:
            self.settings_screen.update_hover(mouse_pos)
        elif self.state == STATE_IN_GAME:
            self._update_in_game(dt, mouse_pos)
        elif self.state == STATE_LOADING_WORLD:
            self._update_loading(dt)

    def _effective_render_distance(self):
        """
        In a networked session the world only extends as far as the server
        streams it (SERVER_VIEW_DISTANCE), so asking for more is asking for
        chunks that are never coming - and fog computed for a 32-chunk view
        would leave 8 chunks of terrain sitting in clear air with a hard edge
        into the void behind it. Clamping is not a limitation being imposed
        here; it is this side admitting what it actually has.
        """
        distance = self.settings["render_distance"]
        if self.client is not None:
            distance = min(distance, SERVER_VIEW_DISTANCE)
        return distance

    def _update_in_game(self, dt, mouse_pos):
        # The link is drained even while paused or in a menu. A paused client is
        # still a connected one: the server keeps ticking, keeps sending, and
        # keeps expecting a keepalive - stop polling and it drops us for a
        # timeout while the player reads the options screen.
        if self.client is not None:
            self.client.poll(dt)
            if not self.client.connected:
                self._on_connection_lost()
                return
            self._apply_client_state()

        if self.player_dead:
            self.death_screen.update_hover(mouse_pos)
            return
        if self.showing_settings_in_game:
            self.in_game_settings_screen.update_hover(mouse_pos)
            return
        if self.crafting_table_open:
            return
        if self.inventory_open:
            return
        if self.paused:
            self.pause_screen.update_hover(mouse_pos)
            return

        self.controls.update_look(self.player, self.input, self.settings["mouse_sensitivity"])
        self.controls.handle_hotbar_scroll(self.inventory, self.input)
        self.controls.handle_hotbar_number_keys(self.inventory, self.input)

        jump_pressed = self.input.was_key_pressed(pygame.K_SPACE)
        now = pygame.time.get_ticks() / 1000.0
        self.player.physics.update(
            self.world, self.input, dt, self.player.game_mode, self.player.yaw,
            space_key=pygame.K_SPACE, shift_key=pygame.K_LSHIFT,
            forward_key=pygame.K_w, backward_key=pygame.K_s,
            left_key=pygame.K_a, right_key=pygame.K_d,
            jump_pressed_this_frame=jump_pressed, now=now,
            damage_callback=self._on_fall,
        )
        self.player.physics.update_step_smoothing(dt)
        # Must come after physics.update(), which is what samples whether the
        # player's head is submerged this frame. Skipped entirely on the
        # network: the server runs the drowning timer for everyone against its
        # own copy of the water (GameServer._update_client_vitals), and two
        # timers racing over one bar is how the bar starts flickering.
        if self.client is None:
            self.player.update_breathing(dt)
        self.water_renderer.update(dt)

        self.controls.update_targeting(self.player, self.world)
        self.controls.update_breaking(self.world, self.inventory, self.player,
                                      self.particles, dt, self.client)
        self.particles.update(dt)
        self.hand_renderer.update(dt, self.player)
        # Keep the swing looping while a survival-mode break is still in
        # progress (holding left-click on a hard block) - a single swing
        # triggered on mouse-down would otherwise finish and leave the arm
        # frozen mid-pose for the remaining mining time, unlike vanilla's
        # continuous punching animation.
        if self.controls.is_breaking and not self.hand_renderer.animation.is_swinging():
            self.hand_renderer.trigger_swing()

        # Void death: falling out of the generated world (below the world's
        # floor, into open air with nothing left to land on) kills the
        # player outright, matching vanilla Minecraft's void damage - this
        # applies regardless of game mode (Java Edition kills even in
        # Creative once far enough into the void), since it exists purely
        # to catch "fell through/around the world" states, not as a normal
        # survival hazard.
        if self.player.physics.y <= config.VOID_DEATH_Y:
            self._kill_player("void")

        if not self.player.alive:
            self.player_dead = True
            self.window.set_mouse_grab(False)
            return

        if self.client is not None:
            self.client.send_move(self.player, dt)
            self.player_renderer.update(self.client.remote_players, dt)
        else:
            # Simulated time. Fixed 1/20 s steps, decoupled from the frame rate:
            # every reactive delay in the world is quoted in ticks, so driving
            # this from dt directly would make water flow faster on a better
            # GPU. Ticks only run offline and not while paused, which matches
            # singleplayer Minecraft - pausing stops the world. On the network
            # the server's thread owns time, and pausing stops nothing, because
            # someone else is still playing.
            self._tick_accumulator += dt
            ticks_run = 0
            while (self._tick_accumulator >= config.TICK_SECONDS
                   and ticks_run < config.MAX_TICKS_PER_FRAME):
                self._tick_accumulator -= config.TICK_SECONDS
                self.world.tick()
                ticks_run += 1
            if ticks_run >= config.MAX_TICKS_PER_FRAME:
                # Owed more ticks than one frame may run (a chunk-load spike, or
                # the window was alt-tabbed for a minute). Paying the whole debt
                # makes the next frame slower still, which owes even more. Drop
                # it: the world skips a moment of simulated time, which is
                # recoverable.
                self._tick_accumulator = 0.0

            self.world.update_streaming(self.player.physics.x, self.player.physics.z,
                                        self.settings["render_distance"])
            self.world.process_generation_budget()

        # Free GPU meshes for chunks the world just unloaded - without this,
        # every chunk ever visited stays resident in the renderer forever
        # (a memory leak) AND keeps being drawn regardless of render
        # distance, which is why changing the render distance setting
        # previously had no visible effect on how far the player could see.
        # On the network the unloads arrive as S_UNLOAD_CHUNK instead of being
        # decided locally, but they land in the same list.
        for cx, cz in self.world.pop_recently_unloaded():
            self._remove_chunk_meshes(cx, cz)

        dirty = self.world.pop_dirty_chunks(config.CHUNK_BUILD_BUDGET_PER_FRAME)
        for cx, cz in dirty:
            chunk = self.world.get_chunk(cx, cz)
            if chunk is not None:
                self._rebuild_chunk_meshes(chunk)
        if self.client is not None:
            # GameClient keeps its own list of chunks that arrived, for the
            # loading screen to budget. In play the dirty queue above already
            # covers them - and covers block changes too, which that list never
            # sees - so this exists only to stop it growing for the rest of the
            # session.
            self.client.pop_pending_meshes(len(self.client.pending_meshes))

        self.camera.position = self.player.camera_eye_position()
        self.camera.set_pitch_yaw(self.player.yaw, self.player.pitch)
        self.camera.far = self.settings["render_distance"] * config.CHUNK_SIZE_X * 1.3 + 32

        self._save_timer += dt
        if self._save_timer > 30.0:
            self._save_timer = 0.0
            if self.session_mode == MODE_SINGLEPLAYER:
                self._save_current_world()
            elif self.session_mode == MODE_HOST:
                self._save_host_player_state()

    def _apply_client_state(self):
        """
        Folds what the server said about US into the local Player.

        Health is copied straight across every frame, with no change-detection.
        That is only correct because nothing on this side subtracts from it any
        more: falling and the void report to the server (see _on_fall /
        _kill_player) and drowning was always the server's. Health has exactly
        one owner, so the local number is a display of the server's, and the
        earlier "only apply when it changed" dance - which existed purely so
        local fall damage would not be healed on the next frame - is gone with
        the thing it was working around.
        """
        client = self.client

        if client.pending_teleport is not None:
            x, y, z = client.pending_teleport
            client.pending_teleport = None
            self.player.physics.teleport(x, y, z)
            # A teleport lands the player somewhere new with whatever velocity
            # they had a moment ago - which after a death is a terminal-velocity
            # fall. Not zeroing it means respawning at spawn and immediately
            # being driven back into the ground at 50 blocks a second.
            self.player.physics.vx = 0.0
            self.player.physics.vy = 0.0
            self.player.physics.vz = 0.0
            self.player.physics.is_falling = False
            self.player.physics.fall_start_y = None

        if client.health is not None:
            self.player.health = client.health
            self.player.alive = client.health > 0
        if client.air is not None:
            self.player.air = client.air

        # The server answering a respawn request is the only thing that can
        # bring a networked player back, so this is where the death screen
        # closes - not in _respawn_player, which only asks.
        if self.player_dead and self.player.alive:
            self.player_dead = False
            self.window.set_mouse_grab(True)

    def _on_connection_lost(self):
        reason = self.client.disconnect_reason or "Lost connection to server"
        was_host = self.session_mode == MODE_HOST
        self._shutdown_session()
        if was_host:
            # The host's own loopback cannot time out, so this is the server
            # thread having died - which it only does on an unhandled exception
            # it has already printed. Nothing to reconnect to.
            self._back_to_main_menu()
            return
        self._go_to_multiplayer()
        self.multiplayer_screen.set_error(reason)

    # -- rendering ------------------------------------------------------------

    def _render(self):
        self.ui.clear()
        w, h = self.window.width, self.window.height

        if self.state == STATE_MAIN_MENU:
            self.main_menu.draw(self.ui.surface, w, h)
        elif self.state == STATE_WORLD_SELECT:
            self.world_select_screen.draw(self.ui.surface, w, h)
        elif self.state == STATE_MULTIPLAYER:
            self.multiplayer_screen.draw(self.ui.surface, w, h)
        elif self.state == STATE_CREATE_WORLD:
            self.create_world_screen.draw(self.ui.surface, w, h)
        elif self.state == STATE_SETTINGS_FROM_MENU:
            self.settings_screen.draw(self.ui.surface, w, h)
        elif self.state == STATE_IN_GAME:
            self._render_in_game()
        elif self.state == STATE_LOADING_WORLD:
            self._render_loading_world()

        self.ui.upload_and_draw()
        self.window.swap_buffers()

    def _render_loading_world(self):
        from OpenGL.GL import glClear, glClearColor, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
        # Explicit, because _render_in_game leaves the clear colour set to the
        # underwater blue whenever the player quit to the title while submerged.
        glClearColor(*config.FOG_COLOR, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        w, h = self.window.width, self.window.height
        self.loading_screen.draw(self.ui.surface, w, h)

    def _render_in_game(self):
        from OpenGL.GL import glClear, glClearColor, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT

        render_distance = self._effective_render_distance()
        underwater = self.player.physics.head_in_water

        # Underwater, fog isn't atmosphere any more - it IS the water, and it
        # has to close in within a few blocks or the sea reads as clear air with
        # a blue filter on it. The clear colour has to move with it, otherwise
        # the sky colour keeps showing wherever nothing was drawn and the fog
        # fades to a horizon that shouldn't exist down there.
        if underwater:
            fog_color = config.UNDERWATER_FOG_COLOR
            fog_start = config.UNDERWATER_FOG_START
            fog_end = config.UNDERWATER_FOG_END
        else:
            fog_color = config.FOG_COLOR
            fog_end = render_distance * config.CHUNK_SIZE_X
            fog_start = fog_end * 0.55
        glClearColor(*fog_color, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        pcx, pcz = world_to_chunk_coords(int(self.player.physics.x), int(self.player.physics.z))
        visible_keys = [
            key for key in self.world_renderer.chunk_meshes.keys()
            if (key[0] - pcx) ** 2 + (key[1] - pcz) ** 2 <= render_distance ** 2
        ]
        self.world_renderer.render(self.camera, visible_keys, render_distance_chunks=render_distance,
                                    fog_color=fog_color, fog_start=fog_start, fog_end=fog_end)
        self.shadow_renderer.render(self.camera, visible_keys)
        self.door_renderer.render(self.camera, visible_keys)
        self.stairs_renderer.render(self.camera, visible_keys)
        self.fence_renderer.render(self.camera, visible_keys)

        # Other players before the water, after the terrain: they are opaque and
        # they can be standing in a lake, so they have to be in the depth buffer
        # before the thing that has to blend against what is behind it.
        if self.client is not None:
            self.player_renderer.render(self.camera, render_distance_chunks=render_distance,
                                        fog_color=fog_color, fog_start=fog_start, fog_end=fog_end)

        # Water last of the world geometry: it's blended and writes no depth, so
        # everything it is meant to be seen THROUGH has to already be in the
        # frame and in the depth buffer.
        self.water_renderer.render(self.camera, visible_keys, fog_color, fog_start, fog_end, underwater)

        if self.controls.targeted_block is not None:
            t = self.controls.targeted_block
            self.outline_renderer.render(self.camera, t["x"], t["y"], t["z"])

        self.particles.render(self.camera)

        held_stack = self.inventory.selected_stack()
        held_id = held_stack["id"] if held_stack else None
        self.hand_renderer.render(self.window.width / self.window.height, held_id)

        w, h = self.window.width, self.window.height
        fps = self.window.get_fps()
        draw_hud(self.ui.surface, w, h, self.player, self.inventory, self.texture_atlas,
                 self.controls, fps, self.settings["show_fps"])

        if self.inventory_open:
            self.inventory_screen.draw(self.ui.surface, w, h, self.inventory, self.player.game_mode)
        elif self.crafting_table_open:
            self.crafting_table_screen.draw(self.ui.surface, w, h, self.inventory)
        elif self.showing_settings_in_game:
            self.in_game_settings_screen.draw(self.ui.surface, w, h)
        elif self.paused:
            self.pause_screen.draw(self.ui.surface, w, h)
        if self.player_dead:
            self.death_screen.draw(self.ui.surface, w, h)


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
