"""
ui/screen_pause.py
In-game pause menu: resume, open to LAN, options, quit to title. Drawn as a
translucent overlay on top of the (frozen) game view.

WHY "OPEN TO LAN" LIVES HERE AND NOWHERE ELSE
---------------------------------------------
Because opening a world is something you do TO a world, and the title screen
does not have one. Vanilla puts it in exactly this place for exactly this
reason. The alternative - a "host a world" button on the title screen - reads
fine right up until you notice it would have to ask which world, which game
mode and which port before it could do anything, i.e. it would have to
re-implement the world list to avoid being a second way of doing the same
thing.

The button is only offered where opening can actually happen: a singleplayer
session (net.MODE_SINGLEPLAYER). A guest (MODE_CLIENT) is standing in someone
else's world and has nothing to open, and a host (MODE_HOST) has already opened
theirs - both get a status line in the same slot instead. The slot is reserved
either way rather than collapsed, so opening a world does not make the Quit
button jump out from under a moving cursor.
"""

import pygame

from net import MODE_CLIENT, MODE_HOST, MODE_SINGLEPLAYER
from net.protocol import DEFAULT_PORT
from ui.widgets import Button, draw_text


class PauseScreen:
    def __init__(self, on_resume, on_options, on_quit_to_title, on_open_to_lan=None):
        self.on_resume = on_resume
        self.on_options = on_options
        self.on_quit_to_title = on_quit_to_title
        self.on_open_to_lan = on_open_to_lan

        self.resume_button = None
        self.open_button = None
        self.options_button = None
        self.quit_button = None

        # Pushed in by main.py via set_network_state. Defaults describe a plain
        # offline world, which is what this screen opens over until told
        # otherwise.
        self.mode = MODE_SINGLEPLAYER
        self.port = DEFAULT_PORT
        self.address = None
        self.player_count = 0
        self.error = None

    def layout(self, width, height):
        cx = width // 2
        y = height // 2 - 100
        self.resume_button = Button((cx - 140, y, 280, 44), "Back to Game")
        self.open_button = Button((cx - 140, y + 56, 280, 44), "Open to LAN")
        self.options_button = Button((cx - 140, y + 112, 280, 44), "Options...")
        self.quit_button = Button((cx - 140, y + 168, 280, 44), "Quit to Title")

    # -- state pushed in by main.py -------------------------------------------

    def set_network_state(self, mode, port=None, player_count=0, error=None, address=None):
        """
        `error` is GameServer.error verbatim - it already names the port and the
        OS's reason (see GameServer.start), which is the entire diagnosis.

        `address` is what net.server.find_lan_address found, or None if this
        machine has no Radmin address to offer.
        """
        self.mode = mode
        if port is not None:
            self.port = port
        self.address = address
        self.player_count = player_count
        self.error = error

    @property
    def _can_open(self):
        return self.mode == MODE_SINGLEPLAYER and self.on_open_to_lan is not None

    # -- input ----------------------------------------------------------------

    def handle_event(self, event, width, height):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            if self.resume_button.rect.collidepoint(mouse_pos):
                self.on_resume()
            elif self._can_open and self.open_button.rect.collidepoint(mouse_pos):
                self.on_open_to_lan()
            elif self.options_button.rect.collidepoint(mouse_pos):
                self.on_options()
            elif self.quit_button.rect.collidepoint(mouse_pos):
                self.on_quit_to_title()

    def update_hover(self, mouse_pos):
        self.resume_button.update_hover(mouse_pos)
        if self._can_open:
            self.open_button.update_hover(mouse_pos)
        self.options_button.update_hover(mouse_pos)
        self.quit_button.update_hover(mouse_pos)

    # -- drawing --------------------------------------------------------------

    def draw(self, surface, width, height):
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        surface.blit(overlay, (0, 0))

        draw_text(surface, "Game Paused", (width // 2, height // 2 - 140), size=26, center=True)

        self.resume_button.draw(surface)
        if self._can_open:
            self.open_button.draw(surface)
        else:
            self._draw_network_status(surface, width)
        self.options_button.draw(surface)
        self.quit_button.draw(surface)

        if self.error:
            draw_text(surface, self.error, (width // 2, self.quit_button.rect.bottom + 24),
                      size=13, center=True, color=(230, 120, 120))

    def _draw_network_status(self, surface, width):
        """Fills the Open-to-LAN slot for the two sessions that cannot use it."""
        cx = width // 2
        slot = self.open_button.rect

        if self.mode == MODE_HOST:
            # The port is different every time this world is opened (the kernel
            # picks a free one - see GameServer.start), so it is not something
            # the host can be expected to know or remember. It has to be ON
            # SCREEN, in the form their friend types, or the feature does not
            # work at all.
            if self.address:
                draw_text(surface, f"{self.address}:{self.port}",
                          (cx, slot.centery - 10), size=20, center=True, color=(160, 230, 160))
                hint = "Tell your friend this - it's in the Multiplayer screen"
            else:
                draw_text(surface, f"Open to LAN on port {self.port}",
                          (cx, slot.centery - 10), size=18, center=True, color=(160, 230, 160))
                # No 26.x.x.x on this machine: either Radmin is off, or it is on
                # and not connected. Saying so beats printing a LAN address the
                # friend cannot reach.
                hint = "Radmin not detected - check it's connected, then reopen"
            draw_text(surface, hint, (cx, slot.centery + 12), size=11, center=True,
                      color=(190, 190, 190), shadow=False)

            if self.player_count > 1:
                # Minus one: the host's own loopback client is in that count.
                # It is a real client to the server and deliberately so (see
                # net/__init__.py) - but "1 player connected" while standing
                # alone in a world would read as a bug rather than as the
                # architecture being honest.
                others = self.player_count - 1
                label = f"{others} player connected" if others == 1 else f"{others} players connected"
            else:
                label = "Waiting for someone to join"
            draw_text(surface, label, (cx, slot.centery + 28), size=11, center=True,
                      color=(150, 150, 150), shadow=False)
        elif self.mode == MODE_CLIENT:
            draw_text(surface, "Connected to a friend's world",
                      (cx, slot.centery), size=14, center=True, color=(190, 190, 190))
