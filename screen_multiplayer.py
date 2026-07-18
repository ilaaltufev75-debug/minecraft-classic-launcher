"""
ui/screen_multiplayer.py
Direct Connect: an address, a name, and a Connect button.

No server list and no LAN discovery, deliberately. Discovery works by
broadcasting on the local subnet, and the whole point of this build is two
people on a Radmin VPN who are not on the same subnet and never will be -
the one case where discovery cannot help is the case it exists for here.
Vanilla's Direct Connect box is the right shape, and typing an address is
something the friend on the other end has to do exactly once.

The address is parsed by net.protocol.parse_address, which raises ValueError
with text already written to be shown to a player - so its message goes on
screen verbatim rather than being re-worded here. Same for the connection
errors main.py feeds back through set_error(): net.client._friendly_connect_error
has already turned the errno into the three things that actually go wrong on a
Radmin LAN, each with a different fix. Restating either of them here would mean
two places to keep honest.

Settings are written into the shared dict, not saved - same contract as
ui/screen_settings.py, where main.py owns the write to disk.
"""

import pygame

from net.protocol import parse_address
from ui.widgets import Button, TextField, draw_text

# The server truncates a username to this (GameServer._handle_hello), so the
# field refuses to take more. A player who types 20 characters and joins as 16
# finds out from the chat line announcing them.
MAX_USERNAME_LENGTH = 16

# Long enough for "255.255.255.255:65535" with room to spare, short enough that
# a stuck key cannot fill the field with a megabyte of text.
MAX_ADDRESS_LENGTH = 48


class MultiplayerScreen:
    def __init__(self, settings: dict, on_connect, on_back):
        """`on_connect(host, port, username)` is only ever called with an
        address that already parsed - the screen never hands main.py a string to
        re-validate."""
        self.settings = settings
        self.on_connect = on_connect
        self.on_back = on_back

        self.address_field = None
        self.name_field = None
        self.connect_button = None
        self.back_button = None

        self.error = None
        # Set by main.py around the connect attempt. The button goes dead while
        # it is up, because the connect can take seconds and a player who does
        # not see it acknowledged will click it again - which on a working
        # server is a second connection, and on a broken one is a second wait.
        self.connecting = False
        self.status = None

    def layout(self, width, height):
        cx = width // 2
        self.address_field = TextField(
            (cx - 200, 160, 400, 38),
            initial_text=str(self.settings.get("last_server_address", "")),
            placeholder="26.x.x.x:port  - from the host's pause menu",
            max_length=MAX_ADDRESS_LENGTH,
        )
        self.name_field = TextField(
            (cx - 200, 250, 400, 38),
            initial_text=str(self.settings.get("username", "")),
            placeholder="Your name",
            max_length=MAX_USERNAME_LENGTH,
        )
        self.connect_button = Button((cx - 160, 320, 320, 46), "Connect")
        self.back_button = Button((20, 20, 100, 36), "Back", font_size=14)

    # -- state pushed in by main.py -------------------------------------------

    def set_error(self, message):
        """Shows a failure and re-enables the button. `message` may contain
        newlines - net.client's connect errors are two lines by design, the
        second one being what to actually do about it."""
        self.error = message
        self.connecting = False
        self.status = None

    def set_connecting(self, host, port):
        self.error = None
        self.connecting = True
        self.status = f"Connecting to {host}:{port}..."

    # -- input ----------------------------------------------------------------

    def handle_event(self, event, width, height):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            self.address_field.handle_click(mouse_pos)
            self.name_field.handle_click(mouse_pos)
            if self.back_button.rect.collidepoint(mouse_pos):
                self.on_back()
                return
            if self.connect_button.rect.collidepoint(mouse_pos):
                self._submit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.address_field.handle_backspace()
                self.name_field.handle_backspace()
            elif event.key == pygame.K_RETURN:
                self._submit()
            elif event.key == pygame.K_ESCAPE:
                self.on_back()
            elif event.key == pygame.K_TAB:
                if self.address_field.focused:
                    self.address_field.focused = False
                    self.name_field.focused = True
                else:
                    self.name_field.focused = False
                    self.address_field.focused = True
        elif event.type == pygame.TEXTINPUT:
            self.address_field.handle_text_input(event.text)
            self.name_field.handle_text_input(event.text)

    def _submit(self):
        if self.connecting:
            return

        username = self.name_field.text.strip()
        if not username:
            self.error = "Enter a name"
            return

        address_text = self.address_field.text.strip()
        if ":" not in address_text:
            # parse_address would happily default this to DEFAULT_PORT, and that
            # default is now a trap rather than a convenience: a host's port is
            # picked fresh by the kernel every time they open their world, so a
            # bare IP is not "the usual port", it is a missing number. Better to
            # say which number is missing than to time out on 25565 and leave
            # them wondering whether the VPN is down.
            self.error = "Add the port too, e.g. 26.31.44.10:54321\nThe host's pause menu shows theirs."
            return

        try:
            host, port = parse_address(address_text)
        except ValueError as exc:
            self.error = str(exc)
            return

        # Remembered before the attempt, not after it succeeds. An address that
        # failed is still the address they meant to type, and having to retype
        # it to try again after a VPN hiccup is the worst possible moment to
        # make someone retype anything.
        self.settings["username"] = username
        self.settings["last_server_address"] = address_text

        self.set_connecting(host, port)
        self.on_connect(host, port, username)

    def update_hover(self, mouse_pos):
        self.connect_button.enabled = not self.connecting
        self.connect_button.update_hover(mouse_pos)
        self.back_button.update_hover(mouse_pos)

    # -- drawing --------------------------------------------------------------

    def draw(self, surface, width, height):
        surface.fill((50, 40, 30))
        cx = width // 2

        draw_text(surface, "Multiplayer", (cx, 60), size=30, center=True)
        draw_text(surface, "Join a friend's world", (cx, 95), size=13, center=True, color=(180, 180, 180))

        draw_text(surface, "Server Address", (cx - 200, 138), size=13, color=(200, 200, 200))
        self.address_field.draw(surface)
        draw_text(surface, "The host's port changes every time they open their world",
                  (cx - 200, 202), size=11, color=(150, 150, 150), shadow=False)

        draw_text(surface, "Your Name", (cx - 200, 228), size=13, color=(200, 200, 200))
        self.name_field.draw(surface)

        self.connect_button.enabled = not self.connecting
        self.connect_button.draw(surface)
        self.back_button.draw(surface)

        if self.status:
            draw_text(surface, self.status, (cx, 390), size=15, center=True, color=(220, 220, 160))
        if self.error:
            self._draw_error(surface, cx, 390 if not self.status else 420)

    def _draw_error(self, surface, cx, top):
        # Split rather than blit: parse_address's and _friendly_connect_error's
        # messages are deliberately multi-line, and pygame's font renderer draws
        # a newline as a glyph box rather than a line break.
        for i, line in enumerate(str(self.error).split("\n")):
            draw_text(surface, line, (cx, top + i * 22), size=14, center=True, color=(230, 120, 120))
