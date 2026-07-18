"""
ui/widgets.py
Minecraft-styled 2D widget drawing helpers, all operating on a plain
pygame.Surface (see ui_renderer.py for how that surface reaches the
screen). The look: chunky beveled buttons (light top/left edge, dark
bottom/right edge - the classic pre-flat-design GUI bevel Minecraft used),
stone-gray palette, drop-shadowed pixel text, and slot grids for inventory/
hotbar rendering. Nothing here does any OpenGL directly - purely 2D pygame
drawing, kept deliberately simple and readable over pixel-perfect asset
recreation.
"""

import pygame
import config

_font_cache = {}


def get_font(size: int, bold: bool = True):
    key = (size, bold)
    if key not in _font_cache:
        # A monospace font reads closest to Minecraft's blocky in-game font
        # without needing to ship/parse an actual bitmap font asset.
        f = pygame.font.SysFont("dejavusansmono,couriernew,monospace", size, bold=bold)
        _font_cache[key] = f
    return _font_cache[key]


def draw_text(surface, text: str, pos, size=16, color=config.COLOR_UI_TEXT,
              shadow=True, center=False, bold=True):
    font = get_font(size, bold)
    x, y = pos
    if shadow:
        shadow_surf = font.render(text, True, config.COLOR_UI_TEXT_SHADOW)
        sx, sy = (x + 2, y + 2)
        if center:
            rect = shadow_surf.get_rect(center=(sx, sy))
            surface.blit(shadow_surf, rect)
        else:
            surface.blit(shadow_surf, (sx, sy))

    text_surf = font.render(text, True, color)
    if center:
        rect = text_surf.get_rect(center=(x, y))
        surface.blit(text_surf, rect)
        return rect
    else:
        surface.blit(text_surf, (x, y))
        return pygame.Rect(x, y, *text_surf.get_size())


def draw_beveled_panel(surface, rect, base_color=config.COLOR_UI_STONE_DARK,
                        light=config.COLOR_UI_STONE_LIGHT, dark=config.COLOR_UI_STONE_SHADOW,
                        bevel=3):
    """Classic inset/outset bevel: light edge top-left, dark edge bottom-right,
    matching the chunky GUI style Minecraft used for panels and slots."""
    x, y, w, h = rect
    pygame.draw.rect(surface, base_color, rect)
    # top & left highlight
    pygame.draw.rect(surface, light, (x, y, w, bevel))
    pygame.draw.rect(surface, light, (x, y, bevel, h))
    # bottom & right shadow
    pygame.draw.rect(surface, dark, (x, y + h - bevel, w, bevel))
    pygame.draw.rect(surface, dark, (x + w - bevel, y, bevel, h))


def draw_inset_panel(surface, rect, base_color=config.COLOR_UI_STONE_DARK,
                      light=config.COLOR_UI_STONE_LIGHT, dark=config.COLOR_UI_STONE_SHADOW,
                      bevel=2):
    """Inverted bevel (dark top-left, light bottom-right) for slots and
    pressed-in surfaces, as opposed to raised buttons."""
    x, y, w, h = rect
    pygame.draw.rect(surface, base_color, rect)
    pygame.draw.rect(surface, dark, (x, y, w, bevel))
    pygame.draw.rect(surface, dark, (x, y, bevel, h))
    pygame.draw.rect(surface, light, (x, y + h - bevel, w, bevel))
    pygame.draw.rect(surface, light, (x + w - bevel, y, bevel, h))


class Button:
    def __init__(self, rect, label, on_click=None, font_size=18):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.font_size = font_size
        self.hovered = False
        self.enabled = True

    def update_hover(self, mouse_pos):
        self.hovered = self.enabled and self.rect.collidepoint(mouse_pos)

    def handle_click(self, mouse_pos) -> bool:
        if self.enabled and self.rect.collidepoint(mouse_pos):
            if self.on_click:
                self.on_click()
            return True
        return False

    def draw(self, surface):
        color = (143, 143, 100) if self.hovered else config.COLOR_UI_STONE_DARK
        if not self.enabled:
            color = (90, 90, 90)
        draw_beveled_panel(surface, self.rect, base_color=color)
        text_color = config.COLOR_UI_TEXT_HOVER if self.hovered else config.COLOR_UI_TEXT
        if not self.enabled:
            text_color = (150, 150, 150)
        draw_text(surface, self.label, self.rect.center, size=self.font_size,
                  color=text_color, center=True)


class Slider:
    def __init__(self, rect, label_fn, value, min_value, max_value, step, on_change=None):
        self.rect = pygame.Rect(rect)
        self.label_fn = label_fn  # callable(value) -> display string
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.on_change = on_change
        self.dragging = False

    def _value_to_x(self):
        t = (self.value - self.min_value) / (self.max_value - self.min_value)
        return self.rect.x + int(t * self.rect.width)

    def _x_to_value(self, x):
        t = max(0.0, min(1.0, (x - self.rect.x) / self.rect.width))
        raw = self.min_value + t * (self.max_value - self.min_value)
        stepped = round(raw / self.step) * self.step
        return max(self.min_value, min(self.max_value, stepped))

    def handle_mouse_down(self, mouse_pos):
        if self.rect.collidepoint(mouse_pos):
            self.dragging = True
            self._update_from_mouse(mouse_pos)

    def handle_mouse_up(self):
        self.dragging = False

    def handle_mouse_motion(self, mouse_pos):
        if self.dragging:
            self._update_from_mouse(mouse_pos)

    def _update_from_mouse(self, mouse_pos):
        new_value = self._x_to_value(mouse_pos[0])
        if new_value != self.value:
            self.value = new_value
            if self.on_change:
                self.on_change(self.value)

    def draw(self, surface):
        draw_inset_panel(surface, self.rect, base_color=(80, 80, 80))
        handle_x = self._value_to_x()
        handle_rect = pygame.Rect(handle_x - 4, self.rect.y - 2, 8, self.rect.height + 4)
        draw_beveled_panel(surface, handle_rect, base_color=config.COLOR_UI_STONE_LIGHT)
        label = self.label_fn(self.value)
        draw_text(surface, label, self.rect.center, size=14, center=True)


class TextField:
    def __init__(self, rect, initial_text="", placeholder="", max_length=None):
        self.rect = pygame.Rect(rect)
        self.text = initial_text
        self.placeholder = placeholder
        # None means unbounded, which is what every existing caller gets. A cap
        # matters where the far end has one of its own: the server truncates a
        # username to 16 characters (see GameServer._handle_hello), so a field
        # that lets someone type 30 shows them a name they will never be called
        # by, and they find that out from a chat line after they have joined.
        self.max_length = max_length
        self.focused = False

    def handle_click(self, mouse_pos):
        self.focused = self.rect.collidepoint(mouse_pos)

    def handle_text_input(self, text: str):
        if not self.focused:
            return
        if self.max_length is not None:
            room = self.max_length - len(self.text)
            if room <= 0:
                return
            text = text[:room]
        self.text += text

    def handle_backspace(self):
        if self.focused and self.text:
            self.text = self.text[:-1]

    def draw(self, surface):
        border_color = config.COLOR_UI_TEXT_HOVER if self.focused else config.COLOR_UI_STONE_LIGHT
        draw_inset_panel(surface, self.rect, base_color=(20, 20, 20), light=border_color)
        display = self.text if self.text else self.placeholder
        color = config.COLOR_UI_TEXT if self.text else (140, 140, 140)
        draw_text(surface, display, (self.rect.x + 8, self.rect.y + self.rect.height // 2),
                  size=16, color=color, center=False, shadow=False)
        # simple blinking-less caret at the end of the text when focused
        if self.focused:
            font = get_font(16)
            text_w = font.size(self.text)[0]
            caret_x = self.rect.x + 8 + text_w + 2
            pygame.draw.line(surface, config.COLOR_UI_TEXT,
                              (caret_x, self.rect.y + 4), (caret_x, self.rect.y + self.rect.height - 4), 2)


class ItemSlot:
    """Draws a single inventory/hotbar slot with an icon (from an atlas tile
    rendered to a small pygame surface once and cached) and stack count."""

    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.highlighted = False

    def draw(self, surface, icon_surface=None, count=None):
        base = (255, 255, 255) if self.highlighted else config.COLOR_UI_STONE_DARK
        draw_inset_panel(surface, self.rect, base_color=(139, 139, 139))
        if self.highlighted:
            pygame.draw.rect(surface, (255, 255, 255), self.rect, width=2)

        if icon_surface is not None:
            icon_rect = icon_surface.get_rect(center=self.rect.center)
            surface.blit(icon_surface, icon_rect)

        if count is not None and count > 1:
            draw_text(surface, str(count), (self.rect.right - 6, self.rect.bottom - 4),
                      size=13, center=False)
