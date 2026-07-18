"""
ui/hud.py
In-game heads-up display: crosshair, hotbar with selected-slot highlight,
health hearts (survival only), break progress bar, and an optional debug
info panel (position, FPS). All drawn each frame onto the shared UI
surface via ui_renderer.py.
"""

import math

import pygame
import config
from ui.widgets import draw_text, draw_inset_panel, ItemSlot
from ui.icon_cache import get_item_icon
from world.blocks import get_stack_size

HOTBAR_SLOT_SIZE = 44
HOTBAR_SLOT_MARGIN = 3


def draw_crosshair(surface, width, height):
    cx, cy = width // 2, height // 2
    size = 10
    color = (255, 255, 255)
    pygame.draw.line(surface, color, (cx - size, cy), (cx + size, cy), 2)
    pygame.draw.line(surface, color, (cx, cy - size), (cx, cy + size), 2)


def draw_hotbar(surface, width, height, inventory, texture_atlas):
    total_w = config.HOTBAR_SIZE * (HOTBAR_SLOT_SIZE + HOTBAR_SLOT_MARGIN * 2)
    x0 = width // 2 - total_w // 2
    y0 = height - HOTBAR_SLOT_SIZE - 16

    for i in range(config.HOTBAR_SIZE):
        slot_x = x0 + i * (HOTBAR_SLOT_SIZE + HOTBAR_SLOT_MARGIN * 2) + HOTBAR_SLOT_MARGIN
        rect = pygame.Rect(slot_x, y0, HOTBAR_SLOT_SIZE, HOTBAR_SLOT_SIZE)
        slot = ItemSlot(rect)
        slot.highlighted = (i == inventory.selected_slot)

        stack = inventory.get_slot(i)
        icon = None
        count = None
        if stack is not None:
            icon = get_item_icon(texture_atlas, stack["id"], size=32)
            count = stack["count"]
        slot.draw(surface, icon_surface=icon, count=count)

        num_color = (255, 255, 255) if slot.highlighted else (210, 210, 210)
        draw_text(surface, str((i + 1) % 10), (slot_x + 3, y0 + 1), size=11, color=num_color)


HEART_FULL_COLOR = (176, 20, 20)
HEART_EMPTY_COLOR = (60, 60, 60)
BUBBLE_COLOR = (86, 172, 232)
BUBBLE_HIGHLIGHT = (226, 244, 255)
BUBBLE_OUTLINE = (18, 40, 66)


def _status_row_geometry(width, height):
    """
    The strip just above the hotbar that hearts and bubbles share, returned as
    (left_x, right_x, y).

    Both rows are anchored to the HOTBAR's edges rather than to the screen
    centre - hearts left, bubbles right, exactly as vanilla lays them out.
    Centring both (which is what the hearts used to do) puts them on top of each
    other the moment a second row exists.
    """
    hotbar_w = config.HOTBAR_SIZE * (HOTBAR_SLOT_SIZE + HOTBAR_SLOT_MARGIN * 2)
    left = width // 2 - hotbar_w // 2 + HOTBAR_SLOT_MARGIN
    right = width // 2 + hotbar_w // 2 - HOTBAR_SLOT_MARGIN
    y = height - HOTBAR_SLOT_SIZE - 16 - 16 - 6
    return left, right, y


def draw_health(surface, width, height, player):
    if player.game_mode != "survival":
        return
    heart_size = 16
    gap = 2
    x0, _right, y0 = _status_row_geometry(width, height)

    half_hearts = player.health
    for i in range(10):
        heart_value = half_hearts - i * 2
        cx = x0 + i * (heart_size + gap) + heart_size // 2
        cy = y0 + heart_size // 2
        color = HEART_FULL_COLOR if heart_value >= 2 else HEART_EMPTY_COLOR
        _draw_heart_shape(surface, (cx, cy), heart_size, color)
        if heart_value == 1:
            # half heart: draw the empty heart then overlay the left half in red
            _draw_heart_half(surface, (cx, cy), heart_size, HEART_FULL_COLOR)


def draw_air(surface, width, height, player):
    """
    The breath row. Hidden entirely while the player has full air - vanilla only
    shows bubbles once you're actually underwater and losing them, so a
    permanently visible row of ten bubbles would read as a bug.
    """
    if player.game_mode != "survival":
        return
    fraction = player.air_fraction()
    if fraction >= 1.0:
        return

    bubble_size = 16
    gap = 2
    _left, right, y0 = _status_row_geometry(width, height)
    total_w = 10 * (bubble_size + gap) - gap
    x0 = right - total_w

    # Round UP, so the last bubble only disappears when the air is genuinely at
    # zero and the row emptying lines up exactly with the damage starting.
    full = int(math.ceil(fraction * 10))
    for i in range(10):
        if i >= full:
            continue
        cx = x0 + i * (bubble_size + gap) + bubble_size // 2
        cy = y0 + bubble_size // 2
        radius = bubble_size // 2 - 1
        pygame.draw.circle(surface, BUBBLE_OUTLINE, (cx, cy), radius)
        pygame.draw.circle(surface, BUBBLE_COLOR, (cx, cy), radius - 1)
        pygame.draw.circle(surface, BUBBLE_HIGHLIGHT, (cx - radius // 3, cy - radius // 3),
                           max(1, radius // 3))


def _draw_heart_shape(surface, center, size, color):
    cx, cy = center
    r = size // 4
    pygame.draw.circle(surface, color, (cx - r, cy - r), r)
    pygame.draw.circle(surface, color, (cx + r, cy - r), r)
    points = [(cx - size // 2, cy - r // 2), (cx, cy + size // 2), (cx + size // 2, cy - r // 2)]
    pygame.draw.polygon(surface, color, points)


def _draw_heart_half(surface, center, size, color):
    cx, cy = center
    clip_rect = pygame.Rect(cx - size // 2, cy - size // 2, size // 2, size)
    prev_clip = surface.get_clip()
    surface.set_clip(clip_rect)
    _draw_heart_shape(surface, center, size, color)
    surface.set_clip(prev_clip)


def draw_break_progress(surface, width, height, progress: float):
    if progress <= 0:
        return
    bar_w, bar_h = 60, 6
    x0 = width // 2 - bar_w // 2
    y0 = height // 2 + 20
    draw_inset_panel(surface, (x0, y0, bar_w, bar_h), base_color=(30, 30, 30))
    fill_w = int(bar_w * min(1.0, progress))
    if fill_w > 0:
        pygame.draw.rect(surface, (220, 220, 220), (x0, y0, fill_w, bar_h))


def draw_debug_info(surface, player, fps: float, show_fps: bool):
    lines = [
        "Minecraft Classic",
        f"Mode: {player.game_mode}",
        f"XYZ: {player.physics.x:.2f} / {player.physics.y:.2f} / {player.physics.z:.2f}",
        f"Ground: {player.physics.on_ground}  Flying: {player.physics.flying}",
        f"Water: {player.physics.in_water}  Submerged: {player.physics.head_in_water}"
        f"  Air: {player.air:.1f}s",
    ]
    if show_fps:
        lines.append(f"{fps:.0f} FPS")

    y = 10
    for line in lines:
        draw_text(surface, line, (10, y), size=14)
        y += 18


def draw_underwater_overlay(surface, width, height):
    """
    Flat blue wash over the whole frame while submerged.

    Deliberately drawn here on the UI surface rather than in the world shaders:
    the UI layer is already composited over the finished 3D frame, so one fill
    tints EVERYTHING - terrain, water, the held item, particles - without every
    shader in the project needing an underwater uniform. Paired with the short
    blue fog the renderers do get (config.UNDERWATER_FOG_*), which is what
    supplies the depth cue this flat tint can't.
    """
    tint = pygame.Surface((width, height), pygame.SRCALPHA)
    tint.fill(config.UNDERWATER_TINT)
    surface.blit(tint, (0, 0))


def draw_hud(surface, width, height, player, inventory, texture_atlas, controls,
             fps: float, show_fps: bool):
    if player.physics.head_in_water:
        draw_underwater_overlay(surface, width, height)
    draw_crosshair(surface, width, height)
    draw_hotbar(surface, width, height, inventory, texture_atlas)
    draw_health(surface, width, height, player)
    draw_air(surface, width, height, player)
    if controls.is_breaking:
        draw_break_progress(surface, width, height, controls.break_progress)
    draw_debug_info(surface, player, fps, show_fps)
