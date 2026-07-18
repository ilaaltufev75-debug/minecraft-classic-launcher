"""
ui/screen_loading.py
"Generating level" overlay shown while a world's spawn area streams in.
Matches the classic Minecraft loading screen: a dirt-textured background,
"Generating level" title, a status line ("Building terrain" / "Growing
trees" / etc.), and a progress bar that fills as chunks/trees finish
generating. Purely a 2D pygame overlay drawn on top of the (already-active)
3D scene - see main.py's STATE_LOADING_WORLD handling for how progress is
fed in frame by frame.
"""

import pygame
import config
from ui.widgets import draw_text, get_font

BG_COLOR = (48, 34, 22)          # dark dirt-brown backdrop
BG_COLOR_2 = (43, 30, 19)        # subtly darker tile for a woven texture look
BAR_WIDTH = 300
BAR_HEIGHT = 10
BAR_BORDER = (128, 128, 128)
BAR_FILL = (64, 210, 64)
BAR_EMPTY = (55, 55, 55)


class LoadingScreen:
    def __init__(self):
        self.progress = 0.0          # 0..1, smoothed
        self.target_progress = 0.0
        self.status_text = "Building terrain"
        self._tile_cache = {}

    def set_progress(self, fraction: float, status_text: str = None):
        self.target_progress = max(0.0, min(1.0, fraction))
        if status_text is not None:
            self.status_text = status_text

    def update(self, dt: float):
        # smooth toward target so the bar doesn't visibly jump between the
        # coarse per-frame generation steps
        diff = self.target_progress - self.progress
        self.progress += diff * min(1.0, dt * 10.0)
        if abs(self.target_progress - self.progress) < 0.002:
            self.progress = self.target_progress

    def is_done(self) -> bool:
        return self.progress >= 0.999 and self.target_progress >= 0.999

    def _get_tile(self):
        if "tile" not in self._tile_cache:
            tile = pygame.Surface((16, 16))
            tile.fill(BG_COLOR)
            # a few darker pixel speckles so it doesn't read as a flat block of color
            import random
            rng = random.Random(1337)
            for _ in range(14):
                x, y = rng.randrange(16), rng.randrange(16)
                tile.set_at((x, y), BG_COLOR_2)
            self._tile_cache["tile"] = tile
        return self._tile_cache["tile"]

    def draw(self, surface, width, height):
        tile = self._get_tile()
        for y in range(0, height, 16):
            for x in range(0, width, 16):
                surface.blit(tile, (x, y))

        cx, cy = width // 2, height // 2

        draw_text(surface, "Generating level", (cx, cy - 34), size=22, color=(255, 255, 255), center=True)

        bar_x = cx - BAR_WIDTH // 2
        bar_y = cy + 6
        draw_text(surface, self.status_text, (bar_x, bar_y - 22), size=14, color=(255, 255, 255), center=False)

        pygame.draw.rect(surface, BAR_EMPTY, (bar_x, bar_y, BAR_WIDTH, BAR_HEIGHT))
        fill_w = int(BAR_WIDTH * self.progress)
        if fill_w > 0:
            pygame.draw.rect(surface, BAR_FILL, (bar_x, bar_y, fill_w, BAR_HEIGHT))
        pygame.draw.rect(surface, BAR_BORDER, (bar_x, bar_y, BAR_WIDTH, BAR_HEIGHT), width=1)
