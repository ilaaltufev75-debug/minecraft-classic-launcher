"""
world/noise.py
Deterministic, seedable noise for infinite world generation, with both a
scalar API (simple, used for single-point queries like tree placement) and
a vectorized numpy API (used by worldgen.py to generate entire chunks in
one shot instead of looping block-by-block in pure Python, which profiling
showed was the dominant cost of chunk generation - about 25ms/chunk scalar
vs under 1ms vectorized for the noise evaluation itself).

Real Perlin/Simplex noise would be nicer, but pulling in an extra dependency
for it isn't worth it here; a few octaves of seeded sine/cosine plus a
hashed pseudo-random function for point features (ore, trees) is enough to
produce convincing rolling hills, mountains, and caves, and - crucially -
is 100% deterministic from (seed, x, z) so the same coordinates always
regenerate identically no matter which chunk loads first (required for a
real infinite/streaming world, unlike the old fixed-size build).
"""

import math
import numpy as np


class WorldNoise:
    def __init__(self, seed: int):
        # Fold the seed into several independent phase offsets so different
        # seeds produce visibly different terrain rather than just shifting
        # the same pattern by a constant.
        self.seed = seed
        self._phase_a = (seed * 12.9898) % 6.283185307
        self._phase_b = (seed * 78.233) % 6.283185307
        self._phase_c = (seed * 37.719) % 6.283185307
        self._phase_d = (seed * 4.1414) % 6.283185307
        self._phase_e = (seed * 91.7561) % 6.283185307

    # -- scalar hashed pseudo-random in [0, 1), stable per integer coordinate --
    def hash2(self, x: int, z: int) -> float:
        n = x * 374761393 + z * 668265263 + self.seed * 2147483647
        n = (n ^ (n >> 13)) * 1274126177
        n = n ^ (n >> 16)
        return (n & 0xFFFFFFFF) / 0xFFFFFFFF

    def hash3(self, x: int, y: int, z: int) -> float:
        n = x * 374761393 + y * 217645177 + z * 668265263 + self.seed * 2147483647
        n = (n ^ (n >> 13)) * 1274126177
        n = n ^ (n >> 16)
        return (n & 0xFFFFFFFF) / 0xFFFFFFFF

    # -- scalar smooth 2D terrain noise (rolling hills base shape) -----------
    def terrain2d(self, x: float, z: float) -> float:
        """Roughly in range [-4.5, 4.5]. Smooth, used for base hill shape."""
        a = self._phase_a
        b = self._phase_b
        return (
            math.sin(x * 0.045 + a) * 2.0
            + math.cos(z * 0.052 + b) * 2.0
            + math.sin((x + z) * 0.021 + a) * 1.2
            + math.cos((x - z) * 0.031 + b) * 0.8
        )

    # -- scalar coarser, sharper mask that decides where mountains cluster ---
    def mountain_mask(self, x: float, z: float) -> float:
        """Sharper peaks than terrain2d; roughly in range [-1.0, 1.3]."""
        c = self._phase_c
        d = self._phase_d
        n1 = math.sin(x * 0.011 + c) * math.cos(z * 0.0097 + d)
        n2 = math.sin((x + z) * 0.0053 + c) * 0.5
        return n1 + n2

    # -- scalar pseudo-3D noise for carving caves ----------------------------
    def cave3d(self, x: float, y: float, z: float) -> float:
        """Normalized roughly into [0, 1]. Used with a threshold to carve air pockets."""
        n = (
            math.sin(x * 0.16 + y * 0.29 + self._phase_a) * math.cos(z * 0.19 - y * 0.11 + self._phase_b)
            + math.sin((x + z) * 0.08 + y * 0.37 + self._phase_c) * 0.6
            + math.cos((x - z) * 0.12 - y * 0.21 + self._phase_d) * 0.5
        )
        return (n + 2.1) / 4.2

    # =========================================================================
    # Vectorized numpy equivalents (identical math, evaluated over whole arrays)
    # =========================================================================

    def terrain2d_grid(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        a, b = self._phase_a, self._phase_b
        return (
            np.sin(X * 0.045 + a) * 2.0
            + np.cos(Z * 0.052 + b) * 2.0
            + np.sin((X + Z) * 0.021 + a) * 1.2
            + np.cos((X - Z) * 0.031 + b) * 0.8
        )

    def mountain_mask_grid(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        c, d = self._phase_c, self._phase_d
        n1 = np.sin(X * 0.011 + c) * np.cos(Z * 0.0097 + d)
        n2 = np.sin((X + Z) * 0.0053 + c) * 0.5
        return n1 + n2

    def cave3d_grid(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        a, b, c, d = self._phase_a, self._phase_b, self._phase_c, self._phase_d
        n = (
            np.sin(X * 0.16 + Y * 0.29 + a) * np.cos(Z * 0.19 - Y * 0.11 + b)
            + np.sin((X + Z) * 0.08 + Y * 0.37 + c) * 0.6
            + np.cos((X - Z) * 0.12 - Y * 0.21 + d) * 0.5
        )
        return (n + 2.1) / 4.2

    def hash3_grid(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        seed = self.seed
        n = (X.astype(np.int64) * 374761393 + Y.astype(np.int64) * 217645177
             + Z.astype(np.int64) * 668265263 + seed * 2147483647)
        n = (n ^ (n >> 13)) * 1274126177
        n = n ^ (n >> 16)
        return (n & 0xFFFFFFFF).astype(np.float64) / 0xFFFFFFFF

    def hash2_grid(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        seed = self.seed
        n = X.astype(np.int64) * 374761393 + Z.astype(np.int64) * 668265263 + seed * 2147483647
        n = (n ^ (n >> 13)) * 1274126177
        n = n ^ (n >> 16)
        return (n & 0xFFFFFFFF).astype(np.float64) / 0xFFFFFFFF

    def hash2_salted_grid(self, X: np.ndarray, Z: np.ndarray, salt: int) -> np.ndarray:
        """
        hash2_grid with an extra integer salt, so several *independent*
        pseudo-random values can be drawn from the same (x, z). Mountain
        cells need this: one cell coordinate has to yield a centre offset,
        a radius, a height and a profile exponent that don't correlate with
        each other, which a single hash per coordinate can't give.
        """
        seed = self.seed
        n = (X.astype(np.int64) * 374761393
             + Z.astype(np.int64) * 668265263
             + np.int64(salt) * np.int64(1013904223)
             + seed * 2147483647)
        n = (n ^ (n >> 13)) * 1274126177
        n = n ^ (n >> 16)
        return (n & 0xFFFFFFFF).astype(np.float64) / 0xFFFFFFFF

    # =========================================================================
    # Additional layers used by the mountain/plains/cave generator
    # =========================================================================

    def detail2d_grid(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        High-frequency roughness, roughly in [-1.3, 1.3]. Layered on top of
        terrain2d_grid so hillsides get small bumps and dips instead of
        reading as one smooth mathematical surface.
        """
        a, b = self._phase_a, self._phase_b
        return (
            np.sin(X * 0.13 + b) * 0.5
            + np.cos(Z * 0.117 + a) * 0.5
            + np.sin((X - Z) * 0.083 + b) * 0.3
        )

    def plains_mask_grid(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Very low frequency mask in roughly [0, 1] marking where the world
        should flatten out into open plains. Frequency is tuned so a patch
        above PLAINS_THRESHOLD spans roughly 6-12 chunks: coarse enough to be
        worth building on, but well short of the ~43-chunk expanses the
        original frequency produced, which read as an endless featureless
        field rather than a clearing in a forest.
        """
        c, d = self._phase_c, self._phase_d
        n = (
            np.sin(X * 0.027 + c) * 0.6
            + np.cos(Z * 0.026 + d) * 0.6
            + np.sin((X - Z) * 0.015 + d) * 0.4
        )
        return (n + 1.6) / 3.2

    def warp_grid(self, X: np.ndarray, Z: np.ndarray, channel: int) -> np.ndarray:
        """
        Domain-warp offset in roughly [-1.6, 1.6], scaled by the caller.
        Mountain footprints are computed from a distance-to-centre test,
        which on its own produces perfect circles; warping the input
        coordinates first bends those circles into irregular, natural
        outlines with spurs and hollows. `channel` selects an independent
        offset field for the X and Z axes.
        """
        p = self._phase_a if channel == 0 else self._phase_b
        q = self._phase_c if channel == 0 else self._phase_d
        return (
            np.sin(X * 0.013 + p) * 0.6
            + np.cos(Z * 0.017 + q) * 0.6
            + np.sin((X + Z) * 0.0071 + p) * 0.4
        )

    def continent_grid(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Continentalness: the lowest-frequency field in the generator, in
        roughly [0, 1]. Decides where the world is land and where it is ocean
        floor; worldgen._column_fields ramps the terrain baseline between
        OCEAN_FLOOR_HEIGHT and BASE_TERRAIN_HEIGHT across it.

        Frequencies sit an order of magnitude below alpine_mask_grid's (0.0016
        vs 0.0043) on purpose. An ocean has to be big enough that its far shore
        is over the horizon - the slowest term here has a wavelength of ~7700
        blocks (~480 chunks), so a continent spans thousands of blocks rather
        than reading as a large lake. It also guarantees the two fields cannot
        lock together: if continents and mountain ranges peaked in the same
        places, every landmass would be a mountain and every ocean flat.

        Uses _phase_e, which nothing else touches, for the same reason
        alpine_mask_grid avoids plains_mask_grid's phases.

        MEASURED (see config.OCEAN_THRESHOLD): mean 0.498, and the cut placing
        the shoreline at 65% land is 0.4133, stable to +-0.003 across seeds.
        Do not nudge the frequencies without re-running the calibration - the
        distribution is bell-shaped, so a threshold that "looks" high on a 0..1
        field is not.
        """
        a, b, e = self._phase_a, self._phase_b, self._phase_e
        n = (
            np.sin(X * 0.0016 + e) * 0.6
            + np.cos(Z * 0.0014 + a) * 0.6
            + np.sin((X + Z) * 0.00081 + b) * 0.4
        )
        return (n + 1.6) / 3.2

    def alpine_mask_grid(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Extremely low frequency mask in roughly [0, 1] marking the rare
        alpine region - the only place the tall, steep massifs are allowed
        to spawn. Frequency is chosen so a region above ALPINE_THRESHOLD
        spans at most ~42 chunks; anything faster produced lone spires
        sticking out of flat forest, which reads as a bug rather than a
        mountain range.

        Deliberately built from different phase/frequency pairs than
        plains_mask_grid: if the two masks shared a field they would peak in
        the same places and the world would only ever offer "plain OR
        mountains" at any given spot, never a plain in one region and a
        range somewhere unrelated.
        """
        a, d = self._phase_a, self._phase_d
        n = (
            np.sin(X * 0.0043 + a) * 0.6
            + np.cos(Z * 0.0039 + d) * 0.6
            + np.sin((X + Z) * 0.0021 + d) * 0.4
        )
        return (n + 1.6) / 3.2

    def cavern3d_grid(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Low-frequency 3D noise in roughly [0, 1], carving large open
        chambers. cave3d_grid alone only ever produces tunnels of a single
        characteristic width because it has one frequency band; running a
        second, much coarser pass alongside it and unioning the results is
        what gives caves a mix of tight passages and rooms you can actually
        stand around in.
        """
        a, b, c = self._phase_a, self._phase_b, self._phase_c
        n = (
            np.sin(X * 0.048 + Y * 0.075 + a) * np.cos(Z * 0.053 - Y * 0.041 + b)
            + np.sin((X + Z) * 0.029 + Y * 0.061 + c) * 0.7
        )
        return (n + 1.7) / 3.4
