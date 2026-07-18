"""
core/camera.py
First-person camera: tracks position/yaw/pitch and builds the view and
projection matrices the renderer needs every frame. All matrices are plain
numpy float32 arrays in column-major layout (matching OpenGL's expectation),
so they can be handed straight to glUniformMatrix4fv without transposing.

Convention: yaw=0, pitch=0 looks down -Z (matches the old Three.js build, and
is the standard OpenGL camera-forward convention), with +Y up and +X right.
"""

import math
import numpy as np

import config


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    # numpy array above is in row-major "math" layout; OpenGL wants column-major
    # when read as a flat buffer, so we transpose before flattening/uploading.
    return m.T.astype(np.float32)


def _look_at_from_yaw_pitch(position, yaw, pitch) -> np.ndarray:
    forward = _forward_vector(yaw, pitch)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    f = forward / np.linalg.norm(forward)
    s = np.cross(f, up)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-8:
        # looking straight up/down: pick an arbitrary stable right vector
        s = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        s = s / s_norm
    u = np.cross(s, f)

    m = np.identity(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(s, position)
    m[1, 3] = -np.dot(u, position)
    m[2, 3] = np.dot(f, position)
    # same row-major -> column-major transpose as the projection matrix
    return m.T.astype(np.float32)


def _forward_vector(yaw: float, pitch: float) -> np.ndarray:
    """Matches the convention used throughout the project: yaw=0 looks down -Z."""
    x = -math.sin(yaw) * math.cos(pitch)
    y = math.sin(pitch)
    z = -math.cos(yaw) * math.cos(pitch)
    return np.array([x, y, z], dtype=np.float32)


class Camera:
    # Defaults are pulled from config rather than hardcoded so the near/far
    # planes track FAR_PLANE automatically. They previously mirrored config's
    # values as literals (70 / 0.05 / 400), which silently went stale the
    # moment FAR_PLANE was raised for a longer render distance - and a far
    # plane shorter than the render distance clips the outer chunks away
    # before the fog has hidden them, ending the world in a hard edge.
    def __init__(self, position=(0.0, 0.0, 0.0), yaw: float = 0.0, pitch: float = 0.0,
                 fov: float = config.FOV_DEFAULT, aspect: float = 16 / 9,
                 near: float = config.NEAR_PLANE, far: float = config.FAR_PLANE):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

        # pitch is clamped just short of the poles to avoid the look-at basis
        # degenerating (gimbal-style flip) when looking perfectly up/down.
        self._pitch_limit = math.pi / 2 - 0.001

    def set_pitch_yaw(self, yaw: float, pitch: float):
        self.yaw = yaw
        self.pitch = max(-self._pitch_limit, min(self._pitch_limit, pitch))

    def forward(self) -> np.ndarray:
        return _forward_vector(self.yaw, self.pitch)

    def right(self) -> np.ndarray:
        f = _forward_vector(self.yaw, 0.0)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = np.cross(f, up)
        return r / np.linalg.norm(r)

    def view_matrix(self) -> np.ndarray:
        return _look_at_from_yaw_pitch(self.position, self.yaw, self.pitch)

    def projection_matrix(self) -> np.ndarray:
        return _perspective(self.fov, self.aspect, self.near, self.far)

    def set_aspect(self, width: int, height: int):
        self.aspect = width / max(1, height)
