"""Camera model utilities for projecting 3D trajectory points into the monocular image plane."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np


def undistort_points(points: np.ndarray, distortion: Iterable[float]) -> np.ndarray:
    """Apply a simple radial-tangential undistortion model.

    The implementation accepts five distortion coefficients ``[k1, k2, p1, p2, k3]``.
    For this design example the default values are zero, but the helper keeps the
    interface consistent with real-world camera models.
    """
    k1, k2, p1, p2, k3 = distortion
    x = points[:, 0]
    y = points[:, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([x_distorted, y_distorted], axis=1)


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    image_width: int = 640
    image_height: int = 480

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points in the camera frame onto the image plane.

        Points behind the camera are filtered out by returning NaNs so that callers
        can drop invalid projections during mask filtering.
        """
        z = points_3d[:, 2]
        valid = z > 1e-6
        x_norm = points_3d[:, 0] / np.maximum(z, 1e-6)
        y_norm = points_3d[:, 1] / np.maximum(z, 1e-6)
        normalized = np.stack([x_norm, y_norm], axis=1)
        undistorted = undistort_points(normalized, self.distortion)
        u = undistorted[:, 0] * self.fx + self.cx
        v = undistorted[:, 1] * self.fy + self.cy
        pixels = np.stack([u, v], axis=1)
        pixels[~valid] = np.nan
        return pixels

    def inside_image(self, uv: np.ndarray) -> np.ndarray:
        """Return a boolean mask indicating which ``(u, v)`` coordinates fall inside the image."""
        in_x = (uv[:, 0] >= 0) & (uv[:, 0] < self.image_width)
        in_y = (uv[:, 1] >= 0) & (uv[:, 1] < self.image_height)
        return in_x & in_y
