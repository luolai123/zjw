"""Generate, evaluate, and adjust motion primitives."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import numpy as np
from .anchors import AnchorPrimitive
from ..camera import CameraIntrinsics


def quintic_polynomial(a: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate quintic polynomial given coefficients ``a = [a0..a5]``."""
    powers = np.stack([np.ones_like(t), t, t**2, t**3, t**4, t**5], axis=-1)
    return powers @ a


def sample_trajectory(coeffs: np.ndarray, duration: float, num_samples: int = 50) -> np.ndarray:
    t = np.linspace(0.0, duration, num_samples)
    return quintic_polynomial(coeffs, t)


@dataclass
class Trajectory3D:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    duration: float

    def stack(self) -> np.ndarray:
        return np.stack([self.x, self.y, self.z], axis=1)


class PrimitiveGenerator:
    """Create candidate 3D trajectories from anchor parameters."""

    def __init__(self, camera: CameraIntrinsics) -> None:
        self.camera = camera

    def make_primitive(self, anchor: AnchorPrimitive, offsets: Iterable[float]) -> Trajectory3D:
        d_az, d_el, d_r, d_v, d_t = offsets
        az = anchor.direction[0] + d_az
        el = anchor.direction[1] + d_el
        distance = anchor.distance + d_r
        velocity = anchor.velocity + d_v
        duration = max(0.5, anchor.duration + d_t)
        time = np.linspace(0.0, duration, 50)
        target = np.array([
            distance * np.cos(el) * np.cos(az),
            distance * np.cos(el) * np.sin(az),
            distance * np.sin(el),
        ])
        coeffs = np.column_stack([
            np.linspace(0, target[0], 6),
            np.linspace(0, target[1], 6),
            np.linspace(0, target[2], 6),
        ])
        x = sample_trajectory(coeffs[:, 0], duration, num_samples=len(time))
        y = sample_trajectory(coeffs[:, 1], duration, num_samples=len(time))
        z = sample_trajectory(coeffs[:, 2], duration, num_samples=len(time))
        scale = np.clip(velocity * duration / distance, 0.3, 2.0)
        return Trajectory3D(x * scale, y * scale, z, duration)

    def filter_by_mask(self, trajectory: Trajectory3D, mask: np.ndarray) -> bool:
        points = trajectory.stack()
        pixels = self.camera.project(points)
        valid = ~np.isnan(pixels[:, 0])
        inside = self.camera.inside_image(pixels)
        valid &= inside
        if not valid.any():
            return False
        pixels = pixels[valid].astype(int)
        safe_pixels = mask[pixels[:, 1], pixels[:, 0]] > 0.5
        return bool(safe_pixels.all())

    def edge_push(self, trajectory: Trajectory3D, mask_grad: np.ndarray, step: float = 0.02) -> Trajectory3D:
        points = trajectory.stack()
        pixels = self.camera.project(points)
        valid = ~np.isnan(pixels[:, 0])
        if not valid.any():
            return trajectory
        grad_x = np.take_along_axis(mask_grad[0], pixels[:, 1].astype(int), axis=0)
        grad_y = np.take_along_axis(mask_grad[1], pixels[:, 0].astype(int), axis=0)
        push = np.stack([grad_x, grad_y], axis=1)
        push = np.nan_to_num(push, nan=0.0)
        offset_norm = np.linalg.norm(push, axis=1, keepdims=True) + 1e-6
        push_unit = push / offset_norm
        adjustment = step * push_unit
        adjusted_pixels = pixels + adjustment
        adjusted_pixels[:, 0] = (adjusted_pixels[:, 0] - self.camera.cx) / self.camera.fx
        adjusted_pixels[:, 1] = (adjusted_pixels[:, 1] - self.camera.cy) / self.camera.fy
        adjusted_points = np.column_stack([
            adjusted_pixels[:, 0] * points[:, 2],
            adjusted_pixels[:, 1] * points[:, 2],
            points[:, 2],
        ])
        return Trajectory3D(adjusted_points[:, 0], adjusted_points[:, 1], adjusted_points[:, 2], trajectory.duration)

    def generate_safe_primitives(
        self,
        anchors: List[AnchorPrimitive],
        offsets: List[np.ndarray],
        mask: np.ndarray,
    ) -> List[Trajectory3D]:
        safe: List[Trajectory3D] = []
        for anchor, offset in zip(anchors, offsets):
            traj = self.make_primitive(anchor, offset)
            if self.filter_by_mask(traj, mask):
                safe.append(traj)
        return safe
