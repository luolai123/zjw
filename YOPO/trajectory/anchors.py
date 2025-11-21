"""Anchor-style motion primitive definitions bound to the image grid."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class AnchorPrimitive:
    grid_index: Tuple[int, int]
    direction: Tuple[float, float]
    distance: float
    velocity: float
    duration: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.direction[0], self.direction[1], self.distance, self.velocity, self.duration])


class AnchorGenerator:
    """Generate anchor primitives covering the field of view."""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (8, 4),
        azimuth_range: Tuple[float, float] = (-30.0, 30.0),
        elevation_range: Tuple[float, float] = (-15.0, 15.0),
        distance: float = 5.0,
        speed_range: Tuple[float, float] = (0.5, 3.0),
        duration: float = 2.0,
    ) -> None:
        self.grid_size = grid_size
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        self.distance = distance
        self.speed_range = speed_range
        self.duration = duration

    def generate(self) -> List[AnchorPrimitive]:
        anchors: List[AnchorPrimitive] = []
        grid_x, grid_y = self.grid_size
        azimuths = np.linspace(self.azimuth_range[0], self.azimuth_range[1], grid_x)
        elevations = np.linspace(self.elevation_range[0], self.elevation_range[1], grid_y)
        speeds = np.linspace(self.speed_range[0], self.speed_range[1], 2)
        for ix, az in enumerate(azimuths):
            for iy, el in enumerate(elevations):
                for speed in speeds:
                    anchors.append(
                        AnchorPrimitive(
                            grid_index=(ix, iy),
                            direction=(np.deg2rad(az), np.deg2rad(el)),
                            distance=self.distance,
                            velocity=float(speed),
                            duration=self.duration,
                        )
                    )
        return anchors
