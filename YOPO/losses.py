"""Loss functions for segmentation and trajectory modules."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)


def trajectory_safety_loss(mask: np.ndarray, pixels: np.ndarray) -> float:
    pixels = pixels.astype(int)
    hit = mask[pixels[:, 1], pixels[:, 0]] < 0.5
    return float(hit.mean())


def trajectory_smoothness_loss(coeffs: np.ndarray) -> float:
    jerk_coeffs = np.array([0, 0, 2, 6, 12, 20])
    jerk = np.sum((coeffs * jerk_coeffs) ** 2)
    return float(jerk)


def total_trajectory_loss(mask: np.ndarray, pixels: np.ndarray, coeffs: np.ndarray) -> float:
    return trajectory_safety_loss(mask, pixels) + 0.01 * trajectory_smoothness_loss(coeffs)
