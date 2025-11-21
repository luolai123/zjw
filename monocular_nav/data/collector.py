"""Synthetic dataset collector for monocular navigation.

The collector mirrors the automatic depth dataset generator shared in the
prompt but adapts it to produce monocular RGB images and binary safety masks.
Randomly placed box-like obstacles are projected through the pinhole camera
model, producing pairs of ``images/*.png`` and ``masks/*.png`` plus a CSV of
camera poses for later use.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw

from monocular_nav.camera import CameraIntrinsics


@dataclass
class CollectorSettings:
    output_dir: Path
    num_frames: int
    obstacles_per_frame: Tuple[int, int]
    depth_range: Tuple[float, float]
    size_range: Tuple[float, float]
    lateral_range: Tuple[float, float]
    vertical_range: Tuple[float, float]
    yaw_range_deg: Tuple[float, float]
    background: Tuple[int, int, int]
    seed: int
    overwrite: bool


def load_camera(camera_config: Path) -> CameraIntrinsics:
    with open(camera_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return CameraIntrinsics(
        fx=cfg["fx"],
        fy=cfg["fy"],
        cx=cfg["cx"],
        cy=cfg["cy"],
        distortion=tuple(cfg.get("distortion", (0.0, 0.0, 0.0, 0.0, 0.0))),
        image_width=int(cfg.get("image_width", 640)),
        image_height=int(cfg.get("image_height", 480)),
    )


def load_settings(config_path: Path, args: argparse.Namespace) -> CollectorSettings:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output or cfg["output_dir"])
    num_frames = args.frames or int(cfg["num_frames"])
    settings = CollectorSettings(
        output_dir=output_dir,
        num_frames=num_frames,
        obstacles_per_frame=tuple(cfg["obstacles_per_frame"]),
        depth_range=tuple(cfg["depth_range"]),
        size_range=tuple(cfg["size_range"]),
        lateral_range=tuple(cfg["lateral_range"]),
        vertical_range=tuple(cfg["vertical_range"]),
        yaw_range_deg=tuple(cfg["yaw_range_deg"]),
        background=tuple(cfg["background"]),
        seed=int(args.seed if args.seed is not None else cfg["seed"]),
        overwrite=bool(cfg.get("overwrite", True)) if args.overwrite is None else args.overwrite,
    )
    return settings


def prepare_output(root: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    images = root / "images"
    masks = root / "masks"
    poses = root / "poses.csv"
    if overwrite and root.exists():
        for child in [images, masks, poses]:
            if child.is_dir():
                for png in child.glob("*.png"):
                    png.unlink()
            elif child.is_file():
                child.unlink()
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    return images, masks, poses


def yaw_to_rotation(yaw_rad: float) -> np.ndarray:
    c = math.cos(-yaw_rad)
    s = math.sin(-yaw_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def project_obstacle(
    camera: CameraIntrinsics,
    obstacle_corners_world: np.ndarray,
    yaw_rad: float,
) -> np.ndarray | None:
    rotation = yaw_to_rotation(yaw_rad)
    corners_cam = obstacle_corners_world @ rotation.T
    pixels = camera.project(corners_cam)
    inside = camera.inside_image(pixels)
    valid = (~np.isnan(pixels[:, 0])) & inside
    if valid.sum() < 3:
        return None
    return pixels[valid]


def sample_obstacle(settings: CollectorSettings) -> Tuple[np.ndarray, float]:
    depth = random.uniform(*settings.depth_range)
    width = random.uniform(*settings.size_range)
    height = random.uniform(*settings.size_range)
    center_x = random.uniform(*settings.lateral_range)
    center_y = random.uniform(*settings.vertical_range)
    corners = np.array(
        [
            [center_x - width / 2.0, center_y - height / 2.0, depth],
            [center_x + width / 2.0, center_y - height / 2.0, depth],
            [center_x + width / 2.0, center_y + height / 2.0, depth],
            [center_x - width / 2.0, center_y + height / 2.0, depth],
        ],
        dtype=float,
    )
    return corners, depth


def draw_obstacle(
    rgb_draw: ImageDraw.ImageDraw,
    mask_draw: ImageDraw.ImageDraw,
    pixels: np.ndarray,
    depth: float,
    settings: CollectorSettings,
) -> None:
    depth_norm = (depth - settings.depth_range[0]) / max(1e-6, settings.depth_range[1] - settings.depth_range[0])
    shade = int(220 - 140 * np.clip(depth_norm, 0.0, 1.0))
    color = (shade, max(0, shade - 25), max(0, shade - 45))
    points = [tuple(map(float, pt)) for pt in pixels]
    rgb_draw.polygon(points, fill=color)
    mask_draw.polygon(points, fill=0)


def render_frame(
    camera: CameraIntrinsics,
    settings: CollectorSettings,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, float, int]:
    image = Image.new("RGB", (camera.image_width, camera.image_height), color=settings.background)
    mask = Image.new("L", (camera.image_width, camera.image_height), color=255)
    rgb_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    yaw_deg = rng.uniform(*settings.yaw_range_deg)
    yaw_rad = math.radians(yaw_deg)
    obstacles_drawn = 0

    num_obstacles = rng.randint(settings.obstacles_per_frame[0], settings.obstacles_per_frame[1])
    for _ in range(num_obstacles):
        corners_world, depth = sample_obstacle(settings)
        projected = project_obstacle(camera, corners_world, yaw_rad)
        if projected is None:
            continue
        draw_obstacle(rgb_draw, mask_draw, projected, depth, settings)
        obstacles_drawn += 1
    return image, mask, yaw_deg, obstacles_drawn


def progress_bar(current: int, total: int) -> str:
    progress = current / total
    bar_width = 30
    pos = int(bar_width * progress)
    bar = "=" * pos + ">" + " " * (bar_width - pos)
    return f"[{bar}] {progress*100:5.1f}%"


def collect_dataset(camera_cfg: Path, settings: CollectorSettings) -> None:
    camera = load_camera(camera_cfg)
    images_dir, masks_dir, pose_path = prepare_output(settings.output_dir, settings.overwrite)

    rng = random.Random(settings.seed)
    np.random.seed(settings.seed)

    with open(pose_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "z", "yaw_deg", "obstacles"])
        for idx in range(settings.num_frames):
            image, mask, yaw_deg, obstacles = render_frame(camera, settings, rng)
            name = f"frame_{idx:05d}.png"
            image.save(images_dir / name)
            mask.save(masks_dir / name)
            writer.writerow([idx, 0.0, 0.0, 0.0, yaw_deg, obstacles])
            print(progress_bar(idx + 1, settings.num_frames), end="\r")
    print(f"\nSaved synthetic dataset to {settings.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic monocular dataset collection")
    parser.add_argument("--config", type=Path, default=Path("monocular_nav/config/collector.yaml"), help="Collector YAML config")
    parser.add_argument("--camera", type=Path, default=Path("monocular_nav/config/camera.yaml"), help="Camera intrinsics YAML")
    parser.add_argument("--frames", type=int, default=None, help="Override number of frames")
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="Force overwrite existing dataset")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Append without deleting old files")
    parser.set_defaults(overwrite=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config, args)
    collect_dataset(args.camera, settings)


if __name__ == "__main__":
    main()
