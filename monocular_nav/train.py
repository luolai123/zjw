"""Two-stage training routine for segmentation then motion generation."""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
from monocular_nav.data.dataset import SegmentationDataset
from monocular_nav.models.segmentation import LightweightUNet
from monocular_nav.trajectory.anchors import AnchorGenerator
from monocular_nav.trajectory.generator import PrimitiveGenerator
from monocular_nav.camera import CameraIntrinsics
from monocular_nav.losses import segmentation_loss, total_trajectory_loss


def _build_dataset(dataset_root: Path) -> SegmentationDataset:
    dataset = SegmentationDataset(dataset_root)
    if len(dataset) == 0:
        images_dir = dataset_root / "images"
        masks_dir = dataset_root / "masks"
        raise FileNotFoundError(
            "No training samples were found. "
            f"Expected matching PNG files under '{images_dir}' and '{masks_dir}'. "
            "Generate data with the collector (python -m monocular_nav.data.collector "
            "--config monocular_nav/config/collector.yaml)."
        )
    return dataset


def train_segmentation(
    dataset_root: Path,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
) -> LightweightUNet:
    model = LightweightUNet()
    dataset = _build_dataset(dataset_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for images, masks in loader:
            optim.zero_grad()
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            loss.backward()
            optim.step()
    return model


def train_motion_module(
    model: LightweightUNet,
    camera: CameraIntrinsics,
    dataset_root: Path,
    epochs: int = 5,
    lr: float = 1e-3,
) -> None:
    model.eval()
    anchor_generator = AnchorGenerator()
    anchors = anchor_generator.generate()
    generator = PrimitiveGenerator(camera)
    params = [torch.zeros(5, requires_grad=True) for _ in anchors]
    optim = torch.optim.Adam(params, lr=lr)
    dataset = _build_dataset(dataset_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for _ in range(epochs):
        for images, masks in loader:
            with torch.no_grad():
                mask_logits = model(images)
                mask = torch.sigmoid(mask_logits)[0, 0].numpy()
            optim.zero_grad()
            losses: List[torch.Tensor] = []
            for anchor, offset in zip(anchors, params):
                traj = generator.make_primitive(anchor, offset.detach().numpy())
                pixels = camera.project(traj.stack())
                pixels = pixels[~np.isnan(pixels[:, 0])]
                loss_value = total_trajectory_loss(mask, pixels, traj.stack())
                losses.append(torch.tensor(loss_value, requires_grad=True))
            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            optim.step()


def run_training(dataset_root: str, camera_cfg: CameraIntrinsics) -> LightweightUNet:
    dataset_path = Path(dataset_root)
    seg_model = train_segmentation(dataset_path)
    train_motion_module(seg_model, camera_cfg, dataset_path)
    return seg_model


if __name__ == "__main__":
    camera = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0)
    run_training("./data/simulated", camera)
