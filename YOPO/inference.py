"""Inference pipeline: from monocular RGB to safe trajectory."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from YOPO.camera import CameraIntrinsics
from YOPO.models.segmentation import LightweightUNet
from YOPO.trajectory.anchors import AnchorGenerator
from YOPO.trajectory.generator import PrimitiveGenerator


def load_model(weights_path: Path) -> LightweightUNet:
    model = LightweightUNet()
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def run_inference(image: torch.Tensor, model: LightweightUNet, camera: CameraIntrinsics):
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        mask = torch.sigmoid(logits)[0, 0].numpy()
    mask_grad = np.gradient(mask)
    anchor_generator = AnchorGenerator()
    anchors = anchor_generator.generate()
    offsets = [np.zeros(5) for _ in anchors]
    generator = PrimitiveGenerator(camera)
    safe = generator.generate_safe_primitives(anchors, offsets, mask)
    adjusted = [generator.edge_push(traj, mask_grad) for traj in safe]
    return adjusted
