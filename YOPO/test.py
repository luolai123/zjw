"""Quick offline test for the YOPO monocular planner."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml

from YOPO.camera import CameraIntrinsics
from YOPO.inference import load_model, run_inference


def load_camera(config_path: Path) -> CameraIntrinsics:
    data = yaml.safe_load(config_path.read_text())
    return CameraIntrinsics(**data)


def load_image(image_path: Path | None) -> torch.Tensor:
    if image_path is None:
        return torch.zeros(3, 224, 224)
    image = Image.open(image_path).convert("RGB")
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=False, default=Path("YOPO/saved/segmentation.pth"))
    parser.add_argument("--image", type=Path, required=False, help="Path to an RGB image for inference")
    parser.add_argument(
        "--camera",
        type=Path,
        default=Path("YOPO/config/camera.yaml"),
        help="Camera intrinsics YAML path",
    )
    args = parser.parse_args()

    camera = load_camera(args.camera)
    model = load_model(args.weights)
    frame = load_image(args.image)
    trajectories = run_inference(frame, model, camera)
    print(f"Safe primitives: {len(trajectories)}")


if __name__ == "__main__":
    main()
