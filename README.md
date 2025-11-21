# YOPO Monocular Anchor-Based Safe Navigation

YOPO is a lightweight monocular navigation prototype that segments safe regions in an image, maps them to anchor-style motion primitives, and outputs optimized trajectories. The code is written in pure Python/PyTorch and does not require a ROS workspace or a CMake/catkin build.

## What's Included
- **monocular_nav/**: Core planner code with training, inference, synthetic data generation, and a ROS 1 node sketch.
- **monocular_nav/config/**: Default configuration for the synthetic collector.
- **monocular_nav/data/**: Generated datasets live here (images, masks, and poses.csv).

## Prerequisites
- Python 3.9+
- PyTorch (CPU is fine for the demo-sized models)
- Optional: ROS 1 for the example node

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install numpy opencv-python pyyaml pillow
```

## Generate a Synthetic Dataset
1. Adjust camera intrinsics in `monocular_nav/config/camera.yaml` if needed.
2. Optionally tweak dataset parameters (frame count, obstacle ranges, seed) in `monocular_nav/config/collector.yaml`.
3. Run the collector to populate `monocular_nav/data/simulated/` with RGB frames, binary masks, and a `poses.csv` log:
   ```bash
   python -m monocular_nav.data.collector \
     --config monocular_nav/config/collector.yaml
   ```
   Use `--frames` or `--seed` to override the defaults defined in the YAML.

## Train the Models
Training is a two-stage routine: semantic segmentation followed by anchor-offset refinement.
```bash
python -m monocular_nav.train
```
- Expects images under `monocular_nav/data/simulated/images/*.png` and matching masks under `monocular_nav/data/simulated/masks/*.png`.
- Writes trained segmentation weights to memory during the run; modify `train_segmentation`/`train_motion_module` in `monocular_nav/train.py` to save checkpoints to disk as needed.

## Run Offline Inference
Use `monocular_nav/inference.py` to load weights and produce safe motion primitives from a monocular frame:
```bash
python - <<'PY'
from pathlib import Path
import torch
from monocular_nav.camera import CameraIntrinsics
from monocular_nav.inference import load_model, run_inference

camera = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0)
model = load_model(Path("./weights/segmentation.pth"))
# Replace with a real RGB tensor in CHW format, normalized to [0,1]
image = torch.zeros(3, camera.image_height, camera.image_width)
trajectories = run_inference(image, model, camera)
print(f"Produced {len(trajectories)} safe trajectories")
PY
```
Swap in an actual image tensor (e.g., from OpenCV) and the weights produced during training.

## ROS 1 Node (Optional)
A minimal ROS 1 integration lives in `monocular_nav/ros_nodes/navigation_node.py`:
```bash
rosrun monocular_nav.navigation_node _weights:=./weights/segmentation.pth \
  _fx:=320.0 _fy:=320.0 _cx:=320.0 _cy:=240.0
```
- Subscribes to `/camera/image_raw` (RGB) and publishes a `PoseArray` on `/monocular_nav/trajectory`.
- Relies on `sensor_msgs`, `geometry_msgs`, `rospy`, and `cv_bridge` to convert images to tensors.

## Tips
- All paths are relative to the repository root; adjust if running from elsewhere.
- The synthetic collector is deterministic with the provided seed, making it easy to regenerate the same dataset for debugging.
- The models are intentionally small to keep experimentation fast on CPU-only machines.
