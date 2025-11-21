# YOPO Monocular Anchor-Based Safe Navigation

YOPO is a lightweight monocular navigation prototype that segments safe regions in an image, maps them to anchor-style motion primitives, and outputs optimized trajectories. The code is written in pure Python/PyTorch and does not require a ROS workspace or a CMake/catkin build.

## What's Included
- **YOPO/**: Core planner code with training, inference, and ROS node sketches.
- **monocular_nav/**: Synthetic data generator and camera helpers used by the YOPO pipeline.

## Quickstart
### 1) Install dependencies
Create a virtual environment (Python 3.9+) and install PyTorch plus common tooling:
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install numpy opencv-python pyyaml
```

### 2) Prepare data and camera config
- Place monocular RGB images under `YOPO/data/simulated/images/*.png` and matching binary safety masks under `YOPO/data/simulated/masks/*.png` (1=safe, 0=obstacle).
- Set pinhole intrinsics and distortion coefficients in `YOPO/config/camera.yaml`.
- To synthesize a toy dataset, run:
  ```bash
  python -m monocular_nav.data.collector --config monocular_nav/config/collector.yaml
  ```
  which writes RGB/mask pairs and a `poses.csv` into `monocular_nav/data/simulated/`.

### 3) Train the models
Run the two-stage training loop (segmentation then motion-offset refinement):
```bash
python -m YOPO.train
```
Trained weights are saved to `YOPO/saved/` for later inference.

### 4) Run an offline inference check
```bash
python YOPO/test.py --weights YOPO/saved/segmentation.pth --image path/to/frame.png
```
If `--image` is omitted, a blank frame is used to verify the end-to-end plumbing.

### 5) ROS usage (optional)
A minimal ROS 1 node sketch is provided for integration, but no ROS build is required here:
```bash
rosrun YOPO navigation_node.py _weights:=YOPO/saved/segmentation.pth
```
The node subscribes to `/camera/image_raw` and publishes optimized trajectories on `/yopo/trajectory` using `rospy`, `sensor_msgs`, `geometry_msgs`, and `cv_bridge`.

## Notes
- Unused placeholder directories for controllers and simulators have been removed to keep the repository focused on the monocular navigation prototype.
- The repository is self-contained Python; there is no separate build step beyond installing dependencies.
