# YOPO Monocular Anchor-Based Safe Navigation

This repository sketches a monocular camera pipeline that segments safe regions, generates anchor-style motion primitives, and outputs optimized trajectories for autonomous navigation. The design keeps training and inference lightweight for ROS deployment.

## Directory Layout
- **YOPO/**: Core planner code, including configuration (**config/**), pretrained weights placeholders (**saved/**), anchor-based trajectory logic, and a quick evaluation script (**test.py**).
- **Controller/**: Controller and dynamics simulator scaffolding (ROS-oriented, adapted from HKUST-Aerial-Robotics/Fast-Planner) for attitude/position control.
- **Simulator/**: Environment and sensor simulation stubs (CUDA-capable), compatible with mock map generation.

## Usage
### 1) Prepare data and camera config
- Place monocular RGB images under `YOPO/data/simulated/images/*.png` and matching binary safety masks under `YOPO/data/simulated/masks/*.png` (1=safe, 0=obstacle).
- Set pinhole intrinsics and distortion coefficients in `YOPO/config/camera.yaml`.

### 2) Train (two-stage)
```bash
# Segmentation then motion-offset refinement
python -m YOPO.train
```
- The script first fits a lightweight U-Net on the masks, freezes it, then optimizes anchor offsets using safety/smoothness trajectory losses.
- Save resulting weights to `YOPO/saved/` as needed for inference.

### 3) Offline inference smoke test
```bash
python YOPO/test.py --weights YOPO/saved/segmentation.pth --image path/to/frame.png
```
- Loads the segmentation model, projects anchor primitives through the camera model, filters against the predicted mask, and prints the count of safe, edge-adjusted trajectories.
- If `--image` is omitted, a blank frame is used to verify the end-to-end plumbing.

### 4) ROS node sketch
```bash
rosrun YOPO navigation_node.py _weights:=YOPO/saved/segmentation.pth
```
- Subscribes to `/camera/image_raw` and publishes optimized trajectories on `/yopo/trajectory`.
- Requires `rospy`, `sensor_msgs`, `geometry_msgs`, and `cv_bridge` in your ROS workspace.

## Key Components (YOPO/)
- `models/segmentation.py`: Simplified U-Net with BCE loss for binary safety masks.
- `trajectory/anchors.py`: Anchor-style primitive definitions bound to image grid cells, mirroring detection-style priors.
- `trajectory/generator.py`: Quintic trajectory synthesis, mask-based filtering, and gradient edge pushing (<10° deviation).
- `losses.py`: Segmentation loss plus trajectory safety and smoothness penalties.
- `train.py`: Two-phase training loop—segmentation then motion refinement with frozen masks.
- `inference.py`: End-to-end inference from RGB to adjusted safe trajectories.
- `ros_nodes/navigation_node.py`: ROS 1 node sketch wiring camera images to published trajectories.
- `test.py`: Minimal script to load a frame, run the planner, and report safe primitive counts.

## Anchor-Style Primitive Design
- The image is divided into an 8×4 grid; each cell binds predefined 3D primitives (direction φ/θ, distance r, speed v, duration T), analogous to detection anchors.
- The network predicts primitive offsets `(Δφ, Δθ, Δr, ΔvT, ΔaT)` and confidence instead of generating trajectories from scratch, constraining search space and enabling real-time mapping from image features to motion.

## Trajectory Representation
Quintic polynomial `s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5`, sampled over `t ∈ [0, T]` (default `T=2s`) to derive 3D points, projected via camera intrinsics `(fx, fy, cx, cy)` for mask filtering.
