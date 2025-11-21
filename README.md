# Monocular Anchor-Based Safe Navigation

This ROS-oriented prototype demonstrates a monocular camera pipeline that segments safe regions, generates anchor-style motion primitives, and outputs optimized trajectories without depth sensors. The design follows a two-stage training flow (segmentation, then motion generation) and supports onboard inference with ROS topics.

## System Overview
1. **Training stage**
   - Train a lightweight U-Net on monocular images with safety-distance annotations.
   - Freeze the segmentation backbone and train anchor-tied motion primitive offsets using trajectory safety/smoothness losses.
2. **Inference stage**
   - Input: monocular RGB frame.
   - Output: filtered and edge-adjusted motion primitives ready for execution.

## Key Modules
- `monocular_nav/models/segmentation.py`: simplified U-Net with BCE loss for binary safety masks.
- `monocular_nav/trajectory/anchors.py`: anchor-style primitive definitions bound to image grid cells, mirroring object-detection priors.
- `monocular_nav/trajectory/generator.py`: quintic trajectory synthesis, mask-based filtering, and gradient edge pushing (<10° deviation) for safety.
- `monocular_nav/losses.py`: segmentation loss plus trajectory safety and smoothness penalties.
- `monocular_nav/train.py`: two-phase training loop—segmentation then motion refinement with frozen masks.
- `monocular_nav/inference.py`: end-to-end inference from RGB to adjusted safe trajectories.
- `monocular_nav/ros_nodes/navigation_node.py`: ROS 1 node sketch subscribing to `/camera/image_raw` and publishing `/monocular_nav/trajectory`.
- `monocular_nav/config/camera.yaml`: default pinhole intrinsics and distortion vector.

## Anchor-Style Primitive Design
- Image is divided into an 8×4 grid; each cell binds predefined 3D primitives (direction φ/θ, distance r, speed v, duration T), analogous to detection anchors.
- Network predicts primitive offsets `(Δφ, Δθ, Δr, ΔvT, ΔaT)` and confidence instead of generating trajectories from scratch, constraining search space and enabling real-time mapping from image features to motion.

## Usage Sketch
```bash
# Train segmentation then motion module (dataset folders: data/simulated/images, data/simulated/masks)
python -m monocular_nav.train

# Run ROS node (requires rospy/sensor_msgs/geometry_msgs/cv_bridge)
rosrun monocular_nav navigation_node.py _weights:=/path/to/segmentation.pth
```

## Trajectory Representation
Quintic polynomial `s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5`, sampled over `t ∈ [0, T]` (default `T=2s`) to derive 3D points, projected via camera intrinsics `(fx, fy, cx, cy)` for mask filtering.
