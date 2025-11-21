"""ROS node sketch wiring monocular inference into motion commands."""
from __future__ import annotations
from pathlib import Path
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import torch
from monocular_nav.camera import CameraIntrinsics
from monocular_nav.inference import load_model, run_inference


class NavigationNode:
    def __init__(self, weights_path: str, camera: CameraIntrinsics) -> None:
        self.bridge = CvBridge()
        self.camera = camera
        self.model = load_model(Path(weights_path))
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1)
        self.trajectory_pub = rospy.Publisher("/monocular_nav/trajectory", PoseArray, queue_size=1)

    def image_callback(self, msg: Image) -> None:
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        tensor = torch.from_numpy(cv_image).permute(2, 0, 1).float() / 255.0
        trajectories = run_inference(tensor, self.model, self.camera)
        pose_array = PoseArray()
        pose_array.header = msg.header
        for traj in trajectories:
            pose = Pose()
            pose.position.x = float(traj.x[-1])
            pose.position.y = float(traj.y[-1])
            pose.position.z = float(traj.z[-1])
            pose_array.poses.append(pose)
        self.trajectory_pub.publish(pose_array)


def main():
    rospy.init_node("monocular_navigation")
    camera = CameraIntrinsics(fx=rospy.get_param("~fx", 320.0), fy=rospy.get_param("~fy", 320.0), cx=rospy.get_param("~cx", 320.0), cy=rospy.get_param("~cy", 240.0))
    weights = rospy.get_param("~weights", "./weights/segmentation.pth")
    NavigationNode(weights, camera)
    rospy.spin()


if __name__ == "__main__":
    main()
