#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from scipy.spatial.transform import Rotation as R_
from sensor_msgs_py import point_cloud2 as pc2   # ROS2 helper for pointcloud2
import math
import time

class ArucoTFPublisher(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher')
        RGB_TOPIC = '/camera/image_raw'
        POINTCLOUD_TOPIC = '/camera/depth/points'
        CAMERA_INFO_TOPIC = '/camera/camera_info'
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.cam_info_callback, 10)
        self.create_subscription(Image, RGB_TOPIC, self.image_callback, 10)
        self.create_subscription(PointCloud2, POINTCLOUD_TOPIC, self.pointcloud_callback, 10)

        self.map = {3:'eYRC#4406_fertilizer_can', 6:'vehicle'}

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        self.latest_pointcloud_msg = None
        self.pointcloud_xyz = None   
        self.pc_width = 0
        self.pc_height = 0

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()

        self.get_logger().info("ArUco TF publisher (RGB + organized PointCloud2) initialized. Waiting for data...")

    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3)).astype(np.float64)
            self.dist_coeffs = np.array(msg.d).astype(np.float64) if msg.d is not None else np.zeros((5,), dtype=np.float64)
            self.cx = self.camera_matrix[0,2]
            self.cy = self.camera_matrix[1,2]
            self.fx = self.camera_matrix[0,0]
            self.fy = self.camera_matrix[1,1]
            self.get_logger().info("Camera calibration received and stored.")

    def pointcloud_callback(self, msg: PointCloud2):
        try:
            H = msg.height
            W = msg.width

            if H < 2 or W < 2:
                self.pointcloud_xyz = None
                return
            gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False)

            cloud = np.empty((H, W, 3), dtype=np.float32)

            idx = 0
            for x, y, z in gen:
                r = idx // W
                c = idx % W
                cloud[r, c] = (x, y, z)
                idx += 1

            self.pointcloud_xyz = cloud
            self.pc_width = W
            self.pc_height = H
            self.latest_pointcloud_msg = msg

        except Exception as e:
            self.get_logger().warn(f"Failed to convert PointCloud2 -> numpy: {e}")
            self.pointcloud_xyz = None


    def _rotate_quat_180_y(self,mid,q):
        q_orig = R_.from_quat(q)
        if mid == 3:
            q_rot = R_.from_euler('y', 180, degrees=True)
            q_new = q_rot * q_orig
        elif mid == 6:
            q_rot = R_.from_euler('z', 180, degrees=True)
            q_new = q_orig * q_rot
        q_new =  q_new.as_quat()
        q_new = q_new/np.linalg.norm(q_new)
        return q_new

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is None or self.pointcloud_xyz is None:
            return

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i].reshape(-1, 2)
            pts3d = []
            valid = True
            for (u_f, v_f) in c:
                u, v = int(round(u_f)), int(round(v_f))
                pt = self.get_point_from_pointcloud(u, v)
                if pt is None:
                    pt = self.get_point_from_pointcloud_neighbourhood(u_f, v_f, k=5)
                if pt is None:
                    valid = False
                    break
                pts3d.append(np.array(pt, dtype=np.float64))

            if not valid:
                self.get_logger().warn(f"3D corner fetch failed for marker {marker_id}")
                continue
            pts3d = np.stack(pts3d, axis=0)
            centroid = np.mean(pts3d, axis=0).reshape((3,1))
            v_x = pts3d[1] - pts3d[0]
            v_y = pts3d[3] - pts3d[0]

            if np.linalg.norm(v_x) < 1e-6 or np.linalg.norm(v_y) < 1e-6:
                continue

            x_axis = v_x / np.linalg.norm(v_x)
            y_temp = v_y / np.linalg.norm(v_y)
            z_axis = np.cross(x_axis, y_temp)
            z_axis /= np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)

            R_cam_marker = np.column_stack((x_axis, y_axis, z_axis))
            U, _, Vt = np.linalg.svd(R_cam_marker)
            R_cam_marker = U @ Vt
            quati = R_.from_matrix(R_cam_marker).as_quat()
            quat = self._rotate_quat_180_y(marker_id,quati)
            quat = quat / np.linalg.norm(quat)

            self.publish_marker_tf(marker_id, centroid.flatten(), quat)

            to_camera = -centroid.flatten()
            R_cam_marker_c  = R_cam_marker.copy()
            # if Z axis is pointing AWAY from the camera (wrong)
            if np.dot(R_cam_marker_c[:, 2], to_camera) < 0:
                # Flip Z, and Y to maintain right-handed frame
                R_cam_marker_c[:, 2] *= -1
                R_cam_marker_c[:, 1] *= -1
            qua = R_.from_matrix(R_cam_marker_c).as_quat()
            t_base_cam = np.array([-1.080, 0.007, 1.090])
            r_cam_marker = R_.from_quat(qua)
            q_base_cam = np.array([0.000, 0.358, 0.000, 0.934])
            r_base_cam = R_.from_quat(q_base_cam)
            R_base_cam = r_base_cam.as_matrix()
            r_base_marker = r_base_cam * r_cam_marker
            t_base_marker = (R_base_cam @ centroid.flatten()) + t_base_cam

            # Get the final results for the base-to-marker pose
            q_base_marker = r_base_marker.as_quat()

            self.publish_marker_tf(marker_id,t_base_cam,q_base_marker,True)

    def get_point_from_pointcloud(self, u: int, v: int):
        if self.pointcloud_xyz is None:
            return None
        if u < 0 or v < 0 or u >= self.pc_width or v >= self.pc_height:
            return None
        pt = self.pointcloud_xyz[v, u]
        if np.any(np.isnan(pt)) or np.linalg.norm(pt) == 0.0:
            return None
        return tuple(pt.tolist())

    def get_point_from_pointcloud_neighbourhood(self, u_f, v_f, k=5):
        if self.pointcloud_xyz is None:
            return None
        u0 = int(round(u_f))
        v0 = int(round(v_f))
        half = k // 2
        vals = []
        for dv in range(-half, half+1):
            for du in range(-half, half+1):
                u = u0 + du
                v = v0 + dv
                if u < 0 or v < 0 or u >= self.pc_width or v >= self.pc_height:
                    continue
                pt = self.pointcloud_xyz[v, u]
                if np.any(np.isnan(pt)) or np.linalg.norm(pt) == 0.0:
                    continue
                vals.append(pt)
        if len(vals) == 0:
            return None
        vals = np.array(vals)
        med = np.median(vals, axis=0)
        if np.any(np.isnan(med)) or np.linalg.norm(med) == 0.0:
            return None
        return tuple(med.tolist())

    def publish_marker_tf(self, marker_id, centroid, quat,name=False):
        if not name:
            try:
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "camera_link"
                t.child_frame_id = f"{self.map.get(int(marker_id), f'marker_{int(marker_id)}')}_s"
                t.transform.translation.x = float(centroid[0])
                t.transform.translation.y = float(centroid[1])
                t.transform.translation.z = float(centroid[2])
                t.transform.rotation.x = float(quat[0])
                t.transform.rotation.y = float(quat[1])
                t.transform.rotation.z = float(quat[2])
                t.transform.rotation.w = float(quat[3])
                self.tf_broadcaster.sendTransform(t)
                self.get_logger().info(f"Published TF: {t.child_frame_id} @ {centroid}")
            except Exception as e:
                self.get_logger().warn(f"Could not publish marker TF: {e}")
        else:
            try:
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "base_link"
                t.child_frame_id = f"{self.map.get(int(marker_id), f'marker_{int(marker_id)}')}"
                t.transform.translation.x = float(centroid[0])
                t.transform.translation.y = float(centroid[1])
                t.transform.translation.z = float(centroid[2])
                t.transform.rotation.x = float(quat[0])
                t.transform.rotation.y = float(quat[1])
                t.transform.rotation.z = float(quat[2])
                t.transform.rotation.w = float(quat[3])
                self.tf_broadcaster.sendTransform(t)
                self.get_logger().info(f"Published TF: {t.child_frame_id} @ {centroid}")
            except Exception as e:
                self.get_logger().warn(f"Could not publish marker TF: {e}")

def main():
    rclpy.init()
    node = ArucoTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
