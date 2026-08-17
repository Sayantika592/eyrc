#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from scipy.spatial.transform import Rotation as R_
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from sensor_msgs_py import point_cloud2 as pc2
import math
import time

class FruitTopDetector(Node):
    def __init__(self):
        super().__init__('fruit_top_detector_node')

        # Hyperparameters for grey detection
        self.LOWER_GREY = np.array([0, 0, 80])
        self.UPPER_GREY = np.array([179, 50, 200])
        self.grey_fruit_top_frame_id = "grey_fruit_top"
        self.LOWER_PURPLE = np.array([140, 100, 80])
        self.UPPER_PURPLE = np.array([160, 255, 255])
        self.bad_fruit_top_frame_id = "bad_fruit_top"
        self.LOWER_GREEN_TOP = np.array([40, 100, 100])
        self.UPPER_GREEN_TOP = np.array([70, 255, 255])


        self.ROI_X_START_PERCENTAGE = 0.0
        self.ROI_X_END_PERCENTAGE = 0.25
        self.ROI_Y_START_PERCENTAGE = 0.3
        self.ROI_Y_END_PERCENTAGE = 0.6


        self.MIN_CONTOUR_AREA_FRUIT = 500 # For the main fruit body (grey or purple)
        self.MIN_CONTOUR_AREA_TOP = 50    # For the smaller green top

        RGB_TOPIC = '/camera/image_raw'
        POINTCLOUD_TOPIC = '/camera/depth/points'
        CAMERA_INFO_TOPIC = '/camera/camera_info'

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.cam_info_callback, 10)
        self.create_subscription(Image, RGB_TOPIC, self.image_callback, 10)
        self.create_subscription(PointCloud2, POINTCLOUD_TOPIC, self.pointcloud_callback, 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_width = 0
        self.image_height = 0

        self.latest_pointcloud_msg = None
        self.pointcloud_xyz = None
        self.pc_width = 0
        self.pc_height = 0

        self.get_logger().info("Fruit Top Detector (Grey + Bad) with ROI initialized. Waiting for data...")

    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3)).astype(np.float64)
            self.dist_coeffs = np.array(msg.d).astype(np.float64) if msg.d is not None else np.zeros((5,), dtype=np.float64)
            self.image_width = msg.width
            self.image_height = msg.height
            self.get_logger().info(f"Camera calibration received. Image resolution: {self.image_width}x{self.image_height}")

    def pointcloud_callback(self, msg: PointCloud2):
        try:
            H = msg.height; W = msg.width
            if H < 2 or W < 2: self.pointcloud_xyz = None; return
            gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False)
            cloud = np.empty((H, W, 3), dtype=np.float32)
            idx = 0
            for x, y, z in gen:
                r = idx // W; c = idx % W
                cloud[r, c] = (x, y, z)
                idx += 1
            self.pointcloud_xyz = cloud
            self.pc_width = W; self.pc_height = H
            self.latest_pointcloud_msg = msg
        except Exception as e:
            self.get_logger().warn(f"Failed to convert PointCloud2 -> numpy: {e}")
            self.pointcloud_xyz = None

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}"); return
        if self.image_width == 0 or self.pointcloud_xyz is None:
            self.get_logger().warn("Waiting for camera info and point cloud...", throttle_duration_sec=2); return

        roi_x_start = int(self.ROI_X_START_PERCENTAGE * self.image_width)
        roi_x_end = int(self.ROI_X_END_PERCENTAGE * self.image_width)
        roi_y_start = int(self.ROI_Y_START_PERCENTAGE * self.image_height)
        roi_y_end = int(self.ROI_Y_END_PERCENTAGE * self.image_height)
        roi_x_start = max(0, roi_x_start); roi_y_start = max(0, roi_y_start)
        roi_x_end = min(self.image_width, roi_x_end); roi_y_end = min(self.image_height, roi_y_end)

        roi_frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        mask_roi_grey = cv2.inRange(hsv_roi, self.LOWER_GREY, self.UPPER_GREY)
        # mask_roi_purple = cv2.inRange(hsv_roi, self.LOWER_PURPLE, self.UPPER_PURPLE)
        mask_roi_green = cv2.inRange(hsv_roi, self.LOWER_GREEN_TOP, self.UPPER_GREEN_TOP)
        mask_grey_clean = cv2.dilate(cv2.erode(mask_roi_grey, None, iterations=2), None, iterations=2)
        # mask_purple_clean = cv2.dilate(cv2.erode(mask_roi_purple, None, iterations=2), None, iterations=2)
        contours_grey, _ = cv2.findContours(mask_grey_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        grey_fruit_count = 0
        if contours_grey:
            for cnt_g in contours_grey:
                area_g = cv2.contourArea(cnt_g)
                if area_g < self.MIN_CONTOUR_AREA_FRUIT: continue

                (x_g_body, y_g_body, w_g_body, h_g_body) = cv2.boundingRect(cnt_g)
                green_top_roi_mask = mask_roi_green[y_g_body:y_g_body+h_g_body, x_g_body:x_g_body+w_g_body]
                contours_green_top, _ = cv2.findContours(green_top_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours_green_top: continue
                
                cnt_gt = max(contours_green_top, key=cv2.contourArea)
                if cv2.contourArea(cnt_gt) < self.MIN_CONTOUR_AREA_TOP: continue
                
                M_gt = cv2.moments(cnt_gt)
                if M_gt["m00"] == 0: continue
                
                cx_gt_roi = int(M_gt["m10"] / M_gt["m00"])
                cy_gt_roi = int(M_gt["m01"] / M_gt["m00"])

                cx_full = cx_gt_roi + x_g_body + roi_x_start
                cy_full = cy_gt_roi + y_g_body + roi_y_start

                pt_3d = self.get_point_from_pointcloud_neighbourhood(cx_full, cy_full, k=5)
                if pt_3d is None: continue

                centroid_3d_gt = np.array(pt_3d)
                identity_quat_xyzw = [ 0.0, -0.934, 0.0, -0.358 ]
                child_frame = f"{self.grey_fruit_top_frame_id}_{grey_fruit_count}"
                grey_fruit_count += 1
                self.publish_object_tf(child_frame, centroid_3d_gt, identity_quat_xyzw)


                qua = identity_quat_xyzw
                t_base_cam = np.array([-1.080, 0.007, 1.090])
                r_cam_marker = R_.from_quat(qua)
                q_base_cam = np.array([0.000, 0.358, 0.000, 0.934])
                r_base_cam = R_.from_quat(q_base_cam)
                R_base_cam = r_base_cam.as_matrix()
                r_base_marker = r_base_cam * r_cam_marker
                q_base_marker = r_base_marker.as_quat()
                t_base_marker = (R_base_cam @ centroid_3d_gt.flatten()) + t_base_cam
                child_frame2 = f"eYRC#4406_bad_fruit_{grey_fruit_count}"
                self.publish_object_tf(child_frame2, t_base_marker, q_base_marker,True)

                cv2.drawContours(frame, [cnt_g + (roi_x_start, roi_y_start)], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx_full, cy_full), 7, (255, 255, 0), -1)
                cv2.putText(frame, child_frame, (cx_full - 20, cy_full - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    def get_point_from_pointcloud(self, u: int, v: int):
        if self.pointcloud_xyz is None: return None
        if u < 0 or v < 0 or u >= self.pc_width or v >= self.pc_height: return None
        pt = self.pointcloud_xyz[v, u]
        if np.any(np.isnan(pt)) or np.linalg.norm(pt) == 0.0: return None
        return tuple(pt.tolist())

    def get_point_from_pointcloud_neighbourhood(self, u_f, v_f, k=5):
        if self.pointcloud_xyz is None: return None
        u0 = int(round(u_f)); v0 = int(round(v_f)); half = k // 2
        vals = []
        for dv in range(-half, half+1):
            for du in range(-half, half+1):
                u = u0 + du; v = v0 + dv
                if u < 0 or v < 0 or u >= self.pc_width or v >= self.pc_height: continue
                pt = self.pointcloud_xyz[v, u]
                if np.any(np.isnan(pt)) or np.linalg.norm(pt) == 0.0: continue
                vals.append(pt)
        if len(vals) == 0: return None
        med = np.median(np.array(vals), axis=0)
        if np.any(np.isnan(med)) or np.linalg.norm(med) == 0.0: return None
        return tuple(med.tolist())

    def publish_object_tf(self, frame_id, centroid, quat,name=False):
        if not name:
            try:
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "camera_link"
                t.child_frame_id = frame_id
                t.transform.translation.x = float(centroid[0])
                t.transform.translation.y = float(centroid[1])
                t.transform.translation.z = float(centroid[2])
                t.transform.rotation.x = float(quat[0])
                t.transform.rotation.y = float(quat[1])
                t.transform.rotation.z = float(quat[2])
                t.transform.rotation.w = float(quat[3])
                self.tf_broadcaster.sendTransform(t)
                self.get_logger().info(f"Published TF: {t.child_frame_id} at [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
            except Exception as e:
                self.get_logger().warn(f"Could not publish object TF: {e}")
        else:
            try:
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "base_link"
                t.child_frame_id = frame_id
                t.transform.translation.x = float(centroid[0])
                t.transform.translation.y = float(centroid[1])
                t.transform.translation.z = float(centroid[2])
                t.transform.rotation.x = float(quat[0])
                t.transform.rotation.y = float(quat[1])
                t.transform.rotation.z = float(quat[2])
                t.transform.rotation.w = float(quat[3])
                self.tf_broadcaster.sendTransform(t)
                # self.get_logger().info(f"Published TF: {t.child_frame_id} at [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
            except Exception as e:
                self.get_logger().warn(f"Could not publish object TF: {e}")

def main():
    rclpy.init()
    node = FruitTopDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()