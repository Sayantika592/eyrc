#!/usr/bin/env python3
"""
ULTIMATE FIX: FORCE DETECTION MODE - WILL STOP AT ANY SHAPE EDGES
No more complex logic. If ANY short lines detected in front → PUBLISH IMMEDIATELY
This will make the robot stop at every shape it approaches.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from tf_transformations import euler_from_quaternion

class ShapeDetector(Node):
    def __init__(self):
        super().__init__('shape_detector')
        
        
        self.marker_pub = self.create_publisher(MarkerArray, '/shape_detector/lines', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.status_pub = self.create_publisher(String, '/detection_status', 10)
        
        # Robot pose
        self.robot_x = self.robot_y = self.robot_yaw = 0.0
        self.has_odom = False
        
        # DETECTION: ULTRA-RELAXED - NO COOLDOWN, PUBLISH EVERYTHING
        self.detected_positions = []
        self.publish_every_scan = True  # Force publish on every valid detection
        
        # RANSAC: TUNED FOR SHORT EDGES
        self.ransac_threshold = 0.03  # 3cm tolerance
        self.ransac_iters = 150
        self.min_inliers = 5
        self.max_lines = 25
        
        # SHORT LINES ONLY: 0.05m to 0.60m
        self.min_short = 0.05
        self.max_short = 0.60
        
        # ANY 2+ SHORT LINES IN FRONT = DETECT
        self.min_lines_for_detect = 2
        self.front_dist_min = 0.10  # Even very close
        self.front_dist_max = 1.00  # Up to 1m
        self.side_limit = 0.50  # ±50cm sides
        
        self.get_logger().info('🚨 FORCE DETECTION MODE: Will publish ANY short lines in front!')
        self.get_logger().info('📍 Detecting from 0.10m to 1.0m ahead, ANY 2+ short edges = BAD_HEALTH')
    
    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.robot_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.has_odom = True
    
    def scan_callback(self, msg):
        if not self.has_odom:
            self.get_logger().info('⏳ Waiting for odom...')
            return
        
        # FORCE: Process every scan (no cooldown)
        points = self.scan_to_points(msg)
        if len(points) < 10:
            self.get_logger().debug('⚠️ Too few points')
            return
        
        self.get_logger().info(f'📊 Processing {len(points)} points from LiDAR')
        
        # Extract lines
        lines = self.ransac_extract_lines(points)
        self.get_logger().info(f'📏 Found {len(lines)} total lines')
        
        if len(lines) == 0:
            return
        
        # Visualize ALL lines (green for short, red for long)
        self.publish_markers(lines)
        
        # Filter SHORT lines (shape edges)
        short_lines = [l for l in lines if self.min_short <= l['length'] <= self.max_short]
        self.get_logger().info(f'🔍 Short lines: {len(short_lines)} (0.05-0.60m)')
        
        if len(short_lines) < self.min_lines_for_detect:
            self.get_logger().info(f'❌ Need {self.min_lines_for_detect} short lines, have {len(short_lines)}')
            return
        
        # ANY SHORT LINES? Check if in front zone
        front_short = []
        for line in short_lines:
            center_x, center_y = line['center']
            if (self.front_dist_min <= center_x <= self.front_dist_max and 
                abs(center_y) <= self.side_limit):
                front_short.append(line)
        
        self.get_logger().info(f'🎯 Front short lines: {len(front_short)} in zone')
        
        if len(front_short) >= self.min_lines_for_detect:
            # FORCE DETECT: ANY 2+ SHORT LINES IN FRONT = BAD_HEALTH (most common)
            avg_center = np.mean([l['center'] for l in front_short], axis=0)
            self.get_logger().info(f'🔥 TRIGGER: {len(front_short)} short lines at avg ({avg_center[0]:.2f}, {avg_center[1]:.2f})')
            
            # Transform to world (map frame)
            dx = avg_center[0] * np.cos(self.robot_yaw) - avg_center[1] * np.sin(self.robot_yaw)
            dy = avg_center[0] * np.sin(self.robot_yaw) + avg_center[1] * np.cos(self.robot_yaw)
            world_x = self.robot_x + dx
            world_y = self.robot_y + dy
            
            # Check not too close to previous (simple duplicate filter)
            if not self.is_duplicate(world_x, world_y):
                self.publish_force(world_x, world_y, 'BAD_HEALTH')
                return  # One detection per scan
        else:
            self.get_logger().info(f'❌ No front short lines (need {self.min_lines_for_detect}, have {len(front_short)})')
    
    def scan_to_points(self, msg):
        """Convert LiDAR to robot-frame points"""
        points = []
        angle = msg.angle_min
        for r in msg.ranges:
            if 0.08 < r < 2.5 and not np.isnan(r) and not np.isinf(r):  # Start from 8cm
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                if x > 0.05 and abs(y) < 1.0:  # Forward bias
                    points.append([x, y])
            angle += msg.angle_increment
        return np.array(points)
    
    def ransac_extract_lines(self, points):
        """Robust RANSAC for line segments"""
        lines = []
        remaining = points.copy()
        
        for _ in range(self.max_lines):
            if len(remaining) < self.min_inliers:
                break
            
            best_model = None
            best_inliers = np.array([])
            
            for _ in range(self.ransac_iters):
                if len(remaining) < 2:
                    break
                idx1, idx2 = np.random.choice(len(remaining), 2, replace=False)
                p1, p2 = remaining[idx1], remaining[idx2]
                
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                dist = np.hypot(dx, dy)
                if dist < 0.02:
                    continue
                
                # Line equation: a*x + b*y + c = 0
                a = -dy / dist
                b = dx / dist
                c = - (a * p1[0] + b * p1[1])
                
                # Inliers
                dists = np.abs(a * remaining[:,0] + b * remaining[:,1] + c)
                inliers = remaining[dists < self.ransac_threshold]
                
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_model = (a, b, c)
            
            if len(best_inliers) >= self.min_inliers:
                # Fit endpoints with PCA
                mean_pt = np.mean(best_inliers, axis=0)
                centered = best_inliers - mean_pt
                if len(centered) < 2:
                    continue
                cov = np.cov(centered.T)
                if cov[0,0] + cov[1,1] < 1e-6:  # Degenerate
                    continue
                _, dir_vec = np.linalg.eigh(cov)
                dir_vec = dir_vec[:, -1]  # Longest direction
                
                projs = np.dot(centered, dir_vec)
                start_pt = mean_pt + dir_vec * projs.min()
                end_pt = mean_pt + dir_vec * projs.max()
                length = np.linalg.norm(end_pt - start_pt)
                
                if length > 0.03:  # At least 3cm
                    angle = np.arctan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0])
                    lines.append({
                        'start': start_pt, 'end': end_pt,
                        'length': length, 'angle': angle,
                        'center': mean_pt
                    })
                    
                    # Remove inliers from remaining
                    dists = np.abs(best_model[0] * remaining[:,0] + best_model[1] * remaining[:,1] + best_model[2])
                    remaining = remaining[dists >= self.ransac_threshold]
        
        return lines
    
    def publish_markers(self, lines):
        """Green for short, red for long edges"""
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i, line in enumerate(lines):
            m = Marker()
            m.header.frame_id = 'ebot_base'  # Consistent with nav
            m.header.stamp = now
            m.ns = 'lines'
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.02
            m.color.a = 1.0
            
            # Color: Green short, Red long
            if line['length'] <= self.max_short:
                m.color.g = 1.0  # Green
            else:
                m.color.r = 1.0  # Red
            
            p1 = Point(x=float(line['start'][0]), y=float(line['start'][1]), z=0.0)
            p2 = Point(x=float(line['end'][0]), y=float(line['end'][1]), z=0.0)
            m.points = [p1, p2]
            markers.markers.append(m)
        
        self.marker_pub.publish(markers)
        self.get_logger().debug(f'📈 Published {len(lines)} markers')
    
    def is_duplicate(self, x, y):
        """Simple duplicate check (last 3 detections)"""
        for px, py in self.detected_positions[-3:]:
            if np.hypot(x - px, y - py) < 0.30:
                return True
        return False
    
    def publish_force(self, world_x, world_y, shape):
        """FORCE PUBLISH - This will trigger nav pause"""
        msg = String()
        msg.data = f"{shape},{world_x:.3f},{world_y:.3f}"  # Exact format from nav
        self.status_pub.publish(msg)
        
        self.detected_positions.append((world_x, world_y))
        
        self.get_logger().info('═' * 60)
        self.get_logger().info(f'🚨 FORCE DETECTED: {shape}')
        self.get_logger().info(f'📍 World Pos: ({world_x:.3f}, {world_y:.3f})')
        self.get_logger().info(f'📤 EXACT MSG: "{msg.data}" ← Nav will receive this')
        self.get_logger().info(f'🛑 Robot SHOULD STOP NOW (paused=True in nav)')
        self.get_logger().info('═' * 60)

def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

