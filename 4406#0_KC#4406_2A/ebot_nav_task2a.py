#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from std_msgs.msg import String

POSE_TOL = 0.18
YAW_TOL = math.radians(10)
TURN_SPEED = 0.5
LIN_SPEED = 0.5

WAYPOINTS = [
    [-1.53, -6.61, 0.0],      # WP3 Start
    [0.26, -1.95, 1.57],      # WP1
    [-1.48, -0.67, -1.57],    # WP2
    [-1.53, -6.61, -1.57],    # WP3 End
]

class EBotNav(Node):
    def __init__(self):
        super().__init__('ebot_nav')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.pose_x = self.pose_y = self.yaw = 0.0
        self.phase = 0
        self.odom_initialized = False
        # wp3 -> wp1 distance for buffer logic
        self.wp3_wp1_dist = math.sqrt(
            (WAYPOINTS[1][0] - WAYPOINTS[0][0]) ** 2 + (WAYPOINTS[1][1] - WAYPOINTS[0][1]) ** 2)
        self.buffer_dist = 0.1 * self.wp3_wp1_dist
        self.segment_dist = 0.3 * self.wp3_wp1_dist
        self.segment_dist_new = 0.35 * self.wp3_wp1_dist
        self.dist_min = None
        self.initial_wp2_dist = None
        self.min_found = False
        
        self.paused = False
        self.last_pause_time = None
        self.det_sub = self.create_subscription(
            String, '/detection_status', self.detector_cb, 10)
        
        self.get_logger().info(f"Buffer distance: {self.buffer_dist:.2f}m")
        self.timer = self.create_timer(0.08, self.loop)
        
    def detector_cb(self, msg):
        self.get_logger().info(f"detector_cb called with: {msg.data}")
        self.get_logger().info(f"Navigation paused by detector: {msg.data}")
        self.paused = True
        self.last_pause_time = self.get_clock().now().nanoseconds / 1e9

    def odom_cb(self, msg):
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        # Dynamically set start buffer position on first odometry received
        if not self.odom_initialized:
            self.start_x = self.pose_x
            self.start_y = self.pose_y
            self.odom_initialized = True
            self.get_logger().info(f"Initialized buffer start at: x={self.start_x:.2f}, y={self.start_y:.2f}")

    def loop(self):
    
        if self.paused:
            # Stop robot while paused
            # Resume after 2 seconds (or adjust as needed)
            if self.last_pause_time and self.get_clock().now().nanoseconds / 1e9 - self.last_pause_time > 2.0:
                self.get_logger().info("Navigation RESUMED after shape detection pause.")
                self.paused = False
            else:
                stop = Twist()
                self.cmd_pub.publish(stop)
                return
        
        if not getattr(self, 'odom_initialized', False):
            return  # wait for first odom so we have a correct start_x/y

        twist = Twist()

        if self.phase == 0:
            d = math.sqrt((self.pose_x - self.start_x)**2 + (self.pose_y - self.start_y)**2)
            if d < self.buffer_dist:
                twist.linear.x = LIN_SPEED
                twist.angular.z = 0.0
            else:
                self.get_logger().info("Phase 0→1: Buffer complete, turning 90°")
                self.phase = 1

        elif self.phase == 1:
            target_yaw = 0.0
            ang_err = self.normalize_angle(target_yaw - self.yaw)
            twist.linear.x = 0.0
            if abs(ang_err) > YAW_TOL:
                twist.angular.z = math.copysign(TURN_SPEED, ang_err)
            else:
                self.get_logger().info("Phase 1→2: Turn complete, driving straight to segment distance for crossing")
                self.phase = 2
                self.segment_start_x = self.pose_x
                self.segment_start_y = self.pose_y

        elif self.phase == 2:
            d = math.sqrt((self.pose_x - self.segment_start_x)**2 + (self.pose_y - self.segment_start_y)**2)
            twist.linear.x = LIN_SPEED
            twist.angular.z = 0.0
            if d > self.segment_dist:
                self.get_logger().info("Phase 2→3: 0.3*wp3_wp1_dist traveled, turning 90°")
                self.phase = 3

        elif self.phase == 3:
            target_yaw = math.pi/2 + math.radians(5)
            ang_err = self.normalize_angle(target_yaw - self.yaw)
            twist.linear.x = 0.0
            if abs(ang_err) > YAW_TOL:
                twist.angular.z = math.copysign(TURN_SPEED, ang_err)
            else:
                self.get_logger().info("Phase 3→4: Turn complete, moving to WP1")
                self.phase = 4

        elif self.phase == 4:
            curr_dist = math.sqrt(
                (self.pose_x - WAYPOINTS[1][0])**2 + (self.pose_y - WAYPOINTS[1][1])**2)
            twist.linear.x = LIN_SPEED
            twist.angular.z = 0.0
            if curr_dist < POSE_TOL:
                self.get_logger().info("Phase 4→5: At WP1, tracking towards WP2")
                self.phase = 5
                self.initial_wp2_dist = math.sqrt(
                    (self.pose_x - WAYPOINTS[2][0])**2 + (self.pose_y - WAYPOINTS[2][1])**2)
                self.dist_min = self.initial_wp2_dist
                self.min_found = False
                self.segment2_start_x = self.pose_x
                self.segment2_start_y = self.pose_y

        elif self.phase == 5:
            d = math.sqrt((self.pose_x - self.segment2_start_x)**2 + (self.pose_y - self.segment2_start_y)**2)
            twist.linear.x = LIN_SPEED
            twist.angular.z = 0.0
            if d > 1.5 * self.initial_wp2_dist:
                self.get_logger().info("Phase 5→6: 1.5x initial WP1->WP2 distance, turning 90°")
                self.phase = 6

        elif self.phase == 6:
            target_yaw = - math.pi
            ang_err = self.normalize_angle(target_yaw - self.yaw)
            twist.linear.x = 0.0
            if abs(ang_err) > YAW_TOL:
                twist.angular.z = math.copysign(TURN_SPEED, ang_err)
            else:
                self.get_logger().info("Phase 6→7: Turn complete, driving to 0.3*wp3_wp1_dist for next crossing")
                self.phase = 7
                self.segment3_start_x = self.pose_x
                self.segment3_start_y = self.pose_y

        elif self.phase == 7:
            d = math.sqrt((self.pose_x - self.segment3_start_x)**2 + (self.pose_y - self.segment3_start_y)**2)
            twist.linear.x = LIN_SPEED
            twist.angular.z = 0.0
            if d > self.segment_dist_new:
                self.get_logger().info("Phase 7→8: 0.3*wp3_wp1_dist traveled (corridor), turning 90° to -y for WP2")
                self.phase = 8

        elif self.phase == 8:
            target_yaw = -math.pi/2 + math.radians(5)
            ang_err = self.normalize_angle(target_yaw - self.yaw)
            twist.linear.x = 0.0
            if abs(ang_err) > YAW_TOL:
                twist.angular.z = math.copysign(TURN_SPEED, ang_err)
            else:
                self.get_logger().info("Phase 8→9: Turn to -y complete, heading to WP2")
                self.phase = 9

        elif self.phase == 9:
            curr_dist = math.sqrt((self.pose_x - WAYPOINTS[2][0])**2 + (self.pose_y - WAYPOINTS[2][1])**2)
            twist.linear.x = LIN_SPEED
            twist.angular.z = 0.0
            if curr_dist < POSE_TOL:
                self.get_logger().info("Phase 9→10: At WP2! [LOG: Reached WP2]")
                self.phase = 10
                
        elif self.phase == 10:
             overshoot = math.radians(7)  # 5 degrees in radians
             target_yaw = WAYPOINTS[3][2] + overshoot
             ang_err = self.normalize_angle(target_yaw - self.yaw)
             twist.linear.x = 0.0
             if abs(ang_err) > YAW_TOL:
                 twist.angular.z = math.copysign(TURN_SPEED, ang_err)
             else:
                 self.get_logger().info("Phase 10→11: Overshoot turn complete, heading straight to WP3")
                 self.phase = 11

        elif self.phase == 11:
            curr_dist = math.sqrt((self.pose_x - WAYPOINTS[3][0])**2 + (self.pose_y - WAYPOINTS[3][1])**2)
            twist.linear.x = LIN_SPEED
            twist.angular.z = 0.0
            if curr_dist < POSE_TOL:
                self.get_logger().info("Phase 11→12: At WP3. [LOG: Reached WP3]")
                self.phase = 12

        # All done: stop
        elif self.phase == 11:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))


def main(args=None):
    rclpy.init(args=args)
    node = EBotNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

