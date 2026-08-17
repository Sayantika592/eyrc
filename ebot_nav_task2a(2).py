#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import heapq
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion


def norm(a):
    return math.atan2(math.sin(a), math.cos(a))


# ---------------- A* ----------------
def astar(grid, start, goal):
    h = lambda p: abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    pq = [(h(start), start, None)]
    visited = {}

    while pq:
        _, cur, parent = heapq.heappop(pq)
        if cur in visited:
            continue
        visited[cur] = parent
        if cur == goal:
            break
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if grid[nx, ny] == 0:
                    heapq.heappush(pq, (h((nx,ny)), (nx,ny), cur))

    path = []
    cur = goal
    while cur:
        path.append(cur)
        cur = visited.get(cur)
    return path[::-1]


# ---------------- NODE ----------------
class AutonomousNav(Node):

    def __init__(self):
        super().__init__('autonomous_nav_distance_lock')

        # Parameters
        self.lookahead = 0.6
        self.max_lin = 0.35
        self.max_ang = 0.6
        self.obs_dist = 0.7

        # Robot state
        self.x = self.y = self.yaw = 0.0
        self.start_pose = None

        # Distance logic
        self.max_dist_from_start = 0.0
        self.return_locked = False

        # Planning
        self.map = None
        self.scan = None
        self.path = []
        self.path_idx = 0
        self.planned = False

        self.goal_world = (2.0, -2.0)

        # ROS
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Autonomous navigation with distance-lock return started")

    # ---------------- Callbacks ----------------
    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        if self.start_pose is None:
            self.start_pose = (self.x, self.y)

    def scan_cb(self, msg):
        self.scan = msg

    def map_cb(self, msg):
        if self.planned:
            return
        self.map = msg
        self.plan_path()

    # ---------------- Mapping ----------------
    def world_to_map(self, x, y):
        res = self.map.info.resolution
        ox = self.map.info.origin.position.x
        oy = self.map.info.origin.position.y
        return int((y-oy)/res), int((x-ox)/res)

    def map_to_world(self, r, c):
        res = self.map.info.resolution
        ox = self.map.info.origin.position.x
        oy = self.map.info.origin.position.y
        return ox + c*res, oy + r*res

    def plan_path(self):
        grid = np.array(self.map.data).reshape(
            self.map.info.height, self.map.info.width)
        grid = np.where(grid > 50, 1, 0)

        s = self.world_to_map(self.x, self.y)
        g = self.world_to_map(*self.goal_world)

        path_cells = astar(grid, s, g)
        self.path = [self.map_to_world(r,c) for r,c in path_cells]
        self.planned = True
        self.get_logger().info("Global path planned")

    # ---------------- Helpers ----------------
    def get_lookahead(self):
        for i in range(self.path_idx, len(self.path)):
            if math.hypot(self.path[i][0]-self.x,
                          self.path[i][1]-self.y) > self.lookahead:
                self.path_idx = i
                return self.path[i]
        return self.path[-1]

    def obstacle_bias(self):
        if not self.scan:
            return 0.0
        ranges = np.array(self.scan.ranges)
        angles = np.linspace(self.scan.angle_min,
                              self.scan.angle_max,
                              len(ranges))
        mask = ranges < self.obs_dist
        if not np.any(mask):
            return 0.0
        return -0.6 * np.sum(np.sin(angles[mask]))

    # ---------------- MAIN LOOP ----------------
    def control_loop(self):
        if not self.planned:
            return

        cmd = Twist()

        # Distance to start
        dist = math.hypot(self.start_pose[0]-self.x,
                          self.start_pose[1]-self.y)

        # Track maximum distance
        self.max_dist_from_start = max(self.max_dist_from_start, dist)

        # 🔒 Lock return once distance starts decreasing
        if dist < self.max_dist_from_start - 0.2:
            self.return_locked = True

        # ---------------- RETURN MODE ----------------
        if self.return_locked:
            dx = self.start_pose[0] - self.x
            dy = self.start_pose[1] - self.y
            yaw_err = norm(math.atan2(dy, dx) - self.yaw)

            cmd.linear.x = self.max_lin * max(0.3, 1.0 - abs(yaw_err))
            cmd.angular.z = max(-self.max_ang,
                                min(self.max_ang, 1.2 * yaw_err))

            self.cmd_pub.publish(cmd)

            if dist < 0.25:
                self.cmd_pub.publish(Twist())
                self.get_logger().info("Returned to start successfully")
            return

        # ---------------- NORMAL PATH FOLLOW ----------------
        target = self.get_lookahead()
        dx = target[0] - self.x
        dy = target[1] - self.y
        yaw_err = norm(math.atan2(dy, dx) - self.yaw)

        curvature = 2.0 * math.sin(yaw_err) / self.lookahead
        bias = self.obstacle_bias()

        cmd.linear.x = self.max_lin * max(0.4, 1.0 - abs(yaw_err))
        cmd.angular.z = max(-self.max_ang,
                            min(self.max_ang,
                                curvature + bias))

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    rclpy.spin(AutonomousNav())
    rclpy.shutdown()


if __name__ == '__main__':
    main()

