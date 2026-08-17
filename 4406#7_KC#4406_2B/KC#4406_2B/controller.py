#!/usr/bin/env python3
# This script will publish to /delta_twist_cmds.

import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R_
from geometry_msgs.msg import Twist
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from task2_control.action import MoveArm
import time
import tf2_ros
import tf_transformations
import numpy as np
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time


class ArmServoNode(Node):
    def __init__(self):
        super().__init__('arm_servo_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)

        self.pose_ee = np.zeros((7,))
        self.pose_ee_prev = np.zeros((7,))

        self._action_server = ActionServer(
            self,
            MoveArm,
            'MoveArm',                   
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.twist_cmd = Twist()
        self.twist_cmd.linear.x = 0.0 
        self.twist_cmd.linear.y = 0.0
        self.twist_cmd.linear.z = 0.0
        self.twist_cmd.angular.x = 0.0
        self.twist_cmd.angular.y = 0.0
        self.twist_cmd.angular.z = 0.0
        
        self.count = 0
        self.isReached = False
        self.err_res = np.zeros((6,))
        self.err_prev = np.zeros((6,))
        self.Kp = 3
        self.Ki = 0.16
        self.Kd = 0.2

        self.Kp_o = 3
        self.Ki_o = 0.16
        self.Kd_o = 0.2

        self.act_max = 1e9
        self.threshold = 0.05
        self.threshold_o = 0.05
        self.err_max = 1.5e7
        self.err_o_max = 1.5e8
        self.err_ori = np.zeros((3,))
        self.err_prev_ori = np.zeros((3,))
        self.act_val_prev = np.zeros((6,))
        self.middle_pts = None
        self.t_count = 0
        self.nxt_pt = True
        self.err_c = 0
        self.finalPoseReached = False
        self.s = 0.15

    def goal_callback(self, goal_request):
        self.get_logger().info(
            f"Received goal -> position: ({goal_request.x:.3f}, {goal_request.y:.3f}, {goal_request.z:.3f}), "
            f"orientation: ({goal_request.qx:.3f}, {goal_request.qy:.3f}, {goal_request.qz:.3f}, {goal_request.qw:.3f})"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().warn("Goal cancelled")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        goal = goal_handle.request

        pose = np.array([goal.x, goal.y, goal.z, goal.qx, goal.qy, goal.qz, goal.qw], dtype=float)

        success = False
        while not success and rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().warn("Goal cancelled by client.")
                return MoveArm.Result(success=False)
            success = self.publish_twist(pose)
            time.sleep(0.02)

        if success:
            goal_handle.succeed()
            result = MoveArm.Result()
            result.success = True
            self.get_logger().info(f"Goal execution completed successfully.")
            return result

    def looktf(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link',  # target_frame
                'wrist_3_link',    # source_frame
                Time()
            )
        except Exception as ex:
            self.get_logger().warn_throttle(2000, f"TF lookup failed: {ex}")
            return

        self.pose_ee[0] = trans.transform.translation.x
        self.pose_ee[1] = trans.transform.translation.y
        self.pose_ee[2] = trans.transform.translation.z
        self.pose_ee[3] = trans.transform.rotation.x
        self.pose_ee[4] = trans.transform.rotation.y
        self.pose_ee[5] = trans.transform.rotation.z
        self.pose_ee[6] = trans.transform.rotation.w

    def calc_val(self, point):
        for i in range(3):
            err = point[i] - self.pose_ee[i]
            self.err_res[i] += err
            self.err_res[i] = np.clip(self.err_res[i], -self.err_max, self.err_max)
            act = (self.Kp * err) + (self.Kd * (err - self.err_prev[i])) + (self.Ki * (self.err_res[i]))
            self.err_prev[i] = err
            act = np.clip(act, -self.act_max, self.act_max)
            setattr(self.twist_cmd.linear, ['x', 'y', 'z'][i], act)

        quat_ee = self.pose_ee[3:]
        quat_pt = point[3:]
        quat_ee_inv = tf_transformations.quaternion_inverse(quat_ee)
        quat_err = tf_transformations.quaternion_multiply(quat_pt, quat_ee_inv)
        err_term = np.sign(quat_err[-1]) * np.array(quat_err[:3])
        self.err_ori += err_term
        self.err_ori = np.clip(self.err_ori, -self.err_o_max, self.err_o_max)
        act = self.Kp_o * err_term + self.Ki_o * self.err_ori + self.Kd_o * (self.act_val_prev[-3:])
        for i in range(3, 6):
            act_val = np.clip(act[i - 3], -self.act_max, self.act_max)
            setattr(self.twist_cmd.angular, ['x', 'y', 'z'][i - 3], act_val)
        self.act_val_prev = np.clip(act, -self.act_max, self.act_max)

    def updateReach(self, point, t1, t2):
        err_vec = self.pose_ee - point
        if (np.abs(err_vec[:3]) < t1).all() and (np.abs(err_vec[3:]) < t2).all():
            return True
        else:
            return False

    def _add_distance_pt(self,s,x):
        t = np.array(x[:3], dtype=np.float64).flatten()
        R_mat = R_.from_quat(x[3:]).as_matrix()
        p_local = np.array([0.0, 0.0, -float(s)], dtype=np.float64)
        p_global = t + R_mat.dot(p_local).flatten()
        q_global = x[3:]
        return np.hstack([p_global,q_global])

    def nlerp(self, X1, X2, num_steps):
        X_add = self._add_distance_pt(self.s,X1)
        X_step = np.linspace(X_add[:3], X2[:3], num_steps)
        q1 = X_add[3:]
        q2 = X2[3:]
        q1 = np.asarray(q1)
        q2 = np.asarray(q2)
        dot = np.dot(q1, q2)
        if dot < 0.0:
            q2 = -q2
        t_steps = np.linspace(0.0, 1.0, num_steps)
        t_broadcast = t_steps[:, np.newaxis]
        q_interp = (1.0 - t_broadcast) * q1 + t_broadcast * q2
        norms = np.linalg.norm(q_interp, axis=1, keepdims=True)
        q_step = q_interp / (norms + 1e-8)

        X_out = np.hstack([X_step, q_step])
        return X_out
    

    def publish_twist(self, point):
        self.pose_ee_prev = self.pose_ee
        self.looktf()

        if self.isReached:
            self.get_logger().info(f"Checkpoint {self.count + 1} reached")
            self.isReached = False
            for i in range(3):
                setattr(self.twist_cmd.linear, ['x', 'y', 'z'][i], 0.0)
                setattr(self.twist_cmd.angular, ['x', 'y', 'z'][i], 0.0)
            self.act_val_prev = np.zeros((6,))
            self.err_res = np.zeros((6,))
            self.err_prev = np.zeros((6,))
            self.err_ori = np.zeros((3,))
            self.err_prev_ori = np.zeros((3,))
            self.count += 1
            self.get_logger().info("Over")
            self.pub.publish(self.twist_cmd)
            self.nxt_pt = True
            return True

        if self.nxt_pt:
            self.get_logger().info(f"Next Checkpoint {point}")
            self.traj = self.nlerp(self.pose_ee, point, 12)
            self.nxt_pt = False
            self.t_count = 0

        self.calc_val(self.traj[self.t_count])
        if self.t_count == len(self.traj)-1:
            t1 = 0.06
            t2 = 0.06
        else:
            t1=0.15
            t2=0.15

        if self.updateReach(self.traj[self.t_count], t1, t2):
            if self.err_c > 20:
                self.err_c = 0
            self.t_count += 1
            self.err_res = np.zeros((6,))
            self.err_prev = np.zeros((6,))
            self.err_ori = np.zeros((3,))
            self.err_prev_ori = np.zeros((3,))
            self.get_logger().warn(f" subpoint {self.traj[self.t_count - 1]} reached")

        self.pub.publish(self.twist_cmd)

        if self.t_count >= len(self.traj):
            self.t_count = 0
            self.isReached = True
            self.get_logger().warn(" entered final point")

        return False


def main(args=None):
    rclpy.init(args=args)
    node = ArmServoNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
