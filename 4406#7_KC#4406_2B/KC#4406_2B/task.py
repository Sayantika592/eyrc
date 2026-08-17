#!/usr/bin/env python3
# This script will publish to /delta_twist_cmds.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient, CancelResponse, GoalResponse
from task2_control.action import MoveArm
from linkattacher_msgs.srv import AttachLink, DetachLink
import time
import tf2_ros
import tf_transformations
import numpy as np
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
import threading

class TaskNode(Node):
    def __init__(self):
        super().__init__('Task_Node')
        
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._client = ActionClient(self, MoveArm, 'MoveArm')
        self._attach_cli = self.create_client(AttachLink, '/attach_link')
        while not self._attach_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(' Waiting for /attach_link service...')
        self.get_logger().info(' /attach_link service available.')

        self._detach_cli = self.create_client(DetachLink, '/detach_link')
        while not self._detach_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(' Waiting for /detach_link service...')
        self.get_logger().info(' /detach_link service available.')

        self.task_success = False
        self.can_pose = np.zeros((7,),dtype=np.float32)
        self.agv_pose = np.zeros((7,),dtype=np.float32)
        self.fruit_poses = []
        self.bin_pose = np.zeros((7,)) #np.array([-0.806 , 0.010 ,0.182 ,-0.684 , 0.726 , 0.050  ,0.008])
        self.pose_ee = np.zeros((7,),dtype=np.float32)
        
    def looktf(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link',  # target_frame
                'ee_link',    # source_frame
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5) # Wait up to 0.5s
            )
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"Waiting for 'can' pose: {ex}")
            return False

        self.pose_ee[0] = trans.transform.translation.x
        self.pose_ee[1] = trans.transform.translation.y
        self.pose_ee[2] = trans.transform.translation.z
        self.pose_ee[3] = trans.transform.rotation.x
        self.pose_ee[4] = trans.transform.rotation.y
        self.pose_ee[5] = trans.transform.rotation.z
        self.pose_ee[6] = trans.transform.rotation.w

        return True

    def attach(self, model1, link1, model2, link2):
        req = AttachLink.Request()
        req.model1_name = model1
        req.link1_name = link1
        req.model2_name = model2
        req.link2_name = link2

        self.get_logger().info(f" Attaching {model1}/{link1} ↔ {model2}/{link2}")
        future = self._attach_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp is not None:
            self.get_logger().info(" Attach service completed successfully.")
        else:
            self.get_logger().error(" Attach service failed.")

    def detach(self, model1, link1, model2, link2):
        req = DetachLink.Request()
        req.model1_name = model1
        req.link1_name = link1
        req.model2_name = model2
        req.link2_name = link2

        self.get_logger().info(f" Detaching {model1}/{link1} ↔ {model2}/{link2}")
        future = self._detach_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp is not None:
            self.get_logger().info(" Detach service completed successfully.")
        else:
            self.get_logger().error(" Detach service failed.")

    def send_goal(self, pose):
        # Wait for action server
        self.get_logger().info("Waiting for action server...")
        self._client.wait_for_server()

        # Construct goal
        goal_msg = MoveArm.Goal()
        goal_msg.x, goal_msg.y, goal_msg.z = pose[:3]
        goal_msg.qx, goal_msg.qy, goal_msg.qz, goal_msg.qw = pose[3:]

        self.get_logger().info(f"Sending goal: {pose}")


        send_future = self._client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_cb
        )
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected.")
            return False

        self.get_logger().info("Goal accepted.")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.success:
            self.get_logger().info("Goal executed successfully.")
            return True
        else:
            self.get_logger().warn("Goal execution failed.")
            return False
        
    def feedback_cb(self, feedback):
        pass

    def _get_can_pose(self):
        try:
            can_tf = self.tf_buffer.lookup_transform(
                'base_link',  # target_frame
                'eYRC#4406_fertilizer_can_s',    # source_frame
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5) # Wait up to 0.5s
            )
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"Waiting for 'can' pose: {ex}")
            return False # Return False on failure
        
        self.can_pose[0] = can_tf.transform.translation.x
        self.can_pose[1] = can_tf.transform.translation.y
        self.can_pose[2] = can_tf.transform.translation.z
        self.can_pose[3] = can_tf.transform.rotation.x
        self.can_pose[4] = can_tf.transform.rotation.y
        self.can_pose[5] = can_tf.transform.rotation.z
        self.can_pose[6] = can_tf.transform.rotation.w

        print('can_pose : ',self.can_pose)
        return True # Return True on success

    def _get_agv_pose(self):
        try:
            agv_tf = self.tf_buffer.lookup_transform(
                'base_link',  # target_frame
                'vehicle_s',    # source_frame
                Time(),
                timeout=rclpy.duration.Duration(seconds=0.5) # Wait up to 0.5s
            )
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"Waiting for 'AGV' pose: {ex}")
            return False # Return False on failure
        
        self.agv_pose[0] = agv_tf.transform.translation.x
        self.agv_pose[1] = agv_tf.transform.translation.y
        self.agv_pose[2] = agv_tf.transform.translation.z
        self.agv_pose[3] = agv_tf.transform.rotation.x
        self.agv_pose[4] = agv_tf.transform.rotation.y
        self.agv_pose[5] = agv_tf.transform.rotation.z
        self.agv_pose[6] = agv_tf.transform.rotation.w

        print('agv_pose : ',self.agv_pose)
        return True # Return True on success
    
    def _get_fruits_poses(self):
        for i in range(3):
            try:

                fruit_tf = self.tf_buffer.lookup_transform(
                    'base_link',  # target_frame
                    f'grey_fruit_top_{i}',    # source_frame
                    Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5) # Wait up to 0.5s
                )
                self.fruit_poses.append(np.array([
                    fruit_tf.transform.translation.x,
                    fruit_tf.transform.translation.y,
                    fruit_tf.transform.translation.z,
                    fruit_tf.transform.rotation.x,
                    fruit_tf.transform.rotation.y,
                    fruit_tf.transform.rotation.z,
                    fruit_tf.transform.rotation.w]))
            except tf2_ros.TransformException as ex:
                self.get_logger().warn(f"Waiting for 'AGV' pose: {ex}")
                return False # Return False on failure
    
        return True
    
    def _get_bin_pose(self):
        try:

            agv_tf = self.tf_buffer.lookup_transform(
                'base_link',  # target_frame
                'bin_link',    # source_frame
                Time(),
                timeout=rclpy.duration.Duration(seconds=0.5) # Wait up to 0.5s
            )
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"Waiting for 'AGV' pose: {ex}")
            return False # Return False on failure
        
        self.bin_pose[0] = agv_tf.transform.translation.x
        self.bin_pose[1] = agv_tf.transform.translation.y
        self.bin_pose[2] = agv_tf.transform.translation.z
        self.bin_pose[3] = agv_tf.transform.rotation.x
        self.bin_pose[4] = agv_tf.transform.rotation.y
        self.bin_pose[5] = agv_tf.transform.rotation.z
        self.bin_pose[6] = agv_tf.transform.rotation.w

        return True
    
    def run(self):
        while not self.looktf():
            if not rclpy.ok():
                return
            time.sleep(0.1)
        self.get_logger().info("Waiting for 'can' pose...")

        while not self._get_can_pose():
            if not rclpy.ok():
                return
            time.sleep(0.1)
        
        self.get_logger().info("Got 'can' pose. Waiting for 'AGV' pose...")
        while not self._get_agv_pose():
            if not rclpy.ok():
                return
            time.sleep(0.1)

        self.get_logger().info("Got 'agv' pose. Waiting for 'fruits' pose...")
        while not self._get_fruits_poses():
            if not rclpy.ok():
                return
            time.sleep(0.1)

        self.get_logger().info("Got 'fruits' poses. Waiting for 'bin' pose...")
        while not self._get_bin_pose():
            if not rclpy.ok():
                return
            time.sleep(0.1)
        self.get_logger().info("Got all poses. Starting arm movement.")

        pose_safe = self.can_pose + np.array([-0.06,0,0,0,0,0,0])
        self.send_goal(pose_safe.astype(float))
        self.attach('fertiliser_can','body','ur5','wrist_3_link')
        pose_safe2 = self.agv_pose + np.array([0,0,0.2,0,0,0,0])
        self.send_goal(pose_safe2.astype(float))
        self.detach('fertiliser_can','body','ur5','wrist_3_link')
        # self.send_goal(np.array([-0.314, -0.332, 0.457,  0.707,  0.028,  0.034,  0.707]))
        # self.send_goal(np.array([0.120, -0.109, 0.445, 0.501, 0.497, 0.503, 0.499]))
        # self.send_goal(np.array([ 0.356,  0.010, 0.522,  0.480,  0.519,  0.524,  0.475]))
        # self.send_goal(np.array([-0.159,  0.501, 0.415,  0.029,  0.997,  0.045,  0.033]))
        # self.send_goal(np.array([0.120, 0.209, 0.445, 0.501, 0.497, 0.503, 0.499]))
        # self.send_goal(np.array([ 0.356,  0.010, 0.522,  0.480,  0.519,  0.524,  0.475]))
        # self.send_goal(np.array([-0.806,  -0.010, 0.182, -0.684,  0.726,  0.050,  0.008]))
        # self.send_goal(np.array([0.120, -0.109, 0.445, 0.501, 0.497, 0.503, 0.499]))
        # self.send_goal(np.array([0.020, 0.109, 0.445, *[0.5*[0.708, -0.003, 0.706, -0.001]+0.5*[0.501, 0.497, 0.503, 0.499]]]))
        # self.send_goal(np.array([[ 0.356,  0.010, 0.522,  0.480,  0.519,  0.524,  0.475]]))
        # self.send_goal(np.array([ 0.356,  0.010, 0.522,  0.480,  0.519,  0.524,  0.475]))
        for i in range(3):
            # pose_0 = np.array([-0.159,  0.501, 0.515, 0.501, 0.497, 0.503, 0.499])
            pose_1 = self.fruit_poses[i] + np.array([0,0,0.05,0,0,0,0])
            # self.send_goal((pose_1+np.array([0.1,0,0.2,0,0,0,0])).astype(float))
            self.send_goal(pose_1.astype(float))
            self.attach('bad_fruit','body','ur5','wrist_3_link')
            self.send_goal((pose_1+np.array([0.0,0,0.1,0,0,0,0])).astype(float))
            self.send_goal((pose_1+np.array([0.0,0,0.01,0,0,0,0])).astype(float))
            self.send_goal((pose_1+np.array([0.0,0,0.1,0,0,0,0])).astype(float))
            pose_2 = self.bin_pose 
            self.send_goal((pose_2+np.array([0.0,0,0.1,0,0,0,0])).astype(float))
            #self.send_goal(pose_2.astype(float))
            self.detach('bad_fruit','body','ur5','wrist_3_link')

        
def main(args=None):
    rclpy.init(args=args)
    node = TaskNode()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()