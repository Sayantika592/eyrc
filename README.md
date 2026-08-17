# eyrc

# Krishi Cobot — e-Yantra Robotics Competition 2025–26

Team KC#4406 | Qualified for Stage 2 (top 100 of 400+ participating teams) | Hosted by IIT Bombay, sponsored by the Ministry of Education

## Overview

Krishi Cobot is an autonomous farm-robotics challenge: a differential-drive eBot has to navigate a simulated field, locate and classify produce, and hand off pick-and-place manipulation to a robotic arm — combining navigation, perception, and manipulation in one pipeline. This repo covers our Task 2A (navigation) and Task 2B (perception + manipulation) submissions, built on ROS 2.

## Task 2A — Autonomous Navigation & Shape Detection

`ebot_nav_task2a.py`
- A* global path planning over a live occupancy grid (world ↔ map coordinate conversion, 4-connected grid search), with pure-pursuit-style lookahead tracking to follow the planned path.
- LiDAR-based obstacle-avoidance bias blended into the steering command, computed from `/scan` ranges within a configurable obstacle distance.
- Distance-lock return logic: tracks the robot's maximum distance from its start pose and automatically switches into a locked return-to-start mode once that distance begins decreasing.

`shape_detector_task2a.py`
- RANSAC-based line/edge detection over 2D LiDAR scans (tuned inlier threshold and iteration count) to detect and localize shape markers along the route, publishing detections as visualization markers.

## Task 2B — Perception & Manipulation

`aruco_detect.py`
- ArUco marker detection fused with RGB + depth (PointCloud2) to compute and broadcast TF frames for semantically labeled objects (e.g. fertilizer can, AGV/vehicle), using camera intrinsics for pose estimation.

`bad_fruit.py`
- HSV + contour-based fruit health classification: distinguishes healthy (grey) from bad (purple) produce via ROI-restricted color segmentation and contour-area filtering, publishing TF frames for detected fruit tops.

`controller.py`
- A custom ROS 2 action server (`MoveArm.action`: target pose in → success flag out) that servos the manipulator toward a goal pose using TF lookups and normalized-interpolation (nlerp) trajectory smoothing, publishing velocity commands to drive the arm.

`task.py`
- Task-orchestration node sequencing the full pipeline: locates the can, AGV, fruits, and bin via TF, sends pose goals to the arm action server, and handles simulated grasp/release through Gazebo model attach/detach.

## Tech stack

ROS 2 · OpenCV · NumPy · SciPy · cv_bridge · tf2 · Gazebo

## Result

Qualified for Stage 2 of e-Yantra Robotics Competition 2025–26 (IIT Bombay), placing in the top 100 of 400+ participating teams nationally.
