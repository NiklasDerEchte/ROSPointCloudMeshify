# Für Unity Docu

## Unity-Integration

### Start Docker-Server

- cd ~/UnityRosPlugins/Unity-Robotics-Hub/tutorials/ros_unity_integration && docker run -it --rm -p 10000:10000 humble_unity_environment /bin/bash

### Start Unity-ROS Server

- ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0

# Basics

## Installation ROS2-Humble

- Follow: https://docs.ros.org/en/humble/Installation.html

### Test-Commands for ROS2

- ros2 run demo_nodes_cpp talker
- ros2 run demo_nodes_cpp listener

## Installation Zivid

- Zivid Studio https://support.zivid.com/en/latest/getting-started/software-installation.html
  - Computer muss im selben Netzwerk sein wie die Camera (172.28.60.XXX)
- Zivid Ros https://github.com/zivid/zivid-ros

### Launch Zivid sample

- ros2 launch zivid_samples sample_with_rviz.launch sample:=sample_capture_cpp

# [C++] ROS2-Humble-PCL

## Installation PCL

- sudo apt install libpcl-dev
- sudo apt install ros-humble-pcl-conversions ros-humble-pcl-ros
- sudo apt install ros-humble-visualization-msgs
- sudo apt install libpcap-dev

## Import modules

- source ros_pcl_module/install/setup.bash
- source ros_zivid_driver/install/setup.bash

## Start Object detection with PCL

- ros2 run object_detection object_detection_node

### Visualizer

- ros2 run object_detection object_detection_visualizer_node

### FirstTry Bounding Boxes

- ros2 run object_detection object_detection_geometryc_visualizer_node

### Mesh-Creator

- ros2 run object_detection object_detection_mesh_creator_node

# [Python] ROS2-Humble-Open3D

## Installation

### Python

- apt install ros-humble-rclpy
- pip install sensor_msgs_py

### Open3D

- pip install open3d
- pip install open3d-cpu
- pip install open3d_ros_helper

## Import modules

- source ros_open3d_module/install/setup.bash
