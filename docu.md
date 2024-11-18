## Installation ROS
- Follow: https://docs.ros.org/en/humble/Installation.html

### Test ROS
- ros2 run demo_nodes_cpp talker
- ros2 run demo_nodes_cpp listener

## Installation Zivid
- Zivid Studio https://support.zivid.com/en/latest/getting-started/software-installation.html
  - Computer muss im selben Netzwerk sein wie die Camera (172.28.60.XXX)
- Zivid Ros https://github.com/zivid/zivid-ros

## Installation PCL
- sudo apt install libpcl-dev
- sudo apt install ros-humble-pcl-conversions ros-humble-pcl-ros
- sudo apt install ros-humble-visualization-msgs
- sudo apt install libpcap-dev

## Import modules
source ros_pcl_module/install/setup.bash
source ros_zivid_driver/install/setup.bash

## Launch Zivid sample
- ros2 launch zivid_samples sample_with_rviz.launch sample:=sample_capture_cpp

## Start Object detection with PCL 
- ros2 run object_detection object_detection_node
### Visualizer
- ros2 run object_detection object_detection_visualizer_node
### FirstTry Bounding Boxes
- ros2 run object_detection object_detection_geometryc_visualizer_node

## Unity-Integration
## Start Docker-Server
- cd ~/UnityRosPlugins/Unity-Robotics-Hub/tutorials/ros_unity_integration && docker run -it --rm -p 10000:10000 humble_unity_environment /bin/bash

## Start Unity-ROS Server
- ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0
