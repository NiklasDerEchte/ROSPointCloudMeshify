## Installed Packages
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
