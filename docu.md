## Installed Packages
- sudo apt install libpcl-dev
- sudo apt install ros-humble-pcl-conversions ros-humble-pcl-ros
- sudo apt install libpcap-dev


## Launch Zivid sample
- ros2 launch zivid_samples sample_with_rviz.launch sample:=sample_capture_cpp

## Start Object detection with PCL 
- ros2 run object_detection object_detection_node
