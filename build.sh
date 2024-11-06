#!/bin/bash
(cd ros_pcl_module && rm -rf build/ install/ log/ && colcon build && source install/setup.bash)
(cd ros_zivid_driver && rm -rf build/ install/ log/ && colcon build && source install/setup.bash)
