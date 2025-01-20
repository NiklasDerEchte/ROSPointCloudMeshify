#!/bin/bash
(cd ros_pcl_module && rm -rf build/ install/ log/ && colcon build)
(cd ros_zivid_driver && rm -rf build/ install/ log/ && colcon build)
# (cd ros_open3d_module && rm -rf build/ install/ log/ && colcon build)
