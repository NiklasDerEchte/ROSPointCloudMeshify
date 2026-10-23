#!/bin/bash
# Release: an empty CMAKE_BUILD_TYPE leaves PCL and the grid mesh unoptimized.
(cd ros_pcl_module && rm -rf build/ install/ log/ && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release)
