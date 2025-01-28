#!/bin/bash
(cd ros_pcl_module && rm -rf build/ install/ log/ && colcon build)
