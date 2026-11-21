#!/bin/bash
(cd meshify_ws && rm -rf build/ install/ log/ && colcon build)
