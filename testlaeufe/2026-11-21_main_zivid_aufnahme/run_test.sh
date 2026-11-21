#!/bin/bash
# Usage: run_test.sh <outdir> <node> <name> [extra ros args...]
source /opt/ros/humble/setup.bash
WS=/home/niklas/ROSPointCloudMeshify/.claude/worktrees/cpu-gpu-logging-patch-ee1d92/meshify_ws
source $WS/install/setup.bash
export ROS_DOMAIN_ID=42
set -u
OUT=$1/$3; NODE=$2; NAME=$3; shift 3
mkdir -p $OUT && cd $OUT && rm -f $NAME.csv
$WS/install/meshify/lib/meshify/$NODE --ros-args -p csv_path:=$NAME.csv "$@" > node.log 2>&1 &
PID=$!
sleep 3
ros2 bag play --wait-for-all-acked 10000 ~/Schreibtisch/zivid_bags/zivid_aufnahme > bag.log 2>&1
for i in $(seq 1 900); do
  [ -f $NAME.csv ] && [ $(($(wc -l < $NAME.csv) - 1)) -ge 46 ] && break
  sleep 1
done
kill -INT $PID; wait $PID
echo "$NAME: $(($(wc -l < $NAME.csv) - 1)) frames"
