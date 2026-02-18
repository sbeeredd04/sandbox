#!/usr/bin/env bash
# publish_goal_cmd_ros1.sh
rostopic pub --once --latch /spinflow/goal_cmd_override std_msgs/String "$*"
