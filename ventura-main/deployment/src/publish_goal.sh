#!/usr/bin/env bash
# publish_goal.sh
#
# Usage: ./publish_goal.sh "My new goal command"

if [ $# -lt 1 ]; then
  echo "Usage: $0 \"GOAL_STRING\""
  exit 1
fi

GOAL="$*"

# publish once and exit
rostopic pub -1 /goal_cmd_override std_msgs/String "{data: \"$GOAL\"}"