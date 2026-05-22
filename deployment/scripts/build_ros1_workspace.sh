#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-noetic}"
ROS1_WS="${ROS1_WS:-${REPO_ROOT}/ros1_ws}"

source "/opt/ros/${ROS_DISTRO}/setup.bash"
cd "${ROS1_WS}"
catkin_make

