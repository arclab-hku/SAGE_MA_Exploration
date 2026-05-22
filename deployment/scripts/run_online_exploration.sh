#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-noetic}"
ROS1_WS="${ROS1_WS:-${REPO_ROOT}/ros1_ws}"
DRONE_NUM="${DRONE_NUM:-2}"
SCENARIO_CONFIG="${SCENARIO_CONFIG:-${REPO_ROOT}/deployment/config/scenarios/office.yaml}"
USE_LEARNED_UTILITY="${USE_LEARNED_UTILITY:-false}"
MODEL_PATH="${MODEL_PATH:-}"
MAX_VEL="${MAX_VEL:-3.0}"
MAX_ACC="${MAX_ACC:-2.5}"
OBSTACLE_INFLATION="${OBSTACLE_INFLATION:-0.05}"

source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${ROS1_WS}/devel/setup.bash"

roslaunch "${REPO_ROOT}/deployment/launch/online_exploration.launch" \
  drone_num:="${DRONE_NUM}" \
  config_file:="${SCENARIO_CONFIG}" \
  use_learned_utility:="${USE_LEARNED_UTILITY}" \
  model_path:="${MODEL_PATH}" \
  max_vel:="${MAX_VEL}" \
  max_acc:="${MAX_ACC}" \
  obstacle_inflation:="${OBSTACLE_INFLATION}"

