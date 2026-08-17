#!/usr/bin/env bash
set -euo pipefail

# All ROS child processes must share the same storage identity as the web
# gateway. Legacy deployments only define USERNAME, so derive its stable hash
# once before launching the ROS graph.
if [[ -z "${HARNESS_USER_ID:-}" ]]; then
  [[ -n "${HARNESS_USERNAME:-}" ]] || {
    echo "Set USERNAME or HARNESS_USER_ID before starting the ROS runtime." >&2
    exit 1
  }
  export HARNESS_USER_ID
  HARNESS_USER_ID="$(printf '%s' "$HARNESS_USERNAME" | sha256sum | awk '{print $1}')"
fi

log_root="${HARNESS_LOG_ROOT:-/workspace/.logs}"
/workspace/.venv/bin/python -m app.ros_migration --log-root "$log_root" --user-id "$HARNESS_USER_ID"

source /workspace/.venv/bin/activate
source /opt/ros/jazzy/setup.bash
source /workspace/ros_ws/install/setup.bash
exec ros2 launch harness_nodes harness_system.launch.py
