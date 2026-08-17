#!/usr/bin/env bash
# Replace the legacy systemd web server with the ROS Compose runtime and roll
# back to the prior runtime when the new container does not become ready.
set -euo pipefail

usage() {
  echo "Usage: $0 [--previous-sha <sha>]" >&2
  exit 2
}

previous_sha=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --previous-sha) previous_sha="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done

base_compose="compose.ros.yaml"
[[ -f "$base_compose" ]] || {
  echo "Run from a Harness checkout containing the ROS Compose files." >&2
  exit 1
}

compose=(docker compose --project-name harness-v2 -f "$base_compose")
release_version="$(tr -d '[:space:]' < VERSION)"
release_revision="$(git rev-parse --short HEAD)"
export HARNESS_VERSION="${release_version}+${release_revision}"
"${compose[@]}" config --quiet
had_existing=false
[[ -n "$("${compose[@]}" ps --quiet 2>/dev/null)" ]] && had_existing=true
legacy_was_active=false
if sudo /usr/bin/systemctl is-active --quiet harness-v2.service; then
  legacy_was_active=true
fi
replacement_started=false

rollback() {
  status=$?
  if [[ "$replacement_started" == true ]]; then
    echo "ROS deployment failed; restoring the prior runtime." >&2
    set +e
    "${compose[@]}" down
    if [[ "$legacy_was_active" == true ]]; then
      git checkout --detach "$previous_sha"
      sudo /usr/bin/systemctl start harness-v2.service
    elif [[ -n "$previous_sha" && "$had_existing" == true ]]; then
      git checkout --detach "$previous_sha"
      "${compose[@]}" up --detach --build --remove-orphans
    elif [[ "$had_existing" == false ]]; then
      true
    fi
  fi
  exit "$status"
}
trap rollback ERR

replacement_started=true
if [[ "$legacy_was_active" == true ]]; then
  sudo /usr/bin/systemctl stop harness-v2.service
fi
"${compose[@]}" up --detach --build --remove-orphans
for _ in {1..30}; do
  if curl --fail --silent --show-error http://127.0.0.1:8000/healthz >/dev/null; then
    if [[ "$legacy_was_active" == true ]]; then
      sudo /usr/bin/systemctl disable harness-v2.service
    fi
    echo "ROS deployment is ready."
    exit 0
  fi
  sleep 1
done
echo "Timed out waiting for the ROS health endpoint." >&2
exit 1
