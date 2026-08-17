# Running the ROS 2 system

The ROS implementation uses ROS 2 Jazzy in Docker because ROS 2 is not
installed on the development host. The browser remains outside ROS and speaks
WebSocket to the `ros_web_gateway` node on port 8000.

## Prepare a session

Create or identify a normal Harness chat first. Its `user_id` is the SHA-256
value printed by `main.py`, and its `chat_id` is printed at the same time.
Ensure llama.cpp is reachable from Docker; on macOS the usual address is
`http://host.docker.internal:PORT`.

```bash
export HARNESS_BASE_URL=http://host.docker.internal:8080
export HARNESS_USER_ID='...'
export HARNESS_CHAT_ID='...'
export HARNESS_MODEL='...'
docker compose -f compose.ros.yaml up --build
```

Then open `http://127.0.0.1:8000`. The browser publishes a text observation
through the ROS gateway. The graph is:

```text
web gateway → /harness/observations/text → attention arbiter
→ session orchestrator → /harness/infer action → action server
→ /harness/outputs/assistant → web gateway
```

Inspect it from the container:

```bash
ros2 topic list
ros2 action list
ros2 service list
ros2 node list
```

## QoS and media

Submitted text, control, decisions, and output use reliable bounded QoS.
Audio/video/image/tool observations use best-effort, depth-five QoS and carry
an object-store URI rather than unbounded bytes. `perception` emits derived
`Percept` messages, while the attention arbiter intentionally treats them as
context only until an explicit salience policy is supplied.

## Operational boundaries

The `RunInference` action is cancellable at the -= goal layer. The current
llama.cpp request API cannot interrupt a decoding call, so cancellation prevents
the result from being published after decoding completes. Adding provider-level
cancel support is the remaining path to immediate GPU/model cancellation.
