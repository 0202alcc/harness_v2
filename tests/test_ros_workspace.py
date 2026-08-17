from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INTERFACES = ROOT / "ros_ws/src/harness_interfaces"
NODES = ROOT / "ros_ws/src/harness_nodes"


def test_ros_interfaces_cover_observation_action_and_control_contracts():
    assert (INTERFACES / "msg/ObservationEnvelope.msg").read_text().find("session_id") >= 0
    assert (INTERFACES / "msg/AssistantOutput.msg").read_text().find("superseded") >= 0
    action = (INTERFACES / "action/RunInference.action").read_text()
    assert action.count("---") == 2
    assert "context_snapshot_id" in action
    assert (INTERFACES / "srv/SessionControl.srv").exists()


def test_ros_launch_graph_and_container_entrypoint_are_present():
    launch = (NODES / "launch/harness_system.launch.py").read_text()
    for executable in (
        "attention_arbiter",
        "perception",
        "inference_action_server",
        "session_orchestrator",
        "ros_web_gateway",
    ):
        assert executable in launch
    dockerfile = (ROOT / "Dockerfile.ros").read_text()
    assert "colcon build" in dockerfile
    assert "ros2 launch harness_nodes" in dockerfile
