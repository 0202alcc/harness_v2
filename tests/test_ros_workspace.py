from pathlib import Path

from app.ros_migration import migrate_orphaned_sessions
from storage import ChatStorage

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
    startup = (ROOT / "scripts/start_ros_system.sh").read_text()
    assert "ros2 launch harness_nodes" in startup


def test_orphaned_ros_session_messages_are_migrated_once(tmp_path):
    store = ChatStorage(tmp_path)
    store.create_chat("chat", "", model="fake")
    store.append_message(chat_id="chat", user_id="", role="user", content="hello")
    store.append_message(chat_id="chat", user_id="", role="assistant", content="hi")

    assert migrate_orphaned_sessions(log_root=str(tmp_path), user_id="real-user") == 2
    repaired = store.get_chat("chat", "real-user")
    assert [message["content"] for message in repaired["messages"]] == ["hello", "hi"]
    assert migrate_orphaned_sessions(log_root=str(tmp_path), user_id="real-user") == 0
