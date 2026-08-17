from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="harness_nodes", executable="lifecycle_manager", output="screen"),
        Node(package="harness_nodes", executable="attention_arbiter", output="screen"),
        Node(package="harness_nodes", executable="perception", output="screen"),
        Node(package="harness_nodes", executable="session_projector", output="screen"),
        Node(package="harness_nodes", executable="inference_action_server", output="screen"),
        Node(package="harness_nodes", executable="session_orchestrator", output="screen"),
        Node(package="harness_nodes", executable="ros_web_gateway", output="screen"),
    ])
