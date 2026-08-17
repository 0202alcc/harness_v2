"""Lifecycle supervisor for managed ROS nodes."""

from __future__ import annotations

import rclpy
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
from rclpy.node import Node


class LifecycleManager(Node):
    """Drive declared managed nodes through configure then activate."""

    def __init__(self) -> None:
        super().__init__("lifecycle_manager")
        self.declare_parameter("managed_nodes", [])
        self._timer = self.create_timer(1.0, self._configure_once)
        self._started = False

    def _configure_once(self) -> None:
        if self._started:
            return
        self._started = True
        for node_name in self.get_parameter("managed_nodes").value:
            self._transition(node_name, Transition.TRANSITION_CONFIGURE)
            self._transition(node_name, Transition.TRANSITION_ACTIVATE)

    def _transition(self, node_name: str, transition_id: int) -> None:
        client = self.create_client(ChangeState, f"{node_name}/change_state")
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"Lifecycle service unavailable: {node_name}")
            return
        request = ChangeState.Request()
        request.transition.id = transition_id
        client.call_async(request)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LifecycleManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
