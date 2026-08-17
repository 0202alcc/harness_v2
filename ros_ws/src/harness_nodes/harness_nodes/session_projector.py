"""Durable session snapshot projection exposed through a ROS service."""

from __future__ import annotations

import json
import os

import rclpy
from rclpy.node import Node

from harness_interfaces.srv import GetSessionSnapshot
from storage import ChatNotFoundError, ChatStorage


class SessionProjector(Node):
    def __init__(self) -> None:
        super().__init__("session_projector")
        self._store = ChatStorage(os.environ.get("HARNESS_LOG_ROOT", "/workspace/.logs"))
        self._user_id = os.environ["HARNESS_USER_ID"]
        self.create_service(GetSessionSnapshot, "/harness/snapshot", self._snapshot)

    def _snapshot(self, request, response):
        try:
            chat = self._store.get_chat(request.session_id, self._user_id)
        except ChatNotFoundError:
            response.found = False
            response.error = "session not found"
            return response
        response.found = True
        response.snapshot_id = f"{request.session_id}:{chat.get('state_version', 0)}"
        response.snapshot_json = json.dumps(chat, ensure_ascii=False)
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SessionProjector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
