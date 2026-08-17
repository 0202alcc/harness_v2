"""ROS-facing runtime for the existing FastAPI/WebSocket gateway.

Run this node in the same environment as the application package. It replaces
the in-process orchestrator with ROS publishers/subscribers while leaving the
browser protocol unchanged.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from app.always_on import SessionEventBroker
from harness_interfaces.msg import AssistantOutput, ObservationEnvelope
from storage import ChatStorage

from .qos import OUTPUT_QOS, TEXT_EVENTS_QOS


class RosGatewayRuntime:
    """Drop-in replacement for ``AlwaysOnOrchestrator`` at the web boundary."""

    def __init__(self, *, store: ChatStorage, user_id: str) -> None:
        self.store = store
        self.user_id = user_id
        self.broker = SessionEventBroker()
        self._seen: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        rclpy.init()
        self.node = Node("web_gateway")
        self._text = self.node.create_publisher(
            ObservationEnvelope, "/harness/observations/text", TEXT_EVENTS_QOS
        )
        self._control = self.node.create_publisher(String, "/harness/control", TEXT_EVENTS_QOS)
        self.node.create_subscription(
            AssistantOutput, "/harness/outputs/assistant", self._on_output, OUTPUT_QOS
        )
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _publish_event(self, session_id: str, event: dict[str, Any]) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self.broker.publish, session_id, {"type": "session_event", "event": event}
            )

    def _on_output(self, output: AssistantOutput) -> None:
        # Harness emits high-frequency annotation and token events while a
        # request is running. They are transport events, not durable chat
        # history, so fan them straight to connected WebSocket clients.
        if not output.final and output.kind == "harness_feedback":
            try:
                feedback = json.loads(output.content)
            except json.JSONDecodeError:
                feedback = {"type": "stream_error", "content": output.content}
            if self._loop is not None:
                self._loop.call_soon_threadsafe(
                    self.broker.publish,
                    output.session_id,
                    {
                        "type": "inference_feedback",
                        "run_id": output.run_id,
                        "event": feedback,
                    },
                )
            return

        if output.kind == "error":
            event = self.store.append_event(
                chat_id=output.session_id,
                user_id=self.user_id,
                event_type="inference_failed",
                turn_id=output.turn_id or None,
                data={
                    "run_id": output.run_id,
                    "message": output.content,
                    "output_id": output.output_id,
                },
            )
            self._publish_event(output.session_id, event)
            return

        event = self.store.append_event(
            chat_id=output.session_id,
            user_id=self.user_id,
            event_type="inference_completed" if output.final else "inference_output",
            turn_id=output.turn_id or None,
            data={
                "run_id": output.run_id,
                "response": output.content,
                "output_id": output.output_id,
                "superseded": output.superseded,
            },
        )
        self._publish_event(output.session_id, event)

    async def submit_text(
        self,
        *,
        session_id: str,
        content: str,
        source_id: str = "web",
        sequence: int = 0,
        observation_id: str | None = None,
        captured_at: str | None = None,
    ) -> dict[str, Any]:
        content = content.strip()
        if not content:
            raise ValueError("Text observations may not be blank.")
        self.store.get_chat(chat_id=session_id, user_id=self.user_id)
        self._loop = asyncio.get_running_loop()
        observation_id = observation_id or str(uuid.uuid4())
        if observation_id in self._seen:
            return {"observation_id": observation_id, "status": "duplicate"}
        self._seen.add(observation_id)
        now = datetime.now(timezone.utc).isoformat()
        event = self.store.append_event(
            chat_id=session_id,
            user_id=self.user_id,
            event_type="observation_received",
            data={
                "observation_id": observation_id,
                "session_id": session_id,
                "source_id": source_id,
                "sequence": sequence,
                "modality": "TEXT",
                "payload": content,
                "captured_at": captured_at or now,
                "received_at": now,
            },
        )
        self._publish_event(session_id, event)
        message = ObservationEnvelope()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.observation_id = observation_id
        message.session_id = session_id
        message.source_id = source_id
        message.sequence = sequence
        message.modality = "TEXT"
        message.correlation_id = observation_id
        message.trace_id = observation_id
        message.payload_text = content
        message.captured_at = self.node.get_clock().now().to_msg()
        message.received_at = self.node.get_clock().now().to_msg()
        message.schema_version = 1
        self._text.publish(message)
        return {"observation_id": observation_id, "status": "accepted"}

    async def cancel(self, session_id: str) -> None:
        message = String()
        message.data = session_id
        self._control.publish(message)

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()
