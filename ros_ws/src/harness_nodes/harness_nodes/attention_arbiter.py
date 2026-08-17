"""Turn accepted observations into bounded, cancellable inference work."""

from __future__ import annotations

import uuid

import rclpy
from rclpy.node import Node

from harness_interfaces.msg import AttentionDecision, ObservationEnvelope, Percept

from .qos import MEDIA_QOS, TEXT_EVENTS_QOS


class AttentionArbiter(Node):
    """A conservative first policy: final text triggers exactly one decision.

    Media percepts are intentionally context-only until a modality-specific
    salience policy is configured. This prevents frame-rate driven LLM calls.
    """

    def __init__(self) -> None:
        super().__init__("attention_arbiter")
        self._decisions = self.create_publisher(
            AttentionDecision, "/harness/attention/decisions", TEXT_EVENTS_QOS
        )
        self.create_subscription(
            ObservationEnvelope, "/harness/observations/text", self._on_text, TEXT_EVENTS_QOS
        )
        self.create_subscription(
            Percept, "/harness/percepts", self._on_percept, MEDIA_QOS
        )

    def _on_text(self, observation: ObservationEnvelope) -> None:
        decision = AttentionDecision()
        decision.header = observation.header
        decision.decision_id = str(uuid.uuid4())
        decision.session_id = observation.session_id
        decision.observation_ids = [observation.observation_id]
        decision.priority = 100
        decision.reason = "submitted_text"
        decision.context_snapshot_id = ""  # projector assigns durable snapshots later.
        decision.trace_id = observation.trace_id
        self._decisions.publish(decision)

    def _on_percept(self, percept: Percept) -> None:
        self.get_logger().debug(
            f"Percept {percept.percept_id} retained as context: {percept.kind}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AttentionArbiter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
