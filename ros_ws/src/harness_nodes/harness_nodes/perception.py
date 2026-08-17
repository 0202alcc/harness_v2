"""Reference multimodal perception node.

Binary media stays in object storage; ROS carries bounded references and emits
small derived percepts. Configure real ASR/vision/OCR implementations behind
this node without changing the rest of the graph.
"""

from __future__ import annotations

import json
import uuid

import rclpy
from rclpy.node import Node

from harness_interfaces.msg import ObservationEnvelope, Percept

from .qos import MEDIA_QOS


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception")
        self._percepts = self.create_publisher(Percept, "/harness/percepts", MEDIA_QOS)
        for modality in ("audio", "video", "image", "tool"):
            self.create_subscription(
                ObservationEnvelope,
                f"/harness/observations/{modality}",
                self._on_observation,
                MEDIA_QOS,
            )

    def _on_observation(self, observation: ObservationEnvelope) -> None:
        percept = Percept()
        percept.header = observation.header
        percept.percept_id = str(uuid.uuid4())
        percept.session_id = observation.session_id
        percept.observation_id = observation.observation_id
        percept.modality = observation.modality
        percept.kind = "media_reference_received"
        percept.confidence = 1.0
        percept.payload_json = json.dumps({"payload_uri": observation.payload_uri})
        percept.trace_id = observation.trace_id
        percept.observed_at = observation.received_at
        self._percepts.publish(percept)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
