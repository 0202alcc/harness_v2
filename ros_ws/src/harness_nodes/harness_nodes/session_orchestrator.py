"""Consumes attention decisions and owns one cancellable action goal/session."""

from __future__ import annotations

import uuid

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from harness_interfaces.action import RunInference
from harness_interfaces.msg import AttentionDecision, ObservationEnvelope

from .qos import TEXT_EVENTS_QOS


class SessionOrchestrator(Node):
    def __init__(self) -> None:
        super().__init__("session_orchestrator")
        self._observations: dict[str, ObservationEnvelope] = {}
        self._active_goals: dict[str, object] = {}
        self._client = ActionClient(self, RunInference, "/harness/infer")
        self.create_subscription(
            ObservationEnvelope, "/harness/observations/text", self._remember, TEXT_EVENTS_QOS
        )
        self.create_subscription(
            AttentionDecision, "/harness/attention/decisions", self._decide, TEXT_EVENTS_QOS
        )
        self.create_subscription(String, "/harness/control", self._control, TEXT_EVENTS_QOS)

    def _remember(self, observation: ObservationEnvelope) -> None:
        self._observations[observation.observation_id] = observation

    def _decide(self, decision: AttentionDecision) -> None:
        if not decision.observation_ids:
            return
        observation = self._observations.get(decision.observation_ids[-1])
        if observation is None:
            self.get_logger().warning("Decision arrived before its observation")
            return
        active = self._active_goals.get(decision.session_id)
        if active is not None:
            active.cancel_goal_async()
        goal = RunInference.Goal()
        goal.session_id = decision.session_id
        goal.run_id = str(uuid.uuid4())
        goal.context_snapshot_id = decision.context_snapshot_id
        goal.observation_ids = decision.observation_ids
        goal.input_text = observation.payload_text
        goal.trace_id = decision.trace_id
        future = self._client.send_goal_async(goal, feedback_callback=self._feedback)
        future.add_done_callback(lambda completed: self._accepted(decision.session_id, completed))

    def _accepted(self, session_id: str, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Inference goal rejected")
            return
        self._active_goals[session_id] = handle
        handle.get_result_async().add_done_callback(
            lambda _: self._active_goals.pop(session_id, None)
        )

    def _feedback(self, feedback_message) -> None:
        self.get_logger().debug(feedback_message.feedback.phase)

    def _control(self, message: String) -> None:
        session_id = message.data
        active = self._active_goals.get(session_id)
        if active is not None:
            active.cancel_goal_async()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SessionOrchestrator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
