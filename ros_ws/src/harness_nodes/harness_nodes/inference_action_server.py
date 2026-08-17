"""ROS action server that runs the existing Harness graph.

This is the deliberate compatibility boundary: inference evolves behind a ROS
action contract while the existing graph/model adapter remains reusable.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from Harness import Harness
from LLManager import LLManager, LlamaCppProvider
from harness_interfaces.action import RunInference
from harness_interfaces.msg import AgentActivity, AssistantOutput
from storage import ChatStorage

from .qos import OUTPUT_QOS


class HarnessExecutor:
    """Builds re-entrant Harness instances from container configuration."""

    def __init__(self) -> None:
        config_path = Path(os.environ.get("HARNESS_CONFIG_PATH", "/workspace/config.json"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.user_id = os.environ["HARNESS_USER_ID"]
        self.model = os.environ["HARNESS_MODEL"]
        self.store = ChatStorage(os.environ.get("HARNESS_LOG_ROOT", "/workspace/.logs"))
        self.llm = LLManager(LlamaCppProvider(
            base_url=os.environ["HARNESS_BASE_URL"],
            api_key=os.environ.get("HARNESS_API_KEY"),
        ))
        self.config: dict[str, Any] = config

    def get_harness(self, session_id: str) -> Harness:
        try:
            self.store.get_chat(chat_id=session_id, user_id=self.user_id)
        except Exception:
            self.store.create_chat(
                chat_id=session_id,
                user_id=self.user_id,
                model=self.model,
                provider="llama.cpp",
            )
        return Harness(
            llm=self.llm,
            store=self.store,
            model=self.model,
            user_id=self.user_id,
            chat_id=session_id,
            annotation_instruction=self.config["ANNOTATION_INSTRUCTION"],
            thought_process_instruction=self.config["THOUGHT_PROCESS_INSTRUCTION"],
            response_instruction=self.config["RESPONSE_INSTRUCTION"],
            markers=self.config.get("MARKERS"),
            thought_process_output_prefix=self.config.get("THOUGHT_PROCESS_OUTPUT_PREFIX"),
        )


class InferenceActionServer(Node):
    def __init__(self) -> None:
        super().__init__("inference_action_server")
        self._executor = HarnessExecutor()
        self._outputs = self.create_publisher(AssistantOutput, "/harness/outputs/assistant", OUTPUT_QOS)
        self._activity = self.create_publisher(AgentActivity, "/harness/activity", OUTPUT_QOS)
        self._server = ActionServer(
            self,
            RunInference,
            "/harness/infer",
            execute_callback=self._execute,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
        )

    def _publish_activity(self, goal, state: str, detail: str = "") -> None:
        message = AgentActivity()
        message.session_id = goal.session_id
        message.run_id = goal.run_id
        message.state = state
        message.detail = detail
        message.trace_id = goal.trace_id
        self._activity.publish(message)

    def _execute(self, goal_handle):
        goal = goal_handle.request
        result = RunInference.Result()
        self._publish_activity(goal, "thinking")

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.superseded = True
            result.error = "cancelled before inference"
            return result

        def feedback(event: dict[str, Any]) -> None:
            # Keep the native action feedback compact for non-web clients, but
            # forward the complete event over the output topic as well.  The
            # web gateway subscribes to that topic and can therefore retain
            # the existing browser streaming protocol without becoming an
            # action client itself.
            message = RunInference.Feedback()
            message.phase = str(event.get("type", "generating"))
            message.content_delta = str(event.get("content", event.get("text", "")))
            goal_handle.publish_feedback(message)

            streamed = AssistantOutput()
            streamed.output_id = str(uuid.uuid4())
            streamed.session_id = goal.session_id
            streamed.run_id = goal.run_id
            streamed.turn_id = goal.turn_id
            streamed.kind = "harness_feedback"
            streamed.content = json.dumps(event, ensure_ascii=False)
            streamed.final = False
            streamed.superseded = False
            streamed.trace_id = goal.trace_id
            self._outputs.publish(streamed)

        try:
            harness = self._executor.get_harness(goal.session_id)
            chat = self._executor.store.get_chat(goal.session_id, self._executor.user_id)
            user_message = self._executor.store.append_message(
                chat_id=goal.session_id,
                user_id=self._executor.user_id,
                role="user",
                content=goal.input_text,
                turn_id=goal.turn_id or None,
            )
            run = harness.handle_message(
                message=goal.input_text,
                turn_id=user_message["turn_id"],
                run_id=goal.run_id,
                on_annotation_event=feedback,
                persist_response=False,
            )
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.superseded = True
                result.error = "cancelled while model was decoding"
                self._publish_activity(goal, "idle", "cancelled")
                return result

            output = AssistantOutput()
            output.output_id = str(uuid.uuid4())
            output.session_id = goal.session_id
            output.run_id = goal.run_id
            output.turn_id = user_message["turn_id"]
            output.kind = "response"
            output.content = run["response"]
            output.final = True
            output.superseded = False
            output.trace_id = goal.trace_id
            self._outputs.publish(output)
            self._executor.store.append_message(
                chat_id=goal.session_id,
                user_id=self._executor.user_id,
                role="assistant",
                content=run["response"],
                turn_id=user_message["turn_id"],
            )
            goal_handle.succeed()
            result.completed = True
            result.response = run["response"]
            result.output_id = output.output_id
            self._publish_activity(goal, "idle")
            return result
        except Exception as exc:
            goal_handle.abort()
            result.error = str(exc)
            self._publish_activity(goal, "error", str(exc))
            return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InferenceActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
