"""Run the browser app with ROS 2 as its internal event transport."""

from __future__ import annotations

import json
import os

import uvicorn

from LLManager import LLManager, LlamaCppProvider
from app.server import create_app
from storage import ChatStorage

from .ros_gateway import RosGatewayRuntime


def main() -> None:
    config_path = os.environ.get("HARNESS_CONFIG_PATH", "/workspace/config.json")
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    user_id = os.environ["HARNESS_USER_ID"]
    chat_id = os.environ["HARNESS_CHAT_ID"]
    model = os.environ["HARNESS_MODEL"]
    store = ChatStorage(os.environ.get("HARNESS_LOG_ROOT", "/workspace/.logs"))
    llm = LLManager(LlamaCppProvider(
        base_url=os.environ["HARNESS_BASE_URL"],
        api_key=os.environ.get("HARNESS_API_KEY"),
    ))
    runtime = RosGatewayRuntime(store=store, user_id=user_id)
    app = create_app(
        chat_id=chat_id,
        user_id=user_id,
        model=model,
        llm=llm,
        store=store,
        annotation_instruction=config["ANNOTATION_INSTRUCTION"],
        thought_process_instruction=config["THOUGHT_PROCESS_INSTRUCTION"],
        response_instruction=config["RESPONSE_INSTRUCTION"],
        markers=config.get("MARKERS"),
        thought_process_output_prefix=config.get("THOUGHT_PROCESS_OUTPUT_PREFIX"),
    )
    app.state.orchestrator = runtime
    try:
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
    finally:
        runtime.close()
        llm.close()
