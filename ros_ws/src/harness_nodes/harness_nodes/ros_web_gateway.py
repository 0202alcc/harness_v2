"""Run the browser app with ROS 2 as its internal event transport."""

from __future__ import annotations

import json
import os
import hashlib
import uuid

import uvicorn

from LLManager import LLManager, LlamaCppProvider
from app.server import create_app
from storage import ChatStorage

from .ros_gateway import RosGatewayRuntime


def resolve_runtime_session(store: ChatStorage, llm: LLManager) -> tuple[str, str, str]:
    """Resolve ROS startup state from the legacy Harness deployment settings."""
    user_id = os.environ.get("HARNESS_USER_ID")
    if not user_id:
        username = os.environ.get("HARNESS_USERNAME")
        if not username:
            raise RuntimeError("Set USERNAME or HARNESS_USER_ID in the deployment environment.")
        user_id = hashlib.sha256(username.encode("utf-8")).hexdigest()

    chat_id = os.environ.get("HARNESS_CHAT_ID")
    if not chat_id:
        chats = store.list_chats(user_id)
        chat_id = chats[0]["chat_id"] if chats else None

    requested_model = os.environ.get("HARNESS_MODEL")
    if chat_id:
        chat = store.get_chat(chat_id=chat_id, user_id=user_id)
        model = requested_model or chat.get("current_model", {}).get("model")
    else:
        models = llm.list_models()
        model = requested_model or next((item.get("id") for item in models if item.get("id")), None)
        if not model:
            raise RuntimeError("llama.cpp returned no usable model for the first ROS session.")
        chat_id = str(uuid.uuid4())
        store.create_chat(
            chat_id=chat_id,
            user_id=user_id,
            provider="llama.cpp",
            model=model,
            system_prompt=None,
        )

    if not model:
        raise RuntimeError(f"Chat {chat_id} has no configured model.")
    return user_id, chat_id, model


def main() -> None:
    config_path = os.environ.get("HARNESS_CONFIG_PATH", "/workspace/config.json")
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    store = ChatStorage(os.environ.get("HARNESS_LOG_ROOT", "/workspace/.logs"))
    llm = LLManager(LlamaCppProvider(
        base_url=os.environ["HARNESS_BASE_URL"],
        api_key=os.environ.get("HARNESS_API_KEY"),
    ))
    user_id, chat_id, model = resolve_runtime_session(store, llm)
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
        github_repository=os.environ.get("HARNESS_GITHUB_REPOSITORY"),
        github_token=os.environ.get("HARNESS_GITHUB_TOKEN"),
    )
    app.state.orchestrator = runtime
    try:
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
    finally:
        runtime.close()
        llm.close()
