# main.py

import argparse
import hashlib
import json
import logging
import uuid

from dotenv import dotenv_values

from app.server import run_server
from LLManager import LLManager, LlamaCppProvider
from storage import ChatNotFoundError, ChatStorage


def make_user_id(username: str) -> str:
    """
    Convert the human-readable username into the stable ID used
    for storage.
    """
    if not username:
        raise ValueError("A username is required.")

    return hashlib.sha256(
        username.encode("utf-8")
    ).hexdigest()


def resolve_model(
    llm: LLManager,
    requested_model: str | None = None,
) -> str:
    """
    Validate an explicitly requested model, or choose the first
    model advertised by llama.cpp.
    """

    models = llm.list_models()

    if not models:
        raise RuntimeError(
            "llama.cpp reported no available models."
        )

    model_ids = [
        model["id"]
        for model in models
        if model.get("id")
    ]

    if requested_model is not None:
        if requested_model not in model_ids:
            raise ValueError(
                f"Requested model {requested_model!r} is not available. "
                f"Available models: {model_ids}"
            )

        return requested_model

    return model_ids[0]


def ensure_model_loaded(
    llm: LLManager,
    model: str,
) -> None:
    """
    Load the selected model if llama.cpp reports it as unloaded.

    This is useful when llama-server is running in router mode with
    model autoloading disabled.
    """

    models = llm.list_models()

    model_info = next(
        (
            item
            for item in models
            if item.get("id") == model
        ),
        None,
    )

    if model_info is None:
        raise ValueError(
            f"Model {model!r} is not known to llama.cpp."
        )

    status = (
        model_info
        .get("status", {})
        .get("value")
    )

    if status == "unloaded":
        logging.info(
            "Loading model %s",
            model,
        )

        llm.load_model(model)


def main(
    *,
    base_url: str,
    username: str,
    annotation_instruction: str,
    thought_process_instruction: str,
    chat_id: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    log_root: str = "./.logs/",
) -> None:

    # ---------------------------------------------------------
    # Resolve user
    # ---------------------------------------------------------

    user_id = make_user_id(username)

    logging.info(
        "Starting Harness for user %s",
        username,
    )

    # ---------------------------------------------------------
    # Construct dependencies
    # ---------------------------------------------------------

    provider = LlamaCppProvider(
        base_url=base_url,
        api_key=api_key,
    )

    llm = LLManager(
        provider=provider,
    )

    store = ChatStorage(
        root_path=log_root,
    )

    try:
        # -----------------------------------------------------
        # Verify llama.cpp
        # -----------------------------------------------------

        health = llm.health()

        if not health.get("ok"):
            raise RuntimeError(
                f"llama.cpp is not ready: {health}"
            )

        # -----------------------------------------------------
        # Existing chat
        # -----------------------------------------------------

        if chat_id is not None:
            try:
                chat = store.get_chat(
                    chat_id=chat_id,
                    user_id=user_id,
                )

            except ChatNotFoundError as exc:
                raise ValueError(
                    f"Chat {chat_id!r} does not exist "
                    f"for user {username!r}."
                ) from exc

            logging.info(
                "Using existing chat %s",
                chat_id,
            )

            stored_model = (
                chat
                .get("current_model", {})
                .get("model")
            )

            # Explicit CLI model overrides stored model.
            selected_model = (
                resolve_model(llm, model)
                if model is not None
                else stored_model
            )

            # Old/incomplete chat state may not have a model.
            if selected_model is None:
                selected_model = resolve_model(llm)

            # If CLI changed the model, persist that change.
            if selected_model != stored_model:
                store.update_model(
                    chat_id=chat_id,
                    user_id=user_id,
                    provider="llama.cpp",
                    model=selected_model,
                )

        # -----------------------------------------------------
        # New chat
        # -----------------------------------------------------

        else:
            chat_id = str(uuid.uuid4())

            selected_model = resolve_model(
                llm,
                requested_model=model,
            )

            chat = store.create_chat(
                chat_id=chat_id,
                user_id=user_id,
                provider="llama.cpp",
                model=selected_model,
                system_prompt=None,
            )

            logging.info(
                "Created new chat %s",
                chat_id,
            )

        # -----------------------------------------------------
        # Ensure selected model is ready
        # -----------------------------------------------------

        ensure_model_loaded(
            llm,
            selected_model,
        )

        print(
            f"Starting server\n"
            f"  user:     {username}\n"
            f"  user_id:  {user_id}\n"
            f"  chat_id:  {chat_id}\n"
            f"  model:    {selected_model}"
        )

        # -----------------------------------------------------
        # Launch interface
        # -----------------------------------------------------

        run_server(
            chat_id=chat_id,
            user_id=user_id,
            model=selected_model,
            llm=llm,
            store=store,
            annotation_instruction=annotation_instruction,
            thought_process_instruction=thought_process_instruction,
        )

    finally:
        llm.close()


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    env = dotenv_values(".env")

    with open(
        "config.json",
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    log_root = (
        config
        .get("PATH", {})
        .get("logs", "./.logs/")
    )

    annotation_instruction = config.get(
        "ANNOTATION_INSTRUCTION"
    )

    if not annotation_instruction:
        raise RuntimeError(
            "ANNOTATION_INSTRUCTION is missing from config.json"
        )
    thought_process_instruction = config.get("THOUGHT_PROCESS_INSTRUCTION")
    if not thought_process_instruction:
        raise RuntimeError("THOUGHT_PROCESS_INSTRUCTION is missing from config.json")
    # ---------------------------------------------------------
    # CLI
    # ---------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Harness v1 - Local LLM Interface"
    )

    parser.add_argument(
        "-cid",
        "--chat_id",
        type=str,
        default=None,
        help="Existing chat ID to resume.",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model to use. Overrides the stored model.",
    )

    parser.add_argument(
        "--username",
        type=str,
        default=env.get("USERNAME"),
        help="Username for the session.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Start application
    # ---------------------------------------------------------

    main(
        base_url=env.get("BASE_URL"),
        api_key=env.get("API_KEY"),
        username=args.username,
        chat_id=args.chat_id,
        model=args.model,
        log_root=log_root,
        annotation_instruction=annotation_instruction,
        thought_process_instruction=thought_process_instruction,
    )
