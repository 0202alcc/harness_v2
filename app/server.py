# app/server.py

from __future__ import annotations

import logging
import asyncio
import json
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from Harness import Harness
from LLManager import LLManager
from storage import ChatStorage


templates = Jinja2Templates(
    directory="./app/templates"
)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def chat_metadata(chat_data: dict[str, Any]) -> dict[str, Any]:
    """Return the debug view without duplicating message bodies."""
    return {
        **chat_data,
        "messages": [
            {
                key: value
                for key, value in message.items()
                if key != "content"
            }
            for message in chat_data.get("messages", [])
        ],
    }


def create_app(
    *,
    chat_id: str,
    user_id: str,
    model: str,
    llm: LLManager,
    store: ChatStorage,
    annotation_instruction: str,
    thought_process_instruction: str,
    response_instruction: str,
) -> FastAPI:
    """
    Construct the FastAPI application and inject the application's
    runtime dependencies.

    server.py is responsible only for the web interface. It does not
    perform direct filesystem access or direct llama.cpp access.
    """

    app = FastAPI()

    # ---------------------------------------------------------
    # Construct session Harness
    # ---------------------------------------------------------

    harness = Harness(
        llm=llm,
        store=store,
        model=model,
        user_id=user_id,
        chat_id=chat_id,
        annotation_instruction=annotation_instruction,
        thought_process_instruction=thought_process_instruction,
        response_instruction=response_instruction,
    )

    # ---------------------------------------------------------
    # Application state
    # ---------------------------------------------------------

    app.state.chat_id = chat_id
    app.state.user_id = user_id
    app.state.model = model

    app.state.llm = llm
    app.state.store = store
    app.state.harness = harness

    # ---------------------------------------------------------
    # Routes
    # ---------------------------------------------------------

    @app.get(
        "/",
        response_class=HTMLResponse,
    )
    async def read_form(
        request: Request,
    ):
        chat_data = store.get_chat(
            chat_id=chat_id,
            user_id=user_id,
        )

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "chat_data": chat_data,
                "chat_metadata": chat_metadata(chat_data),
            },
        )

    @app.post(
        "/send",
        response_class=HTMLResponse,
    )
    async def handle_form(
        request: Request,
        system_prompt: str | None = Form(None),
        disable_system_prompt: bool = Form(False),
        message: str = Form(...),
    ):
        chat_data = store.get_chat(
            chat_id=chat_id,
            user_id=user_id,
        )

        # A blank submission means "keep the already selected prompt".
        # New chats therefore use no system prompt unless the user enters one.
        supplied_system_prompt = (system_prompt or "").strip()
        if disable_system_prompt:
            store.update_system_prompt(
                chat_id=chat_id,
                user_id=user_id,
                system_prompt=None,
            )
        elif (
            supplied_system_prompt
            and chat_data.get("system_prompt") != supplied_system_prompt
        ):
            store.update_system_prompt(
                chat_id=chat_id,
                user_id=user_id,
                system_prompt=supplied_system_prompt,
            )

        user_message = store.append_message(
            chat_id=chat_id,
            user_id=user_id,
            role="user",
            content=message,
        )

        turn_id = user_message["turn_id"]

        loop = asyncio.get_running_loop()
        events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def send_event(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(events.put_nowait, event)

        async def run_harness() -> dict[str, Any]:
            try:
                result = await run_in_threadpool(
                    harness.handle_message,
                    message=message,
                    turn_id=turn_id,
                    on_annotation_event=send_event,
                )
                send_event({
                    "type": "complete",
                    "result": {
                        "run_id": result["run_id"],
                        "total_tokens": result["total_tokens"],
                        "chunk_count": len(result["chunks"]),
                        "annotation_count": len(result["annotations"]),
                        "thought_process_token_count": len(result["thought_process_token_ids"]),
                        "response_token_count": len(result["response_token_ids"]),
                    },
                })
                return result
            except Exception as exc:
                logging.exception("Harness failed")
                send_event({"type": "error", "message": str(exc)})
                return {}

        task = asyncio.create_task(run_harness())

        async def event_stream():
            try:
                while True:
                    event = await events.get()
                    yield f"data: {json.dumps(event)}\n\n"
                    if event["type"] in {"complete", "error"}:
                        break
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    return app



def run_server(
    *,
    chat_id: str,
    user_id: str,
    model: str,
    llm: LLManager,
    store: ChatStorage,
    annotation_instruction: str,
    thought_process_instruction: str,
    response_instruction: str,
) -> None:
    """
    Start the web interface for one Harness chat session.
    """

    import uvicorn

    logging.info(
        "Starting web server: "
        "chat_id=%s user_id=%s model=%s",
        chat_id,
        user_id,
        model,
    )

    app = create_app(
        chat_id=chat_id,
        user_id=user_id,
        model=model,
        llm=llm,
        store=store,
        annotation_instruction=annotation_instruction,
        thought_process_instruction=thought_process_instruction,
        response_instruction=response_instruction,
    )

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
