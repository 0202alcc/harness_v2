# app/server.py

from __future__ import annotations

import logging

from fastapi import FastAPI, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from Harness import Harness
from LLManager import LLManager
import markdown
from storage import ChatStorage


templates = Jinja2Templates(
    directory="./app/templates"
)


def create_app(
    *,
    chat_id: str,
    user_id: str,
    model: str,
    llm: LLManager,
    store: ChatStorage,
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

        for msg in chat_data["messages"]:
            msg["rendered_content"] = markdown.markdown(
                msg["content"],
                extensions=[
                    "fenced_code",
                    "tables",
                ],
            )

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "chat_data": chat_data,
            },
        )

    @app.post(
        "/send",
        response_class=HTMLResponse,
    )
    async def handle_form(
        request: Request,
        system_prompt: str = Form(...),
        message: str = Form(...),
    ):
        # -----------------------------------------------------
        # Update persistent system prompt if necessary
        # -----------------------------------------------------

        chat_data = store.get_chat(
            chat_id=chat_id,
            user_id=user_id,
        )

        if (
            chat_data.get("system_prompt")
            != system_prompt
        ):
            chat_data = store.update_system_prompt(
                chat_id=chat_id,
                user_id=user_id,
                system_prompt=system_prompt,
            )

        for msg in chat_data["messages"]:
            msg["rendered_content"] = markdown.markdown(
                msg["content"],
                extensions=[
                    "fenced_code",
                    "tables",
                ],
            )
        # -----------------------------------------------------
        # Persist incoming user message
        # -----------------------------------------------------

        user_message = store.append_message(
            chat_id=chat_id,
            user_id=user_id,
            role="user",
            content=message,
        )

        turn_id = user_message["turn_id"]

        # -----------------------------------------------------
        # Run Harness
        #
        # For now this only performs your chunking pipeline.
        # Later this becomes harness.handle_message(...)
        # -----------------------------------------------------

        chunk_result = await run_in_threadpool(
            harness.chunk_message,
            message,
        )

        logging.info(
            "Chunked user message: "
            "chat_id=%s turn_id=%s tokens=%d chunks=%d",
            chat_id,
            turn_id,
            chunk_result["total_tokens"],
            len(chunk_result["chunks"]),
        )

        # -----------------------------------------------------
        # Temporary V1 response
        # -----------------------------------------------------

        result_message = (
            f"Message received and split into "
            f"{len(chunk_result['chunks'])} chunk(s) "
            f"({chunk_result['total_tokens']} tokens)."
        )

        # Refresh state because the user message has now been
        # persisted.
        chat_data = store.get_chat(
            chat_id=chat_id,
            user_id=user_id,
        )

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "system_prompt": system_prompt,
                "message": message,
                "reply": result_message,
                "chat_data": chat_data,
            },
        )

    return app


def run_server(
    *,
    chat_id: str,
    user_id: str,
    model: str,
    llm: LLManager,
    store: ChatStorage,
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
    )

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )