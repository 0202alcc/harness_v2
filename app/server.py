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

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def render_chat_messages(
    chat_data: dict,
) -> dict:
    for msg in chat_data.get("messages", []):
        msg["rendered_content"] = markdown.markdown(
            msg["content"],
            extensions=[
                "fenced_code",
                "tables",
            ],
        )

    return chat_data

def create_app(
    *,
    chat_id: str,
    user_id: str,
    model: str,
    llm: LLManager,
    store: ChatStorage,
    annotation_instruction: str,
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

        chat_data = render_chat_messages(chat_data)

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
        chat_data = store.get_chat(
            chat_id=chat_id,
            user_id=user_id,
        )

        if chat_data.get("system_prompt") != system_prompt:
            store.update_system_prompt(
                chat_id=chat_id,
                user_id=user_id,
                system_prompt=system_prompt,
            )

        user_message = store.append_message(
            chat_id=chat_id,
            user_id=user_id,
            role="user",
            content=message,
        )

        turn_id = user_message["turn_id"]

        result = await run_in_threadpool(
            harness.handle_message,
            message=message,
            turn_id=turn_id,
        )

        logging.info(
            "Harness completed: "
            "chat_id=%s turn_id=%s run_id=%s "
            "tokens=%d chunks=%d",
            chat_id,
            turn_id,
            result["run_id"],
            result["total_tokens"],
            len(result["chunks"]),
        )

        result_message = (
            f"Message split into "
            f"{len(result['chunks'])} chunk(s), "
            f"{result['total_tokens']} tokens. "
            f"Generated "
            f"{len(result['annotations'])} annotation(s)."
        )

        chat_data = store.get_chat(
            chat_id=chat_id,
            user_id=user_id,
        )

        chat_data = render_chat_messages(
            chat_data
        )

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
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
    annotation_instruction: str,
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
    )

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )