# app/server.py

from __future__ import annotations

import logging
import asyncio
import json
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from Harness import Harness
from LLManager import LLManager
from storage import ChatNotFoundError, ChatStorage
from app.github_issues import format_issue_body
from app.always_on import AlwaysOnOrchestrator


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


def latest_event_number(store: ChatStorage, *, chat_id: str, user_id: str) -> int:
    events = store.get_events(chat_id=chat_id, user_id=user_id)
    return events[-1]["event_number"] if events else -1


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
    markers: dict[str, str] | None = None,
    thought_process_output_prefix: str | None = None,
    full_bandwidth_feedback: bool = False,
    github_repository: str | None = None,
    github_token: str | None = None,
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
        markers=markers,
        thought_process_output_prefix=thought_process_output_prefix,
        full_bandwidth_feedback=full_bandwidth_feedback,
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
    app.state.github_repository = github_repository
    app.state.github_token = github_token

    def get_harness(selected_chat_id: str) -> Harness:
        """Reuse the startup Harness for its chat; build one for another chat."""
        if selected_chat_id == app.state.chat_id:
            return app.state.harness

        selected_chat = store.get_chat(
            chat_id=selected_chat_id,
            user_id=user_id,
        )
        selected_model = selected_chat.get("current_model", {}).get("model") or model
        return Harness(
            llm=llm,
            store=store,
            model=selected_model,
            user_id=user_id,
            chat_id=selected_chat_id,
            annotation_instruction=annotation_instruction,
            thought_process_instruction=thought_process_instruction,
            response_instruction=response_instruction,
            markers=markers,
            thought_process_output_prefix=thought_process_output_prefix,
            full_bandwidth_feedback=full_bandwidth_feedback,
        )

    app.state.orchestrator = AlwaysOnOrchestrator(
        store=store,
        get_harness=get_harness,
        user_id=user_id,
    )

    # ---------------------------------------------------------
    # Routes
    # ---------------------------------------------------------

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        """Container readiness probe; inference-provider health is separate."""
        return {"ok": True}

    @app.get(
        "/",
        response_class=HTMLResponse,
    )
    async def read_form(
        request: Request,
        chat_id: str | None = None,
    ):
        selected_chat_id = chat_id or app.state.chat_id
        try:
            chat_data = store.get_chat(
                chat_id=selected_chat_id,
                user_id=user_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "chat_data": chat_data,
                "chat_metadata": chat_metadata(chat_data),
                "chat_list": store.list_chats(user_id),
                "selected_chat_id": selected_chat_id,
                "event_cursor": latest_event_number(
                    store, chat_id=selected_chat_id, user_id=user_id
                ),
            },
        )

    @app.post("/chats")
    async def create_chat() -> RedirectResponse:
        new_chat_id = str(uuid.uuid4())
        store.create_chat(
            chat_id=new_chat_id,
            user_id=user_id,
            provider="llama.cpp",
            model=model,
            system_prompt=None,
        )
        return RedirectResponse(url=f"/?chat_id={new_chat_id}", status_code=303)

    @app.post("/sessions/{session_id}/observations/text", status_code=202)
    async def submit_text_observation(
        session_id: str,
        payload: dict[str, Any],
    ) -> JSONResponse:
        """Accept an observation without holding the HTTP request for the LLM."""
        try:
            accepted = await app.state.orchestrator.submit_text(
                session_id=session_id,
                content=str(payload.get("content", "")),
                source_id=str(payload.get("source_id", "web")),
                sequence=int(payload.get("sequence", 0)),
                observation_id=payload.get("observation_id"),
                captured_at=payload.get("captured_at"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ChatNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        return JSONResponse(accepted, status_code=202)

    @app.websocket("/ws/sessions/{session_id}")
    async def session_socket(
        websocket: WebSocket,
        session_id: str,
        after_event_number: int = -1,
    ) -> None:
        """Gateway bridge for live output plus durable-event replay."""
        try:
            store.get_chat(chat_id=session_id, user_id=user_id)
        except ChatNotFoundError:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        queue = app.state.orchestrator.broker.subscribe(session_id)
        try:
            for event in store.get_events(
                chat_id=session_id,
                user_id=user_id,
                after_event_number=after_event_number,
            ):
                await websocket.send_json({"type": "session_event", "event": event})

            async def receive_commands() -> None:
                while True:
                    message = await websocket.receive_json()
                    message_type = message.get("type")
                    if message_type == "text_observation":
                        await app.state.orchestrator.submit_text(
                            session_id=session_id,
                            content=str(message.get("content", "")),
                            source_id=str(message.get("source_id", "web")),
                            sequence=int(message.get("sequence", 0)),
                            observation_id=message.get("observation_id"),
                            captured_at=message.get("captured_at"),
                        )
                    elif message_type == "cancel_inference":
                        await app.state.orchestrator.cancel(session_id)
                    else:
                        await websocket.send_json({
                            "type": "gateway_error",
                            "message": f"Unsupported message type: {message_type!r}",
                        })

            receiver = asyncio.create_task(receive_commands())
            try:
                while True:
                    event = await queue.get()
                    await websocket.send_json(event)
            finally:
                receiver.cancel()
        except WebSocketDisconnect:
            pass
        finally:
            app.state.orchestrator.broker.unsubscribe(session_id, queue)

    @app.post("/issues")
    async def create_issue(
        title: str = Form(...),
        description: str = Form(""),
        chat_id: str = Form(...),
        run_id: str = Form(...),
        include_trace: bool = Form(False),
    ) -> JSONResponse:
        """Create a user-requested GitHub issue for one traced run."""
        if not github_repository or not github_token:
            raise HTTPException(
                status_code=503,
                detail=(
                    "GitHub issue reporting is not configured on this server."
                ),
            )
        if not include_trace:
            raise HTTPException(
                status_code=400,
                detail="Confirm raw trace attachment before submitting an issue.",
            )
        try:
            store.get_chat(chat_id=chat_id, user_id=user_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc

        trace_records = store.get_llama_io_for_run(
            user_id=user_id,
            chat_id=chat_id,
            run_id=run_id,
        )
        if not trace_records:
            raise HTTPException(
                status_code=404,
                detail="No llama.cpp trace was recorded for this run.",
            )

        body = format_issue_body(
            chat_id=chat_id,
            run_id=run_id,
            model=model,
            description=description,
            trace_records=trace_records,
        )
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"https://api.github.com/repos/{github_repository}/issues",
                    headers={
                        "Accept": "application/vnd.github+json",
                        "Authorization": f"Bearer {github_token}",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                    json={"title": title.strip(), "body": body},
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logging.exception("GitHub issue creation failed")
            raise HTTPException(
                status_code=502,
                detail="GitHub could not create the issue.",
            ) from exc

        issue = response.json()
        return JSONResponse({
            "number": issue.get("number"),
            "url": issue.get("html_url"),
        })

    @app.post(
        "/send",
        response_class=HTMLResponse,
    )
    async def handle_form(
        request: Request,
        system_prompt: str | None = Form(None),
        disable_system_prompt: bool = Form(False),
        message: str = Form(...),
        chat_id: str = Form(...),
    ):
        try:
            chat_data = store.get_chat(chat_id=chat_id, user_id=user_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        selected_harness = get_harness(chat_id)

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

        run_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def send_event(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(events.put_nowait, event)

        async def run_harness() -> dict[str, Any]:
            try:
                result = await run_in_threadpool(
                    selected_harness.handle_message,
                    message=message,
                    turn_id=turn_id,
                    run_id=run_id,
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
                send_event({"type": "error", "message": str(exc), "run_id": run_id})
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
    markers: dict[str, str] | None = None,
    thought_process_output_prefix: str | None = None,
    full_bandwidth_feedback: bool = False,
    github_repository: str | None = None,
    github_token: str | None = None,
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
        markers=markers,
        thought_process_output_prefix=thought_process_output_prefix,
        full_bandwidth_feedback=full_bandwidth_feedback,
        github_repository=github_repository,
        github_token=github_token,
    )

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
