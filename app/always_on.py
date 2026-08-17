"""Always-on, ROS-shaped orchestration for the text Harness.

The module intentionally has no ROS dependency yet.  Its typed topics,
long-running inference action, cancellation semantics, lifecycle state, and
event replay boundary are stable seams for a later ROS 2 adapter.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi.concurrency import run_in_threadpool

from Harness import Harness
from storage import ChatStorage


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ObservationEnvelope:
    """A modality-neutral, idempotent input accepted by the system."""

    observation_id: str
    session_id: str
    source_id: str
    modality: str
    sequence: int
    captured_at: str
    received_at: str
    correlation_id: str
    payload: str
    schema_version: int = 1

    def to_event_data(self) -> dict[str, Any]:
        return asdict(self)


class SessionEventBroker:
    """Fan-out for live gateway consumers; storage remains the source of truth."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)

    def subscribe(self, session_id: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers[session_id].add(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers[session_id].discard(queue)

    def publish(self, session_id: str, event: dict[str, Any]) -> None:
        """Best-effort live fanout. Durable events are replayable from storage."""
        for queue in tuple(self._subscribers[session_id]):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A slow websocket must reconnect using its durable cursor.
                self._subscribers[session_id].discard(queue)


@dataclass
class _SessionWork:
    observation: ObservationEnvelope
    turn_id: str
    run_id: str
    generation: int


class AlwaysOnOrchestrator:
    """Owns asynchronous inference goals, not HTTP requests.

    One model goal runs per session.  New submitted text supersedes an active
    goal and replaces any pending goal.  The in-flight model call is allowed to
    finish because the current llama.cpp interface cannot cooperatively cancel
    a decoding request; its output is discarded when superseded.  A future ROS
    action adapter can map ``superseded`` to native goal cancellation.
    """

    def __init__(
        self,
        *,
        store: ChatStorage,
        get_harness: Callable[[str], Harness],
        user_id: str,
    ) -> None:
        self.store = store
        self.get_harness = get_harness
        self.user_id = user_id
        self.broker = SessionEventBroker()
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._pending: dict[str, _SessionWork] = {}
        self._generation: dict[str, int] = defaultdict(int)
        self._seen_observations: set[str] = set()

    def _publish_durable(self, session_id: str, event: dict[str, Any]) -> None:
        self.broker.publish(session_id, {"type": "session_event", "event": event})

    def _append_event(
        self,
        *,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        event = self.store.append_event(
            chat_id=session_id,
            user_id=self.user_id,
            event_type=event_type,
            data=data,
            turn_id=turn_id,
        )
        self._publish_durable(session_id, event)
        return event

    def _observation_exists(self, session_id: str, observation_id: str) -> bool:
        """Check the durable log so idempotency survives a process restart."""
        for event in self.store.get_events(chat_id=session_id, user_id=self.user_id):
            if event.get("event_type") != "observation_received":
                continue
            for change in event.get("changes", []):
                data = change.get("data", {})
                if data.get("observation_id") == observation_id:
                    return True
        return False

    async def submit_text(
        self,
        *,
        session_id: str,
        content: str,
        source_id: str = "web",
        sequence: int = 0,
        observation_id: str | None = None,
        captured_at: str | None = None,
    ) -> dict[str, Any]:
        """Durably accept text and schedule it without waiting for inference."""
        content = content.strip()
        if not content:
            raise ValueError("Text observations may not be blank.")
        # Validate the session before making any mutation.
        self.store.get_chat(chat_id=session_id, user_id=self.user_id)

        observation_id = observation_id or str(uuid.uuid4())
        async with self._locks[session_id]:
            if (
                observation_id in self._seen_observations
                or self._observation_exists(session_id, observation_id)
            ):
                return {"observation_id": observation_id, "status": "duplicate"}
            self._seen_observations.add(observation_id)

            now = utc_now()
            observation = ObservationEnvelope(
                observation_id=observation_id,
                session_id=session_id,
                source_id=source_id,
                modality="TEXT",
                sequence=sequence,
                captured_at=captured_at or now,
                received_at=now,
                correlation_id=observation_id,
                payload=content,
            )
            user_message = self.store.append_message(
                chat_id=session_id,
                user_id=self.user_id,
                role="user",
                content=content,
            )
            self._append_event(
                session_id=session_id,
                event_type="observation_received",
                data=observation.to_event_data(),
                turn_id=user_message["turn_id"],
            )

            self._generation[session_id] += 1
            work = _SessionWork(
                observation=observation,
                turn_id=user_message["turn_id"],
                run_id=str(uuid.uuid4()),
                generation=self._generation[session_id],
            )
            previous = self._pending.get(session_id)
            self._pending[session_id] = work
            if session_id in self._workers and not self._workers[session_id].done():
                self._append_event(
                    session_id=session_id,
                    event_type="inference_superseded",
                    data={"superseded_by": observation_id, "reason": "new_text_observation"},
                )
            elif previous is not None:
                self._append_event(
                    session_id=session_id,
                    event_type="inference_superseded",
                    data={"superseded_by": observation_id, "reason": "pending_replaced"},
                )
            else:
                self._workers[session_id] = asyncio.create_task(self._worker(session_id))

            return {
                "observation_id": observation_id,
                "turn_id": user_message["turn_id"],
                "status": "accepted",
            }

    async def cancel(self, session_id: str) -> None:
        async with self._locks[session_id]:
            self._generation[session_id] += 1
            self._pending.pop(session_id, None)
            self._append_event(
                session_id=session_id,
                event_type="inference_cancel_requested",
                data={"reason": "client_request"},
            )

    async def _worker(self, session_id: str) -> None:
        while True:
            async with self._locks[session_id]:
                work = self._pending.pop(session_id, None)
                if work is None:
                    # Remove the worker before releasing the lock.  Otherwise
                    # a submitter could observe a task that is about to exit,
                    # enqueue work, and never start a replacement worker.
                    self._workers.pop(session_id, None)
                    return
                self._append_event(
                    session_id=session_id,
                    event_type="inference_started",
                    data={"run_id": work.run_id, "observation_id": work.observation.observation_id},
                    turn_id=work.turn_id,
                )

            loop = asyncio.get_running_loop()

            def on_feedback(event: dict[str, Any]) -> None:
                loop.call_soon_threadsafe(
                    self.broker.publish,
                    session_id,
                    {
                        "type": "inference_feedback",
                        "run_id": work.run_id,
                        "observation_id": work.observation.observation_id,
                        "event": event,
                    },
                )

            try:
                result = await run_in_threadpool(
                    self.get_harness(session_id).handle_message,
                    message=work.observation.payload,
                    turn_id=work.turn_id,
                    run_id=work.run_id,
                    on_annotation_event=on_feedback,
                    persist_response=False,
                )
            except Exception as exc:
                self._append_event(
                    session_id=session_id,
                    event_type="inference_failed",
                    data={"run_id": work.run_id, "message": str(exc)},
                    turn_id=work.turn_id,
                )
                continue

            async with self._locks[session_id]:
                if work.generation != self._generation[session_id]:
                    self._append_event(
                        session_id=session_id,
                        event_type="inference_discarded",
                        data={"run_id": work.run_id, "reason": "superseded_or_cancelled"},
                        turn_id=work.turn_id,
                    )
                    continue

                assistant_message = self.store.append_message(
                    chat_id=session_id,
                    user_id=self.user_id,
                    role="assistant",
                    content=result["response"],
                    turn_id=work.turn_id,
                )
                self._append_event(
                    session_id=session_id,
                    event_type="inference_completed",
                    data={
                        "run_id": work.run_id,
                        "response": result["response"],
                        "assistant_message_id": assistant_message["message_id"],
                    },
                    turn_id=work.turn_id,
                )
