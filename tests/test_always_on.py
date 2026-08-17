from __future__ import annotations

import asyncio

from app.always_on import AlwaysOnOrchestrator
from storage import ChatStorage


class FakeHarness:
    def __init__(self, response: str = "assistant reply"):
        self.response = response

    def handle_message(self, *, message, on_annotation_event=None, **kwargs):
        if on_annotation_event:
            on_annotation_event({"type": "response_start"})
            on_annotation_event({"type": "response_delta", "content": self.response})
            on_annotation_event({"type": "response_complete", "text": self.response})
        return {
            "response": self.response,
            "run_id": kwargs["run_id"],
            "chunks": [],
            "annotations": [],
            "total_tokens": 0,
            "thought_process_token_ids": [],
            "response_token_ids": [],
        }


async def wait_for(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for background inference")


def make_orchestrator(tmp_path):
    store = ChatStorage(str(tmp_path))
    store.create_chat("chat", "user", model="fake")
    harness = FakeHarness()
    return store, AlwaysOnOrchestrator(
        store=store,
        get_harness=lambda _: harness,
        user_id="user",
    )


def test_text_observation_is_durable_and_eventually_produces_response(tmp_path):
    async def scenario():
        store, orchestrator = make_orchestrator(tmp_path)
        accepted = await orchestrator.submit_text(
            session_id="chat",
            content="hello",
            observation_id="observation-1",
        )
        assert accepted["status"] == "accepted"

        await wait_for(lambda: len(store.get_chat("chat", "user")["messages"]) == 2)
        messages = store.get_chat("chat", "user")["messages"]
        assert [message["content"] for message in messages] == ["hello", "assistant reply"]
        event_types = [event["event_type"] for event in store.get_events(chat_id="chat", user_id="user")]
        assert "observation_received" in event_types
        assert "inference_completed" in event_types

    asyncio.run(scenario())


def test_duplicate_observation_does_not_create_a_second_turn(tmp_path):
    async def scenario():
        store, orchestrator = make_orchestrator(tmp_path)
        first = await orchestrator.submit_text(
            session_id="chat", content="hello", observation_id="same-id"
        )
        duplicate = await orchestrator.submit_text(
            session_id="chat", content="hello", observation_id="same-id"
        )
        assert first["status"] == "accepted"
        assert duplicate == {"observation_id": "same-id", "status": "duplicate"}
        await wait_for(lambda: len(store.get_chat("chat", "user")["messages"]) == 2)

    asyncio.run(scenario())


def test_duplicate_observation_is_detected_after_orchestrator_restart(tmp_path):
    async def scenario():
        store, first_orchestrator = make_orchestrator(tmp_path)
        await first_orchestrator.submit_text(
            session_id="chat", content="hello", observation_id="durable-id"
        )
        await wait_for(lambda: len(store.get_chat("chat", "user")["messages"]) == 2)

        restarted = AlwaysOnOrchestrator(
            store=store,
            get_harness=lambda _: FakeHarness(),
            user_id="user",
        )
        duplicate = await restarted.submit_text(
            session_id="chat", content="hello", observation_id="durable-id"
        )
        assert duplicate["status"] == "duplicate"
        assert len(store.get_chat("chat", "user")["messages"]) == 2

    asyncio.run(scenario())
