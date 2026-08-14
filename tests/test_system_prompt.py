from fastapi.testclient import TestClient

from app.server import create_app
from storage import ChatStorage


class FakeLLM:
    def tokenize(self, text, **kwargs):
        return [len(text)]


def make_client(tmp_path, system_prompt):
    store = ChatStorage(tmp_path)
    store.create_chat(
        "chat",
        "user",
        model="fake",
        system_prompt=system_prompt,
    )
    app = create_app(
        chat_id="chat",
        user_id="user",
        model="fake",
        llm=FakeLLM(),
        store=store,
        annotation_instruction="Annotate.",
        thought_process_instruction="Think.",
    )
    app.state.harness.handle_message = lambda **_: {
        "run_id": "run",
        "total_tokens": 1,
        "chunks": [{}],
        "annotations": [],
        "thought_process_token_ids": [],
    }
    return TestClient(app), store


def test_system_prompt_is_kept_replaced_or_disabled(tmp_path):
    client, store = make_client(tmp_path, "Existing prompt")

    response = client.post("/send", data={"message": "keep"})
    assert response.status_code == 200
    assert store.get_chat("chat", "user")["system_prompt"] == "Existing prompt"

    response = client.post(
        "/send",
        data={"message": "replace", "system_prompt": "New prompt"},
    )
    assert response.status_code == 200
    assert store.get_chat("chat", "user")["system_prompt"] == "New prompt"

    response = client.post(
        "/send",
        data={"message": "disable", "disable_system_prompt": "on"},
    )
    assert response.status_code == 200
    assert store.get_chat("chat", "user")["system_prompt"] is None
