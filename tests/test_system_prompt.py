from fastapi.testclient import TestClient
from urllib.parse import parse_qs, urlparse

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
        response_instruction="Respond.",
    )
    app.state.harness.handle_message = lambda **_: {
        "run_id": "run",
        "total_tokens": 1,
        "chunks": [{}],
        "annotations": [],
        "thought_process_token_ids": [],
        "response_token_ids": [],
        "response": "response",
    }
    return TestClient(app), store


def test_system_prompt_is_kept_replaced_or_disabled(tmp_path):
    client, store = make_client(tmp_path, "Existing prompt")

    response = client.post("/send", data={"message": "keep", "chat_id": "chat"})
    assert response.status_code == 200
    assert store.get_chat("chat", "user")["system_prompt"] == "Existing prompt"

    response = client.post(
        "/send",
        data={"message": "replace", "system_prompt": "New prompt", "chat_id": "chat"},
    )
    assert response.status_code == 200
    assert store.get_chat("chat", "user")["system_prompt"] == "New prompt"

    response = client.post(
        "/send",
        data={"message": "disable", "disable_system_prompt": "on", "chat_id": "chat"},
    )
    assert response.status_code == 200
    assert store.get_chat("chat", "user")["system_prompt"] is None


def test_chat_picker_lists_existing_chats_and_creates_a_new_one(tmp_path):
    client, store = make_client(tmp_path, None)
    store.create_chat("other-chat", "user", model="fake")

    page = client.get("/")
    assert page.status_code == 200
    assert 'value="chat" selected' in page.text
    assert 'value="other-chat"' in page.text

    response = client.post("/chats", follow_redirects=False)
    assert response.status_code == 303
    new_chat_id = parse_qs(urlparse(response.headers["location"]).query)["chat_id"][0]
    assert store.get_chat(new_chat_id, "user")["messages"] == []
    assert store.get_events(chat_id=new_chat_id, user_id="user")[0]["event_type"] == "chat_initialized"

    new_chat_page = client.get(response.headers["location"])
    assert f'value="{new_chat_id}" selected' in new_chat_page.text
    assert 'id="app-version"' in new_chat_page.text
