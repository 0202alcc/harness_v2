import json

from app.github_issues import MAX_TRACE_CHARACTERS, format_issue_body
from storage import ChatStorage


def test_issue_body_is_run_scoped_and_bounded():
    body = format_issue_body(
        chat_id="chat",
        run_id="run",
        model="model",
        description="The reply stopped early.",
        trace_records=[{"run_id": "run", "content": "x" * (MAX_TRACE_CHARACTERS + 20)}],
    )

    assert "The reply stopped early." in body
    assert "`chat`" in body
    assert "`run`" in body
    assert "```jsonl" in body
    assert "Trace was truncated" in body


def test_storage_returns_only_requested_run_trace(tmp_path):
    store = ChatStorage(tmp_path)
    store.create_chat("chat", "user", model="fake")
    path = store.get_llama_io_path("user", "chat")
    path.write_text(
        "\n".join([
            json.dumps({"run_id": "first", "node": "annotate_chunk"}),
            json.dumps({"run_id": "second", "node": "generate_thought_process"}),
        ]) + "\n",
        encoding="utf-8",
    )

    assert store.get_llama_io_for_run(
        user_id="user", chat_id="chat", run_id="second",
    ) == [{"run_id": "second", "node": "generate_thought_process"}]
