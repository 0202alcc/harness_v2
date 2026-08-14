import json
from pathlib import Path

from Harness.responder import Responder
from storage import ChatStorage


class FakeLLM:
    def __init__(self):
        self.prompt = ""

    def tokenize(self, text, **kwargs):
        self.prompt = text
        return [1, 2, 3]

    def complete(self, **kwargs):
        return {"content": "Here is my answer.", "tokens": [4], "timings": {"cache_n": 0, "prompt_n": 3}}


def test_response_uses_message_and_thought_without_annotation_instruction(tmp_path):
    store = ChatStorage(tmp_path)
    store.create_chat("chat", "user", model="fake")
    llm = FakeLLM()
    responder = Responder(
        llm=llm, store=store, model="fake", user_id="user", chat_id="chat",
        instruction="Now answer the user:",
    )

    result = responder.generate(
        message="What does this mean?",
        thought_process="The message asks for an explanation.",
        system_prompt="Be concise.",
        run_id="run", turn_id="turn",
    )

    assert "What does this mean?" in llm.prompt
    assert "The message asks for an explanation." in llm.prompt
    assert "I just received a message" not in llm.prompt
    assert result["text"] == "Here is my answer."
    [record] = [json.loads(line) for line in Path(store.get_llama_io_path("user", "chat")).read_text().splitlines()]
    assert record["node"] == "generate_final_response"
    assert record["request"]["includes_annotation_instruction"] is False
