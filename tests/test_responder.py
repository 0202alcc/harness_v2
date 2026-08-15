import json
from pathlib import Path

from Harness.responder import Responder
from storage import ChatStorage


class FakeLLM:
    def __init__(self):
        self.prompt = ""
        self.complete_kwargs = {}

    def tokenize(self, text, **kwargs):
        self.prompt = text
        return [1, 2, 3]

    def apply_chat_template(self, messages, **kwargs):
        self.template_prompt = "\n".join(message["content"] for message in messages)
        return self.template_prompt

    def stream_complete(self, **kwargs):
        self.complete_kwargs = kwargs
        yield {"content": '{"response":"Here is', "tokens": [4], "timings": {"cache_n": 0, "prompt_n": 3}}
        yield {"content": ' my answer."}', "tokens": [5], "timings": {"cache_n": 0, "prompt_n": 3}}


def test_response_uses_message_and_thought_without_annotation_instruction(tmp_path):
    store = ChatStorage(tmp_path)
    store.create_chat("chat", "user", model="fake")
    llm = FakeLLM()
    responder = Responder(
        llm=llm, store=store, model="fake", user_id="user", chat_id="chat",
        instruction="Now answer the user:",
        markers={
            "user_message": "\n[user]\n",
            "thought_process": "\n[reasoning]\n",
            "assistant_response": "\n[my msg]\n",
        },
    )

    events = []
    result = responder.generate(
        message="What does this mean?",
        thought_process="The message asks for an explanation.",
        system_prompt="Be concise.",
        conversation_history="[Conversation history]\nAssistant: Earlier answer",
        run_id="run", turn_id="turn",
        on_event=events.append,
    )

    assert "What does this mean?" in llm.template_prompt
    assert "The message asks for an explanation." in llm.template_prompt
    assert "Assistant: Earlier answer" in llm.template_prompt
    assert "I just received a message" not in llm.template_prompt
    assert "Use this internal reasoning to inform your answer; do not repeat it:" in llm.template_prompt
    assert llm.template_prompt.index("Now answer the user:") < llm.template_prompt.index('Return exactly one JSON object')
    assert "[my msg]" not in llm.template_prompt
    assert result["text"] == "Here is my answer."
    assert llm.complete_kwargs["stop"] == ["<turn|>"]
    assert llm.complete_kwargs["json_schema"] == {
        "type": "object",
        "properties": {"response": {"type": "string"}},
        "required": ["response"],
        "additionalProperties": False,
    }
    assert [event["type"] for event in events] == [
        "response_start",
        "response_delta",
        "response_delta",
    ]
    [record] = [json.loads(line) for line in Path(store.get_llama_io_path("user", "chat")).read_text().splitlines()]
    assert record["node"] == "generate_final_response"
    assert record["request"]["includes_annotation_instruction"] is False
    assert record["response"]["generated_content"] == '{"response":"Here is my answer."}'
    assert record["response"]["decoded_response"] == "Here is my answer."
