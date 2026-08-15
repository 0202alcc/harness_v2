import json
from pathlib import Path

from Harness.thought_processor import ThoughtProcessor
from storage import ChatStorage


class FakeLLM:
    def __init__(self):
        self.tokenized_text = []

    def tokenize(self, text, **kwargs):
        self.tokenized_text.append(text)
        return [1, 2, 3]

    def stream_complete(self, **kwargs):
        yield {
            "content": '{"thought_process":"A complete synthesis."}',
            "tokens": [9, 10],
            "id_slot": 0,
            "timings": {"cache_n": 0, "prompt_n": 3},
        }


def test_thought_process_uses_system_and_annotations_not_annotation_instruction(tmp_path):
    store = ChatStorage(tmp_path)
    store.create_chat("chat", "user", model="fake")
    llm = FakeLLM()
    processor = ThoughtProcessor(
        llm=llm,
        store=store,
        model="fake",
        user_id="user",
        chat_id="chat",
        instruction="Create a complete synthesis.",
    )

    events = []
    result = processor.generate(
        message="What happened after the journey?",
        annotations=[{
            "chunk_index": 0,
            "text": "The message describes a journey.",
            "token_ids": [4],
        }],
        system_prompt="Be concise.",
        conversation_history="[Conversation history]\nUser: Earlier question",
        run_id="run",
        turn_id="turn",
        on_event=events.append,
    )

    prompt = llm.tokenized_text[0]
    assert prompt.index("Be concise.") < prompt.index("Chunk 0:")
    assert prompt.index("[Conversation history]") < prompt.index("Chunk 0:")
    assert prompt.index("What happened after the journey?") < prompt.index("Chunk 0:")
    assert prompt.index("Chunk 0:") < prompt.index("[Complete thought process]")
    assert prompt.index("[Complete thought process]") < prompt.index("Create a complete synthesis.")
    assert "Chunk 0: The message describes a journey." in prompt
    assert "I just received a message" not in prompt
    assert result["text"] == "A complete synthesis."
    assert [event["type"] for event in events] == [
        "thought_process_start",
        "thought_process_delta",
    ]

    [record] = [
        json.loads(line)
        for line in Path(store.get_llama_io_path("user", "chat")).read_text().splitlines()
    ]
    assert record["node"] == "generate_thought_process"
    assert record["request"]["includes_annotation_instruction"] is False
    assert record["request"]["json_schema"]["required"] == ["thought_process"]
    assert record["response"]["decoded_thought_process"] == "A complete synthesis."


def test_thought_process_uses_configured_markers(tmp_path):
    store = ChatStorage(tmp_path)
    store.create_chat("chat", "user", model="fake")
    llm = FakeLLM()
    processor = ThoughtProcessor(
        llm=llm,
        store=store,
        model="fake",
        user_id="user",
        chat_id="chat",
        instruction="Think.",
        markers={
            "accumulated_annotations": "\n<notes>\n",
            "thought_process": "\n<reasoning>\n",
        },
    )

    processor.generate(
        message="What should I do next?",
        annotations=[{"chunk_index": 0, "text": "A note.", "token_ids": [4]}],
        system_prompt=None,
        conversation_history=None,
        run_id="run",
        turn_id="turn",
    )

    prompt = llm.tokenized_text[0]
    assert "What should I do next?" in prompt
    assert "<notes>\nChunk 0: A note." in prompt
    assert "<reasoning>" in prompt


def test_thought_process_strips_configured_generated_prefix(tmp_path):
    class PrefixedLLM(FakeLLM):
        def stream_complete(self, **kwargs):
            yield {
                "content": '{"thought_process":"Think first: The actual note."}',
                "tokens": [9],
                "id_slot": 0,
                "timings": {"cache_n": 0, "prompt_n": 3},
            }

    store = ChatStorage(tmp_path)
    store.create_chat("chat", "user", model="fake")
    llm = PrefixedLLM()
    processor = ThoughtProcessor(
        llm=llm,
        store=store,
        model="fake",
        user_id="user",
        chat_id="chat",
        instruction="Think.",
        output_prefix="Think first: ",
    )
    events = []

    result = processor.generate(
        message="What should I do next?",
        annotations=[{"chunk_index": 0, "text": "A note.", "token_ids": [4]}],
        system_prompt=None,
        conversation_history=None,
        run_id="run",
        turn_id="turn",
        on_event=events.append,
    )

    assert result["text"] == "The actual note."
    assert events[-1] == {"type": "thought_process_delta", "content": "The actual note."}
    assert 'must begin exactly with: "Think first: "' in llm.tokenized_text[0]
