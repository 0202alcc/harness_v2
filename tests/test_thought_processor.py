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

    def complete(self, **kwargs):
        return {
            "content": "A complete synthesis.",
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

    result = processor.generate(
        annotations=[{
            "chunk_index": 0,
            "text": "The message describes a journey.",
            "token_ids": [4],
        }],
        system_prompt="Be concise.",
        run_id="run",
        turn_id="turn",
    )

    prompt = llm.tokenized_text[0]
    assert prompt.index("Be concise.") < prompt.index("Chunk 0:")
    assert prompt.index("Chunk 0:") < prompt.index("[Complete thought process]")
    assert prompt.index("[Complete thought process]") < prompt.index("Create a complete synthesis.")
    assert "Chunk 0: The message describes a journey." in prompt
    assert "I just received a message" not in prompt
    assert result["text"] == "A complete synthesis."

    [record] = [
        json.loads(line)
        for line in Path(store.get_llama_io_path("user", "chat")).read_text().splitlines()
    ]
    assert record["node"] == "generate_thought_process"
    assert record["request"]["includes_annotation_instruction"] is False
