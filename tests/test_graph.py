# tests/test_graph.py

class FakeChunker:
    def chunk(self, text):
        return {
            "input_text": text,
            "model": "fake",
            "total_tokens": 3,
            "chunks": [
                {
                    "index": 0,
                    "token_start": 0,
                    "token_end": 3,
                    "token_count": 3,
                    "token_ids": [1, 2, 3],
                    "text": text,
                }
            ],
        }


class FakeAnnotator:
    def initialize(self, *, system_prompt=None):
        return [0]

    def annotate(self, *, thinking_token_ids, chunk, **kwargs):
        return {
            "annotation": {
                "chunk_index": chunk["index"],
                "text": "annotation",
                "token_ids": [4],
            },
            "thinking_token_ids": thinking_token_ids + [4],
        }


class FakeThoughtProcessor:
    def generate(self, **kwargs):
        return {"text": "thought", "token_ids": [5]}


def test_chunk_graph():
    from Harness.graph import build_graph

    graph = build_graph(
        chunker=FakeChunker(),
        annotator=FakeAnnotator(),
        thought_processor=FakeThoughtProcessor(),
    )

    result = graph.invoke({
        "message": "hello world",
        "chat_id": "chat",
        "user_id": "user",
        "turn_id": "turn",
        "run_id": "run",
    })

    assert result["total_tokens"] == 3
    assert len(result["chunks"]) == 1
    assert result["annotations"][0]["text"] == "annotation"
    assert result["thought_process"] == "thought"
