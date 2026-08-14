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


def test_chunk_graph():
    from Harness.graph import build_graph

    graph = build_graph(
        chunker=FakeChunker()
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