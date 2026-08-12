from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from transformers import AutoTokenizer


CHUNK_SIZE = 512

# IMPORTANT:
# Replace this with the tokenizer for the model that will eventually
# consume these chunks.
TOKENIZER_NAME = "YOUR_MODEL_REPO"

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    use_fast=True,
)


class Chunk(TypedDict):
    index: int
    token_start: int
    token_end: int
    token_count: int
    text: str


class ChunkingState(TypedDict):
    input_text: str
    chunks: list[Chunk]
    total_tokens: int


def fixed_token_chunker(state: ChunkingState) -> dict:
    """
    Split input_text into non-overlapping chunks of at most 512 tokens.

    Example:
        1300 input tokens
            ->
        chunk 0: tokens    0-511   (512)
        chunk 1: tokens  512-1023  (512)
        chunk 2: tokens 1024-1299  (276)
    """

    # Tokenize WITHOUT adding BOS/EOS/etc.
    token_ids = tokenizer.encode(
        state["input_text"],
        add_special_tokens=False,
    )

    chunks: list[Chunk] = []

    for start in range(0, len(token_ids), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(token_ids))

        chunk_token_ids = token_ids[start:end]

        chunk_text = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        chunks.append(
            {
                "index": len(chunks),
                "token_start": start,
                "token_end": end,
                "token_count": len(chunk_token_ids),
                "text": chunk_text,
            }
        )

    return {
        "chunks": chunks,
        "total_tokens": len(token_ids),
    }


# -------------------------
# Build LangGraph
# -------------------------

builder = StateGraph(ChunkingState)

builder.add_node("fixed_token_chunker", fixed_token_chunker)

builder.add_edge(START, "fixed_token_chunker")
builder.add_edge("fixed_token_chunker", END)

chunking_graph = builder.compile()


if __name__ == "__main__":
    result = chunking_graph.invoke(
        {
            "input_text": (
                "Your potentially very long user message goes here. "
                "This can contain thousands of tokens."
            ),
            "chunks": [],
            "total_tokens": 0,
        }
    )

    print(f"Total tokens: {result['total_tokens']}")
    print(f"Total chunks: {len(result['chunks'])}")

    for chunk in result["chunks"]:
        print(
            f"\n--- Chunk {chunk['index']} ---\n"
            f"tokens: {chunk['token_start']}:{chunk['token_end']}\n"
            f"count: {chunk['token_count']}\n"
            f"{chunk['text']}"
        )