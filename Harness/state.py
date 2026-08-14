from __future__ import annotations

from typing import TypedDict
from .chunker import Chunk



class Annotation(TypedDict):
    chunk_index: int
    text: str
    token_ids: list[int]


class HarnessState(TypedDict, total=False):
    # Identity
    chat_id: str
    user_id: str
    turn_id: str
    run_id: str

    # Input
    message: str

    # Chunking
    chunks: list[Chunk]
    total_tokens: int

    # Annotation pass
    current_chunk_index: int
    annotations: list[Annotation]

    # Actual incremental LLM context
    thinking_token_ids: list[int]