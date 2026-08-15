from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict
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
    system_prompt: str | None
    conversation_history: str | None

    # Chunking
    chunks: list[Chunk]
    total_tokens: int

    # Annotation pass
    current_chunk_index: int
    annotations: list[Annotation]
    thought_process: str
    thought_process_token_ids: list[int]
    response: str
    response_token_ids: list[int]

    # Actual incremental LLM context
    thinking_token_ids: list[int]

    # UI/server hook. It is deliberately ephemeral and never persisted.
    on_annotation_event: Callable[[dict[str, Any]], None]
