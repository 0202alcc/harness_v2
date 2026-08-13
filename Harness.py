from __future__ import annotations
from LLManager import LLManager
from storage import ChatStorage
from typing import TypedDict

CHUNK_SIZE = 512


class Chunk(TypedDict):
    index: int
    token_start: int
    token_end: int
    token_count: int
    token_ids: list[int]
    text: str


class Harness:
    """
    Main Harness state machine.

    The Harness coordinates:
        - chat state
        - tokenization/chunking
        - future LangGraph reasoning pipeline
        - LLM operations

    It does not directly access llama.cpp HTTP endpoints or storage files.
    """

    def __init__(
        self,
        *,
        llm: LLManager,
        store: ChatStorage,
        model: str,
        user_id: str,
        chat_id: str,
    ):
        self.llm = llm
        self.store = store

        self.model = model
        self.user_id = user_id
        self.chat_id = chat_id

    def get_chat_state(self) -> dict:
        """
        Fetch the current authoritative state of this chat.
        """
        return self.store.get_chat(
            chat_id=self.chat_id,
            user_id=self.user_id,
        )

    def chunk_message(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
    ) -> dict:
        """
        Tokenize a user message with the current model's tokenizer
        and split it into fixed-size source-token chunks.
        """

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        # Use llama.cpp's actual tokenizer.
        token_ids = self.llm.tokenize(
            text,
            model=self.model,
            add_special=False,
        )

        chunks: list[Chunk] = []

        for start in range(
            0,
            len(token_ids),
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                len(token_ids),
            )

            chunk_token_ids = token_ids[start:end]

            chunk_text = self.llm.detokenize(
                chunk_token_ids,
                model=self.model,
            )

            chunks.append(
                {
                    "index": len(chunks),
                    "token_start": start,
                    "token_end": end,
                    "token_count": len(
                        chunk_token_ids
                    ),
                    "token_ids": chunk_token_ids,
                    "text": chunk_text,
                }
            )

        return {
            "input_text": text,
            "model": self.model,
            "total_tokens": len(token_ids),
            "chunks": chunks,
        }