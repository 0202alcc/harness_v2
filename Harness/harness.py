from __future__ import annotations
from .chunker import (
    Chunker,
    ChunkResult,
    DEFAULT_CHUNK_SIZE,
    FixedTokenChunker,
)
from LLManager import LLManager
from storage import ChatStorage

class Harness:
    """
    Main Harness state machine.

    Responsibilities:
        - coordinate Harness pipeline/state
        - access current chat state
        - invoke pipeline components
        - coordinate LLM operations

    Individual pipeline operations such as chunking live in their
    own modules.
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

        # -----------------------------------------------------
        # Pipeline components
        # -----------------------------------------------------

        self.chunker: Chunker = FixedTokenChunker(
            llm=self.llm,
            model=self.model,
            chunk_size=DEFAULT_CHUNK_SIZE,
        )

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
    ) -> ChunkResult:
        """
        Run the configured chunking stage.
        """

        return self.chunker.chunk(text)