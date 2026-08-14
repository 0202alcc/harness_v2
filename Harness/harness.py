from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from LLManager import LLManager
from storage import ChatStorage

from .annotator import Annotator
from .chunker import (
    Chunker,
    DEFAULT_CHUNK_SIZE,
    FixedTokenChunker,
)
from .graph import build_graph
from .state import HarnessState
from .thought_processor import ThoughtProcessor
from .responder import Responder


class Harness:
    """
    Main Harness state machine.
    """

    def __init__(
        self,
        *,
        llm: LLManager,
        store: ChatStorage,
        model: str,
        user_id: str,
        chat_id: str,
        annotation_instruction: str,
        thought_process_instruction: str,
        response_instruction: str,
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

        self.annotator = Annotator(
            llm=self.llm,
            store=self.store,
            model=self.model,
            user_id=self.user_id,
            chat_id=self.chat_id,
            instruction=annotation_instruction,
            n_predict=32,
            temperature=0.4,
        )
        self.thought_processor = ThoughtProcessor(
            llm=self.llm,
            store=self.store,
            model=self.model,
            user_id=self.user_id,
            chat_id=self.chat_id,
            instruction=thought_process_instruction,
        )
        self.responder = Responder(
            llm=self.llm, store=self.store, model=self.model,
            user_id=self.user_id, chat_id=self.chat_id,
            instruction=response_instruction,
        )

        # -----------------------------------------------------
        # Compile Harness graph
        # -----------------------------------------------------

        self.graph = build_graph(
            chunker=self.chunker,
            annotator=self.annotator,
            thought_processor=self.thought_processor,
            responder=self.responder,
        )

    def get_chat_state(self) -> dict:
        return self.store.get_chat(
            chat_id=self.chat_id,
            user_id=self.user_id,
        )

    def handle_message(
        self,
        *,
        message: str,
        turn_id: str,
        on_annotation_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> HarnessState:
        """
        Run one incoming user message through the Harness graph.
        """

        run_id = str(uuid.uuid4())
        chat_state = self.get_chat_state()

        initial_state: HarnessState = {
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "turn_id": turn_id,
            "run_id": run_id,
            "message": message,
            "system_prompt": chat_state.get("system_prompt"),
        }

        if on_annotation_event is not None:
            initial_state["on_annotation_event"] = on_annotation_event

        result = self.graph.invoke(
            initial_state
        )

        print("\n=== HARNESS RESULT ===")
        print(f"run_id: {result['run_id']}")
        print(f"total_tokens: {result['total_tokens']}")
        print(f"chunks: {len(result['chunks'])}")

        for annotation in result["annotations"]:
            print(
                f"\n--- Annotation {annotation['chunk_index']} ---"
            )
            print(annotation["text"])

        print(
            "\nthinking_token_ids:",
            len(result["thinking_token_ids"]),
        )
        print("\n--- Thought process ---")
        print(result["thought_process"])
        print("\n--- Response ---")
        print(result["response"])
        self.store.append_message(
            chat_id=self.chat_id,
            user_id=self.user_id,
            role="assistant",
            content=result["response"],
            turn_id=turn_id,
        )
        return result
