# Harness/graph.py

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from .annotator import Annotator
from .chunker import Chunker
from .state import HarnessState


def build_graph(
    *,
    chunker: Chunker,
    annotator: Annotator,
):

    # ---------------------------------------------------------
    # Chunk message
    # ---------------------------------------------------------

    def chunk_message(
        state: HarnessState,
    ) -> dict:
        result = chunker.chunk(
            state["message"]
        )

        if not result["chunks"]:
            raise ValueError(
                "Message produced no chunks."
            )

        return {
            "chunks": result["chunks"],
            "total_tokens": result["total_tokens"],
        }

    # ---------------------------------------------------------
    # Initialize annotation pass
    # ---------------------------------------------------------

    def initialize_annotation_pass(
        state: HarnessState,
    ) -> dict:
        return {
            "current_chunk_index": 0,
            "annotations": [],
            "thinking_token_ids":
                annotator.initialize(),
        }

    # ---------------------------------------------------------
    # Annotate current chunk
    # ---------------------------------------------------------

    def annotate_chunk(
        state: HarnessState,
    ) -> dict:
        index = state["current_chunk_index"]
        chunk = state["chunks"][index]

        result = annotator.annotate(
            thinking_token_ids=(
                state["thinking_token_ids"]
            ),
            chunk=chunk,
        )

        return {
            "annotations": (
                state["annotations"]
                + [result["annotation"]]
            ),

            "thinking_token_ids": (
                result["thinking_token_ids"]
            ),
        }

    # ---------------------------------------------------------
    # Decide whether annotation pass is finished
    # ---------------------------------------------------------

    def route_after_annotation(
        state: HarnessState,
    ) -> Literal[
        "advance_chunk",
        "end",
    ]:
        current = state["current_chunk_index"]
        last = len(state["chunks"]) - 1

        if current < last:
            return "advance_chunk"

        return "end"

    # ---------------------------------------------------------
    # Advance to next source chunk
    # ---------------------------------------------------------

    def advance_chunk(
        state: HarnessState,
    ) -> dict:
        return {
            "current_chunk_index": (
                state["current_chunk_index"] + 1
            )
        }

    # ---------------------------------------------------------
    # Build graph
    # ---------------------------------------------------------

    builder = StateGraph(HarnessState)

    builder.add_node(
        "chunk_message",
        chunk_message,
    )

    builder.add_node(
        "initialize_annotation_pass",
        initialize_annotation_pass,
    )

    builder.add_node(
        "annotate_chunk",
        annotate_chunk,
    )

    builder.add_node(
        "advance_chunk",
        advance_chunk,
    )

    builder.add_edge(
        START,
        "chunk_message",
    )

    builder.add_edge(
        "chunk_message",
        "initialize_annotation_pass",
    )

    builder.add_edge(
        "initialize_annotation_pass",
        "annotate_chunk",
    )

    builder.add_conditional_edges(
        "annotate_chunk",
        route_after_annotation,
        {
            "advance_chunk": "advance_chunk",
            "end": END,
        },
    )

    builder.add_edge(
        "advance_chunk",
        "annotate_chunk",
    )

    return builder.compile()