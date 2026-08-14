# Harness/annotator.py

from __future__ import annotations

from typing import TypedDict

from LLManager import LLManager

from .chunker import Chunk
from .state import Annotation


# Plain-text delimiters for V1.
#
# These become part of the actual incremental token sequence:
#
# P
# + SOURCE_MARKER + C0 + ANNOTATION_MARKER + A0
# + SOURCE_MARKER + C1 + ANNOTATION_MARKER + A1
# + ...
#
SOURCE_MARKER = "\n\n[Source chunk]\n"
ANNOTATION_MARKER = "\n\n[Annotation]\n"


class AnnotationResult(TypedDict):
    annotation: Annotation
    thinking_token_ids: list[int]


class Annotator:
    """
    Performs the incremental annotation pass.

    The Annotator owns:
        - annotation instruction formatting
        - annotation-specific control delimiters
        - construction of the evolving token sequence
        - llama.cpp completion calls

    It does NOT decide which chunk runs next.
    That belongs to graph.py.
    """

    def __init__(
        self,
        *,
        llm: LLManager,
        model: str,
        instruction: str,
        n_predict: int = 64,
        temperature: float = 0.4,
    ):
        if not instruction:
            raise ValueError(
                "annotation instruction cannot be empty"
            )

        self.llm = llm
        self.model = model
        self.instruction = instruction

        self.n_predict = n_predict
        self.temperature = temperature

        # These are static, so tokenize them once rather than
        # making another /tokenize request for every chunk.
        self._source_marker_tokens = self.llm.tokenize(
            SOURCE_MARKER,
            model=self.model,
            add_special=False,
            parse_special=False,
        )

        self._annotation_marker_tokens = self.llm.tokenize(
            ANNOTATION_MARKER,
            model=self.model,
            add_special=False,
            parse_special=False,
        )

    def initialize(
        self,
    ) -> list[int]:
        """
        Create the initial token sequence for one annotation pass.

        Since subsequent /completion calls use raw token IDs, the
        initial sequence explicitly receives any model-required
        special beginning token here.
        """

        return self.llm.tokenize(
            self.instruction,
            model=self.model,
            add_special=True,
            parse_special=False,
        )

    def annotate(
        self,
        *,
        thinking_token_ids: list[int],
        chunk: Chunk,
    ) -> AnnotationResult:
        """
        Append one source chunk to the evolving context and ask the
        model to generate its annotation.

        Input sequence:

            previous_thinking
            + SOURCE_MARKER
            + current_chunk
            + ANNOTATION_MARKER

        The generated annotation is then appended to that sequence.
        """

        prompt_tokens = (
            thinking_token_ids
            + self._source_marker_tokens
            + chunk["token_ids"]
            + self._annotation_marker_tokens
        )

        print(
            "[Annotator] "
            f"chunk={chunk['index']} "
            f"prior_thinking_tokens={len(thinking_token_ids)} "
            f"source_tokens={len(chunk['token_ids'])} "
            f"prompt_tokens={len(prompt_tokens)}"
        )

        response = self.llm.complete(
            prompt=prompt_tokens,
            model=self.model,
            n_predict=self.n_predict,
            cache_prompt=True,
            return_tokens=True,
            temperature=self.temperature,
            stop=[
                SOURCE_MARKER,
            ],
        )

        print(
            "[Annotator] "
            f"chunk={chunk['index']} completed "
            f"generated_tokens={len(response.get('tokens', []))}"
        )

        annotation_text = response.get("content")
        annotation_tokens = response.get("tokens")

        if not isinstance(annotation_text, str):
            raise RuntimeError(
                "llama.cpp completion did not return "
                "annotation content"
            )

        if not isinstance(annotation_tokens, list):
            raise RuntimeError(
                "llama.cpp completion did not return "
                "generated token IDs"
            )

        annotation: Annotation = {
            "chunk_index": chunk["index"],
            "text": annotation_text,
            "token_ids": annotation_tokens,
        }

        return {
            "annotation": annotation,

            "thinking_token_ids": (
                prompt_tokens
                + annotation_tokens
            ),
        }