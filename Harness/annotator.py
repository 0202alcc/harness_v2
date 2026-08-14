# Harness/annotator.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict
import time

from LLManager import LLManager, ProviderError

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

# There are three initial attempts: the original request plus these retries.
INITIAL_RETRY_DELAYS_SECONDS = (0.25, 1.0)
# A continuation is a new completion request, so give the model server a
# moment to release the interrupted stream before submitting it.
CONTINUATION_RETRY_DELAYS_SECONDS = (0.25, 1.0)


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
        on_event: Callable[[dict[str, Any]], None] | None = None,
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

        if on_event is not None:
            on_event({
                "type": "annotation_start",
                "chunk_index": chunk["index"],
            })

        content_parts: list[str] = []
        annotation_tokens: list[int] = []
        initial_retry_index = 0
        continuation_retry_index = 0

        while True:
            request_prompt = prompt_tokens + annotation_tokens
            remaining_tokens = self.n_predict - len(annotation_tokens)

            if remaining_tokens <= 0:
                break

            received_this_attempt = False

            try:
                for event in self.llm.stream_complete(
                    prompt=request_prompt,
                    model=self.model,
                    n_predict=remaining_tokens,
                    cache_prompt=True,
                    return_tokens=True,
                    temperature=self.temperature,
                    stop=[
                        SOURCE_MARKER,
                    ],
                ):
                    content = event.get("content", "")
                    tokens = event.get("tokens", [])

                    if (
                        isinstance(content, str)
                        and content
                        and not (
                            isinstance(tokens, list)
                            and tokens
                        )
                    ):
                        raise RuntimeError(
                            "llama.cpp streamed annotation content "
                            "without raw token IDs; cannot safely "
                            "resume after a connection drop"
                        )

                    if isinstance(tokens, list) and tokens:
                        annotation_tokens.extend(tokens)
                        received_this_attempt = True

                    if isinstance(content, str) and content:
                        content_parts.append(content)
                        received_this_attempt = True
                        if on_event is not None:
                            on_event({
                                "type": "annotation_delta",
                                "chunk_index": chunk["index"],
                                "content": content,
                            })

                break

            except ProviderError as exc:
                if not exc.retryable:
                    raise

                # No output reached Harness, so the original request can be
                # retried unchanged without affecting the browser display.
                if not annotation_tokens and not received_this_attempt:
                    if initial_retry_index >= len(
                        INITIAL_RETRY_DELAYS_SECONDS
                    ):
                        raise

                    delay = INITIAL_RETRY_DELAYS_SECONDS[
                        initial_retry_index
                    ]
                    initial_retry_index += 1
                    time.sleep(delay)
                    continue

                # We have a partial annotation. Check the server first, then
                # continue from its exact raw token prefix in a new request.
                if continuation_retry_index >= len(
                    CONTINUATION_RETRY_DELAYS_SECONDS
                ):
                    raise

                delay = CONTINUATION_RETRY_DELAYS_SECONDS[
                    continuation_retry_index
                ]
                continuation_retry_index += 1
                time.sleep(delay)

                health = self.llm.health()
                if not health.get("ok"):
                    # A failed health check consumes this bounded retry; the
                    # next pass will wait longer before checking again.
                    continue

        annotation_text = "".join(content_parts)

        print(
            "[Annotator] "
            f"chunk={chunk['index']} completed "
            f"generated_tokens={len(annotation_tokens)}"
        )

        if not annotation_text:
            raise RuntimeError(
                "llama.cpp completion did not return "
                "annotation content"
            )

        if not annotation_tokens:
            raise RuntimeError(
                "llama.cpp completion did not return "
                "generated token IDs"
            )

        if on_event is not None:
            on_event({
                "type": "annotation_complete",
                "chunk_index": chunk["index"],
                "generated_token_count": len(annotation_tokens),
            })

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
