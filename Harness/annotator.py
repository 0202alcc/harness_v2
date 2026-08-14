# Harness/annotator.py

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from typing import Any, TypedDict
import time

from LLManager import LLManager, ProviderError
from storage import ChatStorage, utc_now

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
        store: ChatStorage,
        model: str,
        user_id: str,
        chat_id: str,
        instruction: str,
        n_predict: int = 64,
        temperature: float = 0.4,
    ):
        if not instruction:
            raise ValueError(
                "annotation instruction cannot be empty"
            )

        self.llm = llm
        self.store = store
        self.model = model
        self.user_id = user_id
        self.chat_id = chat_id
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
        *,
        system_prompt: str | None = None,
    ) -> list[int]:
        """
        Create the initial token sequence for one annotation pass.

        Since subsequent /completion calls use raw token IDs, the
        initial sequence explicitly receives any model-required
        special beginning token here.
        """

        initial_instruction = self.instruction
        if system_prompt:
            initial_instruction = (
                f"{system_prompt.rstrip()}\n\n"
                f"{self.instruction}"
            )

        return self.llm.tokenize(
            initial_instruction,
            model=self.model,
            add_special=True,
            parse_special=False,
        )

    def annotate(
        self,
        *,
        thinking_token_ids: list[int],
        chunk: Chunk,
        run_id: str,
        turn_id: str,
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
                "chunk_text": chunk["text"],
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
            attempt_events: list[dict[str, Any]] = []
            attempt_content_parts: list[str] = []
            attempt_tokens: list[int] = []
            started_at = utc_now()
            started_clock = time.perf_counter()
            attempt_kind = (
                "initial"
                if not annotation_tokens
                and initial_retry_index == 0
                else (
                    "initial_retry"
                    if not annotation_tokens
                    else "continuation"
                )
            )

            try:
                for event in self.llm.stream_complete(
                    prompt=request_prompt,
                    model=self.model,
                    n_predict=remaining_tokens,
                    cache_prompt=True,
                    return_tokens=True,
                    return_progress=True,
                    temperature=self.temperature,
                    stop=[
                        SOURCE_MARKER,
                    ],
                ):
                    attempt_events.append(event)
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
                        attempt_tokens.extend(tokens)
                        received_this_attempt = True

                    if isinstance(content, str) and content:
                        content_parts.append(content)
                        attempt_content_parts.append(content)
                        received_this_attempt = True
                        if on_event is not None:
                            on_event({
                                "type": "annotation_delta",
                                "chunk_index": chunk["index"],
                                "content": content,
                            })

                self._trace_completion(
                    run_id=run_id,
                    turn_id=turn_id,
                    chunk_index=chunk["index"],
                    prompt_tokens=request_prompt,
                    n_predict=remaining_tokens,
                    attempt_kind=attempt_kind,
                    initial_retry_index=initial_retry_index,
                    continuation_retry_index=continuation_retry_index,
                    started_at=started_at,
                    started_clock=started_clock,
                    events=attempt_events,
                    generated_tokens=attempt_tokens,
                    generated_content="".join(attempt_content_parts),
                    status="success",
                    error=None,
                )
                break

            except ProviderError as exc:
                self._trace_completion(
                    run_id=run_id,
                    turn_id=turn_id,
                    chunk_index=chunk["index"],
                    prompt_tokens=request_prompt,
                    n_predict=remaining_tokens,
                    attempt_kind=attempt_kind,
                    initial_retry_index=initial_retry_index,
                    continuation_retry_index=continuation_retry_index,
                    started_at=started_at,
                    started_clock=started_clock,
                    events=attempt_events,
                    generated_tokens=attempt_tokens,
                    generated_content="".join(attempt_content_parts),
                    status="error",
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "retryable": exc.retryable,
                    },
                )
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

            except Exception as exc:
                self._trace_completion(
                    run_id=run_id,
                    turn_id=turn_id,
                    chunk_index=chunk["index"],
                    prompt_tokens=request_prompt,
                    n_predict=remaining_tokens,
                    attempt_kind=attempt_kind,
                    initial_retry_index=initial_retry_index,
                    continuation_retry_index=continuation_retry_index,
                    started_at=started_at,
                    started_clock=started_clock,
                    events=attempt_events,
                    generated_tokens=attempt_tokens,
                    generated_content="".join(attempt_content_parts),
                    status="error",
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                raise

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

    def _trace_completion(
        self,
        *,
        run_id: str,
        turn_id: str,
        chunk_index: int,
        prompt_tokens: list[int],
        n_predict: int,
        attempt_kind: str,
        initial_retry_index: int,
        continuation_retry_index: int,
        started_at: str,
        started_clock: float,
        events: list[dict[str, Any]],
        generated_tokens: list[int],
        generated_content: str,
        status: str,
        error: dict[str, Any] | None,
    ) -> None:
        """Persist one native llama.cpp completion attempt as JSONL."""

        def last_value(*keys: str) -> Any:
            for event in reversed(events):
                for key in keys:
                    if event.get(key) is not None:
                        return event[key]
            return None

        prompt_fingerprint = hashlib.sha256(
            json.dumps(prompt_tokens, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        prompt_progress = last_value("prompt_progress")
        timings = last_value("timings")
        llama_tokens_cached = last_value("tokens_cached")
        llama_tokens_evaluated = last_value("tokens_evaluated")
        tokens_cached = None
        tokens_evaluated = None

        # The top-level tokens_cached value is the slot's post-generation
        # cache size, not the reused prefix. Prefer the actual prompt-work
        # counters exposed in timings or stream progress.
        if isinstance(timings, dict):
            tokens_cached = timings.get("cache_n")
            tokens_evaluated = timings.get("prompt_n")

        if isinstance(prompt_progress, dict):
            if tokens_cached is None:
                tokens_cached = prompt_progress.get("cache")
            if tokens_evaluated is None:
                processed = prompt_progress.get("processed")
                cached = prompt_progress.get("cache")
                if isinstance(processed, int) and isinstance(cached, int):
                    tokens_evaluated = processed - cached

        response = {
            "stream_event_count": len(events),
            "generated_token_count": len(generated_tokens),
            "generated_token_ids": generated_tokens,
            "generated_content": generated_content,
            "tokens_cached": tokens_cached,
            "tokens_evaluated": tokens_evaluated,
            "llama_tokens_cached": llama_tokens_cached,
            "llama_tokens_evaluated": llama_tokens_evaluated,
            "prompt_progress": prompt_progress,
            "slot_id": last_value("slot_id", "id_slot", "slot"),
            "timings": timings,
            "stop_type": last_value("stop_type"),
            "truncated": last_value("truncated"),
            "final_event": events[-1] if events else None,
        }

        self.store.append_llama_io(
            chat_id=self.chat_id,
            user_id=self.user_id,
            turn_id=turn_id,
            run_id=run_id,
            node="annotate_chunk",
            provider="llama.cpp",
            model=self.model,
            operation="completion",
            endpoint="/completion",
            request={
                "stream": True,
                "prompt_token_count": len(prompt_tokens),
                "prompt_token_sha256": prompt_fingerprint,
                "n_predict": n_predict,
                "cache_prompt": True,
                "return_tokens": True,
                "return_progress": True,
                "temperature": self.temperature,
                "stop": [SOURCE_MARKER],
                "attempt": {
                    "kind": attempt_kind,
                    "initial_retry_index": initial_retry_index,
                    "continuation_retry_index": continuation_retry_index,
                },
            },
            response=response,
            started_at=started_at,
            finished_at=utc_now(),
            duration_ms=(time.perf_counter() - started_clock) * 1000,
            chunk_index=chunk_index,
            status=status,
            error=error,
        )
