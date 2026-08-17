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
from .markers import resolve_markers
from .state import Annotation
from .structured_output import (
    JSONChunkStreamDecoder,
    classify_terminal_condition,
    chunked_output_instruction,
    chunked_string_schema,
    increased_token_budget,
)


# Plain-text delimiters for V1.
#
# These become part of the actual incremental token sequence:
#
# P
# + SOURCE_MARKER + C0 + ANNOTATION_MARKER + A0
# + SOURCE_MARKER + C1 + ANNOTATION_MARKER + A1
# + ...
#
ANNOTATION_SCHEMA = chunked_string_schema()

# There are three initial attempts: the original request plus these retries.
INITIAL_RETRY_DELAYS_SECONDS = (0.25, 1.0)
# A continuation is a new completion request, so give the model server a
# moment to release the interrupted stream before submitting it.
CONTINUATION_RETRY_DELAYS_SECONDS = (0.25, 1.0)


def _normalise_annotation_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _is_annotation_loop(piece: str, *, source_text: str, prior_annotation: str) -> bool:
    """Detect a model echoing the source or a prior protocol envelope."""
    candidate = _normalise_annotation_text(piece)
    return bool(candidate) and candidate in {
        _normalise_annotation_text(source_text),
        _normalise_annotation_text(prior_annotation),
    }


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
        markers: dict[str, str] | None = None,
        completion_options: dict[str, Any] | None = None,
        max_protocol_chunks: int = 32,
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
        self.markers = resolve_markers(markers)

        self.n_predict = n_predict
        self.temperature = temperature
        self.completion_options = dict(completion_options or {})
        self.max_protocol_chunks = max_protocol_chunks

        # These are static, so tokenize them once rather than
        # making another /tokenize request for every chunk.
        self._source_marker_tokens = self.llm.tokenize(
            self.markers["source_chunk"],
            model=self.model,
            add_special=False,
            parse_special=False,
        )

        self._annotation_marker_tokens = self.llm.tokenize(
            self.markers["annotation"],
            model=self.model,
            add_special=False,
            parse_special=False,
        )

    def initialize(
        self,
        *,
        system_prompt: str | None = None,
        conversation_history: str | None = None,
    ) -> list[int]:
        """
        Create the initial token sequence for one annotation pass.

        Since subsequent /completion calls use raw token IDs, the
        initial sequence explicitly receives any model-required
        special beginning token here.
        """

        initial_parts = []
        if system_prompt:
            initial_parts.append(system_prompt.rstrip())
        if conversation_history:
            initial_parts.append(conversation_history)
        initial_parts.append(self.instruction)
        initial_parts.append(chunked_output_instruction())
        initial_instruction = "\n\n".join(initial_parts)

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

        annotation_text = ""
        annotation_tokens: list[int] = []
        continuation_tokens = self.llm.tokenize(
            "\n\nContinue the annotation above without repeating it. "
            + chunked_output_instruction(),
            model=self.model,
            add_special=False,
            parse_special=False,
        )

        for protocol_chunk_index in range(self.max_protocol_chunks):
            prior_text_tokens = self.llm.tokenize(
                annotation_text,
                model=self.model,
                add_special=False,
                parse_special=False,
            ) if annotation_text else []
            request_prompt = prompt_tokens + prior_text_tokens
            if annotation_text:
                request_prompt += continuation_tokens

            envelope_n_predict = self.n_predict
            for retry_index, delay in enumerate((*INITIAL_RETRY_DELAYS_SECONDS, None)):
                decoder = JSONChunkStreamDecoder()
                attempt_events: list[dict[str, Any]] = []
                attempt_content_parts: list[str] = []
                attempt_tokens: list[int] = []
                emitted_text = ""
                started_at = utc_now()
                started_clock = time.perf_counter()
                try:
                    for event in self.llm.stream_complete(
                        prompt=request_prompt,
                        model=self.model,
                        n_predict=envelope_n_predict,
                        cache_prompt=True,
                        return_tokens=True,
                        return_progress=True,
                        temperature=self.temperature,
                        json_schema=ANNOTATION_SCHEMA,
                        **getattr(self, "completion_options", {}),
                    ):
                        attempt_events.append(event)
                        content = event.get("content", "")
                        tokens = event.get("tokens", [])
                        if isinstance(tokens, list):
                            attempt_tokens.extend(tokens)
                        if isinstance(content, str) and content:
                            attempt_content_parts.append(content)
                            decoded_content = decoder.feed(content)
                            emitted_text += decoded_content
                            if decoded_content and on_event is not None:
                                on_event({"type": "annotation_delta", "chunk_index": chunk["index"], "content": decoded_content})

                    piece, done = decoder.result()
                    if not piece:
                        raise RuntimeError("llama.cpp returned an empty annotation chunk")
                    if _is_annotation_loop(
                        piece,
                        source_text=chunk["text"],
                        prior_annotation=annotation_text,
                    ):
                        fallback = (
                            "The source is the user's request; use the original "
                            "message directly when preparing the response."
                        )
                        self._trace_completion(
                            run_id=run_id, turn_id=turn_id, chunk_index=chunk["index"],
                            prompt_tokens=request_prompt, n_predict=envelope_n_predict,
                            attempt_kind="annotation_echo_fallback",
                            initial_retry_index=retry_index if protocol_chunk_index == 0 else 0,
                            continuation_retry_index=retry_index if protocol_chunk_index else 0,
                            started_at=started_at, started_clock=started_clock,
                            events=attempt_events, generated_tokens=attempt_tokens,
                            generated_content="".join(attempt_content_parts), status="error",
                            error={
                                "type": "AnnotationEcho",
                                "message": "Model repeated source or prior annotation text",
                                "retryable": False,
                            },
                            protocol_chunk_index=protocol_chunk_index,
                            terminal_condition="annotation_echo",
                        )
                        annotation_text = fallback
                        annotation_tokens = self.llm.tokenize(
                            fallback,
                            model=self.model,
                            add_special=False,
                            parse_special=False,
                        )
                        if on_event is not None:
                            on_event({
                                "type": "annotation_replace",
                                "chunk_index": chunk["index"],
                                "content": fallback,
                            })
                        done = True
                        break
                    self._trace_completion(
                        run_id=run_id, turn_id=turn_id, chunk_index=chunk["index"],
                        prompt_tokens=request_prompt, n_predict=envelope_n_predict,
                        attempt_kind=(
                            "initial_retry" if protocol_chunk_index == 0 and retry_index
                            else "initial" if protocol_chunk_index == 0
                            else "protocol_continuation"
                        ),
                        initial_retry_index=retry_index if protocol_chunk_index == 0 else 0,
                        continuation_retry_index=retry_index if protocol_chunk_index else 0,
                        started_at=started_at, started_clock=started_clock,
                        events=attempt_events, generated_tokens=attempt_tokens,
                        generated_content="".join(attempt_content_parts), status="success", error=None,
                        protocol_chunk_index=protocol_chunk_index,
                        terminal_condition="complete",
                    )
                    annotation_text += piece
                    annotation_tokens.extend(attempt_tokens)
                    if done:
                        break
                    break
                except Exception as exc:
                    terminal_condition = classify_terminal_condition(
                        attempt_events, error=exc,
                    )
                    self._trace_completion(
                        run_id=run_id, turn_id=turn_id, chunk_index=chunk["index"],
                        prompt_tokens=request_prompt, n_predict=envelope_n_predict,
                        attempt_kind="protocol_retry",
                        initial_retry_index=retry_index if protocol_chunk_index == 0 else 0,
                        continuation_retry_index=retry_index if protocol_chunk_index else 0,
                        started_at=started_at, started_clock=started_clock,
                        events=attempt_events, generated_tokens=attempt_tokens,
                        generated_content="".join(attempt_content_parts), status="error",
                        error={"type": type(exc).__name__, "message": str(exc), "retryable": getattr(exc, "retryable", False)},
                        protocol_chunk_index=protocol_chunk_index,
                        terminal_condition=terminal_condition,
                    )
                    if terminal_condition == "context_limit":
                        raise RuntimeError("annotation hit the model context limit") from exc
                    if not isinstance(exc, (ProviderError, ValueError)) and not isinstance(exc, RuntimeError):
                        raise RuntimeError(str(exc)) from exc
                    if isinstance(exc, ProviderError) and not exc.retryable:
                        raise RuntimeError(str(exc)) from exc
                    if delay is None:
                        raise RuntimeError(str(exc)) from exc
                    if terminal_condition in {"token_limit", "invalid_json"}:
                        envelope_n_predict = increased_token_budget(
                            envelope_n_predict, self.n_predict,
                        )
                    if emitted_text and on_event is not None:
                        on_event({"type": "annotation_replace", "chunk_index": chunk["index"], "content": annotation_text})
                    time.sleep(delay)
            else:
                raise RuntimeError("llama.cpp could not complete an annotation protocol chunk")

            if done:
                break
        else:
            raise RuntimeError("annotation exceeded the constrained protocol chunk limit")

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

        # JSON is only the transport format. Re-encode the decoded text so
        # future chunks accumulate compact marker text, not JSON syntax.
        compact_annotation_tokens = self.llm.tokenize(
            annotation_text,
            model=self.model,
            add_special=False,
            parse_special=False,
        )

        return {
            "annotation": annotation,

            "thinking_token_ids": (
                prompt_tokens
                + compact_annotation_tokens
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
        protocol_chunk_index: int = 0,
        terminal_condition: str | None = None,
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
            "terminal_condition": terminal_condition,
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
                "stop": [],
                "json_schema": ANNOTATION_SCHEMA,
                **getattr(self, "completion_options", {}),
                "attempt": {
                    "kind": attempt_kind,
                    "initial_retry_index": initial_retry_index,
                    "continuation_retry_index": continuation_retry_index,
                },
                "protocol_chunk_index": protocol_chunk_index,
            },
            response=response,
            started_at=started_at,
            finished_at=utc_now(),
            duration_ms=(time.perf_counter() - started_clock) * 1000,
            chunk_index=chunk_index,
            status=status,
            error=error,
        )
