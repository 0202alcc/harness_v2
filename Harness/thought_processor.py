from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from typing import Any, TypedDict

from LLManager import LLManager, ProviderError
from storage import ChatStorage, utc_now

from .markers import resolve_markers
from .state import Annotation
from .structured_output import (
    JSONChunkStreamDecoder,
    PrefixStripper,
    classify_terminal_condition,
    chunked_output_instruction,
    chunked_string_schema,
    increased_token_budget,
)


THOUGHT_PROCESS_SCHEMA = chunked_string_schema()
PROTOCOL_RETRY_DELAYS_SECONDS = (0.25, 1.0)


class ThoughtProcessResult(TypedDict):
    text: str
    token_ids: list[int]


class ThoughtProcessor:
    """Generate one synthesis pass from completed chunk annotations."""

    def __init__(
        self,
        *,
        llm: LLManager,
        store: ChatStorage,
        model: str,
        user_id: str,
        chat_id: str,
        instruction: str,
        n_predict: int = 256,
        temperature: float = 0.4,
        markers: dict[str, str] | None = None,
        output_prefix: str | None = None,
        completion_options: dict[str, Any] | None = None,
        max_protocol_chunks: int = 32,
    ):
        if not instruction:
            raise ValueError("thought-process instruction cannot be empty")

        self.llm = llm
        self.store = store
        self.model = model
        self.user_id = user_id
        self.chat_id = chat_id
        self.instruction = instruction
        self.n_predict = n_predict
        self.temperature = temperature
        self.markers = resolve_markers(markers)
        self.output_prefix = output_prefix
        self.completion_options = dict(completion_options or {})
        self.max_protocol_chunks = max_protocol_chunks

    def generate(
        self,
        *,
        message: str,
        annotations: list[Annotation],
        system_prompt: str | None,
        conversation_history: str | None,
        run_id: str,
        turn_id: str,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> ThoughtProcessResult:
        annotation_text = "\n".join(
            f"Chunk {annotation['chunk_index']}: {annotation['text']}"
            for annotation in annotations
        )
        prompt_parts: list[str] = []
        if system_prompt:
            prompt_parts.append(system_prompt.rstrip())
        if conversation_history:
            prompt_parts.append(conversation_history)
        prompt_parts.extend([
            f"{self.markers['user_message']}{message}",
            f"{self.markers['accumulated_annotations']}{annotation_text}",
            self.markers["thought_process"],
            self.instruction,
            (
                "The thought_process string must begin exactly with: "
                f"{json.dumps(self.output_prefix)}. Continue the reasoning immediately after it."
                if self.output_prefix else ""
            ),
            chunked_output_instruction(),
        ])
        prompt = "\n\n".join(part for part in prompt_parts if part)
        prompt_tokens = self.llm.tokenize(
            prompt,
            model=self.model,
            add_special=True,
            parse_special=False,
        )

        generated_text = ""
        transport_token_ids: list[int] = []
        prefix_stripper = PrefixStripper(self.output_prefix)
        if on_event is not None:
            on_event({"type": "thought_process_start"})

        for protocol_chunk_index in range(self.max_protocol_chunks):
            request_prompt = prompt_tokens
            if generated_text:
                continuation = (
                    "\n\n[Thought process written so far]\n"
                    f"{generated_text}\n\n"
                    "Continue it without repeating any existing text. "
                    + chunked_output_instruction()
                )
                request_prompt = self.llm.tokenize(
                    prompt + continuation, model=self.model,
                    add_special=True, parse_special=False,
                )

            request = {
                "stream": True,
                "prompt_token_count": len(request_prompt),
                "prompt_token_sha256": hashlib.sha256(json.dumps(request_prompt, separators=(",", ":")).encode("utf-8")).hexdigest(),
                "n_predict": self.n_predict, "cache_prompt": True,
                "return_tokens": True, "temperature": self.temperature,
                "stop": [], "json_schema": THOUGHT_PROCESS_SCHEMA,
                "includes_annotation_instruction": False,
                "protocol_chunk_index": protocol_chunk_index,
                **getattr(self, "completion_options", {}),
            }
            envelope_n_predict = self.n_predict
            for retry_index, delay in enumerate((*PROTOCOL_RETRY_DELAYS_SECONDS, None)):
                events: list[dict[str, Any]] = []
                raw_content_parts: list[str] = []
                attempt_tokens: list[int] = []
                decoder = JSONChunkStreamDecoder()
                started_at, started_clock = utc_now(), time.perf_counter()
                try:
                    for event in self.llm.stream_complete(
                        prompt=request_prompt, model=self.model,
                        n_predict=envelope_n_predict, cache_prompt=True,
                        return_tokens=True, return_progress=True,
                        temperature=self.temperature, json_schema=THOUGHT_PROCESS_SCHEMA,
                        **getattr(self, "completion_options", {}),
                    ):
                        events.append(event)
                        content, tokens = event.get("content", ""), event.get("tokens", [])
                        if isinstance(content, str) and content:
                            raw_content_parts.append(content)
                            visible = prefix_stripper.feed(decoder.feed(content))
                            if visible and on_event is not None:
                                on_event({"type": "thought_process_delta", "content": visible})
                        if isinstance(tokens, list):
                            attempt_tokens.extend(tokens)
                    piece, done = decoder.result()
                    if not piece:
                        raise RuntimeError("llama.cpp returned an empty thought-process chunk")
                    response = {**(events[-1] if events else {}), "content": "".join(raw_content_parts), "tokens": attempt_tokens}
                    response["decoded_thought_process"] = piece
                    request["n_predict"] = envelope_n_predict
                    self._trace(run_id, turn_id, request, response, started_at, started_clock, "success", None, "complete")
                    generated_text += piece
                    transport_token_ids.extend(attempt_tokens)
                    if done:
                        break
                    break
                except Exception as exc:
                    terminal_condition = classify_terminal_condition(events, error=exc)
                    response = {**(events[-1] if events else {}), "content": "".join(raw_content_parts), "tokens": attempt_tokens}
                    request["n_predict"] = envelope_n_predict
                    self._trace(run_id, turn_id, request, response, started_at, started_clock, "error", {"type": type(exc).__name__, "message": str(exc), "retryable": getattr(exc, "retryable", False)}, terminal_condition)
                    if terminal_condition == "context_limit":
                        raise RuntimeError("thought process hit the model context limit") from exc
                    if delay is None or (isinstance(exc, ProviderError) and not exc.retryable):
                        raise RuntimeError(str(exc)) from exc
                    if terminal_condition in {"token_limit", "invalid_json"}:
                        envelope_n_predict = increased_token_budget(envelope_n_predict, self.n_predict)
                    prefix_stripper = PrefixStripper(self.output_prefix)
                    if on_event is not None:
                        on_event({"type": "thought_process_replace", "content": prefix_stripper.strip(generated_text)})
                    time.sleep(delay)
            else:
                raise RuntimeError("llama.cpp could not complete a thought-process protocol chunk")
            if done:
                break
        else:
            raise RuntimeError("thought process exceeded the constrained protocol chunk limit")

        text = prefix_stripper.strip(generated_text)
        if not text or not transport_token_ids:
            error = {
                "type": "InvalidCompletionResponse",
                "message": "llama.cpp did not return thought-process content and tokens",
            }
            raise RuntimeError(error["message"])
        return {"text": text, "token_ids": transport_token_ids}

    def _trace(self, run_id, turn_id, request, response, started_at, started_clock, status, error, terminal_condition=None) -> None:
        timings = response.get("timings") if response else None
        self.store.append_llama_io(
            chat_id=self.chat_id,
            user_id=self.user_id,
            turn_id=turn_id,
            run_id=run_id,
            node="generate_thought_process",
            provider="llama.cpp",
            model=self.model,
            operation="completion",
            endpoint="/completion",
            request=request,
            response={
                "generated_token_count": len(response.get("tokens", [])) if response else 0,
                "generated_token_ids": response.get("tokens", []) if response else [],
                "generated_content": response.get("content") if response else "",
                "decoded_thought_process": response.get("decoded_thought_process") if response else None,
                "stripped_thought_process": response.get("stripped_thought_process") if response else None,
                "output_prefix_matched": response.get("output_prefix_matched") if response else None,
                "tokens_cached": timings.get("cache_n") if isinstance(timings, dict) else None,
                "tokens_evaluated": timings.get("prompt_n") if isinstance(timings, dict) else None,
                "slot_id": response.get("id_slot") if response else None,
                "timings": timings,
                "terminal_condition": terminal_condition,
                "final_event": response,
            },
            started_at=started_at,
            finished_at=utc_now(),
            duration_ms=(time.perf_counter() - started_clock) * 1000,
            status=status,
            error=error,
        )
