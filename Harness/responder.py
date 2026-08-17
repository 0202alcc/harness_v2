from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from typing import Any, TypedDict

from LLManager import LLManager, ProviderError
from storage import ChatStorage, utc_now
from .markers import resolve_markers
from .structured_output import (
    JSONChunkStreamDecoder,
    classify_terminal_condition,
    chunked_output_instruction,
    chunked_string_schema,
    increased_token_budget,
)


GEMMA_END_OF_TURN = "<turn|>"
RESPONSE_SCHEMA = chunked_string_schema()
PROTOCOL_RETRY_DELAYS_SECONDS = (0.25, 1.0)


class ResponseResult(TypedDict):
    text: str
    token_ids: list[int]


class Responder:
    """Generate the user-facing answer after the thought-process pass."""

    def __init__(self, *, llm: LLManager, store: ChatStorage, model: str,
                 user_id: str, chat_id: str, instruction: str,
                 n_predict: int = 512, temperature: float = 0.4,
                 markers: dict[str, str] | None = None,
                 completion_options: dict[str, Any] | None = None,
                 max_protocol_chunks: int = 32):
        self.llm = llm
        self.store = store
        self.model = model
        self.user_id = user_id
        self.chat_id = chat_id
        self.instruction = instruction
        self.n_predict = n_predict
        self.temperature = temperature
        self.markers = resolve_markers(markers)
        self.completion_options = dict(completion_options or {})
        self.max_protocol_chunks = max_protocol_chunks

    def generate(self, *, message: str, thought_process: str,
                 system_prompt: str | None, conversation_history: str | None,
                 run_id: str,
                 turn_id: str,
                 on_event: Callable[[dict[str, Any]], None] | None = None) -> ResponseResult:
        user_parts = []
        if conversation_history:
            user_parts.append(conversation_history)
        user_parts.extend([
            f"{self.markers['user_message']}{message}",
            (
                "Private reasoning notes follow. They may already resemble an "
                "answer, but they are only source material and are never shown "
                "to the user. Use them silently:\n"
                f"{self.markers['thought_process']}{thought_process}"
            ),
            (
                "Write the complete, standalone answer to the original user "
                "message. Do not merely summarize, conclude, continue, or refer "
                "to the private reasoning. Include the explanation, examples, "
                "and level of detail the user requested, even if those details "
                "already appear in the notes."
            ),
            self.instruction,
            chunked_output_instruction(),
        ])
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.rstrip()})
        messages.append({"role": "user", "content": "\n\n".join(user_parts)})

        # Gemma emits an EOG token when asked to answer from a raw prompt.
        # Its native chat template supplies the required assistant-turn prefix.
        formatted_prompt = self.llm.apply_chat_template(messages, model=self.model)
        prompt_tokens = self.llm.tokenize(
            formatted_prompt, model=self.model,
            add_special=False, parse_special=False,
        )
        generated_text = ""
        transport_tokens: list[int] = []
        if on_event is not None:
            on_event({"type": "response_start"})
        for protocol_chunk_index in range(self.max_protocol_chunks):
            request_prompt = prompt_tokens
            if generated_text:
                continuation = (
                    "\n\n[Reply written so far]\n"
                    f"{generated_text}\n\n"
                    "Continue the reply without repeating any existing text. "
                    + chunked_output_instruction()
                )
                continuation_messages = list(messages)
                continuation_messages[-1] = {
                    "role": "user",
                    "content": user_parts[0] if len(user_parts) == 1 else "\n\n".join(user_parts) + continuation,
                }
                continuation_prompt = self.llm.apply_chat_template(continuation_messages, model=self.model)
                request_prompt = self.llm.tokenize(continuation_prompt, model=self.model, add_special=False, parse_special=False)

            request = {
                "stream": True, "prompt_token_count": len(request_prompt),
                "prompt_token_sha256": hashlib.sha256(json.dumps(request_prompt, separators=(",", ":")).encode()).hexdigest(),
                "n_predict": self.n_predict, "cache_prompt": True,
                "return_tokens": True, "temperature": self.temperature,
                "stop": [GEMMA_END_OF_TURN], "json_schema": RESPONSE_SCHEMA,
                "includes_annotation_instruction": False, "uses_chat_template": True,
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
                        return_tokens=True, temperature=self.temperature,
                        return_progress=True, stop=[GEMMA_END_OF_TURN],
                        json_schema=RESPONSE_SCHEMA, **getattr(self, "completion_options", {}),
                    ):
                        events.append(event)
                        content, event_tokens = event.get("content", ""), event.get("tokens", [])
                        if isinstance(content, str) and content:
                            raw_content_parts.append(content)
                            decoded_content = decoder.feed(content)
                            if decoded_content and on_event is not None:
                                on_event({"type": "response_delta", "content": decoded_content})
                        if isinstance(event_tokens, list):
                            attempt_tokens.extend(event_tokens)
                    piece, done = decoder.result()
                    if not done and not piece:
                        raise RuntimeError("llama.cpp returned an empty unfinished response chunk")
                    response = {**(events[-1] if events else {}), "content": "".join(raw_content_parts), "tokens": attempt_tokens, "decoded_response": piece}
                    request["n_predict"] = envelope_n_predict
                    self._trace(run_id, turn_id, request, response, started_at, started_clock, "success", None, "complete")
                    generated_text += piece
                    transport_tokens.extend(attempt_tokens)
                    if done:
                        break
                    break
                except Exception as exc:
                    terminal_condition = classify_terminal_condition(events, error=exc)
                    response = {**(events[-1] if events else {}), "content": "".join(raw_content_parts), "tokens": attempt_tokens}
                    request["n_predict"] = envelope_n_predict
                    self._trace(run_id, turn_id, request, response, started_at, started_clock, "error", {"type": type(exc).__name__, "message": str(exc), "retryable": getattr(exc, "retryable", False)}, terminal_condition)
                    if terminal_condition == "context_limit":
                        raise RuntimeError("response hit the model context limit") from exc
                    if delay is None or (isinstance(exc, ProviderError) and not exc.retryable):
                        raise RuntimeError(str(exc)) from exc
                    if terminal_condition in {"token_limit", "invalid_json"}:
                        envelope_n_predict = increased_token_budget(envelope_n_predict, self.n_predict)
                    if on_event is not None:
                        on_event({"type": "response_replace", "content": generated_text})
                    time.sleep(delay)
            else:
                raise RuntimeError("llama.cpp could not complete a response protocol chunk")
            if done:
                break
        else:
            raise RuntimeError("response exceeded the constrained protocol chunk limit")

        text = generated_text
        if not text.strip() or not transport_tokens:
            error = {"type": "InvalidCompletionResponse", "message": "llama.cpp did not return response content and tokens"}
            raise RuntimeError(error["message"])
        return {"text": text, "token_ids": transport_tokens}

    def _trace(self, run_id, turn_id, request, response, started_at, started_clock, status, error, terminal_condition=None):
        timings = response.get("timings") if response else None
        self.store.append_llama_io(
            chat_id=self.chat_id, user_id=self.user_id, turn_id=turn_id,
            run_id=run_id, node="generate_final_response", provider="llama.cpp",
            model=self.model, operation="completion", endpoint="/completion",
            request=request,
            response={
                "generated_token_count": len(response.get("tokens", [])) if response else 0,
                "generated_token_ids": response.get("tokens", []) if response else [],
                "generated_content": response.get("content") if response else "",
                "decoded_response": response.get("decoded_response") if response else None,
                "tokens_cached": timings.get("cache_n") if isinstance(timings, dict) else None,
                "tokens_evaluated": timings.get("prompt_n") if isinstance(timings, dict) else None,
                "slot_id": response.get("id_slot") if response else None,
                "timings": timings,
                "terminal_condition": terminal_condition,
                "final_event": response,
            },
            started_at=started_at, finished_at=utc_now(),
            duration_ms=(time.perf_counter() - started_clock) * 1000,
            status=status, error=error,
        )
