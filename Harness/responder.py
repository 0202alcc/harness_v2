from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from typing import Any, TypedDict

from LLManager import LLManager
from storage import ChatStorage, utc_now
from .markers import resolve_markers
from .structured_output import (
    JSONFieldStreamDecoder,
    single_string_schema,
    structured_output_instruction,
)


GEMMA_END_OF_TURN = "<turn|>"
RESPONSE_SCHEMA = single_string_schema("response")


class ResponseResult(TypedDict):
    text: str
    token_ids: list[int]


class Responder:
    """Generate the user-facing answer after the thought-process pass."""

    def __init__(self, *, llm: LLManager, store: ChatStorage, model: str,
                 user_id: str, chat_id: str, instruction: str,
                 n_predict: int = 512, temperature: float = 0.4,
                 markers: dict[str, str] | None = None):
        self.llm = llm
        self.store = store
        self.model = model
        self.user_id = user_id
        self.chat_id = chat_id
        self.instruction = instruction
        self.n_predict = n_predict
        self.temperature = temperature
        self.markers = resolve_markers(markers)

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
                "Use this internal reasoning to inform your answer; do not "
                f"repeat it:\n{self.markers['thought_process']}{thought_process}"
            ),
            self.instruction,
            (
                'Return exactly one JSON object with one key, "response". '
                "Its string value must contain only the reply to the user."
            ),
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
        request = {
            "stream": True,
            "prompt_token_count": len(prompt_tokens),
            "prompt_token_sha256": hashlib.sha256(
                json.dumps(prompt_tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            "n_predict": self.n_predict,
            "cache_prompt": True,
            "return_tokens": True,
            "temperature": self.temperature,
            "stop": [GEMMA_END_OF_TURN],
            "json_schema": RESPONSE_SCHEMA,
            "includes_annotation_instruction": False,
            "uses_chat_template": True,
        }
        started_at = utc_now()
        started_clock = time.perf_counter()
        events: list[dict[str, Any]] = []
        raw_content_parts: list[str] = []
        decoder = JSONFieldStreamDecoder("response")
        tokens: list[int] = []
        if on_event is not None:
            on_event({"type": "response_start"})
        try:
            for event in self.llm.stream_complete(
                prompt=prompt_tokens, model=self.model,
                n_predict=self.n_predict, cache_prompt=True,
                return_tokens=True, temperature=self.temperature,
                return_progress=True,
                stop=[GEMMA_END_OF_TURN],
                json_schema=RESPONSE_SCHEMA,
                retry_before_first_token=True,
            ):
                events.append(event)
                content = event.get("content", "")
                event_tokens = event.get("tokens", [])
                if isinstance(content, str) and content:
                    raw_content_parts.append(content)
                    decoded_content = decoder.feed(content)
                    if decoded_content and on_event is not None:
                        on_event({"type": "response_delta", "content": decoded_content})
                if isinstance(event_tokens, list):
                    tokens.extend(event_tokens)
        except Exception as exc:
            self._trace(run_id, turn_id, request, None, started_at, started_clock,
                        "error", {"type": type(exc).__name__, "message": str(exc)})
            raise

        raw_content = "".join(raw_content_parts)
        response = {**(events[-1] if events else {}), "content": raw_content, "tokens": tokens}
        try:
            text = decoder.result()
        except ValueError as exc:
            self._trace(run_id, turn_id, request, response, started_at, started_clock,
                        "error", {"type": type(exc).__name__, "message": str(exc)})
            raise RuntimeError(str(exc)) from exc
        response["decoded_response"] = text
        if not text.strip() or not tokens:
            error = {"type": "InvalidCompletionResponse", "message": "llama.cpp did not return response content and tokens"}
            self._trace(run_id, turn_id, request, response, started_at, started_clock, "error", error)
            raise RuntimeError(error["message"])

        self._trace(run_id, turn_id, request, response, started_at, started_clock, "success", None)
        return {"text": text, "token_ids": tokens}

    def _trace(self, run_id, turn_id, request, response, started_at, started_clock, status, error):
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
                "final_event": response,
            },
            started_at=started_at, finished_at=utc_now(),
            duration_ms=(time.perf_counter() - started_clock) * 1000,
            status=status, error=error,
        )
