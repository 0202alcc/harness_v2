from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from typing import Any, TypedDict

from LLManager import LLManager
from storage import ChatStorage, utc_now

from .markers import resolve_markers
from .state import Annotation
from .structured_output import (
    JSONFieldStreamDecoder,
    PrefixStripper,
    single_string_schema,
    structured_output_instruction,
)


THOUGHT_PROCESS_SCHEMA = single_string_schema("thought_process")


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
            structured_output_instruction("thought_process"),
        ])
        prompt = "\n\n".join(part for part in prompt_parts if part)
        prompt_tokens = self.llm.tokenize(
            prompt,
            model=self.model,
            add_special=True,
            parse_special=False,
        )

        started_at = utc_now()
        started_clock = time.perf_counter()
        request = {
            "stream": True,
            "prompt_token_count": len(prompt_tokens),
            "prompt_token_sha256": hashlib.sha256(
                json.dumps(prompt_tokens, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "n_predict": self.n_predict,
            "cache_prompt": True,
            "return_tokens": True,
            "temperature": self.temperature,
            "stop": [],
            "json_schema": THOUGHT_PROCESS_SCHEMA,
            "includes_annotation_instruction": False,
        }

        events: list[dict[str, Any]] = []
        raw_content_parts: list[str] = []
        decoder = JSONFieldStreamDecoder("thought_process")
        prefix_stripper = PrefixStripper(self.output_prefix)
        token_ids: list[int] = []
        if on_event is not None:
            on_event({"type": "thought_process_start"})
        try:
            for event in self.llm.stream_complete(
                prompt=prompt_tokens,
                model=self.model,
                n_predict=self.n_predict,
                cache_prompt=True,
                return_tokens=True,
                return_progress=True,
                temperature=self.temperature,
                json_schema=THOUGHT_PROCESS_SCHEMA,
                retry_before_first_token=True,
            ):
                events.append(event)
                content = event.get("content", "")
                tokens = event.get("tokens", [])
                if isinstance(content, str) and content:
                    raw_content_parts.append(content)
                    decoded_content = prefix_stripper.feed(decoder.feed(content))
                    if decoded_content and on_event is not None:
                        on_event({"type": "thought_process_delta", "content": decoded_content})
                if isinstance(tokens, list):
                    token_ids.extend(tokens)
        except Exception as exc:
            self._trace(
                run_id, turn_id, request, None, started_at, started_clock,
                "error", {"type": type(exc).__name__, "message": str(exc)},
            )
            raise

        raw_content = "".join(raw_content_parts)
        response = {**(events[-1] if events else {}), "content": raw_content, "tokens": token_ids}
        try:
            generated_text = decoder.result()
        except ValueError as exc:
            self._trace(
                run_id, turn_id, request, response, started_at, started_clock,
                "error", {"type": type(exc).__name__, "message": str(exc)},
            )
            raise RuntimeError(str(exc)) from exc
        text = prefix_stripper.strip(generated_text)
        response["decoded_thought_process"] = generated_text
        response["stripped_thought_process"] = text
        response["output_prefix_matched"] = (
            not self.output_prefix or generated_text.startswith(self.output_prefix)
        )
        if not text or not token_ids:
            error = {
                "type": "InvalidCompletionResponse",
                "message": "llama.cpp did not return thought-process content and tokens",
            }
            self._trace(run_id, turn_id, request, response, started_at, started_clock, "error", error)
            raise RuntimeError(error["message"])

        self._trace(run_id, turn_id, request, response, started_at, started_clock, "success", None)
        return {"text": text, "token_ids": token_ids}

    def _trace(self, run_id, turn_id, request, response, started_at, started_clock, status, error) -> None:
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
                "final_event": response,
            },
            started_at=started_at,
            finished_at=utc_now(),
            duration_ms=(time.perf_counter() - started_clock) * 1000,
            status=status,
            error=error,
        )
