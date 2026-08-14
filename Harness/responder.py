from __future__ import annotations

import hashlib
import json
import time
from typing import TypedDict

from LLManager import LLManager
from storage import ChatStorage, utc_now


USER_MESSAGE_MARKER = "\n\n[User message]\n"
THOUGHT_PROCESS_MARKER = "\n\n[Complete thought process]\n"
RESPONSE_MARKER = "\n\n[Assistant response]\n"


class ResponseResult(TypedDict):
    text: str
    token_ids: list[int]


class Responder:
    """Generate the user-facing answer after the thought-process pass."""

    def __init__(self, *, llm: LLManager, store: ChatStorage, model: str,
                 user_id: str, chat_id: str, instruction: str,
                 n_predict: int = 512, temperature: float = 0.4):
        self.llm = llm
        self.store = store
        self.model = model
        self.user_id = user_id
        self.chat_id = chat_id
        self.instruction = instruction
        self.n_predict = n_predict
        self.temperature = temperature

    def generate(self, *, message: str, thought_process: str,
                 system_prompt: str | None, run_id: str,
                 turn_id: str) -> ResponseResult:
        parts = []
        if system_prompt:
            parts.append(system_prompt.rstrip())
        parts.extend([
            f"{USER_MESSAGE_MARKER}{message}",
            f"{THOUGHT_PROCESS_MARKER}{thought_process}",
            RESPONSE_MARKER,
            self.instruction,
        ])
        prompt_tokens = self.llm.tokenize(
            "\n\n".join(parts), model=self.model,
            add_special=True, parse_special=False,
        )
        request = {
            "stream": False,
            "prompt_token_count": len(prompt_tokens),
            "prompt_token_sha256": hashlib.sha256(
                json.dumps(prompt_tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            "n_predict": self.n_predict,
            "cache_prompt": True,
            "return_tokens": True,
            "temperature": self.temperature,
            "includes_annotation_instruction": False,
        }
        started_at = utc_now()
        started_clock = time.perf_counter()
        try:
            response = self.llm.complete(
                prompt=prompt_tokens, model=self.model,
                n_predict=self.n_predict, cache_prompt=True,
                return_tokens=True, temperature=self.temperature,
            )
        except Exception as exc:
            self._trace(run_id, turn_id, request, None, started_at, started_clock,
                        "error", {"type": type(exc).__name__, "message": str(exc)})
            raise

        text = response.get("content")
        tokens = response.get("tokens")
        if not isinstance(text, str) or not isinstance(tokens, list):
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
