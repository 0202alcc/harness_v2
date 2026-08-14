from __future__ import annotations

import hashlib
import json
import time
from typing import TypedDict

from LLManager import LLManager
from storage import ChatStorage, utc_now

from .state import Annotation


ANNOTATIONS_MARKER = "\n\n[Accumulated annotations]\n"
THOUGHT_PROCESS_MARKER = "\n\n[Complete thought process]\n"


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

    def generate(
        self,
        *,
        annotations: list[Annotation],
        system_prompt: str | None,
        run_id: str,
        turn_id: str,
    ) -> ThoughtProcessResult:
        annotation_text = "\n".join(
            f"Chunk {annotation['chunk_index']}: {annotation['text']}"
            for annotation in annotations
        )
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt.rstrip())
        # The thought-process instruction must be the final prompt prefix.
        # If it precedes the annotations, the model treats it as another item
        # to analyse rather than as the cue to begin its own continuation.
        prompt_parts.extend([
            f"{ANNOTATIONS_MARKER}{annotation_text}",
            THOUGHT_PROCESS_MARKER,
            self.instruction,
        ])
        prompt = "\n\n".join(prompt_parts)
        prompt_tokens = self.llm.tokenize(
            prompt,
            model=self.model,
            add_special=True,
            parse_special=False,
        )

        started_at = utc_now()
        started_clock = time.perf_counter()
        request = {
            "stream": False,
            "prompt_token_count": len(prompt_tokens),
            "prompt_token_sha256": hashlib.sha256(
                json.dumps(prompt_tokens, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "n_predict": self.n_predict,
            "cache_prompt": True,
            "return_tokens": True,
            "temperature": self.temperature,
            "stop": [],
            "includes_annotation_instruction": False,
        }

        try:
            response = self.llm.complete(
                prompt=prompt_tokens,
                model=self.model,
                n_predict=self.n_predict,
                cache_prompt=True,
                return_tokens=True,
                temperature=self.temperature,
            )
        except Exception as exc:
            self._trace(
                run_id, turn_id, request, None, started_at, started_clock,
                "error", {"type": type(exc).__name__, "message": str(exc)},
            )
            raise

        text = response.get("content")
        token_ids = response.get("tokens")
        if not isinstance(text, str) or not isinstance(token_ids, list):
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
