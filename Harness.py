import logging
from functools import lru_cache
from typing import TypedDict


class Chunk(TypedDict):
    index: int
    token_start: int
    token_end: int
    token_count: int
    token_ids: list[int]
    text: str


CHUNK_SIZE = 512

MODEL_TO_TOKENIZER = {
    "qwen2": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-4": "google/gemma-4-E4B-it",
    "gemma4": "google/gemma-4-E4B-it",
    "e4b": "google/gemma-4-E4B-it",
    "gemma-3": "google/gemma-3-4b-it",
    "gemma3": "google/gemma-3-4b-it",
    "gemma-2": "google/gemma-2-9b-it",
    "gemma2": "google/gemma-2-9b-it",
    "gemma": "google/gemma-2-9b-it",
    "llama-3": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-3": "microsoft/Phi-3-medium-128k-instruct",
    "phi3": "microsoft/Phi-3-medium-128k-instruct",
}


_lru_tokenizer_cache = lru_cache(maxsize=8)


@_lru_tokenizer_cache
def _load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except Exception as exc:
        logging.warning(
            "Failed to load tokenizer for %s (%s); falling back to mapping",
            tokenizer_name, exc,
        )
        raise


def resolve_tokenizer_name(model: str) -> str:
    model_lower = model.lower()
    for key, repo in MODEL_TO_TOKENIZER.items():
        if key in model_lower:
            return repo
    return model


class Harness:
    def __init__(self,
                 model: str,
                 user_id: str,
                 *, chat_id: str = None, chat_json: dict = None, log_file: str = None):

        self.model = model
        self.user_id = user_id
        self.chat_id = chat_id
        self.chat_json = chat_json
        self.log_file = log_file

        self._validate_session_consistency()

        self.tokenizer_name = resolve_tokenizer_name(model)

    def _validate_session_consistency(self):
        expected_ids = []

        if self.chat_id:
            expected_ids.append(self.chat_id)

        if self.chat_json:
            json_id = self.chat_json.get("chat_id")
            if json_id:
                expected_ids.append(json_id)
            elif self.chat_id:
                raise ValueError(
                    "chat_json provided but missing 'chat_id' while chat_id was specified"
                )

        if self.log_file:
            import os
            log_id = os.path.splitext(os.path.basename(self.log_file))[0]
            expected_ids.append(log_id)

        if expected_ids:
            first_id = expected_ids[0]
            for cid in expected_ids[1:]:
                if cid != first_id:
                    raise ValueError(
                        "Session mismatch: chat_id, chat_json, and log_file must "
                        "all refer to the same session. Found mismatch between "
                        f"{first_id} and {cid}"
                    )

    def _get_tokenizer(self):
        try:
            tokenizer = _load_tokenizer(self.tokenizer_name)
        except Exception:
            if self.tokenizer_name != self.model:
                logging.info(
                    "Falling back to model name %s for tokenizer resolution",
                    self.model,
                )
                tokenizer = _load_tokenizer(self.model)
            else:
                raise

        self._verify_tokenizer(tokenizer)
        return tokenizer

    def _verify_tokenizer(self, tokenizer) -> None:
        probe = "The quick brown fox jumps over the lazy dog."
        ids = tokenizer.encode(probe, add_special_tokens=False)
        if not ids:
            raise RuntimeError(
                f"Tokenizer {self.tokenizer_name!r} produced no tokens for probe text"
            )
        round_trip = tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if probe not in round_trip and round_trip not in probe:
            raise RuntimeError(
                f"Tokenizer {self.tokenizer_name!r} failed round-trip verification "
                f"for model {self.model!r}"
            )
        logging.info(
            "Tokenizer %s verified (probe=%d tokens, vocab=%d)",
            self.tokenizer_name,
            len(ids),
            getattr(tokenizer, "vocab_size", -1),
        )

    def chunk_message(self, text: str, chunk_size: int = CHUNK_SIZE) -> dict:
        tokenizer = self._get_tokenizer()

        token_ids = tokenizer.encode(text, add_special_tokens=False)

        chunks: list[Chunk] = []
        for start in range(0, len(token_ids), chunk_size):
            end = min(start + chunk_size, len(token_ids))
            chunk_token_ids = token_ids[start:end]
            chunk_text = tokenizer.decode(
                chunk_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            chunks.append({
                "index": len(chunks),
                "token_start": start,
                "token_end": end,
                "token_count": len(chunk_token_ids),
                "token_ids": chunk_token_ids,
                "text": chunk_text,
            })

        return {
            "input_text": text,
            "tokenizer": self.tokenizer_name,
            "total_tokens": len(token_ids),
            "chunks": chunks,
        }


     # The harness is the state machine
