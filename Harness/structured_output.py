"""Shared constrained-output contracts for every Harness generation stage."""

from __future__ import annotations

import json
from typing import Any


def single_string_schema(field: str) -> dict:
    """Return the strict JSON-schema contract used by one pipeline stage."""
    return {
        "type": "object",
        "properties": {field: {"type": "string"}},
        "required": [field],
        "additionalProperties": False,
    }


def chunked_string_schema() -> dict:
    """Schema for one independently valid piece of a streamed stage."""
    return {
        "type": "object",
        "properties": {
            "chunk": {"type": "string"},
            "done": {"type": "boolean"},
        },
        "required": ["chunk", "done"],
        "additionalProperties": False,
    }


def structured_output_instruction(field: str) -> str:
    return (
        f'Return exactly one JSON object with one key, "{field}". '
        "Its string value must contain only the requested content."
    )


def chunked_output_instruction() -> str:
    """Tell the model how to finish a stage across valid JSON envelopes."""
    return (
        'Return exactly one JSON object with keys "chunk" and "done". '
        'Put only the next, non-repeated portion of the requested content in '
        '"chunk". Set "done" to true only when the requested content is '
        'complete; otherwise set it to false.'
    )


class JSONFieldStreamDecoder:
    """Incrementally extract one JSON string field while retaining raw JSON."""

    def __init__(self, field: str) -> None:
        self.field = field
        self.raw = ""
        self.position = 0
        self.started = False
        self.finished = False
        self.escape = ""

    def feed(self, fragment: str) -> str:
        self.raw += fragment
        if not self.started:
            key_index = self.raw.find(json.dumps(self.field))
            if key_index < 0:
                return ""
            colon_index = self.raw.find(":", key_index)
            if colon_index < 0:
                return ""
            quote_index = self.raw.find('"', colon_index + 1)
            if quote_index < 0:
                return ""
            self.started = True
            self.position = quote_index + 1

        output: list[str] = []
        while self.position < len(self.raw) and not self.finished:
            char = self.raw[self.position]
            self.position += 1
            if self.escape:
                self.escape += char
                if self.escape == "\\u":
                    continue
                if self.escape.startswith("\\u"):
                    if len(self.escape) < 6:
                        continue
                    output.append(chr(int(self.escape[2:], 16)))
                else:
                    output.append({
                        '\\\"': '"', '\\\\': '\\', '\\/': '/', '\\b': '\b',
                        '\\f': '\f', '\\n': '\n', '\\r': '\r', '\\t': '\t',
                    }.get(self.escape, self.escape[-1]))
                self.escape = ""
            elif char == "\\":
                self.escape = "\\"
            elif char == '"':
                self.finished = True
            else:
                output.append(char)
        return "".join(output)

    def result(self) -> str:
        try:
            value = json.loads(self.raw)
        except json.JSONDecodeError as exc:
            raise ValueError("llama.cpp returned incomplete constrained JSON") from exc
        extracted = value.get(self.field) if isinstance(value, dict) else None
        if not isinstance(extracted, str):
            raise ValueError(
                f"llama.cpp constrained JSON did not contain a string {self.field!r}"
            )
        return extracted


class JSONChunkStreamDecoder:
    """Decode one ``{\"chunk\": ..., \"done\": ...}`` envelope."""

    def __init__(self) -> None:
        self._chunk_decoder = JSONFieldStreamDecoder("chunk")

    @property
    def raw(self) -> str:
        return self._chunk_decoder.raw

    def feed(self, fragment: str) -> str:
        return self._chunk_decoder.feed(fragment)

    def result(self) -> tuple[str, bool]:
        try:
            value = json.loads(self.raw)
        except json.JSONDecodeError as exc:
            raise ValueError("llama.cpp returned incomplete constrained JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("llama.cpp constrained JSON was not an object")
        chunk = value.get("chunk")
        done = value.get("done")
        if not isinstance(chunk, str) or not isinstance(done, bool):
            raise ValueError(
                'llama.cpp constrained JSON must contain string "chunk" '
                'and boolean "done" fields'
            )
        return chunk, done


def classify_terminal_condition(
    events: list[dict[str, Any]],
    *,
    error: BaseException | None = None,
    json_valid: bool = False,
) -> str:
    """Normalize native completion endings into Harness-level outcomes."""
    if error is not None and getattr(error, "retryable", False):
        return "transport_drop"

    final = events[-1] if events else {}
    reason = " ".join(
        str(final.get(key, ""))
        for key in ("stop_type", "stop_reason", "reason")
    ).lower()
    if "context" in reason or "ctx" in reason:
        return "context_limit"
    if final.get("truncated") or any(
        marker in reason
        for marker in ("token", "length", "limit", "max")
    ):
        return "token_limit"
    if error is not None:
        return "invalid_json"
    return "complete" if json_valid else "invalid_json"


def increased_token_budget(current: int, baseline: int) -> int:
    """Increase one retry budget without allowing unbounded envelope growth."""
    return min(max(current + 16, current * 2), max(baseline * 4, baseline + 16))


class PrefixStripper:
    """Hide an expected generated prefix while preserving live streaming."""

    def __init__(self, prefix: str | None) -> None:
        self.prefix = prefix or ""
        self._buffer = ""
        self.matched = not self.prefix

    def feed(self, content: str) -> str:
        if self.matched:
            return content
        self._buffer += content
        if self.prefix.startswith(self._buffer):
            return ""
        if self._buffer.startswith(self.prefix):
            self.matched = True
            remainder = self._buffer[len(self.prefix):]
            self._buffer = ""
            return remainder

        # Preserve output if the model declines to follow the prefix cue.
        self.matched = True
        remainder = self._buffer
        self._buffer = ""
        return remainder

    def strip(self, content: str) -> str:
        return content.removeprefix(self.prefix)
