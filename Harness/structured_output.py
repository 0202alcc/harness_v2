"""Shared constrained-output contracts for every Harness generation stage."""

from __future__ import annotations

import json


def single_string_schema(field: str) -> dict:
    """Return the strict JSON-schema contract used by one pipeline stage."""
    return {
        "type": "object",
        "properties": {field: {"type": "string"}},
        "required": [field],
        "additionalProperties": False,
    }


def structured_output_instruction(field: str) -> str:
    return (
        f'Return exactly one JSON object with one key, "{field}". '
        "Its string value must contain only the requested content."
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
