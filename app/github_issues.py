"""Create bounded, run-scoped GitHub issue reports."""

from __future__ import annotations

import json
from typing import Any


MAX_TRACE_CHARACTERS = 48_000


def format_issue_body(
    *,
    chat_id: str,
    run_id: str,
    model: str,
    description: str,
    trace_records: list[dict[str, Any]],
) -> str:
    """Build a GitHub-safe issue body, bounded below GitHub's body limit."""
    trace = "\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in trace_records
    )
    truncated = len(trace) > MAX_TRACE_CHARACTERS
    if truncated:
        trace = trace[-MAX_TRACE_CHARACTERS:]

    description = description.strip() or "No additional description provided."
    truncation_note = (
        "\n\n_Trace was truncated to the final 48,000 characters._"
        if truncated else ""
    )
    return (
        "## Harness report\n\n"
        f"{description}\n\n"
        "## Run metadata\n\n"
        f"- Chat: `{chat_id}`\n"
        f"- Run: `{run_id}`\n"
        f"- Model: `{model}`\n\n"
        "## Raw llama.cpp trace\n\n"
        "```jsonl\n"
        f"{trace}\n"
        "```"
        f"{truncation_note}\n"
    )
