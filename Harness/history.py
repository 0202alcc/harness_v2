from __future__ import annotations


HISTORY_MARKER = "[Conversation history]"


def format_conversation_history(
    messages: list[dict],
    *,
    current_turn_id: str,
) -> str | None:
    """Format all completed turns before the message being processed."""
    entries = []
    for message in messages:
        if message.get("turn_id") == current_turn_id:
            continue

        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        entries.append(f"{role.capitalize()}: {content}")

    if not entries:
        return None

    return f"{HISTORY_MARKER}\n" + "\n\n".join(entries)
