"""Repair sessions written by early ROS images without a resolved user ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from storage import ChatNotFoundError, ChatStorage


def migrate_orphaned_sessions(*, log_root: str, user_id: str) -> int:
    """Copy root-level, empty-user sessions into their correct user directory.

    The operation is idempotent: original message IDs are preserved and already
    migrated messages are skipped. Source files are retained as an audit trail.
    """
    store = ChatStorage(log_root)
    migrated = 0
    for state_path in Path(log_root).glob("*/chat_state.json"):
        source = json.loads(state_path.read_text(encoding="utf-8"))
        if source.get("user_id") != "":
            continue
        chat_id = str(source.get("chat_id") or state_path.parent.name)
        try:
            target = store.get_chat(chat_id=chat_id, user_id=user_id)
        except ChatNotFoundError:
            target = store.create_chat(
                chat_id=chat_id,
                user_id=user_id,
                provider=source.get("current_model", {}).get("provider", "llama.cpp"),
                model=source.get("current_model", {}).get("model"),
                system_prompt=source.get("system_prompt"),
            )

        existing_ids = {message.get("message_id") for message in target.get("messages", [])}
        copied = 0
        for message in source.get("messages", []):
            if message.get("message_id") in existing_ids:
                continue
            store.append_message(
                chat_id=chat_id,
                user_id=user_id,
                role=message["role"],
                content=message["content"],
                turn_id=message.get("turn_id"),
                message_id=message.get("message_id"),
                branch_id=message.get("branch_id"),
                parent_message_id=message.get("parent_message_id"),
            )
            copied += 1
        if copied:
            store.append_event(
                chat_id=chat_id,
                user_id=user_id,
                event_type="ros_orphaned_session_migrated",
                data={"copied_message_count": copied},
            )
            migrated += copied
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--user-id", required=True)
    args = parser.parse_args()
    copied = migrate_orphaned_sessions(log_root=args.log_root, user_id=args.user_id)
    if copied:
        print(f"Migrated {copied} message(s) from orphaned ROS sessions.")


if __name__ == "__main__":
    main()
