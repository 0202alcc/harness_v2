# storage.py

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import tempfile
import threading
import uuid

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = "1.0"

CHAT_STATE_FILENAME = "chat_state.json"
EVENTS_FILENAME = "events.jsonl"
LLAMA_IO_FILENAME = "llama_io.jsonl"
LOCK_FILENAME = ".storage.lock"


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------


class StorageError(RuntimeError):
    """Base error for ChatStorage."""


class ChatNotFoundError(StorageError):
    """Requested chat does not exist."""


class ChatAlreadyExistsError(StorageError):
    """Attempted to create a chat that already exists."""


class StorageCorruptionError(StorageError):
    """Stored JSON/JSONL data could not be decoded safely."""


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def utc_now() -> str:
    """
    Return an ISO-8601 UTC timestamp.

    Example:
        2026-08-13T16:43:12.123456+00:00
    """
    return datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()


def local_timezone() -> str:
    return str(
        datetime.datetime.now().astimezone().tzinfo
    )


def make_id() -> str:
    return str(uuid.uuid4())


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------


class ChatStorage:
    """
    File-backed persistence for Harness V1.

    Directory structure:

        <root_path>/
        └── <user_id>/
            └── <chat_id>/
                ├── chat_state.json
                ├── events.jsonl
                ├── llama_io.jsonl
                └── .storage.lock

    Responsibilities:

        chat_state.json
            Materialized current conversation state.

        events.jsonl
            Append-only audit log of persistent state changes.

        llama_io.jsonl
            Append-only trace of llama.cpp requests/responses.
    """

    def __init__(
        self,
        root_path: str = "./.logs/",
    ):
        self.root_path = Path(root_path)

        self.root_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Fallback lock for platforms without fcntl.
        self._thread_lock = threading.RLock()

    # =================================================================
    # Paths
    # =================================================================

    def get_chat_dir(
        self,
        user_id: str,
        chat_id: str,
        *,
        create: bool = False,
    ) -> Path:

        path = (
            self.root_path
            / str(user_id)
            / str(chat_id)
        )

        if create:
            path.mkdir(
                parents=True,
                exist_ok=True,
            )

        return path

    def get_chat_state_path(
        self,
        user_id: str,
        chat_id: str,
    ) -> Path:
        return (
            self.get_chat_dir(user_id, chat_id)
            / CHAT_STATE_FILENAME
        )

    def get_events_path(
        self,
        user_id: str,
        chat_id: str,
    ) -> Path:
        return (
            self.get_chat_dir(user_id, chat_id)
            / EVENTS_FILENAME
        )

    def get_llama_io_path(
        self,
        user_id: str,
        chat_id: str,
    ) -> Path:
        return (
            self.get_chat_dir(user_id, chat_id)
            / LLAMA_IO_FILENAME
        )

    # =================================================================
    # Locking
    # =================================================================

    @contextmanager
    def _chat_lock(
        self,
        user_id: str,
        chat_id: str,
    ) -> Iterator[None]:
        """
        Obtain an exclusive lock for one chat.

        On macOS/Linux this uses fcntl.flock, so separate processes
        cannot mutate the same chat simultaneously.

        The threading lock is a fallback for platforms without fcntl.
        """

        chat_dir = self.get_chat_dir(
            user_id,
            chat_id,
            create=True,
        )

        lock_path = chat_dir / LOCK_FILENAME

        with self._thread_lock:
            lock_file = open(lock_path, "a+")

            try:
                try:
                    import fcntl

                    fcntl.flock(
                        lock_file.fileno(),
                        fcntl.LOCK_EX,
                    )
                except ImportError:
                    # Windows fallback: the in-process RLock above
                    # still provides basic protection.
                    pass

                yield

            finally:
                try:
                    import fcntl

                    fcntl.flock(
                        lock_file.fileno(),
                        fcntl.LOCK_UN,
                    )
                except ImportError:
                    pass

                lock_file.close()

    # =================================================================
    # Low-level file operations
    # =================================================================

    def _read_json_unlocked(
        self,
        path: Path,
    ) -> dict[str, Any]:

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

        except FileNotFoundError:
            raise

        except json.JSONDecodeError as exc:
            raise StorageCorruptionError(
                f"Could not decode JSON file: {path}"
            ) from exc

        if not isinstance(data, dict):
            raise StorageCorruptionError(
                f"Expected JSON object in {path}, "
                f"found {type(data).__name__}"
            )

        return data

    def _atomic_write_json_unlocked(
        self,
        path: Path,
        data: dict[str, Any],
    ) -> None:
        """
        Write JSON atomically.

        Data is written to a temporary file in the same directory,
        flushed to disk, then atomically replaces the previous file.
        """

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:

                temp_path = temp_file.name

                json.dump(
                    data,
                    temp_file,
                    indent=2,
                    ensure_ascii=False,
                )

                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())

            os.replace(
                temp_path,
                path,
            )

        except Exception:
            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            raise

    def _append_jsonl_unlocked(
        self,
        path: Path,
        record: dict[str, Any],
    ) -> None:
        """
        Append exactly one JSON object as one physical line.
        """

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serialized = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(serialized)
            f.write("\n")

            f.flush()
            os.fsync(f.fileno())

    def _next_jsonl_number_unlocked(
        self,
        path: Path,
        field: str,
    ) -> int:
        """
        Determine the next monotonic integer for a JSONL stream.

        V1 implementation scans the file. This is simple and safe.
        If logs become very large, this can later be replaced with
        a counter/index.
        """

        if not path.exists():
            return 0

        last_number = -1

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as f:

                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise StorageCorruptionError(
                            f"Corrupt JSONL record in {path}"
                        ) from exc

                    value = record.get(field)

                    if isinstance(value, int):
                        last_number = max(
                            last_number,
                            value,
                        )

        except OSError as exc:
            raise StorageError(
                f"Could not read {path}"
            ) from exc

        return last_number + 1

    # =================================================================
    # Chat state
    # =================================================================

    def create_chat(
        self,
        chat_id: str,
        user_id: str,
        *,
        model: str | None = None,
        provider: str = "llama.cpp",
        system_prompt: str | None = None,
        language: str = "en",
        timezone: str | None = None,
    ) -> dict[str, Any]:

        chat_id = str(chat_id)
        user_id = str(user_id)

        with self._chat_lock(
            user_id,
            chat_id,
        ):
            state_path = self.get_chat_state_path(
                user_id,
                chat_id,
            )

            if state_path.exists():
                raise ChatAlreadyExistsError(
                    f"Chat {chat_id} already exists "
                    f"for user {user_id}"
                )

            now = utc_now()

            chat = {
                "schema_version": SCHEMA_VERSION,

                "user_id": user_id,
                "chat_id": chat_id,

                "state_version": 0,

                "system_prompt": system_prompt,

                "current_model": {
                    "provider": provider,
                    "model": model,
                },

                "active_branch_id": "main",

                "messages": [],

                "session_metadata": {
                    "language": language,
                    "timezone": (
                        timezone
                        if timezone is not None
                        else local_timezone()
                    ),
                    "created_at": now,
                    "last_updated": now,
                },
            }

            self._atomic_write_json_unlocked(
                state_path,
                chat,
            )

            # Create empty JSONL files.
            self.get_events_path(
                user_id,
                chat_id,
            ).touch(exist_ok=True)

            self.get_llama_io_path(
                user_id,
                chat_id,
            ).touch(exist_ok=True)

            self._append_event_unlocked(
                user_id=user_id,
                chat_id=chat_id,
                event_type="chat_initialized",
                turn_id=None,
                branch_id="main",
                state_version_before=None,
                state_version_after=0,
                changes=[
                    {
                        "op": "initialize",
                        "path": "/",
                        "initial_model": {
                            "provider": provider,
                            "model": model,
                        },
                        "system_prompt_sha256": (
                            sha256_text(system_prompt)
                        ),
                    }
                ],
                timestamp=now,
            )

            logging.info(
                "Created chat %s for user %s",
                chat_id,
                user_id,
            )

            return chat

    def get_chat(
        self,
        chat_id: str,
        user_id: str,
    ) -> dict[str, Any]:

        chat_id = str(chat_id)
        user_id = str(user_id)

        with self._chat_lock(
            user_id,
            chat_id,
        ):
            path = self.get_chat_state_path(
                user_id,
                chat_id,
            )

            if not path.exists():
                raise ChatNotFoundError(
                    f"Chat {chat_id} does not exist "
                    f"for user {user_id}"
                )

            return self._read_json_unlocked(path)

    def save_chat(
        self,
        chat: dict[str, Any],
    ) -> None:
        """
        Atomic low-level state save.

        Prefer dedicated mutation methods such as append_message(),
        update_system_prompt(), and update_model() because they also
        generate appropriate events.
        """

        chat_id = chat.get("chat_id")
        user_id = chat.get("user_id")

        if not chat_id or not user_id:
            raise ValueError(
                "chat must contain user_id and chat_id"
            )

        with self._chat_lock(
            str(user_id),
            str(chat_id),
        ):
            path = self.get_chat_state_path(
                str(user_id),
                str(chat_id),
            )

            self._atomic_write_json_unlocked(
                path,
                chat,
            )

    # =================================================================
    # Message mutations
    # =================================================================

    def append_message(
        self,
        chat_id: str,
        user_id: str,
        *,
        role: str,
        content: str,
        turn_id: str | None = None,
        message_id: str | None = None,
        branch_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> dict[str, Any]:

        chat_id = str(chat_id)
        user_id = str(user_id)

        if role not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError(
                f"Unsupported message role: {role!r}"
            )

        with self._chat_lock(
            user_id,
            chat_id,
        ):
            state_path = self.get_chat_state_path(
                user_id,
                chat_id,
            )

            if not state_path.exists():
                raise ChatNotFoundError(
                    f"Chat {chat_id} does not exist"
                )

            chat = self._read_json_unlocked(
                state_path
            )

            messages = chat.setdefault(
                "messages",
                [],
            )

            if branch_id is None:
                branch_id = chat.get(
                    "active_branch_id",
                    "main",
                )

            # A new user message begins a new turn.
            #
            # Non-user messages inherit the most recent turn unless
            # explicitly supplied.
            if turn_id is None:
                if role == "user":
                    turn_id = make_id()

                elif messages:
                    turn_id = messages[-1].get(
                        "turn_id"
                    )

                if not turn_id:
                    turn_id = make_id()

            if message_id is None:
                message_id = make_id()

            if (
                parent_message_id is None
                and messages
            ):
                # For V1, use the latest message as parent.
                # Explicit branch handling can override this later.
                parent_message_id = messages[-1].get(
                    "message_id"
                )

            now = utc_now()

            sequence_number = len(messages)

            message = {
                "message_id": message_id,
                "sequence_number": sequence_number,
                "turn_id": turn_id,

                "branch_id": branch_id,
                "parent_message_id": (
                    parent_message_id
                ),

                "role": role,
                "content": content,

                "timestamp": now,
            }

            state_before = int(
                chat.get("state_version", 0)
            )

            state_after = state_before + 1

            messages.append(message)

            chat["state_version"] = state_after

            metadata = chat.setdefault(
                "session_metadata",
                {},
            )

            metadata["last_updated"] = now

            self._atomic_write_json_unlocked(
                state_path,
                chat,
            )

            self._append_event_unlocked(
                user_id=user_id,
                chat_id=chat_id,
                event_type="message_added",
                turn_id=turn_id,
                branch_id=branch_id,
                state_version_before=state_before,
                state_version_after=state_after,
                changes=[
                    {
                        "op": "add",
                        "path": "/messages/-",
                        "message_id": message_id,
                        "sequence_number": (
                            sequence_number
                        ),
                        "role": role,
                        "content_sha256": (
                            sha256_text(content)
                        ),
                    }
                ],
                timestamp=now,
            )

            return message

    # =================================================================
    # Other state mutations
    # =================================================================

    def update_system_prompt(
        self,
        chat_id: str,
        user_id: str,
        system_prompt: str | None,
    ) -> dict[str, Any]:

        chat_id = str(chat_id)
        user_id = str(user_id)

        with self._chat_lock(
            user_id,
            chat_id,
        ):
            state_path = self.get_chat_state_path(
                user_id,
                chat_id,
            )

            if not state_path.exists():
                raise ChatNotFoundError(
                    f"Chat {chat_id} does not exist"
                )

            chat = self._read_json_unlocked(
                state_path
            )

            old_prompt = chat.get(
                "system_prompt"
            )

            if old_prompt == system_prompt:
                return chat

            before = int(
                chat.get("state_version", 0)
            )

            after = before + 1
            now = utc_now()

            chat["system_prompt"] = system_prompt
            chat["state_version"] = after

            chat.setdefault(
                "session_metadata",
                {},
            )["last_updated"] = now

            self._atomic_write_json_unlocked(
                state_path,
                chat,
            )

            self._append_event_unlocked(
                user_id=user_id,
                chat_id=chat_id,
                event_type="system_prompt_changed",
                turn_id=None,
                branch_id=chat.get(
                    "active_branch_id",
                    "main",
                ),
                state_version_before=before,
                state_version_after=after,
                changes=[
                    {
                        "op": "replace",
                        "path": "/system_prompt",
                        "old_sha256": (
                            sha256_text(old_prompt)
                        ),
                        "new_sha256": (
                            sha256_text(
                                system_prompt
                            )
                        ),
                    }
                ],
                timestamp=now,
            )

            return chat

    def update_model(
        self,
        chat_id: str,
        user_id: str,
        *,
        provider: str,
        model: str,
    ) -> dict[str, Any]:

        chat_id = str(chat_id)
        user_id = str(user_id)

        with self._chat_lock(
            user_id,
            chat_id,
        ):
            state_path = self.get_chat_state_path(
                user_id,
                chat_id,
            )

            if not state_path.exists():
                raise ChatNotFoundError(
                    f"Chat {chat_id} does not exist"
                )

            chat = self._read_json_unlocked(
                state_path
            )

            old_model = chat.get(
                "current_model"
            )

            new_model = {
                "provider": provider,
                "model": model,
            }

            if old_model == new_model:
                return chat

            before = int(
                chat.get("state_version", 0)
            )

            after = before + 1
            now = utc_now()

            chat["current_model"] = new_model
            chat["state_version"] = after

            chat.setdefault(
                "session_metadata",
                {},
            )["last_updated"] = now

            self._atomic_write_json_unlocked(
                state_path,
                chat,
            )

            self._append_event_unlocked(
                user_id=user_id,
                chat_id=chat_id,
                event_type="model_changed",
                turn_id=None,
                branch_id=chat.get(
                    "active_branch_id",
                    "main",
                ),
                state_version_before=before,
                state_version_after=after,
                changes=[
                    {
                        "op": "replace",
                        "path": "/current_model",
                        "old_value": old_model,
                        "new_value": new_model,
                    }
                ],
                timestamp=now,
            )

            return chat

    # =================================================================
    # Events
    # =================================================================

    def _append_event_unlocked(
        self,
        *,
        user_id: str,
        chat_id: str,
        event_type: str,
        turn_id: str | None,
        branch_id: str,
        state_version_before: int | None,
        state_version_after: int | None,
        changes: list[dict[str, Any]],
        timestamp: str | None = None,
        event_id: str | None = None,
    ) -> dict[str, Any]:

        path = self.get_events_path(
            user_id,
            chat_id,
        )

        event_number = (
            self._next_jsonl_number_unlocked(
                path,
                "event_number",
            )
        )

        event = {
            "schema_version": SCHEMA_VERSION,

            "event_id": event_id or make_id(),
            "event_number": event_number,
            "event_type": event_type,

            "timestamp": (
                timestamp or utc_now()
            ),

            "user_id": str(user_id),
            "chat_id": str(chat_id),
            "branch_id": branch_id,
            "turn_id": turn_id,

            "state_version_before": (
                state_version_before
            ),
            "state_version_after": (
                state_version_after
            ),

            "changes": changes,
        }

        self._append_jsonl_unlocked(
            path,
            event,
        )

        return event

    def append_event(
        self,
        chat_id: str,
        user_id: str,
        *,
        event_type: str,
        data: dict[str, Any] | None = None,
        turn_id: str | None = None,
        branch_id: str = "main",
        state_version_before: int | None = None,
        state_version_after: int | None = None,
    ) -> dict[str, Any]:
        """
        Append a generic persistent-state event.

        For canonical state mutations, prefer append_message(),
        update_system_prompt(), or update_model().

        `data` is wrapped as one custom change payload.
        """

        chat_id = str(chat_id)
        user_id = str(user_id)

        changes = []

        if data is not None:
            changes.append(
                {
                    "op": "custom",
                    "data": data,
                }
            )

        with self._chat_lock(
            user_id,
            chat_id,
        ):
            return self._append_event_unlocked(
                user_id=user_id,
                chat_id=chat_id,
                event_type=event_type,
                turn_id=turn_id,
                branch_id=branch_id,
                state_version_before=(
                    state_version_before
                ),
                state_version_after=(
                    state_version_after
                ),
                changes=changes,
            )

    # =================================================================
    # llama.cpp I/O trace
    # =================================================================

    def append_llama_io(
        self,
        chat_id: str,
        user_id: str,
        *,
        turn_id: str | None,
        run_id: str | None,
        node: str | None,
        provider: str,
        model: str,
        operation: str,
        endpoint: str,
        request: dict[str, Any],
        response: dict[str, Any] | None,
        started_at: str,
        finished_at: str,
        duration_ms: float,
        chunk_index: int | None = None,
        branch_id: str = "main",
        status: str = "success",
        error: dict[str, Any] | None = None,
        io_id: str | None = None,
    ) -> dict[str, Any]:

        chat_id = str(chat_id)
        user_id = str(user_id)

        if status not in {
            "success",
            "error",
        }:
            raise ValueError(
                "status must be 'success' or 'error'"
            )

        with self._chat_lock(
            user_id,
            chat_id,
        ):
            path = self.get_llama_io_path(
                user_id,
                chat_id,
            )

            io_number = (
                self._next_jsonl_number_unlocked(
                    path,
                    "io_number",
                )
            )

            record = {
                "schema_version": SCHEMA_VERSION,

                "io_id": io_id or make_id(),
                "io_number": io_number,

                "user_id": user_id,
                "chat_id": chat_id,
                "branch_id": branch_id,
                "turn_id": turn_id,

                "run_id": run_id,
                "node": node,

                "chunk_index": chunk_index,

                "provider": provider,
                "model": model,

                "operation": operation,
                "endpoint": endpoint,

                "started_at": started_at,
                "finished_at": finished_at,
                "duration_ms": duration_ms,

                "request": request,
                "response": response,

                "status": status,
                "error": error,
            }

            self._append_jsonl_unlocked(
                path,
                record,
            )

            return record