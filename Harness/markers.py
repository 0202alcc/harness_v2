from __future__ import annotations

from typing import Mapping


DEFAULT_MARKERS = {
    "source_chunk": "\n\n[Source chunk]\n",
    "annotation": "\n\n[Annotation]\n",
    "accumulated_annotations": "\n\n[Accumulated annotations]\n",
    "thought_process": "\n\n[Complete thought process]\n",
    "user_message": "\n\n[User message]\n",
    "assistant_response": "\n\n[Assistant response]\n",
}


def resolve_markers(markers: Mapping[str, str] | None = None) -> dict[str, str]:
    """Merge configured labels with defaults and reject empty marker values."""
    resolved = dict(DEFAULT_MARKERS)
    if markers:
        resolved.update(markers)
    invalid = [key for key, value in resolved.items() if not isinstance(value, str) or not value]
    if invalid:
        raise ValueError(f"marker values must be non-empty strings: {', '.join(invalid)}")
    return resolved
