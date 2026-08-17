"""Protocol configuration for feedback-capable transformer backends.

The recurrent hidden state belongs inside the model server.  The Harness only
negotiates whether a backend has a trained full-bandwidth model and, when it
does, asks the server to enable its per-token feedback path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FullBandwidthFeedback:
    """Opt in to the server-side Full-bandwidth Transformer protocol."""

    enabled: bool = False

    def completion_options(self, llm: Any, model: str) -> dict[str, Any]:
        """Return the request extension accepted by a compatible backend.

        The opaque hidden state is intentionally never materialized in Python:
        it must remain next to the model's KV cache for every decoding step.
        """

        if not self.enabled:
            return {}

        if not llm.supports_full_bandwidth(model):
            raise RuntimeError(
                f"Model {model!r} does not advertise full-bandwidth feedback. "
                "Use a server and model trained with the feedback architecture, "
                "or run without --full-bandwidth-feedback."
            )

        return {
            "full_bandwidth_feedback": {
                "enabled": True,
                "protocol_version": 1,
            }
        }
