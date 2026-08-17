"""Build identity shown by the web gateway and health endpoint."""

from __future__ import annotations

import os
from pathlib import Path


def application_version() -> str:
    """Return the deploy-time version, falling back to the source VERSION file."""
    configured = os.environ.get("HARNESS_VERSION", "").strip()
    if configured:
        return configured
    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    return version_file.read_text(encoding="utf-8").strip()
