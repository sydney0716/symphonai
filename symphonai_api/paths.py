"""Locate the user-level SymphonAI directory."""

from __future__ import annotations

import os
from pathlib import Path


def symphonai_home(home: Path | None = None) -> Path:
    if home is not None:
        return Path(home) / ".symphonai"
    override = os.environ.get("SYMPHONAI_HOME", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".symphonai"
