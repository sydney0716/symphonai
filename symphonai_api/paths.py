"""Locate the user-level SymphonAI directory."""

from __future__ import annotations

import os
from pathlib import Path


def symphonai_home(home: Path | None = None) -> Path:
    if home is not None:
        return Path(home) / ".symphonai"
    override = os.environ.get("SYMPHONAI_HOME", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".symphonai"


def ensure_private_directory(path: Path) -> None:
    """Create missing directory components with owner-only permissions."""
    path = Path(path)
    missing = []
    current = path
    while not current.exists():
        if current.parent == current:
            break
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            if not directory.is_dir():
                raise
        else:
            directory.chmod(0o700)
    if not path.is_dir():
        raise NotADirectoryError(path)
