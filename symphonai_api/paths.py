"""Locate the user-level SymphonAI directory."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(path: Path, data: str | bytes, *, mode: int = 0o600, fsync: bool = True) -> None:
    """Replace a file from a private same-directory temporary, cleaning up on failure."""
    path = Path(path)
    encoded = data.encode("utf-8") if isinstance(data, str) else data
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        output = os.fdopen(descriptor, "wb")
        descriptor = None
        with output:
            os.fchmod(output.fileno(), mode)
            output.write(encoded)
            output.flush()
            if fsync:
                os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


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
