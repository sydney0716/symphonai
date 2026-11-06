"""Private filesystem writes, JSONL records and UTC timestamps."""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
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


def utc_iso(ts: float | None = None) -> str:
    """Format a UTC timestamp for new persisted records and wire replies."""
    value = datetime.now(timezone.utc) if ts is None else datetime.fromtimestamp(ts, timezone.utc)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TranscriptError(RuntimeError):
    """A transcript cannot be serialized, persisted, or safely read."""


def read_records(path: Path) -> tuple[list[dict], int]:
    """Read JSONL objects, dropping only a malformed final line."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TranscriptError(f"cannot read transcript {path}: {exc}") from exc
    lines = raw.splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("record is not a JSON object")
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            if index == len(lines) - 1:
                return records, len(line)
            raise TranscriptError(f"malformed transcript record at line {index + 1}: {exc}") from exc
        records.append(value)
    return records, 0


def append_record(output, line: bytes, *, torn_bytes: int = 0, fsync: bool = False) -> None:
    """Append one encoded JSONL line, repairing a torn tail; roll back on failure."""
    output.seek(0, os.SEEK_END)
    if torn_bytes:
        output.truncate(output.tell() - torn_bytes)
    output.seek(0, os.SEEK_END)
    previous = output.tell()
    separator = b""
    if previous:
        output.seek(-1, os.SEEK_END)
        if output.read(1) != b"\n":
            separator = b"\n"
    try:
        output.write(separator + line)
        output.flush()
        if fsync:
            os.fsync(output.fileno())
    except Exception:
        output.seek(previous)
        output.truncate()
        raise
