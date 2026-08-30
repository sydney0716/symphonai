"""Per-registry record of successful file reads for stale-write checks."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReadRecord:
    mtime_ns: int
    full: bool
    content: str | None


class ReadLedger:
    """Track reads shared by concurrency-safe tools under one lock."""

    def __init__(self) -> None:
        self._records: dict[Path, ReadRecord] = {}
        self._lock = threading.Lock()

    def record(self, resolved: Path, *, full: bool, content: str | None) -> None:
        with self._lock:
            self._records[resolved] = ReadRecord(
                mtime_ns=resolved.stat().st_mtime_ns,
                full=full,
                content=content if full else None,
            )

    def check(self, resolved: Path) -> str | None:
        with self._lock:
            record = self._records.get(resolved)
            if record is None:
                return "file has not been read yet; read it with read_file before editing it"

            current_mtime_ns = resolved.stat().st_mtime_ns
            if current_mtime_ns <= record.mtime_ns:
                return None
            if record.full and resolved.read_text(encoding="utf-8") == record.content:
                self._records[resolved] = ReadRecord(
                    mtime_ns=current_mtime_ns,
                    full=True,
                    content=record.content,
                )
                return None
            return "file has changed since it was read; read it again before editing it"
