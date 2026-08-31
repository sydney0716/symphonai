"""Per-registry record of successful file reads for stale-write checks."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path


MAX_LEDGER_ENTRIES = 100
MAX_LEDGER_BYTES = 25_000_000
DEFAULT_READ_OFFSET = 1
DEFAULT_READ_LIMIT = 2_000


@dataclass(frozen=True)
class ReadRecord:
    mtime_ns: int
    full: bool
    content: str | None
    offset: int = DEFAULT_READ_OFFSET
    limit: int = DEFAULT_READ_LIMIT
    partial_view: bool = False
    # These preserve read_file's output exactly when a range is served from cache.
    more_follows: bool = False
    last_line_seen: int = 0


class ReadLedger:
    """Track concurrency-safe reads under one lock and untuned reference bounds."""

    def __init__(self) -> None:
        self._records: OrderedDict[Path, ReadRecord] = OrderedDict()
        self._lock = threading.Lock()

    def _enforce_bounds(self) -> None:
        retained_characters = sum(
            len(record.content)
            for record in self._records.values()
            if record.content is not None
        )
        for resolved, record in list(self._records.items()):
            if retained_characters <= MAX_LEDGER_BYTES:
                break
            if record.content is None:
                continue
            retained_characters -= len(record.content)
            self._records[resolved] = replace(record, content=None)

        while len(self._records) > MAX_LEDGER_ENTRIES:
            self._records.popitem(last=False)

    def record(
        self,
        resolved: Path,
        *,
        full: bool,
        content: str | None,
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
        partial_view: bool = False,
        more_follows: bool = False,
        last_line_seen: int = 0,
    ) -> None:
        with self._lock:
            self._records[resolved] = ReadRecord(
                mtime_ns=resolved.stat().st_mtime_ns,
                full=full,
                content=None if partial_view else content,
                offset=offset,
                limit=limit,
                partial_view=partial_view,
                more_follows=more_follows,
                last_line_seen=last_line_seen,
            )
            self._records.move_to_end(resolved)
            self._enforce_bounds()

    def cached_record(
        self,
        resolved: Path,
        *,
        offset: int,
        limit: int,
    ) -> ReadRecord | None:
        """Return an unchanged matching range while refreshing its LRU position."""
        with self._lock:
            record = self._records.get(resolved)
            if (
                record is None
                or record.partial_view
                or record.content is None
                or record.offset != offset
                or record.limit != limit
                or resolved.stat().st_mtime_ns != record.mtime_ns
            ):
                return None
            self._records.move_to_end(resolved)
            return record

    def check(self, resolved: Path) -> str | None:
        with self._lock:
            record = self._records.get(resolved)
            if record is None:
                return "file has not been read yet; read it with read_file before editing it"
            if record.partial_view:
                return (
                    "only a processed view of this file has been read; "
                    "read it with read_file before editing it"
                )

            current_mtime_ns = resolved.stat().st_mtime_ns
            if current_mtime_ns <= record.mtime_ns:
                self._records.move_to_end(resolved)
                self._enforce_bounds()
                return None
            if (
                record.full
                and record.content is not None
                and resolved.read_text(encoding="utf-8") == record.content
            ):
                self._records[resolved] = replace(record, mtime_ns=current_mtime_ns)
                self._records.move_to_end(resolved)
                self._enforce_bounds()
                return None
            return "file has changed since it was read; read it again before editing it"
