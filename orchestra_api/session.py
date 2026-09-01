"""Append-only transcripts and mutable metadata for one Orchestra session."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from orchestra_api.identity import SCHEMA_VERSION, new_id


_RECORD_TYPES = frozenset(
    {
        "run_started",
        "turn_started",
        "request",
        "message",
        "tool_started",
        "tool_result",
        "turn_finished",
        "cancellation",
        "compaction",
        "run_finished",
        "run_failed",
    }
)


class TranscriptError(RuntimeError):
    """A transcript cannot be serialized, persisted, or safely read."""


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def default_sessions_root() -> Path:
    """The user-level directory runs are written under.

    One function so phase 17's host process changes one line rather than a
    convention. Honours ORCHESTRA_SESSIONS_DIR when it is set and non-empty.
    """

    override = os.environ.get("ORCHESTRA_SESSIONS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".orchestra" / "sessions"


class TranscriptWriter:
    """Append-only, one open file, one lock, flushed per record."""

    def __init__(
        self,
        path: Path,
        *,
        _on_append: Callable[[dict], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._on_append = _on_append
        self._closed = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.path.parent.chmod(0o700)
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            os.chmod(self.path, 0o600)
            self._file = os.fdopen(descriptor, "a", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise TranscriptError(f"cannot open transcript {self.path}: {exc}") from exc

    def append(
        self,
        record_type: str,
        *,
        run_id: str,
        agent_id: str,
        turn_id: str | None,
        data: dict,
    ) -> str:
        if record_type not in _RECORD_TYPES:
            raise TranscriptError(f"unknown transcript record type: {record_type!r}")
        record_id = new_id("rec")
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_id": record_id,
            "ts": _timestamp(),
            "type": record_type,
            "run_id": run_id,
            "agent_id": agent_id,
            "turn_id": turn_id,
            "data": data,
        }
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        except (TypeError, ValueError) as exc:
            raise TranscriptError(
                f"cannot serialize {record_type!r} transcript record: {exc}"
            ) from exc
        with self._lock:
            if self._closed:
                raise TranscriptError(f"transcript is closed: {self.path}")
            try:
                self._file.write(line)
                self._file.flush()
            except (OSError, UnicodeError) as exc:
                raise TranscriptError(
                    f"cannot append transcript record to {self.path}: {exc}"
                ) from exc
        if self._on_append is not None:
            self._on_append(record)
        return record_id

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._file.close()
            except OSError as exc:
                raise TranscriptError(f"cannot close transcript {self.path}: {exc}") from exc
            finally:
                self._closed = True


class SessionStore:
    """Owns one run's directory: transcripts, sidecar, result store dir."""

    def __init__(self, root: Path, run_id: str) -> None:
        self._run_id = run_id
        self._root = Path(root)
        self._directory = self._root / run_id
        self._writers: dict[Path, TranscriptWriter] = {}
        self._writers_lock = threading.Lock()
        self._meta_lock = threading.Lock()
        try:
            self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._root.chmod(0o700)
            self._directory.mkdir(exist_ok=True, mode=0o700)
            self._directory.chmod(0o700)
            tool_results = self._directory / "tool-results"
            tool_results.mkdir(exist_ok=True, mode=0o700)
            tool_results.chmod(0o700)
        except OSError as exc:
            raise TranscriptError(
                f"cannot create session directory {self._directory}: {exc}"
            ) from exc
        now = _timestamp()
        self.write_meta(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "agent_id": None,
                "created_at": now,
                "updated_at": now,
                "title": None,
                "parent_run_id": None,
                "stopped_reason": None,
            }
        )

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def run_id(self) -> str:
        return self._run_id

    def _record_appended(self, record: dict) -> None:
        if record["type"] not in {"run_started", "run_finished", "run_failed"}:
            return
        meta = self.read_meta()
        if record["type"] == "run_started":
            meta["agent_id"] = record["agent_id"]
            meta["parent_run_id"] = record["data"]["parent_run_id"]
        elif record["type"] == "run_finished":
            meta["stopped_reason"] = record["data"]["stopped_reason"]
        else:
            meta["stopped_reason"] = "failed"
        meta["updated_at"] = _timestamp()
        self.write_meta(meta)

    def writer_for(
        self, agent_id: str, *, is_root: bool = False
    ) -> TranscriptWriter:
        path = self._directory / (
            "run.jsonl" if is_root else f"agent-{agent_id}.jsonl"
        )
        with self._writers_lock:
            writer = self._writers.get(path)
            if writer is None:
                writer = TranscriptWriter(
                    path,
                    _on_append=self._record_appended if is_root else None,
                )
                self._writers[path] = writer
            return writer

    def read_meta(self) -> dict:
        path = self._directory / "meta.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TranscriptError(f"cannot read session metadata {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise TranscriptError(f"session metadata is not an object: {path}")
        return value

    def write_meta(self, meta: dict) -> None:
        path = self._directory / "meta.json"
        try:
            encoded = json.dumps(meta, ensure_ascii=False, separators=(",", ":")) + "\n"
        except (TypeError, ValueError) as exc:
            raise TranscriptError(f"cannot serialize session metadata: {exc}") from exc
        with self._meta_lock:
            temporary_path: Path | None = None
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".meta-", dir=self._directory
                )
                temporary_path = Path(temporary_name)
                os.chmod(temporary_path, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(encoded)
                    handle.flush()
                os.replace(temporary_path, path)
                os.chmod(path, 0o600)
            except (OSError, UnicodeError) as exc:
                raise TranscriptError(f"cannot write session metadata {path}: {exc}") from exc
            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        pass

    def close(self) -> None:
        with self._writers_lock:
            writers = list(self._writers.values())
        for writer in writers:
            writer.close()


def read_records(path: Path) -> tuple[list[dict], int]:
    """Return every complete record and the number of trailing bytes dropped."""

    transcript_path = Path(path)
    try:
        raw = transcript_path.read_bytes()
    except OSError as exc:
        raise TranscriptError(f"cannot read transcript {transcript_path}: {exc}") from exc
    lines = raw.splitlines(keepends=True)
    records: list[dict] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("record is not a JSON object")
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            if index == len(lines) - 1:
                return records, len(line)
            raise TranscriptError(
                f"malformed transcript record at line {index + 1}: {exc}"
            ) from exc
        records.append(value)
    return records, 0
