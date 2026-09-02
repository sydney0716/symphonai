"""Append-only transcripts and mutable metadata for one SymphonAI session."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from symphonai_api.identity import SCHEMA_VERSION, new_id

if TYPE_CHECKING:
    from symphonai_api.models import Message


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
        "conversation_rewritten",
        "run_finished",
        "run_failed",
    }
)


class TranscriptError(RuntimeError):
    """A transcript cannot be serialized, persisted, or safely read."""


class SessionError(TranscriptError):
    """A persisted run cannot be loaded, resumed, or forked safely."""


@dataclass(frozen=True)
class LoadedRun:
    run_id: str
    run_count: int
    agent_id: str
    parent_run_id: str | None
    messages: list[Message]
    record_ids: list[str]
    stopped_reason: str | None
    dropped_bytes: int
    meta: dict


class RunState(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    CRASHED = "crashed"


class TurnState(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    EMPTY = "empty"


@dataclass(frozen=True)
class RunDiagnosis:
    state: RunState
    run_id: str
    run_count: int
    stopped_reason: str | None
    turns: tuple[tuple[str, TurnState], ...]
    unanswered_tool_call_ids: tuple[str, ...]
    compactions: int
    dropped_bytes: int


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def default_sessions_root() -> Path:
    """The user-level directory runs are written under.

    One function so phase 17's host process changes one line rather than a
    convention. Honours SYMPHONAI_SESSIONS_DIR when it is set and non-empty.
    """

    override = os.environ.get("SYMPHONAI_SESSIONS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".symphonai" / "sessions"


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

    def __init__(self, root: Path, run_id: str, *, create: bool = True) -> None:
        self._run_id = run_id
        self._root = Path(root)
        self._directory = self._root / run_id
        self._writers: dict[Path, TranscriptWriter] = {}
        self._writers_lock = threading.Lock()
        self._meta_lock = threading.Lock()
        if create:
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
        if create and not (self._directory / "meta.json").exists():
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

    @classmethod
    def open(cls, root: Path, run_id: str) -> "SessionStore":
        """Open an existing run without creating or rewriting anything."""

        directory = Path(root) / run_id
        if not directory.is_dir():
            raise SessionError(
                f"run {run_id!r} cannot be opened: run directory is missing"
            )
        if not (directory / "meta.json").is_file():
            raise SessionError(
                f"run {run_id!r} cannot be opened: meta.json is missing"
            )
        return cls(root, run_id, create=False)

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def tool_results_directory(self) -> Path:
        """`<run>/tool-results/`, created on first access with mode 0o700."""

        path = self._directory / "tool-results"
        try:
            path.mkdir(exist_ok=True, mode=0o700)
            path.chmod(0o700)
        except OSError as exc:
            raise TranscriptError(
                f"cannot create tool-result directory {path}: {exc}"
            ) from exc
        return path

    @property
    def existing_tool_results_directory(self) -> Path | None:
        """Return the existing result directory without changing anything."""

        path = self._directory / "tool-results"
        return path if path.is_dir() else None

    def set_parent_session(self, parent_session_id: str) -> None:
        """Record which session directory this run descends from."""

        meta = self.read_meta()
        meta["parent_session_id"] = parent_session_id
        meta["updated_at"] = _timestamp()
        self.write_meta(meta)

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


def _transcript_path(store: SessionStore, agent_id: str | None) -> Path:
    if agent_id is None:
        return store.directory / "run.jsonl"
    return store.directory / f"agent-{agent_id}.jsonl"


def _checked_records(
    store: SessionStore, agent_id: str | None
) -> tuple[list[dict], int]:
    if not store.directory.is_dir():
        raise SessionError(
            f"run {store.run_id!r} cannot be loaded: run directory is missing"
        )
    path = _transcript_path(store, agent_id)
    if not path.is_file():
        raise SessionError(
            f"run {store.run_id!r} cannot be loaded: transcript {path.name!r} is missing"
        )
    records, dropped_bytes = read_records(path)
    for record in records:
        version = record.get("schema_version")
        if isinstance(version, int) and version > SCHEMA_VERSION:
            raise SessionError(
                f"run {store.run_id!r} uses schema_version {version}, "
                f"but this build supports {SCHEMA_VERSION}"
            )
    return records, dropped_bytes


def load_run(
    store: SessionStore, *, agent_id: str | None = None
) -> LoadedRun:
    """Rebuild one agent's persisted conversation from its transcript."""

    # Local import avoids a cycle: serialization's public errors live here.
    from symphonai_api.serialization import message_from_json

    records, dropped_bytes = _checked_records(store, agent_id)
    messages: list[Message] = []
    record_ids: list[str] = []
    run_id: str | None = None
    run_count = 0
    loaded_agent_id: str | None = None
    parent_run_id: str | None = None
    stopped_reason: str | None = None
    for record in records:
        record_type = record.get("type")
        if record_type == "run_started":
            run_id = record.get("run_id")
            run_count += 1
            loaded_agent_id = record.get("agent_id")
            parent_run_id = record.get("data", {}).get("parent_run_id")
            stopped_reason = None
        elif record_type == "message":
            messages.append(message_from_json(record["data"]))
            record_ids.append(record["record_id"])
        elif record_type == "conversation_rewritten":
            kept_prefix = record.get("data", {}).get("kept_prefix")
            # `bool` is an `int` in Python, and a negative prefix would slice
            # silently from the end -- both are corruption, not a boundary.
            if (
                not isinstance(kept_prefix, int)
                or isinstance(kept_prefix, bool)
                or not 0 <= kept_prefix <= len(messages)
            ):
                raise SessionError(
                    f"run {run_id or store.run_id!r} has conversation rewrite "
                    f"kept_prefix {kept_prefix!r}, but only {len(messages)} "
                    "messages were loaded"
                )
            messages = messages[:kept_prefix]
            record_ids = record_ids[:kept_prefix]
        elif record_type == "run_finished":
            stopped_reason = record.get("data", {}).get("stopped_reason")
        elif record_type == "run_failed":
            stopped_reason = "failed"
    if not isinstance(run_id, str) or not isinstance(loaded_agent_id, str):
        raise SessionError(
            f"run {store.run_id!r} cannot be loaded: transcript has no run_started record"
        )
    return LoadedRun(
        run_id=run_id,
        run_count=run_count,
        agent_id=loaded_agent_id,
        parent_run_id=parent_run_id,
        messages=messages,
        record_ids=record_ids,
        stopped_reason=stopped_reason,
        dropped_bytes=dropped_bytes,
        meta=store.read_meta(),
    )


def resume_run(
    store: SessionStore, *, agent_id: str | None = None
) -> tuple[list[Message], str]:
    """Return the loaded conversation and the run id it continues."""

    loaded = load_run(store, agent_id=agent_id)
    return loaded.messages, loaded.run_id


def classify_run(loaded: LoadedRun, records: Sequence[dict]) -> RunDiagnosis:
    """Classify the final run segment without mutating its conversation."""

    from symphonai_api.repair import unanswered_tool_call_ids
    from symphonai_api.models import Role
    from symphonai_api.serialization import message_from_json

    starts = [
        index
        for index, record in enumerate(records)
        if record.get("type") == "run_started"
    ]
    if not starts or starts[0] != 0:
        raise SessionError("transcript has records before its first run_started")
    segment = records[starts[-1] :]
    run_id = segment[0].get("run_id")
    if not isinstance(run_id, str):
        raise SessionError("final run_started record has no run id")

    failed = any(record.get("type") == "run_failed" for record in segment)
    finishes = [record for record in segment if record.get("type") == "run_finished"]
    if failed:
        state = RunState.FAILED
        stopped_reason = "failed"
    elif finishes:
        stopped_reason = finishes[-1].get("data", {}).get("stopped_reason")
        state = (
            RunState.CANCELLED
            if stopped_reason == "cancelled"
            else RunState.COMPLETED
        )
    else:
        state = RunState.CRASHED
        stopped_reason = None

    turn_starts = [
        index
        for index, record in enumerate(segment)
        if record.get("type") == "turn_started"
    ]
    turns: list[tuple[str, TurnState]] = []
    for position, start in enumerate(turn_starts):
        end = (
            turn_starts[position + 1]
            if position + 1 < len(turn_starts)
            else len(segment)
        )
        turn_id = segment[start].get("turn_id")
        if not isinstance(turn_id, str):
            raise SessionError("turn_started record has no turn id")
        messages = [
            message_from_json(record["data"])
            for record in segment[start:end]
            if record.get("type") == "message"
        ]
        if not any(message.role == Role.ASSISTANT for message in messages):
            turn_state = TurnState.EMPTY
        elif unanswered_tool_call_ids(messages):
            turn_state = TurnState.PARTIAL
        else:
            turn_state = TurnState.COMPLETED
        turns.append((turn_id, turn_state))

    assert not (
        state == RunState.COMPLETED
        and any(turn_state == TurnState.PARTIAL for _, turn_state in turns)
    ), "a completed run cannot contain a partial turn"
    return RunDiagnosis(
        state=state,
        run_id=run_id,
        run_count=len(starts),
        stopped_reason=stopped_reason,
        turns=tuple(turns),
        unanswered_tool_call_ids=tuple(unanswered_tool_call_ids(loaded.messages)),
        compactions=sum(record.get("type") == "compaction" for record in records),
        dropped_bytes=loaded.dropped_bytes,
    )


def load_run_for_resume(
    store: SessionStore, *, agent_id: str | None = None
) -> tuple[LoadedRun, RunDiagnosis, list[str]]:
    """Load, diagnose, and repair a conversation for provider-safe resume."""

    from symphonai_api.repair import repair_unanswered_tool_calls

    loaded = load_run(store, agent_id=agent_id)
    records, _ = _checked_records(store, agent_id)
    diagnosis = classify_run(loaded, records)
    repaired = repair_unanswered_tool_calls(
        loaded.messages,
        error="the session ended before this tool completed",
        cancelled=False,
    )
    return loaded, diagnosis, repaired


def tool_result_search_path(store: SessionStore) -> tuple[Path, ...]:
    """Return existing result directories for a session and its ancestors."""

    paths: list[Path] = []
    current = store
    seen = {store.run_id}
    own_directory = current.existing_tool_results_directory
    if own_directory is not None:
        paths.append(own_directory)
    for _ in range(64):
        try:
            parent_session_id = current.read_meta().get("parent_session_id")
        except TranscriptError:
            break
        if not isinstance(parent_session_id, str) or parent_session_id in seen:
            break
        seen.add(parent_session_id)
        try:
            current = SessionStore.open(store.directory.parent, parent_session_id)
        except SessionError:
            break
        directory = current.existing_tool_results_directory
        if directory is not None:
            paths.append(directory)
    return tuple(paths)


def fork_run(
    store: SessionStore,
    *,
    through_record_id: str,
    new_store: SessionStore,
    agent_id: str | None = None,
) -> LoadedRun:
    """Copy a consistent message-addressed prefix into a descendant run."""

    from symphonai_api.serialization import message_from_json

    records, _ = _checked_records(store, agent_id)
    target_index = next(
        (
            index
            for index, record in enumerate(records)
            if record.get("record_id") == through_record_id
        ),
        None,
    )
    if target_index is None:
        raise SessionError(
            f"run {store.run_id!r} has no record {through_record_id!r}"
        )
    target = records[target_index]
    if target.get("type") != "message":
        raise SessionError(
            f"run {store.run_id!r} record {through_record_id!r} is not a message"
        )
    prefix = records[: target_index + 1]
    prefix_messages = [
        message_from_json(record["data"])
        for record in prefix
        if record.get("type") == "message"
    ]
    from symphonai_api.repair import unanswered_tool_call_ids

    unanswered = unanswered_tool_call_ids(prefix_messages)
    if unanswered:
        raise SessionError(
            f"run {store.run_id!r} fork through {through_record_id!r} has "
            f"unanswered tool call ids: {', '.join(unanswered)}"
        )

    loaded = load_run(store, agent_id=agent_id)
    new_store.set_parent_session(store.run_id)
    source_started = next(
        record for record in records if record.get("type") == "run_started"
    )
    writer = new_store.writer_for(loaded.agent_id, is_root=agent_id is None)
    source_start_data = source_started.get("data", {})
    writer.append(
        "run_started",
        run_id=new_store.run_id,
        agent_id=loaded.agent_id,
        turn_id=None,
        data={
            "agent_name": source_start_data.get("agent_name"),
            "parent_run_id": loaded.run_id,
            "model": source_start_data.get("model"),
        },
    )
    for record in prefix:
        if record.get("type") == "run_started":
            continue
        writer.append(
            record["type"],
            run_id=new_store.run_id,
            agent_id=loaded.agent_id,
            turn_id=record.get("turn_id"),
            data=record.get("data", {}),
        )
    return load_run(new_store, agent_id=agent_id)
