"""Bounded project and user memory persisted independently of run transcripts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid
from typing import Literal

from symphonai_api.identity import SCHEMA_VERSION


MAX_ENTRIES = 32
MAX_ENTRY_CHARS = 500
MAX_TOTAL_CHARS = 8000
MemoryScope = Literal["project", "global"]


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    text: str
    written_at: float
    agent_name: str
    run_id: str
    scope: MemoryScope = "project"
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class MemorySettings:
    enabled: bool = False
    max_entries: int = MAX_ENTRIES

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(self.max_entries) is not int or not 1 <= self.max_entries <= MAX_ENTRIES:
            raise ValueError(f"max_entries must be between 1 and {MAX_ENTRIES}")


class MemoryUnavailable(RuntimeError):
    """A memory operation failed because its backing store was unavailable."""

    def __init__(
        self,
        operation: str,
        path: Path,
        error: OSError | ValueError | str,
        *,
        unreadable: bool = False,
    ) -> None:
        self.path = path
        self.unreadable = unreadable
        if unreadable:
            message = "memory file could not be read; nothing was saved"
        else:
            message = f"memory {operation} failed at {path}: {error}"
        super().__init__(message)


class AgentMemory:
    def __init__(self, root: Path, repo_root: Path | None = None) -> None:
        self._root = Path(root).resolve()
        self._repo_root = Path(repo_root or Path.cwd()).resolve()
        self._project_key = hashlib.sha256(
            str(self._repo_root).encode("utf-8")
        ).hexdigest()
        self._lock = threading.Lock()

    def _path(self, scope: MemoryScope) -> Path:
        if scope == "global":
            return self._root / "global.json"
        if scope == "project":
            return self._root / "projects" / f"{self._project_key}.json"
        raise ValueError("scope must be 'project' or 'global'")

    @staticmethod
    def _decode(path: Path, scope: MemoryScope) -> tuple[MemoryEntry, ...]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ()
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise MemoryUnavailable("read", path, exc, unreadable=True) from exc
        try:
            if not isinstance(payload, list):
                raise ValueError("memory file must contain a list")
            entries: list[MemoryEntry] = []
            for value in payload:
                if (
                    not isinstance(value, dict)
                    or not isinstance(value.get("id"), str)
                    or not value["id"]
                    or not isinstance(value.get("text"), str)
                    or not isinstance(value.get("agent_name"), str)
                    or not isinstance(value.get("run_id"), str)
                    or isinstance(value.get("written_at"), bool)
                    or not isinstance(value.get("written_at"), (int, float))
                    or type(value.get("schema_version")) is not int
                    or value["schema_version"] != SCHEMA_VERSION
                ):
                    raise ValueError("memory entry has invalid fields")
                written_at = float(value["written_at"])
                datetime.fromtimestamp(written_at, timezone.utc)
                entries.append(MemoryEntry(
                    id=value["id"],
                    text=value["text"],
                    written_at=written_at,
                    agent_name=value["agent_name"],
                    run_id=value["run_id"],
                    scope=scope,
                ))
            return tuple(entries)
        except (OSError, TypeError, ValueError, OverflowError) as exc:
            raise MemoryUnavailable("read", path, exc, unreadable=True) from exc

    def read(self, scope: MemoryScope = "project") -> tuple[MemoryEntry, ...]:
        entries, _ = self.read_status(scope)
        return entries

    def read_status(
        self, scope: MemoryScope = "project"
    ) -> tuple[tuple[MemoryEntry, ...], bool]:
        path = self._path(scope)
        with self._lock:
            try:
                return self._decode(path, scope), False
            except MemoryUnavailable:
                return (), True

    def _write_entries(
        self, path: Path, entries: tuple[MemoryEntry, ...] | list[MemoryEntry]
    ) -> None:
        payload = [
            {
                "id": entry.id,
                "text": entry.text,
                "written_at": entry.written_at,
                "agent_name": entry.agent_name,
                "run_id": entry.run_id,
                "schema_version": entry.schema_version,
            }
            for entry in entries
        ]
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            raise MemoryUnavailable("write", path, exc) from exc

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("memory text must not be blank")
        if len(text) > MAX_ENTRY_CHARS:
            raise ValueError(
                f"MAX_ENTRY_CHARS is {MAX_ENTRY_CHARS}; actual length is {len(text)}"
            )

    def write(
        self,
        text: str,
        *,
        agent_name: str,
        run_id: str,
        scope: MemoryScope = "project",
    ) -> MemoryEntry:
        self._validate_text(text)
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("agent_name must be a non-blank string")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-blank string")
        path = self._path(scope)
        entry = MemoryEntry(
            id=uuid.uuid4().hex,
            text=text,
            written_at=time.time(),
            agent_name=agent_name,
            run_id=run_id,
            scope=scope,
        )
        with self._lock:
            entries = self._decode(path, scope)
            if (
                len(entries) >= MAX_ENTRIES
                or sum(len(existing.text) for existing in entries) + len(text)
                > MAX_TOTAL_CHARS
            ):
                raise ValueError(
                    "memory set is full; forget or merge outdated entries, or the person can remove entries in Settings → Memory"
                )
            self._write_entries(path, (*entries, entry))
        return entry

    def update(self, entry_id: str, text: str) -> MemoryEntry | None:
        self._validate_text(text)
        with self._lock:
            unreadable: MemoryUnavailable | None = None
            for scope in ("project", "global"):
                path = self._path(scope)
                try:
                    entries = self._decode(path, scope)
                except MemoryUnavailable as exc:
                    unreadable = exc
                    continue
                for index, existing in enumerate(entries):
                    if existing.id != entry_id:
                        continue
                    updated = replace(existing, text=text)
                    if sum(len(item.text) for item in entries) - len(existing.text) + len(text) > MAX_TOTAL_CHARS:
                        raise ValueError("memory set would exceed its total character limit")
                    changed = list(entries)
                    changed[index] = updated
                    self._write_entries(path, changed)
                    return updated
            if unreadable is not None:
                raise unreadable
        return None

    def delete(self, entry_id: str, expected_text: str | None = None) -> bool:
        with self._lock:
            unreadable: MemoryUnavailable | None = None
            for scope in ("project", "global"):
                path = self._path(scope)
                try:
                    entries = self._decode(path, scope)
                except MemoryUnavailable as exc:
                    unreadable = exc
                    continue
                if expected_text is not None and any(
                    entry.id == entry_id and entry.text != expected_text
                    for entry in entries
                ):
                    return False
                changed = [entry for entry in entries if entry.id != entry_id]
                if len(changed) != len(entries):
                    self._write_entries(path, changed)
                    return True
            if unreadable is not None:
                raise unreadable
        return False
