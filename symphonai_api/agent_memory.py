"""Bounded, per-agent memory persisted independently of run transcripts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time

from symphonai_api.identity import SCHEMA_VERSION


MAX_ENTRIES = 32
MAX_ENTRY_CHARS = 500
MAX_TOTAL_CHARS = 8000


@dataclass(frozen=True)
class MemoryEntry:
    text: str
    agent_name: str
    run_id: str
    written_at: float
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


class AgentMemory:
    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, agent_name: str) -> Path:
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("agent_name must be a non-blank string")
        encoded = agent_name.encode("utf-8").hex()
        return self._root / f"{encoded}.json"

    def _read_path(self, path: Path) -> tuple[MemoryEntry, ...]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("memory file must contain a list")
            entries: list[MemoryEntry] = []
            for value in payload:
                if not isinstance(value, dict):
                    raise ValueError("memory entry must be an object")
                if (
                    not isinstance(value.get("text"), str)
                    or not isinstance(value.get("agent_name"), str)
                    or not isinstance(value.get("run_id"), str)
                    or isinstance(value.get("written_at"), bool)
                    or not isinstance(value.get("written_at"), (int, float))
                    or type(value.get("schema_version")) is not int
                    or value["schema_version"] != SCHEMA_VERSION
                ):
                    raise ValueError("memory entry has invalid fields")
                entries.append(
                    MemoryEntry(
                        text=value["text"],
                        agent_name=value["agent_name"],
                        run_id=value["run_id"],
                        written_at=float(value["written_at"]),
                        schema_version=value["schema_version"],
                    )
                )
            return tuple(entries)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return ()

    def read(self, agent_name: str) -> tuple[MemoryEntry, ...]:
        path = self._path(agent_name)
        with self._lock:
            return self._read_path(path)

    def write(self, agent_name: str, text: str, *, run_id: str) -> MemoryEntry:
        path = self._path(agent_name)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("memory text must not be blank")
        if len(text) > MAX_ENTRY_CHARS:
            raise ValueError(
                f"MAX_ENTRY_CHARS is {MAX_ENTRY_CHARS}; actual length is {len(text)}"
            )
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-blank string")
        entry = MemoryEntry(
            text=text,
            agent_name=agent_name,
            run_id=run_id,
            written_at=time.time(),
        )
        with self._lock:
            entries = list(self._read_path(path))
            entries.append(entry)
            while (
                len(entries) > MAX_ENTRIES
                or sum(len(existing.text) for existing in entries) > MAX_TOTAL_CHARS
            ):
                entries.pop(0)
            payload = [
                {
                    "text": existing.text,
                    "agent_name": existing.agent_name,
                    "run_id": existing.run_id,
                    "written_at": existing.written_at,
                    "schema_version": existing.schema_version,
                }
                for existing in entries
            ]
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(path)
        return entry

    def forget(self, agent_name: str) -> None:
        path = self._path(agent_name)
        with self._lock:
            path.unlink(missing_ok=True)
