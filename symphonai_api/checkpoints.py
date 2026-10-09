"""Append-only per-prompt snapshots of files changed by agent file tools."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointEntry:
    key: str
    path: str
    backup: str | None


class CheckpointStore:
    def __init__(self, directory: Path, repo_root: Path) -> None:
        self.directory = Path(directory)
        self.repo_root = Path(repo_root).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._manifest = self.directory / "manifest.jsonl"
        self._lock = threading.RLock()
        self._key: str | None = None
        self._keys: list[str] = []
        self._labels: dict[str, str] = {}
        self._entries: list[CheckpointEntry] = []
        self._entry_keys: set[tuple[str, str]] = set()
        self._pending: dict[tuple[str, str], str | None] = {}
        self._last_written: dict[str, str] = {}
        self._backup_number = 1
        self._reload()

    def _reload(self) -> None:
        if not self._manifest.exists():
            return
        with self._manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid checkpoint manifest line {line_number}") from exc
                kind = record.get("type") if isinstance(record, dict) else None
                if kind == "begin" and isinstance(record.get("key"), str):
                    self._key = record["key"]
                    self._keys.append(self._key)
                    label = record.get("label")
                    if isinstance(label, str):
                        self._labels[self._key] = label
                elif kind == "write":
                    key, path, backup = record.get("key"), record.get("path"), record.get("backup")
                    if (
                        isinstance(key, str)
                        and isinstance(path, str)
                        and (backup is None or isinstance(backup, str) and Path(backup).name == backup)
                    ):
                        entry = CheckpointEntry(key, path, backup)
                        self._entries.append(entry)
                        self._entry_keys.add((key, path))
                        if backup is not None and backup.startswith("backup-"):
                            try:
                                self._backup_number = max(
                                    self._backup_number,
                                    int(backup.removeprefix("backup-").removesuffix(".bin")) + 1,
                                )
                            except ValueError:
                                pass
                        digest = record.get("sha256")
                        if isinstance(digest, str):
                            self._last_written[path] = digest
                        elif digest is None:
                            self._last_written.pop(path, None)
                elif kind in ("after_write", "restore"):
                    path, digest = record.get("path"), record.get("sha256")
                    if isinstance(path, str):
                        if isinstance(digest, str):
                            self._last_written[path] = digest
                        elif digest is None:
                            self._last_written.pop(path, None)

    def _append(self, record: dict) -> None:
        with self._manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _relative_path(self, path: Path) -> tuple[Path, str]:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(self.repo_root).as_posix()
        except ValueError:
            raise ValueError(f"checkpoint path is outside repository: {path}") from None
        return resolved, relative

    def begin(self, key: str, *, label: str | None = None) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("checkpoint key must be a non-empty string")
        if label is not None and not isinstance(label, str):
            raise ValueError("checkpoint label must be a string")
        with self._lock:
            for backup in self._pending.values():
                if backup is not None:
                    (self.directory / backup).unlink(missing_ok=True)
            self._pending.clear()
            record = {"type": "begin", "key": key}
            if label is not None:
                record["label"] = label
                self._labels[key] = label
            self._append(record)
            self._key = key
            self._keys.append(key)

    def before_write(self, path: Path) -> None:
        with self._lock:
            if self._key is None:
                return
            resolved, relative = self._relative_path(path)
            identity = (self._key, relative)
            if identity in self._entry_keys or identity in self._pending:
                return
            if resolved.exists():
                backup = f"backup-{self._backup_number:08d}.bin"
                self._backup_number += 1
                (self.directory / backup).write_bytes(resolved.read_bytes())
            else:
                backup = None
            self._pending[identity] = backup

    def after_write(self, path: Path) -> None:
        with self._lock:
            if self._key is None:
                return
            resolved, relative = self._relative_path(path)
            identity = (self._key, relative)
            if identity not in self._entry_keys and identity not in self._pending:
                return
            if not resolved.exists():
                if identity in self._entry_keys:
                    self._append({"type": "after_write", "path": relative, "sha256": None})
                else:
                    backup = self._pending[identity]
                    self._append({
                        "type": "write",
                        "key": self._key,
                        "path": relative,
                        "backup": backup,
                        "sha256": None,
                    })
                    self._entries.append(CheckpointEntry(self._key, relative, backup))
                    self._entry_keys.add(identity)
                    self._pending.pop(identity, None)
                self._last_written.pop(relative, None)
                return
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if identity in self._entry_keys:
                self._append({"type": "after_write", "path": relative, "sha256": digest})
            else:
                backup = self._pending[identity]
                self._append({
                    "type": "write",
                    "key": self._key,
                    "path": relative,
                    "backup": backup,
                    "sha256": digest,
                })
                self._entries.append(CheckpointEntry(self._key, relative, backup))
                self._entry_keys.add(identity)
                self._pending.pop(identity, None)
            self._last_written[relative] = digest

    def entries(self) -> tuple[CheckpointEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def last_written(self, path: str) -> str | None:
        with self._lock:
            return self._last_written.get(Path(path).as_posix())

    def restore(self, path: str, content: bytes | None) -> None:
        with self._lock:
            requested = Path(path)
            if requested.is_absolute() or ".." in requested.parts:
                raise ValueError(f"invalid checkpoint path: {path}")
            target = self.repo_root / requested
            parent = target.parent.resolve()
            if not parent.is_relative_to(self.repo_root):
                raise ValueError(f"checkpoint path is outside repository: {path}")
            resolved = parent / target.name
            relative = resolved.relative_to(self.repo_root).as_posix()
            if resolved.is_symlink():
                resolved.unlink()
            if content is None:
                resolved.unlink(missing_ok=True)
                digest = None
            else:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_bytes(content)
                digest = hashlib.sha256(content).hexdigest()
            self._append({"type": "restore", "path": relative, "sha256": digest})
            if digest is None:
                self._last_written.pop(relative, None)
            else:
                self._last_written[relative] = digest

    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._keys)

    def label(self, key: str) -> str | None:
        with self._lock:
            return self._labels.get(key)

    def copy_kept_to(
        self,
        destination: CheckpointStore,
        keys: tuple[str, ...],
        current_files: dict[str, bytes | None],
    ) -> None:
        """Copy the selected prompt snapshots and branch file state."""
        selected = set(keys)
        with self._lock, destination._lock:
            for key in keys:
                destination.begin(key, label=self.label(key))
            for entry in self._entries:
                if entry.key not in selected:
                    continue
                backup = None
                if entry.backup is not None:
                    backup = f"backup-{destination._backup_number:08d}.bin"
                    destination._backup_number += 1
                    (destination.directory / backup).write_bytes(
                        (self.directory / entry.backup).read_bytes()
                    )
                destination._append({
                    "type": "write",
                    "key": entry.key,
                    "path": entry.path,
                    "backup": backup,
                    "sha256": None,
                })
                destination._entries.append(CheckpointEntry(entry.key, entry.path, backup))
                destination._entry_keys.add((entry.key, entry.path))
            for path, content in current_files.items():
                digest = None if content is None else hashlib.sha256(content).hexdigest()
                destination._append({"type": "restore", "path": path, "sha256": digest})
                if digest is None:
                    destination._last_written.pop(path, None)
                else:
                    destination._last_written[path] = digest
