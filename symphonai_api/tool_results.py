"""Bounded storage and preview offload for large tool results.

The character thresholds and store bounds are reasoned starting points, not
measured tuning values. A store is memory-only by default, or can use a session
directory so its content-addressed handles survive process restarts.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from symphonai_api.models import OffloadedResult, ToolResult
from symphonai_api.session import TranscriptError


OFFLOAD_THRESHOLD_CHARS = 4_000
PREVIEW_CHARS = 2_000
MIN_PREVIEW_CHARS = PREVIEW_CHARS // 2
MAX_RESULT_SLICE_CHARS = 20_000
MAX_STORE_ENTRIES = 100
MAX_STORE_CHARS = 25_000_000
_RESULT_ID_PATTERN = re.compile(r"^res_[0-9a-f]{12}$")


@dataclass(frozen=True)
class StoredResult:
    id: str
    tool_name: str
    tool_call_id: str
    content: str


class ToolResultStore:
    """Addressable and memory-bounded, with optional persistent backing."""

    def __init__(
        self,
        *,
        directory: Path | None = None,
        fallback_directories: Sequence[Path] = (),
    ) -> None:
        self._results: OrderedDict[str, StoredResult] = OrderedDict()
        self._characters = 0
        self._lock = threading.Lock()
        self._directory = None if directory is None else Path(directory)
        self._fallback_directories = tuple(Path(path) for path in fallback_directories)

    def store(self, *, tool_name: str, tool_call_id: str, content: str) -> StoredResult:
        result_id = "res_" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        with self._lock:
            existing = self._results.get(result_id)
            if existing is not None:
                self._persist(result_id, content)
                self._results.move_to_end(result_id)
                return existing
            self._persist(result_id, content)
            stored = StoredResult(
                id=result_id,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                content=content,
            )
            self._results[result_id] = stored
            self._characters += len(content)
            self._enforce_bounds()
            return stored

    def get(self, result_id: str) -> StoredResult | None:
        with self._lock:
            stored = self._results.get(result_id)
            if stored is not None:
                self._results.move_to_end(result_id)
                return stored
            if (self._directory is None and not self._fallback_directories) or (
                _RESULT_ID_PATTERN.fullmatch(result_id) is None
            ):
                return None
            content = self._read_persisted(result_id)
            if content is None:
                return None
            stored = StoredResult(
                id=result_id,
                tool_name="",
                tool_call_id="",
                content=content,
            )
            self._results[result_id] = stored
            self._characters += len(content)
            self._enforce_bounds()
            return stored

    def prune(self, max_bytes: int) -> None:
        """Remove oldest primary backing files until their total fits the cap."""

        if max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        if self._directory is None:
            return
        with self._lock:
            try:
                files = [path for path in self._directory.iterdir() if path.is_file()]
                entries = []
                for path in files:
                    file_stat = path.stat()
                    entries.append((file_stat.st_mtime_ns, file_stat.st_size, path))
            except OSError as exc:
                raise TranscriptError(
                    f"cannot inspect tool-result directory {self._directory}: {exc}"
                ) from exc
            total = sum(size for _, size, _ in entries)
            ordered = sorted(entries, key=lambda entry: (entry[0], entry[2].name))
            for _, size, path in ordered:
                if total <= max_bytes:
                    break
                try:
                    path.unlink()
                except FileNotFoundError:
                    total -= size
                    continue
                except OSError as exc:
                    raise TranscriptError(
                        f"cannot prune tool result {path.name}: {exc}"
                    ) from exc
                total -= size

    def __len__(self) -> int:
        with self._lock:
            return len(self._results)

    def _persist(self, result_id: str, content: str) -> None:
        if self._directory is None:
            return
        path = self._directory / result_id
        if path.is_file():
            return
        temporary_path: Path | None = None
        descriptor: int | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{result_id}-",
                dir=self._directory,
            )
            temporary_path = Path(temporary_name)
            os.fchmod(descriptor, 0o600)
            handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
            descriptor = None
            with handle:
                handle.write(content)
                handle.flush()
            os.replace(temporary_path, path)
            os.chmod(path, 0o600)
        except (OSError, UnicodeError) as exc:
            raise TranscriptError(f"cannot write tool result {result_id}: {exc}") from exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _read_persisted(self, result_id: str) -> str | None:
        directories = (self._directory, *self._fallback_directories)
        for directory in directories:
            if directory is None:
                continue
            try:
                # Decode the bytes rather than reading text: universal newlines
                # would rewrite "\r\n" and "\r" to "\n", and content that no
                # longer hashes to the handle it was fetched by is not the
                # content that was stored.
                return (directory / result_id).read_bytes().decode("utf-8")
            except (OSError, UnicodeError):
                continue
        return None

    def _enforce_bounds(self) -> None:
        # Never evict the entry just accepted: the caller is about to hand its
        # id to the model, and a handle that is dead on arrival sends the model
        # back to re-run the tool into the same state. A single result larger
        # than MAX_STORE_CHARS therefore leaves the store holding only it --
        # those bytes are in the process either way, and the alternative is
        # putting them in the conversation.
        while len(self._results) > MAX_STORE_ENTRIES or (
            self._characters > MAX_STORE_CHARS and len(self._results) > 1
        ):
            _, evicted = self._results.popitem(last=False)
            self._characters -= len(evicted.content)


def offload_tool_result(
    result: ToolResult,
    *,
    tool_name: str,
    store: ToolResultStore,
) -> ToolResult:
    """Replace a large successful result's text with a preview and handle."""

    if (
        not result.ok
        or result.cancelled
        or len(result.content) <= OFFLOAD_THRESHOLD_CHARS
    ):
        return result

    stored = store.store(
        tool_name=tool_name,
        tool_call_id=result.tool_call_id,
        content=result.content,
    )
    preview = result.content[:PREVIEW_CHARS]
    newline = preview.rfind("\n")
    # Cut to a line boundary only when a usable preview survives it. A result
    # whose one early newline sits at the front would otherwise be previewed as
    # nothing at all, forcing a second call to learn anything.
    if newline >= MIN_PREVIEW_CHARS:
        preview = preview[:newline]
    marker = (
        f"[tool result offloaded: {len(result.content)} characters stored as {stored.id}; "
        f"the first {len(preview)} are shown above. Call read_tool_result with id "
        f'"{stored.id}" and offset/limit to read the rest.]'
    )
    return replace(
        result,
        content=f"{preview}\n{marker}",
        offloaded=OffloadedResult(
            id=stored.id,
            characters=len(result.content),
            preview_characters=len(preview),
            tool_name=tool_name,
        ),
    )
