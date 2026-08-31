"""Bounded in-process storage and preview offload for large tool results.

The character thresholds and store bounds are reasoned starting points, not
measured tuning values. The store deliberately has no filesystem location;
its lifetime matches the in-memory conversation that holds its handles.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace

from orchestra_api.models import OffloadedResult, ToolResult


OFFLOAD_THRESHOLD_CHARS = 4_000
PREVIEW_CHARS = 2_000
MIN_PREVIEW_CHARS = PREVIEW_CHARS // 2
MAX_RESULT_SLICE_CHARS = 20_000
MAX_STORE_ENTRIES = 100
MAX_STORE_CHARS = 25_000_000


@dataclass(frozen=True)
class StoredResult:
    id: str
    tool_name: str
    tool_call_id: str
    content: str


class ToolResultStore:
    """Addressable, in-process, bounded. No filesystem, one lock."""

    def __init__(self) -> None:
        self._results: OrderedDict[str, StoredResult] = OrderedDict()
        self._characters = 0
        self._lock = threading.Lock()

    def store(self, *, tool_name: str, tool_call_id: str, content: str) -> StoredResult:
        result_id = "res_" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        with self._lock:
            existing = self._results.get(result_id)
            if existing is not None:
                self._results.move_to_end(result_id)
                return existing
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

    def __len__(self) -> int:
        with self._lock:
            return len(self._results)

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
