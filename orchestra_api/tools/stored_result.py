"""Local tool for reading bounded slices of offloaded tool-result text."""

from __future__ import annotations

from orchestra_api.cancellation import CancellationToken
from orchestra_api.models import ToolCall, ToolResult
from orchestra_api.permissions import PermissionPolicy
from orchestra_api.tool_results import MAX_RESULT_SLICE_CHARS, ToolResultStore
from orchestra_api.tools.base import LocalTool
from orchestra_api.tools.metadata import ResultHint, ToolEffect, ToolMetadata


class ReadToolResultTool(LocalTool):
    """Read stored text without a policy consult; the call has no path or side effect."""

    def __init__(self, store: ToolResultStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "read_tool_result"

    @property
    def description(self) -> str:
        return "Read a bounded character slice from an offloaded tool result."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Stored result id from an offload marker.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Zero-based character offset.",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum characters to return.",
                    "default": MAX_RESULT_SLICE_CHARS,
                },
            },
            "required": ["id"],
        }

    def validate(self, arguments: dict) -> str | None:
        if not isinstance(arguments.get("id"), str):
            return "missing or invalid required argument: id (must be a string)"
        for name, default in (("offset", 0), ("limit", MAX_RESULT_SLICE_CHARS)):
            value = arguments.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int):
                return f"{name} must be an integer"
            if value < 0:
                return f"{name} must be >= 0"
        return None

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(),
            result_hint=ResultHint.TEXT,
        )

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        result_id = tool_call.arguments["id"]
        stored = self._store.get(result_id)
        if stored is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=(
                    f"no stored result with id {result_id!r}; "
                    "it may have been evicted -- re-run the tool"
                ),
            )

        offset = tool_call.arguments.get("offset", 0)
        limit = min(
            tool_call.arguments.get("limit", MAX_RESULT_SLICE_CHARS),
            MAX_RESULT_SLICE_CHARS,
        )
        slice_text = stored.content[offset : offset + limit]
        next_offset = offset + len(slice_text)
        more_follows = next_offset < len(stored.content)
        content = slice_text
        if more_follows:
            remaining = len(stored.content) - next_offset
            marker = (
                f"[{remaining} characters remain; call read_tool_result again "
                f"with offset={next_offset}]"
            )
            content = f"{slice_text}\n{marker}" if slice_text else marker
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=True,
            content=content,
            payload={
                "kind": "stored_result_slice",
                "id": result_id,
                "offset": offset,
                "characters": len(slice_text),
                "total_characters": len(stored.content),
                "more_follows": more_follows,
            },
        )
