"""The opt-in tool for recording one durable per-agent lesson."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from symphonai_api.agent_memory import (
    MAX_ENTRY_CHARS,
    AgentMemory,
    MemoryUnavailable,
)
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata

if TYPE_CHECKING:
    from symphonai_api.cancellation import CancellationToken
    from symphonai_api.permissions import PermissionPolicy


class MemoryTool(LocalTool):
    def __init__(
        self,
        store: AgentMemory | None,
        agent_name: str,
        run_id: Callable[[], str | None],
    ) -> None:
        self._store = store
        self._agent_name = agent_name
        self._run_id = run_id

    @property
    def name(self) -> str:
        return "remember"

    @property
    def description(self) -> str:
        return (
            "Record a durable lesson or standing preference the person taught you. "
            "Do not store task state, findings, or a summary of the run."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "maxLength": MAX_ENTRY_CHARS,
                    "description": "The durable lesson to remember.",
                },
            },
            "required": ["text"],
        }

    def validate(self, arguments: dict) -> str | None:
        text = arguments.get("text")
        if not isinstance(text, str) or not text.strip():
            return "text must be a non-blank string"
        if len(text) > MAX_ENTRY_CHARS:
            return (
                f"MAX_ENTRY_CHARS is {MAX_ENTRY_CHARS}; "
                f"actual length is {len(text)}"
            )
        return None

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.DESTRUCTIVE,
            concurrency_safe=False,
            paths=None,
        )

    def _execute(
        self,
        tool_call: ToolCall,
        policy: "PermissionPolicy",
        cancel: "CancellationToken | None" = None,
    ) -> ToolResult:
        if self._store is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error="memory write is unavailable",
            )
        run_id = self._run_id()
        if run_id is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error="memory write has no active run id",
            )
        try:
            self._store.write(
                self._agent_name,
                tool_call.arguments["text"],
                run_id=run_id,
            )
        except (MemoryUnavailable, ValueError) as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        return ToolResult(tool_call_id=tool_call.id, ok=True, content="Remembered.")
