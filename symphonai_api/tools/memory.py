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
        self.forget_tool = ForgetMemoryTool(store)

    @property
    def name(self) -> str:
        return "remember"

    @property
    def description(self) -> str:
        return (
            "Record a durable lesson or standing preference the person taught you. "
            "Use global only for preferences about the person that hold in any "
            "repository. Do not store task state, findings, or a summary of the run."
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
                "scope": {
                    "type": "string",
                    "enum": ["project", "global"],
                    "default": "project",
                    "description": "Use global only for preferences that hold in every repository.",
                },
                "replaces": {
                    "type": "string",
                    "description": "The id of an existing memory to update.",
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
        replaces = arguments.get("replaces")
        if replaces is not None and (not isinstance(replaces, str) or not replaces):
            return "replaces must be a non-blank memory id"
        scope = arguments.get("scope", "project")
        if replaces is None and scope not in ("project", "global"):
            return "scope must be 'project' or 'global'"
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
            replaces = tool_call.arguments.get("replaces")
            if replaces is not None:
                updated = self._store.update(replaces, tool_call.arguments["text"])
                if updated is None:
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        ok=False,
                        error="no memory has that id; nothing was saved",
                    )
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=True,
                    content=f"Updated {updated.id}.",
                )
            entry = self._store.write(
                tool_call.arguments["text"],
                agent_name=self._agent_name,
                run_id=run_id,
                scope=tool_call.arguments.get("scope", "project"),
            )
        except MemoryUnavailable as exc:
            error = (
                "memory file could not be read; nothing was saved"
                if exc.unreadable
                else str(exc)
            )
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=error)
        except ValueError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=True,
            content=f"Remembered as {entry.id}.",
        )


class ForgetMemoryTool(LocalTool):
    def __init__(self, store: AgentMemory | None) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "forget"

    @property
    def description(self) -> str:
        return "Remove a saved memory when asked, or when it is outdated or wrong."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The saved memory id."},
                "text": {
                    "type": "string",
                    "description": "The memory's current text, copied exactly from its seed.",
                },
            },
            "required": ["id", "text"],
        }

    def validate(self, arguments: dict) -> str | None:
        if not isinstance(arguments.get("id"), str) or not arguments["id"]:
            return "id must be a non-blank string"
        if not isinstance(arguments.get("text"), str) or not arguments["text"].strip():
            return "text must be a non-blank string"
        return None

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(effect=ToolEffect.DESTRUCTIVE, concurrency_safe=False, paths=None)

    def _execute(self, tool_call: ToolCall, policy: "PermissionPolicy", cancel=None) -> ToolResult:
        if self._store is None:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error="memory write is unavailable")
        entry_id = tool_call.arguments["id"]
        text = tool_call.arguments["text"]
        try:
            entries = []
            unreadable = False
            for scope in ("project", "global"):
                scope_entries, scope_unreadable = self._store.read_status(scope)
                entries.extend(scope_entries)
                unreadable = unreadable or scope_unreadable
            entry = next((item for item in entries if item.id == entry_id), None)
            if entry is None:
                if unreadable:
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        ok=False,
                        error="memory file could not be read; nothing was removed",
                    )
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error="no memory has that id; nothing was removed",
                )
            if entry.text != text:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error="memory changed since it was read; nothing was removed",
                )
            if not self._store.delete(entry_id, expected_text=text):
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error="memory changed since it was read; nothing was removed",
                )
        except MemoryUnavailable as exc:
            error = (
                "memory file could not be read; nothing was removed"
                if exc.unreadable
                else str(exc)
            )
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=error)
        return ToolResult(tool_call_id=tool_call.id, ok=True, content=f"Forgot {entry_id}.")
