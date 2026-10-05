"""Append current language-server errors to successful file writes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from symphonai_api.cancellation import CancellationToken
from symphonai_api.lsp import LspError, LspManager
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.lsp import _utf16_to_column
from symphonai_api.tools.metadata import ToolMetadata


class DiagnosticsAfterWrite(LocalTool):
    def __init__(self, tool: LocalTool, manager: LspManager) -> None:
        self._tool = tool
        self._manager = manager

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def parameters(self) -> dict:
        return self._tool.parameters

    def validate(self, arguments: dict) -> str | None:
        return self._tool.validate(arguments)

    def metadata(self, arguments: dict) -> ToolMetadata:
        return self._tool.metadata(arguments)

    def _execute(self, tool_call: ToolCall, policy: PermissionPolicy, cancel: CancellationToken | None = None) -> ToolResult:
        result = self._tool.execute(tool_call, policy, cancel=cancel)
        if not result.ok:
            return result
        relative = tool_call.arguments.get("path")
        if not isinstance(relative, str):
            return result
        path = Path(relative)
        path = path if path.is_absolute() else policy.repo_root / path
        path = path.resolve()
        try:
            client = self._manager.client_for(path)
            if client is None:
                return result
            after = client.sync(path)
            diagnostics = client.wait_diagnostics(path, after, 3.0)
            if not isinstance(diagnostics, list):
                return result
            errors = [
                item for item in diagnostics
                if isinstance(item, dict) and type(item.get("severity")) is int and item["severity"] == 1
            ]
            if not errors:
                return result
            errors.sort(key=lambda item: (
                item.get("range", {}).get("start", {}).get("line", 0),
                item.get("range", {}).get("start", {}).get("character", 0),
            ))
            lines = []
            for item in errors[:20]:
                start = item.get("range", {}).get("start", {})
                line = int(start.get("line", 0))
                column = _utf16_to_column(path, line, int(start.get("character", 0)))
                repo_path = path.relative_to(policy.repo_root).as_posix()
                lines.append(f"{repo_path}:{line + 1}:{column}: {item.get('message', '')}")
            if len(errors) > 20:
                lines.append(f"… {len(errors) - 20} more")
            appended = f"\n\nErrors reported by {client.spec.name} after this write ({len(errors)}):\n" + "\n".join(lines)
            payload = result.payload
            if isinstance(payload, dict) and payload.get("kind") == "file_diff":
                payload = dict(payload)
                payload.setdefault("diff", result.content)
            return replace(result, content=result.content + appended, payload=payload)
        except (LspError, OSError, UnicodeError, TypeError, ValueError, KeyError, AttributeError):
            return result
