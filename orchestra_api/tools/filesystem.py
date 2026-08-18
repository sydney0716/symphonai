"""Filesystem local tools: read_file, write_file, list_files.

Every method consults `PermissionPolicy` before touching disk. A denial
returns `ToolResult(ok=False, ...)` with a clear reason instead of raising
or performing the disk access.
"""

from __future__ import annotations

from orchestra_api.models import ToolCall, ToolResult
from orchestra_api.permissions import PermissionPolicy
from orchestra_api.tools.base import LocalTool

MAX_READ_BYTES = 1_000_000  # 1 MB safety cap on a single read_file call


class ReadFileTool(LocalTool):
    """Read the full UTF-8 text contents of a file inside the allowed scope."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the full contents of a text file inside the allowed scope."

    def execute(self, tool_call: ToolCall, policy: PermissionPolicy) -> ToolResult:
        path = tool_call.arguments.get("path")
        if not path:
            return ToolResult(
                tool_call_id=tool_call.id, ok=False, error="missing required argument: path"
            )
        decision = policy.check_read(path)
        if not decision.allowed:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=decision.reason)
        # Reuse the policy's own path resolution so this tool can never
        # disagree with the decision it was just given.
        resolved = policy._resolve_within_root(path)
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        if len(data) > MAX_READ_BYTES:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"file exceeds {MAX_READ_BYTES} byte read limit",
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error="file is not valid UTF-8 text")
        return ToolResult(tool_call_id=tool_call.id, ok=True, content=text)


class WriteFileTool(LocalTool):
    """Write UTF-8 text content to a file inside the explicit allowed write scope."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write text content to a file inside the explicit allowed write scope."

    def execute(self, tool_call: ToolCall, policy: PermissionPolicy) -> ToolResult:
        path = tool_call.arguments.get("path")
        content = tool_call.arguments.get("content")
        if not path:
            return ToolResult(
                tool_call_id=tool_call.id, ok=False, error="missing required argument: path"
            )
        if content is None:
            return ToolResult(
                tool_call_id=tool_call.id, ok=False, error="missing required argument: content"
            )
        decision = policy.check_write(path)
        if not decision.allowed:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=decision.reason)
        resolved = policy._resolve_within_root(path)
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        return ToolResult(
            tool_call_id=tool_call.id, ok=True, content=f"wrote {len(content)} chars to {path}"
        )


class ListFilesTool(LocalTool):
    """List entries of a directory inside the allowed scope."""

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return "List entries in a directory inside the allowed scope."

    def execute(self, tool_call: ToolCall, policy: PermissionPolicy) -> ToolResult:
        path = tool_call.arguments.get("path", ".")
        decision = policy.check_list(path)
        if not decision.allowed:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=decision.reason)
        resolved = policy._resolve_within_root(path)
        try:
            if not resolved.is_dir():
                return ToolResult(tool_call_id=tool_call.id, ok=False, error=f"not a directory: {path}")
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in resolved.iterdir())
        except OSError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        return ToolResult(tool_call_id=tool_call.id, ok=True, content="\n".join(entries))
