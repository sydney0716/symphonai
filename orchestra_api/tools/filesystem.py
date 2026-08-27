"""Filesystem local tools: read_file, write_file, list_files.

Every method consults `PermissionPolicy` before touching disk. A denial
returns `ToolResult(ok=False, ...)` with a clear reason instead of raising
or performing the disk access.
"""

from __future__ import annotations

import itertools

from orchestra_api.cancellation import CancellationToken
from orchestra_api.compaction import estimate_text_tokens
from orchestra_api.models import ToolCall, ToolResult
from orchestra_api.permissions import PermissionPolicy
from orchestra_api.tools.base import LocalTool
from orchestra_api.tools.metadata import ResultHint, ToolEffect, ToolMetadata

MAX_READ_BYTES = 1_000_000  # 1 MB safety cap on a single read_file call
MAX_READ_LINES = 2_000
MAX_READ_TOKENS = 25_000


class ReadFileTool(LocalTool):
    """Read a bounded line range from a UTF-8 file inside the allowed scope."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read a numbered line range from a text file inside the allowed scope."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read, relative to the allowed root.",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based first line to read.",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum lines to read; 0 reads to end of file. Defaults to {MAX_READ_LINES}.",
                    "default": MAX_READ_LINES,
                },
            },
            "required": ["path"],
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        path = arguments.get("path")
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(path,) if isinstance(path, str) else None,
        )

    def validate(self, arguments: dict) -> str | None:
        if not arguments.get("path"):
            return "missing required argument: path"
        offset = arguments.get("offset", 1)
        if isinstance(offset, bool) or not isinstance(offset, int):
            return "offset must be an integer"
        if offset < 1:
            return "offset must be >= 1"
        limit = arguments.get("limit", MAX_READ_LINES)
        if isinstance(limit, bool) or not isinstance(limit, int):
            return "limit must be an integer"
        if limit < 0:
            return "limit must be >= 0"
        return None

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        path = tool_call.arguments.get("path")
        decision = policy.check_read(path)
        if not decision.allowed:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=decision.reason)
        # Reuse the policy's own path resolution so this tool can never
        # disagree with the decision it was just given.
        resolved = policy._resolve_within_root(path)
        is_unranged = "offset" not in tool_call.arguments and "limit" not in tool_call.arguments
        if is_unranged:
            try:
                size = resolved.stat().st_size
            except OSError as exc:
                return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
            if size > MAX_READ_BYTES:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error=(
                        f"file exceeds {MAX_READ_BYTES} byte read limit; "
                        "read part of it with offset and limit"
                    ),
                )
        offset = tool_call.arguments.get("offset", 1)
        limit = tool_call.arguments.get("limit", MAX_READ_LINES)
        slice_start = offset - 1
        slice_stop = None if limit == 0 else slice_start + limit + 1
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                lines = list(itertools.islice(handle, slice_start, slice_stop))
        except UnicodeDecodeError:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error="file is not valid UTF-8 text")
        except OSError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        more_follows = limit != 0 and len(lines) > limit
        selected = lines[:limit] if more_follows else lines
        rendered_lines = [
            f"{lineno}\t{line.removesuffix(chr(10))}"
            for lineno, line in enumerate(selected, start=offset)
        ]
        rendered = "\n".join(rendered_lines)
        tokens = estimate_text_tokens(rendered)
        if tokens > MAX_READ_TOKENS:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=(
                    f"selected range is about {tokens} tokens, over the "
                    f"{MAX_READ_TOKENS} token limit; narrow it with offset and limit"
                ),
            )
        if selected:
            end = offset + len(selected) - 1
            if more_follows:
                rendered_lines.append(
                    f"[lines {offset}-{end}; more follow, pass offset={end + 1}]"
                )
            elif offset > 1:
                rendered_lines.append(f"[lines {offset}-{end}; end of file]")
        return ToolResult(tool_call_id=tool_call.id, ok=True, content="\n".join(rendered_lines))


class WriteFileTool(LocalTool):
    """Write UTF-8 text content to a file inside the explicit allowed write scope."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write text content to a file inside the explicit allowed write scope."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write, relative to the allowed root.",
                },
                "content": {
                    "type": "string",
                    "description": "Full text content to write to the file.",
                },
            },
            "required": ["path", "content"],
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        path = arguments.get("path")
        # Full replacement can discard contents that were never read.
        return ToolMetadata(
            effect=ToolEffect.DESTRUCTIVE,
            concurrency_safe=False,
            paths=(path,) if isinstance(path, str) else None,
        )

    def validate(self, arguments: dict) -> str | None:
        if not arguments.get("path"):
            return "missing required argument: path"
        if arguments.get("content") is None:
            return "missing required argument: content"
        return None

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        path = tool_call.arguments.get("path")
        content = tool_call.arguments.get("content")
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

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list, relative to the allowed root. Defaults to '.' if omitted.",
                },
            },
            "required": [],
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        path = arguments.get("path", ".")
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(path,) if isinstance(path, str) else None,
            result_hint=ResultHint.FILE_LIST,
        )

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
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
