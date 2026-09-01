"""Targeted single- and multi-edit tools with structured diff results."""

from __future__ import annotations

import difflib

from symphonai_api.cancellation import CancellationToken
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools import filesystem as filesystem_tools
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ResultHint, ToolEffect, ToolMetadata
from symphonai_api.tools.read_ledger import ReadLedger


def _edit_validation_error(arguments: dict) -> str | None:
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string")
    if not isinstance(old_string, str):
        return "missing required argument: old_string"
    if not isinstance(new_string, str):
        return "missing required argument: new_string"
    if old_string == "":
        return "old_string must not be empty; use write_file to create a file"
    if old_string == new_string:
        return "old_string and new_string are identical"
    if "replace_all" in arguments and not isinstance(arguments["replace_all"], bool):
        return "replace_all must be a boolean"
    return None


def _diff_result(tool_call_id: str, path: str, old: str, new: str) -> ToolResult:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    content = "\n".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=path,
            tofile=path,
            lineterm="",
            n=3,
        )
    )

    hunks: list[dict] = []
    lines_added = 0
    lines_removed = 0
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    for group in matcher.get_grouped_opcodes(3):
        first_i1 = group[0][1]
        first_j1 = group[0][3]
        hunk_lines: list[dict[str, str]] = []
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                hunk_lines.extend(
                    {"op": "context", "text": line} for line in old_lines[i1:i2]
                )
            elif tag == "delete":
                removed = old_lines[i1:i2]
                lines_removed += len(removed)
                hunk_lines.extend({"op": "remove", "text": line} for line in removed)
            elif tag == "insert":
                added = new_lines[j1:j2]
                lines_added += len(added)
                hunk_lines.extend({"op": "add", "text": line} for line in added)
            else:
                removed = old_lines[i1:i2]
                added = new_lines[j1:j2]
                lines_removed += len(removed)
                lines_added += len(added)
                hunk_lines.extend({"op": "remove", "text": line} for line in removed)
                hunk_lines.extend({"op": "add", "text": line} for line in added)
        hunks.append(
            {
                "old_start": first_i1 + 1,
                "old_lines": group[-1][2] - first_i1,
                "new_start": first_j1 + 1,
                "new_lines": group[-1][4] - first_j1,
                "lines": hunk_lines,
            }
        )

    truncated = len(content) > filesystem_tools.MAX_READ_BYTES
    if truncated:
        content = (
            f"[diff omitted: {lines_added} lines added, {lines_removed} removed, "
            f"over the {filesystem_tools.MAX_READ_BYTES} byte diff limit]"
        )
        hunks = []
    return ToolResult(
        tool_call_id=tool_call_id,
        ok=True,
        content=content,
        payload={
            "kind": "file_diff",
            "path": path,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "truncated": truncated,
            "hunks": hunks,
        },
    )


class EditFileTool(LocalTool):
    """Replace one exact string, or every occurrence, in a UTF-8 file."""

    def __init__(self, ledger: ReadLedger) -> None:
        self._ledger = ledger

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Replace exact text in a file that has been read."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit, relative to the allowed root.",
                },
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence instead of requiring one match.",
                    "default": False,
                },
            },
            "required": ["path", "old_string", "new_string"],
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        path = arguments.get("path")
        return ToolMetadata(
            effect=ToolEffect.MUTATING,
            concurrency_safe=False,
            paths=(path,) if isinstance(path, str) else None,
            result_hint=ResultHint.DIFF,
        )

    def validate(self, arguments: dict) -> str | None:
        path = arguments.get("path")
        if not isinstance(path, str) or not path:
            return "missing required argument: path"
        return _edit_validation_error(arguments)

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        path = tool_call.arguments["path"]
        old_string = tool_call.arguments["old_string"]
        new_string = tool_call.arguments["new_string"]
        replace_all = tool_call.arguments.get("replace_all", False)
        decision = policy.check_write(path)
        if not decision.allowed:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=decision.reason)
        resolved = policy._resolve_within_root(path)
        try:
            content = resolved.read_text(encoding="utf-8")
            stale_error = self._ledger.check(resolved)
        except UnicodeDecodeError:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error="file is not valid UTF-8 text")
        except OSError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        if stale_error is not None:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=stale_error)

        count = content.count(old_string)
        if count == 0:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"old_string not found in {path}",
            )
        if count > 1 and not replace_all:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=(
                    f"old_string matches {count} times in {path}; set replace_all=true to "
                    "change every match, or add surrounding context to identify one"
                ),
            )
        updated = content.replace(old_string, new_string) if replace_all else content.replace(
            old_string, new_string, 1
        )
        try:
            resolved.write_text(updated, encoding="utf-8")
            self._ledger.record(resolved, full=True, content=updated)
        except OSError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        return _diff_result(tool_call.id, path, content, updated)


class MultiEditFileTool(EditFileTool):
    """Apply an ordered, all-or-nothing batch of exact replacements."""

    @property
    def name(self) -> str:
        return "multi_edit_file"

    @property
    def description(self) -> str:
        return "Apply ordered exact-text edits to a file that has been read."

    @property
    def parameters(self) -> dict:
        edit_properties = {
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring one match.",
                "default": False,
            },
        }
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit, relative to the allowed root.",
                },
                "edits": {
                    "type": "array",
                    "description": "Ordered exact-text replacements.",
                    "items": {
                        "type": "object",
                        "properties": edit_properties,
                        "required": ["old_string", "new_string"],
                    },
                },
            },
            "required": ["path", "edits"],
        }

    def validate(self, arguments: dict) -> str | None:
        path = arguments.get("path")
        if not isinstance(path, str) or not path:
            return "missing required argument: path"
        edits = arguments.get("edits")
        if not isinstance(edits, list) or not edits or any(
            not isinstance(edit, dict) for edit in edits
        ):
            return "missing required argument: edits"
        for index, edit in enumerate(edits, start=1):
            error = _edit_validation_error(edit)
            if error is not None:
                return f"edit {index}: {error}"
        return None

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        path = tool_call.arguments["path"]
        decision = policy.check_write(path)
        if not decision.allowed:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=decision.reason)
        resolved = policy._resolve_within_root(path)
        try:
            original = resolved.read_text(encoding="utf-8")
            stale_error = self._ledger.check(resolved)
        except UnicodeDecodeError:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error="file is not valid UTF-8 text")
        except OSError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        if stale_error is not None:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=stale_error)

        updated = original
        for index, edit in enumerate(tool_call.arguments["edits"], start=1):
            old_string = edit["old_string"]
            count = updated.count(old_string)
            if count == 0:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error=f"edit {index}: old_string not found in {path}",
                )
            if count > 1 and not edit.get("replace_all", False):
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error=(
                        f"edit {index}: old_string matches {count} times in {path}; "
                        "set replace_all=true to change every match, or add surrounding "
                        "context to identify one"
                    ),
                )
            updated = (
                updated.replace(old_string, edit["new_string"])
                if edit.get("replace_all", False)
                else updated.replace(old_string, edit["new_string"], 1)
            )
        try:
            resolved.write_text(updated, encoding="utf-8")
            self._ledger.record(resolved, full=True, content=updated)
        except OSError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        return _diff_result(tool_call.id, path, original, updated)
