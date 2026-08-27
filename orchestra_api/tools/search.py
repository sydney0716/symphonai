"""Read-only file search tools with bounded, permission-gated results."""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from orchestra_api.cancellation import CancellationToken
from orchestra_api.models import ToolCall, ToolResult
from orchestra_api.permissions import PermissionPolicy
from orchestra_api.tools.base import LocalTool
from orchestra_api.tools.filesystem import MAX_READ_BYTES
from orchestra_api.tools.metadata import ResultHint, ToolEffect, ToolMetadata

DEFAULT_HEAD_LIMIT = 250
CANCEL_CHECK_INTERVAL_FILES = 64


def apply_head_limit(
    items: list, limit: int | None, offset: int = 0
) -> tuple[list, int | None]:
    """Return a page and the cap only when that cap truncated the results."""
    effective_limit = DEFAULT_HEAD_LIMIT if limit is None else limit
    if effective_limit == 0:
        return items[offset:], None
    page = items[offset : offset + effective_limit]
    applied_limit = effective_limit if offset + len(page) < len(items) else None
    return page, applied_limit


@dataclass(frozen=True)
class _Candidate:
    path: Path
    repo_relative: str
    search_relative: str
    mtime: float


def _matches_path(pattern: str, relative_path: str) -> bool:
    """Match path segments, with ``**`` consuming zero or more segments."""
    pattern_parts = tuple(pattern.split("/"))
    path_parts = tuple(relative_path.split("/"))

    @lru_cache(maxsize=None)
    def matches(pattern_index: int, path_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        part = pattern_parts[pattern_index]
        if part == "**":
            return matches(pattern_index + 1, path_index) or (
                path_index < len(path_parts) and matches(pattern_index, path_index + 1)
            )
        return (
            path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], part)
            and matches(pattern_index + 1, path_index + 1)
        )

    return matches(0, 0)


def _walk_candidates(
    search_root: Path,
    policy: PermissionPolicy,
    cancel: CancellationToken | None,
):
    files_seen = 0
    for directory, dirnames, filenames in os.walk(
        search_root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        dirnames[:] = [
            name
            for name in dirnames
            if not (directory_path / name).is_symlink()
            and policy.check_read(directory_path / name).allowed
        ]
        for filename in filenames:
            if cancel is not None and files_seen % CANCEL_CHECK_INTERVAL_FILES == 0:
                cancel.raise_if_cancelled()
            files_seen += 1
            path = directory_path / filename
            if not policy.check_read(path).allowed:
                continue
            if cancel is not None:
                cancel.raise_if_cancelled()
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0
            yield _Candidate(
                path=path,
                repo_relative=path.relative_to(policy.repo_root).as_posix(),
                search_relative=path.relative_to(search_root).as_posix(),
                mtime=mtime,
            )


def _validate_search_arguments(arguments: dict) -> str | None:
    if not isinstance(arguments.get("pattern"), str):
        return "missing required argument: pattern"
    if "path" in arguments and not isinstance(arguments["path"], str):
        return "path must be a string"
    for name, minimum in (("head_limit", 0), ("offset", 0)):
        value = arguments.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            return f"{name} must be an integer"
        if value is not None and value < minimum:
            return f"{name} must be >= {minimum}"
    return None


def _bounded_content(items: list[str], limit: int | None, offset: int) -> str:
    page, applied_limit = apply_head_limit(items, limit, offset)
    lines = list(page)
    if applied_limit is not None:
        lines.append(
            f"[{len(page)} of {len(items)} results; "
            f"pass offset={offset + len(page)} for the next page]"
        )
    return "\n".join(lines)


class GlobTool(LocalTool):
    """Find readable files whose path matches a glob pattern."""

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return "Find files by a glob pattern inside the allowed scope."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern relative to path; ** crosses directories.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search, defaulting to '.'.",
                    "default": ".",
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Maximum results; 0 returns all results.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of ordered results to skip.",
                    "default": 0,
                },
            },
            "required": ["pattern"],
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        path = arguments.get("path", ".")
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(path,) if isinstance(path, str) else None,
            result_hint=ResultHint.FILE_LIST,
        )

    def validate(self, arguments: dict) -> str | None:
        return _validate_search_arguments(arguments)

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        path = tool_call.arguments.get("path", ".")
        decision = policy.check_read(path)
        if not decision.allowed:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=decision.reason)
        search_root = policy._resolve_within_root(path)
        candidates = [
            candidate
            for candidate in _walk_candidates(search_root, policy, cancel)
            if _matches_path(tool_call.arguments["pattern"], candidate.search_relative)
        ]
        candidates.sort(key=lambda candidate: (-candidate.mtime, candidate.repo_relative))
        if not candidates:
            return ToolResult(tool_call_id=tool_call.id, ok=True, content="no files matched")
        content = _bounded_content(
            [candidate.repo_relative for candidate in candidates],
            tool_call.arguments.get("head_limit"),
            tool_call.arguments.get("offset", 0),
        )
        return ToolResult(tool_call_id=tool_call.id, ok=True, content=content)


class GrepTool(LocalTool):
    """Search readable UTF-8 files with a Python regular expression."""

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return "Search text files with a Python re regular expression."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python re regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search, defaulting to '.'.",
                    "default": ".",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional glob filter relative to path; ** crosses directories.",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Match without regard to case.",
                    "default": False,
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["files_with_matches", "content"],
                    "description": "Return matching files or matching lines.",
                    "default": "files_with_matches",
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Maximum matching files or lines; 0 returns all.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of ordered results to skip.",
                    "default": 0,
                },
            },
            "required": ["pattern"],
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        path = arguments.get("path", ".")
        output_mode = arguments.get("output_mode", "files_with_matches")
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(path,) if isinstance(path, str) else None,
            result_hint=(
                ResultHint.FILE_LIST
                if output_mode == "files_with_matches"
                else ResultHint.TEXT
            ),
        )

    def validate(self, arguments: dict) -> str | None:
        error = _validate_search_arguments(arguments)
        if error is not None:
            return error
        glob_pattern = arguments.get("glob")
        if glob_pattern is not None and not isinstance(glob_pattern, str):
            return "glob must be a string"
        case_insensitive = arguments.get("case_insensitive", False)
        if not isinstance(case_insensitive, bool):
            return "case_insensitive must be a boolean"
        output_mode = arguments.get("output_mode", "files_with_matches")
        if output_mode not in ("files_with_matches", "content"):
            return "output_mode must be 'files_with_matches' or 'content'"
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            re.compile(arguments["pattern"], flags)
        except re.error as exc:
            return f"invalid regular expression: {exc}"
        return None

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        arguments = tool_call.arguments
        path = arguments.get("path", ".")
        decision = policy.check_read(path)
        if not decision.allowed:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=decision.reason)
        flags = re.IGNORECASE if arguments.get("case_insensitive", False) else 0
        expression = re.compile(arguments["pattern"], flags)
        search_root = policy._resolve_within_root(path)
        glob_pattern = arguments.get("glob")
        candidates = [
            candidate
            for candidate in _walk_candidates(search_root, policy, cancel)
            if glob_pattern is None or _matches_path(glob_pattern, candidate.search_relative)
        ]
        candidates.sort(key=lambda candidate: (-candidate.mtime, candidate.repo_relative))

        output_mode = arguments.get("output_mode", "files_with_matches")
        matches: list[str] = []
        for candidate in candidates:
            if cancel is not None:
                cancel.raise_if_cancelled()
            try:
                with candidate.path.open("rb") as handle:
                    data = handle.read(MAX_READ_BYTES + 1)
            except OSError:
                continue
            if len(data) > MAX_READ_BYTES:
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            matching_lines = [
                (lineno, line)
                for lineno, line in enumerate(text.splitlines(), start=1)
                if expression.search(line)
            ]
            if not matching_lines:
                continue
            if output_mode == "files_with_matches":
                matches.append(candidate.repo_relative)
            else:
                matches.extend(
                    f"{candidate.repo_relative}:{lineno}\t{line}"
                    for lineno, line in matching_lines
                )

        if not matches:
            return ToolResult(tool_call_id=tool_call.id, ok=True, content="no matches")
        content = _bounded_content(
            matches, arguments.get("head_limit"), arguments.get("offset", 0)
        )
        return ToolResult(tool_call_id=tool_call.id, ok=True, content=content)
