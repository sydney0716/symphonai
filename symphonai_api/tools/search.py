"""Read-only file search tools with bounded, permission-gated results."""

from __future__ import annotations

from symphonai_api.git import git as _git, WorktreeError

import fnmatch
from collections import deque
import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from symphonai_api.cancellation import CancellationToken
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.filesystem import MAX_READ_BYTES
from symphonai_api.tools.metadata import ToolMetadata

DEFAULT_HEAD_LIMIT = 250
CANCEL_CHECK_INTERVAL_FILES = 64
DEFAULT_SKIP_DIRECTORIES = frozenset({
    ".venv", "__pycache__", "node_modules", "dist", "build",
})


def _default_skipped_directory(name: str) -> bool:
    return name in DEFAULT_SKIP_DIRECTORIES or fnmatch.fnmatchcase(name, "*.egg-info")


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
    is_directory: bool = False


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


def _git_ignored_paths(search_root: Path, policy: PermissionPolicy) -> set[str]:
    try:
        result = _git(policy.repo_root, 'ls-files', '--others', '--ignored', '--exclude-standard', '--directory', '-z', stdin=subprocess.DEVNULL, capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired, WorktreeError):
        return set()
    if result.returncode != 0:
        return set()
    ignored = {os.fsdecode(path).rstrip("/") for path in result.stdout.split(b"\0") if path}
    if search_root != policy.repo_root:
        relative_root = search_root.relative_to(policy.repo_root).as_posix()
        try:
            explicitly_named = _git(policy.repo_root, 'check-ignore', '-q', '--no-index', relative_root, stdin=subprocess.DEVNULL, capture_output=True, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired, WorktreeError):
            return ignored
        if explicitly_named.returncode == 0:
            return set()
    return ignored


def _is_git_ignored(relative_path: str, ignored: set[str]) -> bool:
    parts = relative_path.split("/")
    return any("/".join(parts[:index]) in ignored for index in range(1, len(parts) + 1))


def _walk_candidates(
    search_root: Path, policy: PermissionPolicy, cancel: CancellationToken | None = None,
    *, include_skipped: bool = False, breadth_first: bool = False,
    include_directories: bool = False, include_gitignored: bool = False,
):
    ignored = set() if include_gitignored else _git_ignored_paths(search_root, policy)
    pending = deque([search_root])
    while pending:
        directory = pending.popleft() if breadth_first else pending.pop()
        try:
            entries = [directory] if directory.is_file() else sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        children = []
        for path in entries:
            if cancel is not None:
                cancel.raise_if_cancelled()
            relative = path.relative_to(policy.repo_root).as_posix()
            if _is_git_ignored(relative, ignored) or not policy.check_read(path).allowed:
                continue
            is_directory = path.is_dir()
            if is_directory:
                if path.is_symlink() or (not include_skipped and _default_skipped_directory(path.name)):
                    continue
                children.append(path)
                if not include_directories:
                    continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0
            yield _Candidate(path, relative, path.name if path == search_root else path.relative_to(search_root).as_posix(),
                             mtime, is_directory)
        pending.extend(children if breadth_first else reversed(children))


def _bounded_content(items: list[str], limit: int | None, offset: int) -> str:
    page, applied_limit = apply_head_limit(items, limit, offset)
    if not page:
        return f"[no results at offset {offset}; {len(items)} results total]"
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
                    "minimum": 0,
                    "type": "integer",
                    "description": "Maximum results; 0 returns all results.",
                },
                "offset": {
                    "minimum": 0,
                    "type": "integer",
                    "description": "Number of ordered results to skip.",
                    "default": 0,
                },
            },
            "required": ["pattern"],
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(concurrency_safe=True)

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
        if not search_root.exists():
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=f"no such file or directory: {path}")
        explicit = (
            any(_default_skipped_directory(part) for part in search_root.relative_to(policy.repo_root).parts)
            or any(_default_skipped_directory(part) for part in tool_call.arguments["pattern"].split("/"))
        )
        candidates = [
            candidate
            for candidate in _walk_candidates(
                search_root, policy, cancel, include_skipped=explicit,
            )
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
                    "minimum": 0,
                    "type": "integer",
                    "description": "Maximum matching files or lines; 0 returns all.",
                },
                "offset": {
                    "minimum": 0,
                    "type": "integer",
                    "description": "Number of ordered results to skip.",
                    "default": 0,
                },
            },
            "required": ["pattern"],
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(concurrency_safe=True)

    def validate(self, arguments: dict) -> str | None:
        flags = re.IGNORECASE if arguments.get("case_insensitive", False) else 0
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
        if not search_root.exists():
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=f"no such file or directory: {path}")
        glob_pattern = arguments.get("glob")
        explicit = any(
            _default_skipped_directory(part)
            for part in search_root.relative_to(policy.repo_root).parts
        )
        candidates = [
            candidate
            for candidate in _walk_candidates(
                search_root, policy, cancel, include_skipped=explicit,
            )
            if glob_pattern is None or _matches_path(glob_pattern, candidate.search_relative)
        ]
        candidates.sort(key=lambda candidate: (-candidate.mtime, candidate.repo_relative))

        output_mode = arguments.get("output_mode", "files_with_matches")
        matches: list[str] = []
        skipped_large = 0
        for candidate in candidates:
            if cancel is not None:
                cancel.raise_if_cancelled()
            try:
                with candidate.path.open("rb") as handle:
                    data = handle.read(MAX_READ_BYTES + 1)
            except OSError:
                continue
            if len(data) > MAX_READ_BYTES:
                skipped_large += 1
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            matching_lines = [
                (lineno, line)
                for lineno, line in enumerate(text.removesuffix("\n").split("\n") if text else [], start=1)
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

        content = (_bounded_content(matches, arguments.get("head_limit"), arguments.get("offset", 0))
                   if matches else "no matches")
        if skipped_large:
            content += f"\n[skipped {skipped_large} files over {MAX_READ_BYTES} bytes]"
        return ToolResult(tool_call_id=tool_call.id, ok=True, content=content)
