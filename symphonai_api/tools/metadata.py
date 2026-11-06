"""Per-call concurrency declarations and tool target descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from symphonai_api.cancellation import OperationCancelled
from symphonai_api.tools.shell_parse import parse_command_line

if TYPE_CHECKING:
    from symphonai_api.tools.base import LocalTool


FILE_EDIT_TOOLS = frozenset({"write_file", "edit_file", "multi_edit_file", "apply_patch"})


def edit_tool_names(names, wire_format: int | None = None) -> set[str]:
    """An editing capability selects the model's edit family, or all names for rosters."""
    names = set(names)
    edits = FILE_EDIT_TOOLS - {"write_file"}
    if names & edits:
        family = edits if wire_format is None else {"apply_patch"} if wire_format == 1 else {"edit_file", "multi_edit_file"}
        names = (names - edits) | family
    return names


def patch_paths(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    prefixes = ("*** Add File: ", "*** Update File: ", "*** Delete File: ", "*** Move to: ")
    return list(dict.fromkeys(line[len(prefix):] for line in value.splitlines()
                             for prefix in prefixes if line.startswith(prefix)))


def changed_paths(tool_name: str, arguments: dict) -> list[str]:
    if tool_name == "apply_patch":
        return patch_paths(arguments.get("input"))
    path = arguments.get("path")
    return [path] if tool_name in FILE_EDIT_TOOLS and isinstance(path, str) else []


TARGET_LIMIT = 200


def _plain_target(value: object) -> str:
    return value if isinstance(value, str) else ""


def _argv_program(argv: list) -> str:
    if not isinstance(argv, list) or not argv:
        return ""
    program = argv[0]
    return program if isinstance(program, str) and program else ""


def _url_origin(url: object) -> str:
    if not isinstance(url, str):
        return ""
    try:
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.hostname:
            return ""
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return ""
    if ":" in host:
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"
    return f"{parsed.scheme}://{host}"


_TARGET_KEYS = {
    "read_file": ("path", _plain_target),
    "list_files": ("path", _plain_target),
    "glob": ("pattern", _plain_target),
    "grep": ("pattern", _plain_target),
    "web_search": ("query", _plain_target),
    "use_skill": ("name", _plain_target),
    "dispatch_subagent": ("subagent_name", _plain_target),
    "run_shell": ("argv", _argv_program),
    "web_fetch": ("url", _url_origin),
    "remember": ("text", _plain_target),
    "forget": ("text", _plain_target),
}


def call_target(tool_name: str, arguments: dict) -> str:
    """Return a bounded display target derived safely for one known tool."""
    if not isinstance(tool_name, str) or not isinstance(arguments, dict):
        return ""
    if tool_name in FILE_EDIT_TOOLS:
        return next(iter(changed_paths(tool_name, arguments)), "")[:TARGET_LIMIT]
    if tool_name == "run_shell" and "command" in arguments:
        commands = parse_command_line(arguments["command"])
        if commands is None or not commands[0].argv:
            return ""
        shell_keywords = {
            "if", "then", "elif", "else", "do", "while", "until", "!",
            "for", "case", "select",
        }
        first = commands[0].argv[0]
        target = next(
            (
                command.argv[0]
                for command in commands
                if command.argv[0] not in shell_keywords and command.argv[0] != ":"
            ),
            first,
        )
        return target[:TARGET_LIMIT]
    target_rule = _TARGET_KEYS.get(tool_name)
    if target_rule is None:
        return ""
    key, reduce_target = target_rule
    value = arguments.get(key)
    if not value:
        return ""
    target = reduce_target(value)
    return target[:TARGET_LIMIT]


@dataclass(frozen=True)
class ToolMetadata:
    """What one specific call does. Declared per call, not per tool."""

    concurrency_safe: bool
    """Safe to execute in parallel with other concurrency-safe calls in one turn.

    A wrong True can cause a data race; when uncertain, use False.
    """


FAIL_CLOSED = ToolMetadata(concurrency_safe=False)
"""What a caller must assume when a call's metadata cannot be determined."""


def safe_metadata(tool: "LocalTool", arguments: dict) -> ToolMetadata:
    """`tool.metadata(arguments)`, or `FAIL_CLOSED` if it raises.

    `metadata` receives raw model-supplied arguments and may parse them, so it
    can raise on input no one anticipated. Every caller that decides something
    consequential -- phase 03's scheduler above all -- must go through this,
    never through `tool.metadata` directly.

    `OperationCancelled` is re-raised, never converted. Swallowing it would
    make a cancelled turn look like an ordinary unsafe call, which is the same
    rule `symphonai_api.events.py:emit` follows.
    """
    try:
        return tool.metadata(arguments)
    except OperationCancelled:
        raise
    except Exception:
        return FAIL_CLOSED
