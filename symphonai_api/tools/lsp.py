"""Read-only navigation requests through a configured language server."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlsplit

from symphonai_api.cancellation import CancellationToken
from symphonai_api.lsp import LspError, LspManager
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata

_OPERATIONS = {"definition", "references", "hover", "document_symbols", "workspace_symbols"}
_KINDS = (
    "", "File", "Module", "Namespace", "Package", "Class", "Method", "Property",
    "Field", "Constructor", "Enum", "Interface", "Function", "Variable", "Constant",
    "String", "Number", "Boolean", "Array", "Object", "Key", "Null", "EnumMember",
    "Struct", "Event", "Operator", "TypeParameter",
)


class LspTool(LocalTool):
    def __init__(self, manager: LspManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "lsp"

    @property
    def description(self) -> str:
        return "Navigate source code with its configured language server. Read-only."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": sorted(_OPERATIONS)},
                "path": {"type": "string"},
                "line": {"type": "integer", "minimum": 1},
                "column": {"type": "integer", "minimum": 1},
                "query": {"type": "string"},
            },
            "required": ["operation"],
        }

    def validate(self, arguments: dict) -> str | None:
        operation = arguments.get("operation")
        if operation not in _OPERATIONS:
            return "operation must be definition, references, hover, document_symbols, or workspace_symbols"
        if operation == "workspace_symbols":
            if not isinstance(arguments.get("query"), str):
                return "query must be a string for workspace_symbols"
            return None
        if not isinstance(arguments.get("path"), str) or not arguments["path"]:
            return "path must be a non-empty string"
        if operation in ("definition", "references", "hover"):
            for key in ("line", "column"):
                if type(arguments.get(key)) is not int or arguments[key] < 1:
                    return f"{key} must be a positive 1-based integer"
        return None

    def metadata(self, arguments: dict) -> ToolMetadata:
        operation = arguments.get("operation")
        path = arguments.get("path")
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(path,) if operation != "workspace_symbols" and isinstance(path, str) else (),
        )

    def _execute(self, tool_call: ToolCall, policy: PermissionPolicy, cancel: CancellationToken | None = None) -> ToolResult:
        args = tool_call.arguments
        operation = args["operation"]
        try:
            if operation == "workspace_symbols":
                clients = self._manager.clients
                if not clients:
                    return ToolResult(tool_call_id=tool_call.id, ok=False, error="no language server is running")
                results = []
                for client in clients:
                    results.extend(_as_list(client.request("workspace/symbol", {"query": args["query"]})))
                return _result(tool_call.id, _format_symbols(results, policy))

            path = Path(args["path"])
            decision = policy.check_read(path)
            if not decision.allowed:
                return ToolResult(tool_call_id=tool_call.id, ok=False, error=decision.reason)
            resolved = path if path.is_absolute() else policy.repo_root / path
            resolved = resolved.resolve()
            client = self._manager.client_for(resolved)
            if client is None:
                return ToolResult(
                    tool_call_id=tool_call.id, ok=False,
                    error=f"no language server handles {resolved.suffix or '<no suffix>'}",
                )
            if operation == "document_symbols":
                client.sync(resolved)
                response = client.request("textDocument/documentSymbol", {"textDocument": {"uri": resolved.as_uri()}})
                return _result(tool_call.id, _format_symbols(_as_list(response), policy))
            line, character = _position(resolved, args["line"], args["column"])
            client.sync(resolved)
            position = {"textDocument": {"uri": resolved.as_uri()}, "position": {"line": line, "character": character}}
            if operation == "definition":
                response = client.request("textDocument/definition", position)
                return _result(tool_call.id, _format_locations(_as_list(response), policy))
            if operation == "references":
                response = client.request("textDocument/references", {**position, "context": {"includeDeclaration": True}})
                return _result(tool_call.id, _format_locations(_as_list(response), policy))
            response = client.request("textDocument/hover", position)
            contents = response.get("contents") if isinstance(response, dict) else None
            return _result(tool_call.id, _hover_text(contents) if contents else "No results.")
        except LspError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))


def _result(call_id: str, text: str) -> ToolResult:
    return ToolResult(tool_call_id=call_id, ok=True, content=text)


def _as_list(value: object) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _position(path: Path, line: int, column: int) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if line > len(lines) or column > len(lines[line - 1]) + 1:
        raise ValueError("line or column is outside the file")
    return line - 1, len(lines[line - 1][:column - 1].encode("utf-16-le")) // 2


def _path_from_uri(uri: str) -> Path:
    parts = urlsplit(uri)
    if parts.scheme != "file":
        return Path(unquote(uri))
    return Path(unquote(parts.path))


def _point(location: dict) -> tuple[Path, int, int] | None:
    uri = location.get("uri") or location.get("targetUri")
    range_data = location.get("range") or location.get("targetSelectionRange") or location.get("targetRange")
    start = range_data.get("start") if isinstance(range_data, dict) else None
    if not isinstance(uri, str) or not isinstance(start, dict):
        return None
    return _path_from_uri(uri), int(start.get("line", 0)), int(start.get("character", 0))


def _utf16_to_column(path: Path, line: int, character: int) -> int:
    try:
        text = path.read_text(encoding="utf-8").splitlines()[line]
    except (OSError, UnicodeError, IndexError):
        return character + 1
    units = 0
    for index, char in enumerate(text):
        if units >= character:
            return index + 1
        units += len(char.encode("utf-16-le")) // 2
    return len(text) + 1


def _display_path(path: Path, policy: PermissionPolicy) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(policy.repo_root):
        return resolved.relative_to(policy.repo_root).as_posix()
    return str(resolved)


def _format_locations(locations: list, policy: PermissionPolicy) -> str:
    output = []
    outside = 0
    for location in locations:
        if not isinstance(location, dict):
            continue
        point = _point(location)
        if point is None:
            continue
        path, line, character = point
        decision = policy.check_read(path)
        if not decision.allowed:
            outside += 1
            continue
        path = Path(path).resolve()
        output.append(f"{_display_path(path, policy)}:{line + 1}:{_utf16_to_column(path, line, character)}")
    return _cap(output, outside)


def _symbol_lines(symbols: list, policy: PermissionPolicy, depth: int = 0) -> tuple[list[str], int]:
    output = []
    outside = 0
    for symbol in symbols:
        if not isinstance(symbol, dict):
            continue
        name = symbol.get("name", "")
        kind = symbol.get("kind", 0)
        kind_name = _KINDS[kind] if isinstance(kind, int) and 0 <= kind < len(_KINDS) else "Symbol"
        location = symbol.get("location")
        line = None
        if isinstance(location, dict):
            point = _point(location)
            if point is not None:
                path, line0, _ = point
                decision = policy.check_read(path)
                if not decision.allowed:
                    outside += 1
                    continue
                path = Path(path).resolve()
                line = line0 + 1
        elif isinstance(symbol.get("range"), dict):
            line = int(symbol["range"].get("start", {}).get("line", 0)) + 1
        output.append("  " * depth + f"{kind_name} {name}" + (f" — line {line}" if line is not None else ""))
        children, hidden = _symbol_lines(symbol.get("children", []), policy, depth + 1)
        output.extend(children)
        outside += hidden
    return output, outside


def _format_symbols(symbols: list, policy: PermissionPolicy) -> str:
    lines, outside = _symbol_lines(symbols, policy)
    return _cap(lines, outside)


def _hover_text(contents: object) -> str:
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        return str(contents.get("value", ""))
    if isinstance(contents, list):
        return "\n".join(_hover_text(item) for item in contents)
    return str(contents)


def _cap(items: list[str], outside: int = 0) -> str:
    shown = items[:200]
    if not shown:
        shown.append("No results.")
    if len(items) > 200:
        shown.append(f"… {len(items) - 200} more")
    if outside:
        shown.append(f"({outside} outside the readable repository)")
    return "\n".join(shown)
