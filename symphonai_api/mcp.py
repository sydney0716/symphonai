"""Minimal stdio MCP client and LocalTool adapter."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any

from symphonai_api.config import ConfigError, ResolvedConfig, Scope
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_api.trust import TrustList

if TYPE_CHECKING:
    from symphonai_api.cancellation import CancellationToken
    from symphonai_api.permissions import PermissionPolicy


_CLEANUP_TIMEOUT_SECONDS = 0.2
_MCP_PROTOCOL_VERSION = "2024-11-05"
# Both OpenAI and Anthropic require this function-name shape.
_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SERVER_KEYS = {
    "name",
    "command",
    "enabled",
    "startup_timeout_seconds",
    "call_timeout_seconds",
}
_EOF = object()


class McpError(RuntimeError):
    """A server that failed to start, answer, or speak the protocol."""


@dataclass(frozen=True)
class McpServerSpec:
    name: str
    command: tuple[str, ...]
    enabled: bool = False
    startup_timeout_seconds: float = 10.0
    call_timeout_seconds: float = 30.0
    source: Path | None = None


def _config_error(source: Path | None, key: str, detail: str) -> ConfigError:
    location = str(source) if source is not None else "<session>"
    return ConfigError(f"{location}: {key}: {detail}")


def _positive_timeout(
    source: Path | None,
    key: str,
    value: object,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _config_error(source, key, "must be a positive number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise _config_error(source, key, "must be a positive number")
    return timeout


def mcp_servers_from_config(
    config: ResolvedConfig,
    *,
    repo_root: Path | None = None,
    trust: TrustList | None = None,
) -> tuple[McpServerSpec, ...]:
    """Parse the winning ``mcp.servers`` list with its scope provenance."""
    value = config.get("mcp.servers", [])
    origin = config.provenance.get("mcp.servers")
    source = origin.source if origin is not None else None
    trust_root = (
        "<unspecified>"
        if repo_root is None
        else repr(str(Path(repo_root).resolve()))
    )
    if not isinstance(value, list):
        raise _config_error(source, "mcp.servers", "must be an array of tables")

    specs: list[McpServerSpec] = []
    first_index: dict[str, int] = {}
    for index, entry in enumerate(value):
        key = f"mcp.servers[{index}]"
        if not isinstance(entry, Mapping):
            raise _config_error(source, key, "must be a table")
        unknown = entry.keys() - _SERVER_KEYS
        if unknown:
            raise _config_error(
                source,
                f"{key}.{sorted(unknown)[0]}",
                "unknown key",
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise _config_error(source, f"{key}.name", "must be a non-empty identifier")
        if name in first_index:
            raise _config_error(
                source,
                "mcp.servers",
                f"duplicate name {name!r} at indices {first_index[name]} and {index}",
            )
        first_index[name] = index

        command = entry.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) and part for part in command)
        ):
            raise _config_error(
                source,
                f"{key}.command",
                "must be a non-empty array of non-empty strings",
            )
        enabled = entry.get("enabled", McpServerSpec.enabled)
        if type(enabled) is not bool:
            raise _config_error(source, f"{key}.enabled", "must be a boolean")
        if (
            enabled
            and origin is not None
            and origin.scope in (Scope.PROJECT, Scope.PRIVATE)
            and not (
                repo_root is not None
                and trust is not None
                and trust.allows(repo_root, "mcp")
            )
        ):
            raise _config_error(
                source,
                f"{key}.enabled",
                f"repository server {name!r} may be declared here, but only "
                "~/.symphonai/config.toml or session configuration may enable it; "
                f"the machine owner may grant {trust_root} "
                "under [[trust.repositories]]",
            )
        startup_timeout = _positive_timeout(
            source,
            f"{key}.startup_timeout_seconds",
            entry.get("startup_timeout_seconds", 10.0),
        )
        call_timeout = _positive_timeout(
            source,
            f"{key}.call_timeout_seconds",
            entry.get("call_timeout_seconds", 30.0),
        )
        specs.append(
            McpServerSpec(
                name=name,
                command=tuple(command),
                enabled=enabled,
                startup_timeout_seconds=startup_timeout,
                call_timeout_seconds=call_timeout,
                source=source,
            )
        )
    return tuple(specs)


def _terminate_process_group(proc: subprocess.Popen[str], pgid: int) -> None:
    if proc.poll() is None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
        except (OSError, ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=_CLEANUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    try:
        if hasattr(os, "killpg"):
            os.killpg(pgid, signal.SIGKILL)
        elif proc.poll() is None:
            proc.kill()
    except (OSError, ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=_CLEANUP_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass


class McpClient:
    def __init__(
        self,
        spec: McpServerSpec,
        *,
        cwd: Path,
        reserved_names: Collection[str] = (),
    ) -> None:
        self.spec = spec
        self.cwd = Path(cwd)
        self._reserved_names = frozenset(reserved_names)
        self._process: subprocess.Popen[str] | None = None
        self._process_group: int | None = None
        self._reader: threading.Thread | None = None
        self._frames: queue.Queue[object] = queue.Queue()
        self._request_lock = threading.Lock()
        self._next_id = 1

    def start(self) -> None:
        if not self.spec.enabled:
            raise McpError(f"MCP server {self.spec.name!r} is disabled")
        if self._process is not None and self._process.poll() is None:
            return
        try:
            proc = subprocess.Popen(
                self.spec.command,
                shell=False,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            raise McpError(
                f"MCP server {self.spec.name!r} failed to start: "
                f"{type(exc).__name__}: {exc}"
            ) from None
        self._process = proc
        self._process_group = proc.pid
        self._frames = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_frames,
            name=f"mcp-{self.spec.name}-stdout",
            daemon=True,
        )
        self._reader.start()
        try:
            result = self._request(
                "initialize",
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "symphonai", "version": "1"},
                },
                timeout=self.spec.startup_timeout_seconds,
                phase="startup",
            )
            if not isinstance(result, Mapping):
                raise McpError(
                    f"MCP server {self.spec.name!r} returned an invalid initialize result"
                )
            self._notify("notifications/initialized", {})
        except McpError:
            self.close()
            raise

    def _read_frames(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            self._frames.put(_EOF)
            return
        try:
            while True:
                line = proc.stdout.readline()
                if line == "":
                    self._frames.put(_EOF)
                    return
                self._frames.put(line)
        except Exception as exc:  # noqa: BLE001
            self._frames.put(exc)

    def _running_process(self) -> subprocess.Popen[str]:
        proc = self._process
        if proc is None or proc.poll() is not None:
            raise McpError(f"MCP server {self.spec.name!r} is not running")
        return proc

    def _write_message(self, message: Mapping[str, object]) -> None:
        proc = self._running_process()
        if proc.stdin is None:
            raise McpError(f"MCP server {self.spec.name!r} has no stdin")
        try:
            proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            proc.stdin.flush()
        except (OSError, UnicodeError) as exc:
            self.close()
            raise McpError(
                f"MCP server {self.spec.name!r} could not receive a request: "
                f"{type(exc).__name__}: {exc}"
            ) from None

    def _notify(self, method: str, params: Mapping[str, object]) -> None:
        self._write_message(
            {"jsonrpc": "2.0", "method": method, "params": dict(params)}
        )

    def _request(
        self,
        method: str,
        params: Mapping[str, object],
        *,
        timeout: float,
        phase: str,
    ) -> object:
        with self._request_lock:
            request_id = self._next_id
            self._next_id += 1
            self._write_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": dict(params),
                }
            )
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.close()
                    raise McpError(
                        f"MCP server {self.spec.name!r} {phase} timed out "
                        f"after {timeout:g} seconds"
                    )
                try:
                    frame = self._frames.get(timeout=remaining)
                except queue.Empty:
                    self.close()
                    raise McpError(
                        f"MCP server {self.spec.name!r} {phase} timed out "
                        f"after {timeout:g} seconds"
                    ) from None
                if frame is _EOF:
                    self.close()
                    raise McpError(
                        f"MCP server {self.spec.name!r} exited during {phase}"
                    )
                if isinstance(frame, Exception):
                    self.close()
                    raise McpError(
                        f"MCP server {self.spec.name!r} sent unreadable data: "
                        f"{type(frame).__name__}: {frame}"
                    ) from None
                try:
                    message = json.loads(frame)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    self.close()
                    raise McpError(
                        f"MCP server {self.spec.name!r} sent malformed JSON-RPC: "
                        f"{type(exc).__name__}: {exc}"
                    ) from None
                if not isinstance(message, Mapping) or message.get("jsonrpc") != "2.0":
                    self.close()
                    raise McpError(
                        f"MCP server {self.spec.name!r} sent a malformed JSON-RPC frame"
                    )
                if "id" not in message:
                    if isinstance(message.get("method"), str):
                        continue
                    self.close()
                    raise McpError(
                        f"MCP server {self.spec.name!r} sent a response without an id"
                    )
                if message["id"] != request_id:
                    unknown_id = message["id"]
                    self.close()
                    raise McpError(
                        f"MCP server {self.spec.name!r} sent unknown response id "
                        f"{unknown_id!r}"
                    )
                if "error" in message:
                    self.close()
                    raise McpError(
                        f"MCP server {self.spec.name!r} returned an error for "
                        f"{method}: {message['error']!r}"
                    )
                if "result" not in message:
                    self.close()
                    raise McpError(
                        f"MCP server {self.spec.name!r} response omitted result"
                    )
                return message["result"]

    def list_tools(self) -> tuple["McpTool", ...]:
        result = self._request(
            "tools/list",
            {},
            timeout=self.spec.call_timeout_seconds,
            phase="tools/list",
        )
        if not isinstance(result, Mapping) or not isinstance(result.get("tools"), list):
            raise McpError(
                f"MCP server {self.spec.name!r} returned an invalid tools/list result"
            )
        tools: list[McpTool] = []
        seen: set[str] = set()
        for index, entry in enumerate(result["tools"]):
            if not isinstance(entry, Mapping):
                raise McpError(
                    f"MCP server {self.spec.name!r} tool {index} is not an object"
                )
            tool_name = entry.get("name")
            description = entry.get("description", "")
            parameters = entry.get("inputSchema")
            if not isinstance(tool_name, str) or not tool_name:
                raise McpError(
                    f"MCP server {self.spec.name!r} tool {index} has an invalid name"
                )
            if not isinstance(description, str) or not isinstance(parameters, Mapping):
                raise McpError(
                    f"MCP server {self.spec.name!r} tool {tool_name!r} is malformed"
                )
            adapted = McpTool(
                self,
                server_tool_name=tool_name,
                description=description,
                parameters=dict(parameters),
            )
            if _TOOL_NAME_PATTERN.fullmatch(adapted.name) is None:
                raise McpError(
                    f"MCP server {self.spec.name!r} tool {tool_name!r} composes "
                    f"unusable name {adapted.name!r}; expected "
                    "^[a-zA-Z0-9_-]{1,64}$"
                )
            if adapted.name in self._reserved_names:
                raise McpError(
                    f"MCP server {self.spec.name!r} tool {tool_name!r} collides "
                    f"with reserved name {adapted.name!r}"
                )
            if adapted.name in seen:
                raise McpError(
                    f"MCP server {self.spec.name!r} repeats tool {tool_name!r}"
                )
            seen.add(adapted.name)
            tools.append(adapted)
        return tuple(tools)

    def call(self, tool: str, arguments: Mapping[str, object]) -> str:
        result = self._request(
            "tools/call",
            {"name": tool, "arguments": dict(arguments)},
            timeout=self.spec.call_timeout_seconds,
            phase=f"tool {tool!r} call",
        )
        if not isinstance(result, Mapping):
            raise McpError(
                f"MCP server {self.spec.name!r} returned an invalid tools/call result"
            )
        if result.get("isError") is True:
            raise McpError(
                f"MCP server {self.spec.name!r} tool {tool!r} reported an error: "
                f"{_render_content(result.get('content'))}"
            )
        return _render_content(result.get("content"))

    def close(self) -> None:
        proc = self._process
        pgid = self._process_group
        if proc is None or pgid is None:
            return
        _terminate_process_group(proc, pgid)
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=_CLEANUP_TIMEOUT_SECONDS)
        self._process = None
        self._process_group = None

    def __enter__(self) -> "McpClient":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _render_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    rendered: list[str] = []
    for item in content:
        if isinstance(item, Mapping) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                rendered.append(text)
                continue
        rendered.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return "\n".join(rendered)


class McpTool(LocalTool):
    """One server-side tool, presented as a LocalTool."""

    def __init__(
        self,
        client: McpClient,
        *,
        server_tool_name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self._client = client
        self._server_tool_name = server_tool_name
        self._description = description
        self._parameters = parameters

    @property
    def name(self) -> str:
        return f"mcp__{self._client.spec.name}__{self._server_tool_name}"

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._parameters.copy()

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
        try:
            decision = policy.check_opaque_tool(
                self.name,
                target=self._client.spec.name,
                details=f"call MCP tool {self._server_tool_name!r}",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"MCP approval failed: {type(exc).__name__}: {exc}",
            )
        if not decision.allowed:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=decision.reason or "MCP tool call denied",
            )
        try:
            content = self._client.call(self._server_tool_name, tool_call.arguments)
        except McpError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        return ToolResult(tool_call_id=tool_call.id, ok=True, content=content)
