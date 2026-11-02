"""Stdio and HTTP MCP clients and LocalTool adapter."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from symphonai_api.process import stop_process_group
from symphonai_api.web import valid_service_url
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
_MCP_HTTP_PROTOCOL_VERSION = "2025-06-18"
# Both OpenAI and Anthropic require this function-name shape.
_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SERVER_KEYS = {
    "name",
    "command",
    "url",
    "bearer_token_env_var",
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
    url: str | None = None
    bearer_token_env_var: str | None = None


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
        url = entry.get("url")
        if (command is None) == (url is None):
            raise _config_error(source, key, "exactly one of command or url is required")
        if command is not None and (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) and part for part in command)
        ):
            raise _config_error(
                source,
                f"{key}.command",
                "must be a non-empty array of non-empty strings",
            )
        if url is not None:
            if not isinstance(url, str) or not url:
                raise _config_error(source, f"{key}.url", "must be a non-empty HTTP or HTTPS URL")
            if not valid_service_url(url):
                raise _config_error(source, f"{key}.url", "must use HTTPS or loopback HTTP, without user info")
        bearer_token_env_var = entry.get("bearer_token_env_var")
        if bearer_token_env_var is not None:
            if url is None:
                raise _config_error(source, f"{key}.bearer_token_env_var", "is only valid with url")
            if not isinstance(bearer_token_env_var, str) or not bearer_token_env_var:
                raise _config_error(source, f"{key}.bearer_token_env_var", "must be a non-empty environment variable name")
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
                command=() if command is None else tuple(command),
                enabled=enabled,
                startup_timeout_seconds=startup_timeout,
                call_timeout_seconds=call_timeout,
                source=source,
                url=url,
                bearer_token_env_var=bearer_token_env_var,
            )
        )
    return tuple(specs)




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
        self._http_session_id: str | None = None
        self._http_protocol_version: str | None = None
        self._http_token: str | None = None
        self._http_ready = False
        self.instructions: str | None = None
        self.capabilities: dict = {}
        self.auth_state = "starting" if spec.enabled else "disabled"
        self.oauth = None

    def start(self) -> None:
        if not self.spec.enabled:
            raise McpError(f"MCP server {self.spec.name!r} is disabled")
        if self.spec.url is not None:
            if self._http_ready:
                return
            self.instructions = None
            self._start_http()
            return
        if self._process is not None and self._process.poll() is None:
            return
        self.instructions = None
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
            self._read_capabilities(result)
            self._notify("notifications/initialized", {})
            self.auth_state = "started"
        except McpError as exc:
            self._fail("failed", str(exc))

    def _fail(self, state: str, message: str):
        self.auth_state = state
        self.close()
        raise McpError(message) from None

    def _read_capabilities(self, result):
        if not isinstance(result, Mapping):
            raise McpError(f"MCP server {self.spec.name!r} returned an invalid initialize result")
        self._keep_instructions(result)
        capabilities = result.get("capabilities", {})
        if not isinstance(capabilities, Mapping):
            raise McpError("MCP server returned invalid capabilities")
        self.capabilities = dict(capabilities)

    def _start_http(self) -> None:
        if self.spec.bearer_token_env_var is not None:
            token = os.environ.get(self.spec.bearer_token_env_var)
            if not token:
                raise McpError(
                    f"MCP server {self.spec.name!r} requires environment variable "
                    f"{self.spec.bearer_token_env_var!r}"
                )
            self._http_token = token
        else:
            from symphonai_api.mcp_auth import OAuthSession
            self.oauth = OAuthSession(self.spec.name, self.spec.url)
        try:
            result = self._request(
                "initialize",
                {
                    "protocolVersion": _MCP_HTTP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "symphonai", "version": "1"},
                },
                timeout=self.spec.startup_timeout_seconds,
                phase="startup",
            )
            self._read_capabilities(result)
            version = result.get("protocolVersion")
            if not isinstance(version, str) or not version:
                raise McpError(
                    f"MCP server {self.spec.name!r} returned an invalid protocol version"
                )
            self._http_protocol_version = version
            self._notify("notifications/initialized", {})
            self._http_ready = True
        except McpError as exc:
            self._fail(self.auth_state if self.auth_state == "needs_sign_in" else "failed", str(exc))

    def _keep_instructions(self, result: Mapping) -> None:
        instructions = result.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            self.instructions = instructions[:4000] + (
                "…" if len(instructions) > 4000 else ""
            )

    def _http_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self._http_token is not None:
            headers["Authorization"] = f"Bearer {self._http_token}"
        if self._http_session_id is not None:
            headers["Mcp-Session-Id"] = self._http_session_id
        if self._http_protocol_version is not None:
            headers["MCP-Protocol-Version"] = self._http_protocol_version
        return headers

    def _http_request(
        self,
        method: str,
        params: Mapping[str, object],
        *,
        timeout: float,
        phase: str,
    ) -> object:
        if self.spec.url is None:
            raise McpError(f"MCP server {self.spec.name!r} has no HTTP URL")
        if method != "initialize" and not self._http_ready:
            raise McpError(f"MCP server {self.spec.name!r} is not running")
        with self._request_lock:
            request_id = self._next_id
            self._next_id += 1
            message = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": dict(params),
            }
            from symphonai_api.mcp_auth import AuthError
            oauth = self.oauth
            for attempt in range(2):
                if oauth is not None:
                    try:
                        self._http_token = oauth.access_token(force=attempt == 1)
                    except AuthError:
                        self._fail("needs_sign_in", f"MCP server {self.spec.name!r} needs sign-in from the app")
                request = Request(self.spec.url,
                    data=json.dumps(message, separators=(",", ":")).encode("utf-8"),
                    headers=self._http_headers(), method="POST")
                try:
                    with urlopen(request, timeout=timeout) as response:
                        status = response.status
                        content_type = response.headers.get_content_type()
                        body = response.read()
                        if method == "initialize":
                            session_id = response.headers.get("Mcp-Session-Id")
                            if session_id:
                                self._http_session_id = session_id
                except HTTPError as exc:
                    if exc.code == 401 and oauth is not None:
                        oauth.challenge = exc.headers.get("WWW-Authenticate", "")
                        if attempt == 0 and oauth.record.get("refresh_token"):
                            continue
                        self._fail("needs_sign_in", f"MCP server {self.spec.name!r} needs sign-in from the app")
                    self._fail("failed", f"MCP server {self.spec.name!r} returned HTTP {exc.code} during {phase}")
                except (TimeoutError, URLError, OSError, ValueError):
                    self._fail("failed", f"MCP server {self.spec.name!r} connection failed during {phase}")
                try:
                    result = self._decode_http_response(body, content_type=content_type,
                        request_id=request_id, status=status, phase=phase)
                except McpError as exc:
                    message = (f"MCP server {self.spec.name!r} returned an invalid response"
                               if oauth is not None or self._http_token else str(exc))
                    self._fail("failed", message)
                self.auth_state = "started"
                return result

    def _decode_http_response(
        self,
        body: bytes,
        *,
        content_type: str,
        request_id: int,
        status: int,
        phase: str,
    ) -> object:
        if not 200 <= status < 300:
            raise McpError(
                f"MCP server {self.spec.name!r} returned HTTP {status} during {phase}"
            )
        try:
            text = body.decode("utf-8")
        except UnicodeError:
            raise McpError(
                f"MCP server {self.spec.name!r} returned malformed HTTP {status} during {phase}"
            ) from None

        candidates: list[object] = []
        if content_type == "text/event-stream":
            data_lines: list[str] = []
            for line in (*text.splitlines(), ""):
                if not line:
                    if data_lines:
                        candidates.append("\n".join(data_lines))
                        data_lines = []
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
        else:
            candidates.append(text)

        for candidate in candidates:
            try:
                message = json.loads(candidate)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(message, Mapping) or message.get("jsonrpc") != "2.0":
                continue
            if message.get("id") != request_id:
                if content_type == "text/event-stream":
                    continue
                break
            if "error" in message:
                raise McpError(
                    f"MCP server {self.spec.name!r} returned a JSON-RPC error "
                    f"with HTTP {status} during {phase}"
                )
            if "result" not in message:
                break
            return message["result"]
        raise McpError(
            f"MCP server {self.spec.name!r} returned a malformed response "
            f"with HTTP {status} during {phase}"
        )

    def _http_notification(self, method: str, params: Mapping[str, object]) -> None:
        if self.spec.url is None:
            raise McpError(f"MCP server {self.spec.name!r} has no HTTP URL")
        request = Request(
            self.spec.url,
            data=json.dumps(
                {"jsonrpc": "2.0", "method": method, "params": dict(params)},
                separators=(",", ":"),
            ).encode("utf-8"),
            headers=self._http_headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.spec.startup_timeout_seconds) as response:
                status = response.status
        except HTTPError as exc:
            self.close()
            raise McpError(
                f"MCP server {self.spec.name!r} returned HTTP {exc.code} during {method}"
            ) from None
        except (TimeoutError, URLError, OSError) as exc:
            self.close()
            detail = "timed out" if isinstance(exc, TimeoutError) else "connection failed"
            if isinstance(exc, URLError) and isinstance(exc.reason, TimeoutError):
                detail = "timed out"
            raise McpError(
                f"MCP server {self.spec.name!r} HTTP status unavailable during "
                f"{method}: {detail}"
            ) from None
        if status != 202:
            self.close()
            raise McpError(
                f"MCP server {self.spec.name!r} expected HTTP 202 during {method}, "
                f"received HTTP {status}"
            )

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
        if self.spec.url is not None:
            self._http_notification(method, params)
            return
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
        if self.spec.url is not None:
            return self._http_request(method, params, timeout=timeout, phase=phase)
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

    def list_resources(self, cursor: str | None = None) -> dict:
        return self._list_catalog("resources", cursor)

    def list_prompts(self, cursor: str | None = None) -> dict:
        return self._list_catalog("prompts", cursor)

    def _list_catalog(self, kind: str, cursor: str | None) -> dict:
        if kind not in self.capabilities:
            raise McpError(f"MCP server {self.spec.name!r} does not offer {kind}")
        result = self._request(f"{kind}/list", {} if cursor is None else {"cursor": cursor},
                               timeout=self.spec.call_timeout_seconds, phase=f"{kind}/list")
        if not isinstance(result, Mapping) or not isinstance(result.get(kind), list):
            raise McpError(f"MCP server {self.spec.name!r} returned an invalid {kind} list")
        for entry in result[kind]:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
                raise McpError(f"MCP server {self.spec.name!r} returned an invalid {kind} entry")
            if not isinstance(entry.get("description", ""), str):
                raise McpError("MCP catalog entry has an invalid description")
            if kind == "resources" and not isinstance(entry.get("uri"), str):
                raise McpError("MCP resource has no URI")
            if kind == "prompts":
                arguments = entry.get("arguments", [])
                if not isinstance(arguments, list) or any(not isinstance(argument, Mapping)
                        or not isinstance(argument.get("name"), str)
                        or not isinstance(argument.get("required", False), bool) for argument in arguments):
                    raise McpError("MCP prompt has invalid arguments")
        if result.get("nextCursor") is not None and not isinstance(result["nextCursor"], str):
            raise McpError("MCP catalog has an invalid cursor")
        return dict(result)

    def read_resource(self, uri: str) -> dict:
        if "resources" not in self.capabilities:
            raise McpError("MCP server does not offer resources")
        result = self._request("resources/read", {"uri": uri}, timeout=self.spec.call_timeout_seconds, phase="resources/read")
        if not isinstance(result, Mapping) or not isinstance(result.get("contents"), list):
            raise McpError("MCP server returned invalid resource contents")
        return dict(result)

    def get_prompt(self, name: str, arguments: Mapping[str, str]) -> dict:
        if "prompts" not in self.capabilities:
            raise McpError("MCP server does not offer prompts")
        result = self._request("prompts/get", {"name": name, "arguments": dict(arguments)},
                               timeout=self.spec.call_timeout_seconds, phase="prompts/get")
        if not isinstance(result, Mapping) or not isinstance(result.get("messages"), list):
            raise McpError("MCP server returned invalid prompt messages")
        return dict(result)

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
        if self.spec.url is not None:
            session_id = self._http_session_id
            if session_id is not None:
                try:
                    request = Request(
                        self.spec.url,
                        headers=self._http_headers(),
                        method="DELETE",
                    )
                    with urlopen(request, timeout=_CLEANUP_TIMEOUT_SECONDS):
                        pass
                except Exception:  # noqa: BLE001
                    pass
            self._http_session_id = None
            self._http_protocol_version = None
            self._http_token = None
            self._http_ready = False
            return
        proc = self._process
        pgid = self._process_group
        if proc is None or pgid is None:
            return
        stop_process_group(proc, pgid)
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

    deferrable = True

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
