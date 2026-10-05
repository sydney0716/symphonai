"""Standard-library Language Server Protocol client over framed stdio.

For example, configure pyright with command = ["pyright-langserver", "--stdio"]
for language_id = "python" and extensions = [".py", ".pyi"].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from typing import Any

from symphonai_api.config import ConfigError, ResolvedConfig, Scope


class LspError(RuntimeError):
    """A language server failed to start, answer, or speak LSP."""


@dataclass(frozen=True)
class LspServerSpec:
    name: str
    command: tuple[str, ...]
    language_id: str
    extensions: tuple[str, ...]
    enabled: bool = False
    startup_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 10.0
    source: Path | None = None


def _config_error(source: Path | None, key: str, detail: str) -> ConfigError:
    return ConfigError(f"{source if source is not None else '<session>'}: {key}: {detail}")


def lsp_servers_from_config(
    config: ResolvedConfig,
    *,
    repo_root: Path | None = None,
    trust=None,
) -> tuple[LspServerSpec, ...]:
    value = config.get("lsp.servers", [])
    origin = config.provenance.get("lsp.servers")
    source = origin.source if origin is not None else None
    if not isinstance(value, list):
        raise _config_error(source, "lsp.servers", "must be an array of tables")
    specs = []
    names = set()
    suffixes = set()
    for index, item in enumerate(value):
        key = f"lsp.servers[{index}]"
        if not isinstance(item, Mapping):
            raise _config_error(source, key, "must be a table")
        unknown = item.keys() - {
            "name", "command", "language_id", "extensions", "enabled",
            "startup_timeout_seconds", "request_timeout_seconds",
        }
        if unknown:
            raise _config_error(source, f"{key}.{sorted(unknown)[0]}", "unknown key")
        name = item.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            raise _config_error(source, f"{key}.name", "must be a non-empty identifier")
        if name in names:
            raise _config_error(source, "lsp.servers", f"duplicate name {name!r}")
        names.add(name)
        command = item.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
            raise _config_error(source, f"{key}.command", "must be a non-empty array of non-empty strings")
        language_id = item.get("language_id")
        if not isinstance(language_id, str) or not language_id:
            raise _config_error(source, f"{key}.language_id", "must be a non-empty string")
        extensions = item.get("extensions")
        if not isinstance(extensions, list) or not extensions or not all(
            isinstance(x, str) and x.startswith(".") and len(x) > 1 for x in extensions
        ):
            raise _config_error(source, f"{key}.extensions", "must be a non-empty array of file suffixes")
        enabled = item.get("enabled", False)
        if type(enabled) is not bool:
            raise _config_error(source, f"{key}.enabled", "must be a boolean")
        if enabled and origin is not None and origin.scope in (Scope.PROJECT, Scope.PRIVATE) and not (
            repo_root is not None and trust is not None and trust.allows(repo_root, "lsp")
        ):
            raise _config_error(source, f"{key}.enabled", "repository LSP servers require an lsp trust grant")
        timeouts = []
        for field, default in (("startup_timeout_seconds", 30.0), ("request_timeout_seconds", 10.0)):
            number = item.get(field, default)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
                raise _config_error(source, f"{key}.{field}", "must be a positive number")
            timeouts.append(float(number))
        overlap = set(extensions) & suffixes if enabled else set()
        if overlap:
            raise _config_error(source, f"{key}.extensions", f"enabled servers both claim {sorted(overlap)[0]}")
        if enabled:
            suffixes.update(extensions)
        specs.append(LspServerSpec(name, tuple(command), language_id, tuple(extensions), enabled, *timeouts, source))
    return tuple(specs)


def _frame(message: dict[str, Any]) -> bytes:
    body = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


class LspClient:
    def __init__(self, spec: LspServerSpec, *, root: Path) -> None:
        self.spec = spec
        self.root = Path(root).resolve()
        try:
            self.process = subprocess.Popen(
                spec.command, cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, start_new_session=True,
            )
        except OSError as exc:
            raise LspError(f"could not start language server {spec.name}: {exc}") from None
        self._write_lock = threading.Lock()
        self._state = threading.Condition()
        self._next_id = 0
        self._pending: dict[int, tuple[threading.Event, dict[str, Any]]] = {}
        self._diagnostics: dict[str, tuple[int, list[dict]]] = {}
        self._documents: dict[str, tuple[int, str]] = {}
        self._document_lock = threading.Lock()
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, name=f"lsp-{spec.name}", daemon=True)
        self._reader.start()
        try:
            root_uri = self.root.as_uri()
            self.request("initialize", {
                "processId": os.getpid(), "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": self.root.name}],
                "capabilities": {"textDocument": {
                    "publishDiagnostics": {"relatedInformation": True},
                    "definition": {}, "references": {}, "hover": {}, "documentSymbol": {},
                    "synchronization": {"dynamicRegistration": False, "willSave": False, "didSave": False},
                }, "workspace": {"symbol": {}, "workspaceFolders": True}},
            }, timeout=spec.startup_timeout_seconds)
            self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        except Exception:
            self.close()
            raise

    def _send(self, message: dict[str, Any]) -> None:
        if self.process.poll() is not None:
            raise LspError("language server exited")
        data = _frame(message)
        try:
            with self._write_lock:
                self.process.stdin.write(data)
                self.process.stdin.flush()
        except (OSError, BrokenPipeError) as exc:
            raise LspError(f"language server write failed: {exc}") from None

    def _read_message(self) -> dict[str, Any] | None:
        headers = {}
        while True:
            line = self.process.stdout.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            key, separator, value = line.partition(b":")
            if not separator:
                raise LspError("malformed LSP header")
            headers[key.strip().lower()] = value.strip()
        try:
            length = int(headers[b"content-length"])
            body = self.process.stdout.read(length)
            if len(body) != length:
                raise LspError("truncated LSP message")
            result = json.loads(body.decode("utf-8"))
        except (KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LspError(f"malformed LSP message: {exc}") from None
        if not isinstance(result, dict):
            raise LspError("LSP message must be an object")
        return result

    def _reply_to_server(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        params = message.get("params") or {}
        if method == "workspace/configuration":
            result = [None] * len(params.get("items", []))
            response = {"jsonrpc": "2.0", "id": message.get("id"), "result": result}
        elif method in ("client/registerCapability", "client/unregisterCapability", "window/workDoneProgress/create"):
            response = {"jsonrpc": "2.0", "id": message.get("id"), "result": None}
        else:
            response = {"jsonrpc": "2.0", "id": message.get("id"), "error": {"code": -32601, "message": "Method not found"}}
        self._send(response)

    def _read_loop(self) -> None:
        try:
            while not self._closed:
                message = self._read_message()
                if message is None:
                    break
                if "method" in message:
                    if message.get("method") == "textDocument/publishDiagnostics":
                        params = message.get("params") or {}
                        uri = params.get("uri")
                        if isinstance(uri, str):
                            with self._state:
                                sequence, _ = self._diagnostics.get(uri, (0, []))
                                self._diagnostics[uri] = (sequence + 1, params.get("diagnostics", []))
                                self._state.notify_all()
                    elif "id" in message:
                        self._reply_to_server(message)
                    continue
                message_id = message.get("id")
                with self._state:
                    pending = self._pending.get(message_id)
                    if pending is not None:
                        pending[1].update(message)
                        pending[0].set()
        except Exception:
            pass
        finally:
            with self._state:
                for event, result in self._pending.values():
                    result["_exit"] = True
                    event.set()
                self._state.notify_all()

    def request(self, method: str, params: dict, *, timeout: float | None = None) -> object:
        with self._state:
            self._next_id += 1
            request_id = self._next_id
            event = threading.Event()
            result: dict[str, Any] = {}
            self._pending[request_id] = (event, result)
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        if not event.wait(self.spec.request_timeout_seconds if timeout is None else timeout):
            with self._state:
                self._pending.pop(request_id, None)
            raise LspError(f"language server request timed out: {method}")
        with self._state:
            self._pending.pop(request_id, None)
        if result.get("_exit"):
            raise LspError("language server exited")
        if "error" in result:
            raise LspError(f"language server returned error: {result['error']!r}")
        return result.get("result")

    def sync(self, path: Path) -> int:
        with self._document_lock:
            return self._sync(path)

    def _sync(self, path: Path) -> int:
        path = Path(path).resolve()
        uri = path.as_uri()
        content = path.read_text(encoding="utf-8")
        with self._state:
            sequence = self._diagnostics.get(uri, (0, []))[0]
        previous = self._documents.get(uri)
        if previous is None:
            version = 1
            self._send({"jsonrpc": "2.0", "method": "textDocument/didOpen", "params": {
                "textDocument": {"uri": uri, "languageId": self.spec.language_id, "version": version, "text": content},
            }})
        elif previous[1] != content:
            version = previous[0] + 1
            self._send({"jsonrpc": "2.0", "method": "textDocument/didChange", "params": {
                "textDocument": {"uri": uri, "version": version}, "contentChanges": [{"text": content}],
            }})
        else:
            return sequence
        self._documents[uri] = (version, content)
        return sequence

    def wait_diagnostics(self, path: Path, after: int, timeout: float) -> list[dict] | None:
        uri = Path(path).resolve().as_uri()
        deadline = time.monotonic() + timeout
        with self._state:
            while True:
                sequence, diagnostics = self._diagnostics.get(uri, (0, []))
                if sequence > after:
                    return diagnostics
                if self.process.poll() is not None:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._state.wait(remaining)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            try:
                self.request("shutdown", {}, timeout=0.2)
            except LspError:
                pass
            try:
                self._send({"jsonrpc": "2.0", "method": "exit", "params": {}})
            except LspError:
                pass
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=0.2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except OSError:
                    self.process.kill()
        if threading.current_thread() is not self._reader:
            self._reader.join(timeout=0.5)


class LspManager:
    def __init__(self, specs: Sequence[LspServerSpec], *, root: Path) -> None:
        self.specs = tuple(specs)
        self.root = Path(root).resolve()
        self._lock = threading.Lock()
        self._clients: dict[str, LspClient] = {}
        self._failed: set[str] = set()
        self._closed = False

    @property
    def has_enabled_servers(self) -> bool:
        return any(spec.enabled for spec in self.specs)

    def client_for(self, path: Path) -> LspClient | None:
        suffix = Path(path).suffix.casefold()
        spec = next((item for item in self.specs if item.enabled and suffix in {x.casefold() for x in item.extensions}), None)
        if spec is None:
            return None
        with self._lock:
            if self._closed or suffix in self._failed:
                return None
            client = self._clients.get(spec.name)
            if client is not None:
                return client
            try:
                client = LspClient(spec, root=self.root)
            except LspError:
                self._failed.update(ext.casefold() for ext in spec.extensions)
                raise
            self._clients[spec.name] = client
            return client

    @property
    def clients(self) -> tuple[LspClient, ...]:
        with self._lock:
            return tuple(self._clients.values())

    def close(self) -> None:
        with self._lock:
            self._closed = True
            clients = tuple(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.close()
