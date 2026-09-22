"""Loopback-only HTTP/SSE boundary for a single SymphonAI host run."""

from __future__ import annotations

import json
import os
import secrets
import shlex
import threading
from collections.abc import Mapping
from http.cookies import CookieError, SimpleCookie
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from symphonai_api.compaction import DEFAULT_CONTEXT_TOKEN_BUDGET, DEFAULT_RECENT_TURNS
from symphonai_api.cost import PriceTable
from symphonai_api.extensions import Extensions
from symphonai_api.permissions import PermissionPolicy, _contains_path
from symphonai_api.providers.base import ModelProvider
from symphonai_api.providers.anthropic_provider import API_KEY_ENV_VAR as ANTHROPIC_KEY_ENV_VAR, AnthropicProvider
from symphonai_api.providers.gemini_provider import API_KEY_ENV_VAR as GEMINI_KEY_ENV_VAR, GeminiProvider
from symphonai_api.providers.openai_provider import API_KEY_ENV_VAR as OPENAI_KEY_ENV_VAR, OpenAIProvider
from symphonai_api.session import SessionError, TranscriptError
from symphonai_api.survey import survey_repository
from symphonai_api.tools.base import LocalTool
from symphonai_host.broker import EventBroker, Subscription
from symphonai_host.credentials import CredentialError, apply_to_environment, store
from symphonai_host.protocol import (
    ApprovalRequested,
    HistoryMessage,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_request,
    encode_event,
    encode_frame,
)
from symphonai_host.run import HostRun, ProviderSelectionError, RunActiveError
from symphonai_host.sessions import list_sessions


MAX_FILE_BYTES = 1024 * 1024
APP_CONTENT_TYPES = {
    ".css": "text/css",
    ".html": "text/html",
    ".js": "text/javascript",
}
APP_HANDSHAKE_MARKER = "<!-- symphonai-handshake -->"
APP_COOKIE_NAME = "symphonai_app"
PROVIDERS = (
    ("anthropic", ANTHROPIC_KEY_ENV_VAR, AnthropicProvider),
    ("gemini", GEMINI_KEY_ENV_VAR, GeminiProvider),
    ("openai", OPENAI_KEY_ENV_VAR, OpenAIProvider),
)


def _provider(name: str | None = None, model: str | None = None, base_url: str | None = None) -> ModelProvider | None:
    if name is None:
        name = next((vendor for vendor, key, _ in PROVIDERS if os.environ.get(key, "").strip()), None)
        if name is None:
            return None
    if not isinstance(name, str) or not any(vendor == name for vendor, _, _ in PROVIDERS):
        raise ProviderSelectionError("unknown provider")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ProviderSelectionError("model must be a non-empty string")
    if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
        raise ProviderSelectionError("base_url must be a non-empty string")
    vendor, key, provider_class = next(row for row in PROVIDERS if row[0] == name)
    if not os.environ.get(key, "").strip():
        raise ProviderSelectionError(f"{vendor} has no API key")
    options = {}
    if model is not None:
        options["model"] = model
    if base_url is not None:
        options["base_url"] = base_url
    return provider_class(**options)


def _app_root() -> Path:
    return Path(__file__).resolve().parent.parent / "symphonai_app"


class HostServer:
    """Own a loopback HTTP server, event broker, and one active runtime run."""

    def __init__(
        self,
        provider: ModelProvider | None,
        policy: PermissionPolicy,
        *,
        token: str | None = None,
        broker: EventBroker | None = None,
        keepalive_seconds: float = 15.0,
        system_prompt: str | None = None,
        max_turns: int = 20,
        model: str | None = None,
        approval_timeout: float = 300.0,
        sessions_root: Path | None = None,
        extensions: Extensions | None = None,
        mcp_tools: Mapping[str, LocalTool] | None = None,
        price_table: PriceTable | None = None,
        chat_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
        chat_recent_turns: int = DEFAULT_RECENT_TURNS,
    ) -> None:
        if keepalive_seconds <= 0:
            raise ValueError("keepalive_seconds must be greater than 0")
        self.token = token or secrets.token_urlsafe(32)
        self._repo_root = policy.repo_root
        self._mcp_started = mcp_tools is not None
        self.broker = broker or EventBroker()
        self.run = HostRun(
            provider,
            policy,
            self.broker,
            system_prompt=system_prompt,
            max_turns=max_turns,
            model=model,
            provider_factory=lambda name, selected_model, base_url: _provider(name, selected_model, base_url),
            publish_approval=self._publish_approval,
            approval_timeout=approval_timeout,
            sessions_root=sessions_root,
            extensions=extensions,
            mcp_tools=mcp_tools,
            price_table=price_table,
            chat_token_budget=chat_token_budget,
            chat_recent_turns=chat_recent_turns,
        )
        self.keepalive_seconds = keepalive_seconds
        self._handshake_printed = False
        self._thread: threading.Thread | None = None
        self._serving = False
        self._close_lock = threading.Lock()
        self._closed = False
        handler = self._handler_type()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._httpd.daemon_threads = True

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def address(self) -> tuple[str, int]:
        return "127.0.0.1", self.port

    def handshake(self) -> dict[str, Any]:
        return {"port": self.port, "token": self.token}

    def _publish_approval(self, approval: Any) -> bool:
        if self.broker.subscriber_count == 0:
            return False
        self.broker.publish(
            ApprovalRequested(
                approval_id=approval.approval_id,
                operation=approval.operation,
                target=approval.target,
                details=approval.details,
                tool_call_id=approval.tool_call_id,
            )
        )
        return True

    def pending_approvals(self) -> list[dict[str, str]]:
        return [approval.__dict__ for approval in self.run.approvals.pending()]

    def print_handshake(self) -> None:
        if not self._handshake_printed:
            print(json.dumps(self.handshake()), flush=True)
            self._handshake_printed = True

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self.serve_forever,
                name="symphonai-host-http",
                daemon=True,
            )
            self._thread.start()

    def serve_forever(self) -> None:
        self._serving = True
        try:
            self._httpd.serve_forever()
        finally:
            self._serving = False

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self.run.close()
        self.broker.close()
        if self._serving:
            self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        host = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def handle(self) -> None:
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError):
                    return

            def log_message(self, format: str, *args: object) -> None:
                return

            def _authorized(
                self,
                *,
                allow_app_query: bool = False,
                allow_app_cookie: bool = False,
            ) -> bool:
                supplied = self.headers.get("Authorization", "")
                expected = f"Bearer {host.token}"
                if secrets.compare_digest(supplied, expected):
                    return True
                if allow_app_query:
                    values = parse_qs(
                        urlsplit(self.path).query,
                        keep_blank_values=True,
                    ).get("token", [])
                    if len(values) == 1 and secrets.compare_digest(
                        values[0], host.token
                    ):
                        return True
                if allow_app_cookie:
                    cookie = SimpleCookie()
                    try:
                        cookie.load(self.headers.get("Cookie", ""))
                    except CookieError:
                        cookie.clear()
                    credential = cookie.get(APP_COOKIE_NAME)
                    if credential is not None and secrets.compare_digest(
                        credential.value, host.token
                    ):
                        return True
                self.send_response(HTTPStatus.UNAUTHORIZED)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return False

            def _json(self, status: HTTPStatus, body: object) -> None:
                encoded = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _empty(self, status: HTTPStatus) -> None:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _bytes(
                self,
                status: HTTPStatus,
                body: bytes,
                content_type: str,
                *,
                headers: tuple[tuple[str, str], ...] = (),
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for name, value in headers:
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

            def _app_path(self, requested: str) -> Path | None:
                candidate = Path(unquote(requested))
                if candidate.is_absolute():
                    self._empty(HTTPStatus.FORBIDDEN)
                    return None
                app_root = _app_root()
                if not app_root.is_dir():
                    self._json(
                        HTTPStatus.NOT_FOUND,
                        {"error": "app is not installed"},
                    )
                    return None
                try:
                    resolved = (app_root / candidate).resolve()
                except (OSError, RuntimeError):
                    self._empty(HTTPStatus.FORBIDDEN)
                    return None
                if not _contains_path(app_root, resolved):
                    self._empty(HTTPStatus.FORBIDDEN)
                    return None
                return resolved

            def _serve_app_index(self) -> None:
                resolved = self._app_path("index.html")
                if resolved is None:
                    return
                try:
                    source = resolved.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    self._not_found()
                    return
                if source.count(APP_HANDSHAKE_MARKER) != 1:
                    self._not_found()
                    return
                handshake = json.dumps(host.handshake()).replace("<", "\\u003c")
                script = f"<script>window.__symphonai = {handshake};</script>"
                body = source.replace(APP_HANDSHAKE_MARKER, script).encode("utf-8")
                cookie = (
                    f"{APP_COOKIE_NAME}={host.token}; Path=/app/; "
                    "HttpOnly; SameSite=Strict"
                )
                self._bytes(
                    HTTPStatus.OK,
                    body,
                    APP_CONTENT_TYPES[".html"],
                    headers=(("Set-Cookie", cookie),),
                )

            def _serve_app_asset(self, requested: str) -> None:
                resolved = self._app_path(requested)
                if resolved is None:
                    return
                content_type = APP_CONTENT_TYPES.get(resolved.suffix)
                if content_type is None:
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                try:
                    body = resolved.read_bytes()
                except OSError:
                    self._not_found()
                    return
                self._bytes(HTTPStatus.OK, body, content_type)

            def _serve_file(self) -> None:
                values = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                paths = values.get("path", [])
                if len(paths) != 1:
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                requested = paths[0]
                candidate = Path(requested)
                if candidate.is_absolute():
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                try:
                    resolved = (host._repo_root / candidate).resolve()
                except (OSError, RuntimeError):
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                allowed_roots = (
                    host._repo_root / "specs",
                    host._repo_root / "docs",
                )
                if not _contains_path(host._repo_root, resolved) or not any(
                    _contains_path(root, resolved) for root in allowed_roots
                ):
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                try:
                    with resolved.open("rb") as source:
                        data = source.read(MAX_FILE_BYTES + 1)
                except (OSError, ValueError):
                    self._not_found()
                    return
                if len(data) > MAX_FILE_BYTES:
                    self._empty(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                    return
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    self._empty(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                    return
                self._json(HTTPStatus.OK, {"path": requested, "text": text})

            def _read_object(self) -> dict:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    data = json.loads(self.rfile.read(length).decode("utf-8"))
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProtocolError(f"invalid JSON payload: {exc}") from None
                if not isinstance(data, dict):
                    raise ProtocolError("request payload must be an object")
                return data

            def _not_found(self) -> None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_GET(self) -> None:
                request_url = urlsplit(self.path)
                request_path = request_url.path
                if self.path == "/health":
                    self._json(
                        HTTPStatus.OK,
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "state": "active" if host.run.active else "idle",
                            "run_id": host.run.active_run_id,
                            "runtime_run_id": host.run.runtime_run_id,
                        },
                    )
                    return
                if self.path == "/conversation":
                    if not self._authorized():
                        return
                    self._json(
                        HTTPStatus.OK,
                        {"conversation": host.run.conversation_stats()},
                    )
                    return
                if request_path == "/app":
                    location = "/app/"
                    if request_url.query:
                        location = f"{location}?{request_url.query}"
                    self.send_response(HTTPStatus.FOUND)
                    self.send_header("Location", location)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if request_path == "/app/":
                    if not self._authorized(
                        allow_app_query=True,
                        allow_app_cookie=True,
                    ):
                        return
                    self._serve_app_index()
                    return
                if request_path.startswith("/app/"):
                    if not self._authorized(allow_app_cookie=True):
                        return
                    self._serve_app_asset(request_path.removeprefix("/app/"))
                    return
                if request_path == "/file":
                    if not self._authorized():
                        return
                    self._serve_file()
                    return
                if request_path == "/survey":
                    if not self._authorized():
                        return
                    survey = survey_repository(policy=host.run.policy)
                    relative = lambda path: path.relative_to(  # noqa: E731
                        survey.root
                    ).as_posix()
                    self._json(
                        HTTPStatus.OK,
                        {
                            "survey": {
                                "root": ".",
                                "languages": survey.languages,
                                "by_directory": survey.by_directory,
                                "entry_points": [
                                    relative(path) for path in survey.entry_points
                                ],
                                "docs": [relative(path) for path in survey.docs],
                                "tests": [relative(path) for path in survey.tests],
                                "tree_summary": survey.tree_summary,
                                "truncated_directories": survey.truncated_directories,
                                "stopped": survey.stopped,
                                "file_count": survey.file_count,
                            }
                        },
                    )
                    return
                if request_path == "/project":
                    if not self._authorized():
                        return
                    repo_root = host._repo_root.resolve()
                    self._json(
                        HTTPStatus.OK,
                        {"repo_root": str(repo_root), "name": repo_root.name},
                    )
                    return
                if request_path == "/settings":
                    if not self._authorized():
                        return
                    extensions = host.run.extensions
                    root = host.run.policy.repo_root.resolve()

                    def display_directory(directory: Path) -> str:
                        resolved = directory.resolve()
                        return (
                            resolved.relative_to(root).as_posix()
                            if resolved.is_relative_to(root)
                            else str(resolved)
                        )

                    def roster(kind: str) -> list[dict[str, str]]:
                        if extensions is None:
                            return []
                        members = getattr(extensions, kind)
                        paths = getattr(members, "paths", {})
                        return [
                            {
                                "name": name,
                                "path": display_directory(source) if source is not None else "",
                            }
                            for name in sorted(members)
                            for source in (paths.get(name, getattr(members[name], "path", None)),)
                        ]

                    ceiling = None if extensions is None else extensions.ceiling
                    settings = {
                        "config": [] if extensions is None else [
                            {
                                "key": key,
                                "value": value,
                                "scope": extensions.config.provenance[key].scope.value,
                            }
                            for key, value in sorted(extensions.config.values.items())
                        ],
                        "ceiling": {
                            "allowed_write_scope": None if ceiling is None or ceiling.allowed_write_scope is None else [str(path) for path in ceiling.allowed_write_scope],
                            "shell_enabled": None if ceiling is None else ceiling.shell_enabled,
                            "shell_allowlist": None if ceiling is None or ceiling.shell_allowlist is None else [list(command) for command in ceiling.shell_allowlist],
                            "fetch_enabled": None if ceiling is None else ceiling.fetch_enabled,
                            "fetch_allowlist": None if ceiling is None or ceiling.fetch_allowlist is None else list(ceiling.fetch_allowlist),
                            "modes": None if ceiling is None or ceiling.modes is None else list(ceiling.modes),
                        },
                        "trust": [] if extensions is None else [
                            {"root": str(entry.root), "allow": sorted(entry.allow)}
                            for entry in sorted(extensions.trust.entries, key=lambda entry: str(entry.root))
                        ],
                        "hooks": [] if extensions is None else [
                            {"event": event, "command": shlex.join(hook.command)}
                            for hook in extensions.hooks
                            for event in hook.events
                        ],
                        "mcp_servers": [] if extensions is None else [
                            {"name": spec.name, "command": shlex.join(spec.command), "started": spec.enabled and host._mcp_started}
                            for spec in sorted(extensions.mcp_servers, key=lambda spec: spec.name)
                        ],
                        "agents": roster("agents"),
                        "skills": roster("skills"),
                        "plugins": roster("plugins"),
                        "withheld": [] if extensions is None else [
                            {
                                "scope": offered.scope.value,
                                "directory": display_directory(offered.directory),
                                "names": sorted(offered.names),
                                "reason": f"repository not trusted for {offered.directory.name}",
                            }
                            for offered in sorted(extensions.withheld, key=lambda offered: (offered.scope.value, str(offered.directory)))
                        ],
                        "providers": [
                            {
                                "name": name,
                                "env_var": variable,
                                "key_present": bool(os.environ.get(variable, "").strip()),
                            }
                            for name, variable, _ in PROVIDERS
                        ],
                    }
                    self._json(HTTPStatus.OK, {"settings": settings})
                    return
                if self.path == "/approvals":
                    if not self._authorized():
                        return
                    self._json(HTTPStatus.OK, {"pending": host.pending_approvals()})
                    return
                if request_path == "/sessions":
                    if not self._authorized():
                        return
                    values = parse_qs(request_url.query, keep_blank_values=True).get("limit", [])
                    try:
                        limit = int(values[0]) if len(values) == 1 else None
                    except ValueError:
                        limit = None
                    if limit is not None and limit <= 0:
                        limit = None
                    self._json(HTTPStatus.OK, list_sessions(host.run.sessions_root, limit=limit))
                    return
                if self.path != "/events":
                    self._not_found()
                    return
                if not self._authorized():
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                subscription = host.broker.subscribe()
                try:
                    self._stream_events(subscription)
                except OSError:
                    return
                finally:
                    subscription.close()

            def _stream_events(self, subscription: Subscription) -> None:
                while not subscription.closed:
                    dropped = subscription.take_dropped()
                    if dropped:
                        self._sse(encode_frame("error", {"dropped": dropped}))
                    event = subscription.get(timeout=host.keepalive_seconds)
                    if event is None:
                        if subscription.closed:
                            return
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    if isinstance(event, ApprovalRequested):
                        self._sse(encode_frame("approval_requested", event.__dict__))
                    elif isinstance(event, HistoryMessage):
                        self._sse(encode_frame("event", event.payload()))
                    else:
                        self._sse(encode_frame("event", encode_event(event)))

            def _sse(self, frame: str) -> None:
                self.wfile.write(f"data: {frame}\n\n".encode("utf-8"))
                self.wfile.flush()

            def do_POST(self) -> None:
                credential_route = urlsplit(self.path).path == "/credentials"
                if self.path not in ("/prompt", "/stop", "/approval", "/session/open", "/session/new", "/provider") and not credential_route:
                    self._not_found()
                    return
                if not self._authorized():
                    return
                if credential_route:
                    try:
                        payload = self._read_object()
                    except ProtocolError:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid credential request"})
                        return
                    name = payload.get("name")
                    value = payload.get("value")
                    if name not in (ANTHROPIC_KEY_ENV_VAR, GEMINI_KEY_ENV_VAR, OPENAI_KEY_ENV_VAR):
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "unknown credential name"})
                        return
                    if not isinstance(value, str):
                        self._json(HTTPStatus.BAD_REQUEST, {"error": f"invalid value for {name}"})
                        return
                    try:
                        store(name, value)
                    except CredentialError:
                        self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"credential store unavailable for {name}"})
                        return
                    if value:
                        apply_to_environment(os.environ, {name: value})
                    else:
                        os.environ.pop(name, None)
                    self._json(HTTPStatus.OK, {"stored": True, "name": name})
                    return
                if self.path == "/session/new":
                    try:
                        if self._read_object():
                            raise ProtocolError("session/new takes an empty object")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    try:
                        host.run.end_conversation()
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc), "run_id": exc.run_id})
                        return
                    self._json(HTTPStatus.OK, {"ended": True})
                    return
                if self.path == "/provider":
                    try:
                        choice = self._read_object()
                        if not isinstance(choice.get("name"), str) or set(choice) - {"name", "model", "base_url"}:
                            raise ProviderSelectionError("unknown provider option")
                        provider = _provider(choice.get("name"), choice.get("model"), choice.get("base_url"))
                        if provider is None:
                            raise ProviderSelectionError("choose a provider")
                    except (ProtocolError, ProviderSelectionError) as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    host.run.select_provider(provider, choice.get("model"), choice)
                    self._json(HTTPStatus.OK, {"selected": True})
                    return
                kind = self.path.removeprefix("/")
                try:
                    request = decode_request(kind, self._read_object())
                except ProtocolError as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                if kind == "prompt":
                    try:
                        run_id = host.run.start(request.prompt)
                    except ProviderSelectionError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc), "run_id": exc.run_id})
                        return
                    self._json(HTTPStatus.OK, {"accepted": True, "run_id": run_id})
                    return
                if kind == "approval":
                    if not host.run.approvals.resolve(
                        request.approval_id, allowed=request.allowed, reason=request.reason
                    ):
                        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown approval id"})
                        return
                    self._json(HTTPStatus.OK, {"resolved": True})
                    return
                if kind == "session/open":
                    try:
                        reply = host.run.open_session(request.run_id)
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc), "run_id": exc.run_id})
                        return
                    except SessionError:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "session not found"})
                        return
                    except TranscriptError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    except ProviderSelectionError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, reply)
                    return
                host.run.stop()
                self._json(HTTPStatus.OK, {"accepted": True})

        return Handler
