"""Loopback-only HTTP/SSE boundary for a single SymphonAI host run."""

from __future__ import annotations

import json
import os
import secrets
import shlex
import sys
import subprocess
import threading
import tempfile
from datetime import datetime, timezone
from collections.abc import Mapping
from collections.abc import Callable
from http.cookies import CookieError, SimpleCookie
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from symphonai_api.compaction import DEFAULT_RECENT_TURNS
from symphonai_api.cost import PriceTable
from symphonai_api.agent_memory import MemoryUnavailable
from symphonai_api.agent_file import AgentFileError, load_agent_file
from symphonai_api.extensions import Extensions
from symphonai_api.events import Event
from symphonai_api.lsp import LspManager
from symphonai_api.model_discovery import list_models
from symphonai_api.model_table import model_capabilities
from symphonai_api.permissions import PermissionPolicy, _contains_path
from symphonai_api.providers.base import ModelProvider, ProviderError
from symphonai_api.providers.anthropic_provider import API_KEY_ENV_VAR as ANTHROPIC_KEY_ENV_VAR, AnthropicProvider
from symphonai_api.providers.gemini_provider import API_KEY_ENV_VAR as GEMINI_KEY_ENV_VAR, GeminiProvider
from symphonai_api.providers.openai_provider import API_KEY_ENV_VAR as OPENAI_KEY_ENV_VAR, OpenAIProvider
from symphonai_api.provider_catalog import CATALOG, build_catalog_provider, catalog_keys
from symphonai_api.session import SessionError, SessionStore, TranscriptError
from symphonai_api.paths import symphonai_home
from symphonai_api.survey import survey_repository
from symphonai_api.tools.base import LocalTool
from symphonai_api.web_search import search_endpoint, search_endpoints
from symphonai_host.broker import EventBroker, Subscription
from symphonai_host.approvals import InvalidApprovalAnswer
from symphonai_host.credentials import CredentialError, apply_to_environment, store
from symphonai_host.files import repository_files, repository_tree
from symphonai_host.protocol import (
    ApprovalRequested,
    HistoryMessage,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_request,
    encode_event,
    encode_frame,
)
from symphonai_host.run import (
    AgentControlError,
    ChangedOutsideError,
    CommandActiveError,
    HostRun,
    ModeSelectionError,
    NoConversationError,
    ProviderSelectionError,
    RunActiveError,
    WorktreeApplyConflict,
)
from symphonai_host.sessions import list_sessions, prompt_history
from symphonai_host.spec_run import parse_spec
from symphonai_host.projects import ProjectRuntime, record_recent_project, recent_projects


MAX_FILE_BYTES = 1024 * 1024
APP_CONTENT_TYPES = {
    ".css": "text/css",
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
}
APP_HANDSHAKE_MARKER = "<!-- symphonai-handshake -->"
APP_COOKIE_NAME = "symphonai_app"
PROVIDERS = (
    ("anthropic", ANTHROPIC_KEY_ENV_VAR, AnthropicProvider),
    ("gemini", GEMINI_KEY_ENV_VAR, GeminiProvider),
    ("openai", OPENAI_KEY_ENV_VAR, OpenAIProvider),
) + tuple(
    (key, CATALOG[key].api_key_env_var, None)
    for key in catalog_keys()
)


def _models_with_efforts(provider: str, models: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    efforts = {
        capability.model: [option.id for option in capability.efforts]
        for capability in model_capabilities()
        if capability.provider == provider
    }
    return [
        {"id": model, "efforts": efforts.get(model)}
        for model in models
    ]


def _model_listing(
    host: HostServer,
    provider: str,
    models: list[str] | tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extensions = None if host.run is None else host.run.extensions
    configured = None if extensions is None else extensions.config.get(f"models.{provider}")
    if configured is None:
        return _models_with_efforts(provider, models), {"applied": False, "hidden": 0}
    allowed = set(configured)
    visible = [model for model in models if model in allowed]
    return _models_with_efforts(provider, visible), {
        "applied": True,
        "hidden": len(models) - len(visible),
    }


def _memory_entry_reply(entry) -> dict[str, str]:  # noqa: ANN001
    written_at = (
        datetime.fromtimestamp(entry.written_at, timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    return {
        "id": entry.id,
        "text": entry.text,
        "written_at": written_at,
        "agent_name": entry.agent_name,
    }


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
    if name in CATALOG:
        if base_url is not None:
            raise ProviderSelectionError(f"{name} uses its catalog base URL")
        return build_catalog_provider(name, model)
    options = {}
    if model is not None:
        options["model"] = model
    if base_url is not None:
        options["base_url"] = base_url
    return provider_class(**options)


def _selection_path() -> Path:
    return symphonai_home() / "selection.json"


def _load_provider_selection() -> dict[str, str] | None:
    try:
        with _selection_path().open("r", encoding="utf-8") as source:
            choice = json.load(source)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if (
        not isinstance(choice, dict)
        or set(choice) - {"name", "model", "effort", "base_url"}
        or not isinstance(choice.get("name"), str)
        or not choice["name"]
        or any(
            key in choice and choice[key] is not None
            and (not isinstance(choice[key], str) or not choice[key].strip())
            for key in ("model", "effort", "base_url")
        )
    ):
        return None
    return choice


def _provider_for_start(provider_factory=_provider) -> tuple[ModelProvider | None, dict[str, str] | None]:
    choice = _load_provider_selection()
    if choice is None:
        return provider_factory(), None
    try:
        provider = provider_factory(choice["name"], choice.get("model"), choice.get("base_url"))
    except ProviderSelectionError:
        return provider_factory(), None
    return provider, choice


def _save_provider_selection(choice: Mapping[str, str]) -> None:
    target = _selection_path()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=".selection-", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(dict(choice), temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _app_root() -> Path:
    return Path(__file__).resolve().parent.parent / "symphonai_app"


class HostServer:
    """Own a loopback HTTP server, event broker, and one active runtime run."""

    def __init__(
        self,
        provider: ModelProvider | None,
        policy: PermissionPolicy | None,
        *,
        token: str | None = None,
        broker: EventBroker | None = None,
        keepalive_seconds: float = 15.0,
        system_prompt: str | None = None,
        base_prompt: str = "",
        subagent_base_prompt: str = "",
        working_dir: Path | None = None,
        max_turns: int = 20,
        model: str | None = None,
        approval_timeout: float = 300.0,
        sessions_root: Path | None = None,
        memory_root: Path | None = None,
        memory_extraction_enabled: bool | None = None,
        home: Path | None = None,
        extensions: Extensions | None = None,
        mcp_tools: Mapping[str, LocalTool] | None = None,
        price_table: PriceTable | None = None,
        chat_token_budget: int | None = None,
        chat_recent_turns: int = DEFAULT_RECENT_TURNS,
        lsp: LspManager | None = None,
        project_factory: Callable[[Path, EventBroker, Callable], ProjectRuntime] | None = None,
        runtime: ProjectRuntime | None = None,
        provider_selection: Mapping[str, str] | None = None,
    ) -> None:
        if keepalive_seconds <= 0:
            raise ValueError("keepalive_seconds must be greater than 0")
        self.token = token or secrets.token_urlsafe(32)
        self._current_root = (
            runtime.root.resolve() if runtime is not None else
            policy.repo_root.resolve() if policy is not None else None
        )
        self._repo_root = self._current_root
        self._project_factory = project_factory
        self._default_provider = provider
        self._provider_selection = dict(provider_selection or ({
            "name": provider.name,
            **({"model": provider.model} if getattr(provider, "model", None) else {}),
        } if provider is not None else {}))
        self._home = symphonai_home(home)
        self._model_cache: dict[tuple[str, str | None], tuple[str, ...]] = {}
        self._model_cache_lock = threading.Lock()
        self.broker = broker or EventBroker()
        initial_runtime = runtime or (ProjectRuntime(self._current_root, HostRun(
            provider,
            policy,
            self.broker,
            system_prompt=system_prompt,
            base_prompt=base_prompt,
            subagent_base_prompt=subagent_base_prompt,
            working_dir=working_dir,
            max_turns=max_turns,
            model=model,
            provider_factory=lambda name, selected_model, base_url: _provider(name, selected_model, base_url),
            publish_approval=self._publish_approval,
            approval_timeout=approval_timeout,
            sessions_root=sessions_root,
            memory_root=memory_root,
            memory_extraction_enabled=memory_extraction_enabled,
            extensions=extensions,
            mcp_tools=mcp_tools,
            price_table=price_table,
            chat_token_budget=chat_token_budget,
            chat_recent_turns=chat_recent_turns,
            lsp=lsp,
        )) if policy is not None else None)
        initial_run = None if initial_runtime is None else initial_runtime.run
        self.run = initial_run
        self._runtimes: dict[Path, ProjectRuntime] = {}
        self._runtimes_lock = threading.RLock()
        self._event_session_ids: dict[str, str] = {}
        if initial_runtime is not None and self._current_root is not None:
            self._runtimes[self._current_root] = initial_runtime
        self._watch_stop = threading.Event()
        self._event_watch: threading.Thread | None = None
        self._event_watch_subscription: Subscription | None = None
        if self._project_factory is not None:
            self._event_watch = threading.Thread(target=self._watch_project_events, daemon=True)
            self._event_watch.start()
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
        watcher_count = int(
            self._event_watch_subscription is not None
            and not self._event_watch_subscription.closed
        )
        if self.broker.subscriber_count <= watcher_count:
            return False
        self.broker.publish(
            ApprovalRequested(
                approval_id=approval.approval_id,
                operation=approval.operation,
                target=approval.target,
                details=approval.details,
                tool_call_id=approval.tool_call_id,
                remember=approval.remember,
                session_id=approval.session_id,
                kind=approval.kind,
                options=approval.options,
            )
        )
        return True

    def pending_approvals(self) -> list[dict[str, str]]:
        with self._runtimes_lock:
            runtimes = tuple(self._runtimes.values())
        return [approval.__dict__ for runtime in runtimes for approval in runtime.run.pending_approvals()]

    def resolve_approval(self, approval_id: str, **decision) -> bool:
        with self._runtimes_lock:
            runtimes = tuple(self._runtimes.values())
        return any(runtime.run.resolve_approval(approval_id, **decision) for runtime in runtimes)

    def session_activity(self) -> dict[str, str]:
        activity = {}
        with self._runtimes_lock:
            runtimes = tuple(self._runtimes.values())
        for runtime in runtimes:
            activity.update(runtime.run.session_activity())
        return activity

    def event_session_id(self, event: Event) -> str | None:
        session_id = getattr(event, "session_id", None)
        if isinstance(session_id, str):
            return session_id
        run_id = getattr(event, "run_id", None)
        if isinstance(run_id, str) and run_id in self._event_session_ids:
            return self._event_session_ids[run_id]
        with self._runtimes_lock:
            runtimes = tuple(self._runtimes.values())
        for runtime in runtimes:
            session_id = runtime.run.event_session_id(event)
            if session_id is not None:
                if isinstance(run_id, str):
                    self._event_session_ids[run_id] = session_id
                    if len(self._event_session_ids) > 4096:
                        self._event_session_ids.pop(next(iter(self._event_session_ids)))
                return session_id
        return None

    @property
    def run(self) -> HostRun | None:
        with self._runtimes_lock:
            return None if self._current_root is None else self._runtimes[self._current_root].run

    @run.setter
    def run(self, value: HostRun | None) -> None:
        self._initial_run = value

    def open_project(self, root: Path) -> tuple[ProjectRuntime | None, bool]:
        resolved = root.resolve()
        if self._project_factory is None:
            raise RuntimeError("this host cannot open other projects")
        with self._runtimes_lock:
            runtime = self._runtimes.get(resolved)
        if runtime is None:
            current = self.run
            selection = current.selection() if current is not None else {
                "model": self._provider_selection.get("model"),
                "effort": self._provider_selection.get("effort"),
            }
            provider = current._provider if current is not None else self._default_provider
            runtime = self._project_factory(resolved, self.broker, self._publish_approval)
            with self._runtimes_lock:
                self._runtimes[resolved] = runtime
            if provider is not None:
                runtime.run.select_provider(
                    provider, selection.get("model"), selection.get("effort"),
                )
            created = True
        else:
            created = False
        with self._runtimes_lock:
            previous = self._current_root
            self._current_root = resolved
            self._repo_root = resolved
        if previous is not None and previous != resolved:
            self._close_if_idle(previous)
        record_recent_project(self._home, resolved)
        return runtime, created

    def _close_if_idle(self, root: Path) -> None:
        with self._runtimes_lock:
            runtime = self._runtimes.get(root)
        if runtime is None:
            return
        run = runtime.run
        if run.has_work_in_flight():
            return
        with self._runtimes_lock:
            if root != self._current_root and self._runtimes.get(root) is runtime:
                self._runtimes.pop(root, None)
            else:
                return
        if root != self._current_root:
            runtime.close()

    def _session_belongs_here(self, run_id: str) -> bool:
        try:
            store = SessionStore.open(self.run.sessions_root, run_id)
            try:
                owner = store.read_meta().get("repo_root")
            finally:
                store.close()
        except Exception:
            return True
        return owner is None or (self._current_root is not None and Path(owner).resolve() == self._current_root)

    def _watch_project_events(self) -> None:
        subscription = self.broker.subscribe()
        self._event_watch_subscription = subscription
        try:
            while not self._watch_stop.is_set():
                event = subscription.get(timeout=0.1)
                if event is not None:
                    self.event_session_id(event)
                with self._runtimes_lock:
                    roots = tuple(self._runtimes)
                    current_root = self._current_root
                for root in roots:
                    if root != current_root:
                        self._close_if_idle(root)
        finally:
            subscription.close()
            self._event_watch_subscription = None

    def print_handshake(self) -> None:
        if not self._handshake_printed:
            handshake = self.handshake()
            print(json.dumps(handshake), flush=True)
            print(
                f"http://127.0.0.1:{handshake['port']}/app/?token={handshake['token']}",
                file=sys.stderr,
                flush=True,
            )
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
            self._httpd.serve_forever(poll_interval=0.05)
        finally:
            self._serving = False

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._watch_stop.set()
        with self._runtimes_lock:
            runtimes = tuple(self._runtimes.values())
            self._runtimes.clear()
        for runtime in runtimes:
            runtime.close()
        self.broker.close()
        if self._event_watch is not None:
            self._event_watch.join(timeout=1)
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

            def _serve_tree(self) -> None:
                values = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                paths = values.get("path", [])
                if len(paths) > 1:
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                requested = paths[0] if paths else ""
                candidate = Path(requested)
                if candidate.is_absolute() or ".." in candidate.parts:
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                try:
                    resolved = (host._repo_root / candidate).resolve()
                except (OSError, RuntimeError):
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                if not host.run.policy.check_read(resolved).allowed:
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                if not resolved.is_dir():
                    self._empty(HTTPStatus.NOT_FOUND)
                    return
                entries, truncated = repository_tree(host.run.policy, resolved)
                self._json(HTTPStatus.OK, {
                    "path": requested,
                    "entries": entries,
                    "truncated": truncated,
                })

            def _serve_source(self) -> None:
                values = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                paths = values.get("path", [])
                if len(paths) != 1:
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                requested = paths[0]
                candidate = Path(requested)
                if candidate.is_absolute() or ".." in candidate.parts:
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                try:
                    resolved = (host._repo_root / candidate).resolve()
                except (OSError, RuntimeError):
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                if not host.run.policy.check_read(resolved).allowed:
                    self._empty(HTTPStatus.FORBIDDEN)
                    return
                try:
                    with resolved.open("rb") as source:
                        data = source.read(MAX_FILE_BYTES + 1)
                except (OSError, ValueError):
                    self._empty(HTTPStatus.NOT_FOUND)
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

            def _serve_spec_files(self) -> None:
                specs_root = host._repo_root / "specs"
                paths = []
                for path in specs_root.rglob("*.md"):
                    relative = path.relative_to(host._repo_root)
                    if (
                        len(relative.parts) < 3
                        or relative.parts[1] == "report"
                        or path.name.endswith("-PLAN.md")
                        or not path.is_file()
                        or not host.run.policy.check_read(path).allowed
                    ):
                        continue
                    paths.append(relative.as_posix())
                self._json(HTTPStatus.OK, {"paths": sorted(paths)})

            def _definition_target(
                self,
                name: object,
                scope: object,
            ) -> tuple[str, str, Path] | tuple[None, None, None]:
                if not isinstance(name, str) or not name:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "name must be a non-empty string"})
                    return None, None, None
                if not isinstance(scope, str) or scope not in ("project", "user"):
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "scope must be 'project' or 'user'"})
                    return None, None, None
                if any(separator in name for separator in ("/", "\\")):
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "name must be a single segment"})
                    return None, None, None
                if name in (".", ".."):
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "name must not be a parent reference"})
                    return None, None, None
                if name.endswith(".toml"):
                    stem = name[:-5]
                elif Path(name).suffix:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "name must use the .toml extension"})
                    return None, None, None
                else:
                    stem = name
                if not stem or stem in (".", "..") or Path(stem).name != stem:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "name must be a single segment"})
                    return None, None, None
                filename = f"{stem}.toml"
                directory = (
                    host._repo_root / ".symphonai" / "agents"
                    if scope == "project"
                    else host._home / "agents"
                )
                try:
                    base = (
                        Path(host._repo_root).resolve()
                        if scope == "project"
                        else host._home.resolve()
                    )
                    root = directory.resolve()
                    target = (root / filename).resolve(strict=False)
                except (OSError, RuntimeError):
                    self._empty(HTTPStatus.FORBIDDEN)
                    return None, None, None
                if not _contains_path(base, root) or not _contains_path(root, target):
                    self._empty(HTTPStatus.FORBIDDEN)
                    return None, None, None
                return stem, scope, target

            def _definition_trusted(self, scope: str) -> bool:
                if scope == "user":
                    return True
                extensions = host.run.extensions
                return extensions is not None and extensions.trust.allows(
                    host._repo_root,
                    "agents",
                )

            def _validate_definition(self, target: Path, text: str) -> None:
                try:
                    with tempfile.TemporaryDirectory() as temporary:
                        staged = Path(temporary) / target.name
                        staged.write_text(text, encoding="utf-8")
                        load_agent_file(
                            staged,
                            repo_root=host._repo_root,
                            ceiling=(
                                None
                                if host.run.extensions is None
                                else host.run.extensions.ceiling
                            ),
                        )
                except AgentFileError as exc:
                    message = str(exc)
                    prefix = f"{staged}: "
                    if message.startswith(prefix):
                        message = f"{target}: {message[len(prefix):]}"
                    raise AgentFileError(message) from None

            def _write_definition(self, target: Path, text: str) -> None:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary_name: str | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=target.parent,
                        prefix=f".{target.name}.",
                        suffix=".tmp",
                        delete=False,
                    ) as temporary:
                        temporary_name = temporary.name
                        temporary.write(text)
                        temporary.flush()
                        os.fsync(temporary.fileno())
                    os.replace(temporary_name, target)
                finally:
                    if temporary_name is not None:
                        try:
                            Path(temporary_name).unlink()
                        except FileNotFoundError:
                            pass

            def _serve_definition(self) -> None:
                values = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                if set(values) != {"name", "scope"} or any(
                    len(values[key]) != 1 for key in ("name", "scope")
                ):
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "name and scope are required"})
                    return
                stem, scope, target = self._definition_target(
                    values["name"][0], values["scope"][0]
                )
                if target is None:
                    return
                if not self._definition_trusted(scope):
                    self._json(
                        HTTPStatus.FORBIDDEN,
                        {"error": "repository not trusted for agents"},
                    )
                    return
                try:
                    data = target.read_bytes()
                except OSError:
                    self._not_found()
                    return
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    self._empty(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                    return
                self._json(
                    HTTPStatus.OK,
                    {"name": stem, "scope": scope, "text": text},
                )

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
                if host.run is None and request_path not in (
                    "/health", "/events", "/project", "/projects", "/directories",
                    "/settings", "/models", "/app", "/app/",
                ) and not request_path.startswith("/app/"):
                    if not self._authorized():
                        return
                    self._json(HTTPStatus.CONFLICT, {"error": "open a project first"})
                    return
                if self.path == "/health":
                    self._json(
                        HTTPStatus.OK,
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "state": "active" if host.run is not None and host.run.active else "idle",
                            "run_id": None if host.run is None else host.run.active_run_id,
                            "runtime_run_id": None if host.run is None else host.run.runtime_run_id,
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
                if request_path == "/changes":
                    if not self._authorized():
                        return
                    try:
                        reply = host.run.changes()
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, reply)
                    return
                if request_path == "/memory":
                    if not self._authorized():
                        return
                    memory = host.run.memory_store()
                    if memory is None:
                        self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "memory is unavailable"})
                        return
                    reply = {}
                    for scope in ("project", "global"):
                        entries, unreadable = memory.read_status(scope)
                        reply[scope] = (
                            {"entries": [], "error": "unreadable"}
                            if unreadable
                            else {"entries": [_memory_entry_reply(entry) for entry in entries]}
                        )
                    self._json(HTTPStatus.OK, reply)
                    return
                if request_path == "/models":
                    if not self._authorized():
                        return
                    query = parse_qs(request_url.query, keep_blank_values=True)
                    names = query.get("provider", [])
                    base_urls = query.get("base_url", [])
                    known = {name for name, _, _ in PROVIDERS}
                    if (
                        set(query) - {"provider", "base_url"}
                        or len(names) != 1
                        or names[0] not in known
                        or len(base_urls) > 1
                        or (base_urls and not base_urls[0].strip())
                    ):
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid model listing request"})
                        return
                    name = names[0]
                    base_url = base_urls[0] if base_urls else None
                    cache_key = (name, base_url)
                    with host._model_cache_lock:
                        cached = host._model_cache.get(cache_key)
                    if cached is not None:
                        listed, model_filter = _model_listing(host, name, cached)
                        self._json(HTTPStatus.OK, {
                            "provider": name,
                            "state": "available",
                            "models": listed,
                            "detail": "",
                            "filter": model_filter,
                        })
                        return
                    try:
                        provider = _provider(name, None, base_url)
                        if provider is None:
                            raise ProviderSelectionError(f"{name} is unavailable")
                        models = list_models(provider)
                    except (ProviderSelectionError, ProviderError) as exc:
                        detail = str(exc)
                        variable = next(key for vendor, key, _ in PROVIDERS if vendor == name)
                        secret = os.environ.get(variable, "").strip()
                        if secret:
                            detail = detail.replace(secret, "[redacted]")
                        self._json(HTTPStatus.OK, {
                            "provider": name,
                            "state": "unknown",
                            "models": [],
                            "detail": detail,
                            "filter": _model_listing(host, name, [])[1],
                        })
                        return
                    with host._model_cache_lock:
                        host._model_cache[cache_key] = tuple(models)
                    listed, model_filter = _model_listing(host, name, models)
                    self._json(HTTPStatus.OK, {
                        "provider": name,
                        "state": "available",
                        "models": listed,
                        "detail": "",
                        "filter": model_filter,
                    })
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
                if request_path == "/tree":
                    if not self._authorized():
                        return
                    self._serve_tree()
                    return
                if request_path == "/source":
                    if not self._authorized():
                        return
                    self._serve_source()
                    return
                if request_path == "/spec/files":
                    if not self._authorized():
                        return
                    self._serve_spec_files()
                    return
                if request_path == "/files":
                    if not self._authorized():
                        return
                    values = parse_qs(request_url.query, keep_blank_values=True)
                    query_values = values.get("query", [""])
                    limit_values = values.get("limit", [])
                    try:
                        limit = 20 if not limit_values else int(limit_values[0])
                    except ValueError:
                        limit = 0
                    if len(limit_values) > 1 or not 1 <= limit <= 50:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "limit must be from 1 to 50"})
                        return
                    files, truncated = repository_files(
                        host.run.policy,
                        query_values[0],
                        limit,
                    )
                    self._json(HTTPStatus.OK, {"files": files, "truncated": truncated})
                    return
                if request_path == "/agent":
                    if not self._authorized():
                        return
                    self._serve_definition()
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
                    if host._current_root is None:
                        self._json(HTTPStatus.OK, {"repo_root": None, "name": None})
                        return
                    repo_root = host._repo_root.resolve()
                    self._json(
                        HTTPStatus.OK,
                        {"repo_root": str(repo_root), "name": repo_root.name},
                    )
                    return
                if request_path == "/projects":
                    if not self._authorized():
                        return
                    current_root = host._current_root
                    with host._runtimes_lock:
                        runtimes = tuple(host._runtimes.items())
                    open_projects = [
                        {"repo_root": str(root), "name": root.name,
                         "activity": "working" if any(state == "working" for state in runtime.run.session_activity().values()) else "waiting" if runtime.run.pending_approvals() else "idle"}
                        for root, runtime in runtimes
                    ]
                    recent = recent_projects(host._home)
                    self._json(HTTPStatus.OK, {
                        "current": None if current_root is None else {"repo_root": str(current_root), "name": current_root.name},
                        "open": open_projects,
                        "recent": [{"repo_root": path, "name": Path(path).name,
                                    "exists": Path(path).is_dir()} for path in recent if isinstance(path, str)],
                    })
                    return
                if request_path == "/directories":
                    if not self._authorized():
                        return
                    paths = parse_qs(request_url.query, keep_blank_values=True).get("path", [])
                    if len(paths) > 1:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "path must be one absolute directory"})
                        return
                    target = Path(paths[0]).expanduser() if paths else Path.home()
                    if not target.is_absolute():
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "path must be absolute"})
                        return
                    target = target.resolve()
                    if not target.exists() or not target.is_dir():
                        self._json(HTTPStatus.NOT_FOUND, {"error": "directory not found"})
                        return
                    try:
                        entries = sorted(
                            ({"name": child.name, "path": str(child)} for child in target.iterdir()
                             if not child.name.startswith(".") and child.is_dir()),
                            key=lambda entry: entry["name"].casefold(),
                        )
                    except PermissionError:
                        self._json(HTTPStatus.FORBIDDEN, {"error": "directory is not readable"})
                        return
                    self._json(HTTPStatus.OK, {
                        "path": str(target),
                        "parent": None if target == Path("/") else str(target.parent),
                        "entries": entries,
                    })
                    return
                if request_path == "/settings":
                    if not self._authorized():
                        return
                    if host.run is None:
                        selection = host._provider_selection or None
                        self._json(HTTPStatus.OK, {"settings": {
                            "mode": "ask",
                            "selection": None if selection is None else {
                                "provider": selection.get("name"), "model": selection.get("model"), "effort": selection.get("effort")
                            },
                            "config": [], "ceiling": {}, "trust": [], "hooks": [],
                            "mcp_servers": [], "agents": [], "skills": [], "plugins": [], "withheld": [],
                            "providers": [{"name": name, "env_var": variable,
                                           "key_present": bool(os.environ.get(variable, "").strip())}
                                          for name, variable, _ in PROVIDERS],
                            "search": [],
                        }})
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
                    search_key = None if extensions is None else extensions.config.get("search.endpoint")
                    configured_search = None if search_key is None else search_endpoint(search_key)
                    settings = {
                        "mode": host.run.policy.mode,
                        "selection": host.run.selection(),
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
                            {"name": spec.name, "command": shlex.join(spec.command), "started": spec.enabled and host.run._mcp_tools is not None}
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
                        "search": [] if configured_search is None else [
                            {
                                "name": configured_search.key,
                                "env_var": configured_search.api_key_env_var,
                                "key_present": bool(os.environ.get(configured_search.api_key_env_var, "").strip()),
                            }
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
                    queries = parse_qs(request_url.query, keep_blank_values=True).get("query", [])
                    query = queries[0] if len(queries) == 1 else None
                    try:
                        limit = int(values[0]) if len(values) == 1 else None
                    except ValueError:
                        limit = None
                    if limit is not None and limit <= 0:
                        limit = None
                    self._json(HTTPStatus.OK, list_sessions(
                        host.run.sessions_root, limit=limit, activity=host.session_activity(), query=query,
                    ))
                    return
                if request_path == "/spec/runs":
                    if not self._authorized():
                        return
                    from symphonai_api.session import SessionStore
                    runs = []
                    for directory in host.run.sessions_root.iterdir() if host.run.sessions_root.is_dir() else ():
                        try:
                            store = SessionStore.open(host.run.sessions_root, directory.name)
                            meta = store.read_meta()
                            store.close()
                        except Exception:
                            continue
                        info = meta.get("spec_run")
                        if not isinstance(info, dict):
                            continue
                        if info.get("kind") == "plan":
                            runs.append({
                                "session_id": directory.name, "kind": "plan",
                                "phase": info.get("phase"), "item": info.get("item"),
                                "bound": info.get("bound"),
                                "state": "running" if host.run.session_activity().get(directory.name) in ("working", "waiting") and info.get("state") is None else info.get("state", "running"),
                                "updated_at": meta.get("updated_at"),
                            })
                            continue
                        if info.get("kind") != "implement" or not isinstance(info.get("spec"), str):
                            continue
                        worktree = host.run.sessions_root / directory.name / str(info.get("worktree", "worktree"))
                        active = host.run.session_activity().get(directory.name) in ("working", "waiting")
                        files = []
                        if worktree.is_dir():
                            changed = subprocess.run(
                                ["git", "diff", "--name-only", "-z", "HEAD"],
                                cwd=worktree, capture_output=True, check=False,
                            )
                            untracked = subprocess.run(
                                ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                                cwd=worktree, capture_output=True, check=False,
                            )
                            if changed.returncode == 0 and untracked.returncode == 0:
                                files = sorted({
                                    os.fsdecode(value)
                                    for output in (changed.stdout, untracked.stdout)
                                    for value in output.split(b"\0") if value
                                })
                        state = "running" if active and info.get("state") not in ("finished", "blocked", "stopped") else info.get("state", "running")
                        title = ""
                        try:
                            title = parse_spec(host._repo_root / info["spec"], host._repo_root)["title"]
                        except (ValueError, OSError):
                            pass
                        spec_id = Path(info["spec"]).stem.split("-", 1)[0]
                        after_dash = title.split("—", 1)[1].strip() if "—" in title else title
                        runs.append({
                            "session_id": directory.name, "spec": info["spec"],
                            "kind": info.get("kind", "implement"), "state": state,
                            "report_copied": bool(info.get("report_copied", False)),
                            "files": files, "updated_at": meta.get("updated_at"),
                            "review": meta.get("review"), "committed": meta.get("committed"),
                            "commit_message": f"{spec_id}: {after_dash}".strip(),
                        })
                    runs.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
                    self._json(HTTPStatus.OK, runs)
                    return
                if request_path == "/history":
                    if not self._authorized():
                        return
                    values = parse_qs(request_url.query, keep_blank_values=True).get("limit", [])
                    try:
                        limit = 100 if not values else int(values[0])
                    except ValueError:
                        limit = 0
                    if len(values) > 1 or not 1 <= limit <= 500:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "limit must be from 1 to 500"})
                        return
                    self._json(HTTPStatus.OK, {
                        "prompts": prompt_history(
                            host.run.sessions_root,
                            host.run.policy.repo_root,
                            limit=limit,
                        ),
                    })
                    return
                if self.path != "/events":
                    self._not_found()
                    return
                if not self._authorized():
                    return
                subscription = host.broker.subscribe()
                try:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
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
                    session_id = host.event_session_id(event)
                    if isinstance(event, ApprovalRequested):
                        self._sse(encode_frame("approval_requested", event.__dict__))
                    elif isinstance(event, HistoryMessage):
                        payload = event.payload()
                        if session_id is not None:
                            payload["session_id"] = session_id
                        self._sse(encode_frame("event", payload))
                    else:
                        payload = encode_event(event)
                        if session_id is not None:
                            payload["session_id"] = session_id
                        self._sse(encode_frame("event", payload))

            def _sse(self, frame: str) -> None:
                self.wfile.write(f"data: {frame}\n\n".encode("utf-8"))
                self.wfile.flush()

            def do_POST(self) -> None:
                credential_route = urlsplit(self.path).path == "/credentials"
                if self.path not in ("/project/open", "/prompt", "/stop", "/approval", "/session/open", "/session/fork", "/session/new", "/session/rename", "/session/delete", "/provider", "/mode", "/compact", "/agent", "/agent/control", "/steer", "/command", "/changes/revert", "/worktree/apply", "/worktree/discard", "/goal", "/goal/state", "/spec/run", "/spec/review", "/spec/commit", "/spec/plan", "/memory/update", "/memory/delete") and not credential_route:
                    self._not_found()
                    return
                if not self._authorized():
                    return
                if host.run is None and self.path not in ("/project/open", "/credentials", "/provider"):
                    self._json(HTTPStatus.CONFLICT, {"error": "open a project first"})
                    return
                if self.path == "/project/open":
                    try:
                        payload = self._read_object()
                        path = payload.get("path")
                        if set(payload) != {"path"} or not isinstance(path, str):
                            raise ProtocolError("project/open requires an absolute path")
                        root = Path(path)
                        if not root.is_absolute() or not root.is_dir() or root.resolve() == Path("/"):
                            self._json(HTTPStatus.BAD_REQUEST, {"error": "path must be an existing project directory other than /"})
                            return
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    if host._project_factory is None:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "this host cannot open other projects"})
                        return
                    try:
                        runtime, _created = host.open_project(root)
                    except Exception as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, {"repo_root": str(runtime.root), "name": runtime.root.name})
                    return
                if self.path == "/memory/update":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"id", "text"} or not all(
                            isinstance(payload.get(key), str) for key in ("id", "text")
                        ):
                            raise ProtocolError("memory/update requires id and text")
                        if not payload["text"].strip() or len(payload["text"]) > 500:
                            raise ProtocolError("text must be non-blank and at most 500 characters")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    memory = host.run.memory_store()
                    if memory is None:
                        self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "memory is unavailable"})
                        return
                    try:
                        entry = memory.update(payload["id"], payload["text"])
                    except MemoryUnavailable:
                        self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    except ValueError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    if entry is None:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown memory id"})
                        return
                    self._json(HTTPStatus.OK, {"entry": _memory_entry_reply(entry)})
                    return
                if self.path == "/memory/delete":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"id"} or not isinstance(payload.get("id"), str):
                            raise ProtocolError("memory/delete requires id")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    memory = host.run.memory_store()
                    if memory is None:
                        self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "memory is unavailable"})
                        return
                    try:
                        deleted = memory.delete(payload["id"])
                    except MemoryUnavailable:
                        self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    if not deleted:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown memory id"})
                        return
                    self._json(HTTPStatus.OK, {"deleted": payload["id"]})
                    return
                if self.path == "/spec/plan":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"phase", "item"} or not isinstance(payload.get("phase"), str) or type(payload.get("item")) is not int:
                            raise ProtocolError("spec/plan requires phase and item")
                        roadmap_path = host._repo_root / "docs" / "roadmap.json"
                        roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))
                        phase_id, index = payload["phase"], payload["item"]
                        phase = next((entry for entry in roadmap.get("phases", []) if entry.get("id") == phase_id), None)
                        if phase is None or not isinstance(phase.get("items"), list) or not 0 <= index < len(phase["items"]):
                            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown roadmap item"})
                            return
                        item_value = phase["items"][index]
                        title = item_value if isinstance(item_value, str) else item_value.get("title", "")
                        if (item_value.get("spec") if isinstance(item_value, dict) else None):
                            self._json(HTTPStatus.CONFLICT, {"error": "roadmap item already has a spec"})
                            return
                        if not phase_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in phase_id):
                            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown roadmap phase"})
                            return
                        from symphonai_api.session import SessionStore
                        for directory in host.run.sessions_root.iterdir() if host.run.sessions_root.is_dir() else ():
                            try:
                                session_store = SessionStore.open(host.run.sessions_root, directory.name)
                                info = session_store.read_meta().get("spec_run")
                                session_store.close()
                            except Exception:
                                continue
                            if isinstance(info, dict) and info.get("kind") == "plan" and info.get("phase") == phase_id and info.get("item") == index and host.run.session_activity().get(directory.name) in ("working", "waiting"):
                                self._json(HTTPStatus.CONFLICT, {"error": "a plan for this item is already running"})
                                return
                        phase_root = host._repo_root / "specs" / phase_id
                        baseline = [path.relative_to(host._repo_root).as_posix() for path in phase_root.rglob("*.md")] if phase_root.is_dir() else []
                        phase_plan = phase_root / f"{phase_id}-PLAN.md"
                        plan_text = phase_plan.read_text(encoding="utf-8") if phase_plan.is_file() else ""
                        session_id, run_id = host.run.start_spec_plan(
                            phase_id, index, str(title), str(phase.get("name", "")),
                            plan_text, f"Plan roadmap item: {title}", baseline,
                        )
                    except (ProtocolError, OSError, ValueError, json.JSONDecodeError) as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc), "run_id": exc.run_id})
                        return
                    self._json(HTTPStatus.OK, {"session_id": session_id, "run_id": run_id})
                    return
                if self.path == "/spec/review":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"session_id"} or not isinstance(payload.get("session_id"), str):
                            raise ProtocolError("spec/review requires session_id")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    try:
                        session_id, run_id = host.run.start_spec_review(payload["session_id"])
                    except KeyError:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "not a spec run"})
                        return
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                        return
                    except (OSError, ValueError, RuntimeError) as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, {"session_id": session_id, "run_id": run_id})
                    return
                if self.path == "/spec/commit":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"session_id", "message"} or not all(isinstance(payload.get(key), str) for key in ("session_id", "message")):
                            raise ProtocolError("spec/commit requires session_id and message")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    try:
                        reply = host.run.commit_spec(payload["session_id"], payload["message"])
                    except KeyError:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "not a spec run"})
                        return
                    except ValueError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    except WorktreeApplyConflict as exc:
                        status = HTTPStatus.CONFLICT
                        self._json(status, {"error": str(exc)})
                        return
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, reply)
                    return
                if self.path == "/spec/run":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"path"} or not isinstance(payload.get("path"), str):
                            raise ProtocolError("spec/run requires a path")
                        if Path(payload["path"]).is_absolute():
                            raise ProtocolError("spec/run path must be repository-relative")
                        spec = parse_spec(host._repo_root / payload["path"], host._repo_root)
                    except (ProtocolError, ValueError, OSError) as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    from symphonai_api.session import SessionStore
                    for directory in host.run.sessions_root.iterdir() if host.run.sessions_root.is_dir() else ():
                        try:
                            session_store = SessionStore.open(host.run.sessions_root, directory.name)
                            info = session_store.read_meta().get("spec_run")
                            session_store.close()
                        except Exception:
                            continue
                        if isinstance(info, dict) and info.get("spec") == spec["path"] and host.run.session_activity().get(directory.name) in ("working", "waiting"):
                            self._json(HTTPStatus.CONFLICT, {"error": "this spec is already running"})
                            return
                    try:
                        session_id, run_id = host.run.start_spec_run(spec)
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc), "run_id": exc.run_id})
                        return
                    except (ProviderSelectionError, OSError, RuntimeError) as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, {"session_id": session_id, "run_id": run_id})
                    return
                if self.path == "/goal":
                    try:
                        payload = self._read_object()
                        objective = payload.get("objective")
                        check = payload.get("check", [])
                        max_rounds = payload.get("max_rounds", 10)
                        if (
                            set(payload) - {"objective", "check", "max_rounds"}
                            or not isinstance(objective, str)
                            or not objective.strip()
                            or not isinstance(check, list)
                            or not all(isinstance(arg, str) and arg for arg in check)
                            or type(max_rounds) is not int
                            or not 1 <= max_rounds <= 100
                        ):
                            raise ProtocolError("goal requires a non-blank objective, optional check argv, and max_rounds from 1 to 100")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    try:
                        run_id = host.run.start_goal(objective, tuple(check), max_rounds)
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc), "run_id": exc.run_id})
                        return
                    except ProviderSelectionError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, {
                        "accepted": True,
                        "run_id": run_id,
                        "goal": host.run.goal_snapshot(),
                    })
                    return
                if self.path == "/goal/state":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"action"} or payload.get("action") not in ("pause", "resume", "clear"):
                            raise ProtocolError("goal/state requires pause, resume, or clear")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    try:
                        goal = host.run.goal_state(payload["action"])
                    except KeyError:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "no goal"})
                        return
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, {"goal": goal})
                    return
                if self.path == "/agent":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"name", "scope", "text"}:
                            raise ProtocolError("agent requires name, scope, and text")
                        if not all(isinstance(payload[key], str) for key in ("name", "scope", "text")):
                            raise ProtocolError("agent name, scope, and text must be strings")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    stem, scope, target = self._definition_target(
                        payload["name"], payload["scope"]
                    )
                    if target is None:
                        return
                    if not self._definition_trusted(scope):
                        self._json(
                            HTTPStatus.FORBIDDEN,
                            {"error": "repository not trusted for agents"},
                        )
                        return
                    try:
                        self._validate_definition(target, payload["text"])
                    except AgentFileError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    try:
                        self._write_definition(target, payload["text"])
                    except OSError as exc:
                        self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"could not write definition: {exc}"})
                        return
                    self._json(
                        HTTPStatus.OK,
                        {
                            "written": True,
                            "name": stem,
                            "scope": scope,
                            "path": str(target),
                            "message": "definition saved; it will take effect on the next run",
                        },
                    )
                    return
                if self.path == "/agent/control":
                    try:
                        payload = self._read_object()
                        if (
                            set(payload) - {"agent_id", "action", "text"}
                            or not isinstance(payload.get("agent_id"), str)
                            or not payload["agent_id"]
                            or payload.get("action") not in ("pause", "resume", "redirect", "stop")
                            or ("text" in payload and not isinstance(payload["text"], str))
                            or (payload.get("action") == "redirect" and not str(payload.get("text", "")).strip())
                        ):
                            raise ProtocolError("agent/control requires agent_id, a valid action, and redirect text")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    try:
                        reply = host.run.control_agent(
                            payload["agent_id"], payload["action"], payload.get("text")
                        )
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                        return
                    except AgentControlError as exc:
                        self._json(HTTPStatus(exc.status), {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, reply)
                    return
                if self.path == "/steer":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"text"} or not isinstance(payload.get("text"), str) or not payload["text"].strip():
                            raise ProtocolError("steer requires one non-blank text string")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    try:
                        host.run.steer(payload["text"])
                    except RunActiveError:
                        self._json(HTTPStatus.CONFLICT, {"error": "no turn is running"})
                        return
                    self._json(HTTPStatus.OK, {"steered": True})
                    return
                if self.path == "/command":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"command"} or not isinstance(payload.get("command"), str) or not payload["command"].strip():
                            raise ProtocolError("command requires one non-blank command string")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    try:
                        reply = host.run.run_user_command(payload["command"])
                    except CommandActiveError:
                        self._json(HTTPStatus.CONFLICT, {"error": "a command is already running"})
                        return
                    except RunActiveError:
                        self._json(HTTPStatus.CONFLICT, {"error": "wait for the turn to finish"})
                        return
                    except ValueError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, reply)
                    return
                if self.path == "/changes/revert":
                    try:
                        payload = self._read_object()
                        fields = set(payload)
                        if (
                            "force" in payload and not isinstance(payload["force"], bool)
                        ) or fields - {"path", "key", "force"}:
                            raise ProtocolError("changes/revert accepts path or key and optional force")
                        has_path = isinstance(payload.get("path"), str) and bool(payload["path"].strip())
                        has_key = isinstance(payload.get("key"), str) and bool(payload["key"].strip())
                        if has_path == has_key or ("path" in payload and not has_path) or ("key" in payload and not has_key):
                            raise ProtocolError("changes/revert requires one non-empty path or key")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    try:
                        reply = host.run.revert_changes(
                            path=payload.get("path"),
                            key=payload.get("key"),
                            force=payload.get("force", False),
                        )
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                        return
                    except ChangedOutsideError as exc:
                        self._json(
                            HTTPStatus.CONFLICT,
                            {"error": str(exc), "paths": exc.paths},
                        )
                        return
                    except KeyError:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown change"})
                        return
                    except (OSError, ValueError) as exc:
                        self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, reply)
                    return
                if self.path in ("/worktree/apply", "/worktree/discard"):
                    try:
                        payload = self._read_object()
                        name = payload.get("name")
                        if set(payload) != {"name"} or not isinstance(name, str) or not name.strip() or Path(name).name != name or name in (".", ".."):
                            raise ProtocolError("worktree action requires one valid name")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    try:
                        if self.path == "/worktree/apply":
                            reply = host.run.apply_worktree(name)
                        else:
                            reply = host.run.discard_worktree(name)
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                        return
                    except KeyError:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown worktree"})
                        return
                    except WorktreeApplyConflict as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                        return
                    except (OSError, ValueError, RuntimeError) as exc:
                        self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, reply)
                    return
                if credential_route:
                    try:
                        payload = self._read_object()
                    except ProtocolError:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid credential request"})
                        return
                    name = payload.get("name")
                    value = payload.get("value")
                    if name not in (
                        ANTHROPIC_KEY_ENV_VAR,
                        GEMINI_KEY_ENV_VAR,
                        OPENAI_KEY_ENV_VAR,
                        *(variable for _, variable, _ in PROVIDERS[3:]),
                        *(endpoint.api_key_env_var for endpoint in search_endpoints()),
                    ):
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
                if self.path == "/session/rename":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"run_id", "title"} or any(
                            not isinstance(payload.get(key), str) for key in ("run_id", "title")
                        ):
                            raise ProtocolError("session/rename requires run_id and title strings")
                        run_id = payload["run_id"]
                        title = payload["title"].strip()
                        if not run_id or run_id in (".", "..") or Path(run_id).name != run_id or not title or len(title) > 200:
                            raise ProtocolError("title must be non-blank and at most 200 characters")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    from symphonai_api.session import SessionStore
                    if not (host.run.sessions_root / run_id).is_dir():
                        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown session"})
                        return
                    if not host._session_belongs_here(run_id):
                        self._json(HTTPStatus.CONFLICT, {"error": "that conversation belongs to another project"})
                        return
                    with host.run._lock:
                        opened = host.run._open_conversations.get(run_id)
                        session_store = opened[1] if opened is not None else SessionStore.open(host.run.sessions_root, run_id)
                        try:
                            meta = session_store.read_meta()
                            meta["title"] = title
                            session_store.write_meta(meta)
                        finally:
                            if opened is None:
                                session_store.close()
                    self._json(HTTPStatus.OK, {"renamed": True})
                    return
                if self.path == "/session/delete":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"run_id"} or not isinstance(payload.get("run_id"), str) or not payload["run_id"] or payload["run_id"] in (".", "..") or Path(payload["run_id"]).name != payload["run_id"]:
                            raise ProtocolError("session/delete requires a run_id string")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    if not host._session_belongs_here(payload["run_id"]):
                        self._json(HTTPStatus.CONFLICT, {"error": "that conversation belongs to another project"})
                        return
                    try:
                        deleted = host.run.delete_session(payload["run_id"])
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK if deleted else HTTPStatus.NOT_FOUND, {"deleted": deleted})
                    return
                if self.path == "/session/fork":
                    try:
                        payload = self._read_object()
                        if set(payload) - {"run_id", "record_id", "force"} or any(
                            type(payload[key]) is not str or not payload[key]
                            for key in ("run_id", "record_id")
                        ) or ("force" in payload and type(payload["force"]) is not bool):
                            raise ProtocolError("session/fork requires run_id and record_id strings")
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    if not host._session_belongs_here(payload["run_id"]):
                        self._json(HTTPStatus.CONFLICT, {"error": "that conversation belongs to another project"})
                        return
                    try:
                        reply = host.run.fork_session(
                            payload["run_id"], payload["record_id"],
                            force=payload.get("force", False),
                        )
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc), "run_id": exc.run_id})
                        return
                    except ChangedOutsideError as exc:
                        self._json(
                            HTTPStatus.CONFLICT,
                            {"error": str(exc), "paths": exc.paths},
                        )
                        return
                    except SessionError as exc:
                        self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                        return
                    except TranscriptError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    except ProviderSelectionError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, reply)
                    return
                if self.path == "/provider":
                    try:
                        choice = self._read_object()
                        if not isinstance(choice.get("name"), str) or set(choice) - {"name", "model", "base_url", "effort"}:
                            raise ProviderSelectionError("unknown provider option")
                        effort = choice.get("effort")
                        if effort is not None and (not isinstance(effort, str) or not effort.strip()):
                            raise ProviderSelectionError("effort must be a non-empty string")
                        provider = _provider(choice.get("name"), choice.get("model"), choice.get("base_url"))
                        if provider is None:
                            raise ProviderSelectionError("choose a provider")
                    except (ProtocolError, ProviderSelectionError) as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    if host.run is not None:
                        try:
                            host.run.select_provider(
                                provider,
                                choice.get("model"),
                                effort,
                                choice,
                            )
                        except RunActiveError as exc:
                            self._json(HTTPStatus.CONFLICT, {"error": str(exc), "run_id": exc.run_id})
                            return
                    try:
                        _save_provider_selection(choice)
                    except OSError:
                        self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "provider selection could not be saved"})
                        return
                    host._default_provider = provider
                    host._provider_selection = dict(choice)
                    self._json(HTTPStatus.OK, {"selected": True})
                    return
                if self.path == "/mode":
                    try:
                        payload = self._read_object()
                        if set(payload) != {"mode"}:
                            permitted = ", ".join(host.run.permitted_modes()) or "none"
                            raise ModeSelectionError(f"permitted modes: {permitted}")
                        mode = host.run.select_mode(payload["mode"])
                    except (ProtocolError, ModeSelectionError) as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    self._json(HTTPStatus.OK, {"mode": mode})
                    return
                if self.path == "/compact":
                    try:
                        payload = self._read_object()
                        if (
                            set(payload) - {"instructions"}
                            or (
                                "instructions" in payload
                                and not isinstance(payload["instructions"], str)
                            )
                        ):
                            raise ProtocolError(
                                "compact accepts only an optional string instructions"
                            )
                    except ProtocolError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    try:
                        result = host.run.compact(payload.get("instructions"))
                    except RunActiveError as exc:
                        self._json(
                            HTTPStatus.CONFLICT,
                            {"error": str(exc), "run_id": exc.run_id},
                        )
                        return
                    except NoConversationError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    self._json(
                        HTTPStatus.OK,
                        {
                            "changed": result.changed,
                            "before_tokens": result.before_tokens,
                            "after_tokens": result.after_tokens,
                            "dropped_messages": result.dropped_messages,
                        },
                    )
                    return
                kind = self.path.removeprefix("/")
                try:
                    request = decode_request(kind, self._read_object())
                except ProtocolError as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                if kind == "prompt":
                    try:
                        run_id = host.run.start(request.prompt, attachments=request.attachments)
                    except ProviderSelectionError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    except CommandActiveError:
                        self._json(HTTPStatus.CONFLICT, {"error": "wait for the command to finish"})
                        return
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc), "run_id": exc.run_id})
                        return
                    self._json(HTTPStatus.OK, {"accepted": True, "run_id": run_id})
                    return
                if kind == "approval":
                    try:
                        resolved = host.resolve_approval(
                            request.approval_id,
                            allowed=request.allowed,
                            reason=request.reason,
                            remember=request.remember,
                            answer=request.answer,
                        )
                    except InvalidApprovalAnswer as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    if not resolved:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown approval id"})
                        return
                    self._json(HTTPStatus.OK, {"resolved": True})
                    return
                if kind == "session/open":
                    if not host._session_belongs_here(request.run_id):
                        self._json(HTTPStatus.CONFLICT, {"error": "that conversation belongs to another project"})
                        return
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
