"""Loopback-only HTTP/SSE boundary for a single SymphonAI host run."""

from __future__ import annotations

from symphonai_api.paths import utc_iso

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import secrets
import shlex
import sys
import threading
import time
import tempfile
import tomllib
import re
from collections.abc import Mapping
from collections.abc import Callable
from http.cookies import CookieError, SimpleCookie
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from symphonai_api.agent_memory import MemoryUnavailable
from symphonai_api.agent_file import AgentFileError, load_agent_file
from symphonai_api.events import Event
from symphonai_api.model_discovery import list_models
from symphonai_api.model_table import capability_for_provider
from symphonai_api.permissions import _contains_path, display_path
from symphonai_api.providers.base import ModelProvider, ProviderError
from symphonai_api.session import SessionError, SessionStore, TranscriptError
from symphonai_api.paths import symphonai_home, write_atomic
from symphonai_api.web_search import search_endpoint, search_endpoints
from symphonai_host.git import read_git_status
from symphonai_host.git_actions import git_status, git_diff, git_commit, git_pull_request_info, create_pull_request
from symphonai_host.providers import PROVIDERS, provider as build_provider, save_provider_selection, app_root as web_app_root
from symphonai_host.broker import EventBroker, Subscription
from symphonai_host.approvals import InvalidApprovalAnswer, PermissionGrantError
from symphonai_host.credentials import CredentialError, store
from symphonai_host.files import repository_files, repository_tree
from symphonai_host.protocol import (
    ApprovalReply,
    ApprovalRequested,
    FilesChanged,
    HistoryMessage,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_request,
    encode_event,
    encode_frame,
)
from symphonai_host.run import (
    ChangedOutsideError,
    CommandActiveError,
    HostRun,
    TerminalOutput,
    ModeSelectionError,
    NoConversationError,
    ProviderSelectionError,
    RunActiveError,
    WorktreeApplyConflict,
)
from symphonai_host.terminal import sessions_for
from symphonai_host.sessions import list_sessions, prompt_history
from symphonai_host.usage import UsageReport
from symphonai_host.projects import ProjectRuntime, record_recent_project, recent_projects
from symphonai_host.loops import LoopStore
from symphonai_host.roadmap import InvalidRoadmap, RoadmapStore
from symphonai_api.worktree import WorktreeError


MAX_FILE_BYTES = 1024 * 1024
FILES_CHANGED_LIMIT = 100
FILES_CHANGED_POLL_SECONDS = 0.25
FILES_CHANGED_SETTLE_SECONDS = 1.0
APP_CONTENT_TYPES = {
    ".css": "text/css",
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
}
APP_HANDSHAKE_MARKER = "<!-- symphonai-handshake -->"
APP_COOKIE_NAME = "symphonai_app"


def _working_tree_fingerprint(root: Path) -> dict[str, tuple[str, int, int, int, int, int]] | None:
    snapshot = read_git_status(root)
    if snapshot is None:
        return None
    root = snapshot.root
    paths = [(path, entry.code) for entry in snapshot.entries
             for path in (entry.path, entry.old_path) if path is not None]
    fingerprint = {}
    for relative, status in paths:
        try:
            stat_result = (root / relative).lstat()
            metadata = (
                status, stat_result.st_size, stat_result.st_mtime_ns,
                stat_result.st_ctime_ns, stat_result.st_mode, stat_result.st_ino,
            )
        except FileNotFoundError:
            metadata = (status, -1, -1, -1, -1, -1)
        fingerprint[relative] = metadata
    return fingerprint


def _models_with_efforts(provider: str, models: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    return [{"id": model, "efforts": [option.id for option in capability.efforts] if
             (capability := capability_for_provider(provider, model)) is not None else None}
            for model in models]


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
    written_at = utc_iso(entry.written_at)
    return {
        "id": entry.id,
        "text": entry.text,
        "written_at": written_at,
        "agent_name": entry.agent_name,
    }


@dataclass(frozen=True)
class _Reply:
    status: int
    body: Any


class _ReplyReady(Exception):
    def __init__(self, reply: _Reply):
        self.reply = reply


@dataclass(frozen=True)
class _StreamReply:
    send: Callable[[], None]


@dataclass(frozen=True)
class _Payload:
    fields: Mapping[str, type | tuple[type, ...]]
    required: tuple[str, ...]
    extra: bool = False

    def validate(self, payload: dict, path: str) -> None:
        for key in self.required:
            if key not in payload:
                raise ProtocolError(f"{path} missing required key {key}")
        for key, value in payload.items():
            if key not in self.fields:
                if not self.extra:
                    raise ProtocolError(f"{path} unknown key {key}")
                continue
            declared = self.fields[key]
            types = declared if isinstance(declared, tuple) else (declared,)
            if not isinstance(value, types) or (isinstance(value, bool) and int in types and bool not in types):
                names = " or ".join(kind.__name__ for kind in types)
                raise ProtocolError(f"{path} requires {key} as {names}")


@dataclass(frozen=True)
class _Route:
    method: str
    path: str
    handler: Callable
    match: str = "path"
    auth: bool = True
    app_query: bool = False
    app_cookie: bool = False
    project_free: bool = False
    payload: _Payload | None = None
    # Each phase preserves the original catch boundary and exception order.
    errors: tuple[tuple[tuple[type[Exception] | tuple[type[Exception], ...], str], ...], ...] = ()
    helper_errors: Mapping[str, tuple] = field(default_factory=dict)
    action: str = ""

    def matches_path(self, path: str) -> bool:
        return path.startswith(self.path[:-1]) if self.match == "prefix" else path == self.path

    def matches(self, target: str) -> bool:
        return target == self.path if self.match == "exact" else self.matches_path(urlsplit(target).path)


class HostServer:
    """Own a loopback HTTP server, event broker, and one active runtime run."""

    def __init__(
        self,
        provider: ModelProvider | None,
        runtime: ProjectRuntime | None,
        *,
        token: str | None = None,
        broker: EventBroker | None = None,
        keepalive_seconds: float = 15.0,
        home: Path | None = None,
        project_factory: Callable[[Path, EventBroker, Callable], ProjectRuntime] | None = None,
        provider_selection: Mapping[str, str] | None = None,
    ) -> None:
        if keepalive_seconds <= 0:
            raise ValueError("keepalive_seconds must be greater than 0")
        self.token = token or secrets.token_urlsafe(32)
        self._current_root = (
            runtime.root.resolve() if runtime is not None else None
        )
        self._repo_root = self._current_root
        self._project_factory = project_factory
        self._default_provider = provider
        self._provider_selection = dict(provider_selection or ({
            "name": provider.name,
            **({"model": provider.model} if provider.model else {}),
        } if provider is not None else {}))
        self._home = symphonai_home(home)
        self._config_home = home
        self._extensions_lock = threading.Lock()
        self._model_cache: dict[tuple[str, str | None], tuple[str, ...]] = {}
        self._model_cache_lock = threading.Lock()
        self._usage_report = UsageReport()
        self._mcp_auth_flows = {}
        self.broker = broker or EventBroker()
        initial_runtime = runtime
        initial_run = None if initial_runtime is None else initial_runtime.run
        self._runtimes: dict[Path, ProjectRuntime] = {}
        self._runtimes_lock = threading.RLock()
        self._event_session_ids: dict[str, str] = {}
        if initial_runtime is not None and self._current_root is not None:
            initial_runtime.run.enable_roadmap_tools()
            self._runtimes[self._current_root] = initial_runtime
        self._watch_stop = threading.Event()
        self._event_watch: threading.Thread | None = None
        self._event_watch_subscription: Subscription | None = None
        self._file_watch: threading.Thread | None = None
        self.keepalive_seconds = keepalive_seconds
        self._handshake_printed = False
        self._thread: threading.Thread | None = None
        self._serving = False
        self._close_lock = threading.Lock()
        self._closed = False
        handler = self._handler_type()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._httpd.daemon_threads = True
        if self._project_factory is not None:
            self._event_watch = threading.Thread(target=self._watch_project_events, daemon=True)
            self._event_watch.start()
        self._file_watch = threading.Thread(target=self._watch_files, daemon=True)
        self._file_watch.start()
        if initial_run is not None:
            initial_run.loops.recover_loops()

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
                diff=approval.diff,
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


    def open_project(self, root: Path) -> tuple[ProjectRuntime | None, bool]:
        resolved = root.resolve()
        if self._project_factory is None:
            raise RuntimeError("this host cannot open other projects")
        with self._runtimes_lock:
            if self._closed:
                raise RuntimeError("host is closing")
            runtime = self._runtimes.get(resolved)
            created = runtime is None
            if runtime is None:
                current = self.run
                selection = current.selection() if current is not None else {
                    "model": self._provider_selection.get("model"),
                    "effort": self._provider_selection.get("effort"),
                }
                provider = current.provider if current is not None else self._default_provider
                runtime = self._project_factory(resolved, self.broker, self._publish_approval)
                try:
                    runtime.run.enable_roadmap_tools()
                    if provider is not None:
                        runtime.run.select_provider(
                            provider, selection.get("model"), selection.get("effort"),
                        )
                    runtime.run.loops.recover_loops()
                    if self._closed:
                        raise RuntimeError("host is closing")
                except BaseException:
                    runtime.close()
                    raise
                self._runtimes[resolved] = runtime
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

    def _watch_files(self) -> None:
        watched_root: Path | None = None
        fingerprint: dict[str, tuple[str, int, int, int, int, int]] | None = None
        pending_paths: set[str] = set()
        last_change = 0.0
        last_poll = 0.0
        while not self._watch_stop.wait(FILES_CHANGED_POLL_SECONDS):
            watcher_count = int(
                self._event_watch_subscription is not None
                and not self._event_watch_subscription.closed
            )
            connected = self.broker.subscriber_count > watcher_count
            with self._runtimes_lock:
                root = self._current_root
            root = root if connected else None
            if root is None:
                watched_root = None
                fingerprint = None
                pending_paths.clear()
                continue
            if root != watched_root:
                watched_root = root
                fingerprint = _working_tree_fingerprint(root)
                pending_paths.clear()
                last_poll = time.monotonic()
                continue
            now = time.monotonic()
            if now - last_poll < FILES_CHANGED_POLL_SECONDS:
                continue
            current = _working_tree_fingerprint(root)
            now = time.monotonic()
            last_poll = now
            if current is not None and fingerprint is not None:
                changed = {
                    path for path in fingerprint.keys() | current.keys()
                    if fingerprint.get(path) != current.get(path)
                }
                pending_paths.update(changed)
                if changed:
                    last_change = now
                fingerprint = current
            if pending_paths and now - last_change >= FILES_CHANGED_SETTLE_SECONDS:
                changed_paths = sorted(pending_paths)
                self.broker.publish(FilesChanged(
                    agent_id="", run_id="", paths=changed_paths[:FILES_CHANGED_LIMIT],
                    truncated=len(changed_paths) > FILES_CHANGED_LIMIT,
                ))
                pending_paths.clear()

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
        if self._file_watch is not None:
            self._file_watch.join(timeout=1)
        if self._serving:
            self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        host = self

        def reply_bad_request(exc, name=None):
            return _Reply(HTTPStatus.BAD_REQUEST, {'error': str(exc)})

        def reply_conflict_tail(exc, name=None):
            return _Reply(HTTPStatus.CONFLICT, {'error': str(exc)[-4000:]})

        def reply_conflict(exc, name=None):
            return _Reply(HTTPStatus.CONFLICT, {'error': str(exc)})

        def reply_invalid_roadmap(exc, name=None):
            return _Reply(HTTPStatus.CONFLICT, {'error': f'invalid roadmap: {exc}'})

        def reply_model_unavailable(exc, name=None):
            detail = str(exc)
            variable = next((key for vendor, key, _ in PROVIDERS if vendor == name))
            secret = os.environ.get(variable, '').strip()
            if secret:
                detail = detail.replace(secret, '[redacted]')
            return _Reply(HTTPStatus.OK, {'provider': name, 'state': 'unknown', 'models': [], 'detail': detail, 'filter': _model_listing(host, name, [])[1]})

        def reply_not_found(exc, name=None):
            return _Reply(HTTPStatus.NOT_FOUND, {'error': 'not found'})

        def reply_unreadable_directory(exc, name=None):
            return _Reply(HTTPStatus.FORBIDDEN, {'error': 'directory is not readable'})

        def reply_unavailable(exc, name=None):
            return _Reply(HTTPStatus.SERVICE_UNAVAILABLE, {'error': str(exc)})

        def reply_unexportable_session(exc, name=None):
            return _Reply(HTTPStatus.UNPROCESSABLE_ENTITY, {'error': 'session transcript cannot be exported'})

        def reply_empty_unavailable(exc, name=None):
            return _Reply(HTTPStatus.SERVICE_UNAVAILABLE, None)

        def reply_active_run(exc, name=None):
            return _Reply(HTTPStatus.CONFLICT, {'error': str(exc), 'run_id': exc.run_id})

        def reply_no_goal(exc, name=None):
            return _Reply(HTTPStatus.NOT_FOUND, {'error': 'no goal'})

        def reply_definition_write_failed(exc, name=None):
            return _Reply(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': f'could not write definition: {exc}'})

        def reply_no_running_turn(exc, name=None):
            return _Reply(HTTPStatus.CONFLICT, {'error': 'no turn is running'})

        def reply_command_running(exc, name=None):
            return _Reply(HTTPStatus.CONFLICT, {'error': 'a command is already running'})

        def reply_turn_running(exc, name=None):
            return _Reply(HTTPStatus.CONFLICT, {'error': 'wait for the turn to finish'})

        def reply_changed_outside(exc, name=None):
            return _Reply(HTTPStatus.CONFLICT, {'error': str(exc), 'paths': exc.paths})

        def reply_unknown_change(exc, name=None):
            return _Reply(HTTPStatus.NOT_FOUND, {'error': 'unknown change'})

        def reply_internal_error(exc, name=None):
            return _Reply(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': str(exc)})

        def reply_unknown_worktree(exc, name=None):
            return _Reply(HTTPStatus.NOT_FOUND, {'error': 'unknown worktree'})

        def reply_credential_unavailable(exc, name=None):
            return _Reply(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': f'credential store unavailable for {name}'})

        def reply_missing_session(exc, name=None):
            return _Reply(HTTPStatus.NOT_FOUND, {'error': str(exc)})

        def reply_selection_unsaved(exc, name=None):
            return _Reply(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': 'provider selection could not be saved'})

        def reply_command_finishing(exc, name=None):
            return _Reply(HTTPStatus.CONFLICT, {'error': 'wait for the command to finish'})

        def reply_session_not_found(exc, name=None):
            return _Reply(HTTPStatus.NOT_FOUND, {'error': 'session not found'})

        def reply_empty_forbidden(exc, name=None):
            return _Reply(HTTPStatus.FORBIDDEN, None)

        def reply_empty_not_found(exc, name=None):
            return _Reply(HTTPStatus.NOT_FOUND, None)

        def reply_empty_unsupported(exc, name=None):
            return _Reply(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, None)

        error_replies = {
            "empty_forbidden": reply_empty_forbidden,
            "empty_not_found": reply_empty_not_found,
            "empty_unsupported": reply_empty_unsupported,
            'bad_request': reply_bad_request,
            'conflict_tail': reply_conflict_tail,
            'conflict': reply_conflict,
            'invalid_roadmap': reply_invalid_roadmap,
            'model_unavailable': reply_model_unavailable,
            'not_found': reply_not_found,
            'unreadable_directory': reply_unreadable_directory,
            'unavailable': reply_unavailable,
            'unexportable_session': reply_unexportable_session,
            'empty_unavailable': reply_empty_unavailable,
            'active_run': reply_active_run,
            'no_goal': reply_no_goal,
            'definition_write_failed': reply_definition_write_failed,
            'no_running_turn': reply_no_running_turn,
            'command_running': reply_command_running,
            'turn_running': reply_turn_running,
            'changed_outside': reply_changed_outside,
            'unknown_change': reply_unknown_change,
            'internal_error': reply_internal_error,
            'unknown_worktree': reply_unknown_worktree,
            'credential_unavailable': reply_credential_unavailable,
            'missing_session': reply_missing_session,
            'selection_unsaved': reply_selection_unsaved,
            'command_finishing': reply_command_finishing,
            'session_not_found': reply_session_not_found,
        }

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
                if secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
                    return True
                if allow_app_query:
                    values = parse_qs(
                        urlsplit(self.path).query,
                        keep_blank_values=True,
                    ).get("token", [])
                    if len(values) == 1 and secrets.compare_digest(
                        values[0].encode("utf-8"), host.token.encode("utf-8")
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
                        credential.value.encode("utf-8"), host.token.encode("utf-8")
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
                app_root = web_app_root()
                if not app_root.is_dir():
                    self._json(
                        HTTPStatus.NOT_FOUND,
                        {"error": "app is not installed"},
                    )
                    return None
                with self._errors('app_path'):
                    resolved = (app_root / candidate).resolve()
                if not _contains_path(app_root, resolved):
                    self._empty(HTTPStatus.FORBIDDEN)
                    return None
                return resolved

            def _serve_app_index(self) -> None:
                resolved = self._app_path("index.html")
                if resolved is None:
                    return
                with self._errors('app_index'):
                    source = resolved.read_text(encoding="utf-8")
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
                with self._errors('app_asset'):
                    body = resolved.read_bytes()
                self._bytes(HTTPStatus.OK, body, content_type)

            def _requested_path(self, *, directory: bool) -> tuple[str, Path] | None:
                values = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                paths = values.get("path", [])
                if len(paths) > 1 or (not directory and not paths):
                    self._empty(HTTPStatus.FORBIDDEN)
                    return None
                requested = paths[0] if paths else ""
                candidate = Path(requested)
                if candidate.is_absolute() or ".." in candidate.parts:
                    self._empty(HTTPStatus.FORBIDDEN)
                    return None
                with self._errors('requested_path'):
                    resolved = (host._repo_root / candidate).resolve()
                if not host.run.policy.check_read(resolved).allowed:
                    self._empty(HTTPStatus.FORBIDDEN)
                    return None
                return requested, resolved

            def _serve_tree(self) -> None:
                target = self._requested_path(directory=True)
                if target is None:
                    return
                requested, resolved = target
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
                target = self._requested_path(directory=False)
                if target is None:
                    return
                requested, resolved = target
                with self._errors('source_read'):
                    with resolved.open("rb") as source:
                        data = source.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    self._empty(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                    return
                with self._errors('source_decode'):
                    text = data.decode("utf-8")
                self._json(HTTPStatus.OK, {"path": requested, "text": text})

            def _definition_target(
                self,
                name: object,
                scope: object,
            ) -> tuple[str, str, Path]:
                if not isinstance(name, str) or not name:
                    raise _ReplyReady(_Reply(HTTPStatus.BAD_REQUEST, {'error': 'name must be a non-empty string'}))
                if not isinstance(scope, str) or scope not in ("project", "user"):
                    raise _ReplyReady(_Reply(HTTPStatus.BAD_REQUEST, {'error': "scope must be 'project' or 'user'"}))
                if any(separator in name for separator in ("/", "\\")):
                    raise _ReplyReady(_Reply(HTTPStatus.BAD_REQUEST, {'error': 'name must be a single segment'}))
                if name in (".", ".."):
                    raise _ReplyReady(_Reply(HTTPStatus.BAD_REQUEST, {'error': 'name must not be a parent reference'}))
                if name.endswith(".toml"):
                    stem = name[:-5]
                elif Path(name).suffix:
                    raise _ReplyReady(_Reply(HTTPStatus.BAD_REQUEST, {'error': 'name must use the .toml extension'}))
                else:
                    stem = name
                if not stem or stem in (".", "..") or Path(stem).name != stem:
                    raise _ReplyReady(_Reply(HTTPStatus.BAD_REQUEST, {'error': 'name must be a single segment'}))
                filename = f"{stem}.toml"
                directory = (
                    host._repo_root / ".symphonai" / "agents"
                    if scope == "project"
                    else host._home / "agents"
                )
                with self._errors('definition_target'):
                    base = (
                        Path(host._repo_root).resolve()
                        if scope == "project"
                        else host._home.resolve()
                    )
                    root = directory.resolve()
                    target = (root / filename).resolve(strict=False)
                if not _contains_path(base, root) or not _contains_path(root, target):
                    raise _ReplyReady(_Reply(HTTPStatus.FORBIDDEN, None))
                return stem, scope, target

            def _definition_trusted(self, scope: str) -> bool:
                if scope == "user":
                    return True
                extensions = host.run.extensions
                return extensions is not None and extensions.trust.allows(
                    host._repo_root,
                    "agents",
                )

            def _definition_fields(self, text: str) -> dict:
                data = tomllib.loads(text)
                model = data.get("model", {})
                return {
                    "prompt": data.get("prompt", ""),
                    "tools": data.get("tools", []),
                    "deny_tools": data.get("deny_tools", []),
                    **{key: model.get(key, "") for key in ("provider", "model", "effort")},
                }

            def _edit_definition_fields(self, text: str, fields: dict) -> str:
                current = self._definition_fields(text)
                for key, value in fields.items():
                    if key not in current:
                        raise ValueError("unknown agent field: " + key)
                    valid = (
                        isinstance(value, list) and all(isinstance(item, str) for item in value)
                        if key in ("tools", "deny_tools") else isinstance(value, str)
                    )
                    if not valid:
                        raise ValueError("invalid agent field: " + key)
                changes = {key: value for key, value in fields.items() if value != current[key]}
                lines = text.splitlines(keepends=True)
                ending = "\r\n" if "\r\n" in text else "\n"
                table = ()
                root_end = len(lines)
                model_end = None
                edits = []
                index = 0
                while index < len(lines):
                    line = lines[index]
                    if line.lstrip().startswith("["):
                        header = tomllib.loads(line)
                        table = ()
                        while isinstance(header, dict) and len(header) == 1:
                            key, header = next(iter(header.items()))
                            table += (key,)
                        root_end = min(root_end, index)
                        if model_end is not None and table != ("model",):
                            model_end = min(model_end, index)
                        if table == ("model",):
                            model_end = len(lines)
                        index += 1
                        continue
                    match = re.match(r"(\s*([^=]+?)\s*=\s*)(.*)", line)
                    if match is None or line.lstrip().startswith("#"):
                        index += 1
                        continue
                    end = index + 1
                    while True:
                        statement = "".join(lines[index:end])
                        try:
                            parsed = tomllib.loads(statement)
                            break
                        except tomllib.TOMLDecodeError:
                            if end == len(lines):
                                raise
                            end += 1
                    path = table
                    while isinstance(parsed, dict) and len(parsed) == 1:
                        key, parsed = next(iter(parsed.items()))
                        path += (key,)
                    field = path[-1] if path in (("prompt",), ("tools",), ("deny_tools",),
                        ("model", "provider"), ("model", "model"), ("model", "effort")) else None
                    if field in changes:
                        value = changes.pop(field)
                        comment = ""
                        for offset, character in enumerate(statement):
                            if character != "#":
                                continue
                            try:
                                tomllib.loads(statement[:offset])
                            except tomllib.TOMLDecodeError:
                                continue
                            comment = " " + statement[offset:].rstrip("\r\n")
                            break
                        suffix = ending if lines[end - 1].endswith("\n") else ""
                        replacement = match[1] + json.dumps(value, ensure_ascii=False) + comment + suffix
                        if field != "prompt" and not value:
                            replacement = comment.lstrip() + suffix if comment else ""
                        edits.append((index, end, replacement))
                    index = end
                additions = {}
                for key, value in changes.items():
                    if not value and key != "prompt":
                        continue
                    root = key in ("prompt", "tools", "deny_tools")
                    location = root_end if root else (model_end if model_end is not None else len(lines))
                    additions.setdefault(location, []).append((root, key, value))
                for location, values in additions.items():
                    replacement = ""
                    if location and not lines[location - 1].endswith("\n"):
                        replacement += ending
                    for root, key, value in sorted(values, key=lambda item: not item[0]):
                        if not root and model_end is None:
                            replacement += "[model]" + ending
                            model_end = location
                        replacement += key + " = " + json.dumps(value, ensure_ascii=False) + ending
                    edits.append((location, location, replacement))
                for start, end, replacement in sorted(edits, reverse=True):
                    lines[start:end] = [replacement]
                return "".join(lines)

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
                write_atomic(target, text, mode=0o600, fsync=True)

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
                if not self._definition_trusted(scope):
                    self._json(
                        HTTPStatus.FORBIDDEN,
                        {"error": "repository not trusted for agents"},
                    )
                    return
                with self._errors('definition_read'):
                    data = target.read_bytes()
                with self._errors('definition_decode'):
                    text = data.decode("utf-8")
                try:
                    fields = self._definition_fields(text)
                except (tomllib.TOMLDecodeError, AttributeError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._json(
                    HTTPStatus.OK,
                    {"name": stem, "scope": scope, "text": text, "fields": fields},
                )

            def _read_object(self) -> dict:
                try:
                    # Ten 5 MB attachments expand to ~66.7 MB of base64; leave
                    # headroom for prompt text and JSON metadata (80 MiB total).
                    maximum = 80 * 1024 * 1024
                    lengths = self.headers.get_all("Content-Length", [])
                    if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
                        self.close_connection = True
                        raise ProtocolError("a non-negative Content-Length is required")
                    length = int(lengths[0])
                    if length > maximum:
                        self.close_connection = True
                        raise ProtocolError(f"request body exceeds {maximum} bytes")
                    data = json.loads(self.rfile.read(length).decode("utf-8"))
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProtocolError(f"invalid JSON payload: {exc}") from None
                if not isinstance(data, dict):
                    raise ProtocolError("request payload must be an object")
                return data

            def _not_found(self) -> None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})


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


            @contextmanager
            def _errors(self, phase, **context):
                try:
                    yield
                except Exception as exc:
                    rules = (
                        self._route.helper_errors[phase]
                        if isinstance(phase, str) else self._route.errors[phase]
                    )
                    for exception_type, profile in rules:
                        if isinstance(exc, exception_type):
                            raise _ReplyReady(error_replies[profile](exc, **context)) from None
                    raise

            def _dispatch(self, method):
                request_path = urlsplit(self.path).path
                route = next((route for route in routes if route.method == method and route.matches(self.path)), None)
                # GET's project gate historically precedes matching, including unknown paths.
                project_free = any(route.method == method and route.project_free and route.matches_path(request_path) for route in routes)
                if method == "GET" and host.run is None and not project_free:
                    if self._authorized():
                        self._json(HTTPStatus.CONFLICT, {"error": "open a project first"})
                    return
                # POST historically rejects unknown paths before authentication or project checks.
                if route is None:
                    self._not_found()
                    return
                if route.auth and not self._authorized(allow_app_query=route.app_query, allow_app_cookie=route.app_cookie):
                    return
                if method == "POST" and host.run is None and not (route.project_free and self.path == route.path):
                    self._json(HTTPStatus.CONFLICT, {"error": "open a project first"})
                    return
                self._route = route
                try:
                    try:
                        payload = self._read_object() if route.payload is not None else None
                    except ProtocolError as exc:
                        raise _ReplyReady(_Reply(HTTPStatus.BAD_REQUEST, {"error": f"{route.path}: {exc}"})) from None
                    if route.payload is not None:
                        try:
                            route.payload.validate(payload, route.path)
                        except ProtocolError as exc:
                            raise _ReplyReady(_Reply(HTTPStatus.BAD_REQUEST, {"error": str(exc)})) from None
                    reply = route.handler(self, payload)
                    if isinstance(reply, _StreamReply):
                        reply.send()
                        reply = None
                except _ReplyReady as ready:
                    reply = ready.reply
                if reply is not None:
                    if reply.body is None:
                        self._empty(reply.status)
                    else:
                        self._json(reply.status, reply.body)

            def do_GET(self):
                self._dispatch("GET")

            def do_POST(self):
                self._dispatch("POST")

            def _get_health(self, payload):
                return _Reply(HTTPStatus.OK, {'protocol_version': PROTOCOL_VERSION, 'state': 'active' if host.run is not None and host.run.active else 'idle', 'run_id': None if host.run is None else host.run.active_run_id, 'runtime_run_id': None if host.run is None else host.run.runtime_run_id})

            def _post_preview_start(self, payload):
                try:
                    reply = host.run.start_preview(payload.get('url', ''))
                except (ValueError, OSError):
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'Could not start preview: use a local http URL'})
                return _Reply(HTTPStatus.OK, reply)

            def _post_preview_stop(self, payload):
                return _Reply(HTTPStatus.OK, host.run.stop_preview())

            def _get_preview_marks(self, payload):
                return _Reply(HTTPStatus.OK, host.run.preview_marks())

            def _post_preview_mark(self, payload):
                if 'remove' in payload:
                    # Preview credentials grant creation only, never composer control.
                    if not self._authorized():
                        return None
                    ids, session_id = payload.get('remove'), payload.get('session_id')
                    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids) or not (session_id is None or isinstance(session_id, str)):
                        return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'Invalid mark removal'})
                    return _Reply(HTTPStatus.OK, host.run.remove_preview_marks(ids, session_id))
                try:
                    mark = host.run.add_preview_mark(self.headers.get('X-Preview-Token', ''), payload)
                except PermissionError:
                    return _Reply(HTTPStatus.FORBIDDEN, {'error': 'Invalid preview token'})
                except (ValueError, TypeError):
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'Invalid preview mark'})
                return _Reply(HTTPStatus.OK, mark)

            def _get_usage(self, payload):
                values = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                try:
                    days = int(values.get('days', ['30'])[0])
                    if not 1 <= days <= 90:
                        raise ValueError()
                except ValueError:
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'days must be an integer from 1 to 90'})
                report = host._usage_report.read(
                    host.run.sessions_root, days=days, price_table=host.run.price_table,
                )
                return _Reply(HTTPStatus.OK, report)

            def _get_conversation(self, payload):
                return _Reply(HTTPStatus.OK, {'conversation': host.run.conversation_stats()})

            def _get_git_status(self, payload):
                return _Reply(HTTPStatus.OK, git_status(host.run.policy.repo_root))

            def _get_git_diff(self, payload):
                request_url = urlsplit(self.path)
                values = parse_qs(request_url.query, keep_blank_values=True)
                path = values.get('path', [''])[0]
                with self._errors(0):
                    result = git_diff(host.run.policy.repo_root, path)
                if result is None:
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'not found'})
                return _Reply(HTTPStatus.OK, result)

            def _get_jobs(self, payload):
                return _Reply(HTTPStatus.OK, {'jobs': host.run.list_jobs()})

            def _get_git_pull_request(self, payload):
                with self._errors(0):
                    info = git_pull_request_info(host.run.policy.repo_root)
                return _Reply(HTTPStatus.OK, info)

            def _get_changes(self, payload):
                with self._errors(0):
                    reply = host.run.changes()
                return _Reply(HTTPStatus.OK, reply)

            def _get_memory(self, payload):
                memory = host.run.memory_store()
                if memory is None:
                    return _Reply(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': 'memory is unavailable'})
                reply = {}
                for scope in ('project', 'global'):
                    entries, unreadable = memory.read_status(scope)
                    reply[scope] = {'entries': [], 'error': 'unreadable'} if unreadable else {'entries': [_memory_entry_reply(entry) for entry in entries]}
                    try:
                        reply[scope]['last_consolidation'] = memory.last_consolidation(scope)
                        reply[scope]['has_previous'] = memory.has_previous(scope)
                    except (OSError, ValueError):
                        reply[scope]['consolidation_error'] = 'Could not read consolidation history'
                return _Reply(HTTPStatus.OK, reply)

            def _get_roadmap(self, payload):
                with self._errors(0):
                    roadmap = RoadmapStore(host.run.repo_root).load()
                    reply = RoadmapStore(host.run.repo_root).reply(roadmap)
                return _Reply(HTTPStatus.OK, reply)

            def _get_models(self, payload):
                request_url = urlsplit(self.path)
                query = parse_qs(request_url.query, keep_blank_values=True)
                names = query.get('provider', [])
                base_urls = query.get('base_url', [])
                known = {name for name, _, _ in PROVIDERS}
                if set(query) - {'provider', 'base_url'} or len(names) != 1 or names[0] not in known or (len(base_urls) > 1) or (base_urls and (not base_urls[0].strip())):
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'invalid model listing request'})
                name = names[0]
                base_url = base_urls[0] if base_urls else None
                cache_key = (name, base_url)
                with host._model_cache_lock:
                    cached = host._model_cache.get(cache_key)
                if cached is not None:
                    listed, model_filter = _model_listing(host, name, cached)
                    return _Reply(HTTPStatus.OK, {'provider': name, 'state': 'available', 'models': listed, 'detail': '', 'filter': model_filter})
                with self._errors(0, name=name):
                    provider = build_provider(name, None, base_url)
                    if provider is None:
                        raise ProviderSelectionError(f'{name} is unavailable')
                    models = list_models(provider)
                with host._model_cache_lock:
                    host._model_cache[cache_key] = tuple(models)
                listed, model_filter = _model_listing(host, name, models)
                return _Reply(HTTPStatus.OK, {'provider': name, 'state': 'available', 'models': listed, 'detail': '', 'filter': model_filter})

            def _get_app_redirect(self, payload):
                request_url = urlsplit(self.path)
                location = '/app/'
                if request_url.query:
                    location = f'{location}?{request_url.query}'
                return _StreamReply(lambda: self._redirect_app(location))

            def _redirect_app(self, location):
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _get_app_index(self, payload):
                return _StreamReply(self._serve_app_index)

            def _get_app_asset(self, payload):
                request_url = urlsplit(self.path)
                request_path = request_url.path
                return _StreamReply(lambda: self._serve_app_asset(request_path.removeprefix('/app/')))

            def _get_tree(self, payload):
                return _StreamReply(self._serve_tree)

            def _get_source(self, payload):
                return _StreamReply(self._serve_source)

            def _get_files(self, payload):
                request_url = urlsplit(self.path)
                values = parse_qs(request_url.query, keep_blank_values=True)
                query_values = values.get('query', [''])
                limit_values = values.get('limit', [])
                try:
                    limit = 20 if not limit_values else int(limit_values[0])
                except ValueError:
                    limit = 0
                if len(limit_values) > 1 or not 1 <= limit <= 50:
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'limit must be from 1 to 50'})
                files, truncated = repository_files(host.run.policy, query_values[0], limit)
                return _Reply(HTTPStatus.OK, {'files': files, 'truncated': truncated})

            def _get_agent(self, payload):
                return _StreamReply(self._serve_definition)

            def _get_session_notes(self, payload):
                values = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                session_ids = values.get('session_id', [])
                if len(session_ids) != 1:
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'not found'})
                with self._errors(0):
                    reply = host.run.session_notes(session_ids[0])
                return _Reply(HTTPStatus.OK, reply)

            def _get_session_agent(self, payload):
                request_url = urlsplit(self.path)
                values = parse_qs(request_url.query, keep_blank_values=True)
                session_ids = values.get('session_id', [])
                agent_ids = values.get('agent_id', [])
                if len(session_ids) != 1 or len(agent_ids) != 1:
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'not found'})
                with self._errors(0):
                    reply = host.run.agent_history(session_ids[0], agent_ids[0])
                return _Reply(HTTPStatus.OK, reply)


            def _get_project(self, payload):
                if host._current_root is None:
                    return _Reply(HTTPStatus.OK, {'repo_root': None, 'name': None})
                repo_root = host._repo_root.resolve()
                return _Reply(HTTPStatus.OK, {'repo_root': str(repo_root), 'name': repo_root.name})

            def _get_projects(self, payload):
                current_root = host._current_root
                with host._runtimes_lock:
                    runtimes = tuple(host._runtimes.items())
                open_projects = [{'repo_root': str(root), 'name': root.name, 'activity': 'working' if any((state == 'working' for state in runtime.run.session_activity().values())) else 'waiting' if runtime.run.pending_approvals() else 'idle'} for root, runtime in runtimes]
                recent = recent_projects(host._home)
                return _Reply(HTTPStatus.OK, {'current': None if current_root is None else {'repo_root': str(current_root), 'name': current_root.name}, 'open': open_projects, 'recent': [{'repo_root': path, 'name': Path(path).name, 'exists': Path(path).is_dir()} for path in recent if isinstance(path, str)]})

            def _get_directories(self, payload):
                request_url = urlsplit(self.path)
                paths = parse_qs(request_url.query, keep_blank_values=True).get('path', [])
                if len(paths) > 1:
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'path must be one absolute directory'})
                target = Path(paths[0]).expanduser() if paths else Path.home()
                if not target.is_absolute():
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'path must be absolute'})
                target = target.resolve()
                if not target.exists() or not target.is_dir():
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'directory not found'})
                with self._errors(0):
                    entries = sorted(({'name': child.name, 'path': str(child)} for child in target.iterdir() if not child.name.startswith('.') and child.is_dir()), key=lambda entry: entry['name'].casefold())
                return _Reply(HTTPStatus.OK, {'path': str(target), 'parent': None if target == Path('/') else str(target.parent), 'entries': entries})

            def _get_mcp_prompts(self, payload):
                from symphonai_api.mcp import McpError
                pool = host.run.mcp_pool
                try:
                    return _Reply(HTTPStatus.OK, {'prompts': [] if pool is None else pool.prompts()})
                except McpError:
                    return _Reply(HTTPStatus.SERVICE_UNAVAILABLE, {'error': 'Could not list MCP prompts'})

            def _post_mcp_prompt(self, payload):
                from symphonai_api.mcp import McpError
                pool = host.run.mcp_pool
                if pool is None or payload['server'] not in pool.clients or pool.status(payload['server'])['state'] != 'started' or 'prompts' not in pool.clients[payload['server']].capabilities:
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'Unknown prompt server'})
                arguments = payload['arguments']
                if not all(isinstance(key, str) and isinstance(value, str) for key, value in arguments.items()):
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'Prompt arguments must be strings'})
                try:
                    prompt = next((entry for entry in pool.prompts(payload['server']) if entry['name'] == payload['name']), None)
                    if prompt is None:
                        raise ValueError('Unknown MCP prompt')
                    declared = prompt.get('arguments', [])
                    if any(item.get('required') and not arguments.get(item['name']) for item in declared):
                        raise ValueError('Required prompt argument is missing')
                    if set(arguments) - {item['name'] for item in declared}:
                        raise ValueError('Unknown prompt argument')
                    result = pool.clients[payload['server']].get_prompt(payload['name'], arguments)
                    text = []
                    for message in result['messages']:
                        content = message.get('content', {})
                        blocks = content if isinstance(content, list) else [content]
                        text.extend(block['text'] for block in blocks if isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str))
                    return _Reply(HTTPStatus.OK, {'text': '\n\n'.join(text)})
                except ValueError as exc:
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                except (McpError, KeyError, TypeError, AttributeError):
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'Could not render MCP prompt; check its arguments'})

            def _post_mcp_sign_in(self, payload):
                from symphonai_api.mcp_auth import AuthError
                run = host.run
                pool = run.mcp_pool
                client = None if pool is None else pool.clients.get(payload['server'])
                oauth = None if client is None else client.oauth
                if oauth is None:
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'This server does not support app sign-in'})
                try:
                    url, flow = oauth.begin(f'http://127.0.0.1:{host.port}/mcp/callback')
                except AuthError as exc:
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                with host._runtimes_lock:
                    host._mcp_auth_flows = {state: value for state, value in host._mcp_auth_flows.items()
                                           if value[0]['expires'] > time.monotonic()}
                    host._mcp_auth_flows[flow['state']] = (flow, oauth, run)
                return _Reply(HTTPStatus.OK, {'authorization_url': url})

            def _get_mcp_callback(self, payload):
                from html import escape
                from symphonai_api.mcp_auth import AuthError
                query = parse_qs(urlsplit(self.path).query)
                with host._runtimes_lock:
                    pending = host._mcp_auth_flows.pop(query.get('state', [''])[0], None)
                if pending is None or pending[0]['expires'] <= time.monotonic():
                    self._bytes(HTTPStatus.BAD_REQUEST, b'Invalid or expired sign-in state.', 'text/html; charset=utf-8')
                    return
                flow, oauth, run = pending
                try:
                    if 'error' in query:
                        raise AuthError('Sign-in was refused')
                    oauth.finish(flow, query.get('code', [''])[0])
                    run.restart_mcp(oauth.server)
                    if run.mcp_pool.status(oauth.server)['state'] != 'started':
                        raise AuthError('MCP server did not start')
                except (AuthError, ValueError, RuntimeError):
                    self._bytes(HTTPStatus.BAD_REQUEST, b'Sign-in failed. Please try signing in again.', 'text/html; charset=utf-8')
                    return
                message = f'Signed in to {escape(oauth.server)}. You can close this tab.'
                self._bytes(HTTPStatus.OK, message.encode(), 'text/html; charset=utf-8',
                            headers=(('Referrer-Policy', 'no-referrer'),))

            def _get_settings(self, payload):
                if host.run is None:
                    selection = host._provider_selection or None
                    return _Reply(HTTPStatus.OK, {'settings': {'mode': 'ask', 'selection': None if selection is None else {'provider': selection.get('name'), 'model': selection.get('model'), 'effort': selection.get('effort')}, 'config': [], 'ceiling': {}, 'trust': [], 'hooks': [], 'mcp_servers': [], 'agents': [], 'skills': [], 'plugins': [], 'withheld': [], 'providers': [{'name': name, 'env_var': variable, 'key_present': bool(os.environ.get(variable, '').strip())} for name, variable, _ in PROVIDERS], 'search': []}})
                extensions = host.run.extensions
                root = host.run.policy.repo_root.resolve()

                def display_directory(directory: Path) -> str:
                    return display_path(directory, root)

                def roster(kind: str) -> list[dict[str, str]]:
                    from symphonai_api import data

                    if extensions is None:
                        return []
                    members = getattr(extensions, kind)
                    paths = getattr(members, 'paths', {})
                    bundled_directory = (Path(data.__file__).parent / 'skills').resolve()
                    return [{'name': name, 'path': 'bundled' if kind == 'skills' and source is not None and source.resolve().parent == bundled_directory else display_directory(source) if source is not None else ''} for name in sorted(members) for source in (paths.get(name, getattr(members[name], 'path', None)),)]
                ceiling = None if extensions is None else extensions.ceiling
                search_key = None if extensions is None else extensions.config.get('search.endpoint')
                configured_search = None if search_key is None else search_endpoint(search_key)
                pool = host.run.mcp_pool
                mcp_rows = []
                for spec in (() if extensions is None else extensions.mcp_servers):
                    state = pool.status(spec.name) if pool is not None else {'state': 'disabled', 'error': ''}
                    mcp_rows.append({'name': spec.name, 'command': spec.url or shlex.join(spec.command),
                                     'started': state['state'] == 'started', **state})
                settings = {'mode': host.run.starting_mode, 'selection': host.run.selection(), 'config': [] if extensions is None else [{'key': key, 'value': value, 'scope': extensions.config.provenance[key].scope.value} for key, value in sorted(extensions.config.values.items())], 'ceiling': {'allowed_write_scope': None if ceiling is None or ceiling.allowed_write_scope is None else [str(path) for path in ceiling.allowed_write_scope], 'shell_enabled': None if ceiling is None else ceiling.shell_enabled, 'shell_allowlist': None if ceiling is None or ceiling.shell_allowlist is None else [list(command) for command in ceiling.shell_allowlist], 'fetch_enabled': None if ceiling is None else ceiling.fetch_enabled, 'fetch_allowlist': None if ceiling is None or ceiling.fetch_allowlist is None else list(ceiling.fetch_allowlist), 'modes': None if ceiling is None or ceiling.modes is None else list(ceiling.modes)}, 'trust': [] if extensions is None else [{'root': str(entry.root), 'allow': sorted(entry.allow)} for entry in sorted(extensions.trust.entries, key=lambda entry: str(entry.root))], 'hooks': [] if extensions is None else [{'event': event, 'command': shlex.join(hook.command)} for hook in extensions.hooks for event in hook.events], 'mcp_servers': sorted(mcp_rows, key=lambda row: row['name']), 'agents': roster('agents'), 'skills': roster('skills'), 'plugins': roster('plugins'), 'withheld': [] if extensions is None else [{'scope': offered.scope.value, 'directory': display_directory(offered.directory), 'names': sorted(offered.names), 'reason': f'repository not trusted for {offered.directory.name}'} for offered in sorted(extensions.withheld, key=lambda offered: (offered.scope.value, str(offered.directory)))], 'providers': [{'name': name, 'env_var': variable, 'key_present': bool(os.environ.get(variable, '').strip())} for name, variable, _ in PROVIDERS], 'search': [] if configured_search is None else [{'name': configured_search.key, 'env_var': configured_search.api_key_env_var, 'key_present': bool(os.environ.get(configured_search.api_key_env_var, '').strip())}]}
                config_warnings = [] if extensions is None else list(extensions.config.warnings)
                if config_warnings:
                    settings['config_warnings'] = config_warnings
                return _Reply(HTTPStatus.OK, {'settings': settings})

            def _post_extensions_reload(self, payload):
                with host._runtimes_lock:
                    runtime = host._runtimes[host._current_root]
                with self._errors(0), host._extensions_lock:
                    result = runtime.reload_extensions(home=host._config_home)
                return _Reply(HTTPStatus.BAD_REQUEST if result['error'] else HTTPStatus.OK, result)

            def _post_config_set(self, payload):
                from symphonai_api.config_edit import set_config_value
                with host._runtimes_lock:
                    runtime = host._runtimes[host._current_root]
                with self._errors(0), host._extensions_lock:
                    set_config_value(payload['key'], payload['value'], payload['scope'],
                                     repo_root=runtime.root, home=host._config_home)
                    result = runtime.reload_extensions(home=host._config_home)
                return _Reply(HTTPStatus.BAD_REQUEST if result['error'] else HTTPStatus.OK, result)

            def _get_approvals(self, payload):
                return _Reply(HTTPStatus.OK, {'pending': host.pending_approvals()})

            def _get_permissions(self, payload):
                with self._errors(0):
                    shell, fetch = ((), ()) if host.run is None else host.run.project_permission_grants()
                return _Reply(HTTPStatus.OK, {'shell': [list(prefix) for prefix in shell], 'fetch': list(fetch)})

            def _get_sessions(self, payload):
                request_url = urlsplit(self.path)
                values = parse_qs(request_url.query, keep_blank_values=True).get('limit', [])
                queries = parse_qs(request_url.query, keep_blank_values=True).get('query', [])
                query = queries[0] if len(queries) == 1 else None
                archived_values = parse_qs(request_url.query, keep_blank_values=True).get('archived', [])
                archived = archived_values == ['1']
                try:
                    limit = int(values[0]) if len(values) == 1 else None
                except ValueError:
                    limit = None
                if limit is not None and limit <= 0:
                    limit = None
                return _Reply(HTTPStatus.OK, list_sessions(host.run.sessions_root, limit=limit, activity=host.session_activity(), query=query, archived=archived))

            def _get_session_export(self, payload):
                request_url = urlsplit(self.path)
                values = parse_qs(request_url.query, keep_blank_values=True).get('id', [])
                if len(values) != 1 or not values[0]:
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'not found'})
                with self._errors(0):
                    exported = host.run.export_session(values[0])
                return _Reply(HTTPStatus.OK, exported)

            def _get_loops(self, payload):
                loops = LoopStore(host._repo_root).list()
                return _Reply(HTTPStatus.OK, {'loops': [{key: loop[key] for key in ('id', 'goal', 'state', 'updated_at')} for loop in loops]})

            def _get_loop(self, payload):
                request_url = urlsplit(self.path)
                values = parse_qs(request_url.query, keep_blank_values=True).get('id', [])
                if len(values) != 1:
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'not found'})
                with self._errors(0):
                    record = LoopStore(host._repo_root).load_record(values[0])
                loop = record.to_payload()
                loop['usage'] = host.run.loops.loop_usage(record)
                return _Reply(HTTPStatus.OK, {'loop': loop})

            def _get_loop_card(self, payload):
                request_url = urlsplit(self.path)
                values = parse_qs(request_url.query, keep_blank_values=True)
                loop_ids, card_ids = (values.get('loop', []), values.get('card', []))
                if len(loop_ids) != 1 or len(card_ids) != 1:
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'not found'})
                with self._errors(0):
                    detail = host.run.loops.loop_card_details(loop_ids[0], card_ids[0])
                return _Reply(HTTPStatus.OK, detail)

            def _get_history(self, payload):
                request_url = urlsplit(self.path)
                values = parse_qs(request_url.query, keep_blank_values=True).get('limit', [])
                try:
                    limit = 100 if not values else int(values[0])
                except ValueError:
                    limit = 0
                if len(values) > 1 or not 1 <= limit <= 500:
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'limit must be from 1 to 500'})
                return _Reply(HTTPStatus.OK, {'prompts': prompt_history(host.run.sessions_root, host.run.policy.repo_root, limit=limit)})

            def _get_events(self, payload):
                return _StreamReply(self._serve_events)

            def _serve_events(self):
                subscription = host.broker.subscribe()
                try:
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'keep-alive')
                    self.end_headers()
                    self._stream_events(subscription)
                except OSError:
                    return
                finally:
                    subscription.close()

            def _post_project_open(self, payload):
                with self._errors(0):
                    path = payload.get('path')
                    root = Path(path)
                    if not root.is_absolute() or not root.is_dir() or root.resolve() == Path('/'):
                        return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'path must be an existing project directory other than /'})
                if host._project_factory is None:
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'this host cannot open other projects'})
                with self._errors(1):
                    runtime, _created = host.open_project(root)
                reply = {'repo_root': str(runtime.root), 'name': runtime.root.name}
                extensions = runtime.run.extensions
                config_warnings = [] if extensions is None else list(extensions.config.warnings)
                if config_warnings:
                    reply['config_warnings'] = config_warnings
                return _Reply(HTTPStatus.OK, reply)

            def _post_permissions_remove(self, payload):
                with self._errors(0):
                    kind, value = (payload.get('kind'), payload.get('value'))
                    if kind not in {'shell', 'fetch'}:
                        raise ProtocolError('permissions/remove requires kind and value')
                    if kind == 'shell' and (not isinstance(value, list) or not value or (not all((isinstance(arg, str) for arg in value)))) or (kind == 'fetch' and (not isinstance(value, str) or not value)):
                        raise ProtocolError('permissions/remove value must be a shell prefix or fetch host')
                with self._errors(1):
                    removed = host.run.remove_project_permission_grant(kind, value)
                return _Reply(HTTPStatus.OK, {'removed': removed})

            def _post_git_commit(self, payload):
                with self._errors(0):
                    message = payload.get('message')
                    paths = payload.get('paths')
                    if not message.strip():
                        raise ProtocolError('message must be a non-blank string')
                    if not paths or any((not isinstance(path, str) for path in paths)):
                        raise ProtocolError('paths must be a non-empty list of repository-relative paths')
                    if any((state == 'working' for state in host.run.session_activity().values())):
                        return _Reply(HTTPStatus.CONFLICT, {'error': 'wait for the turn to finish'})
                    result = git_commit(host.run.policy.repo_root, message, paths)
                return _Reply(HTTPStatus.OK, result)

            def _terminals(self):
                return sessions_for(host.run, host._current_root,
                    lambda identifier, data, exited, code, offset: host.broker.publish(TerminalOutput(
                        agent_id='', run_id='',
                        terminal_id=identifier, data=data, offset=offset, exited=exited, exit_code=code,
                    )))

            def _get_terminals(self, payload):
                return _Reply(HTTPStatus.OK, {'terminals': self._terminals().list()})

            def _post_terminal_start(self, payload):
                with self._errors(0):
                    identifier = self._terminals().start()
                return _Reply(HTTPStatus.OK, {'id': identifier})

            def _post_terminal_input(self, payload):
                with self._errors(0):
                    self._terminals().get(payload['id']).input(payload['data'])
                return _Reply(HTTPStatus.OK, {'written': True})

            def _post_terminal_resize(self, payload):
                with self._errors(0):
                    self._terminals().get(payload['id']).resize(payload['cols'], payload['rows'])
                return _Reply(HTTPStatus.OK, {'resized': True})

            def _post_terminal_stop(self, payload):
                with self._errors(0):
                    self._terminals().stop(payload['id'])
                return _Reply(HTTPStatus.OK, {'stopped': True})

            def _post_jobs_start(self, payload):
                with self._errors(0):
                    command = payload.get('command')
                    if not command.strip():
                        raise ProtocolError('jobs/start requires a non-blank command')
                    job_id = host.run.start_job(command)
                return _Reply(HTTPStatus.OK, {'id': job_id})

            def _post_jobs_stop(self, payload):
                with self._errors(0):
                    job_id = payload.get('id')
                    if not job_id:
                        raise ProtocolError('jobs/stop requires an id')
                result = host.run.stop_job(job_id)
                if result is None:
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'unknown job id'})
                return _Reply(HTTPStatus.OK, {'status': result['status'], 'exit_code': result['exit_code']})

            def _post_git_pull_request(self, payload):
                with self._errors(0):
                    title = payload.get('title')
                    body = payload.get('body')
                    if not title.strip():
                        raise ProtocolError('title must be non-blank and body must be a string')
                    if any((state == 'working' for state in host.run.session_activity().values())):
                        return _Reply(HTTPStatus.CONFLICT, {'error': 'wait for the turn to finish'})
                    result = create_pull_request(host.run.policy.repo_root, title, body)
                return _Reply(HTTPStatus.OK, result)

            def _post_memory_update(self, payload):
                with self._errors(0):
                    if not payload['text'].strip() or len(payload['text']) > 500:
                        raise ProtocolError('text must be non-blank and at most 500 characters')
                memory = host.run.memory_store()
                if memory is None:
                    return _Reply(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': 'memory is unavailable'})
                with self._errors(1):
                    entry = memory.update(payload['id'], payload['text'])
                if entry is None:
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'unknown memory id'})
                return _Reply(HTTPStatus.OK, {'entry': _memory_entry_reply(entry)})

            def _post_memory_delete(self, payload):
                memory = host.run.memory_store()
                if memory is None:
                    return _Reply(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': 'memory is unavailable'})
                with self._errors(1):
                    deleted = memory.delete(payload['id'])
                if not deleted:
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'unknown memory id'})
                return _Reply(HTTPStatus.OK, {'deleted': payload['id']})

            def _post_memory_restore(self, payload):
                memory = host.run.memory_store()
                if memory is None:
                    return _Reply(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': 'memory is unavailable'})
                with self._errors(0):
                    memory.restore_previous(payload['scope'])
                return _Reply(HTTPStatus.OK, {'restored': payload['scope']})

            def _post_loop_start(self, payload):
                with self._errors(0):
                    loop = host.run.loops.start_loop(payload['goal']).to_payload()
                return _Reply(HTTPStatus.OK, {'loop': loop, 'session_id': loop['planner_session_id']})

            def _post_loop_plan(self, payload):
                with self._errors(0):
                    loop_id, action = (payload.get('loop_id'), payload.get('action'))
                    if action == 'approve' and set(payload) == {'loop_id', 'action'}:
                        loop = host.run.loops.approve_loop_plan(loop_id).to_payload()
                        reply = {'loop': loop}
                    elif action == 'drop' and set(payload) == {'loop_id', 'action', 'cards'}:
                        cards = payload['cards']
                        if not all((isinstance(card, str) for card in cards)):
                            raise ProtocolError('loop/plan drop requires cards as string ids')
                        loop = host.run.loops.drop_loop_cards(loop_id, cards).to_payload()
                        reply = {'loop': loop}
                    elif action == 'revise' and set(payload) == {'loop_id', 'action', 'note'}:
                        loop = host.run.loops.revise_loop_plan(loop_id, payload['note']).to_payload()
                        reply = {'loop': loop, 'session_id': loop['planner_session_id']}
                    else:
                        raise ProtocolError('loop/plan has an invalid action or body')
                return _Reply(HTTPStatus.OK, reply)

            def _post_loop_cancel(self, payload):
                with self._errors(0):
                    loop = host.run.loops.cancel_loop(payload['loop_id']).to_payload()
                return _Reply(HTTPStatus.OK, {'loop': loop})

            def _post_loop_commit(self, payload):
                with self._errors(0):
                    loop = host.run.loops.commit_loop(payload['loop_id'], payload['action']).to_payload()
                return _Reply(HTTPStatus.OK, {'loop': loop, 'branch': loop.get('branch'), 'commits': loop.get('summary', {}).get('commits', [])})

            def _post_loop_resume(self, payload):
                with self._errors(0):
                    loop = host.run.loops.resume_loop(payload['loop_id']).to_payload()
                return _Reply(HTTPStatus.OK, {'loop': loop})

            def _post_goal(self, payload):
                with self._errors(0):
                    objective = payload.get('objective')
                    check = payload.get('check', [])
                    max_rounds = payload.get('max_rounds', 10)
                    if not objective.strip() or not all((isinstance(arg, str) and arg for arg in check)) or (not 1 <= max_rounds <= 100):
                        raise ProtocolError('goal requires a non-blank objective, optional check argv, and max_rounds from 1 to 100')
                with self._errors(1):
                    run_id = host.run.start_goal(objective, tuple(check), max_rounds)
                return _Reply(HTTPStatus.OK, {'accepted': True, 'run_id': run_id, 'goal': host.run.goal_snapshot()})

            def _post_goal_state(self, payload):
                with self._errors(0):
                    if payload.get('action') not in ('pause', 'resume', 'clear'):
                        raise ProtocolError('goal/state requires pause, resume, or clear')
                with self._errors(1):
                    goal = host.run.goal_state(payload['action'])
                return _Reply(HTTPStatus.OK, {'goal': goal})

            def _post_agent_draft(self, payload):
                description = payload['description']
                if not description.strip() or len(description) > 2000:
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'description must be non-blank and at most 2000 characters'})
                try:
                    text = host.run.draft_agent(description)
                except ProviderSelectionError:
                    return _Reply(HTTPStatus.CONFLICT, {'error': 'no provider selected'})
                except Exception:
                    return _Reply(HTTPStatus.BAD_GATEWAY, {'error': 'agent draft model request failed'})
                try:
                    draft = json.loads(text)
                except (ValueError, TypeError):
                    return _Reply(HTTPStatus.BAD_GATEWAY, {'error': 'agent draft must be a JSON object'})
                if (not isinstance(draft, dict)
                    or not isinstance(draft.get('name'), str)
                    or not isinstance(draft.get('prompt'), str) or not draft['prompt'].strip()
                    or any(not isinstance(draft.get(key), list)
                           or any(not isinstance(value, str) for value in draft[key])
                           for key in ('tools', 'deny_tools'))
                    or any(key in draft and not isinstance(draft[key], str)
                           for key in ('provider', 'model', 'effort'))):
                    return _Reply(HTTPStatus.BAD_GATEWAY, {'error': 'agent draft has invalid fields'})
                try:
                    name, _, _ = self._definition_target(draft['name'], 'project')
                except _ReplyReady:
                    return _Reply(HTTPStatus.BAD_GATEWAY, {'error': 'agent draft has an invalid name'})
                available = host.run.available_tool_names()
                checked = {key: draft[key] for key in ('prompt', 'provider', 'model', 'effort') if key in draft}
                checked['name'] = name
                dropped = []
                for key in ('tools', 'deny_tools'):
                    checked[key] = list(dict.fromkeys(value for value in draft[key] if value in available))
                    dropped.extend(value for value in draft[key] if value not in available)
                checked['dropped'] = list(dict.fromkeys(dropped))
                return _Reply(HTTPStatus.OK, checked)

            def _post_agent(self, payload):
                stem, scope, target = self._definition_target(payload['name'], payload['scope'])
                if not self._definition_trusted(scope):
                    return _Reply(HTTPStatus.FORBIDDEN, {'error': 'repository not trusted for agents'})
                if ('text' in payload) == ('fields' in payload):
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'supply exactly one of text or fields'})
                try:
                    text = payload.get('text')
                    if text is None:
                        original = target.read_bytes().decode("utf-8") if target.exists() else ""
                        text = self._edit_definition_fields(original, payload['fields'])
                except (ValueError, AttributeError, OSError) as exc:
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                with self._errors(1):
                    self._validate_definition(target, text)
                with self._errors(2):
                    self._write_definition(target, text)
                return _Reply(HTTPStatus.OK, {'written': True, 'name': stem, 'scope': scope, 'path': str(target), 'message': 'definition saved; it will take effect on the next run'})

            def _post_steer(self, payload):
                with self._errors(0):
                    if not payload['text'].strip():
                        raise ProtocolError('steer requires one non-blank text string')
                with self._errors(1):
                    host.run.steer(payload['text'])
                return _Reply(HTTPStatus.OK, {'steered': True})

            def _post_command(self, payload):
                with self._errors(0):
                    if not payload['command'].strip():
                        raise ProtocolError('command requires one non-blank command string')
                with self._errors(1):
                    reply = host.run.run_user_command(payload['command'])
                return _Reply(HTTPStatus.OK, reply)

            def _post_changes_revert(self, payload):
                with self._errors(0):
                    has_path = 'path' in payload and bool(payload['path'].strip())
                    has_key = 'key' in payload and bool(payload['key'].strip())
                    if has_path == has_key or ('path' in payload and (not has_path)) or ('key' in payload and (not has_key)):
                        raise ProtocolError('changes/revert requires one non-empty path or key')
                with self._errors(1):
                    reply = host.run.revert_changes(path=payload.get('path'), key=payload.get('key'), force=payload.get('force', False))
                return _Reply(HTTPStatus.OK, reply)

            def _post_worktree_apply(self, payload):
                with self._errors(0):
                    name = payload.get('name')
                    if not name.strip() or Path(name).name != name or name in ('.', '..'):
                        raise ProtocolError('worktree action requires one valid name')
                with self._errors(1):
                    if self._route.action == 'apply':
                        reply = host.run.apply_worktree(name)
                    else:
                        reply = host.run.discard_worktree(name)
                return _Reply(HTTPStatus.OK, reply)

            def _post_credentials(self, payload):
                name = payload.get('name')
                value = payload.get('value')
                if name not in ({variable for _, variable, _ in PROVIDERS} | {endpoint.api_key_env_var for endpoint in search_endpoints()}):
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'unknown credential name'})
                providers = {provider for provider, variable, _ in PROVIDERS if variable == name}
                with host._model_cache_lock:
                    with self._errors(0, name=name):
                        store(name, value)
                    if value:
                        os.environ[name] = value
                    else:
                        os.environ.pop(name, None)
                    host._model_cache = {key: models for key, models in host._model_cache.items()
                                         if key[0] not in providers}
                return _Reply(HTTPStatus.OK, {'stored': True, 'name': name})

            def _post_session_new(self, payload):
                with self._errors(1):
                    host.run.end_conversation()
                return _Reply(HTTPStatus.OK, {'ended': True})

            def _post_session_viewed(self, payload):
                session_id = payload['session_id']
                if not session_id or session_id in ('.', '..') or Path(session_id).name != session_id:
                    return _Reply(HTTPStatus.BAD_REQUEST, {'error': 'invalid session_id'})
                directory = (host.run.sessions_root / session_id).resolve()
                if directory.parent != host.run.sessions_root.resolve() or not directory.is_dir():
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'unknown session'})
                if not host._session_belongs_here(session_id):
                    return _Reply(HTTPStatus.CONFLICT, {'error': 'that conversation belongs to another project'})
                viewed_at = utc_iso()
                host.run.update_session_meta(session_id, lambda meta: meta.update(viewed_at=viewed_at))
                return _Reply(HTTPStatus.OK, {'session_id': session_id, 'viewed_at': viewed_at})

            def _post_session_rename(self, payload):
                with self._errors(0):
                    run_id = payload['run_id']
                    title = payload['title'].strip()
                    if not run_id or run_id in ('.', '..') or Path(run_id).name != run_id or (not title) or (len(title) > 200):
                        raise ProtocolError('title must be non-blank and at most 200 characters')
                if not (host.run.sessions_root / run_id).is_dir():
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'unknown session'})
                if not host._session_belongs_here(run_id):
                    return _Reply(HTTPStatus.CONFLICT, {'error': 'that conversation belongs to another project'})
                host.run.update_session_meta(run_id, lambda meta: meta.update(title=title))
                return _Reply(HTTPStatus.OK, {'renamed': True})

            def _post_session_archive(self, payload):
                with self._errors(0):
                    if not payload['run_id'] or payload['run_id'] in ('.', '..') or Path(payload['run_id']).name != payload['run_id']:
                        raise ProtocolError('session/archive requires run_id and archived')
                if not (host.run.sessions_root / payload['run_id']).is_dir():
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'unknown session'})
                if not host._session_belongs_here(payload['run_id']):
                    return _Reply(HTTPStatus.CONFLICT, {'error': 'that conversation belongs to another project'})
                with self._errors(1):
                    updated = host.run.archive_session(payload['run_id'], payload['archived'])
                return _Reply(HTTPStatus.OK if updated else HTTPStatus.NOT_FOUND, {'archived': payload['archived']} if updated else {'archived': False})

            def _post_session_delete(self, payload):
                with self._errors(0):
                    if not payload['run_id'] or payload['run_id'] in ('.', '..') or Path(payload['run_id']).name != payload['run_id']:
                        raise ProtocolError('session/delete requires a run_id string')
                if not host._session_belongs_here(payload['run_id']):
                    return _Reply(HTTPStatus.CONFLICT, {'error': 'that conversation belongs to another project'})
                with self._errors(1):
                    deleted = host.run.delete_session(payload['run_id'])
                return _Reply(HTTPStatus.OK if deleted else HTTPStatus.NOT_FOUND, {'deleted': deleted})

            def _post_session_fork(self, payload):
                with self._errors(0):
                    if any((not payload[key] for key in ('run_id', 'record_id'))):
                        raise ProtocolError('session/fork requires run_id and record_id strings')
                if not host._session_belongs_here(payload['run_id']):
                    return _Reply(HTTPStatus.CONFLICT, {'error': 'that conversation belongs to another project'})
                with self._errors(1):
                    reply = host.run.fork_session(payload['run_id'], payload['record_id'], force=payload.get('force', False))
                return _Reply(HTTPStatus.OK, reply)

            def _post_provider(self, payload):
                with self._errors(0):
                    choice = payload
                    effort = choice.get('effort')
                    if effort is not None and not effort.strip():
                        raise ProviderSelectionError('effort must be a non-empty string')
                    provider = build_provider(choice.get('name'), choice.get('model'), choice.get('base_url'))
                    if provider is None:
                        raise ProviderSelectionError('choose a provider')
                if host.run is not None:
                    with self._errors(1):
                        host.run.select_provider(provider, choice.get('model'), effort, choice)
                with self._errors(2):
                    save_provider_selection(choice)
                host._default_provider = provider
                host._provider_selection = dict(choice)
                return _Reply(HTTPStatus.OK, {'selected': True})

            def _post_mode(self, payload):
                with self._errors(0):
                    mode = host.run.select_mode(payload['mode'])
                return _Reply(HTTPStatus.OK, {'mode': mode})

            def _post_compact(self, payload):
                with self._errors(1):
                    result = host.run.compact(payload.get('instructions'))
                return _Reply(HTTPStatus.OK, {'changed': result.changed, 'before_tokens': result.before_tokens, 'after_tokens': result.after_tokens, 'dropped_messages': result.dropped_messages})

            def _post_prompt(self, payload):
                with self._errors(0):
                    request = decode_request('prompt', payload)
                with self._errors(1):
                    run_id = host.run.start(request.prompt, attachments=request.attachments)
                return _Reply(HTTPStatus.OK, {'accepted': True, 'run_id': run_id})

            def _post_approval(self, payload):
                kind = 'approval'
                with self._errors(0):
                    remember_project = kind == 'approval' and payload.get('remember') == 'project'
                    if remember_project:
                        payload = {**payload, 'remember': False}
                    request = decode_request(kind, payload)
                    if remember_project:
                        request = ApprovalReply(request.approval_id, request.allowed, request.reason, 'project', request.answer)
                with self._errors(1):
                    resolved = host.resolve_approval(request.approval_id, allowed=request.allowed, reason=request.reason, remember=request.remember, answer=request.answer)
                if not resolved:
                    return _Reply(HTTPStatus.NOT_FOUND, {'error': 'unknown approval id'})
                return _Reply(HTTPStatus.OK, {'resolved': True})

            def _post_session_open(self, payload):
                with self._errors(0):
                    request = decode_request('session/open', payload)
                if not host._session_belongs_here(request.run_id):
                    return _Reply(HTTPStatus.CONFLICT, {'error': 'that conversation belongs to another project'})
                with self._errors(1):
                    reply = host.run.open_session(request.run_id)
                return _Reply(HTTPStatus.OK, reply)

            def _post_session_history(self, payload):
                with self._errors(0):
                    reply = host.run.history_page(payload["replay_id"], payload.get("offset", 0))
                return _Reply(HTTPStatus.OK, reply)

            def _post_stop(self, payload):
                with self._errors(0):
                    decode_request('stop', payload)
                host.run.stop()
                return _Reply(HTTPStatus.OK, {'accepted': True})


        routes = (
            _Route('GET', '/mcp/prompts', Handler._get_mcp_prompts),
            _Route('POST', '/mcp/prompt', Handler._post_mcp_prompt,
                payload=_Payload({'server': str, 'name': str, 'arguments': dict}, ('server', 'name', 'arguments'), extra=False)),
            _Route('POST', '/mcp/sign-in', Handler._post_mcp_sign_in,
                payload=_Payload({'server': str}, ('server',), extra=False)),
            _Route('GET', '/mcp/callback', Handler._get_mcp_callback, auth=False, project_free=True),
            _Route('GET', '/health', Handler._get_health,
                match='exact',
                project_free=True,
            ),
            _Route('POST', '/preview/start', Handler._post_preview_start,
                payload=_Payload({'url': str}, (), extra=False)),
            _Route('POST', '/preview/stop', Handler._post_preview_stop,
                payload=_Payload({}, (), extra=False)),
            _Route('POST', '/preview/mark', Handler._post_preview_mark, auth=False,
                payload=_Payload({}, (), extra=True)),
            _Route('GET', '/preview/marks', Handler._get_preview_marks),
            _Route('GET', '/usage', Handler._get_usage,
            ),
            _Route('GET', '/conversation', Handler._get_conversation,
                match='exact',
            ),
            _Route('GET', '/git/status', Handler._get_git_status,
            ),
            _Route('GET', '/git/diff', Handler._get_git_diff,
                errors=(((ValueError, 'bad_request'),),),
            ),
            _Route('GET', '/jobs', Handler._get_jobs,
            ),
            _Route('GET', '/git/pull-request', Handler._get_git_pull_request,
                errors=(((RuntimeError, 'conflict_tail'),),),
            ),
            _Route('GET', '/changes', Handler._get_changes,
                errors=(((RunActiveError, 'conflict'),),),
            ),
            _Route('GET', '/memory', Handler._get_memory,
            ),
            _Route('GET', '/roadmap', Handler._get_roadmap,
                errors=(((InvalidRoadmap, 'invalid_roadmap'),),),
            ),
            _Route('GET', '/models', Handler._get_models,
                project_free=True,
                errors=((((ProviderSelectionError, ProviderError), 'model_unavailable'),),),
            ),
            _Route('GET', '/app', Handler._get_app_redirect,
                auth=False,
                project_free=True,
            ),
            _Route('GET', '/app/', Handler._get_app_index,
                app_query=True,
                app_cookie=True,
                project_free=True,
                helper_errors={
                    'app_path': (((OSError, RuntimeError), 'empty_forbidden'),),
                    'app_index': (((OSError, UnicodeDecodeError), 'not_found'),),
                },
            ),
            _Route('GET', '/app/*', Handler._get_app_asset,
                match='prefix',
                app_cookie=True,
                project_free=True,
                helper_errors={
                    'app_path': (((OSError, RuntimeError), 'empty_forbidden'),),
                    'app_asset': ((OSError, 'not_found'),),
                },
            ),
            _Route('GET', '/tree', Handler._get_tree,
                helper_errors={
                    'requested_path': (((OSError, RuntimeError), 'empty_forbidden'),),
                },
            ),
            _Route('GET', '/source', Handler._get_source,
                helper_errors={
                    'requested_path': (((OSError, RuntimeError), 'empty_forbidden'),),
                    'source_read': (((OSError, ValueError), 'empty_not_found'),),
                    'source_decode': ((UnicodeDecodeError, 'empty_unsupported'),),
                },
            ),
            _Route('GET', '/files', Handler._get_files,
            ),
            _Route('GET', '/agent', Handler._get_agent,
                helper_errors={
                    'definition_target': (((OSError, RuntimeError), 'empty_forbidden'),),
                    'definition_read': ((OSError, 'not_found'),),
                    'definition_decode': ((UnicodeDecodeError, 'empty_unsupported'),),
                },
            ),
            _Route('GET', '/session/notes', Handler._get_session_notes,
                errors=(((TranscriptError, 'not_found'),),),
            ),
            _Route('GET', '/session/agent', Handler._get_session_agent,
                errors=(((TranscriptError, 'not_found'),),),
            ),
            _Route('GET', '/project', Handler._get_project,
                project_free=True,
            ),
            _Route('GET', '/projects', Handler._get_projects,
                project_free=True,
            ),
            _Route('GET', '/directories', Handler._get_directories,
                project_free=True,
                errors=(((PermissionError, 'unreadable_directory'),),),
            ),
            _Route('GET', '/settings', Handler._get_settings,
                project_free=True,
            ),
            _Route('POST', '/extensions/reload', Handler._post_extensions_reload,
                payload=_Payload({}, (), extra=False),
                errors=((((ValueError, OSError), 'bad_request'),),),
            ),
            _Route('POST', '/config/set', Handler._post_config_set,
                payload=_Payload({'key': str, 'value': (bool, int, float, str), 'scope': str}, ('key', 'value', 'scope'), extra=False),
                errors=((((ValueError, OSError), 'bad_request'),),),
            ),
            _Route('GET', '/approvals', Handler._get_approvals,
                match='exact',
            ),
            _Route('GET', '/permissions', Handler._get_permissions,
                match='exact',
                project_free=True,
                errors=(((PermissionGrantError, 'unavailable'),),),
            ),
            _Route('GET', '/sessions', Handler._get_sessions,
            ),
            _Route('GET', '/session/export', Handler._get_session_export,
                errors=((((KeyError, FileNotFoundError), 'not_found'), ((OSError, TranscriptError, ValueError, TypeError), 'unexportable_session')),),
            ),
            _Route('GET', '/loops', Handler._get_loops,
            ),
            _Route('GET', '/loop', Handler._get_loop,
                errors=((((KeyError, ValueError), 'not_found'),),),
            ),
            _Route('GET', '/loop/card', Handler._get_loop_card,
                errors=((((KeyError, ValueError), 'not_found'),),),
            ),
            _Route('GET', '/history', Handler._get_history,
            ),
            _Route('GET', '/events', Handler._get_events,
                match='exact',
                project_free=True,
            ),
            _Route('POST', '/project/open', Handler._post_project_open,
                match='exact',
                project_free=True,
                payload=_Payload({'path': str}, ('path',), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((Exception, 'bad_request'),)),
            ),
            _Route('POST', '/permissions/remove', Handler._post_permissions_remove,
                match='exact',
                payload=_Payload({'kind': str, 'value': (str, list)}, ('kind', 'value'), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((PermissionGrantError, 'unavailable'),)),
            ),
            _Route('POST', '/git/commit', Handler._post_git_commit,
                match='exact',
                payload=_Payload({'message': str, 'paths': list}, ('message', 'paths'), extra=False),
                errors=(((ProtocolError, 'bad_request'), (ValueError, 'bad_request'), (RuntimeError, 'conflict_tail')),),
            ),
            _Route('GET', '/terminals', Handler._get_terminals),
            _Route('POST', '/terminal/start', Handler._post_terminal_start,
                payload=_Payload({}, ()),
                errors=((((ValueError, OSError), 'bad_request'),),),
            ),
            _Route('POST', '/terminal/input', Handler._post_terminal_input,
                payload=_Payload({'id': str, 'data': str}, ('id', 'data')),
                errors=(((KeyError, 'not_found'), ((ValueError, OSError), 'bad_request')),),
            ),
            _Route('POST', '/terminal/resize', Handler._post_terminal_resize,
                payload=_Payload({'id': str, 'cols': int, 'rows': int}, ('id', 'cols', 'rows')),
                errors=(((KeyError, 'not_found'), ((ValueError, OSError), 'bad_request')),),
            ),
            _Route('POST', '/terminal/stop', Handler._post_terminal_stop,
                payload=_Payload({'id': str}, ('id',)),
                errors=(((KeyError, 'not_found'), ((ValueError, OSError), 'bad_request')),),
            ),
            _Route('POST', '/jobs/start', Handler._post_jobs_start,
                match='exact',
                payload=_Payload({'command': str}, ('command',), extra=False),
                errors=(((ProtocolError, 'bad_request'), (RuntimeError, 'conflict')),),
            ),
            _Route('POST', '/jobs/stop', Handler._post_jobs_stop,
                match='exact',
                payload=_Payload({'id': str}, ('id',), extra=False),
                errors=(((ProtocolError, 'bad_request'),),),
            ),
            _Route('POST', '/git/pull-request', Handler._post_git_pull_request,
                match='exact',
                payload=_Payload({'title': str, 'body': str}, ('title', 'body'), extra=False),
                errors=(((ProtocolError, 'bad_request'), (RuntimeError, 'conflict_tail')),),
            ),
            _Route('POST', '/memory/update', Handler._post_memory_update,
                match='exact',
                payload=_Payload({'id': str, 'text': str}, ('id', 'text'), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((MemoryUnavailable, 'empty_unavailable'), (ValueError, 'bad_request'))),
            ),
            _Route('POST', '/memory/delete', Handler._post_memory_delete,
                match='exact',
                payload=_Payload({'id': str}, ('id',), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((MemoryUnavailable, 'empty_unavailable'),)),
            ),
            _Route('POST', '/memory/restore', Handler._post_memory_restore,
                payload=_Payload({'scope': str}, ('scope',), extra=False),
                errors=((((MemoryUnavailable, ValueError, OSError), 'bad_request'),),),
            ),
            _Route('POST', '/loop/start', Handler._post_loop_start,
                match='exact',
                payload=_Payload({'goal': str}, ('goal',), extra=False),
                errors=(((ProtocolError, 'bad_request'), (RunActiveError, 'active_run'), (ValueError, 'bad_request')),),
            ),
            _Route('POST', '/loop/plan', Handler._post_loop_plan,
                match='exact',
                payload=_Payload({'loop_id': str, 'action': str, 'cards': list, 'note': str}, ('loop_id', 'action'), extra=False),
                errors=(((ProtocolError, 'bad_request'), (KeyError, 'not_found'), (RunActiveError, 'active_run'), (ValueError, 'bad_request')),),
            ),
            _Route('POST', '/loop/cancel', Handler._post_loop_cancel,
                match='exact',
                payload=_Payload({'loop_id': str}, ('loop_id',), extra=False),
                errors=(((ProtocolError, 'bad_request'), (KeyError, 'not_found'), (RunActiveError, 'active_run'), (WorktreeError, 'bad_request')),),
            ),
            _Route('POST', '/loop/commit', Handler._post_loop_commit,
                match='exact',
                payload=_Payload({'loop_id': str, 'action': str}, ('loop_id', 'action'), extra=False),
                errors=(((ProtocolError, 'bad_request'), (KeyError, 'not_found'), (RunActiveError, 'active_run'), (ValueError, 'bad_request')),),
            ),
            _Route('POST', '/loop/resume', Handler._post_loop_resume,
                match='exact',
                payload=_Payload({'loop_id': str}, ('loop_id',), extra=False),
                errors=(((ProtocolError, 'bad_request'), (KeyError, 'not_found'), (RunActiveError, 'active_run'), (ValueError, 'bad_request')),),
            ),
            _Route('POST', '/goal', Handler._post_goal,
                match='exact',
                payload=_Payload({'objective': str, 'check': list, 'max_rounds': int}, ('objective',), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((RunActiveError, 'active_run'), (ProviderSelectionError, 'bad_request'))),
            ),
            _Route('POST', '/goal/state', Handler._post_goal_state,
                match='exact',
                payload=_Payload({'action': str}, ('action',), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((KeyError, 'no_goal'), (RunActiveError, 'conflict'))),
            ),
            _Route('POST', '/agent/draft', Handler._post_agent_draft,
                match='exact',
                payload=_Payload({'description': str}, ('description',), extra=False),
                helper_errors={
                    'definition_target': (((OSError, RuntimeError), 'empty_forbidden'),),
                },
            ),
            _Route('POST', '/agent', Handler._post_agent,
                match='exact',
                payload=_Payload({'name': str, 'scope': str, 'text': str, 'fields': dict}, ('name', 'scope'), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((AgentFileError, 'bad_request'),), ((OSError, 'definition_write_failed'),)),
                helper_errors={
                    'definition_target': (((OSError, RuntimeError), 'empty_forbidden'),),
                },
            ),
            _Route('POST', '/steer', Handler._post_steer,
                match='exact',
                payload=_Payload({'text': str}, ('text',), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((RunActiveError, 'no_running_turn'),)),
            ),
            _Route('POST', '/command', Handler._post_command,
                match='exact',
                payload=_Payload({'command': str}, ('command',), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((CommandActiveError, 'command_running'), (RunActiveError, 'turn_running'), (ValueError, 'bad_request'))),
            ),
            _Route('POST', '/changes/revert', Handler._post_changes_revert,
                match='exact',
                payload=_Payload({'path': str, 'key': str, 'force': bool}, (), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((RunActiveError, 'conflict'), (ChangedOutsideError, 'changed_outside'), (KeyError, 'unknown_change'), ((OSError, ValueError), 'internal_error'))),
            ),
            _Route('POST', '/worktree/apply', Handler._post_worktree_apply,
                match='exact',
                payload=_Payload({'name': str}, ('name',), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((RunActiveError, 'conflict'), (KeyError, 'unknown_worktree'), (WorktreeApplyConflict, 'conflict'), ((OSError, ValueError, RuntimeError), 'internal_error'))),
                action='apply',
            ),
            _Route('POST', '/worktree/discard', Handler._post_worktree_apply,
                match='exact',
                payload=_Payload({'name': str}, ('name',), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((RunActiveError, 'conflict'), (KeyError, 'unknown_worktree'), (WorktreeApplyConflict, 'conflict'), ((OSError, ValueError, RuntimeError), 'internal_error'))),
                action='discard',
            ),
            _Route('POST', '/credentials', Handler._post_credentials,
                project_free=True,
                payload=_Payload({'name': str, 'value': str}, ('name', 'value'), extra=True),
                errors=(((CredentialError, 'credential_unavailable'),),),
            ),
            _Route('POST', '/session/new', Handler._post_session_new,
                match='exact',
                payload=_Payload({}, (), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((RunActiveError, 'active_run'),)),
            ),
            _Route('POST', '/session/viewed', Handler._post_session_viewed,
                match='exact',
                payload=_Payload({'session_id': str}, ('session_id',), extra=False),
                errors=(),
            ),
            _Route('POST', '/session/rename', Handler._post_session_rename,
                match='exact',
                payload=_Payload({'run_id': str, 'title': str}, ('run_id', 'title'), extra=False),
                errors=(((ProtocolError, 'bad_request'),),),
            ),
            _Route('POST', '/session/archive', Handler._post_session_archive,
                match='exact',
                payload=_Payload({'run_id': str, 'archived': bool}, ('run_id', 'archived'), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((RunActiveError, 'conflict'),)),
            ),
            _Route('POST', '/session/delete', Handler._post_session_delete,
                match='exact',
                payload=_Payload({'run_id': str}, ('run_id',), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((RunActiveError, 'conflict'),)),
            ),
            _Route('POST', '/session/fork', Handler._post_session_fork,
                match='exact',
                payload=_Payload({'run_id': str, 'record_id': str, 'force': bool}, ('run_id', 'record_id'), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((RunActiveError, 'active_run'), (ChangedOutsideError, 'changed_outside'), (SessionError, 'missing_session'), (TranscriptError, 'bad_request'), (ProviderSelectionError, 'bad_request'))),
            ),
            _Route('POST', '/provider', Handler._post_provider,
                match='exact',
                project_free=True,
                payload=_Payload({'name': str, 'model': (str, type(None)), 'base_url': (str, type(None)), 'effort': (str, type(None))}, ('name',), extra=False),
                errors=((((ProtocolError, ProviderSelectionError), 'bad_request'),), ((RunActiveError, 'active_run'),), ((OSError, 'selection_unsaved'),)),
            ),
            _Route('POST', '/mode', Handler._post_mode,
                match='exact',
                payload=_Payload({'mode': object}, ('mode',), extra=False),
                errors=((((ProtocolError, ModeSelectionError), 'bad_request'),),),
            ),
            _Route('POST', '/compact', Handler._post_compact,
                match='exact',
                payload=_Payload({'instructions': str}, (), extra=False),
                errors=(((ProtocolError, 'bad_request'),), ((RunActiveError, 'active_run'), (NoConversationError, 'bad_request'))),
            ),
            _Route('POST', '/prompt', Handler._post_prompt,
                match='exact',
                payload=_Payload({'prompt': str, 'attachments': list}, ('prompt',), extra=True),
                errors=(((ProtocolError, 'bad_request'),), ((ProviderSelectionError, 'bad_request'), (CommandActiveError, 'command_finishing'), (RunActiveError, 'active_run'))),
            ),
            _Route('POST', '/approval', Handler._post_approval,
                match='exact',
                payload=_Payload({'approval_id': str, 'allowed': bool, 'reason': str, 'remember': (bool, str), 'answer': str}, ('approval_id', 'allowed'), extra=True),
                errors=(((ProtocolError, 'bad_request'),), ((InvalidApprovalAnswer, 'bad_request'), (PermissionGrantError, 'unavailable'))),
            ),
            _Route('POST', '/session/open', Handler._post_session_open,
                match='exact',
                payload=_Payload({'run_id': str}, ('run_id',), extra=True),
                errors=(((ProtocolError, 'bad_request'),), ((RunActiveError, 'active_run'), (SessionError, 'session_not_found'), (TranscriptError, 'bad_request'), (ProviderSelectionError, 'bad_request'))),
            ),
            _Route('POST', '/session/history', Handler._post_session_history,
                match='exact',
                payload=_Payload({'replay_id': str, 'offset': int}, ('replay_id',), extra=False),
                errors=(((KeyError, 'not_found'), (ValueError, 'bad_request')),),
            ),
            _Route('POST', '/stop', Handler._post_stop,
                match='exact',
                payload=_Payload({'reason': str}, (), extra=True),
                errors=(((ProtocolError, 'bad_request'),),),
            ),
        )

        return Handler
