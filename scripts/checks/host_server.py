"""Loopback transport checks for the SymphonAI host HTTP boundary."""

from __future__ import annotations

import contextlib
import http.client
import io
import inspect
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import get_args, get_type_hints
from urllib.parse import urlencode, urlsplit
from unittest import mock

import symphonai_api.agent_loop as agent_loop
import symphonai_api.mcp as mcp_module
import symphonai_host.__main__ as host_main
import symphonai_host.protocol as protocol_module
import symphonai_host.run as host_run_module
import symphonai_host.server as host_server_module
from symphonai_api.cancellation import CancellationToken
from symphonai_api.events import RunFinished, RunStarted
from symphonai_api.extensions import Extensions, load_extensions
from symphonai_api.hooks import HookRunner
from symphonai_api.identity import RunRef, new_agent_ref
from symphonai_api.mcp import McpServerSpec
from symphonai_api.mcp_pool import McpPool
from symphonai_api.models import Message, ModelResponse, Role, ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import ModelProvider
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.runner import merge_tool_registry, standard_tool_registry
from symphonai_api.session import SessionStore, load_run_for_resume
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_host.broker import EventBroker
from symphonai_host.protocol import decode_event, decode_frame
from symphonai_host.run import HostRun
from symphonai_host.server import HostServer
from scripts.checks.harness import CheckFailed, check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]
_PRE_19B_COMMIT = "08206022734f05d5c2afb9b32c7e2789a892f1ed"
_PRE_19E_COMMIT = "fa9a7dd06eee2b5f29772c1870e57a648dac9cdc"
_FROZEN_HOST_RUN = (
    (
        "RunStarted",
        "PromptSubmitted",
        "TurnStarted",
        "TurnFinished",
        "RunFinished",
    ),
    (("user", "frozen host"), ("assistant", "done")),
    "final_response",
)
_FROZEN_PROTOCOL = (
    1,
    ("ApprovalReply", "OpenSessionRequest", "PromptRequest", "StopRequest"),
    ("approval_requested", "error", "event", "reply"),
)


def _host(
    provider: ModelProvider | None = None,
    *,
    broker: EventBroker | None = None,
    keepalive_seconds: float = 0.05,
    repo_root: Path = REPO_ROOT,
    token: str | None = None,
) -> HostServer:
    host = HostServer(
        provider or FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
        PermissionPolicy(repo_root=repo_root),
        broker=broker,
        keepalive_seconds=keepalive_seconds,
        token=token,
    )
    host.start()
    return host


def _headers(host: HostServer, token: str | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {host.token if token is None else token}"}


def _request(
    host: HostServer,
    method: str,
    path: str,
    *,
    body: dict | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    connection = http.client.HTTPConnection("127.0.0.1", host.port, timeout=2)
    encoded = None if body is None else json.dumps(body)
    request_headers = dict(headers or {})
    if encoded is not None:
        request_headers["Content-Type"] = "application/json"
    connection.request(method, path, body=encoded, headers=request_headers)
    return connection, connection.getresponse()


def _event_stream(host: HostServer) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    connection, response = _request(host, "GET", "/events", headers=_headers(host))
    if response.status != 200 or response.getheader("Content-Type") != "text/event-stream":
        connection.close()
        fail(f"event endpoint did not establish SSE: {response.status}, {response.headers!r}")
    return connection, response


def _next_sse(
    connection: http.client.HTTPConnection,
    response: http.client.HTTPResponse,
    *,
    timeout: float = 1,
    allow_timeout: bool = False,
) -> tuple[str, dict] | str:
    raw = getattr(response.fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is None:
        fail("SSE response did not retain a readable socket")
    sock.settimeout(timeout)
    try:
        while True:
            line = response.fp.readline()
            if line.startswith(b"data: "):
                return decode_frame(line.removeprefix(b"data: ").decode("utf-8").strip())
            if line.startswith(b": keepalive"):
                return "keepalive"
    except socket.timeout:
        if allow_timeout:
            return "timeout"
        fail("timed out waiting for SSE output")
    raise AssertionError("unreachable")


def _await_sse(
    connection,
    response,
    predicate,
    *,
    deadline: float = 5.0,
    what: str = "frame",
) -> tuple[str, dict]:
    """Read frames until one satisfies `predicate`, or fail naming `what`."""
    expires_at = time.monotonic() + deadline
    frames = []
    keepalives = 0
    while True:
        remaining = expires_at - time.monotonic()
        if remaining <= 0:
            fail(
                f"timed out waiting for {what}; frames seen: {frames!r}; "
                f"keepalives: {keepalives}"
            )
        frame = _next_sse(
            connection,
            response,
            timeout=remaining,
            allow_timeout=True,
        )
        if frame == "timeout":
            fail(
                f"timed out waiting for {what}; frames seen: {frames!r}; "
                f"keepalives: {keepalives}"
            )
        if frame == "keepalive":
            keepalives += 1
            continue
        frames.append(frame)
        if predicate(frame):
            return frame


class _ScriptedSSESocket:
    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout


class _ScriptedSSEReader:
    def __init__(self, lines: list[bytes], *, interval: float = 0) -> None:
        self.raw = self
        self._sock = _ScriptedSSESocket()
        self._lines = iter(lines)
        self._interval = interval

    def readline(self) -> bytes:
        if self._interval:
            time.sleep(self._interval)
        try:
            return next(self._lines)
        except StopIteration:
            raise socket.timeout from None


class _ScriptedSSEResponse:
    def __init__(self, lines: list[bytes], *, interval: float = 0) -> None:
        self.fp = _ScriptedSSEReader(lines, interval=interval)


def _sse_line(kind: str, payload: dict) -> bytes:
    frame = {
        "protocol_version": protocol_module.PROTOCOL_VERSION,
        "kind": kind,
        "payload": payload,
    }
    return b"data: " + json.dumps(frame).encode("utf-8") + b"\n"


def _check_await_sse_helper() -> None:
    keepalive = b": keepalive\n"
    distractor = ("reply", {"distractor": True})
    target = ("reply", {"target": True})
    scripted = _ScriptedSSEResponse(
        [keepalive] * 10
        + [_sse_line(*distractor), _sse_line(*target)]
    )
    actual = _await_sse(
        None,
        scripted,
        lambda frame: frame == target,
        deadline=0.2,
        what="target reply",
    )
    if actual != target:
        fail(f"awaited SSE predicate returned the wrong frame: {actual!r}")

    timeout_stream = _ScriptedSSEResponse(
        [keepalive, keepalive, _sse_line(*distractor)]
    )
    try:
        _await_sse(
            None,
            timeout_stream,
            lambda frame: frame == target,
            deadline=0.02,
            what="target reply",
        )
    except CheckFailed as exc:
        message = str(exc)
        for expected in ("target reply", "distractor", "keepalives: 2"):
            if expected not in message:
                fail(f"SSE timeout omitted {expected!r}: {message!r}")
    else:
        fail("SSE wait did not expire when its target was absent")

    fast_stream = _ScriptedSSEResponse(
        [keepalive] * 20 + [_sse_line(*target)],
        interval=0.001,
    )
    if _await_sse(
        None,
        fast_stream,
        lambda frame: frame == target,
        deadline=0.2,
        what="target after rapid keepalives",
    ) != target:
        fail("rapid keepalives spent the SSE wait budget")

    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (
            Path(__file__),
            REPO_ROOT / "scripts" / "checks" / "host_approvals.py",
            REPO_ROOT / "scripts" / "checks" / "host_sessions.py",
        )
    }
    definitions = sum(
        source.count("def _next" + "_sse(")
        + source.count("def _await" + "_sse(")
        for source in sources.values()
    )
    if definitions != 2:
        fail(f"SSE readers were duplicated across host checks: {definitions}")
    for name in ("host_approvals.py", "host_sessions.py"):
        if "_await_sse" not in sources[name]:
            fail(f"{name} did not import the shared SSE reader")


def _wait_until(predicate, message: str, *, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    fail(message)


def _subscribed_stream(host, *, expected: int = 1):
    """Open an event stream and wait until the host has registered it."""
    connection, response = _event_stream(host)
    try:
        _wait_until(
            lambda: host.broker.subscriber_count >= expected,
            f"event stream did not reach {expected} registered subscribers",
        )
    except CheckFailed:
        observed = host.broker.subscriber_count
        connection.close()
        fail(
            f"event stream did not reach {expected} registered subscribers; "
            f"observed subscriber count: {observed}"
        )
    return connection, response


def _check_subscribed_stream_helper() -> None:
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (
            Path(__file__),
            REPO_ROOT / "scripts" / "checks" / "host_approvals.py",
            REPO_ROOT / "scripts" / "checks" / "host_sessions.py",
        )
    }
    if sources["host_approvals.py"].count("_event" + "_stream(") != 0:
        fail("an approval check opens an unsubscribed event stream")
    if sources["host_sessions.py"].count("_event" + "_stream(") != 0:
        fail("a session check opens an unsubscribed event stream")
    if sources["host_server.py"].count("_event" + "_stream(") != 4:
        fail("a host check opens an unsubscribed event stream")

    helper_source = inspect.getsource(_subscribed_stream)
    if "time." + "sleep(" in helper_source:
        fail("subscription helper used time.sleep")
    if "subscriber_count >= expected" not in helper_source:
        fail("subscription wait did not use the caller's expected count")
    if "_wait" + "_until(" not in helper_source:
        fail("subscription helper did not use a predicate wait")

    host = _host()
    wait_until = _wait_until
    try:
        with mock.patch.object(
            sys.modules[__name__],
            "_wait_until",
            side_effect=lambda predicate, message: wait_until(
                predicate, message, timeout=0.05
            ),
        ):
            try:
                connection, _ = _subscribed_stream(host, expected=2)
            except CheckFailed as exc:
                message = str(exc)
                if (
                    "did not reach 2 registered subscribers" not in message
                    or "observed subscriber count: " not in message
                    or not message.rsplit("observed subscriber count: ", 1)[1].isdigit()
                ):
                    fail(f"subscription timeout omitted the observed count: {message!r}")
            else:
                connection.close()
                fail("subscription wait accepted an unreachable subscriber count")
    finally:
        host.close()


@check("host_server.handshake_line")
def check_handshake_line() -> None:
    host = _host()
    try:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            host.print_handshake()
            host.print_handshake()
        lines = output.getvalue().splitlines()
        if len(lines) != 1:
            fail(f"host printed {len(lines)} handshake lines: {lines!r}")
        handshake = json.loads(lines[0])
        if (
            set(handshake) != {"port", "token"}
            or handshake.get("port") != host.port
            or handshake.get("token") != host.token
            or not handshake.get("token")
            or host.address[0] != "127.0.0.1"
        ):
            fail(f"handshake did not expose the loopback ephemeral binding: {handshake!r}")
    finally:
        host.close()


@check("host_server.auth_required")
def check_auth_required() -> None:
    host = _host()
    try:
        health_connection, health = _request(host, "GET", "/health")
        try:
            if health.status != 200:
                fail(f"unauthenticated health request failed: {health.status}")
        finally:
            health_connection.close()
        with mock.patch("symphonai_host.server.secrets.compare_digest", wraps=__import__("secrets").compare_digest) as compare:
            for headers in ({}, _headers(host, "wrong-token")):
                connection, response = _request(host, "GET", "/events", headers=headers)
                try:
                    if response.status != 401 or response.read() != b"":
                        fail(f"unauthorized request leaked a response body: {response.status}")
                finally:
                    connection.close()
            if compare.call_count != 2:
                fail("authentication did not call secrets.compare_digest directly")
        connection, response = _request(host, "GET", "/events", headers=_headers(host, "wrong-token"))
        try:
            if host.token in response.read().decode("utf-8"):
                fail("authentication response exposed the host token")
        finally:
            connection.close()
    finally:
        host.close()


@check("host_server.file_route")
def check_file_route() -> None:
    token = "file-route-token"
    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary)
        root = fixture / "repo"
        sibling = fixture / "repo-sibling"
        specs = root / "specs"
        docs = root / "docs"
        source = root / "symphonai_host"
        git = root / ".git"
        for directory in (
            specs,
            specs / "nested",
            docs,
            source,
            git,
            sibling,
            fixture / "etc",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        spec = specs / "18d.md"
        roadmap = docs / "roadmap.json"
        outside = sibling / "outside.md"
        spec.write_text("spec text", encoding="utf-8")
        roadmap.write_text('{"goal":"fixture"}', encoding="utf-8")
        outside.write_text("outside", encoding="utf-8")
        (fixture / "etc" / "passwd").write_text("outside", encoding="utf-8")
        (root / ".env").write_text("secret", encoding="utf-8")
        (git / "config").write_text("private", encoding="utf-8")
        (source / "server.py").write_text("source", encoding="utf-8")
        (specs / "invalid.md").write_bytes(b"\xff")
        (specs / "large.md").write_bytes(
            b"x" * (host_server_module.MAX_FILE_BYTES + 1)
        )
        (specs / "outside-link.md").symlink_to(outside)

        host = _host(repo_root=root, token=token)
        protected_values = (token, str(fixture))

        def request_file(path: str, *, authorized: bool = True):  # noqa: ANN202
            headers = _headers(host) if authorized else {}
            connection, response = _request(
                host,
                "GET",
                f"/file?{urlencode({'path': path})}",
                headers=headers,
            )
            try:
                result = (response.status, tuple(response.getheaders()), response.read())
                wire = repr(result)
                leaked = [value for value in protected_values if value in wire]
                if leaked:
                    fail(f"file route response leaked a protected value: {leaked!r}")
                return result
            finally:
                connection.close()

        try:
            for path, expected_text in (
                ("specs/18d.md", "spec text"),
                ("docs/roadmap.json", '{"goal":"fixture"}'),
            ):
                status, _, body = request_file(path)
                if status != 200 or json.loads(body) != {
                    "path": path,
                    "text": expected_text,
                }:
                    fail(f"file route returned the wrong document for {path!r}")

            traversal = (
                "../etc/passwd",
                str(spec),
                "specs/nested/../../../repo-sibling/outside.md",
                "specs/outside-link.md",
                "docs/../../repo-sibling/outside.md",
            )
            for path in traversal:
                status, _, body = request_file(path)
                if status != 403 or body != b"":
                    fail(f"file route accepted traversal fixture {path!r}: {status}")

            for path in (".env", ".git/config", "symphonai_host/server.py"):
                status, _, body = request_file(path)
                if status != 403 or body != b"":
                    fail(f"file route served a path outside its allow-list: {path!r}")

            status, _, body = request_file("specs/missing.md")
            if status != 404 or json.loads(body) != {"error": "not found"}:
                fail(f"missing file response was not the generic 404: {status}, {body!r}")
            for path, expected_status in (
                ("specs/invalid.md", 415),
                ("specs/large.md", 413),
            ):
                status, _, body = request_file(path)
                if status != expected_status or body != b"":
                    fail(f"file route handled {path!r} as {status} with {body!r}")

            status, _, body = request_file("specs/18d.md", authorized=False)
            if status != 401 or body != b"":
                fail("file route did not require bearer authorization")

            return_types = get_args(
                get_type_hints(protocol_module.decode_request)["return"]
            )
            actual_protocol = (
                protocol_module.PROTOCOL_VERSION,
                tuple(sorted(item.__name__ for item in return_types)),
                tuple(sorted(protocol_module._FRAME_KINDS)),
            )
            if actual_protocol != _FROZEN_PROTOCOL:
                fail(f"file route changed the frozen protocol: {actual_protocol!r}")
        finally:
            host.close()


@check("host_server.app_routes")
def check_app_routes() -> None:
    token = "browser-route-token"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        app_root = root / "symphonai_app"
        source_root = app_root / "src"
        source_root.mkdir(parents=True)
        index = (
            "<!doctype html><html><head>"
            f"{host_server_module.APP_HANDSHAKE_MARKER}"
            "</head><body><main>app</main>"
            '<script type="module" src="/app/src/app.js"></script>'
            "</body></html>"
        )
        (app_root / "index.html").write_text(index, encoding="utf-8")
        (app_root / "app.css").write_text("body { color: black; }", encoding="utf-8")
        (source_root / "app.js").write_text("export const app = true;", encoding="utf-8")
        (app_root / "secret.py").write_text("secret = True", encoding="utf-8")
        (app_root / "notes.md").write_text("private notes", encoding="utf-8")
        outside = root / "outside.js"
        outside.write_text("outside", encoding="utf-8")
        (source_root / "escape.js").symlink_to(outside)
        docs = root / "docs"
        docs.mkdir()
        (docs / "roadmap.json").write_text("{}", encoding="utf-8")

        host = _host(repo_root=root, token=token)
        responses = []

        def get(path: str, *, authorized: bool = False):  # noqa: ANN202
            headers = _headers(host) if authorized else {}
            connection, response = _request(host, "GET", path, headers=headers)
            try:
                result = (
                    response.status,
                    response.getheader("Content-Type"),
                    response.read(),
                )
                responses.append((path, result))
                return result
            finally:
                connection.close()

        try:
            for path in ("/app", f"/app?token={token}"):
                status, content_type, body = get(
                    path,
                    authorized=path == "/app",
                )
                if status != 200 or content_type != "text/html":
                    fail(f"app index response was wrong: {status}, {content_type!r}")
                text = body.decode("utf-8")
                if text.count("window.__symphonai = ") != 1:
                    fail("app index did not contain exactly one injected handshake")
                encoded = text.split("window.__symphonai = ", 1)[1].split(";</script>", 1)[0]
                if json.loads(encoded) != {"port": host.port, "token": token}:
                    fail(f"app index injected the wrong handshake: {encoded!r}")

            for path, content_type, expected in (
                ("/app/src/app.js", "text/javascript", b"export const app = true;"),
                ("/app/app.css", "text/css", b"body { color: black; }"),
                ("/app/index.html", "text/html", index.encode("utf-8")),
            ):
                status, actual_type, body = get(path, authorized=True)
                if (status, actual_type, body) != (200, content_type, expected):
                    fail(f"app asset response was wrong for {path!r}: {status}, {actual_type!r}, {body!r}")

            for path in ("/app/secret.py", "/app/notes.md"):
                status, _, body = get(path, authorized=True)
                if status != 403 or body != b"":
                    fail(f"app route served a forbidden extension: {path!r}")

            traversal = (
                "/app/../outside.js",
                "/app/src/../../outside.js",
                f"/app/{outside}",
                "/app/src/escape.js",
            )
            for path in traversal:
                status, _, body = get(path, authorized=True)
                if status != 403 or body != b"":
                    fail(f"app route accepted traversal fixture {path!r}: {status}")

            for path in (
                f"/app?token=wrong-{token}",
                f"/app/src/app.js?token={token}",
                f"/file?path=docs/roadmap.json&token={token}",
            ):
                status, _, body = get(path)
                if status != 401 or body != b"":
                    fail(f"query authentication escaped exact /app: {path!r}, {status}")

            protected = (str(root),)
            for path, (_, _, body) in responses:
                text = body.decode("utf-8", errors="replace")
                if any(value in text for value in protected):
                    fail(f"app response exposed an absolute path: {path!r}")
                if token in text and urlsplit(path).path != "/app":
                    fail(f"app response exposed the token outside the index: {path!r}")

            source = inspect.getsource(HostServer._handler_type)
            if "_contains_path(app_root, resolved)" not in source:
                fail("app containment did not use the shared path rule")
        finally:
            host.close()


@check("host_server.event_stream_delivers")
def check_event_stream_delivers() -> None:
    _check_await_sse_helper()
    _check_subscribed_stream_helper()
    host = _host()
    try:
        connection, response = _subscribed_stream(host)
        try:
            host.broker.publish(RunStarted(agent_id="agent", run_id="run", agent_name="agent"))
            frame = _await_sse(
                connection,
                response,
                lambda candidate: isinstance(candidate, tuple)
                and candidate[0] == "event",
                what="event frame",
            )
            if not isinstance(frame, tuple) or frame[0] != "event":
                fail(f"event stream emitted the wrong frame: {frame!r}")
            event = decode_event(frame[1])
            if not isinstance(event, RunStarted) or event.run_id != "run":
                fail(f"event frame was not decodable: {event!r}")
        finally:
            connection.close()
    finally:
        host.close()


@check("host_server.two_subscribers")
def check_two_subscribers() -> None:
    host = _host()
    try:
        first_connection, first = _subscribed_stream(host)
        second_connection, second = _subscribed_stream(host, expected=2)
        try:
            if host.broker.subscriber_count != 2:
                fail(
                    "two-subscriber check published before both subscriptions "
                    f"registered: {host.broker.subscriber_count}"
                )
            # An SSE subscriber can observe another valid frame first; only
            # the shared RunStarted is the assertion this check makes.
            host.broker.publish(
                RunFinished(
                    agent_id="earlier-agent",
                    run_id="earlier-run",
                    agent_name="earlier-agent",
                    stopped_reason="done",
                )
            )
            host.broker.publish(RunStarted(agent_id="agent", run_id="run", agent_name="agent"))
            for label, connection, response in (
                ("first", first_connection, first),
                ("second", second_connection, second),
            ):
                _await_sse(
                    connection,
                    response,
                    lambda frame: isinstance(frame, tuple)
                    and frame[0] == "event"
                    and isinstance(decode_event(frame[1]), RunStarted),
                    what=f"{label} subscriber shared RunStarted",
                )
        finally:
            first_connection.close()
            second_connection.close()
    finally:
        host.close()


@check("host_server.slow_subscriber_drops_oldest")
def check_slow_subscriber_drops_oldest() -> None:
    broker = EventBroker(max_queued_events=2)
    subscriber = broker.subscribe()
    for index in range(4):
        broker.publish(RunStarted(agent_id="agent", run_id=f"run-{index}", agent_name="agent"))
    retained = [subscriber.get(timeout=0.01), subscriber.get(timeout=0.01)]
    if [event.run_id for event in retained if event is not None] != ["run-2", "run-3"]:
        fail(f"slow subscriber did not discard oldest events: {retained!r}")
    if subscriber.take_dropped() != 2:
        fail("slow subscriber did not receive an exact dropped count")
    broker.close()


@check("host_server.subscriber_disconnect")
def check_subscriber_disconnect() -> None:
    host = _host()
    try:
        connection, response = _event_stream(host)
        response.close()
        connection.close()
        host.broker.publish(RunStarted(agent_id="agent", run_id="run", agent_name="agent"))
        _wait_until(lambda: host.broker.subscriber_count == 0, "disconnected subscriber remained registered")
        host.broker.publish(RunFinished(agent_id="agent", run_id="run", agent_name="agent", stopped_reason="done"))
    finally:
        host.close()


@check("host_server.prompt_starts_run")
def check_prompt_starts_run() -> None:
    host = _host()
    try:
        connection, response = _subscribed_stream(host)
        try:
            prompt_connection, prompt = _request(host, "POST", "/prompt", body={"prompt": "hello"}, headers=_headers(host))
            try:
                reply = json.loads(prompt.read())
            finally:
                prompt_connection.close()
            if prompt.status != 200 or not reply.get("accepted") or not reply.get("run_id"):
                fail(f"prompt was not accepted before completion: {prompt.status}, {reply!r}")
            events = []
            deadline = time.monotonic() + 5
            while not events or not isinstance(events[-1], RunFinished):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    fail(f"run did not finish within five seconds; last events: {events!r}")
                frame = _await_sse(
                    connection,
                    response,
                    lambda candidate: isinstance(candidate, tuple)
                    and candidate[0] == "event",
                    deadline=min(1, remaining),
                    what="run event",
                )
                if isinstance(frame, tuple) and frame[0] == "event":
                    events.append(decode_event(frame[1]))
            if not isinstance(events[0], RunStarted) or not isinstance(events[-1], RunFinished):
                fail(f"run did not emit RunStarted through RunFinished: {events!r}")
            if reply["run_id"] == events[0].run_id:
                fail(f"/prompt returned the runtime id rather than a host handle: {reply!r}")
        finally:
            connection.close()
    finally:
        host.close()


class _WaitingProvider(ModelProvider):
    def __init__(self) -> None:
        self.release = threading.Event()

    @property
    def name(self) -> str:
        return "waiting"

    @property
    def wire_format(self) -> int:
        return 4

    def create_response(self, request, *, cancel=None) -> ModelResponse:
        while not self.release.wait(0.01):
            if cancel is not None:
                cancel.raise_if_cancelled()
        if cancel is not None:
            cancel.raise_if_cancelled()
        return ModelResponse(Message(Role.ASSISTANT, "done"))


@check("host_server.second_prompt_conflicts")
def check_second_prompt_conflicts() -> None:
    provider = _WaitingProvider()
    host = _host(provider)
    try:
        first_connection, first = _request(host, "POST", "/prompt", body={"prompt": "one"}, headers=_headers(host))
        try:
            active_id = json.loads(first.read())["run_id"]
        finally:
            first_connection.close()
        second_connection, second = _request(host, "POST", "/prompt", body={"prompt": "two"}, headers=_headers(host))
        try:
            body = json.loads(second.read())
        finally:
            second_connection.close()
        if second.status != 409 or active_id not in body.get("error", ""):
            fail(f"second prompt was accepted or did not name the active run: {second.status}, {body!r}")
        provider.release.set()
        _wait_until(lambda: not host.run.active, "released run never finished")
    finally:
        host.close()


@check("host_server.stop_cancels")
def check_stop_cancels() -> None:
    provider = _WaitingProvider()
    host = _host(provider)
    try:
        connection, response = _subscribed_stream(host)
        try:
            prompt_connection, prompt = _request(host, "POST", "/prompt", body={"prompt": "wait"}, headers=_headers(host))
            prompt.read()
            prompt_connection.close()
            for _ in range(2):
                stop_connection, stop = _request(host, "POST", "/stop", body={}, headers=_headers(host))
                try:
                    if stop.status != 200 or json.loads(stop.read()) != {"accepted": True}:
                        fail("stop was not idempotently accepted")
                finally:
                    stop_connection.close()
            terminal = None
            seen = []
            deadline = time.monotonic() + 5
            while terminal is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    fail(f"stop did not finish within five seconds; last events: {seen!r}")
                frame = _await_sse(
                    connection,
                    response,
                    lambda candidate: isinstance(candidate, tuple)
                    and candidate[0] == "event",
                    deadline=min(1, remaining),
                    what="stopped run event",
                )
                if isinstance(frame, tuple) and frame[0] == "event":
                    event = decode_event(frame[1])
                    seen.append(event)
                    if isinstance(event, RunFinished):
                        terminal = event
            if terminal.stopped_reason != "cancelled":
                fail(f"stopped run did not report cancellation: {terminal!r}")
        finally:
            connection.close()
    finally:
        host.close()


@check("host_server.bad_request_and_unknown_path")
def check_bad_request_and_unknown_path() -> None:
    host = _host()
    try:
        bad_connection, bad = _request(host, "POST", "/prompt", body={"prompt": 7}, headers=_headers(host))
        try:
            body = json.loads(bad.read())
        finally:
            bad_connection.close()
        if bad.status != 400 or "prompt" not in body.get("error", ""):
            fail(f"malformed request did not return ProtocolError text: {bad.status}, {body!r}")
        missing_connection, missing = _request(host, "GET", "/missing")
        try:
            if missing.status != 404:
                fail(f"unknown path did not return 404: {missing.status}")
        finally:
            missing_connection.close()
    finally:
        host.close()


@check("host_server.keepalive")
def check_keepalive() -> None:
    host = _host(keepalive_seconds=0.01)
    try:
        connection, response = _event_stream(host)
        try:
            if _next_sse(connection, response, timeout=1) != "keepalive":
                fail("silent event stream did not send a keepalive comment")
        finally:
            connection.close()
    finally:
        host.close()


@check("host_server.api_untouched")
def check_api_untouched() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "symphonai_api").rglob("*.py"))
        if "symphonai_host" in path.read_text(encoding="utf-8")
    ]
    if offenders:
        fail(f"runtime modules reference the host boundary: {offenders!r}")


@check("host_server.runtime_run_id_preserved")
def check_runtime_run_id_preserved() -> None:
    provider = _WaitingProvider()
    host = _host(provider)
    run_started_emitted = threading.Event()
    release_run_started = threading.Event()
    original_emit = agent_loop.emit

    def delay_run_started(sink, event) -> None:
        if isinstance(event, RunStarted):
            run_started_emitted.set()
            release_run_started.wait(5)
        original_emit(sink, event)

    try:
        connection, response = _subscribed_stream(host)
        try:
            with mock.patch(
                "symphonai_api.agent_loop.new_run_ref",
                side_effect=lambda agent_id, parent_run_id=None: RunRef(
                    "run_runtime_root", agent_id, parent_run_id
                ),
            ), mock.patch("symphonai_api.agent_loop.emit", side_effect=delay_run_started):
                prompt_connection, prompt = _request(
                    host, "POST", "/prompt", body={"prompt": "wait"}, headers=_headers(host)
                )
                try:
                    reply = json.loads(prompt.read())
                finally:
                    prompt_connection.close()
                if not run_started_emitted.wait(5):
                    fail("runtime did not prepare a root RunStarted within five seconds")
                if host.run.runtime_run_id is not None:
                    fail(f"host recorded a runtime id before RunStarted: {host.run.runtime_run_id!r}")
                health_connection, health = _request(host, "GET", "/health")
                try:
                    body = json.loads(health.read())
                finally:
                    health_connection.close()
                if body.get("run_id") != reply["run_id"] or body.get("runtime_run_id") is not None:
                    fail(f"pre-RunStarted health did not distinguish the ids: {body!r}")
                release_run_started.set()
                deadline = time.monotonic() + 5
                root_event = None
                while root_event is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        fail("root RunStarted was not observed within five seconds")
                    frame = _await_sse(
                        connection,
                        response,
                        lambda candidate: isinstance(candidate, tuple)
                        and candidate[0] == "event",
                        deadline=min(1, remaining),
                        what="root RunStarted",
                    )
                    if isinstance(frame, tuple) and frame[0] == "event":
                        event = decode_event(frame[1])
                        if isinstance(event, RunStarted):
                            root_event = event
                if reply["run_id"] == "run_runtime_root" or root_event.run_id != "run_runtime_root":
                    fail(f"runtime run id was not preserved: {reply!r}, {root_event!r}")
                if host.run.runtime_run_id != "run_runtime_root":
                    fail(f"host did not record the root runtime id: {host.run.runtime_run_id!r}")
                health_connection, health = _request(host, "GET", "/health")
                try:
                    body = json.loads(health.read())
                finally:
                    health_connection.close()
                if body.get("run_id") != reply["run_id"] or body.get("runtime_run_id") != "run_runtime_root":
                    fail(f"active health did not expose both run ids: {body!r}")
                provider.release.set()
                _wait_until(lambda: not host.run.active, "runtime run did not finish")
            health_connection, health = _request(host, "GET", "/health")
            try:
                body = json.loads(health.read())
            finally:
                health_connection.close()
            if body.get("state") != "idle" or body.get("run_id") is not None or body.get("runtime_run_id") is not None:
                fail(f"idle health retained a run id: {body!r}")
        finally:
            connection.close()
    finally:
        release_run_started.set()
        host.close()


@check("host_server.subagent_run_ids_distinct")
def check_subagent_run_ids_distinct() -> None:
    provider = _WaitingProvider()
    host = _host(provider)
    root_run_started = threading.Event()
    release_root_run_started = threading.Event()
    original_emit = agent_loop.emit

    def delay_root_run_started(sink, event) -> None:
        if isinstance(event, RunStarted):
            root_run_started.set()
            release_root_run_started.wait(5)
        original_emit(sink, event)

    try:
        connection, response = _subscribed_stream(host)
        try:
            with mock.patch("symphonai_api.agent_loop.emit", side_effect=delay_root_run_started):
                prompt_connection, prompt = _request(
                    host, "POST", "/prompt", body={"prompt": "wait"}, headers=_headers(host)
                )
                try:
                    host_run_id = json.loads(prompt.read())["run_id"]
                finally:
                    prompt_connection.close()
                if not root_run_started.wait(5):
                    fail("runtime did not prepare a root RunStarted within five seconds")
                first_subagent_event = RunStarted(
                    agent_id="agent_subagent_first", run_id="run_subagent_first", agent_name="subagent"
                )
                host.run._publish(host_run_id, first_subagent_event)
                frame = _await_sse(
                    connection,
                    response,
                    lambda candidate: isinstance(candidate, tuple)
                    and candidate[0] == "event"
                    and decode_event(candidate[1]) == first_subagent_event,
                    deadline=1,
                    what="first subagent event",
                )
                if not isinstance(frame, tuple) or decode_event(frame[1]) != first_subagent_event:
                    fail(f"first subagent event did not retain its own identity: {frame!r}")
                if host.run.runtime_run_id is not None:
                    fail(f"subagent RunStarted claimed the root runtime id: {host.run.runtime_run_id!r}")
                release_root_run_started.set()
                deadline = time.monotonic() + 5
                root_event = None
                while root_event is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        fail("root RunStarted was not observed within five seconds")
                    frame = _await_sse(
                        connection,
                        response,
                        lambda candidate: isinstance(candidate, tuple)
                        and candidate[0] == "event",
                        deadline=min(1, remaining),
                        what="root RunStarted",
                    )
                    if isinstance(frame, tuple) and frame[0] == "event":
                        event = decode_event(frame[1])
                        if isinstance(event, RunStarted):
                            root_event = event
            subagent_event = RunStarted(
                agent_id="agent_subagent", run_id="run_subagent", agent_name="subagent"
            )
            host.run._publish(host_run_id, subagent_event)
            deadline = time.monotonic() + 5
            observed_subagent = None
            while observed_subagent is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    fail("subagent RunStarted was not observed within five seconds")
                frame = _await_sse(
                    connection,
                    response,
                    lambda candidate: isinstance(candidate, tuple)
                    and candidate[0] == "event",
                    deadline=min(1, remaining),
                    what="subagent RunStarted",
                )
                if isinstance(frame, tuple) and frame[0] == "event":
                    event = decode_event(frame[1])
                    if event == subagent_event:
                        observed_subagent = event
            if (
                root_event.run_id in {first_subagent_event.run_id, subagent_event.run_id}
                or host.run.runtime_run_id != root_event.run_id
            ):
                fail(
                    "subagent RunStarted replaced the root runtime id: "
                    f"root={root_event!r}, first={first_subagent_event!r}, subagent={subagent_event!r}, "
                    f"recorded={host.run.runtime_run_id!r}"
                )
            provider.release.set()
            _wait_until(lambda: not host.run.active, "subagent identity test run did not finish")
        finally:
            connection.close()
    finally:
        release_root_run_started.set()
        host.close()


class _GatedProvider(ModelProvider):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self._calls = 0
        self.entered = [threading.Event() for _ in responses]
        self.release = [threading.Event() for _ in responses]
        self.requests = []

    @property
    def name(self) -> str:
        return "gated"

    @property
    def wire_format(self) -> int:
        return 4

    def create_response(self, request, *, cancel=None) -> ModelResponse:
        index = min(self._calls, len(self._responses) - 1)
        self._calls += 1
        self.requests.append(request)
        self.entered[index].set()
        if not self.release[index].wait(5):
            raise RuntimeError("host check did not release provider")
        if cancel is not None:
            cancel.raise_if_cancelled()
        return self._responses[index]


class _RecordingTool(LocalTool):
    def __init__(self) -> None:
        self.invocations = 0

    @property
    def name(self) -> str:
        return "recording"

    @property
    def description(self) -> str:
        return "Record whether host tool execution happened."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(ToolEffect.READ_ONLY, True, ())

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel=None,
    ) -> ToolResult:
        self.invocations += 1
        return ToolResult(tool_call_id=tool_call.id, ok=True, content="ran")


class _HookProbe:
    def __call__(self, event) -> None:  # noqa: ANN001
        return

    def pre_tool(self, tool_name: str, tool_call_id: str) -> str | None:
        return None


class _CountingExtensions:
    def __init__(self) -> None:
        self.calls = 0
        self.runners: list[_HookProbe] = []

    def hook_runner(self, *, cwd: Path) -> _HookProbe:
        self.calls += 1
        runner = _HookProbe()
        self.runners.append(runner)
        return runner


def _start_gated(host_run: HostRun, provider: _GatedProvider, prompt: str, index: int) -> str:
    host_run_id = host_run.start(prompt)
    if not provider.entered[index].wait(5):
        fail("host provider was not called within five seconds")
    with host_run._lock:
        active = host_run._active
    if active is None:
        fail("gated host run was not active")
    provider.release[index].set()
    active.thread.join(5)
    if active.thread.is_alive():
        fail("host run did not finish within five seconds")
    return host_run_id


def _host_run_snapshot(
    root: Path,
    extensions: Extensions | None,
    mcp_tools=None,  # noqa: ANN001
) -> tuple[tuple, HostRun, tuple]:
    provider = _GatedProvider(
        [ModelResponse(Message(Role.ASSISTANT, "done"))]
    )
    broker = EventBroker()
    subscription = broker.subscribe()
    run = HostRun(
        provider,
        PermissionPolicy(root),
        broker,
        sessions_root=root / "sessions",
        extensions=extensions,
        mcp_tools=mcp_tools,
    )
    calls: list[tuple] = []
    real_fan_out = host_run_module.fan_out

    def record_fan_out(*sinks):  # noqa: ANN002, ANN202
        combined = real_fan_out(*sinks)
        calls.append((sinks, combined))
        return combined

    with mock.patch.object(host_run_module, "fan_out", side_effect=record_fan_out):
        host_run_id = _start_gated(run, provider, "frozen host", 0)
    events = []
    while True:
        event = subscription.get(timeout=0.01)
        if event is None:
            break
        events.append(event)
    store = SessionStore.open(root / "sessions", host_run_id)
    loaded, _, _ = load_run_for_resume(store)
    terminal = next(
        (event.stopped_reason for event in events if isinstance(event, RunFinished)),
        None,
    )
    snapshot = (
        tuple(type(event).__name__ for event in events),
        tuple((message.role.value, message.text) for message in loaded.messages),
        terminal,
    )
    subscription.close()
    broker.close()
    return snapshot, run, tuple(calls)


@check("host_server.extensions_defaults")
def check_extensions_defaults() -> None:
    for label, configured in (("None", False), ("empty", True)):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extensions = (
                load_extensions(repo_root=root, home=root / "home")
                if configured
                else None
            )
            snapshot, run, fan_out_calls = _host_run_snapshot(root, extensions)
            if snapshot != _FROZEN_HOST_RUN:
                fail(
                    f"extensions={label} changed HostRun from {_PRE_19B_COMMIT}: "
                    f"expected={_FROZEN_HOST_RUN!r}, actual={snapshot!r}"
                )
            if run._hooks is not None:
                fail(f"extensions={label} constructed an empty HookRunner")
            if len(fan_out_calls) != 1:
                fail(f"extensions={label} did not fan out once: {fan_out_calls!r}")
            sinks, combined = fan_out_calls[0]
            if len(sinks) != 2 or sinks[1] is not None or combined is not sinks[0]:
                fail(f"extensions={label} wrapped the publish-only sink")


@check("host_server.extensions_observe_and_veto")
def check_extensions_observe_and_veto() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        event_log = root / "events.jsonl"
        observer = root / "observer.py"
        observer.write_text(
            "import json,sys\n"
            "payload=json.load(sys.stdin)\n"
            "with open(sys.argv[1], 'a', encoding='utf-8') as out:\n"
            " out.write(json.dumps(payload)+'\\n')\n",
            encoding="utf-8",
        )
        extensions = load_extensions(
            repo_root=root,
            home=root / "home",
            session={
                "hooks": [
                    {
                        "on": [
                            "RunStarted",
                            "PromptSubmitted",
                            "TurnStarted",
                            "TurnFinished",
                            "RunFinished",
                        ],
                        "command": [sys.executable, str(observer), str(event_log)],
                    }
                ]
            },
        )
        host = HostServer(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            PermissionPolicy(root),
            sessions_root=root / "sessions",
            extensions=extensions,
        )
        host.start()
        try:
            connection, response = _subscribed_stream(host)
            try:
                prompt_connection, prompt = _request(
                    host,
                    "POST",
                    "/prompt",
                    body={"prompt": "observe"},
                    headers=_headers(host),
                )
                prompt.read()
                prompt_connection.close()
                received = []
                while not received or not isinstance(received[-1], RunFinished):
                    frame = _await_sse(
                        connection,
                        response,
                        lambda candidate: isinstance(candidate, tuple)
                        and candidate[0] == "event",
                        what="observational hook event",
                    )
                    if isinstance(frame, tuple) and frame[0] == "event":
                        received.append(decode_event(frame[1]))
                with host.run._lock:
                    active = host.run._active
                if active is not None:
                    active.thread.join(5)
                    if active.thread.is_alive():
                        fail("observational host run did not finish")
                hook_types = tuple(
                    json.loads(line)["type"]
                    for line in event_log.read_text(encoding="utf-8").splitlines()
                )
                broker_types = tuple(type(event).__name__ for event in received)
                if hook_types != broker_types or not hook_types:
                    fail(
                        "hook and broker did not receive the same host events: "
                        f"hook={hook_types!r}, broker={broker_types!r}"
                    )
            finally:
                connection.close()
        finally:
            host.close()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        guard_log = root / "guard.jsonl"
        guard = root / "guard.py"
        guard.write_text(
            "import json,sys\n"
            "payload=json.load(sys.stdin)\n"
            "with open(sys.argv[1], 'a', encoding='utf-8') as out:\n"
            " out.write(json.dumps(payload)+'\\n')\n"
            "print('deny: host guarded')\n",
            encoding="utf-8",
        )
        extensions = load_extensions(
            repo_root=root,
            home=root / "home",
            session={
                "hooks": [
                    {
                        "on": ["PreToolUse"],
                        "command": [sys.executable, str(guard), str(guard_log)],
                        "blocking": True,
                    }
                ]
            },
        )
        tool = _RecordingTool()
        provider = _GatedProvider(
            [
                ModelResponse(
                    Message(
                        Role.ASSISTANT,
                        tool_calls=[ToolCall(id="record", name=tool.name)],
                    )
                ),
                ModelResponse(Message(Role.ASSISTANT, "done")),
            ]
        )
        provider.release[0].set()
        host = HostServer(
            provider,
            PermissionPolicy(root),
            sessions_root=root / "sessions",
            extensions=extensions,
        )
        host.start()
        try:
            with mock.patch.object(
                host_run_module,
                "standard_tool_registry",
                return_value={tool.name: tool},
            ):
                connection, response = _subscribed_stream(host)
                try:
                    prompt_connection, prompt = _request(
                        host,
                        "POST",
                        "/prompt",
                        body={"prompt": "veto"},
                        headers=_headers(host),
                    )
                    prompt.read()
                    prompt_connection.close()
                    if not provider.entered[1].wait(5):
                        fail("model did not receive the tool result")
                    tool_results = [
                        message.tool_result
                        for message in provider.requests[1].messages
                        if message.tool_result is not None
                    ]
                    if (
                        tool.invocations != 0
                        or len(tool_results) != 1
                        or tool_results[0].ok
                        or tool_results[0].error != "host guarded"
                    ):
                        fail(
                            "host blocking hook did not veto before execution: "
                            f"invocations={tool.invocations}, results={tool_results!r}"
                        )
                    provider.release[1].set()
                    received_terminal = False
                    while not received_terminal:
                        frame = _await_sse(
                            connection,
                            response,
                            lambda candidate: isinstance(candidate, tuple)
                            and candidate[0] == "event",
                            what="guarded run terminal event",
                        )
                        if isinstance(frame, tuple) and frame[0] == "event":
                            received_terminal = isinstance(
                                decode_event(frame[1]), RunFinished
                            )
                    payloads = [
                        json.loads(line)
                        for line in guard_log.read_text(encoding="utf-8").splitlines()
                    ]
                    if [item.get("tool_name") for item in payloads] != [tool.name]:
                        fail(f"host guard saw the wrong tool: {payloads!r}")
                finally:
                    provider.release[1].set()
                    connection.close()
        finally:
            host.close()


@check("host_server.extension_runner_lifetime")
def check_extension_runner_lifetime() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = _GatedProvider(
            [
                ModelResponse(Message(Role.ASSISTANT, "one")),
                ModelResponse(Message(Role.ASSISTANT, "two")),
            ]
        )
        configured = _CountingExtensions()
        run = HostRun(
            provider,
            PermissionPolicy(root),
            EventBroker(),
            sessions_root=root / "sessions",
            extensions=configured,  # type: ignore[arg-type]
        )
        original_runner = run._hooks
        _start_gated(run, provider, "one", 0)
        _start_gated(run, provider, "two", 1)
        if (
            configured.calls != 1
            or len(configured.runners) != 1
            or run._hooks is not original_runner
            or run._hooks is not configured.runners[0]
        ):
            fail(
                "HostRun did not retain one runner across prompts: "
                f"calls={configured.calls}, runners={configured.runners!r}"
            )


@check("host_server.extensions_forward_and_main")
def check_extensions_forward_and_main() -> None:
    marker = _CountingExtensions()
    with mock.patch.object(host_server_module, "HostRun") as host_run_factory:
        server = HostServer(
            FakeModelProvider(),
            PermissionPolicy(REPO_ROOT),
            extensions=marker,  # type: ignore[arg-type]
        )
        try:
            forwarded = host_run_factory.call_args.kwargs.get("extensions")
            if forwarded is not marker:
                fail("HostServer did not forward extensions unchanged")
        finally:
            server.close()

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        home = base / "home"
        repo = base / "repo"
        repo.mkdir()
        malformed = home / ".symphonai" / "config.toml"
        malformed.parent.mkdir(parents=True)
        malformed.write_text("unknown = true\n", encoding="utf-8")
        environment = dict(os.environ)
        environment["HOME"] = str(home)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "symphonai_host",
                "--repo-root",
                str(repo),
            ],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        if (
            completed.returncode != 2
            or not completed.stderr.startswith(
                f"configuration error: {malformed}: unknown:"
            )
            or "Traceback" in completed.stderr
            or completed.stdout != ""
        ):
            fail(
                "malformed startup did not fail cleanly before binding: "
                f"returncode={completed.returncode}, stdout={completed.stdout!r}, "
                f"stderr={completed.stderr!r}"
            )

        malformed.write_text(
            "[[hooks]]\n"
            'on = ["RunStarted"]\n'
            f"command = [{json.dumps(sys.executable)}, \"-c\", \"pass\"]\n",
            encoding="utf-8",
        )
        fake_host = mock.Mock()
        with mock.patch.dict(os.environ, {"HOME": str(home)}), mock.patch.object(
            host_main,
            "_provider",
            return_value=FakeModelProvider(),
        ), mock.patch.object(
            host_main,
            "HostServer",
            return_value=fake_host,
        ) as host_factory:
            host_main.main(["--repo-root", str(repo)])
        resolved = host_factory.call_args.kwargs.get("extensions")
        if (
            not isinstance(resolved, Extensions)
            or len(resolved.hooks) != 1
            or resolved.hooks[0].events != ("RunStarted",)
            or not isinstance(resolved.hook_runner(cwd=repo), HookRunner)
        ):
            fail(f"valid startup did not pass resolved hooks: {resolved!r}")
        fake_host.print_handshake.assert_called_once_with()
        fake_host.serve_forever.assert_called_once_with()
        fake_host.close.assert_called_once_with()


@check("host_server.extensions_protocol_frozen")
def check_extensions_protocol_frozen() -> None:
    return_types = get_args(
        get_type_hints(protocol_module.decode_request)["return"]
    )
    actual = (
        protocol_module.PROTOCOL_VERSION,
        tuple(sorted(item.__name__ for item in return_types)),
        tuple(sorted(protocol_module._FRAME_KINDS)),
    )
    if actual != _FROZEN_PROTOCOL:
        fail(
            "extension wiring changed the host protocol: "
            f"expected={_FROZEN_PROTOCOL!r}, actual={actual!r}"
        )
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "symphonai_api").rglob("*.py"))
        if "symphonai_host" in path.read_text(encoding="utf-8")
    ]
    if offenders:
        fail(f"runtime import direction reversed: {offenders!r}")


_HOST_MCP_SERVER = r'''import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

parent_path = Path(sys.argv[1])
child_path = None if sys.argv[2] == "-" else Path(sys.argv[2])
parent_path.write_text(str(os.getpid()), encoding="utf-8")
if child_path is not None:
    code = (
        "import os,signal,sys,threading;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "open(sys.argv[1], 'w').write(str(os.getpid()));"
        "threading.Event().wait()"
    )
    subprocess.Popen(
        [sys.executable, "-c", code, str(child_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    while not child_path.exists():
        threading.Event().wait(0.01)

def send(request_id, result):
    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)

for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    request_id = message["id"]
    method = message.get("method")
    if method == "initialize":
        send(request_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "host-fake", "version": "1"},
        })
    elif method == "tools/list":
        send(request_id, {"tools": [{
            "name": "search",
            "description": "Search through the host MCP server.",
            "inputSchema": {"type": "object", "properties": {}},
        }]})
'''


def _write_host_mcp(directory: Path) -> Path:
    script = directory / "host_mcp.py"
    script.write_text(_HOST_MCP_SERVER, encoding="utf-8")
    return script


def _write_mcp_config(
    home: Path,
    command: list[str],
    *,
    enabled: bool = True,
) -> Path:
    source = home / ".symphonai" / "config.toml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "[[mcp.servers]]\n"
        'name = "docs"\n'
        f"command = {json.dumps(command)}\n"
        f"enabled = {str(enabled).lower()}\n",
        encoding="utf-8",
    )
    return source


def _wait_condition(predicate, message: str, *, timeout: float = 5) -> None:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.01)
    if not predicate():
        fail(message)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _clean_pid(pid: int | None) -> None:
    if pid is not None and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class _SchemaProvider(ModelProvider):
    def __init__(self) -> None:
        self.requests = []

    @property
    def name(self) -> str:
        return "schema"

    @property
    def wire_format(self) -> int:
        return 4

    def create_response(self, request, *, cancel=None) -> ModelResponse:  # noqa: ANN001
        self.requests.append(request)
        return ModelResponse(Message(Role.ASSISTANT, "done"))


class _HostMcpTool(_RecordingTool):
    @property
    def name(self) -> str:
        return "mcp__docs__search"


@check("host_server.mcp_start_and_schema")
def check_mcp_start_and_schema() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        home = base / "home"
        repo = base / "repo"
        repo.mkdir()
        script = _write_host_mcp(base)
        parent_file = base / "normal-parent.pid"
        child_file = base / "normal-child.pid"
        _write_mcp_config(
            home,
            [sys.executable, str(script), str(parent_file), str(child_file)],
        )
        provider = _SchemaProvider()
        real_host = HostServer
        captured_process = None

        def construct(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            nonlocal captured_process
            tools = kwargs.get("mcp_tools")
            if not isinstance(tools, dict) or tuple(tools) != ("mcp__docs__search",):
                fail(f"main did not hand MCP tools to HostServer: {tools!r}")
            tool = tools["mcp__docs__search"]
            captured_process = tool._client._process
            if captured_process is None or captured_process.poll() is not None:
                fail("HostServer was constructed before its MCP server started")
            host = real_host(*args, **kwargs)

            def serve_one() -> None:
                host.run.start("schema")
                _wait_condition(
                    lambda: not host.run.active,
                    "configured host run did not finish",
                )

            host.serve_forever = serve_one
            return host

        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"HOME": str(home)}),
            mock.patch.object(host_main, "_provider", return_value=provider),
            mock.patch.object(host_main, "HostServer", side_effect=construct),
            mock.patch.object(host_main.signal, "signal"),
            contextlib.redirect_stdout(output),
        ):
            host_main.main(["--repo-root", str(repo), "--permission-mode", "auto"])
        parent = int(parent_file.read_text(encoding="utf-8"))
        child = int(child_file.read_text(encoding="utf-8"))
        try:
            if len(provider.requests) != 1 or not any(
                schema.get("name") == "mcp__docs__search"
                for schema in provider.requests[0].tools
            ):
                fail(
                    "configured MCP tool was absent from provider schemas: "
                    f"{provider.requests!r}"
                )
            if captured_process is None or captured_process.poll() is None:
                fail("normal host shutdown left the MCP parent alive")
            _wait_condition(
                lambda: not _pid_alive(parent) and not _pid_alive(child),
                "normal host shutdown left an MCP process alive",
            )
        finally:
            _clean_pid(parent)
            _clean_pid(child)

        _write_mcp_config(home, ["must-not-start"], enabled=False)
        fake_host = mock.Mock()
        with (
            mock.patch.dict(os.environ, {"HOME": str(home)}),
            mock.patch.object(
                mcp_module.subprocess,
                "Popen",
                side_effect=AssertionError("disabled server spawned"),
            ) as popen,
            mock.patch.object(host_main, "_provider", return_value=FakeModelProvider()),
            mock.patch.object(host_main, "HostServer", return_value=fake_host) as factory,
            mock.patch.object(host_main.signal, "signal"),
        ):
            try:
                host_main.main(["--repo-root", str(repo)])
            except SystemExit as exc:
                fail(f"disabled configured server stopped the host: {exc.code!r}")
        if popen.called:
            fail("disabled configured MCP server reached Popen")
        if factory.call_args.kwargs.get("mcp_tools") != {}:
            fail("disabled configured server contributed a tool")


@check("host_server.mcp_start_failures")
def check_mcp_start_failures() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        home = base / "home"
        repo = base / "repo"
        repo.mkdir()
        source = _write_mcp_config(
            home,
            [str(base / "missing-mcp-server")],
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"HOME": str(home)}),
            mock.patch.object(host_main, "HostServer") as host_factory,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            try:
                host_main.main(["--repo-root", str(repo)])
            except SystemExit as exc:
                if exc.code != 2:
                    fail(f"MCP startup failure exited with {exc.code!r}")
            else:
                fail("MCP startup failure did not exit")
        if (
            not stderr.getvalue().startswith("mcp error: ")
            or "docs" not in stderr.getvalue()
            or "Traceback" in stderr.getvalue()
            or stdout.getvalue() != ""
            or host_factory.called
        ):
            fail(
                "MCP startup failure was not cleanly refused before binding: "
                f"stdout={stdout.getvalue()!r}, stderr={stderr.getvalue()!r}"
            )

        source.write_text("unknown = true\n", encoding="utf-8")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"HOME": str(home)}),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            try:
                host_main.main(["--repo-root", str(repo)])
            except SystemExit as exc:
                if exc.code != 2:
                    fail(f"configuration failure exited with {exc.code!r}")
            else:
                fail("configuration failure did not exit")
        if (
            not stderr.getvalue().startswith("configuration error: ")
            or stderr.getvalue().startswith("mcp error: ")
            or "Traceback" in stderr.getvalue()
            or stdout.getvalue() != ""
        ):
            fail("configuration and MCP startup failures were not distinguishable")


@check("host_server.mcp_sigterm_shutdown")
def check_mcp_sigterm_shutdown() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        home = base / "home"
        repo = base / "repo"
        repo.mkdir()
        script = _write_host_mcp(base)
        parent_file = base / "signal-parent.pid"
        child_file = base / "signal-child.pid"
        _write_mcp_config(
            home,
            [sys.executable, str(script), str(parent_file), str(child_file)],
        )
        environment = dict(os.environ)
        environment["HOME"] = str(home)
        process = subprocess.Popen(
            [sys.executable, "-m", "symphonai_host", "--repo-root", str(repo)],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output: list[str] = []
        ready = threading.Event()
        parent = None
        child = None

        def read_handshake() -> None:
            if process.stdout is not None:
                output.append(process.stdout.readline())
            ready.set()

        reader = threading.Thread(target=read_handshake, daemon=True)
        reader.start()
        try:
            if not ready.wait(5) or not output or not output[0].strip():
                fail("host did not bind after starting its configured MCP server")
            json.loads(output[0])
            parent = int(parent_file.read_text(encoding="utf-8"))
            child = int(child_file.read_text(encoding="utf-8"))
            process.send_signal(signal.SIGTERM)
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                fail("SIGTERM did not stop the host within five seconds")
            if returncode != 0:
                stderr = "" if process.stderr is None else process.stderr.read()
                fail(f"SIGTERM host exit was {returncode}: {stderr!r}")
            _wait_condition(
                lambda: not _pid_alive(parent) and not _pid_alive(child),
                "SIGTERM host shutdown left an MCP process alive",
            )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            _clean_pid(parent)
            _clean_pid(child)


@check("host_server.mcp_close_order")
def check_mcp_close_order() -> None:
    events: list[str] = []
    extensions = mock.Mock(mcp_servers=())
    pool = mock.Mock()
    pool.start.side_effect = lambda: events.append("pool.start") or {}
    pool.close.side_effect = lambda: events.append("pool.close")
    host = mock.Mock()
    host.close.side_effect = lambda: events.append("host.close")

    def construct_host(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        events.append("host.construct")
        return host

    standard = standard_tool_registry()
    with (
        mock.patch.object(host_main, "load_extensions", return_value=extensions),
        mock.patch.object(host_main, "McpPool", return_value=pool) as pool_factory,
        mock.patch.object(host_main, "standard_tool_registry", return_value=standard),
        mock.patch.object(host_main, "_provider", return_value=FakeModelProvider()),
        mock.patch.object(host_main, "HostServer", side_effect=construct_host),
        mock.patch.object(host_main.signal, "signal"),
    ):
        host_main.main(["--repo-root", str(REPO_ROOT)])
    if events != ["pool.start", "host.construct", "host.close", "pool.close"]:
        fail(f"host and MCP pool lifetime order changed: {events!r}")
    if pool.close.call_count != 1:
        fail(f"MCP pool had more than one owner: {pool.close.call_count} closes")
    if pool_factory.call_args.kwargs.get("reserved_names") != set(standard):
        fail("main did not inject the live standard registry keys")


@check("host_server.mcp_pass_through_and_ownership")
def check_mcp_pass_through_and_ownership() -> None:
    marker = {"mcp__docs__search": _HostMcpTool()}
    with mock.patch.object(host_server_module, "HostRun") as host_run_factory:
        server = HostServer(
            FakeModelProvider(),
            PermissionPolicy(REPO_ROOT),
            mcp_tools=marker,
        )
        try:
            if host_run_factory.call_args.kwargs.get("mcp_tools") is not marker:
                fail("HostServer did not forward MCP tools unchanged")
        finally:
            server.close()

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_host_mcp(directory)
        parent_file = directory / "owner-parent.pid"
        pool = McpPool(
            (
                McpServerSpec(
                    "docs",
                    (sys.executable, str(script), str(parent_file), "-"),
                    enabled=True,
                ),
            ),
            cwd=directory,
            reserved_names=set(standard_tool_registry()),
        )
        pool.start()
        tool = pool.tools["mcp__docs__search"]
        process = tool._client._process
        host = HostServer(
            FakeModelProvider(),
            PermissionPolicy(directory),
            mcp_tools=pool.tools,
        )
        try:
            host.close()
            if process is None or process.poll() is not None:
                fail("HostServer.close took ownership of the MCP pool")
        finally:
            host.close()
            pool.close()
        if process is None or process.poll() is None:
            fail("explicit pool owner did not close the MCP server")


@check("host_server.mcp_defaults_merge_and_protocol")
def check_mcp_defaults_merge_and_protocol() -> None:
    for label, tools in (("None", None), ("empty", {})):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot, _, _ = _host_run_snapshot(
                Path(temporary),
                None,
                mcp_tools=tools,
            )
        if snapshot != _FROZEN_HOST_RUN:
            fail(
                f"mcp_tools={label} changed HostRun from {_PRE_19E_COMMIT}: "
                f"expected={_FROZEN_HOST_RUN!r}, actual={snapshot!r}"
            )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = _GatedProvider(
            [ModelResponse(Message(Role.ASSISTANT, "done"))]
        )
        tool = _HostMcpTool()
        host_run = HostRun(
            provider,
            PermissionPolicy(root),
            EventBroker(),
            sessions_root=root / "sessions",
            mcp_tools={tool.name: tool},
        )
        with mock.patch.object(
            host_run_module,
            "merge_tool_registry",
            wraps=merge_tool_registry,
        ) as merge_spy:
            _start_gated(host_run, provider, "schema", 0)
        if merge_spy.call_count != 1:
            fail("HostRun copied the merge instead of calling merge_tool_registry")
        if not any(
            schema.get("name") == tool.name
            for schema in provider.requests[0].tools
        ):
            fail(f"HostRun derived schemas before its MCP merge: {provider.requests[0].tools!r}")

        standard_tool = _RecordingTool()
        extra_tool = _RecordingTool()
        standard = {standard_tool.name: standard_tool}
        collision_run = HostRun(
            FakeModelProvider(),
            PermissionPolicy(root),
            EventBroker(),
            sessions_root=root / "collision-sessions",
            mcp_tools={extra_tool.name: extra_tool},
        )
        session = SessionStore(root / "collision-sessions", "host-collision")
        try:
            with mock.patch.object(
                host_run_module,
                "standard_tool_registry",
                return_value=standard,
            ):
                try:
                    collision_run._run(
                        "host-collision",
                        new_agent_ref("agent"),
                        [Message(Role.USER, "collision")],
                        None,
                        session,
                        (),
                        CancellationToken(),
                    )
                except ValueError as exc:
                    if standard_tool.name not in str(exc):
                        fail(f"host collision omitted the tool name: {exc!r}")
                else:
                    fail("HostRun overwrote a colliding standard tool")
        finally:
            session.close()
        if standard[standard_tool.name] is not standard_tool:
            fail("HostRun collision changed the standard binding")

    return_types = get_args(get_type_hints(protocol_module.decode_request)["return"])
    actual_protocol = (
        protocol_module.PROTOCOL_VERSION,
        tuple(sorted(item.__name__ for item in return_types)),
        tuple(sorted(protocol_module._FRAME_KINDS)),
    )
    if actual_protocol != _FROZEN_PROTOCOL:
        fail(f"MCP host ownership changed the protocol: {actual_protocol!r}")
