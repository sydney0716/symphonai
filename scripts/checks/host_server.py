"""Loopback transport checks for the SymphonAI host HTTP boundary."""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import socket
import threading
import time
from pathlib import Path
from unittest import mock

import symphonai_api.agent_loop as agent_loop
from symphonai_api.events import RunFinished, RunStarted
from symphonai_api.identity import RunRef
from symphonai_api.models import Message, ModelResponse, Role
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import ModelProvider
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_host.broker import EventBroker
from symphonai_host.protocol import decode_event, decode_frame
from symphonai_host.server import HostServer
from scripts.checks.harness import check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]


def _host(
    provider: ModelProvider | None = None,
    *,
    broker: EventBroker | None = None,
    keepalive_seconds: float = 0.05,
) -> HostServer:
    host = HostServer(
        provider or FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
        PermissionPolicy(repo_root=REPO_ROOT),
        broker=broker,
        keepalive_seconds=keepalive_seconds,
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


def _wait_until(predicate, message: str, *, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    fail(message)


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


@check("host_server.event_stream_delivers")
def check_event_stream_delivers() -> None:
    host = _host()
    try:
        connection, response = _event_stream(host)
        try:
            host.broker.publish(RunStarted(agent_id="agent", run_id="run", agent_name="agent"))
            deadline = time.monotonic() + 5
            frame = None
            while time.monotonic() < deadline:
                candidate = _next_sse(connection, response)
                if isinstance(candidate, tuple) and candidate[0] == "event":
                    frame = candidate
                    break
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
        first_connection, first = _event_stream(host)
        second_connection, second = _event_stream(host)
        try:
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
                deadline = time.monotonic() + 5
                last_frame: object = None
                while time.monotonic() < deadline:
                    frame = _next_sse(connection, response, timeout=0.5, allow_timeout=True)
                    last_frame = frame
                    if isinstance(frame, tuple) and frame[0] == "event" and isinstance(
                        decode_event(frame[1]), RunStarted
                    ):
                        break
                else:
                    fail(
                        f"{label} subscriber did not receive the shared event; "
                        f"last frame: {last_frame!r}"
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
        connection, response = _event_stream(host)
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
                frame = _next_sse(
                    connection, response, timeout=min(1, remaining), allow_timeout=True
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
        connection, response = _event_stream(host)
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
                frame = _next_sse(
                    connection, response, timeout=min(1, remaining), allow_timeout=True
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
        connection, response = _event_stream(host)
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
                    frame = _next_sse(connection, response, timeout=min(1, remaining))
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
        connection, response = _event_stream(host)
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
                frame = _next_sse(connection, response, timeout=1)
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
                    frame = _next_sse(connection, response, timeout=min(1, remaining))
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
                frame = _next_sse(connection, response, timeout=min(1, remaining))
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
