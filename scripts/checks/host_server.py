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

from symphonai_api.cancellation import CancellationToken
from symphonai_api.events import RunFinished, RunStarted
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
        fail("timed out waiting for SSE output")
    raise AssertionError("unreachable")


def _wait_until(predicate, message: str) -> None:
    deadline = time.monotonic() + 1
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
        if handshake != host.handshake() or host.address[0] != "127.0.0.1" or not host.port:
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
            frame = _next_sse(connection, response)
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
            host.broker.publish(RunStarted(agent_id="agent", run_id="run", agent_name="agent"))
            for connection, response in ((first_connection, first), (second_connection, second)):
                frame = _next_sse(connection, response)
                if not isinstance(frame, tuple) or not isinstance(decode_event(frame[1]), RunStarted):
                    fail("a subscriber did not receive the shared event")
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
            while not events or not isinstance(events[-1], RunFinished):
                frame = _next_sse(connection, response)
                if isinstance(frame, tuple) and frame[0] == "event":
                    events.append(decode_event(frame[1]))
            if not isinstance(events[0], RunStarted) or not isinstance(events[-1], RunFinished):
                fail(f"run did not emit RunStarted through RunFinished: {events!r}")
            if any(event.run_id != reply["run_id"] for event in events):
                fail(f"event run ids differed from /prompt reply: {events!r}, {reply!r}")
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
            while terminal is None:
                frame = _next_sse(connection, response)
                if isinstance(frame, tuple) and frame[0] == "event":
                    event = decode_event(frame[1])
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
    import subprocess

    changed = subprocess.run(
        ["git", "diff", "--name-only", "--", "symphonai_api"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if changed.returncode or changed.stdout:
        fail(f"phase 17b modified symphonai_api: {changed.stdout or changed.stderr}")
