"""Checks for the standard-library host client."""

from __future__ import annotations

import ast
import io
import json
import threading
from unittest import mock
from pathlib import Path

from symphonai_api.models import Message, ModelResponse, Role
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_host.client import HostAddress, HostClient, HostClientError
from symphonai_host import cli
from symphonai_host.protocol import UnknownEvent, decode_event
from symphonai_host.server import HostServer
from scripts.checks.harness import check, fail


ROOT = Path(__file__).resolve().parents[2]


@check("host_client.handshake_parsing")
def check_handshake_parsing() -> None:
    address = HostAddress.from_handshake('{"port": 1234, "token": "secret"}')
    if address.port != 1234 or address.token != "secret":
        fail(f"handshake parsed incorrectly: {address!r}")
    try:
        HostAddress.from_handshake("not json")
    except HostClientError:
        return
    fail("malformed handshake was accepted")


@check("host_client.sse_iteration")
def check_sse_iteration() -> None:
    frames = list(HostClient._decode_sse([
        b": keepalive\n",
        b'data: {"protocol_version": 1, "kind": "event",\n',
        b'data: "payload": {"type": "Future", "field": 1}}\n',
        b"\n",
        b'data: {"protocol_version": 1, "kind": "error", "payload": {"dropped": 2}}\n',
        b"\n",
    ]))
    if len(frames) != 2 or frames[1] != ("error", {"dropped": 2}):
        fail(f"SSE frames were not preserved: {frames!r}")
    if not isinstance(decode_event(frames[0][1]), UnknownEvent):
        fail(f"unknown event was not preserved: {frames!r}")


@check("host_client.round_trip_against_host")
def check_round_trip_against_host() -> None:
    host = HostServer(FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]), PermissionPolicy(repo_root=ROOT))
    host.start()
    try:
        client = HostClient(HostAddress(host.port, host.token))
        if client.health().get("state") != "idle" or not client.send_prompt("hello").get("accepted"):
            fail("client did not round-trip health and prompt")
        if client.pending_approvals() != []:
            fail("unexpected pending approvals")
        if not client.stop().get("accepted"):
            fail("client did not round-trip stop")
        if host.token in f"{client.address.port}{client.health()}":
            fail("token leaked into a client URL or response")
    finally:
        host.close()


@check("host_client.connection_error")
def check_connection_error() -> None:
    try:
        HostClient(HostAddress(1, "secret"), timeout=0.01).health()
    except HostClientError as exc:
        if "/health" not in str(exc) or "secret" in str(exc):
            fail(f"connection error leaked or omitted endpoint: {exc}")
        return
    fail("unreachable host did not raise HostClientError")


@check("host_client.blank_line_and_eof")
def check_blank_line_and_eof() -> None:
    client = _ScriptedClient(())
    stdin = _CountingInput("\n")
    with mock.patch.object(cli.sys, "stdin", stdin), mock.patch.object(
        cli.select, "select", side_effect=lambda *_: ([stdin], [], [])
    ):
        cli.run(client)
    if client.prompts or client.stops != 1:
        fail("blank line was not ignored before EOF stopped the client")


@check("host_client.stdin_is_read_once")
def check_stdin_is_read_once() -> None:
    approval = {"approval_id": "approval-1", "operation": "write_file", "target": "file.txt"}
    client = _ScriptedClient((("approval_requested", approval),))
    stdin = _CountingInput("yes\n/quit\n")

    def ready(*_):
        if client.approval_seen.wait(1):
            return [stdin], [], []
        return [], [], []

    with mock.patch.object(cli.sys, "stdin", stdin), mock.patch.object(cli.select, "select", side_effect=ready):
        cli.run(client)
    if client.approvals != [("approval-1", True)] or stdin.reads != 2:
        fail(f"approval was not answered solely by main stdin reader: {client!r}")
    source = (ROOT / "symphonai_host" / "cli.py").read_text(encoding="utf-8")
    reader = source.split("def reader()", 1)[1].split("threading.Thread", 1)[0]
    if "input(" in reader or "sys.stdin" in reader:
        fail("event reader accesses stdin")


@check("host_client.no_token_in_argv")
def check_no_token_in_argv() -> None:
    source = (ROOT / "scripts" / "host_client.py").read_text(encoding="utf-8")
    if 'add_argument("--token"' in source or "--token" in source:
        fail("client accepts a bearer token in argv")


@check("host_client.stdlib_only")
def check_stdlib_only() -> None:
    allowed = {"__future__", "argparse", "dataclasses", "http", "json", "queue", "select", "subprocess", "sys", "threading", "typing"}
    for path in (ROOT / "symphonai_host").glob("*.py"):
        if path.name not in {"client.py", "cli.py"}:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.split(".")[0] not in allowed and not name.name.startswith("symphonai_"):
                        fail(f"third-party import in {path.name}: {name.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root not in allowed and not root.startswith("symphonai_"):
                    fail(f"third-party import in {path.name}: {node.module}")


class _ScriptedClient:
    """Minimal client double that exposes which thread consumed stdin."""

    def __init__(self, frames: tuple[tuple[str, dict], ...]) -> None:
        self.frames = frames
        self.approval_seen = threading.Event()
        self.approvals: list[tuple[str, bool]] = []
        self.prompts: list[str] = []
        self.stops = 0

    def events(self):
        for frame in self.frames:
            self.approval_seen.set()
            yield frame

    def send_approval(self, approval_id: str, *, allowed: bool, reason: str = "") -> dict:
        self.approvals.append((approval_id, allowed))
        return {"resolved": True}

    def send_prompt(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        return {"accepted": True}

    def stop(self, reason: str = "") -> dict:
        self.stops += 1
        return {"accepted": True}

    def _request(self, method: str, path: str) -> dict:
        return {"pending": []}


class _CountingInput(io.StringIO):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.reads = 0

    def readline(self, *args, **kwargs) -> str:
        self.reads += 1
        return super().readline(*args, **kwargs)
