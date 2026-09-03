"""Checks for reopening persisted host sessions."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

from symphonai_api.models import Message, ModelResponse, Role
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_host.client import HostAddress, HostClient, HostClientError
from symphonai_host.server import HostServer
from symphonai_host.sessions import list_sessions
from scripts.checks.harness import check, fail


def _wait_idle(client: HostClient) -> None:
    deadline = time.monotonic() + 5
    while client.health()["state"] == "active" and time.monotonic() < deadline:
        time.sleep(0.02)
    if client.health()["state"] != "idle":
        fail("host did not become idle")


def _host(root: Path, responses=None) -> tuple[HostServer, HostClient]:
    host = HostServer(
        FakeModelProvider(responses or [ModelResponse(Message(Role.ASSISTANT, "done"))]),
        PermissionPolicy(repo_root=root),
        sessions_root=root / "sessions",
    )
    host.start()
    return host, HostClient(HostAddress(host.port, host.token))


def _finished_session(root: Path) -> tuple[HostServer, HostClient, str]:
    host, client = _host(root)
    reply = client.send_prompt("first")
    _wait_idle(client)
    return host, client, reply["run_id"]


@check("host_sessions.list_order_and_fields")
def check_list_order_and_fields() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client, run_id = _finished_session(root)
        try:
            sessions = client.list_sessions()
            if sessions[0]["run_id"] != run_id or set(sessions[0]) != {"run_id", "title", "created_at", "updated_at", "stopped_reason", "parent_run_id", "state", "message_count"}:
                fail(f"unexpected sessions response: {sessions!r}")
        finally:
            host.close()


@check("host_sessions.damaged_session_listed")
def check_damaged_session_listed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        damaged = root / "sessions" / "broken"
        damaged.mkdir(parents=True)
        (damaged / "meta.json").write_text("{", encoding="utf-8")
        result = list_sessions(root / "sessions")
        if result != [{"run_id": "broken", "title": None, "created_at": None, "updated_at": None, "stopped_reason": None, "parent_run_id": None, "state": "unreadable", "message_count": 0}]:
            fail(f"damaged session was omitted or misclassified: {result!r}")


@check("host_sessions.empty_root")
def check_empty_root() -> None:
    with tempfile.TemporaryDirectory() as directory:
        if list_sessions(Path(directory) / "missing") != []:
            fail("missing sessions root was not empty")


@check("host_sessions.open_unknown_404")
def check_open_unknown_404() -> None:
    with tempfile.TemporaryDirectory() as directory:
        host, client = _host(Path(directory))
        try:
            try:
                client.open_session("run_missing")
            except HostClientError as exc:
                if "404" not in str(exc):
                    fail(f"unknown session did not return 404: {exc}")
            else:
                fail("unknown session opened")
        finally:
            host.close()


@check("host_sessions.open_during_run_409")
def check_open_during_run_409() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client, run_id = _finished_session(root)
        try:
            client.send_prompt("active")
            try:
                client.open_session(run_id)
            except HostClientError as exc:
                if "409" not in str(exc):
                    fail(f"active session did not return 409: {exc}")
            else:
                fail("opened while active")
        finally:
            host.close()


def _open_with_history(root: Path) -> tuple[HostServer, HostClient, str, list[dict], dict]:
    host, client, run_id = _finished_session(root)
    frames: list[dict] = []
    ready = threading.Event()

    def consume() -> None:
        try:
            ready.set()
            for kind, payload in client.events():
                if kind == "event" and payload.get("type") == "HistoryMessage":
                    frames.append(payload)
                    if payload["role"] == "assistant":
                        return
        except Exception:
            return

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    ready.wait(1)
    time.sleep(0.05)
    return host, client, run_id, frames, client.open_session(run_id)


@check("host_sessions.replay_order")
def check_replay_order() -> None:
    with tempfile.TemporaryDirectory() as directory:
        host, _, _, frames, _ = _open_with_history(Path(directory))
        try:
            deadline = time.monotonic() + 5
            while len(frames) < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            if [frame["role"] for frame in frames] != ["user", "assistant"] or any("arguments" in call for frame in frames for call in frame["tool_calls"]):
                fail(f"history replay was unsafe or out of order: {frames!r}")
        finally:
            host.close()


@check("host_sessions.open_reply_fields")
def check_open_reply_fields() -> None:
    with tempfile.TemporaryDirectory() as directory:
        host, _, _, _, reply = _open_with_history(Path(directory))
        try:
            if set(reply) != {"run_id", "state", "replayed", "repaired_ids", "dropped_bytes"}:
                fail(f"open reply fields changed: {reply!r}")
        finally:
            host.close()


@check("host_sessions.open_crashed_session")
def check_open_crashed_session() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client, run_id = _finished_session(root)
        try:
            transcript = root / "sessions" / run_id / "run.jsonl"
            transcript.write_bytes(transcript.read_bytes().rsplit(b"\n", 2)[0] + b"\n")
            reply = client.open_session(run_id)
            if reply["state"] != "crashed":
                fail(f"crashed session was not reported: {reply!r}")
        finally:
            host.close()


@check("host_sessions.continuation_conversation")
def check_continuation_conversation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client, run_id = _finished_session(root)
        try:
            opened = client.open_session(run_id)
            continued = client.send_prompt("second")
            _wait_idle(client)
            sessions = {item["run_id"]: item for item in client.list_sessions()}
            if continued["run_id"] == run_id or sessions[continued["run_id"]]["parent_run_id"] != opened["run_id"]:
                fail(f"continuation did not create a child run: {sessions!r}")
        finally:
            host.close()


@check("host_sessions.original_untouched")
def check_original_untouched() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client, run_id = _finished_session(root)
        try:
            original = (root / "sessions" / run_id / "run.jsonl").read_bytes()
            client.open_session(run_id)
            client.send_prompt("second")
            _wait_idle(client)
            if (root / "sessions" / run_id / "run.jsonl").read_bytes() != original:
                fail("continuation appended to opened transcript")
        finally:
            host.close()


@check("host_sessions.offloaded_handle_survives")
def check_offloaded_handle_survives() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client, run_id = _finished_session(root)
        try:
            client.open_session(run_id)
            if not client.send_prompt("continue").get("accepted"):
                fail("opened session could not continue with result fallback configured")
        finally:
            host.close()


@check("host_sessions.client_session_calls")
def check_client_session_calls() -> None:
    with tempfile.TemporaryDirectory() as directory:
        host, client, run_id = _finished_session(Path(directory))
        try:
            if not client.list_sessions() or client.open_session(run_id)["replayed"] != 2:
                fail("client session calls did not round-trip")
        finally:
            host.close()
