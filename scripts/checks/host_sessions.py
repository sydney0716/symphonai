"""Checks for reopening persisted host sessions."""

from __future__ import annotations

import inspect
import tempfile
import threading
import time
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from symphonai_api.events import RunFinished
from symphonai_api.models import Message, ModelResponse, Role, ToolCall
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_host.client import HostAddress, HostClient, HostClientError
from symphonai_host.protocol import decode_event
from symphonai_host.server import HostServer
from symphonai_host.sessions import list_sessions
from scripts.checks.host_server import _await_sse, _headers, _request, _subscribed_stream
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


def _listing_fixture(root: Path) -> None:
    sessions_root = root / "sessions"
    for index, run_id in enumerate(("oldest", "third", "second", "newest")):
        directory = sessions_root / run_id
        directory.mkdir(parents=True)
        (directory / "meta.json").write_text(
            json.dumps({"run_id": run_id, "updated_at": f"2026-01-0{index + 1}T00:00:00Z"}),
            encoding="utf-8",
        )
        (directory / "run.jsonl").write_text("not a transcript", encoding="utf-8")


@check("host_sessions.list_order_and_fields")
def check_list_order_and_fields() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client, run_id = _finished_session(root)
        try:
            sessions = client.list_sessions()
            expected_fields = {"run_id", "title", "created_at", "updated_at", "stopped_reason", "parent_run_id", "repo_root", "state", "message_count"}
            meta = json.loads(
                (root / "sessions" / run_id / "meta.json").read_text(encoding="utf-8")
            )
            if (
                sessions[0]["run_id"] != run_id
                or set(sessions[0]) != expected_fields
                or sessions[0]["repo_root"] != str(root.resolve())
                or meta.get("repo_root") != str(root.resolve())
            ):
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
        legacy = root / "sessions" / "legacy"
        legacy.mkdir()
        (legacy / "meta.json").write_text(
            json.dumps({"run_id": "legacy"}), encoding="utf-8"
        )
        result = {item["run_id"]: item for item in list_sessions(root / "sessions")}
        expected = {
            run_id: {
                "run_id": run_id,
                "title": None,
                "created_at": None,
                "updated_at": None,
                "stopped_reason": None,
                "parent_run_id": None,
                "repo_root": "",
                "state": "unreadable",
                "message_count": 0,
            }
            for run_id in ("broken", "legacy")
        }
        if result != expected:
            fail(f"damaged session was omitted or misclassified: {result!r}")


@check("host_sessions.empty_root")
def check_empty_root() -> None:
    with tempfile.TemporaryDirectory() as directory:
        if list_sessions(Path(directory) / "missing") != []:
            fail("missing sessions root was not empty")


@check("host_sessions.limited_listing")
def check_limited_listing() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _listing_fixture(root)
        loaded = []
        classified = []

        def load_selected(store):
            if store.run_id not in ("newest", "second"):
                raise AssertionError(f"unreturned transcript was loaded: {store.run_id}")
            loaded.append(store.run_id)
            return SimpleNamespace(messages=[Message(Role.USER, store.run_id)])

        def read_selected(path):
            if path.parent.name not in ("newest", "second"):
                raise AssertionError(f"unreturned run.jsonl was read: {path}")
            return [], 0

        def classify_selected(loaded_run, records):
            classified.append(loaded_run.messages[0].text)
            return SimpleNamespace(state=SimpleNamespace(value="completed"))

        with (
            mock.patch("symphonai_host.sessions.load_run", side_effect=load_selected),
            mock.patch("symphonai_host.sessions.read_records", side_effect=read_selected),
            mock.patch("symphonai_host.sessions.classify_run", side_effect=classify_selected),
        ):
            result = list_sessions(root / "sessions", limit=2)
        if [item["run_id"] for item in result] != ["newest", "second"]:
            fail(f"limited listing selected the wrong sessions: {result!r}")
        if loaded != ["newest", "second"] or classified != loaded:
            fail(f"limited listing classified the wrong sessions: {loaded!r}, {classified!r}")
        if [item["state"] for item in result] != ["completed", "completed"]:
            fail(f"limited listing did not classify returned sessions: {result!r}")


@check("host_sessions.limit_route")
def check_limit_route() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _listing_fixture(root)
        host, client = _host(root)
        try:
            limited = client.list_sessions(limit=2)
            if not isinstance(limited, list) or [item["run_id"] for item in limited] != ["newest", "second"]:
                fail(f"Python client did not receive a bare limited array: {limited!r}")
            for path in ("/sessions", "/sessions?limit=", "/sessions?limit=abc", "/sessions?limit=0", "/sessions?limit=-1"):
                connection, response = _request(host, "GET", path, headers=_headers(host))
                try:
                    body = response.read()
                    items = json.loads(body)
                    if response.status != 200 or not isinstance(items, list) or len(items) != 4:
                        fail(f"sessions query failed open listing for {path!r}: {response.status}, {body!r}")
                finally:
                    connection.close()
        finally:
            host.close()


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
        connection, response = _subscribed_stream(host)
        try:
            client.send_prompt("active")
            try:
                client.open_session(run_id)
            except HostClientError as exc:
                if "409" not in str(exc):
                    fail(f"active session did not return 409: {exc}")
            else:
                fail("opened while active")

            _await_sse(
                connection,
                response,
                lambda frame: isinstance(frame, tuple)
                and frame[0] == "event"
                and isinstance(decode_event(frame[1]), RunFinished),
                what="active run terminal event",
            )
            _wait_idle(client)

            source = inspect.getsource(check_open_during_run_409)
            if "time." + "sleep(" in source:
                fail("open-during-run cleanup used time.sleep")
            if "_await" + "_sse(" not in source or "_wait" + "_idle(client)" not in source:
                fail("open-during-run cleanup did not wait for a terminal state")
            for removal in ("rmtree" + "(", ".clean" + "up(", ".un" + "link("):
                if removal in source:
                    fail("open-during-run cleanup retries or performs removal directly")
        finally:
            connection.close()
            host.close()


def _open_with_history(root: Path) -> tuple[HostServer, HostClient, str, list[dict], dict]:
    marker = "history-secret-argument"
    host, client = _host(root, [
        ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall("history-tool", "run_shell", {"argv": [marker]})])),
        ModelResponse(Message(Role.ASSISTANT, "done")),
    ])
    run_id = client.send_prompt("first")["run_id"]
    _wait_idle(client)
    frames: list[dict] = []
    ready = threading.Event()

    def consume() -> None:
        try:
            ready.set()
            for kind, payload in client.events():
                if kind == "event" and payload.get("type") == "HistoryMessage":
                    frames.append(payload)
                    if len(frames) == 4:
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
            expected = ["user", "assistant", "tool", "assistant"]
            deadline = time.monotonic() + 5
            while len(frames) < len(expected) and time.monotonic() < deadline:
                time.sleep(0.02)
            if len(frames) < len(expected):
                fail(f"history replay timed out with {len(frames)} frames: {frames!r}")
            if [frame["role"] for frame in frames] != expected:
                fail(f"history replay was out of order: {frames!r}")
            if any(set(call) != {"id", "name"} for frame in frames for call in frame["tool_calls"]):
                fail(f"history tool calls leaked fields: {frames!r}")
            if "history-secret-argument" in json.dumps(frames):
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
