"""Checks for reopening persisted host sessions."""

from __future__ import annotations

import inspect
import contextlib
import io
import os
import shutil
import tempfile
import threading
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from symphonai_api.events import RunFinished, SessionEnded
from symphonai_api.session import SessionStore, load_run, read_records
from symphonai_api.models import Message, ModelResponse, Role, ToolCall
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
import symphonai_host.__main__ as host_main
from symphonai_host.client import HostAddress, HostClient, HostClientError
from symphonai_host.protocol import decode_event
from symphonai_host.server import HostServer
from symphonai_host.sessions import DEFAULT_CLEANUP_PERIOD_DAYS, list_sessions, prune_sessions
from scripts.checks.host_server import _WaitingProvider, _await_sse, _headers, _request, _subscribed_stream
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


@check("host_sessions.title_set_once")
def check_conversation_title_set_once() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client = _host(root, [
            ModelResponse(Message(Role.ASSISTANT, "first answer")),
            ModelResponse(Message(Role.ASSISTANT, "second answer")),
        ])
        try:
            run_id = client.send_prompt("First conversation title")["run_id"]
            _wait_idle(client)
            meta_path = root / "sessions" / run_id / "meta.json"
            first_title = json.loads(meta_path.read_text(encoding="utf-8"))["title"]
            client.send_prompt("A later prompt must not rename it")
            _wait_idle(client)
            second_title = json.loads(meta_path.read_text(encoding="utf-8"))["title"]
            if first_title != "First conversation title" or second_title != first_title:
                fail(f"conversation title was missing or rewritten: {first_title!r}, {second_title!r}")
        finally:
            host.close()


@check("host_sessions.title_normalized")
def check_conversation_title_normalized() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client = _host(root)
        prompt = "  A title\nwith\tseveral   spaces " + "z" * 100
        expected = " ".join(prompt.split())[:80]
        try:
            run_id = client.send_prompt(prompt)["run_id"]
            _wait_idle(client)
            meta = json.loads(
                (root / "sessions" / run_id / "meta.json").read_text(encoding="utf-8")
            )
            listed = {session["run_id"]: session for session in client.list_sessions()}
            if meta["title"] != expected or listed[run_id]["title"] != expected:
                fail(f"normalized title did not reach metadata and listing: {meta!r}, {listed!r}")
            if len(expected) != 80 or "\n" in expected or "\t" in expected:
                fail(f"title was not normalized to the 80-character limit: {expected!r}")
        finally:
            host.close()


@check("host_sessions.current_conversation")
def check_current_conversation() -> None:
    class RecordingProvider(FakeModelProvider):
        def __init__(self) -> None:
            super().__init__([
                ModelResponse(Message(Role.ASSISTANT, "first answer")),
                ModelResponse(Message(Role.ASSISTANT, "second answer")),
            ])
            self.requests = []

        def create_response(self, request, *, cancel=None):  # noqa: ANN001
            self.requests.append(request)
            return super().create_response(request, cancel=cancel)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provider = RecordingProvider()
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        subscription = host.broker.subscribe()
        try:
            first = HostClient(HostAddress(host.port, host.token)).send_prompt("first question")["run_id"]
            client = HostClient(HostAddress(host.port, host.token))
            _wait_idle(client)
            second = client.send_prompt("second question")["run_id"]
            _wait_idle(client)
            if first == second or len(provider.requests) != 2:
                fail("two prompts did not run separately in one conversation")
            sent = [message.text for message in provider.requests[1].messages]
            if sent != ["first question", "first answer", "second question"]:
                fail(f"second provider request lost the first exchange: {sent!r}")
            directories = [path for path in (root / "sessions").iterdir() if path.is_dir()]
            records, _ = read_records(root / "sessions" / first / "run.jsonl")
            loaded = load_run(SessionStore.open(root / "sessions", first))
            if len(directories) != 1 or directories[0].name != first:
                fail(f"two prompts did not share the first run's directory: {directories!r}")
            if sum(record["type"] == "run_started" for record in records) != 2 or loaded.run_count != 2:
                fail("conversation transcript did not contain two runs")
            if [message.text for message in loaded.messages] != [
                "first question", "first answer", "second question", "second answer"
            ]:
                fail(f"conversation transcript did not rebuild in order: {loaded.messages!r}")
            connection, response = _request(host, "POST", "/session/new", body={}, headers=_headers(host))
            try:
                if response.status != 200 or json.loads(response.read()) != {"ended": True}:
                    fail("session/new did not end the current conversation")
            finally:
                connection.close()
            events = []
            while (event := subscription.get(timeout=0.01)) is not None:
                events.append(event)
            if sum(isinstance(event, SessionEnded) for event in events) != 1:
                fail(f"SessionEnded was not emitted once per conversation: {events!r}")
        finally:
            subscription.close()
            host.close()


@check("host_sessions.new_conversation_route")
def check_new_conversation_route() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provider = _WaitingProvider()
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        client = HostClient(HostAddress(host.port, host.token))
        try:
            first = client.send_prompt("first")["run_id"]
            connection, response = _request(host, "POST", "/session/new", body={}, headers=_headers(host))
            try:
                if response.status != 409 or json.loads(response.read()).get("run_id") != first:
                    fail("session/new did not preserve the active-run conflict")
            finally:
                connection.close()
            provider.release.set()
            _wait_idle(client)
            connection, response = _request(host, "POST", "/session/new", body={}, headers=_headers(host))
            try:
                if response.status != 200 or json.loads(response.read()) != {"ended": True}:
                    fail("session/new did not close the finished conversation")
            finally:
                connection.close()
            second = client.send_prompt("second")["run_id"]
            _wait_idle(client)
            loaded = load_run(SessionStore.open(root / "sessions", second))
            if second == first or len(list((root / "sessions").iterdir())) != 2:
                fail("a new conversation did not create a second session directory")
            if [message.text for message in loaded.messages] != ["second", "done"]:
                fail(f"new conversation retained prior history: {loaded.messages!r}")
        finally:
            provider.release.set()
            host.close()


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
            client.open_session(run_id)
            with mock.patch.object(
                host.run._provider,
                "create_response",
                wraps=host.run._provider.create_response,
            ) as response_spy:
                continued = client.send_prompt("second")
                _wait_idle(client)
            sent = [message.text for message in response_spy.call_args.args[0].messages]
            if sent != ["first", "done", "second"]:
                fail(f"reopened conversation did not reach the provider: {sent!r}")
            sessions = {item["run_id"]: item for item in client.list_sessions()}
            loaded = load_run(SessionStore.open(root / "sessions", run_id))
            if continued["run_id"] == run_id or set(sessions) != {run_id} or loaded.run_count != 2:
                fail(f"continuation did not append a second run to the conversation: {sessions!r}")
            if [message.text for message in loaded.messages] != ["first", "done", "second", "done"]:
                fail(f"continued conversation did not rebuild in order: {loaded.messages!r}")
        finally:
            host.close()


@check("host_sessions.instructions_not_reloaded")
def check_instructions_not_reloaded() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        instructions = root / ".symphonai" / "INSTRUCTIONS.md"
        instructions.parent.mkdir()
        instructions.write_text("original convention", encoding="utf-8")
        with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": str(root / "missing-user-home")}):
            host, client = _host(root)
            try:
                run_id = client.send_prompt("first")["run_id"]
                _wait_idle(client)
                instructions.write_text("changed convention", encoding="utf-8")
                client.open_session(run_id)
                with mock.patch.object(host.run._provider, "create_response", wraps=host.run._provider.create_response) as response_spy:
                    client.send_prompt("second")
                    _wait_idle(client)
                sent = response_spy.call_args.args[0].messages
                system = [message.text for message in sent if message.role == Role.SYSTEM]
                if len(system) != 1 or "original convention" not in system[0] or "changed convention" in system[0]:
                    fail(f"reopening reloaded or duplicated instructions: {system!r}")
            finally:
                host.close()


@check("host_sessions.reopened_appends_to_original")
def check_reopened_appends_to_original() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client, run_id = _finished_session(root)
        try:
            original = (root / "sessions" / run_id / "run.jsonl").read_bytes()
            client.open_session(run_id)
            client.send_prompt("second")
            _wait_idle(client)
            updated = (root / "sessions" / run_id / "run.jsonl").read_bytes()
            if not updated.startswith(original) or len(updated) == len(original):
                fail("continuation did not append to the opened transcript")
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


def _prune_fixture(root: Path, name: str, updated_at: str | None) -> None:
    directory = root / name
    directory.mkdir()
    meta = {"run_id": name}
    if updated_at is not None:
        meta["updated_at"] = updated_at
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


@check("host_sessions.prune_selective")
def check_prune_selective() -> None:
    now = datetime(2026, 2, 1, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "sessions"
        root.mkdir()
        _prune_fixture(root, "old", (now - timedelta(days=31)).isoformat())
        _prune_fixture(root, "recent", (now - timedelta(days=29)).isoformat())
        _prune_fixture(root, "undated", None)
        _prune_fixture(root, "unparseable", "not a timestamp")
        broken = root / "invalid-json"
        broken.mkdir()
        (broken / "meta.json").write_text("{", encoding="utf-8")
        (root / "loose-file").write_text("not a session", encoding="utf-8")

        if prune_sessions(root, period_days=30, now=now) != 1:
            fail("pruning did not remove exactly one dated old session")
        if {path.name for path in root.iterdir()} != {
            "recent", "undated", "unparseable", "invalid-json", "loose-file"
        }:
            fail("pruning removed a recent or undated entry")


@check("host_sessions.prune_boundaries")
def check_prune_boundaries() -> None:
    now = datetime(2026, 2, 1, tzinfo=timezone.utc)
    cutoff = now - timedelta(days=30)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "sessions"
        root.mkdir()
        _prune_fixture(root, "exact", cutoff.isoformat())
        _prune_fixture(root, "older", (cutoff - timedelta(seconds=1)).isoformat())
        _prune_fixture(root, "stuck", (cutoff - timedelta(days=1)).isoformat())
        if prune_sessions(root, period_days=0, now=now) != 0:
            fail("disabled pruning removed a session")
        if {path.name for path in root.iterdir()} != {"exact", "older", "stuck"}:
            fail("disabled pruning changed the session set")

        actual_remove = shutil.rmtree

        def remove(directory: Path) -> None:
            if directory.name == "stuck":
                raise OSError("fixture refuses removal")
            actual_remove(directory)

        with mock.patch("symphonai_host.sessions.shutil.rmtree", side_effect=remove):
            removed = prune_sessions(root, period_days=30, now=now)
        if removed != 1 or {path.name for path in root.iterdir()} != {"exact", "stuck"}:
            fail("cutoff or unremovable-directory handling was wrong")
        absent = Path(temporary) / "missing"
        if prune_sessions(absent, period_days=30, now=now) != 0 or absent.exists():
            fail("missing sessions root was created or failed pruning")


@check("host_sessions.prune_startup")
def check_prune_startup() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo = Path(temporary) / "repo"
        repo.mkdir()
        sessions = Path(temporary) / "sessions"
        for values, expected, raises in (
            ({"sessions.cleanup_period_days": 7}, 7, False),
            ({}, DEFAULT_CLEANUP_PERIOD_DAYS, False),
            ({"sessions.cleanup_period_days": 7}, 7, True),
        ):
            events: list[str] = []
            observed: list[tuple[Path, int, datetime]] = []
            extensions = SimpleNamespace(
                config=SimpleNamespace(values=values), mcp_servers=()
            )
            host = mock.Mock()
            host.print_handshake.side_effect = lambda: (events.append("handshake"), print("handshake"))
            host.serve_forever.side_effect = lambda: events.append("serve")
            pool = mock.Mock()
            pool.start.return_value = {}

            def prune(root: Path, *, period_days: int, now: datetime) -> int:
                events.append("prune")
                observed.append((root, period_days, now))
                if raises:
                    raise OSError("fixture pruning failure")
                return 0

            output = io.StringIO()
            with (
                mock.patch.object(host_main, "load", return_value={}),
                mock.patch.object(host_main, "load_extensions", return_value=extensions),
                mock.patch.object(host_main, "default_sessions_root", return_value=sessions),
                mock.patch.object(host_main, "prune_sessions", side_effect=prune),
                mock.patch.object(host_main, "McpPool", return_value=pool),
                mock.patch.object(host_main, "HostServer", return_value=host),
                mock.patch.object(host_main, "_provider"),
                mock.patch.object(host_main, "standard_tool_registry", return_value={}),
                mock.patch.object(host_main.signal, "signal"),
                contextlib.redirect_stdout(output),
            ):
                host_main.main(["--repo-root", str(repo)])
            if events != ["prune", "handshake", "serve"]:
                fail(f"startup pruning or handshake order changed: {events!r}")
            if output.getvalue().splitlines()[0] != "handshake":
                fail("handshake was not the first stdout line")
            if (
                len(observed) != 1
                or observed[0][0] != sessions
                or observed[0][1] != expected
                or observed[0][2].tzinfo is None
            ):
                fail(f"startup pruning arguments were wrong: {observed!r}")
