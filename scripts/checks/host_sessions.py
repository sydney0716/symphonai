"""Checks for reopening persisted host sessions."""

from __future__ import annotations

import inspect
import contextlib
import io
import os
import shutil
import sys
import tempfile
import threading
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from symphonai_api.events import RunFinished, SessionEnded
from symphonai_api.checkpoints import CheckpointStore
from symphonai_api.session import SessionStore, load_run, read_records
from symphonai_api.models import Message, ModelResponse, Role, ToolCall
from symphonai_api.serialization import message_to_json
from symphonai_api.identity import new_id
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
import symphonai_host.__main__ as host_main
from symphonai_host.client import HostAddress, HostClient, HostClientError
from symphonai_host.protocol import decode_event
from symphonai_host.run import ProviderSelectionError
from symphonai_host.server import HostServer
from symphonai_host.sessions import DEFAULT_CLEANUP_PERIOD_DAYS, list_sessions, prompt_history, prune_sessions
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


def _new_session(host: HostServer) -> None:
    connection, response = _request(host, "POST", "/session/new", body={}, headers=_headers(host))
    try:
        if response.status != 200:
            fail(f"session/new returned {response.status}: {response.read()!r}")
        response.read()
    finally:
        connection.close()


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


def _history_fixture(root: Path, name: str, repo_root: Path, updated_at: str, prompts: list[str]) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "meta.json").write_text(json.dumps({
        "run_id": name,
        "repo_root": str(repo_root),
        "updated_at": updated_at,
    }), encoding="utf-8")
    records = []
    for index, prompt in enumerate(prompts):
        records.append({
            "schema_version": 1,
            "record_id": f"rec-{name}-{index}",
            "ts": updated_at,
            "type": "message",
            "run_id": name,
            "agent_id": "agent-1",
            "turn_id": None,
            "data": message_to_json(Message(Role.USER, prompt)),
        })
    (directory / "run.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )


@check("host_sessions.prompt_history_order_and_filtering")
def check_prompt_history_order_and_filtering() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "sessions"
        repo = Path(directory) / "project"
        other_repo = Path(directory) / "other"
        _history_fixture(root, "older", repo, "2026-01-01T00:00:00Z", ["a", "b"])
        _history_fixture(root, "newer", repo, "2026-01-02T00:00:00Z", ["c", "c", "d"])
        _history_fixture(root, "other", other_repo, "2026-01-03T00:00:00Z", ["ignore"])
        _history_fixture(root, "goal", repo, "2026-01-04T00:00:00Z", [
            "objective",
            "Goal check failed (round 1): retry",
            "Round 2 of 10 ended without the goal reported complete.",
            "",
        ])
        prompts = prompt_history(root, repo)
        if prompts != ["objective", "d", "c", "b", "a"]:
            fail(f"prompt history was ordered or filtered incorrectly: {prompts!r}")


@check("host_sessions.prompt_history_limits_and_damage")
def check_prompt_history_limits_and_damage() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sessions = root / "sessions"
        _history_fixture(sessions, "new", root.resolve(), "2099-01-02T00:00:00Z", ["c", "d"])
        unreadable = sessions / "damaged"
        unreadable.mkdir()
        (unreadable / "meta.json").write_text(json.dumps({
            "run_id": "damaged", "repo_root": str(root.resolve()), "updated_at": "2099-01-03T00:00:00Z",
        }), encoding="utf-8")
        (unreadable / "run.jsonl").write_text("not a transcript", encoding="utf-8")
        _history_fixture(sessions, "old", root.resolve(), "2099-01-01T00:00:00Z", ["a", "b"])
        if prompt_history(sessions, root.resolve(), limit=2) != ["d", "c"]:
            fail("history limit did not stop in newest-first order")
        host, _ = _host(root)
        try:
            connection, response = _request(host, "GET", "/history?limit=2", headers=_headers(host))
            payload = json.loads(response.read())
            connection.close()
            if response.status != 200 or payload != {"prompts": ["d", "c"]}:
                fail(f"history route did not skip damage or apply its limit: {response.status}, {payload!r}")
            for value in ("0", "501", "x"):
                connection, response = _request(host, "GET", f"/history?limit={value}", headers=_headers(host))
                response.read()
                connection.close()
                if response.status != 400:
                    fail(f"invalid history limit {value!r} returned {response.status}")
        finally:
            host.close()


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
            sent = [
                message.text for message in provider.requests[1].messages
                if message.role != Role.SYSTEM
            ]
            if sent != ["first question", "first answer", "second question"]:
                fail(f"second provider request lost the first exchange: {sent!r}")
            directories = [path for path in (root / "sessions").iterdir() if path.is_dir()]
            records, _ = read_records(root / "sessions" / first / "run.jsonl")
            loaded = load_run(SessionStore.open(root / "sessions", first))
            if len(directories) != 1 or directories[0].name != first:
                fail(f"two prompts did not share the first run's directory: {directories!r}")
            if sum(record["type"] == "run_started" for record in records) != 2 or loaded.run_count != 2:
                fail("conversation transcript did not contain two runs")
            if [message.text for message in loaded.messages if message.role != Role.SYSTEM] != [
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
    class SplitProvider(FakeModelProvider):
        def __init__(self):
            super().__init__([])
            self.first_started = threading.Event()
            self.release_first = threading.Event()
            self.requests = []

        def create_response(self, request, *, cancel=None):
            self.requests.append(request)
            prompt = next(message.text for message in reversed(request.messages) if message.role is Role.USER)
            if prompt == "first":
                self.first_started.set()
                self.release_first.wait(5)
            return ModelResponse(Message(Role.ASSISTANT, f"answer to {prompt}"))

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provider = SplitProvider()
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        client = HostClient(HostAddress(host.port, host.token))
        stream_connection, stream_response = _subscribed_stream(host)
        try:
            first = client.send_prompt("first")["run_id"]
            if not provider.first_started.wait(2):
                fail("first conversation did not reach the blocking provider")
            connection, response = _request(host, "POST", "/session/new", body={}, headers=_headers(host))
            try:
                if response.status != 200 or json.loads(response.read()) != {"ended": True}:
                    fail("session/new did not succeed while the first run continued")
            finally:
                connection.close()
            second = client.send_prompt("second")["run_id"]
            _wait_idle(client)
            if second == first or first not in host.run._active_by_session:
                fail("second conversation did not finish independently while the first stayed active")
            leader_a = host.run._open_conversations[first][0]
            provider.release_first.set()
            deadline = time.monotonic() + 5
            while first in host.run._active_by_session and time.monotonic() < deadline:
                time.sleep(0.01)
            if first in host.run._active_by_session:
                fail("first conversation did not finish after it was released")
            while first in host.run._open_conversations and time.monotonic() < deadline:
                time.sleep(0.01)
            if first in host.run._open_conversations:
                fail("idle noncurrent conversation stayed open after its run finished")
            host.run.open_session(first)
            if host.run._conversation[0] is leader_a:
                fail("reopening a closed conversation reused its old leader")
            loaded = load_run(SessionStore.open(root / "sessions", first))
            if "answer to first" not in [message.text for message in loaded.messages]:
                fail("reopening the first session did not show its completed answer")
            frames = []
            for expected in (first, second):
                frame = _await_sse(
                    stream_connection, stream_response,
                    lambda item: item[0] == "event" and item[1].get("type") in ("RunFinished", "RunFailed"),
                    what=f"terminal event for {expected}",
                )
                frames.append(frame[1])
            if {item.get("session_id") for item in frames} != {first, second}:
                fail(f"conversation event frames lost their session tags: {frames!r}")
            if len(list(path for path in (root / "sessions").iterdir() if path.is_dir())) != 2:
                fail("a new conversation did not create a second session directory")
            loaded_b = load_run(SessionStore.open(root / "sessions", second))
            if [message.text for message in loaded_b.messages if message.role != Role.SYSTEM] != ["second", "answer to second"]:
                fail(f"new conversation retained prior history: {loaded_b.messages!r}")
        finally:
            provider.release_first.set()
            host.close()


@check("host_sessions.list_order_and_fields")
def check_list_order_and_fields() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client, run_id = _finished_session(root)
        try:
            sessions = client.list_sessions()
            expected_fields = {"run_id", "title", "created_at", "updated_at", "stopped_reason", "parent_run_id", "parent_session_id", "repo_root", "state", "message_count", "activity"}
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
                "parent_session_id": None,
                "repo_root": "",
                "state": "unreadable",
                "message_count": 0,
                "activity": "idle",
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
        try:
            client.send_prompt("active")
            leader = host.run._open_conversations[run_id][0]
            client.open_session(run_id)
            if host.run._conversation[0] is not leader:
                fail("opening a running session rebuilt its leader")
            client.stop()
            _wait_idle(client)
            if run_id in host.run._active_by_session:
                fail("stop did not cancel the current conversation")
        finally:
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
                    if len(frames) == 5:
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
            expected = ["system", "user", "assistant", "tool", "assistant"]
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
            sent = [
                message.text for message in response_spy.call_args.args[0].messages
                if message.role != Role.SYSTEM
            ]
            if sent != ["first", "done", "second"]:
                fail(f"reopened conversation did not reach the provider: {sent!r}")
            sessions = {item["run_id"]: item for item in client.list_sessions()}
            loaded = load_run(SessionStore.open(root / "sessions", run_id))
            if continued["run_id"] == run_id or set(sessions) != {run_id} or loaded.run_count != 2:
                fail(f"continuation did not append a second run to the conversation: {sessions!r}")
            if [message.text for message in loaded.messages if message.role != Role.SYSTEM] != ["first", "done", "second", "done"]:
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
                initial_system: list[str] = []
                with mock.patch.object(
                    host.run._provider,
                    "create_response",
                    wraps=host.run._provider.create_response,
                ) as initial_spy:
                    client.send_prompt("capture initial environment")
                    _wait_idle(client)
                initial_system = [
                    message.text
                    for message in initial_spy.call_args.args[0].messages
                    if message.role == Role.SYSTEM
                ]
                instructions.write_text("changed convention", encoding="utf-8")
                client.open_session(run_id)
                with mock.patch.object(host.run._provider, "create_response", wraps=host.run._provider.create_response) as response_spy:
                    client.send_prompt("second")
                    _wait_idle(client)
                sent = response_spy.call_args.args[0].messages
                system = [message.text for message in sent if message.role == Role.SYSTEM]
                if (
                    len(system) != 2
                    or "original convention" not in system[0]
                    or "changed convention" in system[0]
                    or not system[1].startswith("Environment when this conversation started")
                    or system != initial_system
                ):
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
            if not client.list_sessions() or client.open_session(run_id)["replayed"] != 3:
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
                config=SimpleNamespace(
                    values=values,
                    get=lambda key, default=None: values.get(key, default),
                ),
                mcp_servers=(), lsp_servers=(),
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


def _fork(host: HostServer, run_id: str, record_id: str, *, force: bool = False) -> tuple[int, dict]:
    connection, response = _request(
        host, "POST", "/session/fork",
        body={"run_id": run_id, "record_id": record_id, **({"force": True} if force else {})},
        headers=_headers(host),
    )
    try:
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def _checkpointed_two_prompt_session(root: Path):
    host, client, source_id = _finished_session(root)
    source = SessionStore.open(root / "sessions", source_id)
    try:
        user_agent = next(
            record["agent_id"] for record in read_records(source.directory / "run.jsonl")[0]
            if record.get("type") == "message"
            and record.get("data", {}).get("role") == "user"
        )
        second_turn = new_id("turn")
        writer = source.writer_for(user_agent, is_root=True)
        second_user = writer.append(
            "message", run_id=source_id, agent_id=user_agent, turn_id=second_turn,
            data=message_to_json(Message(Role.USER, "second prompt", turn_id=second_turn)),
        )
        second_answer = writer.append(
            "message", run_id=source_id, agent_id=user_agent, turn_id=second_turn,
            data=message_to_json(Message(Role.ASSISTANT, "second answer", turn_id=second_turn)),
        )
        raw_records, _ = read_records(source.directory / "run.jsonl")
        user_records = [
            record for record in raw_records
            if record.get("type") == "message" and record.get("data", {}).get("role") == "user"
        ]
        keys = ["prompt-one", "prompt-two"]
        marked = []
        for record in raw_records:
            if record in user_records:
                key = keys[user_records.index(record)]
                marker = dict(record)
                marker.update({"record_id": new_id("rec"), "type": "checkpoint", "turn_id": None, "data": {"key": key}})
                marked.append(marker)
            marked.append(record)
        (source.directory / "run.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in marked),
            encoding="utf-8",
        )
    finally:
        source.close()
    a = root / "a.py"
    b = root / "b.py"
    a.write_text("base\n", encoding="utf-8")
    checkpoints = CheckpointStore(root / "sessions" / source_id / "checkpoints", root)
    checkpoints.begin("prompt-one")
    checkpoints.before_write(a)
    a.write_text("after one\n", encoding="utf-8")
    checkpoints.after_write(a)
    checkpoints.begin("prompt-two")
    checkpoints.before_write(a)
    a.write_text("after two\n", encoding="utf-8")
    checkpoints.after_write(a)
    checkpoints.before_write(b)
    b.write_text("new file\n", encoding="utf-8")
    checkpoints.after_write(b)
    return host, client, source_id, user_records[0]["record_id"], second_user, second_answer, a, b


@check("host_sessions.fork_restores_prefix_checkpoints")
def check_fork_restores_prefix_checkpoints() -> None:
    for fork_at, expected_a, expect_b, expected_keys in (
        ("second-user", "after one\n", False, ["prompt-one"]),
        ("first-answer", "after one\n", False, ["prompt-one"]),
        ("second-answer", "after two\n", True, ["prompt-one", "prompt-two"]),
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host, _, source_id, first_user, second_user, second_answer, a, b = _checkpointed_two_prompt_session(root)
            try:
                source = SessionStore.open(root / "sessions", source_id)
                try:
                    records, _ = read_records(source.directory / "run.jsonl")
                    first_answer = next(
                        record["record_id"] for record in records
                        if record.get("type") == "message"
                        and record.get("data", {}).get("role") == "assistant"
                    )
                finally:
                    source.close()
                target = {
                    "second-user": second_user,
                    "first-answer": first_answer,
                    "second-answer": second_answer,
                }[fork_at]
                status, reply = _fork(host, source_id, target)
                if status != 200:
                    fail(f"checkpoint fork failed at {fork_at}: {status}, {reply!r}")
                if a.read_text(encoding="utf-8") != expected_a or b.exists() != expect_b:
                    fail(f"fork at {fork_at} left the wrong files: a={a.read_text()!r}, b={b.exists()}")
                changes = host.run.changes()
                keys = [turn["key"] for turn in changes["turns"]]
                if keys != expected_keys:
                    fail(f"fork at {fork_at} carried the wrong checkpoint turns: {keys!r}")
                if fork_at == "second-user" and [file["path"] for file in changes["files"]] != ["a.py"]:
                    fail(f"fork at prompt two user did not retain only prompt one's file: {changes!r}")
            finally:
                host.close()


@check("host_sessions.fork_conflict_force_and_rollback")
def check_fork_conflict_force_and_rollback() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, _, source_id, _, second_user, _, a, b = _checkpointed_two_prompt_session(root)
        try:
            b.write_text("hand edited\n", encoding="utf-8")
            before_files = (a.read_bytes(), b.read_bytes())
            before_sessions = {path.name for path in (root / "sessions").iterdir() if path.is_dir()}
            status, reply = _fork(host, source_id, second_user)
            after_sessions = {path.name for path in (root / "sessions").iterdir() if path.is_dir()}
            if status != 409 or reply.get("paths") != ["b.py"]:
                fail(f"outside file edit did not block the fork with its path: {status}, {reply!r}")
            if (a.read_bytes(), b.read_bytes()) != before_files or after_sessions != before_sessions:
                fail("conflicted fork changed files or created a session")
            status, _ = _fork(host, source_id, second_user, force=True)
            if status != 200 or a.read_text(encoding="utf-8") != "after one\n" or b.exists():
                fail("forced fork did not restore the requested file state")
        finally:
            host.close()


@check("host_sessions.fork_without_checkpoints_preserves_files")
def check_fork_without_checkpoints_preserves_files() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, _, source_id = _finished_session(root)
        try:
            source = SessionStore.open(root / "sessions", source_id)
            try:
                record_id = load_run(source).record_ids[-1]
                records, _ = read_records(source.directory / "run.jsonl")
                without_checkpoints = [record for record in records if record.get("type") != "checkpoint"]
                (source.directory / "run.jsonl").write_text(
                    "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in without_checkpoints),
                    encoding="utf-8",
                )
                shutil.rmtree(source.directory / "checkpoints", ignore_errors=True)
            finally:
                source.close()
            path = root / "untouched.py"
            path.write_bytes(b"outside checkpoint history\x00\xff")
            before = path.read_bytes()
            status, reply = _fork(host, source_id, record_id)
            if status != 200 or not reply.get("run_id"):
                fail(f"fork without checkpoints failed: {status}, {reply!r}")
            if path.read_bytes() != before:
                fail("fork without checkpoints changed an unrelated file")
        finally:
            host.close()



@check("host_sessions.goal_reopen_and_fork")
def check_goal_reopen_and_fork() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, _, source_id = _finished_session(root)
        try:
            host.run.start_goal(
                "keep working", (sys.executable, "-c", "import time; time.sleep(20)"), 3,
            )
            deadline = time.monotonic() + 5
            while (
                (host.run._goal_check is None or host.run._goal_check.process is None)
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            if host.run._goal_check is None or host.run._goal_check.process is None:
                fail("goal completion check did not start")
            host.run.stop()
            deadline = time.monotonic() + 5
            while host.run._goal_check is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            source = SessionStore.open(root / "sessions", source_id)
            try:
                meta = source.read_meta()
                meta["goal"]["phase"] = "active"
                meta["goal"]["reason"] = ""
                source.write_meta(meta)
                last_record = load_run(source).record_ids[-1]
            finally:
                source.close()
            host.run.open_session(source_id)
            reopened = host.run.conversation_stats()["goal"]
            if host.run._conversation[0] is not host.run._open_conversations[source_id][0]:
                fail("opening an already-open session rebuilt its leader")
            host.run.fork_session(source_id, last_record)
            if host.run.goal_snapshot() is not None:
                fail("fork inherited its source goal")
        finally:
            host.close()


@check("host_sessions.fork_reopen_failure_rolls_back_files")
def check_fork_reopen_failure_rolls_back_files() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, _, source_id, _, second_user, _, a, b = _checkpointed_two_prompt_session(root)
        try:
            before_files = (a.read_bytes(), b.read_bytes())
            before_sessions = {path.name for path in (root / "sessions").iterdir() if path.is_dir()}
            with mock.patch.object(host.run, "open_session", side_effect=RuntimeError("reopen failed")):
                try:
                    host.run.fork_session(source_id, second_user)
                except RuntimeError as exc:
                    if str(exc) != "reopen failed":
                        fail(f"fork changed the open failure: {exc!r}")
                else:
                    fail("fork succeeded despite open_session failure")
            after_sessions = {path.name for path in (root / "sessions").iterdir() if path.is_dir()}
            if (a.read_bytes(), b.read_bytes()) != before_files or after_sessions != before_sessions:
                fail("failed fork did not roll back restored files and destination session")
        finally:
            host.close()


@check("host_sessions.fork_prefix_current_parent")
def check_fork_prefix_current_parent() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client = _host(root, [
            ModelResponse(Message(Role.ASSISTANT, "original answer")),
            ModelResponse(Message(Role.ASSISTANT, "fork answer")),
        ])
        try:
            source_id = client.send_prompt("first")["run_id"]
            _wait_idle(client)
            source_path = root / "sessions" / source_id / "run.jsonl"
            source_bytes = source_path.read_bytes()
            meta_path = source_path.parent / "meta.json"
            source_meta = meta_path.read_bytes()
            source = SessionStore.open(root / "sessions", source_id)
            try:
                source_run = load_run(source)
                first_id = next(
                    record_id
                    for message, record_id in zip(source_run.messages, source_run.record_ids, strict=True)
                    if message.role == Role.USER
                )
            finally:
                source.close()
            stream_connection, stream_response = _subscribed_stream(host)
            try:
                status, reply = _fork(host, source_id, first_id)
                if status != 200 or reply["run_id"] == source_id or reply["replayed"] != 2:
                    fail(f"fork did not create and reopen the prefix: {status}, {reply!r}")
                fork_id = reply["run_id"]
                _, frame = _await_sse(
                    stream_connection, stream_response,
                    lambda frame: frame[0] == "event"
                    and frame[1].get("type") == "HistoryMessage"
                    and frame[1].get("role") == "user",
                    what="fork history",
                )
                if (
                    frame.get("record_id") is None
                    or frame["record_id"] == first_id
                    or "/" in frame["record_id"]
                    or str(root) in json.dumps(frame)
                    or set(frame) != {"type", "role", "text", "tool_calls", "turn_id", "attachments", "record_id", "session_id"}
                ):
                    fail(f"fork history record id was absent or disclosed a path: {frame!r}")
                fork_store = SessionStore.open(root / "sessions", fork_id)
                try:
                    forked = load_run(fork_store)
                finally:
                    fork_store.close()
                if [message.text for message in forked.messages if message.role != Role.SYSTEM] != ["first"]:
                    fail(f"fork copied messages beyond its boundary: {forked.messages!r}")
                listed = {item["run_id"]: item for item in client.list_sessions()}
                if listed[fork_id]["parent_session_id"] != source_id:
                    fail(f"fork parent session was absent from listing: {listed[fork_id]!r}")
                next_run = client.send_prompt("different path")
                _wait_idle(client)
                fork_store = SessionStore.open(root / "sessions", fork_id)
                try:
                    texts = [
                        message.text for message in load_run(fork_store).messages
                        if message.role != Role.SYSTEM
                    ]
                finally:
                    fork_store.close()
                if next_run["run_id"] == fork_id or texts != ["first", "different path", "fork answer"]:
                    fail(f"next prompt did not continue the fork: {texts!r}")
                listed = {item["run_id"]: item for item in client.list_sessions()}
                if listed[fork_id]["parent_session_id"] != source_id:
                    fail("fork parent session disappeared after a later prompt")
                if source_path.read_bytes() != source_bytes or meta_path.read_bytes() != source_meta:
                    fail("fork changed the original session")
            finally:
                stream_connection.close()
        finally:
            host.close()


@check("host_sessions.fork_last_message_inherits_context")
def check_fork_last_message_inherits_context() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        instructions = root / ".symphonai" / "INSTRUCTIONS.md"
        instructions.parent.mkdir()
        instructions.write_text("original fork convention", encoding="utf-8")
        with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": str(root / "missing-home")}):
            host, client = _host(root, [
                ModelResponse(Message(Role.ASSISTANT, "first answer")),
                ModelResponse(Message(Role.ASSISTANT, "second answer")),
            ])
            try:
                source_id = client.send_prompt("first")["run_id"]
                _wait_idle(client)
                source = SessionStore.open(root / "sessions", source_id)
                try:
                    loaded = load_run(source)
                    last_id = loaded.record_ids[-1]
                    source_meta = source.read_meta()
                    source_meta["provider_choice"] = {"name": "fake", "model": "inherited"}
                    source.write_meta(source_meta)
                finally:
                    source.close()
                host.run._provider_factory = lambda name, model, base_url: host.run._provider
                instructions.write_text("changed fork convention", encoding="utf-8")
                status, reply = _fork(host, source_id, last_id)
                if status != 200 or reply["replayed"] != len(loaded.messages):
                    fail(f"last-message fork was not equivalent to source history: {status}, {reply!r}")
                fork_id = reply["run_id"]
                fork_store = SessionStore.open(root / "sessions", fork_id)
                try:
                    forked = load_run(fork_store)
                    if forked.messages != loaded.messages or fork_store.read_meta().get("provider_choice") != source_meta["provider_choice"]:
                        fail("last-message fork lost source messages or provider choice")
                finally:
                    fork_store.close()
                with mock.patch.object(host.run._provider, "create_response", wraps=host.run._provider.create_response) as response_spy:
                    client.send_prompt("continue fork")
                    _wait_idle(client)
                sent = response_spy.call_args.args[0].messages
                system = [message.text for message in sent if message.role == Role.SYSTEM]
                source_system = [
                    message.text for message in loaded.messages
                    if message.role == Role.SYSTEM
                ]
                if (
                    len(system) != 2
                    or "original fork convention" not in system[0]
                    or "changed fork convention" in system[0]
                    or not system[1].startswith("Environment when this conversation started")
                    or system != source_system
                ):
                    fail(f"fork duplicated or reloaded instructions: {system!r}")
                if [message.text for message in sent if message.role != Role.SYSTEM] != [
                    "first", "first answer", "continue fork",
                ]:
                    fail(f"last-message fork did not continue source history: {sent!r}")
            finally:
                host.close()


@check("host_sessions.fork_invalid_and_active")
def check_fork_invalid_and_active() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client, source_id = _finished_session(root)
        try:
            before = {path.name for path in (root / "sessions").iterdir()}
            status, _ = _fork(host, source_id, "rec_missing")
            after = {path.name for path in (root / "sessions").iterdir()}
            if status != 404 or after != before:
                fail("unknown record fork succeeded or left a destination directory")
            connection, response = _request(host, "POST", "/session/new", body={}, headers=_headers(host))
            response.read()
            connection.close()
            waiting = _WaitingProvider()
            host.run.select_provider(waiting)
            active_id = client.send_prompt("busy")["run_id"]
            source = SessionStore.open(root / "sessions", source_id)
            try:
                record_id = load_run(source).record_ids[0]
            finally:
                source.close()
            status, body = _fork(host, source_id, record_id)
            if status != 200 or body.get("run_id") in (None, source_id):
                fail(f"fork refused while another conversation was active: {status}, {body!r}")
            if active_id not in host.run._active_by_session:
                fail("forking another conversation stopped the active run")
            waiting.release.set()
            _wait_idle(client)
        finally:
            host.close()


@check("host_sessions.fork_reopen_failure_leaves_nothing")
def check_fork_reopen_failure_leaves_nothing() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, _, source_id = _finished_session(root)
        try:
            sessions_root = root / "sessions"
            source = SessionStore.open(sessions_root, source_id)
            try:
                record_id = load_run(source).record_ids[-1]
                meta = source.read_meta()
                meta["provider_choice"] = {"name": "unavailable"}
                source.write_meta(meta)
            finally:
                source.close()
            transcript_path = sessions_root / source_id / "run.jsonl"
            meta_path = sessions_root / source_id / "meta.json"
            transcript_before = transcript_path.read_bytes()
            meta_before = meta_path.read_bytes()
            directories_before = {path.name for path in sessions_root.iterdir() if path.is_dir()}
            host.run._provider_factory = lambda name, model, base_url: None
            try:
                host.run.fork_session(source_id, record_id)
            except ProviderSelectionError as exc:
                if str(exc) != "session provider is unavailable":
                    fail(f"fork changed the provider failure: {exc!r}")
            else:
                fail("fork accepted an unavailable recorded provider")
            directories_after = {path.name for path in sessions_root.iterdir() if path.is_dir()}
            if directories_after != directories_before:
                fail(f"failed fork left a session directory: {directories_after!r}")
            status, body = _fork(host, source_id, record_id)
            directories_after = {path.name for path in sessions_root.iterdir() if path.is_dir()}
            if status != 400 or body.get("error") != "session provider is unavailable" or directories_after != directories_before:
                fail(f"failed fork route changed its status or left a session: {status}, {body!r}, {directories_after!r}")
            if transcript_path.read_bytes() != transcript_before or meta_path.read_bytes() != meta_before:
                fail("failed fork changed the source session")
            if host.run._conversation is None or host.run._conversation[1].run_id != source_id:
                fail("failed fork replaced the current source conversation")
        finally:
            host.close()


@check("host_sessions.reopen_failure_keeps_current_store")
def check_reopen_failure_keeps_current_store() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        host, client = _host(root, [
            ModelResponse(Message(Role.ASSISTANT, "first answer")),
            ModelResponse(Message(Role.ASSISTANT, "second answer")),
        ])
        try:
            run_id = client.send_prompt("first")["run_id"]
            _wait_idle(client)
            current = host.run._conversation
            provider = (host.run._provider, host.run._model, host.run._provider_choice)
            rejected = []
            leader_error = RuntimeError("leader construction failed")

            def reject_leader(store):  # noqa: ANN001
                rejected.append(store)
                raise leader_error

            leader = current[0]
            with mock.patch.object(host.run, "_new_leader", side_effect=reject_leader):
                host.run.open_session(run_id)
            if host.run._conversation is not current or current[1]._closed or rejected:
                fail("opening an already-open conversation rebuilt its leader or store")
            if (host.run._provider, host.run._model, host.run._provider_choice) != provider:
                fail("reopening an open conversation changed its provider")

            client.send_prompt("second")
            _wait_idle(client)
            records, _ = read_records(root / "sessions" / run_id / "run.jsonl")
            if sum(record["type"] == "run_started" for record in records) != 2:
                fail("prompt after failed reopen reported success without persisting its run")
            if sum(record["type"] == "run_finished" for record in records) != 2:
                fail("prompt after failed reopen did not persist a normal turn")

            host.run.open_session(run_id)
            if host.run._conversation is not current or host.run._conversation[0] is not leader:
                fail("opening an already-open conversation did not reuse its leader")
        finally:
            host.close()


@check("host_sessions.concurrent_run_limit_and_release")
def check_concurrent_run_limit_and_release() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provider = _WaitingProvider()
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        client = HostClient(HostAddress(host.port, host.token))
        try:
            run_ids = []
            for index in range(4):
                if index:
                    _new_session(host)
                run_ids.append(client.send_prompt(f"run {index}")["run_id"])
            _new_session(host)
            connection, response = _request(
                host, "POST", "/prompt", body={"prompt": "fifth"}, headers=_headers(host)
            )
            try:
                refusal = json.loads(response.read())
                if response.status != 409 or refusal.get("error") != "4 conversations are already running":
                    fail(f"fifth concurrent run was not refused: {response.status}, {refusal!r}")
            finally:
                connection.close()
            client.open_session(run_ids[0])
            client.stop()
            deadline = time.monotonic() + 3
            while run_ids[0] in host.run._active_by_session and time.monotonic() < deadline:
                time.sleep(0.01)
            if run_ids[0] in host.run._active_by_session:
                fail("stopping a conversation did not release a run slot")
            _new_session(host)
            reply = client.send_prompt("replacement")
            if not reply.get("run_id") or reply["run_id"] == run_ids[0]:
                fail("a new run could not start after a slot was freed")
        finally:
            provider.release.set()
            host.close()


@check("host_sessions.session_policy_isolation")
def check_session_policy_isolation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provider = _WaitingProvider()
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        client = HostClient(HostAddress(host.port, host.token))
        try:
            first = client.send_prompt("first")["run_id"]
            policy_a = host.run._open_conversations[first][0]._config.leader_policy
            _new_session(host)
            second = client.send_prompt("second")["run_id"]
            policy_b = host.run._open_conversations[second][0]._config.leader_policy
            host.run.select_mode("plan")
            if policy_a.mode != "ask" or policy_b.mode != "plan":
                fail(f"mode change crossed conversation policies: A={policy_a.mode}, B={policy_b.mode}")
        finally:
            provider.release.set()
            host.close()


@check("host_sessions.stop_only_current_conversation")
def check_stop_only_current_conversation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provider = _WaitingProvider()
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        client = HostClient(HostAddress(host.port, host.token))
        try:
            first = client.send_prompt("first")["run_id"]
            _new_session(host)
            second = client.send_prompt("second")["run_id"]
            client.open_session(first)
            client.stop()
            deadline = time.monotonic() + 3
            while first in host.run._active_by_session and time.monotonic() < deadline:
                time.sleep(0.01)
            if first in host.run._active_by_session or second not in host.run._active_by_session:
                fail(f"stop did not affect only the current conversation: {host.run._active_by_session!r}")
        finally:
            provider.release.set()
            host.close()


@check("host_sessions.active_session_activity")
def check_active_session_activity() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provider = _WaitingProvider()
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        client = HostClient(HostAddress(host.port, host.token))
        try:
            run_id = client.send_prompt("working")["run_id"]
            _new_session(host)
            connection, response = _request(host, "GET", "/sessions", headers=_headers(host))
            try:
                sessions = json.loads(response.read())
                item = next(entry for entry in sessions if entry["run_id"] == run_id)
                if response.status != 200 or item.get("activity") != "working":
                    fail(f"active session was not reported as working: {sessions!r}")
            finally:
                connection.close()
        finally:
            provider.release.set()
            host.close()


@check("host_sessions.idle_session_activity")
def check_idle_session_activity() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _listing_fixture(root)
        sessions = list_sessions(root / "sessions")
        if any(item.get("activity") != "idle" for item in sessions):
            fail(f"inactive sessions did not report idle: {sessions!r}")


@check("host_sessions.goal_check_continues_in_background")
def check_goal_check_continues_in_background() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        release = root / "release-check"
        checker = root / "wait_for_release.py"
        checker.write_text(
            "from pathlib import Path\n"
            f"release = Path({str(release)!r})\n"
            "while not release.exists():\n"
            "    pass\n",
            encoding="utf-8",
        )
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "goal round done")),
            ModelResponse(Message(Role.ASSISTANT, "second conversation done")),
        ])
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        client = HostClient(HostAddress(host.port, host.token))
        try:
            goal_run_id = host.run.start_goal(
                "finish the goal", (sys.executable, str(checker)), max_rounds=2,
            )
            deadline = time.monotonic() + 3
            while goal_run_id not in host.run._goal_checks_by_session and time.monotonic() < deadline:
                time.sleep(0.01)
            if goal_run_id not in host.run._goal_checks_by_session:
                fail("goal check did not remain active after its first round")
            _new_session(host)
            second_id = client.send_prompt("second conversation")["run_id"]
            _wait_idle(client)
            release.touch()
            deadline = time.monotonic() + 3
            while host.run._goals_by_session[goal_run_id].phase != "complete" and time.monotonic() < deadline:
                time.sleep(0.01)
            if host.run._goals_by_session[goal_run_id].phase != "complete":
                fail("background goal check did not complete while another session was current")
            if host.run._conversation[1].run_id != second_id:
                fail("goal completion switched away from the current conversation")
        finally:
            release.touch()
            host.close()
