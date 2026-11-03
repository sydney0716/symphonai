"""Core host flows over HTTP/SSE and the app's public module boundary."""

from __future__ import annotations

import http.client
import io
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

from symphonai_api.extensions import load_extensions
from symphonai_api.models import Message, ModelResponse, Role, ToolCall, Usage
from symphonai_api.providers.base import ProviderError
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_host.headless import main as headless_main
from symphonai_host.loops import LoopStore
from symphonai_host.server import HostServer
from symphonai_host.run import HostRun
from symphonai_host.projects import ProjectRuntime
from symphonai_host.broker import EventBroker
from symphonai_host.providers import provider as build_provider
from scripts.checks.harness import check
from scripts.checks.workspace import handoff_spec, workspace


def reply(text="core reply", *calls):
    return ModelResponse(Message(Role.ASSISTANT, text, tool_calls=list(calls)), usage=Usage(5, 2))


def tool(name, **arguments):
    return ToolCall(f"core-{os.urandom(5).hex()}", name, arguments)


def request(host, path, body=None, *, authenticated=True, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", host.port, timeout=10)
    supplied = {"Authorization": f"Bearer {host.token}"} if authenticated else {}
    supplied.update(headers or {})
    if body is not None:
        supplied["Content-Type"] = "application/json"
    connection.request("GET" if body is None else "POST", path,
                       body=None if body is None else json.dumps(body), headers=supplied)
    response = connection.getresponse()
    raw = response.read()
    status, response_headers = response.status, dict(response.getheaders())
    connection.close()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        value = raw
    return status, value, response_headers


def get(host, path):
    status, value, _ = request(host, path)
    assert status == 200, (path, status, value)
    return value


def post(host, path, body):
    status, value, _ = request(host, path, body)
    assert status == 200, (path, status, value)
    return value


def wait_for(function, predicate, seconds=10):
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        last = function()
        if predicate(last):
            return last
        threading.Event().wait(0.01)
    raise AssertionError(f"condition did not become true; last value: {last!r}")


@contextmanager
def running_host(ws, provider=None, *, mode="allow", sessions=None, extensions=None):
    provider = provider or FakeModelProvider([reply()])
    broker = EventBroker()
    home = ws.outside / "home"
    run = HostRun(provider, replace(
        ws.policy, shell_enabled=True, shell_allowlist=[()],
    ), broker, max_turns=20, provider_factory=build_provider, sessions_root=sessions or ws.outside / "sessions",
        memory_root=ws.outside / "memory", memory_extraction_enabled=False,
        project_grants_root=home / "permissions", extensions=extensions,
        publish_approval=lambda approval: host._publish_approval(approval))
    host = HostServer(provider, ProjectRuntime(ws.policy.repo_root, run), broker=broker,
                      home=home, keepalive_seconds=0.05)
    host.start()
    try:
        post(host, "/mode", {"mode": mode})
        yield host
    finally:
        host.close()


class Stream:
    def __init__(self, host):
        self.connection = http.client.HTTPConnection("127.0.0.1", host.port, timeout=15)
        self.connection.request("GET", "/events", headers={"Authorization": f"Bearer {host.token}"})
        self.response = self.connection.getresponse()
        assert self.response.status == 200
        self.frames = []
        self.queue = queue.Queue()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        try:
            while not self.done.is_set():
                line = self.response.readline()
                if not line:
                    return
                if line.startswith(b"data: "):
                    frame = json.loads(line[6:])
                    self.frames.append(frame)
                    self.queue.put(frame)
        except (OSError, ValueError):
            return

    def until(self, predicate, seconds=10):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                frame = self.queue.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if predicate(frame):
                return frame
        raise AssertionError(f"event not received; got {[f.get('payload', {}).get('type', f.get('kind')) for f in self.frames]}")

    def close(self):
        self.done.set()
        self.connection.close()


def prompt(host, text="core prompt"):
    accepted = post(host, "/prompt", {"prompt": text})
    assert accepted["accepted"] and accepted["run_id"]
    wait_for(lambda: get(host, "/health"), lambda health: health["state"] == "idle")
    return get(host, "/conversation")["conversation"]["session_id"]


@check("host.terminal_input_streams_ordered_output")
def terminal_input_streams_ordered_output():
    with workspace() as ws, mock.patch.dict(os.environ, {"SHELL": "/bin/sh"}), running_host(ws) as host:
        stream = Stream(host)
        identifier = None
        output = []
        try:
            identifier = post(host, "/terminal/start", {})["id"]
            post(host, "/terminal/input", {"id": identifier, "data": "echo hello-probe\r"})

            def echoed(frame):
                event = frame.get("payload", {})
                if event.get("type") != "TerminalOutput" or event.get("terminal_id") != identifier:
                    return False
                assert event["agent_id"] == event["run_id"] == ""
                output.append(event["data"])
                text = "".join(output)
                return "echo hello-probe" in text and text.count("hello-probe\r\n") >= 2

            stream.until(echoed)
            text = "".join(output)
            assert text.index("echo hello-probe") < text.rindex("hello-probe\r\n"), text
        finally:
            if identifier is not None:
                post(host, "/terminal/stop", {"id": identifier})
            stream.close()


@check("host.routes_require_the_token")
def routes_require_the_token():
    gets = ("/health", "/events", "/conversation", "/git/status", "/git/diff", "/git/pull-request",
            "/jobs", "/changes", "/memory", "/roadmap", "/models", "/tree", "/source", "/files",
            "/agent", "/session/agent", "/survey", "/project", "/projects", "/directories",
            "/settings", "/approvals", "/permissions", "/sessions", "/session/export", "/loops",
            "/loop", "/loop/card", "/history", "/app/", "/app/src/app.js")
    posts = ("/project/open", "/git/commit", "/git/pull-request", "/jobs/start", "/jobs/stop", "/prompt",
             "/stop", "/approval", "/permissions/remove", "/session/open", "/session/fork", "/session/new",
             "/session/rename", "/session/delete", "/session/archive", "/provider", "/mode", "/compact",
             "/agent", "/steer", "/command", "/changes/revert", "/worktree/apply", "/worktree/discard",
             "/goal", "/goal/state", "/loop/start", "/loop/plan", "/loop/cancel", "/loop/commit",
             "/loop/resume",
             "/memory/update", "/memory/delete", "/memory/restore", "/credentials", "/extensions/reload", "/config/set")
    with workspace() as ws, running_host(ws) as host:
        exposed = {}
        for method, paths in (("GET", gets), ("POST", posts)):
            for path in paths:
                status, _, _ = request(host, path, {} if method == "POST" else None, authenticated=False)
                if status != 401:
                    exposed[f"{method} {path}"] = status
        assert not exposed, f"routes accessible without a token: {exposed}"
        for path in ("/roadmap/add", "/roadmap/remove", "/roadmap/move", "/roadmap/start"):
            assert request(host, path, {})[0] == 404
        from symphonai_host.roadmap_tools import roadmap_tools

        tools = roadmap_tools(host.run)
        added = tools["add_roadmap_goal"].execute(tool("add_roadmap_goal", goal="Fixture roadmap goal"), ws.policy)
        assert added.ok
        goal_id = json.loads(added.content)["goals"][0]["id"]
        started = tools["start_roadmap_goal"].execute(tool("start_roadmap_goal", goal_id=goal_id), ws.policy)
        assert started.ok
        loop_id = json.loads(started.content)["loop_id"]
        assert get(host, "/roadmap")["goals"][0]["loop_id"] == loop_id
        again = tools["start_roadmap_goal"].execute(tool("start_roadmap_goal", goal_id=goal_id), ws.policy)
        assert not again.ok and "already has" in again.error
        removed = tools["remove_roadmap_goal"].execute(tool("remove_roadmap_goal", goal_id=goal_id), ws.policy)
        assert removed.ok and not json.loads(removed.content)["goals"]
        assert (ws.root / ".symphonai" / "loops" / loop_id).is_dir()
        assert not tools["remove_roadmap_goal"].execute(tool("remove_roadmap_goal", goal_id=goal_id), ws.policy).ok
        config = ws.outside / "home" / ".symphonai" / "config.toml"
        config.parent.mkdir(parents=True, exist_ok=True)
        original = '# Keep header\n[memory]\nextract = true  # Keep comment\n'
        config.write_text(original)
        post(host, "/config/set", {"key": "memory.extract", "value": False, "scope": "user"})
        assert config.read_text() == original.replace("true", "false")
        assert {row["key"]: row["value"] for row in get(host, "/settings")["settings"]["config"]}["memory.extract"] is False
        saved = config.read_bytes()
        for key, value, scope in (("roles.leader.model", "fixture", "project"),
                                  ("memory.extract", "invalid", "user"),
                                  ("unknown.value", 1, "user"), ("memory.model", [], "user")):
            status, reply, _ = request(host, "/config/set", {"key": key, "value": value, "scope": scope})
            assert status == 400 and reply["error"]
            assert config.read_bytes() == saved


@check("host.app_page_and_assets_require_the_page_token_or_cookie")
def app_page_and_assets_require_the_page_token_or_cookie():
    with workspace() as ws, running_host(ws) as host:
        assert request(host, "/app/", authenticated=False)[0] == 401
        status, page, headers = request(host, f"/app/?token={host.token}", authenticated=False)
        assert status == 200 and b"<html" in page.lower()
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        status, source, _ = request(host, "/app/src/app.js", authenticated=False, headers={"Cookie": cookie})
        assert status == 200 and source
        assert request(host, "/app/src/app.js", authenticated=False)[0] == 401


@check("host.prompt_streams_ordered_run_and_tool_events")
def prompt_streams_ordered_run_and_tool_events():
    provider = FakeModelProvider([reply("", tool("read_file", path="existing.txt")), reply("streamed answer")])
    with workspace() as ws, running_host(ws, provider) as host:
        stream = Stream(host)
        try:
            accepted = post(host, "/prompt", {"prompt": "read the fixture"})
            terminal = stream.until(lambda f: f.get("payload", {}).get("type") == "RunFinished")
            events = [f["payload"] for f in stream.frames if f["kind"] == "event"]
            types = [e["type"] for e in events]
            assert accepted["run_id"]
            assert types.index("RunStarted") < types.index("ToolCallStarted") < types.index("ToolCallFinished") < types.index("RunFinished")
            started = next(e for e in events if e["type"] == "RunStarted")
            assert started["run_id"] == terminal["payload"]["run_id"], events
            wait_for(lambda: get(host, "/health"), lambda value: value["state"] == "idle")
            session = get(host, "/conversation")["conversation"]["session_id"]
            assert "streamed answer" in get(host, f"/session/export?id={session}")["markdown"]
        finally:
            stream.close()


def approval_flow(allowed):
    provider = FakeModelProvider([reply("", tool("run_shell", command="printf allowed > approved.txt")), reply()])
    with workspace() as ws, running_host(ws, provider, mode="ask") as host:
        stream = Stream(host)
        try:
            post(host, "/prompt", {"prompt": "run the command"})
            frame = stream.until(lambda f: f["kind"] == "approval_requested")
            approval = frame["payload"]
            assert approval["approval_id"] and approval["operation"] == "run_shell"
            assert post(host, "/approval", {"approval_id": approval["approval_id"], "allowed": allowed})["resolved"]
            stream.until(lambda f: f.get("payload", {}).get("type") == "RunFinished")
            assert (ws.root / "approved.txt").exists() is allowed
        finally:
            stream.close()


@check("host.approval_answer_allows_the_call")
def approval_answer_allows_the_call():
    approval_flow(True)


@check("host.approval_rejection_prevents_the_call")
def approval_rejection_prevents_the_call():
    approval_flow(False)


@check("host.session_lifecycle_preserves_the_conversation")
def session_lifecycle_preserves_the_conversation():
    with workspace() as ws, running_host(ws) as host:
        session = prompt(host, "remember this conversation")
        post(host, "/session/rename", {"run_id": session, "title": "Core conversation"})
        post(host, "/session/new", {})
        assert get(host, "/conversation")["conversation"] is None
        post(host, "/session/open", {"run_id": session})
        exported = get(host, f"/session/export?id={session}")["markdown"]
        assert "remember this conversation" in exported and "core reply" in exported
        post(host, "/session/archive", {"run_id": session, "archived": True})
        assert any(item["run_id"] == session and item["title"] == "Core conversation" for item in get(host, "/sessions?archived=1"))
        post(host, "/session/archive", {"run_id": session, "archived": False})
        assert any(item["run_id"] == session for item in get(host, "/sessions"))
        post(host, "/session/open", {"run_id": session})
        restored = get(host, f"/session/export?id={session}")["markdown"]
        assert "remember this conversation" in restored and "core reply" in restored
        post(host, "/session/new", {})
        assert post(host, "/session/delete", {"run_id": session})["deleted"]
        assert not any(item["run_id"] == session for item in get(host, "/sessions"))


@check("host.session_from_another_project_is_refused")
def session_from_another_project_is_refused():
    with workspace() as first, workspace() as other:
        shared = first.outside / "sessions"
        with running_host(other, sessions=shared) as host:
            session = prompt(host)
        with running_host(first, sessions=shared) as host:
            status, value, _ = request(host, "/session/open", {"run_id": session})
            assert status == 409 and "error" in value
            assert get(host, "/conversation")["conversation"] is None
            from symphonai_host.session_search import session_tools

            past = []
            for text in ("retry rule changed", "notes decision", "RETRY discussion"):
                past.append(prompt(host, text))
                post(host, "/session/new", {})
            current = prompt(host, "retry current conversation")
            tools = session_tools(shared, first.root, current)
            found = tools["search_sessions"].execute(tool("search_sessions", query="retry"), first.policy)
            assert found.ok
            matches = json.loads(found.content)["sessions"]
            assert {item["session_id"] for item in matches} == {past[0], past[2]}
            assert all(item["snippets"] for item in matches)
            page = tools["read_session"].execute(tool("read_session", session_id=past[0], limit=1), first.policy)
            assert page.ok and json.loads(page.content)["messages"][0]["text"] == "retry rule changed"
            assert json.loads(page.content)["next_offset"] == 1
            next_page = tools["read_session"].execute(
                tool("read_session", session_id=past[0], offset=1, limit=1), first.policy,
            )
            assert next_page.ok and json.loads(next_page.content)["messages"][0]["role"] == "assistant"
            for session_id in (session, "missing", "../outside"):
                assert not tools["read_session"].execute(tool("read_session", session_id=session_id), first.policy).ok


def git(root, *arguments):
    return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True).stdout


def init_git(ws):
    git(ws.root, "init", "--quiet")
    git(ws.root, "config", "user.email", "core@example.invalid")
    git(ws.root, "config", "user.name", "Core checks")
    git(ws.root, "add", "existing.txt")
    git(ws.root, "commit", "--quiet", "-m", "fixture")


def editing_provider():
    return FakeModelProvider([
        reply("", tool("dispatch_subagent", subagent_name="implementer", task=handoff_spec("create changed.txt", "test -f changed.txt"))),
        reply("", tool("write_file", path="changed.txt", content="agent change\n")),
        reply("implemented"), reply("complete"),
    ])


@check("host.changes_show_the_agent_edit_and_diff")
def changes_show_the_agent_edit_and_diff():
    with workspace() as ws:
        init_git(ws)
        with running_host(ws, editing_provider()) as host:
            prompt(host)
            changes = get(host, "/changes")
            assert any("changed.txt" in turn["paths"] for turn in changes["turns"])
            assert "agent change" in json.dumps(changes) and "changed.txt" in json.dumps(changes)


@check("host.commit_includes_only_selected_paths")
def commit_includes_only_selected_paths():
    with workspace() as ws:
        init_git(ws)
        (ws.root / "selected.txt").write_text("selected")
        (ws.root / "other.txt").write_text("leave this alone")
        with running_host(ws) as host:
            post(host, "/git/commit", {"message": "selected only", "paths": ["selected.txt"]})
            assert git(ws.root, "show", "--format=", "--name-only", "HEAD").decode().splitlines() == ["selected.txt"]
            assert b"other.txt" in git(ws.root, "ls-files", "--others", "--exclude-standard")


@check("host.revert_restores_the_turn_files")
def revert_restores_the_turn_files():
    with workspace() as ws:
        init_git(ws)
        with running_host(ws, editing_provider()) as host:
            prompt(host)
            (ws.root / "outside.txt").write_text("outside change")
            turn = next(t for t in get(host, "/changes")["turns"] if "changed.txt" in t["paths"])
            post(host, "/changes/revert", {"key": turn["key"]})
            assert not (ws.root / "changed.txt").exists()
            assert (ws.root / "outside.txt").read_text() == "outside change"


class LoopScript(FakeModelProvider):
    @property
    def name(self):
        return "openai"

    def __init__(self, root, *, cards=3, fail=False, hold=False):
        super().__init__()
        self.root, self.card_count, self.fail, self.hold = root, cards, fail, hold
        self.started = threading.Event()
        self.parallel = threading.Barrier(2) if cards == 3 else None
        self.log = []
        self.lock = threading.Lock()

    def create_response(self, request, *, cancel=None):
        stage = request.model
        if stage == "core-planner":
            if any(m.tool_result for m in request.messages):
                return reply("plan ready")
            loop = LoopStore(self.root).list()[0]
            folder = LoopStore(self.root).path(loop["id"], ".")
            cards = []
            calls = []
            for index, card in enumerate("abc"[:self.card_count]):
                lane, after = (1, ["a"]) if card == "b" else (index + 1 if card == "c" else 1, [])
                validation = f"test -f {card}.txt" + (" && test -f a.txt" if card == "b" else "")
                spec = (f"## Goal\ncore-card-{card}\n\n## Scope\n{card}.txt\n\n## Context\nTemporary core fixture\n\n"
                        f"## Contract\nCreate {card}.txt\n\n## Acceptance criteria\nFile exists\n\n## Validation\n```sh\n{validation}\n```\n\n## Report\nReport the result\n")
                calls.append(tool("write_file", path=str(folder / "specs" / f"{card}-core.md"), content=spec))
                cards.append({"id": card, "title": f"Card {card}", "spec": f"specs/{card}-core.md", "lane": lane, "after": after})
            calls.append(tool("write_file", path=str(folder / "plan.json"), content=json.dumps({"cards": cards})))
            return reply("", *calls)
        task = next(m.text for m in request.messages if m.role is Role.USER)
        card = re.search(r"core-card-([abc])", task)[1]
        if stage == "core-reviewer":
            with self.lock:
                self.log.append(("review", card))
            return reply("Verdict: pass")
        assert stage == "core-implementer", f"unexpected scripted stage: {stage}"
        if any(m.tool_result for m in request.messages):
            return reply("implemented")
        with self.lock:
            self.log.append(("implement", card))
        self.started.set()
        if self.hold:
            assert cancel is not None and cancel.wait(10)
            cancel.raise_if_cancelled()
        if self.fail:
            raise ProviderError("fixture implementation failure")
        if self.parallel and card in "ac":
            self.parallel.wait(timeout=5)
        return reply("", tool("write_file", path=f"{card}.txt", content=f"card {card}\n"))


@contextmanager
def loop_host(ws, **options):
    init_git(ws)
    home = ws.outside / "home"
    config = home / ".symphonai" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('[memory]\nextract = false\n' + "\n".join(
        f'[roles.{role}]\nprovider = "openai"\nmodel = "core-{role}"\n'
        for role in ("planner", "implementer", "reviewer")
    ))
    extensions = load_extensions(repo_root=ws.root, home=home)
    provider = LoopScript(ws.root, **options)
    with running_host(ws, provider, extensions=extensions) as host:
        yield host, provider


def loop_state(host, loop_id, state):
    return wait_for(lambda: get(host, f"/loop?id={loop_id}")["loop"], lambda loop: loop["state"] == state, seconds=20)


def start_and_approve(host):
    loop = post(host, "/loop/start", {"goal": "build core fixture cards"})["loop"]
    loop_state(host, loop["id"], "awaiting_plan")
    return post(host, "/loop/plan", {"loop_id": loop["id"], "action": "approve"})["loop"]["id"]


@check("host.spec_loop_plans_runs_lanes_reviews_validates_and_commits")
def spec_loop_plans_runs_lanes_reviews_validates_and_commits():
    with workspace() as ws, loop_host(ws) as (host, provider):
        loop_id = start_and_approve(host)
        loop = loop_state(host, loop_id, "awaiting_commit")
        assert all(card["state"] == "done" for card in loop["cards"])
        assert provider.log.index(("review", "a")) < provider.log.index(("implement", "b"))
        assert not provider.parallel.broken
        branch = loop["branch"]
        for card in "abc":
            assert git(ws.root, "show", f"{branch}:{card}.txt").decode() == f"card {card}\n"
        assert all(not (ws.root / f"{card}.txt").exists() for card in "abc")
        committed = post(host, "/loop/commit", {"loop_id": loop_id, "action": "approve"})
        assert committed["loop"]["state"] == "done" and committed["commits"]


@check("host.failed_loop_card_can_be_resumed")
def failed_loop_card_can_be_resumed():
    with workspace() as ws, loop_host(ws, cards=1, fail=True) as (host, provider):
        loop_id = start_and_approve(host)
        loop = loop_state(host, loop_id, "needs_attention")
        assert loop["cards"][0]["state"] != "done"
        provider.fail = False
        post(host, "/loop/resume", {"loop_id": loop_id})
        loop = loop_state(host, loop_id, "awaiting_commit")
        assert loop["cards"][0]["state"] == "done"


@check("host.cancelling_a_loop_removes_its_worktrees_and_branch")
def cancelling_a_loop_removes_its_worktrees_and_branch():
    with workspace() as ws, loop_host(ws, cards=1, hold=True) as (host, provider):
        loop_id = start_and_approve(host)
        assert provider.started.wait(5)
        branch = get(host, f"/loop?id={loop_id}")["loop"]["branch"]
        post(host, "/loop/cancel", {"loop_id": loop_id})
        loop_state(host, loop_id, "cancelled")
        wait_for(lambda: git(ws.root, "worktree", "list", "--porcelain").decode().count("worktree "), lambda count: count == 1)
        assert branch not in git(ws.root, "branch", "--format=%(refname:short)").decode().splitlines()


class HTTPFixture(io.BytesIO):
    status = 200
    headers = {}

    def __init__(self, response):
        super().__init__(("event: response.completed\ndata: " + json.dumps({"type": "response.completed", "response": response}) + "\n\n").encode())


def headless_case(kind):
    with workspace() as ws:
        home = ws.outside / "home"
        home.mkdir()
        config = home / "config.toml"
        config.write_text('[memory]\nextract = false\n')
        base_response = {"id": "core-response", "status": "completed", "model": "core-model",
                         "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "CLI result"}]}],
                         "usage": {"input_tokens": 2, "output_tokens": 3}}
        if kind == "stopped":
            base_response["output"] = [{"type": "function_call", "call_id": "call-core", "name": "list_files", "arguments": "{}"}]
        elif kind == "refused":
            base_response["output"][0]["content"] = [{"type": "refusal", "refusal": "fixture refusal"}]
        def respond(request, **options):
            if kind == "failed":
                raise HTTPError(request.full_url, 400, "fixture", {}, io.BytesIO(b'{"error":{"message":"fixture error"}}'))
            return HTTPFixture(base_response)
        output = io.StringIO()
        environment = {"SYMPHONAI_HOME": str(home), "SYMPHONAI_SESSIONS_DIR": str(ws.outside / "cli-sessions"), "OPENAI_API_KEY": "core-inert-key"}
        with mock.patch.dict(os.environ, environment), mock.patch("urllib.request.urlopen", side_effect=respond), redirect_stdout(output), redirect_stderr(io.StringIO()):
            code = headless_main(["run", "core task", "--repo-root", str(ws.root), "--provider", "openai", "--model", "core-model", "--mode", "allow", "--max-turns", "1", "--json"])
        result = json.loads(output.getvalue().splitlines()[-1])
        assert {"status", "stopped_reason", "error", "usage", "files_changed"} <= result.keys()
        assert isinstance(result["usage"], dict) and isinstance(result["files_changed"], list)
        expected = {"finished": (0, "finished"), "stopped": (3, "stopped"), "refused": (3, "stopped"), "failed": (1, "failed")}
        assert (code, result["status"]) == expected[kind], (code, result)
        if kind == "finished":
            assert result["result"] == "CLI result" and result["usage"]["total_tokens"] == 5
        if kind == "refused":
            assert result["stopped_reason"] == "refused" and result["error"]
        if kind == "failed":
            assert result["error"]


@check("host.headless_finished_run_exits_zero_with_json")
def headless_finished_run_exits_zero_with_json():
    headless_case("finished")


@check("host.headless_stopped_run_exits_three_with_json")
def headless_stopped_run_exits_three_with_json():
    headless_case("stopped")


@check("host.headless_refusal_exits_three_with_json")
def headless_refusal_exits_three_with_json():
    headless_case("refused")


@check("host.headless_failed_run_exits_nonzero_with_json")
def headless_failed_run_exits_nonzero_with_json():
    headless_case("failed")


@check("host.app_core")
def app_core():
    node = shutil.which("node")
    assert node is not None, "Node 18 or later is required"
    version = subprocess.run([node, "--version"], capture_output=True, text=True, check=True).stdout
    assert int(version.lstrip("v").split(".")[0]) >= 18
    result = subprocess.run([node, "--experimental-default-type=module", "--test", "symphonai_app/test/core.test.js"],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


class FailureFollowUpScript(LoopScript):
    def __init__(self, root):
        super().__init__(root)
        self.parallel = None
        self.failure_review = threading.Event()

    def create_response(self, request, *, cancel=None):
        task = next((m.text for m in request.messages if m.role is Role.USER), "")
        if request.model == "core-reviewer" and task.startswith("Review this failed card"):
            self.failure_review.set()
            if any(m.tool_result for m in request.messages):
                return reply(f"Verdict: follow-ups: {self.follow_path}")
            loop = LoopStore(self.root).list()[0]
            self.follow_path = f".symphonai/loops/{loop['id']}/specs/a-fix.md"
            spec = ("## Goal\ncore-card-a failure-followup\n\n## Scope\na.txt\n\n"
                    "## Contract\nCreate a.txt\n\n## Validation\n```sh\ntest -f a.txt\n```\n")
            return reply("", tool("write_file", path=self.follow_path, content=spec))
        if request.model == "core-implementer":
            card = re.search(r"core-card-([abc])", task)[1]
            if card == "a" and "failure-followup" not in task:
                self.log.append(("failed-validation", "a"))
                return reply("implementation stopped without creating the required file")
            if card == "c" and not any(m.tool_result for m in request.messages):
                self.log.append(("independent-started", "c"))
                assert self.failure_review.wait(10), "independent card did not survive failure review"
            if card == "a" and "failure-followup" in task:
                self.log.append(("replacement", "a"))
        return super().create_response(request, cancel=cancel)


@check("host.failed_card_follow_up_unblocks_dependents")
def failed_card_follow_up_unblocks_dependents():
    with workspace() as ws, loop_host(ws) as (host, provider):
        script = FailureFollowUpScript(ws.root)
        provider.create_response = script.create_response
        loop_id = start_and_approve(host)
        loop = loop_state(host, loop_id, "awaiting_commit")
        assert {card["id"]: card["state"] for card in loop["cards"]} == {
            "a": "failed", "aF": "done", "b": "done", "c": "done",
        }
        assert {card["id"] for card in loop["cards"]} == {"a", "b", "c", "aF"}
        assert ("independent-started", "c") in script.log
        assert script.log.index(("replacement", "a")) < script.log.index(("implement", "b"))
        assert loop.get("reason", "") == ""


class ItemLoopScript(LoopScript):
    def __init__(self, root):
        super().__init__(root)
        self.parallel = None
        self.items_ready = threading.Event()
        self.items_started = set()
        self.items_reviewed = set()

    def create_response(self, request, *, cancel=None):
        task = next((m.text for m in request.messages if m.role is Role.USER), "")
        if request.model == "core-planner":
            if any(m.tool_result for m in request.messages):
                return reply("plan ready")
            loop = LoopStore(self.root).list()[0]
            folder = LoopStore(self.root).path(loop["id"], ".")
            template = "## Goal\ncore-item-{item}\n\n## Scope\n{item}.txt\n\n## Validation\n```sh\ntest -f {item}.txt\n```\n"
            dependent = "## Goal\ncore-card-b\n\n## Scope\nb.txt\n\n## Validation\n```sh\ntest -f b.txt\n```\n"
            cards = [
                {"id": "a", "title": "Each target", "spec": "specs/a-items.md", "lane": 1, "after": [], "items": ["x", "y", "z"]},
                {"id": "b", "title": "Dependent", "spec": "specs/b-dependent.md", "lane": 1, "after": ["a"]},
            ]
            return reply("", tool("write_file", path=str(folder / "specs/a-items.md"), content=template),
                         tool("write_file", path=str(folder / "specs/b-dependent.md"), content=dependent),
                         tool("write_file", path=str(folder / "plan.json"), content=json.dumps({"cards": cards})))
        match = re.search(r"core-item-([xyz])", task)
        if match:
            item = match[1]
            if request.model == "core-reviewer":
                self.items_reviewed.add(item)
                return reply("Verdict: pass")
            if any(m.tool_result for m in request.messages):
                return reply("implemented")
            with self.lock:
                self.items_started.add(item)
                if len(self.items_started) == 3:
                    self.items_ready.set()
            assert self.items_ready.wait(10), "three item implementations were not concurrent"
            return reply("", tool("write_file", path=f"{item}.txt", content=item))
        if request.model == "core-implementer":
            assert self.items_reviewed == {"x", "y", "z"}, "dependent started before all item reviews"
        return super().create_response(request, cancel=cancel)


@check("host.loop_expands_items_and_waits_for_all")
def loop_expands_items_and_waits_for_all():
    with workspace() as ws, loop_host(ws) as (host, provider):
        script = ItemLoopScript(ws.root)
        provider.create_response = script.create_response
        loop_id = post(host, "/loop/start", {"goal": "Apply one change to three targets"})["loop"]["id"]
        planned = loop_state(host, loop_id, "awaiting_plan")
        assert {card["id"] for card in planned["cards"]} == {"a1", "a2", "a3", "b"}
        assert len({card["lane"] for card in planned["cards"] if card["id"].startswith("a")}) == 3
        dependent = next(card for card in planned["cards"] if card["id"] == "b")
        assert dependent["after"] == ["a1", "a2", "a3"]
        post(host, "/loop/plan", {"loop_id": loop_id, "action": "approve"})
        finished = loop_state(host, loop_id, "awaiting_commit")
        assert all(card["state"] == "done" for card in finished["cards"])
        assert script.items_started == {"x", "y", "z"}
