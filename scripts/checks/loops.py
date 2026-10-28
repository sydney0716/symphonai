"""Checks for durable spec loop storage."""

from __future__ import annotations

import json
import http.client
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from unittest import mock
import threading

from symphonai_api.models import Message, ModelResponse, Role, ToolCall, Usage
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import ProviderError
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.session import SessionStore
from symphonai_host.broker import EventBroker
from scripts.checks.harness import check, fail
from symphonai_host.loops import LoopStore
from symphonai_host.roadmap import RoadmapStore
from symphonai_host.projects import ProjectRuntime
from symphonai_host.run import HostRun, LoopChanged, WorktreeApplyConflict
from symphonai_host.server import HostServer
from symphonai_host.headless import _print_loop_result
from symphonai_api.worktree import create_worktree, remove_worktree

_TEST_LOOP_ID = "loop_000000000000"


class _RecordingProvider(FakeModelProvider):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__(responses)
        self.requests = []

    def create_response(self, request, *, cancel=None):
        self.requests.append(request)
        return super().create_response(request, cancel=cancel)


def _write(path: str, content: str, call_id: str) -> ModelResponse:
    if path == "plan.json" or path.startswith("specs/"):
        path = f".symphonai/loops/{_TEST_LOOP_ID}/{path}"
    return ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
        call_id, "write_file", {"path": path, "content": content},
    )]))


def _finish() -> ModelResponse:
    return ModelResponse(Message(Role.ASSISTANT, "Plan complete."))


def _run_loop(root: Path, provider: FakeModelProvider):
    root.mkdir(parents=True, exist_ok=True)
    broker = EventBroker()
    subscription = broker.subscribe()
    run = HostRun(
        provider, PermissionPolicy(root, allowed_write_scope=[root]), broker,
        sessions_root=root / "sessions",
    )
    original_hook = threading.excepthook
    threading.excepthook = lambda _args: None
    try:
        with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000000"):
            loop = run.start_loop("Add a small capability")
        store = LoopStore(root)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            loop = store.load(loop["id"])
            if loop["state"] != "planning":
                break
            time.sleep(0.01)
        if loop["state"] == "planning":
            fail("planner did not finish within 5 seconds")
        events = []
        target_change = "planned" if loop["state"] == "awaiting_plan" else "state"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            event = subscription.get(timeout=0.05)
            if event is None:
                continue
            if isinstance(event, LoopChanged):
                events.append(event)
                if event.change == target_change:
                    break
        return loop, events, run
    finally:
        run.close()
        threading.excepthook = original_hook


@check("loops.reused_pid_is_not_a_runner")
def reused_pid_is_not_a_runner() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        store = LoopStore(root)
        loop = store.create("Recover a loop with a reused pid")
        loop.update(
            state="running",
            runner_pid=os.getpid(),
            runner_lock=True,
            cards=[{"id": "card", "title": "Card", "state": "running", "session_id": None}],
            lanes=[{"id": 1, "cards": ["card"]}],
        )
        store.save(loop)
        run = HostRun(
            FakeModelProvider(), PermissionPolicy(root), EventBroker(),
            sessions_root=root / "sessions",
        )
        try:
            with mock.patch.object(HostRun, "_advance_loop", lambda _self, _loop_id: None):
                run.recover_loops()
            recovered = store.load(loop["id"])
            if recovered["cards"][0]["state"] != "planned":
                fail(f"a reused live pid kept the loop runner alive: {recovered!r}")
        finally:
            run.close()


def _start_host(root: Path, provider: FakeModelProvider) -> HostServer:
    root.mkdir(parents=True, exist_ok=True)
    host = HostServer(
        provider, PermissionPolicy(root, allowed_write_scope=[root]),
        sessions_root=root / "sessions",
    )
    host.start()
    return host


def _http(host: HostServer, method: str, path: str, body: dict | None = None):
    connection = http.client.HTTPConnection(*host.address, timeout=5)
    headers = {"Authorization": f"Bearer {host.token}"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body)
    connection.request(method, path, body=data, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, json.loads(raw) if raw else {}


def _wait_loop_state(host: HostServer, loop_id: str, state: str) -> dict:
    store = LoopStore(host._repo_root)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        loop = store.load(loop_id)
        if loop["state"] == state:
            return loop
        time.sleep(0.01)
    fail(f"loop did not reach {state}: {store.load(loop_id)!r}")


def _git_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for arguments in (
        ("init", "--quiet"),
        ("config", "user.email", "loops-check@example.test"),
        ("config", "user.name", "Loops Check"),
    ):
        result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=False)
        if result.returncode:
            fail(f"git {' '.join(arguments)} failed: {result.stderr.decode(errors='replace')}")
    (root / "a.py").write_text("value = 1\n", encoding="utf-8")
    for arguments in (("add", "a.py"), ("commit", "--quiet", "-m", "initial")):
        result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=False)
        if result.returncode:
            fail(f"git {' '.join(arguments)} failed: {result.stderr.decode(errors='replace')}")


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=False)
    if result.returncode:
        fail(f"git {' '.join(arguments)} failed: {result.stderr.decode(errors='replace')}")
    return result.stdout


@check("loops.create_layout")
def create_layout() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet", str(root)], check=True)
        store = LoopStore(root)
        loop = store.create("Add compound durations to parse")
        folder = store.path(loop["id"], ".")
        if (folder / ".gitignore").read_text(encoding="utf-8") != "*\n":
            fail("loop .gitignore did not contain exactly '*\\n'")
        if loop["slug"] != "add-compound-durations-to-parse":
            fail(f"goal slug was incorrect: {loop['slug']!r}")
        if not (folder / "loop.json").is_file() or not (folder / "specs").is_dir() or not (folder / "reports").is_dir():
            fail("loop layout was incomplete")
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, check=True,
        ).stdout
        if ".symphonai/loops/" in status:
            fail(f"loop files appeared in git status: {status!r}")


@check("loops.slug_is_never_empty")
def slug_is_never_empty() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "project"
        _git_repo(root)
        store = LoopStore(root)
        with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="3f9a12000000"):
            loop = store.create("修改")
        expected_slug = "loop-3f9a12"
        if loop["slug"] != expected_slug or re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", loop["slug"]) is None:
            fail(f"goal without ASCII words had an invalid slug: {loop['slug']!r}")
        loop.update(
            state="awaiting_plan",
            cards=[{"id": "a", "title": "A", "state": "planned", "after": []}],
        )
        store.save(loop)
        run = HostRun(
            FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]), EventBroker(),
            sessions_root=base / "sessions",
        )
        try:
            approved = run.approve_loop_plan(loop["id"])
            expected_branch = f"symphonai/{expected_slug}"
            if approved.get("branch") != expected_branch or approved.get("state") != "running":
                fail(f"approve did not create the fallback loop branch: {approved!r}")
            integration = base / "sessions" / "loops" / loop["id"] / "integration"
            if not integration.is_dir() or _git(integration, "branch", "--show-current").decode().strip() != expected_branch:
                fail("approve did not create the integration worktree on the fallback branch")
        finally:
            run.close()
    with tempfile.TemporaryDirectory() as temporary:
        with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000001"):
            loop = LoopStore(Path(temporary)).create("Fix the parser")
        if loop["slug"] != "fix-the-parser":
            fail(f"ASCII goal slug changed unexpectedly: {loop['slug']!r}")


@check("loops.usage_per_stage")
def usage_per_stage() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        _git_repo(root)
        loop_id = _TEST_LOOP_ID
        report_path = f".symphonai/loops/{loop_id}/reports/a-change-a-report.md"
        spec = (
            "# Change a\n\n## Goal\n\nChange a value.\n\n## Scope\n\n`new.py`.\n\n"
            "## Contract\n\nSet the value.\n\n## Validation\n\n```sh\ntrue\n```\n\n"
            f"## Report\n\n`{report_path}`\n"
        )

        def priced(response: ModelResponse, usage: Usage) -> ModelResponse:
            return ModelResponse(response.message, usage)

        provider = _RecordingProvider([
            priced(_write("specs/a-change.md", spec, "plan-spec"), Usage(10, 1, 2, 3)),
            priced(_write("plan.json", json.dumps({"cards": [{
                "id": "a", "title": "Change a", "spec": "specs/a-change.md",
                "lane": 1, "after": [],
            }]}), "plan"), Usage(20, 2, 3, 4)),
            priced(_finish(), Usage(30, 3, 4, 5)),
            priced(_write("specs/a-change.md", spec, "revised-plan-spec"), Usage(10, 1, 2, 3)),
            priced(_write("plan.json", json.dumps({"cards": [{
                "id": "a", "title": "Change a", "spec": "specs/a-change.md",
                "lane": 1, "after": [],
            }]}), "revised-plan"), Usage(20, 2, 3, 4)),
            priced(_finish(), Usage(30, 3, 4, 5)),
            priced(ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
                "new-file", "write_file", {"path": "new.py", "content": "value = 2\\n"},
            )])), Usage(40, 4, 5, 6)),
            priced(ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
                "report", "write_file", {"path": report_path, "content": "Passed.\\n"},
            )])), Usage(50, 5, 6, 7)),
            priced(ModelResponse(Message(Role.ASSISTANT, "Implemented.")), Usage(60, 6, 7, 8)),
            priced(ModelResponse(Message(Role.ASSISTANT, "Verdict: pass")), Usage(70, 7, 8, 9)),
        ])
        host = _start_host(root, provider)
        try:
            with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000000"):
                loop = host.run.start_loop("Change a")
            _wait_loop_state(host, loop["id"], "awaiting_plan")
            prior_planner_session = loop["planner_session_id"]
            loop = host.run.revise_loop_plan(loop["id"], "Keep the same small change")
            _wait_loop_state(host, loop["id"], "awaiting_plan")
            if loop.get("previous_planner_session_ids") != [prior_planner_session]:
                fail(f"revised plan did not retain its previous planner session: {loop!r}")
            host.run.approve_loop_plan(loop["id"])
            _wait_loop_state(host, loop["id"], "awaiting_commit")
            status, reply = _http(host, "GET", f"/loop?id={loop['id']}")
            if status != 200:
                fail(f"GET /loop failed to return usage: {status}, {reply!r}")
            expected = {
                "planning": {
                    "input_tokens": 120, "output_tokens": 12, "calls": 6,
                    "cache_read_tokens": 18, "cache_write_tokens": 24,
                    "cache_write_1h_tokens": 0,
                },
                "cards": {"a": {
                    "implement": {
                        "input_tokens": 150, "output_tokens": 15, "calls": 3,
                        "cache_read_tokens": 18, "cache_write_tokens": 21,
                        "cache_write_1h_tokens": 0,
                    },
                    "review": {
                        "input_tokens": 70, "output_tokens": 7, "calls": 1,
                        "cache_read_tokens": 8, "cache_write_tokens": 9,
                        "cache_write_1h_tokens": 0,
                    },
                }},
                "total": {
                    "input_tokens": 340, "output_tokens": 34, "calls": 10,
                    "cache_read_tokens": 44, "cache_write_tokens": 54,
                    "cache_write_1h_tokens": 0,
                },
            }
            if reply.get("loop", {}).get("usage") != expected:
                fail(f"GET /loop returned incorrect per-stage usage: {reply!r}")
            json_output = io.StringIO()
            with redirect_stdout(json_output):
                _print_loop_result(reply["loop"], True)
            if json.loads(json_output.getvalue()).get("usage") != expected:
                fail("headless --json loop result did not preserve per-stage usage")
            text_output = io.StringIO()
            with redirect_stdout(text_output):
                _print_loop_result(reply["loop"], False)
            if "Tokens: 340 in · 34 out · 10 calls" not in text_output.getvalue():
                fail(f"headless loop text omitted its token total: {text_output.getvalue()!r}")

            loop_store = LoopStore(root)
            loop = loop_store.load(loop["id"])
            previous_attempt = loop["cards"][0]["session_id"]
            loop["state"] = "needs_attention"
            loop["reason"] = "card a failed"
            loop["cards"][0]["state"] = "failed"
            loop_store.save(loop)
            host.run.start_spec_run = lambda _info: ("resume-run", "resume-session")
            resumed = host.run.resume_loop(loop["id"])
            if resumed["cards"][0].get("previous_session_ids") != [previous_attempt]:
                fail(f"resumed card did not retain its previous implementation session: {resumed!r}")
            extra_session = SessionStore(root / "sessions", "resume-session", repo_root=root)
            extra_session.writer_for("leader", is_root=True).append(
                "usage", run_id="resume-run", agent_id="leader", turn_id=None,
                data={
                    "model": "test-model", "agent_name": "leader", "input_tokens": 13,
                    "output_tokens": 2, "calls": 1, "cache_read_tokens": 3,
                    "cache_write_tokens": 4,
                },
            )
            extra_session.close()
            resumed["state"] = "awaiting_commit"
            resumed["cards"][0]["state"] = "done"
            loop_store.save(resumed)
            resumed_usage = host.run.loop_usage(resumed)
            if resumed_usage["cards"]["a"]["implement"] != {
                "input_tokens": 163, "output_tokens": 17, "calls": 4,
                "cache_read_tokens": 21, "cache_write_tokens": 25,
                "cache_write_1h_tokens": 0,
            }:
                fail(f"resumed implementation usage omitted an attempt: {resumed_usage!r}")
            summed = {
                key: resumed_usage["planning"][key] + resumed_usage["cards"]["a"]["implement"][key]
                     + resumed_usage["cards"]["a"]["review"][key]
                for key in ("input_tokens", "output_tokens", "calls", "cache_read_tokens", "cache_write_tokens", "cache_write_1h_tokens")
            }
            if resumed_usage["total"] != summed:
                fail(f"loop total did not equal the sum of its stages: {resumed_usage!r}")
        finally:
            host.close()


@check("loops.roadmap_goals_round_trip")
def roadmap_goals_round_trip() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        host = _start_host(root, FakeModelProvider([]))
        try:
            status, first = _http(host, "POST", "/roadmap/add", {"goal": "  First goal  "})
            if status != 200 or first["goals"][0]["goal"] != "First goal":
                fail(f"roadmap/add did not trim and append the goal: {status}, {first!r}")
            first_id = first["goals"][0]["id"]
            _, second = _http(host, "POST", "/roadmap/add", {"goal": "Second goal"})
            second_id = second["goals"][1]["id"]
            _, third = _http(host, "POST", "/roadmap/add", {"goal": "Third goal"})
            third_id = third["goals"][2]["id"]
            status, moved = _http(host, "POST", "/roadmap/move", {"id": third_id, "index": 0})
            if status != 200 or [goal["id"] for goal in moved["goals"]] != [third_id, first_id, second_id]:
                fail(f"roadmap/move returned the wrong order: {status}, {moved!r}")
            status, removed = _http(host, "POST", "/roadmap/remove", {"id": second_id})
            if status != 200 or [goal["id"] for goal in removed["goals"]] != [third_id, first_id]:
                fail(f"roadmap/remove returned the wrong goals: {status}, {removed!r}")
            _, appended = _http(host, "POST", "/roadmap/add", {"goal": "Fourth goal"})
            fourth_id = appended["goals"][-1]["id"]
            status, reloaded = _http(host, "GET", "/roadmap")
            expected_goals = [
                {"id": third_id, "goal": "Third goal", "loop_id": None, "done": False, "state": "not_started"},
                {"id": first_id, "goal": "First goal", "loop_id": None, "done": False, "state": "not_started"},
                {"id": fourth_id, "goal": "Fourth goal", "loop_id": None, "done": False, "state": "not_started"},
            ]
            if status != 200 or reloaded.get("goals") != expected_goals:
                fail(f"GET /roadmap did not preserve file order: {status}, {reloaded!r}")
            raw = json.loads(RoadmapStore(root).path.read_text(encoding="utf-8"))
            expected_records = [{key: item[key] for key in ("id", "goal", "loop_id", "done")} for item in expected_goals]
            if raw != {"schema_version": 1, "goals": expected_records}:
                fail(f"roadmap file has the wrong persisted shape or order: {raw!r}")

            loop = LoopStore(root).create("Third goal")
            store = LoopStore(root)
            for state in ("awaiting_plan", "running", "validating", "awaiting_commit"):
                store.set_state(loop, state)
            roadmap_store = RoadmapStore(root)
            roadmap = roadmap_store.load()
            roadmap_store.find(roadmap, third_id)["loop_id"] = loop["id"]
            roadmap_store.save(roadmap)
            host.run.commit_loop(loop["id"], "approve")
            status, committed = _http(host, "GET", "/roadmap")
            goal = next(item for item in committed["goals"] if item["id"] == third_id)
            if status != 200 or not goal["done"]:
                fail(f"approving the loop did not mark its roadmap goal done: {status}, {goal!r}")
            shutil.rmtree(store._folder(loop["id"]))
            status, missing_loop = _http(host, "GET", "/roadmap")
            goal = next(item for item in missing_loop["goals"] if item["id"] == third_id)
            if status != 200 or goal["state"] != "done":
                fail(f"a done goal lost its state after its loop folder was removed: {status}, {goal!r}")
        finally:
            host.close()


@check("loops.roadmap_start_links_one_loop")
def roadmap_start_links_one_loop() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        host = _start_host(root, FakeModelProvider([]))
        try:
            status, added = _http(host, "POST", "/roadmap/add", {"goal": "Concurrent start"})
            if status != 200:
                fail(f"could not add roadmap goal: {status}, {added!r}")
            goal_id = added["goals"][0]["id"]
            def start_planner(loop: dict, _change: str) -> dict:
                loop["planner_session_id"] = "session_fake"
                return loop

            host.run._start_loop_planner = start_planner
            barrier = threading.Barrier(2)
            results = []
            failures = []

            def start() -> None:
                try:
                    barrier.wait(timeout=5)
                    results.append(_http(host, "POST", "/roadmap/start", {"id": goal_id}))
                except Exception as exc:
                    failures.append(exc)

            threads = [threading.Thread(target=start) for _ in range(2)]
            with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000000"):
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)
            if failures or any(thread.is_alive() for thread in threads):
                fail(f"concurrent roadmap starts did not finish: {failures!r}")
            statuses = sorted(status for status, _ in results)
            loop_folders = list((root / ".symphonai" / "loops").glob("loop_*"))
            if statuses != [200, 409] or len(loop_folders) != 1:
                fail(f"concurrent start created multiple loops or returned wrong statuses: {results!r}, {loop_folders!r}")
            status, reply = _http(host, "GET", "/roadmap")
            if status != 200 or reply["goals"][0]["loop_id"] != "loop_000000000000":
                fail(f"the winning roadmap start did not link its loop: {status}, {reply!r}")
        finally:
            host.close()


@check("loops.roadmap_file_is_not_overwritten")
def roadmap_file_is_not_overwritten() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        host = _start_host(root, FakeModelProvider([]))
        try:
            path = RoadmapStore(root).path
            path.parent.mkdir(parents=True, exist_ok=True)
            original = b'{"goals": "x"}\n'
            path.write_bytes(original)
            status, reply = _http(host, "POST", "/roadmap/add", {"goal": "Keep malformed bytes"})
            if status != 409 or not reply.get("error", "").startswith("invalid roadmap: "):
                fail(f"malformed roadmap add did not return 409: {status}, {reply!r}")
            if path.read_bytes() != original:
                fail("malformed roadmap was changed by the rejected add")
        finally:
            host.close()


@check("loops.approve_with_an_invalid_roadmap")
def approve_with_an_invalid_roadmap() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        host = _start_host(root, FakeModelProvider([]))
        subscription = host.broker.subscribe()
        try:
            loop = LoopStore(root).create("Approve with malformed roadmap")
            loop["state"] = "awaiting_commit"
            LoopStore(root).save(loop)
            path = RoadmapStore(root).path
            path.parent.mkdir(parents=True, exist_ok=True)
            original = b'{"goals": "x"}\n'
            path.write_bytes(original)
            status, reply = _http(host, "POST", "/loop/commit", {
                "loop_id": loop["id"], "action": "approve",
            })
            if status != 200 or reply.get("loop", {}).get("state") != "done":
                fail(f"approval failed with an invalid roadmap: {status}, {reply!r}")
            if path.read_bytes() != original:
                fail("approval changed the malformed roadmap file")
            event = subscription.get(timeout=1)
            if not isinstance(event, LoopChanged) or event.loop_id != loop["id"] or event.state != "done":
                fail(f"approval did not publish the done event: {event!r}")
        finally:
            subscription.close()
            host.close()


@check("loops.round_trip")
def round_trip() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        store = LoopStore(root)
        loop = store.create("First loop")
        loop["reason"] = "saved reason"
        store.save(loop)
        if store.load(loop["id"]) != loop:
            fail("save then load did not preserve the loop record")
        restarted = LoopStore(root)
        if restarted.load(loop["id"]) != loop:
            fail("a new LoopStore did not see the persisted record")
        loop["created_at"] = "2000-01-01T00:00:00+00:00"
        store.save(loop)
        newest = restarted.create("Second loop")
        if [item["id"] for item in restarted.list()] != [newest["id"], loop["id"]]:
            fail("list did not return loops newest first")
        malformed = store.path(loop["id"], "loop.json")
        malformed.write_text("{", encoding="utf-8")
        if any(item["id"] == loop["id"] for item in restarted.list()):
            fail("list did not skip a malformed loop record")
        try:
            restarted.load(loop["id"])
        except ValueError:
            pass
        else:
            fail("load did not reject a malformed loop record")


@check("loops.state_transitions")
def state_transitions() -> None:
    transitions = {
        "planning": ("awaiting_plan", "needs_attention", "cancelled"),
        "awaiting_plan": ("planning", "running", "cancelled"),
        "running": ("validating", "needs_attention", "cancelled"),
        "validating": ("awaiting_commit", "needs_attention", "cancelled"),
        "awaiting_commit": ("done", "cancelled"),
        "needs_attention": ("planning", "running", "validating", "cancelled"),
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        store = LoopStore(root)
        for source, targets in transitions.items():
            for target in targets:
                loop = store.create("Transition check")
                loop["state"] = source
                store.save(loop)
                if store.set_state(loop, target, "transition reason")["state"] != target:
                    fail(f"allowed transition was rejected: {source} -> {target}")
                if loop["reason"] != "transition reason":
                    fail(f"transition reason was not saved: {source} -> {target}")
        for source, target in (("planning", "running"), ("done", "planning")):
            loop = store.create("Invalid transition check")
            loop["state"] = source
            store.save(loop)
            try:
                store.set_state(loop, target)
            except ValueError:
                continue
            fail(f"invalid transition was accepted: {source} -> {target}")


@check("loops.ids_and_paths_are_safe")
def ids_and_paths_are_safe() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        store = LoopStore(root)
        loop = store.create("Path safety")
        outside = Path(temporary) / "outside.txt"
        outside.write_text("untouched", encoding="utf-8")
        for invalid_id in ("../x", "loop_ZZ"):
            try:
                store.load(invalid_id)
            except KeyError:
                continue
            fail(f"invalid loop id was accepted: {invalid_id!r}")
        try:
            store.path(loop["id"], "../../../../outside.txt")
        except ValueError:
            pass
        else:
            fail("path escaping the loop folder was accepted")
        if outside.read_text(encoding="utf-8") != "untouched":
            fail("path validation touched a file outside the loop folder")


@check("loops.planner_writes_a_plan")
def planner_writes_a_plan() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        template = root / "specs" / "TEMPLATE.md"
        template.parent.mkdir(parents=True)
        template.write_text("Repository-specific template that SymphonAI must ignore.\n", encoding="utf-8")
        spec_a = "# a — Read files\n\n## Validation\n\n```sh\ntrue\n```\n"
        spec_b = "# b — Review errors\n\n## Validation\n\n```sh\ntrue\n```\n"
        plan = {"cards": [
            {"id": "a", "title": "Read files", "spec": "specs/a-read-files.md", "lane": 1, "after": []},
            {"id": "b", "title": "Review errors", "spec": "specs/b-review-errors.md", "lane": 2, "after": ["a"]},
        ]}
        provider = _RecordingProvider([
            _write("src/x.py", "should be denied\n", "outside"),
            _write("specs/a-read-files.md", spec_a, "spec-a"),
            _write("specs/b-review-errors.md", spec_b, "spec-b"),
            _write("plan.json", json.dumps(plan), "plan"),
            _finish(),
        ])
        loop, events, run = _run_loop(root, provider)
        if loop["state"] != "awaiting_plan" or [card["state"] for card in loop["cards"]] != ["planned", "planned"]:
            tool_errors = [
                message.tool_result.error or message.tool_result.content
                for request in provider.requests
                for message in request.messages
                if message.tool_result is not None
            ]
            fail(f"valid planner output did not await plan approval: {loop!r}, requests={len(provider.requests)}, tool_results={tool_errors!r}")
        if [card["lane"] for card in loop["cards"]] != [1, 2] or [lane["cards"] for lane in loop["lanes"]] != [["a"], ["b"]]:
            fail(f"planner lanes were not recorded: {loop['lanes']!r}, {loop['cards']!r}")
        if [event.change for event in events] != ["created", "planned"]:
            fail(f"loop events did not announce creation then plan: {[event.change for event in events]!r}")
        if not loop.get("planner_session_id"):
            fail("planner session id was not recorded")
        if (root / "src" / "x.py").exists():
            fail("planner wrote outside specs and plan.json")
        refused = [
            message.tool_result.error or message.tool_result.content
            for message in provider.requests[1].messages
            if message.tool_result is not None
        ]
        if not refused or any("outside the explicit allowed write scope" not in str(result) for result in refused):
            fail(f"out-of-scope planner write was not refused: {refused!r}")
        from symphonai_api.tools.shell import _sandbox_available

        if run._policy.shell_enabled != _sandbox_available() or run._policy.mode != "allow":
            fail("planner shell availability did not match the sandbox in allow mode")
        expected_scope = [
            LoopStore(root).path(loop["id"], "specs"),
            LoopStore(root).path(loop["id"], "plan.json"),
        ]
        if run._policy.allowed_write_scope != expected_scope:
            fail(f"planner write scope was wider than specs and plan.json: {run._policy.allowed_write_scope!r}")
        session = SessionStore.open(root / "sessions", loop["planner_session_id"])
        try:
            meta = session.read_meta()
        finally:
            session.close()
        if meta.get("spec_run") != {
            "kind": "loop_plan", "loop_id": loop["id"], "role": "planner",
        } or meta.get("title") != "Plan: Add a small capability":
            fail(f"planner session metadata or title was incorrect: {meta!r}")
        if not any(
            message.role is Role.USER and "Write plan.json last" in message.text
            for message in provider.requests[0].messages
        ):
            fail("planner task omitted the plan instructions")
        if any("TEMPLATE.md" in message.text for message in provider.requests[0].messages):
            fail("planner request named the project's TEMPLATE.md")
        if not any(
            message.role is Role.USER and "one fenced `sh` block" in message.text
            and "nothing else in that section is run" in message.text
            for message in provider.requests[0].messages
        ):
            fail("planner task omitted the Validation command format")


@check("loops.planner_shell_is_scoped")
def planner_shell_is_scoped() -> None:
    from symphonai_api.tools.shell import _sandbox_available

    if not _sandbox_available():
        print("SKIP: shell sandbox is unavailable")
        return
    with tempfile.TemporaryDirectory(dir="/private/var/tmp") as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        (root / "README.md").write_text("project is readable\n", encoding="utf-8")
        folder = root / ".symphonai" / "loops" / _TEST_LOOP_ID
        specs = folder / "specs"
        read = ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
            "planner-read", "run_shell", {"argv": ["cat", "README.md"]},
        )]))
        inside = ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
            "planner-inside", "run_shell", {
                "command": f"printf inside > {str(specs / 'shell.txt')!r}",
            },
        )]))
        outside = ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
            "planner-outside", "run_shell", {
                "command": f"printf outside > {str(root / 'outside.txt')!r}",
            },
        )]))
        spec = "# a — Read files\n\n## Validation\n\n```sh\ntrue\n```\n"
        plan = {"cards": [{
            "id": "a", "title": "Read files", "spec": "specs/a-read-files.md",
            "lane": 1, "after": [],
        }]}
        provider = _RecordingProvider([
            read, inside, outside,
            _write("specs/a-read-files.md", spec, "spec"),
            _write("plan.json", json.dumps(plan), "plan"),
            _finish(),
        ])
        loop, _events, run = _run_loop(root, provider)
        try:
            results = {
                message.tool_result.tool_call_id: message.tool_result
                for request in provider.requests for message in request.messages
                if message.tool_result is not None
            }
            if loop["state"] != "awaiting_plan":
                fail(f"sandboxed planner did not finish its plan: {loop!r}")
            if not results.get("planner-read") or not results["planner-read"].ok or "project is readable" not in results["planner-read"].content:
                fail(f"planner shell could not read the project: {results.get('planner-read')!r}")
            if not results.get("planner-inside") or not results["planner-inside"].ok:
                fail(f"planner shell could not write its specs folder: {results.get('planner-inside')!r}")
            if not results.get("planner-outside") or results["planner-outside"].ok:
                fail(f"planner shell wrote outside its scope: {results.get('planner-outside')!r}")
            if (specs / "shell.txt").read_text(encoding="utf-8") != "inside":
                fail("planner shell did not create its in-scope file")
            if (root / "outside.txt").exists():
                fail("planner shell created a file outside its specs folder")
        finally:
            run.close()


@check("loops.planner_without_sandbox_hides_shell")
def planner_without_sandbox_hides_shell() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        provider = _RecordingProvider([
            _write("specs/a.md", "# A\n\n## Validation\n\n```sh\ntrue\n```\n", "spec"),
            _write("plan.json", json.dumps({"cards": [{
                "id": "a", "title": "A", "spec": "specs/a.md", "lane": 1, "after": [],
            }]}), "plan"),
            _finish(),
        ])
        with mock.patch("symphonai_api.tools.shell._sandbox_available", return_value=False):
            loop, _events, run = _run_loop(root, provider)
        try:
            first = provider.requests[0]
            tools = {tool.get("function", {}).get("name") for tool in first.tools or []}
            if "run_shell" in tools:
                fail("planner without a sandbox was given run_shell")
            if not any(
                message.role is Role.USER and "commands are unavailable" in message.text
                for message in first.messages
            ):
                fail("planner without a sandbox was not told commands are unavailable")
            if loop["state"] != "awaiting_plan":
                fail(f"no-sandbox planner did not complete: {loop!r}")
        finally:
            run.close()


@check("loops.cards_need_validation_commands")
def cards_need_validation_commands() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        provider = _RecordingProvider([
            _write("specs/a.md", "# A\n\n## Validation\n\n- `true`\n", "spec"),
            _write("plan.json", json.dumps({"cards": [
                {"id": "a", "title": "A", "spec": "specs/a.md", "lane": 1, "after": []},
            ]}), "plan"),
            _finish(),
        ])
        loop, _events, run = _run_loop(root, provider)
        if loop["state"] != "needs_attention" or loop["reason"] != "card a has no Validation commands":
            fail(f"plan without executable Validation commands was accepted: {loop!r}")
        if not any(
            message.role is Role.USER and "one fenced `sh` block" in message.text
            and "nothing else in that section is run" in message.text
            for message in provider.requests[0].messages
        ):
            fail("planner request omitted the Validation command format")
        run.close()

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        card = {"id": "a", "title": "A", "lane": 1, "state": "review"}
        run, store, loop = _lane_loop(root, sessions, [card])
        started = []
        run.start_spec_run = lambda spec: (started.append(spec) or ("run", "unexpected"))
        try:
            loop = run.approve_loop_plan(loop["id"])
            source_id = "parent-card"
            create_worktree(root, sessions / source_id / "worktree", start=loop["branch"])
            _loop_session(sessions, root, source_id, {
                "kind": "implement", "loop_id": loop["id"], "card_id": "a",
            })
            follow_spec = store.path(loop["id"], "specs/a-follow-up.md")
            follow_spec.write_text("# Follow-up\n\n## Validation\n\n- `true`\n", encoding="utf-8")
            follow_path = f".symphonai/loops/{loop['id']}/specs/a-follow-up.md"
            source = SessionStore.open(sessions, source_id)
            try:
                meta = source.read_meta()
                meta["review"] = {"verdict": "follow-ups", "follow_ups": [follow_path]}
                source.write_meta(meta)
            finally:
                source.close()
            current = store.load(loop["id"])
            current["state"] = "running"
            current["cards"][0].update(state="review", session_id=source_id)
            store.save(current)
            run._finish_loop_card_review({"loop_id": loop["id"], "card_id": "a", "of": source_id})
            result = store.load(loop["id"])
            follow = next(item for item in result["cards"] if item["id"] == "aF")
            if result["state"] != "needs_attention" or follow["state"] != "failed" or "card aF" not in result["reason"]:
                fail(f"follow-up without commands did not fail its card and loop: {result!r}")
            if started:
                fail(f"implementer conversation started for a follow-up without commands: {started!r}")
        finally:
            run.close()


@check("loops.bad_plans_need_attention")
def bad_plans_need_attention() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        cases = [
            ("missing", [_finish()], "missing"),
            ("outside", [_write("plan.json", json.dumps({"cards": [
                {"id": "a", "title": "Outside", "spec": "../../outside.md", "lane": 1, "after": []},
            ]}), "plan"), _finish()], "escapes"),
            ("too-many", [_write("plan.json", json.dumps({"cards": [
                {"id": card_id, "title": card_id, "spec": "specs/a.md", "lane": 1, "after": []}
                for card_id in "abcdefghi"
            ]}), "plan"), _finish()], "1 to 8"),
            ("cycle", [
                _write("specs/a.md", "# A\n\n## Validation\n\n```sh\ntrue\n```\n", "spec"),
                _write("plan.json", json.dumps({"cards": [
                    {"id": "a", "title": "A", "spec": "specs/a.md", "lane": 1, "after": ["b"]},
                    {"id": "b", "title": "B", "spec": "specs/a.md", "lane": 1, "after": ["a"]},
                ]}), "plan"),
                _finish(),
            ], "cycle"),
        ]
        for name, responses, reason in cases:
            root = base / name
            loop, _events, _run = _run_loop(root, _RecordingProvider(responses))
            if loop["state"] != "needs_attention" or reason not in loop["reason"].lower():
                fail(f"bad plan {name!r} did not name its problem: {loop!r}")

        class _FailingProvider(_RecordingProvider):
            def create_response(self, request, *, cancel=None):
                self.requests.append(request)
                raise ProviderError("scripted planner failure")

        failed, _events, _run = _run_loop(base / "failed", _FailingProvider([]))
        if failed["state"] != "needs_attention" or "failed" not in failed["reason"].lower():
            fail(f"failed planner did not need attention: {failed!r}")


@check("loops.routes_start_read_list")
def routes_start_read_list() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        plan = {"cards": [
            {"id": "a", "title": "First", "spec": "specs/a-first.md", "lane": 1, "after": []},
            {"id": "b", "title": "Second", "spec": "specs/b-second.md", "lane": 2, "after": []},
        ]}
        provider = _RecordingProvider([
            _write("specs/a-first.md", "# First\n\n## Validation\n\n```sh\ntrue\n```\n", "a"),
            _write("specs/b-second.md", "# Second\n\n## Validation\n\n```sh\ntrue\n```\n", "b"),
            _write("plan.json", json.dumps(plan), "plan"),
            _finish(),
        ])
        host = _start_host(root, provider)
        try:
            with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000000"):
                status, reply = _http(host, "POST", "/loop/start", {"goal": "A routed loop"})
            if status != 200 or not reply.get("session_id") or reply.get("loop", {}).get("state") != "planning":
                fail(f"loop/start did not return its loop and planner session: {status}, {reply!r}")
            loop_id = reply["loop"]["id"]
            _wait_loop_state(host, loop_id, "awaiting_plan")
            status, detail = _http(host, "GET", f"/loop?id={loop_id}")
            if status != 200 or detail.get("loop", {}).get("state") != "awaiting_plan":
                fail(f"GET /loop did not return the planned loop: {status}, {detail!r}")
            status, listing = _http(host, "GET", "/loops")
            summary = next((loop for loop in listing.get("loops", []) if loop.get("id") == loop_id), None)
            if status != 200 or summary != {
                "id": loop_id, "goal": "A routed loop", "state": "awaiting_plan",
                "updated_at": detail["loop"]["updated_at"],
            }:
                fail(f"GET /loops omitted the loop summary: {status}, {listing!r}")
        finally:
            host.close()


@check("loops.plan_checkpoint")
def plan_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        _git_repo(root)
        host = _start_host(root, _RecordingProvider([_finish()]))
        try:
            store = LoopStore(root)
            loop = store.create("Checkpoint plan")
            status, error = _http(host, "POST", "/loop/plan", {
                "loop_id": loop["id"], "action": "approve",
            })
            if status != 409 or "planning" not in error.get("error", ""):
                fail(f"approve in planning did not return 409 with its state: {status}, {error!r}")

            loop["state"] = "awaiting_plan"
            loop["cards"] = [
                {"id": "a", "title": "First", "after": [], "state": "planned"},
                {"id": "b", "title": "Second", "after": ["a"], "state": "planned"},
            ]
            store.save(loop)
            status, error = _http(host, "POST", "/loop/plan", {
                "loop_id": loop["id"], "action": "drop", "cards": ["missing"],
            })
            if status != 400 or "unknown card" not in error.get("error", ""):
                fail(f"unknown card drop was not refused: {status}, {error!r}")
            status, error = _http(host, "POST", "/loop/plan", {
                "loop_id": loop["id"], "action": "drop", "cards": ["a"],
            })
            if status != 400 or "b" not in error.get("error", ""):
                fail(f"dropping a dependency did not name the kept card: {status}, {error!r}")
            status, reply = _http(host, "POST", "/loop/plan", {
                "loop_id": loop["id"], "action": "drop", "cards": ["b"],
            })
            if status != 200 or reply.get("loop", {}).get("cards", [])[1].get("state") != "dropped":
                fail(f"drop did not mark the selected card: {status}, {reply!r}")
            status, reply = _http(host, "POST", "/loop/plan", {
                "loop_id": loop["id"], "action": "approve",
            })
            accepted = reply.get("loop", {})
            if status != 200 or accepted.get("state") != "running" or accepted.get("cards", [])[0].get("state") != "planned":
                fail(f"approve did not move the kept plan to running: {status}, {reply!r}")

            empty = store.create("Empty checkpoint plan")
            empty["state"] = "awaiting_plan"
            empty["cards"] = [{"id": "a", "state": "planned", "after": []}]
            store.save(empty)
            status, _reply = _http(host, "POST", "/loop/plan", {
                "loop_id": empty["id"], "action": "drop", "cards": ["a"],
            })
            if status != 200:
                fail(f"dropping every card was refused: {status}")
            status, error = _http(host, "POST", "/loop/plan", {
                "loop_id": empty["id"], "action": "approve",
            })
            if status != 400 or "nothing left to run" not in error.get("error", ""):
                fail(f"approving an empty plan was not refused: {status}, {error!r}")

            status, _reply = _http(host, "GET", "/loop?id=../x")
            if status != 404:
                fail(f"malformed loop id did not return 404: {status}")

            revision_root = root / "revision"
            revision_plan = {"cards": [
                {"id": "a", "title": "Old card", "spec": "specs/a-old-card.md", "lane": 1, "after": []},
            ]}
            revised_plan = {"cards": [
                {"id": "b", "title": "New card", "spec": "specs/b-new-card.md", "lane": 1, "after": []},
            ]}
            revision_provider = _RecordingProvider([
                _write("specs/a-old-card.md", "# Old card\n\n## Validation\n\n```sh\ntrue\n```\n", "old-spec"),
                _write("plan.json", json.dumps(revision_plan), "old-plan"),
                _finish(),
                _write("specs/b-new-card.md", "# New card\n\n## Validation\n\n```sh\ntrue\n```\n", "new-spec"),
                _write("plan.json", json.dumps(revised_plan), "new-plan"),
                _finish(),
            ])
            revision_host = _start_host(revision_root, revision_provider)
            try:
                with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000000"):
                    status, initial = _http(revision_host, "POST", "/loop/start", {"goal": "Revise this plan"})
                if status != 200:
                    fail(f"initial revision plan did not start: {status}, {initial!r}")
                loop_id = initial["loop"]["id"]
                before = _wait_loop_state(revision_host, loop_id, "awaiting_plan")
                status, revised = _http(revision_host, "POST", "/loop/plan", {
                    "loop_id": loop_id, "action": "revise", "note": "please split out the new work",
                })
                if status != 200 or not revised.get("session_id"):
                    fail(f"plan revision did not start a new planner: {status}, {revised!r}")
                after = _wait_loop_state(revision_host, loop_id, "awaiting_plan")
                if ([card["id"] for card in after["cards"]] != ["b"]
                        or revised["session_id"] == before["planner_session_id"]
                        or after["planner_session_id"] != revised["session_id"]):
                    fail(f"revision did not replace the plan: {after!r}")
                request_messages = [
                    message.text
                    for request in revision_provider.requests[3:]
                    for message in request.messages
                    if message.role is Role.USER
                ]
                if not any("please split out the new work" in text and '"id": "a"' in text for text in request_messages):
                    fail(f"revision note and prior plan were missing from the planner task: {request_messages!r}")
                if not LoopStore(revision_root).path(loop_id, "specs/a-old-card.md").is_file():
                    fail("revision removed the existing specs folder contents")
            finally:
                revision_host.close()
        finally:
            host.close()


@check("loops.cancel")
def cancel() -> None:
    class _BlockingProvider(_RecordingProvider):
        def __init__(self) -> None:
            super().__init__([_finish()])
            self.entered = threading.Event()

        def create_response(self, request, *, cancel=None):
            self.requests.append(request)
            self.entered.set()
            while True:
                if cancel is not None:
                    cancel.raise_if_cancelled()
                time.sleep(0.01)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        provider = _BlockingProvider()
        host = _start_host(root, provider)
        try:
            with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000000"):
                status, reply = _http(host, "POST", "/loop/start", {"goal": "Cancel this plan"})
            if status != 200:
                fail(f"loop/start failed before cancellation: {status}, {reply!r}")
            if not provider.entered.wait(2):
                fail("planner did not begin before cancellation")
            loop_id = reply["loop"]["id"]
            status, cancelled = _http(host, "POST", "/loop/cancel", {"loop_id": loop_id})
            if status != 200 or cancelled.get("loop", {}).get("state") != "cancelled":
                fail(f"cancel did not stop the planner loop: {status}, {cancelled!r}")
            status, repeated = _http(host, "POST", "/loop/cancel", {"loop_id": loop_id})
            if status != 409:
                fail(f"second cancellation was not refused: {status}, {repeated!r}")
        finally:
            host.close()


@check("loops.approve_makes_a_branch")
def approve_makes_a_branch() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "project"
        _git_repo(root)
        _git(root, "branch", "symphonai/fix-x")
        (root / "README.md").write_text("user work\n", encoding="utf-8")
        status_before = _git(root, "status", "--porcelain")
        branch_before = _git(root, "branch", "--show-current")
        sessions = base / "sessions"
        store = LoopStore(root)
        loop = store.create("Fix X")
        loop["state"] = "awaiting_plan"
        loop["cards"] = [{"id": "a", "title": "Fix the value", "state": "planned", "after": []}]
        store.save(loop)
        run = HostRun(
            FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]), EventBroker(),
            sessions_root=sessions,
        )
        try:
            approved = run.approve_loop_plan(loop["id"])
            expected_branch = "symphonai/fix-x-2"
            if approved.get("branch") != expected_branch or approved.get("state") != "running":
                fail(f"approve did not record the suffixed loop branch: {approved!r}")
            integration = sessions / "loops" / loop["id"] / "integration"
            if not integration.is_dir() or _git(integration, "branch", "--show-current").decode().strip() != expected_branch:
                fail("approve did not create the attached integration worktree")
            if _git(root, "status", "--porcelain") != status_before or _git(root, "branch", "--show-current") != branch_before:
                fail("approve changed the person's status or current branch")
            if (root / "README.md").read_text(encoding="utf-8") != "user work\n":
                fail("approve changed the person's uncommitted work")
        finally:
            run.close()


def _lane_loop(root: Path, sessions: Path, cards: list[dict]) -> tuple[HostRun, LoopStore, dict]:
    _git_repo(root)
    store = LoopStore(root)
    loop = store.create("Run lane checks")
    loop.update(state="awaiting_plan", cards=cards)
    lane_ids = sorted({card["lane"] for card in cards})
    loop["lanes"] = [{"id": lane, "cards": [card["id"] for card in cards if card["lane"] == lane]} for lane in lane_ids]
    for card in cards:
        card.setdefault("state", "planned")
        card.setdefault("after", [])
        card.setdefault("rounds", 0)
        card.setdefault("follow_up_of", None)
        spec_path = store.path(loop["id"], f"specs/{card['id']}.md")
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(f"# {card['id']}\n\n## Validation\n\n```sh\ntrue\n```\n", encoding="utf-8")
        card["spec"] = f"specs/{card['id']}.md"
        card["report"] = f"reports/{card['id']}-report.md"
    store.save(loop)
    return HostRun(
        FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]), EventBroker(),
        sessions_root=sessions,
    ), store, loop


def _loop_session(sessions: Path, root: Path, session_id: str, info: dict) -> None:
    session = SessionStore(sessions, session_id, repo_root=root)
    try:
        meta = session.read_meta()
        meta["spec_run"] = info
        session.write_meta(meta)
    finally:
        session.close()


@check("loops.lanes_run_in_order_and_in_parallel")
def lanes_run_in_order_and_in_parallel() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        cards = [
            {"id": "a", "title": "A", "lane": 1},
            {"id": "b", "title": "B", "lane": 1, "after": ["a"]},
            {"id": "c", "title": "C", "lane": 2},
        ]
        run, store, loop = _lane_loop(base / "project", base / "sessions", cards)
        started = []
        run.start_spec_run = lambda spec: (started.append(spec) or ("run", f"s{len(started)}"))
        worktrees = []
        try:
            loop = run.approve_loop_plan(loop["id"])
            if [spec["card_id"] for spec in started] != ["a", "c"]:
                fail(f"independent lanes did not start together: {started!r}")
            integration = run._loop_integration_root(loop["id"])
            work_a = create_worktree(base / "project", base / "sessions/cards/a", start=started[0]["start_ref"])
            worktrees.append(base / "sessions/cards/a")
            (work_a / "a.py").write_text("value = 2\n", encoding="utf-8")
            run.integrate(loop, cards[0], work_a)
            persisted = store.load(loop["id"])
            persisted["cards"][0]["state"] = "done"
            store.save(persisted)
            run._advance_loop(loop["id"])
            if [spec["card_id"] for spec in started] != ["a", "c", "b"]:
                fail(f"dependent card did not wait for its predecessor: {started!r}")
            work_b = create_worktree(base / "project", base / "sessions/cards/b", start=started[2]["start_ref"])
            worktrees.append(base / "sessions/cards/b")
            if (work_b / "a.py").read_text(encoding="utf-8") != "value = 2\n":
                fail("dependent card did not start from its predecessor's integrated tip")
            for card_id, worktree in (("c", create_worktree(base / "project", base / "sessions/cards/c", start=started[1]["start_ref"])), ("b", work_b)):
                worktrees.append(base / f"sessions/cards/{card_id}")
                (worktree / f"{card_id}.txt").write_text(card_id, encoding="utf-8")
                run.integrate(loop, {"id": card_id, "title": card_id.upper()}, worktree)
            subjects = _git(integration, "log", "--format=%s", "-3").decode().splitlines()
            if subjects != ["b: B", "c: C", "a: A"]:
                fail(f"integration commits did not follow finish order: {subjects!r}")
        finally:
            for path in worktrees + [base / "sessions/loops" / loop["id"] / "integration"]:
                if path.exists():
                    try:
                        remove_worktree(base / "project", path)
                    except Exception:
                        pass
            run.close()

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        cards = [{"id": card_id, "title": card_id, "lane": lane} for lane, card_id in enumerate("abc", 1)]
        run, _store, loop = _lane_loop(base / "project", base / "sessions", cards)
        started = []
        def start_fake(spec):
            started.append(spec)
            run._active_by_session[f"card-{spec['card_id']}"] = object()
            return "run", spec["card_id"]
        run.start_spec_run = start_fake
        try:
            run._active_by_session.update({"leader": object(), "other-one": object(), "other-two": object()})
            loop = run.approve_loop_plan(loop["id"])
            if len(started) != 1:
                fail(f"scheduler exceeded the four-conversation cap: {len(started)} cards")
            run._active_by_session.pop("other-one")
            run._advance_loop(loop["id"])
            if len(started) != 2:
                fail(f"a waiting lane did not start after one slot opened: {started!r}")
            run._active_by_session.pop("other-two")
            run._advance_loop(loop["id"])
            if [spec["card_id"] for spec in started] != ["a", "b", "c"]:
                fail(f"waiting lane did not start when a conversation slot opened: {started!r}")
        finally:
            run._active_by_session.clear()
            run.close()


@check("loops.review_reason")
def review_reason() -> None:
    cases = (
        ({"verdict": "tree-changed", "error": "An error occurred while processing the request."},
         "card a review failed: An error occurred while processing the request."),
        ({"verdict": "no-verdict"}, "card a review gave no verdict"),
        ({"verdict": "tree-changed"}, "card a review changed files other than its follow-up specs"),
    )
    for index, (review, expected) in enumerate(cases):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, sessions = base / "project", base / "sessions"
            run, store, loop = _lane_loop(
                root, sessions, [{"id": "a", "title": "A", "lane": 1, "state": "review"}],
            )
            try:
                loop = store.load(loop["id"])
                loop["state"] = "running"
                store.save(loop)
                session_id = f"review-{index}"
                _loop_session(sessions, root, session_id, {"kind": "implement", "loop_id": loop["id"], "card_id": "a"})
                source = SessionStore.open(sessions, session_id)
                try:
                    meta = source.read_meta()
                    meta["review"] = review
                    source.write_meta(meta)
                finally:
                    source.close()
                run._finish_loop_card_review({"loop_id": loop["id"], "card_id": "a", "of": session_id})
                result = store.load(loop["id"])
                if result.get("reason") != expected:
                    fail(f"review result {review!r} had reason {result.get('reason')!r}, expected {expected!r}")
            finally:
                run.close()


@check("loops.review_pass_integrates")
def review_pass_integrates() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        card = {"id": "a", "title": "A", "lane": 1, "state": "review"}
        run, store, loop = _lane_loop(root, sessions, [card])
        try:
            loop = run.approve_loop_plan(loop["id"])
            session_id = "card-session"
            worktree = create_worktree(root, sessions / session_id / "worktree", start=loop["branch"])
            (worktree / "a.py").write_text("value = 2\n", encoding="utf-8")
            _loop_session(sessions, root, session_id, {"kind": "implement", "loop_id": loop["id"], "card_id": "a"})
            source = SessionStore.open(sessions, session_id)
            try:
                meta = source.read_meta()
                meta["review"] = {"verdict": "passed", "follow_ups": []}
                source.write_meta(meta)
            finally:
                source.close()
            persisted = store.load(loop["id"])
            persisted["state"] = "running"
            persisted["cards"][0]["state"] = "review"
            store.save(persisted)
            run._finish_loop_card_review({"loop_id": loop["id"], "card_id": "a", "of": session_id})
            result = store.load(loop["id"])
            if result["cards"][0]["state"] != "done" or result["state"] != "validating":
                fail(f"passing review did not complete the card and loop: {result!r}")
            if _git(run._loop_integration_root(loop["id"]), "show", "HEAD:a.py").decode() != "value = 2\n":
                fail("passing review did not integrate the implementation")
        finally:
            run.close()


@check("loops.follow_ups_are_capped")
def follow_ups_are_capped() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        card = {"id": "a", "title": "A", "lane": 1, "state": "review"}
        run, store, loop = _lane_loop(root, sessions, [card])
        started = []
        run.start_spec_run = lambda spec: (started.append(spec) or ("run", f"follow-{len(started)}"))
        try:
            loop = run.approve_loop_plan(loop["id"])
            source_id = "first-card"
            card_worktree = create_worktree(root, sessions / source_id / "worktree", start=loop["branch"])
            _loop_session(sessions, root, source_id, {
                "kind": "implement", "loop_id": loop["id"], "card_id": "a",
                "spec": f".symphonai/loops/{loop['id']}/specs/a.md",
            })
            follow_path = f".symphonai/loops/{loop['id']}/specs/a-follow-up.md"
            (card_worktree / follow_path).parent.mkdir(parents=True, exist_ok=True)
            (card_worktree / follow_path).write_text(
                "# Follow-up\n\n## Validation\n\n```sh\ntrue\n```\n", encoding="utf-8",
            )
            review_id = "first-review"
            _loop_session(sessions, root, review_id, {
                "kind": "review", "loop_id": loop["id"], "card_id": "a", "of": source_id,
                "worktree_path": str(card_worktree), "baseline_files": {},
            })
            review = SessionStore.open(sessions, review_id)
            try:
                meta = review.read_meta()
                meta["spec_run"].update(worktree_path=str(card_worktree))
                review.write_meta(meta)
            finally:
                review.close()
            source = SessionStore.open(sessions, source_id)
            try:
                meta = source.read_meta()
                meta["spec_run"].update(spec=f".symphonai/loops/{loop['id']}/specs/a.md")
                source.write_meta(meta)
            finally:
                source.close()
            current = store.load(loop["id"])
            current["state"] = "running"
            current["cards"][0].update(state="review", session_id=source_id)
            store.save(current)
            run._finish_spec_review(
                review_id,
                type("ReviewLeader", (), {"_chat_messages": [Message(Role.ASSISTANT, f"Verdict: follow-ups: {follow_path}")]})(),
                None,
            )
            current = store.load(loop["id"])
            follow = next((item for item in current["cards"] if item["id"] == "aF"), None)
            if follow is None:
                fail(f"review follow-up was not added: {current!r}")
            if (follow["lane"] != 1 or follow["after"] != ["a"] or follow["rounds"] != 1
                    or follow["state"] != "running" or [spec["card_id"] for spec in started] != ["aF"]):
                fail(f"first follow-up was not added and run in its lane: {current!r}, {started!r}")
            follow_session = "follow-card"
            follow_tree = create_worktree(root, sessions / follow_session / "worktree", start=loop["branch"])
            (follow_tree / "follow.py").write_text("done\n", encoding="utf-8")
            _loop_session(sessions, root, follow_session, {"kind": "implement", "loop_id": loop["id"], "card_id": "aF"})
            source = SessionStore.open(sessions, follow_session)
            try:
                meta = source.read_meta()
                meta["review"] = {"verdict": "passed", "follow_ups": []}
                source.write_meta(meta)
            finally:
                source.close()
            current = store.load(loop["id"])
            current["cards"][1].update(state="review", session_id=follow_session)
            store.save(current)
            run._finish_loop_card_review({"loop_id": loop["id"], "card_id": "aF", "of": follow_session})
            current = store.load(loop["id"])
            if current["state"] != "validating" or current["cards"][1]["state"] != "done":
                fail(f"passing follow-up review did not continue the loop: {current!r}")
        finally:
            run.close()


@check("loops.a_card_end_to_end")
def a_card_end_to_end() -> None:
    def run_case(base: Path, with_follow_up: bool) -> None:
        root, sessions = base / "project", base / "sessions"
        _git_repo(root)
        loop_id = "loop_000000000000"
        spec_path = f".symphonai/loops/{loop_id}/specs/a-change.md"
        report_path = f".symphonai/loops/{loop_id}/reports/a-change-a-report.md"
        spec_text = (
            "# Change a\n\n## Validation\n\n```sh\ntrue\n```\n\n"
            f"## Report\n\n`{report_path}`\n"
        )
        plan = {"cards": [{
            "id": "a", "title": "Change a", "spec": "specs/a-change.md", "lane": 1, "after": [],
        }]}
        responses = [
            _write("specs/a-change.md", spec_text, "plan-spec"),
            _write("plan.json", json.dumps(plan), "plan"),
            _finish(),
            ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
                "new-file", "write_file", {"path": "new.py", "content": "value = 2\n"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
                "report", "write_file", {"path": report_path, "content": "Implementation passed.\n"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "Implemented the change.")),
        ]
        follow_path = f".symphonai/loops/{loop_id}/specs/aF-more.md"
        if with_follow_up:
            follow_spec = (
                "# Follow-up\n\n## Validation\n\n```sh\ntrue\n```\n\n"
                "## Contract\n\nCreate follow.py.\n"
            )
            responses.extend([
                ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
                    "follow-up-spec", "write_file", {"path": follow_path, "content": follow_spec},
                )])),
                ModelResponse(Message(Role.ASSISTANT, f"Verdict: follow-ups: {follow_path}")),
                ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
                    "follow-file", "write_file", {"path": "follow.py", "content": "followed\n"},
                )])),
                ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
                    "follow-report", "write_file", {
                        "path": f".symphonai/loops/{loop_id}/reports/aF-follow-up-change-a-report.md",
                        "content": "Follow-up passed.\n",
                    },
                )])),
                ModelResponse(Message(Role.ASSISTANT, "Implemented the follow-up.")),
                ModelResponse(Message(Role.ASSISTANT, "Verdict: pass")),
            ])
        else:
            responses.append(ModelResponse(Message(Role.ASSISTANT, "Verdict: pass")))
        provider = _RecordingProvider(responses)
        run = HostRun(
            provider, PermissionPolicy(root, allowed_write_scope=[root]), EventBroker(),
            sessions_root=sessions,
        )
        store = LoopStore(root)
        try:
            with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000000"):
                loop = run.start_loop("Change a")
            _wait_state(store, loop["id"], "awaiting_plan")
            run.approve_loop_plan(loop["id"])
            loop = _wait_state(store, loop["id"], "awaiting_commit")
            expected_files = ["follow.py", "new.py"] if with_follow_up else ["new.py"]
            if loop.get("summary", {}).get("files") != expected_files:
                fail(f"loop summary included loop artifacts or missed project files: {loop.get('summary')!r}")
            integration = sessions / "loops" / loop["id"] / "integration"
            committed = set()
            for entry in loop["summary"]["commits"]:
                paths = _git(
                    integration, "show", "--format=", "--name-only", entry["commit"],
                ).decode().splitlines()
                committed.update(item for item in paths if item)
            committed = sorted(committed)
            if committed != expected_files:
                fail(f"card commits included loop artifacts or missed project files: {committed!r}")
            if not (root / report_path).is_file():
                fail(f"implementer report was not copied into the project loop folder: {report_path}")
            if with_follow_up:
                follow = next((item for item in loop["cards"] if item["id"] == "aF"), None)
                if follow is None or follow["state"] != "done":
                    fail(f"review follow-up did not create and run card aF: {loop!r}")
            else:
                if committed != ["new.py"] or loop["summary"].get("files") != ["new.py"]:
                    fail(f"initial card branch did not contain only the project change: {loop!r}")
            run.commit_loop(loop["id"], "approve")
            worktrees = [
                line.removeprefix("worktree ")
                for line in _git(root, "worktree", "list", "--porcelain").decode().splitlines()
                if line.startswith("worktree ")
            ]
            if worktrees != [str(root.resolve())]:
                fail(f"loop worktrees remained after checkpoint approval: {worktrees!r}")
        finally:
            run.close()

    with tempfile.TemporaryDirectory() as temporary:
        run_case(Path(temporary) / "without-follow-up", False)
    with tempfile.TemporaryDirectory() as temporary:
        run_case(Path(temporary) / "with-follow-up", True)

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        card = {"id": "aF2", "title": "Follow up", "lane": 1, "state": "review", "follow_up_of": "a", "rounds": 2}
        run, store, loop = _lane_loop(root, sessions, [card])
        try:
            loop = run.approve_loop_plan(loop["id"])
            session_id = "card-session"
            create_worktree(root, sessions / session_id / "worktree", start=loop["branch"])
            _loop_session(sessions, root, session_id, {"kind": "implement", "loop_id": loop["id"], "card_id": "aF2"})
            persisted = store.load(loop["id"])
            persisted.update(state="running")
            persisted["cards"][0]["state"] = "review"
            store.save(persisted)
            source = SessionStore.open(sessions, session_id)
            try:
                meta = source.read_meta()
                meta["review"] = {"verdict": "follow-ups", "follow_ups": ["ignored"]}
                source.write_meta(meta)
            finally:
                source.close()
            run._finish_loop_card_review({"loop_id": loop["id"], "card_id": "aF2", "of": session_id})
            result = store.load(loop["id"])
            if result["state"] != "needs_attention" or result["reason"] != "card a still has findings after 2 follow-up rounds":
                fail(f"follow-up cap did not stop after aF2: {result!r}")
            if any(item["id"] == "aF3" for item in result["cards"]):
                fail("third follow-up card was created")
        finally:
            run.close()


@check("loops.failed_card_needs_attention")
def failed_card_needs_attention() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        cards = [
            {"id": "a", "title": "A", "lane": 1, "state": "running"},
            {"id": "b", "title": "B", "lane": 2, "after": ["a"]},
        ]
        run, store, loop = _lane_loop(root, sessions, cards)
        started = []
        run.start_spec_run = lambda spec: (started.append(spec) or ("run", "unexpected"))
        try:
            loop = run.approve_loop_plan(loop["id"])
            _loop_session(sessions, root, "failed-session", {"kind": "implement", "loop_id": loop["id"], "card_id": "a"})
            run._finish_loop_card_run("failed-session", "blocked")
            result = store.load(loop["id"])
            if result["state"] != "needs_attention" or result["reason"] != "card a could not finish: blocked":
                fail(f"blocked implementation did not stop the loop: {result!r}")
            if result["cards"][0]["state"] != "failed" or any(spec["card_id"] == "b" for spec in started):
                fail(f"failure state or scheduling was wrong: {result!r}, {started!r}")
        finally:
            run.close()


@check("loops.card_run_reason")
def card_run_reason() -> None:
    from symphonai_host.goal import Goal, GoalCheck

    failure = "scripted implementer failure"

    class _FailingProvider(FakeModelProvider):
        def create_response(self, request, *, cancel=None):
            raise ProviderError(failure)

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        run, store, loop = _lane_loop(
            root, sessions, [{"id": "a", "title": "A", "lane": 1}],
        )
        run._provider = _FailingProvider()
        original_hook = threading.excepthook
        threading.excepthook = lambda _args: None
        try:
            loop = run.approve_loop_plan(loop["id"])
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                result = store.load(loop["id"])
                if result["state"] == "needs_attention":
                    break
                time.sleep(0.01)
            if result.get("reason") != f"card a could not finish: {failure}":
                fail(f"failed implementation reported the wrong cause: {result!r}")
            session_id = result["cards"][0].get("session_id")
            source = SessionStore.open(sessions, session_id)
            try:
                goal = source.read_meta().get("goal", {})
            finally:
                source.close()
            if goal.get("reason") != "failed":
                fail(f"card goal reason changed while reporting the error: {goal!r}")
        finally:
            run.close()
            threading.excepthook = original_hook

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        run, store, loop = _lane_loop(
            root, sessions,
            [{"id": "a", "title": "A", "lane": 1, "state": "running"}],
        )
        try:
            loop["state"] = "running"
            store.save(loop)
            session_id = "blocked-card"
            _loop_session(sessions, root, session_id, {
                "kind": "implement", "loop_id": loop["id"], "card_id": "a",
            })
            source = SessionStore.open(sessions, session_id)
            try:
                meta = source.read_meta()
                meta["goal"] = {"reason": "rounds exhausted"}
                source.write_meta(meta)
            finally:
                source.close()
            run._finish_spec_run(session_id, "blocked")
            result = store.load(loop["id"])
            if result.get("reason") != "card a could not finish: rounds exhausted":
                fail(f"blocked implementation reported the wrong cause: {result!r}")
            source = SessionStore.open(sessions, session_id)
            try:
                goal = source.read_meta()["goal"]
            finally:
                source.close()
            if goal.get("reason") != "rounds exhausted":
                fail(f"blocked goal reason changed while reporting the card: {goal!r}")
        finally:
            run.close()

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        run, store, loop = _lane_loop(
            root, sessions,
            [{"id": "a", "title": "A", "lane": 1, "state": "running"}],
        )
        try:
            loop["state"] = "running"
            session_id = "checked-card"
            loop["cards"][0]["session_id"] = session_id
            store.save(loop)
            _loop_session(sessions, root, session_id, {
                "kind": "implement", "loop_id": loop["id"], "card_id": "a",
            })
            goal = Goal(
                "finish implementation", ("/bin/sh", "-c", "set -e\ntrue\nfalse"),
                rounds=5, max_rounds=5,
            )
            run._save_goal(session_id, goal)
            run._goals_by_session[session_id] = goal
            run._policy_by_session[session_id] = PermissionPolicy(root, allowed_write_scope=[root])
            run._perform_goal_check(
                GoalCheck(goal, session_id, threading.Event(), "check-agent"), "check-run",
            )
            result = store.load(loop["id"])
            expected = "card a could not finish: validation still failing after 5 rounds (exit 1): true; false"
            if result.get("reason") != expected:
                fail(f"exhausted validation did not name its failing checks: {result!r}")
            if goal.reason != "rounds exhausted" or goal.last_check != {"exit": 1, "ok": False, "output": ""}:
                fail(f"goal state was changed or the failing check was not run: {goal!r}")
        finally:
            run.close()


def _validation_loop(root: Path, sessions: Path, commands: list[str]) -> tuple[HostRun, LoopStore, dict]:
    card = {"id": "a", "title": "A", "lane": 1, "state": "running", "commit": None}
    run, store, loop = _lane_loop(root, sessions, [card])
    loop = store.load(loop["id"])
    spec = store.path(loop["id"], card["spec"])
    spec.write_text("# A\n\n## Validation\n\n```sh\n" + "\n".join(commands) + "\n```\n", encoding="utf-8")
    loop = run.approve_loop_plan(loop["id"])
    integration = run._loop_integration_root(loop["id"])
    (integration / "a.py").write_text("value = 2\n", encoding="utf-8")
    _git(integration, "add", "a.py")
    _git(integration, "commit", "--quiet", "-m", "a: A")
    loop = store.load(loop["id"])
    loop["cards"][0].update(state="done", commit=_git(integration, "rev-parse", "HEAD").decode().strip())
    loop["state"] = "validating"
    store.save(loop)
    return run, store, loop


def _wait_state(store: LoopStore, loop_id: str, state: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        loop = store.load(loop_id)
        if loop["state"] == state:
            return loop
        time.sleep(0.01)
    fail(f"loop did not reach {state}: {store.load(loop_id)!r}")


@check("loops.final_validation")
def final_validation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        run, store, loop = _validation_loop(base / "project", base / "sessions", ["true", "test -f a.py"])
        try:
            loop = store.load(loop["id"])
            loop["state"] = "validating"
            store.save(loop)
            run._start_loop_validation(loop["id"])
            result = _wait_state(store, loop["id"], "awaiting_commit")
            summary = result.get("summary", {})
            if ([entry["command"] for entry in summary.get("validation", [])] != ["true", "test -f a.py"]
                    or any(entry["exit"] != 0 for entry in summary["validation"])):
                fail(f"combined validation results were not recorded: {summary!r}")
            if not summary.get("commits") or summary["files"] != ["a.py"]:
                fail(f"validation summary omitted commits or changed files: {summary!r}")
        finally:
            run.close()

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        run, store, loop = _validation_loop(base / "project", base / "sessions", ["false"])
        try:
            loop = store.load(loop["id"])
            loop["state"] = "validating"
            store.save(loop)
            run._start_loop_validation(loop["id"])
            result = _wait_state(store, loop["id"], "needs_attention")
            if result["reason"] != "validation failed: false" or result["summary"]["validation"][0]["exit"] != 1:
                fail(f"failed combined validation did not preserve its output: {result!r}")
        finally:
            run.close()


@check("loops.commit_checkpoint")
def commit_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        _git_repo(root)
        store = LoopStore(root)
        run = HostRun(FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]), EventBroker(), sessions_root=sessions)
        host = HostServer(
            FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]),
            runtime=ProjectRuntime(root, run),
        )
        host.start()
        try:
            branches = []
            for action in ("approve", "decline"):
                loop = store.create(f"Checkpoint {action}")
                loop["state"] = "awaiting_commit"
                loop["cards"] = [{"id": "a", "title": "A", "state": "done", "after": []}]
                loop["base_commit"] = _git(root, "rev-parse", "HEAD").decode().strip()
                branch = f"symphonai/{loop['slug']}"
                _git(root, "branch", branch)
                loop["branch"] = branch
                store.save(loop)
                integration_admin = sessions / "loops" / loop["id"] / "integration"
                create_worktree(root, integration_admin, start=branch)
                branches.append(branch)
                integration = run._loop_integration_root(loop["id"])
                (integration / "a.py").write_text(f"{action}\n", encoding="utf-8")
                _git(integration, "add", "a.py")
                _git(integration, "commit", "--quiet", "-m", "a: A")
                loop = store.load(loop["id"])
                loop["summary"] = {"commits": [{"card": "a"}]}
                store.save(loop)
                person_branch = _git(root, "branch", "--show-current")
                (root / "uncommitted.txt").write_text("person work\n", encoding="utf-8")
                person_status = _git(root, "status", "--porcelain")
                status, reply = _http(host, "POST", "/loop/commit", {"loop_id": loop["id"], "action": action})
                result = reply.get("loop", {})
                if status != 200:
                    fail(f"loop commit route refused {action}: {status}, {reply!r}")
                if action == "approve":
                    if result["state"] != "done" or integration.exists() or branch not in _git(root, "branch", "--list").decode():
                        fail(f"approval did not keep the branch and remove its worktree: {result!r}")
                elif result["state"] != "cancelled" or branch in _git(root, "branch", "--list").decode():
                    fail(f"decline did not delete the loop branch: {result!r}")
                if _git(root, "branch", "--show-current") != person_branch or _git(root, "status", "--porcelain") != person_status:
                    fail("checkpoint action changed the person's branch or working tree")
        finally:
            host.close()


@check("loops.cancel_while_running")
def cancel_while_running() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        cards = [
            {"id": card_id, "title": card_id, "lane": lane, "state": "running", "session_id": f"session-{card_id}"}
            for lane, card_id in enumerate(("a", "b"), 1)
        ]
        run, store, loop = _lane_loop(root, sessions, cards)
        workers = []
        worktrees = []
        host = None
        try:
            loop = run.approve_loop_plan(loop["id"])
            for card in cards:
                session_id = card["session_id"]
                _loop_session(sessions, root, session_id, {"kind": "implement", "loop_id": loop["id"], "card_id": card["id"]})
                path = sessions / session_id / "worktree"
                create_worktree(root, path, start=loop["branch"])
                worktrees.append(path)
                stopped = threading.Event()
                thread = threading.Thread(target=stopped.wait, daemon=True)
                thread.start()
                workers.append((thread, stopped))
                run._active_by_session[session_id] = type("Active", (), {
                    "cancel": type("Cancel", (), {"cancel": stopped.set})(), "thread": thread,
                })()
            branch = loop["branch"]
            host = HostServer(
                FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]),
                runtime=ProjectRuntime(root, run),
            )
            host.start()
            status, reply = _http(host, "POST", "/loop/cancel", {"loop_id": loop["id"]})
            result = reply.get("loop", {})
            if status != 200:
                fail(f"running loop cancellation route refused the request: {status}, {reply!r}")
            if result["state"] != "cancelled" or _git(root, "branch", "--list").decode().find(branch) >= 0:
                fail(f"running loop cancellation did not delete its branch: {result!r}")
            if any(thread.is_alive() for thread, _event in workers) or any(path.exists() for path in worktrees):
                fail("cancellation left an active card conversation or worktree")
            if (sessions / "loops" / loop["id"] / "integration").exists():
                fail("cancellation left the integration worktree")
        finally:
            run._active_by_session.clear()
            if host is None:
                run.close()
            else:
                host.close()


@check("loops.resume_after_restart")
def resume_after_restart() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        card = {"id": "a", "title": "A", "lane": 1, "state": "failed", "session_id": "old-session"}
        run, store, loop = _lane_loop(root, sessions, [card])
        old_worktree = sessions / "old-session" / "worktree"
        started = []
        run.start_spec_run = lambda spec: (started.append(spec) or ("run", "retry-session"))
        host = None
        try:
            loop = run.approve_loop_plan(loop["id"])
            # Simulate a prior failed run after approval and preserve its worktree for cleanup.
            create_worktree(root, old_worktree, start=loop["branch"])
            loop = store.load(loop["id"])
            loop["state"] = "needs_attention"
            loop["reason"] = "card a could not finish: blocked"
            loop["cards"][0].update(state="failed", session_id="old-session")
            store.save(loop)
            host = HostServer(
                FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]),
                runtime=ProjectRuntime(root, run),
            )
            host.start()
            status, reply = _http(host, "POST", "/loop/resume", {"loop_id": loop["id"]})
            result = reply.get("loop", {})
            if status != 200:
                fail(f"loop resume route refused the failed card: {status}, {reply!r}")
            if result["state"] != "running" or result["cards"][0]["state"] != "running" or len(started) != 1:
                fail(f"resume did not start the failed card again: {result!r}, {started!r}")
            if old_worktree.exists():
                fail("resume left the failed card's old worktree")
            result["cards"][0]["state"] = "done"
            store.save(result)
            run._advance_loop(loop["id"])
            finished = _wait_state(store, loop["id"], "awaiting_commit")
            if finished.get("summary", {}).get("commits") != []:
                fail(f"a done card without a commit was included in the summary: {finished.get('summary')!r}")
        finally:
            if host is None:
                run.close()
            else:
                host.close()


@check("loops.card_details")
def card_details() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        cards = [
            {"id": "a", "title": "A", "lane": 1, "state": "done", "session_id": "done-session"},
            {"id": "b", "title": "B", "lane": 1, "state": "running", "session_id": "running-session"},
            {"id": "c", "title": "C", "lane": 1, "state": "planned", "session_id": None},
            {"id": "d", "title": "D", "lane": 1, "state": "failed", "session_id": "failed-session"},
            {"id": "e", "title": "E", "lane": 1, "state": "review", "session_id": "review-card-session"},
            {"id": "aF", "title": "Follow-up", "lane": 1, "state": "done", "follow_up_of": "a", "after": ["a"]},
        ]
        run, store, loop = _lane_loop(root, sessions, cards)
        host = None
        try:
            branch = "symphonai/card-details"
            base_commit = _git(root, "rev-parse", "HEAD").decode().strip()
            _git(root, "branch", branch)
            integration_admin = sessions / "loops" / loop["id"] / "integration"
            integration = create_worktree(root, integration_admin, start=branch)
            _git(integration, "switch", branch)
            (integration / "a.py").write_text("value = 2\n", encoding="utf-8")
            _git(integration, "add", "a.py")
            _git(integration, "commit", "--quiet", "-m", "a: A")
            commit = _git(integration, "rev-parse", "HEAD").decode().strip()
            spec = store.path(loop["id"], "specs/a.md")
            report = store.path(loop["id"], "reports/a-report.md")
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("Implementation passed.\n", encoding="utf-8")
            card_worktree = create_worktree(root, sessions / "running-session" / "worktree", start=branch)
            (card_worktree / "working.py").write_text("current change\n", encoding="utf-8")
            failed_worktree = create_worktree(root, sessions / "failed-session" / "worktree", start=branch)
            (failed_worktree / "failed.py").write_text("failed change\n", encoding="utf-8")
            review_worktree = create_worktree(root, sessions / "review-card-session" / "worktree", start=branch)
            (review_worktree / "review.py").write_text("review change\n", encoding="utf-8")
            _loop_session(sessions, root, "done-session", {
                "kind": "implement", "loop_id": loop["id"], "card_id": "a",
                "spec": f".symphonai/loops/{loop['id']}/specs/a.md",
            })
            _loop_session(sessions, root, "review-session", {"kind": "review", "of": "done-session"})
            source = SessionStore.open(sessions, "done-session")
            try:
                meta = source.read_meta()
                meta["review"] = {"session_id": "review-session", "verdict": "passed", "follow_ups": []}
                source.write_meta(meta)
            finally:
                source.close()
            review_store = SessionStore(sessions, "review-session", repo_root=root)
            try:
                writer = review_store.writer_for("review-agent", is_root=True)
                writer.append("run_started", run_id="review-session", agent_id="review-agent", turn_id=None, data={"parent_run_id": None})
                from symphonai_api.serialization import message_to_json
                writer.append("message", run_id="review-session", agent_id="review-agent", turn_id=None,
                              data=message_to_json(Message(Role.ASSISTANT, "Reviewed the change.\nVerdict: pass")))
                writer.append("run_finished", run_id="review-session", agent_id="review-agent", turn_id=None,
                              data={"stopped_reason": "final_response"})
            finally:
                review_store.close()
            loop.update(state="needs_attention", branch=branch, base_commit=base_commit)
            loop["cards"][0].update(commit=commit, spec="specs/a.md", report="reports/a-report.md")
            loop["cards"][1].update(spec="specs/b.md", report="reports/b-report.md")
            loop["cards"][2].update(spec="specs/c.md", report="reports/c-report.md")
            store.save(loop)

            host = HostServer(
                FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]),
                runtime=ProjectRuntime(root, run),
            )
            host.start()
            status, done = _http(host, "GET", f"/loop/card?loop={loop['id']}&card=a")
            if status != 200 or not done.get("spec", "").startswith("# a\n") or done.get("report") != "Implementation passed.\n":
                fail(f"done card detail omitted its spec or report: {status}, {done!r}")
            if ("+value = 2" not in done.get("diff", "") or done.get("review", {}).get("verdict") != "passed"
                    or done.get("review", {}).get("text", "").find("Verdict: pass") < 0
                    or done.get("review", {}).get("follow_ups") != ["aF"]
                    or done.get("session_id") != "done-session"):
                fail(f"done card detail omitted its commit diff or review: {done!r}")

            status, running = _http(host, "GET", f"/loop/card?loop={loop['id']}&card=b")
            if status != 200 or "+current change" not in running.get("diff", "") or running.get("report") or running.get("review", {}).get("text"):
                fail(f"running card detail did not return only its current patch: {status}, {running!r}")
            status, reviewing = _http(host, "GET", f"/loop/card?loop={loop['id']}&card=e")
            if status != 200 or "+review change" not in reviewing.get("diff", ""):
                fail(f"review card detail omitted its worktree patch: {status}, {reviewing!r}")
            status, failed = _http(host, "GET", f"/loop/card?loop={loop['id']}&card=d")
            if status != 200 or "+failed change" not in failed.get("diff", ""):
                fail(f"failed card detail omitted its worktree patch: {status}, {failed!r}")
            status, planned = _http(host, "GET", f"/loop/card?loop={loop['id']}&card=c")
            if status != 200 or planned.get("diff") or planned.get("report") or planned.get("review", {}).get("text"):
                fail(f"planned card detail was not empty: {status}, {planned!r}")
            status, _unknown = _http(host, "GET", f"/loop/card?loop={loop['id']}&card=../x")
            if status != 404:
                fail(f"malformed card id did not return 404: {status}")
            status, _unknown = _http(host, "GET", "/loop/card?loop=loop_ffffffffffff&card=a")
            if status != 404:
                fail(f"unknown loop id did not return 404: {status}")
        finally:
            if host is not None:
                host.close()
            else:
                run.close()

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        card = {"id": "a", "title": "A", "lane": 1, "state": "running", "session_id": "gone-session"}
        run, store, loop = _lane_loop(root, sessions, [card])
        host = None
        try:
            loop = run.approve_loop_plan(loop["id"])
            # Restore a durable running state as if the host process stopped here.
            persisted = store.load(loop["id"])
            persisted["cards"][0].update(state="running", session_id="gone-session")
            store.save(persisted)
            stale = create_worktree(root, sessions / "gone-session" / "worktree", start=loop["branch"])
            (stale / "stale.txt").write_text("old worker\n", encoding="utf-8")
            run.close()
            restarted = HostRun(
                FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]), EventBroker(),
                sessions_root=sessions,
            )
            started = []
            restarted.start_spec_run = lambda spec: (started.append(spec) or ("run", "new-session"))
            host = HostServer(
                FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]),
                runtime=ProjectRuntime(root, restarted),
            )
            host.start()
            result = store.load(loop["id"])
            if result["cards"][0]["session_id"] != "new-session" or len(started) != 1 or stale.exists():
                fail(f"startup recovery did not restart the orphaned card: {result!r}, {started!r}")
            result["cards"][0]["state"] = "done"
            store.save(result)
            restarted._advance_loop(loop["id"])
            _wait_state(store, loop["id"], "awaiting_commit")
        finally:
            if host is None:
                run.close()
            else:
                host.close()

        non_repo = base / "not-a-repo"
        non_repo.mkdir()
        failed_store = LoopStore(non_repo)
        failed_loop = failed_store.create("Cannot branch")
        failed_loop["state"] = "awaiting_plan"
        failed_loop["cards"] = [{"id": "a", "title": "Card", "state": "planned", "after": []}]
        failed_store.save(failed_loop)
        failed_run = HostRun(
            FakeModelProvider(), PermissionPolicy(non_repo, allowed_write_scope=[non_repo]), EventBroker(),
            sessions_root=base / "failed-sessions",
        )
        try:
            try:
                failed_run.approve_loop_plan(failed_loop["id"])
            except ValueError:
                pass
            else:
                fail("approval succeeded outside a git repository")
            persisted = failed_store.load(failed_loop["id"])
            if persisted["state"] != "awaiting_plan" or persisted["branch"] is not None:
                fail(f"failed branch creation changed the loop: {persisted!r}")
            if (base / "failed-sessions" / "loops" / failed_loop["id"] / "integration").exists():
                fail("failed approval left an integration worktree")
        finally:
            failed_run.close()


@check("loops.integrate_commits_a_card")
def integrate_commits_a_card() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "project"
        _git_repo(root)
        sessions = base / "sessions"
        store = LoopStore(root)
        loop = store.create("Update value")
        loop["state"] = "awaiting_plan"
        card = {"id": "a", "title": "Update the value", "state": "planned", "after": []}
        store.save({**loop, "cards": [card]})
        loop = store.load(loop["id"])
        run = HostRun(
            FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]), EventBroker(),
            sessions_root=sessions,
        )
        card_a_admin = sessions / "cards" / "a"
        stale_admin = sessions / "cards" / "stale"
        second_admin = sessions / "cards" / "second"
        integration_admin = sessions / "loops" / loop["id"] / "integration"
        try:
            loop = run.approve_loop_plan(loop["id"])
            main_status = _git(root, "status", "--porcelain")
            main_branch = _git(root, "branch", "--show-current")
            card_a = create_worktree(root, card_a_admin, start=loop["branch"])
            stale = create_worktree(root, stale_admin, start=loop["branch"])
            (card_a / "a.py").write_text("value = 2\n", encoding="utf-8")
            (stale / "a.py").write_text("value = 3\n", encoding="utf-8")
            commit_id = run.integrate(loop, card, card_a)
            integration = run._loop_integration_root(loop["id"])
            subject = _git(integration, "show", "-s", "--format=%s", "HEAD").decode().strip()
            changed = _git(integration, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").decode().splitlines()
            if not commit_id or subject != "a: Update the value" or changed != ["a.py"]:
                fail(f"card integration did not create the exact commit: {commit_id!r}, {subject!r}, {changed!r}")
            if (integration / "a.py").read_text(encoding="utf-8") != "value = 2\n":
                fail("integration commit did not contain the card's change")
            second = create_worktree(root, second_admin, start=loop["branch"])
            if (second / "a.py").read_text(encoding="utf-8") != "value = 2\n":
                fail("a later card worktree did not start from the integrated tip")
            before_conflict_head = _git(integration, "rev-parse", "HEAD")
            before_conflict_status = _git(integration, "status", "--porcelain")
            try:
                run.integrate(loop, {"id": "b", "title": "Conflicting edit"}, stale)
            except WorktreeApplyConflict:
                pass
            else:
                fail("conflicting card patch was integrated")
            if (_git(integration, "rev-parse", "HEAD") != before_conflict_head
                    or _git(integration, "status", "--porcelain") != before_conflict_status
                    or (integration / "a.py").read_text(encoding="utf-8") != "value = 2\n"):
                fail("a conflicting card patch changed the integration worktree")
            if _git(root, "status", "--porcelain") != main_status or _git(root, "branch", "--show-current") != main_branch:
                fail("card integration changed the person's status or current branch")
            if run.integrate(loop, {"id": "empty", "title": "No changes"}, second) != "":
                fail("empty card worktree created an integration commit")
        finally:
            for path in (card_a_admin, stale_admin, second_admin, integration_admin):
                if path.exists():
                    try:
                        remove_worktree(root, path)
                    except Exception:
                        pass
            run.close()


@check("loops.integrate_ignores_dependency_links")
def integrate_ignores_dependency_links() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, sessions = base / "project", base / "sessions"
        _git_repo(root)
        (root / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
        (root / "node_modules").mkdir()
        (root / "node_modules" / "package.json").write_text("{}\n", encoding="utf-8")
        _git(root, "add", ".gitignore")
        _git(root, "commit", "--quiet", "-m", "ignore node_modules")
        store = LoopStore(root)
        loop = store.create("Integrate through linked dependencies")
        loop.update(
            state="awaiting_plan",
            cards=[{
                "id": "a", "title": "Add a file", "lane": 1, "after": [], "state": "review",
            }],
        )
        store.save(loop)
        run = HostRun(
            FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]), EventBroker(),
            sessions_root=sessions,
        )
        card_admin, second_admin = sessions / "cards" / "a", sessions / "cards" / "b"
        integration_admin = sessions / "loops" / loop["id"] / "integration"
        try:
            loop = run.approve_loop_plan(loop["id"])
            integration = run._loop_integration_root(loop["id"])
            if not (integration / "node_modules").is_symlink():
                fail("integration worktree did not link the project node_modules folder")
            card_worktree = create_worktree(root, card_admin, start=loop["branch"])
            (card_worktree / "new.py").write_text("value = 2\n", encoding="utf-8")
            commit = run.integrate(loop, loop["cards"][0], card_worktree)
            changed = _git(integration, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).decode().splitlines()
            if changed != ["new.py"] or (integration / "new.py").read_text(encoding="utf-8") != "value = 2\n":
                fail(f"integration commit contained unexpected paths: {changed!r}")
            if _git(integration, "ls-tree", "-r", "--name-only", commit).decode().splitlines().count("node_modules"):
                fail("integration commit tracked the linked node_modules folder")

            second = create_worktree(root, second_admin, start=loop["branch"])
            (second / "other.py").write_text("value = 3\n", encoding="utf-8")
            (integration / "unrelated.txt").write_text("uncommitted\n", encoding="utf-8")
            before_head = _git(integration, "rev-parse", "HEAD")
            try:
                run.integrate(loop, {"id": "b", "title": "Add another file"}, second)
            except RuntimeError as exc:
                if str(exc) != "integration worktree has uncommitted changes":
                    raise
            else:
                fail("an unrelated untracked integration file did not stop the card")
            if _git(integration, "rev-parse", "HEAD") != before_head or (integration / "other.py").exists():
                fail("rejected integration changed the branch")
        finally:
            for path in (card_admin, second_admin, integration_admin):
                if path.exists():
                    try:
                        remove_worktree(root, path)
                    except Exception:
                        pass
            run.close()
