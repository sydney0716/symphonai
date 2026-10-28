"""Loopback transport checks for the SymphonAI host HTTP boundary."""

from __future__ import annotations

import contextlib
import base64
import hashlib
import http.client
import io
import inspect
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit
from unittest import mock

import symphonai_api.agent_loop as agent_loop
import symphonai_api.leader as leader_module
import symphonai_api.mcp as mcp_module
import symphonai_host.__main__ as host_main
import symphonai_host.headless as headless_module
import symphonai_host.protocol as protocol_module
import symphonai_host.run as host_run_module
import symphonai_host.server as host_server_module
import symphonai_host.goal as goal_module
import symphonai_host.plan as plan_module
import symphonai_host.questions as questions_module
from symphonai_api.agent_run import RunNode
from symphonai_api.agent_file import load_agent_file
from symphonai_api.agent_spec import ModelSelector
from symphonai_api.config import ConfigError, ResolvedConfig
from symphonai_api.events import (
    AssistantTextDelta,
    CompactionApplied,
    JobChanged,
    ModelFallback,
    RunFailed,
    RunFinished,
    RunStarted,
    SubagentSpawned,
)
from symphonai_api.cost import ModelPrice, PriceTable, UsageTotals
from symphonai_api.extensions import Extensions, load_extensions
from symphonai_api.hooks import HookRunner
from symphonai_api.identity import RunRef
from symphonai_api.instructions import MAX_INSTRUCTION_FILE_CHARS, load_instructions
from symphonai_api.mcp import McpServerSpec
from symphonai_api.mcp_pool import McpPool
from symphonai_api.models import DocumentBlock, ImageBlock, Message, ModelResponse, Role, TextBlock, ToolCall, ToolResult, Usage
from symphonai_api.permissions import DenialReason, PermissionDecision, PermissionPolicy
from symphonai_api.provider_catalog import CATALOG, catalog_keys
from symphonai_host.files import repository_files
from symphonai_api.providers.base import ModelProvider, ProviderError, ProviderUnavailableError
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.retry import read_with_retry
from symphonai_api.roles import LEADER_BASE_PROMPT, SUBAGENT_BASE_PROMPT
from symphonai_api.runner import merge_tool_registry, standard_tool_registry
from symphonai_api.session import SessionStore, load_run, load_run_for_resume
from symphonai_api.streaming import StreamCompleted, TextDelta, open_stream_with_retry
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.filesystem import ReadLedger, WriteFileTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_api.tools.shell import RunShellTool, shell_environment
import symphonai_api.tools.shell as shell_module
from symphonai_api.tools.web_fetch import WebFetchTool
from symphonai_api.worktree import create_worktree, worktree_diff
from symphonai_host.broker import EventBroker
from symphonai_host.goal import GoalChanged
from symphonai_host.loops import LoopStore
from symphonai_host.plan import PlanChanged
from symphonai_host.protocol import ApprovalRequested, FilesChanged, decode_event, decode_frame
from symphonai_host.run import HostRun, RunActiveError
from symphonai_host.projects import ProjectRuntime
from symphonai_host.projects import choose_start_root
from symphonai_host.server import HostServer
from symphonai_host.spec_run import patch_digest, review_verdict, spec_validation
from scripts.checks.harness import CheckFailed, check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]


@check("host_server.close_is_prompt")
def check_host_server_close_is_prompt() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = HostServer(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            PermissionPolicy(repo_root=root, allowed_write_scope=[root]),
            sessions_root=root / "sessions",
        )
        host.start()
        started = time.monotonic()
        host.close()
        elapsed = time.monotonic() - started
        if elapsed >= 0.2:
            fail(f"host close took {elapsed:.3f}s")


@check("host_server.files_changed_event")
def check_files_changed_event() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
        (root / "tracked.txt").write_text("initial\n", encoding="utf-8")
        (root / ".gitignore").write_text("ignored.log\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.txt", ".gitignore"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
        broker = EventBroker()
        host = SimpleNamespace(
            _watch_stop=threading.Event(),
            _event_watch_subscription=None,
            _runtimes_lock=threading.RLock(),
            _current_root=root,
            broker=broker,
        )
        subscription = None
        watcher = threading.Thread(target=HostServer._watch_files, args=(host,), daemon=True)
        try:
            watcher.start()
            (root / "tracked.txt").write_text("changed with no app\n", encoding="utf-8")
            time.sleep(0.4)
            subscription = host.broker.subscribe()
            time.sleep(0.4)
            if subscription.get(timeout=0.05) is not None:
                fail("file watcher published changes made before an app connected")

            (root / "tracked.txt").write_text("first connected edit\n", encoding="utf-8")
            event = subscription.get(timeout=2.0)
            if not isinstance(event, FilesChanged) or event.paths != ["tracked.txt"] or event.truncated:
                fail(f"tracked file change was not published correctly: {event!r}")

            for value in ("burst one\n", "burst two\n", "burst three\n"):
                (root / "tracked.txt").write_text(value, encoding="utf-8")
                time.sleep(0.1)
            event = subscription.get(timeout=2.0)
            if not isinstance(event, FilesChanged) or event.paths != ["tracked.txt"]:
                fail(f"settled burst was not coalesced into one file event: {event!r}")
            if subscription.get(timeout=1.2) is not None:
                fail("three quick writes produced more than one file event")

            (root / "ignored.log").write_text("ignored\n", encoding="utf-8")
            (root / ".git" / "watch-check").write_text("git metadata\n", encoding="utf-8")
            if subscription.get(timeout=1.4) is not None:
                fail("ignored or .git changes produced a FilesChanged event")
        finally:
            if subscription is not None:
                subscription.close()
            host._watch_stop.set()
            broker.close()
            watcher.join(timeout=1)


@check("host_server.spec_validation_commands")
def check_loop_spec_validation() -> None:
    commands = ["python3 -m unittest", ".venv/bin/python scripts/check.py"]
    for heading in ("# Validation", "## Validation", "### Validation", "## vAlIdAtIoN"):
        text = f"# Card spec\n\n{heading}\n\n```sh\n" + "\n".join(commands) + "\n```\n\n## Report\n"
        if spec_validation(text) != commands:
            fail(f"loop spec validation commands were not parsed under {heading!r}")
    nested = (
        "## Validation\n\n### Notes\n\nThis subsection remains inside Validation.\n\n"
        "```sh\npython3 -m unittest\n```\n\n## Report\n"
        "```sh\nnot-a-validation-command\n```\n"
    )
    if spec_validation(nested) != ["python3 -m unittest"]:
        fail("a lower-level subsection ended Validation or Report commands were included")
    if spec_validation("## Validation\nProse only\n"):
        fail("validation prose without a command fence was accepted")


@check("host_server.spec_run_goal_checks_in_its_worktree")
def check_spec_run_goal_checks_in_its_worktree() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
        (root / "a.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "a.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
        run = HostRun(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            PermissionPolicy(repo_root=root, allowed_write_scope=[root]),
            EventBroker(), sessions_root=root / "sessions",
        )
        try:
            session_id, _ = run.start_spec_run({
                "path": "specs/39/39z-check.md", "report": "specs/report/39/39z-check-report.md",
                "text": "Implement the requested change.", "validation": ["pwd"],
            })
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                goal = run._goal_for_session(session_id)
                if goal and goal["phase"] == "complete":
                    break
                time.sleep(0.01)
            goal = run._goal_for_session(session_id)
            expected = str(root / "sessions" / session_id / "worktree")
            if not goal or goal["phase"] != "complete" or expected not in goal["last_check"]["output"]:
                fail(f"spec validation did not run in its worktree: {goal!r}")
        finally:
            run.close()


@check("host_server.spec_goal_check_sandbox_policy")
def check_spec_goal_check_sandbox_policy() -> None:
    from threading import Event

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "worktree"
        root.mkdir()
        policy = PermissionPolicy(repo_root=root, mode="allow")

        def execute(argv, *, sandboxed, run_policy=policy):
            return goal_module.run_check(
                tuple(argv), root, cancelled=Event(), set_process=lambda _proc: None,
                timeout=5, policy=run_policy, sandboxed=sandboxed,
            )

        inside = root / "inside-marker"
        inside_result = execute([
            sys.executable, "-c", "from pathlib import Path; Path('inside-marker').touch()",
        ], sandboxed=True)
        if not inside_result["ok"] or not inside.exists():
            fail(f"sandboxed spec check could not write in its worktree: {inside_result!r}")

        outside = root.parent / "goal-marker"
        user_goal_result = goal_module.run_check(
            (sys.executable, "-c", f"from pathlib import Path; Path({str(outside)!r}).touch()"),
            root, cancelled=Event(), set_process=lambda _proc: None, timeout=5,
        )
        if not user_goal_result["ok"] or not outside.exists():
            fail(f"user-supplied /goal check was not left unsandboxed: {user_goal_result!r}")

        strict = replace(policy, shell_sandbox=True)
        with mock.patch.object(shell_module, "_sandbox_available", return_value=False):
            unavailable = execute([sys.executable, "-c", "pass"], sandboxed=True, run_policy=strict)
        if unavailable["ok"] or "sandbox requested but unavailable on this platform" not in unavailable["output"]:
            fail(f"strict sandbox request did not fail closed: {unavailable!r}")

        with mock.patch.object(shell_module.sys, "platform", "linux"):
            linux_result = execute([
                sys.executable, "-c", "from pathlib import Path; Path('linux-marker').touch()",
            ], sandboxed=True)
        if not linux_result["ok"] or not (root / "linux-marker").exists():
            fail(f"Linux spec validation did not run unconfined: {linux_result!r}")

        if shell_module._sandbox_available():
            home_marker = Path.home() / f".symphonai-spec-check-{os.getpid()}"
            try:
                outside_result = execute([
                    sys.executable, "-c", f"from pathlib import Path; Path({str(home_marker)!r}).touch()",
                ], sandboxed=True)
                if outside_result["ok"] or home_marker.exists():
                    fail(f"sandboxed spec check wrote outside its worktree: {outside_result!r}")
            finally:
                home_marker.unlink(missing_ok=True)
        else:
            print("SKIP: sandbox-exec unavailable; macOS confinement check not run")


@check("host_server.spec_run_starts_with_full_spec_and_keeps_title")
def check_spec_run_starts_with_full_spec_and_keeps_title() -> None:
    class RecordingProvider(FakeModelProvider):
        def __init__(self):
            super().__init__([ModelResponse(Message(Role.ASSISTANT, "done"))])
            self.requests = []

        def create_response(self, request, *, cancel=None):
            self.requests.append(request)
            return super().create_response(request, cancel=cancel)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
        (root / "seed.txt").write_text("seed\n", encoding="utf-8")
        subprocess.run(["git", "add", "seed.txt"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
        provider = RecordingProvider()
        host = HostServer(
            provider, PermissionPolicy(repo_root=root, allowed_write_scope=[root]),
            sessions_root=root / "sessions",
        )
        host.start()
        spec_text = "# 39bF — exact implementer prompt\n\nImplement this exact text."
        spec_path = "specs/39/39bF-test.md"
        report_path = "specs/report/39/39bF-test-report.md"
        try:
            session_id, _ = host.run.start_spec_run({
                "path": spec_path, "report": report_path, "text": spec_text, "validation": [],
            })
            _wait_until(lambda: bool(provider.requests), "spec provider was not called")
            prompt = next(
                message.text for message in provider.requests[0].messages
                if message.role is Role.USER
            )
            if prompt != f"{spec_text}\n\nWrite your report at {report_path}.":
                fail(f"first spec-run message was not the complete spec: {prompt!r}")
            store = SessionStore.open(root / "sessions", session_id)
            try:
                meta = store.read_meta()
            finally:
                store.close()
            expected_title = f"Run {Path(spec_path).name}"
            if meta.get("title") != expected_title:
                fail(f"spec-run title changed: {meta.get('title')!r}")
            connection, response = _request(host, "GET", "/sessions", headers=_headers(host))
            sessions = json.loads(response.read())
            connection.close()
            listed = next((item for item in sessions if item.get("run_id") == session_id), None)
            if not listed or listed.get("title") != expected_title:
                fail(f"GET /sessions did not retain the spec-run title: {listed!r}")
        finally:
            host.close()


@check("host_server.spec_roles_use_the_routine_model")
def check_spec_roles_use_the_routine_model() -> None:
    class RecordingProvider(FakeModelProvider):
        def __init__(self, name, model, responses=None, routine_model=None) -> None:
            super().__init__(responses or [ModelResponse(Message(Role.ASSISTANT, "done"))])
            self.provider_name = name
            self.model = model
            self.DEFAULT_MODEL = model
            self.ROUTINE_MODEL = routine_model
            self.requests: list[tuple[str | None, str | None]] = []

        @property
        def name(self):
            return self.provider_name

        def create_response(self, request, *, cancel=None):
            self.requests.append((request.model, request.effort))
            return super().create_response(request, cancel=cancel)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = RecordingProvider("anthropic", "claude-opus-5-5", routine_model="claude-sonnet-5-5")
        run = HostRun(
            provider, PermissionPolicy(root), EventBroker(),
            sessions_root=root / "sessions", memory_extraction_enabled=False,
        )
        try:
            for role, expected in (
                ("implementer", "claude-sonnet-5-5"),
                ("reviewer", "claude-sonnet-5-5"),
                ("planner", "claude-opus-5-5"),
            ):
                session = SessionStore(
                    root / "sessions", f"role-{role}", repo_root=root,
                )
                meta = session.read_meta()
                meta["spec_run"] = {"role": role}
                session.write_meta(meta)
                try:
                    leader = run._new_leader(session)
                    leader.run(f"run {role}")
                finally:
                    session.close()
                if provider.requests[-1][0] != expected:
                    fail(
                        f"spec {role} used {provider.requests[-1][0]!r}; "
                        f"expected {expected!r}"
                    )
        finally:
            run.close()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        home = root / "home"
        (home / ".symphonai").mkdir(parents=True)
        (home / ".symphonai" / "config.toml").write_text(
            '[roles.implementer]\nprovider = "openai"\nmodel = "gpt-6-luna"\neffort = "high"\n'
            '[roles.worker]\nprovider = "openai"\nmodel = "gpt-6-luna"\neffort = "low"\n',
            encoding="utf-8",
        )
        anthropic = RecordingProvider("anthropic", "claude-opus-5-5")
        openai = RecordingProvider(
            "openai",
            "gpt-6-luna",
            [
                ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
                    "dispatch-worker", "dispatch_subagent", {
                        "subagent_name": "worker", "task": "inspect the implementation stage",
                    },
                )])),
                ModelResponse(Message(Role.ASSISTANT, "worker done")),
                ModelResponse(Message(Role.ASSISTANT, "implementer done")),
            ],
        )

        def provider_factory(name, _model, _base_url):
            return openai if name == "openai" else anthropic if name == "anthropic" else None

        extensions = load_extensions(repo_root=root, home=home)
        run = HostRun(
            anthropic, PermissionPolicy(root), EventBroker(),
            model="claude-picked", provider_factory=provider_factory,
            sessions_root=root / "sessions", memory_extraction_enabled=False,
            extensions=extensions,
        )
        try:
            session = SessionStore(root / "sessions", "role-implementer", repo_root=root)
            meta = session.read_meta()
            meta["spec_run"] = {"role": "implementer", "kind": "implement"}
            session.write_meta(meta)
            try:
                run._new_leader(session).run("run implementer")
            finally:
                session.close()
            if openai.requests != [
                ("gpt-6-luna", "high"),
                ("gpt-6-luna", "low"),
                ("gpt-6-luna", "high"),
            ]:
                fail(f"loop implementer and dispatched worker used the wrong provider/model/effort: {openai.requests!r}")

            for role in ("planner", "reviewer"):
                session = SessionStore(root / "sessions", f"role-picked-{role}", repo_root=root)
                meta = session.read_meta()
                meta["spec_run"] = {"role": role, "kind": role}
                session.write_meta(meta)
                try:
                    run._new_leader(session).run(f"run {role}")
                finally:
                    session.close()
                if anthropic.requests[-1][0] != "claude-picked":
                    fail(f"unconfigured {role} did not use the picked model: {anthropic.requests[-1]!r}")
        finally:
            run.close()


@check("host_server.agent_history_route")
def check_agent_history_route() -> None:
    def first_user_message(messages):
        return next((message for message in messages if message["role"] == "user"), None)

    class BlockingProvider(FakeModelProvider):
        def __init__(self) -> None:
            super().__init__([
                ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
                    "dispatch", "dispatch_subagent", {
                        "subagent_name": "explorer",
                        "task": "inspect live explorer history",
                    },
                )])),
                ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
                    "files", "list_files", {"path": "."},
                )])),
                ModelResponse(Message(Role.ASSISTANT, "explorer finished")),
                ModelResponse(Message(Role.ASSISTANT, "leader finished")),
            ])
            self.explorer_started = threading.Event()
            self.release_explorer = threading.Event()

        def create_response(self, request, *, cancel=None):
            latest_user = next(
                (message.text for message in reversed(request.messages) if message.role is Role.USER),
                "",
            )
            if latest_user == "inspect live explorer history" and not self.explorer_started.is_set():
                self.explorer_started.set()
                if not self.release_explorer.wait(5):
                    raise RuntimeError("explorer test gate timed out")
            return super().create_response(request, cancel=cancel)

    provider = BlockingProvider()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        try:
            connection, response = _request(
                host, "POST", "/prompt", body={"prompt": "dispatch explorer now"},
                headers=_headers(host),
            )
            response.read()
            connection.close()
            if not provider.explorer_started.wait(3):
                fail("explorer did not enter its blocked model request")
            conversation = host.run.conversation_stats()
            if conversation is None:
                fail("running conversation disappeared")
            explorer = next(
                (agent for agent in conversation["agents"] if agent["name"] == "explorer"),
                None,
            )
            if explorer is None:
                fail(f"running explorer was not listed: {conversation!r}")
            session_id = conversation["session_id"]
            agent_id = explorer["agent_id"]
            subscription = host.broker.subscribe()
            try:
                path = "/session/agent?" + urlencode({
                    "session_id": session_id, "agent_id": agent_id,
                })
                connection, response = _request(host, "GET", path, headers=_headers(host))
                partial = json.loads(response.read())
                connection.close()
                task_message = first_user_message(partial.get("messages", []))
                if response.status != 200 or task_message is None or task_message["text"] != "inspect live explorer history":
                    fail(f"running history did not return the explorer task: {partial!r}")
                if subscription.get(timeout=0.02) is not None:
                    fail("agent history GET published an event")
            finally:
                subscription.close()
                provider.release_explorer.set()
            _wait_until(lambda: not host.run.active, "explorer conversation did not finish")
            path = "/session/agent?" + urlencode({
                "session_id": session_id, "agent_id": agent_id,
            })
            connection, response = _request(host, "GET", path, headers=_headers(host))
            history = json.loads(response.read())
            connection.close()
            messages = history.get("messages", [])
            task_message = first_user_message(messages)
            if response.status != 200 or task_message is None or task_message["text"] != "inspect live explorer history":
                fail(f"closed history did not start with the explorer task: {history!r}")
            tool_calls = [call for message in messages for call in message["tool_calls"]]
            if not any(call["name"] == "list_files" and call.get("output") for call in tool_calls):
                fail(f"explorer tool call/result was not replayed: {history!r}")
            if not any(message["text"] == "explorer finished" for message in messages):
                fail(f"explorer final answer was not replayed: {history!r}")
            if any(message["text"] == "leader finished" for message in messages):
                fail("leader transcript leaked into the explorer history")
            for query in (
                {"session_id": session_id, "agent_id": "../run"},
                {"session_id": session_id, "agent_id": "agent_00000000000000000000000000000000"},
                {"session_id": "run_00000000000000000000000000000000", "agent_id": agent_id},
            ):
                connection, response = _request(
                    host, "GET", "/session/agent?" + urlencode(query), headers=_headers(host),
                )
                response.read()
                connection.close()
                if response.status != 404:
                    fail(f"invalid/unknown agent history query was not 404: {query!r}")
        finally:
            provider.release_explorer.set()
            host.close()


@check("host_server.spec_review_pass_verdict")
def check_spec_review_pass_verdict() -> None:
    if review_verdict("Review looks good.\nVerdict: pass") != ("passed", []):
        fail("final pass verdict was not recognized")


@check("host_server.spec_review_followup_verdict")
def check_spec_review_followup_verdict() -> None:
    expected = ("follow-ups", ["specs/39/39zF-fix.md", "specs/39/39zF2-more.md"])
    if review_verdict("Verdict: follow-ups: specs/39/39zF-fix.md, specs/39/39zF2-more.md") != expected:
        fail("follow-up verdict paths were not parsed in order")


@check("host_server.spec_review_requires_last_line")
def check_spec_review_requires_last_line() -> None:
    if review_verdict("Verdict: pass\nA later explanation") != ("no-verdict", []):
        fail("a verdict before the last non-empty line was accepted")


@check("host_server.spec_review_rejects_empty_followups")
def check_spec_review_rejects_empty_followups() -> None:
    if review_verdict("Verdict: follow-ups:") != ("no-verdict", []):
        fail("empty follow-up verdict was accepted")


@check("host_server.spec_review_ignores_nonfinal_text")
def check_spec_review_ignores_nonfinal_text() -> None:
    if review_verdict("Verdict: pass\n\n") != ("passed", []):
        fail("trailing blank lines changed a final verdict")


@check("host_server.spec_review_unknown_verdict")
def check_spec_review_unknown_verdict() -> None:
    if review_verdict("Verdict: maybe") != ("no-verdict", []):
        fail("unknown verdict was not treated as no-verdict")


@check("host_server.spec_review_missing_verdict")
def check_spec_review_missing_verdict() -> None:
    if review_verdict("Review is complete, with two points to consider.") != ("no-verdict", []):
        fail("review prose without a verdict was not rejected")


@check("host_server.spec_patch_digest_stable")
def check_spec_patch_digest_stable() -> None:
    if patch_digest("patch\n") != patch_digest("patch\n") or patch_digest("patch\n") == patch_digest("other\n"):
        fail("review baseline digest did not distinguish patch changes")


@check("host_server.spec_review_detects_tree_change")
def check_spec_review_detects_tree_change() -> None:
    from types import SimpleNamespace
    from symphonai_api.worktree import create_worktree, worktree_diff
    from symphonai_host.spec_run import patch_digest

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
        (root / "a.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "a.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
        sessions = root / "sessions"
        source_id, review_id = "spec-source", "spec-review"
        worktree = create_worktree(root, sessions / source_id / "worktree")
        (worktree / "a.py").write_text("value = 2\n", encoding="utf-8")
        baseline = worktree_diff(worktree)
        baseline_files = {"a.py": hashlib.sha256((worktree / "a.py").read_bytes()).hexdigest()}
        (worktree / "a.py").write_text("value = 3\n", encoding="utf-8")
        source_store = SessionStore(sessions, source_id, repo_root=root)
        source_meta = source_store.read_meta()
        source_meta["spec_run"] = {"kind": "implement", "spec": "specs/39/39b.md", "report": "specs/report/39/39b-report.md", "worktree": "worktree"}
        source_store.write_meta(source_meta)
        source_store.close()
        review_store = SessionStore(sessions, review_id, repo_root=root)
        review_meta = review_store.read_meta()
        review_meta["spec_run"] = {
            "kind": "review", "of": source_id, "worktree_path": str(worktree),
            "baseline": patch_digest(baseline.patch), "baseline_files": baseline_files,
        }
        review_store.write_meta(review_meta)
        review_store.close()
        run = HostRun(None, PermissionPolicy(repo_root=root), EventBroker(), sessions_root=sessions)
        try:
            run._finish_spec_review(review_id, SimpleNamespace(
                _chat_messages=[Message(Role.ASSISTANT, "Verdict: pass")],
            ), RunFinished(agent_id="reviewer", run_id=review_id, stopped_reason="final_response"))
            source_store = SessionStore.open(sessions, source_id)
            verdict = source_store.read_meta().get("review", {}).get("verdict")
            source_store.close()
            if verdict != "tree-changed":
                fail(f"review verdict did not prioritize a modified worktree: {verdict!r}")
        finally:
            run.close()


@check("host_server.model_fallback")
def check_model_fallback() -> None:
    class _Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def read(self) -> bytes:
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _HttpProvider(ModelProvider):
        model = "model-a"

        def __init__(self) -> None:
            self.requests: list[str | None] = []

        @property
        def name(self) -> str:
            return "anthropic"

        @property
        def wire_format(self) -> int:
            return 2

        def create_response(self, request, *, cancel=None):
            self.requests.append(request.model)
            request_bytes = json.dumps({"model": request.model or self.model}).encode()
            raw = read_with_retry(
                urllib.request.Request("https://mock.invalid/messages", data=request_bytes),
                timeout=600,
                api_key="test-key",
                operation="mock provider",
                fallback_configured=bool(getattr(self, "_symphonai_fallback_models", ())),
            )
            return ModelResponse(Message(Role.ASSISTANT, raw.decode("utf-8")))

    def wait_terminal(subscription):
        events = []
        while True:
            event = subscription.get(timeout=0.05)
            if event is not None:
                events.append(event)
                if isinstance(event, (RunFinished, RunFailed)):
                    return events

    def run_prompt(root, provider, *, fallback, outcomes):
        source = root / ".symphonai" / "config.toml"
        if fallback:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text('[fallback]\nanthropic = ["model-b"]\n', encoding="utf-8")
        extensions = load_extensions(repo_root=root, home=root / "home")
        broker = EventBroker()
        subscription = broker.subscribe()
        host = HostRun(
            provider,
            PermissionPolicy(repo_root=root),
            broker,
            sessions_root=root / "sessions",
            extensions=extensions,
            memory_extraction_enabled=False,
        )
        try:
            host.start("first prompt")
            events = wait_terminal(subscription)
            if not isinstance(events[-1], outcomes):
                fail(f"host run did not reach expected terminal event: {events!r}")
            return host, events, subscription
        except Exception:
            host.close()
            subscription.close()
            raise

    sent_models = []

    def mocked_open(url_request, timeout):
        body = json.loads(url_request.data.decode("utf-8"))
        model = body["model"]
        sent_models.append(model)
        if model == "model-a" or always_unavailable:
            raise urllib.error.HTTPError(
                url_request.full_url, 529, "overloaded", {}, io.BytesIO(b"busy")
            )
        return _Response(b"ok")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = _HttpProvider()
        always_unavailable = False
        retry_clock = SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _delay: None)
        with mock.patch("symphonai_api.retry.urllib.request.urlopen", side_effect=mocked_open), mock.patch(
            "symphonai_api.retry.time", retry_clock
        ):
            host, events, subscription = run_prompt(
                root, provider, fallback=True, outcomes=(RunFinished,)
            )
            try:
                fallback_events = [event for event in events if isinstance(event, ModelFallback)]
                if len(fallback_events) != 1 or (
                    fallback_events[0].from_model,
                    fallback_events[0].to_model,
                    fallback_events[0].reason,
                ) != ("model-a", "model-b", "529"):
                    fail(f"fallback event was missing or incorrect: {fallback_events!r}")
                host.start("second prompt")
                second_events = wait_terminal(subscription)
                if not isinstance(second_events[-1], RunFinished):
                    fail(f"second host run did not finish: {second_events!r}")
                if sent_models != ["model-a"] * 3 + ["model-b"] + ["model-a"] * 3 + ["model-b"]:
                    fail(f"next turn did not start from model-a: {sent_models!r}")
                if provider.requests != [None, "model-b", None, "model-b"]:
                    fail(f"primary request model was changed from None: {provider.requests!r}")
            finally:
                host.close()
                subscription.close()

    for fallback, status in ((False, 529), (True, 529), (True, 400)):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = _HttpProvider()
            always_unavailable = True
            sent_models = []

            def fail_open(url_request, timeout):
                sent_models.append(json.loads(url_request.data.decode("utf-8"))["model"])
                raise urllib.error.HTTPError(
                    url_request.full_url, status, "provider error", {}, io.BytesIO(b"failure")
                )

            retry_clock = SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _delay: None)
            with mock.patch("symphonai_api.retry.urllib.request.urlopen", side_effect=fail_open), mock.patch(
                "symphonai_api.retry.time", retry_clock
            ):
                host, events, subscription = run_prompt(
                    root, provider, fallback=fallback, outcomes=(RunFailed,)
                )
                try:
                    if any(isinstance(event, ModelFallback) for event in events):
                        fail(f"HTTP {status} emitted a fallback without a successful model")
                    if status == 400 and (provider.requests != [None] or sent_models != ["model-a"]):
                        fail(f"HTTP 400 attempted fallback: {provider.requests!r}")
                    expected = (
                        ["model-a"] * 3 + (["model-b"] * 3 if fallback else [])
                        if fallback else ["model-a"] * 10
                    )
                    if status == 529 and sent_models != expected:
                        fail(f"fallback chain did not try the expected models: {sent_models!r}")
                finally:
                    host.close()
                    subscription.close()


@check("host_server.model_fallback_streamed")
def check_model_fallback_streamed() -> None:
    class _StreamResponse:
        def __iter__(self):
            return iter([b"data: reply\n", b"\n"])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _StreamProvider(ModelProvider):
        model = "model-a"

        def __init__(self) -> None:
            self.request_models: list[str | None] = []
            self.endpoint_models: list[str] = []

        @property
        def name(self) -> str:
            return "anthropic"

        @property
        def wire_format(self) -> int:
            return 2

        def create_response(self, request, *, cancel=None):
            raise AssertionError("streaming host called create_response")

        def create_response_stream(self, request, *, cancel=None):
            self.request_models.append(request.model)
            model = request.model or self.model
            self.endpoint_models.append(model)
            body = json.dumps({"model": model}).encode("utf-8")
            lines = open_stream_with_retry(
                urllib.request.Request("https://mock.invalid/stream", data=body),
                timeout=5,
                max_attempts=3,
                api_key="test-key",
                operation="mock stream",
                cancel=cancel,
            )
            for _line in lines:
                pass
            yield StreamCompleted(ModelResponse(Message(Role.ASSISTANT, "streamed answer")))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = root / ".symphonai" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text('[fallback]\nanthropic = ["model-b"]\n', encoding="utf-8")
        extensions = load_extensions(repo_root=root, home=root / "home")
        provider = _StreamProvider()
        broker = EventBroker()
        subscription = broker.subscribe()
        host = HostRun(
            provider,
            PermissionPolicy(repo_root=root),
            broker,
            sessions_root=root / "sessions",
            extensions=extensions,
            memory_extraction_enabled=False,
        )

        def open_stream(url_request, timeout):
            model = json.loads(url_request.data.decode("utf-8"))["model"]
            if model == "model-a":
                raise urllib.error.HTTPError(
                    url_request.full_url, 529, "overloaded", {}, io.BytesIO(b"busy")
                )
            return _StreamResponse()

        def terminal_events():
            events = []
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                event = subscription.get(timeout=0.05)
                if event is not None:
                    events.append(event)
                    if isinstance(event, (RunFinished, RunFailed)):
                        return events
            fail("streamed fallback run did not reach a terminal event")

        try:
            with mock.patch("symphonai_api.streaming.urllib.request.urlopen", side_effect=open_stream), mock.patch(
                "symphonai_api.streaming.time.sleep"
            ):
                host.start("first prompt")
                first_events = terminal_events()
                if not isinstance(first_events[-1], RunFinished):
                    fail(f"streamed fallback failed: {first_events[-1]!r}")
                first_fallbacks = [event for event in first_events if isinstance(event, ModelFallback)]
                if len(first_fallbacks) != 1 or (
                    first_fallbacks[0].from_model,
                    first_fallbacks[0].to_model,
                    first_fallbacks[0].reason,
                ) != ("model-a", "model-b", "529"):
                    fail(f"streamed fallback event was missing or incorrect: {first_fallbacks!r}")
                if any(
                    isinstance(event, AssistantTextDelta) for event in first_events
                ):
                    fail("failed primary stream emitted text before the fallback notice")
                deadline = time.monotonic() + 5
                while host._active is not None and time.monotonic() < deadline:
                    time.sleep(0.01)
                while subscription.get(timeout=0) is not None:
                    pass
                host.start("second prompt")
                second_events = terminal_events()
                if not isinstance(second_events[-1], RunFinished):
                    fail(f"second streamed fallback failed: {second_events[-1]!r}")
                if provider.endpoint_models != ["model-a", "model-b", "model-a", "model-b"]:
                    fail(f"next streamed turn did not restart from model-a: {provider.endpoint_models!r}")
                if provider.request_models != [None, "model-b", None, "model-b"]:
                    fail(f"primary request model was changed from None: {provider.request_models!r}")
        finally:
            subscription.close()
            host.close()
_APP_HTML_RESOURCE = re.compile(r"""(?:src|href)=[\"']([^\"']+)[\"']""")
_APP_STATIC_IMPORT = re.compile(
    r"""\b(?:import|export)\s+(?:[^;\"']*?\s+from\s*)?[\"']([^\"']+)[\"']"""
)
_APP_DYNAMIC_IMPORT = re.compile(r"""\bimport\(\s*[\"']([^\"']+)[\"']\s*\)""")
_APP_CSS_REFERENCE = re.compile(
    r"""(?:@import\s+(?:url\()?\s*|url\(\s*)[\"']?([^\"'()\s;]+)"""
)


def _app_import_references(content_type: str, text: str) -> list[str]:
    if content_type == "text/html":
        return _APP_HTML_RESOURCE.findall(text)
    if content_type == "text/javascript":
        return [*_APP_STATIC_IMPORT.findall(text), *_APP_DYNAMIC_IMPORT.findall(text)]
    if content_type == "text/css":
        return [
            reference
            for reference in _APP_CSS_REFERENCE.findall(text)
            if not reference.startswith(("data:", "blob:", "#"))
        ]
    return []


def _check_app_import_graph(get, host, page: str, headers: dict[str, str]) -> set[str]:
    pending = [
        urljoin("/app/", reference)
        for reference in _app_import_references("text/html", page)
    ]
    fetched: set[str] = set()
    while pending:
        asset_url = pending.pop()
        parsed = urlsplit(asset_url)
        if parsed.scheme or parsed.netloc or not parsed.path.startswith("/app/"):
            fail(f"app import escaped the served app directory: {asset_url!r}")
        asset_path = parsed.path
        if asset_path in fetched:
            continue
        expected_type = host_server_module.APP_CONTENT_TYPES.get(Path(asset_path).suffix)
        if expected_type is None:
            fail(f"app import graph names an unserved asset: {asset_path!r}")
        fetched.add(asset_path)
        status, response_headers, response_body = get(host, asset_url, headers=headers)
        if status != 200:
            fail(f"app import graph could not fetch {asset_path!r}: {status}")
        content_types = [
            value
            for key, value in response_headers
            if key.casefold() == "content-type"
        ]
        if content_types != [expected_type]:
            fail(
                f"app import graph got the wrong content type for {asset_path!r}: "
                f"{content_types!r}, expected {expected_type!r}"
            )
        text = response_body.decode("utf-8")
        pending.extend(
            urljoin(asset_url, reference)
            for reference in _app_import_references(expected_type, text)
        )
    for required in (
        "/app/keys.default.json",
        "/app/src/picker.js",
        "/app/src/json.js",
    ):
        if required not in fetched:
            fail(f"app import graph did not reach {required!r}")
    return fetched


class _RecordingWireFakeProvider(FakeModelProvider):
    def __init__(self, name: str, wire_format: int, responses: list[ModelResponse]) -> None:
        super().__init__(responses)
        self._provider_name = name
        self._wire_format = wire_format
        self.requests = []

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def wire_format(self) -> int:
        return self._wire_format

    def create_response(self, request, *, cancel=None):
        self.requests.append(request)
        return super().create_response(request, cancel=cancel)


@check("host_server.prompt_attachments")
def check_prompt_attachments() -> None:
    class RecordingProvider(FakeModelProvider):
        def __init__(self):
            super().__init__([ModelResponse(Message(Role.ASSISTANT, "done"))])
            self.requests = []

        def create_response(self, request, *, cancel=None):
            self.requests.append(request)
            return super().create_response(request, cancel=cancel)

    png = b"\x89PNG\r\n\x1a\n\x00"
    encoded_png = base64.b64encode(png).decode("ascii")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provider = RecordingProvider()
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        try:
            connection, response = _request(
                host, "POST", "/prompt",
                body={"prompt": "what is this", "attachments": [{"data": encoded_png}]},
                headers=_headers(host),
            )
            reply = json.loads(response.read())
            connection.close()
            if response.status != 200:
                fail(f"PNG prompt was rejected: {response.status}, {reply!r}")
            _wait_until(lambda: host.run._active is None, "PNG prompt did not finish")
            user = next(
                message for message in reversed(provider.requests[0].messages)
                if message.role == Role.USER
            )
            if user.content != (TextBlock("what is this"), ImageBlock(encoded_png, "image/png")):
                fail(f"PNG prompt reached the provider with the wrong content: {user.content!r}")

            stream_connection, stream_response = _subscribed_stream(host)
            try:
                connection, opened = _request(
                    host, "POST", "/session/open", body={"run_id": reply["run_id"]},
                    headers=_headers(host),
                )
                if opened.status != 200:
                    fail(f"attachment session did not reopen: {opened.status}, {opened.read()!r}")
                opened.read()
                connection.close()
                frame = _await_sse(
                    stream_connection, stream_response,
                    lambda item: item[1].get("type") == "HistoryMessage"
                    and item[1].get("role") == "user",
                    what="attachment history",
                )[1]
                expected = [{"kind": "image", "media_type": "image/png", "filename": None}]
                if frame.get("attachments") != expected or encoded_png in json.dumps(frame):
                    fail(f"attachment replay exposed data or omitted safe metadata: {frame!r}")
            finally:
                stream_connection.close()
        finally:
            host.close()


    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provider = RecordingProvider()
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        try:
            encoded_pdf = base64.b64encode(b"%PDF-1.7 sample").decode("ascii")
            connection, response = _request(
                host, "POST", "/prompt",
                body={"prompt": "", "attachments": [{"data": encoded_pdf, "filename": "spec.pdf"}]},
                headers=_headers(host),
            )
            reply = json.loads(response.read())
            connection.close()
            if response.status != 200:
                fail(f"attachment-only PDF prompt was rejected: {response.status}, {reply!r}")
            _wait_until(lambda: host.run._active is None, "PDF prompt did not finish")
            user = next(
                message for message in reversed(provider.requests[0].messages)
                if message.role == Role.USER
            )
            if user.content != (DocumentBlock(encoded_pdf, filename="spec.pdf"),):
                fail(f"PDF prompt reached the provider with the wrong content: {user.content!r}")
            title = json.loads(
                (root / "sessions" / reply["run_id"] / "meta.json").read_text(encoding="utf-8")
            )["title"]
            if title != "spec.pdf":
                fail(f"attachment-only prompt got the wrong title: {title!r}")
        finally:
            host.close()


@check("host_server.prompt_attachment_errors")
def check_prompt_attachment_errors() -> None:
    class RecordingProvider(FakeModelProvider):
        def __init__(self):
            super().__init__([ModelResponse(Message(Role.ASSISTANT, "done"))])
            self.requests = []

        def create_response(self, request, *, cancel=None):
            self.requests.append(request)
            return super().create_response(request, cancel=cancel)

    png = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00").decode("ascii")
    too_large = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * (5_000_001 - 8)).decode("ascii")
    cases = (
        ([{"data": "%%%"}], "attachment 0"),
        ([{"data": too_large}], "attachment 0"),
        ([{"data": base64.b64encode(b"plain text").decode("ascii")}], "attachment 0"),
        ([{"data": png}] * 11, "attachment 10"),
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provider = RecordingProvider()
        host = _host(provider, repo_root=root)
        try:
            for attachments, expected in cases:
                connection, response = _request(
                    host, "POST", "/prompt",
                    body={"prompt": "should not run", "attachments": attachments},
                    headers=_headers(host),
                )
                reply = json.loads(response.read())
                connection.close()
                if response.status != 400 or expected not in reply.get("error", ""):
                    fail(f"invalid attachment response was wrong: {response.status}, {reply!r}")
            if provider.requests or host.run._active is not None:
                fail("invalid attachment request started a run")
        finally:
            host.close()


@check("host_server.files_ranked_search")
def check_files_ranked_search() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for relative, text in (
            ("src/parser.py", "source"),
            ("tests/test_parser.py", "test"),
            ("docs/parse.md", "docs"),
            (".env", "secret"),
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        host = _host(repo_root=root, sessions_root=root / "sessions")
        try:
            connection, response = _request(host, "GET", "/files?query=parser", headers=_headers(host))
            result = json.loads(response.read())
            connection.close()
            if response.status != 200 or result["files"][:2] != ["src/parser.py", "tests/test_parser.py"] or ".env" in result["files"]:
                fail(f"ranked file search returned unexpected paths: {response.status}, {result!r}")
        finally:
            host.close()


@check("host_server.files_subsequence_and_forbidden")
def check_files_subsequence_and_forbidden() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "src").mkdir()
        (root / "src" / "parser.py").write_text("source", encoding="utf-8")
        (root / ".env").write_text("secret", encoding="utf-8")
        host = _host(repo_root=root, sessions_root=root / "sessions")
        try:
            connection, response = _request(host, "GET", "/files?query=spp", headers=_headers(host))
            result = json.loads(response.read())
            connection.close()
            if response.status != 200 or result["files"] != ["src/parser.py"]:
                fail(f"subsequence search or forbidden file filtering failed: {response.status}, {result!r}")
        finally:
            host.close()


@check("host_server.files_limits_and_truncation")
def check_files_limits_and_truncation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ("a.py", "b.py", "c.py"):
            (root / name).write_text("file", encoding="utf-8")
        host = _host(repo_root=root, sessions_root=root / "sessions")
        try:
            for value in ("0", "51", "x"):
                connection, response = _request(host, "GET", f"/files?limit={value}", headers=_headers(host))
                response.read()
                connection.close()
                if response.status != 400:
                    fail(f"invalid /files limit {value!r} returned {response.status}")
            with mock.patch("symphonai_host.files.MAX_SEARCH_FILES", 2):
                connection, response = _request(host, "GET", "/files?limit=50", headers=_headers(host))
                result = json.loads(response.read())
                connection.close()
            if response.status != 200 or not result["truncated"] or len(result["files"]) != 2:
                fail(f"file walk did not report its cap: {response.status}, {result!r}")
        finally:
            host.close()


@check("host_server.tree_and_source_policy")
def check_tree_and_source_policy() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "repo"
        root.mkdir()
        source = root / "src"
        source.mkdir()
        (source / "a.py").write_text("print('hello')\n", encoding="utf-8")
        (root / ".env").write_text("secret", encoding="utf-8")
        git = root / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (root / "big.bin").write_bytes(b"x" * (2 * 1024 * 1024))
        (root / "binary.bin").write_bytes(b"\xff\xfe")
        outside = base / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        (root / "linked-dir").symlink_to(source, target_is_directory=True)
        (root / "escape.txt").symlink_to(outside)
        host = _host(repo_root=root, sessions_root=base / "sessions")
        try:
            for route in ("/tree", "/source?path=src%2Fa.py"):
                connection, response = _request(host, "GET", route)
                response.read()
                connection.close()
                if response.status != 401:
                    fail(f"{route} did not require bearer authorization")

            connection, response = _request(host, "GET", "/tree", headers=_headers(host))
            tree = json.loads(response.read())
            connection.close()
            if response.status != 200 or tree != {
                "path": "",
                "entries": [
                    {"name": "src", "kind": "dir"},
                    {"name": "big.bin", "kind": "file"},
                    {"name": "binary.bin", "kind": "file"},
                ],
                "truncated": False,
            }:
                fail(f"root tree exposed denied paths or was not sorted: {response.status}, {tree!r}")

            connection, response = _request(
                host, "GET", f"/tree?{urlencode({'path': 'src'})}", headers=_headers(host),
            )
            src_tree = json.loads(response.read())
            connection.close()
            if response.status != 200 or src_tree["entries"] != [{"name": "a.py", "kind": "file"}]:
                fail(f"subdirectory tree returned the wrong entries: {response.status}, {src_tree!r}")

            connection, response = _request(
                host, "GET", f"/source?{urlencode({'path': 'src/a.py'})}", headers=_headers(host),
            )
            source_reply = json.loads(response.read())
            connection.close()
            if response.status != 200 or source_reply != {"path": "src/a.py", "text": "print('hello')\n"}:
                fail(f"source read returned the wrong reply: {response.status}, {source_reply!r}")

            for path in (
                ".env", ".ENV", ".GIT/HEAD", "../outside.txt", str(outside), "escape.txt"
            ):
                connection, response = _request(
                    host, "GET", f"/source?{urlencode({'path': path})}", headers=_headers(host),
                )
                body = response.read()
                connection.close()
                if response.status != 403 or body != b"":
                    fail(f"denied source path {path!r} returned {response.status}, {body!r}")

            for path, expected in (("big.bin", 413), ("binary.bin", 415), ("missing.txt", 404)):
                connection, response = _request(
                    host, "GET", f"/source?{urlencode({'path': path})}", headers=_headers(host),
                )
                body = response.read()
                connection.close()
                if response.status != expected or body != b"":
                    fail(f"source path {path!r} returned {response.status}, expected {expected}")
        finally:
            host.close()


@check("host_server.tree_truncation")
def check_tree_truncation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "repo"
        root.mkdir()
        for index in range(2_001):
            (root / f"{index:04}.txt").touch()
        host = _host(repo_root=root, sessions_root=base / "sessions")
        try:
            connection, response = _request(host, "GET", "/tree", headers=_headers(host))
            tree = json.loads(response.read())
            connection.close()
            names = [entry["name"] for entry in tree.get("entries", [])]
            if response.status != 200 or len(names) != 2_000 or not tree.get("truncated"):
                fail(f"tree did not cap its listing: {response.status}, {len(names)}, {tree.get('truncated')}")
            if names != sorted(names) or names[0] != "0000.txt" or names[-1] != "1999.txt":
                fail(f"capped tree was not sorted by name: {names[:2]!r}, {names[-2:]!r}")
        finally:
            host.close()


@check("host_server.files_prunes_forbidden_directories")
def check_files_prunes_forbidden_directories() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "src" / "parser.py"
        source.parent.mkdir()
        source.write_text("source", encoding="utf-8")
        hidden = root / ".venv"
        hidden.mkdir()
        for index in range(25_000):
            (hidden / f"file-{index:05}.py").touch()
        paths, truncated = repository_files(PermissionPolicy(repo_root=root), "parser", 20)
        if paths != ["src/parser.py"] or truncated:
            fail(f"forbidden files consumed the search cap: {paths!r}, truncated={truncated}")


@check("host_server.files_breadth_first_cap")
def check_files_breadth_first_cap() -> None:
    from symphonai_host import files as files_module

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "src" / "parser.py"
        source.parent.mkdir()
        source.write_text("source", encoding="utf-8")
        deep = root / "vendor" / "deep" / "a" / "b"
        deep.mkdir(parents=True)
        for index in range(3):
            (deep / f"file-{index:05}.py").touch()
        with mock.patch.object(files_module, "MAX_SEARCH_FILES", 3):
            paths, truncated = repository_files(PermissionPolicy(repo_root=root), "parser", 20)
        if paths != ["src/parser.py"] or not truncated:
            fail(f"breadth-first cap omitted a shallow file: {paths!r}, truncated={truncated}")


@check("host_server.files_candidate_cache_ttl")
def check_files_candidate_cache_ttl() -> None:
    from symphonai_host import files as files_module

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "a.py").write_text("a", encoding="utf-8")
        policy = PermissionPolicy(repo_root=root)
        clock = iter((100.0, 100.5, 101.0, 111.0, 111.5))
        with mock.patch.object(files_module, "_walk_candidates", wraps=files_module._walk_candidates) as walk, \
                mock.patch.object(files_module.time, "monotonic", side_effect=lambda: next(clock)):
            repository_files(policy, "a", 20)
            repository_files(policy, "b", 20)
            repository_files(policy, "a", 20)
        if walk.call_count != 2:
            fail(f"candidate cache walked {walk.call_count} times across its TTL")


@check("host_server.files_search_reaches_repository")
def check_files_search_reaches_repository() -> None:
    paths, _ = repository_files(PermissionPolicy(repo_root=REPO_ROOT), "leader", 20)
    if not paths or paths[0] != "symphonai_api/leader.py":
        fail(f"repository search did not rank the runtime leader first: {paths[:5]!r}")


def _host(
    provider: ModelProvider | None = None,
    *,
    broker: EventBroker | None = None,
    keepalive_seconds: float = 0.05,
    repo_root: Path = REPO_ROOT,
    token: str | None = None,
    sessions_root: Path | None = None,
    memory_extraction_enabled: bool | None = False,
) -> HostServer:
    host = HostServer(
        provider or FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
        PermissionPolicy(repo_root=repo_root),
        broker=broker,
        keepalive_seconds=keepalive_seconds,
        token=token,
        sessions_root=sessions_root,
        memory_extraction_enabled=memory_extraction_enabled,
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


@check("host_server.handshake_url")
def check_handshake_url() -> None:
    host = object.__new__(HostServer)
    host._httpd = mock.Mock(server_address=("127.0.0.1", 51234))
    host.token = "handshake-test-token"
    host._handshake_printed = False
    writes: list[str] = []

    class RecordedOutput(io.StringIO):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def write(self, text: str) -> int:
            writes.append(self.name)
            return super().write(text)

    output = RecordedOutput("stdout")
    errors = RecordedOutput("stderr")
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        host.print_handshake()
        host.print_handshake()
    lines = output.getvalue().splitlines()
    if len(lines) != 1:
        fail(f"host printed {len(lines)} stdout lines: {lines!r}")
    handshake = json.loads(lines[0])
    expected = (
        f"http://127.0.0.1:{handshake['port']}/app/?token={handshake['token']}"
    )
    if (
        set(handshake) != {"port", "token"}
        or handshake != {"port": host.port, "token": host.token}
        or errors.getvalue().splitlines() != [expected]
        or list(dict.fromkeys(writes)) != ["stdout", "stderr"]
    ):
        fail(
            "host startup output did not include the matching app URL on stderr: "
            f"stdout={output.getvalue()!r}, stderr={errors.getvalue()!r}"
        )


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
        connection, response = _request(host, "GET", "/conversation")
        try:
            if response.status != 401 or response.read() != b"":
                fail("unauthorized conversation request leaked a response body")
        finally:
            connection.close()
    finally:
        host.close()


@check("host_server.search_settings_credentials_and_registry")
def check_search_settings_credentials_and_registry() -> None:
    secret = "recognisable-search-secret-25ff"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        agents = root / ".symphonai" / "agents"
        agents.mkdir(parents=True)
        user_config = home / ".symphonai" / "config.toml"
        user_config.parent.mkdir(parents=True)
        user_config.write_text(
            f'[[trust.repositories]]\nroot = {json.dumps(str(root))}\nallow = ["agents"]\n',
            encoding="utf-8",
        )
        (root / ".symphonai" / "config.toml").write_text(
            '[search]\nendpoint = "brave"\n', encoding="utf-8",
        )
        (agents / "searcher.toml").write_text(
            'prompt = "Search."\ntools = ["read_file", "web_search"]\n[model]\nprovider = "fake"\n', encoding="utf-8",
        )
        (agents / "reader.toml").write_text(
            'prompt = "Read."\ntools = ["read_file"]\n[model]\nprovider = "fake"\n', encoding="utf-8",
        )
        extensions = load_extensions(repo_root=root, home=home)
        if set(extensions.agents) != {"searcher", "reader"}:
            fail("trusted project search definitions did not load")
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "searcher", "dispatch_subagent", {"subagent_name": "searcher", "task": "inspect"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "searcher done")),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "reader", "dispatch_subagent", {"subagent_name": "reader", "task": "inspect"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "reader done")),
            ModelResponse(Message(Role.ASSISTANT, "leader done")),
        ])
        host = HostServer(
            provider, PermissionPolicy(root), sessions_root=root / "sessions",
            extensions=extensions,
        )
        host.start()
        try:
            with mock.patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}):
                connection, response = _request(host, "GET", "/settings", headers=_headers(host))
                try:
                    settings = json.loads(response.read())["settings"]
                    if response.status != 200 or settings["search"] != [{
                        "name": "brave", "env_var": "BRAVE_SEARCH_API_KEY", "key_present": False,
                    }]:
                        fail(f"unkeyed search settings were wrong: {settings['search']!r}")
                finally:
                    connection.close()
                with mock.patch.object(host_server_module, "store") as stored:
                    connection, response = _request(
                        host, "POST", "/credentials",
                        body={"name": "BRAVE_SEARCH_API_KEY", "value": secret},
                        headers=_headers(host),
                    )
                    try:
                        credential_body = response.read()
                        if response.status != 200 or secret.encode() in credential_body:
                            fail(f"search credential route failed or disclosed its value: {response.status}")
                    finally:
                        connection.close()
                    stored.assert_called_once_with("BRAVE_SEARCH_API_KEY", secret)
                connection, response = _request(host, "GET", "/settings", headers=_headers(host))
                try:
                    body = response.read()
                    settings = json.loads(body)["settings"]
                    if response.status != 200 or secret.encode() in body or settings["search"] != [{
                        "name": "brave", "env_var": "BRAVE_SEARCH_API_KEY", "key_present": True,
                    }]:
                        fail(f"keyed search settings were wrong or disclosed the key: {settings['search']!r}")
                finally:
                    connection.close()
                session = SessionStore(root / "sessions", "search-registry", repo_root=root)
                try:
                    leader = host.run._new_leader(session)
                    if "web_search" not in leader._agent._tools:
                        fail("configured host leader did not receive web_search")
                    leader.run("delegate")
                    for name, expected in (("searcher", True), ("reader", False)):
                        record = leader.subagents.get(name)
                        if record is None or ("web_search" in record.agent._tools) != expected:
                            fail(f"project {name} search registry was wrong: {record!r}")
                    backend = host.run._search_backend
                    assert backend is not None
                    backend.max_attempts = 1
                    requests = []

                    def urlopen(request, *, timeout):
                        requests.append(request)
                        return io.BytesIO(b'{"web":{"results":[]}}')

                    output = io.StringIO()
                    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output), mock.patch(
                        "urllib.request.urlopen", side_effect=urlopen,
                    ):
                        backend.search("fixture query", limit=1)
                    if len(requests) != 1 or secret in requests[0].full_url:
                        fail("search key entered the request URL")
                    if requests[0].get_header("X-subscription-token") != secret:
                        fail("search key did not reach its preset header")
                    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output), mock.patch(
                        "urllib.request.urlopen", side_effect=ConnectionError(secret),
                    ):
                        try:
                            backend.search("failure query", limit=1)
                        except Exception as exc:
                            if secret in str(exc):
                                fail("search exception disclosed the key")
                        else:
                            fail("mocked search failure was accepted")
                    if secret in output.getvalue():
                        fail("search logs disclosed the key")
                finally:
                    session.close()
        finally:
            host.close()
        (root / ".symphonai" / "config.toml").write_text("", encoding="utf-8")
        unconfigured = load_extensions(repo_root=root, home=home)
        run = HostRun(
            FakeModelProvider(), PermissionPolicy(root), EventBroker(),
            sessions_root=root / "unconfigured-sessions", extensions=unconfigured,
        )
        session = SessionStore(root / "unconfigured-sessions", "no-search", repo_root=root)
        try:
            leader = run._new_leader(session)
            if "web_search" in leader._agent._tools:
                fail("unconfigured host leader acquired web_search")
            result = leader._dispatch_tool.execute(
                ToolCall("missing-search", "dispatch_subagent", {
                    "subagent_name": "searcher", "task": "inspect",
                }),
                PermissionPolicy(root),
            )
            if result.ok or "search is not configured" not in (result.error or ""):
                fail(f"unconfigured project search definition escaped dispatch refusal: {result!r}")
        finally:
            session.close()


@check("host_server.skills_reach_leader")
def check_skills_reach_leader() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        skill_dir = root / ".symphonai" / "skills"
        skill_dir.mkdir(parents=True)
        user_config = home / ".symphonai" / "config.toml"
        user_config.parent.mkdir(parents=True)
        user_config.write_text(
            f'[[trust.repositories]]\nroot = {json.dumps(str(root))}\nallow = ["skills"]\n',
            encoding="utf-8",
        )
        for name in ("review", "release"):
            (skill_dir / f"{name}.md").write_text(
                f'+++\nname = "{name}"\ndescription = "{name} steps."\n'
                f'when_to_use = "Use for {name}."\n+++\n\n# {name.title()} private body\n',
                encoding="utf-8",
            )
        extensions = load_extensions(repo_root=root, home=home)
        host = HostRun(
            FakeModelProvider(), PermissionPolicy(root), EventBroker(),
            sessions_root=root / "sessions", extensions=extensions,
        )
        session = SessionStore(root / "sessions", "skills", repo_root=root)
        try:
            leader = host._new_leader(session)
            tool = leader._agent._tools.get("use_skill")
            if tool is None:
                fail("host conversation leader did not receive use_skill")
            expected_roster = (
                "Load the full instructions of a skill. Call it when the task matches a skill's when_to_use."
                "\n\nAvailable skills:\n\n"
                "name: release\ndescription: release steps.\nwhen_to_use: Use for release."
                "\n\nname: review\ndescription: review steps.\nwhen_to_use: Use for review."
            )
            if tool.description != expected_roster:
                fail(f"host leader skill roster was not sorted and body-free: {tool.description!r}")
            if "private body" in tool.description:
                fail("host leader description exposed a skill body")
            original_description = tool.description
            release_path = skill_dir / "release.md"
            release_path.write_text(
                release_path.read_text(encoding="utf-8") + "Appended private procedure.\n",
                encoding="utf-8",
            )
            if tool.description != original_description or "Appended private" in tool.description:
                fail("host leader description changed after a body append")
            loaded_body = tool.execute(
                ToolCall("skill-body", "use_skill", {"name": "release"}),
                PermissionPolicy(root, mode="plan"),
            )
            if not loaded_body.ok or loaded_body.content != (
                "\n# Release private body\nAppended private procedure.\n"
            ):
                fail(f"host leader could not load a skill in plan mode: {loaded_body!r}")
        finally:
            session.close()

        no_skill_root = Path(temporary) / "no-skills"
        agent_dir = no_skill_root / ".symphonai" / "agents"
        agent_dir.mkdir(parents=True)
        no_skill_home = Path(temporary) / "no-skill-home"
        trust_config = no_skill_home / ".symphonai" / "config.toml"
        trust_config.parent.mkdir(parents=True)
        trust_config.write_text(
            f'[[trust.repositories]]\nroot = {json.dumps(str(no_skill_root))}\nallow = ["agents"]\n',
            encoding="utf-8",
        )
        (agent_dir / "leader.toml").write_text(
            'prompt = "Configured leader."\ntools = ["use_skill"]\n'
            '[model]\nprovider = "fake"\n',
            encoding="utf-8",
        )
        no_skill_extensions = load_extensions(repo_root=no_skill_root, home=no_skill_home)
        no_skill_run = HostRun(
            FakeModelProvider(), PermissionPolicy(no_skill_root), EventBroker(),
            sessions_root=no_skill_root / "sessions", extensions=no_skill_extensions,
        )
        no_skill_session = SessionStore(
            no_skill_root / "sessions", "no-skills", repo_root=no_skill_root,
        )
        try:
            try:
                no_skill_run._new_leader(no_skill_session)
            except host_run_module.ProviderSelectionError as exc:
                if str(exc) != "leader cannot use_skill: no skills are available":
                    fail(f"host no-skill refusal wording differed: {exc!r}")
            else:
                fail("host accepted a defined leader requiring unavailable use_skill")
        finally:
            no_skill_session.close()


@check("host_server.steer_route")
def check_steer_route() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        blocker = _HostControlTool()

        class RecordingProvider(FakeModelProvider):
            def __init__(self):
                super().__init__([
                    ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                        "wait", blocker.name, {},
                    )])),
                    ModelResponse(Message(Role.ASSISTANT, "done")),
                ])
                self.requests = []

            def create_response(self, request, *, cancel=None):
                self.requests.append(request)
                return super().create_response(request, cancel=cancel)

        provider = RecordingProvider()
        host = HostServer(
            provider, PermissionPolicy(root), sessions_root=root / "sessions",
            mcp_tools={blocker.name: blocker},
        )
        host.start()

        def post(body):
            connection, response = _request(host, "POST", "/steer", body=body, headers=_headers(host))
            try:
                raw = response.read()
                return response.status, json.loads(raw) if raw else {}
            finally:
                connection.close()

        try:
            if post({"text": "use b.py"}) != (409, {"error": "no turn is running"}):
                fail("steer without a running turn did not return 409")
            if post({"text": " "})[0] != 400 or post({"text": "use b.py", "extra": True})[0] != 400:
                fail("steer accepted blank text or an extra body field")
            connection, response = _request(
                host, "POST", "/prompt", body={"prompt": "start"}, headers=_headers(host),
            )
            try:
                if response.status != 200:
                    fail(f"steer test prompt did not start: {response.status}")
                response.read()
            finally:
                connection.close()
            if not blocker.entered.wait(3):
                fail("leader did not reach the blocking tool")
            status, body = post({"text": "use b.py"})
            if status != 200 or body != {"steered": True}:
                fail(f"steer did not accept a redirect: {status}, {body!r}")
            blocker.release.set()
            _wait_until(lambda: not host.run.active, "steered turn did not finish")
            if len(provider.requests) < 2:
                fail("steered run did not reach its next model request")
            messages = provider.requests[1].messages
            result_index = next((
                index for index, item in enumerate(messages)
                if item.tool_result is not None and item.tool_result.content == "released"
            ), None)
            steer_index = next((
                index for index, item in enumerate(messages)
                if item.role is Role.USER and item.text == "use b.py"
            ), None)
            if result_index is None or steer_index is None or steer_index <= result_index:
                fail(f"steer was not delivered after the tool result: {messages!r}")
            if post({"text": "too late"}) != (409, {"error": "no turn is running"}):
                fail("steer after the run ended did not return 409")
        finally:
            blocker.release.set()
            host.close()


@check("host_server.user_command_route")
def check_user_command_route() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        class RecordingProvider(FakeModelProvider):
            def __init__(self):
                super().__init__([ModelResponse(Message(Role.ASSISTANT, "done"))])
                self.requests = []

            def create_response(self, request, *, cancel=None):
                self.requests.append(request)
                return super().create_response(request, cancel=cancel)

        provider = RecordingProvider()
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()

        def post(body):
            connection, response = _request(host, "POST", "/command", body=body, headers=_headers(host))
            try:
                raw = response.read()
                return response.status, json.loads(raw) if raw else {}
            finally:
                connection.close()

        try:
            status, reply = post({"command": "printf hi; exit 3"})
            if status != 200 or reply != {"exit_code": 3, "output": "hi", "timed_out": False, "stopped": False}:
                fail(f"command route returned the wrong exit and output: {status}, {reply!r}")
            if (
                host_run_module.shell_environment is not shell_module.shell_environment
                or host_run_module.shell_executable is not shell_module.shell_executable
            ):
                fail("host command and run_shell do not share their shell helpers")
            with mock.patch.dict(os.environ, {"FAKE_PROBE_TOKEN": "abc"}):
                status, reply = post({"command": "printenv FAKE_PROBE_TOKEN"})
            if status != 200 or reply.get("output") != "":
                fail(f"command received a filtered environment value: {status}, {reply!r}")
            if post({"command": " "})[0] != 400 or post({"command": "true", "extra": True})[0] != 400:
                fail("command route accepted a blank command or extra field")
            connection, response = _request(
                host, "POST", "/prompt", body={"prompt": "continue"}, headers=_headers(host),
            )
            try:
                if response.status != 200:
                    fail(f"command context prompt did not start: {response.status}")
                response.read()
            finally:
                connection.close()
            _wait_until(lambda: not host.run.active, "command context prompt did not finish")
            messages = provider.requests[0].messages
            user_messages = [message.text for message in messages if message.role is Role.USER]
            if len(user_messages) < 3 or not all(
                value in user_messages[0] for value in (
                    "$ printf hi; exit 3", "hi", "(exit code 3)",
                )
            ) or user_messages[-1] != "continue":
                fail(f"command result was not added before the next prompt: {user_messages!r}")
        finally:
            host.close()


@check("host_server.removed_spec_runner_routes_are_not_found")
def check_removed_spec_runner_routes_are_not_found() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = HostServer(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            PermissionPolicy(root), sessions_root=root / "sessions",
        )
        host.start()
        try:
            for method, path in (
                ("GET", "/file?path=docs/roadmap.json"),
                ("GET", "/spec/files"),
                ("GET", "/spec/runs"),
                ("POST", "/spec/plan"),
                ("POST", "/spec/run"),
                ("POST", "/spec/review"),
                ("POST", "/spec/commit"),
            ):
                connection, response = _request(
                    host, method, path, body={} if method == "POST" else None,
                    headers=_headers(host),
                )
                try:
                    payload = json.loads(response.read())
                    if response.status != 404 or payload != {"error": "not found"}:
                        fail(f"removed {method} {urlsplit(path).path} route returned {response.status}: {payload!r}")
                finally:
                    connection.close()
        finally:
            host.close()


@check("host_server.user_command_output_that_is_not_utf8")
def check_user_command_output_that_is_not_utf8() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        run = HostRun(None, PermissionPolicy(root), EventBroker(), sessions_root=root / "sessions")
        result = run.run_user_command("printf 'ok \\247 done'")
    output = result["output"]
    if "ok" not in output or "\ufffd" not in output or "done" not in output:
        fail(f"user command did not replace invalid UTF-8 output bytes: {result!r}")


@check("host_server.user_command_does_not_block")
def check_user_command_does_not_block() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "ready"))])
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        subscription = host.broker.subscribe()
        command_reply = []
        command_error = []

        def request_command(command):
            try:
                connection, response = _request(
                    host, "POST", "/command", body={"command": command}, headers=_headers(host),
                )
                try:
                    raw = response.read()
                    command_reply.append((response.status, json.loads(raw) if raw else {}))
                finally:
                    connection.close()
            except Exception as exc:  # noqa: BLE001
                command_error.append(exc)

        try:
            host.run.start("seed conversation")
            _wait_until(lambda: not host.run.active, "seed conversation did not finish")
            leader = host.run._conversation[0]
            while subscription.get(timeout=0) is not None:
                pass
            thread = threading.Thread(target=request_command, args=("sleep 3",), daemon=True)
            thread.start()
            _wait_until(lambda: host.run._user_command is not None, "user command did not start")

            started = time.monotonic()
            connection, response = _request(host, "GET", "/settings", headers=_headers(host))
            try:
                response.read()
                if response.status != 200 or time.monotonic() - started >= 0.5:
                    fail("settings waited for the user command")
            finally:
                connection.close()

            other_event = RunStarted(
                agent_id=leader.agent_ref.agent_id, run_id="other-conversation-run", agent_name="leader",
            )
            started = time.monotonic()
            host.run._publish_active(other_event)
            observed = subscription.get(timeout=0.5)
            if observed != other_event or time.monotonic() - started >= 0.5:
                fail("runtime event publication waited for the user command")

            connection, response = _request(
                host, "POST", "/command", body={"command": "true"}, headers=_headers(host),
            )
            try:
                if response.status != 409 or json.loads(response.read()) != {"error": "a command is already running"}:
                    fail("second command did not return its 409")
            finally:
                connection.close()
            connection, response = _request(
                host, "POST", "/prompt", body={"prompt": "wait"}, headers=_headers(host),
            )
            try:
                if response.status != 409 or json.loads(response.read()) != {"error": "wait for the command to finish"}:
                    fail("prompt during a user command did not return its 409")
            finally:
                connection.close()

            started = time.monotonic()
            connection, response = _request(host, "POST", "/stop", body={}, headers=_headers(host))
            try:
                response.read()
            finally:
                connection.close()
            thread.join(2)
            if (
                thread.is_alive()
                or command_error
                or time.monotonic() - started >= 2
                or command_reply != [(200, {"exit_code": None, "output": "", "timed_out": False, "stopped": True})]
            ):
                fail(f"stopped command did not return promptly: {command_reply!r}, {command_error!r}")

            pid_path = root / "child.pid"
            command = f"sleep 30 & echo $! > {shlex.quote(str(pid_path))}; wait"
            command_reply.clear()
            thread = threading.Thread(target=request_command, args=(command,), daemon=True)
            thread.start()
            _wait_until(pid_path.exists, "grandchild command did not create its pid file")
            child_pid = int(pid_path.read_text())
            connection, response = _request(host, "POST", "/stop", body={}, headers=_headers(host))
            try:
                response.read()
            finally:
                connection.close()
            thread.join(2)
            if thread.is_alive() or command_reply[0][1].get("stopped") is not True:
                fail(f"stopped descendant command did not finish: {command_reply!r}")
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                pass
            else:
                fail(f"stopped command left child process {child_pid} running")
        finally:
            subscription.close()
            host.run.stop()
            host.close()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        blocker = _HostControlTool()
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "wait", blocker.name, {},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        host = HostServer(
            provider, PermissionPolicy(root), sessions_root=root / "sessions",
            mcp_tools={blocker.name: blocker},
        )
        host.start()
        try:
            connection, response = _request(
                host, "POST", "/prompt", body={"prompt": "start"}, headers=_headers(host),
            )
            response.read()
            connection.close()
            if not blocker.entered.wait(3):
                fail("command test leader did not reach the blocking tool")
            connection, response = _request(
                host, "POST", "/command", body={"command": "true"}, headers=_headers(host),
            )
            try:
                body = json.loads(response.read())
                if response.status != 409 or body != {"error": "wait for the turn to finish"}:
                    fail(f"command route did not reject an active turn: {response.status}, {body!r}")
            finally:
                connection.close()
        finally:
            blocker.release.set()
            host.close()


@check("host_server.agent_control_route")
def check_agent_control_route() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        blocker = _HostControlTool()
        provider = _HostControlProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "dispatch", "dispatch_subagent", {
                    "subagent_name": "worker", "task": "inspect",
                },
            )])),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "block", blocker.name, {},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "child done")),
            ModelResponse(Message(Role.ASSISTANT, "leader done")),
        ])
        host = HostServer(
            provider,
            PermissionPolicy(root),
            sessions_root=root / "sessions",
            mcp_tools={blocker.name: blocker},
        )
        host.start()

        def post(payload):
            connection, response = _request(
                host, "POST", "/agent/control", body=payload, headers=_headers(host),
            )
            try:
                body = response.read()
                return response.status, json.loads(body) if body else {}
            finally:
                connection.close()

        try:
            if post({"agent_id": "missing", "action": "pause"})[0] != 409:
                fail("control route did not reject a request with no active run")
            if post({"agent_id": "missing", "action": "unknown"})[0] != 400:
                fail("control route accepted an unknown action")
            if post({"agent_id": "missing", "action": "redirect"})[0] != 400:
                fail("control route accepted redirect without text")

            connection, response = _request(
                host, "POST", "/prompt", body={"prompt": "start"}, headers=_headers(host),
            )
            try:
                if response.status != 200:
                    fail(f"control route prompt did not start: {response.status}")
                response.read()
            finally:
                connection.close()
            if not blocker.entered.wait(3):
                fail("subagent did not reach the control tool")
            leader = host.run._conversation[0]
            child_id = leader._dispatch_tool.pool["worker"].agent_ref.agent_id
            paused_status, paused = post({"agent_id": child_id, "action": "pause"})
            if paused_status != 200 or paused != {"agent_id": child_id, "state": "paused"}:
                fail(f"control route did not pause the subagent: {paused_status}, {paused!r}")
            if post({"agent_id": child_id, "action": "pause"})[0] != 409:
                fail("control route accepted a repeated pause")
            resumed_status, resumed = post({"agent_id": child_id, "action": "resume"})
            if resumed_status != 200 or resumed != {"agent_id": child_id, "state": "running"}:
                fail(f"control route did not resume the subagent: {resumed_status}, {resumed!r}")
            if post({"agent_id": "unknown-agent", "action": "pause"})[0] != 404:
                fail("control route did not return 404 for an unknown agent")
            if post({"agent_id": child_id, "action": "redirect"})[0] != 400:
                fail("control route accepted redirect without text during a run")

            blocker.release.set()
            if not provider.final_entered.wait(3):
                fail("leader did not reach its final model request")
            if post({"agent_id": child_id, "action": "pause"})[0] != 404:
                fail("control route did not return 404 for a finished agent")

            active = host.run._active
            leader_id = leader.agent_ref.agent_id
            stopped_status, stopped = post({"agent_id": leader_id, "action": "stop"})
            if stopped_status != 200 or stopped != {"agent_id": leader_id, "state": "stopping"}:
                fail(f"control route did not stop the leader: {stopped_status}, {stopped!r}")
            provider.release_final.set()
            assert active is not None
            active.thread.join(5)
            if active.thread.is_alive() or not isinstance(active.terminal_event, RunFinished):
                fail("leader control stop did not end the host run")
            if active.terminal_event.stopped_reason != "cancelled":
                fail(f"leader control stop differed from /stop: {active.terminal_event!r}")
            if post({"agent_id": leader_id, "action": "pause"})[0] != 409:
                fail("control route did not reject a request after the run ended")
        finally:
            blocker.release.set()
            provider.release_final.set()
            host.close()


@check("host_server.survey_route")
def check_survey_route() -> None:
    token = "survey-route-token"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "docs").mkdir()
        (root / "tests").mkdir()
        (root / "README.md").write_text("read me", encoding="utf-8")
        (root / "docs" / "guide.md").write_text("guide", encoding="utf-8")
        (root / "main.py").write_text("", encoding="utf-8")
        host = _host(repo_root=root, token=token)
        try:
            for path in ("/survey", f"/survey?token={token}"):
                connection, response = _request(host, "GET", path)
                try:
                    body = response.read()
                    if response.status != 401 or body != b"":
                        fail(f"survey route accepted unauthenticated path {path!r}")
                finally:
                    connection.close()

            connection, response = _request(
                host, "GET", "/survey", headers=_headers(host)
            )
            try:
                body = response.read()
                if response.status != 200:
                    fail(f"authorized survey failed: {response.status}, {body!r}")
                payload = json.loads(body)
            finally:
                connection.close()
            survey = payload.get("survey", {})
            expected_fields = {
                "root",
                "languages",
                "by_directory",
                "entry_points",
                "docs",
                "tests",
                "tree_summary",
                "truncated_directories",
                "stopped",
                "file_count",
            }
            if not expected_fields <= set(survey):
                fail(f"survey route returned the wrong fields: {survey!r}")
            if survey.get("root") != ".":
                fail(f"survey root was not repository-relative: {survey!r}")
            if survey.get("languages") != [[".md", 2], [".py", 1]]:
                fail(f"survey route returned wrong languages: {survey!r}")
            if survey.get("docs") != ["README.md", "docs/guide.md"]:
                fail(f"survey route returned wrong documentation: {survey!r}")
            if survey.get("tests") != ["tests"]:
                fail(f"survey route returned wrong test directories: {survey!r}")
            if survey.get("by_directory") != [
                ["docs", [[".md", 1]]],
                ["tests", []],
            ]:
                fail(f"survey route returned wrong directory counts: {survey!r}")
            if survey.get("stopped") is not False or survey.get("file_count") != 3:
                fail(f"survey route returned wrong truncation state: {survey!r}")

            protocol = (REPO_ROOT / "symphonai_host" / "PROTOCOL.md").read_text(
                encoding="utf-8"
            )
            survey_paragraph = protocol.partition(
                "An authenticated `GET /survey`"
            )[2].partition("\n\n")[0]
            missing = [
                field for field in expected_fields if f"`{field}`" not in survey_paragraph
            ]
            if missing:
                fail(f"survey protocol paragraph omitted fields: {missing!r}")
        finally:
            host.close()


@check("host_server.project_route")
def check_project_route() -> None:
    token = "project-route-token"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "named-project"
        root.mkdir()
        host = _host(repo_root=root, token=token)
        try:
            for path in ("/project", f"/project?token={token}"):
                connection, response = _request(host, "GET", path)
                try:
                    body = response.read()
                    if response.status != 401 or body != b"":
                        fail(f"project route accepted unauthenticated path {path!r}")
                finally:
                    connection.close()

            connection, response = _request(
                host, "GET", "/project", headers={"Cookie": f"symphonai_app={token}"}
            )
            try:
                body = response.read()
                if response.status != 401 or body != b"":
                    fail("project route accepted an app cookie")
            finally:
                connection.close()

            connection, response = _request(
                host, "GET", "/project", headers=_headers(host)
            )
            try:
                body = response.read()
                expected = {
                    "repo_root": str(root.resolve()),
                    "name": root.resolve().name,
                }
                if response.status != 200 or json.loads(body) != expected:
                    fail(
                        f"authorized project response was wrong: "
                        f"{response.status}, {body!r}"
                    )
            finally:
                connection.close()
        finally:
            host.close()


@check("host_server.settings_route")
def check_settings_route() -> None:
    token = "settings-route-token"
    secret = "recognisable-settings-secret-84f1"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        trusted_other = Path(temporary) / "other"
        project_config = root / ".symphonai"
        user_config = home / ".symphonai"
        project_config.mkdir(parents=True)
        user_config.mkdir(parents=True)
        (project_config / "skills").mkdir()
        (project_config / "skills" / "blocked.md").write_text("offered", encoding="utf-8")
        (project_config / "config.toml").write_text(
            "[agents.ceiling]\nfetch_enabled = false\n", encoding="utf-8"
        )
        mcp_script = _write_host_mcp(Path(temporary))
        mcp_command = [
            sys.executable,
            str(mcp_script),
            str(Path(temporary) / "settings-mcp.pid"),
            "-",
        ]
        shown_command = shlex.join(mcp_command)
        (user_config / "config.toml").write_text(
            "[agents.ceiling]\nshell_enabled = false\n"
            "[[hooks]]\non = [\"RunStarted\"]\ncommand = [\"echo\", \"observed\"]\n"
            f"[[mcp.servers]]\nname = \"sample\"\ncommand = {json.dumps(mcp_command)}\nenabled = true\n"
            f"[[trust.repositories]]\nroot = {json.dumps(str(trusted_other))}\nallow = [\"skills\"]\n",
            encoding="utf-8",
        )
        extensions = load_extensions(repo_root=root, home=home)
        host = HostServer(
            FakeModelProvider(),
            PermissionPolicy(repo_root=root),
            token=token,
            sessions_root=root / "sessions",
            extensions=extensions,
        )
        host.start()
        try:
            if host.run.extensions is not extensions:
                fail("host run did not retain the resolved extensions")
            for path in ("/settings", f"/settings?token={token}"):
                connection, response = _request(host, "GET", path)
                try:
                    body = response.read()
                    if response.status != 401 or body != b"":
                        fail(f"settings route accepted unauthenticated path {path!r}")
                finally:
                    connection.close()

            catalog_variables = {entry.api_key_env_var: "" for entry in CATALOG.values()}
            with mock.patch.dict(
                os.environ,
                {**catalog_variables, "OPENAI_API_KEY": secret, "ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": " "},
            ):
                connection, response = _request(
                    host, "GET", "/settings", headers=_headers(host)
                )
                try:
                    body = response.read()
                    if response.status != 200 or secret.encode() in body:
                        fail(f"authorized settings status or secret disclosure: {response.status}")
                    payload = json.loads(body)
                finally:
                    connection.close()
            if set(payload) != {"settings"}:
                fail(f"settings envelope changed: {payload!r}")
            settings = payload["settings"]
            expected_fields = {
                "config", "ceiling", "trust", "hooks", "mcp_servers",
                "agents", "skills", "plugins", "withheld", "providers", "selection",
            }
            if (
                not expected_fields | {"search", "mode"} <= set(settings)
                or settings["search"] != []
                or settings["mode"] != "ask"
            ):
                fail(f"settings route returned the wrong fields: {settings!r}")
            config = {entry["key"]: entry for entry in settings["config"]}
            if (
                any(set(entry) != {"key", "value", "scope"} for entry in settings["config"])
                or config["agents.ceiling.shell_enabled"] != {
                    "key": "agents.ceiling.shell_enabled", "value": False, "scope": "user"
                }
                or config["agents.ceiling.fetch_enabled"] != {
                    "key": "agents.ceiling.fetch_enabled", "value": False, "scope": "project"
                }
            ):
                fail(f"settings config lost per-value provenance: {config!r}")
            if set(settings["ceiling"]) != {
                "allowed_write_scope", "shell_enabled", "shell_allowlist",
                "fetch_enabled", "fetch_allowlist", "modes",
            } or settings["ceiling"]["shell_enabled"] is not False:
                fail(f"settings ceiling is not resolved: {settings['ceiling']!r}")
            if settings["trust"] != [{"root": str(trusted_other.resolve()), "allow": ["skills"]}]:
                fail(f"settings trust listing changed: {settings['trust']!r}")
            if settings["hooks"] != [{"event": "RunStarted", "command": "echo observed"}]:
                fail(f"settings hooks changed: {settings['hooks']!r}")
            if settings["mcp_servers"] != [{"name": "sample", "command": shown_command, "started": False}]:
                fail(f"settings MCP listing changed: {settings['mcp_servers']!r}")
            if settings["withheld"] != [{
                "scope": "project",
                "directory": ".symphonai/skills",
                "names": ["blocked"],
                "reason": "repository not trusted for skills",
            }]:
                fail(f"settings withheld diagnostics changed: {settings['withheld']!r}")
            empty_rosters = {"agents": [], "skills": [], "plugins": []}
            if {kind: settings[kind] for kind in empty_rosters} != empty_rosters:
                fail("settings exposed untrusted extension names as loaded")
            providers = settings["providers"]
            expected_providers = [
                {"name": "anthropic", "env_var": "ANTHROPIC_API_KEY", "key_present": False},
                {"name": "gemini", "env_var": "GEMINI_API_KEY", "key_present": False},
                {"name": "openai", "env_var": "OPENAI_API_KEY", "key_present": True},
                *[
                    {"name": key, "env_var": CATALOG[key].api_key_env_var, "key_present": False}
                    for key in catalog_keys()
                ],
            ]
            if providers != expected_providers:
                fail(f"settings provider presence changed: {providers!r}")

            protocol = (REPO_ROOT / "symphonai_host" / "PROTOCOL.md").read_text(
                encoding="utf-8"
            )
            paragraph = protocol.partition("An authenticated `GET /settings`")[2].partition("\n\n")[0]
            if any(f"`{field}`" not in paragraph for field in expected_fields):
                fail("settings protocol paragraph omitted a response field")
        finally:
            host.close()


        pool = McpPool(extensions.mcp_servers, cwd=root)
        try:
            mcp_tools = pool.start()
            if "mcp__sample__search" not in mcp_tools:
                fail("settings MCP fixture did not start its server")
            started_host = HostServer(
                FakeModelProvider(),
                PermissionPolicy(repo_root=root),
                sessions_root=root / "sessions",
                extensions=extensions,
                mcp_tools=mcp_tools,
            )
            started_host.start()
            try:
                connection, response = _request(
                    started_host, "GET", "/settings", headers=_headers(started_host)
                )
                try:
                    settings = json.loads(response.read())["settings"]
                    if response.status != 200 or settings["mcp_servers"] != [
                        {"name": "sample", "command": shown_command, "started": True}
                    ]:
                        fail(f"settings did not report a started MCP server: {settings!r}")
                finally:
                    connection.close()
            finally:
                started_host.close()
        finally:
            pool.close()


@check("host_server.catalog_providers")
def check_catalog_providers() -> None:
    secret = "deepseek-fixture-key"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = _host(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "fallback"))]),
            repo_root=root,
            sessions_root=root / "sessions",
        )
        try:
            with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
                connection, response = _request(host, "POST", "/provider", body={"name": "deepseek", "model": "deepseek-chat"}, headers=_headers(host))
                try:
                    body = json.loads(response.read())
                    if response.status != 400 or "deepseek has no API key" not in body.get("error", ""):
                        fail(f"missing catalog credential was not named: {response.status}, {body!r}")
                finally:
                    connection.close()

            requests = []

            def urlopen(request, *, timeout):
                requests.append(request)
                return io.BytesIO(
                    b'data: {"choices":[{"delta":{"role":"assistant","content":"done"},"finish_reason":null}]}\n\n'
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    b'data: [DONE]\n\n'
                )

            with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": secret}), mock.patch(
                "urllib.request.urlopen", side_effect=urlopen,
            ):
                connection, response = _request(host, "POST", "/provider", body={"name": "deepseek", "model": "deepseek-chat"}, headers=_headers(host))
                try:
                    if response.status != 200:
                        fail(f"catalog provider selection failed: {response.status}, {response.read()!r}")
                finally:
                    connection.close()
                connection, response = _request(host, "POST", "/prompt", body={"prompt": "use selected provider"}, headers=_headers(host))
                try:
                    if response.status != 200:
                        fail(f"catalog provider prompt failed: {response.status}, {response.read()!r}")
                    response.read()
                finally:
                    connection.close()
                _wait_until(lambda: not host.run.active, "catalog provider turn did not finish")
            if (
                len(requests) != 1
                or requests[0].full_url != "https://api.deepseek.com/v1/chat/completions"
                or requests[0].get_header("Authorization") != f"Bearer {secret}"
            ):
                fail(f"catalog provider sent the wrong request: {[(req.full_url, req.header_items()) for req in requests]!r}")

            with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), mock.patch.object(host_server_module, "store") as store:
                connection, response = _request(host, "POST", "/credentials", body={"name": "OPENROUTER_API_KEY", "value": "x"}, headers=_headers(host))
                try:
                    if response.status != 200:
                        fail(f"catalog credential was refused: {response.status}")
                    response.read()
                finally:
                    connection.close()
                store.assert_called_once_with("OPENROUTER_API_KEY", "x")
                connection, response = _request(host, "POST", "/credentials", body={"name": "UNRELATED_KEY", "value": "x"}, headers=_headers(host))
                try:
                    if response.status != 400:
                        fail("unrelated credential name was accepted")
                    response.read()
                finally:
                    connection.close()

            with mock.patch.dict(os.environ, {"GROQ_API_KEY": "groq-fixture"}), mock.patch.object(
                host_server_module, "list_models", return_value=["groq-chat"],
            ) as listed:
                connection, response = _request(host, "GET", "/models?provider=groq", headers=_headers(host))
                try:
                    body = json.loads(response.read())
                    if response.status != 200 or [item["id"] for item in body.get("models", [])] != ["groq-chat"]:
                        fail(f"catalog model listing failed: {response.status}, {body!r}")
                    listed.assert_called_once()
                finally:
                    connection.close()
        finally:
            host.close()


@check("host_server.settings_selection")
def check_settings_selection() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        host = _host(repo_root=Path(temporary))
        class AnthropicFake(FakeModelProvider):
            @property
            def name(self):
                return "anthropic"

        try:
            connection, response = _request(
                host, "GET", "/settings", headers=_headers(host)
            )
            try:
                selection = json.loads(response.read())["settings"]["selection"]
                if response.status != 200 or selection is None or selection.get("provider") != "fake":
                    fail(f"initial selection omitted the configured provider: {selection!r}")
            finally:
                connection.close()

            with mock.patch.object(host_server_module, "_provider", return_value=AnthropicFake()):
                connection, response = _request(
                    host, "POST", "/provider", body={"name": "anthropic", "model": "m-1"},
                    headers=_headers(host),
                )
                try:
                    if response.status != 200:
                        fail(f"provider selection returned {response.status}")
                finally:
                    connection.close()
            connection, response = _request(
                host, "GET", "/settings", headers=_headers(host)
            )
            try:
                selection = json.loads(response.read())["settings"]["selection"]
                if response.status != 200 or selection != {
                    "provider": "anthropic", "model": "m-1", "effort": None,
                }:
                    fail(f"settings did not report the next-turn selection: {selection!r}")
            finally:
                connection.close()
        finally:
            host.close()


@check("host_server.model_listing_route")
def check_model_listing_route() -> None:
    secret = "recognisable-model-listing-secret-25i"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = _host(repo_root=root)

        def get(path: str, *, authorized: bool = True) -> tuple[int, bytes, dict | None]:
            connection, response = _request(
                host,
                "GET",
                path,
                headers=_headers(host) if authorized else None,
            )
            try:
                body = response.read()
                return response.status, body, json.loads(body) if body else None
            finally:
                connection.close()

        try:
            status, body, _ = get("/models?provider=openai", authorized=False)
            if status != 401 or body != b"":
                fail("model listing accepted an unauthenticated request")

            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}), mock.patch.object(
                host_server_module, "list_models"
            ) as listed:
                status, _, reply = get("/models?provider=openai")
                if (
                    status != 200
                    or reply is None
                    or reply.get("state") != "unknown"
                    or reply.get("models") != []
                    or "no API key" not in reply.get("detail", "")
                    or listed.call_count != 0
                ):
                    fail(f"missing-key model listing was wrong: {status}, {reply!r}")

            calls = []

            def available(provider: ModelProvider) -> list[str]:
                calls.append(provider)
                return ["gpt-listed", "gpt-second"]

            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": secret}), mock.patch.object(
                host_server_module, "list_models", side_effect=available,
            ):
                first = get("/models?provider=openai")
                second = get("/models?provider=openai")
                alternate = get(
                    "/models?provider=openai&base_url=http%3A%2F%2F127.0.0.1%3A9000%2Fv1"
                )
            expected = {
                "provider": "openai",
                "state": "available",
                "models": [
                    {"id": "gpt-listed", "efforts": None},
                    {"id": "gpt-second", "efforts": None},
                ],
                "detail": "",
                "filter": {"applied": False, "hidden": 0},
            }
            if any(status != 200 or reply != expected for status, _, reply in (first, second, alternate)):
                fail(f"available model listing response changed: {first!r}, {second!r}, {alternate!r}")
            if len(calls) != 2 or getattr(calls[1], "base_url", None) != "http://127.0.0.1:9000/v1":
                fail(f"successful model listings were not cached by base URL: {calls!r}")

            failure_url = "/models?provider=openai&base_url=http%3A%2F%2F127.0.0.1%3A9001%2Fv1"
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": secret}), mock.patch.object(
                host_server_module,
                "list_models",
                side_effect=ProviderError(f"catalogue failed with {secret}"),
            ) as failed:
                failures = [get(failure_url), get(failure_url)]
            for status, body, reply in failures:
                if (
                    status != 200
                    or reply is None
                    or reply.get("state") != "unknown"
                    or reply.get("models") != []
                    or "catalogue failed" not in reply.get("detail", "")
                    or secret.encode() in body
                ):
                    fail(f"failed model listing was unsafe or malformed: {status}, {body!r}")
            if failed.call_count != 2:
                fail("an unknown model listing result was cached")

            with mock.patch.object(host_server_module, "list_models") as listed:
                status, _, reply = get("/settings")
                if status != 200 or reply is None or listed.call_count != 0:
                    fail("settings performed model discovery")

            selected = FakeModelProvider()
            with mock.patch.object(host_server_module, "_provider", return_value=selected) as factory:
                connection, response = _request(
                    host,
                    "POST",
                    "/provider",
                    body={
                        "name": "openai",
                        "model": "gpt-custom",
                        "base_url": "http://127.0.0.1:9002/v1",
                    },
                    headers=_headers(host),
                )
                try:
                    body = response.read()
                    if response.status != 200 or json.loads(body) != {"selected": True}:
                        fail(f"provider payload was rejected: {response.status}, {body!r}")
                finally:
                    connection.close()
                connection, response = _request(
                    host,
                    "POST",
                    "/provider",
                    body={"name": "openai", "extra": True},
                    headers=_headers(host),
                )
                try:
                    response.read()
                    if response.status != 400:
                        fail("provider route accepted an unknown field")
                finally:
                    connection.close()
                if factory.call_args_list != [mock.call(
                    "openai", "gpt-custom", "http://127.0.0.1:9002/v1"
                )]:
                    fail(f"provider route changed its accepted payload: {factory.call_args_list!r}")
        finally:
            host.close()


@check("host_server.model_listing_efforts_cached")
def check_model_listing_efforts_cached() -> None:
    secret = "recognisable-model-effort-secret-25q"
    host = _host()

    def get() -> dict:
        connection, response = _request(
            host,
            "GET",
            "/models?provider=anthropic",
            headers=_headers(host),
        )
        try:
            body = response.read()
            if response.status != 200:
                fail(f"effort model listing returned {response.status}: {body!r}")
            return json.loads(body)
        finally:
            connection.close()

    try:
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": secret}), mock.patch.object(
            host_server_module,
            "list_models",
            return_value=["claude-sonnet-5", "claude-haiku-4-5", "vendor-private"],
        ) as listed:
            first = get()
            second = get()
        expected_models = [
            {
                "id": "claude-sonnet-5",
                "efforts": ["low", "medium", "high", "xhigh", "max"],
            },
            {"id": "claude-haiku-4-5", "efforts": []},
            {"id": "vendor-private", "efforts": None},
        ]
        if first.get("models") != expected_models or second.get("models") != expected_models:
            fail(f"model efforts were missing or changed: {first!r}, {second!r}")
        if listed.call_count != 1:
            fail(f"adding efforts changed model listing cache behavior: {listed.call_count}")
    finally:
        host.close()


@check("host_server.model_effort_knowledge_states")
def check_model_effort_knowledge_states() -> None:
    host = _host()

    def get(provider: str) -> dict:
        connection, response = _request(
            host,
            "GET",
            f"/models?provider={provider}",
            headers=_headers(host),
        )
        try:
            body = response.read()
            if response.status != 200:
                fail(f"{provider} model listing returned {response.status}: {body!r}")
            return json.loads(body)
        finally:
            connection.close()

    def listed(provider: ModelProvider) -> list[str]:
        if provider.name == "anthropic":
            return ["claude-sonnet-5", "claude-haiku-4-5"]
        return ["openai-model-with-no-table-row"]

    try:
        with mock.patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "anthropic-model-state-key",
                "OPENAI_API_KEY": "openai-model-state-key",
            },
        ), mock.patch.object(host_server_module, "list_models", side_effect=listed) as discovery:
            anthropic = get("anthropic")
            openai = get("openai")
        states = {
            (reply["provider"], model["id"]): model["efforts"]
            for reply in (anthropic, openai)
            for model in reply["models"]
        }
        if states != {
            ("anthropic", "claude-sonnet-5"): [
                "low", "medium", "high", "xhigh", "max"
            ],
            ("anthropic", "claude-haiku-4-5"): [],
            ("openai", "openai-model-with-no-table-row"): None,
        }:
            fail(f"model effort knowledge states were collapsed: {states!r}")
        if discovery.call_count != 2:
            fail(f"model state lookup made {discovery.call_count} discovery calls")
    finally:
        host.close()


@check("host_server.provider_effort_reaches_request")
def check_provider_effort_reaches_request() -> None:
    def received(choice: dict) -> tuple[str | None, str | None, bool]:
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            host = HostServer(
                None,
                PermissionPolicy(root),
                sessions_root=root / "sessions",
                memory_extraction_enabled=False,
            )
            host.start()
            try:
                with mock.patch.object(
                    host_server_module,
                    "_provider",
                    return_value=provider,
                ), mock.patch.object(
                    provider,
                    "create_response",
                    wraps=provider.create_response,
                ) as create_response:
                    connection, response = _request(
                        host,
                        "POST",
                        "/provider",
                        body=choice,
                        headers=_headers(host),
                    )
                    try:
                        body = response.read()
                        if response.status != 200:
                            fail(f"effort provider choice was rejected: {response.status}, {body!r}")
                    finally:
                        connection.close()
                    _send_host_prompt(host, "use this choice")
                if create_response.call_count != 1:
                    fail(f"provider received {create_response.call_count} requests")
                request = create_response.call_args.args[0]
                leader = host.run._conversation[0]
                provider_is_unwrapped = (
                    leader._config.leader_provider is provider
                    and leader._config.subagent_provider is provider
                )
                return request.model, request.effort, provider_is_unwrapped
            finally:
                host.close()

    selected = received({"name": "openai", "model": "gpt-listed", "effort": "high"})
    if selected != ("gpt-listed", "high", True):
        fail(f"selected effort did not reach the provider request: {selected!r}")
    unlisted = received({"name": "openai", "model": "private-model"})
    if unlisted != ("private-model", None, True):
        fail(f"an unlisted typed model acquired an effort: {unlisted!r}")


@check("host_server.model_listing_filter")
def check_model_listing_filter() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / ".symphonai" / "config.toml"
        source.parent.mkdir(parents=True)
        source.write_text('[models]\nopenai = ["gpt-second", "not-discovered"]\n', encoding="utf-8")
        host = HostServer(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            PermissionPolicy(repo_root=root),
            keepalive_seconds=0.05,
            extensions=load_extensions(repo_root=root),
        )
        host.start()
        try:
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "model-filter-test-key"}), mock.patch.object(
                host_server_module, "list_models", return_value=["gpt-first", "gpt-second", "gpt-third"],
            ):
                connection, response = _request(host, "GET", "/models?provider=openai", headers=_headers(host))
                try:
                    body = response.read()
                    reply = json.loads(body)
                finally:
                    connection.close()
            if (
                response.status != 200
                or [row["id"] for row in reply.get("models", [])] != ["gpt-second"]
                or reply.get("filter") != {"applied": True, "hidden": 2}
            ):
                fail(f"configured model list did not filter discovered models: {reply!r}")
        finally:
            host.close()


@check("host_server.unknown_model_effort_can_be_set")
def check_unknown_model_effort_can_be_set() -> None:
    provider = _RecordingWireFakeProvider(
        "openai",
        1,
        [ModelResponse(Message(Role.ASSISTANT, "done"))],
    )
    host = _host()
    try:
        with mock.patch.object(host_server_module, "_provider", return_value=provider):
            connection, response = _request(
                host,
                "POST",
                "/provider",
                body={
                    "name": "openai",
                    "model": "private-model",
                    "effort": "experimental",
                },
                headers=_headers(host),
            )
            try:
                body = response.read()
                if response.status != 200:
                    fail(f"unknown model effort was rejected: {response.status}, {body!r}")
            finally:
                connection.close()
        _send_host_prompt(host, "use the custom effort")
        if len(provider.requests) != 1:
            fail(f"unknown model effort made {len(provider.requests)} requests")
        request = provider.requests[0]
        if request.model != "private-model" or request.effort != "experimental":
            fail(f"unknown model effort was not forwarded: {request!r}")
    finally:
        host.close()


@check("host_server.memory_extraction_can_be_disabled")
def check_memory_extraction_can_be_disabled() -> None:
    provider = _RecordingWireFakeProvider(
        "openai", 1, [ModelResponse(Message(Role.ASSISTANT, "done"))]
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = HostServer(
            provider,
            PermissionPolicy(root),
            sessions_root=root / "sessions",
            memory_root=root / "memory",
            memory_extraction_enabled=False,
        )
        host.start()
        try:
            memory = host.run.memory_store()
            if memory is None:
                fail("disabled extraction made the memory store unavailable")
            memory.write(
                "seeded memory survives extraction-off", agent_name="leader", run_id="seed"
            )
            _send_host_prompt(host, "answer without background extraction")
            if len(provider.requests) != 1:
                fail(f"disabled extraction made {len(provider.requests) - 1} background requests")
            tool_names = {tool["function"]["name"] for tool in provider.requests[0].tools}
            if not {"remember", "forget"}.issubset(tool_names):
                fail(f"disabling extraction removed memory tools: {tool_names!r}")
            prompt_text = "\n".join(message.text for message in provider.requests[0].messages)
            if "Memory persists across conversations" not in prompt_text:
                fail("disabling extraction removed the memory guide")
            if "seeded memory survives extraction-off" not in prompt_text:
                fail("disabling extraction removed seeded memory")
        finally:
            host.close()

    class RecordingProvider(FakeModelProvider):
        def __init__(self):
            super().__init__([
                ModelResponse(Message(Role.ASSISTANT, "leader final reply")),
                ModelResponse(Message(Role.ASSISTANT, "extraction complete")),
            ])
            self.requests = []

        def create_response(self, request, *, cancel=None):
            self.requests.append(request)
            return super().create_response(request, cancel=cancel)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = RecordingProvider()
        host = HostServer(
            provider,
            PermissionPolicy(root),
            sessions_root=root / "sessions",
            memory_root=root / "memory",
        )
        host.start()
        try:
            _send_host_prompt(host, "remember the preference")
            _wait_until(lambda: len(provider.requests) >= 2, "memory extraction did not run")
            if len(provider.requests) != 2:
                fail(f"enabled extraction made {len(provider.requests)} requests")
            extraction_messages = provider.requests[1].messages
            if not any(
                message.role is Role.ASSISTANT and message.text == "leader final reply"
                for message in extraction_messages
            ) or not any(
                message.role is Role.USER and "Analyze the most recent" in message.text
                for message in extraction_messages
            ):
                fail(f"extraction did not fork the completed leader history: {extraction_messages!r}")
        finally:
            host.close()


@check("host_server.memory_extraction_streams_after_turn")
def check_memory_extraction_streams_after_turn() -> None:
    provider = _RecordingWireFakeProvider("fake", 4, [
        ModelResponse(Message(Role.ASSISTANT, "leader answer")),
        ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
            "extract-save", "remember", {"text": "Run checks."},
        )])),
        ModelResponse(Message(Role.ASSISTANT, "extraction complete")),
    ])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = HostServer(
            provider,
            PermissionPolicy(root, mode="allow"),
            sessions_root=root / "sessions",
            memory_root=root / "memory",
            memory_extraction_enabled=True,
        )
        host.start()
        connection, response = _subscribed_stream(host)
        try:
            prompt_connection, prompt = _request(
                host, "POST", "/prompt", body={"prompt": "remember Run checks."},
                headers=_headers(host),
            )
            try:
                reply = json.loads(prompt.read())
                if prompt.status != 200:
                    fail(f"memory prompt was rejected: {prompt.status}, {reply!r}")
            finally:
                prompt_connection.close()

            streamed = []
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                frame = _await_sse(
                    connection,
                    response,
                    lambda candidate: isinstance(candidate, tuple)
                    and candidate[0] == "event",
                    deadline=min(1, deadline - time.monotonic()),
                    what="memory extraction event",
                )
                streamed.append(frame[1])
                if frame[1].get("type") == "RunFinished":
                    _, conversation = _conversation_reply(host)
                    leader = next(
                        (agent for agent in conversation.get("conversation", {}).get("agents", [])
                         if agent.get("name") == "leader"),
                        None,
                    )
                    if leader is None or leader.get("state") == "running":
                        fail(f"conversation still reported a running leader after RunFinished: {conversation!r}")
                if (
                    frame[1].get("type") == "ToolCallFinished"
                    and frame[1].get("tool_call_id") == "extract-save"
                ):
                    break
            matching = [
                (index, event)
                for index, event in enumerate(streamed)
                if event.get("tool_call_id") == "extract-save"
                and event.get("type") in {"ToolCallStarted", "ToolCallFinished"}
            ]
            start = next((index for index, event in matching if event["type"] == "ToolCallStarted"), None)
            finish = next((index for index, event in matching if event["type"] == "ToolCallFinished"), None)
            terminal = next(
                (index for index, event in enumerate(streamed) if event.get("type") == "RunFinished"),
                None,
            )
            if terminal is None or start is None or finish is None or not terminal < start < finish:
                fail(f"extractor events were missing or preceded RunFinished: {streamed!r}")
            if any(
                event.get("session_id") != reply.get("run_id")
                for _, event in matching
            ):
                fail(f"extractor events were not routed to their conversation: {matching!r}")
        finally:
            connection.close()
            host.close()


@check("host_server.background_usage_is_announced")
def check_background_usage_is_announced() -> None:
    host = _host()
    connection, response = _subscribed_stream(host)
    try:
        _send_host_prompt(host, "finish before recording background usage")
        session_id = host.run.conversation_stats()["session_id"]
        host.run._record_background_usage(session_id, "memory-model", UsageTotals(12, 8, 1))
        frame = _await_sse(
            connection,
            response,
            lambda candidate: isinstance(candidate, tuple)
            and candidate[0] == "event"
            and candidate[1].get("type") == "BackgroundUsageRecorded",
            what="background usage event",
        )
        if not isinstance(frame, tuple) or frame[1].get("session_id") != session_id:
            fail(f"background usage event omitted its conversation id: {frame!r}")
    finally:
        connection.close()
        host.close()


@check("host_server.settings_roster_paths")
def check_settings_roster_paths() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        root.mkdir()
        user_base = home / ".symphonai"
        project_base = root / ".symphonai"

        def write_members(base: Path, name: str) -> None:
            agents = base / "agents"
            skills = base / "skills"
            plugin = base / "plugins" / name
            agents.mkdir(parents=True)
            skills.mkdir()
            plugin.mkdir(parents=True)
            (agents / f"{name}.toml").write_text(
                f'prompt = "{name}"\n[model]\nprovider = "fake"\n', encoding="utf-8"
            )
            (skills / f"{name}.md").write_text(
                "+++\n" f'name = "{name}"\n' f'description = "{name}"\n'
                f'when_to_use = "{name}"\n' "+++\nbody\n",
                encoding="utf-8",
            )
            (plugin / "plugin.toml").write_text(
                f'name = "{name}"\nversion = "1"\ndescription = "{name}"\n',
                encoding="utf-8",
            )

        write_members(user_base, "zeta")
        write_members(project_base, "alpha")
        (user_base / "config.toml").write_text(
            f'[[trust.repositories]]\nroot = {json.dumps(str(root))}\n'
            'allow = ["agents", "skills", "plugins"]\n',
            encoding="utf-8",
        )
        extensions = load_extensions(repo_root=root, home=home)
        host = HostServer(
            FakeModelProvider(), PermissionPolicy(repo_root=root),
            sessions_root=root / "sessions", extensions=extensions,
        )
        host.start()
        try:
            connection, response = _request(host, "GET", "/settings", headers=_headers(host))
            try:
                settings = json.loads(response.read())["settings"]
                if response.status != 200:
                    fail("settings roster route did not respond successfully")
            finally:
                connection.close()
            for kind, suffix in (("agents", ".toml"), ("skills", ".md"), ("plugins", "")):
                expected = [
                    {"name": "alpha", "path": f".symphonai/{kind}/alpha{suffix}"},
                    {"name": "zeta", "path": str((user_base / kind / f"zeta{suffix}").resolve())},
                ]
                if settings[kind] != expected:
                    fail(f"settings {kind} roster lost sorted names or source paths: {settings[kind]!r}")
        finally:
            host.close()

        unknown_source = replace(extensions, agents={"orphan": extensions.agents["alpha"]})
        unknown_host = HostServer(
            FakeModelProvider(), PermissionPolicy(repo_root=root),
            sessions_root=root / "other-sessions", extensions=unknown_source,
        )
        unknown_host.start()
        try:
            connection, response = _request(unknown_host, "GET", "/settings", headers=_headers(unknown_host))
            try:
                settings = json.loads(response.read())["settings"]
                if response.status != 200 or settings["agents"] != [{"name": "orphan", "path": ""}]:
                    fail("settings dropped an agent with unknown source path")
            finally:
                connection.close()
        finally:
            unknown_host.close()

    source = Path(host_server_module.__file__ or "").read_text(encoding="utf-8")
    if '"/file"' in source.partition("def do_POST")[2]:
        fail("host POST handler offers a file write route")


@check("host_server.app_routes")
def check_app_routes() -> None:
    token = "browser-route-token"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        project_root = root / "project"
        decoy = project_root / "symphonai_app"
        decoy_source = decoy / "src"
        decoy_source.mkdir(parents=True)
        marker = b"user-project-app-marker"
        (decoy / "index.html").write_text(
            "<!doctype html><html><head>"
            f"{host_server_module.APP_HANDSHAKE_MARKER}"
            '<link rel="stylesheet" href="app.css">'
            "</head><body><main>decoy</main>"
            '<script type="module" src="src/app.js"></script>'
            "</body></html>",
            encoding="utf-8",
        )
        (decoy / "app.css").write_text("body { color: red; }", encoding="utf-8")
        (decoy_source / "app.js").write_text(
            "export const decoy = true;",
            encoding="utf-8",
        )
        (decoy / "project-marker.js").write_bytes(marker)
        docs = project_root / "docs"
        docs.mkdir()
        (docs / "roadmap.json").write_text("{}", encoding="utf-8")
        outside = root / "outside.js"
        outside.write_text("outside", encoding="utf-8")

        responses = []

        def request(
            active_host,
            method: str,
            path: str,
            *,
            headers: dict[str, str] | None = None,
            body: dict | None = None,
        ):  # noqa: ANN202
            connection, response = _request(
                active_host,
                method,
                path,
                headers=headers,
                body=body,
            )
            try:
                result = (
                    response.status,
                    tuple(response.getheaders()),
                    response.read(),
                )
                responses.append((method, path, result))
                return result
            finally:
                connection.close()

        def get(
            active_host,
            path: str,
            *,
            headers: dict[str, str] | None = None,
        ):  # noqa: ANN202
            return request(active_host, "GET", path, headers=headers)

        def header_values(headers, name: str) -> list[str]:  # noqa: ANN001
            return [value for key, value in headers if key.casefold() == name.casefold()]

        host = _host(repo_root=project_root, token=token)
        try:
            for path, location in (
                ("/app", "/app/"),
                (f"/app?token={token}", f"/app/?token={token}"),
            ):
                status, headers, body = get(host, path)
                if (
                    status != 302
                    or header_values(headers, "Location") != [location]
                    or header_values(headers, "Set-Cookie")
                    or body != b""
                ):
                    fail(
                        "app canonical redirect was wrong: "
                        f"{path!r}, {status}, {headers!r}, {body!r}"
                    )

            served_html = None
            cookie_header = None
            for path, request_headers in (
                ("/app/", _headers(host)),
                (f"/app/?token={token}", None),
            ):
                status, headers, body = get(host, path, headers=request_headers)
                content_types = header_values(headers, "Content-Type")
                cookies = header_values(headers, "Set-Cookie")
                if status != 200 or content_types != ["text/html"]:
                    fail(f"app index response was wrong: {status}, {headers!r}")
                if len(cookies) != 1:
                    fail(f"app index set {len(cookies)} cookies instead of one")
                cookie = cookies[0]
                if not cookie.startswith(f"symphonai_app={token};"):
                    fail(f"app cookie did not carry the host token: {cookie!r}")
                if "Path=/app/" not in cookie:
                    fail(f"app cookie omitted Path=/app/: {cookie!r}")
                if "HttpOnly" not in cookie:
                    fail(f"app cookie omitted HttpOnly: {cookie!r}")
                if "SameSite=Strict" not in cookie:
                    fail(f"app cookie omitted SameSite=Strict: {cookie!r}")
                if "max-age" in cookie.casefold() or "expires" in cookie.casefold():
                    fail(f"app cookie was persistent: {cookie!r}")
                text = body.decode("utf-8")
                if text.count("window.__symphonai = ") != 1:
                    fail("app index did not contain exactly one injected handshake")
                encoded = text.split("window.__symphonai = ", 1)[1].split(
                    ";</script>", 1
                )[0]
                if json.loads(encoded) != {"port": host.port, "token": token}:
                    fail(f"app index injected the wrong handshake: {encoded!r}")
                served_html = text
                cookie_header = cookie.split(";", 1)[0]

            if served_html is None or cookie_header is None:
                fail("app index did not yield browser credentials")
            status, _, body = get(host, "/app/")
            if status != 401 or body != b"":
                fail("app page loaded without credentials")
            status, _, body = get(
                host,
                "/app/",
                headers={"Cookie": cookie_header},
            )
            if status != 200 or b"window.__symphonai = " not in body:
                fail("app cookie did not reload the page route")

            json_status, json_headers, _ = get(
                host,
                "/app/keys.default.json",
                headers={"Cookie": cookie_header},
            )
            if json_status != 200 or header_values(json_headers, "Content-Type") != [
                "application/json"
            ]:
                fail(
                    "app JSON module did not load as application/json: "
                    f"{json_status}, {json_headers!r}"
                )

            for reference in re.findall(r'(?:src|href)="([^"]+)"', served_html):
                asset_path = urljoin("/app/", reference)
                status, _, _ = get(
                    host,
                    asset_path,
                    headers={"Cookie": cookie_header},
                )
                if status != 200:
                    fail(f"browser cookie did not load app resource {asset_path!r}: {status}")

            for request_headers, label in (
                ({"Cookie": cookie_header}, "cookie"),
                (_headers(host), "header"),
            ):
                status, headers, _ = get(
                    host,
                    "/app/src/app.js",
                    headers=request_headers,
                )
                if status != 200 or header_values(headers, "Content-Type") != [
                    "text/javascript"
                ]:
                    fail(f"app JavaScript rejected {label} authentication")

            for path in ("/app/secret.py", "/app/notes.md"):
                status, _, body = get(host, path, headers=_headers(host))
                if status != 403 or body != b"":
                    fail(f"app route served a forbidden extension: {path!r}")

            traversal = (
                "/app/../outside.js",
                "/app/src/../../outside.js",
                f"/app/{outside}",
            )
            for path in traversal:
                status, _, body = get(host, path, headers=_headers(host))
                if status != 403 or body != b"":
                    fail(f"app route accepted traversal fixture {path!r}: {status}")

            for path in (
                f"/app/?token=wrong-{token}",
                f"/app/src/app.js?token={token}",
                f"/roadmap?token={token}",
            ):
                status, _, body = get(host, path)
                if status != 401 or body != b"":
                    fail(f"query authentication escaped exact /app: {path!r}, {status}")

            wrong_cookie = {"Cookie": "symphonai_app=wrong-token"}
            status, _, body = get(host, "/app/src/app.js", headers=wrong_cookie)
            if status != 401 or body != b"":
                fail("app asset accepted a cookie carrying the wrong token")

            for method, path, body in (
                ("GET", "/roadmap", None),
                ("GET", "/events", None),
                ("POST", "/prompt", {"prompt": "must not run"}),
            ):
                status, _, response_body = request(
                    host,
                    method,
                    path,
                    headers={"Cookie": cookie_header},
                    body=body,
                )
                if status != 401 or response_body != b"":
                    fail(f"app cookie escaped to {method} {path}: {status}")

            status, _, body = get(
                host,
                "/app/project-marker.js",
                headers={"Cookie": cookie_header},
            )
            if status != 404 or marker in body:
                fail("host served a symphonai_app directory from the user project")

            source = inspect.getsource(HostServer._handler_type)
            if "_contains_path(app_root, resolved)" not in source:
                fail("app containment did not use the shared path rule")
            if "app_root = _app_root()" not in source or (
                'app_root = host._repo_root / "symphonai_app"' in source
            ):
                fail("app assets were resolved from the user project")
        finally:
            host.close()

        packaged_fixture = root / "packaged" / "symphonai_app"
        packaged_fixture.mkdir(parents=True)
        (packaged_fixture / "escape.js").symlink_to(outside)
        with mock.patch.object(
            host_server_module,
            "_app_root",
            return_value=packaged_fixture,
        ):
            host = _host(repo_root=project_root, token=token)
            try:
                status, _, body = get(
                    host,
                    "/app/escape.js",
                    headers=_headers(host),
                )
                if status != 403 or body != b"":
                    fail("app route accepted an escaping symlink")
            finally:
                host.close()

        missing_app = root / "missing" / "symphonai_app"
        with mock.patch.object(
            host_server_module,
            "_app_root",
            return_value=missing_app,
        ):
            host = _host(repo_root=project_root, token=token)
            try:
                status, headers, body = get(host, f"/app/?token={token}")
                if (
                    status != 404
                    or json.loads(body) != {"error": "app is not installed"}
                    or header_values(headers, "Set-Cookie")
                    or marker in body
                ):
                    fail(f"missing packaged app fell back to the project: {status}, {body!r}")
            finally:
                host.close()

        protected_paths = (
            str(root),
            str(project_root),
            str(REPO_ROOT),
            str(host_server_module._app_root()),
        )
        for method, path, (_, headers, body) in responses:
            response_text = repr(headers) + body.decode("utf-8", errors="replace")
            if any(protected in response_text for protected in protected_paths):
                fail(f"app response exposed an absolute path: {method} {path}")
            request_path = urlsplit(path).path
            app_redirect = method == "GET" and request_path == "/app"
            app_page = method == "GET" and request_path == "/app/"
            if token in body.decode("utf-8", errors="replace") and not app_page:
                fail(f"app response body exposed the token: {method} {path}")
            token_headers = [
                (name, value)
                for name, value in headers
                if token in value
            ]
            if token_headers:
                names = [name.casefold() for name, _ in token_headers]
                allowed = (
                    (app_redirect and names == ["location"])
                    or (app_page and names == ["set-cookie"])
                )
                if not allowed:
                    fail(f"app response header exposed the token: {method} {path}")


@check("host_server.app_import_graph")
def check_app_import_graph() -> None:
    token = "app-import-graph-token"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        host = _host(repo_root=root, token=token)

        def get(
            active_host,
            path: str,
            *,
            headers: dict[str, str] | None = None,
        ):  # noqa: ANN202
            connection, response = _request(
                active_host,
                "GET",
                path,
                headers=headers,
            )
            try:
                return response.status, tuple(response.getheaders()), response.read()
            finally:
                connection.close()

        try:
            status, headers, body = get(host, "/app/", headers=_headers(host))
            if status != 200:
                fail(f"app import graph could not load /app/: {status}")
            content_types = [
                value
                for key, value in headers
                if key.casefold() == "content-type"
            ]
            if content_types != ["text/html"]:
                fail(f"app page had the wrong content type: {content_types!r}")
            fetched = _check_app_import_graph(
                get,
                host,
                body.decode("utf-8"),
                _headers(host),
            )
            if "/app/keys.default.json" not in fetched:
                fail("app import graph omitted keys.default.json")
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


@check("host_server.events_subscribe_before_reply")
def check_events_subscribe_before_reply() -> None:
    host = _host()
    handler_type = host._httpd.RequestHandlerClass
    original_end_headers = handler_type.end_headers

    def publish_after_headers(handler) -> None:
        original_end_headers(handler)
        if handler.path == "/events":
            host.broker.publish(
                RunStarted(agent_id="agent", run_id="immediate", agent_name="agent")
            )

    handler_type.end_headers = publish_after_headers
    try:
        connection, response = _request(host, "GET", "/events", headers=_headers(host))
        try:
            frame = _await_sse(
                connection,
                response,
                lambda candidate: isinstance(candidate, tuple)
                and candidate[0] == "event",
                what="event published as response headers arrive",
            )
            if not isinstance(frame, tuple) or decode_event(frame[1]) != RunStarted(
                agent_id="agent", run_id="immediate", agent_name="agent"
            ):
                fail(f"event published with the SSE headers was lost: {frame!r}")
        finally:
            connection.close()
    finally:
        handler_type.end_headers = original_end_headers
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
            run_events = [event for event in events if isinstance(event, (RunStarted, RunFinished))]
            if not isinstance(run_events[0], RunStarted) or not isinstance(run_events[-1], RunFinished):
                fail(f"run did not emit RunStarted through RunFinished: {events!r}")
            if reply["run_id"] == run_events[0].run_id:
                fail(f"/prompt returned the runtime id rather than a host handle: {reply!r}")
        finally:
            connection.close()
    finally:
        host.close()


@check("host_server.assistant_text_reaches_the_stream")
def check_assistant_text_reaches_the_stream() -> None:
    reply = "ping"
    provider = FakeModelProvider(streams=[(
        TextDelta("pi"),
        TextDelta("ng"),
        StreamCompleted(ModelResponse(Message(Role.ASSISTANT, reply))),
    )])
    with tempfile.TemporaryDirectory() as temporary:
        with mock.patch.dict(os.environ, {"SYMPHONAI_SESSIONS_DIR": str(Path(temporary) / "sessions")}):
            host = _host(provider)
            try:
                connection, response = _subscribed_stream(host)
                try:
                    prompt_connection, prompt = _request(
                        host, "POST", "/prompt", body={"prompt": "say ping"}, headers=_headers(host)
                    )
                    try:
                        prompt.read()
                    finally:
                        prompt_connection.close()
                    if prompt.status != 200:
                        fail(f"prompt was not accepted: {prompt.status}")
                    deltas = []
                    while True:
                        frame = _await_sse(
                            connection,
                            response,
                            lambda candidate: isinstance(candidate, tuple) and candidate[0] == "event",
                            what="run event",
                        )
                        event = decode_event(frame[1])
                        if isinstance(event, AssistantTextDelta):
                            deltas.append(event.text)
                        if isinstance(event, RunFinished):
                            break
                    if not deltas or "".join(deltas) != reply:
                        fail(f"assistant text did not reach the event stream: {deltas!r}")
                finally:
                    connection.close()
            finally:
                host.close()


@check("host_server.leader_delegates")
def check_leader_delegates() -> None:
    provider = FakeModelProvider(streams=[
        [StreamCompleted(ModelResponse(Message(Role.ASSISTANT, tool_calls=[
            ToolCall("delegate", "dispatch_subagent", {"subagent_name": "worker", "task": "inspect"})
        ])))],
        [TextDelta("child"), StreamCompleted(ModelResponse(Message(Role.ASSISTANT, "")))],
        [TextDelta("leader"), StreamCompleted(ModelResponse(Message(Role.ASSISTANT, "")))],
    ])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        subscription = host.broker.subscribe()
        try:
            connection, response = _request(host, "POST", "/prompt", body={"prompt": "delegate"}, headers=_headers(host))
            try:
                if response.status != 200:
                    fail(f"host rejected delegation prompt: {response.status}")
                response.read()
            finally:
                connection.close()
            events = []
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                event = subscription.get(timeout=0.1)
                if event is not None:
                    events.append(event)
                    if isinstance(event, RunFinished) and event.agent_name == "leader":
                        break
            spawned = [event for event in events if isinstance(event, SubagentSpawned)]
            if len(spawned) != 1 or spawned[0].subagent_name != "worker":
                fail(f"leader dispatch did not publish a subagent spawn: {events!r}")
            leader_id = spawned[0].agent_id
            text = "".join(
                event.text for event in events
                if isinstance(event, AssistantTextDelta) and event.agent_id == leader_id
            )
            if text != "leader":
                fail(f"leader streamed text changed after delegation: {text!r}")
        finally:
            subscription.close()
            host.close()


@check("host_server.changes_report_and_revert")
def check_changes_report_and_revert() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "a.py"
        source.write_text("original\n")
        provider = FakeModelProvider(
            [
                ModelResponse(Message(Role.ASSISTANT, tool_calls=[
                    ToolCall("read-a-1", "read_file", {"path": "a.py"}),
                ])),
                ModelResponse(Message(Role.ASSISTANT, tool_calls=[
                    ToolCall("edit-a-1", "edit_file", {
                        "path": "a.py", "old_string": "original", "new_string": "prompt one",
                    }),
                ])),
                ModelResponse(Message(Role.ASSISTANT, tool_calls=[
                    ToolCall("write-b", "write_file", {"path": "b.py", "content": "new file\n"}),
                ])),
                ModelResponse(Message(Role.ASSISTANT, "first done")),
                ModelResponse(Message(Role.ASSISTANT, tool_calls=[
                    ToolCall("read-a-2", "read_file", {"path": "a.py"}),
                ])),
                ModelResponse(Message(Role.ASSISTANT, tool_calls=[
                    ToolCall("edit-a-2", "edit_file", {
                        "path": "a.py", "old_string": "prompt one", "new_string": "prompt two",
                    }),
                ])),
                ModelResponse(Message(Role.ASSISTANT, "second done")),
            ]
        )
        host = HostServer(
            provider,
            PermissionPolicy(repo_root=root, allowed_write_scope=[root], mode="allow"),
            sessions_root=root / "sessions",
            memory_extraction_enabled=False,
        )
        host.start()
        try:
            host.run.select_mode("allow")
            host.run.start("prompt one")
            _wait_until(lambda: not host.run.active, "first changes prompt did not finish")
            host.run.start("prompt two")
            _wait_until(lambda: not host.run.active, "second changes prompt did not finish")
            connection, response = _request(host, "GET", "/changes", headers=_headers(host))
            try:
                data = json.loads(response.read())
                if response.status != 200:
                    fail(f"changes route returned {response.status}: {data!r}")
            finally:
                connection.close()
            if (
                [turn["prompt"] for turn in data["turns"]] != ["prompt one", "prompt two"]
                or [turn["paths"] for turn in data["turns"]] != [["a.py", "b.py"], ["a.py"]]
                or [item["path"] for item in data["files"]] != ["a.py", "b.py"]
                or data["files"][0]["status"] != "modified"
                or "original" not in data["files"][0]["diff"]
                or "prompt two" not in data["files"][0]["diff"]
                or data["files"][1]["status"] != "added"
            ):
                leader_messages = host.run._conversation[0]._chat_messages
                tool_results = [
                    message.tool_result for message in leader_messages
                    if message.tool_result is not None
                ]
                fail(
                    "changes report did not describe both prompts: "
                    f"{data!r}; tool_results={tool_results!r}"
                )

            second_key = data["turns"][1]["key"]
            connection, response = _request(
                host, "POST", "/changes/revert", body={"key": second_key}, headers=_headers(host)
            )
            try:
                reverted = json.loads(response.read())
                if response.status != 200 or reverted != {"reverted": ["a.py"]}:
                    fail(f"prompt revert returned {response.status}: {reverted!r}")
            finally:
                connection.close()
            if source.read_text() != "prompt one\n" or (root / "b.py").read_text() != "new file\n":
                fail("reverting prompt two did not preserve prompt one's files")

            connection, response = _request(
                host, "POST", "/changes/revert", body={"path": "b.py"}, headers=_headers(host)
            )
            try:
                reverted_file = json.loads(response.read())
                if response.status != 200 or reverted_file != {"reverted": ["b.py"]}:
                    fail(f"file revert returned {response.status}: {reverted_file!r}")
            finally:
                connection.close()
            if (root / "b.py").exists():
                fail("reverting the added file did not delete it")
            connection, response = _request(host, "GET", "/changes", headers=_headers(host))
            try:
                after_reverts = json.loads(response.read())
                if response.status != 200 or [item["path"] for item in after_reverts["files"]] != ["a.py"]:
                    fail(f"changes report after reverts was incorrect: {after_reverts!r}")
            finally:
                connection.close()
        finally:
            host.close()


@check("host_server.prompt_checkpoint_key_live_and_reopened")
def check_prompt_checkpoint_key_live_and_reopened() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "a.py").write_text("alpha = 1\n")
        (root / "b.py").write_text("beta = 1\n")
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[
                ToolCall("read-a", "read_file", {"path": "a.py"}),
                ToolCall("read-b", "read_file", {"path": "b.py"}),
            ])),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[
                ToolCall("edit-a", "edit_file", {
                    "path": "a.py", "old_string": "alpha = 1", "new_string": "alpha = 2",
                }),
                ToolCall("edit-b", "edit_file", {
                    "path": "b.py", "old_string": "beta = 1", "new_string": "beta = 3",
                }),
            ])),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        host = HostServer(
            provider,
            PermissionPolicy(repo_root=root, allowed_write_scope=[root], mode="allow"),
            sessions_root=root / "sessions",
        )
        host.start()
        host.run.select_mode("allow")
        stream_connection, stream_response = _subscribed_stream(host)
        try:
            connection, response = _request(
                host, "POST", "/prompt", body={"prompt": "checkpoint prompt"},
                headers=_headers(host),
            )
            reply = json.loads(response.read())
            connection.close()
            if response.status != 200:
                fail(f"checkpoint prompt failed: {response.status}, {reply!r}")
            _wait_until(lambda: host.run._active is None, "checkpoint prompt did not finish")
            live = _await_sse(
                stream_connection, stream_response,
                lambda item: item[1].get("type") == "PromptSubmitted",
                what="live prompt checkpoint key",
            )[1]
            connection, response = _request(host, "GET", "/changes", headers=_headers(host))
            changes = json.loads(response.read())
            connection.close()
            if response.status != 200 or len(changes["turns"]) != 1:
                fail(f"checkpoint changes listing was invalid: {changes!r}")
            key = changes["turns"][0]["key"]
            if live.get("checkpoint_key") != key:
                fail(f"live prompt key did not match /changes: {live!r}, {changes!r}")

            connection, response = _request(
                host, "POST", "/session/open", body={"run_id": reply["run_id"]},
                headers=_headers(host),
            )
            response.read()
            connection.close()
            reopened = _await_sse(
                stream_connection, stream_response,
                lambda item: item[1].get("type") == "HistoryMessage"
                and item[1].get("role") == "user",
                what="reopened prompt checkpoint key",
            )[1]
            if reopened.get("checkpoint_key") != key:
                fail(f"reopened prompt key did not match /changes: {reopened!r}, {changes!r}")
            edits = reopened.get("edits")
            if edits != [
                {"path": "a.py", "added": 1, "removed": 1},
                {"path": "b.py", "added": 1, "removed": 1},
            ]:
                fail(f"reopened prompt edits did not match the two file diffs: {reopened!r}")
        finally:
            stream_connection.close()
            host.close()


@check("host_server.replayed_subagent_checkpoint_edits")
def check_replayed_subagent_checkpoint_edits() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "repo"
        root.mkdir()
        (root / "c.py").write_text("value = 1\n", encoding="utf-8")
        (root / "d.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
        subprocess.run(["git", "add", "c.py", "d.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "dispatch-shared", "dispatch_subagent",
                {"subagent_name": "worker", "task": "edit c.py"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "read-c", "read_file", {"path": "c.py"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "edit-c", "edit_file", {
                    "path": "c.py", "old_string": "value = 1", "new_string": "value = 2",
                },
            )])),
            ModelResponse(Message(Role.ASSISTANT, "shared edit done")),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "dispatch-worktree", "dispatch_subagent",
                {"subagent_name": "planner", "task": "edit d.py", "isolation": "worktree"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "read-d", "read_file", {"path": "d.py"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "edit-d", "edit_file", {
                    "path": "d.py", "old_string": "value = 1", "new_string": "value = 3",
                },
            )])),
            ModelResponse(Message(Role.ASSISTANT, "worktree edit done")),
            ModelResponse(Message(Role.ASSISTANT, "leader done")),
        ])
        host = HostServer(
            provider,
            PermissionPolicy(repo_root=root, allowed_write_scope=[root], mode="allow"),
            sessions_root=root / "sessions",
        )
        host.start()
        host.run.select_mode("allow")
        stream_connection, stream_response = _subscribed_stream(host)
        try:
            connection, response = _request(
                host, "POST", "/prompt", body={"prompt": "edit through subagents"},
                headers=_headers(host),
            )
            reply = json.loads(response.read())
            connection.close()
            if response.status != 200:
                fail(f"subagent edit prompt failed: {response.status}, {reply!r}")
            _wait_until(lambda: host.run._active is None, "subagent edit prompt did not finish")

            connection, response = _request(host, "GET", "/changes", headers=_headers(host))
            changes = json.loads(response.read())
            connection.close()
            if response.status != 200 or len(changes["turns"]) != 1:
                fail(f"subagent checkpoint changes were invalid: {changes!r}")
            turn = changes["turns"][0]
            if turn["paths"] != ["c.py"]:
                fail(f"checkpoint included the wrong subagent edits: {turn!r}")

            connection, response = _request(
                host, "POST", "/session/open", body={"run_id": reply["run_id"]},
                headers=_headers(host),
            )
            response.read()
            connection.close()
            replay = _await_sse(
                stream_connection, stream_response,
                lambda item: item[1].get("type") == "HistoryMessage"
                and item[1].get("role") == "user",
                what="subagent checkpoint edit replay",
            )[1]
            expected = [{"path": "c.py", "added": 1, "removed": 1}]
            if replay.get("checkpoint_key") != turn["key"] or replay.get("edits") != expected:
                fail(f"shared subagent edit did not replay under its checkpoint: {replay!r}")
            if any(edit.get("path") == "d.py" for edit in replay.get("edits", [])):
                fail(f"worktree subagent edit was included in checkpoint replay: {replay!r}")
        finally:
            stream_connection.close()
            host.close()


@check("host_server.changes_external_edit_refused")
def check_changes_external_edit_refused() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "a.py"
        source.write_text("original\n")
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[
                ToolCall("read-a", "read_file", {"path": "a.py"}),
            ])),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[
                ToolCall("edit-a", "edit_file", {
                    "path": "a.py", "old_string": "original", "new_string": "agent edit",
                }),
            ])),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        host = HostServer(
            provider,
            PermissionPolicy(repo_root=root, allowed_write_scope=[root], mode="allow"),
            sessions_root=root / "sessions",
        )
        host.start()
        try:
            host.run.select_mode("allow")
            host.run.start("edit a.py")
            _wait_until(lambda: not host.run.active, "external-edit prompt did not finish")
            source.write_text("hand edit\n")
            connection, response = _request(host, "GET", "/changes", headers=_headers(host))
            try:
                changes = json.loads(response.read())
                if response.status != 200 or changes["files"][0]["changed_outside"] is not True:
                    fail(f"manual edit was not marked outside: {changes!r}")
            finally:
                connection.close()
            connection, response = _request(
                host, "POST", "/changes/revert", body={"path": "a.py"}, headers=_headers(host)
            )
            try:
                conflict = json.loads(response.read())
                if response.status != 409 or conflict.get("paths") != ["a.py"]:
                    fail(f"outside edit was not refused: {response.status}, {conflict!r}")
            finally:
                connection.close()
            if source.read_text() != "hand edit\n":
                fail("refused revert changed the hand-edited file")
            connection, response = _request(
                host,
                "POST",
                "/changes/revert",
                body={"path": "a.py", "force": True},
                headers=_headers(host),
            )
            try:
                forced = json.loads(response.read())
                if response.status != 200 or forced != {"reverted": ["a.py"]}:
                    fail(f"forced revert returned {response.status}: {forced!r}")
            finally:
                connection.close()
            if source.read_text() != "original\n":
                fail("forced revert did not restore the checkpoint bytes")
        finally:
            host.close()


@check("host_server.changes_empty_active_and_invalid")
def check_changes_empty_active_and_invalid() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = HostServer(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            PermissionPolicy(repo_root=root),
            sessions_root=root / "sessions",
        )
        host.start()
        try:
            connection, response = _request(host, "GET", "/changes", headers=_headers(host, "bad"))
            try:
                if response.status != 401:
                    fail(f"unauthorized changes request returned {response.status}")
            finally:
                connection.close()
            connection, response = _request(host, "GET", "/changes", headers=_headers(host))
            try:
                empty = json.loads(response.read())
                if response.status != 200 or empty != {"turns": [], "files": [], "worktrees": []}:
                    fail(f"changes without a conversation were not empty: {empty!r}")
            finally:
                connection.close()
            for body in ({}, {"path": "a.py", "key": "key"}, {"path": 3}):
                connection, response = _request(
                    host, "POST", "/changes/revert", body=body, headers=_headers(host)
                )
                try:
                    response.read()
                    if response.status != 400:
                        fail(f"invalid revert body returned {response.status}: {body!r}")
                finally:
                    connection.close()
            for body in ({"path": "missing.py"}, {"key": "missing-key"}):
                connection, response = _request(
                    host, "POST", "/changes/revert", body=body, headers=_headers(host)
                )
                try:
                    response.read()
                    if response.status != 404:
                        fail(f"unknown revert target returned {response.status}: {body!r}")
                finally:
                    connection.close()
            active = type("Active", (), {"run_id": "active-run"})()
            host.run._active = active
            connection, response = _request(host, "GET", "/changes", headers=_headers(host))
            try:
                response.read()
                if response.status != 409:
                    fail(f"active changes request returned {response.status}")
            finally:
                connection.close()
            connection, response = _request(
                host, "POST", "/changes/revert", body={"path": "a.py"}, headers=_headers(host)
            )
            try:
                response.read()
                if response.status != 409:
                    fail(f"active revert request returned {response.status}")
            finally:
                connection.close()
            host.run._active = None
        finally:
            host.run._active = None
            host.close()


def _conversation_reply(host: HostServer) -> tuple[bytes, dict]:
    connection, response = _request(host, "GET", "/conversation", headers=_headers(host))
    try:
        body = response.read()
        if response.status != 200:
            fail(f"conversation route returned {response.status}: {body!r}")
        return body, json.loads(body)
    finally:
        connection.close()


def _wait_goal_state(host: HostServer, phase: str, *, timeout: float = 8) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, reply = _conversation_reply(host)
        goal = reply.get("conversation", {}).get("goal")
        if goal is not None and goal.get("phase") == phase:
            return goal
        time.sleep(0.02)
    fail(f"goal did not reach {phase!r}: {_conversation_reply(host)[1]!r}")


def _send_host_prompt(host: HostServer, prompt: str) -> None:
    connection, response = _request(
        host, "POST", "/prompt", body={"prompt": prompt}, headers=_headers(host)
    )
    try:
        body = response.read()
        if response.status != 200:
            fail(f"host rejected usage prompt: {response.status}, {body!r}")
    finally:
        connection.close()
    _wait_until(lambda: not host.run.active, "usage prompt did not finish")


class _BackgroundGoalProvider(ModelProvider):
    def __init__(self) -> None:
        self.goal_round_started = threading.Event()
        self.release_goal_round = threading.Event()
        self.feedback_round_started = threading.Event()
        self.release_feedback_round = threading.Event()

    @property
    def name(self) -> str:
        return "background-goal"

    @property
    def wire_format(self) -> int:
        return 4

    def create_response(self, request, *, cancel=None) -> ModelResponse:
        prompt = next(message.text for message in reversed(request.messages) if message.role is Role.USER)
        if prompt == "finish goal A":
            self.goal_round_started.set()
            gate = self.release_goal_round
        elif prompt.startswith("Goal check failed (round "):
            self.feedback_round_started.set()
            gate = self.release_feedback_round
        else:
            gate = None
        while gate is not None and not gate.wait(0.01):
            if cancel is not None:
                cancel.raise_if_cancelled()
        if cancel is not None:
            cancel.raise_if_cancelled()
        return ModelResponse(Message(Role.ASSISTANT, f"answer to {prompt}"))


def _background_goal_fixture(root: Path):
    provider = _BackgroundGoalProvider()
    host = _host(provider, repo_root=root, sessions_root=root / "sessions")
    checker = "import time; time.sleep(2); raise SystemExit(1)"
    connection, response = _request(
        host,
        "POST",
        "/goal",
        body={"objective": "finish goal A", "check": [sys.executable, "-c", checker], "max_rounds": 3},
        headers=_headers(host),
    )
    try:
        if response.status != 200:
            fail(f"background goal was rejected: {response.status}, {response.read()!r}")
        response.read()
    finally:
        connection.close()
    if not provider.goal_round_started.wait(2):
        fail("goal A did not start its first round")
    session_id = host.run._goal_session_id
    connection, response = _request(host, "POST", "/session/new", body={}, headers=_headers(host))
    try:
        response.read()
        if response.status != 200:
            fail(f"new conversation was refused while goal A ran: {response.status}")
    finally:
        connection.close()
    _send_host_prompt(host, "first prompt in B")
    provider.release_goal_round.set()
    _wait_until(
        lambda: session_id in host.run._goal_checks_by_session,
        "goal A did not enter its check while B was current",
    )
    return host, provider, session_id, host.run._goal_checks_by_session[session_id]


@check("host_server.background_goal_prompt_is_session_scoped")
def check_background_goal_prompt_is_session_scoped() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        host, provider, session_id, context = _background_goal_fixture(Path(temporary))
        try:
            if host.run._conversation[1].run_id == session_id:
                fail("conversation B was not current during A's check")
            _send_host_prompt(host, "second prompt in B")
            if context.interrupted or context.cancel.is_set():
                fail(f"a prompt in B interrupted goal A's check: {context.interrupted!r}, {context.cancel.is_set()}")
            if not provider.feedback_round_started.wait(4):
                fail("goal A did not start its next round after the failing check")
            goal = host.run._goals_by_session[session_id]
            if goal.phase != "active" or session_id not in host.run._active_by_session:
                fail(f"goal A did not remain active in its next round: {goal.payload()!r}")
        finally:
            host.close()


@check("host_server.background_goal_stop_is_session_scoped")
def check_background_goal_stop_is_session_scoped() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        host, provider, session_id, context = _background_goal_fixture(Path(temporary))
        try:
            connection, response = _request(host, "POST", "/stop", body={}, headers=_headers(host))
            response.read()
            connection.close()
            if response.status != 200 or context.cancel.is_set():
                fail("/stop in B cancelled A's goal check")
            connection, response = _request(
                host, "POST", "/session/open", body={"run_id": session_id}, headers=_headers(host)
            )
            response.read()
            connection.close()
            if response.status != 200:
                fail(f"opening A during its check failed: {response.status}")
            connection, response = _request(host, "POST", "/stop", body={}, headers=_headers(host))
            response.read()
            connection.close()
            goal = _wait_goal_state(host, "paused")
            if (
                response.status != 200
                or not context.cancel.is_set()
                or goal["reason"] != "cancelled"
            ):
                fail(f"/stop in A did not cancel A's own check: {goal!r}")
        finally:
            host.close()


@check("host_server.background_goal_new_chat_is_session_scoped")
def check_background_goal_new_chat_is_session_scoped() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        host, provider, session_id, context = _background_goal_fixture(Path(temporary))
        try:
            connection, response = _request(host, "POST", "/session/new", body={}, headers=_headers(host))
            response.read()
            connection.close()
            if response.status != 200 or context.interrupted or context.cancel.is_set():
                fail(
                    "new chat in B interrupted goal A's check: "
                    f"status={response.status}, interrupted={context.interrupted!r}, "
                    f"cancelled={context.cancel.is_set()}"
                )
            _send_host_prompt(host, "new chat prompt in B")
            if not provider.feedback_round_started.wait(4):
                fail("goal A did not continue after a new chat in B")
            goal = host.run._goals_by_session[session_id]
            if goal.phase != "active" or context.interrupted:
                fail(f"new chat in B changed goal A: {goal.payload()!r}")
        finally:
            host.close()


@check("host_server.goal_rounds_until_check_passes")
def check_goal_rounds_until_check_passes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        counter = root / "check-count"
        code = (
            "from pathlib import Path; import sys; "
            f"p=Path({str(counter)!r}); n=int(p.read_text())+1 if p.exists() else 1; "
            "p.write_text(str(n)); print(f'failed-{n}'); sys.exit(0 if n == 3 else 1)"
        )
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "round one")),
            ModelResponse(Message(Role.ASSISTANT, "round two")),
            ModelResponse(Message(Role.ASSISTANT, "round three")),
        ])
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        connection, response = _event_stream(host)
        try:
            request, reply = _request(
                host, "POST", "/goal",
                body={"objective": "finish the task", "check": [sys.executable, "-c", code]},
                headers=_headers(host),
            )
            try:
                accepted = json.loads(reply.read())
                if reply.status != 200 or accepted["goal"]["rounds"] != 1:
                    fail(f"goal was not accepted at round one: {reply.status}, {accepted!r}")
            finally:
                request.close()
            goal = _wait_goal_state(host, "complete")
            if goal["rounds"] != 3 or goal["last_check"] != {"exit": 0, "ok": True, "output": "failed-3\n"}:
                fail(f"goal did not complete on the passing third check: {goal!r}")
            if provider.call_count != 3 or counter.read_text(encoding="utf-8") != "3":
                fail(f"goal ran the wrong number of rounds or checks: {provider.call_count}")
            session_id = host.run._conversation[1].run_id
            store = SessionStore.open(host.run.sessions_root, session_id)
            try:
                users = [message.text for message in load_run(store).messages if message.role is Role.USER]
            finally:
                store.close()
            if len(users) != 3 or "round 1 of 10" not in users[1] or "failed-1" not in users[1] or "round 2 of 10" not in users[2] or "failed-2" not in users[2]:
                fail(f"subsequent goal prompts omitted check feedback: {users!r}")
            seen = []
            check_events = []
            goal_events = []
            while len(check_events) < 3:
                frame = _next_sse(connection, response, timeout=5)
                if frame == "keepalive":
                    continue
                kind, payload = frame
                if kind != "event":
                    continue
                seen.append(payload.get("type"))
                if payload.get("type") == "GoalChanged":
                    goal_events.append(payload)
                if payload.get("type") == "GoalChanged" and payload.get("change") == "check":
                    if seen.count("RunFinished") <= len(check_events):
                        fail(f"goal check event preceded its RunFinished: {seen!r}")
                    check_events.append(payload)
            if [event["phase"] for event in check_events] != ["active", "active", "complete"]:
                fail(f"goal check events had the wrong phases: {check_events!r}")
            if [event["change"] for event in goal_events] != ["set", "check", "check", "check"]:
                fail(f"goal state events were missing or out of order: {goal_events!r}")
        finally:
            connection.close()
            host.close()


@check("host_server.goal_round_limit_blocks")
def check_goal_round_limit_blocks() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        try:
            code = "print('still failing'); raise SystemExit(1)"
            connection, response = _request(
                host, "POST", "/goal",
                body={"objective": "finish", "check": [sys.executable, "-c", code], "max_rounds": 2},
                headers=_headers(host),
            )
            try:
                if response.status != 200:
                    fail(f"goal route rejected a valid two-round goal: {response.status}")
                response.read()
            finally:
                connection.close()
            goal = _wait_goal_state(host, "blocked")
            if goal["rounds"] != 2 or goal["reason"] != "rounds exhausted" or provider.call_count != 2:
                fail(f"goal round limit did not block after two rounds: {goal!r}, calls={provider.call_count}")
        finally:
            host.close()


@check("host_server.goal_agent_updates")
def check_goal_agent_updates() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "complete-goal", "update_goal",
                {"status": "complete", "message": "implemented the requested change"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "The requested change is complete.")),
        ])
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        try:
            connection, response = _request(
                host, "POST", "/goal", body={"objective": "finish"}, headers=_headers(host),
            )
            accepted = json.loads(response.read())
            connection.close()
            if response.status != 200 or accepted["goal"]["check"] != []:
                fail(f"check-less goal was not accepted: {response.status}, {accepted!r}")
            goal = _wait_goal_state(host, "complete")
            if goal["rounds"] != 1 or goal["reason"] != "implemented the requested change" or provider.call_count != 2:
                fail(f"agent completion did not finish the check-less goal in one round: {goal!r}")
        finally:
            host.close()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        marker = root / "checked"
        check_code = (
            "from pathlib import Path; import sys; "
            f"p=Path({str(marker)!r}); n=int(p.read_text())+1 if p.exists() else 1; "
            "p.write_text(str(n)); sys.exit(0 if n == 2 else 1)"
        )
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "defer-completion", "update_goal",
                {"status": "complete", "message": "the work looks done"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "Run the check.")),
            ModelResponse(Message(Role.ASSISTANT, "The check now passes.")),
        ])
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        try:
            connection, response = _request(
                host, "POST", "/goal",
                body={"objective": "finish", "check": [sys.executable, "-c", check_code]},
                headers=_headers(host),
            )
            response.read()
            connection.close()
            goal = _wait_goal_state(host, "complete")
            if goal["rounds"] != 2 or goal["last_check"]["exit"] != 0 or marker.read_text() != "2":
                fail(f"agent completion bypassed the configured check: {goal!r}")
        finally:
            host.close()


@check("host_server.goal_agent_blocked")
def check_goal_agent_blocked() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        marker = root / "should-not-run"
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "block-goal", "update_goal",
                {"status": "blocked", "message": "the upstream service is unavailable"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "I cannot finish without the service.")),
        ])
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        try:
            code = f"from pathlib import Path; Path({str(marker)!r}).touch()"
            connection, response = _request(
                host, "POST", "/goal",
                body={"objective": "finish", "check": [sys.executable, "-c", code]},
                headers=_headers(host),
            )
            response.read()
            connection.close()
            goal = _wait_goal_state(host, "blocked")
            if goal["reason"] != "the upstream service is unavailable" or marker.exists() or provider.call_count != 2:
                fail(f"agent block did not stop the goal loop before its check: {goal!r}")
        finally:
            host.close()


@check("host_server.goal_without_check_round_limit")
def check_goal_without_check_round_limit() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        class RecordingProvider(FakeModelProvider):
            def __init__(self) -> None:
                super().__init__([
                    ModelResponse(Message(Role.ASSISTANT, "still working")),
                    ModelResponse(Message(Role.ASSISTANT, "still not done")),
                ])
                self.requests = []

            def create_response(self, request, *, cancel=None):  # noqa: ANN001
                self.requests.append(request)
                return super().create_response(request, cancel=cancel)

        provider = RecordingProvider()
        run = HostRun(
            provider, PermissionPolicy(root), EventBroker(), sessions_root=root / "sessions",
            memory_extraction_enabled=False,
        )
        try:
            run.start_goal("finish the work", (), 2)
            deadline = time.monotonic() + 8
            goal = run.goal_snapshot()
            while time.monotonic() < deadline and (goal is None or goal["phase"] != "blocked"):
                time.sleep(0.02)
                goal = run.goal_snapshot()
            expected_prompt = (
                "Round 1 of 2 ended without the goal reported complete.\n"
                "Keep working toward the goal: finish the work\n"
                'If it is done, call update_goal with status "complete"; '
                'if you cannot finish it, call update_goal with status "blocked".'
            )
            user_prompts = [
                message.text
                for message in provider.requests[1].messages
                if message.role is Role.USER
            ] if len(provider.requests) > 1 else []
            if (
                goal is None
                or goal["phase"] != "blocked"
                or goal["rounds"] != 2
                or goal["reason"] != "rounds exhausted"
                or user_prompts[-1:] != [expected_prompt]
                or provider.call_count != 2
            ):
                fail(f"check-less goal did not continue and block at its limit: {goal!r}, {user_prompts!r}")
        finally:
            run.close()


@check("host_server.goal_tool_reads_and_rejects_inactive_updates")
def check_goal_tool_reads_and_rejects_inactive_updates() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sessions = root / "sessions"
        session_id = "goal-tool-session"
        session = SessionStore(sessions, session_id, repo_root=root)
        session.close()
        run = HostRun(
            FakeModelProvider(), PermissionPolicy(root), EventBroker(),
            sessions_root=sessions,
        )
        run._goal = goal_module.Goal("finish", (), rounds=1)
        run._goal_session_id = session_id
        run._goals_by_session[session_id] = run._goal
        tools = goal_module.goal_tools(
            lambda: run._goal_for_session(session_id),
            lambda status, message: run._update_goal_for_session(session_id, status, message),
        )
        read = tools["get_goal"].execute(
            ToolCall("read-goal", "get_goal", {}), run.policy,
        )
        if not read.ok or json.loads(read.content).get("objective") != "finish":
            fail(f"get_goal did not return the current goal JSON: {read!r}")

        run._goal.phase = "paused"
        run._save_goal(session_id, run._goal)
        paused = tools["update_goal"].execute(
            ToolCall("update-paused", "update_goal", {"status": "blocked", "message": "blocked"}),
            run.policy,
        )
        run._goal = None
        run._goal_session_id = None
        run._goals_by_session.pop(session_id, None)
        missing = tools["update_goal"].execute(
            ToolCall("update-missing", "update_goal", {"status": "complete", "message": "done"}),
            run.policy,
        )
        if paused.ok or "paused" not in (paused.error or ""):
            fail(f"update_goal accepted a paused goal: {paused!r}")
        if missing.ok or "No goal" not in (missing.error or ""):
            fail(f"update_goal accepted a missing goal: {missing!r}")


@check("host_server.plan_tool")
def check_plan_tool() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sessions = root / "sessions"
        plan = [
            {"step": "Read the spec", "status": "completed"},
            {"step": "Implement the feature", "status": "in_progress"},
            {"step": "Review the result", "status": "pending"},
        ]
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "plan-call", "update_plan", {"plan": plan, "explanation": "Three steps."},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        broker = EventBroker()
        subscription = broker.subscribe()
        run = HostRun(provider, PermissionPolicy(root), broker, sessions_root=sessions)
        closed = False
        try:
            run.start("make a multi-step change")
            _wait_until(lambda: not run.active, "plan tool run did not finish")
            session_id = run._conversation[1].run_id
            expected = {"steps": plan, "explanation": "Three steps."}
            conversation = run.conversation_stats()
            if conversation is None or conversation.get("plan") != expected:
                fail(f"conversation stats lost the plan: {conversation!r}")
            changed = []
            while (event := subscription.get(timeout=0.05)) is not None:
                if isinstance(event, PlanChanged):
                    changed.append(event)
            if len(changed) != 1 or changed[0].steps != plan or changed[0].session_id != session_id:
                fail(f"plan update did not publish PlanChanged: {changed!r}")
            run.close()
            closed = True
            session = SessionStore.open(sessions, session_id)
            try:
                if session.read_meta().get("plan") != expected:
                    fail("plan was not saved in session metadata")
            finally:
                session.close()

            published = []
            tool = plan_module.plan_tools(lambda steps, explanation: published.append((steps, explanation)))["update_plan"]
            invalid = (
                {"plan": []},
                {"plan": [*plan[:2], {"step": "third", "status": "in_progress"}]},
                {"plan": [{"step": "x" * 201, "status": "pending"}]},
                {"plan": [{"step": "one", "status": "pending", "extra": True}]},
            )
            for index, arguments in enumerate(invalid):
                result = tool.execute(ToolCall(f"invalid-{index}", "update_plan", arguments), run.policy)
                if result.ok:
                    fail(f"invalid plan was accepted: {arguments!r}")
            if published:
                fail("invalid plans invoked the update callback")
        finally:
            subscription.close()
            if not closed:
                run.close()


@check("host_server.ask_user_round_trip")
def check_ask_user_round_trip() -> None:
    checked = []
    question_tool = questions_module.question_tools(
        lambda prompt, options, tool_call_id: checked.append((prompt, options, tool_call_id))
    )["ask_user"]
    for index, arguments in enumerate((
        {"question": ""},
        {"question": "Pick", "options": ["same", "same"]},
        {"question": "Pick", "options": ["only one"]},
        {"question": "Pick", "unexpected": True},
    )):
        invalid = question_tool.execute(
            ToolCall(f"invalid-question-{index}", "ask_user", arguments),
            PermissionPolicy(REPO_ROOT),
        )
        if invalid.ok:
            fail(f"invalid ask_user arguments were accepted: {arguments!r}")
    if checked:
        fail("invalid ask_user arguments invoked the answer callback")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        def post(host, path: str, body: dict) -> tuple[int, dict]:  # noqa: ANN001
            connection, response = _request(
                host, "POST", path, body=body, headers=_headers(host),
            )
            try:
                raw = response.read()
                return response.status, json.loads(raw) if raw else {}
            finally:
                connection.close()

        class RecordingProvider(FakeModelProvider):
            def __init__(self, responses):
                super().__init__(responses)
                self.requests = []

            def create_response(self, request, *, cancel=None):
                self.requests.append(request)
                return super().create_response(request, cancel=cancel)

        provider = RecordingProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "ask-one", "ask_user", {"question": "Choose a file", "options": ["a.py", "b.py"]},
            )])),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "ask-two", "ask_user", {"question": "Continue?", "options": ["yes", "no"]},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        host = HostServer(
            provider, PermissionPolicy(root), sessions_root=root / "sessions",
            approval_timeout=2,
        )
        host.start()
        stream_connection = None
        try:
            stream_connection, stream_response = _subscribed_stream(host)
            host.run.start("ask me before choosing")
            first = _await_sse(
                stream_connection, stream_response,
                lambda item: isinstance(item, tuple)
                and item[0] == "approval_requested" and item[1].get("kind") == "question",
                what="first user question",
            )[1]
            if (
                first.get("operation") != "ask_user"
                or first.get("target") != "Choose a file"
                or first.get("options") != ["a.py", "b.py"]
            ):
                fail(f"question request carried the wrong fields: {first!r}")
            status, reply = post(
                host, "/approval", {"approval_id": first["approval_id"], "allowed": True, "answer": ""},
            )
            if status != 400:
                fail(f"blank allowed answer returned {status}: {reply!r}")
            status, reply = post(
                host, "/approval", {"approval_id": first["approval_id"], "allowed": True, "answer": "B"},
            )
            if status != 200 or reply != {"resolved": True}:
                fail(f"non-blank question answer was not accepted: {status}, {reply!r}")

            second = _await_sse(
                stream_connection, stream_response,
                lambda item: isinstance(item, tuple)
                and item[0] == "approval_requested" and item[1].get("approval_id") != first["approval_id"],
                what="second user question",
            )[1]
            status, reply = post(
                host, "/approval", {"approval_id": second["approval_id"], "allowed": False},
            )
            if status != 200 or reply != {"resolved": True}:
                fail(f"question skip was not accepted: {status}, {reply!r}")
            _wait_until(lambda: not host.run.active, "question run did not finish")
            tool_results = [
                message.tool_result.content
                for request in provider.requests
                for message in request.messages
                if message.tool_result is not None
            ]
            if (
                "The person answered: B" not in tool_results
                or "The person skipped the question." not in tool_results
            ):
                fail(f"question answers did not reach the model: {tool_results!r}")
        finally:
            if stream_connection is not None:
                stream_connection.close()
            host.close()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
            "ask-stop", "ask_user", {"question": "Continue?"},
        )]))])
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        connection = None
        try:
            connection, response = _subscribed_stream(host)
            host.run.start("wait for my answer")
            question = _await_sse(
                connection, response,
                lambda item: isinstance(item, tuple)
                and item[0] == "approval_requested" and item[1].get("kind") == "question",
                what="question before stop",
            )[1]
            status, reply = post(host, "/stop", {"reason": "person stopped"})
            if status != 200 or reply.get("accepted") is not True:
                fail(f"stop did not accept a waiting question: {status}, {reply!r}")
            _wait_until(lambda: not host.run.active, "stopped question run did not finish")
            if host.run.pending_approvals():
                fail(f"stopped question remained pending: {question!r}")
        finally:
            if connection is not None:
                connection.close()
            host.close()


@check("host_server.questions_carry_their_tool_call")
def check_questions_carry_their_tool_call() -> None:
    for call, mode, kind, answer in (
        (ToolCall("q-1", "ask_user", {"question": "Continue?"}), "ask", "question", "yes"),
        (ToolCall("p-1", "propose_plan", {"plan": "# Make the change"}), "plan", "plan", "ask"),
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = FakeModelProvider([
                ModelResponse(Message(Role.ASSISTANT, tool_calls=[call])),
                ModelResponse(Message(Role.ASSISTANT, "done")),
            ])
            host = HostServer(
                provider, PermissionPolicy(root), sessions_root=root / "sessions",
                memory_extraction_enabled=False,
            )
            host.start()
            stream_connection = None
            try:
                host.run.select_mode(mode)
                stream_connection, stream_response = _subscribed_stream(host)
                host.run.start("ask for my decision")
                frame = _await_sse(
                    stream_connection, stream_response,
                    lambda item: isinstance(item, tuple)
                    and item[0] == "approval_requested" and item[1].get("kind") == kind,
                    what=f"{kind} with its tool call id",
                )[1]
                if frame.get("tool_call_id") != call.id:
                    fail(f"{call.name} event lost its tool call id: {frame!r}")
                connection, response = _request(host, "GET", "/approvals", headers=_headers(host))
                try:
                    pending = json.loads(response.read())
                    if response.status != 200:
                        fail(f"pending {kind} request failed: {response.status}, {pending!r}")
                finally:
                    connection.close()
                approvals = pending.get("pending", [])
                if (
                    len(approvals) != 1
                    or approvals[0].get("approval_id") != frame["approval_id"]
                    or approvals[0].get("tool_call_id") != call.id
                ):
                    fail(f"{call.name} pending approval lost its tool call id: {pending!r}")
                connection, response = _request(host, "POST", "/approval", headers=_headers(host), body={
                    "approval_id": frame["approval_id"], "allowed": True, "answer": answer,
                })
                try:
                    resolved = json.loads(response.read())
                    if response.status != 200 or resolved != {"resolved": True}:
                        fail(f"{kind} answer failed: {response.status}, {resolved!r}")
                finally:
                    connection.close()
                _wait_until(lambda: not host.run.active, f"{kind} run did not finish")
            finally:
                if stream_connection is not None:
                    stream_connection.close()
                host.close()


@check("host_server.plan_approval")
def check_plan_approval() -> None:
    plan_text = "# Change parser\n\n1. Update token handling."

    def post(host, path: str, body: dict) -> tuple[int, dict]:  # noqa: ANN001
        connection, response = _request(host, "POST", path, body=body, headers=_headers(host))
        try:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
        finally:
            connection.close()

    def run_plan(answer: str, *, reason: str = "", modes=("ask", "plan", "allow")):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        loaded_extensions = load_extensions(repo_root=root, home=root / "home")
        extensions = replace(loaded_extensions, ceiling=replace(loaded_extensions.ceiling, modes=modes))

        class RecordingProvider(FakeModelProvider):
            def __init__(self, responses):
                super().__init__(responses)
                self.requests = []

            def create_response(self, request, *, cancel=None):
                self.requests.append(request)
                return super().create_response(request, cancel=cancel)

        provider = RecordingProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "propose", "propose_plan", {"plan": plan_text},
            )])),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "write", "write_file", {"path": "result.txt", "content": "written"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        host = HostServer(
            provider,
            PermissionPolicy(root, allowed_write_scope=[root]),
            sessions_root=root / "sessions",
            extensions=extensions,
            approval_timeout=2,
            memory_extraction_enabled=False,
        )
        host.start()
        connection = None
        try:
            host.run.select_mode("plan")
            connection, response = _subscribed_stream(host)
            host.run.start("make the change")
            proposal = _await_sse(
                connection, response,
                lambda item: isinstance(item, tuple)
                and item[0] == "approval_requested" and item[1].get("kind") == "plan",
                what="plan proposal",
            )[1]
            expected_options = [mode for mode in ("ask", "allow") if mode in modes]
            if proposal.get("target") != plan_text or proposal.get("options") != expected_options:
                fail(f"plan proposal carried wrong Markdown or choices: {proposal!r}")
            status, body = post(host, "/approval", {
                "approval_id": proposal["approval_id"], "allowed": True, "answer": answer,
            })
            if answer not in expected_options:
                if status != 400:
                    fail(f"unoffered mode {answer!r} returned {status}: {body!r}")
                host.run.stop()
                _wait_until(lambda: not host.run.active, "invalid plan mode did not stop")
                return provider, host, connection, response, temporary
            if status != 200:
                fail(f"plan approval failed: {status}, {body!r}")
            if answer == "ask":
                write_approval = _await_sse(
                    connection, response,
                    lambda item: isinstance(item, tuple)
                    and item[0] == "approval_requested" and item[1].get("operation") == "write_file",
                    what="write approval after choosing ask",
                )[1]
                status, body = post(host, "/approval", {
                    "approval_id": write_approval["approval_id"], "allowed": True,
                })
                if status != 200:
                    fail(f"write approval after choosing ask failed: {status}, {body!r}")
            _wait_until(lambda: not host.run.active, "plan approval run did not finish")
            connection2, response2 = _request(host, "GET", "/conversation", headers=_headers(host))
            try:
                stats = json.loads(response2.read()).get("conversation")
            finally:
                connection2.close()
            if stats is None or stats.get("mode") != answer:
                fail(f"conversation mode did not change to {answer!r}: {stats!r}")
            if answer == "allow" and not (root / "result.txt").exists():
                fail("write did not run after approving allow mode")
            return provider, host, connection, response, temporary
        except BaseException:
            if connection is not None:
                connection.close()
            host.close()
            temporary.cleanup()
            raise

    provider, host, connection, response, temporary = run_plan("ask")
    try:
        results = [
            message.tool_result.content
            for request in provider.requests
            for message in request.messages
            if message.tool_result is not None
        ]
        if not any("The person approved the plan. The conversation is now in ask mode" in item for item in results):
            fail(f"plan approval did not report ask mode: {results!r}")
    finally:
        connection.close()
        host.close()
        temporary.cleanup()

    published = []
    proposal_tool = questions_module.plan_proposal_tools(
        lambda approval: published.append(approval) or (
            PermissionDecision.deny("smaller steps", denial=DenialReason.DENIED_BY_USER), "",
        ),
        lambda: ("ask", "allow"),
    )["propose_plan"]
    policy = PermissionPolicy(Path.cwd(), mode="plan")
    kept = proposal_tool.execute(
        ToolCall("keep-plan", "propose_plan", {"plan": plan_text}), policy,
    )
    if not kept.ok or kept.content != "The person wants to keep planning: smaller steps" or policy.mode != "plan":
        fail(f"plan feedback did not keep the conversation in plan mode: {kept!r}")

    published.clear()
    outside = proposal_tool.execute(
        ToolCall("outside-plan", "propose_plan", {"plan": plan_text}),
        PermissionPolicy(Path.cwd(), mode="ask"),
    )
    if outside.ok or outside.error != "propose_plan is for plan mode; this conversation is in ask mode" or published:
        fail(f"propose_plan ran outside plan mode: {outside!r}, {published!r}")

    provider, host, connection, response, temporary = run_plan("allow")
    try:
        results = [
            message.tool_result.content
            for request in provider.requests
            for message in request.messages
            if message.tool_result is not None
        ]
        if not any("now in allow mode" in item for item in results):
            fail(f"plan approval did not report allow mode: {results!r}")
    finally:
        connection.close()
        host.close()
        temporary.cleanup()

    provider, host, connection, response, temporary = run_plan("allow", modes=("ask", "plan"))
    try:
        if provider.call_count != 1:
            fail("a disallowed plan mode reached the next model request")
    finally:
        connection.close()
        host.close()
        temporary.cleanup()


@check("host_server.goal_stop_and_resume")
def check_goal_stop_and_resume() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = _WaitingProvider()
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        marker = root / "checked"
        code = f"from pathlib import Path; Path({str(marker)!r}).write_text('ok')"
        try:
            connection, response = _request(
                host, "POST", "/goal",
                body={"objective": "finish", "check": [sys.executable, "-c", code]},
                headers=_headers(host),
            )
            response.read()
            connection.close()
            _wait_until(lambda: host.run.active, "goal round did not start")
            connection, response = _request(host, "POST", "/stop", body={}, headers=_headers(host))
            response.read()
            connection.close()
            goal = _wait_goal_state(host, "paused")
            if goal["reason"] != "cancelled" or marker.exists():
                fail(f"stopped goal round did not pause without a check: {goal!r}")
            connection, response = _request(
                host, "POST", "/goal/state", body={"action": "resume"}, headers=_headers(host),
            )
            response.read()
            connection.close()
            goal = _wait_goal_state(host, "complete")
            if not marker.exists() or goal["last_check"]["exit"] != 0:
                fail(f"resuming the goal did not run its check: {goal!r}")
        finally:
            provider.release.set()
            host.close()


@check("host_server.goal_resume_mid_round")
def check_goal_resume_mid_round() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        counter = root / "check-count"
        code = (
            "from pathlib import Path; import sys; "
            f"p=Path({str(counter)!r}); n=int(p.read_text())+1 if p.exists() else 1; "
            "p.write_text(str(n)); sys.exit(0 if n == 2 else 1)"
        )
        provider = _WaitingProvider()
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        try:
            connection, response = _request(
                host, "POST", "/goal",
                body={"objective": "finish", "check": [sys.executable, "-c", code]},
                headers=_headers(host),
            )
            response.read()
            connection.close()
            _wait_until(lambda: host.run.active, "goal round did not start")
            connection, response = _request(
                host, "POST", "/goal/state", body={"action": "pause"}, headers=_headers(host),
            )
            response.read()
            connection.close()
            connection, response = _request(
                host, "POST", "/goal/state", body={"action": "resume"}, headers=_headers(host),
            )
            resumed = json.loads(response.read())
            connection.close()
            if response.status != 200 or resumed["goal"]["phase"] != "active":
                fail(f"mid-round resume was not accepted: {response.status}, {resumed!r}")
            provider.release.set()
            goal = _wait_goal_state(host, "complete")
            if goal["rounds"] != 2 or goal["last_check"]["exit"] != 0 or counter.read_text() != "2":
                fail(f"resumed goal did not check round one and continue: {goal!r}")
        finally:
            provider.release.set()
            host.close()


@check("host_server.goal_pause_mid_round_stays_paused")
def check_goal_pause_mid_round_stays_paused() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        marker = root / "check-ran"
        code = f"from pathlib import Path; Path({str(marker)!r}).touch()"
        provider = _WaitingProvider()
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        try:
            connection, response = _request(
                host, "POST", "/goal",
                body={"objective": "finish", "check": [sys.executable, "-c", code]},
                headers=_headers(host),
            )
            response.read()
            connection.close()
            _wait_until(lambda: host.run.active, "goal round did not start")
            connection, response = _request(
                host, "POST", "/goal/state", body={"action": "pause"}, headers=_headers(host),
            )
            response.read()
            connection.close()
            provider.release.set()
            goal = _wait_goal_state(host, "paused")
            _wait_until(lambda: not host.run.active, "paused goal round did not finish")
            if goal["rounds"] != 1 or goal["reason"] != "paused" or marker.exists():
                fail(f"a paused round ran its check or changed state: {goal!r}")
        finally:
            provider.release.set()
            host.close()


@check("host_server.goal_check_timeout_kills_group")
def check_goal_check_timeout_kills_group() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        marker = root / "child-finished"
        child = (
            "import time; from pathlib import Path; time.sleep(0.7); "
            f"Path({str(marker)!r}).write_text('alive')"
        )
        parent = (
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(10)"
        )
        host = _host(repo_root=root, sessions_root=root / "sessions")
        try:
            with mock.patch.object(goal_module, "GOAL_CHECK_TIMEOUT_SECONDS", 0.1):
                connection, response = _request(
                    host, "POST", "/goal",
                    body={"objective": "finish", "check": [sys.executable, "-c", parent], "max_rounds": 1},
                    headers=_headers(host),
                )
                response.read()
                connection.close()
                goal = _wait_goal_state(host, "blocked")
            time.sleep(0.8)
            if goal["last_check"]["exit"] is not None or marker.exists():
                fail(f"timed out goal check did not kill its process group: {goal!r}, child={marker.exists()}")
        finally:
            host.close()


@check("host_server.goal_pause_and_clear")
def check_goal_pause_and_clear() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = _WaitingProvider()
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        try:
            connection, response = _request(
                host, "POST", "/goal",
                body={"objective": "finish", "check": [sys.executable, "-c", "pass"]},
                headers=_headers(host),
            )
            response.read()
            connection.close()
            _wait_until(lambda: host.run.active, "goal round did not start")
            connection, response = _request(
                host, "POST", "/goal/state", body={"action": "pause"}, headers=_headers(host),
            )
            paused = json.loads(response.read()).get("goal")
            connection.close()
            if response.status != 200 or paused["phase"] != "paused":
                fail(f"goal pause route failed: {response.status}, {paused!r}")
            connection, response = _request(
                host, "POST", "/goal/state", body={"action": "clear"}, headers=_headers(host),
            )
            cleared = json.loads(response.read()).get("goal", "missing")
            connection.close()
            if response.status != 200 or cleared is not None:
                fail(f"goal clear route failed: {response.status}, {cleared!r}")
            provider.release.set()
            _wait_until(lambda: not host.run.active, "cleared goal round did not finish")
            if host.run.conversation_stats()["goal"] is not None:
                fail("cleared goal returned to conversation state")
        finally:
            provider.release.set()
            host.close()


@check("host_server.goal_check_interrupted_by_prompt")
def check_goal_check_interrupted_by_prompt() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
        host = _host(provider, repo_root=root, sessions_root=root / "sessions")
        try:
            code = "import time; print('failed'); time.sleep(0.4); raise SystemExit(1)"
            connection, response = _request(
                host, "POST", "/goal",
                body={"objective": "finish", "check": [sys.executable, "-c", code]},
                headers=_headers(host),
            )
            response.read()
            connection.close()
            _wait_until(lambda: host.run._goal_check is not None and host.run._goal_check.process is not None, "goal check did not start")
            connection, response = _request(
                host, "POST", "/prompt", body={"prompt": "continue manually"}, headers=_headers(host),
            )
            if response.status != 200:
                fail(f"prompt during goal check was rejected: {response.status}")
            response.read()
            connection.close()
            goal = _wait_goal_state(host, "paused")
            _wait_until(lambda: not host.run.active, "manual prompt did not finish")
            if goal["reason"] != "interrupted" or provider.call_count != 2:
                fail(f"failed check continued after a new prompt: {goal!r}, calls={provider.call_count}")
        finally:
            host.close()


@check("host_server.goal_routes_validate")
def check_goal_routes_validate() -> None:
    provider = _WaitingProvider()
    host = _host(provider)
    try:
        for body in (
            {},
            {"objective": " ", "check": [sys.executable]},
            {"objective": "x", "check": None},
            {"objective": "x", "check": [""]},
            {"objective": "x", "check": [sys.executable], "max_rounds": True},
            {"objective": "x", "check": [sys.executable], "max_rounds": 101},
            {"objective": "x", "check": [sys.executable], "other": 1},
        ):
            connection, response = _request(host, "POST", "/goal", body=body, headers=_headers(host))
            response.read()
            connection.close()
            if response.status != 400:
                fail(f"invalid goal body was accepted: {body!r}, status={response.status}")
        connection, response = _request(
            host, "POST", "/goal/state", body={"action": "pause"}, headers=_headers(host),
        )
        response.read()
        connection.close()
        if response.status != 404:
            fail(f"goal state route without a goal returned {response.status}")
        connection, response = _request(
            host, "POST", "/prompt", body={"prompt": "busy"}, headers=_headers(host),
        )
        response.read()
        connection.close()
        _wait_until(lambda: host.run.active, "busy prompt did not start")
        connection, response = _request(
            host, "POST", "/goal",
            body={"objective": "x", "check": [sys.executable, "-c", "pass"]},
            headers=_headers(host),
        )
        response.read()
        connection.close()
        if response.status != 409:
            fail(f"goal route during a run returned {response.status}")
    finally:
        provider.release.set()
        host.run.stop()
        host.close()


@check("host_server.session_routes_clear_shell_grants")
def check_session_routes_clear_shell_grants() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "first")),
            ModelResponse(Message(Role.ASSISTANT, "second")),
        ])
        host = HostServer(
            provider,
            PermissionPolicy(root),
            sessions_root=root / "sessions",
        )
        host.start()
        stream_connection = None
        try:
            stream_connection, stream_response = _subscribed_stream(host)

            def shell_question():
                result = []
                thread = threading.Thread(
                    target=lambda: result.append(host.run.policy.check_shell(["pytest", "-x"]))
                )
                thread.start()
                frame = _await_sse(
                    stream_connection,
                    stream_response,
                    lambda candidate: isinstance(candidate, tuple)
                    and candidate[0] == "approval_requested",
                    what="shell approval after session transition",
                )
                return result, thread, frame[1]

            def answer_shell(result, thread, payload, *, remember):
                if payload.get("remember") != "pytest":
                    fail(f"shell approval offered the wrong grant prefix: {payload!r}")
                connection, response = _request(
                    host,
                    "POST",
                    "/approval",
                    body={
                        "approval_id": payload["approval_id"],
                        "allowed": remember,
                        "reason": "no" if not remember else "",
                        "remember": remember,
                    },
                    headers=_headers(host),
                )
                try:
                    body = response.read()
                    if response.status != 200:
                        fail(f"shell approval answer failed: {response.status}, {body!r}")
                finally:
                    connection.close()
                thread.join(1)
                if thread.is_alive() or not result or result[0].allowed != remember:
                    fail(f"shell approval response was not applied: {result!r}")

            def grant_and_check_transition():
                result, thread, payload = shell_question()
                answer_shell(result, thread, payload, remember=True)

            def require_prompt_after_transition():
                result, thread, payload = shell_question()
                answer_shell(result, thread, payload, remember=False)

            _send_host_prompt(host, "first session")
            first_session_id = host.run._conversation[1].run_id
            grant_and_check_transition()
            connection, response = _request(
                host, "POST", "/session/new", body={}, headers=_headers(host)
            )
            try:
                if response.status != 200:
                    fail(f"session/new failed: {response.status}, {response.read()!r}")
                response.read()
            finally:
                connection.close()
            require_prompt_after_transition()

            _send_host_prompt(host, "second session")
            grant_and_check_transition()
            connection, response = _request(
                host,
                "POST",
                "/session/open",
                body={"run_id": first_session_id},
                headers=_headers(host),
            )
            try:
                if response.status != 200:
                    fail(f"session/open failed: {response.status}, {response.read()!r}")
                response.read()
            finally:
                connection.close()
            require_prompt_after_transition()

            store = SessionStore.open(root / "sessions", first_session_id)
            try:
                record_id = load_run(store).record_ids[0]
            finally:
                store.close()
            grant_and_check_transition()
            connection, response = _request(
                host,
                "POST",
                "/session/fork",
                body={"run_id": first_session_id, "record_id": record_id},
                headers=_headers(host),
            )
            try:
                if response.status != 200:
                    fail(f"session/fork failed: {response.status}, {response.read()!r}")
                response.read()
            finally:
                connection.close()
            require_prompt_after_transition()
        finally:
            if stream_connection is not None:
                stream_connection.close()
            host.close()


@check("host_server.project_instructions_seeded")
def check_project_instructions_seeded() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "CLAUDE.md").write_text("legacy rule must stay absent", encoding="utf-8")
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
        with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": str(root / "missing-user-home")}):
            host = HostServer(provider, PermissionPolicy(root), system_prompt="host baseline", sessions_root=root / "sessions", memory_extraction_enabled=False)
            host.start()
            try:
                with mock.patch.object(provider, "create_response", wraps=provider.create_response) as response_spy:
                    _send_host_prompt(host, "plain")
                    plain = [(message.role.value, message.text) for message in response_spy.call_args.args[0].messages]
                    plain = [item for item in plain if "Memory persists across conversations" not in (item[1] or "")]
                    if (
                        len(plain) != 3
                        or plain[0] != ("system", "host baseline")
                        or plain[1][0] != "system"
                        or not plain[1][1].startswith("Environment when this conversation started")
                        or "In the person's messages, @<path> names a file in this repository. Read it with read_file before relying on its contents." not in plain[1][1]
                        or plain[2] != ("user", "plain")
                    ):
                        fail(f"empty hierarchy or CLAUDE.md changed the provider request: {plain!r}")
                    host.run.end_conversation()
                    instructions = root / ".symphonai" / "INSTRUCTIONS.md"
                    instructions.parent.mkdir()
                    instructions.write_text("project convention", encoding="utf-8")
                    _send_host_prompt(host, "with instructions")
                    sent = [(message.role.value, message.text) for message in response_spy.call_args.args[0].messages]
                    sent = [item for item in sent if "Memory persists across conversations" not in (item[1] or "")]
                    if (
                        len(sent) != 4
                        or sent[0] != ("system", "host baseline")
                        or sent[1] != ("system", "# instructions: project .symphonai/INSTRUCTIONS.md\nproject convention")
                        or sent[2][0] != "system"
                        or not sent[2][1].startswith("Environment when this conversation started")
                        or "In the person's messages, @<path> names a file in this repository. Read it with read_file before relying on its contents." not in sent[2][1]
                        or sent[3] != ("user", "with instructions")
                    ):
                        fail(f"project instructions or system prompt missed the first request: {sent!r}")
            finally:
                host.close()


@check("host_server.environment_seeded_once")
def check_environment_seeded_once() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        instruction_path = root / ".symphonai" / "INSTRUCTIONS.md"
        instruction_path.parent.mkdir()
        instruction_path.write_text("project convention", encoding="utf-8")
        environment_text = (
            "Environment when this conversation started (it does not update):\n"
            "- Working directory: /fixed/workdir\n"
            "- Repository root: /fixed/repo\n"
            "- Platform: fixed platform\n"
            "- Date: 2026-10-02\n"
            "- Model: fake test-model"
        )
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "first answer")),
            ModelResponse(Message(Role.ASSISTANT, "second answer")),
        ])
        with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": str(root / "missing-home")}), mock.patch(
            "symphonai_host.run.capture_environment", return_value=environment_text
        ) as capture:
            host = HostServer(
                provider,
                PermissionPolicy(root),
                sessions_root=root / "sessions",
            )
            host.start()
            try:
                with mock.patch.object(
                    provider, "create_response", wraps=provider.create_response
                ) as response_spy:
                    _send_host_prompt(host, "first")
                    first = list(response_spy.call_args.args[0].messages)
                    _send_host_prompt(host, "second")
                    second = list(response_spy.call_args.args[0].messages)
                for messages in (first, second):
                    systems = [message.text for message in messages if message.role == Role.SYSTEM]
                    memory_messages = [text for text in systems if "Memory persists across conversations" in text]
                    systems = [text for text in systems if "Memory persists across conversations" not in text]
                    environment_messages = [
                        text for text in systems
                        if text.startswith("Environment when this conversation started")
                    ]
                    if (
                        len(memory_messages) != 1
                        or len(systems) != 2
                        or "project convention" not in systems[0]
                        or not systems[1].startswith(environment_text)
                        or "In the person's messages, @<path> names a file in this repository. Read it with read_file before relying on its contents." not in systems[1]
                        or len(environment_messages) != 1
                    ):
                        fail(f"request did not carry one memory guide and one environment block after instructions: {messages!r}")
                if capture.call_count != 1:
                    fail(f"environment was captured {capture.call_count} times in one conversation")
                first_system = [message for message in first if message.role == Role.SYSTEM]
                second_system = [message for message in second if message.role == Role.SYSTEM]
                if first_system != second_system:
                    fail(f"conversation environment changed between prompts: {first_system!r}, {second_system!r}")
            finally:
                host.close()


@check("host_server.instruction_scope_and_warning")
def check_instruction_scope_and_warning() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        nested = root / "src"
        for directory in (root, nested):
            (directory / ".symphonai").mkdir(parents=True)
        project_text = "project rule\n" + "x" * MAX_INSTRUCTION_FILE_CHARS
        (root / ".symphonai" / "INSTRUCTIONS.md").write_text(project_text, encoding="utf-8")
        (nested / ".symphonai" / "INSTRUCTIONS.md").write_text("directory rule", encoding="utf-8")
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
        with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": str(root / "missing-user-home")}):
            host = HostServer(provider, PermissionPolicy(root), working_dir=nested, sessions_root=root / "sessions", chat_token_budget=1_000_000, memory_extraction_enabled=False)
            host.start()
            try:
                stderr = io.StringIO()
                with mock.patch.object(provider, "create_response", wraps=provider.create_response) as response_spy, contextlib.redirect_stderr(stderr):
                    _send_host_prompt(host, "check scopes")
                sent = response_spy.call_args.args[0].messages
                sent = [message for message in sent if "Memory persists across conversations" not in (message.text or "")]
                rendered = sent[0].text if sent and sent[0].role == Role.SYSTEM else ""
                if len(sent) != 3 or "# instructions: project .symphonai/INSTRUCTIONS.md\n" not in rendered or project_text not in rendered:
                    fail("project instruction text or scope did not reach the provider")
                if "# instructions: directory src/.symphonai/INSTRUCTIONS.md\ndirectory rule" not in rendered or not sent[1].text.startswith("Environment when this conversation started") or "In the person's messages, @<path> names a file in this repository. Read it with read_file before relying on its contents." not in sent[1].text:
                    fail("directory instruction text or scope did not reach the provider")
                if "instruction warning:" not in stderr.getvalue() or "loaded in full" not in stderr.getvalue() or provider.call_count != 1:
                    fail("oversize warning was hidden or the run did not complete")
            finally:
                host.close()

        user_home = root / "user-home"
        user_home.mkdir()
        instruction_file = user_home / "INSTRUCTIONS.md"
        forbidden_lower = user_home / ".env"
        forbidden_upper = user_home / ".ENV"
        instruction_file.write_text("@.env\n@.ENV\nuser rule\n", encoding="utf-8")
        forbidden_lower.write_text("lower secret", encoding="utf-8")
        forbidden_upper.write_text("upper secret", encoding="utf-8")
        loaded = load_instructions(PermissionPolicy(root), user_home=user_home)
        denial_warnings = [warning for warning in loaded.warnings if "forbidden pattern" in warning]
        if len(denial_warnings) != 2 or denial_warnings[0] != denial_warnings[1].replace(
            str(forbidden_upper), str(forbidden_lower)
        ):
            fail(f"case-variant user instruction denial warning changed: {loaded.warnings!r}")
        if "lower secret" in loaded.render() or "upper secret" in loaded.render():
            fail("case-variant user instruction include leaked forbidden content")


@check("host_server.instruction_warnings_reported")
def check_instruction_warnings_reported() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        instruction_file = root / ".symphonai" / "INSTRUCTIONS.md"
        instruction_file.parent.mkdir()
        instruction_file.write_text("x" * (MAX_INSTRUCTION_FILE_CHARS + 1), encoding="utf-8")
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
        with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": str(root / "missing-home")}):
            host = _host(
                provider, repo_root=root, sessions_root=root / "sessions",
            )
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    _send_host_prompt(host, "load instruction warnings")
                conversation = host.run.conversation_stats()
                expected = (
                    f"instruction file {instruction_file.resolve()} has {MAX_INSTRUCTION_FILE_CHARS + 1} "
                    f"characters, over the {MAX_INSTRUCTION_FILE_CHARS} character reference limit; "
                    "loaded in full"
                )
                if conversation is None or conversation.get("instruction_warnings") != [expected]:
                    fail(f"conversation did not report its instruction warning: {conversation!r}")
            finally:
                host.close()

        clean_root = root / "clean"
        clean_root.mkdir()
        with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": str(root / "missing-home")}):
            clean_host = _host(
                FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
                repo_root=clean_root, sessions_root=clean_root / "sessions",
            )
            try:
                _send_host_prompt(clean_host, "no instruction warnings")
                conversation = clean_host.run.conversation_stats()
                if conversation is None or conversation.get("instruction_warnings") != []:
                    fail(f"clean conversation did not report an empty warning list: {conversation!r}")
            finally:
                clean_host.close()


@check("host_server.provider_default_and_rejection")
def check_provider_default_and_rejection() -> None:
    keys = {
        "ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": "", "OPENAI_API_KEY": "",
        **{entry.api_key_env_var: "" for entry in CATALOG.values()},
    }
    with mock.patch.dict(os.environ, keys), tempfile.TemporaryDirectory() as temporary:
        if host_main._provider() is not None:
            fail("a host without keys chose a default provider")
        host = HostServer(None, PermissionPolicy(Path(temporary)), sessions_root=Path(temporary) / "sessions")
        host.start()
        try:
            for path, body in (
                ("/prompt", {"prompt": "must refuse"}),
                ("/provider", {"name": "openai"}),
                ("/provider", {"name": "unknown"}),
                ("/provider", {}),
            ):
                connection, response = _request(host, "POST", path, body=body, headers=_headers(host))
                try:
                    if response.status != 400:
                        fail(f"{path} accepted a missing-key or invalid choice: {response.status}")
                    response.read()
                finally:
                    connection.close()
            if host.run._conversation is not None:
                fail("a refused prompt opened a conversation")
        finally:
            host.close()
        with mock.patch.dict(os.environ, {
            "GEMINI_API_KEY": "fixture-key", "OPENAI_API_KEY": "fixture-key",
            **{entry.api_key_env_var: "" for entry in CATALOG.values()},
        }):
            if host_main._provider().name != "gemini":
                fail("the first keyed vendor in Settings order was not the default")
        for name, key, expected_model in (
            ("anthropic", "ANTHROPIC_API_KEY", "claude-opus-5-5"),
            ("openai", "OPENAI_API_KEY", "gpt-6.1-sol"),
            ("gemini", "GEMINI_API_KEY", "gemini-pro-latest"),
        ):
            root = Path(temporary) / name
            root.mkdir()
            host = HostServer(
                None, PermissionPolicy(root), sessions_root=root / "sessions",
                memory_extraction_enabled=False,
            )
            host.start()
            try:
                with mock.patch.dict(os.environ, {key: "fixture-key"}):
                    connection, response = _request(
                        host, "POST", "/provider", body={"name": name}, headers=_headers(host),
                    )
                    try:
                        reply = json.loads(response.read())
                        if response.status != 200 or reply != {"selected": True}:
                            fail(f"default {name} selection failed: {response.status}, {reply!r}")
                    finally:
                        connection.close()
                    with mock.patch.object(
                        host.run._provider, "create_response_stream",
                        return_value=iter([StreamCompleted(ModelResponse(Message(Role.ASSISTANT, "done")))]),
                    ) as respond:
                        _send_host_prompt(host, "use the default model")
                    if respond.call_count != 1:
                        fail(f"default {name} conversation did not call its selected provider")
                    conversation = host.run.conversation_stats()
                    if conversation is None or (
                        conversation.get("provider"), conversation.get("model")
                    ) != (name, expected_model):
                        fail(f"default {name} conversation used the wrong model: {conversation!r}")
            finally:
                host.close()
    for flag in ("--provider", "--model", "--base-url"):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                host_main._arguments([flag, "openai"])
            except SystemExit as exc:
                if exc.code != 2:
                    fail(f"removed launch flag {flag} exited with {exc.code!r}")
            else:
                fail(f"removed launch flag {flag} was accepted")


@check("host_server.provider_choice_per_conversation")
def check_provider_choice_per_conversation() -> None:
    first = _RecordingWireFakeProvider(
        "anthropic", 2, [ModelResponse(Message(Role.ASSISTANT, "first"))]
    )
    second = _RecordingWireFakeProvider(
        "gemini", 3, [ModelResponse(Message(Role.ASSISTANT, "second"))]
    )
    providers = {"anthropic": first, "gemini": second}
    with tempfile.TemporaryDirectory() as temporary:
        host = HostServer(None, PermissionPolicy(Path(temporary)), sessions_root=Path(temporary) / "sessions", memory_extraction_enabled=False)
        host.start()
        try:
            with mock.patch.object(host_server_module, "_provider", side_effect=lambda name, model, base_url: providers[name]):
                def select(name: str) -> None:
                    connection, response = _request(host, "POST", "/provider", body={"name": name}, headers=_headers(host))
                    try:
                        if response.status != 200:
                            fail(f"provider {name} was not selected: {response.status}")
                        response.read()
                    finally:
                        connection.close()

                select("anthropic")
                _send_host_prompt(host, "first prompt")
                leader = host.run._conversation[0]
                first_session_id = host.run._conversation[1].run_id
                if leader._config.leader_provider is not first or leader._config.subagent_provider is not first:
                    fail("first conversation did not use its provider for both Leader roles")
                select("gemini")
                changed_leader = host.run._conversation[0]
                if (
                    changed_leader is leader
                    or host.run._conversation[1].run_id != first_session_id
                    or changed_leader._config.leader_provider is not second
                ):
                    fail("provider selection did not rebuild the leader in the same session")
                _send_host_prompt(host, "same conversation")
                if first.call_count != 1 or second.call_count != 1 or host.run._conversation[0] is not changed_leader:
                    fail("the next turn did not use the new provider")
                connection, response = _request(host, "POST", "/session/new", body={}, headers=_headers(host))
                try:
                    if response.status != 200:
                        fail("new conversation was rejected")
                    response.read()
                finally:
                    connection.close()
                _send_host_prompt(host, "new conversation")
                leader = host.run._conversation[0]
                if second.call_count != 2 or leader._config.leader_provider is not second or leader._config.subagent_provider is not second:
                    fail("next conversation did not use the new provider for both Leader roles")
                connection, response = _request(host, "POST", "/session/open", body={"run_id": first_session_id}, headers=_headers(host))
                try:
                    if response.status != 200:
                        fail(f"selected conversation could not be reopened: {response.status}")
                    response.read()
                finally:
                    connection.close()
                if host.run._conversation[0]._config.leader_provider is not second:
                    fail("reopening a conversation did not restore its updated provider choice")
        finally:
            host.close()


@check("host_server.provider_change_rebuilds_schema_and_keeps_history")
def check_provider_change_rebuilds_schema_and_keeps_history() -> None:
    first = _RecordingWireFakeProvider("anthropic", 2, [
        ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
            "first-call",
            "read_file",
            {"path": "note.txt"},
            provider_metadata={"thoughtSignature": "old-vendor-state"},
            vendor_id="old-vendor-id",
        )])),
        ModelResponse(Message(Role.ASSISTANT, "first answer")),
    ])
    second = _RecordingWireFakeProvider("gemini", 3, [
        ModelResponse(Message(Role.ASSISTANT, "second answer")),
    ])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "note.txt").write_text("tool result", encoding="utf-8")
        run = HostRun(
            first,
            PermissionPolicy(root),
            EventBroker(),
            model="anthropic-before",
            sessions_root=root / "sessions",
            memory_extraction_enabled=False,
        )
        try:
            run.start("first provider turn")
            _wait_until(lambda: not run.active, "first provider turn did not finish")
            old_leader, session = run._conversation
            old_history = list(old_leader._chat_messages)
            expected_history = [
                replace(
                    message,
                    tool_calls=[
                        replace(call, provider_metadata={}, vendor_id=None)
                        for call in message.tool_calls
                    ],
                )
                if message.tool_calls
                else message
                for message in old_history
            ]
            run.select_provider(
                second,
                "gemini-next",
                "high",
                {"name": "gemini", "model": "gemini-next", "effort": "high"},
            )
            new_leader, new_session = run._conversation
            if new_session is not session or new_leader is old_leader:
                fail("provider change did not rebuild the leader in its existing session")
            if old_leader._agent._tool_schemas == new_leader._agent._tool_schemas:
                fail("provider change did not rebuild tool schemas for the new wire format")
            if new_leader._chat_messages != expected_history:
                fail("provider change did not preserve history while clearing old vendor state")

            if new_session.read_meta().get("provider_state_reset") is not True:
                fail("provider change did not record its vendor-state reset for reopening")
            run.start("second provider turn")
            _wait_until(lambda: not run.active, "second provider turn did not finish")
            if len(second.requests) != 1:
                fail(f"new provider received {len(second.requests)} requests")
            request = second.requests[0]
            if request.model != "gemini-next" or request.effort != "high":
                fail(f"new provider did not receive its selected model and effort: {request!r}")
            if request.tools != new_leader._agent._tool_schemas:
                fail("new provider request did not carry the rebuilt tool schemas")
            if request.messages[:-1] != expected_history:
                fail("new provider request did not carry the prior conversation history")
            vendor_calls = [
                call
                for message in request.messages
                for call in message.tool_calls
            ]
            if any(call.vendor_id is not None or call.provider_metadata for call in vendor_calls):
                fail("old vendor-specific call state reached the new provider")
            run.open_session(session.run_id)
            reopened_calls = [
                call
                for message in run._conversation[0]._chat_messages
                for call in message.tool_calls
            ]
            if any(call.vendor_id is not None or call.provider_metadata for call in reopened_calls):
                fail("reopening the changed provider session restored old vendor state")
        finally:
            run.close()


@check("host_server.provider_change_refused_while_active")
def check_provider_change_refused_while_active() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingProvider(_RecordingWireFakeProvider):
        def create_response(self, request, *, cancel=None):
            entered.set()
            if not release.wait(3):
                fail("blocking provider was not released")
            return super().create_response(request, cancel=cancel)

    first = BlockingProvider("anthropic", 2, [ModelResponse(Message(Role.ASSISTANT, "done"))])
    second = _RecordingWireFakeProvider("gemini", 3, [ModelResponse(Message(Role.ASSISTANT, "unused"))])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = HostRun(first, PermissionPolicy(root), EventBroker(), sessions_root=root / "sessions")
        try:
            run.start("hold the run")
            if not entered.wait(2):
                fail("provider request did not become active")
            try:
                run.select_provider(second, choice={"name": "gemini"})
            except RunActiveError as exc:
                if exc.run_id != run.active_run_id:
                    fail("active provider refusal reported the wrong run id")
            else:
                fail("provider change was accepted while a run was active")
            if run._provider is not first or run._conversation[0]._config.leader_provider is not first:
                fail("refused provider change replaced the current provider")
        finally:
            release.set()
            run.close()


@check("host_server.manual_compaction_endpoint")
def check_manual_compaction_endpoint() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "answer one")),
            ModelResponse(Message(Role.ASSISTANT, "answer two")),
            ModelResponse(Message(Role.ASSISTANT, "answer three")),
            ModelResponse(
                Message(Role.ASSISTANT, "short summary"),
                usage=Usage(input_tokens=100, output_tokens=50),
            ),
        ])
        host = HostServer(
            provider,
            PermissionPolicy(root),
            sessions_root=root / "sessions",
            chat_token_budget=100_000,
            chat_recent_turns=4,
        )
        host.start()
        try:
            for index in range(3):
                _send_host_prompt(host, f"request {index} " + "x" * 1_200)
            before_conversation = _conversation_reply(host)[1]["conversation"]
            before = before_conversation["context"]["used_tokens"]
            before_leader = next(
                agent for agent in before_conversation["agents"] if agent["name"] == "leader"
            )

            connection, response = _request(
                host, "POST", "/compact",
                body={"instructions": "keep the API names"},
                headers=_headers(host),
            )
            try:
                compact_body = json.loads(response.read())
                if response.status != 200:
                    fail(f"manual compaction returned {response.status}: {compact_body!r}")
            finally:
                connection.close()
            if (
                compact_body.get("changed") is not True
                or compact_body.get("after_tokens", before) >= before
                or compact_body.get("dropped_messages", 0) < 1
            ):
                fail(f"manual compaction did not reduce context: {compact_body!r}, before={before}")
            conversation = _conversation_reply(host)[1]["conversation"]
            leader = next(agent for agent in conversation["agents"] if agent["name"] == "leader")
            if (
                conversation["context"]["used_tokens"] >= before
                or leader.get("input_tokens") != before_leader.get("input_tokens", 0) + 100
                or leader.get("output_tokens") != before_leader.get("output_tokens", 0) + 50
                or leader.get("calls") != before_leader.get("calls", 0) + 1
            ):
                fail(f"conversation stats omitted compacted context or summary usage: {conversation!r}")

            connection, response = _request(
                host, "POST", "/compact", body={"x": 1}, headers=_headers(host)
            )
            try:
                invalid = json.loads(response.read())
                if response.status != 400:
                    fail(f"invalid compact key returned {response.status}: {invalid!r}")
            finally:
                connection.close()
            connection, response = _request(
                host, "POST", "/compact", body={"instructions": 3}, headers=_headers(host)
            )
            try:
                if response.status != 400:
                    fail(f"non-string compact instructions returned {response.status}")
            finally:
                connection.close()
        finally:
            host.close()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        empty = HostServer(
            FakeModelProvider(), PermissionPolicy(root), sessions_root=root / "sessions"
        )
        empty.start()
        try:
            connection, response = _request(
                empty, "POST", "/compact", body={}, headers=_headers(empty)
            )
            try:
                body = json.loads(response.read())
                if response.status != 400 or body != {"error": "no conversation to compact"}:
                    fail(f"pre-conversation compaction response was incorrect: {response.status}, {body!r}")
            finally:
                connection.close()
        finally:
            empty.close()

    entered = threading.Event()
    release = threading.Event()

    class BlockingProvider(FakeModelProvider):
        def create_response(self, request, *, cancel=None):
            entered.set()
            if not release.wait(3):
                fail("active compact provider was not released")
            return ModelResponse(Message(Role.ASSISTANT, "done"))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        active = HostServer(
            BlockingProvider(), PermissionPolicy(root), sessions_root=root / "sessions"
        )
        active.start()
        try:
            connection, response = _request(
                active, "POST", "/prompt", body={"prompt": "hold"}, headers=_headers(active)
            )
            accepted = json.loads(response.read())
            connection.close()
            if response.status != 200 or not accepted.get("accepted"):
                fail(f"active compact fixture did not start: {response.status}, {accepted!r}")
            if not entered.wait(2):
                fail("active compact fixture did not reach its provider")
            connection, response = _request(
                active, "POST", "/compact", body={}, headers=_headers(active)
            )
            try:
                body = json.loads(response.read())
                if response.status != 409 or body.get("run_id") != accepted.get("run_id"):
                    fail(f"active compaction did not return 409 and run id: {response.status}, {body!r}")
            finally:
                connection.close()
        finally:
            release.set()
            active.close()


@check("host_server.conversation_reports_model_selection")
def check_conversation_reports_model_selection() -> None:
    provider = _RecordingWireFakeProvider(
        "openai", 1,
        [ModelResponse(Message(Role.ASSISTANT, "first")), ModelResponse(Message(Role.ASSISTANT, "second"))],
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = HostRun(provider, PermissionPolicy(root), EventBroker(), sessions_root=root / "sessions", memory_extraction_enabled=False)
        try:
            run.start("first turn")
            _wait_until(lambda: not run.active, "first model turn did not finish")
            run.select_provider(
                provider,
                "live-model",
                "xhigh",
                {"name": "openai", "model": "live-model", "effort": "xhigh"},
            )
            conversation = run.conversation_stats()
            if (conversation.get("provider"), conversation.get("model"), conversation.get("effort")) != (
                "openai", "live-model", "xhigh"
            ):
                fail(f"conversation did not report its live selection: {conversation!r}")
            run.start("second turn")
            _wait_until(lambda: not run.active, "second model turn did not finish")
            if provider.requests[-1].model != "live-model" or provider.requests[-1].effort != "xhigh":
                fail("reported model selection did not reach the following request")
        finally:
            run.close()


@check("host_server.context_budget_tracks_model")
def check_context_budget_tracks_model() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = _RecordingWireFakeProvider(
            "anthropic", 2, [ModelResponse(Message(Role.ASSISTANT, "done"))]
        )
        host = HostServer(
            provider,
            PermissionPolicy(root),
            model="claude-opus-4-8",
            sessions_root=root / "default-sessions",
        )
        host.start()
        try:
            _send_host_prompt(host, "hello")
            conversation = _conversation_reply(host)[1]["conversation"]
            budget = conversation["context"]["budget_tokens"]
            if budget != 955_000:
                fail(f"default model context budget was {budget}, expected 955000")
        finally:
            host.close()

        explicit = HostServer(
            _RecordingWireFakeProvider(
                "anthropic", 2, [ModelResponse(Message(Role.ASSISTANT, "done"))]
            ),
            PermissionPolicy(root),
            model="claude-opus-4-8",
            sessions_root=root / "explicit-sessions",
            chat_token_budget=170,
        )
        explicit.start()
        try:
            _send_host_prompt(explicit, "hello")
            conversation = _conversation_reply(explicit)[1]["conversation"]
            budget = conversation["context"]["budget_tokens"]
            if budget != 170:
                fail(f"explicit host context budget was {budget}, expected 170")
        finally:
            explicit.close()


@check("host_server.context_window_reporting")
def check_context_window_reporting() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for index, (model, expected_window) in enumerate((
            ("claude-opus-4-8", 1_000_000),
            ("model-without-a-window", None),
        )):
            host = HostServer(
                _RecordingWireFakeProvider(
                    "anthropic", 2, [ModelResponse(Message(Role.ASSISTANT, "done"))]
                ),
                PermissionPolicy(root),
                model=model,
                sessions_root=root / f"sessions-{index}",
            )
            host.start()
            try:
                _send_host_prompt(host, "measure context")
                conversation = _conversation_reply(host)[1]["conversation"]
                actual_window = conversation["context"].get("window_tokens", "missing")
                if actual_window != expected_window:
                    fail(
                        f"{model} reported context window {actual_window!r}, "
                        f"expected {expected_window!r}"
                    )
            finally:
                host.close()


@check("host_server.permission_mode_control")
def check_permission_mode_control() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        extensions = load_extensions(repo_root=root, home=root / "home")
        empty_modes = replace(
            extensions,
            ceiling=replace(extensions.ceiling, modes=()),
        )
        try:
            HostServer(
                FakeModelProvider(), PermissionPolicy(root),
                sessions_root=root / "empty-sessions", extensions=empty_modes,
                memory_extraction_enabled=False,
            )
        except host_run_module.ModeSelectionError as exc:
            if "agents.ceiling.modes" not in str(exc):
                fail(f"empty modes error omitted its configuration key: {exc}")
        else:
            fail("an empty modes ceiling started a host")
        error = io.StringIO()
        with (
            mock.patch.object(host_main, "load_extensions", return_value=empty_modes),
            mock.patch.object(host_main, "prune_sessions"),
            mock.patch.object(host_main, "_provider", return_value=FakeModelProvider()),
            contextlib.redirect_stderr(error),
        ):
            try:
                host_main.main(["--repo-root", str(root)])
            except SystemExit as exc:
                if exc.code != 2:
                    fail(f"empty modes ceiling exited with {exc.code!r}")
            else:
                fail("the host entry point accepted an empty modes ceiling")
        if (
            "configuration error:" not in error.getvalue()
            or "agents.ceiling.modes" not in error.getvalue()
        ):
            fail(f"empty modes startup error was unclear: {error.getvalue()!r}")

        for modes, expected in (
            (("plan",), "plan"),
            (("allow", "plan"), "plan"),
        ):
            configured = replace(
                extensions,
                ceiling=replace(extensions.ceiling, modes=modes),
            )
            candidate = HostServer(
                FakeModelProvider(), PermissionPolicy(root, mode="allow"),
                sessions_root=root / f"sessions-{'-'.join(modes)}",
                extensions=configured,
                memory_extraction_enabled=False,
            )
            try:
                if candidate.run.policy.mode != expected:
                    fail(f"ceiling {modes!r} started in {candidate.run.policy.mode!r}")
                if candidate.run.policy.mode not in candidate.run.permitted_modes():
                    fail(f"ceiling {modes!r} started outside its permitted modes")
                if modes == ("allow", "plan"):
                    candidate.run.select_mode("allow")
                    candidate.run.select_mode("plan")
                    if candidate.run.policy.mode != "plan":
                        fail("the starting mode became a one-way door")
            finally:
                candidate.close()

        extensions = replace(
            extensions,
            ceiling=replace(extensions.ceiling, modes=("ask", "plan")),
        )
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "conversation open")),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "mode-child", "dispatch_subagent", {
                    "subagent_name": "worker", "task": "inspect",
                },
            )])),
            ModelResponse(Message(Role.ASSISTANT, "child done")),
            ModelResponse(Message(Role.ASSISTANT, "leader done")),
        ])
        policy = PermissionPolicy(root, allowed_write_scope=[root], mode="ask")
        host = HostServer(
            provider,
            policy,
            sessions_root=root / "sessions",
            extensions=extensions,
            memory_extraction_enabled=False,
        )
        host.start()

        def select(mode: object) -> tuple[int, dict]:
            connection, response = _request(
                host, "POST", "/mode", body={"mode": mode}, headers=_headers(host)
            )
            try:
                return response.status, json.loads(response.read())
            finally:
                connection.close()

        try:
            connection, response = _request(host, "POST", "/mode", body={"mode": "plan"})
            try:
                if response.status != 401 or response.read() != b"":
                    fail("mode route accepted an unauthenticated request")
            finally:
                connection.close()

            _send_host_prompt(host, "open a conversation")
            leader, session = host.run._conversation
            session_id = session.run_id
            policy = host.run._policy

            status, reply = select("plan")
            if status != 200 or reply != {"mode": "plan"}:
                fail(f"plan mode was not selected: {status}, {reply!r}")
            denied = policy.check_write(root / "plan-denied.txt")
            if denied.allowed or "plan mode" not in denied.reason:
                fail(f"plan mode permitted a write: {denied!r}")

            policy.approval_callback = lambda _: True
            status, reply = select("ask")
            if status != 200 or reply != {"mode": "ask"}:
                fail(f"ask mode was not selected: {status}, {reply!r}")
            if host.run._conversation[0] is not leader:
                fail("changing mode rebuilt the active conversation")
            allowed = policy.check_write(root / "ask-approved.txt")
            if not allowed.allowed:
                fail(f"ask mode did not permit an approved write: {allowed!r}")

            for refused in ("unknown", "allow"):
                status, reply = select(refused)
                error = reply.get("error", "")
                if (
                    status != 400
                    or "ask" not in error
                    or "plan" not in error
                    or policy.mode != "ask"
                ):
                    fail(f"mode {refused!r} was not refused atomically: {status}, {reply!r}")

            connection, response = _request(
                host, "POST", "/mode",
                body={"mode": "plan", "extra": True}, headers=_headers(host),
            )
            try:
                reply = json.loads(response.read())
                if (
                    response.status != 400
                    or "ask" not in reply.get("error", "")
                    or policy.mode != "ask"
                ):
                    fail(f"mode route accepted unknown fields: {response.status}, {reply!r}")
            finally:
                connection.close()

            select("plan")
            _send_host_prompt(host, "dispatch after changing mode")
            child = leader.subagents.get("worker")
            if child is None or child.agent._policy.mode != "plan":
                fail(f"next subagent dispatch missed the live mode: {child!r}")

            _, conversation = _conversation_reply(host)
            if conversation.get("conversation", {}).get("mode") != "plan":
                fail(f"conversation omitted its current mode: {conversation!r}")

            connection, response = _request(
                host, "POST", "/session/open",
                body={"run_id": session_id}, headers=_headers(host),
            )
            try:
                body = response.read()
                if response.status != 200:
                    fail(f"session did not reopen: {response.status}, {body!r}")
            finally:
                connection.close()
            _, reopened = _conversation_reply(host)
            if reopened.get("conversation", {}).get("mode") != "plan" or policy.mode != "plan":
                fail(f"reopening the open session did not retain its mode: {reopened!r}")
            select("plan")
            host.run.end_conversation()
            if policy.mode != "ask" or host.run.conversation_stats() is not None:
                fail("ending a conversation did not restore the starting mode")
        finally:
            host.close()


@check("host_server.conversation_usage")
def check_conversation_usage() -> None:
    secret = "fixture-secret-token-24f"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "recognisable-absolute-path-24f"
        root.mkdir()
        provider = FakeModelProvider([
            ModelResponse(
                Message(Role.ASSISTANT, tool_calls=[ToolCall(
                    "dispatch", "dispatch_subagent",
                    {"subagent_name": "worker", "task": "inspect"},
                )]),
                usage=Usage(1050, 20, 900, 50),
            ),
            ModelResponse(Message(Role.ASSISTANT, "child"), usage=Usage(input_tokens=3, output_tokens=2)),
            ModelResponse(Message(Role.ASSISTANT, "first"), usage=Usage(input_tokens=20, output_tokens=4)),
            ModelResponse(Message(Role.ASSISTANT, "second"), usage=Usage(input_tokens=5, output_tokens=1)),
        ])
        provider.model = "metered"
        host = HostServer(
            provider,
            PermissionPolicy(root),
            token=secret,
            sessions_root=root / "sessions",
            memory_extraction_enabled=False,
        )
        host.start()
        terminal_stats = []
        publish = host.broker.publish

        def capture_terminal(event, **kwargs) -> None:  # noqa: ANN001
            if isinstance(event, RunFinished) and event.agent_name == "leader":
                terminal_stats.append(host.run.conversation_stats())
            publish(event, **kwargs)

        host.broker.publish = capture_terminal
        try:
            _, empty = _conversation_reply(host)
            if empty != {"conversation": None}:
                fail(f"unused host exposed conversation totals: {empty!r}")
            _send_host_prompt(host, "first short prompt")
            if not terminal_stats or terminal_stats[-1] is None:
                fail("leader terminal event was published before conversation totals")
            first_body, first = _conversation_reply(host)
            _send_host_prompt(host, "second short prompt")
            second_body, second = _conversation_reply(host)
            first_stats = first["conversation"]
            second_stats = second["conversation"]
            if second_stats["context"]["used_tokens"] <= first_stats["context"]["used_tokens"]:
                fail(f"context usage did not grow across prompts: {first_stats!r}, {second_stats!r}")
            if second_stats["usage"]["total_tokens"] != 1105:
                fail(f"conversation usage did not accumulate across prompts: {second_stats!r}")
            if (
                second_stats["usage"]["cache_read_tokens"] != 900
                or second_stats["usage"]["cache_write_tokens"] != 50
            ):
                fail(f"aggregate cache usage was wrong: {second_stats['usage']!r}")
            agents = {agent["name"]: agent for agent in second_stats["agents"]}
            if set(agents) != {"leader", "worker"}:
                fail(f"per-agent usage did not name leader and child: {agents!r}")
            if agents["leader"]["total_tokens"] != 1100 or agents["worker"]["total_tokens"] != 5:
                fail(f"per-agent totals were wrong: {agents!r}")
            if (
                agents["leader"]["cache_read_tokens"] != 900
                or agents["leader"]["cache_write_tokens"] != 50
            ):
                fail(f"leader cache usage was wrong: {agents['leader']!r}")
            if "cost" in second_stats["usage"] or any("cost" in agent for agent in agents.values()):
                fail(f"missing price table rendered zero cost: {second_stats!r}")
            encoded = first_body + second_body
            if secret.encode() in encoded or str(root).encode() in encoded:
                fail("conversation payload leaked a token or absolute repository path")
        finally:
            host.close()


def _delegating_conversation_host(root: Path) -> HostServer:
    provider = FakeModelProvider([
        ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
            "dispatch", "dispatch_subagent", {"subagent_name": "worker", "task": "inspect"},
        )])),
        ModelResponse(Message(Role.ASSISTANT, "child")),
        ModelResponse(Message(Role.ASSISTANT, "done")),
    ])
    host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
    host.start()
    return host


@check("host_server.replay_dispatch_task_and_turn_times")
def check_replay_dispatch_task_and_turn_times() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        host = _delegating_conversation_host(Path(temporary))
        stream_connection, stream_response = _subscribed_stream(host)
        try:
            _send_host_prompt(host, "dispatch")
            _wait_until(lambda: host.run._active is None, "dispatch run did not finish")
            run_id = host.run._conversation[1].run_id
            connection, response = _request(
                host, "POST", "/session/open", body={"run_id": run_id}, headers=_headers(host),
            )
            try:
                if response.status != 200:
                    fail(f"session reopen failed: {response.status}, {response.read()!r}")
                response.read()
            finally:
                connection.close()
            prompt = _await_sse(
                stream_connection, stream_response,
                lambda item: item[1].get("type") == "HistoryMessage" and item[1].get("role") == "user",
                what="replayed timed prompt",
            )[1]
            dispatch = _await_sse(
                stream_connection, stream_response,
                lambda item: item[1].get("type") == "HistoryMessage"
                and any(call.get("name") == "dispatch_subagent" for call in item[1].get("tool_calls", [])),
                what="replayed subagent dispatch",
            )[1]
            times = (prompt.get("turn_started_at"), prompt.get("turn_ended_at"))
            if not isinstance(times[0], str) or "turn_ended_at" in prompt or "turn_outcome" in prompt:
                fail(f"replayed user turn has misplaced end metadata: {prompt!r}")
            call = next(call for call in dispatch["tool_calls"] if call.get("name") == "dispatch_subagent")
            if call.get("subagent_name") != "worker" or call.get("task") != "inspect":
                fail(f"replayed dispatch omitted its safe name/task: {dispatch!r}")
            closing = _await_sse(
                stream_connection, stream_response,
                lambda item: item[1].get("type") == "HistoryMessage"
                and item[1].get("role") == "assistant"
                and isinstance(item[1].get("turn_ended_at"), str),
                what="replayed closing assistant message",
            )[1]
            if closing.get("turn_outcome") != "finished" or "turn_started_at" in closing:
                fail(f"replayed closing assistant has incorrect turn metadata: {closing!r}")
            if closing["turn_ended_at"] <= times[0]:
                fail(f"replayed root turn end did not follow its start: {closing!r}")
        finally:
            stream_connection.close()
            host.close()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = FakeModelProvider()

        def fail_before_reply(request, *, cancel=None):  # noqa: ANN001
            raise ProviderError("scripted failure before reply")

        provider.create_response = fail_before_reply
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        stream_connection, stream_response = _subscribed_stream(host)
        try:
            with mock.patch.object(threading, "excepthook", lambda args: None):
                _send_host_prompt(host, "fail before any reply")
            run_id = host.run._conversation[1].run_id
            connection, response = _request(
                host, "POST", "/session/open", body={"run_id": run_id}, headers=_headers(host),
            )
            try:
                if response.status != 200:
                    fail(f"failed session reopen failed: {response.status}, {response.read()!r}")
                response.read()
            finally:
                connection.close()
            prompt = _await_sse(
                stream_connection, stream_response,
                lambda item: item[1].get("type") == "HistoryMessage" and item[1].get("role") == "user",
                what="replayed failed user prompt",
            )[1]
            if (
                not isinstance(prompt.get("turn_started_at"), str)
                or not isinstance(prompt.get("turn_ended_at"), str)
                or prompt.get("turn_outcome") != "failed"
            ):
                fail(f"failed run without an assistant reply omitted end metadata on its user record: {prompt!r}")
            records, _ = host_run_module.read_records(
                host.run._conversation[1].directory / "run.jsonl"
            )
            if any(
                record.get("type") == "message"
                and isinstance(record.get("data"), dict)
                and record["data"].get("role") == Role.ASSISTANT.value
                for record in records
            ):
                fail("scripted failure unexpectedly persisted an assistant message")
        finally:
            stream_connection.close()
            host.close()


@check("host_server.conversation_live_parentage")
def check_conversation_live_parentage() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        host = _delegating_conversation_host(Path(temporary))
        try:
            _send_host_prompt(host, "dispatch")
            conversation = _conversation_reply(host)[1]["conversation"]
            agents = {agent["name"]: agent for agent in conversation["agents"]}
            if set(agents) != {"leader", "worker"}:
                fail(f"live conversation omitted an agent: {agents!r}")
            if agents["leader"]["parent_agent_id"] is not None or agents["worker"]["parent_agent_id"] != agents["leader"]["agent_id"]:
                fail(f"live conversation lost dispatch parentage: {agents!r}")
            if "usage" not in conversation or any("total_tokens" not in agent for agent in agents.values()):
                fail(f"live conversation lost accounted usage: {conversation!r}")
        finally:
            host.close()


@check("host_server.conversation_agent_state")
def check_conversation_agent_state() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "dispatch-one", "dispatch_subagent", {"subagent_name": "explorer", "task": "inspect one"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "first result")),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "dispatch-two", "dispatch_subagent", {"subagent_name": "explorer", "task": "inspect two"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "second result")),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        try:
            _send_host_prompt(host, "dispatch twice")
            conversation = _conversation_reply(host)[1]["conversation"]
            agents = {agent["name"]: agent for agent in conversation["agents"]}
            if set(agents) != {"leader", "explorer"}:
                fail(f"repeated dispatch duplicated an agent: {conversation!r}")
            if any(agent.get("state") != "finished" for agent in agents.values()):
                fail(f"completed runs omitted finished agent state: {agents!r}")
        finally:
            host.close()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = _GatedProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions")
        host.start()
        try:
            host.run.start("cancel this run")
            if not provider.entered[0].wait(5):
                fail("cancellation provider did not start")
            with host.run._lock:
                active = host.run._active
            if active is None:
                fail("cancelled run was not active")
            host.run.stop()
            provider.release[0].set()
            active.thread.join(5)
            if active.thread.is_alive():
                fail("cancelled run did not finish")
            conversation = _conversation_reply(host)[1]["conversation"]
            agents = {agent["name"]: agent for agent in conversation["agents"]}
            if agents.get("leader", {}).get("state") != "stopped":
                fail(f"cancelled run did not report stopped state: {agents!r}")
        finally:
            host.close()


@check("host_server.conversation_reopened_parentage")
def check_conversation_reopened_parentage() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        host = _delegating_conversation_host(Path(temporary))
        try:
            _send_host_prompt(host, "dispatch")
            run_id = host.run._conversation[1].run_id
            call_count = host.run._provider.call_count
            connection, response = _request(
                host, "POST", "/session/open", body={"run_id": run_id}, headers=_headers(host),
            )
            try:
                if response.status != 200:
                    fail(f"session reopen failed: {response.status}, {response.read()!r}")
                response.read()
            finally:
                connection.close()
            conversation = _conversation_reply(host)[1]["conversation"]
            if conversation is None or not {"agents", "mode", "goal", "provider", "model", "effort", "context", "usage"}.issubset(conversation):
                fail(f"reopened conversation did not retain its session usage: {conversation!r}")
            agents = {agent["name"]: agent for agent in conversation["agents"]}
            if set(agents) != {"leader", "worker"}:
                fail(f"reopened conversation omitted an agent: {agents!r}")
            if agents["leader"]["parent_agent_id"] is not None or agents["worker"]["parent_agent_id"] != agents["leader"]["agent_id"]:
                fail(f"reopened conversation lost persisted parentage: {agents!r}")
            if not conversation["usage"]["calls"] or any("calls" not in agent for agent in agents.values()):
                fail(f"reopened conversation lost its recorded usage: {conversation!r}")
            if host.run._provider.call_count != call_count:
                fail("reopening ran a new model turn")
        finally:
            host.close()


@check("host_server.conversation_repeated_agent_parentage")
def check_conversation_repeated_agent_parentage() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = HostServer(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            PermissionPolicy(root), sessions_root=root / "sessions",
        )
        host.start()
        try:
            if _conversation_reply(host)[1] != {"conversation": None}:
                fail("a host without a conversation reported agents")
            _send_host_prompt(host, "one root")
            host.run.open_session(host.run._conversation[1].run_id)
            root_only = _conversation_reply(host)[1]["conversation"]
            if len(root_only["agents"]) != 1 or root_only["agents"][0]["parent_agent_id"] is not None:
                fail(f"reopened conversation without subagents lost its root: {root_only!r}")

            graph = (
                RunNode("r1", "leader-old", "leader", None, None, False, (
                    RunNode("r4", "c", "C", "r1", None, False),
                    RunNode("r2", "a", "A", "r1", None, False, (
                        RunNode("r3", "b", "B", "r2", None, False),
                    )),
                    RunNode("r8", "explorer-old", "explorer", "r1", None, False),
                )),
                RunNode("r5", "leader-new", "leader", None, None, False, (
                    RunNode("r6", "explorer-new", "explorer", "r5", None, False),
                )),
                RunNode("r7", "orphan", "Orphan", "missing", None, False),
            )
            records = [
                {"type": "run_started", "run_id": f"r{index}", "ts": f"2026-01-01T00:00:0{index}Z"}
                for index in range(1, 8)
            ]
            with mock.patch.object(host.run._conversation[0], "run_graph", return_value=graph), mock.patch.object(
                host_run_module, "read_records", return_value=(records, 0),
            ):
                host.run._usage_by_session[host.run._conversation[1].run_id] = {
                    "leader-old": ("leader", {"model": UsageTotals(1, 2, 1)}),
                    "leader-new": ("leader", {"model": UsageTotals(3, 4, 1)}),
                    "explorer-old": ("explorer", {"model": UsageTotals(5, 6, 1)}),
                    "explorer-new": ("explorer", {"model": UsageTotals(7, 8, 1)}),
                }
                agents = _conversation_reply(host)[1]["conversation"]["agents"]
            if [agent["agent_id"] for agent in agents] != ["leader-new", "a", "b", "c", "explorer-new", "orphan"]:
                fail(f"spawn order or repeated-agent folding changed: {agents!r}")
            if [agent["parent_agent_id"] for agent in agents] != [None, "leader-new", "a", "leader-new", "leader-new", None]:
                fail(f"nested or unknown parent was misplaced: {agents!r}")
            if agents[0]["input_tokens"] != 4 or agents[0]["output_tokens"] != 6 or agents[0]["calls"] != 2:
                fail(f"leader usage was not summed across root ids: {agents[0]!r}")
            explorer = next(agent for agent in agents if agent["name"] == "explorer")
            if explorer["input_tokens"] != 12 or explorer["output_tokens"] != 14 or explorer["calls"] != 2:
                fail(f"explorer usage was not summed across ids: {explorer!r}")
        finally:
            host.close()


@check("host_server.conversation_agents_keep_identity_after_restart")
def check_conversation_agents_keep_identity_after_restart() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sessions_root = root / "sessions"
        first_host = HostServer(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "first reply"))]),
            PermissionPolicy(root), sessions_root=sessions_root,
        )
        first_host.start()
        try:
            _send_host_prompt(first_host, "first prompt")
            first_conversation = _conversation_reply(first_host)[1]["conversation"]
            first_leaders = [agent for agent in first_conversation["agents"] if agent["name"] == "leader"]
            if len(first_leaders) != 1:
                fail(f"first conversation did not have one leader: {first_conversation!r}")
            session_id = first_conversation["session_id"]
            first_agent_id = first_leaders[0]["agent_id"]
        finally:
            first_host.close()

        second_host = HostServer(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "second reply"))]),
            PermissionPolicy(root), sessions_root=sessions_root,
        )
        second_host.start()
        try:
            second_host.run.open_session(session_id)
            _send_host_prompt(second_host, "second prompt")
            reopened = _conversation_reply(second_host)[1]["conversation"]
            leaders = [agent for agent in reopened["agents"] if agent["name"] == "leader"]
            if len(leaders) != 1 or leaders[0]["agent_id"] != first_agent_id:
                fail(f"host restart changed or duplicated leader identity: {reopened!r}")
        finally:
            second_host.close()


@check("host_server.conversation_cost_and_context")
def check_conversation_cost_and_context() -> None:
    table = PriceTable(
        prices={"priced": ModelPrice(Decimal("1"), Decimal("2"))},
        currency="USD",
    )

    def priced_payload(root: Path, model: str) -> dict:
        provider = FakeModelProvider([
            ModelResponse(
                Message(Role.ASSISTANT, "priced reply"),
                usage=Usage(input_tokens=1_000_000, output_tokens=2_000_000),
            )
        ])
        provider.model = model
        host = HostServer(
            provider,
            PermissionPolicy(root),
            sessions_root=root / "sessions",
            price_table=table,
            memory_extraction_enabled=False,
        )
        host.start()
        try:
            _send_host_prompt(host, "price this")
            return _conversation_reply(host)[1]["conversation"]
        finally:
            host.close()

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        priced_root = base / "priced"
        unknown_root = base / "unknown"
        compact_root = base / "compact"
        for root in (priced_root, unknown_root, compact_root):
            root.mkdir()
        priced = priced_payload(priced_root, "priced")
        unknown = priced_payload(unknown_root, "unpriced")
        expected_cost = {"amount": "5", "currency": "USD"}
        if priced["usage"].get("cost") != expected_cost:
            fail(f"priced model cost was missing or inexact: {priced!r}")
        if "cost" in unknown["usage"] or any("cost" in agent for agent in unknown["agents"]):
            fail(f"unpriced model produced a cost: {unknown!r}")

        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "reply " + "y" * 40))
        ])
        context_budget = 8_000
        host = HostServer(
            provider,
            PermissionPolicy(compact_root),
            sessions_root=compact_root / "sessions",
            chat_token_budget=context_budget,
            chat_recent_turns=1,
            memory_extraction_enabled=False,
        )
        host.start()
        subscription = host.broker.subscribe()
        used = []
        try:
            for index in range(6):
                _send_host_prompt(host, f"prompt {index} " + "x" * 6_000)
                used.append(_conversation_reply(host)[1]["conversation"]["context"]["used_tokens"])
            if host.run._conversation is None or not host.run._conversation[0]._config.model_summary:
                fail("host-built leader did not enable model summaries")
            events = []
            while (event := subscription.get(timeout=0.01)) is not None:
                events.append(event)
        finally:
            subscription.close()
            host.close()
        compactions = [event for event in events if isinstance(event, CompactionApplied)]
        if not compactions or not all(
            event.after_tokens < event.before_tokens for event in compactions
        ):
            fail(f"runtime did not report a context reduction: {compactions!r}")
        if used[-1] > context_budget:
            fail(f"visible context exceeded its configured budget: {used!r}")
        if used[-1] >= sum(used[:3]):
            fail(f"visible context kept a cumulative total after compaction: {used!r}")


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


@check("host_server.decisions_replay")
def check_decisions_replay() -> None:
    calls = [
        ToolCall("answered", "ask_user", {"question": "Pick one", "options": ["A", "B"]}),
        ToolCall("skipped", "ask_user", {"question": "Continue?"}),
        ToolCall("approved", "propose_plan", {"plan": "# Do it"}),
        ToolCall("kept", "propose_plan", {"plan": "# Rework"}),
        ToolCall("failed", "ask_user", {"question": "Unavailable?"}),
    ]
    messages = [
        Message(Role.ASSISTANT, tool_calls=calls),
        Message(Role.TOOL, tool_result=ToolResult("answered", True, "The person answered: B")),
        Message(Role.TOOL, tool_result=ToolResult("skipped", True, "The person skipped the question.")),
        Message(Role.TOOL, tool_result=ToolResult(
            "approved", True,
            "The person approved the plan. The conversation is now in allow mode; carry out the plan.",
        )),
        Message(Role.TOOL, tool_result=ToolResult("kept", True, "The person wants to keep planning: smaller")),
        Message(Role.TOOL, tool_result=ToolResult("failed", False, error="question was stopped")),
    ]
    results = HostRun._replay_tool_results(messages)
    replayed = HostRun._replay_tool_calls(messages[0], results)
    expected = [
        {"id": "answered", "name": "ask_user", "question": "Pick one", "options": ["A", "B"], "answer": "B"},
        {"id": "skipped", "name": "ask_user", "question": "Continue?", "skipped": True},
        {"id": "approved", "name": "propose_plan", "plan": "# Do it", "decision": "allow"},
        {"id": "kept", "name": "propose_plan", "plan": "# Rework", "decision": "keep_planning", "feedback": "smaller"},
        {"id": "failed", "name": "ask_user", "question": "Unavailable?"},
    ]
    if replayed != expected:
        fail(f"replayed decisions were wrong or leaked tool result text: {replayed!r}")


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


class _HostControlTool(LocalTool):
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    @property
    def name(self) -> str:
        return "wait_for_control"

    @property
    def description(self) -> str:
        return "Wait until the host control check releases this tool."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(ToolEffect.READ_ONLY, True, ())

    def _execute(self, tool_call, policy, cancel=None):
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError("host control check did not release tool")
        return ToolResult(tool_call_id=tool_call.id, ok=True, content="released")


class _HostControlProvider(FakeModelProvider):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__(responses)
        self.final_entered = threading.Event()
        self.release_final = threading.Event()

    def create_response(self, request, *, cancel=None) -> ModelResponse:
        if self.call_count == 3:
            self.final_entered.set()
            if not self.release_final.wait(5):
                raise RuntimeError("host control check did not release final response")
        return super().create_response(request, cancel=cancel)


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
        self.agents = {}
        self.skills = {}
        self.config = ResolvedConfig({}, {})
        self.ceiling = None

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


def _host_run_probe(
    root: Path,
    extensions: Extensions | None,
    mcp_tools=None,  # noqa: ANN001
) -> tuple[HostRun, tuple]:
    provider = _GatedProvider(
        [ModelResponse(Message(Role.ASSISTANT, "done"))]
    )
    broker = EventBroker()
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
    broker.close()
    return run, tuple(calls)


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
            run, fan_out_calls = _host_run_probe(root, extensions)
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
                broker_types = tuple(
                    type(event).__name__ for event in received
                    if type(event).__name__ in hook_types
                )
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
                leader_module,
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


@check("host_server.main_passes_base_prompts")
def main_passes_base_prompts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        home = root / "home"
        repo = root / "repo"
        home.mkdir()
        repo.mkdir()
        fake_host = mock.Mock()
        fake_pool = mock.Mock()
        fake_pool.start.return_value = {}
        fake_pool.instructions = ()
        with mock.patch.dict(os.environ, {"HOME": str(home)}), mock.patch.object(
            host_main, "_provider", return_value=FakeModelProvider(),
        ), mock.patch.object(
            host_main, "HostServer", return_value=fake_host,
        ) as host_factory, mock.patch.object(
            host_main, "McpPool", return_value=fake_pool,
        ), mock.patch.object(
            host_main, "LspManager", return_value=mock.Mock(),
        ), mock.patch.object(
            host_main, "prune_sessions",
        ):
            host_main.main(["--repo-root", str(repo)])
        if host_factory.call_args.kwargs.get("base_prompt") != LEADER_BASE_PROMPT:
            fail("host main did not pass the leader base prompt")
        if host_factory.call_args.kwargs.get("subagent_base_prompt") != SUBAGENT_BASE_PROMPT:
            fail("host main did not pass the subagent base prompt")
        fake_host.close.assert_called_once_with()


@check("host_server.selection_is_remembered")
def selection_is_remembered() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        home = root / "post-home"
        repo = root / "repo"
        home.mkdir()
        repo.mkdir()
        choice = {"name": "openai", "model": "gpt-5.4-mini", "effort": "high"}
        with mock.patch.dict(os.environ, {
            "HOME": str(home), "SYMPHONAI_HOME": "", "OPENAI_API_KEY": "test-key",
        }, clear=True):
            host = HostServer(FakeModelProvider(), PermissionPolicy(repo), sessions_root=root / "sessions")
            host.start()
            try:
                connection, response = _request(
                    host, "POST", "/provider", body=choice, headers=_headers(host),
                )
                reply = json.loads(response.read())
                connection.close()
                if response.status != 200 or reply != {"selected": True}:
                    fail(f"provider selection failed: {response.status}, {reply!r}")
                selection_path = home / ".symphonai" / "selection.json"
                if json.loads(selection_path.read_text(encoding="utf-8")) != choice:
                    fail("provider selection was not atomically saved with its chosen fields")
                connection, response = _request(host, "GET", "/settings", headers=_headers(host))
                settings = json.loads(response.read())
                connection.close()
                if response.status != 200 or settings.get("settings", {}).get("selection") != {
                    "provider": "openai", "model": "gpt-5.4-mini", "effort": "high",
                }:
                    fail(f"settings did not report the applied selection: {settings!r}")
            finally:
                host.close()

        def run_main(test_home: Path, environment: dict[str, str]):
            fake_host = mock.Mock()
            fake_pool = mock.Mock()
            fake_pool.start.return_value = {}
            fake_pool.instructions = ()
            with mock.patch.dict(os.environ, {
                "HOME": str(test_home), "PATH": os.defpath,
                "SYMPHONAI_HOME": "", **environment,
            }, clear=True), mock.patch.object(host_main, "HostServer", return_value=fake_host) as host_factory, mock.patch.object(
                host_main, "McpPool", return_value=fake_pool,
            ), mock.patch.object(host_main, "LspManager", return_value=mock.Mock()), mock.patch.object(
                host_main, "prune_sessions",
            ):
                host_main.main(["--repo-root", str(repo)])
            return host_factory.call_args.args[0], fake_host

        keyed_home = root / "keyed-home"
        keyed_home.mkdir()
        (keyed_home / ".symphonai").mkdir()
        (keyed_home / ".symphonai" / "selection.json").write_text(
            json.dumps(choice), encoding="utf-8",
        )
        selected_provider, selected_host = run_main(keyed_home, {"OPENAI_API_KEY": "test-key"})
        if selected_provider.name != "openai" or selected_provider.model != "gpt-5.4-mini":
            fail(f"main did not start with the remembered provider and model: {selected_provider!r}")
        selected_host.run.select_provider.assert_called_once_with(
            selected_provider, "gpt-5.4-mini", "high", choice,
        )

        fallback_home = root / "fallback-home"
        fallback_home.mkdir()
        (fallback_home / ".symphonai").mkdir()
        (fallback_home / ".symphonai" / "selection.json").write_text(
            json.dumps(choice), encoding="utf-8",
        )
        fallback_provider, fallback_host = run_main(
            fallback_home, {"ANTHROPIC_API_KEY": "test-key"},
        )
        if fallback_provider.name != "anthropic" or fallback_host.run.select_provider.called:
            fail(f"main did not silently use the keyed default provider: {fallback_provider!r}")

        malformed_home = root / "malformed-home"
        malformed_home.mkdir()
        (malformed_home / ".symphonai").mkdir()
        (malformed_home / ".symphonai" / "selection.json").write_text("not json", encoding="utf-8")
        malformed_provider, _ = run_main(malformed_home, {"ANTHROPIC_API_KEY": "test-key"})
        if malformed_provider.name != "anthropic":
            fail(f"main did not start with today's default after malformed selection: {malformed_provider!r}")




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
            kwargs["runtime"].run._memory_extraction_enabled = False
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
            host_main.main(["--repo-root", str(repo)])
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


@check("host_server.mcp_instructions_seed_the_conversation")
def check_mcp_instructions_seed_the_conversation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        server = root / "mcp.py"
        server.write_text(
            "import json, sys\n"
            "for line in sys.stdin:\n"
            " message = json.loads(line)\n"
            " if 'id' not in message: continue\n"
            " result = {'protocolVersion':'2024-11-05','capabilities':{},"
            "'serverInfo':{'name':'test','version':'1'},"
            "'instructions':'Call search before fetch.'}\n"
            " if message.get('method') == 'initialize':\n"
            "  print(json.dumps({'jsonrpc':'2.0','id':message['id'],'result':result}), flush=True)\n"
            " elif message.get('method') == 'tools/list':\n"
            "  print(json.dumps({'jsonrpc':'2.0','id':message['id'],'result':{'tools':[]}}), flush=True)\n",
            encoding="utf-8",
        )
        pool = McpPool((McpServerSpec(
            name="docs",
            command=(sys.executable, str(server)),
            enabled=True,
        ),), cwd=root)
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
        run = HostRun(
            provider,
            PermissionPolicy(root),
            EventBroker(),
            system_prompt="host baseline",
            sessions_root=root / "sessions",
            memory_extraction_enabled=False,
            mcp_tools=pool.start(),
            mcp_instructions=pool.instructions,
        )
        host = HostServer(provider, PermissionPolicy(root), runtime=ProjectRuntime(root, run, pool))
        environment = "fixed environment"
        try:
            host.start()
            with (
                mock.patch.object(host_run_module, "capture_environment", return_value=environment),
                mock.patch.object(provider, "create_response", wraps=provider.create_response) as response_spy,
            ):
                _send_host_prompt(host, "hello")
                actual = [(message.role.value, message.text) for message in response_spy.call_args.args[0].messages]
                actual = [item for item in actual if "Memory persists across conversations" not in (item[1] or "")]
                expected = [
                    ("system", "host baseline"),
                    ("system", environment + "\n\nIn the person's messages, @<path> names a file in this repository. "
                     "Read it with read_file before relying on its contents."),
                    ("system", "MCP servers' instructions:\n\n## docs\nCall search before fetch."),
                    ("user", "hello"),
                ]
                if actual != expected:
                    fail(f"MCP instructions were not seeded after the environment block: {actual!r}")
        finally:
            host.close()

        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
        host = HostServer(
            provider,
            PermissionPolicy(root),
            system_prompt="host baseline",
            sessions_root=root / "no-mcp-sessions",
            memory_extraction_enabled=False,
        )
        try:
            host.start()
            with (
                mock.patch.object(host_run_module, "capture_environment", return_value=environment),
                mock.patch.object(provider, "create_response", wraps=provider.create_response) as response_spy,
            ):
                _send_host_prompt(host, "hello")
                actual = [(message.role.value, message.text) for message in response_spy.call_args.args[0].messages]
                actual = [item for item in actual if "Memory persists across conversations" not in (item[1] or "")]
                expected = [
                    ("system", "host baseline"),
                    ("system", environment + "\n\nIn the person's messages, @<path> names a file in this repository. "
                     "Read it with read_file before relying on its contents."),
                    ("user", "hello"),
                ]
                if actual != expected:
                    fail(f"opening messages changed without MCP instructions: {actual!r}")
        finally:
            host.close()


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
    with tempfile.TemporaryDirectory() as temporary:
        events: list[str] = []
        pool = mock.Mock()
        pool.instructions = ()
        pool.close.side_effect = lambda: events.append("pool.close")
        lsp = mock.Mock()
        lsp.close.side_effect = lambda: events.append("lsp.close")
        extensions = mock.Mock(config=mock.Mock(values={}))
        run = mock.Mock(
            policy=PermissionPolicy(REPO_ROOT),
            extensions=extensions,
            _mcp_tools={},
            sessions_root=Path(temporary) / "sessions",
        )
        run.close.side_effect = lambda: events.append("run.close")
        runtime = ProjectRuntime(REPO_ROOT, run, pool, lsp)
        host = mock.Mock()
        host.run = mock.Mock()
        host.close.side_effect = lambda: (events.append("host.close"), runtime.close())

        def build(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            events.append("project.build")
            return runtime

        def construct_host(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            events.append("host.construct")
            return host

        with (
            mock.patch.object(host_main, "build_project", side_effect=build),
            mock.patch.object(host_main, "_provider", return_value=FakeModelProvider()),
            mock.patch.object(host_main, "HostServer", side_effect=construct_host),
            mock.patch.object(host_main, "prune_sessions"),
            mock.patch.object(host_main, "symphonai_home", return_value=Path(temporary) / "home" / ".symphonai"),
            mock.patch.object(host_main.signal, "signal"),
        ):
            host_main.main(["--repo-root", str(REPO_ROOT)])
        if events != ["project.build", "host.construct", "host.close", "run.close", "lsp.close", "pool.close"]:
            fail(f"project runtime resource lifetime order changed: {events!r}")
        if pool.close.call_count != 1:
            fail(f"MCP pool had more than one owner: {pool.close.call_count} closes")


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
            _, _ = _host_run_probe(
                Path(temporary),
                None,
                mcp_tools=tools,
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
            leader_module,
            "merge_tool_registry",
            wraps=merge_tool_registry,
        ) as merge_spy:
            _start_gated(host_run, provider, "schema", 0)
        if merge_spy.call_count != 1:
            fail("Leader copied the merge instead of calling merge_tool_registry")
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
        with mock.patch.object(
            leader_module,
            "standard_tool_registry",
            return_value=standard,
        ):
            try:
                collision_run.start("collision")
            except ValueError as exc:
                if standard_tool.name not in str(exc):
                    fail(f"host collision omitted the tool name: {exc!r}")
            else:
                fail("HostRun overwrote a colliding standard tool")
        if standard[standard_tool.name] is not standard_tool:
            fail("HostRun collision changed the standard binding")



@check("host_server.builtin_roster_without_definitions")
def check_builtin_roster_without_definitions() -> None:
    for empty_directory in (False, True):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            home = Path(temporary) / "home"
            root.mkdir()
            if empty_directory:
                (root / ".symphonai" / "agents").mkdir(parents=True)
            extensions = load_extensions(repo_root=root, home=home)
            if extensions.agents:
                fail("empty project unexpectedly discovered agent definitions")
            provider = FakeModelProvider([
                ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                    id="worker", name="dispatch_subagent",
                    arguments={"subagent_name": "worker", "task": "work"},
                )])),
                ModelResponse(Message(Role.ASSISTANT, "worker done")),
                ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                    id="explorer", name="dispatch_subagent",
                    arguments={"subagent_name": "explorer", "task": "inspect"},
                )])),
                ModelResponse(Message(Role.ASSISTANT, "explorer done")),
                ModelResponse(Message(Role.ASSISTANT, "done")),
            ])
            run = HostRun(provider, PermissionPolicy(root), EventBroker(),
                          sessions_root=root / "sessions", extensions=extensions)
            session = SessionStore(root / "sessions", "roster", repo_root=root)
            try:
                leader = run._new_leader(session)
                leader.run("delegate")
                if set(leader.subagents) != {"worker", "explorer"}:
                    fail(f"host could not dispatch built-ins with empty={empty_directory}: {leader.subagents!r}")
                worker_tools = set(leader.subagents["worker"].agent._tools)
                explorer_tools = set(leader.subagents["explorer"].agent._tools)
                if worker_tools != set(standard_tool_registry()) | {"read_tool_result", "job_output", "job_stop"}:
                    fail(f"host worker lost its stored-result tool: {worker_tools!r}")
                if explorer_tools != {"read_file", "glob", "grep", "list_files", "web_fetch", "read_tool_result"}:
                    fail(f"host explorer registry was wrong: {explorer_tools!r}")
                roster = leader_module.builtin_subagent_specs(
                    FakeModelProvider(), PermissionPolicy(root), jobs=leader.background_jobs,
                )
                for name in ("explorer", "planner"):
                    if {"job_output", "job_stop"} & set(roster[name].tool_names or ()):
                        fail(f"built-in {name} received job tools without run_shell")
                for name in ("worker", "implementer", "reviewer"):
                    tools = set(roster[name].tool_names or ())
                    if not {"run_shell", "job_output", "job_stop"} <= tools:
                        fail(f"built-in {name} lost job tools: {tools!r}")
                if not {"job_output", "job_stop"} <= set(leader._agent._tools):
                    fail("leader lost its job tools")
                missing = leader._dispatch_tool.execute(ToolCall(
                    id="missing", name="dispatch_subagent",
                    arguments={"subagent_name": "missing", "task": "work"},
                ), PermissionPolicy(root))
                error = missing.error or ""
                if missing.ok or not all(name in error for name in ("missing", "worker", "explorer")):
                    fail(f"undefined host dispatch omitted roster names: {missing!r}")
                if set(leader.subagents) != {"worker", "explorer"}:
                    fail("undefined host dispatch created pool state")
            finally:
                session.close()


@check("host_server.defined_worker_reaches_pool")
def check_defined_worker_reaches_pool() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        agents = root / ".symphonai" / "agents"
        agents.mkdir(parents=True)
        user_base = home / ".symphonai"
        user_base.mkdir(parents=True)
        (user_base / "config.toml").write_text(
            f'[[trust.repositories]]\nroot = {json.dumps(str(root))}\nallow = ["agents"]\n',
            encoding="utf-8",
        )
        (agents / "worker.toml").write_text(
            'prompt = "defined worker prompt"\n'
            'tools = ["read_file"]\n'
            'deadline_seconds = 5\n'
            '[model]\nprovider = "fake"\n'
            '[budget]\nmax_turns = 2\n',
            encoding="utf-8",
        )
        extensions = load_extensions(repo_root=root, home=home)
        if "worker" not in extensions.agents:
            fail("trusted worker definition was not discovered")
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                id="worker", name="dispatch_subagent",
                arguments={"subagent_name": "worker", "task": "inspect"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "child done")),
            ModelResponse(Message(Role.ASSISTANT, "leader done")),
        ])
        run = HostRun(provider, PermissionPolicy(root), EventBroker(),
                      sessions_root=root / "sessions", extensions=extensions)
        session = SessionStore(root / "sessions", "defined-worker", repo_root=root)
        try:
            leader = run._new_leader(session)
            leader.run("delegate")
            record = leader.subagents.get("worker")
            if record is None or set(record.agent._tools) != {"read_file", "read_tool_result"}:
                fail("worker definition did not replace the built-in tool registry")
            if not record.messages or record.messages[0].text != "defined worker prompt":
                fail("worker definition prompt did not reach the spawned agent")
            if record.agent._budget is None or record.agent._budget.max_turns != 2:
                fail("worker definition budget did not reach the spawned agent")
            if not record.runs or record.runs[0].spec.deadline_seconds != 5:
                fail("worker definition deadline did not reach the child run")
        finally:
            session.close()


@check("host_server.agent_memory_survives_new_host")
def check_agent_memory_survives_new_host() -> None:
    class RecordingProvider(FakeModelProvider):
        def __init__(self, responses):  # noqa: ANN001
            super().__init__(responses)
            self.requests = []

        def create_response(self, request, *, cancel=None):  # noqa: ANN001
            self.requests.append(request)
            return super().create_response(request, cancel=cancel)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        memory_root = Path(temporary) / "memory"
        agents = root / ".symphonai" / "agents"
        agents.mkdir(parents=True)
        user_base = home / ".symphonai"
        user_base.mkdir(parents=True)
        (user_base / "config.toml").write_text(
            f'[[trust.repositories]]\nroot = {json.dumps(str(root))}\nallow = ["agents"]\n',
            encoding="utf-8",
        )
        (agents / "reviewer.toml").write_text(
            'prompt = "review carefully"\n'
            '[memory]\nenabled = true\n'
            '[model]\nprovider = "fake"\n',
            encoding="utf-8",
        )
        extensions = load_extensions(repo_root=root, home=home)
        first_provider = RecordingProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "dispatch", "dispatch_subagent",
                {"subagent_name": "reviewer", "task": "review"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "remember", "remember", {"text": "Prefer concise findings."},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "saved")),
            ModelResponse(Message(Role.ASSISTANT, "first done")),
        ])
        first_host = HostRun(
            first_provider,
            PermissionPolicy(root),
            EventBroker(),
            sessions_root=root / "sessions-one",
            memory_root=memory_root,
            extensions=extensions,
        )
        first_session = SessionStore(root / "sessions-one", "first", repo_root=root)
        try:
            first_leader = first_host._new_leader(first_session)
            result = first_leader.run("delegate")
            if result.final_answer != "first done":
                fail(f"first host could not write agent memory: {result!r}")
        finally:
            first_session.close()

        second_provider = RecordingProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "dispatch", "dispatch_subagent",
                {"subagent_name": "reviewer", "task": "review"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "read")),
            ModelResponse(Message(Role.ASSISTANT, "second done")),
        ])
        second_host = HostRun(
            second_provider,
            PermissionPolicy(root),
            EventBroker(),
            sessions_root=root / "sessions-two",
            memory_root=memory_root,
            extensions=extensions,
        )
        second_session = SessionStore(root / "sessions-two", "second", repo_root=root)
        try:
            second_leader = second_host._new_leader(second_session)
            result = second_leader.run("delegate")
            seeded = [message.text for message in second_provider.requests[1].messages]
            if result.final_answer != "second done" or not any("Prefer concise findings." in text for text in seeded):
                fail(f"new host did not reopen the agent memory root: {seeded!r}")
            if second_host._memory is first_host._memory:
                fail("host persistence test reused the same AgentMemory object")
        finally:
            second_session.close()


@check("host_server.memory_routes")
def check_memory_routes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        host = _host(repo_root=root, sessions_root=root / "sessions")
        try:
            memory = host.run.memory_store()
            if memory is None:
                fail("host memory store was unavailable")
            project_entry = memory.write(
                "project note", agent_name="leader", run_id="project-run"
            )
            global_entry = memory.write(
                "global note", agent_name="reviewer", run_id="global-run", scope="global"
            )

            for method, path, body in (
                ("GET", "/memory", None),
                ("POST", "/memory/update", {"id": project_entry.id, "text": "changed"}),
                ("POST", "/memory/delete", {"id": project_entry.id}),
            ):
                connection, response = _request(host, method, path, body=body)
                response.read()
                connection.close()
                if response.status != 401:
                    fail(f"{method} {path} did not require bearer auth: {response.status}")

            connection, response = _request(host, "GET", "/memory", headers=_headers(host))
            listing = json.loads(response.read())
            connection.close()
            if response.status != 200 or set(listing) != {"project", "global"}:
                fail(f"memory listing had the wrong shape: {response.status}, {listing!r}")
            if [entry["id"] for entry in listing["project"]["entries"]] != [project_entry.id]:
                fail(f"project memory listing omitted its id: {listing!r}")
            if [entry["id"] for entry in listing["global"]["entries"]] != [global_entry.id]:
                fail(f"global memory listing omitted its id: {listing!r}")
            serialized = json.dumps(listing)
            if str(root) in serialized or str(Path(temporary)) in serialized:
                fail(f"memory listing exposed an absolute path: {listing!r}")
            listed_entry = listing["project"]["entries"][0]
            if set(listed_entry) != {"id", "text", "written_at", "agent_name"} or not listed_entry["written_at"].endswith("Z"):
                fail(f"memory entry had invalid public fields: {listed_entry!r}")

            connection, response = _request(
                host, "POST", "/memory/update",
                body={"id": project_entry.id, "text": "updated note"},
                headers=_headers(host),
            )
            updated = json.loads(response.read())
            connection.close()
            if response.status != 200 or updated.get("entry", {}).get("text") != "updated note":
                fail(f"memory update failed: {response.status}, {updated!r}")

            for text in ("", "x" * 501):
                connection, response = _request(
                    host, "POST", "/memory/update",
                    body={"id": project_entry.id, "text": text}, headers=_headers(host),
                )
                response.read()
                connection.close()
                if response.status != 400:
                    fail(f"invalid memory text returned {response.status}")

            connection, response = _request(
                host, "POST", "/memory/update",
                body={"id": "unknown", "text": "valid"}, headers=_headers(host),
            )
            response.read()
            connection.close()
            if response.status != 404:
                fail(f"unknown memory update id returned {response.status}")

            connection, response = _request(
                host, "POST", "/memory/delete", body={"id": project_entry.id},
                headers=_headers(host),
            )
            deleted = json.loads(response.read())
            connection.close()
            if response.status != 200 or deleted != {"deleted": project_entry.id}:
                fail(f"memory delete failed: {response.status}, {deleted!r}")
            connection, response = _request(
                host, "POST", "/memory/delete", body={"id": "unknown"},
                headers=_headers(host),
            )
            response.read()
            connection.close()
            if response.status != 404:
                fail(f"unknown memory delete id returned {response.status}")

            unreadable_entry = memory.write(
                "keep this memory", agent_name="leader", run_id="unreadable-run"
            )
            project_path = memory._path("project")
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload[0]["schema_version"] = 999
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            connection, response = _request(host, "GET", "/memory", headers=_headers(host))
            unreadable_listing = json.loads(response.read())
            connection.close()
            if (
                response.status != 200
                or unreadable_listing.get("project")
                != {"entries": [], "error": "unreadable"}
                or [entry["id"] for entry in unreadable_listing["global"]["entries"]]
                != [global_entry.id]
            ):
                fail(f"memory listing did not isolate its unreadable set: {unreadable_listing!r}")

            connection, response = _request(
                host, "POST", "/memory/update",
                body={"id": unreadable_entry.id, "text": "changed"},
                headers=_headers(host),
            )
            body = response.read()
            connection.close()
            if response.status != 503 or body:
                fail(f"update against unreadable memory returned {response.status}, {body!r}")

            connection, response = _request(
                host, "POST", "/memory/delete", body={"id": unreadable_entry.id},
                headers=_headers(host),
            )
            body = response.read()
            connection.close()
            if response.status != 503 or body:
                fail(f"delete against unreadable memory returned {response.status}, {body!r}")

            memory._write_entries(project_path, (unreadable_entry,))
            global_path = memory._path("global")
            global_payload = json.loads(global_path.read_text(encoding="utf-8"))
            global_payload[0]["written_at"] = 1.7e12
            global_path.write_text(json.dumps(global_payload), encoding="utf-8")
            connection, response = _request(host, "GET", "/memory", headers=_headers(host))
            timestamp_listing = json.loads(response.read())
            connection.close()
            if (
                response.status != 200
                or [entry["id"] for entry in timestamp_listing["project"]["entries"]]
                != [unreadable_entry.id]
                or timestamp_listing.get("global") != {"entries": [], "error": "unreadable"}
            ):
                fail(f"timestamp made the memory route unreadable: {timestamp_listing!r}")
        finally:
            host.close()


@check("host_server.defined_leader_provider_model_and_tools")
def check_defined_leader_provider_model_and_tools() -> None:
    class RecordingProvider(FakeModelProvider):
        def __init__(self, name, model, responses):  # noqa: ANN001
            super().__init__(responses)
            self._name = name
            self.model = model
            self.requests = []

        @property
        def name(self) -> str:
            return self._name

        def create_response(self, request, *, cancel=None):  # noqa: ANN001
            self.requests.append(request)
            return super().create_response(request, cancel=cancel)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        agents = root / ".symphonai" / "agents"
        agents.mkdir(parents=True)
        user_base = home / ".symphonai"
        user_base.mkdir(parents=True)
        (user_base / "config.toml").write_text(
            f'[[trust.repositories]]\nroot = {json.dumps(str(root))}\nallow = ["agents"]\n',
            encoding="utf-8",
        )
        (agents / "leader.toml").write_text(
            'prompt = "defined leader prompt"\n'
            'tools = ["read_file"]\n'
            '[model]\nprovider = "defined"\nmodel = "defined-model"\n',
            encoding="utf-8",
        )
        extensions = load_extensions(repo_root=root, home=home)
        conversation = RecordingProvider("conversation", "app-model", [
            ModelResponse(Message(Role.ASSISTANT, "child done")),
        ])
        defined = RecordingProvider("defined", "defined-model", [
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                id="worker", name="dispatch_subagent",
                arguments={"subagent_name": "worker", "task": "inspect"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "leader done")),
        ])
        selections = []

        def select(name, model, base_url):  # noqa: ANN001
            selections.append((name, model, base_url))
            return defined if name == "defined" else None

        run = HostRun(conversation, PermissionPolicy(root), EventBroker(),
                      sessions_root=root / "sessions", extensions=extensions,
                      model="app-model", provider_factory=select)
        session = SessionStore(root / "sessions", "defined-leader", repo_root=root)
        try:
            leader = run._new_leader(session)
            leader.run("delegate")
            if selections != [("defined", "defined-model", None)]:
                fail(f"leader definition did not select its provider and model: {selections!r}")
            if not defined.requests or defined.requests[0].model != "defined-model":
                fail("leader definition model did not reach the provider request")
            if defined.requests[0].messages[0].text != "defined leader prompt":
                fail("leader definition prompt did not reach the provider request")
            if set(leader._agent._tools) != {
                "dispatch_subagent", "read_file", "read_tool_result", "get_goal", "update_goal",
                "update_plan", "ask_user", "propose_plan",
            }:
                fail("leader definition tools did not reach the actual registry")
            if set(leader._leader_spec.tool_names or ()) != set(leader._agent._tools):
                fail("defined leader run spec did not report its actual registry")
            if leader._leader_spec.model.provider != "defined" or leader._leader_spec.model.model != "defined-model":
                fail("defined leader run spec did not report its actual provider and model")
            if not conversation.requests or conversation.requests[0].model != "app-model":
                fail("unmodeled worker did not retain the conversation provider and model")
            if leader.subagents["worker"].agent._provider is not conversation:
                fail("worker switched to the leader definition's provider")
        finally:
            session.close()


@check("host_server.configured_conversation_and_subagent_budgets")
def check_configured_conversation_and_subagent_budgets() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        root.mkdir()
        plain = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "plain"))])
        no_limits = HostRun(plain, PermissionPolicy(root), EventBroker(),
                            max_turns=7, sessions_root=root / "plain-sessions",
                            extensions=load_extensions(repo_root=root, home=home))
        plain_session = SessionStore(root / "plain-sessions", "plain", repo_root=root)
        try:
            plain_leader = no_limits._new_leader(plain_session)
            with mock.patch.object(plain, "create_response", wraps=plain.create_response) as sent:
                plain_leader.run("plain prompt")
            if plain_leader._agent._budget is not None or plain_leader._agent._max_turns != 7:
                fail("omitting budgets changed the host launch turn limit")
            sent_messages = [
                (item.role, item.text)
                for item in sent.call_args.args[0].messages
                if "Memory persists across conversations" not in (item.text or "")
            ]
            if sent_messages != [(Role.USER, "plain prompt")]:
                fail("omitting budgets changed the provider messages")
        finally:
            plain_session.close()

        config_file = root / ".symphonai" / "config.toml"
        config_file.parent.mkdir()
        config_file.write_text(
            '[budgets.leader]\nmax_turns = 3\nwall_seconds = 30\n'
            'max_total_tokens = 100\nmax_cost = "0.50"\n'
            '[budgets.subagent]\nmax_turns = 1\nwall_seconds = 20\n'
            'max_total_tokens = 50\nmax_cost = "0.25"\n',
            encoding="utf-8",
        )
        agents = root / ".symphonai" / "agents"
        agents.mkdir()
        (agents / "worker.toml").write_text(
            'prompt = "worker"\n[model]\nprovider = "fake"\n'
            '[budget]\nmax_turns = 2\n', encoding="utf-8",
        )
        user_base = home / ".symphonai"
        user_base.mkdir(parents=True)
        (user_base / "config.toml").write_text(
            f'[[trust.repositories]]\nroot = {json.dumps(str(root))}\nallow = ["agents"]\n',
            encoding="utf-8",
        )
        table = PriceTable({"priced": ModelPrice(Decimal(1), Decimal(2))}, "USD")
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                id="worker", name="dispatch_subagent",
                arguments={"subagent_name": "worker", "task": "inspect"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "child done")),
            ModelResponse(Message(Role.ASSISTANT, "leader done")),
        ])
        provider.model = "priced"
        run = HostRun(provider, PermissionPolicy(root), EventBroker(),
                      max_turns=9, sessions_root=root / "limited-sessions",
                      extensions=load_extensions(repo_root=root, home=home),
                      price_table=table)
        session = SessionStore(root / "limited-sessions", "limited", repo_root=root)
        try:
            leader = run._new_leader(session)
            leader.run("delegate")
            active = leader._agent._budget
            child = leader.subagents["worker"].agent._budget
            if active is None or (active.max_turns, active.wall_seconds, active.max_total_tokens, active.max_cost, active.price_table) != (3, 30, 100, Decimal("0.50"), table):
                fail(f"configured leader budget did not reach the agent: {active!r}")
            if leader._agent._max_turns != 3 or leader._leader_spec.budget is not active:
                fail("configured leader max_turns did not override launch max_turns")
            if child is None or (child.max_turns, child.wall_seconds, child.max_total_tokens, child.max_cost, child.price_table) != (1, 20, 50, Decimal("0.25"), table):
                fail(f"configured subagent ceiling did not narrow its definition: {child!r}")
            if leader.subagents["worker"].runs[0].spec.budget is not child:
                fail("child run spec lost the configured ceiling")
        finally:
            session.close()
        config_file.write_text('[budgets.subagent]\nwall_seconds = 20\n', encoding="utf-8")
        partial_provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                id="worker", name="dispatch_subagent",
                arguments={"subagent_name": "worker", "task": "inspect"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "child done")),
            ModelResponse(Message(Role.ASSISTANT, "leader done")),
        ])
        partial = HostRun(partial_provider, PermissionPolicy(root), EventBroker(),
                          sessions_root=root / "partial-sessions",
                          extensions=load_extensions(repo_root=root, home=home))
        partial_session = SessionStore(root / "partial-sessions", "partial", repo_root=root)
        try:
            partial_leader = partial._new_leader(partial_session)
            partial_leader.run("delegate")
            child_budget = partial_leader.subagents["worker"].agent._budget
            if child_budget is None or child_budget.max_turns != 2 or child_budget.wall_seconds != 20:
                fail(f"partial ceiling changed an agent's explicit turn limit: {child_budget!r}")
        finally:
            partial_session.close()
        config_file.write_text('[budgets.leader]\nmax_cost = "0.50"\n', encoding="utf-8")
        try:
            HostRun(FakeModelProvider(), PermissionPolicy(root), EventBroker(),
                    sessions_root=root / "refused-sessions",
                    extensions=load_extensions(repo_root=root, home=home))
        except ConfigError as exc:
            if str(config_file) not in str(exc) or "budgets.leader.max_cost" not in str(exc):
                fail(f"host cost refusal lacked the config source and key: {exc}")
        else:
            fail("host accepted a cost ceiling without a price table")


@check("host_server.no_turn_limit_by_default")
def check_no_turn_limit_by_default() -> None:
    if host_main._arguments([]).max_turns is not None:
        fail("host CLI still sets a default --max-turns")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        responses = [
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                f"tool-{index}", "missing_tool", {},
            )]))
            for index in range(30)
        ]
        responses.append(ModelResponse(Message(Role.ASSISTANT, "done")))
        provider = FakeModelProvider(responses)
        host = HostRun(provider, PermissionPolicy(root), EventBroker(), sessions_root=root / "sessions")
        session = SessionStore(root / "sessions", "unlimited", repo_root=root)
        try:
            result = host._new_leader(session).run("work through 30 tool calls")
            if result.stopped_reason != "final_response" or provider.call_count != 31:
                fail(f"default host limit interrupted the 31 turn run: {result!r}")
        finally:
            session.close()
            host.close()

        capped_provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                f"limited-{index}", "missing_tool", {},
            )]))
            for index in range(5)
        ])
        capped = HostRun(capped_provider, PermissionPolicy(root), EventBroker(), max_turns=4,
                         sessions_root=root / "capped-sessions")
        capped_session = SessionStore(root / "capped-sessions", "capped", repo_root=root)
        try:
            result = capped._new_leader(capped_session).run("stop after four turns")
            if result.stopped_reason != "max_turns" or capped_provider.call_count != 4:
                fail(f"explicit host max_turns=4 did not stop at four: {result!r}")
        finally:
            capped_session.close()
            capped.close()


@check("host_server.budget_stop_survives_next_prompt")
def check_budget_stop_survives_next_prompt() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        root.mkdir()
        config_file = root / ".symphonai" / "config.toml"
        config_file.parent.mkdir()
        config_file.write_text('[budgets.leader]\nmax_total_tokens = 5\n', encoding="utf-8")
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                id="missing", name="missing_tool", arguments={},
            )]), usage=Usage(input_tokens=8, output_tokens=0)),
            ModelResponse(Message(Role.ASSISTANT, "second prompt succeeded"),
                          usage=Usage(input_tokens=1, output_tokens=1)),
        ])
        host = HostServer(provider, PermissionPolicy(root),
                          sessions_root=root / "sessions",
                          extensions=load_extensions(repo_root=root, home=home), memory_extraction_enabled=False)
        host.start()
        connection, response = _subscribed_stream(host)
        try:
            with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": str(home / ".symphonai")}):
                _send_host_prompt(host, "first")
                leader = host.run._conversation[0]
                _, stopped = _await_sse(
                    connection, response,
                    lambda frame: frame[0] == "event" and frame[1].get("type") == "RunFinished"
                    and frame[1].get("agent_name") == "leader",
                    what="first budget stop",
                )
                if stopped.get("stopped_reason") != "budget_tokens":
                    fail(f"budget stop reason did not reach the event stream: {stopped!r}")
                _send_host_prompt(host, "second")
                _, finished = _await_sse(
                    connection, response,
                    lambda frame: frame[0] == "event" and frame[1].get("type") == "RunFinished"
                    and frame[1].get("agent_name") == "leader",
                    what="second prompt finish",
                )
                if finished.get("stopped_reason") != "final_response" or provider.call_count != 2:
                    fail(f"next prompt did not finish normally: {finished!r}")
                if host.run._conversation[0] is not leader:
                    fail("budget stop replaced the conversation")
        finally:
            connection.close()
            host.close()


@check("host_server.agent_definition_read_write")
def check_agent_definition_read_write() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        (home / ".symphonai").mkdir(parents=True)
        (home / ".symphonai" / "config.toml").write_text(
            f'[[trust.repositories]]\nroot = {json.dumps(str(root))}\nallow = ["agents"]\n',
            encoding="utf-8",
        )
        extensions = load_extensions(repo_root=root, home=home)
        text = (
            'prompt = "Project reviewer."\n'
            'tools = ["read_file"]\n'
            '[model]\nprovider = "fake"\n'
        )
        host = HostServer(
            FakeModelProvider(),
            PermissionPolicy(root),
            sessions_root=root / "sessions",
            home=home,
            extensions=extensions,
        )
        host.start()
        try:
            connection, response = _request(
                host,
                "POST",
                "/agent",
                body={"name": "reviewer", "scope": "project", "text": text},
                headers=_headers(host),
            )
            try:
                written = json.loads(response.read())
                if response.status != 200 or not written.get("written"):
                    fail(f"trusted project definition was not written: {response.status}, {written!r}")
                if written.get("message") != "definition saved; it will take effect on the next run":
                    fail(f"write response did not defer activation: {written!r}")
            finally:
                connection.close()
            connection, response = _request(
                host,
                "GET",
                f"/agent?{urlencode({'name': 'reviewer', 'scope': 'project'})}",
                headers=_headers(host),
            )
            try:
                opened = json.loads(response.read())
                if response.status != 200 or opened != {
                    "name": "reviewer", "scope": "project", "text": text
                }:
                    fail(f"definition read was not byte-for-byte: {response.status}, {opened!r}")
            finally:
                connection.close()
        finally:
            host.close()
        loaded = load_extensions(repo_root=root, home=home)
        spec = loaded.agents.get("reviewer")
        if (
            spec is None
            or spec.prompt != "Project reviewer."
            or spec.tool_names != ("read_file",)
        ):
            fail(f"next extension load did not see the written definition: {loaded.agents!r}")


@check("host_server.agent_definition_validation_and_atomicity")
def check_agent_definition_validation_and_atomicity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        project_agents = root / ".symphonai" / "agents"
        project_agents.mkdir(parents=True)
        (home / ".symphonai").mkdir(parents=True)
        (home / ".symphonai" / "config.toml").write_text(
            f'[[trust.repositories]]\nroot = {json.dumps(str(root))}\nallow = ["agents"]\n',
            encoding="utf-8",
        )
        target = project_agents / "reviewer.toml"
        original = 'prompt = "keep this file"\n[model]\nprovider = "fake"\n'
        target.write_text(original, encoding="utf-8")
        host = HostServer(
            FakeModelProvider(),
            PermissionPolicy(root),
            sessions_root=root / "sessions",
            home=home,
            extensions=load_extensions(repo_root=root, home=home),
        )
        host.start()
        try:
            invalid = 'prompt = "broken"\n[mystery]\nvalue = true\n'
            connection, response = _request(
                host,
                "POST",
                "/agent",
                body={"name": "reviewer", "scope": "project", "text": invalid},
                headers=_headers(host),
            )
            try:
                body = json.loads(response.read())
                error = body.get("error", "")
                if response.status != 400 or str(target) not in error or "mystery" not in error:
                    fail(f"invalid definition did not preserve loader error: {response.status}, {body!r}")
            finally:
                connection.close()
            if target.read_text(encoding="utf-8") != original:
                fail("invalid replacement changed the existing definition")

            missing = project_agents / "new-agent.toml"
            connection, response = _request(
                host,
                "POST",
                "/agent",
                body={"name": "new-agent", "scope": "project", "text": invalid},
                headers=_headers(host),
            )
            try:
                body = json.loads(response.read())
                if response.status != 400 or missing.exists() or str(missing) not in body.get("error", ""):
                    fail(f"invalid creation reached its target: {response.status}, {body!r}")
            finally:
                connection.close()
        finally:
            host.close()


@check("host_server.agent_definition_trust_and_user_scope")
def check_agent_definition_trust_and_user_scope() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        project_agents = root / ".symphonai" / "agents"
        project_agents.mkdir(parents=True)
        (project_agents / "offered.toml").write_text(
            'prompt = "offered"\n[model]\nprovider = "fake"\n',
            encoding="utf-8",
        )
        (home / ".symphonai").mkdir(parents=True)
        extensions = load_extensions(repo_root=root, home=home)
        before = sorted(path.name for path in project_agents.iterdir())
        host = HostServer(
            FakeModelProvider(),
            PermissionPolicy(root),
            sessions_root=root / "sessions",
            home=home,
            extensions=extensions,
        )
        host.start()
        try:
            connection, response = _request(
                host,
                "POST",
                "/agent",
                body={
                    "name": "blocked",
                    "scope": "project",
                    "text": 'prompt = "blocked"\n[model]\nprovider = "fake"\n',
                },
                headers=_headers(host),
            )
            try:
                body = json.loads(response.read())
                if response.status != 403 or "not trusted for agents" not in body.get("error", ""):
                    fail(f"untrusted project write was accepted: {response.status}, {body!r}")
            finally:
                connection.close()
            after = sorted(path.name for path in project_agents.iterdir())
            if before != after:
                fail(f"untrusted project write changed its directory: {before!r} -> {after!r}")

            user_text = 'prompt = "private"\n[model]\nprovider = "fake"\n'
            connection, response = _request(
                host,
                "POST",
                "/agent",
                body={"name": "private", "scope": "user", "text": user_text},
                headers=_headers(host),
            )
            try:
                body = json.loads(response.read())
                if response.status != 200 or not body.get("written"):
                    fail(f"user definition did not bypass repository trust: {response.status}, {body!r}")
            finally:
                connection.close()
            if (home / ".symphonai" / "agents" / "private.toml").read_text(encoding="utf-8") != user_text:
                fail("user definition text was not written exactly")
            connection, response = _request(
                host,
                "GET",
                f"/agent?{urlencode({'name': 'private', 'scope': 'user'})}",
                headers=_headers(host),
            )
            try:
                body = json.loads(response.read())
                if response.status != 200 or body != {
                    "name": "private", "scope": "user", "text": user_text
                }:
                    fail(f"user definition did not round-trip through read route: {response.status}, {body!r}")
            finally:
                connection.close()
        finally:
            host.close()


@check("host_server.agent_definition_name_validation_and_auth")
def check_agent_definition_name_validation_and_auth() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        home = Path(temporary) / "home"
        host = HostServer(
            FakeModelProvider(),
            PermissionPolicy(root),
            sessions_root=root / "sessions",
            home=home,
        )
        host.start()
        try:
            for method, path, body in (
                ("GET", "/agent?name=bad.txt&scope=project", None),
                ("POST", "/agent", {"name": "../escape", "scope": "project", "text": "x"}),
            ):
                connection, response = _request(host, method, path, body=body)
                try:
                    if response.status != 401 or response.read() != b"":
                        fail(f"unauthenticated definition route succeeded: {method} {path}")
                finally:
                    connection.close()

            for name in ("../escape", "nested/name", "bad.txt", "agent.toml.bak"):
                connection, response = _request(
                    host,
                    "POST",
                    "/agent",
                    body={"name": name, "scope": "project", "text": "x"},
                    headers=_headers(host),
                )
                try:
                    if response.status != 400:
                        fail(f"invalid definition name was not refused: {name!r}, {response.status}")
                finally:
                    connection.close()
            if (root / ".symphonai").exists() or (home / ".symphonai").exists():
                fail("invalid names touched a definition directory")

            protocol = (REPO_ROOT / "symphonai_host" / "PROTOCOL.md").read_text(encoding="utf-8")
            for phrase in ("GET /agent?name=", "POST /agent", "load_agent_file", "next run"):
                if phrase not in protocol:
                    fail(f"agent definition protocol omitted {phrase!r}")
        finally:
            host.close()


def _worktree_route_fixture(root: Path) -> tuple[HostServer, str, Path]:
    root.mkdir(parents=True)
    source = root / "a.py"
    source.write_text("original\n")
    for args in (("init", "-q"), ("config", "user.email", "checks@example.test"),
                 ("config", "user.name", "Checks"), ("add", "a.py"),
                 ("commit", "-qm", "base")):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
        if result.returncode:
            fail(f"git {' '.join(args)} failed: {result.stderr!r}")
    host = HostServer(
        FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "ready"))]),
        PermissionPolicy(repo_root=root, allowed_write_scope=[root], mode="allow"),
        sessions_root=root / "sessions",
    )
    host.start()
    host.run.start("open conversation")
    _wait_until(lambda: not host.run.active, "worktree fixture prompt did not finish")
    session = host.run._conversation[1]
    return host, session.run_id, session.directory


def _create_named_worktree(root: Path, session_directory: Path, name: str = "w1") -> Path:
    admin_path = session_directory / "worktrees" / name
    worktree_root = create_worktree(root, admin_path)
    (worktree_root / "a.py").write_text("worktree\n")
    (worktree_root / "b.py").write_text("new file\n")
    return admin_path


@check("host_server.worktree_apply_checkpoint_and_revert")
def check_worktree_apply_checkpoint_and_revert() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        host, _, session_directory = _worktree_route_fixture(root)
        worktree = _create_named_worktree(root, session_directory)
        try:
            connection, response = _request(host, "GET", "/changes", headers=_headers(host))
            try:
                changes = json.loads(response.read())
                if response.status != 200 or changes["worktrees"][0]["files"] != ["a.py", "b.py"]:
                    fail(f"worktree did not appear in Changes: {response.status}, {changes!r}")
            finally:
                connection.close()
            connection, response = _request(
                host, "POST", "/worktree/apply", body={"name": "w1"}, headers=_headers(host)
            )
            try:
                applied = json.loads(response.read())
                if response.status != 200 or applied != {"applied": ["a.py", "b.py"]}:
                    fail(f"worktree apply failed: {response.status}, {applied!r}")
            finally:
                connection.close()
            if (root / "a.py").read_text() != "worktree\n" or (root / "b.py").read_text() != "new file\n":
                fail("apply did not bring worktree files into the main tree")
            if worktree.exists():
                fail("apply left the worktree directory behind")
            connection, response = _request(host, "GET", "/changes", headers=_headers(host))
            try:
                changes = json.loads(response.read())
                turn = next(item for item in changes["turns"] if item["prompt"] == "Applied worktree w1")
                if turn["paths"] != ["a.py", "b.py"]:
                    fail(f"applied worktree was not checkpointed: {changes!r}")
            finally:
                connection.close()
            connection, response = _request(
                host, "POST", "/changes/revert", body={"key": turn["key"]}, headers=_headers(host)
            )
            try:
                reverted = json.loads(response.read())
                if response.status != 200 or reverted["reverted"] != ["a.py", "b.py"]:
                    fail(f"applied worktree could not be reverted: {response.status}, {reverted!r}")
            finally:
                connection.close()
            if (root / "a.py").read_text() != "original\n" or (root / "b.py").exists():
                fail("reverting the applied worktree did not restore the original tree")
        finally:
            host.close()


@check("host_server.worktree_apply_deletion_and_revert")
def check_worktree_apply_deletion_and_revert() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        host, _, session_directory = _worktree_route_fixture(root)
        (root / "b.py").write_text("original b\n")
        subprocess.run(["git", "add", "b.py"], cwd=root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-qm", "add b"], cwd=root, check=True, capture_output=True,
        )
        directory = session_directory / "worktrees" / "w1"
        worktree_root = create_worktree(root, directory)
        (worktree_root / "a.py").unlink()
        (worktree_root / "b.py").write_text("changed b\n")
        try:
            connection, response = _request(
                host, "POST", "/worktree/apply", body={"name": "w1"}, headers=_headers(host)
            )
            try:
                applied = json.loads(response.read())
                if response.status != 200 or applied != {"applied": ["a.py", "b.py"]}:
                    fail(f"deletion worktree apply failed: {response.status}, {applied!r}")
            finally:
                connection.close()
            if (root / "a.py").exists() or (root / "b.py").read_text() != "changed b\n":
                fail("deletion worktree apply produced the wrong main tree")
            connection, response = _request(host, "GET", "/changes", headers=_headers(host))
            try:
                changes = json.loads(response.read())
                statuses = {item["path"]: item["status"] for item in changes["files"]}
                turn = next(item for item in changes["turns"] if item["prompt"] == "Applied worktree w1")
                if statuses.get("a.py") != "deleted" or statuses.get("b.py") != "modified":
                    fail(f"applied deletion was absent from Changes: {changes!r}")
            finally:
                connection.close()
            connection, response = _request(
                host, "POST", "/changes/revert", body={"key": turn["key"]}, headers=_headers(host)
            )
            try:
                reverted = json.loads(response.read())
                if response.status != 200 or set(reverted["reverted"]) != {"a.py", "b.py"}:
                    fail(f"deletion apply could not be reverted: {response.status}, {reverted!r}")
            finally:
                connection.close()
            if (root / "a.py").read_bytes() != b"original\n" or (root / "b.py").read_text() != "original b\n":
                fail("reverting the deletion apply did not restore both original files")
        finally:
            host.close()


@check("host_server.worktree_apply_subdirectory_paths")
def check_worktree_apply_subdirectory_paths() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        top = Path(temporary) / "top"
        root = top / "app"
        root.mkdir(parents=True)
        (root / "x.py").write_text("original\n")
        for args in (
            ("init", "-q"), ("config", "user.email", "checks@example.test"),
            ("config", "user.name", "Checks"), ("add", "app/x.py"),
            ("commit", "-qm", "base"),
        ):
            result = subprocess.run(["git", *args], cwd=top, capture_output=True, check=False)
            if result.returncode:
                fail(f"git {' '.join(args)} failed: {result.stderr!r}")
        host = HostServer(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "ready"))]),
            PermissionPolicy(repo_root=root, allowed_write_scope=[root], mode="allow"),
            sessions_root=top / "sessions",
        )
        host.start()
        host.run.start("open conversation")
        _wait_until(lambda: not host.run.active, "subdirectory worktree fixture prompt did not finish")
        session = host.run._conversation[1]
        directory = session.directory / "worktrees" / "w1"
        worktree_root = create_worktree(root, directory)
        (worktree_root / "x.py").write_text("edited\n")
        try:
            connection, response = _request(
                host, "POST", "/worktree/apply", body={"name": "w1"}, headers=_headers(host)
            )
            try:
                applied = json.loads(response.read())
                if response.status != 200 or applied != {"applied": ["app/x.py"]}:
                    fail(f"subdirectory worktree apply failed: {response.status}, {applied!r}")
            finally:
                connection.close()
            connection, response = _request(host, "GET", "/changes", headers=_headers(host))
            try:
                changes = json.loads(response.read())
                turn = next(item for item in changes["turns"] if item["prompt"] == "Applied worktree w1")
                if turn["paths"] != ["x.py"] or (root / "x.py").read_text() != "edited\n":
                    fail(f"subdirectory path was not mapped to repo_root: {changes!r}")
            finally:
                connection.close()
            connection, response = _request(
                host, "POST", "/changes/revert", body={"key": turn["key"]}, headers=_headers(host)
            )
            try:
                if response.status != 200:
                    fail(f"subdirectory apply revert failed: {response.status}, {response.read()!r}")
                response.read()
            finally:
                connection.close()
            if (root / "x.py").read_text() != "original\n":
                fail("subdirectory apply revert did not restore x.py")
        finally:
            host.close()


@check("host_server.worktree_apply_conflict_keeps_worktree")
def check_worktree_apply_conflict_keeps_worktree() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        host, _, session_directory = _worktree_route_fixture(root)
        worktree = _create_named_worktree(root, session_directory)
        (root / "a.py").write_text("main tree\n")
        try:
            connection, response = _request(
                host, "POST", "/worktree/apply", body={"name": "w1"}, headers=_headers(host)
            )
            try:
                conflict = json.loads(response.read())
                if response.status != 409 or "error" not in conflict or not conflict["error"]:
                    fail(f"worktree conflict did not return git's reason: {response.status}, {conflict!r}")
            finally:
                connection.close()
            if (root / "a.py").read_text() != "main tree\n" or not worktree.exists():
                fail("conflicting apply changed the main tree or removed its worktree")
        finally:
            host.close()


@check("host_server.worktree_discard_and_reopen")
def check_worktree_discard_and_reopen() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        host, session_id, session_directory = _worktree_route_fixture(root)
        worktree = _create_named_worktree(root, session_directory)
        host.close()
        reopened = HostServer(
            FakeModelProvider(), PermissionPolicy(repo_root=root, allowed_write_scope=[root]),
            sessions_root=root / "sessions",
        )
        reopened.start()
        try:
            reopened.run.open_session(session_id)
            connection, response = _request(reopened, "GET", "/changes", headers=_headers(reopened))
            try:
                changes = json.loads(response.read())
                if response.status != 200 or [item["name"] for item in changes["worktrees"]] != ["w1"]:
                    fail(f"unacted worktree did not survive reopening: {response.status}, {changes!r}")
            finally:
                connection.close()
            connection, response = _request(
                reopened, "POST", "/worktree/discard", body={"name": "w1"}, headers=_headers(reopened)
            )
            try:
                discarded = json.loads(response.read())
                if response.status != 200 or discarded != {"discarded": "w1"}:
                    fail(f"worktree discard failed: {response.status}, {discarded!r}")
            finally:
                connection.close()
            if worktree.exists() or (root / "a.py").read_text() != "original\n" or (root / "b.py").exists():
                fail("discard changed the main tree or retained the worktree")
        finally:
            reopened.close()


@check("host_server.worktree_unknown_and_active")
def check_worktree_unknown_and_active() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        host, _, _ = _worktree_route_fixture(root)
        try:
            for route in ("/worktree/apply", "/worktree/discard"):
                connection, response = _request(
                    host, "POST", route, body={"name": "missing"}, headers=_headers(host)
                )
                try:
                    response.read()
                    if response.status != 404:
                        fail(f"unknown worktree returned {response.status} on {route}")
                finally:
                    connection.close()
            host.run._active = type("Active", (), {"run_id": "active-run"})()
            for method, route, body in (
                ("GET", "/changes", None),
                ("POST", "/worktree/apply", {"name": "missing"}),
                ("POST", "/worktree/discard", {"name": "missing"}),
            ):
                connection, response = _request(host, method, route, body=body, headers=_headers(host))
                try:
                    response.read()
                    if response.status != 409:
                        fail(f"active worktree request returned {response.status} on {route}")
                finally:
                    connection.close()
        finally:
            host.run._active = None
            host.close()


@check("host_server.allow_mode_full_access_from_main")
def check_allow_mode_full_access_from_main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / ".symphonai").mkdir()
        (root / ".symphonai" / "config.toml").write_text(
            "[sandbox]\nshell = false\nnetwork = false\n", encoding="utf-8"
        )
        extensions = load_extensions(repo_root=root, home=root / "home")
        fake_host = mock.Mock()
        with (
            mock.patch.object(host_main, "apply_to_environment"),
            mock.patch.object(host_main, "load_extensions", return_value=extensions),
            mock.patch.object(host_main, "prune_sessions"),
            mock.patch.object(host_main, "_provider", return_value=FakeModelProvider()),
            mock.patch.object(host_main, "McpPool") as pool_class,
            mock.patch.object(host_main, "LspManager"),
            mock.patch.object(host_main, "HostServer", return_value=fake_host) as host_class,
            mock.patch.object(host_main.signal, "signal"),
            mock.patch.dict(os.environ, {"HOME": str(root / "home"), "SYMPHONAI_HOME": ""}),
        ):
            pool_class.return_value.start.return_value = {}
            host_main.main(["--repo-root", str(root)])
        policy = host_class.call_args.args[1]
        if (
            policy.repo_root != root.resolve()
            or policy.allowed_write_scope != [root.resolve()]
            or not policy.shell_enabled
            or policy.shell_allowlist != [()]
            or not policy.fetch_enabled
            or policy.shell_sandbox
            or policy.sandbox_network
        ):
            fail(f"host entry point did not build the allow policy: {policy!r}")

        written = WriteFileTool(ReadLedger()).execute(
            ToolCall("allow-write", "write_file", {"path": "a.py", "content": "ok\n"}),
            policy,
        )
        if not written.ok or (root / "a.py").read_text(encoding="utf-8") != "ok\n":
            fail(f"allow-mode write_file failed: {written!r}")
        if policy.check_write(".env").allowed or policy.check_write(root.parent / "outside").allowed:
            fail("allow mode permitted a forbidden or out-of-repository write")

        shell = RunShellTool().execute(
            ToolCall(
                "allow-shell", "run_shell",
                {"argv": [sys.executable, "--version"], "timeout_seconds": 5},
            ),
            policy,
        )
        if not shell.ok:
            fail(f"allow-mode run_shell failed: {shell!r}")
        if policy.check_shell(["rm", "-rf", "/"]).allowed:
            fail("allow mode permitted an always-denied command")

        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = {"Content-Type": "text/plain", "Content-Length": "2"}
        response.read.return_value = b"ok"
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch("symphonai_api.web.urllib.request.build_opener", return_value=opener):
            fetched = WebFetchTool().execute(
                ToolCall("allow-fetch", "web_fetch", {"url": "https://public.example.invalid/"}),
                policy,
            )
        if not fetched.ok or "ok" not in fetched.content:
            fail(f"allow-mode public fetch failed: {fetched!r}")
        if policy.check_fetch("http://127.0.0.1/").allowed:
            fail("allow mode permitted a local fetch host")

        requests = []
        ask_policy = replace(
            policy,
            mode="ask",
            approval_callback=lambda request: requests.append(request.operation) or True,
        )
        ask_decisions = (
            ask_policy.check_write("ask.txt"),
            ask_policy.check_shell(["pytest", "--version"]),
            ask_policy.check_fetch("https://public.example.invalid/"),
        )
        if not all(decision.allowed for decision in ask_decisions) or requests != [
            "write_file", "run_shell", "web_fetch",
        ]:
            fail(f"ask mode did not ask for all three operations: {requests!r}, {ask_decisions!r}")

        plan_policy = replace(policy, mode="plan")
        plan_decisions = (
            plan_policy.check_write("plan.txt"),
            plan_policy.check_shell(["pytest", "--version"]),
        )
        if any(decision.allowed for decision in plan_decisions):
            fail(f"plan mode permitted a side effect: {plan_decisions!r}")


@check("host_server.builtin_subagents_keep_allow_access")
def check_builtin_subagents_keep_allow_access() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        policy = PermissionPolicy(
            repo_root=root, allowed_write_scope=[root], shell_enabled=True,
            shell_allowlist=[()], fetch_enabled=True,
        )
        specs = leader_module.builtin_subagent_specs(FakeModelProvider(), policy)
        for name in ("worker", "implementer", "reviewer"):
            spec = specs[name]
            if "write_file" not in spec.tool_names or "run_shell" not in spec.tool_names:
                fail(f"built-in {name} lacks expected write/shell tools: {spec.tool_names!r}")
            effective = policy.narrowed(spec.policy_ceiling)
            if not effective.check_write("child.py").allowed or not effective.check_shell(["pytest", "--version"]).allowed:
                fail(f"built-in {name} lost allow-mode access: {effective!r}")
            written = WriteFileTool(ReadLedger()).execute(
                ToolCall(
                    f"{name}-write", "write_file",
                    {"path": f"{name}.py", "content": "ok\n"},
                ),
                effective,
            )
            shell = RunShellTool().execute(
                ToolCall(
                    f"{name}-shell", "run_shell",
                    {"argv": [sys.executable, "--version"], "timeout_seconds": 5},
                ),
                effective,
            )
            if not written.ok or not shell.ok:
                fail(f"built-in {name} could not write or run a command: {written!r}, {shell!r}")


@check("host_server.capability_ceiling_bounds_conversation")
def check_capability_ceiling_bounds_conversation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config_file = root / ".symphonai" / "config.toml"
        config_file.parent.mkdir()

        def policy_for(content: str) -> tuple[PermissionPolicy, Extensions]:
            config_file.write_text(content, encoding="utf-8")
            extensions = load_extensions(repo_root=root, home=root / "home")
            fake_host = mock.Mock()
            with (
                mock.patch.object(host_main, "apply_to_environment"),
                mock.patch.object(host_main, "load_extensions", return_value=extensions),
                mock.patch.object(host_main, "prune_sessions"),
                mock.patch.object(host_main, "_provider", return_value=FakeModelProvider()),
                mock.patch.object(host_main, "McpPool") as pool_class,
                mock.patch.object(host_main, "LspManager"),
                mock.patch.object(host_main, "HostServer", return_value=fake_host) as host_class,
                mock.patch.object(host_main.signal, "signal"),
                mock.patch.dict(os.environ, {"HOME": str(root / "home")}),
            ):
                pool_class.return_value.start.return_value = {}
                host_main.main(["--repo-root", str(root)])
            return host_class.call_args.args[1], extensions

        no_shell, extension = policy_for(
            "[agents.ceiling]\nshell_enabled = false\n"
        )
        if no_shell.check_shell(["pytest"]).allowed:
            fail("conversation policy exceeded shell_enabled=false")
        agent_path = root / "worker.toml"
        agent_path.write_text('prompt = "work"\n', encoding="utf-8")
        inherited = load_agent_file(
            agent_path,
            repo_root=root,
            default_model=ModelSelector("fake"),
            ceiling=extension.ceiling,
        )
        if no_shell.narrowed(inherited.policy_ceiling).check_shell(["pytest"]).allowed:
            fail("table-less agent exceeded the conversation shell ceiling")

        write_limited, _ = policy_for(
            '[agents.ceiling]\nallowed_write_scope = ["src"]\n'
        )
        (root / "src").mkdir(exist_ok=True)
        src_write = WriteFileTool(ReadLedger()).execute(
            ToolCall("ceiling-src", "write_file", {"path": "src/a.py", "content": "ok\n"}),
            write_limited,
        )
        outside_write = WriteFileTool(ReadLedger()).execute(
            ToolCall("ceiling-outside", "write_file", {"path": "b.py", "content": "no\n"}),
            write_limited,
        )
        if not src_write.ok or outside_write.ok or (root / "b.py").exists():
            fail(f"write ceiling did not restrict the conversation: {src_write!r}, {outside_write!r}")

        shell_limited, _ = policy_for(
            '[agents.ceiling]\nshell_allowlist = [["git"]]\n'
        )
        subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
        git_status = RunShellTool().execute(
            ToolCall("ceiling-git", "run_shell", {"argv": ["git", "status"]}),
            shell_limited,
        )
        pytest = RunShellTool().execute(
            ToolCall("ceiling-pytest", "run_shell", {"argv": ["pytest", "--version"]}),
            shell_limited,
        )
        if not git_status.ok or pytest.ok:
            fail(f"shell ceiling did not restrict the conversation: {git_status!r}, {pytest!r}")

        fetch_limited, _ = policy_for(
            '[agents.ceiling]\nfetch_allowlist = ["example.com"]\n'
        )
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = {"Content-Type": "text/plain", "Content-Length": "2"}
        response.read.return_value = b"ok"
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch("symphonai_api.web.urllib.request.build_opener", return_value=opener):
            allowed_fetch = WebFetchTool().execute(
                ToolCall("ceiling-fetch-ok", "web_fetch", {"url": "https://example.com/"}),
                fetch_limited,
            )
            denied_fetch = WebFetchTool().execute(
                ToolCall("ceiling-fetch-no", "web_fetch", {"url": "https://other.org/"}),
                fetch_limited,
            )
        if not allowed_fetch.ok or denied_fetch.ok or opener.open.call_count != 1:
            fail(f"fetch ceiling did not restrict network requests: {allowed_fetch!r}, {denied_fetch!r}")


class _ClosedMarker:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _project_runtime(root: Path, broker: EventBroker, sessions_root: Path, *, provider=None, policy=None, pool=None, lsp=None, approval=None) -> ProjectRuntime:
    run = HostRun(
        provider or FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
        policy or PermissionPolicy(repo_root=root),
        broker,
        sessions_root=sessions_root,
        memory_extraction_enabled=False,
        publish_approval=approval,
    )
    return ProjectRuntime(root.resolve(), run, pool, lsp)


@check("host_server.refused_project_still_opens")
def check_refused_project_still_opens() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        initial = base / "initial"
        project = base / "hook-project"
        home = base / "home"
        initial.mkdir()
        project.mkdir()
        config_path = project / ".symphonai" / "config.toml"
        config_path.parent.mkdir()
        marker = base / "hook-ran"
        command = [
            sys.executable, "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
        ]
        config_path.write_text(
            "[[hooks]]\non = [\"RunStarted\"]\ncommand = "
            + json.dumps(command) + "\n",
            encoding="utf-8",
        )
        private_config_path = project / ".symphonai" / "config.local.toml"
        private_config_path.write_text('[permissions]\nshell = [["python3"]]\n', encoding="utf-8")
        sessions_root = base / "sessions"
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "done")),
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "private-shell", "run_shell", {"argv": ["python3", "--version"]},
            )])),
        ])

        def factory(root, broker, approval):
            extensions = load_extensions(repo_root=root, home=home)
            run = HostRun(
                provider, PermissionPolicy(
                    repo_root=root, shell_enabled=True, shell_allowlist=[], mode="ask",
                ), broker,
                sessions_root=sessions_root,
                memory_extraction_enabled=False,
                extensions=extensions,
                publish_approval=approval,
            )
            return ProjectRuntime(root.resolve(), run)

        host = HostServer(
            provider, PermissionPolicy(
                repo_root=initial, shell_enabled=True, shell_allowlist=[], mode="ask",
            ), sessions_root=sessions_root,
            project_factory=factory,
        )
        host.start()
        try:
            connection, response = _request(
                host, "POST", "/project/open", body={"path": str(project)},
                headers=_headers(host),
            )
            try:
                reply = json.loads(response.read())
                if response.status != 200 or reply.get("repo_root") != str(project.resolve()):
                    fail(f"project with refused hooks failed to open: {response.status}, {reply!r}")
                warnings = reply.get("config_warnings", [])
                if not any(str(config_path) in warning and "hooks were refused" in warning for warning in warnings):
                    fail(f"project/open did not explain the refused hooks: {reply!r}")
                if not any(str(private_config_path) in warning and "permissions" in warning for warning in warnings):
                    fail(f"project/open did not explain the private permissions refusal: {reply!r}")
            finally:
                connection.close()
            connection, response = _request(host, "GET", "/settings", headers=_headers(host))
            try:
                settings = json.loads(response.read()).get("settings", {})
                if settings.get("config_warnings") != warnings:
                    fail(f"settings did not expose config warnings before a conversation: {settings!r}")
            finally:
                connection.close()
            if host.run.extensions.hooks:
                fail(f"refused project hooks were applied: {host.run.extensions.hooks!r}")
            if host.run.project_permission_grants()[0]:
                fail("private project permission grants were honored")
            _send_host_prompt(host, "confirm refused hook does not run")
            if marker.exists():
                fail("a hook from the untrusted project configuration ran")
            conversation = host.run.conversation_stats()
            warnings = [] if conversation is None else conversation.get("config_warnings", [])
            if not any(str(config_path) in warning and "hooks were refused" in warning for warning in warnings):
                fail(f"conversation stats did not explain the refused hooks: {conversation!r}")
            if not any(str(private_config_path) in warning and "permissions" in warning for warning in warnings):
                fail(f"conversation stats omitted the project config warning: {conversation!r}")
        finally:
            host.close()


@check("host_server.project_open_and_switch")
def check_project_open_and_switch() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        a, b, bad = (base / name for name in ("a", "b", "bad"))
        for root in (a, b, bad): root.mkdir()
        (a / "a.py").write_text("a\n")
        (b / "b.py").write_text("b\n")
        sessions_root = base / "sessions"
        created: list[Path] = []
        def factory(root, broker, approval):
            if root == bad.resolve():
                raise ValueError("invalid project config")
            created.append(root)
            return _project_runtime(root, broker, sessions_root, approval=approval)
        host = HostServer(FakeModelProvider(), PermissionPolicy(repo_root=a), sessions_root=sessions_root, project_factory=factory)
        host.start()
        try:
            connection, response = _request(host, "POST", "/project/open", body={"path": str(b)}, headers=_headers(host))
            body = response
            reply = json.loads(body.read())
            connection.close()
            if response.status != 200 or reply != {"repo_root": str(b.resolve()), "name": "b"}:
                fail(f"project switch returned {response.status}: {reply!r}")
            _, project = _request(host, "GET", "/project", headers=_headers(host))
            project_reply = json.loads(project.read())
            _, files = _request(host, "GET", "/files", headers=_headers(host))
            files_reply = json.loads(files.read())
            if project_reply.get("repo_root") != str(b.resolve()) or "b.py" not in json.dumps(files_reply):
                fail(f"current-project routes did not move to B: {project_reply!r}, {files_reply!r}")
            host.open_project(a)
            if host.run.policy.repo_root != a.resolve():
                fail("opening an already-open A did not switch back to A")
            before = host._current_root
            for path in ("relative", "/"):
                connection, response = _request(host, "POST", "/project/open", body={"path": path}, headers=_headers(host))
                response.read(); connection.close()
                if response.status != 400 or host._current_root != before:
                    fail(f"invalid project path changed current project: {path!r}")
            connection, response = _request(host, "POST", "/project/open", body={"path": str(bad)}, headers=_headers(host))
            error = json.loads(response.read()); connection.close()
            if response.status != 400 or error.get("error") != "invalid project config" or host._current_root != before:
                fail(f"invalid config changed current project: {response.status}, {error!r}")
            if len(created) != 2:
                fail(f"unexpected project factory calls: {created!r}")
        finally:
            host.close()
        host = HostServer(FakeModelProvider(), PermissionPolicy(repo_root=a), sessions_root=sessions_root)
        host.start()
        try:
            connection, response = _request(host, "POST", "/project/open", body={"path": str(b)}, headers=_headers(host))
            error = json.loads(response.read()); connection.close()
            if response.status != 400 or error.get("error") != "this host cannot open other projects":
                fail(f"one-project host allowed project opening: {response.status}, {error!r}")
        finally:
            host.close()


@check("host_server.project_keeps_running_in_background")
def check_project_keeps_running_in_background() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary); a = base / "a"; b = base / "b"
        a.mkdir(); b.mkdir(); sessions_root = base / "sessions"
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall("ask-shell", "run_shell", {"argv": ["true"]})])),
            ModelResponse(Message(Role.ASSISTANT, "finished")),
        ])
        policy = PermissionPolicy(repo_root=a, shell_enabled=True, shell_allowlist=[], mode="ask")
        pool_marker, lsp_marker = _ClosedMarker(), _ClosedMarker()
        broker = EventBroker(); subscription = broker.subscribe()
        def factory(root, event_broker, approval):
            if root == a.resolve():
                return _project_runtime(root, event_broker, sessions_root, provider=provider, policy=policy,
                                        pool=pool_marker, lsp=lsp_marker, approval=approval)
            return _project_runtime(root, event_broker, sessions_root, approval=approval)
        host = HostServer(None, None, broker=broker, sessions_root=sessions_root, project_factory=factory)
        initial, _ = host.open_project(a)
        assert initial is not None
        host.start()
        try:
            connection, response = _request(host, "POST", "/prompt", body={"prompt": "wait for approval"}, headers=_headers(host))
            response.read(); connection.close()
            approval = None
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and approval is None:
                pending = host.pending_approvals()
                approval = pending[0] if pending else None
                time.sleep(0.01)
            if approval is None:
                fail("background project did not request shell approval")
            host.open_project(b)
            if not host.run.pending_approvals() and not host.pending_approvals():
                fail("background approval disappeared after switching projects")
            connection, response = _request(host, "POST", "/approval", body={
                "approval_id": approval["approval_id"], "allowed": True, "reason": "",
            }, headers=_headers(host))
            result = json.loads(response.read()); connection.close()
            if response.status != 200 or not result.get("resolved"):
                fail(f"background approval did not resolve: {response.status}, {result!r}")
            _wait_until(lambda: not initial.run.active, "background run did not finish")
            terminal = None
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and terminal is None:
                event = subscription.get(timeout=0.05)
                if isinstance(event, RunFinished): terminal = event
            if terminal is None or host.event_session_id(terminal) is None:
                fail("background RunFinished did not reach the shared broker with its session")
            _wait_until(lambda: a.resolve() not in host._runtimes, "finished background project was not closed")
            if not pool_marker.closed or not lsp_marker.closed:
                fail("background MCP or LSP resource was not closed")
        finally:
            subscription.close(); host.close()


@check("host_server.project_closes_when_idle")
def check_project_closes_when_idle() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary); a = base / "a"; b = base / "b"
        a.mkdir(); b.mkdir(); sessions_root = base / "sessions"
        pool_marker, lsp_marker = _ClosedMarker(), _ClosedMarker()
        broker = EventBroker()
        initial = _project_runtime(a, broker, sessions_root, pool=pool_marker, lsp=lsp_marker)
        host = HostServer(FakeModelProvider(), initial.run.policy, broker=broker, sessions_root=sessions_root, runtime=initial,
                          project_factory=lambda root, event_broker, approval: _project_runtime(root, event_broker, sessions_root, approval=approval))
        host.start()
        try:
            host.open_project(b)
            if a.resolve() in host._runtimes or not pool_marker.closed or not lsp_marker.closed:
                fail("idle project resources remained open after switching away")
        finally:
            host.close()


@check("host_server.project_with_background_conversation_stays_open")
def check_project_with_background_conversation_stays_open() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary); a = base / "a"; b = base / "b"
        a.mkdir(); b.mkdir(); sessions_root = base / "sessions"
        entered, release = threading.Event(), threading.Event()
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "finished"))])
        original_create_response = provider.create_response

        def slow_response(request, *, cancel=None):
            entered.set()
            if not release.wait(5):
                fail("slow fake provider was not released")
            return original_create_response(request, cancel=cancel)

        provider.create_response = slow_response
        def factory(root, broker, approval):
            return _project_runtime(
                root, broker, sessions_root,
                provider=provider if root == a.resolve() else None,
                approval=approval,
            )
        host = HostServer(None, None, sessions_root=sessions_root, project_factory=factory)
        initial, _ = host.open_project(a)
        assert initial is not None
        subscription = host.broker.subscribe()
        host.start()
        try:
            connection, response = _request(
                host, "POST", "/prompt", body={"prompt": "slow turn"}, headers=_headers(host),
            )
            response.read(); connection.close()
            if not entered.wait(2):
                fail("slow provider did not start")
            connection, response = _request(host, "POST", "/session/new", body={}, headers=_headers(host))
            response.read(); connection.close()
            host.open_project(b)
            if a.resolve() not in host._runtimes:
                fail("project with a background conversation closed during its turn")
            release.set()
            terminal = None
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and terminal is None:
                event = subscription.get(timeout=0.05)
                if isinstance(event, RunFinished):
                    terminal = event
            if terminal is None or terminal.stopped_reason != "final_response":
                fail(f"background turn did not finish normally: {terminal!r}")
            _wait_until(lambda: a.resolve() not in host._runtimes, "finished project was not closed")
        finally:
            release.set(); subscription.close(); host.close()


@check("host_server.approval_without_client_refuses")
def check_approval_without_client_refuses() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); root.mkdir(exist_ok=True)
        sessions_root = root / "sessions"
        ask_shell = lambda call_id: ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
            call_id, "run_shell", {"argv": ["true"]},
        )]))
        provider = FakeModelProvider([
            ask_shell("no-client"), ModelResponse(Message(Role.ASSISTANT, "done")),
            ask_shell("with-client"), ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        def factory(project, broker, approval):
            return _project_runtime(
                project, broker, sessions_root, provider=provider,
                policy=PermissionPolicy(repo_root=project, shell_enabled=True, shell_allowlist=[], mode="ask"),
                approval=approval,
            )
        host = HostServer(None, None, sessions_root=sessions_root, project_factory=factory)
        runtime, _ = host.open_project(root)
        assert runtime is not None
        host.start()
        try:
            connection, response = _request(host, "POST", "/prompt", body={"prompt": "ask"}, headers=_headers(host))
            response.read(); connection.close()
            _wait_until(
                lambda: provider.call_count >= 2 and not runtime.run.pending_approvals(),
                "approval without an app client remained pending",
                timeout=1,
            )
            client = host.broker.subscribe()
            try:
                connection, response = _request(host, "POST", "/session/new", body={}, headers=_headers(host))
                response.read(); connection.close()
                connection, response = _request(host, "POST", "/prompt", body={"prompt": "ask with client"}, headers=_headers(host))
                response.read(); connection.close()
                event = None
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and event is None:
                    value = client.get(timeout=0.05)
                    if isinstance(value, ApprovalRequested):
                        event = value
                if event is None:
                    fail("approval was not published with a connected client")
            finally:
                client.close()
        finally:
            host.close()


@check("host_server.git_status_route")
def check_git_status_route() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
        for name in ("a.py", "c.py", "deleted.py"):
            (root / name).write_text(f"{name} = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "a.py", "c.py", "deleted.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
        (root / "a.py").write_text("a.py = 2\n", encoding="utf-8")
        (root / "deleted.py").unlink()
        (root / "staged.py").write_text("staged = True\n", encoding="utf-8")
        subprocess.run(["git", "add", "staged.py"], cwd=root, check=True)
        (root / "untracked.py").write_text("untracked = True\n", encoding="utf-8")
        host = _host(repo_root=root, sessions_root=root / "sessions")
        try:
            def status():
                connection, response = _request(host, "GET", "/git/status", headers=_headers(host))
                result = json.loads(response.read())
                connection.close()
                if response.status != 200:
                    fail(f"git status route returned {response.status}: {result!r}")
                return result

            before_status = subprocess.run(
                ["git", "status", "--porcelain", "-z"], cwd=root, capture_output=True, check=True,
            ).stdout
            index_path = root / ".git" / "index"
            before_index = index_path.read_bytes()
            result = status()
            files = {item["path"]: item for item in result["files"]}
            expected = {
                "a.py": ("modified", False),
                "deleted.py": ("deleted", False),
                "staged.py": ("added", True),
                "untracked.py": ("untracked", False),
            }
            if {path: (files[path]["status"], files[path]["staged"]) for path in expected} != expected:
                fail(f"git status route returned incorrect file states: {files!r}")
            if list(files) != sorted(files):
                fail(f"git status route did not sort files by path: {list(files)!r}")
            if "+untracked = True" not in files["untracked.py"]["diff"]:
                fail(f"untracked diff did not show its added line: {files['untracked.py']!r}")
            if (root / ".git" / "index").read_bytes() != before_index:
                fail("git status route changed the index file")
            after_status = subprocess.run(
                ["git", "status", "--porcelain", "-z"], cwd=root, capture_output=True, check=True,
            ).stdout
            if after_status != before_status:
                fail("git status route changed git status")

            (root / "a.py").write_text("z" * 30_000, encoding="utf-8")
            long_file = next(item for item in status()["files"] if item["path"] == "a.py")
            if len(long_file["diff"]) != 20_000 or long_file["diff_truncated"] is not True:
                fail(f"long git diff was not capped at 20,000 characters: {len(long_file['diff'])}, {long_file['diff_truncated']!r}")

            entered, release = threading.Event(), threading.Event()
            provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
            original = provider.create_response
            def slow_response(request, *, cancel=None):
                entered.set()
                if not release.wait(5):
                    fail("slow provider was not released")
                return original(request, cancel=cancel)
            provider.create_response = slow_response
            host.close()
            host = _host(provider, repo_root=root, sessions_root=root / "sessions")
            connection, response = _request(host, "POST", "/prompt", body={"prompt": "slow"}, headers=_headers(host))
            response.read(); connection.close()
            if not entered.wait(2):
                fail("slow turn did not reach the provider")
            if status().get("repo") is not True:
                fail("git status route did not answer during an active turn")
            release.set()
            _wait_until(lambda: not host.run.active, "slow turn did not finish")
        finally:
            if "release" in locals():
                release.set()
            host.close()

        non_repo = Path(temporary) / "not-repo"
        non_repo.mkdir()
        host = _host(repo_root=non_repo, sessions_root=Path(temporary) / "sessions")
        try:
            connection, response = _request(host, "GET", "/git/status", headers=_headers(host))
            result = json.loads(response.read())
            connection.close()
            if response.status != 200 or result != {"repo": False}:
                fail(f"non-repository git status was not reported: {response.status}, {result!r}")
        finally:
            host.close()


@check("host_server.git_status_many_files")
def check_git_status_many_files() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
        tracked = root / "tracked.txt"
        tracked.write_text("before\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
        tracked.write_text("after\n", encoding="utf-8")
        for index in range(2_000):
            (root / f"u{index:04}.txt").write_text(f"value {index}\n", encoding="utf-8")
        host = _host(repo_root=root, sessions_root=root / "sessions")
        try:
            before_status = subprocess.run(
                ["git", "status", "--porcelain", "-z"], cwd=root,
                capture_output=True, check=True,
            ).stdout
            index_path = root / ".git" / "index"
            before_index = index_path.read_bytes()
            started = time.perf_counter()
            connection, response = _request(host, "GET", "/git/status", headers=_headers(host))
            result = json.loads(response.read())
            connection.close()
            elapsed = time.perf_counter() - started
            if response.status != 200:
                fail(f"large git status returned {response.status}: {result!r}")
            files = result.get("files", [])
            if len(files) != 2_001:
                fail(f"large git status listed {len(files)} files, expected 2001")
            diffed = [item for item in files if item.get("diff") is not None]
            deferred = [item for item in files if item.get("diff_deferred") is True]
            if len(diffed) != 200 or len(deferred) != 1_801:
                fail(f"large git status diffed {len(diffed)} and deferred {len(deferred)} files")
            if elapsed >= 3:
                fail(f"large git status took {elapsed:.3f}s, expected under 3s")
            if index_path.read_bytes() != before_index:
                fail("large git status changed the index")
            after_status = subprocess.run(
                ["git", "status", "--porcelain", "-z"], cwd=root,
                capture_output=True, check=True,
            ).stdout
            if after_status != before_status:
                fail("large git status changed the working tree")

            deferred_path = next(item["path"] for item in files if item.get("diff_deferred"))
            connection, response = _request(
                host, "GET", f"/git/diff?{urlencode({'path': deferred_path})}",
                headers=_headers(host),
            )
            result = json.loads(response.read())
            connection.close()
            if response.status != 200 or f"+value {int(deferred_path[1:5])}" not in result.get("diff", ""):
                fail(f"deferred diff route returned an incorrect diff: {response.status}, {result!r}")
            connection, response = _request(
                host, "GET", f"/git/diff?{urlencode({'path': '../x'})}",
                headers=_headers(host),
            )
            response.read(); connection.close()
            if response.status != 400:
                fail(f"outside diff path returned {response.status}, expected 400")
            connection, response = _request(
                host, "GET", f"/git/diff?{urlencode({'path': 'unchanged.txt'})}",
                headers=_headers(host),
            )
            response.read(); connection.close()
            if response.status != 404:
                fail(f"unchanged diff path returned {response.status}, expected 404")
        finally:
            host.close()


@check("host_server.git_status_untracked_links_and_size")
def check_git_status_untracked_links_and_size() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
        (root / "inside.txt").write_text("inside target secret\n", encoding="utf-8")
        subprocess.run(["git", "add", "inside.txt"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
        (root / "outside-link").symlink_to("/etc/passwd")
        (root / "inside-link").symlink_to("inside.txt")
        large_file = root / "large.txt"
        with large_file.open("wb") as stream:
            block = b"x\n" * (1024 * 512)
            for _ in range(50):
                stream.write(block)
        large_stat = large_file.stat()
        host = _host(repo_root=root, sessions_root=root / "sessions")
        try:
            original_fdopen = os.fdopen
            bytes_read = 0

            class ReadTracker:
                def __init__(self, stream):
                    self.stream = stream

                def __enter__(self):
                    self.stream.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.stream.__exit__(*args)

                def fileno(self):
                    return self.stream.fileno()

                def read(self, size=-1):
                    nonlocal bytes_read
                    content = self.stream.read(size)
                    bytes_read += len(content)
                    return content

            def tracked_fdopen(descriptor, *args, **kwargs):
                stream = original_fdopen(descriptor, *args, **kwargs)
                opened_stat = os.fstat(descriptor)
                if (opened_stat.st_dev, opened_stat.st_ino) == (large_stat.st_dev, large_stat.st_ino):
                    return ReadTracker(stream)
                return stream

            def get(path):
                connection, response = _request(host, "GET", path, headers=_headers(host))
                body = json.loads(response.read())
                connection.close()
                if response.status != 200:
                    fail(f"{path} returned {response.status}: {body!r}")
                return body

            with mock.patch.object(host_server_module.os, "fdopen", new=tracked_fdopen):
                result = get("/git/status")
            files = {item["path"]: item for item in result["files"]}
            for name, target in (("outside-link", "/etc/passwd"), ("inside-link", "inside.txt")):
                diff = files[name]["diff"]
                if "new file mode 120000" not in diff or f"+{target}\n" not in diff:
                    fail(f"untracked symlink {name} was not diffed as a link: {diff!r}")
                if "inside target secret" in diff or "+root:" in diff:
                    fail(f"untracked symlink {name} exposed target contents: {diff!r}")
            expected_link_diff = (
                "diff --git a/outside-link b/outside-link\n"
                "new file mode 120000\n"
                "--- /dev/null\n"
                "+++ b/outside-link\n"
                "@@ -0,0 +1 @@\n"
                "+/etc/passwd\n"
                "\\ No newline at end of file\n"
            )
            if files["outside-link"]["diff"] != expected_link_diff:
                fail(f"outside symlink diff did not match git format: {files['outside-link']['diff']!r}")
            if bytes_read > 1024 * 1024:
                fail(f"untracked file reader consumed {bytes_read} bytes, over 1 MB")
            if bytes_read != 1024 * 1024:
                fail(f"untracked file reader did not track the bounded read: {bytes_read}")
            large_diff = files["large.txt"]
            if len(large_diff["diff"]) != 20_000 or large_diff["diff_truncated"] is not True:
                fail(f"large untracked diff was not capped: {len(large_diff['diff'])}, {large_diff['diff_truncated']!r}")
            if "@@ -0,0 +1,524288 @@" not in large_diff["diff"]:
                fail(f"large untracked diff hunk did not count the lines read: {large_diff['diff'][:100]!r}")

            link_diff = get(f"/git/diff?{urlencode({'path': 'outside-link'})}")
            if link_diff.get("diff") != expected_link_diff:
                fail(f"single-file route did not show the outside symlink itself: {link_diff!r}")
        finally:
            host.close()


@check("host_server.git_commit_route")
def check_git_commit_route() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
        for name in ("a.py", "c.py", "unchanged.py"):
            (root / name).write_text(f"{name} = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "a.py", "c.py", "unchanged.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
        (root / "a.py").write_text("a.py = 2\n", encoding="utf-8")
        (root / "b.py").write_text("b.py = 2\n", encoding="utf-8")
        (root / "c.py").write_text("c.py = staged\n", encoding="utf-8")
        subprocess.run(["git", "add", "c.py"], cwd=root, check=True)
        host = _host(repo_root=root, sessions_root=root / "sessions")

        def post(payload):
            connection, response = _request(host, "POST", "/git/commit", body=payload, headers=_headers(host))
            result = json.loads(response.read())
            connection.close()
            return response.status, result

        try:
            status, result = post({"message": "selected files", "paths": ["a.py", "b.py"]})
            if status != 200 or result.get("paths") != ["a.py", "b.py"]:
                fail(f"git commit did not commit the selected paths: {status}, {result!r}")
            committed_paths = subprocess.run(
                ["git", "show", "--pretty=format:", "--name-only", "HEAD"],
                cwd=root, capture_output=True, check=True,
            ).stdout.decode().splitlines()
            if sorted(path for path in committed_paths if path) != ["a.py", "b.py"]:
                fail(f"commit included paths outside its selection: {committed_paths!r}")
            porcelain = subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, capture_output=True, check=True,
            ).stdout.decode()
            if "M  c.py" not in porcelain or "a.py" in porcelain or "b.py" in porcelain:
                fail(f"unselected staged changes did not remain staged and uncommitted: {porcelain!r}")

            for path in ("unchanged.py", "../x"):
                invalid_status, error = post({"message": "bad path", "paths": [path]})
                if invalid_status != 400:
                    fail(f"unchanged or outside path was accepted: {path!r}, {invalid_status}, {error!r}")

            (root / "a.py").write_text("a.py = 3\n", encoding="utf-8")
            hook = root / ".git" / "hooks" / "pre-commit"
            hook.write_text("#!/bin/sh\necho hook refused >&2\nexit 1\n", encoding="utf-8")
            hook.chmod(0o755)
            index_path = root / ".git" / "index"
            before_index = index_path.read_bytes()
            failed_status, error = post({"message": "hook failure", "paths": ["a.py"]})
            if failed_status != 409 or "hook refused" not in error.get("error", ""):
                fail(f"commit hook failure was not returned: {failed_status}, {error!r}")
            if index_path.read_bytes() != before_index:
                fail("failed commit changed the person's Git index")
            hook.unlink()

            entered, release = threading.Event(), threading.Event()
            provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
            original = provider.create_response
            def slow_response(request, *, cancel=None):
                entered.set()
                if not release.wait(5):
                    fail("slow provider was not released")
                return original(request, cancel=cancel)
            provider.create_response = slow_response
            host.close()


            host = _host(provider, repo_root=root, sessions_root=root / "sessions")
            connection, response = _request(host, "POST", "/prompt", body={"prompt": "slow"}, headers=_headers(host))
            response.read(); connection.close()
            if not entered.wait(2):
                fail("slow turn did not reach the provider")
            head_before = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True,
            ).stdout
            busy_status, error = post({"message": "wait", "paths": ["a.py"]})
            if busy_status != 409 or error.get("error") != "wait for the turn to finish":
                fail(f"commit during a running turn was not refused: {busy_status}, {error!r}")
            head_after = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True,
            ).stdout
            if head_after != head_before:
                fail("commit during a turn changed HEAD")
            release.set()
            _wait_until(lambda: not host.run.active, "slow turn did not finish")
        finally:
            if "release" in locals():
                release.set()
            host.close()


@check("host_server.git_commit_from_subfolder")
def check_git_commit_from_subfolder() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repository = Path(temporary) / "repo"
        project = repository / "pkg"
        project.mkdir(parents=True)
        subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
        subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=repository, check=True)
        subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=repository, check=True)
        changed = project / "m.py"
        changed.write_text("before = True\n", encoding="utf-8")
        subprocess.run(["git", "add", "pkg/m.py"], cwd=repository, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=repository, check=True)
        changed.write_text("after = True\n", encoding="utf-8")
        host = _host(repo_root=project, sessions_root=Path(temporary) / "sessions")
        try:
            connection, response = _request(
                host, "POST", "/git/commit",
                body={"message": "subfolder change", "paths": ["pkg/m.py"]},
                headers=_headers(host),
            )
            result = json.loads(response.read())
            connection.close()
            if response.status != 200 or result.get("paths") != ["pkg/m.py"]:
                fail(f"subfolder commit failed: {response.status}, {result!r}")
            committed = subprocess.run(
                ["git", "show", "--pretty=format:", "--name-only", "HEAD"],
                cwd=repository, capture_output=True, check=True,
            ).stdout.decode().splitlines()
            if [path for path in committed if path] != ["pkg/m.py"]:
                fail(f"subfolder commit included incorrect paths: {committed!r}")
        finally:
            host.close()


def _pull_request_repo(base: Path, *, origin: bool = True, branch: str = "feature", ahead: bool = True, detached: bool = False):
    root = base / "repo"
    root.mkdir()
    bare = base / "origin.git"
    if origin:
        subprocess.run(["git", "init", "--bare", "--quiet", "--initial-branch=main", str(bare)], check=True)
    subprocess.run(["git", "init", "--quiet", "--initial-branch=main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
    (root / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "base commit"], cwd=root, check=True)
    if origin:
        subprocess.run(["git", "remote", "add", "origin", str(bare)], cwd=root, check=True)
        subprocess.run(["git", "push", "--quiet", "-u", "origin", "main"], cwd=root, check=True)
        subprocess.run(["git", "remote", "set-head", "origin", "main"], cwd=root, check=True)
    if branch != "main":
        subprocess.run(["git", "checkout", "--quiet", "-b", branch], cwd=root, check=True)
    if ahead:
        (root / "feature.py").write_text("feature = True\n", encoding="utf-8")
        subprocess.run(["git", "add", "feature.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "feature change"], cwd=root, check=True)
    if detached:
        subprocess.run(["git", "checkout", "--quiet", "--detach", "HEAD"], cwd=root, check=True)
    return root, bare if origin else None


def _fake_gh(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    git_path = shutil.which("git")
    if git_path is None:
        fail("git executable was not found for fake PATH")
    (bin_dir / "git").symlink_to(git_path)
    fake = bin_dir / "gh"
    fake.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = auth ]; then\n"
        "  if [ \"${GH_SIGNED_OUT:-0}\" = 1 ]; then echo signed out >&2; exit 1; fi\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = pr ]; then\n"
        "  printf '%s\\n' \"$@\" >> \"$GH_ARGS_LOG\"\n"
        "  echo \"${GH_PR_URL:-https://github.com/example/repo/pull/1}\"\n"
        "  exit \"${GH_CREATE_EXIT:-0}\"\n"
        "fi\n"
        "echo unexpected gh command >&2; exit 2\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)


@check("host_server.pull_request_preflight")
def check_pull_request_preflight() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        bin_dir = base / "bin"
        _fake_gh(bin_dir)
        path_value = str(bin_dir)
        cases = [
            ("gh-missing", {"origin": True, "branch": "feature", "ahead": True},
             path_value, {}, "The GitHub CLI (gh) is not installed."),
            ("signed-out", {"origin": True, "branch": "feature", "ahead": True},
             path_value, {"GH_SIGNED_OUT": "1"}, "gh is not signed in. Run gh auth login."),
            ("no-origin", {"origin": False, "branch": "feature", "ahead": True},
             path_value, {}, "This repository has no origin remote."),
            ("detached", {"origin": True, "branch": "feature", "ahead": True, "detached": True},
             path_value, {}, "Check out a branch first."),
            ("on-base", {"origin": True, "branch": "main", "ahead": False},
             path_value, {}, "Create a branch first; a pull request comes from a branch other than main."),
            ("no-ahead", {"origin": True, "branch": "feature", "ahead": False},
             path_value, {}, "feature has no commits that main does not."),
        ]
        for name, options, route_path, extra_env, expected in cases:
            case_root = base / name
            case_root.mkdir()
            root, bare = _pull_request_repo(case_root, **options)
            if name == "gh-missing":
                missing_bin = case_root / "git-only"
                missing_bin.mkdir()
                (missing_bin / "git").symlink_to(shutil.which("git"))
                route_path = str(missing_bin)
            env = {"PATH": route_path, "GH_SIGNED_OUT": "0", **extra_env}
            host = _host(repo_root=root, sessions_root=case_root / "sessions")
            try:
                with mock.patch.dict(os.environ, env):
                    connection, response = _request(host, "GET", "/git/pull-request", headers=_headers(host))
                    info = json.loads(response.read())
                    connection.close()
                    if response.status != 200 or info.get("problem") != expected:
                        fail(f"pull-request preflight {name} returned {response.status}, {info!r}")
                    before = None
                    if bare is not None:
                        before_result = subprocess.run(
                            ["git", "--git-dir", str(bare), "rev-parse", "--verify", "refs/heads/feature"],
                            capture_output=True, check=False,
                        )
                        before = before_result.stdout if before_result.returncode == 0 else None
                    connection, response = _request(
                        host, "POST", "/git/pull-request",
                        body={"title": "title", "body": "body"}, headers=_headers(host),
                    )
                    refused = json.loads(response.read())
                    connection.close()
                    if response.status != 409 or refused.get("error") != expected:
                        fail(f"POST did not refuse preflight case {name}: {response.status}, {refused!r}")
                    if bare is not None:
                        after_result = subprocess.run(
                            ["git", "--git-dir", str(bare), "rev-parse", "--verify", "refs/heads/feature"],
                            capture_output=True, check=False,
                        )
                        after = after_result.stdout if after_result.returncode == 0 else None
                        if after != before:
                            fail(f"preflight problem {name} pushed a branch")
            finally:
                host.close()


@check("host_server.pull_request_create")
def check_pull_request_create() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, bare = _pull_request_repo(base, branch="feature", ahead=True)
        (root / "uncommitted.txt").write_text("keep local\n", encoding="utf-8")
        bin_dir = base / "bin"
        _fake_gh(bin_dir)
        args_log = base / "gh-args.log"
        with mock.patch.dict(os.environ, {
            "PATH": str(bin_dir), "GH_ARGS_LOG": str(args_log),
            "GH_PR_URL": "https://github.com/example/repo/pull/42",
        }):
            host = _host(repo_root=root, sessions_root=base / "sessions")
            try:
                connection, response = _request(host, "GET", "/git/pull-request", headers=_headers(host))
                info = json.loads(response.read()); connection.close()
                if response.status != 200 or {
                    key: info.get(key) for key in ("branch", "base", "title", "ahead", "problem")
                } != {
                    "branch": "feature", "base": "main", "title": "feature change", "ahead": 1,
                    "problem": None,
                }:
                    fail(f"pull-request preflight returned wrong branch information: {info!r}")
                connection, response = _request(
                    host, "POST", "/git/pull-request",
                    body={"title": "Fix parser", "body": "Fixes parsing."}, headers=_headers(host),
                )
                created = json.loads(response.read()); connection.close()
                if response.status != 200 or created.get("url") != "https://github.com/example/repo/pull/42":
                    fail(f"pull request was not created through fake gh: {response.status}, {created!r}")
                pushed = subprocess.run(
                    ["git", "--git-dir", str(bare), "rev-parse", "refs/heads/feature"],
                    capture_output=True, check=True,
                ).stdout
                head = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True,
                ).stdout
                if pushed != head:
                    fail("current branch was not pushed to the local origin")
                if args_log.read_text(encoding="utf-8").splitlines() != [
                    "pr", "create", "--title", "Fix parser", "--body", "Fixes parsing.",
                    "--base", "main", "--head", "feature",
                ]:
                    fail(f"gh pr create received the wrong arguments: {args_log.read_text(encoding='utf-8')!r}")
                if not (root / "uncommitted.txt").exists():
                    fail("PR creation changed the uncommitted working tree")
                remote_file = subprocess.run(
                    ["git", "--git-dir", str(bare), "show", "refs/heads/feature:uncommitted.txt"],
                    capture_output=True, check=False,
                )
                if remote_file.returncode == 0:
                    fail("uncommitted content was pushed")

                entered, release = threading.Event(), threading.Event()
                provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
                original = provider.create_response
                def slow_response(request, *, cancel=None):
                    entered.set()
                    if not release.wait(5):
                        fail("slow provider was not released")
                    return original(request, cancel=cancel)
                provider.create_response = slow_response
                host.close()
                host = _host(provider, repo_root=root, sessions_root=base / "sessions")
                connection, response = _request(host, "POST", "/prompt", body={"prompt": "slow"}, headers=_headers(host))
                response.read(); connection.close()
                if not entered.wait(2):
                    fail("slow turn did not reach the provider")
                remote_before = subprocess.run(
                    ["git", "--git-dir", str(bare), "rev-parse", "refs/heads/feature"],
                    capture_output=True, check=True,
                ).stdout
                connection, response = _request(
                    host, "POST", "/git/pull-request",
                    body={"title": "busy", "body": ""}, headers=_headers(host),
                )
                busy = json.loads(response.read()); connection.close()
                if response.status != 409 or busy.get("error") != "wait for the turn to finish":
                    fail(f"pull request during a turn was not refused: {response.status}, {busy!r}")
                remote_after = subprocess.run(
                    ["git", "--git-dir", str(bare), "rev-parse", "refs/heads/feature"],
                    capture_output=True, check=True,
                ).stdout
                if remote_after != remote_before:
                    fail("pull request during a turn pushed the branch")
                release.set()
                _wait_until(lambda: not host.run.active, "slow turn did not finish")
            finally:
                if "release" in locals():
                    release.set()
                host.close()


@check("host_server.pull_request_never_prompts")
def check_pull_request_never_prompts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, _ = _pull_request_repo(base, branch="feature", ahead=True)
        bin_dir = base / "bin"
        _fake_gh(bin_dir)
        real_git = shutil.which("git")
        if real_git is None:
            fail("git executable was not found for non-interactive push check")
        git = bin_dir / "git"
        git.unlink()
        git.write_text(
            f"#!{sys.executable}\n"
            "import json, os, select, sys, time\n"
            "if sys.argv[1:2] == ['push']:\n"
            "    ready, _, _ = select.select([sys.stdin], [], [], 1.5)\n"
            "    stdin_closed = bool(ready) and sys.stdin.read(1) == ''\n"
            "    with open(os.environ['PR_ENV_LOG'], 'w', encoding='utf-8') as output:\n"
            "        json.dump({'terminal': os.environ.get('GIT_TERMINAL_PROMPT'), 'ssh': os.environ.get('GIT_SSH_COMMAND'), 'stdin_closed': stdin_closed}, output)\n"
            "    if not stdin_closed:\n"
            "        print('stdin was not closed', file=sys.stderr)\n"
            "        sys.exit(2)\n"
            "    if os.environ.get('PR_PUSH_SLEEP'):\n"
            "        time.sleep(float(os.environ['PR_PUSH_SLEEP']))\n"
            "    print('missing credentials', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            f"os.execv({real_git!r}, [{real_git!r}, *sys.argv[1:]])\n",
            encoding="utf-8",
        )
        git.chmod(0o755)
        environment = {
            "PATH": str(bin_dir),
            "GH_SIGNED_OUT": "0",
            "PR_ENV_LOG": str(base / "push-env.json"),
            "GIT_SSH_COMMAND": "ssh custom-batch-command",
        }
        host = _host(repo_root=root, sessions_root=base / "sessions")
        try:
            with mock.patch.dict(os.environ, environment):
                started = time.perf_counter()
                connection, response = _request(
                    host, "POST", "/git/pull-request",
                    body={"title": "title", "body": "body"}, headers=_headers(host),
                )
                result = json.loads(response.read())
                connection.close()
                elapsed = time.perf_counter() - started
                if response.status != 409 or "missing credentials" not in result.get("error", ""):
                    fail(f"non-interactive push did not return its error: {response.status}, {result!r}")
                if elapsed >= 2:
                    fail(f"push waited {elapsed:.3f}s for terminal input")
                recorded = json.loads((base / "push-env.json").read_text(encoding="utf-8"))
                if recorded != {
                    "terminal": "0", "ssh": "ssh custom-batch-command", "stdin_closed": True,
                }:
                    fail(f"push did not receive non-interactive settings: {recorded!r}")

                with mock.patch.object(host_server_module, "PULL_REQUEST_TIMEOUT_SECONDS", 0.1):
                    with mock.patch.dict(os.environ, {**environment, "PR_PUSH_SLEEP": "1"}):
                        connection, response = _request(
                            host, "POST", "/git/pull-request",
                            body={"title": "title", "body": "body"}, headers=_headers(host),
                        )
                        result = json.loads(response.read())
                        connection.close()
                if response.status != 409 or "git push timed out after 120 s" not in result.get("error", ""):
                    fail(f"push timeout was not named in the response: {response.status}, {result!r}")
        finally:
            host.close()


@check("host_server.session_belongs_to_project")
def check_session_belongs_to_project() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary); a = base / "a"; b = base / "b"
        a.mkdir(); b.mkdir(); sessions_root = base / "sessions"
        host = HostServer(FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]), PermissionPolicy(repo_root=a),
                          sessions_root=sessions_root,
                          project_factory=lambda root, broker, approval: _project_runtime(root, broker, sessions_root, approval=approval))
        host.start()
        try:
            connection, response = _request(host, "POST", "/prompt", body={"prompt": "create A session"}, headers=_headers(host))
            session_id = json.loads(response.read())["run_id"]; connection.close()
            _wait_until(lambda: not host.run.active, "A session did not finish")
            host.open_project(b)
            connection, response = _request(host, "POST", "/session/open", body={"run_id": session_id}, headers=_headers(host))
            error = json.loads(response.read()); connection.close()
            if response.status != 409 or error.get("error") != "that conversation belongs to another project":
                fail(f"cross-project session open was not rejected: {response.status}, {error!r}")
        finally:
            host.close()


@check("host_server.start_project_rule")
def check_start_project_rule() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary); home = base / "home"; project_home = base / ".symphonai"
        project = base / "repo"; home.mkdir(); project.mkdir(); project_home.mkdir()
        (project_home / "projects.json").write_text(json.dumps([str(project), str(base / "missing")]))
        if choose_start_root(None, Path("/"), home, project_home) != project.resolve():
            fail("startup from / did not choose the first existing recent project")
        (project_home / "projects.json").unlink()
        if choose_start_root(None, home, home, project_home) is not None:
            fail("startup from home without a recent project did not remain projectless")
        if choose_start_root(None, project, home, project_home) != project.resolve():
            fail("startup inside a repository did not choose the current directory")
        if choose_start_root(project, Path("/"), home, project_home) != project.resolve():
            fail("explicit --repo-root did not take precedence")


@check("host_server.no_project_routes")
def check_no_project_routes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary); repo = base / "repo"; repo.mkdir(); sessions = base / "sessions"
        host = HostServer(
            None, None, sessions_root=sessions,
            project_factory=lambda root, broker, approval: _project_runtime(root, broker, sessions, approval=approval),
        )
        host.start()
        try:
            _, response = _request(host, "GET", "/project", headers=_headers(host))
            project = json.loads(response.read())
            if response.status != 200 or project != {"repo_root": None, "name": None}:
                fail(f"projectless /project reply was wrong: {response.status}, {project!r}")
            for method, path, body in (("POST", "/prompt", {"prompt": "hello"}), ("GET", "/files", None)):
                connection, response = _request(host, method, path, body=body, headers=_headers(host))
                error = json.loads(response.read()); connection.close()
                if response.status != 409 or error.get("error") != "open a project first":
                    fail(f"projectless {method} {path} was not gated: {response.status}, {error!r}")
            _, response = _request(host, "GET", "/settings", headers=_headers(host))
            settings = json.loads(response.read())
            if response.status != 200 or settings.get("settings", {}).get("skills") != []:
                fail(f"projectless settings were unavailable or project-scoped: {settings!r}")
            _, response = _request(host, "GET", "/models?provider=anthropic", headers=_headers(host))
            if response.status != 200:
                fail(f"projectless models returned {response.status}")
            connection, response = _request(host, "POST", "/project/open", body={"path": str(repo)}, headers=_headers(host))
            response.read(); connection.close()
            connection, response = _request(host, "POST", "/prompt", body={"prompt": "now open"}, headers=_headers(host))
            result = json.loads(response.read()); connection.close()
            if response.status != 200 or not result.get("run_id"):
                fail(f"prompt did not work after opening a project: {response.status}, {result!r}")
        finally:
            host.close()


@check("host_server.projects_and_directories")
def check_projects_and_directories() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary); root = base / "repo"; home = base / "home"; root.mkdir(); home.mkdir()
        tree = base / "tree"; tree.mkdir()
        for name in ("a", "B", ".git"): (tree / name).mkdir()
        (tree / "file.txt").write_text("not read")
        sessions = base / "sessions"; missing = base / "deleted-project"
        host = HostServer(FakeModelProvider(), PermissionPolicy(repo_root=root), sessions_root=sessions, home=home)
        host.start()
        try:
            recent_path = host._home / "projects.json"
            recent_path.parent.mkdir(parents=True, exist_ok=True)
            recent_path.write_text(json.dumps([str(root), str(missing)]))
            _, response = _request(host, "GET", "/projects", headers=_headers(host))
            projects = json.loads(response.read())
            if response.status != 200 or projects.get("current", {}).get("repo_root") != str(root.resolve()) or projects.get("recent", [{}])[1].get("exists") is not False:
                fail(f"project listing omitted current or deleted recent roots: {projects!r}")
            _, response = _request(host, "GET", f"/directories?{urlencode({'path': str(tree)})}", headers=_headers(host))
            listing = json.loads(response.read())
            if response.status != 200 or [entry["name"] for entry in listing.get("entries", [])] != ["a", "B"]:
                fail(f"directory listing included hidden paths or wrong order: {listing!r}")
            _, response = _request(host, "GET", f"/directories?{urlencode({'path': str(tree / 'file.txt')})}", headers=_headers(host))
            if response.status != 404:
                fail(f"file path was not rejected as a missing directory: {response.status}")
            _, response = _request(host, "GET", f"/directories?{urlencode({'path': '/'})}", headers=_headers(host))
            root_reply = json.loads(response.read())
            if response.status != 200 or root_reply.get("parent") is not None:
                fail(f"root directory parent was not null: {root_reply!r}")
        finally:
            host.close()


@check("host_server.jobs_live_per_conversation")
def check_jobs_live_per_conversation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = _host(repo_root=root, sessions_root=root / "sessions")
        try:
            host.run.start("create conversation")
            _wait_until(lambda: not host.run.active, "initial conversation did not finish")
            leader = host.run._conversation[0]
            manager = leader.background_jobs
            if manager is None:
                fail("host conversation did not receive a job manager")
            job_id = manager.start(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                str(root), shell_environment(),
            )
            process = manager.get(job_id).process
            connection, response = _request(
                host, "POST", "/session/new", body={}, headers=_headers(host),
            )
            response.read(); connection.close()
            if response.status != 200:
                fail(f"new session did not end its conversation: {response.status}")
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                fail("ending the conversation left its background job process running")
            result = manager.read(job_id)
            if result is None or result["status"] != "stopped":
                fail(f"closed conversation job did not end as stopped: {result!r}")
        finally:
            host.close()


@check("host_server.jobs_routes")
def check_jobs_routes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = _host(repo_root=root, sessions_root=root / "sessions")
        events = host.broker.subscribe()
        try:
            connection, response = _request(
                host, "POST", "/jobs/start", body={"command": "echo hi; sleep 30"},
                headers=_headers(host),
            )
            started = json.loads(response.read()); connection.close()
            if response.status != 200 or not started.get("id", "").startswith("job_"):
                fail(f"person job did not start: {response.status}, {started!r}")
            start_event = events.get(timeout=1)
            if not isinstance(start_event, JobChanged) or start_event.job_id != started["id"] or start_event.status != "running" or start_event.session_id is not None:
                fail(f"person job start event was malformed: {start_event!r}")
            _wait_until(
                lambda: any(job["id"] == started["id"] and "hi" in job["output_tail"] for job in host.run.list_jobs()),
                "person job output was not captured",
            )
            connection, response = _request(host, "GET", "/jobs", headers=_headers(host))
            listing = json.loads(response.read()); connection.close()
            person = next((job for job in listing.get("jobs", []) if job["id"] == started["id"]), None)
            if response.status != 200 or person is None or person["owner"] != "person" or person["label"] != "echo hi; sleep 30" or person["status"] != "running" or "hi" not in person["output_tail"]:
                fail(f"person job was not listed with its output: {listing!r}")

            connection, response = _request(
                host, "POST", "/jobs/stop", body={"id": started["id"]}, headers=_headers(host),
            )
            stopped = json.loads(response.read()); connection.close()
            if response.status != 200 or stopped.get("status") != "stopped":
                fail(f"person job did not stop: {response.status}, {stopped!r}")
            end_event = events.get(timeout=1)
            if not isinstance(end_event, JobChanged) or end_event.job_id != started["id"] or end_event.status != "stopped":
                fail(f"person job end event was malformed: {end_event!r}")
            connection, response = _request(
                host, "POST", "/jobs/stop", body={"id": "job_unknown"}, headers=_headers(host),
            )
            response.read(); connection.close()
            if response.status != 404:
                fail(f"unknown job stop did not return 404: {response.status}")

            host.run.start("open a conversation")
            _wait_until(lambda: not host.run.active, "agent-job conversation did not finish")
            events.close()
            events = host.broker.subscribe()
            leader, session = host.run._conversation
            manager = leader.background_jobs
            if manager is None:
                fail("agent conversation had no background job manager")
            agent_id = manager.start(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                str(root), shell_environment(), label="python",
            )
            agent_event = events.get(timeout=1)
            if not isinstance(agent_event, JobChanged) or agent_event.job_id != agent_id or agent_event.session_id != session.run_id:
                fail(f"agent job event omitted its session: {agent_event!r}")
            connection, response = _request(host, "GET", "/jobs", headers=_headers(host))
            listing = json.loads(response.read()); connection.close()
            agent = next((job for job in listing.get("jobs", []) if job["id"] == agent_id), None)
            if agent is None or agent["owner"] != "agent" or agent["session_id"] != session.run_id or agent["label"] != "python" or "sleep 30" in json.dumps(agent):
                fail(f"agent job metadata leaked command or omitted label/session: {agent!r}")
            host.run.stop_job(agent_id)
        finally:
            events.close()
            host.close()


@check("host_server.jobs_keep_a_project_open")
def check_jobs_keep_a_project_open() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        a, b = base / "a", base / "b"
        a.mkdir(); b.mkdir()
        sessions = base / "sessions"

        def factory(root, broker, approval):
            return _project_runtime(root, broker, sessions, approval=approval)

        host = HostServer(None, None, sessions_root=sessions, project_factory=factory)
        runtime_a, _ = host.open_project(a)
        assert runtime_a is not None
        host.start()
        try:
            runtime_a.run.start("open conversation")
            _wait_until(lambda: not runtime_a.run.active, "project A conversation did not finish")
            leader = runtime_a.run._conversation[0]
            agent_manager = leader.background_jobs
            agent_id = agent_manager.start(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                str(a), shell_environment(), label="python",
            )
            person_id = runtime_a.run.start_job("sleep 30")
            host.open_project(b)
            if a.resolve() not in host._runtimes:
                fail("project A closed while an agent job was running")
            runtime_a.run.stop_job(agent_id)
            if a.resolve() not in host._runtimes:
                fail("project A closed while the person's job was running")
            runtime_a.run.stop_job(person_id)
            _wait_until(lambda: a.resolve() not in host._runtimes, "idle project A stayed open after its jobs stopped")
        finally:
            host.close()


def _headless_invoke(
    root: Path, provider, arguments: list[str], *, stdin_text: str = "",
    provider_factory=None,
):
    home = root.parent / "home"
    sessions = root.parent / "sessions"
    stdout, stderr = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, {
        "SYMPHONAI_HOME": str(home),
        "SYMPHONAI_SESSIONS_DIR": str(sessions),
    }):
        provider_patch = (
            mock.patch.object(headless_module, "_provider", side_effect=provider_factory)
            if provider_factory is not None
            else mock.patch.object(headless_module, "_provider", return_value=provider)
        )
        with provider_patch:
            with mock.patch.object(headless_module.sys, "stdin", io.StringIO(stdin_text)):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = headless_module.main(arguments)
    return code, stdout.getvalue(), stderr.getvalue(), sessions


@check("host_server.headless_prompt")
def check_headless_prompt() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        code, output, error, _ = _headless_invoke(
            root, FakeModelProvider(), ["prompt", "worker", "--repo-root", str(root)]
        )
        if code != 0 or "You are a subagent in SymphonAI." not in output or "built-in: SUBAGENT_BASE_PROMPT" not in error:
            fail(f"prompt command did not show worker layers: {code}, {output!r}, {error!r}")
        code, _output, error, _ = _headless_invoke(
            root, FakeModelProvider(), ["prompt", "nobody", "--repo-root", str(root)]
        )
        if code != 2 or "known roles:" not in error or "worker" not in error:
            fail(f"unknown prompt role did not list known roles: {code}, {error!r}")


@check("host_server.headless_review")
def check_headless_review() -> None:
    class RecordingProvider(FakeModelProvider):
        def __init__(self, responses=None) -> None:
            super().__init__(responses or [ModelResponse(Message(Role.ASSISTANT, "No findings."))])
            self.requests = []

        def create_response(self, request, *, cancel=None):
            self.requests.append(request)
            return super().create_response(request, cancel=cancel)

    def app_review_prompt(source: str) -> str | None:
        prefix = "export const REVIEW_PROMPT = `"
        suffix = "`;\n\nfunction promptCommand"
        if prefix not in source or suffix not in source:
            return None
        prompt = source.split(prefix, 1)[1].split(suffix, 1)[0]
        return prompt.replace("\\`", "`")

    def request_tool_names(request) -> set[str]:
        return {
            schema.get("function", {}).get("name", schema.get("name"))
            for schema in request.tools
        }

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        modes = []
        original_select_mode = HostRun.select_mode

        def record_mode(run, mode):
            modes.append(mode)
            return original_select_mode(run, mode)

        provider = RecordingProvider()
        with mock.patch.object(HostRun, "select_mode", record_mode):
            code, output, _error, _sessions = _headless_invoke(
                root, provider,
                ["review", "--json", "--repo-root", str(root), "--provider", "openai"],
            )
        result = json.loads(output)
        user_messages = [
            message.text
            for message in provider.requests[0].messages
            if message.role is Role.USER
        ]
        if (code != 0 or result.get("status") != "finished"
                or result.get("result") != "No findings."
                or user_messages != [headless_module.REVIEW_PROMPT] or modes != ["plan"]
                or "dispatch_subagent" in request_tool_names(provider.requests[0])):
            fail(f"headless review did not use the app prompt and plan mode: {code}, {result!r}, {user_messages!r}, {modes!r}")

        provider = RecordingProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "dispatch", "dispatch_subagent", {"subagent_name": "worker", "task": "inspect"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "No findings.")),
        ])
        code, _output, _error, _sessions = _headless_invoke(
            root, provider,
            ["review", "--json", "--repo-root", str(root), "--provider", "openai"],
        )
        tool_results = [
            message.tool_result.error or message.tool_result.content
            for message in provider.requests[1].messages
            if message.tool_result is not None
        ]
        if (code != 0 or "dispatch_subagent" in request_tool_names(provider.requests[0])
                or not tool_results or any("unknown tool" not in str(value) for value in tool_results)):
            fail(f"headless review did not refuse delegation by default: {code}, {tool_results!r}")

        provider = RecordingProvider()
        code, _output, _error, _sessions = _headless_invoke(
            root, provider,
            ["review", "--json", "--repo-root", str(root), "--provider", "openai",
             "--allowed-tools", "read_file,dispatch_subagent"],
        )
        if code != 0 or request_tool_names(provider.requests[0]) != {"read_file", "dispatch_subagent"}:
            fail(f"review --allowed-tools did not preserve the explicit delegation allow: {code}, {provider.requests!r}")

        provider = RecordingProvider()
        code, _output, _error, _sessions = _headless_invoke(
            root, provider,
            ["run", "task", "--json", "--repo-root", str(root), "--provider", "openai"],
        )
        if code != 0 or "dispatch_subagent" not in request_tool_names(provider.requests[0]):
            fail(f"headless run unexpectedly lost dispatch_subagent: {code}, {provider.requests!r}")

        provider = RecordingProvider()
        code, _output, _error, _sessions = _headless_invoke(
            root, provider,
            ["review", "--repo-root", str(root), "--provider", "openai",
             "--base", "origin/main", "--focus", "errors"],
        )
        messages = [
            message.text
            for message in provider.requests[0].messages
            if message.role is Role.USER
        ]
        based_prompt = messages[0] if messages else ""
        if (code != 0 or not based_prompt.startswith(
                "Review the changes between origin/main and the working tree (git diff origin/main)."
        ) or not based_prompt.endswith("Focus on: errors")):
            fail(f"headless review did not apply --base and --focus: {code}, {based_prompt!r}")

        app_source = (REPO_ROOT / "symphonai_app" / "src" / "app.js").read_text(encoding="utf-8")
        if app_review_prompt(app_source) != headless_module.REVIEW_PROMPT:
            fail("headless review prompt drifted from REVIEW_PROMPT in app.js")
        with tempfile.TemporaryDirectory() as temporary_copy:
            edited_source = app_source.replace(
                "Report findings, most severe first.",
                "Report issues, most severe first.",
                1,
            )
            edited_path = Path(temporary_copy) / "app.js"
            edited_path.write_text(edited_source, encoding="utf-8")
            if app_review_prompt(edited_path.read_text(encoding="utf-8")) == headless_module.REVIEW_PROMPT:
                fail("the review prompt drift check accepted an edited app.js prompt")


@check("host_server.headless_output_schema")
def check_headless_output_schema() -> None:
    schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        schema_path = root / "schema.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, '{"ok": true}'))])
        code, output, _error, _sessions = _headless_invoke(
            root, provider,
            ["run", "return json", "--repo-root", str(root), "--provider", "openai",
             "--output-schema", str(schema_path), "--json"],
        )
        result = json.loads(output)
        if code != 0 or result.get("output") != {"ok": True} or result.get("status") != "finished":
            fail(f"valid output schema was not returned: {code}, {result!r}")

        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "hello"))])
        code, output, _error, _sessions = _headless_invoke(
            root, provider,
            ["run", "return json", "--repo-root", str(root), "--provider", "openai",
             "--output-schema", str(schema_path), "--json"],
        )
        result = json.loads(output)
        if (code != 1 or result.get("status") != "failed"
                or not result.get("error", "").startswith("final answer did not match the output schema:")):
            fail(f"invalid output schema result was accepted: {code}, {result!r}")

        schema_path.write_text("[]", encoding="utf-8")
        code, _output, error, _sessions = _headless_invoke(
            root, FakeModelProvider([]),
            ["run", "task", "--repo-root", str(root), "--provider", "openai",
             "--output-schema", str(schema_path)],
        )
        if code != 2 or "invalid output schema" not in error:
            fail(f"non-object output schema did not exit 2: {code}, {error!r}")


@check("host_server.headless_tool_lists")
def check_headless_tool_lists() -> None:
    class RecordingProvider(FakeModelProvider):
        def __init__(self, responses) -> None:
            super().__init__(responses)
            self.requests = []

        def create_response(self, request, *, cancel=None):
            self.requests.append(request)
            return super().create_response(request, cancel=cancel)

    def request_tool_names(request) -> set[str]:
        return {
            schema.get("function", {}).get("name", schema.get("name"))
            for schema in request.tools
        }

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        provider = RecordingProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
        code, _output, _error, _sessions = _headless_invoke(
            root, provider,
            ["run", "task", "--repo-root", str(root), "--provider", "openai", "--json",
             "--allowed-tools", "read_file,grep"],
        )
        if code != 0 or request_tool_names(provider.requests[0]) != {"read_file", "grep"}:
            fail(f"--allowed-tools did not constrain the leader schemas: {code}, {provider.requests!r}")

        provider = RecordingProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "blocked-shell", "run_shell", {"argv": ["echo", "unsafe"]},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        code, _output, _error, _sessions = _headless_invoke(
            root, provider,
            ["run", "task", "--repo-root", str(root), "--provider", "openai", "--json",
             "--allowed-tools", "read_file,grep"],
        )
        tool_results = [
            message.tool_result.error or message.tool_result.content
            for message in provider.requests[1].messages
            if message.tool_result is not None
        ]
        if code != 0 or any("unknown tool" not in str(value) for value in tool_results):
            fail(f"a removed tool call was not refused: {code}, {tool_results!r}")

        provider = RecordingProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "dispatch", "dispatch_subagent", {"subagent_name": "worker", "task": "inspect"},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "worker done")),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        code, _output, _error, _sessions = _headless_invoke(
            root, provider,
            ["run", "task", "--repo-root", str(root), "--provider", "openai", "--json",
             "--disallowed-tools", "run_shell"],
        )
        if (code != 0 or len(provider.requests) < 2
                or "run_shell" in request_tool_names(provider.requests[0])
                or "run_shell" in request_tool_names(provider.requests[1])):
            fail(f"--disallowed-tools did not constrain the leader and worker: {code}, {provider.requests!r}")

        code, _output, error, _sessions = _headless_invoke(
            root, FakeModelProvider([]),
            ["run", "task", "--repo-root", str(root), "--provider", "openai", "--allowed-tools", "nope"],
        )
        if code != 2 or "nope" not in error or "read_file" not in error:
            fail(f"unknown tool did not report the valid tools: {code}, {error!r}")


@check("host_server.doctor")
def check_doctor() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "project"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet", str(root)], check=True)
        home = base / ".symphonai"
        home.mkdir(mode=0o700)
        home.chmod(0o700)
        env_name = headless_module.PROVIDERS[0][1]
        secret = "doctor-secret-must-not-appear"
        isolated_env = {"PATH": os.environ.get("PATH", ""), "SYMPHONAI_HOME": str(home), env_name: secret}
        output, error = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, isolated_env, clear=True), \
                mock.patch.object(headless_module, "_sandbox_available", return_value=True), \
                mock.patch.object(headless_module, "_app_root", return_value=root), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = headless_module.main(["doctor", "--repo-root", str(root)])
        if code != 0 or secret in output.getvalue() or "ok API keys" not in output.getvalue():
            fail(f"doctor did not pass safely with a configured key: {code}, {output.getvalue()!r}")

        output = io.StringIO()
        with mock.patch.dict(os.environ, {"PATH": os.environ.get("PATH", ""), "SYMPHONAI_HOME": str(home)}, clear=True), \
                mock.patch.object(headless_module, "_sandbox_available", return_value=False), \
                mock.patch.object(headless_module, "_app_root", return_value=root), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            code = headless_module.main(["doctor", "--repo-root", str(root), "--json"])
        result = json.loads(output.getvalue())
        key_check = next(item for item in result["checks"] if item["name"] == "API keys")
        if code != 1 or key_check["status"] != "fail" or set(result) != {"checks"}:
            fail(f"doctor did not report missing keys in the documented JSON shape: {code}, {result!r}")


@check("host_server.no_sleep_while_running")
def check_no_sleep_while_running() -> None:
    class KeeperProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.terminated = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.terminate()

    class BlockingProvider(FakeModelProvider):
        def __init__(self) -> None:
            super().__init__([ModelResponse(Message(Role.ASSISTANT, "done"))])
            self.entered = threading.Event()
            self.release = threading.Event()
            self.calls = 0
            self.calls_lock = threading.Lock()

        def create_response(self, request, *, cancel=None):
            with self.calls_lock:
                self.calls += 1
                calls = self.calls
            if calls >= 2:
                self.entered.set()
            self.release.wait(timeout=5)
            return super().create_response(request, cancel=cancel)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        def make_run(provider, sessions, repo_root=root):
            return HostRun(
                provider, PermissionPolicy(repo_root=repo_root), EventBroker(),
                sessions_root=sessions, memory_extraction_enabled=False,
            )

        provider = BlockingProvider()
        host = make_run(provider, root / "sessions")
        second_root = root / "second-project"
        second_root.mkdir()
        second_host = make_run(provider, root / "second-sessions", second_root)
        keeper_processes = []

        def start_keeper(*args, **kwargs):
            if args[0][0] != "/usr/bin/caffeinate":
                return real_popen(*args, **kwargs)
            process = KeeperProcess()
            keeper_processes.append(process)
            return process

        real_isfile = os.path.isfile
        real_popen = subprocess.Popen
        with mock.patch.object(host_run_module.sys, "platform", "darwin"), \
                mock.patch.object(host_run_module.os.path, "isfile", side_effect=lambda path: path == "/usr/bin/caffeinate" or real_isfile(path)), \
                mock.patch.object(host_run_module.subprocess, "Popen", side_effect=start_keeper):
            host.start("first")
            _wait_until(lambda: provider.calls >= 1, "first run did not reach the provider")
            second_host.start("second")
            if not provider.entered.wait(timeout=2):
                fail("second concurrent run did not reach the provider")
            if len(keeper_processes) != 1 or keeper_processes[0].terminated:
                fail(f"concurrent runs did not share one active keeper: {len(keeper_processes)}")
            provider.release.set()
            _wait_until(
                lambda: not host._active_by_session and not second_host._active_by_session,
                "concurrent runs did not finish",
            )
            if not keeper_processes[0].terminated:
                fail("keeper remained after the last conversation ended")
        host.close()
        second_host.close()

        linux_provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
        linux_host = make_run(linux_provider, root / "linux-sessions")
        linux_starts = []
        def record_linux_keeper(*args, **kwargs):
            if args[0][0] == "/usr/bin/caffeinate":
                linux_starts.append(args)
                return KeeperProcess()
            return real_popen(*args, **kwargs)
        with mock.patch.object(host_run_module.sys, "platform", "linux"), \
                mock.patch.object(host_run_module.os.path, "isfile", return_value=True), \
                mock.patch.object(host_run_module.subprocess, "Popen", side_effect=record_linux_keeper):
            linux_host.start("linux run")
            _wait_until(lambda: not linux_host._active_by_session, "Linux run did not finish")
        linux_host.close()
        if linux_starts:
            fail(f"Linux started a sleep keeper: {linux_starts!r}")

        shutdown_provider = BlockingProvider()
        shutdown_host = make_run(shutdown_provider, root / "shutdown-sessions")
        shutdown_keeper = []
        def start_shutdown_keeper(*args, **kwargs):
            if args[0][0] != "/usr/bin/caffeinate":
                return real_popen(*args, **kwargs)
            shutdown_keeper.append(KeeperProcess())
            return shutdown_keeper[-1]
        with mock.patch.object(host_run_module.sys, "platform", "darwin"), \
                mock.patch.object(host_run_module.os.path, "isfile", side_effect=lambda path: path == "/usr/bin/caffeinate" or real_isfile(path)), \
                mock.patch.object(host_run_module.subprocess, "Popen", side_effect=start_shutdown_keeper):
            shutdown_host.start("shutdown during run")
            _wait_until(lambda: shutdown_provider.calls >= 1, "shutdown run did not reach the provider")
            closer = threading.Thread(target=shutdown_host.close)
            closer.start()
            _wait_until(lambda: bool(shutdown_keeper) and shutdown_keeper[0].terminated, "host shutdown left the keeper running")
            shutdown_provider.release.set()
            closer.join(timeout=3)
            if closer.is_alive():
                fail("host shutdown did not finish after the active run was released")


@check("host_server.headless_key_errors")
def check_headless_key_errors() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        home = root.parent / "home"
        sessions = root.parent / "sessions"
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {
            "SYMPHONAI_HOME": str(home), "SYMPHONAI_SESSIONS_DIR": str(sessions),
        }):
            with mock.patch.object(headless_module, "_provider_for_start", return_value=(None, None)):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = headless_module.main([
                        "run", "no key", "--json", "--repo-root", str(root),
                    ])
        message = "no API key found; set one of " + ", ".join(
            dict.fromkeys(variable for _, variable, _ in headless_module.PROVIDERS)
        ) + " in the environment"
        if (code != 2 or json.loads(stdout.getvalue()) != {"status": "failed", "error": message}
                or stderr.getvalue().splitlines() != [f"symphonai: {message}"]):
            fail(f"headless missing-key JSON result was wrong: {code}, {stdout.getvalue()!r}, {stderr.getvalue()!r}")

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {
            "SYMPHONAI_HOME": str(home), "SYMPHONAI_SESSIONS_DIR": str(sessions),
        }):
            with mock.patch.object(headless_module, "_provider_for_start", return_value=(None, None)):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = headless_module.main([
                        "run", "no key", "--stream-json", "--repo-root", str(root),
                    ])
        stream_missing = [json.loads(line) for line in stdout.getvalue().splitlines()]
        if (code != 2 or stream_missing != [{"type": "result", "status": "failed", "error": message}]
                or stderr.getvalue().splitlines() != [f"symphonai: {message}"]):
            fail(f"headless missing-key stream JSON result was wrong: {code}, {stream_missing!r}, {stderr.getvalue()!r}")

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {
            "SYMPHONAI_HOME": str(home), "SYMPHONAI_SESSIONS_DIR": str(sessions),
        }):
            with mock.patch.object(
                headless_module, "_provider",
                side_effect=headless_module.ProviderSelectionError("openai has no API key"),
            ):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = headless_module.main([
                        "run", "no key", "--json", "--repo-root", str(root), "--provider", "openai",
                    ])
        explicit_message = "no API key found; set OPENAI_API_KEY in the environment"
        if (code != 2 or json.loads(stdout.getvalue()).get("error") != explicit_message
                or stderr.getvalue().splitlines() != [f"symphonai: {explicit_message}"]):
            fail(f"headless provider-specific missing-key message was wrong: {code}, {stdout.getvalue()!r}, {stderr.getvalue()!r}")

        class RejectedProvider(FakeModelProvider):
            @property
            def name(self):
                return "openai"

            @property
            def wire_format(self):
                return 1

            def create_response(self, request, *, cancel=None):
                raise ProviderUnavailableError("HTTP 401: raw vendor response body", status=401)

        code, output, error, _sessions = _headless_invoke(
            root, RejectedProvider(),
            ["run", "reject key", "--json", "--repo-root", str(root), "--provider", "openai"],
        )
        result = json.loads(output)
        rejected_message = "openai rejected the API key in OPENAI_API_KEY (HTTP 401)"
        if (code != 2 or result.get("status") != "failed" or result.get("error") != rejected_message
                or "raw vendor response body" in output + error
                or error.splitlines() != [f"symphonai: {rejected_message}"]):
            fail(f"headless rejected-key JSON result was wrong: {code}, {output!r}, {error!r}")

        stream_code, stream_output, stream_error, _sessions = _headless_invoke(
            root, RejectedProvider(),
            ["run", "reject key", "--stream-json", "--repo-root", str(root), "--provider", "openai"],
        )
        records = [json.loads(line) for line in stream_output.splitlines()]
        if (stream_code != 2 or records[-1].get("type") != "result"
                or records[-1].get("error") != rejected_message
                or any("raw vendor response body" in line for line in stream_output.splitlines())
                or stream_error.splitlines() != [f"symphonai: {rejected_message}"]):
            fail(f"headless rejected-key stream JSON result was wrong: {stream_code}, {records!r}, {stream_error!r}")


@check("host_server.symphonai_app_command")
def check_symphonai_app_command() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        served_hosts = []

        def serve_then_interrupt(host):  # noqa: ANN001
            served_hosts.append(host)
            server_thread = threading.Thread(target=host._httpd.serve_forever, daemon=True)
            server_thread.start()
            connection, response = _request(
                host, "GET", f"/app/?token={host.token}",
            )
            body = response.read().decode("utf-8")
            connection.close()
            if response.status != 200 or "SymphonAI" not in body:
                fail(f"symphonai app did not serve the app: {response.status}, {body[:200]!r}")
            raise KeyboardInterrupt

        for no_open, browser_result in ((True, False), (False, True), (False, False)):
            stdout, stderr = io.StringIO(), io.StringIO()
            browser = mock.Mock(return_value=browser_result)
            args = ["app", "--repo-root", str(root)] + (["--no-open"] if no_open else [])
            with mock.patch.dict(os.environ, {
                "SYMPHONAI_HOME": str(root / ".symphonai"),
                "SYMPHONAI_SESSIONS_DIR": str(root / "sessions"),
            }):
                with mock.patch.object(HostServer, "serve_forever", serve_then_interrupt):
                    with mock.patch.object(headless_module.webbrowser, "open", browser):
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            code = headless_module.main(args)
            if code != 0 or not served_hosts[-1]._closed:
                fail(f"symphonai app did not stop cleanly after Ctrl+C: {code}, closed={served_hosts[-1]._closed}")
            handshake = json.loads(stdout.getvalue().splitlines()[0])
            address = f"http://127.0.0.1:{handshake['port']}/app/?token={handshake['token']}"
            lines = stderr.getvalue().splitlines()
            if f"SymphonAI is running at {address}" not in lines or "Press Ctrl+C to stop." not in lines:
                fail(f"symphonai app status was missing: {lines!r}")
            if len(lines) != 2 or stderr.getvalue().count(address) != 1:
                fail(f"symphonai app printed an unexpected browser diagnostic: {lines!r}")
            if no_open and browser.called:
                fail("symphonai app opened a browser with --no-open")
            if not no_open and browser.call_args_list != [mock.call(address)]:
                fail(f"symphonai app did not open the app address once: {browser.call_args_list!r}")

        help_output, help_error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(help_output), contextlib.redirect_stderr(help_error):
            code = headless_module.main(["--help"])
        if code != 0 or "app" not in help_output.getvalue() or "run" not in help_output.getvalue():
            fail(f"symphonai help omitted a subcommand: {code}, {help_output.getvalue()!r}")


@check("host_server.headless_text_and_json")
def check_headless_text_and_json() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "hello from headless"))])
        code, output, _error, sessions = _headless_invoke(
            root, provider,
            ["run", "say hi", "--repo-root", str(root), "--provider", "openai"],
        )
        if code != 0 or output != "hello from headless\n":
            fail(f"headless text run returned {code}, {output!r}")

        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "json answer"))])
        provider.model = "fake-default"
        code, output, _error, sessions = _headless_invoke(
            root, provider,
            ["run", "--json", "say hi", "--repo-root", str(root), "--provider", "openai", "--model", "m-1"],
        )
        lines = output.splitlines()
        if code != 0 or len(lines) != 1:
            fail(f"headless JSON output was not one line: {code}, {output!r}")
        try:
            result = json.loads(lines[0])
        except ValueError as exc:
            fail(f"headless JSON was invalid: {exc}")
        expected_fields = {
            "session_id", "status", "stopped_reason", "result", "usage",
            "cost", "duration_ms", "files_changed", "error", "provider", "model",
        }
        if (set(result) != expected_fields or result["status"] != "finished"
                or result["result"] != "json answer" or result["error"] is not None
                or result["provider"] != "fake" or result["model"] != "m-1"):

            fail(f"headless JSON result fields or values were wrong: {result!r}")
        default_provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "default model answer"))])
        default_provider.model = "fake-default"
        default_code, default_output, _default_error, _default_sessions = _headless_invoke(
            root, default_provider,
            ["run", "--json", "say hi", "--repo-root", str(root), "--provider", "openai"],
        )
        default_result = json.loads(default_output)
        if (default_code != 0 or default_result.get("provider") != "fake"
                or default_result.get("model") != "fake-default"):
            fail(f"headless JSON omitted provider default model: {default_code}, {default_result!r}")

        class FallbackProvider(FakeModelProvider):
            @property
            def name(self):
                return "openai"

            @property
            def wire_format(self):
                return 1

            def create_response(self, request, *, cancel=None):
                if request.model == "m-1":
                    raise ProviderUnavailableError("primary unavailable", status=529)
                return super().create_response(request, cancel=cancel)

        fallback_provider = FallbackProvider([ModelResponse(Message(Role.ASSISTANT, "fallback answer"))])
        fallback_provider.model = "m-1"
        config_dir = root / ".symphonai"
        config_dir.mkdir(exist_ok=True)
        (config_dir / "config.toml").write_text('[fallback]\nopenai = ["m-2"]\n', encoding="utf-8")
        fallback_code, fallback_output, _fallback_error, _fallback_sessions = _headless_invoke(
            root, fallback_provider,
            ["run", "say hi", "--json", "--repo-root", str(root), "--provider", "openai"],
        )
        if fallback_code != 0:
            fail(f"headless fallback run failed: {fallback_code}, {fallback_output!r}, {_fallback_error!r}")
        fallback_result = json.loads(fallback_output)
        if (fallback_result.get("provider") != "openai" or fallback_result.get("model") != "m-2"):
            fail(f"headless JSON omitted fallback model: {fallback_result!r}")

        session = SessionStore.open(sessions, result["session_id"])
        try:
            if session.read_meta().get("headless") is not True:
                fail("headless session did not record headless=true")
        finally:
            session.close()

        for mode, filename, should_write in (("ask", "denied.txt", False), ("allow", "allowed.txt", True)):
            provider = FakeModelProvider([
                ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                    id=f"write-{mode}", name="write_file",
                    arguments={"path": filename, "content": "created\n"},
                )])),
                ModelResponse(Message(Role.ASSISTANT, f"{mode} finished")),
            ])
            code, output, error, _sessions = _headless_invoke(
                root, provider,
                ["run", "write a file", "--repo-root", str(root), "--provider", "openai", "--mode", mode],
            )
            exists = (root / filename).exists()
            if code != 0 or exists != should_write or f"{mode} finished" not in output:
                fail(f"headless {mode} mode result was wrong: {code}, {output!r}, {error!r}, exists={exists}")
            if should_write:
                json_provider = FakeModelProvider([
                    ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                        id="json-write", name="write_file",
                        arguments={"path": "json-created.txt", "content": "created\n"},
                    )])),
                    ModelResponse(Message(Role.ASSISTANT, "allow result")),
                ])
                json_code, json_output, _, _ = _headless_invoke(
                    root, json_provider,
                    ["run", "report file", "--json", "--repo-root", str(root), "--provider", "openai", "--mode", "allow"],
                )
                file_result = json.loads(json_output)
                if json_code != 0 or file_result["files_changed"] != ["json-created.txt"]:
                    fail(f"headless JSON omitted the changed file: {file_result!r}")

        code, output, _error, _sessions = _headless_invoke(
            root,
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "stdin answer"))]),
            ["run", "-", "--repo-root", str(root), "--provider", "openai"],
            stdin_text="task from stdin",
        )
        if code != 0 or output != "stdin answer\n":
            fail(f"headless stdin task returned {code}, {output!r}")


@check("host_server.headless_reports_failures_and_files")
def check_headless_reports_failures_and_files() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()

        class FailingProvider(FakeModelProvider):
            def create_response(self, request, *, cancel=None):
                raise ProviderUnavailableError(
                    "mock provider returned HTTP 529 after 3 attempts: busy", status=529
                )

        code, output, error, _sessions = _headless_invoke(
            root,
            FailingProvider(),
            ["run", "fail", "--json", "--repo-root", str(root), "--provider", "openai"],
        )
        result = json.loads(output)
        if code != 1 or result.get("error") != "mock provider returned HTTP 529 after 3 attempts: busy":
            fail(f"headless JSON omitted provider failure: {code}, {result!r}")
        if error.splitlines() != [f"symphonai: {result['error']}"] or "Exception in thread" in error:
            fail(f"headless failure stderr was not a single diagnostic line: {error!r}")

        stream_code, stream_output, stream_error, _sessions = _headless_invoke(
            root,
            FailingProvider(),
            ["run", "fail", "--stream-json", "--repo-root", str(root), "--provider", "openai"],
        )
        stream_records = [json.loads(line) for line in stream_output.splitlines()]
        if (stream_code != 1 or stream_records[-1].get("type") != "result"
                or stream_records[-1].get("error") != result["error"]
                or stream_error.splitlines() != [f"symphonai: {result['error']}"]):
            fail(f"headless stream JSON omitted provider failure: {stream_code}, {stream_records!r}")

        subprocess.run(["git", "init", "-q", str(root)], check=True)
        tracked = root / "preexisting.txt"
        tracked.write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "preexisting.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com",
             "commit", "-qm", "base"],
            check=True,
        )
        tracked.write_text("already changed\n", encoding="utf-8")
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "shell-write", "run_shell",
                {"argv": ["/bin/sh", "-c", "echo hi > out.txt"]},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        code, output, error, _sessions = _headless_invoke(
            root,
            provider,
            ["run", "write through shell", "--json", "--repo-root", str(root),
             "--provider", "openai", "--mode", "allow"],
        )
        result = json.loads(output)
        if code != 0 or result.get("files_changed") != ["out.txt"]:
            fail(f"headless git status diff was wrong: {code}, {result!r}, {error!r}")


@check("host_server.headless_exit_codes")
def check_headless_exit_codes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()

        class FailingProvider(FakeModelProvider):
            def create_response(self, request, *, cancel=None):
                raise ProviderError("provider failed")

        code, _output, _error, _sessions = _headless_invoke(
            root, FailingProvider(),
            ["run", "fail", "--repo-root", str(root), "--provider", "openai"],
        )
        if code != 1:
            fail(f"headless provider failure exited {code}, expected 1")

        limited = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
            id="limited-write", name="write_file", arguments={"path": "x", "content": "x"},
        )]))])
        code, output, _error, _sessions = _headless_invoke(
            root, limited,
            ["run", "stop at limit", "--repo-root", str(root), "--provider", "openai", "--max-turns", "1"],
        )
        if code != 3:
            fail(f"headless turn limit exited {code}, expected 3: {output!r}")

        missing = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "should not run"))])
        with tempfile.TemporaryDirectory() as home_dir:
            home = Path(home_dir) / "home"
            sessions = Path(home_dir) / "sessions"
            stdout, stderr = io.StringIO(), io.StringIO()
            with mock.patch.dict(os.environ, {
                "SYMPHONAI_HOME": str(home), "SYMPHONAI_SESSIONS_DIR": str(sessions),
            }):
                with mock.patch.object(
                    headless_module, "_provider",
                    side_effect=headless_module.ProviderSelectionError("openai has no API key"),
                ):
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        code = headless_module.main([
                            "run", "no key", "--repo-root", str(root), "--provider", "openai",
                        ])
            if code != 2 or missing.call_count != 0 or stdout.getvalue() or len(stderr.getvalue().splitlines()) != 1:
                fail(f"missing key did not fail before a request with one error line: {code}, {stderr.getvalue()!r}")

        class InterruptBroker(EventBroker):
            def subscribe(self):
                subscription = super().subscribe()

                class InterruptOnce:
                    interrupted = False

                    def get(self, timeout=None):
                        if not self.interrupted:
                            self.interrupted = True
                            raise KeyboardInterrupt
                        return subscription.get(timeout)

                    def close(self):
                        subscription.close()

                return InterruptOnce()

        class WaitingProvider(FakeModelProvider):
            def create_response(self, request, *, cancel=None):
                while cancel is not None:
                    cancel.raise_if_cancelled()
                    time.sleep(0.005)
                return ModelResponse(Message(Role.ASSISTANT, "unexpected"))

        with mock.patch.object(headless_module, "EventBroker", InterruptBroker):
            code, _output, _error, _sessions = _headless_invoke(
                root, WaitingProvider(),
                ["run", "interrupt", "--repo-root", str(root), "--provider", "openai"],
            )
        if code != 130:
            fail(f"headless interrupt exited {code}, expected 130")


@check("host_server.headless_stream_json")
def check_headless_stream_json() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "stream answer"))])
        provider.model = "fake-default"
        code, output, _error, _sessions = _headless_invoke(
            root, provider,
            ["run", "stream", "--repo-root", str(root), "--provider", "openai", "--model", "m-1", "--stream-json"],
        )
        try:
            records = [json.loads(line) for line in output.splitlines()]
        except ValueError as exc:
            fail(f"headless stream JSON had an invalid line: {exc}")
        if code != 0 or not records or records[0].get("type") not in {"PromptSubmitted", "RunStarted"}:
            fail(f"headless stream JSON did not begin with a run event: {code}, {records!r}")
        if (records[-1].get("type") != "result" or records[-1].get("result") != "stream answer"
                or records[-1].get("error") is not None or records[-1].get("provider") != "fake"
                or records[-1].get("model") != "m-1"):

            fail(f"headless stream JSON did not end with the result line: {records[-1]!r}")

        default_provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "default stream answer"))])
        default_provider.model = "fake-default"
        default_code, default_output, _default_error, _default_sessions = _headless_invoke(
            root, default_provider,
            ["run", "stream", "--repo-root", str(root), "--provider", "openai", "--stream-json"],
        )
        default_records = [json.loads(line) for line in default_output.splitlines()]
        if (default_code != 0 or default_records[-1].get("model") != "fake-default"
                or default_records[-1].get("provider") != "fake"):
            fail(f"headless stream JSON omitted provider default model: {default_code}, {default_records[-1]!r}")


@check("host_server.headless_resume")
def check_headless_resume() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "first answer"))])
        code, output, _error, sessions = _headless_invoke(
            root, provider,
            ["run", "first task", "--repo-root", str(root), "--provider", "openai", "--json"],
        )
        if code != 0:
            fail(f"initial headless run failed: {output!r}")
        session_id = json.loads(output)["session_id"]

        class RecordingProvider(FakeModelProvider):
            def __init__(self):
                super().__init__([ModelResponse(Message(Role.ASSISTANT, "second answer"))])
                self.requests = []

            def create_response(self, request, *, cancel=None):
                self.requests.append(request)
                return super().create_response(request, cancel=cancel)

        resumed = RecordingProvider()
        code, output, _error, _sessions = _headless_invoke(
            root, resumed,
            ["run", "second task", "--repo-root", str(root), "--provider", "openai", "--resume", session_id, "--json"],
        )
        if code != 0:
            fail(f"headless resume failed: {output!r}")
        request_texts = [
            "\n".join(message.text for message in request.messages)
            for request in resumed.requests
        ]
        if not any(
            "first task" in text and "first answer" in text and "second task" in text
            for text in request_texts
        ):
            fail(f"headless resume omitted stored history or the new task: {request_texts!r}")
        result = json.loads(output)
        if result["session_id"] != session_id:
            fail(f"headless resume changed the session id: {result!r}")


@check("host_server.environment_names_the_venv")
def check_environment_names_the_venv() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        for name in ("venv", ".venv"):
            venv = root / name
            (venv / "bin").mkdir(parents=True)
            (venv / "pyvenv.cfg").touch()
            (venv / "bin/python").touch()
            provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
            host = HostServer(provider, PermissionPolicy(root), sessions_root=root / f"sessions-{name}", memory_extraction_enabled=False)
            host.start()
            try:
                with mock.patch.object(provider, "create_response", wraps=provider.create_response) as spy:
                    _send_host_prompt(host, "inspect environment")
                systems = [message.text for message in spy.call_args.args[0].messages if message.role is Role.SYSTEM]
                expected = f"- Repository root: {root}\n- Python virtual environment: {name} (on PATH for shell commands)\n- Platform:"
                if not any(expected in text for text in systems):
                    fail(f"project virtualenv missing from environment: {systems!r}")
            finally:
                host.close()


@check("host_server.empty_chat_mode_is_one_chat")
def check_empty_chat_mode_is_one_chat() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "done")) for _ in range(3)
        ])
        host = HostServer(provider, PermissionPolicy(root), sessions_root=root / "sessions",
                          memory_extraction_enabled=False)
        host.start()
        entered = threading.Event()
        release = threading.Event()

        def request(method, path, body=None):
            connection, response = _request(host, method, path, body=body, headers=_headers(host))
            try:
                reply = json.loads(response.read())
                if response.status != 200:
                    fail(f"mode fixture request failed: {method} {path}: {response.status}, {reply!r}")
                return reply
            finally:
                connection.close()

        def configured_mode():
            if request("GET", "/settings").get("settings", {}).get("mode") != "ask":
                fail("temporary conversation mode leaked into settings")

        def conversation_mode(expected):
            _, reply = _conversation_reply(host)
            if reply.get("conversation", {}).get("mode") != expected:
                fail(f"expected conversation mode {expected!r}: {reply!r}")
            configured_mode()

        try:
            configured_mode()
            request("POST", "/mode", {"mode": "plan"})
            configured_mode()
            _send_host_prompt(host, "plan this chat")
            conversation_mode("plan")
            first_session = host.run._conversation[1].run_id
            request("POST", "/session/new", {})
            configured_mode()

            respond = provider.create_response

            def blocked_response(model_request, *, cancel=None):
                entered.set()
                if not release.wait(5):
                    fail("mode fixture did not release the active turn")
                return respond(model_request, cancel=cancel)

            with mock.patch.object(provider, "create_response", side_effect=blocked_response):
                request("POST", "/prompt", {"prompt": "ordinary new chat"})
                if not entered.wait(2):
                    fail("new chat did not reach its model request")
                conversation_mode("ask")
                request("POST", "/mode", {"mode": "plan"})
                conversation_mode("plan")
                release.set()
                _wait_until(lambda: not host.run.active, "mode fixture turn did not finish")

            request("POST", "/session/new", {})
            _send_host_prompt(host, "another ordinary chat")
            conversation_mode("ask")
            request("POST", "/session/new", {})
            request("POST", "/mode", {"mode": "plan"})
            request("POST", "/session/new", {})
            if host.run.policy.mode != "ask":
                fail("new chat retained an unused temporary mode")
            request("POST", "/mode", {"mode": "plan"})
            request("POST", "/session/open", {"run_id": first_session})
            request("POST", "/session/new", {})
            if host.run.policy.mode != "ask":
                fail("switching conversations retained a temporary mode")
            configured_mode()
        finally:
            release.set()
            host.close()


def _headless_loop_provider(
    validation_command: str = "true", loop_id: str = "loop_000000000000",
) -> FakeModelProvider:
    spec_path = f".symphonai/loops/{loop_id}/specs/a-change.md"
    report_path = f".symphonai/loops/{loop_id}/reports/a-change-a-report.md"
    spec = (
        "# Change a\n\n## Validation\n\n```sh\n"
        f"{validation_command}\n```\n\n## Report\n\n`{report_path}`\n"
    )
    plan = {"cards": [{
        "id": "a", "title": "Change a", "spec": "specs/a-change.md",
        "lane": 1, "after": [],
    }]}
    responses = [
        ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
            "plan-spec", "write_file", {"path": spec_path, "content": spec},
        )])),
        ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
            "plan", "write_file", {
                "path": f".symphonai/loops/{loop_id}/plan.json",
                "content": json.dumps(plan),
            },
        )])),
        ModelResponse(Message(Role.ASSISTANT, "Plan complete.")),
        ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
            "new-file", "write_file", {"path": "new.py", "content": "value = 2\n"},
        )])),
        ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
            "report", "write_file", {"path": report_path, "content": "Implementation passed.\n"},
        )])),
        ModelResponse(Message(Role.ASSISTANT, "Implemented the change.")),
        ModelResponse(Message(Role.ASSISTANT, "Verdict: pass")),
    ]
    return FakeModelProvider([
        ModelResponse(response.message, Usage(10, 2, 3, 4))
        for response in responses
    ])


@check("host_server.headless_loop_checks_role_providers_before_planning")
def check_headless_loop_checks_role_providers_before_planning() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        home = root.parent / "home"
        home.mkdir()
        (home / "config.toml").write_text(
            '[roles.implementer]\nprovider = "anthropic"\nmodel = "claude-sonnet-4-5"\n',
            encoding="utf-8",
        )
        openai = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "unexpected"))])

        def provider_factory(name, _model=None, _base_url=None):
            return openai if name == "openai" else None

        code, _output, error, _sessions = _headless_invoke(
            root,
            openai,
            ["loop", "Run the configured role", "--json", "--repo-root", str(root), "--provider", "openai"],
            provider_factory=provider_factory,
        )
        if code != 2 or "implementer" not in error or "anthropic" not in error:
            fail(f"unavailable role provider did not stop the headless loop: {code}, {error!r}")
        if openai.call_count:
            fail(f"headless loop called the conversation provider before role preflight: {openai.call_count}")
        if (root / ".symphonai" / "loops").exists():
            fail("headless loop created a loop before role preflight completed")


@check("host_server.headless_loop")
def check_headless_loop() -> None:
    if headless_module._loop_retry_target("card aF could not finish: provider error") != ("implementer", "aF"):
        fail("failed follow-up card was not classified for an implementer retry")
    if headless_module._loop_retry_target("card aF review gave no verdict") != ("reviewer", "aF"):
        fail("follow-up review without a verdict was not classified for a reviewer retry")
    if headless_module._loop_retry_target(
        "card aF could not finish: validation still failing after 3 rounds"
    ) != (None, None):
        fail("follow-up card with exhausted validation was incorrectly classified for retry")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        for arguments in (
            ("init", "--quiet"),
            ("config", "user.email", "headless-loop@example.test"),
            ("config", "user.name", "Headless Loop Check"),
        ):
            subprocess.run(["git", *arguments], cwd=root, check=True)
        (root / "base.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "base.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "base"], cwd=root, check=True)

        provider = _headless_loop_provider()
        with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000000"):
            code, output, error, sessions = _headless_invoke(
                root,
                provider,
                ["loop", "Add new.py", "--json", "--repo-root", str(root), "--provider", "openai"],
            )
        try:
            result = json.loads(output)
        except ValueError as exc:
            fail(f"headless loop did not emit JSON: {code}, {output!r}, {error!r}: {exc}")
        loop = LoopStore(root).load(result.get("loop_id", ""))
        if (code != 0 or result.get("state") != "awaiting_commit"
                or loop.get("state") != "awaiting_commit" or not loop.get("branch")
                or result.get("branch") != loop.get("branch")
                or not result.get("cards") or not result["cards"][0].get("commit")
                or loop["cards"][0].get("state") != "done"
                or loop["cards"][0].get("commit") != result["cards"][0].get("commit")
                or not result.get("summary", {}).get("commits")
                or result["summary"]["commits"][0].get("card") != "a"
                or not result.get("summary", {}).get("validation")
                or result["summary"]["validation"][0].get("command") != "true"
                or result["summary"]["validation"][0].get("exit") != 0
                or not (sessions / "loops" / loop["id"] / "integration").is_dir()):
            fail(f"headless loop did not stop at a committed checkpoint: {code}, {result!r}, {loop!r}, {error!r}")
        expected_usage = {
            "cards": {
                "a": {
                    "implement": {
                        "input_tokens": 30, "output_tokens": 6, "calls": 3,
                        "cache_read_tokens": 9, "cache_write_tokens": 12,
                    },
                    "review": {
                        "input_tokens": 10, "output_tokens": 2, "calls": 1,
                        "cache_read_tokens": 3, "cache_write_tokens": 4,
                    },
                },
            },
            "planning": {
                "input_tokens": 30, "output_tokens": 6, "calls": 3,
                "cache_read_tokens": 9, "cache_write_tokens": 12,
            },
            "total": {
                "input_tokens": 70, "output_tokens": 14, "calls": 7,
                "cache_read_tokens": 21, "cache_write_tokens": 28,
            },
        }
        if result.get("usage") != expected_usage:
            fail(f"headless --json loop omitted real usage: {result.get('usage')!r}")
        branch_ref = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{loop['branch']}"],
            cwd=root, check=False,
        )
        if branch_ref.returncode != 0:
            fail(f"headless loop branch was missing at checkpoint: {loop['branch']!r}")
        integration = sessions / "loops" / loop["id"] / "integration"
        if not (integration / "new.py").is_file() or "card a" not in error:
            fail(f"headless loop did not run its card or report card changes: {error!r}")

        committer = HostRun(
            FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]),
            EventBroker(), sessions_root=sessions,
        )
        try:
            committer.commit_loop(loop["id"], "approve")
        finally:
            committer.close()
        if LoopStore(root).load(loop["id"])["state"] != "done":
            fail("later checkpoint approval did not finish the headless loop")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        for arguments in (
            ("init", "--quiet"),
            ("config", "user.email", "headless-retry@example.test"),
            ("config", "user.name", "Headless Retry Check"),
        ):
            subprocess.run(["git", *arguments], cwd=root, check=True)
        (root / "base.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "base.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "base"], cwd=root, check=True)

        class FailImplementerOnce(FakeModelProvider):
            def create_response(self, request, *, cancel=None):
                is_implementer = any(
                    "Write your report at" in message.text for message in request.messages
                )
                if is_implementer and not getattr(self, "failed_once", False):
                    self.failed_once = True
                    raise ProviderError("scripted provider failure")
                return super().create_response(request, cancel=cancel)

        provider = FailImplementerOnce(
            _headless_loop_provider(loop_id="loop_000000000003")._responses
        )
        with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000003"):
            code, output, _error, _sessions = _headless_invoke(
                root, provider,
                ["loop", "Add new.py", "--json", "--repo-root", str(root), "--provider", "openai"],
            )
        result = json.loads(output)
        retries = result.get("retries")
        if (code != 0 or result.get("state") != "awaiting_commit"
                or not isinstance(retries, list) or len(retries) != 1
                or retries[0].get("stage") != "implementer"
                or retries[0].get("card") != "a"
                or "scripted provider failure" not in retries[0].get("reason", "")):
            fail(f"headless loop did not retry the failed implementer stage: {code}, {result!r}")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        for arguments in (
            ("init", "--quiet"),
            ("config", "user.email", "headless-text-usage@example.test"),
            ("config", "user.name", "Headless Text Usage Check"),
        ):
            subprocess.run(["git", *arguments], cwd=root, check=True)
        (root / "base.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "base.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "base"], cwd=root, check=True)
        with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000000"):
            code, output, _error, _sessions = _headless_invoke(
                root,
                _headless_loop_provider(),
                ["loop", "Add new.py", "--repo-root", str(root), "--provider", "openai"],
            )
        if code != 0 or "Tokens: 70 in · 14 out · 7 calls" not in output:
            fail(f"headless loop text omitted actual usage: {code}, {output!r}")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        for arguments in (
            ("init", "--quiet"),
            ("config", "user.email", "headless-plan-usage@example.test"),
            ("config", "user.name", "Headless Plan Usage Check"),
        ):
            subprocess.run(["git", *arguments], cwd=root, check=True)
        (root / "base.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "base.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "base"], cwd=root, check=True)

        with mock.patch.object(
            HostRun, "approve_loop_plan", side_effect=RunActiveError("approval unavailable"),
        ), mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000002"):
            code, output, _error, _sessions = _headless_invoke(
                root,
                _headless_loop_provider(loop_id="loop_000000000002"),
                ["loop", "Plan only", "--json", "--repo-root", str(root), "--provider", "openai"],
            )
        result = json.loads(output)
        if (code != 1 or result.get("state") != "awaiting_plan"
                or result.get("usage", {}).get("planning") != {
                    "input_tokens": 30, "output_tokens": 6, "calls": 3,
                    "cache_read_tokens": 9, "cache_write_tokens": 12,
                }
                or result.get("usage", {}).get("total") != {
                    "input_tokens": 30, "output_tokens": 6, "calls": 3,
                    "cache_read_tokens": 9, "cache_write_tokens": 12,
                }):
            fail(f"headless planning stop omitted planner usage: {code}, {result!r}")

    class TerminalInput(io.StringIO):
        def isatty(self):
            return True

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(headless_module.sys, "stdin", TerminalInput()):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = headless_module.main([
                    "loop", "--repo-root", str(root), "--provider", "openai",
                ])
        if code != 2 or "goal is required when stdin is a terminal" not in stderr.getvalue():
            fail(f"headless loop without a terminal goal did not return usage: {code}, {stderr.getvalue()!r}")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        for arguments in (
            ("init", "--quiet"),
            ("config", "user.email", "headless-loop@example.test"),
            ("config", "user.name", "Headless Loop Check"),
        ):
            subprocess.run(["git", *arguments], cwd=root, check=True)
        (root / "base.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "base.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "base"], cwd=root, check=True)
        with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000000"):
            code, output, error, _sessions = _headless_invoke(
                root, _headless_loop_provider("false"),
                ["loop", "Add blocked.py", "--json", "--repo-root", str(root), "--provider", "openai"],
            )
        result = json.loads(output)
        if (code != 1 or result.get("state") != "needs_attention"
                or not result.get("reason") or result["reason"] not in output
                or "validation still failing after" not in result.get("reason", "")
                or "loop " not in error or result.get("retries") != []):
            fail(f"blocked loop did not report needs_attention: {code}, {result!r}, {error!r}")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        for arguments in (
            ("init", "--quiet"),
            ("config", "user.email", "headless-loop@example.test"),
            ("config", "user.name", "Headless Loop Check"),
        ):
            subprocess.run(["git", *arguments], cwd=root, check=True)
        (root / "base.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "base.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "base"], cwd=root, check=True)

        class BlockingProvider(FakeModelProvider):
            def __init__(self) -> None:
                super().__init__(_headless_loop_provider()._responses)
                self.card_started = threading.Event()
                self.card_cancelled = threading.Event()
                self.release_card = threading.Event()

            def create_response(self, request, *, cancel=None):
                if self.call_count == 3:
                    self.card_started.set()
                    deadline = time.monotonic() + 10
                    while not self.release_card.wait(0.01) and time.monotonic() < deadline:
                        if cancel is not None and cancel.cancelled:
                            self.card_cancelled.set()
                            break
                    if cancel is not None and cancel.cancelled:
                        self.card_cancelled.set()
                return super().create_response(request, cancel=cancel)

        provider = BlockingProvider()
        command_result = []
        command_errors = []

        def invoke_loop() -> None:
            try:
                command_result.append(_headless_invoke(
                    root, provider,
                    ["loop", "Add new.py", "--json", "--repo-root", str(root), "--provider", "openai"],
                ))
            except BaseException as exc:
                command_errors.append(exc)

        store = LoopStore(root)
        with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000000"):
            command_thread = threading.Thread(target=invoke_loop, daemon=True)
            command_thread.start()
            deadline = time.monotonic() + 5
            loop = None
            while time.monotonic() < deadline:
                try:
                    loop = store.list()[0]
                except IndexError:
                    time.sleep(0.01)
                    continue
                if loop.get("state") == "running" and provider.card_started.is_set():
                    break
                time.sleep(0.01)
            if loop is None or loop.get("state") != "running" or not provider.card_started.is_set():
                provider.release_card.set()
                command_thread.join(timeout=3)
                fail(f"headless loop did not block in its card: {loop!r}")

            canceller = HostRun(
                FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]),
                EventBroker(), sessions_root=root.parent / "sessions",
            )
            try:
                canceller.cancel_loop(loop["id"])
            finally:
                canceller.close()
            command_thread.join(timeout=3)
            if command_thread.is_alive():
                provider.release_card.set()
                cancelled = store.load(loop["id"])
                cancelled.update(state="needs_attention", reason="test cleanup")
                store.save(cancelled)
                command_thread.join(timeout=3)
                fail("headless command did not exit after another HostRun cancelled its loop")
            if command_errors:
                fail(f"headless cancellation command raised: {command_errors[0]!r}")
            code, output, _error, _sessions = command_result[0]
            result = json.loads(output)
            if code != 1 or result.get("state") != "cancelled":
                fail(f"headless cancellation returned the wrong result: {code}, {result!r}")
            if not provider.card_cancelled.is_set():
                fail("headless cancellation did not stop the active card conversation")


@check("host_server.headless_loop_resume")
def check_headless_loop_resume() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        for arguments in (
            ("init", "--quiet"),
            ("config", "user.email", "headless-resume@example.test"),
            ("config", "user.name", "Headless Resume Check"),
        ):
            subprocess.run(["git", *arguments], cwd=root, check=True)
        (root / "base.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "base.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "base"], cwd=root, check=True)

        class FailDuringCardProvider(FakeModelProvider):
            def __init__(self) -> None:
                super().__init__(_headless_loop_provider()._responses[:3])

            def create_response(self, request, *, cancel=None):
                if self.call_count >= 3:
                    raise ProviderError("scripted card failure")
                return super().create_response(request, cancel=cancel)

        with mock.patch("symphonai_host.loops.secrets.token_hex", return_value="000000000000"):
            code, output, _error, sessions = _headless_invoke(
                root, FailDuringCardProvider(),
                ["loop", "Add new.py", "--json", "--repo-root", str(root), "--provider", "openai"],
            )
        failed_result = json.loads(output)
        if code != 1 or failed_result.get("state") != "needs_attention":
            fail(f"fixture loop did not stop at its failed card: {code}, {failed_result!r}")
        loop_id = failed_result["loop_id"]
        store = LoopStore(root)

        report_path = f".symphonai/loops/{loop_id}/reports/a-change-a-report.md"
        class BlockingResumeProvider(FakeModelProvider):
            def __init__(self) -> None:
                super().__init__([
                    ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
                        "new-file", "write_file", {"path": "new.py", "content": "value = 2\n"},
                    )])),
                    ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(
                        "report", "write_file", {"path": report_path, "content": "Implementation passed.\n"},
                    )])),
                    ModelResponse(Message(Role.ASSISTANT, "Implemented the change.")),
                    ModelResponse(Message(Role.ASSISTANT, "Verdict: pass")),
                ])
                self.requests = []
                self.card_started = threading.Event()
                self.release_card = threading.Event()

            def create_response(self, request, *, cancel=None):
                self.requests.append(request)
                if self.call_count == 0:
                    self.card_started.set()
                    self.release_card.wait(5)
                return super().create_response(request, cancel=cancel)

        provider = BlockingResumeProvider()
        command_result = []
        command_errors = []

        def resume() -> None:
            try:
                command_result.append(_headless_invoke(
                    root, provider,
                    ["loop", "--resume", loop_id, "--json", "--repo-root", str(root), "--provider", "openai"],
                ))
            except BaseException as exc:
                command_errors.append(exc)

        thread = threading.Thread(target=resume, daemon=True)
        thread.start()
        try:
            if not provider.card_started.wait(5):
                fail("resumed loop did not start its failed card")
            running = store.load(loop_id)
            if running.get("runner_pid") != os.getpid() or running.get("state") != "running":
                fail(f"resumed loop did not record its active runner: {running!r}")
        finally:
            provider.release_card.set()
            thread.join(timeout=10)
        if thread.is_alive():
            fail("resumed headless loop did not exit")
        if command_errors:
            fail(f"resumed headless command raised: {command_errors[0]!r}")
        code, output, _error, _sessions = command_result[0]
        result = json.loads(output)
        if code != 0 or result.get("state") != "awaiting_commit":
            fail(f"resumed loop did not reach its commit checkpoint: {code}, {result!r}")
        if any(
            message.role is Role.USER and "Plan this goal:" in message.text
            for request in provider.requests for message in request.messages
        ):
            fail("resumed provider received a planning request")

        committer = HostRun(
            FakeModelProvider(), PermissionPolicy(root, allowed_write_scope=[root]),
            EventBroker(), sessions_root=sessions,
        )
        try:
            committer.commit_loop(loop_id, "approve")
        finally:
            committer.close()
        loop_path = store.path(loop_id, "loop.json")
        done_bytes = loop_path.read_bytes()
        code, _output, error, _sessions = _headless_invoke(
            root, FakeModelProvider(),
            ["loop", "--resume", loop_id, "--json", "--repo-root", str(root), "--provider", "openai"],
        )
        if code != 1 or loop_id not in error or "done" not in error:
            fail(f"resuming a done loop did not name its state: {code}, {error!r}")
        if loop_path.read_bytes() != done_bytes:
            fail("rejecting resume of a done loop changed loop.json")

        unknown = "loop_unknown0000"
        code, _output, error, _sessions = _headless_invoke(
            root, FakeModelProvider(),
            ["loop", "--resume", unknown, "--json", "--repo-root", str(root), "--provider", "openai"],
        )
        if code != 1 or unknown not in error or "unknown" not in error:
            fail(f"unknown loop resume did not name the id and state: {code}, {error!r}")

        code, _output, error, _sessions = _headless_invoke(
            root, FakeModelProvider(),
            ["loop", "also a goal", "--resume", loop_id, "--repo-root", str(root), "--provider", "openai"],
        )
        if code != 2 or "cannot be combined" not in error:
            fail(f"resume accepted a positional goal: {code}, {error!r}")


@check("host_server.headless_loop_is_not_recovered_while_running")
def check_headless_loop_is_not_recovered_while_running() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "project"
        sessions = Path(temporary) / "sessions"
        root.mkdir()
        for arguments in (
            ("init", "--quiet"),
            ("config", "user.email", "loop-recovery@example.test"),
            ("config", "user.name", "Loop Recovery Check"),
        ):
            subprocess.run(["git", *arguments], cwd=root, check=True)
        store = LoopStore(root)
        loop = store.create("Recover a running card")
        loop.update(
            state="running",
            runner_pid=os.getpid(),
            runner_lock=True,
            cards=[{"id": "a", "title": "A", "state": "running", "session_id": "run_card"}],
            lanes=[{"id": 1, "cards": ["a"]}],
        )
        store.save(loop)
        lock_owner = HostRun(
            FakeModelProvider(), PermissionPolicy(root), EventBroker(),
            sessions_root=sessions,
        )
        lock_fd = lock_owner._acquire_loop_runner_lock(loop["id"])
        if lock_fd is None:
            fail("could not acquire the headless runner lock")
        lock_owner._loop_runner_lock_fds[loop["id"]] = lock_fd

        with mock.patch.object(HostRun, "_advance_loop", lambda _self, _loop_id: None):
            live_host = HostServer(
                FakeModelProvider(), PermissionPolicy(root), sessions_root=sessions,
            )
            try:
                live_loop = store.load(loop["id"])
                if live_loop["cards"][0]["state"] != "running":
                    fail(f"host recovered a card while its headless runner was alive: {live_loop!r}")
            finally:
                live_host.close()
            lock_owner.close()

            stale = store.load(loop["id"])
            stale["runner_pid"] = os.getpid() + 10_000_000
            store.save(stale)
            recovered_host = HostServer(
                FakeModelProvider(), PermissionPolicy(root), sessions_root=sessions,
            )
            try:
                recovered = store.load(loop["id"])
                if recovered["cards"][0]["state"] != "planned":
                    fail(f"host did not recover the card after its runner exited: {recovered!r}")
            finally:
                recovered_host.close()
