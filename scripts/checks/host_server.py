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
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import get_args, get_type_hints
from urllib.parse import urlencode, urljoin, urlsplit
from unittest import mock

import symphonai_api.agent_loop as agent_loop
import symphonai_api.leader as leader_module
import symphonai_api.mcp as mcp_module
import symphonai_host.__main__ as host_main
import symphonai_host.protocol as protocol_module
import symphonai_host.run as host_run_module
import symphonai_host.server as host_server_module
import symphonai_host.goal as goal_module
from symphonai_api.agent_run import RunNode
from symphonai_api.agent_file import load_agent_file
from symphonai_api.agent_spec import ModelSelector
from symphonai_api.config import ConfigError, ResolvedConfig
from symphonai_api.events import (
    AssistantTextDelta,
    CompactionApplied,
    RunFinished,
    RunStarted,
    SubagentSpawned,
)
from symphonai_api.cost import ModelPrice, PriceTable
from symphonai_api.extensions import Extensions, load_extensions
from symphonai_api.hooks import HookRunner
from symphonai_api.identity import RunRef
from symphonai_api.instructions import MAX_INSTRUCTION_FILE_CHARS
from symphonai_api.mcp import McpServerSpec
from symphonai_api.mcp_pool import McpPool
from symphonai_api.models import DocumentBlock, ImageBlock, Message, ModelResponse, Role, TextBlock, ToolCall, ToolResult, Usage
from symphonai_api.permissions import PermissionPolicy
from symphonai_host.files import repository_files
from symphonai_api.providers.base import ModelProvider, ProviderError
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.runner import merge_tool_registry, standard_tool_registry
from symphonai_api.session import SessionStore, load_run, load_run_for_resume
from symphonai_api.streaming import StreamCompleted, TextDelta
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.filesystem import ReadLedger, WriteFileTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_api.tools.shell import RunShellTool
from symphonai_api.tools.web_fetch import WebFetchTool
from symphonai_api.worktree import create_worktree, worktree_diff
from symphonai_host.broker import EventBroker
from symphonai_host.goal import GoalChanged
from symphonai_host.protocol import decode_event, decode_frame
from symphonai_host.run import HostRun, RunActiveError
from symphonai_host.server import HostServer
from symphonai_host.spec_run import bind_roadmap_item, mark_roadmap_spec_done, parse_spec, patch_digest, review_verdict
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


@check("host_server.spec_parser_title_and_validation")
def check_spec_parser_title_and_validation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = root / "specs" / "39" / "39z-sample.md"
        spec.parent.mkdir(parents=True)
        spec.write_text("# 39z — sample\n\n## Validation\n\n```bash\none\n\n two\n```\n")
        parsed = parse_spec(spec, root)
        if parsed["title"] != "39z — sample" or parsed["validation"] != ["one", " two"]:
            fail(f"spec title or validation block parsed incorrectly: {parsed!r}")


@check("host_server.spec_parser_report_path")
def check_spec_parser_report_path() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = root / "specs" / "39" / "39z-sample.md"
        spec.parent.mkdir(parents=True)
        spec.write_text("# Sample\n\n## Report\n\n`specs/report/39/sample.md`\n")
        if parse_spec(spec, root)["report"] != "specs/report/39/sample.md":
            fail("spec report path did not use the first code span")


@check("host_server.spec_parser_derived_report")
def check_spec_parser_derived_report() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = root / "specs" / "39" / "39z-sample.md"
        spec.parent.mkdir(parents=True)
        spec.write_text("# Sample\n")
        if parse_spec(spec, root)["report"] != "specs/report/39/39z-sample-report.md":
            fail("missing report section did not derive a report path")


@check("host_server.spec_parser_requires_specs_markdown")
def check_spec_parser_requires_specs_markdown() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bad = root / "other.md"
        bad.write_text("# No")
        for path in (bad, root / "specs" / "missing.md"):
            try:
                parse_spec(path, root)
            except ValueError:
                continue
            fail(f"invalid spec path was accepted: {path}")


@check("host_server.spec_parser_checkless_validation")
def check_spec_parser_checkless_validation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = root / "specs" / "39" / "39z-sample.md"
        spec.parent.mkdir(parents=True)
        spec.write_text("# Sample\n\n## Validation\n\nNo commands.\n")
        if parse_spec(spec, root)["validation"]:
            fail("prose without a fenced validation block became commands")


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


@check("host_server.spec_run_review_commit_end_to_end")
def check_spec_run_review_commit_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
        (root / ".gitignore").write_text("specs/\n", encoding="utf-8")
        (root / "a.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", ".gitignore", "a.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
        spec_path = root / "specs" / "39" / "39b-run-a-spec.md"
        spec_path.parent.mkdir(parents=True)
        spec_path.write_text(
            "# 39b — run a spec\n\n## Validation\n\n```bash\npwd\n```\n\n"
            "## Report\n\n`specs/report/39/39b-run-a-spec-report.md`\n",
            encoding="utf-8",
        )
        docs = root / "docs"
        docs.mkdir()
        roadmap = docs / "roadmap.json"
        roadmap.write_text(json.dumps({"phases": [{
            "id": "39", "status": "in_progress",
            "items": [{"title": "run a spec", "spec": ["specs/39/39b-run-a-spec.md"]}],
        }]}), encoding="utf-8")
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("outside-worktree", "write_file", {
                "path": "../outside-worktree.txt", "content": "must be refused\n",
            })])),
            ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("read", "read_file", {"path": "a.py"})])),
            ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("edit", "write_file", {"path": "a.py", "content": "value = 2\n"})])),
            ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("report", "write_file", {"path": "specs/report/39/39b-run-a-spec-report.md", "content": "report"})])),
            ModelResponse(Message(Role.ASSISTANT, "implemented")),
            ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("review-outside-scope", "write_file", {
                "path": "a.py", "content": "reviewer must not write this\n",
            })])),
            ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("follow-up", "write_file", {
                "path": "specs/39/39bF-fix.md", "content": "# Follow-up\n",
            })])),
            ModelResponse(Message(Role.ASSISTANT, "Verdict: follow-ups: specs/39/39bF-fix.md")),
        ])
        host = HostServer(
            provider,
            PermissionPolicy(repo_root=root, allowed_write_scope=[root]),
            sessions_root=root / "sessions",
        )
        host.start()
        try:
            for invalid_path in ("specs/../a.py", "specs/39/missing.md"):
                connection, response = _request(host, "POST", "/spec/run", body={"path": invalid_path}, headers=_headers(host))
                body = response.read()
                connection.close()
                if response.status != 400:
                    fail(f"invalid spec path {invalid_path!r} returned {response.status}: {body!r}")
            connection, response = _request(host, "POST", "/spec/run", body={"path": "specs/39/39b-run-a-spec.md"}, headers=_headers(host))
            started = json.loads(response.read())
            connection.close()
            if response.status != 200:
                fail(f"spec/run returned {response.status}: {started!r}")
            connection, response = _request(host, "POST", "/spec/run", body={"path": "specs/39/39b-run-a-spec.md"}, headers=_headers(host))
            duplicate = json.loads(response.read())
            connection.close()
            if response.status != 409:
                fail(f"duplicate running spec was not refused: {response.status}, {duplicate!r}")

            def spec_run_state():
                connection, response = _request(host, "GET", "/spec/runs", headers=_headers(host))
                values = json.loads(response.read())
                connection.close()
                return next(item for item in values if item["session_id"] == started["session_id"])

            deadline = time.monotonic() + 8
            state = spec_run_state()
            while state["state"] == "running" and time.monotonic() < deadline:
                time.sleep(0.02)
                state = spec_run_state()
            if state["state"] != "finished" or not state["report_copied"] or state["files"] != ["a.py"]:
                worktree_file = root / "sessions" / started["session_id"] / "worktree" / "a.py"
                actual = worktree_file.read_text(encoding="utf-8") if worktree_file.exists() else "<missing>"
                fail(f"spec run did not finish with an isolated edit and copied report: {state!r}; worktree a.py={actual!r}; provider calls={provider.call_count}")
            if (root / "a.py").read_text(encoding="utf-8") != "value = 1\n":
                fail("spec implementation changed the main tree before review and commit")
            if (root / "sessions" / started["session_id"] / "outside-worktree.txt").exists() or host.pending_approvals():
                fail("spec implementation wrote outside its worktree or asked for approval in allow mode")
            if state.get("commit_message") != "39b: run a spec":
                fail(f"spec run default commit message was wrong: {state.get('commit_message')!r}")

            connection, response = _request(host, "POST", "/spec/review", body={"session_id": started["session_id"]}, headers=_headers(host))
            reviewed = json.loads(response.read())
            connection.close()
            if response.status != 200:
                fail(f"spec/review returned {response.status}: {reviewed!r}")
            deadline = time.monotonic() + 5
            state = spec_run_state()
            while (not state.get("review") or state["review"].get("verdict") == "running") and time.monotonic() < deadline:
                time.sleep(0.02)
                state = spec_run_state()
            if state.get("review", {}).get("verdict") != "follow-ups":
                fail(f"review verdict was not recorded: {state!r}")
            if state["review"].get("follow_ups") != ["specs/39/39bF-fix.md"] or not (root / "specs/39/39bF-fix.md").is_file():
                fail(f"eligible reviewer follow-up was not copied to the main tree: {state['review']!r}")

            (root / "staged.txt").write_text("staged\n", encoding="utf-8")
            subprocess.run(["git", "add", "staged.txt"], cwd=root, check=True)
            connection, response = _request(host, "POST", "/spec/commit", body={
                "session_id": started["session_id"], "message": "blocked while staged",
            }, headers=_headers(host))
            refusal = json.loads(response.read())
            connection.close()
            if response.status != 409 or refusal.get("error") != "the main tree has staged changes":
                fail(f"commit with an existing staged path was not refused exactly: {response.status}, {refusal!r}")
            if (root / "a.py").read_text(encoding="utf-8") != "value = 1\n":
                fail("staged-index refusal applied the worktree prematurely")
            subprocess.run(["git", "reset", "--", "staged.txt"], cwd=root, check=True, capture_output=True)
            (root / "staged.txt").unlink()

            connection, response = _request(host, "POST", "/spec/commit", body={
                "session_id": started["session_id"], "message": "39b: run a spec",
            }, headers=_headers(host))
            committed = json.loads(response.read())
            connection.close()
            if response.status != 200 or committed.get("paths") != ["a.py"] or not committed.get("commit"):
                fail(f"spec commit did not commit exactly a.py: {response.status}, {committed!r}")
            if (root / "a.py").read_text(encoding="utf-8") != "value = 2\n":
                fail("committed spec change was not applied to the main tree")
            names = subprocess.run(["git", "show", "--pretty=format:", "--name-only", "HEAD"], cwd=root, capture_output=True, check=True).stdout.decode().splitlines()
            if names != ["a.py"]:
                fail(f"spec commit included unexpected paths: {names!r}")
            committed_state = spec_run_state()
            phase = json.loads(roadmap.read_text(encoding="utf-8"))["phases"][0]
            if committed_state.get("committed", {}).get("sha") != committed["commit"] or phase["status"] != "done" or phase["items"][0].get("done") is not True:
                fail("successful spec commit did not record its SHA and mark the bound roadmap item done")
        finally:
            host.close()


def _spec_commit_route_fixture(root: Path, *, add_file: bool = False):
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
    (root / "a.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    (root / "specs" / "39").mkdir(parents=True)
    (root / "specs" / "39" / "39b.md").write_text("# 39b — test spec\n", encoding="utf-8")
    (root / "specs" / "39" / "exists.md").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py", "specs"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
    sessions = root / "sessions"
    host = _host(FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]), repo_root=root, sessions_root=sessions)
    host.run.start("seed conversation")
    _wait_until(lambda: host.run._active is None, "seed conversation did not finish")
    source_id, review_id = "spec-source", "spec-review"
    source = SessionStore(sessions, source_id, repo_root=root)
    meta = source.read_meta()
    meta["spec_run"] = {"kind": "implement", "state": "finished", "spec": "specs/39/39b.md", "report": "specs/report/39/39b-report.md", "worktree": "worktree"}
    meta["review"] = {"session_id": review_id, "verdict": "passed", "follow_ups": []}
    source.write_meta(meta)
    source.close()
    worktree = create_worktree(root, sessions / source_id / "worktree")
    if add_file:
        (worktree / "new.py").write_text("created = True\n", encoding="utf-8")
    else:
        (worktree / "a.py").write_text("a = 100\nb = 2\n", encoding="utf-8")
    return host, sessions, source_id, review_id, worktree


def _post_spec_commit(host, session_id: str, message: str = "spec commit"):
    connection, response = _request(
        host, "POST", "/spec/commit", body={"session_id": session_id, "message": message},
        headers=_headers(host),
    )
    body = json.loads(response.read())
    connection.close()
    return response.status, body


@check("host_server.spec_commit_refuses_unstaged_path_edits")
def check_spec_commit_refuses_unstaged_path_edits() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        host, _, source_id, _, worktree = _spec_commit_route_fixture(root)
        try:
            (root / "a.py").write_text("a = 1\nb = 200\n", encoding="utf-8")
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True).stdout
            status, body = _post_spec_commit(host, source_id)
            if status != 409 or body.get("error") != "uncommitted changes in: a.py":
                fail(f"unstaged edit was not named on refusal: {status}, {body!r}")
            if (root / "a.py").read_text(encoding="utf-8") != "a = 1\nb = 200\n" or not worktree.is_dir():
                fail("unstaged edit refusal changed the main file or removed the worktree")
            if subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True).stdout != head:
                fail("unstaged edit refusal changed HEAD")
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode != 0:
                fail("unstaged edit refusal changed the index")
        finally:
            host.close()


@check("host_server.spec_commit_refuses_untracked_addition_path")
def check_spec_commit_refuses_untracked_addition_path() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        host, _, source_id, _, worktree = _spec_commit_route_fixture(root, add_file=True)
        try:
            (root / "new.py").write_text("person's file\n", encoding="utf-8")
            status, body = _post_spec_commit(host, source_id)
            if status != 409 or body.get("error") != "uncommitted changes in: new.py":
                fail(f"untracked collision was not named on refusal: {status}, {body!r}")
            if (root / "new.py").read_text(encoding="utf-8") != "person's file\n" or not worktree.is_dir():
                fail("untracked collision refusal applied the run patch")
        finally:
            host.close()


@check("host_server.spec_commit_hook_failure_leaves_applied_files")
def check_spec_commit_hook_failure_leaves_applied_files() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        host, _, source_id, _, worktree = _spec_commit_route_fixture(root)
        try:
            hook = root / ".git" / "hooks" / "pre-commit"
            hook.write_text("#!/bin/sh\necho hook failed >&2\nexit 1\n", encoding="utf-8")
            hook.chmod(0o755)
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True).stdout
            status, body = _post_spec_commit(host, source_id)
            if status != 409 or "hook failed" not in body.get("error", ""):
                fail(f"pre-commit failure was not returned: {status}, {body!r}")
            if (root / "a.py").read_text(encoding="utf-8") != "a = 100\nb = 2\n" or worktree.exists():
                fail("hook failure did not leave applied files with the worktree gone")
            if subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True).stdout != head:
                fail("hook failure changed HEAD")
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode != 0:
                fail("hook failure left paths staged")
        finally:
            host.close()


@check("host_server.spec_commit_stale_patch_is_not_applied")
def check_spec_commit_stale_patch_is_not_applied() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        host, _, source_id, _, worktree = _spec_commit_route_fixture(root)
        try:
            (root / "a.py").write_text("a = 9\nb = 2\n", encoding="utf-8")
            subprocess.run(["git", "add", "a.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "--quiet", "-m", "conflicting edit"], cwd=root, check=True)
            expected = subprocess.run(["git", "apply", "--check", "--binary"], input=worktree_diff(worktree).patch.encode(), cwd=root, capture_output=True).stderr.decode().strip()
            status, body = _post_spec_commit(host, source_id)
            if status != 409 or not expected or expected not in body.get("error", ""):
                fail(f"stale patch did not return git's refusal: {status}, {body!r}, expected {expected!r}")
            if (root / "a.py").read_text(encoding="utf-8") != "a = 9\nb = 2\n" or not worktree.is_dir():
                fail("stale patch refusal changed the main tree or removed the worktree")
        finally:
            host.close()


@check("host_server.spec_review_refuses_second_active_review")
def check_spec_review_refuses_second_active_review() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        host, sessions, source_id, review_id, _ = _spec_commit_route_fixture(root)
        try:
            source = SessionStore.open(sessions, source_id)
            try:
                meta = source.read_meta()
                meta["review"] = {"session_id": review_id, "verdict": "running", "follow_ups": [], "not_copied": []}
                source.write_meta(meta)
            finally:
                source.close()
            host.run._active_by_session[review_id] = object()
            connection, response = _request(host, "POST", "/spec/review", body={"session_id": source_id}, headers=_headers(host))
            body = json.loads(response.read())
            connection.close()
            if response.status != 409 or "review is already running" not in body.get("error", ""):
                fail(f"second active review was not refused: {response.status}, {body!r}")
        finally:
            host.run._active_by_session.pop(review_id, None)
            host.close()


@check("host_server.spec_review_lists_ineligible_followups")
def check_spec_review_lists_ineligible_followups() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
        (root / "a.py").write_text("a = 1\n", encoding="utf-8")
        (root / "specs" / "39").mkdir(parents=True)
        (root / "specs" / "39" / "39b.md").write_text("# 39b — test spec\n", encoding="utf-8")
        (root / "specs" / "39" / "exists.md").write_text("original\n", encoding="utf-8")
        subprocess.run(["git", "add", "a.py", "specs"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
        sessions = root / "sessions"
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "seed")),
            ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("outside", "write_file", {
                "path": "specs/40/outside.md", "content": "outside\n",
            })])),
            ModelResponse(Message(Role.ASSISTANT, "Verdict: follow-ups: specs/40/outside.md, specs/39/exists.md")),
        ])
        host = _host(provider, repo_root=root, sessions_root=sessions)
        try:
            host.run.start("seed")
            _wait_until(lambda: host.run._active is None, "seed conversation did not finish")
            source_id = "spec-source"
            source = SessionStore(sessions, source_id, repo_root=root)
            meta = source.read_meta()
            meta["spec_run"] = {
                "kind": "implement", "state": "finished", "spec": "specs/39/39b.md",
                "report": "specs/report/39/39b-report.md", "worktree": "worktree",
            }
            source.write_meta(meta)
            source.close()
            worktree = create_worktree(root, sessions / source_id / "worktree")
            (worktree / "a.py").write_text("a = 2\n", encoding="utf-8")
            (worktree / "specs" / "40").mkdir(parents=True)
            connection, response = _request(host, "POST", "/spec/review", body={"session_id": source_id}, headers=_headers(host))
            reply = json.loads(response.read())
            connection.close()
            if response.status != 200:
                fail(f"spec review did not start: {response.status}, {reply!r}")
            def review_finished():
                store = SessionStore.open(sessions, source_id)
                try:
                    return store.read_meta().get("review", {}).get("verdict") != "running"
                finally:
                    store.close()
            _wait_until(review_finished, "spec reviewer did not finish")
            source = SessionStore.open(sessions, source_id)
            try:
                review = source.read_meta().get("review", {})
            finally:
                source.close()
            expected = ["specs/40/outside.md", "specs/39/exists.md"]
            if review.get("verdict") != "follow-ups" or review.get("not_copied") != expected or review.get("follow_ups"):
                fail(f"ineligible follow-ups were not reported precisely: {review!r}")
            if (root / "specs/39/exists.md").read_text(encoding="utf-8") != "original\n":
                fail("reviewer follow-up overwrote an existing main-tree spec")
        finally:
            host.close()


@check("host_server.spec_plan_binds_one_new_spec")
def check_spec_plan_binds_one_new_spec() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        docs = root / "docs"
        docs.mkdir()
        roadmap = docs / "roadmap.json"
        roadmap.write_text('{"goal":"test","phases":[{"id":"39","name":"Phase 39","status":"in_progress","items":["New capability","Leave unchanged"]}]}', encoding="utf-8")
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("outside", "write_file", {
                "path": "README.md", "content": "should be denied\n",
            })])),
            ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("shell", "run_shell", {"argv": ["touch", "planner-shell.txt"]})])),
            ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("new-spec", "write_file", {
                "path": "specs/39/39z-new-capability.md", "content": "# 39z — new capability\n",
            })])),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        host = HostServer(provider, PermissionPolicy(repo_root=root, allowed_write_scope=[root]), sessions_root=root / "sessions")
        host.start()
        try:
            connection, response = _request(host, "POST", "/spec/plan", body={"phase": "39", "item": 0}, headers=_headers(host))
            reply = json.loads(response.read())
            connection.close()
            if response.status != 200:
                fail(f"spec/plan returned {response.status}: {reply!r}")
            deadline = time.monotonic() + 5
            bound = None
            while time.monotonic() < deadline:
                connection, response = _request(host, "GET", "/spec/runs", headers=_headers(host))
                runs = json.loads(response.read())
                connection.close()
                bound = next((item for item in runs if item["session_id"] == reply["session_id"]), None)
                if bound and bound.get("state") != "running":
                    break
                time.sleep(0.02)
            result = json.loads(roadmap.read_text(encoding="utf-8"))
            items = result["phases"][0]["items"]
            expected = {"title": "New capability", "spec": ["specs/39/39z-new-capability.md"]}
            if items[0] != expected or items[1] != "Leave unchanged":
                fail(f"planner did not bind exactly its new spec or changed a neighbor: {items!r}")
            if not bound or bound.get("kind") != "plan" or bound.get("bound") != expected["spec"][0]:
                fail(f"plan session did not record its binding: {bound!r}")
            if (root / "README.md").exists() or (root / "planner-shell.txt").exists():
                fail("planner wrote outside specs or ran a shell command")
        finally:
            host.close()


@check("host_server.spec_plan_records_zero_and_multiple_specs_without_binding")
def check_spec_plan_records_zero_and_multiple_specs_without_binding() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        docs = root / "docs"
        docs.mkdir()
        roadmap = docs / "roadmap.json"
        initial_roadmap = {
            "phases": [{"id": "39", "status": "in_progress", "items": ["New capability"]}],
        }
        roadmap.write_text(json.dumps(initial_roadmap), encoding="utf-8")
        specs = root / "specs" / "39"
        specs.mkdir(parents=True)
        sessions = root / "sessions"
        run = HostRun(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            PermissionPolicy(repo_root=root, allowed_write_scope=[root]), EventBroker(),
            sessions_root=sessions,
        )
        try:
            for session_id, created in (("plan-none", []), ("plan-many", ["39a.md", "39b.md"])):
                for name in ("39a.md", "39b.md"):
                    path = specs / name
                    if name in created:
                        path.write_text(f"# {name}\n", encoding="utf-8")
                    else:
                        path.unlink(missing_ok=True)
                store = SessionStore(sessions, session_id, repo_root=root)
                meta = store.read_meta()
                meta["spec_run"] = {"kind": "plan", "phase": "39", "item": 0, "baseline_specs": []}
                store.write_meta(meta)
                store.close()
                run._finish_spec_plan(session_id)
                result = SessionStore.open(sessions, session_id)
                try:
                    outcome = result.read_meta()["spec_run"]
                finally:
                    result.close()
                expected = [f"specs/39/{name}" for name in created]
                if outcome.get("created") != expected or outcome.get("bound") is not None:
                    fail(f"planner output was not recorded without binding: {outcome!r}")
            if json.loads(roadmap.read_text(encoding="utf-8")) != initial_roadmap:
                fail("zero or multiple planner specs changed the roadmap")
        finally:
            run.close()


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
            ))
            source_store = SessionStore.open(sessions, source_id)
            verdict = source_store.read_meta().get("review", {}).get("verdict")
            source_store.close()
            if verdict != "tree-changed":
                fail(f"review verdict did not prioritize a modified worktree: {verdict!r}")
        finally:
            run.close()


@check("host_server.roadmap_binding_preserves_other_bytes")
def check_roadmap_binding_preserves_other_bytes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        docs = root / "docs"
        docs.mkdir()
        raw = '{\n "goal":"g",\n "phases":[{"id":"39","name":"Phase","status":"in_progress","items":["first", "second"]}]\n}\n'
        path = docs / "roadmap.json"
        path.write_text(raw)
        bind_roadmap_item(root, "39", 1, "specs/39/39b.md")
        updated = path.read_text()
        if not updated.startswith(raw[:raw.index('"second"')]) or not updated.endswith(raw[raw.index('"second"') + len('"second"'):]):
            fail("binding changed bytes outside the selected roadmap item")


@check("host_server.roadmap_binding_converts_string_item")
def check_roadmap_binding_converts_string_item() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "docs").mkdir()
        path = root / "docs" / "roadmap.json"
        path.write_text('{"phases":[{"id":"39","items":["Thing"]}]}')
        bind_roadmap_item(root, "39", 0, "specs/39/39a.md")
        item = json.loads(path.read_text())["phases"][0]["items"][0]
        if item != {"title": "Thing", "spec": ["specs/39/39a.md"]}:
            fail(f"string roadmap item was not bound as an object: {item!r}")


@check("host_server.roadmap_binding_rejects_unknown_item")
def check_roadmap_binding_rejects_unknown_item() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "docs").mkdir()
        (root / "docs" / "roadmap.json").write_text('{"phases":[{"id":"39","items":["Thing"]}]}')
        try:
            bind_roadmap_item(root, "39", 2, "specs/39/39a.md")
        except ValueError:
            return
        fail("unknown roadmap item index was accepted")


@check("host_server.roadmap_commit_marks_done_and_phase")
def check_roadmap_commit_marks_done_and_phase() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "docs").mkdir()
        path = root / "docs" / "roadmap.json"
        path.write_text('{"phases":[{"id":"39","status":"in_progress","items":[{"title":"A","spec":["specs/39/a.md"]},{"title":"B","spec":["specs/39/b.md"],"done":true}]}]}')
        mark_roadmap_spec_done(root, "specs/39/a.md")
        phase = json.loads(path.read_text())["phases"][0]
        if phase["status"] != "done" or phase["items"][0].get("done") is not True:
            fail("commit did not mark its roadmap item and completed phase done")
_APP_HTML_RESOURCE = re.compile(r"""(?:src|href)=[\"']([^\"']+)[\"']""")
_APP_STATIC_IMPORT = re.compile(
    r"""\b(?:import|export)\s+(?:[^;\"']*?\s+from\s*)?[\"']([^\"']+)[\"']"""
)
_APP_DYNAMIC_IMPORT = re.compile(r"""\bimport\(\s*[\"']([^\"']+)[\"']\s*\)""")
_APP_CSS_REFERENCE = re.compile(
    r"""(?:@import\s+(?:url\()?\s*|url\(\s*)[\"']?([^\"'()\s;]+)"""
)
_PRE_19B_COMMIT = "08206022734f05d5c2afb9b32c7e2789a892f1ed"
_PRE_19E_COMMIT = "fa9a7dd06eee2b5f29772c1870e57a648dac9cdc"
_FROZEN_HOST_RUN = (
    (
        "RunStarted",
        "PromptSubmitted",
        "TurnStarted",
        "TurnFinished",
        "RunFinished",
    ),
    (("user", "frozen host"), ("assistant", "done")),
    "final_response",
)
_FROZEN_PROTOCOL = (
    1,
    ("ApprovalReply", "OpenSessionRequest", "PromptRequest", "StopRequest"),
    ("approval_requested", "error", "event", "reply"),
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
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "src" / "parser.py"
        source.parent.mkdir()
        source.write_text("source", encoding="utf-8")
        deep = root / "vendor" / "deep" / "a" / "b"
        deep.mkdir(parents=True)
        for index in range(20_005):
            (deep / f"file-{index:05}.py").touch()
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
) -> HostServer:
    host = HostServer(
        provider or FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
        PermissionPolicy(repo_root=repo_root),
        broker=broker,
        keepalive_seconds=keepalive_seconds,
        token=token,
        sessions_root=sessions_root,
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

    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (
            Path(__file__),
            REPO_ROOT / "scripts" / "checks" / "host_approvals.py",
            REPO_ROOT / "scripts" / "checks" / "host_sessions.py",
        )
    }
    definitions = sum(
        source.count("def _next" + "_sse(")
        + source.count("def _await" + "_sse(")
        for source in sources.values()
    )
    if definitions != 2:
        fail(f"SSE readers were duplicated across host checks: {definitions}")
    for name in ("host_approvals.py", "host_sessions.py"):
        if "_await_sse" not in sources[name]:
            fail(f"{name} did not import the shared SSE reader")


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
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (
            Path(__file__),
            REPO_ROOT / "scripts" / "checks" / "host_approvals.py",
            REPO_ROOT / "scripts" / "checks" / "host_sessions.py",
        )
    }
    if sources["host_approvals.py"].count("_event" + "_stream(") != 0:
        fail("an approval check opens an unsubscribed event stream")
    if sources["host_sessions.py"].count("_event" + "_stream(") != 0:
        fail("a session check opens an unsubscribed event stream")
    if sources["host_server.py"].count("_event" + "_stream(") != 5:
        fail("a host check opens an unsubscribed event stream")

    helper_source = inspect.getsource(_subscribed_stream)
    if "time." + "sleep(" in helper_source:
        fail("subscription helper used time.sleep")
    if "subscriber_count >= expected" not in helper_source:
        fail("subscription wait did not use the caller's expected count")
    if "_wait" + "_until(" not in helper_source:
        fail("subscription helper did not use a predicate wait")

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


@check("host_server.file_route")
def check_file_route() -> None:
    token = "file-route-token"
    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary)
        root = fixture / "repo"
        sibling = fixture / "repo-sibling"
        specs = root / "specs"
        docs = root / "docs"
        source = root / "symphonai_host"
        git = root / ".git"
        for directory in (
            specs,
            specs / "nested",
            docs,
            source,
            git,
            sibling,
            fixture / "etc",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        spec = specs / "18d.md"
        roadmap = docs / "roadmap.json"
        outside = sibling / "outside.md"
        spec.write_text("spec text", encoding="utf-8")
        roadmap.write_text('{"goal":"fixture"}', encoding="utf-8")
        outside.write_text("outside", encoding="utf-8")
        (fixture / "etc" / "passwd").write_text("outside", encoding="utf-8")
        (root / ".env").write_text("secret", encoding="utf-8")
        (git / "config").write_text("private", encoding="utf-8")
        (source / "server.py").write_text("source", encoding="utf-8")
        (specs / "invalid.md").write_bytes(b"\xff")
        (specs / "large.md").write_bytes(
            b"x" * (host_server_module.MAX_FILE_BYTES + 1)
        )
        (specs / "outside-link.md").symlink_to(outside)

        host = _host(repo_root=root, token=token)
        protected_values = (token, str(fixture))

        def request_file(path: str, *, authorized: bool = True):  # noqa: ANN202
            headers = _headers(host) if authorized else {}
            connection, response = _request(
                host,
                "GET",
                f"/file?{urlencode({'path': path})}",
                headers=headers,
            )
            try:
                result = (response.status, tuple(response.getheaders()), response.read())
                wire = repr(result)
                leaked = [value for value in protected_values if value in wire]
                if leaked:
                    fail(f"file route response leaked a protected value: {leaked!r}")
                return result
            finally:
                connection.close()

        try:
            for path, expected_text in (
                ("specs/18d.md", "spec text"),
                ("docs/roadmap.json", '{"goal":"fixture"}'),
            ):
                status, _, body = request_file(path)
                if status != 200 or json.loads(body) != {
                    "path": path,
                    "text": expected_text,
                }:
                    fail(f"file route returned the wrong document for {path!r}")

            traversal = (
                "../etc/passwd",
                str(spec),
                "specs/nested/../../../repo-sibling/outside.md",
                "specs/outside-link.md",
                "docs/../../repo-sibling/outside.md",
            )
            for path in traversal:
                status, _, body = request_file(path)
                if status != 403 or body != b"":
                    fail(f"file route accepted traversal fixture {path!r}: {status}")

            for path in (".env", ".git/config", "symphonai_host/server.py"):
                status, _, body = request_file(path)
                if status != 403 or body != b"":
                    fail(f"file route served a path outside its allow-list: {path!r}")

            status, _, body = request_file("specs/missing.md")
            if status != 404 or json.loads(body) != {"error": "not found"}:
                fail(f"missing file response was not the generic 404: {status}, {body!r}")
            for path, expected_status in (
                ("specs/invalid.md", 415),
                ("specs/large.md", 413),
            ):
                status, _, body = request_file(path)
                if status != expected_status or body != b"":
                    fail(f"file route handled {path!r} as {status} with {body!r}")

            status, _, body = request_file("specs/18d.md", authorized=False)
            if status != 401 or body != b"":
                fail("file route did not require bearer authorization")

            return_types = get_args(
                get_type_hints(protocol_module.decode_request)["return"]
            )
            actual_protocol = (
                protocol_module.PROTOCOL_VERSION,
                tuple(sorted(item.__name__ for item in return_types)),
                tuple(sorted(protocol_module._FRAME_KINDS)),
            )
            if actual_protocol != _FROZEN_PROTOCOL:
                fail(f"file route changed the frozen protocol: {actual_protocol!r}")
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
            if set(survey) != expected_fields:
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

            with mock.patch.dict(
                os.environ,
                {"OPENAI_API_KEY": secret, "ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": " "},
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
                "agents", "skills", "plugins", "withheld", "providers",
            }
            if (
                set(settings) != expected_fields | {"search", "mode"}
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
            if providers != [
                {"name": "anthropic", "env_var": "ANTHROPIC_API_KEY", "key_present": False},
                {"name": "gemini", "env_var": "GEMINI_API_KEY", "key_present": False},
                {"name": "openai", "env_var": "OPENAI_API_KEY", "key_present": True},
            ]:
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
                f"/file?path=docs/roadmap.json&token={token}",
            ):
                status, _, body = get(host, path)
                if status != 401 or body != b"":
                    fail(f"query authentication escaped exact /app: {path!r}, {status}")

            wrong_cookie = {"Cookie": "symphonai_app=wrong-token"}
            status, _, body = get(host, "/app/src/app.js", headers=wrong_cookie)
            if status != 401 or body != b"":
                fail("app asset accepted a cookie carrying the wrong token")

            for method, path, body in (
                ("GET", "/file?path=docs/roadmap.json", None),
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


@check("host_server.event_broker_publish_event_only")
def check_event_broker_publish_event_only() -> None:
    if tuple(inspect.signature(EventBroker.publish).parameters) != ("self", "event"):
        fail(f"EventBroker.publish still accepts session identity: {inspect.signature(EventBroker.publish)}")


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
                kind, payload = _next_sse(connection, response, timeout=5)
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
            host = HostServer(provider, PermissionPolicy(root), system_prompt="host baseline", sessions_root=root / "sessions")
            host.start()
            try:
                with mock.patch.object(provider, "create_response", wraps=provider.create_response) as response_spy:
                    _send_host_prompt(host, "plain")
                    plain = [(message.role.value, message.text) for message in response_spy.call_args.args[0].messages]
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
                    environment_messages = [
                        text for text in systems
                        if text.startswith("Environment when this conversation started")
                    ]
                    if (
                        len(systems) != 2
                        or "project convention" not in systems[0]
                        or not systems[1].startswith(environment_text)
                        or "In the person's messages, @<path> names a file in this repository. Read it with read_file before relying on its contents." not in systems[1]
                        or len(environment_messages) != 1
                    ):
                        fail(f"request did not carry one environment block after instructions: {messages!r}")
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
            host = HostServer(provider, PermissionPolicy(root), working_dir=nested, sessions_root=root / "sessions", chat_token_budget=1_000_000)
            host.start()
            try:
                stderr = io.StringIO()
                with mock.patch.object(provider, "create_response", wraps=provider.create_response) as response_spy, contextlib.redirect_stderr(stderr):
                    _send_host_prompt(host, "check scopes")
                sent = response_spy.call_args.args[0].messages
                rendered = sent[0].text if sent and sent[0].role == Role.SYSTEM else ""
                if len(sent) != 3 or "# instructions: project .symphonai/INSTRUCTIONS.md\n" not in rendered or project_text not in rendered:
                    fail("project instruction text or scope did not reach the provider")
                if "# instructions: directory src/.symphonai/INSTRUCTIONS.md\ndirectory rule" not in rendered or not sent[1].text.startswith("Environment when this conversation started") or "In the person's messages, @<path> names a file in this repository. Read it with read_file before relying on its contents." not in sent[1].text:
                    fail("directory instruction text or scope did not reach the provider")
                if "instruction warning:" not in stderr.getvalue() or "loaded in full" not in stderr.getvalue() or provider.call_count != 1:
                    fail("oversize warning was hidden or the run did not complete")
            finally:
                host.close()


@check("host_server.provider_default_and_rejection")
def check_provider_default_and_rejection() -> None:
    keys = {"ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": "", "OPENAI_API_KEY": ""}
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
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "fixture-key", "OPENAI_API_KEY": "fixture-key"}):
            if host_main._provider().name != "gemini":
                fail("the first keyed vendor in Settings order was not the default")
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
        host = HostServer(None, PermissionPolicy(Path(temporary)), sessions_root=Path(temporary) / "sessions")
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
        run = HostRun(provider, PermissionPolicy(root), EventBroker(), sessions_root=root / "sessions")
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
                RunNode("r1", "leader", "leader", None, None, False, (
                    RunNode("r4", "c", "C", "r1", None, False),
                    RunNode("r2", "a", "A", "r1", None, False, (
                        RunNode("r3", "b", "B", "r2", None, False),
                    )),
                )),
                RunNode("r5", "leader", "leader", None, None, False, (
                    RunNode("r6", "b", "B", "r5", None, False),
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
                host.run._usage_by_session[host.run._conversation[1].run_id] = {}
                agents = _conversation_reply(host)[1]["conversation"]["agents"]
            if [agent["agent_id"] for agent in agents] != ["leader", "a", "b", "c", "orphan"]:
                fail(f"spawn order or repeated-agent folding changed: {agents!r}")
            if [agent["parent_agent_id"] for agent in agents] != [None, "leader", "a", "leader", None]:
                fail(f"nested or unknown parent was misplaced: {agents!r}")
        finally:
            host.close()


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
        context_budget = 280
        host = HostServer(
            provider,
            PermissionPolicy(compact_root),
            sessions_root=compact_root / "sessions",
            chat_token_budget=context_budget,
            chat_recent_turns=1,
        )
        host.start()
        subscription = host.broker.subscribe()
        used = []
        try:
            for index in range(6):
                _send_host_prompt(host, f"prompt {index} " + "x" * 80)
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


def _host_run_snapshot(
    root: Path,
    extensions: Extensions | None,
    mcp_tools=None,  # noqa: ANN001
) -> tuple[tuple, HostRun, tuple]:
    provider = _GatedProvider(
        [ModelResponse(Message(Role.ASSISTANT, "done"))]
    )
    broker = EventBroker()
    subscription = broker.subscribe()
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
    events = []
    while True:
        event = subscription.get(timeout=0.01)
        if event is None:
            break
        events.append(event)
    store = SessionStore.open(root / "sessions", host_run_id)
    loaded, _, _ = load_run_for_resume(store)
    terminal = next(
        (event.stopped_reason for event in events if isinstance(event, RunFinished)),
        None,
    )
    snapshot = (
        tuple(type(event).__name__ for event in events if type(event).__name__ != "SessionStarted"),
        tuple(
            (message.role.value, message.text)
            for message in loaded.messages
            if not (
                message.role == Role.SYSTEM
                and (
                    message.text.startswith("Environment when this conversation started")
                    or message.text.startswith("In the person's messages, @<path>")
                )
            )
        ),
        terminal,
    )
    subscription.close()
    broker.close()
    return snapshot, run, tuple(calls)


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
            snapshot, run, fan_out_calls = _host_run_snapshot(root, extensions)
            if snapshot != _FROZEN_HOST_RUN:
                fail(
                    f"extensions={label} changed HostRun from {_PRE_19B_COMMIT}: "
                    f"expected={_FROZEN_HOST_RUN!r}, actual={snapshot!r}"
                )
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


@check("host_server.extensions_protocol_frozen")
def check_extensions_protocol_frozen() -> None:
    return_types = get_args(
        get_type_hints(protocol_module.decode_request)["return"]
    )
    actual = (
        protocol_module.PROTOCOL_VERSION,
        tuple(sorted(item.__name__ for item in return_types)),
        tuple(sorted(protocol_module._FRAME_KINDS)),
    )
    if actual != _FROZEN_PROTOCOL:
        fail(
            "extension wiring changed the host protocol: "
            f"expected={_FROZEN_PROTOCOL!r}, actual={actual!r}"
        )
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "symphonai_api").rglob("*.py"))
        if "symphonai_host" in path.read_text(encoding="utf-8")
    ]
    if offenders:
        fail(f"runtime import direction reversed: {offenders!r}")


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
    events: list[str] = []
    extensions = mock.Mock(mcp_servers=(), lsp_servers=())
    pool = mock.Mock()
    pool.start.side_effect = lambda: events.append("pool.start") or {}
    pool.close.side_effect = lambda: events.append("pool.close")
    host = mock.Mock()
    host.close.side_effect = lambda: events.append("host.close")

    def construct_host(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        events.append("host.construct")
        return host

    standard = standard_tool_registry()
    with (
        mock.patch.object(host_main, "load_extensions", return_value=extensions),
        mock.patch.object(host_main, "McpPool", return_value=pool) as pool_factory,
        mock.patch.object(host_main, "standard_tool_registry", return_value=standard),
        mock.patch.object(host_main, "_provider", return_value=FakeModelProvider()),
        mock.patch.object(host_main, "HostServer", side_effect=construct_host),
        mock.patch.object(host_main.signal, "signal"),
    ):
        host_main.main(["--repo-root", str(REPO_ROOT)])
    if events != ["pool.start", "host.construct", "host.close", "pool.close"]:
        fail(f"host and MCP pool lifetime order changed: {events!r}")
    if pool.close.call_count != 1:
        fail(f"MCP pool had more than one owner: {pool.close.call_count} closes")
    if pool_factory.call_args.kwargs.get("reserved_names") != set(standard):
        fail("main did not inject the live standard registry keys")


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
            snapshot, _, _ = _host_run_snapshot(
                Path(temporary),
                None,
                mcp_tools=tools,
            )
        if snapshot != _FROZEN_HOST_RUN:
            fail(
                f"mcp_tools={label} changed HostRun from {_PRE_19E_COMMIT}: "
                f"expected={_FROZEN_HOST_RUN!r}, actual={snapshot!r}"
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

    return_types = get_args(get_type_hints(protocol_module.decode_request)["return"])
    actual_protocol = (
        protocol_module.PROTOCOL_VERSION,
        tuple(sorted(item.__name__ for item in return_types)),
        tuple(sorted(protocol_module._FRAME_KINDS)),
    )
    if actual_protocol != _FROZEN_PROTOCOL:
        fail(f"MCP host ownership changed the protocol: {actual_protocol!r}")


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
                if worker_tools != set(standard_tool_registry()) | {"read_tool_result"}:
                    fail(f"host worker lost its stored-result tool: {worker_tools!r}")
                if explorer_tools != {"read_file", "glob", "grep", "list_files", "web_fetch", "read_tool_result"}:
                    fail(f"host explorer registry was wrong: {explorer_tools!r}")
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
            if [(item.role, item.text) for item in sent.call_args.args[0].messages] != [(Role.USER, "plain prompt")]:
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
                          extensions=load_extensions(repo_root=root, home=home))
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
