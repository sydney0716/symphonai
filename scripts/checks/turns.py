"""Core turn behaviour through scripted agents and real temporary tools."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from symphonai_api.agent_loop import ApiAgent
from symphonai_api.budgets import RunBudget
from symphonai_api.cancellation import CancellationToken
from symphonai_api.cost import ModelPrice, PriceTable
from symphonai_api.hooks import HookRunner, HookSpec
from symphonai_api.jobs import JobManager
from symphonai_api.leader import Leader, LeaderConfig
from symphonai_api.models import Message, ModelResponse, Role, ToolCall, ToolResult, Usage
from symphonai_api.providers.base import ProviderUnavailableError
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.runner import standard_tool_registry
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_api.tools.read_ledger import ReadLedger
from symphonai_host.broker import EventBroker
from symphonai_host.run import HostRun
from scripts.checks.harness import check
from scripts.checks.workspace import handoff_spec, workspace


class Script(FakeModelProvider):
    def __init__(self, responses=None, before=None):
        super().__init__(responses)
        self.requests = []
        self.before = before

    def create_response(self, request, *, cancel=None):
        self.requests.append(request)
        if self.before is not None:
            self.before(request, len(self.requests))
        return super().create_response(request, cancel=cancel)


def answer(text="finished", *calls, usage=None):
    return ModelResponse(Message(Role.ASSISTANT, text, tool_calls=list(calls)), usage=usage or Usage())


def call(name, **arguments):
    return ToolCall(f"{name}-{os.urandom(4).hex()}", name, arguments)


def results(run):
    messages = run.leader_messages if hasattr(run, "leader_messages") else run.messages
    return [message.tool_result for message in messages if message.tool_result is not None]


def run_tools(ws, responses, *, before=None, extra=None, budget=None, max_turns=None, hooks=None, policy=None):
    ledger = ReadLedger()
    tools = standard_tool_registry(ledger=ledger)
    tools.update(extra or {})
    provider = Script(responses, before)
    agent = ApiAgent(provider, tools, policy or ws.policy, max_turns=max_turns,
                     budget=budget, read_ledger=ledger)
    return agent.run([Message(Role.USER, "Do the fixture task")], hooks=hooks), provider


class Action(LocalTool):
    def __init__(self, action):
        self.action = action

    name = "fixture_action"
    description = "Exercise an externally controlled tool."
    parameters = {"type": "object", "properties": {}}

    def metadata(self, arguments):
        return ToolMetadata(ToolEffect.MUTATING, concurrency_safe=False, paths=None)

    def _execute(self, tool_call, policy, cancel=None):
        self.action(cancel)
        return ToolResult(tool_call.id, ok=True, content="action completed")


def leader(ws, parent, child, **options):
    return Leader(LeaderConfig(parent, child, str(ws.root), leader_policy=replace(
        ws.policy, shell_enabled=True, shell_allowlist=[()],
    ), memory_extraction_enabled=False, **options))


@check("turns.tool_result_reaches_the_next_request")
def tool_result_reaches_the_next_request():
    with workspace() as ws:
        tool_call = call("read_file", path="existing.txt")
        run, provider = run_tools(ws, [answer("", tool_call), answer("answer returned")])
        sent = [m.tool_result for m in provider.requests[1].messages if m.tool_result]
        assert any(r.tool_call_id == tool_call.id and r.ok and "hello from disk" in r.content for r in sent)
        assert run.stopped_reason == "final_response" and run.final_response.message.text == "answer returned"
        assert provider.call_count == 2


@check("turns.turn_limit_stops")
def turn_limit_stops():
    with workspace() as ws:
        run, provider = run_tools(ws, [answer("", call("list_files"))], max_turns=2)
        assert run.stopped_reason == "max_turns" and provider.call_count == 2


@check("turns.token_limit_stops")
def token_limit_stops():
    with workspace() as ws:
        run, provider = run_tools(ws, [answer("", call("list_files"), usage=Usage(10, 2))],
                                  budget=RunBudget(max_total_tokens=10))
        assert run.stopped_reason == "budget_tokens" and provider.call_count == 1


@check("turns.cost_limit_stops")
def cost_limit_stops():
    with workspace() as ws:
        table = PriceTable({"unknown": ModelPrice(Decimal(1_000_000), Decimal(1_000_000))}, "USD")
        run, provider = run_tools(ws, [answer("", call("list_files"), usage=Usage(2, 1))],
                                  budget=RunBudget(max_cost=Decimal(1), price_table=table))
        assert run.stopped_reason == "budget_cost" and provider.call_count == 1


@check("turns.unpriced_cost_limit_stops")
def unpriced_cost_limit_stops():
    with workspace() as ws:
        run, provider = run_tools(ws, [answer()], budget=RunBudget(
            max_cost=Decimal(1), price_table=PriceTable({}, "USD"),
        ))
        assert run.stopped_reason == "budget_cost_unpriced" and provider.call_count == 0


@check("turns.wall_limit_stops")
def wall_limit_stops():
    with workspace() as ws:
        run, provider = run_tools(ws, [answer("", call("fixture_action"))],
            extra={"fixture_action": Action(lambda cancel: threading.Event().wait(0.03))},
            budget=RunBudget(wall_seconds=0.02))
        assert run.stopped_reason == "budget_wall_time" and provider.call_count == 1


@check("turns.cancellation_answers_calls_and_allows_the_next_prompt")
def cancellation_answers_calls_and_allows_the_next_prompt():
    with workspace() as ws:
        token = CancellationToken()
        active = call("fixture_action")
        pending = call("fixture_action")
        executed = []
        provider = Script([answer("", active, pending), answer("recovered")])
        chat = leader(ws, provider, FakeModelProvider(), extra_tools={
            "fixture_action": Action(lambda cancel: (executed.append(True), token.cancel(), cancel.raise_if_cancelled())),
        })
        stopped = chat.chat("start", cancel=token)
        assert stopped.stopped_reason == "cancelled"
        assert {r.tool_call_id for r in results(stopped)} >= {active.id, pending.id}
        assert len(executed) == 1 and all(r.cancelled for r in results(stopped))
        resumed = chat.chat("another prompt")
        assert resumed.stopped_reason == "final_response" and resumed.final_answer == "recovered"
        assert any(m.role is Role.USER and m.text == "another prompt" for m in provider.requests[-1].messages)


@check("turns.overload_uses_and_accounts_for_the_fallback")
def overload_uses_and_accounts_for_the_fallback():
    class Overloaded(Script):
        model = "primary"
        fallback_models = ("backup",)

        def create_response(self, request, *, cancel=None):
            self.requests.append(request)
            if request.model == "primary":
                raise ProviderUnavailableError("busy", status=529)
            return answer("fallback answer", usage=Usage(7, 3))

    with workspace() as ws:
        provider = Overloaded()
        run = ApiAgent(provider, {}, ws.policy).run([Message(Role.USER, "go")], model="primary")
        assert [r.model for r in provider.requests] == ["primary", "backup"]
        assert run.final_response.message.text == "fallback answer"
        assert run.usage_by_model["backup"].total_tokens == 10 and "primary" not in run.usage_by_model


@check("turns.steering_reaches_the_next_request")
def steering_reaches_the_next_request():
    with workspace() as ws:
        parent = Script([answer("", call("fixture_action")), answer()])
        chat = leader(ws, parent, FakeModelProvider(), extra_tools={
            "fixture_action": Action(lambda cancel: chat.redirect("use the smaller solution")),
        })
        chat.chat("start")
        assert any(m.role is Role.USER and m.text == "use the smaller solution" for m in parent.requests[-1].messages)


@check("turns.leader_has_no_file_editing_tools")
def leader_has_no_file_editing_tools():
    with workspace() as ws:
        provider = Script([answer()])
        leader(ws, provider, FakeModelProvider()).run("inspect")
        names = {tool["name"] for tool in provider.requests[0].tools}
        assert "dispatch_subagent" in names
        assert not names.intersection({"write_file", "edit_file", "multi_edit_file"})


@check("turns.dispatch_runs_the_spec_and_its_validation")
def dispatch_runs_the_spec_and_its_validation():
    with workspace() as ws:
        task = handoff_spec("Write output.txt", "test -f output.txt && printf checked > validation-ran")
        parent = Script([answer("", call("dispatch_subagent", subagent_name="implementer", task=task)), answer()])
        child = Script([answer("", call("write_file", path="output.txt", content="implemented")), answer("child result")])
        run = leader(ws, parent, child).run("delegate")
        assert any(m.role is Role.USER and m.text == task for m in child.requests[0].messages)
        assert (ws.root / "output.txt").read_text() == "implemented"
        assert (ws.root / "validation-ran").read_text() == "checked"
        assert results(run)[0].ok and "child result" in results(run)[0].content


@check("turns.failed_validation_is_retried_until_it_passes")
def failed_validation_is_retried_until_it_passes():
    with workspace() as ws:
        task = handoff_spec("Create done.txt", "test -f done.txt")
        parent = Script([answer("", call("dispatch_subagent", subagent_name="implementer", task=task)), answer()])
        child = Script([answer("first finish"), answer("", call("write_file", path="done.txt", content="done")), answer("fixed")])
        run = leader(ws, parent, child).run("delegate")
        feedback = [m.text for m in child.requests[1].messages if m.role is Role.USER and m.text != task]
        assert any("test -f done.txt" in text and "1" in text for text in feedback)
        assert results(run)[0].ok and len(run.subagents["implementer"].runs) == 2


@check("turns.failed_validation_stops_after_five_rounds")
def failed_validation_stops_after_five_rounds():
    with workspace() as ws:
        task = handoff_spec("Cannot pass", "false")
        parent = Script([answer("", call("dispatch_subagent", subagent_name="implementer", task=task)), answer()])
        child = Script([answer("not fixed")])
        run = leader(ws, parent, child).run("delegate")
        assert child.call_count == 5 and not results(run)[0].ok
        assert "false" in results(run)[0].content


def git(root, *arguments):
    return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True).stdout


def init_repository(root):
    git(root, "init", "--quiet")
    git(root, "config", "user.email", "core@example.invalid")
    git(root, "config", "user.name", "Core checks")
    git(root, "add", "existing.txt")
    git(root, "commit", "--quiet", "-m", "fixture")


def worktree_run(ws, validation):
    class Routed(Script):
        def create_response(self, request, *, cancel=None):
            self.requests.append(request)
            is_leader = any(t["name"] == "dispatch_subagent" for t in request.tools)
            if is_leader:
                if not any(m.tool_result for m in request.messages):
                    return answer("", call("dispatch_subagent", subagent_name="implementer", isolation="worktree",
                                           task=handoff_spec("Create isolated.txt", validation)))
                return answer()
            if not any(m.tool_result for m in request.messages):
                return answer("", call("write_file", path="isolated.txt", content="isolated change"))
            return answer("child finished")

    init_repository(ws.root)
    policy = replace(ws.policy, shell_enabled=True, shell_allowlist=[()])
    host = HostRun(Routed(), policy, EventBroker(), sessions_root=ws.outside / "sessions", memory_extraction_enabled=False)
    host.select_mode("allow")
    host.start("delegate in a worktree")
    deadline = time.monotonic() + 10
    while host.active and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    assert not host.active, "worktree run did not finish"
    return host


@check("turns.worktree_changes_require_reviewed_apply")
def worktree_changes_require_reviewed_apply():
    with workspace() as ws:
        host = worktree_run(ws, "test -f isolated.txt")
        try:
            assert not (ws.root / "isolated.txt").exists()
            changes = host.changes()
            assert "isolated.txt" in str(changes) and "isolated change" in str(changes)
            applied = host.apply_worktree("implementer")
            assert "isolated.txt" in applied["applied"]
            assert (ws.root / "isolated.txt").read_text() == "isolated change"
        finally:
            host.close()


@check("turns.failed_worktree_leaves_the_project_unchanged")
def failed_worktree_leaves_the_project_unchanged():
    with workspace() as ws:
        host = worktree_run(ws, "false")
        try:
            assert not (ws.root / "isolated.txt").exists()
            assert git(ws.root, "diff", "HEAD") == b""
        finally:
            host.close()


@check("turns.edit_replaces_one_match_and_refuses_others")
def edit_replaces_one_match_and_refuses_others():
    with workspace() as ws:
        path = ws.root / "sample.txt"
        path.write_text("unique repeated repeated")
        run, _ = run_tools(ws, [answer("", call("read_file", path=path.name)),
            answer("", call("edit_file", path=path.name, old_string="unique", new_string="changed")),
            answer("", call("edit_file", path=path.name, old_string="repeated", new_string="ambiguous")),
            answer("", call("edit_file", path=path.name, old_string="absent", new_string="missing")), answer()])
        assert [r.ok for r in results(run)] == [True, True, False, False]
        assert path.read_text() == "changed repeated repeated"


@check("turns.stale_read_prevents_editing")
def stale_read_prevents_editing():
    with workspace() as ws:
        path = ws.root / "existing.txt"
        def external_change(request, index):
            if index == 2:
                path.write_text("outside change")
        run, _ = run_tools(ws, [answer("", call("read_file", path=path.name)),
            answer("", call("edit_file", path=path.name, old_string="hello", new_string="changed")), answer()], before=external_change)
        assert not results(run)[1].ok and path.read_text() == "outside change"


@check("turns.multi_edit_is_atomic")
def multi_edit_is_atomic():
    with workspace() as ws:
        path = ws.root / "existing.txt"
        original = path.read_text()
        run, _ = run_tools(ws, [answer("", call("read_file", path=path.name)), answer("", call("multi_edit_file", path=path.name, edits=[
            {"old_string": "hello", "new_string": "updated"},
            {"old_string": "missing", "new_string": "bad"},
        ])), answer()])
        assert not results(run)[1].ok and path.read_text() == original


@check("turns.shell_output_is_bounded_with_notice")
def shell_output_is_bounded_with_notice():
    with workspace() as ws:
        policy = replace(ws.policy, shell_enabled=True, shell_allowlist=[()], shell_output_limit_chars=1000)
        run, _ = run_tools(ws, [answer("", call("run_shell", argv=[sys.executable, "-c", "print('x'*5000)"])), answer()], policy=policy)
        output = results(run)[0]
        assert output.ok and 1000 <= len(output.content) < 1500 and "truncat" in output.content.lower()


@check("turns.shell_timeout_kills_descendants")
def shell_timeout_kills_descendants():
    with workspace() as ws:
        marker = ws.root / "survived"
        code = "import subprocess,time; subprocess.Popen(['sh','-c','sleep 0.4; touch survived']); time.sleep(30)"
        policy = replace(ws.policy, shell_enabled=True, shell_allowlist=[()])
        run, _ = run_tools(ws, [answer("", call("run_shell", argv=[sys.executable, "-c", code], timeout_seconds=0.15)), answer()], policy=policy)
        assert not results(run)[0].ok
        threading.Event().wait(0.5)
        assert not marker.exists(), "a descendant survived the timed-out shell"


@check("turns.background_jobs_can_be_read_and_stopped")
def background_jobs_can_be_read_and_stopped():
    with workspace() as ws:
        jobs = JobManager()
        policy = replace(ws.policy, shell_enabled=True, shell_allowlist=[()])
        class JobScript(Script):
            def create_response(self, request, *, cancel=None):
                self.requests.append(request)
                if len(self.requests) == 1:
                    return answer("", call("run_shell", argv=[sys.executable, "-u", "-c", "import time; print('READY',flush=True); time.sleep(30)"], background=True))
                if len(self.requests) == 2:
                    deadline = time.monotonic() + 3
                    job = jobs.list()[0]
                    while "READY" not in jobs.read(job.id)["output"] and time.monotonic() < deadline:
                        threading.Event().wait(0.01)
                    return answer("", call("job_output", job_id=job.id), call("job_stop", job_id=job.id))
                return answer()
        try:
            agent = ApiAgent(JobScript(), standard_tool_registry(jobs=jobs), policy)
            run = agent.run([Message(Role.USER, "run a job")])
            assert all(r.ok for r in results(run))
            assert "READY" in results(run)[1].content
            assert jobs.list()[0].stopped and jobs.list()[0].status != "running"
        finally:
            jobs.close()


def hook_refusal(command):
    with workspace() as ws:
        hooks = HookRunner((HookSpec(("PreToolUse",), command, blocking=True),), cwd=ws.root)
        run, _ = run_tools(ws, [answer("", call("write_file", path="blocked.txt", content="no")), answer()], hooks=hooks)
        assert not results(run)[0].ok and not (ws.root / "blocked.txt").exists()


@check("turns.pre_tool_hook_can_block")
def pre_tool_hook_can_block():
    hook_refusal((sys.executable, "-c", "print('deny: blocked by fixture')"))


@check("turns.failed_pre_tool_hook_fails_closed")
def failed_pre_tool_hook_fails_closed():
    hook_refusal(("/nonexistent-core-hook",))


@check("turns.outside_file_changes_are_announced")
def outside_file_changes_are_announced():
    with workspace() as ws:
        def change(cancel):
            (ws.root / "existing.txt").write_text("external contents")
        run, provider = run_tools(ws, [answer("", call("read_file", path="existing.txt")),
            answer("", call("fixture_action")), answer()], extra={"fixture_action": Action(change)})
        notices = [m for m in provider.requests[-1].messages if m.role is Role.USER]
        assert any("existing.txt" in m.text and "chang" in m.text.lower() for m in notices)
