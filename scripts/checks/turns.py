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


def call(tool_name, /, **arguments):
    return ToolCall(f"{tool_name}-{os.urandom(4).hex()}", tool_name, arguments)


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


class BackgroundScript(FakeModelProvider):
    """Route scripted child requests by their configured model, with real gates."""
    def __init__(self, respond):
        super().__init__()
        self.respond = respond
        self.requests = {}
        self.lock = threading.Lock()

    def create_response(self, request, *, cancel=None):
        with self.lock:
            requests = self.requests.setdefault(request.model, [])
            requests.append(request)
            count = len(requests)
        return self.respond(request, count, cancel)


def await_background_gate(gate, cancel):
    deadline = time.monotonic() + 5
    while not gate.wait(0.01):
        if cancel is not None:
            cancel.raise_if_cancelled()
        assert time.monotonic() < deadline, "background fixture gate did not open"
    if cancel is not None:
        cancel.raise_if_cancelled()


def background_leader(ws, provider, children, *, names=("first",), events=None,
                      leader_tools=None, extra_tools=None, editable=False):
    from symphonai_api.agent_spec import AgentSpec, Isolation, ModelSelector
    policy = replace(ws.policy, shell_enabled=True, shell_allowlist=[()])
    specs = {name: AgentSpec(
        name=name, prompt="", model=ModelSelector(provider="fake", model=name),
        policy_ceiling=policy, tool_names=("write_file",) if editable else ("read_file",),
        isolation=Isolation(workspace_prefix=name),
    ) for name in names}
    return Leader(LeaderConfig(
        provider, children, str(ws.root), leader_policy=policy,
        subagent_specs=specs, events=events, leader_tools=leader_tools,
        extra_tools=extra_tools, memory_extraction_enabled=False,
    ))


@check("turns.background_dispatch_keeps_working_and_waits_for_blocking_results")
def background_dispatch_keeps_working_and_waits_for_blocking_results():
    import json
    gate = threading.Event()
    started = {name: threading.Event() for name in ("first", "second")}
    def respond(request, count, cancel):
        started[request.model].set()
        await_background_gate(gate, cancel)
        return answer(f"{request.model} result")
    children = BackgroundScript(respond)
    task = handoff_spec("Return the scripted result")
    with workspace() as ws:
        def working(request, count):
            if count == 2:
                assert all(event.wait(5) for event in started.values())
                assert not gate.is_set(), "dispatch blocked instead of returning"
        provider = Script([
            answer("", call("dispatch_subagent", subagent_name="first", task=task, background=True),
                   call("dispatch_subagent", subagent_name="second", task=task, background=True)),
            answer("The leader keeps working", call("fixture_action"),
                   call("wait_subagents", names=["first"], timeout_seconds=5),
                   call("wait_subagents", names=["second"], timeout_seconds=5)),
            answer(),
        ], working)
        leader = background_leader(ws, provider, children, names=("first", "second"), editable=True,
                                   leader_tools={"fixture_action": Action(lambda _: gate.set())})
        try:
            run = leader.run("Start both workers")
        finally:
            gate.set()
        collected = {}
        for result in results(run):
            if result.payload and "finished" in result.payload:
                collected.update(result.payload["finished"])
        assert set(collected) == {"first", "second"}
        for name, finished in collected.items():
            blocking = background_leader(ws, Script([
                answer("", call("dispatch_subagent", subagent_name=name, task=task)), answer(),
            ]), BackgroundScript(lambda *_: answer(f"{name} result")), names=(name,), editable=True)
            expected = results(blocking.run("Run the same worker"))[0]
            assert finished["ok"] == expected.ok
            assert finished["content"] == expected.content and "Validation:" in expected.content
            assert finished["error"] == expected.error and finished["payload"] == expected.payload
        assert not any(m.text.startswith("[subagent ") for m in run.leader_messages if m.role is Role.USER)
        assert all(result.ok for result in results(run)), json.dumps([r.error for r in results(run)])


@check("turns.background_results_continue_the_leader_and_cancellation_stops_children")
def background_results_continue_the_leader_and_cancellation_stops_children():
    gate, started = threading.Event(), threading.Event()
    def respond(request, count, cancel):
        started.set()
        await_background_gate(gate, cancel)
        return answer("child result")
    with workspace() as ws:
        def working(request, count):
            if count == 2:
                assert started.wait(5)
                gate.set()
            if count == 3:
                assert any(m.text.startswith("[subagent first finished]") for m in request.messages)
        provider = Script([
            answer("", call("dispatch_subagent", subagent_name="first", task="Read", background=True)),
            answer("My initial answer"), answer("Answer after the child"),
        ], working)
        run = background_leader(ws, provider, BackgroundScript(respond)).run("Start a worker")
        assert run.final_answer == "Answer after the child" and len(provider.requests) == 3

        token, exited = CancellationToken(), threading.Event()
        def cancelled_child(request, count, cancel):
            try:
                token.cancel()
                await_background_gate(threading.Event(), cancel)
            finally:
                exited.set()
        provider = Script([answer("", call("dispatch_subagent", subagent_name="first", task="Read", background=True)), answer()])
        run = background_leader(ws, provider, BackgroundScript(cancelled_child)).run("Cancel", cancel=token)
        assert run.stopped_reason == "cancelled" and exited.wait(5)


@check("turns.closing_background_agents_suppresses_delivery_and_enforces_the_cap")
def closing_background_agents_suppresses_delivery_and_enforces_the_cap():
    names = ("first", "second", "third", "fourth", "fifth")
    exited = {name: threading.Event() for name in names[:4]}
    started = {name: threading.Event() for name in names[:4]}
    def respond(request, count, cancel):
        started[request.model].set()
        try:
            await_background_gate(threading.Event(), cancel)
        finally:
            exited[request.model].set()
    with workspace() as ws:
        def all_started(request, count):
            if count == 2:
                assert all(event.wait(5) for event in started.values())
        provider = Script([
            answer("", *(call("dispatch_subagent", subagent_name=name, task="Read", background=True) for name in names[:4])),
            answer("", call("dispatch_subagent", subagent_name="first", task="Read", background=True),
                   call("dispatch_subagent", subagent_name="fifth", task="Read", background=True),
                   *(call("close_subagent", name=name) for name in names[:4]), call("list_subagents")),
            answer(),
        ], all_started)
        notices = []
        run = background_leader(ws, provider, BackgroundScript(respond), names=names, events=notices.append).run("Start and close workers")
        tool_results = results(run)
        assert "first" in tool_results[4].error and "already running" in tool_results[4].error
        assert all(name in tool_results[5].error for name in names[:4])
        assert all(event.is_set() for event in exited.values())
        rows = tool_results[-1].payload["subagents"]
        assert len(rows) == 4 and all(row["state"] == "closed" for row in rows)
        assert all(r.payload["state"] == "closed" for r in tool_results[6:10])
        assert not any(getattr(event, "kind", None) == "finished" for event in notices)
        assert not any(m.text.startswith("[subagent ") for m in run.leader_messages if m.role is Role.USER)


@check("turns.messages_reach_running_subagents_and_finished_agents_refuse_them")
def messages_reach_running_subagents_and_finished_agents_refuse_them():
    release = threading.Event()
    def respond(request, count, cancel):
        if count == 1:
            return answer("", call("fixture_action"))
        assert any(m.text.startswith("[message from the leader]\nNew input") for m in request.messages)
        return answer("Received input")
    with workspace() as ws:
        provider = Script([
            answer("", call("dispatch_subagent", subagent_name="first", task="Read", background=True)),
            answer("", call("message_subagent", name="first", text="New input"), call("fixture_action"),
                   call("wait_subagents", names=["first"], timeout_seconds=5)),
            answer("", call("message_subagent", name="first", text="Too late")), answer(),
        ])
        run = background_leader(ws, provider, BackgroundScript(respond),
            leader_tools={"fixture_action": Action(lambda _: release.set())},
            extra_tools={"fixture_action": Action(lambda cancel: await_background_gate(release, cancel))}).run("Send input")
        sent = [r for r in results(run) if r.payload and r.payload.get("sent")]
        assert len(sent) == 1
        refusal = results(run)[-1]
        assert not refusal.ok and "first" in refusal.error and "finished" in refusal.error


@check("turns.subagent_questions_wake_waits_and_progress_is_listed")
def subagent_questions_wake_waits_and_progress_is_listed():
    release_question, release_answer = threading.Event(), threading.Event()
    def respond(request, count, cancel):
        if count == 1:
            return answer("", call("fixture_action"),
                          call("message_leader", text="Reading the files", kind="progress"),
                          call("message_leader", text="Which file?", kind="question"))
        if count == 2:
            return answer("", call("fixture_action"))
        assert any(m.text.startswith("[message from the leader]\nUse existing.txt") for m in request.messages)
        return answer()
    child_actions = iter((release_question, release_answer))
    with workspace() as ws:
        provider = Script([
            answer("", call("dispatch_subagent", subagent_name="first", task="Read", background=True)),
            answer("", call("fixture_action"), call("wait_subagents", names=["first"], timeout_seconds=5),
                   call("list_subagents"), call("message_subagent", name="first", text="Use existing.txt"),
                   call("fixture_action")),
            answer(), answer(),
        ])
        leader_actions = iter((release_question, release_answer))
        notices = []
        run = background_leader(ws, provider, BackgroundScript(respond), events=notices.append,
            leader_tools={"fixture_action": Action(lambda _: next(leader_actions).set())},
            extra_tools={"fixture_action": Action(lambda cancel: await_background_gate(next(child_actions), cancel))}).run("Ask a question")
        waits = [r.payload for r in results(run) if r.payload and "questions" in r.payload]
        assert waits and waits[0]["questions"] == [{"name": "first", "text": "Which file?"}]
        assert waits[0]["running"] == ["first"]
        listed = next(r.payload["subagents"] for r in results(run) if r.payload and "subagents" in r.payload)
        assert listed[0]["progress"] == "Reading the files"
        assert any(m.text.startswith("[subagent first progress]") for m in provider.requests[-1].messages)
        assert any(m.text.startswith("[subagent first asks]") for m in provider.requests[-1].messages)
        assert {getattr(event, "kind", None) for event in notices} >= {"progress", "question", "message"}


@check("turns.session_notes_compaction_requires_coverage_of_every_dropped_message")
def session_notes_compaction_requires_coverage_of_every_dropped_message():
    from contextlib import closing
    from unittest import TestCase
    from symphonai_api.cancellation import OperationCancelled
    from symphonai_api.call_class import CallClass
    from symphonai_api.compaction import MODEL_SUMMARY_PROMPT
    from symphonai_api.events import SessionNotesUpdated
    from symphonai_api.providers.base import ProviderError
    from symphonai_api.session import SessionStore

    # Only the final answer crosses the threshold; close must let its update finish.
    with workspace() as ws, closing(SessionStore(ws.outside / "sessions", "closing-notes")) as store:
        release = threading.Event()
        started = threading.Event()
        notes_path = ws.outside / "notes.md"

        class ClosingNotesScript(FakeModelProvider):
            def create_response(self, request, *, cancel=None):
                if request.call_class is CallClass.BACKGROUND:
                    started.set()
                    assert release.wait(5)
                    assert not cancel.cancelled
                    return answer("FINAL NOTES", usage=Usage(input_tokens=123, output_tokens=7))
                return answer("x" * 44000)

        chat = Leader(LeaderConfig(
            ClosingNotesScript(), FakeModelProvider(), str(ws.root), leader_policy=ws.policy,
            session_notes=notes_path, memory_extraction_enabled=False,
        ), session=store)
        try:
            result = chat.run("Finish this short task")
            assert started.wait(5) and not notes_path.exists()
            timer = threading.Timer(0.05, release.set)
            timer.start()
            chat.close()
            timer.join()
            assert notes_path.read_text(encoding="utf-8") == "FINAL NOTES"
            import json
            records = [json.loads(line) for path in (ws.outside / "sessions").rglob("*.jsonl")
                       for line in path.read_text(encoding="utf-8").splitlines()]
            assert any(record["type"] == "usage" and record["data"].get("input_tokens") == 123
                       for record in records)
            assert result.final_answer == "x" * 44000
        finally:
            release.set()
            chat.close()

    for failure, cause in (
        (ProviderError("fixture provider failure"), "fixture provider failure"),
        (OperationCancelled("fixture cancellation"), "cancellation"),
        (answer("", call("read_file", path="existing.txt")), "tool-call"),
        (answer(" "), "empty"),
        (answer("x" * 48001), "oversized"),
    ):
        with workspace() as ws:
            notes_path = ws.outside / "notes.md"
            notes_path.write_text("OLD NOTES", encoding="utf-8")

            class FailingNotesScript(FakeModelProvider):
                foreground = 0

                def create_response(self, request, *, cancel=None):
                    if request.call_class is CallClass.BACKGROUND:
                        if isinstance(failure, Exception):
                            raise failure
                        return failure
                    self.foreground += 1
                    if self.foreground == 1:
                        return answer("", *[call("read_file", path="existing.txt") for _ in range(3)])
                    return answer("x" * 44000)

            chat = Leader(LeaderConfig(
                FailingNotesScript(), FakeModelProvider(), str(ws.root), leader_policy=ws.policy,
                session_notes=notes_path, memory_extraction_enabled=False,
            ))
            with TestCase().assertLogs("symphonai_api.agent_loop", level="WARNING") as logged:
                try:
                    chat.run("Finish")
                finally:
                    chat.close()
            assert any(cause in line for line in logged.output)
            assert notes_path.read_text(encoding="utf-8") == "OLD NOTES"

    for covered in (True, False):
        with workspace() as ws:
            folder = ws.outside / "session"
            folder.mkdir()
            notes_path = folder / "notes.md"
            if not covered:
                notes_path.write_text("STALE NOTES", encoding="utf-8")
            written = threading.Event()
            foreground_entered = threading.Event()

            class NotesScript(FakeModelProvider):
                def __init__(self):
                    super().__init__()
                    self.foreground = 0
                    self.summary_requests = []
                    self.notes_requests = []

                def create_response(self, request, *, cancel=None):
                    if request.call_class is CallClass.BACKGROUND:
                        self.notes_requests.append(request)
                        assert foreground_entered.wait(5), "the notes update blocked the next leader turn"
                        return answer("COVERED NOTES")
                    if request.messages[0].text == MODEL_SUMMARY_PROMPT:
                        self.summary_requests.append(request)
                        return answer("FALLBACK SUMMARY")
                    self.foreground += 1
                    if self.foreground == 1:
                        calls = [call("read_file", path="existing.txt") for _ in range(3 if covered else 1)]
                        return answer("x" * 44000, *calls)
                    if self.foreground == 2:
                        foreground_entered.set()
                        if covered:
                            assert written.wait(5), "notes did not finish while the next turn was running"
                        return answer("y" * 16000, call("read_file", path="existing.txt"))
                    return answer("Finished after compaction")

            def observe(event):
                if isinstance(event, SessionNotesUpdated):
                    written.set()

            provider = NotesScript()
            chat = Leader(LeaderConfig(
                provider, FakeModelProvider(), str(ws.root),
                leader_policy=ws.policy, session_notes=notes_path,
                model_summary=True, compaction_limit=14000, chat_recent_turns=2,
                events=observe, memory_extraction_enabled=False,
            ))
            try:
                result = chat.run("Keep working after compaction")
            finally:
                chat.close()
            summaries = [message.text for message in result.leader_messages
                         if message.role is Role.SYSTEM and message.text.startswith(
                             "Summary of the earlier conversation, written when it was compacted:")]
            assert summaries
            if covered:
                assert len(provider.notes_requests) == 1 and not provider.summary_requests
                assert summaries[-1].endswith("COVERED NOTES")
                assert notes_path.read_text(encoding="utf-8") == "COVERED NOTES"
            else:
                assert not provider.notes_requests and len(provider.summary_requests) == 1
                assert summaries[-1].endswith("FALLBACK SUMMARY")
                assert notes_path.read_text(encoding="utf-8") == "STALE NOTES"

    # Exercise the same coverage rule through chat-boundary and forced compaction.
    for mode in ("chat", "force", "instructions"):
        with workspace() as ws:
            folder = ws.outside / "chat-session"
            folder.mkdir()
            written = threading.Event()

            class ChatNotesScript(FakeModelProvider):
                def __init__(self):
                    super().__init__()
                    self.foreground = 0
                    self.summary_requests = []
                    self.notes_requests = []
                    self.foreground_requests = []

                def create_response(self, request, *, cancel=None):
                    if request.messages[0].text.startswith(MODEL_SUMMARY_PROMPT):
                        self.summary_requests.append(request)
                        return answer("INSTRUCTED SUMMARY")
                    if request.call_class is CallClass.BACKGROUND:
                        self.notes_requests.append(request)
                        return answer("CHAT NOTES")
                    self.foreground_requests.append(request)
                    self.foreground += 1
                    if self.foreground == 1:
                        return answer("x" * 32000, call("read_file", path="existing.txt"))
                    if self.foreground == 2:
                        return answer("y" * 28000)
                    return answer("Next turn after compaction")

            def observe_chat(event):
                if isinstance(event, SessionNotesUpdated):
                    written.set()

            provider = ChatNotesScript()
            chat = Leader(LeaderConfig(
                provider, FakeModelProvider(), str(ws.root), leader_policy=ws.policy,
                session_notes=folder / "notes.md", model_summary=True,
                compaction_limit=14000 if mode == "chat" else 20000,
                chat_recent_turns=1, events=observe_chat, memory_extraction_enabled=False,
            ))
            if mode != "chat":
                chat.seed_chat([
                    Message(Role.USER, "Original goal"),
                    Message(Role.ASSISTANT, "Earlier context to compact"),
                    Message(Role.USER, "Earlier request"),
                ])
            try:
                chat.chat("Read the file and finish this turn")
                assert written.wait(5), "the complete first chat was not covered by notes"
                (ws.root / "existing.txt").write_text("Current contents after the first chat", encoding="utf-8")
                if mode == "chat":
                    chat.chat("Continue with the current file")
                    messages = provider.foreground_requests[-1].messages
                else:
                    instructions = "Focus on the exact file result" if mode == "instructions" else None
                    compacted, _ = chat.force_compact_chat(instructions=instructions)
                    assert compacted.changed
                    messages = compacted.messages
                summary = next(message.text for message in messages if message.role is Role.SYSTEM
                               and message.text.startswith("Summary of the earlier conversation, written when it was compacted:"))
                restore = next(message.text for message in messages
                               if message.text.startswith("Restored after compaction:"))
                assert "existing.txt" in restore and "Current contents after the first chat" in restore
                if mode == "instructions":
                    assert len(provider.summary_requests) == 1
                    assert "Focus on the exact file result" in provider.summary_requests[0].messages[0].text
                    assert summary.endswith("INSTRUCTED SUMMARY")
                else:
                    assert not provider.summary_requests and summary.endswith("CHAT NOTES")
                    if mode == "force":
                        chat.chat("One more request")
                        compacted_again, _ = chat.force_compact_chat()
                        assert compacted_again.changed and not provider.summary_requests
                        assert any(message.text.endswith("CHAT NOTES") for message in compacted_again.messages
                                   if message.role is Role.SYSTEM)
                assert len(provider.notes_requests) == 1
            finally:
                chat.close()


@check("turns.deferred_mcp_tools_load_execute_and_survive_compaction")
def deferred_mcp_tools_load_execute_and_survive_compaction():
    from contextlib import closing
    from symphonai_api.session import SessionStore, load_run
    from symphonai_api.tool_schema import tool_registry_schemas

    class McpFixture(LocalTool):
        deferrable = True
        parameters = {"type": "object", "properties": {}}

        def __init__(self, name):
            self._name = name
            self._description = "Create an issue" if name == "github_create_issue" else "An unrelated MCP operation"
            self.calls = 0

        @property
        def name(self):
            return self._name

        @property
        def description(self):
            return self._description

        def metadata(self, arguments):
            return ToolMetadata(ToolEffect.READ_ONLY, concurrency_safe=True, paths=())

        def _execute(self, tool_call, policy, cancel=None):
            self.calls += 1
            return ToolResult(tool_call.id, ok=True, content="issue created")

    with workspace() as ws, closing(SessionStore(ws.outside / "sessions", "mcp-loading")) as store:
        tools = {name: McpFixture(name) for name in ["github_create_issue", *[f"other_{i}" for i in range(59)]]}
        provider = Script([
            answer("", call("other_0")),
            answer("", call("search_tools", query="issue")),
            answer("", call("github_create_issue")),
            answer("first run finished"), answer("continued after compaction"),
        ])
        budget = [100000]
        agent = ApiAgent(provider, tools, ws.policy,
                         tool_schemas=tool_registry_schemas(tools, provider.wire_format),
                         context_budget=lambda model: budget[0],
                         context_recent_turns=1, transcript=store.writer_for("mcp-agent", is_root=True))
        first = agent.run([Message(Role.SYSTEM, "Preserve instructions"), Message(Role.USER, "Original goal")])
        names = lambda request: {schema.get("function", schema)["name"] for schema in request.tools}
        assert names(provider.requests[0]) == {"search_tools"}
        refused = next(message.tool_result for message in first.messages
                       if message.tool_result is not None and not message.tool_result.ok)
        assert "search_tools" in refused.error and tools["other_0"].calls == 0
        assert "github_create_issue" in names(provider.requests[2])
        assert tools["github_create_issue"].calls == 1
        budget[0] = 600
        agent.run([*first.messages, Message(Role.ASSISTANT, "old detail " * 2000),
                            Message(Role.USER, "Continue with the recent request")])
        assert "github_create_issue" in names(provider.requests[-1])
        assert not any(message.tool_result is not None and isinstance(message.tool_result.payload, dict)
                       and message.tool_result.payload.get("kind") == "tool_search"
                       for message in provider.requests[-1].messages), "compaction did not drop the search result"
        resumed_provider = Script([answer("", call("github_create_issue")), answer()])
        resumed = ApiAgent(resumed_provider, tools, ws.policy,
                           tool_schemas=tool_registry_schemas(tools, resumed_provider.wire_format),
                           transcript=store.writer_for("mcp-agent", is_root=True))
        resumed.run(load_run(store).messages + [Message(Role.USER, "Use the previously loaded tool")])
        assert "github_create_issue" in names(resumed_provider.requests[0])
        assert tools["github_create_issue"].calls == 2
        small_tools = dict(list(tools.items())[:5])
        small_provider = Script([answer()])
        ApiAgent(small_provider, small_tools, ws.policy,
                 tool_schemas=tool_registry_schemas(small_tools, small_provider.wire_format)).run([Message(Role.USER, "A small MCP set")])
        assert names(small_provider.requests[0]) == set(small_tools)


@check("turns.hook_rewrites_still_validate_and_check_permissions")
def hook_rewrites_still_validate_and_check_permissions():
    import json

    with workspace() as ws:
        hooks = HookRunner((HookSpec(("PreToolUse",), prompt="Allow this read.", blocking=True),), cwd=ws.root)
        for reply in (
            {"decision": "allow", "updated_input": None, "context": {"note": "unused"}},
            {"decision": "allow", "updated_input": {"path": "existing.txt"}},
            {"decision": "allow", "updated_input": None, "context": None},
            {"decision": "allow", "context": {"note": "unused"}},
            {"decision": "allow"},
        ):
            text = json.dumps(reply)
            for output in (text, "```json\n" + text + "\n```"):
                assert hooks.pre_tool("read_file", "fixture", {"path": "existing.txt"},
                                      ask_model=lambda prompt, cancel: answer(output)) is None
                result, provider = run_tools(ws, [answer("", call("read_file", path="existing.txt")),
                                                 answer(output), answer()], hooks=hooks)
                assert results(result)[0].ok and "Hook note:" not in results(result)[0].content
                assert len(provider.requests) == 3 and provider.requests[1].effort is None
        for reply in ({}, {"decision": "maybe"}, {"decision": "allow", "updated_input": {"path": 42}},
                      {"decision": "allow", "updated_input": "invalid"}):
            result, _ = run_tools(ws, [answer("", call("read_file", path="existing.txt")),
                                      answer(json.dumps(reply)), answer()], hooks=hooks)
            assert not results(result)[0].ok
        stop_hooks = HookRunner((HookSpec(("Stop",), prompt="Check completion.", blocking=True),), cwd=ws.root)
        assert stop_hooks.stop(agent_name="fixture", session_id=None, final_text="done", turns_used=1,
                               stop_continues=0, ask_model=lambda prompt, cancel:
                               answer('```json\n{"decision":"continue","reason":"work remains"}\n```')) == "work remains"
        for updated in ({"path": ".env"}, {"path": 42}):
            command = (sys.executable, "-c", "import json,sys; p=json.load(sys.stdin); "
                       "assert 'arguments' in p and p['agent_name']; "
                       f"print({json.dumps({'decision': 'allow', 'updated_input': updated})!r})")
            hooks = HookRunner((HookSpec(("PreToolUse",), command, blocking=True),), cwd=ws.root)
            result, _ = run_tools(ws, [answer("", call("read_file", path="existing.txt")), answer()], hooks=hooks)
            refused = results(result)[0]
            assert not refused.ok
            if updated["path"] == ".env":
                assert ".env" in refused.error and "forbidden" in refused.error
            else:
                assert sys.executable in refused.error and "updated_input" in refused.error


@check("turns.stop_hook_continuation_is_bounded_and_failure_finishes")
def stop_hook_continuation_is_bounded_and_failure_finishes():
    with workspace() as ws:
        command = (sys.executable, "-c", 'print(\'{"decision":"continue","reason":"work remains"}\')')
        hooks = HookRunner((HookSpec(("Stop",), command, blocking=True),), cwd=ws.root)
        result, provider = run_tools(ws, [answer("answer") for _ in range(4)], hooks=hooks)
        assert len(provider.requests) == 4 and result.turns_used == 4
        assert sum(message.role is Role.USER and message.text.startswith("[Stop hook] ") for message in result.messages) == 3
        assert "(Stop hook asked to continue; limit of 3 reached)" in result.final_response.message.text
        assert result.messages[-1] == result.final_response.message
        timeout = HookSpec(("Stop",), (sys.executable, "-c", "import time; time.sleep(30)"),
                           blocking=True, timeout_seconds=.01)
        result, provider = run_tools(ws, [answer("finished")], hooks=HookRunner((timeout,), cwd=ws.root))
        assert len(provider.requests) == 1 and result.stopped_reason == "final_response"
        assert result.final_response.message.text == "finished"
