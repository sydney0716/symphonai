#!/usr/bin/env python3
"""Smoke test for the Leader / DispatchSubagentTool, using FakeModelProvider only.

Verifies:
  - a full Leader.run(): a dispatch_subagent tool-call turn followed by a
    final answer, stopped_reason == "final_response"
  - dispatch with a new subagent_name creates a subagent
  - dispatch with the same subagent_name reuses it (same object identity,
    message history grows across calls)
  - a different subagent_name creates an isolated second pool entry
  - max_subagents is enforced once the limit is reached
  - typed events preserve the ordered run/subagent lifecycle sequence
  - leader and subagent failures and turn exhaustion emit terminal events
  - separate run() calls never reuse pooled subagents; explicit and chat
    clearing empty the pool and clear_subagents() returns the removed count
  - Leader.chat() persists conversation across calls -- a second chat()
    call's leader_messages include the first call's exchange
  - regression check: a real AnthropicProvider used as leader_provider
    actually includes the dispatch_subagent tool definition in its
    outgoing request body (via mocked urllib.request.urlopen) -- guards
    against ApiAgent silently never telling a real model any tool exists
  - regression check: a real subagent's own outgoing request (behind a
    real leader dispatching to it) includes schemas for all six standard
    tools (read_file/write_file/list_files/glob/grep/run_shell)
  - regression check: a real GeminiProvider leader's outgoing
    dispatch_subagent tool declaration carries a non-empty sanitized
    parameters object
  - regression check: a real OpenAICompatibleProvider leader and its real
    OpenAICompatibleProvider subagent both send OpenAI-shaped tool schemas

No real network call is ever made in this script. FakeModelProvider is
used for every check except the real-provider regression checks, which
exercise request-building code against a mocked HTTP layer.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
import unittest.mock as mock
from dataclasses import fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.cancellation import CancellationToken, OperationCancelled  # noqa: E402
from orchestra_api.events import (  # noqa: E402
    CollectingSink,
    CompactionApplied,
    RunFailed,
    RunFinished,
    RunStarted,
    SubagentSpawned,
    ToolCallStarted,
)
import orchestra_api.leader as leader_module  # noqa: E402
from orchestra_api.leader import DispatchSubagentTool, Leader, LeaderConfig  # noqa: E402
from orchestra_api.models import Message, ModelRequest, ModelResponse, Role, ToolCall, ToolResult  # noqa: E402
from orchestra_api.permissions import PermissionPolicy  # noqa: E402
from orchestra_api.providers.anthropic_provider import API_KEY_ENV_VAR, AnthropicProvider  # noqa: E402
from orchestra_api.providers.fake import FakeModelProvider  # noqa: E402
from orchestra_api.providers.gemini_provider import (  # noqa: E402
    API_KEY_ENV_VAR as GEMINI_API_KEY_ENV_VAR,
)
from orchestra_api.providers.gemini_provider import GeminiProvider  # noqa: E402
from orchestra_api.providers.openai_compatible import OpenAICompatibleProvider  # noqa: E402
from orchestra_api.providers.openai_provider import _build_request_body as _build_openai_body  # noqa: E402
from orchestra_api.tools.base import LocalTool  # noqa: E402
from orchestra_api.tools.metadata import (  # noqa: E402
    InterruptBehavior,
    ResultHint,
    ToolEffect,
    ToolMetadata,
)

OPENAI_COMPATIBLE_API_KEY_ENV_VAR = "ORCHESTRA_OPENAI_COMPATIBLE_SMOKE_KEY"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def lifecycle(events: CollectingSink) -> list[tuple[str, str, str | None]]:
    summary: list[tuple[str, str, str | None]] = []
    for event in events.events:
        if isinstance(event, SubagentSpawned):
            summary.append(("spawned", event.subagent_name, None))
        elif isinstance(event, RunStarted):
            summary.append(("started", event.agent_name, None))
        elif isinstance(event, RunFinished):
            summary.append(("finished", event.agent_name, event.stopped_reason))
        elif isinstance(event, RunFailed):
            summary.append(("failed", event.agent_name, None))
    return summary


class _CancellingSubagentTool(LocalTool):
    @property
    def name(self) -> str:
        return "cancel_work"

    @property
    def description(self) -> str:
        return "Cancel the active subagent turn."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.DESTRUCTIVE,
            concurrency_safe=False,
            paths=None,
        )

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        assert cancel is not None
        cancel.cancel()
        raise OperationCancelled


class _RecordingFakeProvider(FakeModelProvider):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__(responses)
        self.requests: list[ModelRequest] = []

    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        self.requests.append(request)
        return super().create_response(request, cancel=cancel)


def _assert_openai_tool_calls_answered(request: ModelRequest, context: str) -> None:
    wire_messages = _build_openai_body(request, "test-model")["messages"]
    for index, message in enumerate(wire_messages):
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            continue
        expected_ids = [tool_call["id"] for tool_call in tool_calls]
        actual_ids = [
            candidate.get("tool_call_id")
            for candidate in wire_messages[index + 1 : index + 1 + len(expected_ids)]
            if candidate.get("role") == "tool"
        ]
        if actual_ids != expected_ids:
            fail(
                f"{context} left unanswered OpenAI tool calls: "
                f"expected={expected_ids!r}, actual={actual_ids!r}, body={wire_messages!r}"
            )


def main() -> None:
    # Names are split so this file does not itself match the validation grep
    # in specs/01c-tui-events-and-stop.md, which asserts the repo is free of
    # these identifiers. Do not join them back into literals.
    retired_names = [
        "on_" + "status",
        "Status" + "Callback",
        "_" + "report",
        *("STATUS_" + state for state in ("PENDING", "WORKING", "DONE", "FAILED", "EXHAUSTED")),
    ]
    present = [name for name in retired_names if hasattr(leader_module, name)]
    if present:
        fail(f"leader still exposes retired compatibility names: {present!r}")
    retired_field = "on_" + "status"
    if retired_field in {item.name for item in fields(LeaderConfig)}:
        fail("LeaderConfig still exposes the retired compatibility field")
    ok("leader compatibility names and configuration field are removed")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # -- full Leader.run(): dispatch then final answer --
        subagent_provider = FakeModelProvider(
            responses=[ModelResponse(message=Message(role=Role.ASSISTANT, content="the sky is blue"))]
        )
        leader_provider = FakeModelProvider(
            responses=[
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="lc1",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "researcher", "task": "why is the sky blue?"},
                            ),
                            ToolCall(
                                id="lc2",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "reviewer", "task": "check the explanation"},
                            ),
                        ],
                    )
                ),
                ModelResponse(message=Message(role=Role.ASSISTANT, content="Answer: the sky is blue.")),
            ]
        )
        identity_events = CollectingSink()
        config = LeaderConfig(
            leader_provider=leader_provider,
            subagent_provider=subagent_provider,
            repo_root=str(root),
            events=identity_events,
        )
        leader = Leader(config)
        result = leader.run("why is the sky blue?")

        if result.stopped_reason != "final_response":
            fail(f"expected stopped_reason='final_response', got {result.stopped_reason!r}")
        if set(result.subagents) != {"researcher", "reviewer"}:
            fail(f"expected two distinct subagents in the pool, got {result.subagents!r}")
        if not result.final_answer:
            fail("expected a non-empty final answer")
        subagent_refs = [record.agent_ref for record in result.subagents.values()]
        if any(ref.parent_agent_id != result.agent.agent_id for ref in subagent_refs):
            fail(f"subagent parent links do not point to the leader: {subagent_refs!r}")
        if len({ref.agent_id for ref in subagent_refs}) != 2:
            fail(f"distinct subagents reused an agent id: {subagent_refs!r}")
        if result.run.agent_id != result.agent.agent_id:
            fail(f"leader run owner link is inconsistent: {result!r}")
        agent_ids = [result.agent.agent_id, *(ref.agent_id for ref in subagent_refs)]
        if len(set(agent_ids)) != 3:
            fail(f"leader and subagent ids are not unique: {agent_ids!r}")
        if not all(agent_id.startswith("agent_") for agent_id in agent_ids):
            fail(f"agent identity prefixes are invalid: {agent_ids!r}")
        if not result.run.run_id.startswith("run_"):
            fail(f"run identity prefix is invalid: {result.run.run_id!r}")
        spawned = identity_events.of_type(SubagentSpawned)
        if len(spawned) != 2:
            fail(f"expected two SubagentSpawned events, got {identity_events.events!r}")
        if any(
            event.agent_id != result.agent.agent_id
            or event.run_id != result.run.run_id
            or event.subagent_agent_id not in {ref.agent_id for ref in subagent_refs}
            for event in spawned
        ):
            fail(f"subagent spawn identity is incorrect: {spawned!r}")
        dispatch_starts = {
            event.tool_call_id: event
            for event in identity_events.of_type(ToolCallStarted)
            if event.tool_name == "dispatch_subagent"
        }
        if any(
            event.turn_id != dispatch_starts[tool_call_id].turn_id
            for event, tool_call_id in zip(spawned, ("lc1", "lc2"), strict=True)
        ):
            fail(f"subagent spawn turn did not match its dispatch call: {spawned!r}")
        started_pairs = {
            (event.agent_id, event.run_id)
            for event in identity_events.of_type(RunStarted)
        }
        expected_agent_ids = {result.agent.agent_id, *(ref.agent_id for ref in subagent_refs)}
        if {agent_id for agent_id, _ in started_pairs} != expected_agent_ids:
            fail(f"run events do not identify leader and subagents: {identity_events.events!r}")
        terminal_pairs = [
            (event.agent_id, event.run_id)
            for event in identity_events.events
            if isinstance(event, (RunFinished, RunFailed))
        ]
        if any(terminal_pairs.count(pair) != 1 for pair in started_pairs):
            fail(f"leader/subagent terminal event cardinality is invalid: {identity_events.events!r}")
        if any(
            (event.agent_id, event.run_id) not in started_pairs
            for event in identity_events.events
            if not isinstance(event, SubagentSpawned)
        ):
            fail(f"event run identity has no matching RunStarted: {identity_events.events!r}")
        allowed_turns = {
            result.agent.agent_id: {
                message.turn_id
                for message in result.leader_messages
                if message.turn_id is not None
            }
        }
        allowed_turns.update(
            {
                record.agent_ref.agent_id: {
                    message.turn_id
                    for message in record.messages
                    if message.turn_id is not None
                }
                for record in result.subagents.values()
            }
        )
        if any(
            event.turn_id is not None
            and event.turn_id not in allowed_turns.get(event.agent_id, set())
            for event in identity_events.events
        ):
            fail(f"event turn identity has no matching message: {identity_events.events!r}")
        ok(f"Leader.run() dispatched a subagent and reached a final answer: {result.final_answer!r}")

        compaction_events = CollectingSink()
        compaction_leader = Leader(
            LeaderConfig(
                leader_provider=FakeModelProvider(
                    [ModelResponse(Message(Role.ASSISTANT, "seeded"))]
                ),
                subagent_provider=FakeModelProvider(),
                repo_root=str(root),
                chat_token_budget=140,
                chat_recent_turns=1,
                events=compaction_events,
            )
        )
        seed_result = compaction_leader.run("seed")
        compaction_leader._chat_messages = [
            Message(Role.SYSTEM, "system prompt must stay"),
            Message(Role.USER, "earliest user goal must stay"),
            Message(Role.ASSISTANT, "old assistant detail " * 120),
            Message(Role.USER, "old follow-up " * 120),
            Message(Role.ASSISTANT, "old analysis " * 120),
            Message(Role.USER, "latest request must stay"),
        ]
        compacted = compaction_leader.compact_chat()
        compaction_applied = compaction_events.of_type(CompactionApplied)
        if not compacted.changed or len(compaction_applied) != 1:
            fail(f"changed compaction did not emit once: {compaction_events.events!r}")
        if (
            compaction_applied[0].agent_id != seed_result.agent.agent_id
            or compaction_applied[0].run_id != seed_result.run.run_id
            or compaction_applied[0].before_tokens != compacted.before_tokens
            or compaction_applied[0].after_tokens != compacted.after_tokens
        ):
            fail(f"compaction event payload is incorrect: {compaction_applied[0]!r}")
        ok("leader compaction emits canonical token and identity data")

        cancellation_typed_events = CollectingSink()
        cancellation_token = CancellationToken()
        cancellation_subagent_provider = FakeModelProvider(
            responses=[
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[ToolCall(id="cancel-sub", name="cancel_work")],
                    )
                )
            ]
        )
        cancellation_leader_provider = _RecordingFakeProvider(
            responses=[
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="cancel-dispatch",
                                name="dispatch_subagent",
                                arguments={
                                    "subagent_name": "cancellable",
                                    "task": "start cancellable work",
                                },
                            )
                        ],
                    )
                ),
                ModelResponse(
                    message=Message(role=Role.ASSISTANT, content="continued safely")
                ),
            ]
        )
        cancellation_leader = Leader(
            LeaderConfig(
                leader_provider=cancellation_leader_provider,
                subagent_provider=cancellation_subagent_provider,
                repo_root=str(root),
                events=cancellation_typed_events,
            )
        )
        cancelling_tool = _CancellingSubagentTool()
        with mock.patch(
            "orchestra_api.leader.standard_tool_registry",
            return_value={cancelling_tool.name: cancelling_tool},
        ):
            cancellation_result = cancellation_leader.chat(
                "delegate cancellable work", cancel=cancellation_token
            )
        if cancellation_result.stopped_reason != "cancelled":
            fail(f"leader did not carry subagent cancellation through: {cancellation_result!r}")
        expected_cancellation_events = [
            ("started", "leader", None),
            ("spawned", "cancellable", None),
            ("started", "cancellable", None),
            ("finished", "cancellable", "cancelled"),
            ("finished", "leader", "cancelled"),
        ]
        cancellation_lifecycle = lifecycle(cancellation_typed_events)
        if cancellation_lifecycle != expected_cancellation_events:
            fail(
                f"expected cancelled lifecycle {expected_cancellation_events!r}, "
                f"got {cancellation_lifecycle!r}"
            )
        cancelled_ref = cancellation_result.subagents["cancellable"].agent_ref
        cancelled_run_events = [
            event
            for event in cancellation_typed_events.of_type(RunFinished)
            if event.agent_id == cancelled_ref.agent_id
        ]
        if (
            len(cancelled_run_events) != 1
            or cancelled_run_events[0].stopped_reason != "cancelled"
        ):
            fail(
                "cancelled subagent did not emit one cancelled RunFinished: "
                f"{cancellation_typed_events.events!r}"
            )
        leader_tool_results = [
            message.tool_result
            for message in cancellation_result.leader_messages
            if message.tool_result is not None
        ]
        if (
            len(leader_tool_results) != 1
            or leader_tool_results[0].ok
            or not leader_tool_results[0].cancelled
        ):
            fail(f"leader cancellation did not synthesize a cancelled tool result: {leader_tool_results!r}")
        cancelled_record = cancellation_result.subagents["cancellable"]
        if not any(message.role == Role.ASSISTANT for message in cancelled_record.messages):
            fail(f"cancelled subagent lost its partial conversation: {cancelled_record.messages!r}")
        if not any(
            message.tool_result is not None and message.tool_result.cancelled
            for message in cancelled_record.messages
        ):
            fail(f"cancelled subagent transcript was not repaired: {cancelled_record.messages!r}")

        continued_result = cancellation_leader.chat("continue after cancellation")
        if continued_result.stopped_reason != "final_response":
            fail(f"leader could not continue after cancellation: {continued_result!r}")
        _assert_openai_tool_calls_answered(
            cancellation_leader_provider.requests[-1],
            "leader chat after subagent cancellation",
        )
        ok("leader cancellation preserves partial work and leaves a reusable transcript")

        # -- chat() reports cancellation the same way run() does --
        chat_cancel_token = CancellationToken()
        chat_cancel_token.cancel()
        chat_cancel_leader = Leader(
            LeaderConfig(
                leader_provider=FakeModelProvider(
                    responses=[
                        ModelResponse(message=Message(role=Role.ASSISTANT, content="unused"))
                    ]
                ),
                subagent_provider=FakeModelProvider(),
                repo_root=root,
            )
        )
        try:
            chat_cancelled = chat_cancel_leader.chat("do work", cancel=chat_cancel_token)
        except OperationCancelled:
            fail("Leader.chat() raised OperationCancelled; run() returns a cancelled result")
        if chat_cancelled.stopped_reason != "cancelled":
            fail(f"Leader.chat() did not report cancellation: {chat_cancelled!r}")
        ok("Leader.chat() reports cancellation as a result, matching Leader.run()")

        # -- create / reuse / isolate / max_subagents, exercised directly on the tool --
        policy = PermissionPolicy(repo_root=root)
        if "events" in inspect.signature(DispatchSubagentTool).parameters:
            fail("DispatchSubagentTool.__init__ still exposes an unwired events argument")
        try:
            DispatchSubagentTool(
                FakeModelProvider(),
                policy,
                **{"events": CollectingSink()},
            )
        except TypeError:
            pass
        else:
            fail("DispatchSubagentTool accepted events without leader identity context")

        standalone_tool = DispatchSubagentTool(
            FakeModelProvider(
                [ModelResponse(Message(Role.ASSISTANT, "standalone complete"))]
            ),
            policy,
        )
        with mock.patch("orchestra_api.leader.emit") as standalone_emit:
            standalone_result = standalone_tool.execute(
                ToolCall(
                    id="standalone-dispatch",
                    name="dispatch_subagent",
                    arguments={"subagent_name": "standalone", "task": "work"},
                ),
                policy,
            )
        if not standalone_result.ok or standalone_emit.called:
            fail(
                "standalone dispatch emitted an event without leader context: "
                f"result={standalone_result!r}, calls={standalone_emit.call_args_list!r}"
            )
        ok("standalone dispatch has no event sink or invalid empty-run event path")

        pool_provider = FakeModelProvider(
            responses=[ModelResponse(message=Message(role=Role.ASSISTANT, content="sub reply"))]
        )
        tool = DispatchSubagentTool(
            subagent_provider=pool_provider, subagent_policy=policy, max_subagents=2, subagent_max_turns=3
        )
        dispatch_metadata = tool.metadata(
            {"subagent_name": "worker", "task": "inspect metadata"}
        )
        expected_dispatch_metadata = ToolMetadata(
            effect=ToolEffect.DESTRUCTIVE,
            concurrency_safe=False,
            paths=None,
            result_hint=ResultHint.TEXT,
            interrupt_behavior=InterruptBehavior.CANCEL,
        )
        if dispatch_metadata != expected_dispatch_metadata:
            fail(
                "dispatch_subagent metadata did not match the literal contract: "
                f"actual={dispatch_metadata!r}, expected={expected_dispatch_metadata!r}"
            )
        if type(tool).execute is not LocalTool.execute:
            fail("dispatch_subagent bypassed the base validation pipeline")
        invalid_dispatch = tool.execute(
            ToolCall(
                id="invalid-dispatch",
                name="dispatch_subagent",
                arguments={"subagent_name": "worker"},
            ),
            policy,
        )
        if (
            invalid_dispatch.ok
            or invalid_dispatch.error
            != "missing required argument: subagent_name and/or task"
            or tool.pool
        ):
            fail(f"dispatch_subagent validation behavior changed: {invalid_dispatch!r}")
        ok("dispatch_subagent declares conservative per-call metadata")

        r1 = tool.execute(
            ToolCall(id="c1", name="dispatch_subagent", arguments={"subagent_name": "worker", "task": "task one"}),
            policy,
        )
        if not r1.ok:
            fail(f"expected first dispatch to succeed: {r1.error}")
        worker_record = tool.pool.get("worker")
        if worker_record is None:
            fail("expected a 'worker' entry in the pool after first dispatch")
        ok("dispatch with a new name creates a subagent")

        r2 = tool.execute(
            ToolCall(id="c2", name="dispatch_subagent", arguments={"subagent_name": "worker", "task": "task two"}),
            policy,
        )
        if not r2.ok:
            fail(f"expected second dispatch to succeed: {r2.error}")
        if tool.pool["worker"] is not worker_record:
            fail("expected reuse to keep the same SubagentRecord object identity")
        if len(worker_record.messages) < 4:
            fail(f"expected message history to grow across both calls, got {len(worker_record.messages)}")
        ok("dispatch with the same name reuses the subagent (same identity, growing history)")

        r3 = tool.execute(
            ToolCall(id="c3", name="dispatch_subagent", arguments={"subagent_name": "helper", "task": "task three"}),
            policy,
        )
        if not r3.ok or tool.pool.get("helper") is worker_record:
            fail("expected a different name to create an isolated second subagent")
        ok("a different name creates an isolated second pool entry")

        r4 = tool.execute(
            ToolCall(id="c4", name="dispatch_subagent", arguments={"subagent_name": "third", "task": "x"}), policy
        )
        if r4.ok or len(tool.pool) != 2:
            fail("expected max_subagents to be enforced once the limit is reached")
        ok("max_subagents is enforced")

        # -- typed events preserve the expected ordered lifecycle --
        status_events = CollectingSink()
        status_subagent_provider = FakeModelProvider(
            responses=[ModelResponse(message=Message(role=Role.ASSISTANT, content="the sky is blue"))]
        )
        status_leader_provider = FakeModelProvider(
            responses=[
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="lc1",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "researcher", "task": "why blue?"},
                            )
                        ],
                    )
                ),
                ModelResponse(message=Message(role=Role.ASSISTANT, content="Answer: blue.")),
            ]
        )
        status_config = LeaderConfig(
            leader_provider=status_leader_provider,
            subagent_provider=status_subagent_provider,
            repo_root=str(root),
            events=status_events,
        )
        Leader(status_config).run("why is the sky blue?")
        expected_events = [
            ("started", "leader", None),
            ("spawned", "researcher", None),
            ("started", "researcher", None),
            ("finished", "researcher", "final_response"),
            ("finished", "leader", "final_response"),
        ]
        actual_events = lifecycle(status_events)
        if actual_events != expected_events:
            fail(f"expected lifecycle event sequence {expected_events}, got {actual_events}")
        ok(f"typed events preserve the expected lifecycle: {actual_events}")

        # Event delivery isolates consumer bugs from the run.
        def _raising_events(event) -> None:  # noqa: ANN001
            raise RuntimeError("event consumer is broken")

        raising_result = Leader(
            LeaderConfig(
                leader_provider=FakeModelProvider(
                    responses=[
                        ModelResponse(message=Message(role=Role.ASSISTANT, content="answer"))
                    ]
                ),
                subagent_provider=FakeModelProvider(),
                repo_root=root,
                events=_raising_events,
            )
        ).run("goal")
        if raising_result.stopped_reason != "final_response":
            fail(f"a raising event sink broke the run: {raising_result!r}")
        ok("a raising event sink cannot fail the run")

        # -- a leader provider exception emits failed before propagating. --
        failed_leader_events = CollectingSink()
        failed_leader_provider = FakeModelProvider()
        failed_leader = Leader(
            LeaderConfig(
                leader_provider=failed_leader_provider,
                subagent_provider=FakeModelProvider(),
                repo_root=str(root),
                events=failed_leader_events,
            )
        )
        with mock.patch.object(
            failed_leader_provider,
            "create_response",
            side_effect=RuntimeError("leader provider failed"),
        ):
            try:
                failed_leader.run("fail now")
            except RuntimeError:
                pass
            else:
                fail("expected leader provider exception to propagate")
        expected_failed_leader_events = [
            ("started", "leader", None),
            ("failed", "leader", None),
        ]
        actual_failed_leader_events = lifecycle(failed_leader_events)
        if actual_failed_leader_events != expected_failed_leader_events:
            fail(
                f"expected failed leader events {expected_failed_leader_events}, "
                f"got {actual_failed_leader_events}"
            )
        ok("leader exception emits exactly one failed terminal status")

        # -- a subagent provider exception terminates both the subagent and
        # the enclosing leader turn as failed. --
        failed_subagent_events = CollectingSink()
        failed_subagent_provider = FakeModelProvider()
        dispatch_then_fail_provider = FakeModelProvider(
            responses=[
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="failed-dispatch",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "broken", "task": "fail"},
                            )
                        ],
                    )
                )
            ]
        )
        failed_subagent_leader = Leader(
            LeaderConfig(
                leader_provider=dispatch_then_fail_provider,
                subagent_provider=failed_subagent_provider,
                repo_root=str(root),
                events=failed_subagent_events,
            )
        )
        with mock.patch.object(
            failed_subagent_provider,
            "create_response",
            side_effect=RuntimeError("subagent provider failed"),
        ):
            try:
                failed_subagent_leader.run("dispatch broken")
            except RuntimeError:
                pass
            else:
                fail("expected subagent provider exception to propagate")
        expected_failed_subagent_events = [
            ("started", "leader", None),
            ("spawned", "broken", None),
            ("started", "broken", None),
            ("failed", "broken", None),
            ("failed", "leader", None),
        ]
        actual_failed_subagent_events = lifecycle(failed_subagent_events)
        if actual_failed_subagent_events != expected_failed_subagent_events:
            fail(
                f"expected failed subagent events {expected_failed_subagent_events}, "
                f"got {actual_failed_subagent_events}"
            )
        ok("subagent exception emits failed for subagent and leader")

        # -- leader max_turns is exhausted, not done or failed. --
        exhausted_leader_events = CollectingSink()
        exhausted_leader_provider = FakeModelProvider(
            responses=[
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="missing-dispatch-args",
                                name="dispatch_subagent",
                                arguments={},
                            )
                        ],
                    )
                )
            ]
        )
        exhausted_result = Leader(
            LeaderConfig(
                leader_provider=exhausted_leader_provider,
                subagent_provider=FakeModelProvider(),
                repo_root=str(root),
                max_leader_turns=1,
                events=exhausted_leader_events,
            )
        ).run("exhaust")
        if exhausted_result.stopped_reason != "max_turns":
            fail(f"expected leader max_turns, got {exhausted_result.stopped_reason!r}")
        expected_exhausted_leader_events = [
            ("started", "leader", None),
            ("finished", "leader", "max_turns"),
        ]
        actual_exhausted_leader_events = lifecycle(exhausted_leader_events)
        if actual_exhausted_leader_events != expected_exhausted_leader_events:
            fail(
                "expected leader exhausted lifecycle, got "
                f"{actual_exhausted_leader_events}"
            )
        ok("leader max_turns emits exactly one exhausted terminal status")

        # -- subagent max_turns emits exhausted while the leader can consume
        # the failed tool result and finish normally. --
        exhausted_subagent_events = CollectingSink()
        exhausted_subagent_provider = FakeModelProvider(
            responses=[
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="subagent-list",
                                name="list_files",
                                arguments={"path": "."},
                            )
                        ],
                    )
                )
            ]
        )
        exhausted_subagent_leader_provider = FakeModelProvider(
            responses=[
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="exhausted-dispatch",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "limited", "task": "list forever"},
                            )
                        ],
                    )
                ),
                ModelResponse(message=Message(role=Role.ASSISTANT, content="handled exhaustion")),
            ]
        )
        exhausted_subagent_result = Leader(
            LeaderConfig(
                leader_provider=exhausted_subagent_leader_provider,
                subagent_provider=exhausted_subagent_provider,
                repo_root=str(root),
                subagent_max_turns=1,
                events=exhausted_subagent_events,
            )
        ).run("dispatch limited")
        expected_exhausted_subagent_events = [
            ("started", "leader", None),
            ("spawned", "limited", None),
            ("started", "limited", None),
            ("finished", "limited", "max_turns"),
            ("finished", "leader", "final_response"),
        ]
        if exhausted_subagent_result.stopped_reason != "final_response":
            fail("expected leader to finish after receiving exhausted subagent result")
        actual_exhausted_subagent_events = lifecycle(exhausted_subagent_events)
        if actual_exhausted_subagent_events != expected_exhausted_subagent_events:
            fail(
                f"expected exhausted subagent events {expected_exhausted_subagent_events}, "
                f"got {actual_exhausted_subagent_events}"
            )
        ok("subagent max_turns emits exhausted and leader still reaches done")

        # -- run() is one-shot: clear the previous pool at entry, even when
        # the same subagent name is dispatched again. --
        pool_reset_leader_provider = FakeModelProvider(
            responses=[
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="pool-first",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "worker", "task": "first"},
                            )
                        ],
                    )
                ),
                ModelResponse(message=Message(role=Role.ASSISTANT, content="first done")),
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="pool-second",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "worker", "task": "second"},
                            )
                        ],
                    )
                ),
                ModelResponse(message=Message(role=Role.ASSISTANT, content="second done")),
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="pool-third",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "worker", "task": "third"},
                            )
                        ],
                    )
                ),
                ModelResponse(message=Message(role=Role.ASSISTANT, content="third done")),
            ]
        )
        pool_reset_leader = Leader(
            LeaderConfig(
                leader_provider=pool_reset_leader_provider,
                subagent_provider=FakeModelProvider(
                    responses=[
                        ModelResponse(message=Message(role=Role.ASSISTANT, content="fresh reply"))
                    ]
                ),
                repo_root=str(root),
                max_subagents=1,
            )
        )
        pool_reset_leader.run("first run")
        first_worker = pool_reset_leader.subagents.get("worker")
        if first_worker is None:
            fail("expected first run to create worker")
        pool_reset_leader.run("second run")
        second_worker = pool_reset_leader.subagents.get("worker")
        if second_worker is None or second_worker is first_worker:
            fail("run() reused a stale subagent from the previous one-shot run")
        ok("separate run() calls create fresh subagents")

        cleared_count = pool_reset_leader.clear_subagents()
        if cleared_count != 1 or pool_reset_leader.subagents:
            fail(
                f"clear_subagents() should clear one pooled agent, got count={cleared_count} "
                f"and pool={pool_reset_leader.subagents!r}"
            )
        if pool_reset_leader.clear_subagents() != 0:
            fail("clear_subagents() should return zero for an already-empty pool")
        ok("clear_subagents() returns the number removed and empties the pool")

        pool_reset_leader.run("third run")
        if len(pool_reset_leader.subagents) != 1:
            fail("expected third run to repopulate one subagent")
        pool_reset_leader.clear_chat()
        if pool_reset_leader.subagents:
            fail("clear_chat() did not clear the subagent pool")
        ok("clear_chat() clears the subagent pool")

        # -- Leader.chat() persists conversation across calls --
        chat_provider = FakeModelProvider(
            responses=[
                ModelResponse(message=Message(role=Role.ASSISTANT, content="hi there")),
                ModelResponse(message=Message(role=Role.ASSISTANT, content="yes, I remember you said hello")),
            ]
        )
        chat_leader = Leader(
            LeaderConfig(leader_provider=chat_provider, subagent_provider=FakeModelProvider(), repo_root=str(root))
        )
        first = chat_leader.chat("hello")
        if len(first.leader_messages) != 2:
            fail(f"expected 2 messages after first chat() call, got {len(first.leader_messages)}")
        second = chat_leader.chat("do you remember what I said?")
        if len(second.leader_messages) != 4:
            fail(f"expected 4 messages after second chat() call (history carried forward), got {len(second.leader_messages)}")
        contents = [m.text for m in second.leader_messages]
        if "hello" not in contents:
            fail("expected the first call's user message to still be present in the second call's context")
        ok("Leader.chat() persists conversation across calls")

        # -- regression: a real leader provider's outgoing request must
        # actually include the dispatch_subagent tool definition --
        os.environ[API_KEY_ENV_VAR] = "sk-ant-fake-test-key-do-not-use"
        captured: dict = {}

        class _FakeHttpResponse:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def read(self) -> bytes:
                return self._body

            def __enter__(self) -> "_FakeHttpResponse":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        def _fake_urlopen(request, timeout=None):  # noqa: ANN001
            captured["body"] = json.loads(request.data.decode("utf-8"))
            payload = json.dumps(
                {
                    "content": [{"type": "text", "text": "no tool needed"}],
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                    "stop_reason": "end_turn",
                }
            ).encode("utf-8")
            return _FakeHttpResponse(payload)

        real_leader = Leader(
            LeaderConfig(
                leader_provider=AnthropicProvider(),
                subagent_provider=AnthropicProvider(),
                repo_root=str(root),
            )
        )
        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            real_leader.run("hello")
        del os.environ[API_KEY_ENV_VAR]

        sent_tools = captured.get("body", {}).get("tools")
        if not sent_tools or sent_tools[0].get("name") != "dispatch_subagent":
            fail(f"expected outgoing request to include the dispatch_subagent tool, got tools={sent_tools!r}")
        ok("real leader provider's outgoing request includes the dispatch_subagent tool definition")

        # -- regression: a real SUBAGENT's outgoing request must include
        # schemas for all six standard tools (read_file/write_file/
        # list_files/glob/grep/run_shell), not just the leader's own tool --
        os.environ[API_KEY_ENV_VAR] = "sk-ant-fake-test-key-do-not-use"
        subagent_requests: list[dict] = []
        call_count = [0]

        def _fake_urlopen_dispatch(request, timeout=None):  # noqa: ANN001
            call_count[0] += 1
            body = json.loads(request.data.decode("utf-8"))
            if call_count[0] == 1:
                # leader's turn: decide to dispatch to a subagent
                payload = {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "c1",
                            "name": "dispatch_subagent",
                            "input": {"subagent_name": "researcher", "task": "read a.txt"},
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "tool_use",
                }
            elif call_count[0] == 2:
                # the subagent's own turn -- this is the request we care about
                subagent_requests.append(body)
                payload = {
                    "content": [{"type": "text", "text": "done"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "end_turn",
                }
            else:
                # leader's final answer
                payload = {
                    "content": [{"type": "text", "text": "final answer"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "end_turn",
                }
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

        dispatch_leader = Leader(
            LeaderConfig(
                leader_provider=AnthropicProvider(),
                subagent_provider=AnthropicProvider(),
                repo_root=str(root),
            )
        )
        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen_dispatch):
            dispatch_leader.run("please read a.txt via researcher")
        del os.environ[API_KEY_ENV_VAR]

        if not subagent_requests:
            fail("expected the subagent's own request to have been captured")
        sent_tool_names = {t.get("name") for t in subagent_requests[0].get("tools", [])}
        expected_tool_names = {
            "read_file",
            "write_file",
            "list_files",
            "glob",
            "grep",
            "run_shell",
        }
        if sent_tool_names != expected_tool_names:
            fail(f"expected subagent request to include {expected_tool_names}, got {sent_tool_names!r}")
        ok("real subagent's outgoing request includes all six standard tool schemas")

        # -- regression: a real GeminiProvider leader must send
        # dispatch_subagent as a Gemini function declaration with sanitized,
        # non-empty parameters --
        os.environ[GEMINI_API_KEY_ENV_VAR] = "AIza-fake-test-key-do-not-use"
        gemini_leader_captured: dict = {}

        def _fake_gemini_urlopen(request, timeout=None):  # noqa: ANN001
            gemini_leader_captured["body"] = json.loads(request.data.decode("utf-8"))
            payload = {
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "no dispatch"}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            }
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

        gemini_leader = Leader(
            LeaderConfig(
                leader_provider=GeminiProvider(),
                subagent_provider=FakeModelProvider(),
                repo_root=str(root),
            )
        )
        with mock.patch("urllib.request.urlopen", side_effect=_fake_gemini_urlopen):
            gemini_leader.run("hello")
        del os.environ[GEMINI_API_KEY_ENV_VAR]

        declarations = (gemini_leader_captured.get("body", {}).get("tools") or [{}])[0].get(
            "functionDeclarations", []
        )
        dispatch_declaration = next(
            (d for d in declarations if d.get("name") == "dispatch_subagent"), None
        )
        if dispatch_declaration is None:
            fail(f"expected Gemini leader request to declare dispatch_subagent, got {declarations!r}")
        parameters = dispatch_declaration.get("parameters")
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        if (
            not isinstance(parameters, dict)
            or parameters.get("type") != "object"
            or set(properties) != {"subagent_name", "task"}
            or parameters.get("required") != ["subagent_name", "task"]
        ):
            fail(f"expected non-empty sanitized Gemini dispatch parameters, got {parameters!r}")
        ok("real Gemini leader request carries non-empty sanitized dispatch_subagent parameters")

        # -- regression: OpenAI-compatible providers must use OpenAI tool
        # schema shape even when their provider labels are vendor names --
        os.environ[OPENAI_COMPATIBLE_API_KEY_ENV_VAR] = "sk-compatible-fake-test-key-do-not-use"
        compatible_leader_requests: list[dict] = []
        compatible_subagent_requests: list[dict] = []
        compatible_call_count = [0]

        def _fake_openai_compatible_urlopen(request, timeout=None):  # noqa: ANN001
            compatible_call_count[0] += 1
            body = json.loads(request.data.decode("utf-8"))
            if compatible_call_count[0] == 1:
                compatible_leader_requests.append(body)
                payload = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "dispatch_subagent",
                                            "arguments": json.dumps(
                                                {"subagent_name": "researcher", "task": "read a.txt"}
                                            ),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            elif compatible_call_count[0] == 2:
                compatible_subagent_requests.append(body)
                payload = {
                    "choices": [
                        {"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            else:
                payload = {
                    "choices": [
                        {"message": {"role": "assistant", "content": "final answer"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

        compatible_leader = Leader(
            LeaderConfig(
                leader_provider=OpenAICompatibleProvider(
                    api_key_env_var=OPENAI_COMPATIBLE_API_KEY_ENV_VAR,
                    base_url="https://example.invalid/v1",
                    model="compatible-leader-test",
                    provider_label="grok",
                ),
                subagent_provider=OpenAICompatibleProvider(
                    api_key_env_var=OPENAI_COMPATIBLE_API_KEY_ENV_VAR,
                    base_url="https://example.invalid/v1",
                    model="compatible-subagent-test",
                    provider_label="kimi",
                ),
                repo_root=str(root),
            )
        )
        with mock.patch("urllib.request.urlopen", side_effect=_fake_openai_compatible_urlopen):
            compatible_leader.run("please read a.txt via researcher")
        del os.environ[OPENAI_COMPATIBLE_API_KEY_ENV_VAR]

        if not compatible_leader_requests:
            fail("expected the OpenAI-compatible leader request to have been captured")
        leader_tools = compatible_leader_requests[0].get("tools", [])
        leader_dispatch_tool = leader_tools[0] if leader_tools else {}
        leader_function = leader_dispatch_tool.get("function")
        if (
            leader_dispatch_tool.get("type") != "function"
            or not isinstance(leader_function, dict)
            or leader_function.get("name") != "dispatch_subagent"
            or leader_function.get("parameters", {}).get("type") != "object"
        ):
            fail(f"expected OpenAI-shaped leader dispatch tool, got {leader_dispatch_tool!r}")

        if not compatible_subagent_requests:
            fail("expected the OpenAI-compatible subagent request to have been captured")
        subagent_tools = compatible_subagent_requests[0].get("tools", [])
        subagent_tool_names = {
            t.get("function", {}).get("name")
            for t in subagent_tools
            if t.get("type") == "function" and isinstance(t.get("function"), dict)
        }
        if subagent_tool_names != expected_tool_names or len(subagent_tool_names) != len(subagent_tools):
            fail(f"expected OpenAI-shaped subagent tools for {expected_tool_names}, got {subagent_tools!r}")
        ok("real OpenAI-compatible leader and subagent requests use OpenAI tool schema shape")

        print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
