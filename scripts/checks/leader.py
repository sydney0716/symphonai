"""Registered checks for Leader and DispatchSubagentTool behavior."""

from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path
import unittest.mock as mock
from dataclasses import fields

from symphonai_api.agent_memory import MAX_ENTRY_CHARS, AgentMemory, MemorySettings
from symphonai_api.agent_run import RunPhase, new_agent_run
from symphonai_api.budgets import RunBudget
from symphonai_api.agent_spec import (
    AgentSpec,
    ContextInheritance,
    IOContract,
    Isolation,
    ModelSelector,
)
from symphonai_api.call_class import CallClass
from symphonai_api.cost import UsageTotals
from symphonai_api.cancellation import (
    CancelReason,
    CancellationToken,
    OperationCancelled,
)
from symphonai_api.events import (
    AssistantTextDelta,
    CollectingSink,
    CompactionApplied,
    RunFailed,
    RunFinished,
    RunStarted,
    SubagentSpawned,
    ToolCallStarted,
)
from symphonai_api.compaction import ContextCompactionError, estimate_messages_tokens
import symphonai_api.leader as leader_module
from symphonai_api.leader import DispatchSubagentTool, Leader, LeaderConfig, builtin_subagent_specs
from symphonai_api.leases import LeaseConflict, WorkspaceLeases
from symphonai_api.models import (
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.anthropic_provider import API_KEY_ENV_VAR, AnthropicProvider
from symphonai_api.providers.base import ContextLengthExceededError
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.providers.gemini_provider import (
    API_KEY_ENV_VAR as GEMINI_API_KEY_ENV_VAR,
)
from symphonai_api.providers.gemini_provider import GeminiProvider
from symphonai_api.providers.openai_compatible import OpenAICompatibleProvider
from symphonai_api.providers.openai_provider import (
    _build_request_body as _build_openai_body,
)
from symphonai_api.runner import standard_tool_registry
from symphonai_api.session import SessionStore
from symphonai_api.streaming import StreamCompleted, TextDelta
from symphonai_api.tool_results import ToolResultStore
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import (
    InterruptBehavior,
    ResultHint,
    ToolEffect,
    ToolMetadata,
)
from symphonai_api.web_search import SearchBackend

from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


OPENAI_COMPATIBLE_API_KEY_ENV_VAR = "SYMPHONAI_OPENAI_COMPATIBLE_SMOKE_KEY"


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


class _AnthropicFakeProvider(FakeModelProvider):
    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def wire_format(self) -> int:
        return 2


class _WindowCountingProvider(_AnthropicFakeProvider):
    def __init__(self, window: int = 200_000) -> None:
        super().__init__([ModelResponse(Message(Role.ASSISTANT, "recovered"))])
        self.window = window
        self.estimates: list[int] = []
        self.actual_counts: list[int] = []

    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        estimate = estimate_messages_tokens(request.messages)
        actual = (estimate * 3 + 1) // 2
        self.estimates.append(estimate)
        self.actual_counts.append(actual)
        if actual > self.window:
            raise ContextLengthExceededError(
                "request exceeded the model context window",
                actual_tokens=actual,
                limit_tokens=self.window,
            )
        return super().create_response(request, cancel=cancel)


class _GrowingRequestProvider(_AnthropicFakeProvider):
    def __init__(self, window: int = 200_000) -> None:
        super().__init__()
        self.window = window
        self.estimates: list[int] = []
        self.actual_counts: list[int] = []
        self.overflowed = False

    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        estimate = estimate_messages_tokens(request.messages)
        actual = int(1.2 * estimate)
        self.estimates.append(estimate)
        self.actual_counts.append(actual)
        if actual > self.window and not self.overflowed:
            self.overflowed = True
            raise ContextLengthExceededError(
                "request exceeded the model context window",
                actual_tokens=actual,
                limit_tokens=self.window,
            )
        if self.overflowed:
            return ModelResponse(Message(Role.ASSISTANT, "recovered"))
        index = len(self.estimates) - 1
        return ModelResponse(
            Message(
                Role.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        f"grow-{index}",
                        "read_file",
                        {"path": f"growth-{index}.txt"},
                    )
                ],
            )
        )


class _RaisingProvider(FakeModelProvider):
    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        raise RuntimeError("scripted provider failure")


class _ContextOverflowProvider(FakeModelProvider):
    def __init__(self, overflows: list[bool]) -> None:
        super().__init__([ModelResponse(Message(Role.ASSISTANT, "recovered"))])
        self._overflows = iter(overflows)
        self.overflow_error = ContextLengthExceededError(
            "request exceeded the model context window"
        )
        self.requests: list[ModelRequest] = []

    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        self.requests.append(request)
        if next(self._overflows, False):
            raise self.overflow_error
        return super().create_response(request, cancel=cancel)


class _DeadlineProvider(FakeModelProvider):
    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        assert cancel is not None
        cancel.wait(1.0)
        cancel.raise_if_cancelled()
        raise AssertionError("deadline did not cancel the provider")


class _LeaseObservingProvider(FakeModelProvider):
    def __init__(self, leases: WorkspaceLeases, prefix: str) -> None:
        super().__init__([ModelResponse(Message(Role.ASSISTANT, "leased"))])
        self._leases = leases
        self._prefix = prefix
        self.observed_holder: str | None = None

    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        self.observed_holder = self._leases.holder_for(self._prefix)
        return super().create_response(request, cancel=cancel)


def _spec(
    root: Path,
    name: str,
    *,
    policy: PermissionPolicy | None = None,
    prompt: str = "",
    isolation: Isolation = Isolation(),
    io: IOContract = IOContract(),
    deadline_seconds: float | None = None,
    max_depth: int = 0,
    effort: str | None = None,
    model: str = "test-model",
    memory: MemorySettings = MemorySettings(),
) -> AgentSpec:
    return AgentSpec(
        name=name,
        prompt=prompt,
        model=ModelSelector("fake", model, effort=effort),
        policy_ceiling=policy or PermissionPolicy(root),
        deadline_seconds=deadline_seconds,
        isolation=isolation,
        io=io,
        memory=memory,
        call_class=CallClass.BACKGROUND,
        max_depth=max_depth,
    )


def _dispatch(name: str, task: str, call_id: str = "dispatch") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="dispatch_subagent",
        arguments={"subagent_name": name, "task": task},
    )


@check("leader.spec_effort_reaches_requests")
def check_spec_effort_reaches_requests() -> None:
    with workspace() as ws:
        captured: list[dict] = []

        def effort_urlopen(request, timeout=None):  # noqa: ANN001
            captured.append(json.loads(request.data.decode("utf-8")))
            payload = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

        child_provider = OpenAICompatibleProvider(
            api_key_env_var=OPENAI_COMPATIBLE_API_KEY_ENV_VAR,
            base_url="https://mock.invalid/v1",
            provider_label="grok",
        )
        specs = {
            "quick": _spec(
                ws.root, "quick", effort="low", model="gpt-5.4-mini"
            ),
            "deep": _spec(
                ws.root, "deep", effort="high", model="gpt-5.4-mini"
            ),
        }
        dispatching_provider = FakeModelProvider(
            [
                ModelResponse(
                    Message(
                        Role.ASSISTANT,
                        tool_calls=[
                            _dispatch("quick", "quick work", "quick-call"),
                            _dispatch("deep", "deep work", "deep-call"),
                        ],
                    )
                ),
                ModelResponse(Message(Role.ASSISTANT, "all done")),
            ]
        )
        dispatching_leader = Leader(
            LeaderConfig(
                leader_provider=dispatching_provider,
                subagent_provider=child_provider,
                repo_root=str(ws.root),
                subagent_specs=specs,
            )
        )
        with mock.patch.dict(
            os.environ,
            {OPENAI_COMPATIBLE_API_KEY_ENV_VAR: "compatible-effort-key"},
        ), mock.patch("urllib.request.urlopen", side_effect=effort_urlopen):
            dispatch_result = dispatching_leader.run("delegate both tasks")
        if dispatch_result.final_answer != "all done":
            fail(f"effort test leader did not finish: {dispatch_result!r}")
        efforts = [body.get("reasoning_effort") for body in captured]
        if efforts != ["low", "high"]:
            fail(f"subagent specs did not keep independent efforts: {captured!r}")

        leader_provider = _RecordingFakeProvider(
            [ModelResponse(Message(Role.ASSISTANT, "leader done"))]
        )
        leader_spec = _spec(ws.root, "leader", effort="xhigh")
        leader = Leader(
            LeaderConfig(
                leader_provider=leader_provider,
                subagent_provider=FakeModelProvider(),
                repo_root=str(ws.root),
                subagent_specs={"leader": leader_spec},
            )
        )
        result = leader.run("work")
        if result.final_answer != "leader done":
            fail(f"leader effort run did not finish: {result!r}")
        if [request.effort for request in leader_provider.requests] != ["xhigh"]:
            fail(f"leader spec effort did not reach its request: {leader_provider.requests!r}")


@check("leader.conversation_effort_stays_with_leader")
def check_conversation_effort_stays_with_leader() -> None:
    with workspace() as ws:
        leader_provider = _RecordingFakeProvider([
            ModelResponse(Message(
                Role.ASSISTANT,
                tool_calls=[
                    _dispatch("configured", "configured work", "configured-call"),
                    _dispatch("plain", "plain work", "plain-call"),
                ],
            )),
            ModelResponse(Message(Role.ASSISTANT, "all done")),
        ])
        child_provider = _RecordingFakeProvider([
            ModelResponse(Message(Role.ASSISTANT, "configured done")),
            ModelResponse(Message(Role.ASSISTANT, "plain done")),
        ])
        leader = Leader(LeaderConfig(
            leader_provider=leader_provider,
            subagent_provider=child_provider,
            repo_root=str(ws.root),
            leader_effort="high",
            subagent_specs={
                "configured": _spec(
                    ws.root,
                    "configured",
                    effort="low",
                    model="configured-model",
                ),
                "plain": _spec(ws.root, "plain", model="plain-model"),
            },
        ))
        result = leader.run("delegate both tasks")
        if result.final_answer != "all done":
            fail(f"conversation effort run did not finish: {result!r}")
        if [request.effort for request in leader_provider.requests] != ["high", "high"]:
            fail(f"conversation effort did not stay on leader requests: {leader_provider.requests!r}")
        child_efforts = {
            request.model: request.effort
            for request in child_provider.requests
        }
        if child_efforts != {"configured-model": "low", "plain-model": None}:
            fail(f"conversation effort leaked into a dispatched agent: {child_provider.requests!r}")


@check("leader.definition_effort_overrides_conversation")
def check_definition_effort_overrides_conversation() -> None:
    with workspace() as ws:
        provider = _RecordingFakeProvider([
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        leader = Leader(LeaderConfig(
            leader_provider=provider,
            subagent_provider=FakeModelProvider(),
            repo_root=str(ws.root),
            leader_effort="high",
            subagent_specs={
                "leader": _spec(ws.root, "leader", effort="low"),
            },
        ))
        result = leader.run("work")
        if result.final_answer != "done":
            fail(f"defined leader effort run did not finish: {result!r}")
        if [request.effort for request in provider.requests] != ["low"]:
            fail(f"conversation effort replaced the leader definition: {provider.requests!r}")


# Captured from commit c11c7c3 -- the last tree before leader control-plane
# wiring -- by running the probe below. Frozen rather than recomputed from
# repository history: that would compare this change with itself once
# committed, and that commit is not in this repository's history at all.
_DEFAULT_SPECS_PRE_07G_OUTPUT = json.loads(
    r'''
{
  "final": "final",
  "lifecycle": [
    ["started", "leader", null],
    ["spawned", "worker", null],
    ["started", "worker", null],
    ["finished", "worker", "final_response"],
    ["finished", "leader", "final_response"]
  ],
  "messages": [
    {"calls": [], "content": "system", "result": null, "role": "system"},
    {"calls": [], "content": "goal", "result": null, "role": "user"},
    {
      "calls": [["d1", "dispatch_subagent", {"subagent_name": "worker", "task": "inspect"}]],
      "content": "",
      "result": null,
      "role": "assistant"
    },
    {
      "calls": [],
      "content": "",
      "result": ["d1", true, "child", null, false],
      "role": "tool"
    },
    {"calls": [], "content": "final", "result": null, "role": "assistant"}
  ],
  "pool": {
    "worker": {
      "messages": [
        {"calls": [], "content": "inspect", "result": null, "role": "user"},
        {"calls": [], "content": "child", "result": null, "role": "assistant"}
      ],
      "turns": 1,
      "usage": [["unknown", {"calls": 1, "input_tokens": 0, "output_tokens": 0}]]
    }
  },
  "stop": "final_response",
  "usage": [
    [["unknown", {"calls": 1, "input_tokens": 0, "output_tokens": 0}]],
    [["unknown", {"calls": 2, "input_tokens": 0, "output_tokens": 0}]]
  ]
}
'''
)


def _default_specs_probe() -> tuple[dict, int]:
    def message(value: Message) -> dict:
        result = value.tool_result
        return {
            "role": value.role.value,
            "content": value.text,
            "calls": [
                (call.id, call.name, call.arguments) for call in value.tool_calls
            ],
            "result": (
                None
                if result is None
                else (
                    result.tool_call_id,
                    result.ok,
                    result.content,
                    result.error,
                    result.cancelled,
                )
            ),
        }

    def usage_totals(value) -> dict[str, int]:
        return {
            key: getattr(value, key)
            for key in ("calls", "input_tokens", "output_tokens")
        }

    events = CollectingSink()
    leader = Leader(
        LeaderConfig(
            leader_provider=FakeModelProvider(
                [
                    ModelResponse(
                        Message(
                            Role.ASSISTANT,
                            tool_calls=[
                                ToolCall(
                                    id="d1",
                                    name="dispatch_subagent",
                                    arguments={
                                        "subagent_name": "worker",
                                        "task": "inspect",
                                    },
                                )
                            ],
                        )
                    ),
                    ModelResponse(Message(Role.ASSISTANT, "final")),
                ]
            ),
            subagent_provider=FakeModelProvider(
                [ModelResponse(Message(Role.ASSISTANT, "child"))]
            ),
            repo_root=".",
            events=events,
        )
    )
    outcome = leader.run("goal", system_prompt="system")
    lifecycle_summary = []
    for event in events.events:
        if isinstance(event, SubagentSpawned):
            lifecycle_summary.append(("spawned", event.subagent_name, None))
        elif isinstance(event, RunStarted):
            lifecycle_summary.append(("started", event.agent_name, None))
        elif isinstance(event, RunFinished):
            lifecycle_summary.append(
                ("finished", event.agent_name, event.stopped_reason)
            )
        elif isinstance(event, RunFailed):
            lifecycle_summary.append(("failed", event.agent_name, None))
    pool = {
        name: {
            "messages": [message(item) for item in record.messages],
            "turns": record.turns_used,
            "usage": sorted(
                (model, usage_totals(usage))
                for model, usage in record.usage_by_model.items()
            ),
        }
        for name, record in outcome.subagents.items()
    }
    measured = {
        "final": outcome.final_answer,
        "stop": outcome.stopped_reason,
        "messages": [message(item) for item in outcome.leader_messages],
        "pool": pool,
        "lifecycle": lifecycle_summary,
        "usage": sorted(
            (
                sorted(
                    (model, usage_totals(usage))
                    for model, usage in totals.items()
                )
                for totals in outcome.usage_by_agent.values()
            ),
            key=lambda item: json.dumps(item, sort_keys=True),
        ),
    }
    normalized = json.loads(json.dumps(measured, sort_keys=True))
    default_max_depth = outcome.subagents["worker"].runs[0].spec.max_depth
    return normalized, default_max_depth


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


class _FakeHttpResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@check("leader.compatibility_names_removed")
def check_compatibility_names_removed() -> None:
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


@check("leader.run_dispatches_subagent")
def check_run_dispatches_subagent() -> None:
    with workspace() as ws:
        root = ws.root
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


@check("leader.compaction_identity")
def check_compaction_identity() -> None:
    with workspace() as ws:
        root = ws.root
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


@check("leader.cancellation_transcript")
def check_cancellation_transcript() -> None:
    with workspace() as ws:
        root = ws.root
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
            "symphonai_api.leader.standard_tool_registry",
            return_value={cancelling_tool.name: cancelling_tool},
        ):
            cancellation_result = cancellation_leader.chat(
                "delegate cancellable work", cancel=cancellation_token
            )
        if cancellation_result.stopped_reason != "final_response":
            fail(f"leader did not continue after child cancellation: {cancellation_result!r}")
        expected_cancellation_events = [
            ("started", "leader", None),
            ("spawned", "cancellable", None),
            ("started", "cancellable", None),
            ("finished", "cancellable", "cancelled"),
            ("finished", "leader", "final_response"),
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
            or "explicitly" not in (leader_tool_results[0].error or "")
        ):
            fail(f"child cancellation did not return a distinct tool error: {leader_tool_results!r}")
        cancelled_record = cancellation_result.subagents["cancellable"]
        if not any(message.role == Role.ASSISTANT for message in cancelled_record.messages):
            fail(f"cancelled subagent lost its partial conversation: {cancelled_record.messages!r}")
        if not any(
            message.tool_result is not None and message.tool_result.cancelled
            for message in cancelled_record.messages
        ):
            fail(f"cancelled subagent transcript was not repaired: {cancelled_record.messages!r}")

        _assert_openai_tool_calls_answered(
            cancellation_leader_provider.requests[-1],
            "leader turn after subagent cancellation",
        )


@check("leader.chat_cancellation")
def check_chat_cancellation() -> None:
    with workspace() as ws:
        root = ws.root
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


@check("leader.standalone_dispatch")
def check_standalone_dispatch() -> None:
    with workspace() as ws:
        root = ws.root
        # -- create / reuse / isolate / max_subagents, exercised directly on the tool --
        policy = ws.policy
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
        with mock.patch("symphonai_api.leader.emit") as standalone_emit:
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


@check("leader.dispatch_metadata")
def check_dispatch_metadata() -> None:
    with workspace() as ws:
        root = ws.root
        policy = ws.policy
        pool_provider = FakeModelProvider(
            responses=[ModelResponse(message=Message(role=Role.ASSISTANT, content="sub reply"))]
        )
        tool = DispatchSubagentTool(
            subagent_provider=pool_provider, leader_policy=policy, max_subagents=2, subagent_max_turns=3
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


@check("leader.dispatch_pool")
def check_dispatch_pool() -> None:
    with workspace() as ws:
        root = ws.root
        policy = ws.policy
        pool_provider = FakeModelProvider(
            responses=[ModelResponse(message=Message(role=Role.ASSISTANT, content="sub reply"))]
        )
        tool = DispatchSubagentTool(
            subagent_provider=pool_provider, leader_policy=policy, max_subagents=2, subagent_max_turns=3
        )
        r1 = tool.execute(
            ToolCall(id="c1", name="dispatch_subagent", arguments={"subagent_name": "worker", "task": "task one"}),
            policy,
        )
        if not r1.ok:
            fail(f"expected first dispatch to succeed: {r1.error}")
        worker_record = tool.pool.get("worker")
        if worker_record is None:
            fail("expected a 'worker' entry in the pool after first dispatch")

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

        r3 = tool.execute(
            ToolCall(id="c3", name="dispatch_subagent", arguments={"subagent_name": "helper", "task": "task three"}),
            policy,
        )
        if not r3.ok or tool.pool.get("helper") is worker_record:
            fail("expected a different name to create an isolated second subagent")

        r4 = tool.execute(
            ToolCall(id="c4", name="dispatch_subagent", arguments={"subagent_name": "third", "task": "x"}), policy
        )
        if r4.ok or len(tool.pool) != 2:
            fail("expected max_subagents to be enforced once the limit is reached")


@check("leader.typed_event_lifecycle")
def check_typed_event_lifecycle() -> None:
    with workspace() as ws:
        root = ws.root
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


@check("leader.event_sink_isolation")
def check_event_sink_isolation() -> None:
    with workspace() as ws:
        root = ws.root
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


@check("leader.leader_failure_events")
def check_leader_failure_events() -> None:
    with workspace() as ws:
        root = ws.root
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


@check("leader.subagent_failure_events")
def check_subagent_failure_events() -> None:
    with workspace() as ws:
        root = ws.root
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


@check("leader.leader_max_turns")
def check_leader_max_turns() -> None:
    with workspace() as ws:
        root = ws.root
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


@check("leader.configured_budget_overrides_turn_limit")
def check_configured_budget_overrides_turn_limit() -> None:
    with workspace() as ws:
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                id="invalid", name="dispatch_subagent", arguments={},
            )])),
        ])
        budget = RunBudget(max_turns=1, max_total_tokens=100)
        leader = Leader(LeaderConfig(
            provider, FakeModelProvider(), str(ws.root),
            max_leader_turns=4, leader_budget=budget,
        ))
        result = leader.run("bounded")
        if leader._agent._budget is not budget or leader._leader_spec.budget is not budget:
            fail("leader budget did not reach the running agent and run spec")
        if result.stopped_reason != "max_turns" or provider.call_count != 1:
            fail(f"budget max_turns did not override the launch turn limit: {result.stopped_reason!r}")


@check("leader.subagent_max_turns")
def check_subagent_max_turns() -> None:
    with workspace() as ws:
        root = ws.root
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


@check("leader.fresh_run_subagents")
def check_fresh_run_subagents() -> None:
    with workspace() as ws:
        root = ws.root
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


@check("leader.pool_reset")
def check_pool_reset() -> None:
    with workspace() as ws:
        root = ws.root
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
        pool_reset_leader.run("second run")

        cleared_count = pool_reset_leader.clear_subagents()
        if cleared_count != 1 or pool_reset_leader.subagents:
            fail(
                f"clear_subagents() should clear one pooled agent, got count={cleared_count} "
                f"and pool={pool_reset_leader.subagents!r}"
            )
        if pool_reset_leader.clear_subagents() != 0:
            fail("clear_subagents() should return zero for an already-empty pool")

        pool_reset_leader.run("third run")
        if len(pool_reset_leader.subagents) != 1:
            fail("expected third run to repopulate one subagent")
        pool_reset_leader.clear_chat()
        if pool_reset_leader.subagents:
            fail("clear_chat() did not clear the subagent pool")


@check("leader.chat_history")
def check_chat_history() -> None:
    with workspace() as ws:
        root = ws.root
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


@check("leader.selection_updates_next_request")
def check_selection_updates_next_request() -> None:
    with workspace() as ws:
        provider = _RecordingFakeProvider([
            ModelResponse(message=Message(Role.ASSISTANT, "first")),
            ModelResponse(message=Message(Role.ASSISTANT, "second")),
        ])
        leader = Leader(LeaderConfig(
            leader_provider=provider,
            subagent_provider=FakeModelProvider(),
            repo_root=str(ws.root),
            leader_model="original-model",
        ))
        agent = leader._agent
        leader.chat("first turn")
        leader.select_model("next-model", "high")
        leader.chat("second turn")

        if len(provider.requests) != 2:
            fail(f"selection made {len(provider.requests)} provider requests")
        next_request = provider.requests[1]
        if (next_request.model, next_request.effort) != ("next-model", "high"):
            fail(
                "model and effort selection did not reach the next provider request: "
                f"{next_request.model!r}, {next_request.effort!r}"
            )
        if leader._agent is not agent:
            fail("model and effort selection rebuilt the leader agent")


def _large_file_history() -> list[Message]:
    messages = [Message(Role.SYSTEM, "system prompt")]
    for index in range(7):
        call_id = f"read-{index}"
        # Keep the approximate total and preserved tail described by 28b.
        messages.extend(
            [
                Message(Role.USER, f"prompt {index} " + "x" * 240),
                Message(
                    Role.ASSISTANT,
                    tool_calls=[ToolCall(call_id, "read_file", {"path": f"{index}.py"})],
                ),
                Message(
                    Role.TOOL,
                    tool_result=ToolResult(
                        tool_call_id=call_id,
                        ok=True,
                        content="line of code\n" * 1200,
                    ),
                ),
            ]
        )
    return messages


@check("leader.model_window_sets_compaction_budget")
def check_model_window_sets_compaction_budget() -> None:
    with workspace() as ws:
        default_events = CollectingSink()
        default = Leader(
            LeaderConfig(
                _AnthropicFakeProvider([ModelResponse(Message(Role.ASSISTANT, "reply"))]),
                FakeModelProvider(),
                str(ws.root),
                leader_model="claude-opus-4-8",
                # Include four completed file results plus the new chat prompt.
                chat_recent_turns=5,
                events=default_events,
            )
        )
        default.seed_chat(_large_file_history(), persisted=True)
        default.chat("one more prompt " + "x" * 240)
        if default_events.of_type(CompactionApplied):
            fail("model-window budget compacted the large but in-window history")
        if default._automatic_compaction_breaker.consecutive_failures != 0:
            fail("model-window budget recorded an automatic compaction failure")

        small = Leader(
            LeaderConfig(
                _AnthropicFakeProvider([ModelResponse(Message(Role.ASSISTANT, "reply"))]),
                FakeModelProvider(),
                str(ws.root),
                leader_model="claude-opus-4-8",
                chat_token_budget=16_000,
                chat_recent_turns=5,
            )
        )
        small.seed_chat(_large_file_history(), persisted=True)
        small.chat("one more prompt " + "x" * 240)
        if small._automatic_compaction_breaker.consecutive_failures < 1:
            fail("explicit 16,000-token budget did not record the expected compaction failure")


@check("leader.model_window_budget_tracks_selection")
def check_model_window_budget_tracks_selection() -> None:
    with workspace() as ws:
        leader = Leader(
            LeaderConfig(
                _AnthropicFakeProvider([ModelResponse(Message(Role.ASSISTANT, "reply"))]),
                FakeModelProvider(),
                str(ws.root),
                leader_model="claude-opus-4-8",
            )
        )
        agent = leader._agent
        if leader.chat_token_budget != 955_000:
            fail(f"Opus context budget was {leader.chat_token_budget}, expected 955000")
        leader.select_model("claude-haiku-4-5", None)
        if leader.chat_token_budget != 155_000:
            fail(f"selected Haiku context budget was {leader.chat_token_budget}, expected 155000")
        if leader._agent is not agent:
            fail("changing the model to update the budget rebuilt the leader")


@check("leader.context_overflow_compacts_and_recovers")
def check_context_overflow_compacts_and_recovers() -> None:
    with workspace() as ws:
        provider = _ContextOverflowProvider([True, False, False])
        leader = Leader(LeaderConfig(
            provider,
            FakeModelProvider(),
            str(ws.root),
            chat_token_budget=10_000,
            chat_recent_turns=1,
        ))
        leader.seed_chat([
            Message(Role.USER, "first goal must stay"),
            Message(Role.ASSISTANT, "old answer " * 80),
            Message(Role.USER, "old follow-up " * 80),
            Message(Role.ASSISTANT, "old analysis " * 80),
        ])

        recovered = leader.chat("current request")
        if recovered.final_answer != "recovered" or len(provider.requests) != 2:
            fail("context overflow did not compact and retry exactly once")
        if len(provider.requests[1].messages) >= len(provider.requests[0].messages):
            fail("overflow retry did not send fewer messages than the rejected request")
        if not any(
            "Earlier conversation compacted" in message.text
            for message in provider.requests[1].messages
        ):
            fail("overflow retry did not carry the compacted conversation summary")

        following = leader.chat("following request")
        if following.final_answer != "recovered" or len(provider.requests) != 3:
            fail("conversation was not usable after overflow recovery")
        following_text = [message.text for message in provider.requests[2].messages]
        if "current request" not in following_text or "following request" not in following_text:
            fail(f"following request did not use compacted history: {following_text!r}")


def _vendor_overflow_history() -> list[Message]:
    messages = [Message(Role.SYSTEM, "system instructions")]
    for index in range(8):
        call_id = f"large-read-{index}"
        messages.extend(
            [
                Message(Role.USER, f"inspect file {index}"),
                Message(
                    Role.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            call_id,
                            "read_file",
                            {"path": f"source-{index}.py"},
                        )
                    ],
                ),
                Message(
                    Role.TOOL,
                    tool_result=ToolResult(
                        tool_call_id=call_id,
                        ok=True,
                        content="x" * 72_000,
                    ),
                ),
            ]
        )
    return messages


@check("leader.vendor_counts_recover_and_adjust_budget")
def check_vendor_counts_recover_and_adjust_budget() -> None:
    with workspace() as ws:
        provider = _WindowCountingProvider(window=200_000)
        leader = Leader(
            LeaderConfig(
                provider,
                FakeModelProvider(),
                str(ws.root),
                leader_model="claude-haiku-4-5",
            )
        )
        leader.seed_chat(_vendor_overflow_history(), persisted=True)
        initial_estimate = estimate_messages_tokens(leader._chat_messages)
        if not 140_000 <= initial_estimate <= 150_000:
            fail(f"seeded history estimate was {initial_estimate}, expected about 145000")

        first = leader.chat("summarize the inspected files")
        if first.final_answer != "recovered" or len(provider.estimates) != 2:
            fail(f"counted overflow did not recover in exactly one retry: {provider.estimates!r}")
        if provider.actual_counts[0] <= provider.window or provider.actual_counts[1] > provider.window:
            fail(f"overflow and retry did not straddle the fake window: {provider.actual_counts!r}")
        if leader._automatic_compaction_breaker.consecutive_failures != 0:
            fail("successful overflow recovery left a compaction breaker failure")

        ratio = provider.actual_counts[0] / provider.estimates[0]
        if leader._token_ratio != ratio:
            fail(f"leader token ratio was {leader._token_ratio}, expected {ratio}")
        expected_budget = math.floor(155_000 / ratio)
        if leader.chat_token_budget != expected_budget:
            fail(f"adjusted budget was {leader.chat_token_budget}, expected {expected_budget}")

        second = leader.chat("one more question")
        if second.final_answer != "recovered" or len(provider.estimates) != 3:
            fail(f"later prompt overflowed or retried: {provider.estimates!r}")
        leader.select_model("claude-haiku-4-5", None)
        if leader.chat_token_budget != 155_000 or leader._token_ratio != 1.0:
            fail("select_model did not reset the learned model-specific ratio")


@check("leader.mid_turn_overflow_uses_failed_request_ratio")
def check_mid_turn_overflow_uses_failed_request_ratio() -> None:
    with workspace() as ws:
        for index in range(12):
            (ws.root / f"growth-{index}.txt").write_text(("x" * 400 + "\n") * 200)
        provider = _GrowingRequestProvider()
        leader = Leader(
            LeaderConfig(
                provider,
                FakeModelProvider(),
                str(ws.root),
                leader_model="claude-haiku-4-5",
            )
        )
        history = [Message(Role.SYSTEM, "system instructions")]
        for index in range(4):
            call_id = f"history-read-{index}"
            history.extend(
                [
                    Message(Role.USER, f"inspect history file {index}"),
                    Message(
                        Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                call_id,
                                "read_file",
                                {"path": f"history-{index}.txt"},
                            )
                        ],
                    ),
                    Message(
                        Role.TOOL,
                        tool_result=ToolResult(
                            tool_call_id=call_id,
                            ok=True,
                            content="h" * 24_000,
                        ),
                    ),
                ]
            )
        leader.seed_chat(history, persisted=True)
        initial_estimate = estimate_messages_tokens(leader._chat_messages)
        if not 23_000 <= initial_estimate <= 26_000:
            fail(f"seeded history estimate was {initial_estimate}, expected about 24164")
        first = leader.chat("read the requested files")
        if first.final_answer != "recovered" or not provider.overflowed:
            fail("mid-turn overflow did not recover")
        if len(provider.estimates) < 3:
            fail(f"provider did not grow the request through tool reads: {provider.estimates!r}")
        failed_estimate = provider.estimates[-2]
        failed_actual = provider.actual_counts[-2]
        if failed_actual <= provider.window:
            fail(f"last pre-recovery request did not overflow: {provider.actual_counts!r}")
        if not math.isclose(leader._token_ratio, 1.2, abs_tol=0.01):
            fail(f"mid-turn token ratio was {leader._token_ratio}, expected about 1.2")
        expected_budget = math.floor(155_000 / 1.2)
        if abs(leader.chat_token_budget - expected_budget) > 1:
            fail(
                f"mid-turn budget was {leader.chat_token_budget}, expected about {expected_budget}"
            )
        if abs(failed_actual / failed_estimate - 1.2) > 0.01:
            fail("failed request's fake vendor count did not match its estimate ratio")


@check("leader.uncounted_overflow_uses_three_quarters")
def check_uncounted_overflow_uses_three_quarters() -> None:
    with workspace() as ws:
        leader = Leader(
            LeaderConfig(
                _AnthropicFakeProvider([]),
                FakeModelProvider(),
                str(ws.root),
                leader_model="claude-haiku-4-5",
            )
        )
        leader.seed_chat(
            [
                Message(Role.USER, "goal"),
                Message(Role.ASSISTANT, "old context " * 500),
            ],
            persisted=True,
        )
        before_tokens = estimate_messages_tokens(leader._chat_messages)
        observed_budgets: list[int] = []

        def refuse_compaction(budget: int, **kwargs) -> None:
            observed_budgets.append(budget)
            raise ContextCompactionError("fixture refuses compaction")

        with mock.patch.object(
            leader,
            "_compact_chat_to_budget",
            side_effect=refuse_compaction,
        ):
            try:
                leader._compact_after_context_overflow(
                    ContextLengthExceededError("unknown vendor count")
                )
            except ContextCompactionError:
                pass
            else:
                fail("uncompacted context did not propagate the fixture failure")
        expected = min(leader.chat_token_budget, max(1, before_tokens * 3 // 4))
        if observed_budgets != [expected] or expected > before_tokens * 3 // 4:
            fail(f"uncounted overflow budget was {observed_budgets!r}, expected {expected}")


@check("leader.context_overflow_narrows_recent_window")
def check_context_overflow_narrows_recent_window() -> None:
    with workspace() as ws:
        provider = _ContextOverflowProvider([True, False])
        leader = Leader(LeaderConfig(
            provider,
            FakeModelProvider(),
            str(ws.root),
            chat_token_budget=10_000,
            chat_recent_turns=3,
        ))
        leader.seed_chat([
            Message(Role.USER, "first goal must stay"),
            Message(Role.ASSISTANT, "first large answer " * 200),
            Message(Role.ASSISTANT, "second large answer " * 200),
            Message(Role.USER, "recent request must stay"),
            Message(Role.ASSISTANT, "recent answer must stay"),
        ])

        result = leader.chat("current request must stay")
        if result.final_answer != "recovered" or len(provider.requests) != 2:
            fail("narrowing the recent window did not recover the overflow")
        first_count = len(provider.requests[0].messages)
        second_messages = provider.requests[1].messages
        if len(second_messages) >= first_count:
            fail("narrowed overflow retry did not contain fewer messages")
        second_text = [message.text for message in second_messages]
        for preserved in (
            "first goal must stay",
            "recent request must stay",
            "recent answer must stay",
            "current request must stay",
        ):
            if preserved not in second_text:
                fail(f"narrowed overflow retry lost required context: {preserved!r}")


@check("leader.context_overflow_one_turn_failure")
def check_context_overflow_one_turn_failure() -> None:
    with workspace() as ws:
        provider = _ContextOverflowProvider([True])
        leader = Leader(LeaderConfig(
            provider,
            FakeModelProvider(),
            str(ws.root),
            chat_token_budget=10_000,
            chat_recent_turns=4,
        ))
        try:
            leader.chat("x" * 4_000)
        except ContextLengthExceededError as exc:
            if exc is not provider.overflow_error:
                fail("one-turn compaction did not propagate the original overflow")
        else:
            fail("one-turn overflow unexpectedly recovered")
        if len(provider.requests) != 1:
            fail("one-turn overflow retried without a compactable message")
        if leader._automatic_compaction_breaker.consecutive_failures != 1:
            fail("one-turn overflow did not record a compaction failure")


@check("leader.budget_compaction_keeps_recent_window")
def check_budget_compaction_keeps_recent_window() -> None:
    with workspace() as ws:
        leader = Leader(LeaderConfig(
            FakeModelProvider(),
            FakeModelProvider(),
            str(ws.root),
            chat_token_budget=100,
            chat_recent_turns=2,
        ))
        recent_window = [
            Message(Role.USER, "recent request one"),
            Message(Role.ASSISTANT, "recent answer one"),
            Message(Role.USER, "recent request two"),
            Message(Role.ASSISTANT, "recent answer two"),
        ]
        leader.seed_chat([
            Message(Role.USER, "first goal"),
            Message(Role.ASSISTANT, "old answer " * 400),
            *recent_window,
        ])

        compacted = leader.compact_chat()
        if not compacted.changed or compacted.recent_turns != 2:
            fail(f"ordinary compaction did not use the configured window: {compacted!r}")
        if compacted.messages[-len(recent_window):] != recent_window:
            fail("ordinary compaction changed the configured recent window")


@check("leader.compaction_strips_anthropic_thinking_only_when_changed")
def check_compaction_strips_anthropic_thinking_only_when_changed() -> None:
    def history() -> list[Message]:
        return [
            Message(Role.USER, "first goal"),
            Message(Role.ASSISTANT, "old context " * 300),
            Message(Role.USER, "recent request"),
            Message(
                Role.ASSISTANT,
                "tool request",
                tool_calls=[ToolCall(
                    "read-call",
                    "read_file",
                    {"path": "notes.txt"},
                    provider_metadata={
                        "anthropic_content": [
                            {"type": "thinking", "thinking": "", "signature": "S1"},
                            {"type": "tool_use", "id": "read-call", "name": "read_file", "input": {"path": "notes.txt"}},
                        ],
                        "thoughtSignature": "gemini-thought",
                    },
                )],
            ),
        ]

    with workspace() as ws:
        unchanged = Leader(LeaderConfig(
            FakeModelProvider(), FakeModelProvider(), str(ws.root),
            chat_token_budget=10_000, chat_recent_turns=1,
        ))
        unchanged.seed_chat(history())
        unchanged_result = unchanged.compact_chat()
        if unchanged_result.changed:
            fail("under-budget compaction unexpectedly changed history")
        untouched_call = unchanged._chat_messages[-1].tool_calls[0]
        if "anthropic_content" not in untouched_call.provider_metadata or (
            untouched_call.provider_metadata.get("thoughtSignature") != "gemini-thought"
        ):
            fail(f"unchanged compaction stripped vendor state: {untouched_call!r}")

        compacted = Leader(LeaderConfig(
            FakeModelProvider(), FakeModelProvider(), str(ws.root),
            chat_token_budget=100, chat_recent_turns=1,
        ))
        compacted.seed_chat(history())
        changed_result = compacted.compact_chat()
        if not changed_result.changed:
            fail("large seeded history did not compact")
        calls = [call for message in compacted._chat_messages for call in message.tool_calls]
        if len(calls) != 1:
            fail(f"compaction lost the retained tool call: {compacted._chat_messages!r}")
        if "anthropic_content" in calls[0].provider_metadata:
            fail(f"changed compaction retained stale Anthropic thinking: {calls[0]!r}")
        if calls[0].provider_metadata != {"thoughtSignature": "gemini-thought"}:
            fail(f"changed compaction removed unrelated provider metadata: {calls[0]!r}")


@check("leader.model_summary_request_and_usage")
def check_model_summary_request_and_usage() -> None:
    class RecordingProvider(FakeModelProvider):
        def __init__(self) -> None:
            super().__init__([
                ModelResponse(
                    Message(Role.ASSISTANT, "compressed history"),
                    usage=Usage(input_tokens=100, output_tokens=50),
                ),
                ModelResponse(Message(Role.ASSISTANT, "done")),
            ])
            self.requests: list[ModelRequest] = []

        def create_response(self, request, *, cancel=None) -> ModelResponse:
            self.requests.append(request)
            return super().create_response(request, cancel=cancel)

    with workspace() as ws:
        provider = RecordingProvider()
        provider.model = "leader-summary-model"
        leader = Leader(LeaderConfig(
            provider,
            FakeModelProvider(),
            str(ws.root),
            leader_model="leader-summary-model",
            chat_token_budget=150,
            chat_recent_turns=1,
            model_summary=True,
        ))
        leader.seed_chat([
            Message(Role.USER, "first request"),
            Message(Role.ASSISTANT, "dropped decision and file details " * 100),
            Message(Role.USER, "previous turn"),
        ])
        result = leader.chat("current request")
        if len(provider.requests) != 2:
            fail(f"model compaction did not add exactly one provider call: {len(provider.requests)}")
        summary_request = provider.requests[0]
        if (
            summary_request.call_class is not CallClass.BACKGROUND
            or summary_request.max_tokens != 20_000
            or summary_request.tools
        ):
            fail(f"summary request had incorrect call settings: {summary_request!r}")
        if len(summary_request.messages) != 2 or summary_request.messages[0].role is not Role.SYSTEM:
            fail(f"summary request did not contain the system prompt and transcript: {summary_request.messages!r}")
        if "every explicit request" not in summary_request.messages[0].text:
            fail("summary request omitted the required summary prompt")
        transcript = summary_request.messages[1]
        if transcript.role is not Role.USER or "dropped decision and file details" not in transcript.text:
            fail(f"summary request did not include dropped conversation text: {transcript!r}")
        usage = result.usage_by_agent.get(leader._agent_ref.agent_id, {}).get("leader-summary-model")
        if usage is None or (usage.input_tokens, usage.output_tokens, usage.calls) != (100, 50, 2):
            fail(f"summary usage was not included under the leader: {result.usage_by_agent!r}")

        default_provider = RecordingProvider()
        default_leader = Leader(LeaderConfig(
            default_provider,
            FakeModelProvider(),
            str(ws.root),
            chat_token_budget=150,
            chat_recent_turns=1,
        ))
        default_leader.seed_chat([
            Message(Role.USER, "first request"),
            Message(Role.ASSISTANT, "dropped decision and file details " * 100),
            Message(Role.USER, "previous turn"),
        ])
        default_leader.chat("current request")
        if len(default_provider.requests) != 1:
            fail(f"default leader unexpectedly made a model summary call: {len(default_provider.requests)}")

        post_provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "final answer " * 15)),
            ModelResponse(
                Message(Role.ASSISTANT, "shortened history"),
                usage=Usage(input_tokens=100, output_tokens=50),
            ),
        ])
        post_provider.model = "leader-summary-model"
        post_leader = Leader(LeaderConfig(
            post_provider,
            FakeModelProvider(),
            str(ws.root),
            leader_model="leader-summary-model",
            chat_token_budget=150,
            chat_recent_turns=1,
            model_summary=True,
        ))
        post_leader.seed_chat([
            Message(Role.USER, "first request"),
            Message(Role.ASSISTANT, "old context " * 30),
            Message(Role.USER, "previous turn"),
        ])
        post_result = post_leader.chat("current request")
        post_usage = post_result.usage_by_agent.get(post_leader._agent_ref.agent_id, {}).get(
            "leader-summary-model"
        )
        if post_provider.call_count != 2 or post_usage is None or (
            post_usage.input_tokens, post_usage.output_tokens
        ) != (100, 50):
            fail(f"post-run compaction summary usage was not returned: {post_result.usage_by_agent!r}")


@check("leader.forced_compaction_instructions_and_usage")
def check_forced_compaction_instructions_and_usage() -> None:
    class RecordingProvider(FakeModelProvider):
        def __init__(self) -> None:
            super().__init__([
                ModelResponse(
                    Message(Role.ASSISTANT, "short summary"),
                    usage=Usage(input_tokens=100, output_tokens=50),
                ),
                ModelResponse(Message(Role.ASSISTANT, "next answer")),
            ])
            self.requests: list[ModelRequest] = []

        def create_response(self, request, *, cancel=None):
            self.requests.append(request)
            return super().create_response(request, cancel=cancel)

    with workspace() as ws:
        provider = RecordingProvider()
        provider.model = "leader-summary-model"
        events = CollectingSink()
        leader = Leader(LeaderConfig(
            provider,
            FakeModelProvider(),
            str(ws.root),
            leader_model="leader-summary-model",
            chat_token_budget=10_000,
            chat_recent_turns=4,
            model_summary=True,
            events=events,
        ))
        leader.seed_chat([
            Message(Role.SYSTEM, "system prompt"),
            Message(Role.USER, "first goal"),
            Message(Role.ASSISTANT, "earlier details " * 40),
            Message(Role.USER, "middle request"),
            Message(Role.ASSISTANT, "more earlier details " * 40),
            Message(Role.USER, "latest request"),
            Message(Role.ASSISTANT, "latest answer"),
        ])
        ordinary = leader.compact_chat()
        if ordinary.changed:
            fail(f"ordinary compact_chat changed an under-budget conversation: {ordinary!r}")
        compacted, usage = leader.force_compact_chat("keep the API names")
        if not compacted.changed or len(provider.requests) != 1:
            fail(f"forced compaction did not summarize the under-budget history: {compacted!r}")
        if not provider.requests[0].messages[0].text.endswith(
            "Additional instructions from the user:\nkeep the API names"
        ):
            fail(f"summary request omitted the supplied instructions: {provider.requests[0]!r}")
        if usage.get("leader-summary-model") != UsageTotals(
            input_tokens=100, output_tokens=50, calls=1
        ):
            fail(f"forced summary usage was not returned: {usage!r}")
        if len(events.of_type(CompactionApplied)) != 1:
            fail(f"forced compaction skipped the normal compaction event: {events.events!r}")

        leader.chat("next prompt")
        if usage.get("leader-summary-model") != UsageTotals(
            input_tokens=100, output_tokens=50, calls=1
        ):
            fail(f"the next chat mutated or cleared returned summary usage: {usage!r}")


@check("leader.context_overflow_retry_failure")
def check_context_overflow_retry_failure() -> None:
    with workspace() as ws:
        provider = _ContextOverflowProvider([True, True, True, True, True])
        leader = Leader(LeaderConfig(
            provider,
            FakeModelProvider(),
            str(ws.root),
            chat_token_budget=10_000,
            chat_recent_turns=1,
            max_consecutive_compaction_failures=2,
        ))
        leader.seed_chat([
            Message(Role.USER, "first goal must stay"),
            Message(Role.ASSISTANT, "old answer " * 80),
            Message(Role.USER, "old follow-up " * 80),
            Message(Role.ASSISTANT, "old analysis " * 80),
        ])

        try:
            leader.chat("current request")
        except ContextLengthExceededError:
            pass
        else:
            fail("second context overflow did not propagate")
        if len(provider.requests) != 2:
            fail(f"overflow recovery looped instead of retrying once: {len(provider.requests)} requests")
        breaker = leader._automatic_compaction_breaker
        if breaker.consecutive_failures != 1 or breaker.is_open:
            fail("second overflow did not record a failed automatic repair")

        try:
            leader.chat("still too large")
        except ContextLengthExceededError:
            pass
        else:
            fail("another second overflow did not propagate")
        if len(provider.requests) != 3 or not breaker.is_open:
            fail("consecutive overflow repairs did not open the breaker")

        try:
            leader.chat("breaker is open")
        except ContextLengthExceededError:
            pass
        else:
            fail("open compaction breaker hid the next overflow")
        if len(provider.requests) != 4:
            fail("open compaction breaker allowed another compact-and-retry attempt")


@check("leader.standard_tools")
def check_leader_standard_tools() -> None:
    with workspace() as ws:
        provider = _RecordingFakeProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[
                ToolCall("read", "read_file", {"path": "existing.txt"}),
                ToolCall("denied", "read_file", {"path": str(ws.outside / "outside.txt")}),
            ])),
            ModelResponse(Message(Role.ASSISTANT, "finished")),
        ])
        leader = Leader(LeaderConfig(provider, FakeModelProvider(), str(ws.root)))
        result = leader.run("read existing.txt")
        tool_results = {
            message.tool_result.tool_call_id: message.tool_result
            for message in result.leader_messages
            if message.tool_result is not None
        }
        if result.final_answer != "finished" or result.subagents:
            fail("leader did not finish the direct tool run without subagents")
        if not tool_results.get("read") or not tool_results["read"].ok or "hello from disk" not in tool_results["read"].content:
            fail(f"leader did not read the file with its standard tool: {tool_results!r}")
        if not tool_results.get("denied") or tool_results["denied"].ok:
            fail(f"leader tool escaped its workspace policy: {tool_results!r}")
        if "dispatch_subagent" not in leader._agent._tools or "read_file" not in leader._agent._tools:
            fail("leader execution registry omitted dispatch or a standard tool")
        sent_tools = provider.requests[0].tools
        if [schema.get("name") for schema in sent_tools] != list(leader._agent._tools):
            fail(f"leader model tool list did not match its execution registry: {sent_tools!r}")


@check("leader.streaming")
def check_leader_streaming() -> None:
    with workspace() as ws:
        sink = CollectingSink()
        leader_provider = FakeModelProvider(streams=[
            [StreamCompleted(ModelResponse(Message(Role.ASSISTANT, tool_calls=[
                _dispatch("worker", "answer", "stream-dispatch")
            ])))],
            [TextDelta("lead"), TextDelta("er"), StreamCompleted(ModelResponse(Message(Role.ASSISTANT, "")))],
        ])
        child_provider = FakeModelProvider(streams=[
            [TextDelta("chi"), TextDelta("ld"), StreamCompleted(ModelResponse(Message(Role.ASSISTANT, "")))],
        ])
        leader = Leader(LeaderConfig(leader_provider, child_provider, str(ws.root), events=sink, stream=True))
        result = leader.run("delegate")
        child = result.subagents.get("worker")
        if child is None or "dispatch_subagent" in child.agent._tools:
            fail("spawned subagent was missing or could delegate recursively")
        deltas = sink.of_type(AssistantTextDelta)
        leader_text = "".join(event.text for event in deltas if event.agent_id == result.agent.agent_id)
        child_text = "".join(event.text for event in deltas if event.agent_id == child.agent_ref.agent_id)
        if result.final_answer != "leader" or leader_text != "leader":
            fail(f"leader streamed text did not match its reply: {leader_text!r}")
        if child_text != "child" or child.messages[-1].text != "child":
            fail(f"subagent streamed text did not match its reply: {child_text!r}")
        if LeaderConfig(FakeModelProvider(), FakeModelProvider(), str(ws.root)).stream is not False:
            fail("leader streaming default changed")


@check("leader.seeded_chat")
def check_leader_seeded_chat() -> None:
    with workspace() as ws:
        provider = _RecordingFakeProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[_dispatch("worker", "start")])),
            ModelResponse(Message(Role.ASSISTANT, "first run")),
            ModelResponse(Message(Role.ASSISTANT, "third reply")),
        ])
        sink = CollectingSink()
        leader = Leader(LeaderConfig(
            provider,
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "child"))]),
            str(ws.root),
            events=sink,
        ))
        leader.run("start")
        existing_child = leader.subagents["worker"]
        event_count = len(sink.events)
        call_count = provider.call_count
        seeded = [Message(Role.USER, "first"), Message(Role.ASSISTANT, "second")]
        leader.seed_chat(seeded)
        if len(sink.events) != event_count or provider.call_count != call_count:
            fail("seeding chat started a run")
        if leader.subagents.get("worker") is not existing_child:
            fail("seeding chat changed the subagent pool")
        leader.chat("third")
        sent = [(message.role, message.text) for message in provider.requests[-1].messages]
        if sent != [(Role.USER, "first"), (Role.ASSISTANT, "second"), (Role.USER, "third")]:
            fail(f"seeded history did not reach the next model request: {sent!r}")


@check("leader.host_tools_and_compaction")
def check_host_tools_and_compaction() -> None:
    class ExtraTool(_CancellingSubagentTool):
        @property
        def name(self) -> str:
            return "mcp__test__lookup"

    with workspace() as ws:
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[_dispatch("worker", "inspect")])),
            ModelResponse(Message(Role.ASSISTANT, "child")),
            ModelResponse(Message(Role.ASSISTANT, "leader")),
        ])
        result_store = ToolResultStore(directory=ws.root / "results")
        leader = Leader(LeaderConfig(
            provider, provider, str(ws.root),
            result_store=result_store,
            extra_tools={"mcp__test__lookup": ExtraTool()},
            subagent_specs=builtin_subagent_specs(provider, ws.policy),
        ))
        child = leader.run("delegate").subagents["worker"].agent
        for agent in (leader._agent, child):
            if not {"read_tool_result", "mcp__test__lookup"} <= agent._tools.keys():
                fail(f"host tools did not reach both agent registries: {agent._tools!r}")
            if agent._result_store is not result_store:
                fail("host result store was not passed to both agents")

        events = CollectingSink()
        compact_provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "reply " + "y" * 40))])
        compacting = Leader(LeaderConfig(
            compact_provider, FakeModelProvider(), str(ws.root),
            chat_token_budget=150, chat_recent_turns=1, events=events,
        ))
        for index in range(6):
            result = compacting.chat(f"prompt {index} " + "x" * 80)
        if result.final_answer != "reply " + "y" * 40 or compact_provider.call_count != 6:
            fail("long conversation did not continue through compaction")
        if not events.of_type(CompactionApplied) or len(compacting._chat_messages) >= 12:
            fail("long conversation did not shrink after exceeding its token budget")


@check("leader.anthropic_leader_tool_schema")
def check_anthropic_leader_tool_schema() -> None:
    with workspace() as ws:
        root = ws.root
        previous_api_key = os.environ.get(API_KEY_ENV_VAR)
        try:
            os.environ[API_KEY_ENV_VAR] = 'sk-ant-fake-test-key-do-not-use'
            # -- regression: a real leader provider's outgoing request must
            # actually include the dispatch_subagent tool definition --
            captured: dict = {}
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
            sent_tools = captured.get("body", {}).get("tools")
            if not sent_tools or sent_tools[0].get("name") != "dispatch_subagent":
                fail(f"expected outgoing request to include the dispatch_subagent tool, got tools={sent_tools!r}")
        finally:
            if previous_api_key is None:
                os.environ.pop(API_KEY_ENV_VAR, None)
            else:
                os.environ[API_KEY_ENV_VAR] = previous_api_key


@check("leader.anthropic_subagent_tool_schemas")
def check_anthropic_subagent_tool_schemas() -> None:
    with workspace() as ws:
        root = ws.root
        previous_api_key = os.environ.get(API_KEY_ENV_VAR)
        try:
            os.environ[API_KEY_ENV_VAR] = 'sk-ant-fake-test-key-do-not-use'
            # -- regression: a real SUBAGENT's outgoing request must include
            # schemas for all eight standard tools, not just the leader's own tool --
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
            if not subagent_requests:
                fail("expected the subagent's own request to have been captured")
            sent_tool_names = {t.get("name") for t in subagent_requests[0].get("tools", [])}
            expected_tool_names = {
                "read_file",
                "write_file",
                "edit_file",
                "multi_edit_file",
                "list_files",
                "glob",
                "grep",
                "run_shell",
                "web_fetch",
            }
            if sent_tool_names != expected_tool_names:
                fail(f"expected subagent request to include {expected_tool_names}, got {sent_tool_names!r}")
        finally:
            if previous_api_key is None:
                os.environ.pop(API_KEY_ENV_VAR, None)
            else:
                os.environ[API_KEY_ENV_VAR] = previous_api_key


@check("leader.subagent_tool_subsets")
def check_subagent_tool_subsets() -> None:
    with workspace() as ws:
        root = ws.root
        previous_api_key = os.environ.get(API_KEY_ENV_VAR)
        try:
            os.environ[API_KEY_ENV_VAR] = 'sk-ant-fake-test-key-do-not-use'
            # -- a caller-fixed narrowed registry must narrow both execution and wire schemas --
            narrowed_subagent_requests: list[dict] = []
            narrowed_call_count = [0]

            def _fake_urlopen_narrowed_dispatch(request, timeout=None):  # noqa: ANN001
                narrowed_call_count[0] += 1
                body = json.loads(request.data.decode("utf-8"))
                if narrowed_call_count[0] == 1:
                    payload = {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "narrow-c1",
                                "name": "dispatch_subagent",
                                "input": {
                                    "subagent_name": "reader",
                                    "task": "inspect a.txt",
                                },
                            }
                        ],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                        "stop_reason": "tool_use",
                    }
                elif narrowed_call_count[0] == 2:
                    narrowed_subagent_requests.append(body)
                    payload = {
                        "content": [{"type": "text", "text": "done"}],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                        "stop_reason": "end_turn",
                    }
                else:
                    payload = {
                        "content": [{"type": "text", "text": "final answer"}],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                        "stop_reason": "end_turn",
                    }
                return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

            narrowed_leader = Leader(
                LeaderConfig(
                    leader_provider=AnthropicProvider(),
                    subagent_provider=AnthropicProvider(),
                    repo_root=str(root),
                    subagent_tool_names=["read_file", "glob", "grep"],
                )
            )
            with mock.patch(
                "urllib.request.urlopen", side_effect=_fake_urlopen_narrowed_dispatch
            ):
                narrowed_leader.run("delegate read-only inspection")
            if not narrowed_subagent_requests:
                fail("expected the narrowed subagent request to be captured")
            narrowed_tool_names = {
                tool.get("name")
                for tool in narrowed_subagent_requests[0].get("tools", [])
            }
            expected_narrowed_names = {"read_file", "glob", "grep"}
            if narrowed_tool_names != expected_narrowed_names or "run_shell" in narrowed_tool_names:
                fail(
                    "narrowed subagent schemas exceeded the caller-fixed set: "
                    f"expected={expected_narrowed_names!r}, actual={narrowed_tool_names!r}"
                )
        finally:
            if previous_api_key is None:
                os.environ.pop(API_KEY_ENV_VAR, None)
            else:
                os.environ[API_KEY_ENV_VAR] = previous_api_key


@check("leader.gemini_dispatch_schema")
def check_gemini_dispatch_schema() -> None:
    with workspace() as ws:
        root = ws.root
        previous_api_key = os.environ.get(GEMINI_API_KEY_ENV_VAR)
        try:
            os.environ[GEMINI_API_KEY_ENV_VAR] = 'AIza-fake-test-key-do-not-use'
            # -- regression: a real GeminiProvider leader must send
            # dispatch_subagent as a Gemini function declaration with sanitized,
            # non-empty parameters --
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
        finally:
            if previous_api_key is None:
                os.environ.pop(GEMINI_API_KEY_ENV_VAR, None)
            else:
                os.environ[GEMINI_API_KEY_ENV_VAR] = previous_api_key


@check("leader.openai_compatible_tool_schemas")
def check_openai_compatible_tool_schemas() -> None:
    with workspace() as ws:
        root = ws.root
        previous_api_key = os.environ.get(OPENAI_COMPATIBLE_API_KEY_ENV_VAR)
        try:
            os.environ[OPENAI_COMPATIBLE_API_KEY_ENV_VAR] = 'sk-compatible-fake-test-key-do-not-use'
            # -- regression: OpenAI-compatible providers must use OpenAI tool
            # schema shape even when their provider labels are vendor names --
            expected_tool_names = {
                "read_file",
                "write_file",
                "edit_file",
                "multi_edit_file",
                "list_files",
                "glob",
                "grep",
                "run_shell",
                "web_fetch",
            }
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
        finally:
            if previous_api_key is None:
                os.environ.pop(OPENAI_COMPATIBLE_API_KEY_ENV_VAR, None)
            else:
                os.environ[OPENAI_COMPATIBLE_API_KEY_ENV_VAR] = previous_api_key


@check("leader.default_specs_match_old_behaviour")
def check_default_specs_match_old_behaviour() -> None:
    with mock.patch.object(
        leader_module,
        "seed_messages",
        wraps=leader_module.seed_messages,
    ) as seed_messages_spy:
        actual, default_max_depth = _default_specs_probe()
    if actual != _DEFAULT_SPECS_PRE_07G_OUTPUT:
        fail(
            "default behavior differs from the pre-07g baseline: "
            f"expected={_DEFAULT_SPECS_PRE_07G_OUTPUT!r}, actual={actual!r}"
        )
    if default_max_depth != 0:
        fail(f"default max_depth changed: expected=0, actual={default_max_depth}")
    if seed_messages_spy.call_count != 1:
        fail(
            "default first dispatch did not seed exactly once: "
            f"calls={seed_messages_spy.call_count}"
        )
    default_config = LeaderConfig(FakeModelProvider(), FakeModelProvider(), ".")
    if default_config.permission_mode != "allow":
        fail(f"LeaderConfig's non-interactive default changed: {default_config.permission_mode!r}")


@check("leader.subagent_policy_is_narrowed")
def check_subagent_policy_is_narrowed() -> None:
    with workspace() as ws:
        root = ws.root
        allowed = PermissionPolicy(
            root,
            shell_enabled=True,
            shell_allowlist=[("git", "status")],
        )
        denied = PermissionPolicy(root)
        outside = PermissionPolicy(root.parent)
        forbidden = PermissionPolicy(root / ".git")
        tighter_mode = PermissionPolicy(root, mode="ask")
        specs = {
            "ceiling-denies": _spec(root, "ceiling-denies", policy=denied),
            "leader-denies": _spec(root, "leader-denies", policy=allowed),
            "outside": _spec(root, "outside", policy=outside),
            "forbidden": _spec(root, "forbidden", policy=forbidden),
            "tighter-mode": _spec(root, "tighter-mode", policy=tighter_mode),
            "still-works": _spec(root, "still-works", policy=denied),
        }
        ceiling_tool = DispatchSubagentTool(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            allowed,
            subagent_specs=specs,
        )
        first = ceiling_tool.execute(_dispatch("ceiling-denies", "work"), allowed)
        if not first.ok or ceiling_tool.pool["ceiling-denies"].agent._policy.check_shell(["git", "status"]).allowed:
            fail("a child policy escaped its denying AgentSpec ceiling")

        leader_tool = DispatchSubagentTool(
            FakeModelProvider([
                ModelResponse(Message(Role.ASSISTANT, "done")),
                ModelResponse(Message(Role.ASSISTANT, "done")),
                ModelResponse(Message(Role.ASSISTANT, "done")),
            ]),
            denied,
            subagent_specs=specs,
        )
        second = leader_tool.execute(_dispatch("leader-denies", "work"), denied)
        if not second.ok or leader_tool.pool["leader-denies"].agent._policy.check_shell(["git", "status"]).allowed:
            fail("a child policy escaped its denying leader policy")
        for name, fragment in (
            ("outside", "outside root"),
            ("forbidden", "forbidden root"),
        ):
            rejected = leader_tool.execute(_dispatch(name, "work", name), denied)
            if rejected.ok or fragment not in (rejected.error or "") or name in leader_tool.pool:
                fail(f"invalid policy meet did not fail closed for {name!r}: {rejected!r}")
        narrowed_mode = leader_tool.execute(_dispatch("tighter-mode", "work", "mode"), denied)
        if not narrowed_mode.ok or leader_tool.pool["tighter-mode"].agent._policy.mode != "ask":
            fail(f"tighter child mode was not accepted: {narrowed_mode!r}")
        continued = leader_tool.execute(_dispatch("still-works", "work"), denied)
        if not continued.ok:
            fail(f"leader could not continue after an invalid policy meet: {continued!r}")


@check("leader.unknown_spec_name")
def check_unknown_spec_name_fails() -> None:
    with workspace() as ws:
        specs = {
            "coder": _spec(ws.root, "coder"),
            "reviewer": _spec(ws.root, "reviewer"),
        }
        tool = DispatchSubagentTool(
            FakeModelProvider(), ws.policy, subagent_specs=specs
        )
        result = tool.execute(_dispatch("missing", "work"), ws.policy)
        error = result.error or ""
        if result.ok or "missing" not in error or "coder" not in error or "reviewer" not in error:
            fail(f"unknown mapped name was not actionable: {result!r}")
        if tool.pool:
            fail(f"unknown mapped name created pool state: {tool.pool!r}")


@check("leader.child_token_isolation")
def check_child_token_isolation() -> None:
    with workspace() as ws:
        policy = ws.policy
        explicit_parent = CancellationToken()
        explicit = DispatchSubagentTool(
            FakeModelProvider([
                ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(id="cancel", name="cancel_work")]))
            ]),
            policy,
        )
        cancelling_tool = _CancellingSubagentTool()
        with mock.patch(
            "symphonai_api.leader.standard_tool_registry",
            return_value={cancelling_tool.name: cancelling_tool},
        ):
            explicit_result = explicit.execute(
                _dispatch("worker", "cancel"), policy, cancel=explicit_parent
            )
        if explicit_result.ok or "explicitly" not in (explicit_result.error or ""):
            fail(f"explicit child cancellation was not distinct: {explicit_result!r}")
        if explicit_parent.cancelled:
            fail("explicit child cancellation cancelled its parent token")

        parent = CancellationToken()

        class ParentCancellingProvider(FakeModelProvider):
            def create_response(self, request, *, cancel=None):  # noqa: ANN001
                parent.cancel()
                assert cancel is not None and cancel.reason is CancelReason.PARENT
                cancel.raise_if_cancelled()
                raise AssertionError("parent cancellation did not reach child")

        propagated = DispatchSubagentTool(ParentCancellingProvider(), policy)
        try:
            propagated.execute(_dispatch("worker", "cancel parent"), policy, cancel=parent)
        except OperationCancelled:
            pass
        else:
            fail("parent cancellation did not propagate as OperationCancelled")

        deadline_parent = CancellationToken()
        deadline = DispatchSubagentTool(
            _DeadlineProvider(),
            policy,
            subagent_specs={
                "worker": _spec(
                    ws.root, "worker", deadline_seconds=0.01
                )
            },
        )
        deadline_result = deadline.execute(
            _dispatch("worker", "wait"), policy, cancel=deadline_parent
        )
        if deadline_result.ok or "deadline" not in (deadline_result.error or ""):
            fail(f"deadline cancellation was not distinct: {deadline_result!r}")
        if deadline_parent.cancelled:
            fail("child deadline cancelled its parent token")

        reusable_parent = CancellationToken()
        baseline = len(reusable_parent._callbacks)
        normal = DispatchSubagentTool(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            policy,
        )
        for index in range(20):
            outcome = normal.execute(
                _dispatch("worker", f"task {index}", f"normal-{index}"),
                policy,
                cancel=reusable_parent,
            )
            if not outcome.ok:
                fail(f"normal dispatch {index} failed: {outcome!r}")
        if len(reusable_parent._callbacks) != baseline:
            fail("normal child runs leaked parent cancellation listeners")

        raising_parent = CancellationToken()
        raising_baseline = len(raising_parent._callbacks)
        raising = DispatchSubagentTool(_RaisingProvider(), policy)
        try:
            raising.execute(
                _dispatch("worker", "raise"), policy, cancel=raising_parent
            )
        except RuntimeError:
            pass
        else:
            fail("scripted provider failure did not escape the dispatch")
        if len(raising_parent._callbacks) != raising_baseline:
            fail("raising child run leaked a parent cancellation listener")


@check("leader.dispatch_records_a_run")
def check_dispatch_records_a_run() -> None:
    with workspace() as ws:
        failing = DispatchSubagentTool(_RaisingProvider(), ws.policy)
        try:
            failing.execute(_dispatch("worker", "fail"), ws.policy)
        except RuntimeError:
            pass
        else:
            fail("expected the failing child provider to raise")
        failed_run = failing.pool["worker"].runs[0]
        if failed_run.phase is not RunPhase.FAILED or "scripted provider failure" not in (failed_run.error or ""):
            fail(f"child exception did not terminalize its AgentRun: {failed_run!r}")
        if failed_run.agent is not failing.pool["worker"].agent_ref:
            fail("the pooled ApiAgent does not own its recorded AgentRun")

        session = SessionStore(ws.root / "sessions", "leader-graph")
        leader = Leader(
            LeaderConfig(
                leader_provider=FakeModelProvider([
                    ModelResponse(Message(Role.ASSISTANT, tool_calls=[_dispatch("worker", "work", "graph-dispatch")])),
                    ModelResponse(Message(Role.ASSISTANT, "final")),
                ]),
                subagent_provider=FakeModelProvider([
                    ModelResponse(Message(Role.ASSISTANT, "child"))
                ]),
                repo_root=str(ws.root),
            ),
            session=session,
        )
        leader.run("goal")
        graph = leader.run_graph()
        if len(graph) != 1 or graph[0].agent_name != "leader":
            fail(f"session graph did not have one leader root: {graph!r}")
        if len(graph[0].children) != 1 or graph[0].children[0].agent_name != "worker":
            fail(f"session graph did not parent the child under the leader: {graph!r}")
        control_run = leader.subagents["worker"].runs[0]
        if control_run.run.run_id != graph[0].children[0].run_id:
            fail("recorded child AgentRun did not use the transcript run identity")

        no_session = Leader(
            LeaderConfig(
                leader_provider=FakeModelProvider(),
                subagent_provider=FakeModelProvider(),
                repo_root=str(ws.root),
            )
        )
        if no_session.run_graph() != ():
            fail("leader without a session returned a non-empty run graph")


@check("leader.child_context_seeding")
def check_child_context_seeding() -> None:
    with workspace() as ws:
        parent_messages = [
            Message(Role.SYSTEM, "parent system"),
            Message(Role.USER, "old user"),
            Message(Role.ASSISTANT, "old answer"),
            Message(Role.USER, "recent user"),
            Message(Role.ASSISTANT, "recent answer"),
        ]
        provider = _RecordingFakeProvider([
            ModelResponse(Message(Role.ASSISTANT, "done"))
        ])
        specs = {
            "fresh": _spec(ws.root, "fresh", prompt="fresh prompt"),
            "all": _spec(
                ws.root,
                "all",
                prompt="all prompt",
                isolation=Isolation(inherit=ContextInheritance.ALL),
            ),
            "tail": _spec(
                ws.root,
                "tail",
                prompt="tail prompt",
                isolation=Isolation(
                    inherit=ContextInheritance.TAIL, inherit_tail=1
                ),
            ),
        }
        parent_run = new_agent_run(_spec(ws.root, "leader"))
        tool = DispatchSubagentTool(
            provider,
            ws.policy,
            subagent_specs=specs,
            parent_run=parent_run,
        )
        tool.set_parent_context(parent_run, parent_messages)
        for name in ("fresh", "all", "tail"):
            result = tool.execute(_dispatch(name, f"{name} task", name), ws.policy)
            if not result.ok:
                fail(f"{name} context dispatch failed: {result!r}")

        request_messages = [list(request.messages) for request in provider.requests]
        fresh_contents = [message.text for message in request_messages[0]]
        all_contents = [message.text for message in request_messages[1]]
        tail_contents = [message.text for message in request_messages[2]]
        if fresh_contents != ["fresh prompt", "fresh task"]:
            fail(f"FRESH inherited parent context: {fresh_contents!r}")
        if all_contents != [
            "all prompt", "old user", "old answer", "recent user", "recent answer", "all task"
        ]:
            fail(f"ALL did not inherit non-system parent context: {all_contents!r}")
        if tail_contents != ["tail prompt", "recent user", "recent answer", "tail task"]:
            fail(f"TAIL did not inherit exactly the recent parent turn: {tail_contents!r}")

        followup = tool.execute(_dispatch("all", "follow up", "all-again"), ws.policy)
        followup_contents = [message.text for message in provider.requests[3].messages]
        if (
            not followup.ok
            or followup_contents.count("old user") != 1
            or followup_contents.count("done") != 1
            or followup_contents[-1] != "follow up"
        ):
            fail(f"reused child re-inherited context instead of appending: {followup_contents!r}")


@check("leader.dispatch_holds_a_lease")
def check_dispatch_holds_a_lease() -> None:
    with workspace() as ws:
        leases = WorkspaceLeases(ws.root)
        observing = _LeaseObservingProvider(leases, "work")
        specs = {
            "worker": _spec(
                ws.root, "worker", isolation=Isolation(workspace_prefix="work")
            )
        }
        tool = DispatchSubagentTool(
            observing, ws.policy, subagent_specs=specs, leases=leases
        )
        result = tool.execute(_dispatch("worker", "work"), ws.policy)
        recorded_run = tool.pool["worker"].runs[0]
        if not result.ok or observing.observed_holder != recorded_run.run.run_id:
            fail("dispatch did not hold its workspace lease for the child run")
        if leases.holder_for("work") is not None:
            fail("successful dispatch did not release its workspace lease")

        raising = DispatchSubagentTool(
            _RaisingProvider(), ws.policy, subagent_specs=specs, leases=leases
        )
        try:
            raising.execute(_dispatch("worker", "raise"), ws.policy)
        except RuntimeError:
            pass
        else:
            fail("raising provider did not raise during lease check")
        if leases.holder_for("work") is not None:
            fail("exceptional dispatch did not release its workspace lease")

        external = leases.acquire("external-holder", "work")
        try:
            conflicted = tool.execute(
                _dispatch("worker", "conflict", "conflict"), ws.policy
            )
        finally:
            leases.release(external)
        error = conflicted.error or ""
        if conflicted.ok or "work" not in error or "external-holder" not in error:
            fail(f"lease conflict was not an actionable ToolResult: {conflicted!r}")


@check("leader.max_depth_refusal")
def check_max_depth_is_enforced() -> None:
    with workspace() as ws:
        tool = DispatchSubagentTool(
            FakeModelProvider(),
            ws.policy,
            subagent_specs={"leaf": _spec(ws.root, "leaf", max_depth=0)},
            dispatching_depth=0,
        )
        result = tool.execute(_dispatch("leaf", "recurse"), ws.policy)
        if result.ok or "depth" not in (result.error or "") or "leaf" not in (result.error or ""):
            fail(f"max depth denial was not actionable: {result!r}")
        if tool.pool:
            fail("depth-denied dispatch created pool state")
        if "dispatch_subagent" in standard_tool_registry():
            fail("dispatch_subagent leaked into the standard child tool registry")


@check("leader.typed_output_contract")
def check_typed_subagent_output() -> None:
    with workspace() as ws:
        schema = IOContract(
            output_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        )
        specs = {
            "valid": _spec(ws.root, "valid", io=schema),
            "invalid": _spec(ws.root, "invalid", io=schema),
        }
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, '{"answer":"yes"}')),
            ModelResponse(Message(Role.ASSISTANT, '{"answer":7}')),
        ])
        tool = DispatchSubagentTool(
            provider, ws.policy, subagent_specs=specs
        )
        valid = tool.execute(_dispatch("valid", "work", "valid"), ws.policy)
        if not valid.ok or valid.content != '{"answer":"yes"}':
            fail(f"valid typed output was changed or rejected: {valid!r}")
        invalid = tool.execute(_dispatch("invalid", "work", "invalid"), ws.policy)
        if invalid.ok or "output.answer must be string" not in (invalid.error or ""):
            fail(f"invalid typed output was accepted: {invalid!r}")
        if tool.pool["invalid"].breaker.consecutive_failures != 1:
            fail("typed-output validation failure did not advance the child breaker")


@check("leader.builtin_roster_tool_registries")
def check_builtin_roster_tool_registries() -> None:
    with workspace() as ws:
        standard = set(standard_tool_registry())
        explorer_standard = {"read_file", "glob", "grep", "list_files", "web_fetch"}
        for result_store in (None, ToolResultStore()):
            provider = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))])
            tool = DispatchSubagentTool(
                provider, ws.policy,
                subagent_specs=builtin_subagent_specs(provider, ws.policy),
                result_store=result_store,
            )
            for name in ("explorer", "worker"):
                result = tool.execute(_dispatch(name, "inspect", name), ws.policy)
                if not result.ok:
                    fail(f"built-in {name} could not dispatch: {result!r}")
            stored_tool = {"read_tool_result"} if result_store is not None else set()
            explorer = set(tool.pool["explorer"].agent._tools)
            worker = set(tool.pool["worker"].agent._tools)
            if explorer != explorer_standard | stored_tool:
                fail(f"explorer's actual registry was wrong for store={result_store is not None}: {explorer!r}")
            if worker != standard | stored_tool:
                fail(f"worker's actual registry was wrong for store={result_store is not None}: {worker!r}")


@check("leader.default_spec_reports_actual_tools")
def check_default_spec_reports_actual_tools() -> None:
    with workspace() as ws:
        leader = Leader(LeaderConfig(FakeModelProvider(), FakeModelProvider(), str(ws.root)))
        actual = set(leader._agent._tools)
        reported = set(leader._leader_spec.tool_names or ())
        if actual != {"dispatch_subagent", *standard_tool_registry()} or reported != actual:
            fail(f"leader spec did not report its actual dispatch and standard tools: {reported!r}, {actual!r}")


@check("leader.search_roster_dispatch")
def check_search_roster_dispatch() -> None:
    class Backend(SearchBackend):
        @property
        def name(self) -> str:
            return "fake-search"

        def search(self, query: str, *, limit: int, cancel=None) -> list:
            return []

    with workspace() as ws:
        backend = Backend()
        provider = FakeModelProvider([
            ModelResponse(Message(Role.ASSISTANT, "done")) for _ in range(4)
        ])
        roster = builtin_subagent_specs(provider, ws.policy, backend)
        roster["project_search"] = _spec(ws.root, "project_search").with_overrides(
            tool_names=("read_file", "web_search")
        )
        roster["project_read"] = _spec(ws.root, "project_read").with_overrides(
            tool_names=("read_file",)
        )
        tool = DispatchSubagentTool(
            provider, ws.policy, subagent_specs=roster, search_backend=backend,
        )
        for name, should_search in (
            ("worker", True), ("explorer", True),
            ("project_search", True), ("project_read", False),
        ):
            result = tool.execute(_dispatch(name, "inspect", name), ws.policy)
            if not result.ok:
                fail(f"configured {name} did not dispatch: {result!r}")
            actual = "web_search" in tool.pool[name].agent._tools
            if actual != should_search:
                fail(f"configured {name} search registry mismatch: {tool.pool[name].agent._tools!r}")
        missing = DispatchSubagentTool(
            FakeModelProvider(), ws.policy,
            subagent_specs={"project_search": roster["project_search"]},
        )
        result = missing.execute(_dispatch("project_search", "inspect"), ws.policy)
        if result.ok or "search is not configured" not in (result.error or "") or missing.pool:
            fail(f"unconfigured search definition did not fail at dispatch: {result!r}")


@check("leader.memory_registry_and_seeding")
def check_memory_registry_and_seeding() -> None:
    with workspace() as ws:
        store = AgentMemory(ws.root / "memory")
        store.write("enabled", "older lesson", run_id="old-1")
        store.write("enabled", "latest lesson", run_id="old-2")
        store.write("leader", "leader preference", run_id="old-3")
        enabled = _spec(
            ws.root,
            "enabled",
            prompt="enabled prompt",
            memory=MemorySettings(enabled=True, max_entries=1),
        )
        disabled = _spec(ws.root, "disabled", prompt="disabled prompt")
        provider = _RecordingFakeProvider([
            ModelResponse(Message(Role.ASSISTANT, "enabled done")),
            ModelResponse(Message(Role.ASSISTANT, "disabled done")),
        ])
        dispatch = DispatchSubagentTool(
            provider,
            ws.policy,
            subagent_specs={"enabled": enabled, "disabled": disabled},
            memory=store,
        )
        for name in ("enabled", "disabled"):
            result = dispatch.execute(_dispatch(name, "same task", name), ws.policy)
            if not result.ok:
                fail(f"{name} memory dispatch failed: {result!r}")
        if "remember" not in dispatch.pool["enabled"].agent._tools:
            fail("memory-enabled agent registry omitted remember")
        if "remember" in dispatch.pool["disabled"].agent._tools:
            fail("memory-disabled agent registry exposed remember")
        enabled_text = [message.text for message in provider.requests[0].messages]
        if (
            enabled_text[0] != "enabled prompt"
            or "latest lesson" not in enabled_text[1]
            or "older lesson" in enabled_text[1]
            or enabled_text[-1] != "same task"
        ):
            fail(f"enabled memory seed was wrong: {enabled_text!r}")
        disabled_text = [message.text for message in provider.requests[1].messages]
        if disabled_text != ["disabled prompt", "same task"]:
            fail(f"disabled memory changed context seeding: {disabled_text!r}")

        leader_spec = _spec(
            ws.root,
            "leader",
            prompt="leader prompt",
            memory=MemorySettings(enabled=True),
        )
        leader_provider = _RecordingFakeProvider([
            ModelResponse(Message(Role.ASSISTANT, "leader done")),
        ])
        leader = Leader(LeaderConfig(
            leader_provider,
            FakeModelProvider(),
            str(ws.root),
            subagent_specs={"leader": leader_spec},
            memory=store,
        ))
        leader.run("lead")
        if "remember" not in leader._agent._tools:
            fail("memory-enabled leader registry omitted remember")
        leader_text = [message.text for message in leader_provider.requests[0].messages]
        if (
            leader_text[0] != "leader prompt"
            or "leader preference" not in leader_text[1]
            or leader_text[-1] != "lead"
        ):
            fail(f"leader did not follow memory seeding: {leader_text!r}")


@check("leader.memory_round_trip_and_isolation")
def check_memory_round_trip_and_isolation() -> None:
    with workspace() as ws:
        store = AgentMemory(ws.root / "memory")
        spec = _spec(
            ws.root,
            "reviewer",
            memory=MemorySettings(enabled=True),
        )
        writing = _RecordingFakeProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "remember-call",
                "remember",
                {"text": "Use the project's terse report style."},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "written")),
        ])
        first = DispatchSubagentTool(
            writing,
            ws.policy,
            subagent_specs={"reviewer": spec},
            memory=store,
        )
        result = first.execute(_dispatch("reviewer", "review", "first"), ws.policy)
        entries = store.read("reviewer")
        if not result.ok or len(entries) != 1:
            fail(f"memory tool did not complete its dispatch: {result!r}, {entries!r}")
        current_run_id = first.pool["reviewer"].runs[0].run.run_id
        if entries[0].run_id != current_run_id:
            fail(f"memory used {entries[0].run_id!r}, expected current run {current_run_id!r}")

        reading = _RecordingFakeProvider([
            ModelResponse(Message(Role.ASSISTANT, "read")),
        ])
        second = DispatchSubagentTool(
            reading,
            ws.policy,
            subagent_specs={"reviewer": spec},
            memory=store,
        )
        result = second.execute(_dispatch("reviewer", "review", "second"), ws.policy)
        seeded = [message.text for message in reading.requests[0].messages]
        if not result.ok or not any(entries[0].text in text for text in seeded):
            fail(f"later dispatch was not seeded from the write: {seeded!r}")

        other_spec = _spec(
            ws.root,
            "analyzer",
            memory=MemorySettings(enabled=True),
        )
        isolated_provider = _RecordingFakeProvider([
            ModelResponse(Message(Role.ASSISTANT, "isolated")),
        ])
        isolated = DispatchSubagentTool(
            isolated_provider,
            ws.policy,
            subagent_specs={"analyzer": other_spec},
            memory=store,
        )
        result = isolated.execute(_dispatch("analyzer", "review", "other"), ws.policy)
        other_seed = [message.text for message in isolated_provider.requests[0].messages]
        if not result.ok or any(entries[0].text in text for text in other_seed):
            fail(f"another agent received reviewer's memory: {other_seed!r}")


@check("leader.memory_failures_do_not_fail_run")
def check_memory_failures_do_not_fail_run() -> None:
    with workspace() as ws:
        spec = _spec(
            ws.root,
            "reviewer",
            prompt="review prompt",
            memory=MemorySettings(enabled=True),
        )
        oversized = "x" * (MAX_ENTRY_CHARS + 1)
        store = AgentMemory(ws.root / "memory")
        limited_provider = _RecordingFakeProvider([
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                "too-long", "remember", {"text": oversized},
            )])),
            ModelResponse(Message(Role.ASSISTANT, "continued")),
        ])
        limited = DispatchSubagentTool(
            limited_provider,
            ws.policy,
            subagent_specs={"reviewer": spec},
            memory=store,
        )
        result = limited.execute(_dispatch("reviewer", "review", "limited"), ws.policy)
        tool_results = [
            message.tool_result
            for message in limited_provider.requests[1].messages
            if message.tool_result is not None
        ]
        if (
            not result.ok
            or len(tool_results) != 1
            or tool_results[0].ok
            or "MAX_ENTRY_CHARS" not in (tool_results[0].error or "")
            or store.read("reviewer")
        ):
            fail(f"overlong memory did not fail only its tool call: {result!r}, {tool_results!r}")

        unavailable_root = ws.root / "unavailable-memory"
        unavailable = AgentMemory(unavailable_root)
        original_mode = unavailable_root.stat().st_mode
        unavailable_root.chmod(0o500)
        try:
            unavailable_provider = _RecordingFakeProvider([
                ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall(
                    "unavailable", "remember", {"text": "durable lesson"},
                )])),
                ModelResponse(Message(Role.ASSISTANT, "continued")),
            ])
            dispatch = DispatchSubagentTool(
                unavailable_provider,
                ws.policy,
                subagent_specs={"reviewer": spec},
                memory=unavailable,
            )
            result = dispatch.execute(_dispatch("reviewer", "review", "unavailable"), ws.policy)
            unavailable_results = [
                message.tool_result
                for message in unavailable_provider.requests[1].messages
                if message.tool_result is not None
            ]
            initial = [message.text for message in unavailable_provider.requests[0].messages]
            if (
                not result.ok
                or len(unavailable_results) != 1
                or unavailable_results[0].ok
                or initial != ["review prompt", "review"]
            ):
                fail(
                    "unavailable memory did not degrade to an empty seed and failed tool: "
                    f"{result!r}, {initial!r}, {unavailable_results!r}"
                )
        finally:
            unavailable_root.chmod(original_mode)
