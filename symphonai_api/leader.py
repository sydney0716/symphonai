"""The Leader: digests a user goal and dispatches/reuses named subagents.

The leader is itself an `ApiAgent`, but with exactly one tool available:
`dispatch_subagent`. Calling it with a new `subagent_name` creates a fresh
subagent (its own `ApiAgent`, backed by the one configured subagent
provider, with capabilities and limits taken from its `AgentSpec`); calling it
again with the same name continues that subagent's existing conversation
instead of starting over -- that is the "reuse" behavior.

The leader never chooses which vendor/model backs itself or its
subagents. Both are fixed by `LeaderConfig`, set by the caller, never
decided by a model. See `docs/symphonai-api-runtime.md`.

Subagent pool state lives only in memory for the duration of one
`Leader.run()` call -- there is no cross-run persistence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from symphonai_api.agent_loop import DEFAULT_MAX_TURNS, ApiAgent
from symphonai_api.agent_run import (
    AgentRun,
    RunNode,
    RunPhase,
    new_agent_run,
    read_run_graph,
)
from symphonai_api.agent_spec import AgentSpec, Isolation, ModelSelector, validate_output
from symphonai_api.budgets import RunBudget
from symphonai_api.call_class import CallClass
from symphonai_api.cancellation import (
    CancelReason,
    CancellationToken,
    OperationCancelled,
)
from symphonai_api.child_context import seed_messages
from symphonai_api.circuit_breaker import (
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    ConsecutiveFailureBreaker,
)
from symphonai_api.cost import UsageTotals
from symphonai_api.compaction import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_RECENT_TURNS,
    CompactionResult,
    ContextCompactionError,
    compact_messages_for_budget,
)
from symphonai_api.gemini_schema import sanitize_for_gemini
from symphonai_api.events import (
    CompactionApplied,
    Event,
    EventSink,
    RunStarted,
    SubagentSpawned,
    ToolCallStarted,
    emit,
)
from symphonai_api.identity import AgentRef, RunRef, new_agent_ref
from symphonai_api.leases import LeaseConflict, WorkspaceLeases
from symphonai_api.models import Message, Role, ToolCall, ToolResult
from symphonai_api.permissions import ApprovalCallback, PermissionMode, PermissionPolicy
from symphonai_api.providers.base import ModelProvider
from symphonai_api.runner import standard_tool_registry
from symphonai_api.session import SessionStore
from symphonai_api.tool_schema import tool_registry_schemas
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata

DISPATCH_TOOL_NAME = "dispatch_subagent"
DEFAULT_MAX_SUBAGENTS = 5
DEFAULT_SUBAGENT_MAX_TURNS = 5

class _LeaderEventSink:
    """Fan out events and preserve parent identity for subagent spawning."""

    def __init__(self, events: EventSink | None) -> None:
        self._events = events
        self._dispatch_tool: DispatchSubagentTool | None = None

    def bind_dispatch_tool(self, dispatch_tool: DispatchSubagentTool) -> None:
        self._dispatch_tool = dispatch_tool
        dispatch_tool.attach_event_sink(self)

    def __call__(self, event: Event) -> None:
        if self._dispatch_tool is not None:
            self._dispatch_tool.observe_event(event)
        if (
            isinstance(event, ToolCallStarted)
            and event.tool_name == DISPATCH_TOOL_NAME
            and self._dispatch_tool is not None
        ):
            self._dispatch_tool._set_event_context(
                agent_id=event.agent_id,
                run_id=event.run_id,
                turn_id=event.turn_id,
            )

        emit(self._events, event)

_DISPATCH_DESCRIPTION = (
    "Dispatch a task to a named subagent. If subagent_name has not been "
    "used yet in this run, a new subagent is created. If it has, the same "
    "subagent continues its existing conversation with this new task "
    "instead of starting over -- use the same name to follow up with the "
    "same subagent."
)
_DISPATCH_PROPERTIES = {
    "subagent_name": {
        "type": "string",
        "description": (
            "A short, stable identifier for this subagent, e.g. "
            "'researcher' or 'coder'. Reuse the same name to continue a "
            "conversation with the same subagent."
        ),
    },
    "task": {
        "type": "string",
        "description": "The task or follow-up message to give this subagent.",
    },
}
_DISPATCH_REQUIRED = ["subagent_name", "task"]


def _dispatch_parameters_schema() -> dict:
    return {
        "type": "object",
        "properties": _DISPATCH_PROPERTIES,
        "required": _DISPATCH_REQUIRED,
    }


def dispatch_subagent_tool_schema(wire_format: int) -> dict:
    """Build the dispatch_subagent tool definition in one provider's native shape.

    This is deliberately narrow -- a hand-written schema for this one tool,
    kept separate from the general LocalTool schema formatter in
    symphonai_api.tool_schema.
    """
    parameters = _dispatch_parameters_schema()
    if wire_format == 1:
        return {
            "type": "function",
            "function": {
                "name": DISPATCH_TOOL_NAME,
                "description": _DISPATCH_DESCRIPTION,
                "parameters": parameters,
            },
        }
    if wire_format == 2:
        return {
            "name": DISPATCH_TOOL_NAME,
            "description": _DISPATCH_DESCRIPTION,
            "input_schema": parameters,
        }
    if wire_format == 3:
        return {
            "name": DISPATCH_TOOL_NAME,
            "description": _DISPATCH_DESCRIPTION,
            "parameters": sanitize_for_gemini(parameters),
        }
    # Other/unclassified providers: keep schemas self-describing for debugging.
    return {
        "name": DISPATCH_TOOL_NAME,
        "description": _DISPATCH_DESCRIPTION,
        "parameters": parameters,
    }


@dataclass
class SubagentRecord:
    """One named subagent's live state within a single leader run."""

    agent: ApiAgent
    agent_ref: AgentRef
    breaker: ConsecutiveFailureBreaker
    messages: list[Message] = field(default_factory=list)
    turns_used: int = 0
    usage_by_model: dict[str, UsageTotals] = field(default_factory=dict)
    runs: list[AgentRun] = field(default_factory=list)


class DispatchSubagentTool(LocalTool):
    """The leader's only tool: create-or-reuse a named subagent and run it.

    A child's effective policy is the meet of the leader policy and its
    `AgentSpec` ceiling. Its first dispatch seeds context according to the
    spec; later dispatches append to the named child's existing conversation
    instead of inheriting the parent again.
    """

    def __init__(
        self,
        subagent_provider: ModelProvider,
        leader_policy: PermissionPolicy,
        *,
        max_subagents: int = DEFAULT_MAX_SUBAGENTS,
        subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS,
        subagent_tool_names: Sequence[str] | None = None,
        parent_agent_id: str | None = None,
        subagent_budget: RunBudget | None = None,
        max_consecutive_subagent_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        session: SessionStore | None = None,
        subagent_specs: Mapping[str, AgentSpec] | None = None,
        leases: WorkspaceLeases | None = None,
        parent_run: AgentRun | None = None,
        dispatching_depth: int = -1,
    ) -> None:
        self._subagent_provider = subagent_provider
        self._leader_policy = leader_policy
        self._max_subagents = max_subagents
        self._subagent_max_turns = subagent_max_turns
        self._subagent_tool_names = (
            None if subagent_tool_names is None else tuple(subagent_tool_names)
        )
        self._parent_agent_id = parent_agent_id
        # Every child receives the same immutable limits but tracks its own
        # spend; sharing drawdown needs phase 07's cross-agent coordination.
        self._subagent_budget = subagent_budget
        self._max_consecutive_subagent_failures = max_consecutive_subagent_failures
        self._session = session
        self._subagent_specs = subagent_specs
        self._leases = leases or WorkspaceLeases(leader_policy.repo_root)
        self._parent_run = parent_run
        self._parent_messages: list[Message] = []
        self._dispatching_depth = dispatching_depth
        self._active_run: AgentRun | None = None
        self._events: EventSink | None = None
        self._event_agent_id = parent_agent_id or ""
        self._event_run_id: str | None = None
        self._event_turn_id: str | None = None
        self.pool: dict[str, SubagentRecord] = {}

    def set_parent_context(
        self,
        run: AgentRun,
        messages: list[Message],
        *,
        depth: int = -1,
    ) -> None:
        self._parent_run = run
        self._parent_messages = messages
        self._dispatching_depth = depth

    def observe_event(self, event: Event) -> None:
        if isinstance(event, RunStarted):
            if (
                self._active_run is not None
                and event.agent_id == self._active_run.agent.agent_id
            ):
                self._active_run.run = RunRef(
                    run_id=event.run_id,
                    agent_id=event.agent_id,
                    parent_run_id=(
                        None
                        if self._parent_run is None
                        else self._parent_run.run.run_id
                    ),
                )
            elif (
                self._parent_run is not None
                and event.agent_id == self._parent_run.agent.agent_id
            ):
                self._parent_run.run = RunRef(
                    run_id=event.run_id,
                    agent_id=event.agent_id,
                    parent_run_id=None,
                )

    def attach_event_sink(self, sink: EventSink) -> None:
        self._events = sink

    def _set_event_context(
        self, *, agent_id: str, run_id: str, turn_id: str | None
    ) -> None:
        self._event_agent_id = agent_id
        self._event_run_id = run_id
        self._event_turn_id = turn_id

    @property
    def name(self) -> str:
        return DISPATCH_TOOL_NAME

    @property
    def description(self) -> str:
        return _DISPATCH_DESCRIPTION

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": _DISPATCH_PROPERTIES,
            "required": _DISPATCH_REQUIRED,
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        # A dispatched child may use any of its tools, so inherit the worst case.
        return ToolMetadata(
            effect=ToolEffect.DESTRUCTIVE,
            concurrency_safe=False,
            paths=None,
        )

    def validate(self, arguments: dict) -> str | None:
        if not arguments.get("subagent_name") or not arguments.get("task"):
            return "missing required argument: subagent_name and/or task"
        return None

    def _spec_for(self, subagent_name: str) -> AgentSpec | None:
        if self._subagent_specs is not None:
            return self._subagent_specs.get(subagent_name)
        provider_name = getattr(self._subagent_provider, "name", None) or "unknown"
        return AgentSpec(
            name=subagent_name,
            prompt="",
            model=ModelSelector(
                provider=provider_name,
                model=getattr(self._subagent_provider, "model", None),
            ),
            policy_ceiling=self._leader_policy,
            tool_names=self._subagent_tool_names,
            budget=self._subagent_budget,
            isolation=Isolation(),
            call_class=CallClass.BACKGROUND,
            max_depth=0,
        )

    def _new_run(self, spec: AgentSpec, agent_ref: AgentRef) -> AgentRun:
        run = new_agent_run(spec, parent=self._parent_run)
        run.agent = agent_ref
        run.run = RunRef(
            run_id=run.run.run_id,
            agent_id=agent_ref.agent_id,
            parent_run_id=(
                None if self._parent_run is None else self._parent_run.run.run_id
            ),
        )
        return run

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        subagent_name = tool_call.arguments.get("subagent_name")
        task = tool_call.arguments.get("task")

        spec = self._spec_for(subagent_name)
        if spec is None:
            known = ", ".join(sorted(self._subagent_specs or ())) or "none"
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"unknown subagent {subagent_name!r}; known subagents: {known}",
            )
        if self._dispatching_depth >= spec.max_depth:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=(
                    f"subagent {subagent_name!r} cannot dispatch at depth "
                    f"{self._dispatching_depth}; max_depth is {spec.max_depth}"
                ),
            )
        try:
            effective_policy = self._leader_policy.narrowed(spec.policy_ceiling)
        except ValueError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))

        record = self.pool.get(subagent_name)
        if record is None:
            if len(self.pool) >= self._max_subagents:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error=(
                        f"max_subagents ({self._max_subagents}) reached; "
                        f"cannot create new subagent {subagent_name!r}"
                    ),
                )
            subagent_tools = standard_tool_registry(spec.tool_names)
            agent_ref = new_agent_ref(subagent_name, self._parent_agent_id)
            if self._events is not None:
                if self._event_run_id is None:
                    raise RuntimeError("dispatch event context was never set")
                emit(
                    self._events,
                    SubagentSpawned(
                        agent_id=self._event_agent_id,
                        run_id=self._event_run_id,
                        turn_id=self._event_turn_id,
                        subagent_name=subagent_name,
                        subagent_agent_id=agent_ref.agent_id,
                    ),
                )
            record = SubagentRecord(
                agent=ApiAgent(
                    provider=self._subagent_provider,
                    tools=subagent_tools,
                    policy=effective_policy,
                    max_turns=self._subagent_max_turns,
                    tool_schemas=tool_registry_schemas(subagent_tools, self._subagent_provider.wire_format),
                    agent_ref=agent_ref,
                    events=self._events,
                    budget=spec.budget,
                    call_class=spec.call_class,
                    transcript=(
                        None
                        if self._session is None
                        else self._session.writer_for(agent_ref.agent_id)
                    ),
                ),
                agent_ref=agent_ref,
                breaker=ConsecutiveFailureBreaker(
                    f"subagent {subagent_name}",
                    max_consecutive_failures=self._max_consecutive_subagent_failures,
                ),
            )
            self.pool[subagent_name] = record
        else:
            # The ceiling is mutable even though AgentSpec is frozen, so take
            # the meet again for every dispatch instead of retaining authority.
            record.agent._policy = effective_policy

        if record.breaker.is_open:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=(
                    f"subagent {subagent_name!r} failed "
                    f"{record.breaker.consecutive_failures} times in a row; "
                    "not dispatching again this run"
                ),
            )

        child_run = self._new_run(spec, record.agent_ref)
        record.runs.append(child_run)
        token = (
            cancel.child(deadline_seconds=spec.deadline_seconds)
            if cancel is not None
            else CancellationToken(deadline_seconds=spec.deadline_seconds)
        )
        child_run.start(token)
        if record.messages:
            record.messages.append(Message(role=Role.USER, content=task))
        else:
            record.messages = seed_messages(
                spec,
                task,
                parent_messages=self._parent_messages,
            )
        run_result = None
        failure: str | None = None
        self._active_run = child_run
        try:
            with self._leases.held(child_run.run.run_id, spec.isolation.workspace_prefix):
                run_result = record.agent.run(
                    record.messages,
                    model=spec.model.model,
                    parent_run_id=(
                        None
                        if self._parent_run is None
                        else self._parent_run.run.run_id
                    ),
                    cancel=token,
                )
        except LeaseConflict as exc:
            failure = str(exc)
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=failure)
        except Exception as exc:
            failure = str(exc)
            raise
        finally:
            self._active_run = None
            if child_run.phase is RunPhase.RUNNING:
                if run_result is None:
                    if token.cancelled:
                        child_run.cancel()
                    else:
                        child_run.fail(failure or "subagent dispatch failed")
                elif run_result.stopped_reason == "cancelled":
                    child_run.cancel()
                else:
                    child_run.finish(run_result)
        assert run_result is not None
        record.messages = run_result.messages
        record.turns_used += run_result.turns_used
        for model, usage in run_result.usage_by_model.items():
            record.usage_by_model[model] = record.usage_by_model.get(
                model, UsageTotals()
            ).merged(usage)
        if run_result.stopped_reason == "cancelled":
            if token.reason is CancelReason.PARENT:
                raise OperationCancelled
            record.breaker.record_failure()
            if token.reason is CancelReason.DEADLINE:
                error = "subagent deadline elapsed before a final answer"
            else:
                error = "subagent was cancelled explicitly before a final answer"
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                content=run_result.final_response.message.text,
                error=error,
            )

        succeeded = run_result.stopped_reason == "final_response"
        output_error = None
        if succeeded and spec.io.output_schema is not None:
            _, output_error = validate_output(
                spec.io,
                run_result.final_response.message.text,
            )
            succeeded = output_error is None
        if succeeded:
            record.breaker.record_success()
        else:
            record.breaker.record_failure()
        if output_error is not None:
            error = output_error
        elif succeeded:
            error = None
        elif run_result.stopped_reason == "max_turns":
            error = "subagent reached max_turns without a final answer"
        else:
            error = (
                f"subagent stopped because {run_result.stopped_reason} "
                "before a final answer"
            )
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=succeeded,
            content=run_result.final_response.message.text,
            error=error,
        )


@dataclass
class LeaderConfig:
    """Fixed, user-chosen configuration for a leader run.

    The leader never picks its own or its subagents' provider/model --
    both are fixed here by the caller, once, before the run starts.
    Each subagent independently accounts against ``subagent_budget``; there is
    no shared drawdown. Optional ``subagent_specs`` declare named child roles;
    omitting them preserves the legacy synthesized defaults.
    """

    leader_provider: ModelProvider
    subagent_provider: ModelProvider
    repo_root: str
    max_leader_turns: int = DEFAULT_MAX_TURNS
    max_subagents: int = DEFAULT_MAX_SUBAGENTS
    subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS
    subagent_budget: RunBudget | None = None
    subagent_tool_names: Sequence[str] | None = None
    subagent_specs: Mapping[str, AgentSpec] | None = None
    permission_mode: PermissionMode = "auto"
    approval_callback: ApprovalCallback | None = None
    chat_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET
    chat_recent_turns: int = DEFAULT_RECENT_TURNS
    events: EventSink | None = None
    max_consecutive_compaction_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    max_consecutive_subagent_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES


@dataclass
class LeaderRunResult:
    """The outcome of `Leader.run()`."""

    final_answer: str
    leader_messages: list[Message]
    stopped_reason: str
    subagents: dict[str, SubagentRecord]
    run: RunRef
    agent: AgentRef
    usage_by_agent: dict[str, dict[str, UsageTotals]]
    stopped_repairs: tuple[str, ...] = ()


class Leader:
    """Digests a user goal, dispatching/reusing named subagents as needed.

    `run()` is one-shot: each call starts a fresh conversation. `chat()` is
    the multi-turn version: it keeps its own running message list across
    calls, so a later `chat()` call sees everything said in earlier ones.
    Use `run()` for a single task; use `chat()` for an interactive session.
    """

    def __init__(self, config: LeaderConfig, session: SessionStore | None = None) -> None:
        self._config = config
        self._session = session
        self._agent_ref = new_agent_ref("leader")
        self._event_sink = _LeaderEventSink(config.events)
        self._last_run_id: str | None = None
        leader_policy = PermissionPolicy(
            repo_root=config.repo_root,
            mode=config.permission_mode,
            approval_callback=config.approval_callback,
        )
        self._leader_policy = leader_policy
        self._leases = WorkspaceLeases(config.repo_root)
        provider_name = getattr(config.leader_provider, "name", None) or "unknown"
        self._leader_spec = AgentSpec(
            name="leader",
            prompt="",
            model=ModelSelector(
                provider=provider_name,
                model=getattr(config.leader_provider, "model", None),
            ),
            policy_ceiling=leader_policy,
            tool_names=(DISPATCH_TOOL_NAME,),
            call_class=CallClass.FOREGROUND,
            max_depth=0,
        )
        self._leader_run: AgentRun | None = None
        self._dispatch_tool = DispatchSubagentTool(
            subagent_provider=config.subagent_provider,
            leader_policy=leader_policy,
            max_subagents=config.max_subagents,
            subagent_max_turns=config.subagent_max_turns,
            subagent_tool_names=config.subagent_tool_names,
            parent_agent_id=self._agent_ref.agent_id,
            subagent_budget=config.subagent_budget,
            max_consecutive_subagent_failures=(
                config.max_consecutive_subagent_failures
            ),
            session=session,
            subagent_specs=config.subagent_specs,
            leases=self._leases,
        )
        self._event_sink.bind_dispatch_tool(self._dispatch_tool)
        self._agent = ApiAgent(
            provider=config.leader_provider,
            tools={DISPATCH_TOOL_NAME: self._dispatch_tool},
            policy=leader_policy,
            max_turns=config.max_leader_turns,
            tool_schemas=[dispatch_subagent_tool_schema(config.leader_provider.wire_format)],
            agent_ref=self._agent_ref,
            events=self._event_sink,
            call_class=CallClass.FOREGROUND,
            transcript=(
                None
                if session is None
                else session.writer_for(self._agent_ref.agent_id, is_root=True)
            ),
        )
        self._chat_messages: list[Message] = []
        self._automatic_compaction_breaker = ConsecutiveFailureBreaker(
            "automatic compaction",
            max_consecutive_failures=config.max_consecutive_compaction_failures,
        )

    @property
    def subagents(self) -> dict[str, SubagentRecord]:
        return self._dispatch_tool.pool

    def _stopped_repairs(self) -> tuple[str, ...]:
        breakers = [
            self._automatic_compaction_breaker,
            *(record.breaker for record in self._dispatch_tool.pool.values()),
        ]
        return tuple(sorted(breaker.name for breaker in breakers if breaker.is_open))

    def _run_messages(
        self, messages: list[Message], *, cancel: CancellationToken | None = None
    ) -> LeaderRunResult:
        leader_run = new_agent_run(self._leader_spec)
        leader_run.agent = self._agent_ref
        leader_run.run = RunRef(
            run_id=leader_run.run.run_id,
            agent_id=self._agent_ref.agent_id,
        )
        leader_run.start(CancellationToken())
        self._leader_run = leader_run
        self._dispatch_tool.set_parent_context(leader_run, list(messages))
        try:
            result = self._agent.run(messages, cancel=cancel)
        except OperationCancelled:
            leader_run.cancel()
            raise
        except Exception as exc:
            leader_run.fail(str(exc))
            raise
        if result.stopped_reason == "cancelled":
            leader_run.cancel()
        else:
            leader_run.finish(result)
        leader_run.run = result.run
        self._last_run_id = result.run.run_id
        usage_by_agent = {result.agent.agent_id: dict(result.usage_by_model)}
        usage_by_agent.update(
            {
                record.agent_ref.agent_id: dict(record.usage_by_model)
                for record in self._dispatch_tool.pool.values()
            }
        )
        return LeaderRunResult(
            final_answer=result.final_response.message.text,
            leader_messages=result.messages,
            stopped_reason=result.stopped_reason,
            subagents=self._dispatch_tool.pool,
            run=result.run,
            agent=result.agent,
            usage_by_agent=usage_by_agent,
            stopped_repairs=self._stopped_repairs(),
        )

    def run_graph(self) -> tuple[RunNode, ...]:
        """The run graph of this leader's session, or () without a session."""

        if self._session is None:
            return ()
        return read_run_graph(self._session)

    def run(
        self,
        goal: str,
        *,
        system_prompt: str | None = None,
        cancel: CancellationToken | None = None,
    ) -> LeaderRunResult:
        """Run a single, one-shot task. Each call starts a fresh conversation."""
        self.clear_subagents()
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        messages.append(Message(role=Role.USER, content=goal))
        return self._run_messages(messages, cancel=cancel)

    def clear_chat(self) -> int:
        """Clear the persisted chat state and subagent pool.

        Returns how many subagents were cleared, so a caller that wants to
        report it does not have to call `clear_subagents()` separately and
        thereby clear the pool twice.
        """

        self._chat_messages.clear()
        self._automatic_compaction_breaker.reset()
        return self.clear_subagents()

    def clear_subagents(self) -> int:
        """Clear all dispatched subagents and return how many were removed."""

        count = len(self._dispatch_tool.pool)
        self._dispatch_tool.pool.clear()
        return count

    def compact_chat(self, *, cancel: CancellationToken | None = None) -> CompactionResult:
        """Apply context compaction to the persisted multi-turn chat state."""

        result = compact_messages_for_budget(
            self._chat_messages,
            budget=self._config.chat_token_budget,
            recent_turns=self._config.chat_recent_turns,
            cancel=cancel,
        )
        self._chat_messages = result.messages
        self._automatic_compaction_breaker.record_success()
        if result.changed:
            if self._session is not None:
                self._session.writer_for(
                    self._agent_ref.agent_id, is_root=True
                ).append(
                    "compaction",
                    run_id=self._last_run_id or self._session.run_id,
                    agent_id=self._agent_ref.agent_id,
                    turn_id=None,
                    data={
                        "before_tokens": result.before_tokens,
                        "after_tokens": result.after_tokens,
                        "dropped_messages": result.dropped_messages,
                    },
                )
            emit(
                self._event_sink,
                CompactionApplied(
                    agent_id=self._agent_ref.agent_id,
                    run_id=self._last_run_id or "",
                    before_tokens=result.before_tokens,
                    after_tokens=result.after_tokens,
                    dropped_messages=result.dropped_messages,
                ),
            )
        return result

    def _automatic_compact_chat(
        self, *, cancel: CancellationToken | None = None
    ) -> None:
        if self._automatic_compaction_breaker.is_open:
            return
        try:
            self.compact_chat(cancel=cancel)
        except ContextCompactionError:
            self._automatic_compaction_breaker.record_failure()
        except OperationCancelled:
            # Cancellation is a caller decision, not a failed repair attempt.
            pass

    def chat(
        self, message: str, *, cancel: CancellationToken | None = None
    ) -> LeaderRunResult:
        """Continue an ongoing conversation with the leader.

        Unlike `run()`, this keeps its own running message history across
        calls -- a later `chat()` call includes everything said in earlier
        ones, so the leader (and its view of already-dispatched subagents)
        has full context of the conversation so far.
        """
        self._chat_messages.append(Message(role=Role.USER, content=message))
        self._automatic_compact_chat(cancel=cancel)
        result = self._run_messages(self._chat_messages, cancel=cancel)
        self._chat_messages = result.leader_messages
        self._automatic_compact_chat(cancel=cancel)
        result.stopped_repairs = self._stopped_repairs()
        return result
