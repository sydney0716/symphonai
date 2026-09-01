"""The Leader: digests a user goal and dispatches/reuses named subagents.

The leader is itself an `ApiAgent`, but with exactly one tool available:
`dispatch_subagent`. Calling it with a new `subagent_name` creates a fresh
subagent (its own `ApiAgent`, backed by the one configured subagent
provider, with the standard local tool registry and a deny-by-default
`PermissionPolicy`); calling it again with the same name continues that
subagent's existing conversation instead of starting over -- that is the
"reuse" behavior.

The leader never chooses which vendor/model backs itself or its
subagents. Both are fixed by `LeaderConfig`, set by the caller, never
decided by a model. See `docs/orchestra-api-runtime.md`.

Subagent pool state lives only in memory for the duration of one
`Leader.run()` call -- there is no cross-run persistence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from orchestra_api.agent_loop import DEFAULT_MAX_TURNS, ApiAgent
from orchestra_api.budgets import RunBudget
from orchestra_api.call_class import CallClass
from orchestra_api.cancellation import CancellationToken, OperationCancelled
from orchestra_api.circuit_breaker import (
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    ConsecutiveFailureBreaker,
)
from orchestra_api.cost import UsageTotals
from orchestra_api.compaction import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_RECENT_TURNS,
    CompactionResult,
    ContextCompactionError,
    compact_messages_for_budget,
)
from orchestra_api.gemini_schema import sanitize_for_gemini
from orchestra_api.events import (
    CompactionApplied,
    Event,
    EventSink,
    SubagentSpawned,
    ToolCallStarted,
    emit,
)
from orchestra_api.identity import AgentRef, RunRef, new_agent_ref
from orchestra_api.models import Message, Role, ToolCall, ToolResult
from orchestra_api.permissions import ApprovalCallback, PermissionMode, PermissionPolicy
from orchestra_api.providers.base import ModelProvider
from orchestra_api.runner import standard_tool_registry
from orchestra_api.session import SessionStore
from orchestra_api.tool_schema import tool_registry_schemas
from orchestra_api.tools.base import LocalTool
from orchestra_api.tools.metadata import ToolEffect, ToolMetadata

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
    orchestra_api.tool_schema.
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


class DispatchSubagentTool(LocalTool):
    """The leader's only tool: create-or-reuse a named subagent and run it.

    Note: the `policy` argument `execute()` receives is the *leader's own*
    policy (passed in by the leader's ApiAgent), which this tool
    deliberately ignores -- each subagent runs under its own
    `subagent_policy`, fixed at construction time, independent of whatever
    the leader itself is permitted to touch.
    """

    def __init__(
        self,
        subagent_provider: ModelProvider,
        subagent_policy: PermissionPolicy,
        *,
        max_subagents: int = DEFAULT_MAX_SUBAGENTS,
        subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS,
        subagent_tool_names: Sequence[str] | None = None,
        parent_agent_id: str | None = None,
        subagent_budget: RunBudget | None = None,
        max_consecutive_subagent_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        session: SessionStore | None = None,
    ) -> None:
        self._subagent_provider = subagent_provider
        self._subagent_policy = subagent_policy
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
        self._events: EventSink | None = None
        self._event_agent_id = parent_agent_id or ""
        self._event_run_id: str | None = None
        self._event_turn_id: str | None = None
        self.pool: dict[str, SubagentRecord] = {}

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

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        subagent_name = tool_call.arguments.get("subagent_name")
        task = tool_call.arguments.get("task")

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
            subagent_tools = standard_tool_registry(self._subagent_tool_names)
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
                    policy=self._subagent_policy,
                    max_turns=self._subagent_max_turns,
                    tool_schemas=tool_registry_schemas(subagent_tools, self._subagent_provider.wire_format),
                    agent_ref=agent_ref,
                    events=self._events,
                    budget=self._subagent_budget,
                    call_class=CallClass.BACKGROUND,
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

        record.messages.append(Message(role=Role.USER, content=task))
        try:
            run_result = record.agent.run(record.messages, cancel=cancel)
        except OperationCancelled:
            raise
        except Exception:
            raise
        record.messages = run_result.messages
        record.turns_used += run_result.turns_used
        for model, usage in run_result.usage_by_model.items():
            record.usage_by_model[model] = record.usage_by_model.get(
                model, UsageTotals()
            ).merged(usage)
        if run_result.stopped_reason == "cancelled":
            raise OperationCancelled

        succeeded = run_result.stopped_reason == "final_response"
        if succeeded:
            record.breaker.record_success()
        else:
            record.breaker.record_failure()
        if succeeded:
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
    no shared drawdown because coordinating that across agents belongs to
    phase 07.
    """

    leader_provider: ModelProvider
    subagent_provider: ModelProvider
    repo_root: str
    max_leader_turns: int = DEFAULT_MAX_TURNS
    max_subagents: int = DEFAULT_MAX_SUBAGENTS
    subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS
    subagent_budget: RunBudget | None = None
    subagent_tool_names: Sequence[str] | None = None
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
        subagent_policy = PermissionPolicy(
            repo_root=config.repo_root,
            mode=config.permission_mode,
            approval_callback=config.approval_callback,
        )
        self._dispatch_tool = DispatchSubagentTool(
            subagent_provider=config.subagent_provider,
            subagent_policy=subagent_policy,
            max_subagents=config.max_subagents,
            subagent_max_turns=config.subagent_max_turns,
            subagent_tool_names=config.subagent_tool_names,
            parent_agent_id=self._agent_ref.agent_id,
            subagent_budget=config.subagent_budget,
            max_consecutive_subagent_failures=(
                config.max_consecutive_subagent_failures
            ),
            session=session,
        )
        self._event_sink.bind_dispatch_tool(self._dispatch_tool)
        leader_policy = PermissionPolicy(
            repo_root=config.repo_root,
            mode=config.permission_mode,
            approval_callback=config.approval_callback,
        )
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
        try:
            result = self._agent.run(messages, cancel=cancel)
        except OperationCancelled:
            raise
        except Exception:
            raise
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
