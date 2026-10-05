"""The Leader: digests a user goal and dispatches/reuses named subagents.

The leader is itself an `ApiAgent` with standard tools and
`dispatch_subagent`. Calling dispatch with a new `subagent_name` creates a fresh
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

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from symphonai_api.agent_loop import DEFAULT_MAX_TURNS, ApiAgent, _message_digest
from symphonai_api.agent_memory import AgentMemory, MemoryEntry, MemorySettings
from symphonai_api.agent_run import (
    AgentRun,
    PauseGate,
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
from symphonai_api.checkpoints import CheckpointStore
from symphonai_api.child_context import seed_messages
from symphonai_api.circuit_breaker import (
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    ConsecutiveFailureBreaker,
)
from symphonai_api.cost import UsageTotals
from symphonai_api.compaction import (
    DEFAULT_RECENT_TURNS,
    MODEL_SUMMARY_PROMPT,
    CompactionResult,
    ContextCompactionError,
    budget_for_model,
    compact_messages_for_budget,
    estimate_messages_tokens,
    render_dropped_messages,
)
from symphonai_api.gemini_schema import sanitize_for_gemini
from symphonai_api.events import (
    CompactionApplied,
    Event,
    EventSink,
    RunStarted,
    SubagentSpawned,
    SubagentStopped,
    ToolCallStarted,
    emit,
    fan_out,
)
from symphonai_api.extensions import Extensions
from symphonai_api.hooks import HookRunner
from symphonai_api.identity import AgentRef, RunRef, new_agent_ref, new_id
from symphonai_api.leases import LeaseConflict, WorkspaceLeases
from symphonai_api.lsp import LspManager
from symphonai_api.models import ContentInput, Message, ModelRequest, Role, ToolCall, ToolResult
from symphonai_api.permissions import ApprovalCallback, PermissionMode, PermissionPolicy
from symphonai_api.providers.base import ContextLengthExceededError, ModelProvider
from symphonai_api.runner import merge_tool_registry, standard_tool_registry
from symphonai_api.roles import IMPLEMENTER_PROMPT, PLANNER_PROMPT, REVIEWER_PROMPT
from symphonai_api.session import SessionStore
from symphonai_api.skills import Skill
from symphonai_api.streaming import StreamAssembler
from symphonai_api.tool_schema import tool_registry_schemas
from symphonai_api.tool_results import ToolResultStore
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.filesystem import MAX_READ_BYTES
from symphonai_api.tools.memory import MemoryTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_api.web_search import SearchBackend
from symphonai_api.worktree import WorktreeError, create_worktree, remove_worktree, worktree_diff


class AgentControlError(RuntimeError):
    """A live agent could not accept a control request."""

    def __init__(self, message: str, *, status: int = 409) -> None:
        super().__init__(message)
        self.status = status

DISPATCH_TOOL_NAME = "dispatch_subagent"
DEFAULT_MAX_SUBAGENTS = 5
DEFAULT_SUBAGENT_MAX_TURNS = 5
EXPLORER_TOOL_NAMES = ("read_file", "glob", "grep", "list_files", "web_fetch")


def _memory_entries(
    store: AgentMemory | None,
    spec: AgentSpec,
) -> tuple[MemoryEntry, ...]:
    if store is None or not spec.memory.enabled:
        return ()
    return store.read(spec.name)[-spec.memory.max_entries:]


def builtin_subagent_specs(
    provider: ModelProvider,
    policy: PermissionPolicy,
    search_backend: SearchBackend | None = None,
    skills: Mapping[str, Skill] | None = None,
    lsp: LspManager | None = None,
) -> dict[str, AgentSpec]:
    selector = ModelSelector(
        provider=getattr(provider, "name", None) or "unknown",
        model=getattr(provider, "model", None),
    )
    specs = {
        name: AgentSpec(
            name=name,
            prompt="",
            model=selector,
            policy_ceiling=policy,
            tool_names=tools,
            call_class=CallClass.BACKGROUND,
        )
        for name, tools in (
            ("worker", tuple(standard_tool_registry(search_backend=search_backend, skills=skills, lsp=lsp))),
            (
                "explorer",
                EXPLORER_TOOL_NAMES
                + (("web_search",) if search_backend is not None else ())
                + (("use_skill",) if skills else ())
                + (("lsp",) if lsp is not None and lsp.has_enabled_servers else ()),
            ),
        )
    }
    optional = (("web_search",) if search_backend is not None else ()) + (("use_skill",) if skills else ())
    specs.update({
        "planner": AgentSpec(
            name="planner", prompt=PLANNER_PROMPT, model=selector,
            policy_ceiling=policy,
            tool_names=EXPLORER_TOOL_NAMES + ("write_file", "edit_file") + optional,
            call_class=CallClass.BACKGROUND,
        ),
        "implementer": AgentSpec(
            name="implementer", prompt=IMPLEMENTER_PROMPT, model=selector,
            policy_ceiling=policy,
            tool_names=tuple(standard_tool_registry(search_backend=search_backend, skills=skills, lsp=lsp)),
            call_class=CallClass.BACKGROUND,
        ),
        "reviewer": AgentSpec(
            name="reviewer", prompt=REVIEWER_PROMPT, model=selector,
            policy_ceiling=policy,
            tool_names=EXPLORER_TOOL_NAMES + ("run_shell", "write_file") + optional,
            call_class=CallClass.BACKGROUND,
        ),
    })
    return specs

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
    "same subagent. Set isolation to worktree to run from the last commit "
    "without uncommitted changes; its diff comes back in the result."
)
_BUILTIN_PURPOSES = {
    "worker": "implements coding tasks",
    "explorer": "reads and searches the repository",
    "planner": "writes implementation specs",
    "implementer": "implements a provided spec and reports validation",
    "reviewer": "reviews changes and writes follow-up specs",
}


def _dispatch_description(subagents: Mapping[str, AgentSpec] | None = None) -> str:
    names = _BUILTIN_PURPOSES if subagents is None else {
        name: purpose for name, purpose in _BUILTIN_PURPOSES.items() if name in subagents
    }
    available = [f"{name}: {purpose}." for name, purpose in names.items()]
    if subagents is not None:
        available.extend(sorted(set(subagents) - set(_BUILTIN_PURPOSES)))
    return _DISPATCH_DESCRIPTION + " Available subagents: " + "; ".join(available) + "."
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
    "isolation": {
        "type": "string",
        "enum": ["worktree"],
        "description": "Run the subagent in its own git worktree.",
    },
}
_DISPATCH_REQUIRED = ["subagent_name", "task"]


def _dispatch_parameters_schema() -> dict:
    return {
        "type": "object",
        "properties": _DISPATCH_PROPERTIES,
        "required": _DISPATCH_REQUIRED,
    }


def dispatch_subagent_tool_schema(
    wire_format: int,
    subagents: Mapping[str, AgentSpec] | None = None,
) -> dict:
    """Build the dispatch_subagent tool definition in one provider's native shape.

    This is deliberately narrow -- a hand-written schema for this one tool,
    kept separate from the general LocalTool schema formatter in
    symphonai_api.tool_schema.
    """
    parameters = _dispatch_parameters_schema()
    description = _dispatch_description(subagents)
    if wire_format == 1:
        return {
            "type": "function",
            "function": {
                "name": DISPATCH_TOOL_NAME,
                "description": description,
                "parameters": parameters,
            },
        }
    if wire_format == 2:
        return {
            "name": DISPATCH_TOOL_NAME,
            "description": description,
            "input_schema": parameters,
        }
    if wire_format == 3:
        return {
            "name": DISPATCH_TOOL_NAME,
            "description": description,
            "parameters": sanitize_for_gemini(parameters),
        }
    # Other/unclassified providers: keep schemas self-describing for debugging.
    return {
        "name": DISPATCH_TOOL_NAME,
        "description": description,
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
    pause_gate: PauseGate | None = None
    worktree_path: Path | None = None
    worktree_admin_path: Path | None = None
    running: bool = False


class DispatchSubagentTool(LocalTool):
    """Create-or-reuse a named subagent and run it.

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
        hooks: HookRunner | None = None,
        stream: bool = False,
        result_store: ToolResultStore | None = None,
        search_backend: SearchBackend | None = None,
        extra_tools: Mapping[str, LocalTool] | None = None,
        memory: AgentMemory | None = None,
        skills: Mapping[str, Skill] | None = None,
        checkpoints: CheckpointStore | None = None,
        lsp: LspManager | None = None,
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
        self._hooks = hooks
        self._stream = stream
        self._result_store = result_store
        self._search_backend = search_backend
        self._extra_tools = extra_tools
        self._memory = memory
        self._skills = skills
        self._checkpoints = checkpoints
        self._lsp = lsp
        self._pool_lock = threading.RLock()
        self._reserved_names: set[str] = set()
        self._active_runs_lock = threading.Lock()
        self._active_runs: dict[str, AgentRun] = {}
        self._events: EventSink | None = None
        self._event_agent_id = parent_agent_id or ""
        self._event_run_id: str | None = None
        self._event_turn_id: str | None = None
        self.pool: dict[str, SubagentRecord] = {}

    def _active_run_id(self, agent_id: str) -> str | None:
        with self._active_runs_lock:
            run = self._active_runs.get(agent_id)
            return None if run is None else run.run.run_id

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
            with self._active_runs_lock:
                active_run = self._active_runs.get(event.agent_id)
            if active_run is not None:
                active_run.run = RunRef(
                    run_id=event.run_id,
                    agent_id=event.agent_id,
                    parent_run_id=(None if self._parent_run is None else self._parent_run.run.run_id),
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
        return _dispatch_description(self._subagent_specs)

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": _DISPATCH_PROPERTIES,
            "required": _DISPATCH_REQUIRED,
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        name = arguments.get("subagent_name")
        with self._pool_lock:
            record = self.pool.get(name) if isinstance(name, str) else None
            isolated = (
                arguments.get("isolation") == "worktree"
                or (record is not None and record.worktree_path is not None)
            )
        return ToolMetadata(
            effect=ToolEffect.DESTRUCTIVE,
            concurrency_safe=isolated,
            paths=None,
        )

    def validate(self, arguments: dict) -> str | None:
        if not arguments.get("subagent_name") or not arguments.get("task"):
            return "missing required argument: subagent_name and/or task"
        if arguments.get("isolation") not in (None, "worktree"):
            return "isolation must be 'worktree' when provided"
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
        if self._search_backend is None and "web_search" in (spec.tool_names or ()):
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"subagent {subagent_name!r} cannot use web_search: search is not configured",
            )
        if not self._skills and "use_skill" in (spec.tool_names or ()):
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"subagent {subagent_name!r} cannot use use_skill: no skills are available",
            )
        if (self._lsp is None or not self._lsp.has_enabled_servers) and "lsp" in (spec.tool_names or ()):
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"subagent {subagent_name!r} cannot use lsp: no language server is configured",
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

        wants_worktree = tool_call.arguments.get("isolation") == "worktree"
        if wants_worktree and self._session is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error="worktree isolation requires a session",
            )
        if wants_worktree and (
            not isinstance(subagent_name, str)
            or subagent_name in (".", "..")
            or Path(subagent_name).name != subagent_name
            or "/" in subagent_name
            or "\\" in subagent_name
        ):
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error="subagent name cannot be used as a worktree path",
            )

        with self._pool_lock:
            record = self.pool.get(subagent_name)
            if subagent_name in self._reserved_names or (record is not None and record.running):
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error=(
                        f"subagent {subagent_name} is already running; "
                        "dispatch it again after it finishes"
                    ),
                )
            if record is not None and record.breaker.is_open:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error=(
                        f"subagent {subagent_name!r} failed "
                        f"{record.breaker.consecutive_failures} times in a row; "
                        "not dispatching again this run"
                    ),
                )
            if record is not None and wants_worktree and record.worktree_path is None:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error=f"subagent {subagent_name!r} already works in the shared tree",
                )
            if record is None:
                if len(self.pool) + len(self._reserved_names) >= self._max_subagents:
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        ok=False,
                        error=(
                            f"max_subagents ({self._max_subagents}) reached; "
                            f"cannot create new subagent {subagent_name!r}"
                        ),
                    )
                self._reserved_names.add(subagent_name)
            else:
                record.running = True
                if record.worktree_path is not None:
                    effective_policy = effective_policy.rerooted(record.worktree_path)
                    record.agent._policy = effective_policy

        if record is None and wants_worktree:
            worktree_admin_path = self._session.directory / "worktrees" / subagent_name
            try:
                worktree_root = create_worktree(
                    effective_policy.repo_root, worktree_admin_path
                )
            except WorktreeError as exc:
                with self._pool_lock:
                    self._reserved_names.discard(subagent_name)
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error=str(exc),
                )
            effective_policy = effective_policy.rerooted(worktree_root)
        else:
            worktree_admin_path = None
            worktree_root = None

        if record is None:
            tool_names = spec.tool_names
            if tool_names is not None and self._result_store is not None:
                tool_names = (*tool_names, "read_tool_result")
            agent_ref = new_agent_ref(subagent_name, self._parent_agent_id)
            memory_tool = (
                MemoryTool(
                    self._memory,
                    spec.name,
                    lambda agent_id=agent_ref.agent_id: self._active_run_id(agent_id),
                )
                if spec.memory.enabled
                else None
            )
            subagent_tools = merge_tool_registry(
                standard_tool_registry(
                    tool_names,
                    result_store=self._result_store,
                    search_backend=self._search_backend,
                    memory_tool=memory_tool,
                    skills=self._skills,
                    lsp=self._lsp,
                    checkpoints=(
                        None if worktree_root is not None else self._checkpoints
                    ),
                ),
                self._extra_tools,
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
                    stream=self._stream,
                    result_store=self._result_store,
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
                worktree_path=worktree_root,
                worktree_admin_path=worktree_admin_path,
                running=True,
            )
            with self._pool_lock:
                self.pool[subagent_name] = record
                self._reserved_names.discard(subagent_name)
        else:
            # The ceiling is mutable even though AgentSpec is frozen, so take
            # the meet again for every dispatch instead of retaining authority.
            record.agent._policy = effective_policy

        child_run = self._new_run(spec, record.agent_ref)
        record.runs.append(child_run)
        token = (
            cancel.child(deadline_seconds=spec.deadline_seconds)
            if cancel is not None
            else CancellationToken(deadline_seconds=spec.deadline_seconds)
        )
        child_run.start(token)
        pause_gate = PauseGate()
        record.pause_gate = pause_gate
        if record.messages:
            record.messages.append(Message(role=Role.USER, content=task))
        else:
            record.messages = seed_messages(
                spec,
                task,
                parent_messages=self._parent_messages,
                memory=_memory_entries(self._memory, spec),
            )
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
                    subagent_agent_id=record.agent_ref.agent_id,
                ),
            )
        run_result = None
        failure: str | None = None
        execution_error: Exception | None = None
        with self._active_runs_lock:
            self._active_runs[record.agent_ref.agent_id] = child_run
        try:
            if record.worktree_path is not None:
                run_result = record.agent.run(
                    record.messages,
                    model=spec.model.model,
                    effort=spec.model.effort,
                    parent_run_id=(
                        None
                        if self._parent_run is None
                        else self._parent_run.run.run_id
                    ),
                    cancel=token,
                    run=child_run,
                    pause=pause_gate,
                    hooks=self._hooks,
                )
            else:
                with self._leases.held(child_run.run.run_id, spec.isolation.workspace_prefix):
                    run_result = record.agent.run(
                        record.messages,
                        model=spec.model.model,
                        effort=spec.model.effort,
                        parent_run_id=(
                            None
                            if self._parent_run is None
                            else self._parent_run.run.run_id
                        ),
                        cancel=token,
                        run=child_run,
                        pause=pause_gate,
                        hooks=self._hooks,
                    )
        except LeaseConflict as exc:
            failure = str(exc)
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=failure)
        except Exception as exc:
            failure = str(exc)
            if record.worktree_path is None:
                raise
            execution_error = exc
        finally:
            with self._active_runs_lock:
                if self._active_runs.get(record.agent_ref.agent_id) is child_run:
                    self._active_runs.pop(record.agent_ref.agent_id, None)
            if record.worktree_path is None:
                with self._pool_lock:
                    record.running = False
            if child_run.phase in (RunPhase.RUNNING, RunPhase.PAUSED):
                if run_result is None:
                    if token.cancelled:
                        child_run.cancel()
                    else:
                        child_run.fail(failure or "subagent dispatch failed")
                elif run_result.stopped_reason == "cancelled":
                    child_run.cancel()
                else:
                    if child_run.phase is RunPhase.PAUSED:
                        child_run.resume()
                        pause_gate.resume()
                    child_run.finish(run_result)
            if self._event_run_id is not None:
                emit(
                    self._events,
                    SubagentStopped(
                        agent_id=self._event_agent_id,
                        run_id=self._event_run_id,
                        turn_id=self._event_turn_id,
                        subagent_name=subagent_name,
                        subagent_agent_id=record.agent_ref.agent_id,
                    ),
                )
        if run_result is None:
            try:
                content, payload = self._worktree_output(subagent_name, record, "")
            finally:
                with self._pool_lock:
                    record.running = False
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                content=content,
                error=failure or str(execution_error) or "subagent dispatch failed",
                payload=payload,
            )
        record.messages = run_result.messages
        record.turns_used += run_result.turns_used
        for model, usage in run_result.usage_by_model.items():
            record.usage_by_model[model] = record.usage_by_model.get(
                model, UsageTotals()
            ).merged(usage)
        if run_result.stopped_reason == "cancelled":
            if token.reason is CancelReason.PARENT:
                with self._pool_lock:
                    record.running = False
                raise OperationCancelled
            record.breaker.record_failure()
            if token.reason is CancelReason.DEADLINE:
                error = "subagent deadline elapsed before a final answer"
            else:
                error = "subagent was cancelled explicitly before a final answer"
            try:
                content, payload = self._worktree_output(
                    subagent_name, record, run_result.final_response.message.text
                )
            finally:
                with self._pool_lock:
                    record.running = False
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                content=content,
                error=error,
                payload=payload,
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
        try:
            content, payload = self._worktree_output(
                subagent_name, record, run_result.final_response.message.text
            )
        finally:
            if record.worktree_path is not None:
                with self._pool_lock:
                    record.running = False
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=succeeded,
            content=content,
            error=error,
            payload=payload,
        )

    @staticmethod
    def _worktree_output(
        subagent_name: str,
        record: SubagentRecord,
        output: str,
    ) -> tuple[str, dict | None]:
        if record.worktree_path is None:
            return output, None
        diff = worktree_diff(record.worktree_path)
        payload = {
            "kind": "worktree_diff",
            "subagent": subagent_name,
            "worktree": str(record.worktree_path),
            "files": list(diff.files),
        }
        if not diff.files:
            summary = f"Worktree {record.worktree_path}: no changes"
        else:
            patch = diff.patch
            truncation_note = ""
            if len(patch) > MAX_READ_BYTES:
                patch = patch[:MAX_READ_BYTES]
                truncation_note = f"\n[patch truncated at {MAX_READ_BYTES} characters]"
            summary = (
                f"Worktree {record.worktree_path}: {len(diff.files)} files changed\n"
                + "\n".join(diff.files)
                + ("\n" + patch if patch else "")
                + truncation_note
            )
        content = f"{output}\n\n{summary}" if output else summary
        return content, payload


@dataclass
class LeaderConfig:
    """User-chosen configuration for a leader conversation.

    The leader never picks its own or its subagents' provider/model --
    they are set by the caller. A host may update the leader's model and effort
    between turns without rebuilding its tool registry.
    Each subagent independently accounts against ``subagent_budget``; there is
    no shared drawdown. Optional ``subagent_specs`` declare named child roles;
    omitting them preserves the legacy synthesized defaults.
    """

    leader_provider: ModelProvider
    subagent_provider: ModelProvider
    repo_root: str
    max_leader_turns: int = DEFAULT_MAX_TURNS
    leader_budget: RunBudget | None = None
    max_subagents: int = DEFAULT_MAX_SUBAGENTS
    subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS
    subagent_budget: RunBudget | None = None
    subagent_tool_names: Sequence[str] | None = None
    subagent_specs: Mapping[str, AgentSpec] | None = None
    permission_mode: PermissionMode = "allow"
    approval_callback: ApprovalCallback | None = None
    chat_token_budget: int | None = None
    chat_recent_turns: int = DEFAULT_RECENT_TURNS
    events: EventSink | None = None
    max_consecutive_compaction_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    max_consecutive_subagent_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    extensions: Extensions | None = None
    stream: bool = False
    result_store: ToolResultStore | None = None
    search_backend: SearchBackend | None = None
    extra_tools: Mapping[str, LocalTool] | None = None
    leader_policy: PermissionPolicy | None = None
    leader_model: str | None = None
    leader_effort: str | None = None
    hook_runner: HookRunner | None = None
    memory: AgentMemory | None = None
    model_summary: bool = False
    checkpoints: CheckpointStore | None = None
    leader_tools: Mapping[str, LocalTool] | None = None
    lsp: LspManager | None = None


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


class _LeaderTranscript:
    def __init__(self, writer) -> None:  # noqa: ANN001
        self._writer = writer
        self._checkpoint_key: str | None = None
        self._lock = threading.Lock()

    def begin_checkpoint(self, key: str) -> None:
        with self._lock:
            self._checkpoint_key = key

    def append(
        self,
        record_type: str,
        *,
        run_id: str,
        agent_id: str,
        turn_id: str | None,
        data: dict,
    ) -> str:
        record_id = self._writer.append(
            record_type,
            run_id=run_id,
            agent_id=agent_id,
            turn_id=turn_id,
            data=data,
        )
        if record_type == "run_started":
            with self._lock:
                key = self._checkpoint_key
                self._checkpoint_key = None
            if key is not None:
                self._writer.append(
                    "checkpoint",
                    run_id=run_id,
                    agent_id=agent_id,
                    turn_id=None,
                    data={"key": key},
                )
        return record_id


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
        self._hook_runner = config.hook_runner or (
            None
            if config.extensions is None
            else config.extensions.hook_runner(cwd=Path(config.repo_root))
        )
        self._event_sink = _LeaderEventSink(
            fan_out(config.events, self._hook_runner)
        )
        self._last_run_id: str | None = None
        self._transcript = (
            None
            if session is None
            else _LeaderTranscript(
                session.writer_for(self._agent_ref.agent_id, is_root=True)
            )
        )
        leader_policy = config.leader_policy or PermissionPolicy(
            repo_root=config.repo_root,
            mode=config.permission_mode,
            approval_callback=config.approval_callback,
        )
        self._leader_policy = leader_policy
        self._leases = WorkspaceLeases(config.repo_root)
        defined_leader = None if config.subagent_specs is None else config.subagent_specs.get("leader")
        self._leader_prompt = "" if defined_leader is None else defined_leader.prompt
        leader_provider_name = getattr(config.leader_provider, "name", None) or "unknown"
        self._leader_model_locked = bool(
            defined_leader is not None
            and (
                defined_leader.model.model is not None
                or defined_leader.model.provider != leader_provider_name
            )
        )
        self._leader_effort_locked = bool(
            defined_leader is not None and defined_leader.model.effort is not None
        )
        self._leader_model = (
            config.leader_model
            if defined_leader is None or defined_leader.model.model is None
            else defined_leader.model.model
        )
        self._leader_effort = (
            config.leader_effort
            if defined_leader is None or defined_leader.model.effort is None
            else defined_leader.model.effort
        )
        self._leader_run: AgentRun | None = None
        self._leader_pause_gate: PauseGate | None = None
        self._control_lock = threading.RLock()
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
            hooks=self._hook_runner,
            stream=config.stream,
            result_store=config.result_store,
            search_backend=config.search_backend,
            extra_tools=config.extra_tools,
            memory=config.memory,
            skills=(None if config.extensions is None else config.extensions.skills),
            checkpoints=config.checkpoints,
            lsp=config.lsp,
        )
        self._event_sink.bind_dispatch_tool(self._dispatch_tool)
        leader_tools = {DISPATCH_TOOL_NAME: self._dispatch_tool}
        tool_names = None if defined_leader is None else defined_leader.tool_names
        if tool_names is not None and config.result_store is not None:
            tool_names = (*tool_names, "read_tool_result")
        leader_memory_tool = (
            MemoryTool(
                config.memory,
                "leader",
                self._leader_run_id,
            )
            if defined_leader is not None and defined_leader.memory.enabled
            else None
        )
        standard_tools = merge_tool_registry(
            standard_tool_registry(
                tool_names,
                result_store=config.result_store,
                search_backend=config.search_backend,
                memory_tool=leader_memory_tool,
                skills=(None if config.extensions is None else config.extensions.skills),
                checkpoints=config.checkpoints,
                lsp=config.lsp,
            ),
            config.extra_tools,
        )
        leader_tools.update(standard_tools)
        leader_only_tools = dict(config.leader_tools or {})
        leader_tools.update(leader_only_tools)
        provider_name = getattr(config.leader_provider, "name", None) or "unknown"
        self._leader_spec = AgentSpec(
            name="leader",
            prompt=self._leader_prompt,
            model=ModelSelector(
                provider=provider_name,
                model=(
                    self._leader_model
                    if self._leader_model is not None
                    else getattr(config.leader_provider, "model", None)
                ),
                effort=self._leader_effort,
            ),
            policy_ceiling=leader_policy,
            tool_names=tuple(leader_tools),
            budget=config.leader_budget,
            memory=(
                defined_leader.memory
                if defined_leader is not None
                else MemorySettings()
            ),
            call_class=CallClass.FOREGROUND,
            max_depth=0,
        )
        self._agent = ApiAgent(
            provider=config.leader_provider,
            tools=leader_tools,
            policy=leader_policy,
            max_turns=config.max_leader_turns,
            budget=config.leader_budget,
            tool_schemas=[
                dispatch_subagent_tool_schema(
                    config.leader_provider.wire_format, config.subagent_specs,
                ),
                *tool_registry_schemas(
                    standard_tools,
                    config.leader_provider.wire_format,
                ),
                *tool_registry_schemas(
                    leader_only_tools,
                    config.leader_provider.wire_format,
                ),
            ],
            agent_ref=self._agent_ref,
            events=self._event_sink,
            stream=config.stream,
            result_store=config.result_store,
            call_class=CallClass.FOREGROUND,
            transcript=(
                self._transcript
            ),
        )
        self._chat_messages = self._initial_leader_messages()
        self._compaction_usage_by_model: dict[str, UsageTotals] = {}
        self._automatic_compaction_breaker = ConsecutiveFailureBreaker(
            "automatic compaction",
            max_consecutive_failures=config.max_consecutive_compaction_failures,
        )
        self._context_overflow_repair_failed = False
        self._token_ratio = 1.0

    @property
    def subagents(self) -> dict[str, SubagentRecord]:
        return self._dispatch_tool.pool

    @property
    def agent_ref(self) -> AgentRef:
        return self._agent_ref

    @property
    def chat_token_budget(self) -> int:
        """The explicit or model-derived budget currently used for chat compaction."""
        base = (
            self._config.chat_token_budget
            if self._config.chat_token_budget is not None
            else budget_for_model(
                self._config.leader_provider.wire_format,
                self._leader_spec.model.model,
            )
        )
        return math.floor(base / self._token_ratio)

    def _stopped_repairs(self) -> tuple[str, ...]:
        with self._dispatch_tool._pool_lock:
            subagent_breakers = [
                record.breaker for record in self._dispatch_tool.pool.values()
            ]
        breakers = [
            self._automatic_compaction_breaker,
            *subagent_breakers,
        ]
        return tuple(sorted(breaker.name for breaker in breakers if breaker.is_open))

    def _leader_run_id(self) -> str | None:
        return None if self._leader_run is None else self._leader_run.run.run_id

    def select_model(self, model: str | None, effort: str | None) -> None:
        """Apply a host model choice to the next request without rebuilding tools."""
        self._token_ratio = 1.0
        self._config.leader_model = model
        self._config.leader_effort = effort
        if not self._leader_model_locked:
            self._leader_model = model
        if not self._leader_effort_locked:
            self._leader_effort = effort
        actual_model = self._leader_model
        if actual_model is None:
            actual_model = getattr(self._config.leader_provider, "model", None)
        self._leader_spec = replace(
            self._leader_spec,
            model=replace(
                self._leader_spec.model,
                model=actual_model,
                effort=self._leader_effort,
            ),
        )

    def _initial_leader_messages(self) -> list[Message]:
        if not self._leader_spec.memory.enabled:
            return (
                [Message(role=Role.SYSTEM, content=self._leader_prompt)]
                if self._leader_prompt
                else []
            )
        messages = seed_messages(
            self._leader_spec,
            "",
            memory=_memory_entries(self._config.memory, self._leader_spec),
        )
        return messages[:-1]

    def _run_messages(
        self, messages: list[Message], *, cancel: CancellationToken | None = None
    ) -> LeaderRunResult:
        leader_run = new_agent_run(self._leader_spec)
        leader_run.agent = self._agent_ref
        leader_run.run = RunRef(
            run_id=leader_run.run.run_id,
            agent_id=self._agent_ref.agent_id,
        )
        leader_token = cancel if cancel is not None else CancellationToken()
        leader_pause_gate = PauseGate()
        leader_run.start(leader_token)
        self._leader_run = leader_run
        self._leader_pause_gate = leader_pause_gate
        self._dispatch_tool.set_parent_context(leader_run, list(messages))
        try:
            result = self._agent.run(
                messages,
                model=self._leader_model,
                effort=self._leader_effort,
                cancel=leader_token,
                run=leader_run,
                pause=leader_pause_gate,
                hooks=self._hook_runner,
            )
        except OperationCancelled:
            if leader_run.phase in (RunPhase.RUNNING, RunPhase.PAUSED):
                leader_run.cancel()
            raise
        except Exception as exc:
            leader_run.fail(str(exc))
            raise
        if result.stopped_reason == "cancelled":
            if leader_run.phase in (RunPhase.RUNNING, RunPhase.PAUSED):
                leader_run.cancel()
        else:
            if leader_run.phase is RunPhase.PAUSED:
                leader_run.resume()
                leader_pause_gate.resume()
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

    def _begin_checkpoint(self) -> None:
        if self._config.checkpoints is None:
            return
        key = new_id("chk")
        self._config.checkpoints.begin(key)
        if self._transcript is not None:
            self._transcript.begin_checkpoint(key)

    def control_agent(self, agent_id: str, action: str, text: str | None = None) -> str:
        """Control one currently live run by its stable agent id."""
        if action not in ("pause", "resume", "redirect", "stop"):
            raise ValueError(f"unknown agent action {action!r}")
        if action == "redirect" and (not isinstance(text, str) or not text.strip()):
            raise ValueError("redirect text must not be blank")
        with self._control_lock:
            run = None
            gate = None
            if agent_id == self._agent_ref.agent_id:
                run, gate = self._leader_run, self._leader_pause_gate
            else:
                with self._dispatch_tool._pool_lock:
                    for record in self._dispatch_tool.pool.values():
                        if record.agent_ref.agent_id == agent_id:
                            run = record.runs[-1] if record.runs else None
                            gate = record.pause_gate
                            break
            if run is None or run.phase not in (RunPhase.RUNNING, RunPhase.PAUSED):
                raise AgentControlError(f"agent {agent_id!r} is not running", status=404)
            if action == "pause":
                if run.phase is RunPhase.PAUSED:
                    raise AgentControlError(f"agent {agent_id!r} is already paused")
                assert gate is not None
                gate.pause()
                try:
                    run.pause()
                except ValueError as exc:
                    gate.resume()
                    raise AgentControlError(str(exc)) from exc
                return "paused"
            if action == "resume":
                if run.phase is not RunPhase.PAUSED:
                    raise AgentControlError(f"agent {agent_id!r} is not paused")
                assert gate is not None
                run.resume()
                gate.resume()
                return "running"
            if action == "redirect":
                try:
                    run.redirect(text or "")
                except ValueError as exc:
                    raise AgentControlError(str(exc)) from exc
                return run.phase.value
            run.cancel()
            assert gate is not None
            gate.resume()
            return "stopping"

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
        self._begin_checkpoint()
        self.clear_subagents()
        messages = self._initial_leader_messages()
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

        self._chat_messages = self._initial_leader_messages()
        self._automatic_compaction_breaker.reset()
        self._context_overflow_repair_failed = False
        return self.clear_subagents()

    def seed_chat(self, messages: Sequence[Message], *, persisted: bool = False) -> None:
        """Set the history used by the next chat without changing subagents."""

        self._chat_messages = (
            list(messages)
            if persisted
            else [*self._initial_leader_messages(), *messages]
        )
        if persisted:
            self._agent._persisted_digests = [
                _message_digest(message) for message in messages
            ]

    def clear_subagents(self) -> int:
        """Clear all dispatched subagents and return how many were removed."""

        with self._dispatch_tool._pool_lock:
            count = len(self._dispatch_tool.pool)
            self._dispatch_tool.pool.clear()
            return count

    def forget_subagent(self, name: str) -> bool:
        """Remove one named subagent from the current leader pool."""
        with self._dispatch_tool._pool_lock:
            return self._dispatch_tool.pool.pop(name, None) is not None

    def compact_chat(self, *, cancel: CancellationToken | None = None) -> CompactionResult:
        """Apply context compaction to the persisted multi-turn chat state."""

        result = self._compact_chat_to_budget(
            self.chat_token_budget,
            cancel=cancel,
        )
        self._context_overflow_repair_failed = False
        return result

    def force_compact_chat(
        self,
        instructions: str | None = None,
        *,
        cancel: CancellationToken | None = None,
    ) -> tuple[CompactionResult, dict[str, UsageTotals]]:
        """Compact the earlier conversation now and return summary usage."""
        self._compaction_usage_by_model.clear()
        try:
            result = self._compact_chat_to_budget(
                self.chat_token_budget,
                cancel=cancel,
                recent_turns=1,
                force=True,
                instructions=instructions,
            )
            self._context_overflow_repair_failed = False
            return result, dict(self._compaction_usage_by_model)
        finally:
            self._compaction_usage_by_model.clear()

    def _write_model_summary(
        self,
        dropped: list[Message],
        *,
        cancel: CancellationToken | None = None,
        instructions: str | None = None,
    ) -> str:
        model = self._leader_spec.model.model
        system_prompt = MODEL_SUMMARY_PROMPT
        if instructions is not None:
            system_prompt += (
                "\n\nAdditional instructions from the user:\n" + instructions
            )
        request = ModelRequest(
            messages=[
                Message(Role.SYSTEM, system_prompt),
                Message(Role.USER, render_dropped_messages(dropped)),
            ],
            model=model,
            tools=[],
            max_tokens=20_000,
            call_class=CallClass.BACKGROUND,
        )
        if self._config.stream:
            chunks = (
                self._config.leader_provider.create_response_stream(request)
                if cancel is None
                else self._config.leader_provider.create_response_stream(
                    request, cancel=cancel
                )
            )
            assembler = StreamAssembler()
            for chunk in chunks:
                assembler.add(chunk)
            response = assembler.finish()
        elif cancel is None:
            response = self._config.leader_provider.create_response(request)
        else:
            response = self._config.leader_provider.create_response(
                request, cancel=cancel
            )
        model_key = model or getattr(self._config.leader_provider, "model", None) or "unknown"
        totals = UsageTotals.from_usage(response.usage)
        self._compaction_usage_by_model[model_key] = self._compaction_usage_by_model.get(
            model_key, UsageTotals()
        ).merged(totals)
        return response.message.text

    def _compact_chat_to_budget(
        self,
        budget: int,
        *,
        cancel: CancellationToken | None = None,
        record_success: bool = True,
        recent_turns: int | None = None,
        force: bool = False,
        instructions: str | None = None,
    ) -> CompactionResult:
        compact_kwargs = {
            "budget": budget,
            "recent_turns": (
                self._config.chat_recent_turns
                if recent_turns is None
                else recent_turns
            ),
            "cancel": cancel,
        }
        if force:
            compact_kwargs["force"] = True
        if self._config.model_summary:
            compact_kwargs["summarize"] = lambda dropped: self._write_model_summary(
                dropped, cancel=cancel, instructions=instructions
            )
        result = compact_messages_for_budget(self._chat_messages, **compact_kwargs)
        if result.changed:
            stripped_messages: list[Message] = []
            for message in result.messages:
                tool_calls = []
                for call in message.tool_calls:
                    if "anthropic_content" not in call.provider_metadata:
                        tool_calls.append(call)
                        continue
                    metadata = dict(call.provider_metadata)
                    del metadata["anthropic_content"]
                    tool_calls.append(replace(call, provider_metadata=metadata))
                stripped_messages.append(
                    replace(message, tool_calls=tool_calls)
                    if tool_calls != message.tool_calls
                    else message
                )
            result = replace(result, messages=stripped_messages)
        self._chat_messages = result.messages
        if record_success:
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

    def _compact_after_context_overflow(
        self,
        overflow: ContextLengthExceededError,
        *,
        cancel: CancellationToken | None = None,
    ) -> bool:
        before_tokens = estimate_messages_tokens(self._chat_messages)
        if overflow.actual_tokens is not None:
            forced_budget = min(
                self.chat_token_budget,
                max(1, before_tokens - 1),
            )
        else:
            forced_budget = min(
                self.chat_token_budget,
                max(1, before_tokens * 3 // 4),
            )
        user_turns = sum(
            message.role is Role.USER for message in self._chat_messages
        )
        starting_recent_turns = min(
            self._config.chat_recent_turns,
            max(1, user_turns),
        )
        last_error: ContextCompactionError | None = None
        for recent_turns in range(starting_recent_turns, 0, -1):
            try:
                return self._compact_chat_to_budget(
                    forced_budget,
                    cancel=cancel,
                    record_success=False,
                    recent_turns=recent_turns,
                ).changed
            except ContextCompactionError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _automatic_compact_chat(
        self, *, cancel: CancellationToken | None = None
    ) -> None:
        if self._automatic_compaction_breaker.is_open:
            return
        try:
            self._compact_chat_to_budget(
                self.chat_token_budget,
                cancel=cancel,
                record_success=not self._context_overflow_repair_failed,
            )
        except ContextCompactionError:
            self._automatic_compaction_breaker.record_failure()
        except OperationCancelled:
            # Cancellation is a caller decision, not a failed repair attempt.
            pass

    def chat(
        self, message: ContentInput, *, cancel: CancellationToken | None = None
    ) -> LeaderRunResult:
        """Continue an ongoing conversation with the leader.

        Unlike `run()`, this keeps its own running message history across
        calls -- a later `chat()` call includes everything said in earlier
        ones, so the leader (and its view of already-dispatched subagents)
        has full context of the conversation so far.
        """
        self._begin_checkpoint()
        self._chat_messages.append(Message(role=Role.USER, content=message))
        self._compaction_usage_by_model.clear()
        self._automatic_compact_chat(cancel=cancel)
        recovered_overflow = False
        try:
            result = self._run_messages(self._chat_messages, cancel=cancel)
        except ContextLengthExceededError as overflow:
            before_tokens = estimate_messages_tokens(self._chat_messages)
            request_tokens = overflow.request_tokens
            denominator = (
                request_tokens
                if request_tokens is not None and request_tokens > 0
                else before_tokens
            )
            if overflow.actual_tokens is not None and denominator > 0:
                self._token_ratio = max(
                    self._token_ratio,
                    overflow.actual_tokens / denominator,
                )
            if self._automatic_compaction_breaker.is_open:
                raise
            try:
                changed = self._compact_after_context_overflow(
                    overflow,
                    cancel=cancel,
                )
            except ContextCompactionError:
                self._automatic_compaction_breaker.record_failure()
                self._context_overflow_repair_failed = True
                raise overflow from None
            if not changed:
                self._automatic_compaction_breaker.record_failure()
                self._context_overflow_repair_failed = True
                raise
            try:
                result = self._run_messages(self._chat_messages, cancel=cancel)
            except ContextLengthExceededError:
                self._automatic_compaction_breaker.record_failure()
                self._context_overflow_repair_failed = True
                raise
            recovered_overflow = True
        if recovered_overflow or self._context_overflow_repair_failed:
            self._automatic_compaction_breaker.record_success()
            self._context_overflow_repair_failed = False
        self._chat_messages = result.leader_messages
        self._automatic_compact_chat(cancel=cancel)
        leader_usage = result.usage_by_agent.setdefault(
            self._agent_ref.agent_id, {}
        )
        for model, usage in self._compaction_usage_by_model.items():
            leader_usage[model] = leader_usage.get(model, UsageTotals()).merged(usage)
        self._compaction_usage_by_model.clear()
        result.stopped_repairs = self._stopped_repairs()
        return result
