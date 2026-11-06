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

from symphonai_api.models import _without_vendor_state

import json
from contextlib import nullcontext
import math
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from symphonai_api.agent_loop import ApiAgent
from symphonai_api.serialization import message_digest
from symphonai_api.agent_memory import AgentMemory, MemoryEntry, MemorySettings
from symphonai_api.agent_run import (
    AgentRun,
    RunNode,
    RunPhase,
    new_agent_run,
    read_run_graph,
)
from symphonai_api.agent_spec import AgentSpec, Isolation, ModelSelector, validate_input, validate_output
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
    SubagentNotice,
    PersonNotice,
    ToolCallStarted,
    emit,
    fan_out,
)
from symphonai_api.extensions import Extensions
from symphonai_api.hooks import HookRunner
from symphonai_api.identity import AgentRef, RunRef, new_agent_ref, new_id
from symphonai_api.jobs import JobManager
from symphonai_api.leases import LeaseConflict, WorkspaceLeases
from symphonai_api.lsp import LspManager
from symphonai_api.models import ContentInput, Message, ModelRequest, Role, ToolCall, ToolResult
from symphonai_api.memory_extract import MemoryExtractor
from symphonai_api.permissions import ApprovalCallback, DenialReason, PermissionMode, PermissionPolicy
from symphonai_api.providers.base import ContextLengthExceededError, ModelProvider
from symphonai_api.model_table import capability_for_model, capability_for_provider, accepts_effort
from symphonai_api.runner import merge_tool_registry, standard_tool_registry
from symphonai_api.spec_text import spec_validation
from symphonai_api.tools.shell import shell_executable
from symphonai_api.roles import (
    EXPLORER_PROMPT,
    IMPLEMENTER_PROMPT,
    PLANNER_PROMPT,
    REVIEWER_PROMPT,
    ROLE_GUIDANCE,
    WORKER_PROMPT,
)
from symphonai_api.session import SessionStore, TranscriptError, load_run, read_records
from symphonai_api.secrets import mask_secrets, mask_tool_result
from symphonai_api.skills import Skill
from symphonai_api.streaming import StreamAssembler
from symphonai_api.tool_schema import tool_registry_schemas
from symphonai_api.tool_results import ToolResultStore
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.filesystem import MAX_READ_BYTES
from symphonai_api.tools.memory import MemoryTool
from symphonai_api.tools.metadata import FILE_EDIT_TOOLS, ToolMetadata, changed_paths
from symphonai_api.tools.read_ledger import ReadLedger
from symphonai_api.web_search import SearchBackend
from symphonai_api.worktree import WorktreeError, create_worktree, worktree_diff


def _run_handoff_validation(
    commands: Sequence[str], policy: PermissionPolicy,
) -> list[dict[str, object]]:
    from symphonai_api.process import run_check_command
    results: list[dict[str, object]] = []
    for index, command in enumerate(commands):
        # Check unconditional refusals without requesting approval twice. The
        # real tool still enforces the original policy before running anything.
        decision = replace(policy, mode="allow").check_shell(
            [shell_executable(), "-c", command], cwd=policy.repo_root,
        )
        if not decision.allowed and decision.denial in (
            DenialReason.UNCHECKABLE_COMMAND, DenialReason.ALWAYS_DENY,
        ):
            results.append({
                "command": command, "exit": 126, "output": decision.reason,
                "broken": True,
            })
            break
        result = run_check_command([shell_executable(), "-c", command], policy.repo_root,
                                   policy=policy, enforce_policy=True, sandboxed=True,
                                   timeout=min(120, policy.shell_timeout_seconds), output_limit=4000)
        code = result["exit"] if result["exit"] is not None else 126
        output = result["output"]
        results.append({
            "command": command, "exit": code, "output": output[-4000:],
            "broken": code == 127,
        })
        if code != 0:
            break
    return results


class AgentControlError(RuntimeError):
    """A live agent could not accept a control request."""

    def __init__(self, message: str, *, status: int = 409) -> None:
        super().__init__(message)
        self.status = status




def _session_subagent_transcripts(
    session: SessionStore,
) -> dict[str, list[tuple[str, str, Path]]]:
    transcripts: dict[str, list[tuple[str, str, Path]]] = {}
    for path in session.directory.glob("agent-*.jsonl"):
        try:
            records, _ = read_records(path)
        except (OSError, TranscriptError):
            continue
        first_start = next(
            (record for record in records if record.get("type") == "run_started"),
            None,
        )
        if first_start is None:
            continue
        data = first_start.get("data")
        name = data.get("agent_name") if isinstance(data, dict) else None
        if isinstance(name, str):
            agent_id = path.name[len("agent-") : -len(".jsonl")]
            transcripts.setdefault(name, []).append(
                (str(first_start.get("ts", "")), agent_id, path)
            )
    return transcripts

DISPATCH_TOOL_NAME = "dispatch_subagent"
DEFAULT_MAX_SUBAGENTS = 5
EXPLORER_TOOL_NAMES = ("read_file", "glob", "grep", "list_files", "web_fetch")


def _memory_seed(
    store: AgentMemory | None,
    spec: AgentSpec,
) -> tuple[tuple[MemoryEntry, ...], tuple[str, ...]]:
    if store is None or not spec.memory.enabled:
        return (), ()
    limit = spec.memory.max_entries
    global_entries, global_unreadable = store.read_status("global")
    project_entries, project_unreadable = store.read_status("project")
    entries = (
        *reversed(global_entries[-limit:]),
        *reversed(project_entries[-limit:]),
    )
    unreadable = tuple(
        scope
        for scope, failed in (
            ("global", global_unreadable),
            ("project", project_unreadable),
        )
        if failed
    )
    return entries, unreadable


def builtin_subagent_specs(
    provider: ModelProvider,
    policy: PermissionPolicy,
    search_backend: SearchBackend | None = None,
    skills: Mapping[str, Skill] | None = None,
    lsp: LspManager | None = None,
    jobs: JobManager | None = None,
    *,
    model_override: str | None = None,
    role_selectors: Mapping[str, ModelSelector] | None = None,
) -> dict[str, AgentSpec]:
    job_tools = ("job_output", "job_stop") if jobs is not None else ()
    provider_name = provider.name or "unknown"

    def selector(name: str) -> ModelSelector:
        configured = None if role_selectors is None else role_selectors.get(name)
        model = model_override or provider.model
        if configured is not None:
            model = configured.model or (model if configured.provider == provider_name else None)
            if configured.effort is not None:
                return replace(configured, model=model)
            return replace(configured, model=model, effort=_default_effort(
                provider, model, "low", provider_name=configured.provider,
            ))
        return ModelSelector(provider=provider_name, model=model,
                             effort=_default_effort(provider, model, "low"))

    specs = {
        name: AgentSpec(
            name=name,
            prompt={"worker": WORKER_PROMPT, "explorer": EXPLORER_PROMPT}[name],
            model=selector(name),
            policy_ceiling=policy,
            tool_names=tools,
            mcp_tool_names=None if name == "worker" else (),
            call_class=CallClass.BACKGROUND,
        )
        for name, tools in (
            ("worker", tuple(standard_tool_registry(search_backend=search_backend, skills=skills, lsp=lsp, jobs=jobs))),
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
            name="planner", prompt=PLANNER_PROMPT, model=selector("planner"),
            policy_ceiling=policy,
            tool_names=EXPLORER_TOOL_NAMES + ("run_shell", "write_file", "edit_file") + optional,
            mcp_tool_names=(),
            call_class=CallClass.BACKGROUND,
        ),
        "implementer": AgentSpec(
            name="implementer", prompt=IMPLEMENTER_PROMPT, model=selector("implementer"),
            policy_ceiling=policy,
            tool_names=tuple(standard_tool_registry(search_backend=search_backend, skills=skills, lsp=lsp, jobs=jobs)),
            mcp_tool_names=None,
            call_class=CallClass.BACKGROUND,
        ),
        "reviewer": AgentSpec(
            name="reviewer", prompt=REVIEWER_PROMPT, model=selector("reviewer"),
            policy_ceiling=policy,
            tool_names=EXPLORER_TOOL_NAMES + ("run_shell", "write_file") + optional + job_tools,
            mcp_tool_names=(),
            call_class=CallClass.BACKGROUND,
        ),
    })
    return specs


def _default_effort(provider: ModelProvider, model: str | None, level: str,
                    *, provider_name: str | None = None) -> str | None:
    capability = (capability_for_provider(provider_name, model)
                  if provider_name is not None and provider_name != provider.name
                  else capability_for_model(provider.wire_format, model))
    return level if accepts_effort(capability, level) else None


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
    "same subagent. Set isolation to worktree only for a task that should "
    "start from the last commit and stay apart from the person's tree. Such "
    "a subagent does not see uncommitted changes, including yours, and its "
    "diff comes back in the result for you to apply. Explorers, and "
    "follow-ups on uncommitted work, share the tree."
)
def _dispatch_description(subagents: Mapping[str, AgentSpec] | None = None) -> str:
    names = ROLE_GUIDANCE if subagents is None else {
        name: ROLE_GUIDANCE.get(name, "") for name in subagents
    }
    available = []
    for name, purpose in names.items():
        spec = None if subagents is None else subagents[name]
        description = spec.description if spec is not None else None
        purpose = description or purpose
        entry = f"{name}: {purpose}." if purpose else name
        if spec is not None and spec.io.input_schema is not None:
            def json_value(value):
                if isinstance(value, Mapping):
                    return {key: json_value(item) for key, item in value.items()}
                if isinstance(value, (tuple, list)):
                    return [json_value(item) for item in value]
                return value
            schema = json.dumps(json_value(spec.io.input_schema), separators=(",", ":"))
            entry += f" (task must be JSON matching {schema})"
        available.append(entry)
    return _DISPATCH_DESCRIPTION + " Available subagents: " + "; ".join(available) + "."
SUBAGENT_FINISHED_PREFIX = "[subagent {name} finished]"
SUBAGENT_PROGRESS_PREFIX = "[subagent {name} progress]"
SUBAGENT_QUESTION_PREFIX = "[subagent {name} asks]"
LEADER_MESSAGE_PREFIX = "[message from the leader]"
SUBAGENT_NOTICE_PATTERN = re.compile(
    r"^(?:\[subagent .+ (?:finished|progress|asks)\]|"
    + re.escape(LEADER_MESSAGE_PREFIX) + r")(?:\n|$)"
)

_DISPATCH_PROPERTIES = {
    "background": {
        "type": "boolean",
        "description": "Start in the background and collect the result with wait_subagents.",
    },
    "subagent_name": {
        "minLength": 1,
        "type": "string",
        "description": (
            "Choose a configured subagent: the built-in roles are explorer, "
            "implementer, planner, reviewer, and worker. Dispatching the same "
            "name again continues that subagent's conversation."
        ),
    },
    "task": {
        "minLength": 1,
        "type": "string",
        "description": "The task or follow-up message to give this subagent.",
    },
    "isolation": {
        "type": "string",
        "enum": ["worktree"],
        "description": "Run the subagent in its own git worktree, from the last commit.",
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
    if subagents is not None:
        parameters["properties"] = {
            **parameters["properties"],
            "subagent_name": {
                **parameters["properties"]["subagent_name"],
                "enum": list(subagents),
                "description": (
                    "Choose one of the configured subagent names in the enum. "
                    "Dispatching the same name again continues that subagent's conversation."
                ),
            },
        }
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
        parameters = sanitize_for_gemini(parameters)
        if subagents is not None:
            parameters["properties"]["subagent_name"]["enum"] = list(subagents)
        return {
            "name": DISPATCH_TOOL_NAME,
            "description": description,
            "parameters": parameters,
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
    worktree_path: Path | None = None
    worktree_admin_path: Path | None = None
    running: bool = False
    state: str = "idle"
    background: bool = False
    result: ToolResult | None = None
    delivered: bool = False
    token: CancellationToken | None = None
    thread: threading.Thread | None = None
    current_run: AgentRun | None = None


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
        subagent_max_turns: int | None = None,
        subagent_base_prompt: str = "",
        subagent_tool_names: Sequence[str] | None = None,
        parent_agent_id: str | None = None,
        subagent_budget: RunBudget | None = None,
        compaction_limit: int | None = None,
        max_consecutive_subagent_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        session: SessionStore | None = None,
        subagent_specs: Mapping[str, AgentSpec] | None = None,
        leases: WorkspaceLeases | None = None,
        parent_run: AgentRun | None = None,
        hooks: HookRunner | None = None,
        stream: bool = False,
        result_store: ToolResultStore | None = None,
        search_backend: SearchBackend | None = None,
        extra_tools: Mapping[str, LocalTool] | None = None,
        memory: AgentMemory | None = None,
        skills: Mapping[str, Skill] | None = None,
        checkpoints: CheckpointStore | None = None,
        lsp: LspManager | None = None,
        jobs: JobManager | None = None,
        provider_for: Callable[[str, str | None], ModelProvider | None] | None = None,
        model_summary: bool = False,
    ) -> None:
        self._subagent_provider = subagent_provider
        self._leader_policy = leader_policy
        self._max_subagents = max_subagents
        self._subagent_max_turns = subagent_max_turns
        self._subagent_base_prompt = subagent_base_prompt
        self._subagent_tool_names = (
            None if subagent_tool_names is None else tuple(subagent_tool_names)
        )
        self._parent_agent_id = parent_agent_id
        # Every child receives the same immutable limits but tracks its own
        # spend; sharing drawdown needs phase 07's cross-agent coordination.
        self._subagent_budget = subagent_budget
        self._compaction_limit = compaction_limit
        self._max_consecutive_subagent_failures = max_consecutive_subagent_failures
        self._session = session
        self._subagent_specs = subagent_specs
        self._leases = leases or WorkspaceLeases(leader_policy.repo_root)
        self._parent_run = parent_run
        self._parent_messages: list[Message] = []
        self._hooks = hooks
        self._stream = stream
        self._result_store = result_store
        self._search_backend = search_backend
        self._extra_tools = extra_tools
        self._memory = memory
        self._skills = skills
        self._checkpoints = checkpoints
        self._lsp = lsp
        self._jobs = jobs
        self._provider_for = provider_for
        self._model_summary = model_summary
        self._pool_lock = threading.RLock()
        self._reserved_names: set[str] = set()
        self._background_names: set[str] = set()
        self._progress: dict[str, str] = {}
        self._questions: list[dict] = []
        self._changed = threading.Condition(self._pool_lock)
        self._active_runs_lock = threading.Lock()
        self._active_runs: dict[str, AgentRun] = {}
        self._events: EventSink | None = None
        self._event_agent_id = parent_agent_id or ""
        self._event_run_id: str | None = None
        self._event_turn_id: str | None = None
        self.pool: dict[str, SubagentRecord] = {}

    def _notice(self, name: str, record: SubagentRecord, kind: str, text: str) -> None:
        parent = self._parent_run
        if parent is not None:
            emit(self._events, SubagentNotice(
                agent_id=parent.agent.agent_id, run_id=parent.run.run_id,
                subagent_name=name, subagent_agent_id=record.agent_ref.agent_id,
                kind=kind, text=mask_secrets(text),
            ))

    def _collect(self, name: str, record: SubagentRecord) -> dict | None:
        # Called under _changed, including by deferred redirect delivery.
        if record.delivered or record.state == "closed" or record.result is None:
            return None
        record.delivered = True
        result = asdict(mask_tool_result(record.result))
        self._notice(name, record, "finished", json.dumps(result))
        return result

    def _deliver_result(self, name: str, record: SubagentRecord) -> str | None:
        with self._changed:
            result = self._collect(name, record)
            if result is None:
                return None
            return SUBAGENT_FINISHED_PREFIX.format(name=name) + "\n" + json.dumps(result)

    def wait_for_background(self, budget_exceeded) -> None:
        with self._changed:
            while self._background_names:
                if budget_exceeded():
                    return
                parent = self._parent_run
                if parent is not None and parent.token is not None:
                    parent.token.raise_if_cancelled()
                if parent is not None and parent.has_redirects():
                    return
                if any(r.result is not None and not r.delivered and r.state != "closed"
                       for r in self.pool.values() if r.background):
                    return
                self._changed.wait(0.05)

    def wait_subagents(self, names=None, timeout_seconds=600, *, cancel=None) -> dict:
        deadline = time.monotonic() + timeout_seconds
        with self._changed:
            selected = list(names) if names is not None else [
                name for name, record in self.pool.items()
                if record.background and (record.state == "running" or not record.delivered)
            ]
            unknown = [name for name in selected if name not in self.pool]
            if unknown:
                raise ValueError("unknown subagents: " + ", ".join(unknown))
            while True:
                if cancel is not None:
                    cancel.raise_if_cancelled()
                finished = {}
                for name in selected:
                    result = self._collect(name, self.pool[name])
                    if result is not None:
                        finished[name] = result
                questions = [item for item in self._questions
                             if item["name"] in selected and not item["delivered"] and not item["waited"]]
                for item in questions:
                    item["waited"] = True
                running = [name for name in selected if name in self._background_names]
                if questions:
                    return {"finished": finished, "running": running,
                            "questions": [{"name": item["name"], "text": item["text"]}
                                          for item in questions]}
                parent = self._parent_run
                wall_limit = None if parent is None else parent.budget.wall_seconds
                budget_elapsed = (wall_limit is not None and parent.started_monotonic is not None
                                  and time.monotonic() - parent.started_monotonic >= wall_limit)
                if finished or not running or time.monotonic() >= deadline or budget_elapsed:
                    return {"finished": finished, "running": running}
                self._changed.wait(min(0.05, max(0, deadline - time.monotonic())))

    def list_subagents(self) -> dict:
        with self._changed:
            return {"subagents": [
                {"name": name, "role": record.current_run.spec.name if record.current_run else record.agent_ref.name,
                 "state": record.state, "background": record.background,
                 "turns_used": record.turns_used,
                 "progress": self._progress.get(name, "")}
                for name, record in self.pool.items()
            ]}

    def close_subagent(self, name: str) -> dict:
        with self._changed:
            record = self.pool.get(name)
            if record is None:
                raise ValueError(f"unknown subagent {name}")
            record.state = "closed"
            record.delivered = True
            if record.token is not None:
                record.token.cancel()
            thread = record.thread
            self._changed.notify_all()
        if thread is not None:
            thread.join()
        return {"name": name, "state": "closed"}

    def cancel_background(self) -> None:
        with self._changed:
            names = list(self._background_names)
        for name in names:
            self.close_subagent(name)

    def _active_run_id(self, agent_id: str) -> str | None:
        with self._active_runs_lock:
            run = self._active_runs.get(agent_id)
            return None if run is None else run.run.run_id

    def set_parent_context(
        self,
        run: AgentRun,
        messages: list[Message],
    ) -> None:
        self._parent_run = run
        self._parent_messages = messages
        run.wait_for_background = self.wait_for_background

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
        name = arguments.get('subagent_name')
        with self._pool_lock:
            record = self.pool.get(name) if isinstance(name, str) else None
            isolated = arguments.get('isolation') == 'worktree' or (record is not None and record.worktree_path is not None)
        return ToolMetadata(concurrency_safe=isolated)


    def _spec_for(self, subagent_name: str) -> AgentSpec | None:
        if self._subagent_specs is not None:
            spec = self._subagent_specs.get(subagent_name)
            if spec is not None and spec.model.effort is None:
                model = spec.model.model or (
                    self._subagent_provider.model
                    if spec.model.provider == self._subagent_provider.name else None
                )
                return replace(spec, model=replace(spec.model, effort=_default_effort(
                    self._subagent_provider, model, "low", provider_name=spec.model.provider,
                )))
            return spec
        provider_name = self._subagent_provider.name or "unknown"
        model = self._subagent_provider.model
        return AgentSpec(
            name=subagent_name,
            prompt="",
            model=ModelSelector(
                provider=provider_name,
                model=model,
                effort=_default_effort(self._subagent_provider, model, "low"),
            ),
            policy_ceiling=self._leader_policy,
            tool_names=self._subagent_tool_names,
            budget=self._subagent_budget,
            isolation=Isolation(),
            call_class=CallClass.BACKGROUND,
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
        background = tool_call.arguments.get("background", False)
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
        if self._subagent_base_prompt:
            spec = replace(spec, prompt="\n\n".join(
                prompt for prompt in (self._subagent_base_prompt, spec.prompt) if prompt
            ))
        _, input_error = validate_input(spec.io, task)
        if input_error is not None:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=(
                    f"subagent {subagent_name} input does not match its contract: "
                    f"{input_error}"
                ),
            )
        selected_tool_names = (
            spec.tool_names if spec.tool_names is not None else self._subagent_tool_names
        )
        editable_tools = FILE_EDIT_TOOLS
        can_edit = selected_tool_names is None or bool(
            editable_tools.intersection(selected_tool_names)
        )
        document_role = subagent_name in ("planner", "reviewer")
        try:
            validation_commands = spec_validation(task) if can_edit and not document_role else []
        except ValueError as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"subagent {subagent_name}: {exc}",
            )
        if can_edit and not document_role and not validation_commands:
            defect = ("the task has no ## Validation heading" if not re.search(
                r"(?im)^##[ \t]+Validation[ \t]*(?:#+[ \t]*)?$", task,
            ) else "the ## Validation section has no fenced sh/bash block of commands")
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=(f"subagent {subagent_name}: {defect}. Put the spec in the task, with "
                       "## Validation and a fenced sh block of commands, or dispatch planner first to write one."),
            )
        available_extra = self._extra_tools or {}
        named_mcp_tools = spec.mcp_tool_names
        if named_mcp_tools is not None:
            unavailable = next((name for name in named_mcp_tools if name not in available_extra), None)
            if unavailable is not None:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error=f"subagent {subagent_name} names unavailable tool {unavailable}",
                )
            selected_extra = {name: available_extra[name] for name in named_mcp_tools}
        else:
            selected_extra = {
                name: tool for name, tool in available_extra.items()
                if name not in spec.denied_mcp_tool_names
            }
        child_provider = self._subagent_provider
        if spec.model.provider != self._subagent_provider.name:
            child_provider = (
                None if self._provider_for is None
                else self._provider_for(spec.model.provider, spec.model.model)
            )
            if child_provider is None:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error=(
                        f"subagent {subagent_name} needs provider "
                        f"{spec.model.provider}, which is not configured"
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
            if subagent_name in self._reserved_names or subagent_name in self._background_names or (record is not None and record.running):
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error=(
                        f"subagent {subagent_name} is already running; "
                        "dispatch it again after it finishes"
                    ),
                )
            if background and len(self._background_names) >= 4:
                return ToolResult(tool_call.id, ok=False, error=(
                    "at most 4 background subagents; running: "
                    + ", ".join(sorted(self._background_names))))
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

        worker_started = False
        token = None
        lease = None
        try:
            if record is not None and record.worktree_path is not None:
                effective_policy = effective_policy.rerooted(record.worktree_path)
                record.agent._policy = effective_policy
            resumed_messages = None
            resumed_agent_id = None
            if record is None and self._session is not None:
                forgotten = self._session.read_meta().get("forgotten_subagents", {})
                forgotten_ids = set(
                    forgotten.get(subagent_name, ())
                    if isinstance(forgotten, dict)
                    else ()
                )
                candidates = [
                    (started_at, agent_id, path)
                    for started_at, agent_id, path
                    in _session_subagent_transcripts(self._session).get(subagent_name, ())
                    if agent_id not in forgotten_ids
                ]
                if candidates:
                    _, resumed_agent_id, _ = max(candidates, key=lambda item: item[0])
                    resumed_messages = load_run(
                        self._session, agent_id=resumed_agent_id
                    ).messages

            worktree_admin_path = None
            worktree_root = None
            if record is None and self._session is not None:
                candidate_path = self._session.directory / "worktrees" / subagent_name
                if candidate_path.is_dir():
                    from symphonai_api.worktree import _top_level

                    relative_root = effective_policy.repo_root.resolve().relative_to(
                        _top_level(effective_policy.repo_root)
                    )
                    worktree_admin_path = candidate_path
                    worktree_root = candidate_path / relative_root
                    effective_policy = effective_policy.rerooted(worktree_root)
                elif wants_worktree:
                    worktree_admin_path = candidate_path
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

            if record is None:
                tool_names = selected_tool_names
                if tool_names is not None and self._result_store is not None:
                    tool_names = (*tool_names, "read_tool_result")
                child_jobs = self._jobs if not document_role and (
                    tool_names is None or "job_output" in tool_names
                ) else None
                optional_tools = {
                    "web_search": self._search_backend is not None,
                    "use_skill": bool(self._skills),
                    "lsp": self._lsp is not None and self._lsp.has_enabled_servers,
                    "read_tool_result": self._result_store is not None,
                    "job_output": child_jobs is not None,
                    "job_stop": child_jobs is not None,
                }
                if tool_names is not None:
                    tool_names = tuple(name for name in tool_names if optional_tools.get(name, True))
                agent_ref = AgentRef(
                    resumed_agent_id or new_agent_ref(
                        subagent_name, self._parent_agent_id
                    ).agent_id,
                    subagent_name,
                    self._parent_agent_id,
                )
                memory_tool = (
                    MemoryTool(
                        self._memory,
                        spec.name,
                        lambda agent_id=agent_ref.agent_id: self._active_run_id(agent_id),
                    )
                    if spec.memory.enabled and self._memory is not None
                    else None
                )
                read_ledger = ReadLedger()
                standard_tools = (
                    standard_tool_registry(
                        tool_names,
                        wire_format=child_provider.wire_format,
                        ledger=read_ledger,
                        result_store=self._result_store,
                        search_backend=self._search_backend,
                        memory_tool=memory_tool,
                        skills=self._skills,
                        lsp=self._lsp,
                        jobs=child_jobs,
                        checkpoints=(
                            None if worktree_root is not None else self._checkpoints
                        ),
                    )
                    if tool_names is None or tool_names
                    else (
                        {
                            memory_tool.name: memory_tool,
                            memory_tool.forget_tool.name: memory_tool.forget_tool,
                        }
                        if memory_tool is not None
                        else {}
                    )
                )
                subagent_tools = merge_tool_registry(
                    standard_tools,
                    selected_extra,
                )
                if document_role:
                    # Opaque tools and jobs must not bypass the document write boundary.
                    document_tools = set(EXPLORER_TOOL_NAMES) | FILE_EDIT_TOOLS | {
                        "run_shell", "web_search", "use_skill", "read_tool_result",
                    }
                    subagent_tools = {name: tool for name, tool in subagent_tools.items() if name in document_tools}
                    for name, tool in tuple(subagent_tools.items()):
                        if name in FILE_EDIT_TOOLS or name == "run_shell":
                            subagent_tools[name] = _DocumentRoleTool(tool, subagent_name)
                subagent_tools["message_leader"] = _MessageLeaderTool(self, subagent_name)
                record = SubagentRecord(
                    agent=ApiAgent(
                        provider=child_provider,
                        tools=subagent_tools,
                        policy=effective_policy,
                        max_turns=self._subagent_max_turns,
                        tool_schemas=tool_registry_schemas(subagent_tools, child_provider.wire_format),
                        agent_ref=agent_ref,
                        events=self._events,
                        stream=self._stream,
                        result_store=self._result_store,
                        read_ledger=read_ledger,
                        budget=spec.budget,
                        call_class=spec.call_class,
                        model_summary=self._model_summary,
                        context_budget=lambda model: budget_for_model(
                            child_provider.wire_format, model, self._compaction_limit,
                        ),
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
                if resumed_messages is not None:
                    if self._session.read_meta().get("provider_state_reset") is True:
                        record.messages = _without_vendor_state(resumed_messages)
                    else:
                        record.messages = resumed_messages
                with self._pool_lock:
                    self.pool[subagent_name] = record
                    self._reserved_names.discard(subagent_name)
            else:
                # The ceiling is mutable even though AgentSpec is frozen, so take
                # the meet again for every dispatch instead of retaining authority.
                record.agent._policy = effective_policy

            with self._changed:
                if background and len(self._background_names) >= 4:
                    record.running = False
                    return ToolResult(tool_call.id, ok=False, error=(
                        "at most 4 background subagents; running: "
                        + ", ".join(sorted(self._background_names))))
                # Keep a previous dispatch's deferred delivery attached to that dispatch.
                if record.background:
                    record = replace(record)
                    self.pool[subagent_name] = record
                record.background = background
                record.state = "running"
                record.result = None
                record.delivered = False
                if background:
                    self._background_names.add(subagent_name)

            event_run_id = self._event_run_id
            event_agent_id = self._event_agent_id
            event_turn_id = self._event_turn_id

            def new_child_run() -> AgentRun:
                child_run = self._new_run(spec, record.agent_ref)
                record.runs.append(child_run)
                record.current_run = child_run
                return child_run

            def prepare_child() -> None:
                if record.messages:
                    record.messages.append(Message(role=Role.USER, content=task))
                else:
                    memory, unreadable_memory = _memory_seed(self._memory, spec)
                    record.messages = seed_messages(
                        spec,
                        task,
                        parent_messages=self._parent_messages,
                        memory=memory,
                        memory_available=self._memory is not None,
                        unreadable_memory=unreadable_memory,
                    )
                if self._events is not None:
                    if event_run_id is None:
                        raise RuntimeError("dispatch event context was never set")
                    emit(
                        self._events,
                        SubagentSpawned(
                            agent_id=event_agent_id,
                            run_id=event_run_id,
                            turn_id=event_turn_id,
                            subagent_name=subagent_name,
                            subagent_agent_id=record.agent_ref.agent_id,
                        ),
                    )

            def run_child(
                child_run: AgentRun, *, initial: bool = False,
            ) -> tuple[AgentRunResult | None, str | None, Exception | None]:
                round_token = token.child()
                result = None
                failure = None
                execution_error = None
                with self._active_runs_lock:
                    self._active_runs[record.agent_ref.agent_id] = child_run
                try:
                    child_run.start(round_token)
                    if initial and not background:
                        prepare_child()
                    lease = (
                        nullcontext() if record.worktree_path is not None or background
                        else self._leases.held(child_run.run.run_id, spec.isolation.workspace_prefix)
                    )
                    with lease:
                        result = record.agent.run(
                            record.messages, model=spec.model.model, effort=spec.model.effort,
                            parent_run_id=None if self._parent_run is None else self._parent_run.run.run_id,
                            cancel=round_token, run=child_run, hooks=self._hooks,
                        )
                except OperationCancelled:
                    raise
                except Exception as exc:
                    failure = str(exc)
                    execution_error = exc
                finally:
                    with self._active_runs_lock:
                        if self._active_runs.get(record.agent_ref.agent_id) is child_run:
                            self._active_runs.pop(record.agent_ref.agent_id, None)
                    if child_run.phase is RunPhase.RUNNING:
                        if result is None:
                            if token.cancelled:
                                child_run.cancel()
                            else:
                                child_run.fail(failure or "subagent dispatch failed")
                        elif result.stopped_reason == "cancelled":
                            child_run.cancel()
                        else:
                            child_run.finish(result)
                    round_token.close()
                    if initial and event_run_id is not None:
                        emit(self._events, SubagentStopped(
                            agent_id=event_agent_id, run_id=event_run_id,
                            turn_id=event_turn_id, subagent_name=subagent_name,
                            subagent_agent_id=record.agent_ref.agent_id,
                        ))
                return result, failure, execution_error

            def record_child_result(result: AgentRunResult) -> None:
                record.messages = result.messages
                record.turns_used += result.turns_used
                for model, usage in result.usage_by_model.items():
                    record.usage_by_model[model] = record.usage_by_model.get(
                        model, UsageTotals()
                    ).merged(usage)

            child_run = new_child_run()
            token = (
                cancel.child(deadline_seconds=spec.deadline_seconds)
                if cancel is not None
                else CancellationToken(deadline_seconds=spec.deadline_seconds)
            )
            record.token = token

            def dispatch_result() -> ToolResult:
                nonlocal child_run
                document_shell = record.agent._tools.get("run_shell") if document_role else None
                if isinstance(document_shell, _DocumentRoleTool):
                    document_shell.changed_paths.clear()
                run_result, failure, execution_error = run_child(child_run, initial=True)
                if isinstance(execution_error, LeaseConflict):
                    return ToolResult(tool_call_id=tool_call.id, ok=False, error=failure)
                if run_result is None:
                    try:
                        content, payload = self._worktree_output(subagent_name, record, "")
                    finally:
                        with self._pool_lock:
                            record.running = False
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        ok=False,
                        content=document_shell.report(content) if isinstance(document_shell, _DocumentRoleTool) else content,
                        error=failure or str(execution_error) or "subagent dispatch failed",
                        payload=payload,
                    )
                record_child_result(run_result)
                validation_passed = not validation_commands
                validation_evidence: list[dict[str, object]] = []
                validation_round = 1
                while (
                    validation_commands
                    and not token.cancelled
                    and run_result.stopped_reason == "final_response"
                ):
                    validation_evidence = _run_handoff_validation(
                        validation_commands, self._leader_policy.rerooted(effective_policy.repo_root)
                    )
                    failed_check = next(
                        (item for item in validation_evidence if item["exit"] != 0), None
                    )
                    if failed_check is None:
                        validation_passed = True
                        break
                    if token.cancelled or failed_check.get("broken") or validation_round == 5:
                        break
                    record.messages.append(Message(
                        role=Role.USER,
                        content=(
                            f"Validation failed (round {validation_round} of 5): "
                            f"{failed_check['command']} exited {failed_check['exit']}\n"
                            f"{str(failed_check['output'])[-4000:]}\n"
                            "Fix the cause and finish again; do not edit the spec."
                        ),
                    ))
                    validation_round += 1
                    child_run = new_child_run()
                    retry_result, retry_failure, _ = run_child(child_run)
                    if retry_result is None:
                        failure = retry_failure or "validation retry failed"
                        break
                    run_result = retry_result
                    record_child_result(run_result)
                with self._pool_lock:
                    record.running = False
                if token.cancelled or run_result.stopped_reason == "cancelled":
                    if token.reason is CancelReason.PARENT:
                        with self._pool_lock:
                            record.running = False
                        raise OperationCancelled
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
                        content=document_shell.report(content) if isinstance(document_shell, _DocumentRoleTool) else content,
                        error=error,
                        payload=payload,
                    )

                succeeded = run_result.stopped_reason == "final_response" and validation_passed
                output_error = None
                if succeeded and spec.io.output_schema is not None:
                    _, output_error = validate_output(
                        spec.io,
                        run_result.final_response.message.text,
                    )
                    succeeded = output_error is None
                if output_error is not None:
                    error = output_error
                elif succeeded:
                    error = None
                elif run_result.stopped_reason == "refused":
                    category = run_result.final_response.refusal_category
                    error = (f"subagent was refused ({category}) before a final answer"
                             if category else "subagent was refused before a final answer")
                elif run_result.stopped_reason != "final_response":
                    error = f"subagent stopped because {run_result.stopped_reason} before a final answer"
                elif validation_commands and not validation_passed:
                    last_failure = next(
                        (item for item in validation_evidence if item["exit"] != 0), None
                    )
                    error = (
                        f"Validation is broken: {last_failure['command']}\n{last_failure['output']}"
                        if last_failure is not None and last_failure.get("broken")
                        else "validation failed after 5 rounds"
                        if last_failure is not None and validation_round == 5
                        else failure or "validation retry failed"
                    )
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
                if validation_commands:
                    evidence_text = "\n".join(
                        f"{item['command']} exited {item['exit']}\n{item['output']}"
                        for item in validation_evidence
                    )
                    content = f"{content}\n\nValidation:\n{evidence_text}"
                    if succeeded:
                        content += (
                            "\nThe host ran these Validation commands after the subagent "
                            "finished, and all passed. Do not run them again, or any "
                            "equivalent of them; read the diff instead."
                        )
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=succeeded,
                    content=document_shell.report(content) if isinstance(document_shell, _DocumentRoleTool) else content,
                    error=error,
                    payload=payload,
                )

            def execute_dispatch() -> ToolResult:
                try:
                    outcome = dispatch_result()
                except OperationCancelled:
                    raise
                except Exception as exc:
                    outcome = ToolResult(tool_call.id, ok=False, error=str(exc))
                if token.cancelled and token.reason is CancelReason.PARENT:
                    raise OperationCancelled
                if outcome.ok:
                    record.breaker.record_success()
                else:
                    record.breaker.record_failure()
                return outcome

            if not background:
                try:
                    outcome = execute_dispatch()
                    record.state = "idle" if outcome.ok else "failed"
                    return outcome
                except Exception:
                    record.state = "failed"
                    raise
                finally:
                    record.running = False

            # Acquire the shared-tree lease before acknowledging the background start.
            lease = None
            try:
                if record.worktree_path is None:
                    lease = self._leases.acquire(child_run.run.run_id, spec.isolation.workspace_prefix)
                prepare_child()
            except OperationCancelled:
                raise
            except Exception as exc:
                record.breaker.record_failure()
                token.close()
                with self._changed:
                    record.running = False
                    record.state = "failed"
                    self._background_names.discard(subagent_name)
                    self._changed.notify_all()
                return ToolResult(tool_call.id, ok=False, error=str(exc))

            def worker() -> None:
                try:
                    outcome = execute_dispatch()
                except OperationCancelled:
                    outcome = ToolResult(tool_call.id, ok=False, cancelled=True, error="subagent was cancelled")
                except Exception as exc:
                    outcome = ToolResult(tool_call.id, ok=False, error=str(exc))
                finally:
                    if lease is not None:
                        self._leases.release(lease)
                    token.close()
                with self._changed:
                    record.running = False
                    if record.state != "closed":
                        record.result = outcome
                        record.state = "finished" if outcome.ok else "failed"
                        if self._parent_run is not None:
                            self._parent_run.redirect_deferred(
                                lambda: self._deliver_result(subagent_name, record))
                    self._background_names.discard(subagent_name)
                    self._changed.notify_all()

            record.thread = threading.Thread(target=worker, name=f"subagent-{subagent_name}", daemon=True)
            record.thread.start()
            worker_started = True
            return ToolResult(tool_call.id, ok=True, content=f"subagent {subagent_name} is running",
                              payload={"name": subagent_name, "state": "running", "background": True})
        except OperationCancelled:
            raise
        except Exception as exc:
            if record is not None:
                record.state = "failed"
                record.breaker.record_failure()
            return ToolResult(tool_call.id, ok=False, error=str(exc))
        finally:
            with self._changed:
                self._reserved_names.discard(subagent_name)
                if not worker_started:
                    self._background_names.discard(subagent_name)
                    if record is not None:
                        record.running = False
                self._changed.notify_all()
            if token is not None and not worker_started:
                token.close()
            if lease is not None and not worker_started:
                self._leases.release(lease)

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
    base_prompt: str = ""
    subagent_base_prompt: str = ""
    max_leader_turns: int | None = None
    leader_budget: RunBudget | None = None
    max_subagents: int = DEFAULT_MAX_SUBAGENTS
    subagent_max_turns: int | None = None
    subagent_budget: RunBudget | None = None
    subagent_tool_names: Sequence[str] | None = None
    subagent_specs: Mapping[str, AgentSpec] | None = None
    permission_mode: PermissionMode = "allow"
    approval_callback: ApprovalCallback | None = None
    chat_token_budget: int | None = None
    compaction_limit: int | None = None
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
    memory_provider: ModelProvider | None = None
    memory_model: str | None = None
    memory_extraction_enabled: bool = True
    memory_usage_callback: Callable[[str, UsageTotals], None] | None = None
    model_summary: bool = False
    session_notes: Path | None = None
    checkpoints: CheckpointStore | None = None
    leader_tools: Mapping[str, LocalTool] | None = None
    lsp: LspManager | None = None
    provider_for: Callable[[str, str | None], ModelProvider | None] | None = None
    leases: WorkspaceLeases | None = None
    jobs: JobManager | None = None


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


def _project_snapshot(root: Path) -> dict[str, tuple[int, int]] | None:
    """Capture file size and mtime for a reasonably sized project tree."""
    from symphonai_api.tools.search import _walk_candidates

    snapshot: dict[str, tuple[int, int]] = {}
    root = root.resolve()
    policy = PermissionPolicy(repo_root=root)
    for index, candidate in enumerate(_walk_candidates(root, policy, include_directories=True, include_gitignored=True), 1):
        if index > 50_000:
            return None
        if candidate.is_directory:
            continue
        try:
            info = candidate.path.lstat()
        except OSError:
            continue
        snapshot[candidate.repo_relative] = (info.st_size, info.st_mtime_ns)
    return snapshot


class _DocumentRoleTool(LocalTool):
    """Constrain leader-dispatched document roles without changing loop stages."""

    def __init__(self, delegate, role):
        self._delegate = delegate
        self._role = role
        self.changed_paths = set()

    name = property(lambda self: self._delegate.name)
    description = property(lambda self: self._delegate.description)
    parameters = property(lambda self: self._delegate.parameters)

    def metadata(self, arguments):
        return self._delegate.metadata(arguments)

    def report(self, content):
        if self.changed_paths:
            content += (f"\n\nThe {self._role}'s shell changed non-Markdown project files: "
                        + ", ".join(sorted(self.changed_paths)))
        return content

    def _execute(self, tool_call, policy, cancel=None):
        if self.name == "run_shell":
            return _LeaderShellTool(
                self._delegate, root=policy.repo_root, role=self._role,
                on_change=self.changed_paths.update,
            ).execute(tool_call, policy, cancel=cancel)
        paths = changed_paths(self.name, tool_call.arguments)
        for path in paths:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = policy.repo_root / candidate
            if not path.endswith(".md") or candidate.resolve().suffix != ".md":
                return ToolResult(tool_call.id, ok=False, error=(
                    f"the {self._role} writes specs (Markdown files) only; code goes to the implementer"
                ))
        return self._delegate.execute(tool_call, policy, cancel=cancel)


class _LeaderShellTool(LocalTool):
    """Run the normal shell while recording changes to existing project files."""

    def __init__(self, delegate: LocalTool, *, root: Path, leader: "Leader | None" = None,
                 role: str | None = None, on_change=None) -> None:
        self._delegate = delegate
        self._root = root.resolve()
        self._leader = leader
        self._role = role
        self._on_change = on_change

    @property
    def name(self) -> str:
        return self._delegate.name

    @property
    def description(self) -> str:
        return self._delegate.description

    @property
    def parameters(self) -> dict:
        return self._delegate.parameters

    def validate(self, arguments: dict) -> str | None:
        return self._delegate.validate(arguments)

    def metadata(self, arguments: dict) -> ToolMetadata:
        return self._delegate.metadata(arguments)

    def _execute(
        self, tool_call: ToolCall, policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        before = _project_snapshot(self._root)
        result = self._delegate.execute(tool_call, policy, cancel=cancel)
        after = _project_snapshot(self._root)
        skipped = before is None or after is None
        changed_existing: list[str] = []
        new_files_count = 0
        if not skipped:
            changed_existing = sorted(
                path for path, details in before.items()
                if path not in after or after[path] != details
            )
            new_files_count = sum(path not in before for path in after)
        if self._role is not None:
            changed = ([] if skipped else sorted(
                path for path in before.keys() | after.keys()
                if not path.endswith(".md") and before.get(path) != after.get(path)
            ))
            if self._on_change is not None:
                self._on_change(changed)
            if changed:
                suffix = (f"your command changed {', '.join(changed)}; the {self._role} "
                          "writes no code — say so in your report")
                result = replace(result, content=f"{result.content}\n{suffix}" if result.content else suffix)
            elif skipped:
                result = replace(result, content=(
                    f"{result.content}\nProject change snapshot skipped: tree exceeds 50000 entries; "
                    "say what your command changed in your report"
                ))
            return result
        transcript = self._leader._transcript
        run_id = self._leader._leader_run_id()
        if transcript is not None and run_id is not None:
            data = {
                "changed_existing_paths": changed_existing,
                "new_files_count": new_files_count,
            }
            if skipped:
                data["snapshot_skipped"] = "project tree exceeds 50000 entries"
            transcript.append(
                "leader_shell_changes",
                run_id=run_id,
                agent_id=self._leader._agent_ref.agent_id,
                turn_id=None,
                data=data,
            )
        if changed_existing:
            listed = ", ".join(changed_existing[:20])
            if len(changed_existing) > 20:
                listed += f", and {len(changed_existing) - 20} more"
            suffix = f"This command changed existing project files: {listed}"
            result = replace(result, content=f"{result.content}\n{suffix}" if result.content else suffix)
        return result

class _SubagentControlTool(LocalTool):
    def __init__(self, dispatch, name, properties, required, description):
        self.dispatch = dispatch
        self._name = name
        self._description = description
        self._parameters = {"type": "object", "properties": properties, "required": required}

    name = property(lambda self: self._name)
    description = property(lambda self: self._description)
    parameters = property(lambda self: self._parameters)

    def metadata(self, arguments):
        return ToolMetadata(concurrency_safe=False)

    def _execute(self, tool_call, policy, cancel=None):
        try:
            arguments = dict(tool_call.arguments)
            if self.name == "wait_subagents":
                timeout = arguments.get("timeout_seconds", 600)
                if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 <= timeout <= 3600:
                    raise ValueError("timeout_seconds must be between 0 and 3600")
                arguments["cancel"] = cancel
            if self.name == "message_subagent":
                name, text = arguments["name"], arguments["text"]
                if not text.strip():
                    raise ValueError("message text must not be blank")
                with self.dispatch._changed:
                    record = self.dispatch.pool.get(name)
                    state = "unknown" if record is None else record.state
                    if record is None or state != "running" or record.current_run is None:
                        raise ValueError(f"subagent {name} is {state}")
                    notice = LEADER_MESSAGE_PREFIX + "\n" + text
                    try:
                        record.current_run.redirect(notice)
                    except ValueError:
                        raise ValueError(f"subagent {name} is no longer running") from None
                    # Keep the recipient in the root transcript for notice replay.
                    if self.dispatch._parent_run is not None:
                        self.dispatch._parent_run.redirect(
                            LEADER_MESSAGE_PREFIX + f"\nTo {name}: " + text)
                    self.dispatch._notice(name, record, "message", text)
                    payload = {"name": name, "sent": True}
            else:
                payload = getattr(self.dispatch, self.name)(**arguments)
            return ToolResult(tool_call.id, ok=True, content=json.dumps(payload), payload=payload)
        except ValueError as exc:
            return ToolResult(tool_call.id, ok=False, error=str(exc))


class _NotifyPersonTool(LocalTool):
    name = "notify_person"
    description = "Send the person a short progress note without waiting for a reply."
    parameters = {"type": "object", "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 500},
    }, "required": ["text"], "additionalProperties": False}

    def __init__(self, leader):
        self.leader = leader

    def metadata(self, arguments):
        return ToolMetadata(concurrency_safe=False)

    def validate(self, arguments):
        return None if arguments["text"].strip() else "note text must not be blank"

    def _execute(self, tool_call, policy, cancel=None):
        run_id = self.leader._leader_run_id()
        if run_id is None:
            return ToolResult(tool_call.id, ok=False, error="the leader is not running")
        emit(self.leader._event_sink, PersonNotice(
            agent_id=self.leader.agent_ref.agent_id, run_id=run_id,
            text=mask_secrets(tool_call.arguments["text"]),
        ))
        return ToolResult(tool_call.id, ok=True, content="sent")


class _MessageLeaderTool(LocalTool):
    name = "message_leader"
    description = "Send the leader a progress note or question, then keep working."
    parameters = {"type": "object", "properties": {
        "text": {"type": "string", "minLength": 1},
        "kind": {"type": "string", "enum": ["progress", "question"]},
    }, "required": ["text", "kind"]}

    def __init__(self, dispatch, name):
        self.dispatch = dispatch
        self.subagent_name = name

    def metadata(self, arguments):
        return ToolMetadata(concurrency_safe=False)

    def _execute(self, tool_call, policy, cancel=None):
        text, kind = tool_call.arguments["text"], tool_call.arguments["kind"]
        if not text.strip():
            return ToolResult(tool_call.id, ok=False, error="message text must not be blank")
        dispatch = self.dispatch
        with dispatch._changed:
            record = dispatch.pool[self.subagent_name]
            parent = dispatch._parent_run
            if record.state == "closed" or parent is None or parent.phase is not RunPhase.RUNNING:
                return ToolResult(tool_call.id, ok=False, error="the leader is no longer running")
            prefix = SUBAGENT_QUESTION_PREFIX if kind == "question" else SUBAGENT_PROGRESS_PREFIX
            notice = prefix.format(name=self.subagent_name) + "\n" + text
            if kind == "progress":
                dispatch._progress[self.subagent_name] = text
                parent.redirect(notice)
            else:
                question = {"name": self.subagent_name, "text": text, "delivered": False, "waited": False}
                dispatch._questions.append(question)

                def deliver():
                    with dispatch._changed:
                        if question["delivered"] or record.state == "closed":
                            return None
                        question["delivered"] = True
                        return notice

                parent.redirect_deferred(deliver)
            dispatch._notice(self.subagent_name, record, kind, text)
            dispatch._changed.notify_all()
        return ToolResult(tool_call.id, ok=True, content="message sent to the leader")


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
        persisted_agent_id = None if session is None else session.read_meta().get("agent_id")
        self._agent_ref = (
            AgentRef(persisted_agent_id, "leader")
            if isinstance(persisted_agent_id, str) and persisted_agent_id
            else new_agent_ref("leader")
        )
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
        self._leases = config.leases or WorkspaceLeases(config.repo_root)
        defined_leader = None if config.subagent_specs is None else config.subagent_specs.get("leader")
        is_loop_stage = defined_leader is not None and defined_leader.name != "leader"
        restrict_leader = not is_loop_stage
        self._leader_jobs = config.jobs or (JobManager() if restrict_leader else None)
        self._owns_leader_jobs = restrict_leader and config.jobs is None
        leader_agent_policy = leader_policy
        self._leader_prompt = "\n\n".join(
            prompt for prompt in (
                config.base_prompt,
                "" if defined_leader is None else defined_leader.prompt,
            ) if prompt
        )
        leader_provider_name = config.leader_provider.name or "unknown"
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
            defined_leader.model.effort
            if defined_leader is not None and defined_leader.model.effort is not None
            else None if is_loop_stage else config.leader_effort
        )
        if self._leader_effort is None:
            self._leader_effort = _default_effort(
                config.leader_provider, self._leader_model or config.leader_provider.model,
                "low" if is_loop_stage else "medium",
            )
        self._leader_run: AgentRun | None = None
        self._control_lock = threading.RLock()
        self._dispatch_tool = DispatchSubagentTool(
            subagent_provider=config.subagent_provider,
            leader_policy=leader_policy,
            max_subagents=config.max_subagents,
            subagent_max_turns=config.subagent_max_turns,
            subagent_base_prompt=config.subagent_base_prompt,
            subagent_tool_names=config.subagent_tool_names,
            parent_agent_id=self._agent_ref.agent_id,
            subagent_budget=config.subagent_budget,
            compaction_limit=config.compaction_limit,
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
            jobs=config.jobs,
            provider_for=config.provider_for,
            model_summary=config.model_summary,
        )
        self._event_sink.bind_dispatch_tool(self._dispatch_tool)
        control_tools = {
            name: _SubagentControlTool(self._dispatch_tool, name, properties, required, description)
            for name, properties, required, description in (
                ("wait_subagents", {"names": {"type": "array", "items": {"type": "string"}},
                                    "timeout_seconds": {"type": "number", "minimum": 0, "maximum": 3600}},
                 [], "Wait for a background subagent to finish, or until the timeout."),
                ("message_subagent", {"name": {"type": "string"}, "text": {"type": "string", "minLength": 1}},
                 ["name", "text"], "Send input to a running subagent at its next turn boundary."),
                ("list_subagents", {}, [], "List this leader's subagents and their states."),
                ("close_subagent", {"name": {"type": "string"}}, ["name"], "Cancel and close a subagent."),
            )
        }
        leader_tools = {DISPATCH_TOOL_NAME: self._dispatch_tool, **control_tools}
        tool_names = None if defined_leader is None else defined_leader.tool_names
        if tool_names is not None and config.result_store is not None:
            tool_names = (*tool_names, "read_tool_result")
        leader_memory_tool = (
            MemoryTool(
                config.memory,
                "leader",
                self._leader_run_id,
            )
            if config.memory is not None
            and (defined_leader is None or defined_leader.memory.enabled)
            else None
        )
        read_ledger = ReadLedger()
        standard_tools = merge_tool_registry(
            standard_tool_registry(
                tool_names,
                wire_format=config.leader_provider.wire_format,
                ledger=read_ledger,
                result_store=config.result_store,
                search_backend=config.search_backend,
                memory_tool=leader_memory_tool,
                skills=(None if config.extensions is None else config.extensions.skills),
                checkpoints=config.checkpoints,
                lsp=config.lsp,
                jobs=self._leader_jobs if restrict_leader else config.jobs,
            ),
            config.extra_tools,
        )
        if restrict_leader:
            for name in FILE_EDIT_TOOLS:
                standard_tools.pop(name, None)
        leader_tools.update(standard_tools)
        leader_only_tools = dict(config.leader_tools or {})
        if restrict_leader:
            for name in FILE_EDIT_TOOLS:
                leader_only_tools.pop(name, None)
        leader_tools.update(leader_only_tools)
        leader_tools["notify_person"] = _NotifyPersonTool(self)
        if restrict_leader:
            for name in FILE_EDIT_TOOLS:
                leader_tools.pop(name, None)
            if "run_shell" in leader_tools:
                leader_tools["run_shell"] = _LeaderShellTool(
                    leader_tools["run_shell"], root=Path(config.repo_root), leader=self,
                )
        provider_name = config.leader_provider.name or "unknown"
        self._leader_spec = AgentSpec(
            name="leader",
            prompt=self._leader_prompt,
            model=ModelSelector(
                provider=provider_name,
                model=(
                    self._leader_model
                    if self._leader_model is not None
                    else config.leader_provider.model
                ),
                effort=self._leader_effort,
            ),
            policy_ceiling=leader_agent_policy,
            tool_names=tuple(leader_tools),
            budget=config.leader_budget,
            memory=(
                defined_leader.memory
                if defined_leader is not None
                else MemorySettings(enabled=True)
            ),
            call_class=CallClass.FOREGROUND,
        )
        self._agent = ApiAgent(
            provider=config.leader_provider,
            tools=leader_tools,
            policy=leader_agent_policy,
            max_turns=config.max_leader_turns,
            budget=config.leader_budget,
            tool_schemas=[
                dispatch_subagent_tool_schema(
                    config.leader_provider.wire_format, config.subagent_specs,
                ),
                *tool_registry_schemas(control_tools, config.leader_provider.wire_format),
                *tool_registry_schemas(
                    standard_tools,
                    config.leader_provider.wire_format,
                ),
                *tool_registry_schemas(
                    leader_only_tools,
                    config.leader_provider.wire_format,
                ),
            ],
            tool_history=None if session is None else session.directory / "run.jsonl",
            agent_ref=self._agent_ref,
            events=self._event_sink,
            stream=config.stream,
            result_store=config.result_store,
            read_ledger=read_ledger,
            call_class=CallClass.FOREGROUND,
            transcript=(
                self._transcript
            ),
            model_summary=config.model_summary,
            session_notes=config.session_notes,
            context_budget=lambda model: math.floor(
                budget_for_model(
                    config.leader_provider.wire_format, model,
                    config.compaction_limit,
                )
                / self._token_ratio
            ),
            context_overflow=self._record_context_overflow,
            context_recent_turns=config.chat_recent_turns,
        )
        self._chat_messages = self._initial_leader_messages()
        self._memory_extractor = (
            MemoryExtractor(
                config.memory,
                self._leader_spec,
                leader_policy,
                agent_id=self._agent_ref.agent_id,
                events=self._event_sink,
                usage_callback=config.memory_usage_callback,
            )
            if config.memory is not None
            and self._leader_spec.memory.enabled
            and config.memory_extraction_enabled
            else None
        )
        if self._memory_extractor is not None:
            self._memory_extractor.set_cursor(self._chat_messages)
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
                self._config.compaction_limit,
            )
        )
        if self._config.compaction_limit is not None:
            base = min(base, self._config.compaction_limit)
        return math.floor(base / self._token_ratio)

    def _record_context_overflow(
        self, request_tokens: int, actual_tokens: int | None
    ) -> None:
        if actual_tokens is not None and request_tokens > 0:
            self._token_ratio = max(
                self._token_ratio, actual_tokens / request_tokens
            )

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
            actual_model = self._config.leader_provider.model
        self._leader_spec = replace(
            self._leader_spec,
            model=replace(
                self._leader_spec.model,
                model=actual_model,
                effort=self._leader_effort,
            ),
        )

    def _initial_leader_messages(self) -> list[Message]:
        if not self._leader_spec.memory.enabled or self._config.memory is None:
            return (
                [Message(role=Role.SYSTEM, content=self._leader_prompt)]
                if self._leader_prompt
                else []
            )
        memory, unreadable = _memory_seed(self._config.memory, self._leader_spec)
        messages = seed_messages(
            self._leader_spec,
            "",
            memory=memory,
            memory_available=True,
            unreadable_memory=unreadable,
        )
        return messages[:-1]

    def _run_messages(
        self, messages: list[Message], *, cancel: CancellationToken | None = None
    ) -> LeaderRunResult:
        with self._dispatch_tool._pool_lock:
            for record in self._dispatch_tool.pool.values():
                record.breaker.reset()
        leader_run = new_agent_run(self._leader_spec)
        leader_run.agent = self._agent_ref
        leader_run.run = RunRef(
            run_id=leader_run.run.run_id,
            agent_id=self._agent_ref.agent_id,
        )
        leader_token = cancel if cancel is not None else CancellationToken()
        leader_run.start(leader_token)
        self._leader_run = leader_run
        self._dispatch_tool.set_parent_context(leader_run, list(messages))
        try:
            result = self._agent.run(
                messages,
                model=self._leader_model,
                effort=self._leader_effort,
                cancel=leader_token,
                run=leader_run,
                hooks=self._hook_runner,
            )
        except OperationCancelled:
            if leader_run.phase is RunPhase.RUNNING:
                leader_run.cancel()
            raise
        except Exception as exc:
            leader_run.fail(str(exc))
            raise
        finally:
            self._dispatch_tool.cancel_background()
        if result.stopped_reason == "cancelled":
            if leader_run.phase is RunPhase.RUNNING:
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
        )

    def _begin_checkpoint(self) -> None:
        if self._config.checkpoints is None:
            return
        key = new_id("chk")
        self._config.checkpoints.begin(key)
        if self._transcript is not None:
            self._transcript.begin_checkpoint(key)

    def redirect(self, text: str) -> None:
        """Queue steering for the root agent's live run."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("redirect text must not be blank")
        with self._control_lock:
            run = self._leader_run
            if run is None or run.phase is not RunPhase.RUNNING:
                raise AgentControlError("no turn is running", status=404)
            try:
                run.redirect(text)
            except ValueError as exc:
                raise AgentControlError(str(exc)) from exc

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


    def seed_chat(self, messages: Sequence[Message], *, persisted: bool = False) -> None:
        """Set the history used by the next chat without changing subagents."""

        self._chat_messages = (
            list(messages)
            if persisted
            else [*self._initial_leader_messages(), *messages]
        )
        if persisted:
            self._agent._persisted_digests = [
                message_digest(message) for message in messages
            ]
        if self._memory_extractor is not None:
            self._memory_extractor.set_cursor(self._chat_messages)

    def clear_subagents(self) -> int:
        """Clear all dispatched subagents and return how many were removed."""

        with self._dispatch_tool._pool_lock:
            count = len(self._dispatch_tool.pool)
            names = set(self._dispatch_tool.pool)
            if self._session is not None:
                names.update(_session_subagent_transcripts(self._session))
                self._remember_forgotten_subagents(names)
            self._dispatch_tool.pool.clear()
            return count

    def forget_subagent(self, name: str) -> bool:
        """Remove one named subagent from the current leader pool."""
        with self._dispatch_tool._pool_lock:
            if self._session is not None:
                self._remember_forgotten_subagents({name})
            record = self._dispatch_tool.pool.pop(name, None)
            return record is not None

    def _remember_forgotten_subagents(self, names: set[str]) -> None:
        if self._session is None or not names:
            return
        def change(meta):
            forgotten = meta.get("forgotten_subagents")
            if not isinstance(forgotten, dict):
                forgotten = {}
            transcripts = _session_subagent_transcripts(self._session)
            for name in names:
                agent_ids = {agent_id for _, agent_id, _ in transcripts.get(name, ())}
                record = self._dispatch_tool.pool.get(name)
                if record is not None:
                    agent_ids.add(record.agent_ref.agent_id)
                previous = forgotten.get(name, ())
                if isinstance(previous, list):
                    agent_ids.update(item for item in previous if isinstance(item, str))
                forgotten[name] = sorted(agent_ids)
            meta["forgotten_subagents"] = forgotten
        self._session.update_meta(change)


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
                self._config.leader_provider.create_response_stream(request, cancel=cancel)
            )
            assembler = StreamAssembler()
            for chunk in chunks:
                assembler.add(chunk)
            response = assembler.finish()
        else:
            response = self._config.leader_provider.create_response(
                request, cancel=cancel
            )
        model_key = model or self._config.leader_provider.model or "unknown"
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
        from symphonai_api.compaction import estimate_messages_tokens

        notes_summary = None

        def summarize(dropped: list[Message]) -> str:
            nonlocal notes_summary
            if instructions is None:
                notes_summary = self._agent._compaction_notes_summary(
                    self._chat_messages, dropped, cancel=cancel,
                )
                if notes_summary is not None:
                    return notes_summary
            return self._write_model_summary(dropped, cancel=cancel, instructions=instructions)

        compact_kwargs = {
            "budget": budget,
            "recent_turns": (
                self._config.chat_recent_turns
                if recent_turns is None
                else recent_turns
            ),
            "cancel": cancel,
            "recent_files": (
                self._agent._compaction_recent_files(self._chat_messages)
                if force or estimate_messages_tokens(self._chat_messages) > budget else []
            ),
        }
        if force:
            compact_kwargs["force"] = True
        if self._config.model_summary or self._config.session_notes is not None or instructions is not None:
            compact_kwargs["summarize"] = summarize
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
            self._agent._compaction_notes_rebased(
                result.messages, result.before_tokens, result.after_tokens, notes_summary,
            )
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
        try:
            result = self._run_messages(self._chat_messages, cancel=cancel)
        except ContextLengthExceededError:
            self._automatic_compaction_breaker.record_failure()
            self._context_overflow_repair_failed = True
            raise
        if self._context_overflow_repair_failed:
            self._automatic_compaction_breaker.record_success()
            self._context_overflow_repair_failed = False
        self._chat_messages = result.leader_messages
        extraction_messages = list(result.leader_messages)
        if (
            result.stopped_reason == "final_response"
            and self._memory_extractor is not None
        ):
            self._memory_extractor.submit(
                extraction_messages,
                leader_provider=self._config.leader_provider,
                leader_model=self._leader_model,
                leader_effort=self._leader_effort,
                leader_tool_schemas=list(self._agent._tool_schemas),
                memory_provider=self._config.memory_provider,
                memory_model=self._config.memory_model,
            )
        self._automatic_compact_chat(cancel=cancel)
        leader_usage = result.usage_by_agent.setdefault(
            self._agent_ref.agent_id, {}
        )
        for model, usage in self._compaction_usage_by_model.items():
            leader_usage[model] = leader_usage.get(model, UsageTotals()).merged(usage)
        self._compaction_usage_by_model.clear()
        return result

    def add_context(self, text: str) -> None:
        """Append a user message for the next chat call without running the model."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("context text must be non-blank")
        self._chat_messages.append(Message(role=Role.USER, content=text))

    def close(self) -> None:
        self._agent.close_session_notes()
        if self._owns_leader_jobs and self._leader_jobs is not None:
            self._leader_jobs.close()
        elif self._config.jobs is not None:
            self._config.jobs.close()
        if self._memory_extractor is not None:
            self._memory_extractor.close()

    @property
    def background_jobs(self) -> JobManager | None:
        return self._leader_jobs
