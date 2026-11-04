"""Threaded conversation runs owned by a SymphonAI host process."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from symphonai_api.agent_memory import AgentMemory
from symphonai_api.checkpoints import CheckpointEntry, CheckpointStore
from symphonai_api.agent_spec import AgentSpec, ModelSelector
from symphonai_api.budgets import RunBudget
from symphonai_api.cancellation import CancellationToken
from symphonai_api.compaction import CompactionResult, DEFAULT_RECENT_TURNS
from symphonai_api.config import permission_grants, restrict_permission_grants, resolve_run_budgets
from symphonai_api.context_report import ContextReport, account_context
from symphonai_api.cost import PriceTable, UsageTotals, total_cost
from symphonai_api.events import AutoReviewed, PreviewMarkAdded, Event, JobChanged, PromptSubmitted, RunFailed, RunFinished, RunStarted, SessionNotesUpdated, SubagentSpawned, fan_out
from symphonai_api.extensions import Extensions
from symphonai_api.environment import capture_environment
from symphonai_api.identity import new_id
from symphonai_api.jobs import BackgroundJob, JobManager
from symphonai_api.instructions import load_instructions
from symphonai_api.lsp import LspManager
from symphonai_api.leader import (
    AgentControlError, Leader, LeaderConfig, LeaderRunResult,
    builtin_subagent_specs, SUBAGENT_NOTICE_PATTERN, SUBAGENT_FINISHED_PREFIX,
)
from symphonai_api.leases import WorkspaceLeases
from symphonai_api.model_table import context_window_for_model
from symphonai_api.models import ContentInput, DocumentBlock, ImageBlock, Message, Role, TextBlock
from symphonai_api.permissions import ALWAYS_DENY_COMMANDS, PermissionPolicy
from symphonai_api.paths import symphonai_home
from symphonai_api.role_prompts import load_role_additions
from symphonai_api.providers.base import ModelProvider, tool_result_text
from symphonai_api.runner import standard_tool_registry
from symphonai_api.session import (
    SessionError,
    SessionStore,
    TranscriptError,
    default_memory_root,
    default_sessions_root,
    fork_run,
    load_run,
    load_run_for_resume,
    read_records,
    tool_result_search_path,
)
from symphonai_api.serialization import message_from_json
from symphonai_api.skills import Skill
from symphonai_api.tool_results import ToolResultStore
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.edit import _diff_result
from symphonai_api.tools.filesystem import MAX_READ_BYTES
from symphonai_api.tools.metadata import FILE_EDIT_TOOLS, call_target
from symphonai_api.tools.shell import shell_environment, shell_executable
from symphonai_api.worktree import remove_worktree, worktree_diff
from symphonai_api.worktree import create_worktree
from symphonai_api.web_search import HttpJsonSearchBackend, search_endpoint
from symphonai_host.broker import EventBroker
from symphonai_host.approvals import (
    ApprovalBroker,
    InvalidApprovalAnswer,
    PermissionGrantError,
    load_project_permission_grants,
    remove_project_permission_grant,
    save_project_permission_grant,
)
from symphonai_host.goal import Goal, GoalCheck, GoalChanged, goal_tools, run_check
from symphonai_host.plan import PlanChanged, plan_tools
from symphonai_host.questions import plan_proposal_tools, question_tools
from symphonai_host.protocol import HistoryMessage
from symphonai_host.loops import LoopStore
from symphonai_host.spec_loop import Loop


def _usage_record(record: dict) -> tuple[str, str, UsageTotals] | None:
    if record.get("type") != "usage":
        return None
    data = record.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("model"), str):
        return None
    fields = (
        "input_tokens", "output_tokens", "calls",
        "cache_read_tokens", "cache_write_tokens",
    )
    values = [data.get(field) for field in fields]
    values.extend(data.get(field, 0) for field in (
        "long_context_input_tokens", "long_context_output_tokens",
        "long_context_cache_read_tokens", "long_context_cache_write_tokens",
    ))
    values.append(data.get("cache_write_1h_tokens", 0))
    if any(type(value) is not int or value < 0 for value in values):
        return None
    agent_id = record.get("agent_id")
    if not isinstance(agent_id, str):
        return None
    name = data.get("agent_name")
    return agent_id, name if isinstance(name, str) else agent_id, UsageTotals(*values)


MAX_ACTIVE_CONVERSATIONS = 4


@dataclass(frozen=True)
class _SkillSnapshot(Skill):
    saved_body: str = field(repr=False, compare=False)

    def body(self) -> str:
        return self.saved_body


def _snapshot_extensions(extensions):
    if extensions is None:
        return None
    return replace(extensions, skills={
        name: _SkillSnapshot(skill.name, skill.description, skill.when_to_use,
                             skill.path, skill.always_on_tokens, skill.body())
        for name, skill in extensions.skills.items()
    })


class RunActiveError(RuntimeError):
    """A run or another named activity prevents the requested operation."""

    def __init__(self, run_id: str | None = None, *, description: str | None = None) -> None:
        super().__init__(description if description is not None else f"run already active: {run_id}")
        self.run_id = run_id if run_id is not None else description
        self.description = description


class LoopStateError(RunActiveError):
    """A loop operation needs a different state."""

    def __init__(self, loop_id: str, state: str, expected: str) -> None:
        self.loop_id, self.state, self.expected = loop_id, state, expected
        super().__init__(state, description=f"loop {loop_id} is {state}; this needs {expected}")


class CommandActiveError(RuntimeError):
    """A user command is already running in this host."""


class WorktreeApplyConflict(RuntimeError):
    """A worktree patch no longer applies to the main repository."""


@dataclass(frozen=True)
class BackgroundUsageRecorded(Event):
    session_id: str = ""


@dataclass(frozen=True)
class LoopChanged(Event):
    loop_id: str = ""
    state: str = ""
    change: str = ""
    card_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class TerminalOutput(Event):
    terminal_id: str = ""
    data: str = ""
    offset: int = 0
    exited: bool = False
    exit_code: int | None = None


class NoConversationError(RuntimeError):
    """A conversation-only operation was requested before the first prompt."""


class ProviderSelectionError(ValueError):
    """A conversation cannot start with the requested provider."""


def _unchecked_goal_prompt(goal: Goal) -> str:
    return (
        f"Round {goal.rounds} of {goal.max_rounds} ended without the goal reported complete.\n"
        f"Keep working toward the goal: {goal.objective}\n"
        'If it is done, call update_goal with status "complete"; '
        'if you cannot finish it, call update_goal with status "blocked".'
    )


class MissingAPIKeyError(ProviderSelectionError):
    """The selected provider has no configured API key."""


class ModeSelectionError(ValueError):
    """The requested permission mode is not available to this host."""


class ChangedOutsideError(RuntimeError):
    def __init__(self, paths: list[str]) -> None:
        super().__init__("files changed outside the agent: " + ", ".join(paths))
        self.paths = paths


def _checkpoint_file_bytes(checkpoints: CheckpointStore, path: str) -> bytes | None:
    try:
        target = (checkpoints.repo_root / path).resolve()
    except (OSError, RuntimeError):
        return None
    if not target.is_relative_to(checkpoints.repo_root) or not target.is_file():
        return None
    return target.read_bytes()


PERMISSION_MODES = ("ask", "plan", "auto", "allow")
_LOG = logging.getLogger(__name__)


def _schema_tool_name(schema: dict) -> str | None:
    function = schema.get("function")
    return function.get("name") if isinstance(function, dict) else schema.get("name")


@dataclass(frozen=True)
class ForkableHistoryMessage(HistoryMessage):
    record_id: str

    def payload(self) -> dict:
        return {**super().payload(), "record_id": self.record_id}


CONVERSATION_TITLE_LIMIT = 80


def _provider_identity(provider: ModelProvider) -> tuple[str, int, str | None]:
    return (
        provider.name,
        provider.wire_format,
        provider.base_url,
    )


def _without_vendor_state(messages: list[Message]) -> list[Message]:
    return [
        replace(
            message,
            tool_calls=[
                replace(call, provider_metadata={}, vendor_id=None)
                for call in message.tool_calls
            ],
        )
        if message.tool_calls
        else message
        for message in messages
    ]


def _conversation_title(prompt: str) -> str:
    return " ".join(prompt.split())[:CONVERSATION_TITLE_LIMIT]


def _narrow_budget(
    existing: RunBudget | None,
    ceiling: RunBudget | None,
    *,
    turn_limit_configured: bool,
) -> RunBudget | None:
    if existing is None or ceiling is None:
        return existing or ceiling

    def tighter(left, right):  # noqa: ANN001
        return right if left is None else left if right is None else min(left, right)

    if (
        existing.max_cost is not None and ceiling.max_cost is not None
        and existing.price_table != ceiling.price_table
    ):
        raise ValueError("subagent cost budgets use different price tables")
    return RunBudget(
        max_turns=(
            tighter(existing.max_turns, ceiling.max_turns)
            if turn_limit_configured else existing.max_turns
        ),
        wall_seconds=tighter(existing.wall_seconds, ceiling.wall_seconds),
        max_total_tokens=tighter(existing.max_total_tokens, ceiling.max_total_tokens),
        max_cost=tighter(existing.max_cost, ceiling.max_cost),
        price_table=ceiling.price_table if ceiling.max_cost is not None else existing.price_table,
    )


@dataclass
class _ActiveRun:
    run_id: str
    cancel: CancellationToken
    thread: threading.Thread
    root_agent_id: str
    goal_round: bool = False
    runtime_run_id: str | None = None
    terminal_event: RunFinished | RunFailed | None = None
    deferred_memory_events: list[Event] = field(default_factory=list)


@dataclass
class _UserCommand:
    session_id: str | None = None
    stop_requested: bool = False
    process: subprocess.Popen | None = None
    next: _UserCommand | None = None

    def for_session(self, session_id: str | None) -> _UserCommand | None:
        command = self
        while command is not None:
            if command.session_id == session_id:
                return command
            command = command.next
        return None


class HostRun:
    """Run one prompt at a time in a persistent Leader conversation."""

    def __init__(
        self,
        provider: ModelProvider | None,
        policy: PermissionPolicy,
        broker: EventBroker,
        *,
        system_prompt: str | None = None,
        base_prompt: str = "",
        subagent_base_prompt: str = "",
        working_dir: Path | None = None,
        max_turns: int | None = None,
        model: str | None = None,
        provider_factory: Callable[[str | None, str | None, str | None], ModelProvider | None] | None = None,
        publish_approval=None,
        approval_timeout: float = 300.0,
        sessions_root: Path | None = None,
        memory_root: Path | None = None,
        memory_extraction_enabled: bool | None = None,
        project_grants_root: Path | None = None,
        extensions: Extensions | None = None,
        mcp_tools: Mapping[str, LocalTool] | None = None,
        mcp_pool=None,
        mcp_instructions: tuple[tuple[str, str], ...] = (),
        price_table: PriceTable | None = None,
        chat_token_budget: int | None = None,
        chat_recent_turns: int = DEFAULT_RECENT_TURNS,
        lsp: LspManager | None = None,
        reload_policy: PermissionPolicy | None = None,
    ) -> None:
        self._reload_policy = replace(reload_policy or policy)
        self._next_policy = None
        self._roadmap_tools_enabled = False
        self._memory_extraction_override = memory_extraction_enabled
        self._reload_price_table = price_table
        extensions = _snapshot_extensions(extensions)
        if extensions is not None:
            lifted = set(extensions.config.get("always_deny.lift", []))
            if lifted:
                policy = replace(policy, always_deny_commands=tuple(
                    command for command in ALWAYS_DENY_COMMANDS
                    if " ".join(command) not in lifted
                ))
            policy = replace(
                policy,
                sandbox_readable=tuple(extensions.config.get("sandbox.readable", policy.sandbox_readable)),
                allow_uncheckable_shell=extensions.config.get(
                    "always_deny.allow_uncheckable_shell", False
                ),
                allow_discarding_work=extensions.config.get(
                    "always_deny.allow_discarding_work", False
                ),
            )
        self._provider = provider
        self._policy = policy
        self._repo_root = policy.repo_root
        self._leases = WorkspaceLeases(policy.repo_root)
        self._base_policy = replace(policy)
        self._extensions = extensions
        config_shell_grants, config_fetch_grants = (
            permission_grants(extensions.config, extensions.ceiling)
            if extensions is not None else ((), ())
        )
        project_root = Path(policy.repo_root).resolve()
        grant_key = hashlib.sha256(str(project_root).encode("utf-8")).hexdigest()
        self._project_grants_path = (
            Path(project_grants_root) if project_grants_root is not None
            else symphonai_home() / "permissions"
        ) / f"{grant_key}.json"
        try:
            stored_shell_grants, stored_fetch_grants = load_project_permission_grants(
                self._project_grants_path,
            )
        except PermissionGrantError as exc:
            _LOG.error("Ignoring invalid project permission grants: %s", exc)
            stored_shell_grants, stored_fetch_grants = (), ()
        self._configured_shell_grants = config_shell_grants
        self._configured_fetch_grants = config_fetch_grants
        self._stored_project_shell_grants = stored_shell_grants
        self._stored_project_fetch_grants = stored_fetch_grants
        self._project_shell_grants, self._project_fetch_grants = restrict_permission_grants(
            (*config_shell_grants, *stored_shell_grants),
            (*config_fetch_grants, *stored_fetch_grants),
            None if extensions is None else extensions.ceiling,
        )
        self._policy.fetch_allowlist = list(dict.fromkeys([
            *self._policy.fetch_allowlist, *self._project_fetch_grants,
        ]))
        self._base_policy.fetch_allowlist = list(self._policy.fetch_allowlist)
        permitted = self.permitted_modes()
        if not permitted:
            raise ModeSelectionError("agents.ceiling.modes must permit at least one mode")
        self._starting_mode = "ask" if "ask" in permitted else permitted[0]
        self._policy.mode = self._starting_mode
        self._broker = broker
        self._person_jobs = JobManager(
            policy.shell_output_limit_chars,
            owner="person",
            on_change=lambda job: self._job_changed(job, None),
        )
        self._system_prompt = system_prompt
        self._base_prompt = base_prompt
        self._subagent_base_prompt = subagent_base_prompt
        if working_dir is None:
            current = Path.cwd().resolve()
            self._working_dir = current if current.is_relative_to(policy.repo_root) else policy.repo_root
        else:
            self._working_dir = Path(working_dir)
        self._max_turns = max_turns
        self._model = model
        self._effort: str | None = None
        self._provider_factory = provider_factory
        memory_config = None if extensions is None else extensions.config
        self._memory_model = None if memory_config is None else memory_config.get("memory.model")
        configured_memory_provider = (
            None if memory_config is None else memory_config.get("memory.provider")
        )
        self._memory_provider: ModelProvider | None = None
        self._memory_extraction_enabled = (
            True if memory_config is None else memory_config.get("memory.extract", True)
        )
        if memory_extraction_enabled is not None:
            self._memory_extraction_enabled = memory_extraction_enabled
        if configured_memory_provider is not None:
            try:
                self._memory_provider = (
                    None
                    if provider_factory is None
                    else provider_factory(
                        configured_memory_provider, self._memory_model, None
                    )
                )
            except Exception:
                self._memory_provider = None
            if self._memory_provider is None:
                print(
                    "memory extraction disabled: configured provider is unavailable",
                    file=sys.stderr,
                )
                self._memory_extraction_enabled = False
        self._provider_choice = (
            {"name": provider.name, "model": model, "base_url": provider.base_url}
            if provider is not None and provider.name in ("anthropic", "gemini", "openai")
            else None
        )
        endpoint_key = None if extensions is None else extensions.config.get("search.endpoint")
        self._search_backend = (
            None if endpoint_key is None else HttpJsonSearchBackend(search_endpoint(endpoint_key))
        )
        self._hooks = (
            None
            if extensions is None
            else getattr(extensions, "hook_runner", lambda **_: None)(cwd=policy.repo_root)
        )
        self._mcp_tools = mcp_tools
        self.mcp_pool = mcp_pool
        self._mcp_instructions = tuple(mcp_instructions)
        self._lsp = lsp
        self._restricted_tool_names: frozenset[str] | None = None
        self._price_table = price_table
        self._leader_budget = None
        self._subagent_budget = None
        self._subagent_turn_limit_configured = False
        if extensions is not None:
            self._subagent_turn_limit_configured = "budgets.subagent.max_turns" in extensions.config.values
            self._leader_budget, self._subagent_budget, self._price_table = resolve_run_budgets(
                extensions.config,
                repo_root=policy.repo_root,
                leader_max_turns=max_turns,
                subagent_max_turns=None,
                price_table=price_table,
            )
        self._chat_token_budget = chat_token_budget
        self._chat_recent_turns = chat_recent_turns
        self._active: _ActiveRun | None = None
        self._user_command: _UserCommand | None = None
        self._active_by_session: dict[str, _ActiveRun] = {}
        self._open_conversations: dict[str, tuple[Leader, SessionStore]] = {}
        self._instruction_warnings_by_session: dict[str, list[str]] = {}
        self._plans_by_session: dict[str, dict] = {}
        self._pending_user_context: list[str] = []
        self._policy_by_session: dict[str, PermissionPolicy] = {}
        self._approvals_by_session: dict[str, ApprovalBroker] = {}
        self._session_by_root_agent: dict[str, str] = {}
        self._session_by_runtime_run: dict[str, str] = {}
        self._goal: Goal | None = None
        self._goal_session_id: str | None = None
        self._goal_check: GoalCheck | None = None
        self._goal_check_thread: threading.Thread | None = None
        self._goals_by_session: dict[str, Goal] = {}
        self._goal_checks_by_session: dict[str, GoalCheck] = {}
        self._conversation: tuple[Leader, SessionStore] | None = None
        self._context_report: ContextReport | None = None
        self._usage_by_agent: dict[str, tuple[str, dict[str, UsageTotals]]] = {}
        self._context_by_session: dict[str, ContextReport | None] = {}
        self._usage_by_session: dict[str, dict[str, tuple[str, dict[str, UsageTotals]]]] = {}
        self._closing = False
        self._preview = None
        self._preview_lock = threading.RLock()
        self._preview_last_url = ""
        self._preview_marks = {}
        from symphonai_host.spec_loop import LoopController
        self._loop_store = LoopStore(self._repo_root)
        self._loops = LoopController(self, self._loop_store)
        self._sessions_root = default_sessions_root() if sessions_root is None else Path(sessions_root)
        self._memory_root = default_memory_root() if memory_root is None else Path(memory_root)
        self._memory: AgentMemory | None = None
        self._memory_open_attempted = False
        self._lock = threading.RLock()
        self._publish_approval_callback = publish_approval or (lambda _: False)
        self._approval_timeout = approval_timeout
        self.approvals = self._new_approval_broker(None)
        self._policy.approval_callback = self.approvals.callback
        from symphonai_host.keep_awake import KeepAwake
        self._keep_awake = KeepAwake(
            enabled=extensions is None or extensions.config.get("power.keep_awake", True),
            busy=self._keep_awake_busy,
        )

    def restart_mcp(self, server: str) -> None:
        pool = self.mcp_pool
        if pool is None:
            raise ValueError("MCP is unavailable")
        with self._lock:
            if self._closing:
                raise ValueError("Project is closing")
        tools = pool.start(server)
        tools.update(pool.resource_tools)
        with self._lock:
            if self._closing:
                raise ValueError("Project is closing")
            self._mcp_tools = tools
            self._mcp_instructions = pool.instructions

    def replace_extensions(self, extensions, pool, tools) -> bool:
        extensions = _snapshot_extensions(extensions)
        config = extensions.config
        memory_provider = None
        memory_model = config.get("memory.model")
        enabled = (config.get("memory.extract", True) if self._memory_extraction_override is None
                   else self._memory_extraction_override)
        name = config.get("memory.provider")
        if name is not None:
            memory_provider = (None if self._provider_factory is None else
                               self._provider_factory(name, memory_model, None))
            enabled = enabled and memory_provider is not None
        budgets = resolve_run_budgets(config, repo_root=self._repo_root,
                                      leader_max_turns=self._max_turns,
                                      subagent_max_turns=None, price_table=self._reload_price_table)
        shell, fetch = permission_grants(config, extensions.ceiling)
        policy = extensions.ceiling.bound(replace(
            self._reload_policy,
            shell_sandbox=config.get("sandbox.shell", False),
            sandbox_network=config.get("sandbox.network", False),
            sandbox_readable=tuple(config.get("sandbox.readable", ())),
            always_deny_commands=tuple(command for command in ALWAYS_DENY_COMMANDS
                                      if " ".join(command) not in config.get("always_deny.lift", [])),
            allow_uncheckable_shell=config.get("always_deny.allow_uncheckable_shell", False),
            allow_discarding_work=config.get("always_deny.allow_discarding_work", False),
        ))
        endpoint = config.get("search.endpoint")
        hooks = extensions.hook_runner(cwd=self._repo_root)
        modes = tuple(mode for mode in PERMISSION_MODES
                      if extensions.ceiling.modes is None or mode in extensions.ceiling.modes)
        if not modes:
            raise ValueError("agents.ceiling.modes must permit at least one mode")
        with self._lock:
            if self._closing:
                raise ValueError("Project is closing")
            retained = bool(self._open_conversations)
            self._extensions = extensions
            self.mcp_pool = pool
            self._mcp_tools = tools
            self._mcp_instructions = pool.instructions
            self._hooks = hooks
            self._memory_provider, self._memory_model = memory_provider, memory_model
            self._memory_extraction_enabled = enabled
            self._search_backend = None if endpoint is None else HttpJsonSearchBackend(search_endpoint(endpoint))
            self._leader_budget, self._subagent_budget, self._price_table = budgets
            self._subagent_turn_limit_configured = "budgets.subagent.max_turns" in config.values
            self._configured_shell_grants, self._configured_fetch_grants = shell, fetch
            self._project_shell_grants, self._project_fetch_grants = restrict_permission_grants(
                (*shell, *self._stored_project_shell_grants),
                (*fetch, *self._stored_project_fetch_grants), extensions.ceiling,
            )
            policy.fetch_allowlist = list(self._project_fetch_grants)
            self._starting_mode = "ask" if "ask" in modes else modes[0]
            policy.mode = self._starting_mode
            self._next_policy = policy
            self._base_policy = replace(policy)
            if self._conversation is None:
                self._policy = policy
                self.approvals = self._new_approval_broker(None)
                policy.approval_callback = self.approvals.callback
            return retained

    def _new_approval_broker(self, session_id: str | None) -> ApprovalBroker:
        def review(request):
            from symphonai_host.auto_review import review_call
            from symphonai_api.permissions import DenialReason, PermissionDecision
            from symphonai_api.secrets import mask_secrets

            with self._lock:
                policy = self._policy_by_session.get(session_id, self._policy)
                if policy.mode != "auto":
                    return None
                conversation = self._open_conversations.get(session_id)
                active = self._active_by_session.get(session_id)
            headless = False
            try:
                if conversation is None:
                    raise ValueError("No conversation for review")
                leader, store = conversation
                headless = store.read_meta().get("headless") is True
                records, _ = read_records(store.directory / "run.jsonl")
                person = ""
                for record in records:
                    if record.get("type") == "message":
                        message = message_from_json(record["data"])
                        if (message.role is Role.USER and not SUBAGENT_NOTICE_PATTERN.match(message.text)
                            and not message.text.startswith(("[Stop hook] ", "Goal check failed (round ", "Round "))):
                            person = message.text
                found = None
                for path in (store.directory / "run.jsonl", *sorted(store.directory.glob("agent-*.jsonl"))):
                    transcript, _ = read_records(path)
                    last_text = ""
                    for record in transcript:
                        if record.get("type") != "message":
                            continue
                        message = message_from_json(record["data"])
                        if message.role is Role.ASSISTANT:
                            last_text = message.text or last_text
                            if any(call.id == request.tool_call_id for call in message.tool_calls):
                                found = (path, record, last_text)
                    if found is not None:
                        break
                if found is None:
                    raise ValueError("Call context unavailable")
                path, record, last_text = found
                owner = leader._agent if path.name == "run.jsonl" else next(
                    (item.agent for item in leader._dispatch_tool.pool.values()
                     if item.agent_ref.agent_id == record["agent_id"]), None)
                if owner is None:
                    raise ValueError("Call owner unavailable")
                if owner._policy.mode != "auto":
                    return None
                provider = leader._config.leader_provider
                model = leader._agent._previous_answering_model or leader._leader_spec.model.model or provider.model
                decision, reason = review_call(
                    provider, model, request, user_text=person, agent_text=last_text,
                    repo_root=owner._policy.repo_root,
                    on_usage=lambda model, usage: self._record_background_usage(session_id, model or "unknown", usage),
                    cancel=None if active is None else active.cancel,
                )
                event = AutoReviewed(
                    agent_id=record["agent_id"], run_id=record["run_id"], turn_id=record.get("turn_id"),
                    operation=request.operation, target=mask_secrets(request.target),
                    decision=decision, reason=reason, tool_call_id=request.tool_call_id,
                )
                store.writer_for(record["agent_id"], is_root=path.name == "run.jsonl").append(
                    "auto_reviewed", run_id=event.run_id, agent_id=event.agent_id,
                    turn_id=event.turn_id, data={"operation": event.operation, "target": event.target,
                        "decision": event.decision, "reason": event.reason, "tool_call_id": event.tool_call_id},
                )
                self._publish_session(event, session_id)
                if decision == "approve":
                    return PermissionDecision.allow()
                return PermissionDecision.deny(f"Auto reviewer denied: {reason}", denial=DenialReason.DENIED_BY_USER)
            except Exception:
                if headless or self._closing or (active is not None and active.cancel.cancelled):
                    return PermissionDecision.deny("Auto reviewer failed; call denied", denial=DenialReason.APPROVAL_FAILED)
                return None

        return ApprovalBroker(
            self._publish_approval_callback,
            timeout=self._approval_timeout,
            session_id=session_id,
            shell_grants=self._project_shell_grants,
            fetch_grants=self._project_fetch_grants,
            review=review,
        )

    @property
    def provider(self) -> ModelProvider | None:
        """The provider currently selected for the leader."""
        with self._lock:
            return self._provider

    @property
    def policy(self) -> PermissionPolicy:
        return self._policy

    @property
    def starting_mode(self) -> str:
        return self._starting_mode

    @property
    def extensions(self) -> Extensions | None:
        return self._extensions

    @property
    def active_run_id(self) -> str | None:
        with self._lock:
            return None if self._active is None else self._active.run_id

    @property
    def active(self) -> bool:
        return self.active_run_id is not None

    @property
    def runtime_run_id(self) -> str | None:
        """The root runtime id once its RunStarted event has been published."""
        with self._lock:
            return None if self._active is None else self._active.runtime_run_id

    @property
    def sessions_root(self):
        return self._sessions_root

    def select_provider(
        self,
        provider: ModelProvider,
        model: str | None = None,
        effort: str | None = None,
        choice: dict | None = None,
    ) -> None:
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            conversation = self._conversation
            provider_changed = (
                conversation is not None
                and _provider_identity(self._provider) != _provider_identity(provider)
            )
            previous = (self._provider, self._model, self._effort, self._provider_choice)
            self._provider = provider
            self._model = model
            self._effort = effort
            self._provider_choice = choice
            if conversation is None:
                return
            leader, session = conversation
            try:
                if provider_changed:
                    replacement = self._new_leader(session, jobs=leader.background_jobs)
                    replacement.seed_chat(
                        _without_vendor_state(leader._chat_messages),
                        persisted=True,
                    )
                    self._conversation = replacement, session
                    self._open_conversations[session.run_id] = self._conversation
                else:
                    leader.select_model(
                        model if model is not None else provider.model,
                        effort,
                    )
                def change(metadata):
                    metadata["provider_choice"] = (
                        choice
                        if choice is not None
                        else {"name": provider.name, "model": model, "effort": effort}
                    )
                    if provider_changed:
                        metadata["provider_state_reset"] = True
                session.update_meta(change)
            except Exception:
                self._provider, self._model, self._effort, self._provider_choice = previous
                self._conversation = conversation
                self._open_conversations[conversation[1].run_id] = conversation
                raise

    def selection(self) -> dict | None:
        """Return the model selected for the next turn."""
        with self._lock:
            if self._provider is None:
                return None
            return {
                "provider": self._provider.name,
                "model": self._model if self._model is not None else self._provider.model,
                "effort": self._effort,
            }

    def draft_agent(self, description: str) -> str:
        """Make one tool-free call using the currently selected model."""
        from symphonai_api.models import ModelRequest
        from symphonai_api.roles import agent_drafting_prompt

        with self._lock:
            provider = self._provider
            model = self._model
            effort = self._effort
            tools = sorted(self.available_tool_names())
        if provider is None:
            raise ProviderSelectionError("no provider selected")
        return provider.create_response(ModelRequest(
            messages=[Message(Role.USER, agent_drafting_prompt(description, tools))],
            model=model, effort=effort,
        )).message.text

    def permitted_modes(self) -> tuple[str, ...]:
        ceiling = None if self._extensions is None else getattr(self._extensions, "ceiling", None)
        modes = None if ceiling is None else ceiling.modes
        return tuple(
            mode for mode in PERMISSION_MODES
            if modes is None or mode in modes
        )

    def select_mode(self, mode: object) -> str:
        with self._lock:
            permitted = self.permitted_modes()
            if mode not in permitted:
                choices = ", ".join(permitted) or "none"
                raise ModeSelectionError(f"permitted modes: {choices}")
            self._policy.mode = mode
            return mode

    def start_loop(self, goal: str, *, runner_pid: int | None=None, runner_lock: bool=False) -> dict:
        return self._loops.start_loop(goal, runner_pid=runner_pid, runner_lock=runner_lock).to_payload()

    def start_roadmap_goal(self, goal_id: str) -> tuple[dict, dict]:
        roadmap, loop = self._loops.start_roadmap_goal(goal_id)
        return roadmap, loop.to_payload()

    def approve_loop_plan(self, loop_id: str) -> dict:
        return self._loops.approve_loop_plan(loop_id).to_payload()

    def drop_loop_cards(self, loop_id: str, card_ids: list[str]) -> dict:
        return self._loops.drop_loop_cards(loop_id, card_ids).to_payload()

    def revise_loop_plan(self, loop_id: str, note: str) -> dict:
        return self._loops.revise_loop_plan(loop_id, note).to_payload()

    def cancel_loop(self, loop_id: str) -> dict:
        return self._loops.cancel_loop(loop_id).to_payload()

    def commit_loop(self, loop_id: str, action: str) -> dict:
        return self._loops.commit_loop(loop_id, action).to_payload()

    def resume_loop(self, loop_id: str, *, runner_pid: int | None=None, runner_lock: bool=False) -> dict:
        return self._loops.resume_loop(loop_id, runner_pid=runner_pid, runner_lock=runner_lock).to_payload()

    def loop_card_details(self, loop_id: str, card_id: str) -> dict:
        return self._loops.loop_card_details(loop_id, card_id)

    def recover_loops(self) -> None:
        return self._loops.recover_loops()

    def loop_usage(self, loop: dict) -> dict:
        return self._loops.loop_usage(Loop.from_payload(loop))

    def _save_goal(self, session_id: str, goal: Goal | None) -> None:
        current = self._conversation
        if current is not None and current[1].run_id == session_id:
            store = current[1]
            close = False
        else:
            store = SessionStore.open(self._sessions_root, session_id)
            close = True
        try:
            def change(metadata):
                if goal is None:
                    metadata.pop("goal", None)
                else:
                    metadata["goal"] = goal.payload()
            store.update_meta(change)
        finally:
            if close:
                store.close()

    def _goal_event(
        self, change: str, goal: Goal | None, run_id: str = "", agent_id: str = "",
        session_id: str | None = None,
    ) -> None:
        session_id = self._goal_session_id if session_id is None else session_id
        if goal is not None:
            session_id = next((sid for sid, item in self._goals_by_session.items() if item is goal), session_id)
        self._broker.publish(GoalChanged(
            agent_id=agent_id,
            run_id=run_id,
            change=change,
            phase="" if goal is None else goal.phase,
            rounds=0 if goal is None else goal.rounds,
            max_rounds=0 if goal is None else goal.max_rounds,
            reason="" if goal is None else goal.reason,
            last_check=None if goal is None else goal.last_check,
            session_id=session_id,
        ))

    def _save_plan(self, session_id: str, plan: dict) -> None:
        current = self._conversation
        if current is not None and current[1].run_id == session_id:
            store = current[1]
            close = False
        else:
            store = SessionStore.open(self._sessions_root, session_id)
            close = True
        try:
            def change(metadata):
                metadata["plan"] = plan
            store.update_meta(change)
        finally:
            if close:
                store.close()

    def _update_plan_for_session(
        self, session_id: str, steps: list[dict], explanation: str,
    ) -> None:
        with self._lock:
            plan = {"steps": [dict(step) for step in steps], "explanation": explanation}
            self._plans_by_session[session_id] = plan
            self._save_plan(session_id, plan)
            leader = self._open_conversations.get(session_id, (None, None))[0]
            active = self._active_by_session.get(session_id)
            self._publish(
                "" if active is None else active.run_id,
                PlanChanged(
                    agent_id="" if leader is None else leader.agent_ref.agent_id,
                    run_id="" if active is None else active.run_id,
                    steps=plan["steps"],
                    explanation=explanation,
                    session_id=session_id,
                ),
                session_id=session_id,
            )

    def start_goal(self, objective: str, check: tuple[str, ...], max_rounds: int) -> str:
        with self._lock:
            session_id = None if self._conversation is None else self._conversation[1].run_id
            if session_id is not None and session_id in self._goal_checks_by_session:
                raise RunActiveError("goal check")
            goal = Goal(objective, check, max_rounds=max_rounds)
            return self.start(objective, _new_goal=goal)


    def goal_snapshot(self) -> dict | None:
        with self._lock:
            if self._goal is None or self._goal_session_id is None:
                return None
            return self._goal.payload()

    def _goal_for_session(self, session_id: str) -> dict | None:
        with self._lock:
            goal = self._goals_by_session.get(session_id)
            if goal is None:
                return None
            return goal.payload()

    def _update_goal_for_session(
        self, session_id: str, status: str, message: str,
    ) -> tuple[bool, str]:
        with self._lock:
            goal = self._goals_by_session.get(session_id)
            if goal is None:
                return False, "No goal is set."
            if goal.phase != "active":
                return False, f"Goal is {goal.phase}."
            if status == "complete" and goal.check:
                return True, "The check decides completion; it runs when this round ends."
            goal.phase = status
            goal.reason = message
            self._save_goal(session_id, goal)
            if status in ("complete", "blocked"):
                self._loops._finish_spec_run(session_id, status)
            if self._goal_session_id == session_id:
                self._goal = goal
            active = self._active_by_session.get(session_id)
            run_id = "" if active is None else active.run_id
            leader = self._open_conversations.get(session_id, (None, None))[0]
            agent_id = "" if leader is None else leader.agent_ref.agent_id
            self._goal_event("update", goal, run_id, agent_id)
            return True, f"Goal marked {status}."

    def goal_state(self, action: str) -> dict | None:
        with self._lock:
            if self._goal is None or self._goal_session_id is None:
                raise KeyError("no goal")
            goal = self._goal
            session_id = self._goal_session_id
            if action == "clear":
                self._save_goal(session_id, None)
                context = self._goal_checks_by_session.get(session_id)
                if context is not None and context.goal is goal:
                    context.cleared = True
                    context.cancel.set()
                self._goal = None
                self._goal_session_id = None
                self._goals_by_session.pop(session_id, None)
                self._goal_event("clear", None, session_id=session_id)
                return None
            if action == "pause":
                goal.phase = "paused"
                goal.reason = "paused"
                context = self._goal_checks_by_session.get(session_id)
                if context is not None and context.goal is goal:
                    context.interrupted = "interrupted"
                self._save_goal(session_id, goal)
                self._goal_event("pause", goal, agent_id=self._root_agent_id())
                return goal.payload()
            if action != "resume":
                raise ValueError("unknown goal action")
            if session_id in self._goal_checks_by_session:
                raise RunActiveError("goal check")
            if goal.phase in ("active", "complete"):
                raise RunActiveError("goal is already active or complete")
            active = self._active
            if active is not None:
                conversation = self._open_conversations.get(session_id)
                if (
                    not active.goal_round
                    or conversation is None
                    or conversation[1].run_id != session_id
                ):
                    raise RunActiveError(active.run_id)
                goal.phase = "active"
                goal.reason = ""
                self._save_goal(session_id, goal)
                self._goal_event("resume", goal, active.run_id, active.root_agent_id)
                return goal.payload()
            goal.phase = "active"
            goal.reason = ""
            self._save_goal(session_id, goal)
            self._goal_event("resume", goal, agent_id=self._root_agent_id())
            if not goal.check:
                if goal.rounds >= goal.max_rounds:
                    goal.phase = "blocked"
                    goal.reason = "rounds exhausted"
                    self._save_goal(session_id, goal)
                    self._goal_event("round", goal, agent_id=self._root_agent_id())
                    return goal.payload()
                self._start_for_session(session_id, _unchecked_goal_prompt(goal))
                self._goal_event("round", goal, agent_id=self._root_agent_id())
                return goal.payload()
            leader = self._open_conversations[session_id][0]
            context = GoalCheck(goal, session_id, threading.Event(), leader.agent_ref.agent_id)
            self._goal_check = context
            self._goal_checks_by_session[session_id] = context
            thread = threading.Thread(
                target=self._perform_goal_check,
                args=(context, ""),
                name="symphonai-goal-check",
                daemon=True,
            )
            self._goal_check_thread = thread
            thread.start()
            return goal.payload()

    def _root_agent_id(self) -> str:
        return "" if self._conversation is None else self._conversation[0].agent_ref.agent_id

    def _start_for_session(self, session_id: str, prompt: str) -> str:
        return self.start(prompt, _goal_round=True, session_id=session_id)

    def _publish_session(self, event: Event, session_id: str) -> None:
        self._remember_event_session(event, session_id)
        self._broker.publish(event)

    @staticmethod
    def _checkpoint_edit_totals(checkpoints, selected_key=None):
        """Count each turn against its before/after checkpoint file versions."""
        edits = {}
        after_by_path = {}
        for entry in reversed(checkpoints.entries()):
            if entry.path not in after_by_path:
                after_by_path[entry.path] = _checkpoint_file_bytes(checkpoints, entry.path)
            after = after_by_path[entry.path]
            before = None if entry.backup is None else (checkpoints.directory / entry.backup).read_bytes()
            after_by_path[entry.path] = before
            if selected_key is not None and entry.key != selected_key:
                continue
            rows = edits.setdefault(entry.key, [])
            if before == after:
                continue
            if b"\x00" in (before or b"") or b"\x00" in (after or b""):
                added = removed = 0
            else:
                diff = _diff_result("turn-edits", entry.path,
                                    (before or b"").decode("utf-8", errors="replace"),
                                    (after or b"").decode("utf-8", errors="replace"))
                added = diff.payload["lines_added"]
                removed = diff.payload["lines_removed"]
            rows.append({"path": entry.path, "added": added, "removed": removed})
        return {key: sorted(rows, key=lambda row: row["path"]) for key, rows in edits.items()}

    def _record_turn_edits(self, session_id, event):
        conversation = self._open_conversations.get(session_id)
        if conversation is None or event.agent_id != conversation[0].agent_ref.agent_id:
            return
        leader, session = conversation
        checkpoints = leader._config.checkpoints
        keys = () if checkpoints is None else checkpoints.keys()
        if not keys:
            return
        key = keys[-1]
        edits = self._checkpoint_edit_totals(checkpoints, key).get(key, [])
        session.writer_for(event.agent_id, is_root=True).append(
            "checkpoint", run_id=event.run_id, agent_id=event.agent_id, turn_id=None,
            data={"key": key, "edits": edits},
        )
        self._broker.publish(HistoryMessage(
            role="assistant", text="", tool_calls=[], turn_id=None, session_id=session_id,
            checkpoint_key=key, edits=edits,
        ))

    @staticmethod
    def _checkpoint_replay_data(
        records: list[dict], subagent_records: list[list[dict]] | None = None, checkpoints=None,
    ) -> tuple[dict[str, str], dict[str, list[dict[str, str | int]]]]:
        keys: dict[str, str] = {}
        root_run_keys: dict[str, str] = {}
        edits: dict[str, list[dict[str, str | int]]] = {}
        pending_key: str | None = None
        current_root_run: str | None = None
        recorded_edits = {}
        for record in records:
            if record.get("type") == "run_started":
                data = record.get("data")
                if isinstance(data, dict) and data.get("parent_run_id") is None:
                    current_root_run = record.get("run_id")
            elif record.get("type") == "checkpoint":
                key = record.get("data", {}).get("key")
                if isinstance(key, str) and isinstance(record.get("data", {}).get("edits"), list):
                    recorded_edits[key] = record["data"]["edits"]
                    continue
                pending_key = key if isinstance(key, str) else None
                if current_root_run is not None and pending_key is not None:
                    root_run_keys[current_root_run] = pending_key
            elif pending_key is not None and record.get("type") == "message":
                try:
                    message = message_from_json(record.get("data", {}))
                except (KeyError, TypeError, ValueError):
                    continue
                if message.role is Role.USER:
                    record_id = record.get("record_id")
                    if isinstance(record_id, str):
                        keys[record_id] = pending_key
                    pending_key = None

        transcripts = [records, *(subagent_records or [])]
        runs: dict[str, dict[str, str | None]] = {}
        worktree_dispatches: set[tuple[str, str]] = set()
        for transcript in transcripts:
            for record in transcript:
                run_id = record.get("run_id")
                data = record.get("data")
                if record.get("type") == "run_started" and isinstance(run_id, str) and isinstance(data, dict):
                    parent_run_id = data.get("parent_run_id")
                    runs[run_id] = {
                        "agent_id": record.get("agent_id"),
                        "agent_name": data.get("agent_name"),
                        "parent_run_id": parent_run_id if isinstance(parent_run_id, str) else None,
                    }
                if record.get("type") != "message" or not isinstance(data, dict):
                    continue
                try:
                    message = message_from_json(data)
                except (KeyError, TypeError, ValueError):
                    continue
                if message.role is Role.ASSISTANT and isinstance(run_id, str):
                    for call in message.tool_calls:
                        if (
                            call.name == "dispatch_subagent"
                            and call.arguments.get("isolation") == "worktree"
                            and isinstance(call.arguments.get("subagent_name"), str)
                        ):
                            worktree_dispatches.add((run_id, call.arguments["subagent_name"]))

        worktree_agents = {
            run["agent_id"]
            for run in runs.values()
            if isinstance(run["agent_id"], str)
            and isinstance(run["parent_run_id"], str)
            and isinstance(run["agent_name"], str)
            and (run["parent_run_id"], run["agent_name"]) in worktree_dispatches
        }
        run_keys = dict(root_run_keys)
        excluded_runs: set[str] = set()
        for _ in range(len(runs)):
            changed = False
            for run_id, run in runs.items():
                if run_id in run_keys or run_id in excluded_runs:
                    continue
                parent_run_id = run["parent_run_id"]
                if parent_run_id in excluded_runs:
                    excluded_runs.add(run_id)
                    changed = True
                elif parent_run_id in run_keys:
                    if (
                        run["agent_id"] in worktree_agents
                        or (parent_run_id, run["agent_name"]) in worktree_dispatches
                    ):
                        excluded_runs.add(run_id)
                    else:
                        run_keys[run_id] = run_keys[parent_run_id]
                    changed = True
            if not changed:
                break

        ordered_records = [
            (index, record)
            for index, record in enumerate(record for transcript in transcripts for record in transcript)
            if record.get("type") == "message"
        ]
        ordered_records.sort(key=lambda item: (item[1].get("ts", ""), item[0]))
        for _, record in ordered_records:
            run_id = record.get("run_id")
            key = run_keys.get(run_id) if isinstance(run_id, str) else None
            if key is None:
                continue
            try:
                message = message_from_json(record.get("data", {}))
            except (KeyError, TypeError, ValueError):
                continue
            result = message.tool_result
            payload = None if result is None else result.payload
            if (
                message.role is Role.TOOL
                and result is not None
                and result.ok
                and isinstance(payload, dict)
                and payload.get("kind") == "file_diff"
                and isinstance(payload.get("path"), str)
                and type(payload.get("lines_added")) is int
                and type(payload.get("lines_removed")) is int
            ):
                path = payload["path"]
                by_path = {edit["path"]: edit for edit in edits.setdefault(key, [])}
                edit = by_path.get(path)
                if edit is None:
                    edit = {"path": path, "added": 0, "removed": 0}
                    edits[key].append(edit)
                edit["added"] += payload["lines_added"]
                edit["removed"] += payload["lines_removed"]
        edits = {key: sorted(rows, key=lambda row: row["path"]) for key, rows in edits.items()}
        edits.update(recorded_edits)
        return keys, edits

    @staticmethod
    def _turn_replay_data(records: list[dict]) -> dict[str, dict[str, str | None]]:
        roots: dict[str, dict[str, str | None]] = {}
        messages: dict[str, list[tuple[str, object]]] = {}
        for record in records:
            run_id = record.get("run_id")
            record_type = record.get("type")
            data = record.get("data")
            if record_type == "run_started" and isinstance(run_id, str) and isinstance(data, dict):
                if data.get("parent_run_id") is None:
                    roots[run_id] = {"started_at": record.get("ts"), "ended_at": None, "outcome": None}
            elif record_type in {"run_finished", "run_failed"} and isinstance(run_id, str) and run_id in roots:
                roots[run_id]["ended_at"] = record.get("ts")
                roots[run_id]["outcome"] = (
                    "failed" if record_type == "run_failed"
                    else "cancelled" if isinstance(data, dict) and data.get("stopped_reason") == "cancelled"
                    else "finished"
                )
            elif record_type == "message" and isinstance(run_id, str) and isinstance(data, dict):
                try:
                    message = message_from_json(data)
                except (KeyError, TypeError, ValueError):
                    continue
                if run_id in roots:
                    messages.setdefault(run_id, []).append((record.get("record_id", ""), message))

        result: dict[str, dict[str, str | None]] = {}
        for run_id, run in roots.items():
            root_messages = messages.get(run_id, [])
            user_ids = [record_id for record_id, message in root_messages if message.role is Role.USER]
            assistant_ids = [record_id for record_id, message in root_messages if message.role is Role.ASSISTANT]
            for record_id in user_ids:
                if isinstance(record_id, str):
                    result[record_id] = {"turn_started_at": run["started_at"]}
            last_record_id = (assistant_ids or user_ids)[-1] if assistant_ids or user_ids else None
            if isinstance(last_record_id, str):
                result.setdefault(last_record_id, {}).update({
                    "turn_ended_at": run["ended_at"],
                    "turn_outcome": run["outcome"],
                })
        return result

    @staticmethod
    def _replay_tool_call(call, result=None, review=None) -> dict:
        from symphonai_api.secrets import mask_secrets

        replay = {"id": call.id, "name": call.name}
        if review is not None:
            replay["auto_reviews"] = review
        if call.name == "dispatch_subagent":
            for key in ("subagent_name", "task"):
                value = call.arguments.get(key)
                if isinstance(value, str):
                    replay[key] = value
            if result is not None:
                replay["ok"] = result.ok
                if result.ok:
                    text = mask_secrets(tool_result_text(result))
                    replay["output"] = text[:4000]
                    replay["output_truncated"] = len(text) > 4000
                else:
                    replay["error"] = mask_secrets(result.error or "tool call failed")
        elif call.name in FILE_EDIT_TOOLS:
            replay["target"] = mask_secrets(call_target(call.name, call.arguments))
            if result is not None:
                replay["ok"] = result.ok
        elif call.name == "notify_person":
            if result is not None and result.ok:
                replay["text"] = mask_secrets(call.arguments.get("text", ""))
        elif call.name == "update_plan":
            for key in ("plan", "explanation"):
                value = call.arguments.get(key)
                if isinstance(value, (list, str)):
                    replay[key] = value
        elif call.name == "ask_user":
            question = call.arguments.get("question")
            options = call.arguments.get("options")
            if isinstance(question, str):
                replay["question"] = question
            if isinstance(options, list) and all(isinstance(option, str) for option in options):
                replay["options"] = options
            if result is not None and result.ok:
                if result.content.startswith("The person answered: "):
                    replay["answer"] = result.content.removeprefix("The person answered: ")
                elif result.content == "The person skipped the question.":
                    replay["skipped"] = True
        elif call.name == "propose_plan":
            plan = call.arguments.get("plan")
            if isinstance(plan, str):
                replay["plan"] = plan
            if result is not None and result.ok:
                for mode in ("ask", "allow"):
                    if result.content == (
                        "The person approved the plan. The conversation is now in "
                        f"{mode} mode; carry out the plan."
                    ):
                        replay["decision"] = mode
                        break
                if result.content == "The person wants to keep planning.":
                    replay["decision"] = "keep_planning"
                elif result.content.startswith("The person wants to keep planning: "):
                    replay["decision"] = "keep_planning"
                    replay["feedback"] = result.content.removeprefix(
                        "The person wants to keep planning: "
                    )
        hidden_output_tools = {
            "dispatch_subagent", "update_plan", "ask_user", "propose_plan",
            *FILE_EDIT_TOOLS,
        }
        if call.name not in hidden_output_tools:
            replay["target"] = call_target(call.name, call.arguments)
            if result is not None:
                output = ""
                output_truncated = False
                if not (isinstance(result.payload, dict) and result.payload.get("kind") == "file_diff"):
                    text = tool_result_text(result)
                    output = text[:4000]
                    output_truncated = len(text) > 4000
                replay.update({
                    "ok": result.ok,
                    "output": output,
                    "output_truncated": output_truncated,
                })
        return replay

    @staticmethod
    def _replay_tool_calls(message: Message, results: Mapping[str, object], reviews=None) -> list[dict]:
        return [
            HostRun._replay_tool_call(call, results.get(call.id), (reviews or {}).get(call.id))
            for call in message.tool_calls
        ]

    @staticmethod
    def _saved_auto_reviews(records):
        reviews = {}
        for record in records:
            data = record.get("data")
            if record.get("type") == "auto_reviewed" and isinstance(data, dict) and isinstance(data.get("tool_call_id"), str):
                reviews.setdefault(data["tool_call_id"], []).append(data)
        return reviews

    @staticmethod
    def _replay_tool_results(messages: list[Message]) -> dict[str, object]:
        return {
            message.tool_result.tool_call_id: message.tool_result
            for message in messages
            if message.tool_result is not None and message.tool_result.tool_call_id
        }

    def _remember_event_session(self, event: Event, session_id: str) -> None:
        runtime_run_id = getattr(event, "run_id", None)
        if isinstance(runtime_run_id, str) and runtime_run_id:
            self._session_by_runtime_run[runtime_run_id] = session_id
            if len(self._session_by_runtime_run) > 4096:
                self._session_by_runtime_run.pop(next(iter(self._session_by_runtime_run)))

    def event_session_id(self, event: Event) -> str | None:
        session_id = getattr(event, "session_id", None)
        if isinstance(session_id, str):
            return session_id
        run_id = getattr(event, "run_id", None)
        return self._session_by_runtime_run.get(run_id) if isinstance(run_id, str) else None

    def pending_approvals(self) -> tuple:
        with self._lock:
            brokers = tuple(dict.fromkeys(self._approvals_by_session.values()))
            if self.approvals not in brokers:
                brokers += (self.approvals,)
        return tuple(item for broker in brokers for item in broker.pending())

    def session_activity(self) -> dict[str, str]:
        with self._lock:
            activity = {session_id: "idle" for session_id in self._open_conversations}
            for session_id in self._active_by_session:
                activity[session_id] = "working"
            for session_id in self._goal_checks_by_session:
                activity[session_id] = "working"
            for broker in self._approvals_by_session.values():
                for approval in broker.pending():
                    if approval.session_id is not None:
                        activity[approval.session_id] = "waiting"
            return activity

    def _keep_awake_busy(self) -> bool:
        # Loop cards use conversation sessions; persisted card states can be stale.
        working = any(state == "working" for state in self.session_activity().values())
        with self._lock:
            if self._closing:
                return False
            validating = bool(self._loops._loop_validation_cancel)
            conversations = tuple(self._open_conversations.values())
        return working or validating or self._person_jobs.has_running() or any(
            leader.background_jobs is not None and leader.background_jobs.has_running()
            for leader, _session in conversations
        )

    def has_work_in_flight(self) -> bool:
        with self._lock:
            conversational_work = bool(
                self._active_by_session
                or self._goal_checks_by_session
                or self._loops._loop_validation_cancel
                or self._user_command is not None
                or any(broker.pending() for broker in self._approvals_by_session.values())
                or self.approvals.pending()
            )
            conversations = tuple(self._open_conversations.values())
        return conversational_work or self._person_jobs.has_running() or any(
            leader.background_jobs is not None and leader.background_jobs.has_running()
            for leader, _session in conversations
        )

    def list_jobs(self) -> list[dict]:
        with self._lock:
            conversations = tuple(self._open_conversations.values())
        managers = [(self._person_jobs, None)] + [
            (leader.background_jobs, session.run_id)
            for leader, session in conversations
            if leader.background_jobs is not None
        ]
        result = []
        for manager, session_id in managers:
            for job in manager.list():
                result.append({
                    "id": job.id,
                    "owner": job.owner,
                    "session_id": session_id if job.owner == "agent" else None,
                    "label": job.label,
                    "status": job.status,
                    "exit_code": job.exit_code,
                    "started_at": datetime.fromtimestamp(job.started_at, timezone.utc).isoformat(),
                    "ended_at": None if job.ended_at is None else datetime.fromtimestamp(job.ended_at, timezone.utc).isoformat(),
                    "output_tail": job.output[-4000:],
                    "output_dropped": job.total_chars > 4000,
                })
        return sorted(result, key=lambda job: job["started_at"], reverse=True)

    def start_job(self, command: str) -> str:
        return self._person_jobs.start(
            [shell_executable(), "-c", command], str(self._policy.repo_root),
            shell_environment(self._policy.repo_root), label=command,
        )

    def stop_job(self, job_id: str) -> dict | None:
        result = self._person_jobs.stop(job_id)
        if result is not None:
            return result
        with self._lock:
            conversations = tuple(self._open_conversations.values())
        for leader, _session in conversations:
            manager = leader.background_jobs
            result = None if manager is None else manager.stop(job_id)
            if result is not None:
                return result
        return None

    def _job_changed(self, job: BackgroundJob, session_id: str | None) -> None:
        self._broker.publish(JobChanged(
            agent_id="",
            run_id="" if session_id is None else session_id,
            job_id=job.id,
            status=job.status,
            session_id=session_id,
        ))

    def resolve_approval(self, approval_id: str, **decision) -> bool:
        with self._lock:
            brokers = tuple(dict.fromkeys(self._approvals_by_session.values())) + (self.approvals,)
            if decision.get("remember") == "project":
                match = next((
                    (broker, pending)
                    for broker in brokers
                    if (pending := broker.pending_grant(approval_id)) is not None
                ), None)
                if match is None:
                    return False
                broker, (approval, shell_prefixes, fetch_hosts) = match
                if not decision.get("allowed", False):
                    return broker.resolve(
                        approval_id, allowed=False, reason=decision.get("reason", ""),
                        remember=False, answer=decision.get("answer", ""),
                    )
                if not approval.remember:
                    raise InvalidApprovalAnswer("this approval cannot be remembered")
                if approval.operation == "run_shell":
                    try:
                        self._persist_project_grant("shell", shell_prefixes)
                    except PermissionGrantError as exc:
                        _LOG.error("Could not save project permission grant: %s", exc)
                        return broker.resolve(
                            approval_id, allowed=True, reason=decision.get("reason", ""),
                            remember=False, answer=decision.get("answer", ""),
                        )
                    updated_shell, updated_fetch = restrict_permission_grants(
                        tuple(dict.fromkeys((*self._project_shell_grants, *shell_prefixes))),
                        self._project_fetch_grants,
                        None if self._extensions is None else self._extensions.ceiling,
                    )
                    new_shell = tuple(item for item in updated_shell if item not in self._project_shell_grants)
                    self._project_shell_grants, self._project_fetch_grants = updated_shell, updated_fetch
                elif approval.operation == "web_fetch":
                    try:
                        self._persist_project_grant("fetch", fetch_hosts)
                    except PermissionGrantError as exc:
                        _LOG.error("Could not save project permission grant: %s", exc)
                        return broker.resolve(
                            approval_id, allowed=True, reason=decision.get("reason", ""),
                            remember=False, answer=decision.get("answer", ""),
                        )
                    updated_shell, updated_fetch = restrict_permission_grants(
                        self._project_shell_grants,
                        tuple(dict.fromkeys((*self._project_fetch_grants, *fetch_hosts))),
                        None if self._extensions is None else self._extensions.ceiling,
                    )
                    new_fetch = tuple(item for item in updated_fetch if item not in self._project_fetch_grants)
                    self._project_shell_grants, self._project_fetch_grants = updated_shell, updated_fetch
                    policies = (self._base_policy, self._policy, *self._policy_by_session.values())
                    seen_policies: set[int] = set()
                    for policy in policies:
                        if id(policy) in seen_policies:
                            continue
                        seen_policies.add(id(policy))
                        policy.fetch_allowlist = list(dict.fromkeys([
                            *policy.fetch_allowlist, *new_fetch,
                        ]))
                else:
                    raise InvalidApprovalAnswer("this approval cannot be remembered")
                for active_broker in dict.fromkeys((*brokers, broker)):
                    active_broker.add_project_grant(
                        approval.operation,
                        shell_prefixes=new_shell if approval.operation == "run_shell" else (),
                        fetch_hosts=new_fetch if approval.operation == "web_fetch" else (),
                    )
                return broker.resolve(
                    approval_id, allowed=True, reason=decision.get("reason", ""),
                    remember=False, answer=decision.get("answer", ""),
                )
        remember = decision.get("remember", False) is True
        return any(
            broker.resolve(approval_id, **{**decision, "remember": remember})
            for broker in brokers
        )

    def _persist_project_grant(self, kind: str, grant: object) -> None:
        if not grant:
            raise InvalidApprovalAnswer("this approval has no rememberable grant")
        save_project_permission_grant(self._project_grants_path, kind, grant)
        self._stored_project_shell_grants, self._stored_project_fetch_grants = (
            load_project_permission_grants(self._project_grants_path)
        )

    def project_permission_grants(self) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
        return load_project_permission_grants(self._project_grants_path)

    def remove_project_permission_grant(self, kind: str, value: object) -> bool:
        removed = remove_project_permission_grant(self._project_grants_path, kind, value)
        if not removed:
            return False
        previous_fetch_grants = self._project_fetch_grants
        self._stored_project_shell_grants, self._stored_project_fetch_grants = (
            load_project_permission_grants(self._project_grants_path)
        )
        self._project_shell_grants, self._project_fetch_grants = restrict_permission_grants(
            (*self._configured_shell_grants, *self._stored_project_shell_grants),
            (*self._configured_fetch_grants, *self._stored_project_fetch_grants),
            None if self._extensions is None else self._extensions.ceiling,
        )
        brokers = tuple(dict.fromkeys((*self._approvals_by_session.values(), self.approvals)))
        for broker in brokers:
            broker.set_project_grants(self._project_shell_grants, self._project_fetch_grants)
        policies = []
        for policy in (self._base_policy, self._policy, *self._policy_by_session.values()):
            if all(policy is not existing for existing in policies):
                policies.append(policy)
        for policy in policies:
            policy.fetch_allowlist = list(dict.fromkeys([
                *(host for host in policy.fetch_allowlist if host not in previous_fetch_grants),
                *self._project_fetch_grants,
            ]))
        return True

    def start(
        self,
        prompt: str,
        *,
        attachments: tuple[ImageBlock | DocumentBlock, ...] = (),
        _goal_round: bool = False,
        _new_goal: Goal | None = None,
        _session_meta: dict | None = None,
        _title: str | None = None,
        session_policy: PermissionPolicy | None = None,
        session_id: str | None = None,
        tool_restriction: frozenset[str] | None = None,
    ) -> str:
        with self._lock:
            current_id = None if self._conversation is None else self._conversation[1].run_id
            detached = session_policy is not None or (session_id is not None and session_id != current_id)
            if session_id is not None:
                conversation = self._open_conversations.get(session_id)
                if conversation is None:
                    raise SessionError(f"session {session_id!r} is not open")
                policy = self._policy_by_session[session_id]
                approvals = self._approvals_by_session[session_id]
                goal = self._goals_by_session.get(session_id)
                goal_session_id = session_id if goal is not None else None
                active = self._active_by_session.get(session_id)
                if active is not None:
                    raise RunActiveError(active.run_id)
            else:
                conversation = None if detached else self._conversation
                policy = session_policy if detached else self._policy
                approvals = self.approvals
                goal = None if detached else self._goal
                goal_session_id = None if detached else self._goal_session_id
            command_session = None if conversation is None else conversation[1].run_id
            if (self._user_command is not None
                    and (conversation is not None or not detached)
                    and self._user_command.for_session(command_session) is not None):
                raise CommandActiveError("wait for the command to finish")
            if not detached and self._active is not None:
                raise RunActiveError(self._active.run_id)
            if len(self._active_by_session) >= MAX_ACTIVE_CONVERSATIONS:
                raise RunActiveError(description=f"{MAX_ACTIVE_CONVERSATIONS} conversations are already running")
            run_id = new_id("run")
            cancel = CancellationToken()
            creating_conversation = conversation is None
            session = None
            leader = None
            worktree_admin = None
            previous_conversation = self._conversation
            previous_policy, previous_approvals = self._policy, self.approvals
            previous_context, previous_usage = self._context_report, dict(self._usage_by_agent)
            previous_goal, previous_goal_session = self._goal, self._goal_session_id
            pending_context = list(self._pending_user_context)
            try:
                if conversation is None:
                    if not detached and self._next_policy is not None:
                        policy = self._next_policy
                    if self._provider is None and self._provider_factory is not None:
                        self._provider = self._provider_factory(None, None, None)
                        if self._provider is not None:
                            self._provider_choice = {"name": self._provider.name}
                    if self._provider is None:
                        raise ProviderSelectionError("no configured provider; add an API key in Settings")
                    policy = replace(policy)
                    spec_info = None if not _session_meta else _session_meta.get("spec_run")
                    if isinstance(spec_info, dict) and spec_info.get("kind") == "implement":
                        worktree_admin = self._sessions_root / run_id / "worktree"
                        start_ref = spec_info.get("start_ref", "HEAD")
                        if not isinstance(start_ref, str) or not start_ref:
                            raise ValueError("spec run start_ref must be a non-blank string")
                        worktree_root = create_worktree(self._repo_root, worktree_admin, start=start_ref)
                        policy = policy.rerooted(worktree_root)
                        policy.mode = "allow"
                    approvals = self._new_approval_broker(run_id)
                    policy.approval_callback = approvals.callback
                    session = SessionStore(
                        self._sessions_root,
                        run_id,
                        repo_root=self._repo_root if _session_meta and "spec_run" in _session_meta else policy.repo_root,
                        events=fan_out(
                            lambda event, sid=run_id: self._publish_session(event, sid),
                            self._hooks,
                        ),
                    )
                    if _session_meta:
                        def change(meta):
                            meta.update(_session_meta)
                        session.update_meta(change)
                    leader = self._new_leader(session, policy=policy, approvals=approvals, tool_restriction=tool_restriction)
                    seeded = []
                    if self._system_prompt:
                        seeded.append(Message(role=Role.SYSTEM, content=self._system_prompt))
                    instruction_policy = policy
                    if _session_meta and "spec_run" in _session_meta:
                        instruction_policy = replace(policy, repo_root=self._repo_root)
                    instructions = load_instructions(instruction_policy, working_dir=self._working_dir)
                    for warning in instructions.warnings:
                        print(f"instruction warning: {warning}", file=sys.stderr)
                    self._instruction_warnings_by_session[run_id] = list(instructions.warnings)
                    rendered = instructions.render()
                    if rendered:
                        seeded.append(Message(role=Role.SYSTEM, content=rendered))
                    seeded.append(Message(
                        role=Role.SYSTEM,
                        content=(
                            capture_environment(
                                working_dir=self._working_dir,
                                repo_root=policy.repo_root,
                                provider=leader._config.leader_provider.name,
                                model=leader._leader_spec.model.model,
                            )
                            + "\n\nIn the person's messages, @<path> names a file in this repository. "
                            "Read it with read_file before relying on its contents."
                        ),
                    ))
                    if self._mcp_instructions:
                        sections = "\n\n".join(
                            f"## {name}\n{instructions}"
                            for name, instructions in self._mcp_instructions
                        )
                        seeded.append(Message(
                            role=Role.SYSTEM,
                            content=f"MCP servers' instructions:\n\n{sections}",
                        ))
                    leader.seed_chat(seeded)
                    def change(meta):
                        attachment_title = (
                            attachments[0].filename
                            if attachments and isinstance(attachments[0], DocumentBlock)
                            else None
                        )
                        meta["title"] = _title or _conversation_title(prompt) or attachment_title or (
                            "Attachment" if attachments else ""
                        )
                        if _session_meta:
                            meta.update(_session_meta)
                        if self._provider_choice is not None:
                            meta["provider_choice"] = self._provider_choice
                    session.update_meta(change)
                    conversation = (leader, session)
                    self._open_conversations[session.run_id] = conversation
                    self._policy_by_session[session.run_id] = policy
                    self._approvals_by_session[session.run_id] = approvals
                    self._session_by_root_agent[leader.agent_ref.agent_id] = session.run_id
                    if not detached:
                        self._conversation = conversation
                        self._policy = policy
                        self.approvals = approvals
                        self._context_report = None
                        self._usage_by_agent.clear()
                    self._context_by_session[session.run_id] = None
                    self._usage_by_session[session.run_id] = {}
                else:
                    leader, _ = conversation
                session_id = conversation[1].run_id
                chat_message: ContentInput = prompt
                if attachments:
                    chat_message = ([TextBlock(prompt)] if prompt else []) + list(attachments)
                previous_rounds = None
                if _new_goal is not None:
                    _new_goal.rounds = 1
                    goal = _new_goal
                    goal_session_id = session_id
                    self._goals_by_session[session_id] = _new_goal
                    self._save_goal(session_id, _new_goal)
                    self._goal_event("set", _new_goal, agent_id=leader.agent_ref.agent_id)
                elif _goal_round and goal is not None and goal_session_id == session_id:
                    previous_rounds = goal.rounds
                    goal.rounds += 1
                    self._goals_by_session[session_id] = goal
                    self._save_goal(session_id, goal)
                elif not _goal_round:
                    context = self._goal_checks_by_session.get(session_id)
                    if context is not None:
                        context.interrupted = "interrupted"
                is_goal_round = _goal_round or _new_goal is not None
                if not detached and self._pending_user_context:
                    for context in self._pending_user_context:
                        leader.add_context(context)
                    self._pending_user_context.clear()
                thread = threading.Thread(
                    target=self._run,
                    args=(run_id, leader, chat_message, cancel, is_goal_round, session_id),
                    name=f"symphonai-host-{run_id}",
                    daemon=True,
                )
                active = _ActiveRun(
                    run_id, cancel, thread, leader.agent_ref.agent_id, goal_round=is_goal_round,
                )
                if not detached:
                    self._active = active
                    self._goal = goal
                    self._goal_session_id = goal_session_id
                self._active_by_session[session_id] = active
                try:
                    thread.start()
                except Exception:
                    if not detached:
                        self._active = None
                    self._active_by_session.pop(session_id, None)
                    if _new_goal is not None:
                        self._save_goal(session_id, None)
                        self._goals_by_session.pop(session_id, None)
                        goal = None
                        goal_session_id = None
                        self._goal_event("clear", None, agent_id=leader.agent_ref.agent_id, session_id=session_id)
                    elif previous_rounds is not None and goal is not None:
                        goal.rounds = previous_rounds
                        self._save_goal(session_id, goal)
                    if not detached:
                        self._goal = goal
                        self._goal_session_id = goal_session_id
                    raise
                return run_id
            except BaseException:
                cancel.close()
                if creating_conversation:
                    self._active_by_session.pop(run_id, None)
                    self._open_conversations.pop(run_id, None)
                    self._policy_by_session.pop(run_id, None)
                    self._approvals_by_session.pop(run_id, None)
                    self._context_by_session.pop(run_id, None)
                    self._usage_by_session.pop(run_id, None)
                    self._instruction_warnings_by_session.pop(run_id, None)
                    self._goals_by_session.pop(run_id, None)
                    if leader is not None:
                        self._session_by_root_agent.pop(leader.agent_ref.agent_id, None)
                    if not detached:
                        self._conversation = previous_conversation
                        self._policy, self.approvals = previous_policy, previous_approvals
                        self._context_report, self._usage_by_agent = previous_context, previous_usage
                        self._goal, self._goal_session_id = previous_goal, previous_goal_session
                        self._pending_user_context = pending_context
                        self._active = (None if previous_conversation is None else
                                        self._active_by_session.get(previous_conversation[1].run_id))
                    try:
                        if leader is not None:
                            leader.close()
                    finally:
                        try:
                            if session is not None:
                                session.close()
                        finally:
                            if worktree_admin is not None and worktree_admin.exists():
                                remove_worktree(self._repo_root, worktree_admin)
                            directory = self._sessions_root / run_id
                            if directory.exists():
                                shutil.rmtree(directory)
                raise

    def memory_store(self) -> AgentMemory | None:
        """Return the shared project and global memory store if it opens."""
        if not self._memory_open_attempted:
            self._memory_open_attempted = True
            try:
                self._memory = AgentMemory(
                    self._memory_root, self._policy.repo_root
                )
            except OSError:
                self._memory = None
        return self._memory

    def _memory_for(
        self,
        roster: Mapping[str, AgentSpec],
        *,
        leader_enabled: bool = True,
    ) -> AgentMemory | None:
        if not leader_enabled and not any(spec.memory.enabled for spec in roster.values()):
            return None
        return self.memory_store()

    def _close_idle_conversations_locked(self) -> None:
        current_id = None if self._conversation is None else self._conversation[1].run_id
        for session_id, conversation in tuple(self._open_conversations.items()):
            if (
                session_id == current_id
                or session_id in self._active_by_session
                or session_id in self._goal_checks_by_session
                or (
                    conversation[0].background_jobs is not None
                    and conversation[0].background_jobs.has_running()
                )
            ):
                continue
            conversation[1].close()
            conversation[0].close()
            self._open_conversations.pop(session_id, None)
            self._policy_by_session.pop(session_id, None)
            self._approvals_by_session.pop(session_id, None)
            self._goals_by_session.pop(session_id, None)
            self._context_by_session.pop(session_id, None)
            self._usage_by_session.pop(session_id, None)
            for agent_id, owner in tuple(self._session_by_root_agent.items()):
                if owner == session_id:
                    self._session_by_root_agent.pop(agent_id, None)
            for runtime_id, owner in tuple(self._session_by_runtime_run.items()):
                if owner == session_id:
                    self._session_by_runtime_run.pop(runtime_id, None)

    def _new_leader(
        self, session: SessionStore, *, jobs: JobManager | None = None,
        policy: PermissionPolicy | None = None, approvals: ApprovalBroker | None = None,
        tool_restriction: frozenset[str] | None = None,
    ) -> Leader:
        extensions = self._extensions
        policy = self._policy if policy is None else policy
        approvals = self.approvals if approvals is None else approvals
        tool_restriction = self._restricted_tool_names if tool_restriction is None else tool_restriction
        if jobs is None:
            jobs = JobManager(
                policy.shell_output_limit_chars,
                owner="agent",
                on_change=lambda job: self._job_changed(job, session.run_id),
            )
        checkpoints = CheckpointStore(
            session.directory / "checkpoints",
            policy.repo_root,
        )
        result_store = ToolResultStore(
            directory=session.tool_results_directory,
            fallback_directories=tool_result_search_path(session),
        )
        if (
            tool_restriction is not None
            and "read_tool_result" not in tool_restriction
        ):
            result_store = None
        skills = None if self._extensions is None else self._extensions.skills
        role_selectors = {}
        if self._extensions is not None:
            for name in ("planner", "implementer", "reviewer", "worker", "explorer"):
                provider_name = self._extensions.config.get(f"roles.{name}.provider")
                if provider_name is not None:
                    role_selectors[name] = ModelSelector(
                        provider=provider_name,
                        model=self._extensions.config.get(f"roles.{name}.model"),
                        effort=self._extensions.config.get(f"roles.{name}.effort"),
                    )
        roster = builtin_subagent_specs(
            self._provider,
            policy,
            self._search_backend,
            skills=skills,
            lsp=self._lsp,
            jobs=jobs,
            model_override=self._model,
            role_selectors=role_selectors,
        )
        if self._extensions is not None:
            roster.update(self._extensions.agents)
        for name, spec in tuple(roster.items()):
            if name == "leader":
                continue
            addition, _, warnings = load_role_additions(
                name, repo_root=policy.repo_root, policy=policy
            )
            for warning in warnings:
                logging.warning("%s", warning)
            if addition:
                roster[name] = replace(spec, prompt="\n\n".join(filter(None, (spec.prompt, addition))))
        if tool_restriction is not None:
            available_extra = set(self._mcp_tools or {})
            restricted_roster = {}
            for name, spec in roster.items():
                if name == "leader":
                    restricted_roster[name] = spec
                    continue
                tool_names = (
                    tuple(
                        tool for tool in standard_tool_registry(
                            result_store=(result_store if "read_tool_result" in tool_restriction else None),
                            search_backend=self._search_backend,
                            skills=skills,
                            lsp=self._lsp,
                            jobs=jobs,
                        )
                        if tool in tool_restriction
                    ) if spec.tool_names is None
                    else tuple(tool for tool in spec.tool_names if tool in tool_restriction)
                )
                mcp_names = (
                    tuple(tool for tool in available_extra if tool in tool_restriction)
                    if spec.mcp_tool_names is None
                    else tuple(tool for tool in spec.mcp_tool_names if tool in tool_restriction)
                )
                if tool_names == () and not mcp_names:
                    continue
                memory = spec.memory
                if not {"remember", "forget_memory"}.issubset(tool_restriction):
                    memory = replace(memory, enabled=False)
                restricted_roster[name] = replace(
                    spec, tool_names=tool_names, mcp_tool_names=mcp_names,
                    denied_mcp_tool_names=(), memory=memory,
                )
            roster = restricted_roster
        if self._subagent_budget is not None:
            roster = {
                name: spec if name == "leader" else spec.with_overrides(
                    budget=_narrow_budget(
                        spec.budget, self._subagent_budget,
                        turn_limit_configured=self._subagent_turn_limit_configured,
                    )
                )
                for name, spec in roster.items()
            }
        from symphonai_host.session_search import session_tools
        from symphonai_host.roadmap_tools import roadmap_tools

        meta = session.read_meta()
        history_tools = {} if "spec_run" in meta else session_tools(
            self._sessions_root, policy.repo_root, session.run_id,
        )
        role = meta.get("spec_run", {}).get("role") if isinstance(meta.get("spec_run"), dict) else None
        leader_effort = self._effort
        if isinstance(role, str) and role in roster:
            roster["leader"] = roster[role]
            leader_effort = roster[role].model.effort
        defined_leader = roster.get("leader")
        leader_base_prompt = self._base_prompt
        if not isinstance(role, str) or role not in roster:
            leader_addition, _, warnings = load_role_additions(
                "leader", repo_root=policy.repo_root, policy=policy
            )
            for warning in warnings:
                logging.warning("%s", warning)
            if defined_leader is not None and leader_addition:
                defined_leader = replace(
                    defined_leader,
                    prompt="\n\n".join(filter(None, (defined_leader.prompt, leader_addition))),
                )
                roster["leader"] = defined_leader
            elif leader_addition:
                leader_base_prompt = "\n\n".join((leader_base_prompt, leader_addition))
        if (
            defined_leader is not None
            and not skills
            and "use_skill" in (defined_leader.tool_names or ())
        ):
            raise ProviderSelectionError("leader cannot use_skill: no skills are available")
        if (
            defined_leader is not None
            and self._search_backend is None
            and "web_search" in (defined_leader.tool_names or ())
        ):
            raise ProviderSelectionError("leader cannot use web_search: search is not configured")
        if (
            defined_leader is not None
            and (self._lsp is None or not self._lsp.has_enabled_servers)
            and "lsp" in (defined_leader.tool_names or ())
        ):
            raise ProviderSelectionError("leader cannot use lsp: no language server is configured")
        leader_provider = self._provider
        leader_model = self._model
        def configure_fallback(provider: ModelProvider | None) -> ModelProvider | None:
            if provider is not None:
                provider.fallback_models = tuple(
                    () if extensions is None else
                    extensions.config.get(f"fallback.{provider.name}", ())
                )
            return provider

        configure_fallback(leader_provider)
        conversation_approvals = approvals
        if defined_leader is not None:
            selector = defined_leader.model
            if selector.provider != self._provider.name:
                if self._provider_factory is None:
                    raise ProviderSelectionError(f"leader provider {selector.provider!r} is unavailable")
                leader_provider = self._provider_factory(selector.provider, selector.model, None)
                if leader_provider is None:
                    raise ProviderSelectionError(f"leader provider {selector.provider!r} is unavailable")
                configure_fallback(leader_provider)
                leader_model = selector.model
            elif selector.model is not None:
                leader_model = selector.model
        child_providers: dict[tuple[str, str | None], ModelProvider | None] = {}
        approval_broker = approvals

        def provider_for(name: str, model: str | None) -> ModelProvider | None:
            key = (name, model)
            if key not in child_providers:
                child_providers[key] = (
                    None
                    if self._provider_factory is None
                    else self._provider_factory(name, model, None)
                )
                child_providers[key] = configure_fallback(child_providers[key])
            return child_providers[key]

        leader = Leader(
            LeaderConfig(
                leader_provider=leader_provider,
                subagent_provider=self._provider,
                repo_root=str(policy.repo_root),
                base_prompt=leader_base_prompt,
                subagent_base_prompt=self._subagent_base_prompt,
                max_leader_turns=self._max_turns,
                leader_budget=self._leader_budget,
                chat_token_budget=self._chat_token_budget,
                compaction_limit=(
                    None if self._extensions is None
                    else self._extensions.config.get("compaction.limit")
                ),
                chat_recent_turns=self._chat_recent_turns,
                permission_mode=policy.mode,
                approval_callback=approvals.callback,
                events=lambda event: (
                    self._broker.publish(event) if isinstance(event, SessionNotesUpdated)
                    else self._publish_active(event)
                ),
                extensions=self._extensions,
                stream=True,
                result_store=result_store,
                search_backend=self._search_backend,
                extra_tools=self._mcp_tools,
                leader_policy=policy,
                leader_model=leader_model,
                leader_effort=leader_effort,
                model_summary=True,
                session_notes=None if "spec_run" in meta else session.directory / "notes.md",
                subagent_specs=roster,
                subagent_budget=self._subagent_budget,
                hook_runner=self._hooks,
                memory=self._memory_for(
                    roster,
                    leader_enabled=(
                        "leader" not in roster or roster["leader"].memory.enabled
                    ),
                ),
                memory_provider=self._memory_provider,
                memory_model=self._memory_model,
                memory_extraction_enabled=self._memory_extraction_enabled,
                memory_usage_callback=lambda model, usage: self._record_background_usage(
                    session.run_id, model, usage
                ),
                checkpoints=checkpoints,
                provider_for=provider_for,
                leases=self._leases,
                lsp=self._lsp,
                jobs=jobs,
                leader_tools=(roadmap_tools(self) if self._roadmap_tools_enabled and "spec_run" not in meta else {}) | history_tools | goal_tools(
                    lambda: self._goal_for_session(session.run_id),
                    lambda status, message: self._update_goal_for_session(
                        session.run_id, status, message,
                    ),
                ) | plan_tools(
                    lambda plan, explanation: self._update_plan_for_session(
                        session.run_id, plan, explanation,
                    ),
                ) | question_tools(approval_broker.ask_user) | plan_proposal_tools(
                    lambda approval: conversation_approvals.request_interaction(approval),
                    self.permitted_modes,
                    session_id=session.run_id,
                ),
            ),
            session=session,
        )
        if tool_restriction is not None:
            selected = set(tool_restriction)
            leader._agent.restrict_tools(selected)
            leader._leader_spec = replace(
                leader._leader_spec,
                tool_names=tuple(name for name in leader._leader_spec.tool_names if name in selected),
            )
        return leader

    def available_tool_names(self) -> set[str]:
        names = set(standard_tool_registry())
        names.update({
            "web_search", "use_skill", "lsp", "read_tool_result", "job_output",
            "job_stop", "remember", "forget_memory", "dispatch_subagent",
            "get_goal", "update_goal", "update_plan", "ask_user", "propose_plan",
        })
        names.update(self._mcp_tools or {})
        if self._extensions is not None:
            for spec in self._extensions.agents.values():
                names.update(spec.tool_names or ())
                names.update(spec.mcp_tool_names or ())
        return names

    def restrict_tool_names(self, names: set[str]) -> None:
        self._restricted_tool_names = frozenset(names)

    def _publish_active(self, event: Event) -> None:
        with self._lock:
            session_id = self._session_by_root_agent.get(event.agent_id)
            if isinstance(event, SubagentSpawned) and session_id is not None:
                self._session_by_root_agent[event.subagent_agent_id] = session_id
            if session_id is None and len(self._active_by_session) == 1:
                session_id = next(iter(self._active_by_session))
            active = self._active_by_session.get(session_id) if session_id is not None else None
            run_id = None if active is None else active.run_id
            event_run_id = getattr(event, "run_id", None)
            if (
                active is not None
                and isinstance(event_run_id, str)
                and event_run_id.startswith("memory_run_")
            ):
                active.deferred_memory_events.append(event)
                return
            if (
                active is not None
                and event.agent_id == active.root_agent_id
                and isinstance(event, (RunFinished, RunFailed))
            ):
                active.terminal_event = event
                return
        if run_id is not None:
            self._publish(run_id, event, session_id=session_id)
        elif session_id is not None and isinstance(event_run_id, str):
            self._publish(event_run_id, event, session_id=session_id)

    @staticmethod
    def _waited_subagent_notices(message: Message, wait_ids: set[str], session_id: str):
        """Replay collected results without delivering them to the model again."""
        result = message.tool_result
        if result is None or result.tool_call_id not in wait_ids or not isinstance(result.payload, dict):
            return
        for name, finished in result.payload.get("finished", {}).items():
            yield HistoryMessage(
                role="user", notice=True,
                text=SUBAGENT_FINISHED_PREFIX.format(name=name) + "\n" + json.dumps(finished),
                tool_calls=[], turn_id=message.turn_id, session_id=session_id,
            )

    def open_session(self, run_id: str, *, provider_choice_override: dict | None = None) -> dict:
        """Load and replay a finished transcript without ever rewriting it."""
        from dataclasses import asdict

        @dataclass(frozen=True)
        class ReplayedHistoryMessage(ForkableHistoryMessage):
            record_id: str | None
            replay_id: str = field(kw_only=True)

            def payload(self) -> dict:
                payload = {**super().payload(), "replay_id": self.replay_id}
                if self.record_id is None:
                    payload.pop("record_id")
                return payload

        replay_id = new_id("replay")
        history_frames = 0

        def publish_history(message):
            nonlocal history_frames
            values = asdict(message)
            values.setdefault("record_id", None)
            self._broker.publish(ReplayedHistoryMessage(**values, replay_id=replay_id))
            history_frames += 1

        with self._lock:
            reader = SessionStore.open(self._sessions_root, run_id)
            try:
                loaded, diagnosis, repaired_ids = load_run_for_resume(reader)
                tool_results = self._replay_tool_results(loaded.messages)
                records, _ = read_records(reader.directory / "run.jsonl")
                subagent_records = []
                for path in sorted(reader.directory.glob("agent-*.jsonl")):
                    try:
                        agent_records, _ = read_records(path)
                    except (OSError, TranscriptError):
                        continue
                    subagent_records.append(agent_records)
                persisted_usage: dict[str, tuple[str, dict[str, UsageTotals]]] = {}
                for transcript in (records, *subagent_records):
                    for record in transcript:
                        entry = _usage_record(record)
                        if entry is None:
                            continue
                        agent_id, agent_name, totals = entry
                        model = record["data"]["model"]
                        name, by_model = persisted_usage.get(agent_id, (agent_name, {}))
                        updated = dict(by_model)
                        updated[model] = updated.get(model, UsageTotals()).merged(totals)
                        persisted_usage[agent_id] = (name, updated)
                auto_reviews = self._saved_auto_reviews(records)
                metadata = reader.read_meta()
                checkpoint_keys, replay_edits = self._checkpoint_replay_data(
                    records, subagent_records,
                )
                replay_turns = self._turn_replay_data(records)
                undone_keys = metadata.get("undone_checkpoint_keys", [])
                if not isinstance(undone_keys, list):
                    undone_keys = []
                choice = (metadata.get("provider_choice") if provider_choice_override is None
                          else dict(provider_choice_override))
                provider_state_reset = metadata.get("provider_state_reset") is True
                if provider_choice_override is not None:
                    saved_choice = metadata.get("provider_choice")
                    if not isinstance(saved_choice, dict):
                        saved_choice = {}
                    provider_state_reset = provider_state_reset or any(
                        saved_choice.get(key) != choice.get(key) for key in ("name", "base_url")
                    )
                goal = Goal.from_payload(metadata.get("goal"))
                saved_plan = metadata.get("plan")
                if (
                    not isinstance(saved_plan, dict)
                    or not isinstance(saved_plan.get("steps"), list)
                    or not isinstance(saved_plan.get("explanation", ""), str)
                ):
                    saved_plan = None
            finally:
                reader.close()
            existing = self._open_conversations.get(run_id)
            if existing is not None:
                self._conversation = existing
                self._provider = existing[0]._config.leader_provider
                self._model = existing[0]._leader_spec.model.model
                self._effort = existing[0]._leader_spec.model.effort
                self._policy = self._policy_by_session[run_id]
                self.approvals = self._approvals_by_session[run_id]
                self._active = self._active_by_session.get(run_id)
                self._goal = self._goals_by_session.get(run_id, goal)
                self._goal_session_id = run_id if self._goal is not None else None
                self._context_report = self._context_by_session.get(run_id)
                self._usage_by_agent = dict(self._usage_by_session.get(run_id, {}))
                self._plans_by_session.pop(run_id, None)
                if saved_plan is not None:
                    self._plans_by_session[run_id] = saved_plan
                if self._goal is not None:
                    self._goals_by_session[run_id] = self._goal
                wait_ids = {call.id for saved in loaded.messages for call in saved.tool_calls
                            if call.name == "wait_subagents"}
                for message, record_id in zip(loaded.messages, loaded.record_ids):
                    checkpoint_key = checkpoint_keys.get(record_id)
                    publish_history(ForkableHistoryMessage(
                        role=message.role.value,
                        text=message.text,
                        notice=message.role is Role.USER and SUBAGENT_NOTICE_PATTERN.match(message.text) is not None,
                        reasoning=message.reasoning,
                        tool_calls=self._replay_tool_calls(message, tool_results, auto_reviews),
                        turn_id=message.turn_id, record_id=record_id,
                        attachments=[
                            {
                                "kind": "image" if isinstance(block, ImageBlock) else "document",
                                "media_type": block.media_type,
                                "filename": block.filename if isinstance(block, DocumentBlock) else None,
                            }
                            for block in message.content
                            if isinstance(block, (ImageBlock, DocumentBlock))
                        ],
                        checkpoint_key=checkpoint_key,
                        edits=[] if checkpoint_key is None else replay_edits.get(checkpoint_key, []),
                        undone=checkpoint_key in undone_keys,
                        session_id=run_id,
                        **replay_turns.get(record_id, {}),
                    ))
                    for notice in self._waited_subagent_notices(message, wait_ids, run_id):
                        publish_history(notice)
                self._close_idle_conversations_locked()
                return {
                    "run_id": loaded.run_id, "state": diagnosis.state.value,
                    "replayed": len(loaded.record_ids), "repaired_ids": repaired_ids,
                    "dropped_bytes": loaded.dropped_bytes,
                    "replay_id": replay_id, "history_frames": history_frames,
                }
            previous_provider = (
                self._provider,
                self._model,
                self._effort,
                self._provider_choice,
            )
            previous_policy = self._policy
            previous_approvals = self.approvals
            previous_mode = self._policy.mode
            provider, model, effort, provider_choice = previous_provider
            if isinstance(choice, dict) and self._provider_factory is not None:
                provider = self._provider_factory(choice.get("name"), choice.get("model"), choice.get("base_url"))
                if provider is None:
                    raise ProviderSelectionError("session provider is unavailable")
                model = choice.get("model")
                effort = choice.get("effort")
                provider_choice = choice
            elif provider is None and self._provider_factory is not None:
                provider = self._provider_factory(None, None, None)
                if provider is None:
                    raise ProviderSelectionError("no configured provider; add an API key in Settings")
            store = SessionStore.open(
                self._sessions_root,
                run_id,
                events=fan_out(
                    lambda event, sid=run_id: self._publish_session(event, sid),
                    self._hooks,
                ),
            )
            try:
                self._provider, self._model, self._effort, self._provider_choice = (
                    provider,
                    model,
                    effort,
                    provider_choice,
                )
                self._policy = replace(self._next_policy or self._policy)
                self._policy.mode = self._starting_mode
                self.approvals = self._new_approval_broker(run_id)
                self._policy.approval_callback = self.approvals.callback
                if provider_choice_override is not None:
                    def save_choice(meta):
                        meta["provider_choice"] = dict(provider_choice_override)
                        if provider_state_reset:
                            meta["provider_state_reset"] = True
                    store.update_meta(save_choice)
                leader = self._new_leader(store)
                messages = (
                    _without_vendor_state(loaded.messages)
                    if provider_state_reset
                    else loaded.messages
                )
                leader.seed_chat(messages, persisted=True)
                reopened_goal = goal is not None and goal.phase == "active"
                if reopened_goal:
                    goal.phase = "paused"
                    goal.reason = "reopened"
                    store.update_meta(lambda meta: meta.update({"goal": goal.payload()}))
            except Exception:
                self._provider, self._model, self._effort, self._provider_choice = previous_provider
                self._policy = previous_policy
                self.approvals = previous_approvals
                self._policy.mode = previous_mode
                store.close()
                raise
            self._conversation = (leader, store)
            self._open_conversations[run_id] = self._conversation
            self._policy_by_session[run_id] = self._policy
            self._approvals_by_session[run_id] = self.approvals
            self._session_by_root_agent[leader.agent_ref.agent_id] = run_id
            self._active = self._active_by_session.get(run_id)
            self._goal = goal
            self._goal_session_id = run_id if goal is not None else None
            self._context_report = None
            self._usage_by_agent = dict(persisted_usage)
            self._context_by_session[run_id] = None
            self._usage_by_session[run_id] = dict(persisted_usage)
            self._plans_by_session.pop(run_id, None)
            if saved_plan is not None:
                self._plans_by_session[run_id] = saved_plan
            if goal is not None:
                self._goals_by_session[run_id] = goal
            goal_agent_id = leader.agent_ref.agent_id
            self._close_idle_conversations_locked()
        if reopened_goal:
            self._goal_event("pause", goal, agent_id=goal_agent_id)
        wait_ids = {call.id for saved in loaded.messages for call in saved.tool_calls
                    if call.name == "wait_subagents"}
        for message, record_id in zip(loaded.messages, loaded.record_ids):
            checkpoint_key = checkpoint_keys.get(record_id)
            publish_history(ForkableHistoryMessage(
                role=message.role.value,
                text=message.text,
                notice=message.role is Role.USER and SUBAGENT_NOTICE_PATTERN.match(message.text) is not None,
                reasoning=message.reasoning,
                tool_calls=self._replay_tool_calls(message, tool_results, auto_reviews),
                turn_id=message.turn_id,
                record_id=record_id,
                attachments=[
                    {
                        "kind": "image" if isinstance(block, ImageBlock) else "document",
                        "media_type": block.media_type,
                        "filename": block.filename if isinstance(block, DocumentBlock) else None,
                    }
                    for block in message.content
                    if isinstance(block, (ImageBlock, DocumentBlock))
                ],
                checkpoint_key=checkpoint_key,
                edits=[] if checkpoint_key is None else replay_edits.get(checkpoint_key, []),
                undone=checkpoint_key in undone_keys,
                session_id=run_id,
                **replay_turns.get(record_id, {}),
            ))
            for notice in self._waited_subagent_notices(message, wait_ids, run_id):
                publish_history(notice)
        return {
            "run_id": loaded.run_id,
            "state": diagnosis.state.value,
            "replayed": len(loaded.record_ids),
            "repaired_ids": repaired_ids,
            "dropped_bytes": loaded.dropped_bytes,
            "replay_id": replay_id, "history_frames": history_frames,
        }

    def session_notes(self, session_id: str) -> dict:
        """Read a session's notes without changing the session or publishing events."""
        if re.fullmatch(r"run_[0-9a-f]{32}", session_id) is None:
            raise SessionError("session not found")
        reader = SessionStore.open(self._sessions_root, session_id)
        try:
            path = reader.directory / "notes.md"
            if path.is_symlink():
                raise SessionError("session notes not found")
            try:
                with path.open(encoding="utf-8") as source:
                    text = source.read()
                    updated_at = datetime.fromtimestamp(os.fstat(source.fileno()).st_mtime, timezone.utc).isoformat()
            except FileNotFoundError:
                return {"text": "", "updated_at": None}
            except (OSError, UnicodeError) as exc:
                raise SessionError("session notes unavailable") from exc
            return {"text": text, "updated_at": updated_at}
        finally:
            reader.close()

    def agent_history(self, session_id: str, agent_id: str) -> dict:
        """Return one subagent's display-safe transcript without publishing it."""
        if re.fullmatch(r"run_[0-9a-f]{32}", session_id) is None:
            raise SessionError("session not found")
        if re.fullmatch(r"agent_[0-9a-f]{32}", agent_id) is None:
            raise SessionError("agent not found")
        reader = SessionStore.open(self._sessions_root, session_id)
        try:
            loaded = load_run(reader, agent_id=agent_id)
            if loaded.agent_id != agent_id:
                raise SessionError("agent not found")
            records, _ = read_records(reader.directory / f"agent-{agent_id}.jsonl")
            tool_results = self._replay_tool_results(loaded.messages)
            auto_reviews = self._saved_auto_reviews(records)
            replay_turns = self._turn_replay_data(records)
            messages = []
            wait_ids = {call.id for saved in loaded.messages for call in saved.tool_calls
                        if call.name == "wait_subagents"}
            for message, record_id in zip(loaded.messages, loaded.record_ids):
                turn = replay_turns.get(record_id, {})
                messages.append(HistoryMessage(
                    role=message.role.value,
                    text=message.text,
                    notice=message.role is Role.USER and SUBAGENT_NOTICE_PATTERN.match(message.text) is not None,
                    reasoning=message.reasoning,
                    tool_calls=self._replay_tool_calls(message, tool_results, auto_reviews),
                    turn_id=message.turn_id,
                    attachments=[
                        {
                            "kind": "image" if isinstance(block, ImageBlock) else "document",
                            "media_type": block.media_type,
                            "filename": block.filename if isinstance(block, DocumentBlock) else None,
                        }
                        for block in message.content
                        if isinstance(block, (ImageBlock, DocumentBlock))
                    ],
                    session_id=session_id,
                    turn_started_at=turn.get("turn_started_at"),
                    turn_ended_at=turn.get("turn_ended_at"),
                    turn_outcome=turn.get("turn_outcome"),
                ).payload())
                messages.extend(notice.payload() for notice in
                                self._waited_subagent_notices(message, wait_ids, session_id))
            return {
                "session_id": session_id,
                "agent_id": agent_id,
                "messages": messages,
            }
        finally:
            reader.close()

    def fork_session(self, run_id: str, record_id: str, *, force: bool = False) -> dict:
        """Restore the file state at a message prefix, then reopen its fork."""
        with self._lock:
            if run_id in self._active_by_session:
                raise RunActiveError(self._active_by_session[run_id].run_id)
            source = SessionStore.open(self._sessions_root, run_id)
            destination = None
            checkpoints = None
            original_files: dict[str, bytes | None] = {}
            restored = False
            try:
                loaded = load_run(source)
                if record_id not in loaded.record_ids:
                    raise SessionError(f"run {run_id!r} has no current message record {record_id!r}")
                source_meta = source.read_meta()
                checkpoint_directory = source.directory / "checkpoints"
                if checkpoint_directory.is_dir():
                    checkpoints = CheckpointStore(checkpoint_directory, self._policy.repo_root)
                kept_keys: list[str] = []
                undone_entries: list[CheckpointEntry] = []
                branch_files: dict[str, bytes | None] = {}
                if checkpoints is not None:
                    records, _ = read_records(source.directory / "run.jsonl")
                    cutoff = next(
                        index for index, record in enumerate(records)
                        if record.get("record_id") == record_id
                    )
                    prompt_records: dict[str, int] = {}
                    pending_key = None
                    for index, record in enumerate(records):
                        if record.get("type") == "checkpoint":
                            key = record.get("data", {}).get("key")
                            if isinstance(key, str):
                                pending_key = key
                        elif pending_key is not None and record.get("type") == "message":
                            message = message_from_json(record["data"])
                            if message.role is Role.USER:
                                prompt_records.setdefault(pending_key, index)
                                pending_key = None
                    kept_keys = [
                        key for key in checkpoints.keys()
                        if prompt_records.get(key, len(records)) < cutoff
                    ]
                    kept_set = set(kept_keys)
                    undone_entries = [
                        entry for entry in checkpoints.entries() if entry.key not in kept_set
                    ]
                    first_undone: dict[str, CheckpointEntry] = {}
                    for entry in undone_entries:
                        first_undone.setdefault(entry.path, entry)
                    outside = []
                    for path in first_undone:
                        current = _checkpoint_file_bytes(checkpoints, path)
                        digest = None if current is None else hashlib.sha256(current).hexdigest()
                        if digest != checkpoints.last_written(path):
                            outside.append(path)
                    if outside and not force:
                        raise ChangedOutsideError(outside)
                    original_files = {
                        path: _checkpoint_file_bytes(checkpoints, path)
                        for path in first_undone
                    }
                    restored = bool(first_undone)
                    for path, entry in first_undone.items():
                        content = (
                            None if entry.backup is None
                            else (checkpoints.directory / entry.backup).read_bytes()
                        )
                        checkpoints.restore(path, content)
                    branch_files = {
                        path: _checkpoint_file_bytes(checkpoints, path)
                        for path in {
                            entry.path for entry in checkpoints.entries()
                            if entry.key in kept_set
                        } | set(first_undone)
                    }
                new_run_id = new_id("run")
                destination = SessionStore(
                    self._sessions_root, new_run_id, repo_root=self._policy.repo_root,
                )
                try:
                    fork_run(source, through_record_id=record_id, new_store=destination)
                    def change(meta):
                        meta["title"] = source_meta.get("title")
                        if "provider_choice" in source_meta:
                            meta["provider_choice"] = source_meta["provider_choice"]
                    destination.update_meta(change)
                    if checkpoints is not None:
                        destination_checkpoints = CheckpointStore(
                            destination.directory / "checkpoints", self._policy.repo_root,
                        )
                        checkpoints.copy_kept_to(
                            destination_checkpoints, tuple(kept_keys), branch_files,
                        )
                except Exception:
                    destination.close()
                    shutil.rmtree(destination.directory)
                    raise
                else:
                    destination.close()
            except Exception:
                if restored and checkpoints is not None:
                    for path, content in original_files.items():
                        checkpoints.restore(path, content)
                if destination is not None:
                    shutil.rmtree(destination.directory, ignore_errors=True)
                raise
            finally:
                source.close()
            try:
                return self.open_session(new_run_id)
            except Exception:
                if restored and checkpoints is not None:
                    for path, content in original_files.items():
                        checkpoints.restore(path, content)
                shutil.rmtree(destination.directory, ignore_errors=True)
                raise

    def end_conversation(self) -> None:
        with self._lock:
            conversation = self._conversation
            session_id = None if conversation is None else conversation[1].run_id
            context = (
                None if session_id is None else self._goal_checks_by_session.get(session_id)
            )
            if context is not None:
                context.interrupted = "interrupted"
                context.cancel.set()
                if context.process is not None:
                    self._kill_user_command(context.process)
            self._goal_check = None
            self._goal_check_thread = None
            if session_id is not None and session_id in self._active_by_session:
                self._conversation = None

                self._active = None
                self._goal = None
                self._goal_session_id = None
                self._goal_check = None
                self._goal_check_thread = None
                self._policy = replace(self._next_policy or self._policy)
                self._policy.mode = self._starting_mode
                self._policy.approval_callback = self.approvals.callback
                self._context_report = None
                self._usage_by_agent.clear()
                return
            self._goal = None
            self._goal_session_id = None
            self._conversation = None
            self._active = None
            self._policy = replace(self._next_policy or self._policy)
            self._policy.mode = self._starting_mode
            self._context_report = None
            self._usage_by_agent.clear()
            self.approvals.clear_grants()
        if conversation is not None:
            conversation[0].close()
            conversation[1].close()
            self._open_conversations.pop(conversation[1].run_id, None)
            self._policy_by_session.pop(conversation[1].run_id, None)
            self._approvals_by_session.pop(conversation[1].run_id, None)

    def delete_session(self, session_id: str) -> bool:
        """Close an idle session before removing its stored conversation."""
        with self._lock:
            if session_id in (".", "..") or Path(session_id).name != session_id:
                return False
            directory = self._sessions_root / session_id
            if not directory.is_dir():
                return False
            broker = self._approvals_by_session.get(session_id)
            if (
                session_id in self._active_by_session
                or session_id in self._goal_checks_by_session
                or (broker is not None and any(item.session_id == session_id for item in broker.pending()))
            ):
                raise RunActiveError(session_id)
            if self._conversation is not None and self._conversation[1].run_id == session_id:
                self.end_conversation()
            else:
                conversation = self._open_conversations.pop(session_id, None)
                if conversation is not None:
                    conversation[1].close()
                    conversation[0].close()
            self._policy_by_session.pop(session_id, None)
            self._approvals_by_session.pop(session_id, None)
            self._goals_by_session.pop(session_id, None)
            self._context_by_session.pop(session_id, None)
            self._usage_by_session.pop(session_id, None)
            for agent_id, owner in tuple(self._session_by_root_agent.items()):
                if owner == session_id:
                    self._session_by_root_agent.pop(agent_id, None)
            for runtime_id, owner in tuple(self._session_by_runtime_run.items()):
                if owner == session_id:
                    self._session_by_runtime_run.pop(runtime_id, None)
            shutil.rmtree(directory)
            return True

    def export_session(self, session_id: str) -> dict:
        """Return a display-safe Markdown export of one session transcript."""
        if session_id in (".", "..") or Path(session_id).name != session_id:
            raise KeyError(session_id)
        if not (self._sessions_root / session_id).is_dir():
            raise KeyError(session_id)
        store = SessionStore.open(self._sessions_root, session_id)
        try:
            loaded = load_run(store)
        except (OSError, TranscriptError, SessionError, ValueError, KeyError, TypeError):
            store.close()
            raise
        try:
            title = loaded.meta.get("title")
            title = title.strip() if isinstance(title, str) else ""
            title = title or session_id
            created_at = loaded.meta.get("created_at")
            date = created_at[:10] if isinstance(created_at, str) else ""
            results = self._replay_tool_results(loaded.messages)
            lines = [f"# {title}", "", date]
            for message in loaded.messages:
                if message.role is Role.USER:
                    lines.extend(("", "## You", "", message.text))
                    for attachment in message.content:
                        if isinstance(attachment, DocumentBlock):
                            lines.append(f"\n_Attachment: {attachment.filename or 'document'}_")
                        elif isinstance(attachment, ImageBlock):
                            lines.append("\n_Attachment: image")
                elif message.role is Role.ASSISTANT:
                    if message.text:
                        lines.extend(("", "## Assistant", "", message.text))
                    for call in message.tool_calls:
                        result = results.get(call.id)
                        outcome = "ok" if result is not None and result.ok else "failed"
                        target = call_target(call.name, call.arguments)
                        suffix = f" {target}" if target else ""
                        lines.append(f"- ran `{call.name}`{suffix} — {outcome}")
            return {"filename": f"{title}.md", "markdown": "\n".join(lines).rstrip() + "\n"}
        finally:
            store.close()

    def archive_session(self, session_id: str, archived: bool) -> bool:
        """Set a session's archived metadata flag without closing it."""
        if session_id in (".", "..") or Path(session_id).name != session_id:
            return False
        directory = self._sessions_root / session_id
        if not directory.is_dir():
            return False
        with self._lock:
            opened = self._open_conversations.get(session_id)
            store = opened[1] if opened is not None else SessionStore.open(self._sessions_root, session_id)
            try:
                def change(meta):
                    meta["archived"] = archived
                store.update_meta(change)
            finally:
                if opened is None:
                    store.close()
        return True

    def start_preview(self, url: str = "") -> dict:
        from symphonai_host.preview import Preview, target_url

        with self._preview_lock:
            url = url or self._preview_last_url
            target_url(url)
            self.stop_preview()
            with self._lock:
                if self._closing:
                    raise ValueError("Project is closing")
                self._preview = Preview(url, self._policy, self._add_preview_mark)
                self._preview_last_url = url
                return {"preview_url": self._preview.url}

    def stop_preview(self) -> dict:
        with self._preview_lock:
            with self._lock:
                preview, self._preview = self._preview, None
            if preview is not None:
                preview.close()
        return {"stopped": True}

    def _add_preview_mark(self, mark: dict) -> None:
        with self._lock:
            if self._closing:
                raise ValueError("Project is closing")
            session_id = None if self._conversation is None else self._conversation[1].run_id
            self._preview_marks.setdefault(session_id, []).append(mark)
            event = PreviewMarkAdded(
                agent_id="", run_id="", mark=mark, session_id=session_id, repo_root=str(self._repo_root),
            )
            if session_id is None:
                self._broker.publish(event)
            else:
                self._publish_session(event, session_id)

    def add_preview_mark(self, token: str, value: dict) -> dict:
        with self._lock:
            preview = self._preview
        if preview is None:
            raise PermissionError("Preview is not running")
        return preview.mark(token, value)

    def preview_marks(self) -> dict:
        with self._lock:
            session_id = None if self._conversation is None else self._conversation[1].run_id
            return {"session_id": session_id, "marks": list(self._preview_marks.get(session_id, []))}

    def remove_preview_marks(self, ids: list[str], session_id: str | None) -> dict:
        with self._lock:
            self._preview_marks[session_id] = [mark for mark in self._preview_marks.get(session_id, [])
                                               if mark["id"] not in ids]
        return {"removed": True}

    def close(self) -> None:
        from symphonai_host.terminal import close_host
        self._closing = True
        self.stop_preview()
        self._keep_awake.close()
        close_host(self)
        with self._lock:
            active_runs = tuple(self._active_by_session.values())
            approval_brokers = tuple(dict.fromkeys(self._approvals_by_session.values()))
            validations = tuple(self._loops._loop_validation_cancel.values())
            validation_processes = tuple(self._loops._loop_validation_processes.values())
            goal_checks = tuple(self._goal_checks_by_session.values())
            self._goal_check = None
            self._goal_check_thread = None
            for context in goal_checks:
                context.interrupted = "cancelled"
                context.cancel.set()
        for context in goal_checks:
            if context.process is not None:
                self._kill_user_command(context.process)
        for active_run in active_runs:
            active_run.cancel.cancel()
        for cancelled in validations:
            cancelled.set()
        for process in validation_processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
        for broker in approval_brokers:
            broker.cancel_all(reason="host closing")
        self.stop()
        with self._lock:
            active = tuple(self._active_by_session.values())
        for active_run in active:
            if active_run.thread is not threading.current_thread():
                active_run.thread.join(timeout=2)
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            with self._lock:
                remaining = tuple(self._goal_checks_by_session.values())
                for context in remaining:
                    context.interrupted = "cancelled"
                    context.cancel.set()
            if not remaining:
                break
            for context in remaining:
                if context.process is not None:
                    self._kill_user_command(context.process)
            time.sleep(0.01)
        with self._lock:
            conversations = tuple(self._open_conversations.values())
            self._open_conversations.clear()
        if self._lsp is not None:
            self._lsp.close()
        for leader, session in conversations:
            leader.close()
            session.close()
        self._person_jobs.close()
        with self._lock:
            runner_lock_fds = tuple(self._loops._loop_runner_lock_fds.values())
            self._loops._loop_runner_lock_fds.clear()
        for descriptor in runner_lock_fds:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def stop(self) -> None:
        with self._lock:
            active = self._active
            session_id = None if self._conversation is None else self._conversation[1].run_id
            command = None if self._user_command is None else self._user_command.for_session(session_id)
            if command is not None:
                command.stop_requested = True
                process = command.process
            else:
                process = None
            goal_check = None if session_id is None else self._goal_checks_by_session.get(session_id)
        if active is not None:
            active.cancel.cancel()
        if process is not None:
            self._kill_user_command(process)
        if goal_check is not None:
            goal_check.interrupted = "cancelled"
            goal_check.cancel.set()
            if goal_check.process is not None:
                self._kill_user_command(goal_check.process)
        self.approvals.cancel_all(reason="stopped")


    def steer(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("steer text must be non-blank")
        with self._lock:
            active = self._active
            conversation = self._conversation
            if active is None or conversation is None:
                raise RunActiveError("no turn is running")
            leader = conversation[0]
        try:
            leader.redirect(text)
        except AgentControlError as exc:
            raise RunActiveError("no turn is running") from exc

    def run_user_command(self, command: str) -> dict:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be non-blank")
        with self._lock:
            if self._active is not None:
                raise RunActiveError("wait for the turn to finish")
            conversation = self._conversation
            session_id = None if conversation is None else conversation[1].run_id
            if self._user_command is not None and self._user_command.for_session(session_id) is not None:
                raise CommandActiveError("a command is already running")
            repo_root = self._repo_root if conversation is None else self._policy.repo_root
            limit = self._policy.shell_output_limit_chars
            running = _UserCommand(session_id=session_id, next=self._user_command)
            self._user_command = running
        try:
            from symphonai_api.process import run_process
            def started(process):
                with self._lock:
                    running.process = process
                    should_stop = running.stop_requested
                if should_stop:
                    self._kill_user_command(process)
            result = run_process(
                [shell_executable(), "-c", command], repo_root, env=shell_environment(repo_root),
                timeout=120, cancel=lambda: running.stop_requested, output_limit=None, on_start=started,
            )
            if result.error is not None:
                raise result.error
            output = result.stdout.decode("utf-8", errors="replace")
            timed_out, stopped, exit_code = result.timed_out, result.cancelled, result.exit_code
            if len(output) > limit:
                output = output[:limit] + f"\n[output truncated: {len(output)} chars, over the {limit} char limit]"
            outcome = "stopped" if stopped else "timed out after 120 s" if timed_out else f"exit code {exit_code}"
            context = (
                "The person ran a command in the repository:\n"
                f"$ {command}\n{output}\n({outcome})"
            )
            if conversation is None:
                with self._lock:
                    self._pending_user_context.append(context)
            else:
                conversation[0].add_context(context)
            return {
                "exit_code": exit_code,
                "output": output,
                "timed_out": timed_out,
                "stopped": stopped,
            }
        finally:
            with self._lock:
                if self._user_command is running:
                    self._user_command = running.next
                else:
                    previous = self._user_command
                    while previous is not None and previous.next is not running:
                        previous = previous.next
                    if previous is not None:
                        previous.next = running.next

    @staticmethod
    def _kill_user_command(process: subprocess.Popen) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _merge_usage(
        current: dict[str, UsageTotals], incoming: Mapping[str, UsageTotals]
    ) -> dict[str, UsageTotals]:
        merged = dict(current)
        for model, totals in incoming.items():
            merged[model] = merged.get(model, UsageTotals()).merged(totals)
        return merged

    def _session_usage(self, session_id: str) -> UsageTotals:
        directory = self._sessions_root / session_id
        paths = [directory / "run.jsonl", *sorted(directory.glob("agent-*.jsonl"))]
        totals = UsageTotals()
        for path in paths:
            if not path.is_file():
                continue
            try:
                records, _ = read_records(path)
            except (OSError, TranscriptError):
                continue
            for record in records:
                entry = _usage_record(record)
                if entry is not None:
                    totals = totals.merged(entry[2])
        return totals


    def _record_background_usage(
        self, session_id: str, model: str, usage: UsageTotals
    ) -> None:
        with self._lock:
            conversation = self._open_conversations.get(session_id)
            if conversation is None and self._conversation is not None and self._conversation[1].run_id == session_id:
                conversation = self._conversation
            if conversation is None:
                return
            agent_id = conversation[0].agent_ref.agent_id
            usage_by_agent = self._usage_by_session.setdefault(session_id, {})
            name, by_model = usage_by_agent.get(agent_id, ("leader", {}))
            updated = dict(by_model)
            updated[model] = updated.get(model, UsageTotals()).merged(usage)
            usage_by_agent[agent_id] = (name, updated)
            conversation[1].writer_for(agent_id, is_root=True).append(
                "usage",
                run_id=session_id,
                agent_id=agent_id,
                turn_id=None,
                data={
                    "model": model,
                    "agent_name": conversation[0].agent_ref.name,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "calls": usage.calls,
                    "cache_read_tokens": usage.cache_read_tokens,
                    "cache_write_tokens": usage.cache_write_tokens,
                    "cache_write_1h_tokens": usage.cache_write_1h_tokens,
                },
            )
            if self._conversation is not None and self._conversation[1].run_id == session_id:
                self._usage_by_agent = dict(usage_by_agent)
        self._publish_session(
            BackgroundUsageRecorded(
                agent_id=agent_id,
                run_id=new_id("memory_usage_"),
                session_id=session_id,
            ),
            session_id,
        )

    def _record_result(
        self, leader: Leader, result: LeaderRunResult, session_id: str | None = None,
    ) -> None:
        if session_id is None:
            session_id = self._session_by_root_agent.get(result.agent.agent_id)
        usage_by_agent = (
            self._usage_by_agent
            if session_id is None
            else self._usage_by_session.setdefault(session_id, {})
        )
        root_id = result.agent.agent_id
        root_current = usage_by_agent.get(root_id, (result.agent.name, {}))[1]
        usage_by_agent[root_id] = (
            result.agent.name,
            self._merge_usage(root_current, result.usage_by_agent.get(root_id, {})),
        )
        for name, record in result.subagents.items():
            usage_by_agent[record.agent_ref.agent_id] = (
                name,
                dict(record.usage_by_model),
            )
        context_report = account_context(
            leader._chat_messages,
            budget=leader.chat_token_budget,
        )
        if session_id is None or self._conversation is not None and self._conversation[1].run_id == session_id:
            self._context_report = context_report
            self._usage_by_agent = dict(usage_by_agent)
        if session_id is not None:
            self._context_by_session[session_id] = context_report

    def compact(self, instructions: str | None = None) -> CompactionResult:
        """Force compact the current conversation without locking other sessions."""
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            if self._conversation is None:
                raise NoConversationError("no conversation to compact")
            conversation = self._conversation
            leader, session = conversation
            session_id = session.run_id
            cancel = CancellationToken()
            active = _ActiveRun(new_id("run"), cancel, threading.current_thread(), leader.agent_ref.agent_id)
            self._active_by_session[session_id] = active
            self._active = active
        try:
            result, summary_usage = leader.force_compact_chat(instructions, cancel=cancel)
            with self._lock:
                usage_by_agent = dict(self._usage_by_session.get(session_id, {}))
                agent_id = leader.agent_ref.agent_id
                name, current_usage = usage_by_agent.get(
                    agent_id, (leader.agent_ref.name, {})
                )
                usage_by_agent[agent_id] = (
                    name,
                    self._merge_usage(current_usage, summary_usage),
                )
                context_report = account_context(
                    leader._chat_messages,
                    budget=leader.chat_token_budget,
                )
                for model, usage in summary_usage.items():
                    session.writer_for(agent_id, is_root=True).append(
                        "usage",
                        run_id=session_id,
                        agent_id=agent_id,
                        turn_id=None,
                        data={
                            "model": model,
                            "agent_name": name,
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "calls": usage.calls,
                            "cache_read_tokens": usage.cache_read_tokens,
                            "cache_write_tokens": usage.cache_write_tokens,
                            "cache_write_1h_tokens": usage.cache_write_1h_tokens,
                        },
                    )
                self._context_by_session[session_id] = context_report
                self._usage_by_session[session_id] = dict(usage_by_agent)
                if self._conversation is conversation:
                    self._usage_by_agent = dict(usage_by_agent)
                    self._context_report = context_report
                return result
        finally:
            cancel.close()
            with self._lock:
                if self._active_by_session.get(session_id) is active:
                    self._active_by_session.pop(session_id, None)
                if self._active is active:
                    self._active = None
                for event in active.deferred_memory_events:
                    self._publish(getattr(event, "run_id", active.run_id), event, session_id=session_id)
                active.deferred_memory_events.clear()

    def conversation_stats(self) -> dict | None:
        with self._lock:
            conversation = self._conversation
            if conversation is None:
                return None
            session_id = conversation[1].run_id
            graph_cache = getattr(self, "_stats_graph_cache", None)
            leader = conversation[0]
            active = self._active_by_session.get(session_id)
            active_root_agent_id = active.root_agent_id if active is not None else None
            active_runtime_run_id = active.runtime_run_id if active is not None else None
            selection = {
                "provider": leader._config.leader_provider.name,
                "model": leader._leader_spec.model.model,
                "effort": leader._leader_spec.model.effort,
            }
            report = self._context_by_session.get(session_id, self._context_report)
            policy = self._policy_by_session.get(session_id, self._policy)
            mode = policy.mode
            goal_payload = (
                self._goal.payload()
                if self._goal is not None and self._goal_session_id == conversation[1].run_id
                else None
            )
            plan = self._plans_by_session.get(session_id)
            instruction_warnings = list(self._instruction_warnings_by_session.get(session_id, []))
            config_warnings = (
                list(self._extensions.config.warnings) if self._extensions is not None else []
            )
            window_format = leader._config.leader_provider.wire_format
            price_table = self._price_table
            usage_by_agent = {
                agent_id: (name, dict(by_model))
                for agent_id, (name, by_model) in self._usage_by_session.get(
                    session_id, self._usage_by_agent,
                ).items()
            }

        store = conversation[1]
        started_at = store.run_start_times()
        # Graph construction also parses transcripts. Reuse it while their files
        # are unchanged, and do all filesystem work outside the host lock.
        try:
            signature = tuple(
                (path, stat.st_mtime_ns, stat.st_size)
                for path in sorted(store.directory.glob("*.jsonl"))
                for stat in (path.stat(),)
            )
        except OSError:
            signature = None
        if signature is not None and graph_cache is not None and graph_cache[:2] == (leader, signature):
            graph = graph_cache[2]
        else:
            graph = leader.run_graph()
            self._stats_graph_cache = (leader, signature, graph)

        def usage_fields(by_model: Mapping[str, UsageTotals]) -> dict:
            totals = UsageTotals()
            for usage in by_model.values():
                totals = totals.merged(usage)
            fields = {**totals.to_record(), "total_tokens": totals.total_tokens}
            cost = total_cost(by_model, price_table)
            if price_table is not None:
                unpriced = next((model for model in by_model if model not in price_table.prices), None)
                if unpriced is not None:
                    fields["unpriced_model"] = unpriced
            if cost is not None and price_table is not None:
                fields["cost"] = {
                    "amount": str(cost),
                    "currency": price_table.currency,
                    "estimated": any(
                        price_table.long_context_is_estimated(model, usage)
                        or price_table.cache_write_1h_is_estimated(model, usage)
                        for model, usage in by_model.items()
                    ),
                }
            return fields

        agents = []
        groups = {}
        agent_groups = {}

        def run_order(node):  # noqa: ANN001
            return node.run_id not in started_at, started_at.get(node.run_id, "")

        latest_runs = {}
        active_run_ids = set()

        def collect_run_state(node, in_active_run=False):  # noqa: ANN001
            is_active = in_active_run or (
                active_runtime_run_id is not None and node.run_id == active_runtime_run_id
            )
            previous = latest_runs.get(node.agent_id)
            if previous is None or run_order(previous[1]) <= run_order(node):
                latest_runs[node.agent_id] = (run_order(node), node)
            if is_active:
                active_run_ids.add(node.run_id)
            for child in node.children:
                collect_run_state(child, is_active)

        for root in graph:
            collect_run_state(root)

        def state_for_agent(agent_id):
            latest = latest_runs.get(agent_id)
            if latest is None:
                return "running" if agent_id == active_root_agent_id else None
            node = latest[1]
            if node.failed:
                return "failed"
            if node.stopped_reason is not None:
                return "stopped" if node.stopped_reason == "cancelled" else "finished"
            if node.run_id in active_run_ids or agent_id == active_root_agent_id:
                return "running"
            return None

        def add_node(node, parent_key=None, is_root=False):  # noqa: ANN001
            is_leader_root = is_root and node.parent_run_id is None
            name = "leader" if is_leader_root else usage_by_agent.get(
                node.agent_id, (node.agent_name, None)
            )[0]
            key = ("leader",) if is_leader_root else (parent_key, name)
            group = groups.get(key)
            if group is None:
                group = {"name": name, "agent_id": node.agent_id, "parent_key": parent_key, "parts": set()}
                groups[key] = group
                agents.append(group)
            group["agent_id"] = node.agent_id
            group["parts"].add(node.agent_id)
            agent_groups[node.agent_id] = key
            for child in sorted(node.children, key=run_order):
                add_node(child, key)

        for root in sorted(graph, key=run_order):
            add_node(root, is_root=True)
        for agent_id, (name, _) in usage_by_agent.items():
            if agent_id not in agent_groups:
                key = (None, name)
                group = groups.get(key)
                if group is None:
                    group = {"name": name, "agent_id": agent_id, "parent_key": None, "parts": set()}
                    groups[key] = group
                    agents.append(group)
                group["agent_id"] = agent_id
                group["parts"].add(agent_id)
                agent_groups[agent_id] = key

        folded_agents = []
        for group in agents:
            by_model = {}
            has_usage = False
            for agent_id in group["parts"]:
                usage = usage_by_agent.get(agent_id)
                if usage is not None:
                    has_usage = True
                    by_model = self._merge_usage(by_model, usage[1])
            folded = {
                "agent_id": group["agent_id"],
                "name": group["name"],
                "parent_agent_id": (
                    None if group["parent_key"] is None
                    else groups[group["parent_key"]]["agent_id"]
                ),
            }
            latest_states = [
                (latest_runs[agent_id][0], state_for_agent(agent_id))
                for agent_id in group["parts"] if agent_id in latest_runs
            ]
            if group["agent_id"] == active_root_agent_id and not latest_states:
                folded["state"] = "running"
            elif latest_states:
                _, state = max(latest_states, key=lambda item: item[0])
                if state is not None:
                    folded["state"] = state
            if has_usage:
                folded.update(usage_fields(by_model))
            folded_agents.append(folded)
        agents = folded_agents

        result = {
            "session_id": conversation[1].run_id,
            "agents": agents,
            "mode": mode,
            "goal": goal_payload,
            "plan": plan,
            "instruction_warnings": instruction_warnings,
            **selection,
        }
        if config_warnings:
            result["config_warnings"] = config_warnings
        all_models: dict[str, UsageTotals] = {}
        for _, by_model in usage_by_agent.values():
            all_models = self._merge_usage(all_models, by_model)
        result["usage"] = usage_fields(all_models)
        if report is None:
            return result
        result["context"] = {
            "used_tokens": report.total_tokens,
            "budget_tokens": report.budget,
            "remaining_tokens": report.remaining_tokens,
            "window_tokens": context_window_for_model(
                window_format, selection["model"],
            ),
            "by_source": {
                source.value: tokens for source, tokens in report.by_source().items()
            },
        }
        return result

    def changes(self) -> dict:
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            if self._conversation is None:
                return {"turns": [], "files": [], "worktrees": []}
            leader, session = self._conversation
            checkpoints = leader._config.checkpoints
            if checkpoints is None:
                return {"turns": [], "files": [], "worktrees": []}
            entries = checkpoints.entries()
            records, _ = read_records(session.directory / "run.jsonl")

            prompts: dict[str, str] = {}
            pending_key: str | None = None
            for record in records:
                if record.get("type") == "checkpoint":
                    key = record.get("data", {}).get("key")
                    if isinstance(key, str):
                        prompts.setdefault(key, "")
                        pending_key = key
                elif pending_key is not None and record.get("type") == "message":
                    message = message_from_json(record["data"])
                    if message.role is Role.USER:
                        prompts[pending_key] = message.text[:80]
                        pending_key = None

            paths_by_key: dict[str, list[str]] = {
                key: [] for key in checkpoints.keys()
            }
            first_entries: dict[str, CheckpointEntry] = {}
            for entry in entries:
                first_entries.setdefault(entry.path, entry)
                changed_paths = paths_by_key.setdefault(entry.key, [])
                if entry.path not in changed_paths:
                    changed_paths.append(entry.path)
            turns = [
                {
                    "key": key,
                    "prompt": prompts.get(key) or checkpoints.label(key) or "",
                    "paths": paths_by_key.get(key, []),
                }
                for key in checkpoints.keys()
            ]
            files: list[dict] = []
            for path, entry in first_entries.items():
                current = _checkpoint_file_bytes(checkpoints, path)
                backup = (
                    None
                    if entry.backup is None
                    else (checkpoints.directory / entry.backup).read_bytes()
                )
                if current == backup:
                    continue
                if current is None:
                    status = "deleted"
                elif backup is None:
                    status = "added"
                else:
                    status = "modified"
                current_digest = (
                    None if current is None else hashlib.sha256(current).hexdigest()
                )
                changed_outside = current_digest != checkpoints.last_written(path)
                truncated = False
                if b"\x00" in (current or b"") or b"\x00" in (backup or b""):
                    diff = "Binary file differs; diff omitted."
                else:
                    try:
                        before_text = "" if backup is None else backup.decode("utf-8")
                        after_text = "" if current is None else current.decode("utf-8")
                    except UnicodeDecodeError:
                        diff = "Binary file differs; diff omitted."
                    else:
                        result = _diff_result("changes", path, before_text, after_text)
                        diff = result.content
                        truncated = bool(result.payload["truncated"])
                files.append({
                    "path": path,
                    "status": status,
                    "changed_outside": changed_outside,
                    "diff": diff,
                    "truncated": truncated,
                })
            worktrees = []
            worktree_root = session.directory / "worktrees"
            if worktree_root.is_dir():
                for directory in sorted(item for item in worktree_root.iterdir() if item.is_dir()):
                    diff = worktree_diff(directory)
                    patch_bytes = diff.patch.encode("utf-8", errors="surrogateescape")
                    truncated = len(patch_bytes) > MAX_READ_BYTES
                    display_diff = (
                        os.fsdecode(patch_bytes[:MAX_READ_BYTES]) + "\n[diff truncated]"
                        if truncated else diff.patch
                    )
                    worktrees.append({
                        "name": directory.name,
                        "files": list(diff.files),
                        "diff": display_diff,
                        "truncated": truncated,
                    })
            return {"turns": turns, "files": files, "worktrees": worktrees}

    def apply_worktree(
        self, name: str, *, directory: Path | None = None,
        checkpoints_override: CheckpointStore | None = None,
    ) -> dict:
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            if self._conversation is None:
                raise KeyError(name)
            leader, session = self._conversation
            directory = (session.directory / "worktrees" / name) if directory is None else Path(directory)
            if not directory.is_dir():
                raise KeyError(name)
            checkpoints = checkpoints_override or leader._config.checkpoints
            if checkpoints is None:
                raise RuntimeError("checkpoint store is unavailable")
            diff = worktree_diff(directory)
            root = checkpoints.repo_root
            top_result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True,
                check=False,
            )
            if top_result.returncode:
                message = top_result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(message or "could not locate git repository root")
            git_root = Path(os.fsdecode(top_result.stdout.strip())).resolve()
            checkpoint_paths: list[tuple[str, Path]] = []
            for path in diff.files:
                target = (git_root / path).resolve()
                try:
                    target.relative_to(root)
                except ValueError:
                    continue
                checkpoint_paths.append((path, target))
            checked = subprocess.run(
                ["git", "apply", "--check", "--binary"],
                input=diff.patch.encode("utf-8", errors="surrogateescape"), cwd=git_root, capture_output=True,
                check=False,
            )
            if checked.returncode:
                message = checked.stderr.decode("utf-8", errors="replace").strip()
                raise WorktreeApplyConflict(message or "git apply --check failed")
            checkpoints.begin(new_id("chk"), label=f"Applied worktree {name}")
            for _, target in checkpoint_paths:
                checkpoints.before_write(target)
            applied = subprocess.run(
                ["git", "apply", "--binary"], input=diff.patch.encode("utf-8", errors="surrogateescape"),
                cwd=git_root, capture_output=True, check=False,
            )
            if applied.returncode:
                message = applied.stderr.decode("utf-8", errors="replace").strip()
                raise WorktreeApplyConflict(message or "git apply failed")
            for _, target in checkpoint_paths:
                checkpoints.after_write(target)
            remove_worktree(git_root, directory)
            if checkpoints_override is None:
                leader.forget_subagent(name)
            return {"applied": list(diff.files)}

    def discard_worktree(self, name: str) -> dict:
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            if self._conversation is None:
                raise KeyError(name)
            leader, session = self._conversation
            directory = session.directory / "worktrees" / name
            if not directory.is_dir():
                raise KeyError(name)
            remove_worktree(self._policy.repo_root, directory)
            leader.forget_subagent(name)
            return {"discarded": name}

    def revert_changes(
        self,
        *,
        path: str | None = None,
        key: str | None = None,
        force: bool = False,
    ) -> dict:
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            if self._conversation is None:
                raise KeyError(path if path is not None else key)
            leader, session = self._conversation
            checkpoints = leader._config.checkpoints
            if checkpoints is None:
                raise KeyError(path if path is not None else key)
            entries = checkpoints.entries()
            if path is not None:
                selected = next((entry for entry in entries if entry.path == path), None)
                if selected is None:
                    raise KeyError(path)
                targets = [selected]
            else:
                keys = checkpoints.keys()
                if key not in keys:
                    raise KeyError(key)
                first_index = keys.index(key)
                key_indexes = {item: index for index, item in enumerate(keys)}
                earliest: dict[str, CheckpointEntry] = {}
                for entry in entries:
                    if key_indexes.get(entry.key, -1) >= first_index:
                        earliest.setdefault(entry.path, entry)
                targets = list(earliest.values())
            outside: list[str] = []
            for entry in targets:
                current = _checkpoint_file_bytes(checkpoints, entry.path)
                digest = None if current is None else hashlib.sha256(current).hexdigest()
                if digest != checkpoints.last_written(entry.path):
                    outside.append(entry.path)
            if outside and not force:
                raise ChangedOutsideError(outside)
            for entry in targets:
                content = (
                    None
                    if entry.backup is None
                    else (checkpoints.directory / entry.backup).read_bytes()
                )
                checkpoints.restore(entry.path, content)
            if key is not None:
                def change(metadata):
                    undone_keys = metadata.get("undone_checkpoint_keys", [])
                    if not isinstance(undone_keys, list):
                        undone_keys = []
                    first_index = checkpoints.keys().index(key)
                    for reverted_key in checkpoints.keys()[first_index:]:
                        if reverted_key not in undone_keys:
                            undone_keys.append(reverted_key)
                    metadata["undone_checkpoint_keys"] = undone_keys
                session.update_meta(change)
            return {"reverted": [entry.path for entry in targets]}

    def _publish(self, host_run_id: str, event: Event, *, session_id: str | None = None) -> None:
        if session_id is None:
            with self._lock:
                session_id = next((
                    sid for sid, active in self._active_by_session.items()
                    if active.run_id == host_run_id
                ), None)
        if isinstance(event, PromptSubmitted) and session_id is not None:
            conversation = self._open_conversations.get(session_id)
            if conversation is not None and event.agent_id == conversation[0].agent_ref.agent_id:
                checkpoints = conversation[0]._config.checkpoints
                keys = () if checkpoints is None else checkpoints.keys()
                if keys:
                    event = replace(event, checkpoint_key=keys[-1])
        if isinstance(event, (RunFinished, RunFailed)) and session_id is not None:
            try:
                self._record_turn_edits(session_id, event)
            except (OSError, ValueError):
                _LOG.warning("Could not record turn edit totals for %s", session_id)
        if isinstance(event, RunStarted):
            try:
                with self._lock:
                    active = self._active_by_session.get(session_id or "")
                    if (
                        active is not None
                        and active.run_id == host_run_id
                        and active.root_agent_id == event.agent_id
                        and active.runtime_run_id is None
                    ):
                        active.runtime_run_id = event.run_id
            except Exception:
                # Observation must survive a bookkeeping failure in the host.
                pass
        if session_id is None:
            self._broker.publish(event)
        else:
            self._remember_event_session(event, session_id)
            self._broker.publish(event)

    def _perform_goal_check(self, context: GoalCheck, run_id: str) -> None:
        def set_process(process) -> None:  # noqa: ANN001
            with self._lock:
                context.process = process

        session_store = SessionStore.open(self._sessions_root, context.session_id)
        try:
            meta = session_store.read_meta()
        finally:
            session_store.close()
        spec_run = meta.get("spec_run")
        is_spec_run = isinstance(spec_run, dict) and spec_run.get("kind") == "implement"
        check_policy = self._policy_by_session[context.session_id]
        result = run_check(
            context.goal.check,
            check_policy.repo_root,
            cancelled=context.cancel,
            set_process=set_process,
            policy=check_policy,
            sandboxed=is_spec_run,
        )
        with self._lock:
            goal = context.goal
            if context.cleared:
                if self._goal_check is context:
                    self._goal_check = None
                    self._goal_check_thread = None
                self._goal_checks_by_session.pop(context.session_id, None)
                self._close_idle_conversations_locked()
                return
            goal.last_check = result
            next_prompt = None
            if result["ok"]:
                goal.phase = "complete"
                goal.reason = ""
            elif context.interrupted:
                goal.phase = "paused"
                goal.reason = context.interrupted
            elif context.cancel.is_set():
                goal.phase = "paused"
                goal.reason = "cancelled"
            elif goal.phase != "active":
                goal.phase = "paused"
                if not goal.reason:
                    goal.reason = "interrupted"
            elif goal.rounds < goal.max_rounds:
                round_number = goal.rounds
                rendered_check = " ".join(goal.check)
                next_prompt = (
                    f"Goal check failed (round {round_number} of {goal.max_rounds}): "
                    f"{rendered_check} exited {result['exit']}.\n{result['output']}\n"
                    f"Keep working toward the goal: {goal.objective}"
                )
            else:
                goal.phase = "blocked"
                goal.reason = "rounds exhausted"
            self._save_goal(context.session_id, goal)
            self._goals_by_session[context.session_id] = goal
            if self._goal is goal and self._goal_session_id == context.session_id:
                self._goal = goal
            if self._goal_check is context:
                self._goal_check = None
                self._goal_check_thread = None
            self._goal_checks_by_session.pop(context.session_id, None)
            self._goal_event("check", goal, run_id, context.agent_id)
            if goal.phase in ("complete", "blocked", "paused"):
                self._loops._finish_spec_run(context.session_id, goal.phase)
            if next_prompt is not None and not self._closing:
                try:
                    self._start_for_session(context.session_id, next_prompt)
                except Exception as exc:
                    goal.phase = "paused"
                    goal.reason = str(exc) or "could not start next round"
                    self._save_goal(context.session_id, goal)
                    self._goal_event("pause", goal, run_id, self._root_agent_id())
        with self._lock:
            self._close_idle_conversations_locked()


    def _run(
        self,
        run_id: str,
        leader: Leader,
        prompt: ContentInput,
        cancel: CancellationToken,
        goal_round: bool = False,
        session_id: str = "",
    ) -> None:
        terminal_event = None
        failure_published = False
        synthesized_failure = False
        active_to_finish = None
        failure: Exception | None = None
        try:
            result = leader.chat(prompt, cancel=cancel)
            with self._lock:
                if self._open_conversations.get(session_id, (None, None))[0] is leader:
                    self._record_result(leader, result, session_id)
        except Exception as exc:
            failure = exc
        finally:
            with self._lock:
                active = self._active_by_session.get(session_id)
                if active is not None and active.run_id == run_id:
                    terminal_event = active.terminal_event
                    active_to_finish = active
                    if terminal_event is None:
                        synthesized_failure = True
                        terminal_event = RunFailed(
                            agent_id=leader.agent_ref.agent_id, run_id=run_id,
                            agent_name="leader",
                            error=str(failure) if failure is not None else "host run ended without a terminal event",
                        )
                if (
                    active_to_finish is not None
                    and self._active_by_session.get(session_id) is active_to_finish
                ):
                    self._active_by_session.pop(session_id, None)
                    if self._conversation is not None and self._conversation[1].run_id == session_id:
                        self._active = None
                    if terminal_event is not None:
                        self._publish(run_id, terminal_event, session_id=session_id)
                        failure_published = isinstance(terminal_event, RunFailed)
                    for event in active_to_finish.deferred_memory_events:
                        event_run_id = getattr(event, "run_id", run_id)
                        self._publish(event_run_id, event, session_id=session_id)
                    active_to_finish.deferred_memory_events.clear()
                if self._closing and self._conversation is not None and self._conversation[1].run_id == session_id:
                    self._conversation = None
                    self._active = None
        if failure_published and not synthesized_failure:
            error = " ".join(terminal_event.error.splitlines())
            print(f"run {run_id} failed: {error}", file=sys.stderr, flush=True)
        elif synthesized_failure and failure is not None:
            import traceback
            traceback.print_exception(failure, file=sys.stderr)
        if not goal_round:
            self._loops._finish_spec_review(session_id, leader, terminal_event)
            self._loops._finish_loop_plan(session_id, terminal_event, error=failure)
            with self._lock:
                self._close_idle_conversations_locked()
            if failure is not None and not failure_published:
                raise failure
            return
        next_prompt = None
        with self._lock:
            goal = self._goals_by_session.get(session_id)
            if goal is None or session_id is None or goal.phase != "active":
                self._close_idle_conversations_locked()
                if failure is not None and not failure_published:
                    raise failure
                return
            if isinstance(terminal_event, RunFinished) and terminal_event.stopped_reason == "final_response":
                if goal.check:
                    context = GoalCheck(
                        goal, session_id, threading.Event(), leader.agent_ref.agent_id,
                    )
                    if self._conversation is not None and self._conversation[1].run_id == session_id:
                        self._goal_check = context
                        self._goal_check_thread = threading.current_thread()
                    self._goal_checks_by_session[session_id] = context
                    run_check_now = True
                elif goal.rounds < goal.max_rounds:
                    next_prompt = _unchecked_goal_prompt(goal)
                    run_check_now = False
                else:
                    goal.phase = "blocked"
                    goal.reason = "rounds exhausted"
                    self._save_goal(session_id, goal)
                    self._goal_event("round", goal, run_id, leader.agent_ref.agent_id)
                    run_check_now = False
            else:
                reason = (
                    terminal_event.stopped_reason
                    if isinstance(terminal_event, RunFinished)
                    else "cancelled" if cancel.cancelled else "failed"
                )
                goal.phase = "paused"
                goal.reason = reason
                self._save_goal(session_id, goal)
                self._loops._finish_spec_run(
                    session_id, "paused",
                    error=terminal_event.error if isinstance(terminal_event, RunFailed) else "",
                )
                self._goal_event("pause", goal, run_id, leader.agent_ref.agent_id)
                run_check_now = False
        if run_check_now:
            self._perform_goal_check(context, run_id)
        elif next_prompt is not None:
            try:
                self._start_for_session(session_id, next_prompt)
                with self._lock:
                    if self._goals_by_session.get(session_id) is goal:
                        self._goal_event("round", goal, run_id, leader.agent_ref.agent_id)
            except Exception as exc:
                with self._lock:
                    goal.phase = "paused"
                    goal.reason = str(exc) or "could not start next round"
                    self._save_goal(session_id, goal)
                    self._goal_event("pause", goal, run_id, self._root_agent_id())
        if failure is not None:
            if not failure_published:
                raise failure
            return
        with self._lock:
            self._close_idle_conversations_locked()
