"""Threaded conversation runs owned by a SymphonAI host process."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from symphonai_api.agent_memory import AgentMemory
from symphonai_api.checkpoints import CheckpointEntry, CheckpointStore
from symphonai_api.agent_spec import AgentSpec
from symphonai_api.budgets import RunBudget
from symphonai_api.cancellation import CancellationToken
from symphonai_api.compaction import CompactionResult, DEFAULT_RECENT_TURNS
from symphonai_api.config import resolve_run_budgets
from symphonai_api.context_report import ContextReport, account_context
from symphonai_api.cost import PriceTable, UsageTotals, total_cost
from symphonai_api.events import Event, JobChanged, PromptSubmitted, RunFailed, RunFinished, RunStarted, SubagentSpawned, fan_out
from symphonai_api.extensions import Extensions
from symphonai_api.environment import capture_environment
from symphonai_api.identity import new_id
from symphonai_api.jobs import BackgroundJob, JobManager
from symphonai_api.instructions import load_instructions
from symphonai_api.lsp import LspManager
from symphonai_api.leader import (
    AgentControlError, Leader, LeaderConfig, LeaderRunResult,
    builtin_subagent_specs,
)
from symphonai_api.leases import WorkspaceLeases
from symphonai_api.model_table import context_window_for_model
from symphonai_api.models import ContentInput, DocumentBlock, ImageBlock, Message, Role, TextBlock
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import ModelProvider, tool_result_text
from symphonai_api.runner import standard_tool_registry
from symphonai_api.session import (
    SessionError,
    SessionStore,
    default_memory_root,
    default_sessions_root,
    fork_run,
    load_run,
    load_run_for_resume,
    read_records,
    tool_result_search_path,
)
from symphonai_api.serialization import message_from_json
from symphonai_api.tool_results import ToolResultStore
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.edit import _diff_result
from symphonai_api.tools.filesystem import MAX_READ_BYTES
from symphonai_api.tools.metadata import call_target
from symphonai_api.tools.shell import shell_environment, shell_executable
from symphonai_api.worktree import WorktreeDiff, WorktreeError, _linked_paths, remove_worktree, worktree_diff
from symphonai_api.worktree import create_worktree
from symphonai_api.web_search import HttpJsonSearchBackend, search_endpoint
from symphonai_host.broker import EventBroker
from symphonai_host.approvals import ApprovalBroker
from symphonai_host.goal import Goal, GoalCheck, GoalChanged, goal_tools, run_check
from symphonai_host.plan import PlanChanged, plan_tools
from symphonai_host.questions import plan_proposal_tools, question_tools
from symphonai_host.protocol import HistoryMessage
from symphonai_host.loops import LoopStore


class RunActiveError(RuntimeError):
    """A client attempted to start a second run while one is active."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            run_id if run_id == "4 conversations are already running"
            else f"run already active: {run_id}"
        )
        self.run_id = run_id


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


PERMISSION_MODES = ("ask", "plan", "allow")
_sleep_keeper_lock = threading.Lock()
_sleep_keeper_holders: set[int] = set()
_sleep_keeper_process: subprocess.Popen | None = None


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
        getattr(provider, "base_url", None),
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


def _loop_git(cwd: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=cwd, capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(str(exc)) from exc
    if check and result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or f"git {' '.join(arguments)} failed")
    return result


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
    stop_requested: bool = False
    process: subprocess.Popen | None = None


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
        extensions: Extensions | None = None,
        mcp_tools: Mapping[str, LocalTool] | None = None,
        mcp_instructions: tuple[tuple[str, str], ...] = (),
        price_table: PriceTable | None = None,
        chat_token_budget: int | None = None,
        chat_recent_turns: int = DEFAULT_RECENT_TURNS,
        lsp: LspManager | None = None,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._repo_root = policy.repo_root
        self._leases = WorkspaceLeases(policy.repo_root)
        self._base_policy = replace(policy)
        self._extensions = extensions
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
            {"name": provider.name, "model": model, "base_url": getattr(provider, "base_url", None)}
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
        self._loop_validation_cancel: dict[str, threading.Event] = {}
        self._loop_validation_processes: dict[str, subprocess.Popen] = {}
        self._sleep_keeper_released = False
        self._sessions_root = default_sessions_root() if sessions_root is None else Path(sessions_root)
        self._memory_root = default_memory_root() if memory_root is None else Path(memory_root)
        self._memory: AgentMemory | None = None
        self._memory_open_attempted = False
        self._lock = threading.RLock()
        self.approvals = ApprovalBroker(publish_approval or (lambda _: False), timeout=approval_timeout)
        self._publish_approval_callback = publish_approval or (lambda _: False)
        self._approval_timeout = approval_timeout
        self._policy.approval_callback = self.approvals.callback

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
                        model if model is not None else getattr(provider, "model", None),
                        effort,
                    )
                metadata = session.read_meta()
                metadata["provider_choice"] = (
                    choice
                    if choice is not None
                    else {"name": provider.name, "model": model, "effort": effort}
                )
                if provider_changed:
                    metadata["provider_state_reset"] = True
                session.write_meta(metadata)
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
                "model": self._model if self._model is not None else getattr(self._provider, "model", None),
                "effort": self._effort,
            }

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

    def _save_goal(self, session_id: str, goal: Goal | None) -> None:
        current = self._conversation
        if current is not None and current[1].run_id == session_id:
            store = current[1]
            close = False
        else:
            store = SessionStore.open(self._sessions_root, session_id)
            close = True
        try:
            metadata = store.read_meta()
            if goal is None:
                metadata.pop("goal", None)
            else:
                metadata["goal"] = goal.payload()
            store.write_meta(metadata)
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
            metadata = store.read_meta()
            metadata["plan"] = plan
            store.write_meta(metadata)
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

    def start_spec_run(self, spec: dict) -> tuple[str, str]:
        """Start an isolated implementation conversation for a parsed spec."""
        with self._lock:
            if len(self._active_by_session) >= 4:
                raise RunActiveError("4 conversations are already running")
            previous = (self._conversation, self._policy, self._active)
            self._conversation = None
            self._active = None
            self._policy = replace(self._base_policy, repo_root=self._repo_root)
            self._policy.mode = "allow"
            objective = f"{spec['text']}\n\nWrite your report at {spec['report']}."
            if spec["validation"]:
                check = ("/bin/sh", "-c", "set -e\n" + "\n".join(spec["validation"]))
            else:
                check = ()
            goal = Goal(objective, check, max_rounds=5)
            try:
                run_id = self.start(
                    objective, _new_goal=goal,
                    _session_meta={"spec_run": {
                        "spec": spec["path"], "report": spec["report"],
                        "kind": "implement", "worktree": "worktree", "role": "implementer",
                        **({"loop_id": spec["loop_id"], "card_id": spec["card_id"]}
                           if spec.get("loop_id") and spec.get("card_id") else {}),
                        **({"start_ref": spec["start_ref"]} if spec.get("start_ref") else {}),
                    }},
                    _title=f"Run {Path(spec['path']).name}",
                )
                return self._conversation[1].run_id, run_id
            except Exception:
                self._conversation, self._policy, self._active = previous
                raise

    def start_spec_review(self, session_id: str) -> tuple[str, str]:
        from symphonai_host.spec_run import parse_spec, patch_digest

        with self._lock:
            source = SessionStore.open(self._sessions_root, session_id)
            try:
                source_meta = source.read_meta()
            finally:
                source.close()
            info = source_meta.get("spec_run")
            if not isinstance(info, dict) or info.get("kind") != "implement":
                raise KeyError(session_id)
            if info.get("state") not in ("finished", "blocked", "stopped"):
                raise RunActiveError("spec run has not finished")
            if session_id in self._active_by_session or session_id in self._goal_checks_by_session:
                raise RunActiveError("spec run is still active")
            prior_review = source_meta.get("review")
            if isinstance(prior_review, dict) and prior_review.get("session_id") in self._active_by_session:
                raise RunActiveError("review is already running")
            worktree = self._sessions_root / session_id / "worktree"
            diff = worktree_diff(worktree)
            baseline_files = {}
            for name in diff.files:
                file_path = worktree / name
                baseline_files[name] = (
                    hashlib.sha256(file_path.read_bytes()).hexdigest()
                    if file_path.is_file() else None
                )
            if info.get("loop_id"):
                spec_path = LoopStore(self._repo_root).path(info["loop_id"], Path(info["spec"]).relative_to(f".symphonai/loops/{info['loop_id']}").as_posix())
                spec = {"text": spec_path.read_text(encoding="utf-8")}
            else:
                spec = parse_spec(self._repo_root / info["spec"], self._repo_root)
            report_path = self._repo_root / info["report"]
            if not report_path.is_file():
                report_path = worktree / info["report"]
            report = report_path.read_text(encoding="utf-8") if report_path.is_file() else "No report was written."
            patch = diff.patch[:MAX_READ_BYTES]
            prompt = (
                f"Review this spec:\n\n{spec['text']}\n\nReport:\n\n{report}\n\n"
                f"Patch:\n\n{patch}\n\nReview the implementation and report. You may write follow-up specs under {Path(info['spec']).parent}/. "
                "Any follow-up spec's Validation section must hold one fenced `sh` block with one shell command "
                "per line, run from the project root; nothing else in that section is run. "
                "End the final answer with a last line `Verdict: pass` or `Verdict: follow-ups: <path>, <path>`."
            )
            previous = (self._conversation, self._policy, self._active)
            self._conversation = None
            self._active = None
            self._policy = replace(self._base_policy, repo_root=worktree)
            review_scope = worktree / "specs"
            if info.get("loop_id"):
                review_scope = worktree / Path(info["spec"]).parent
            self._policy.allowed_write_scope = [review_scope]
            self._policy.mode = "allow"
            meta = {
                "spec": info["spec"], "report": info["report"], "kind": "review",
                "of": session_id, "worktree_path": str(worktree),
                "baseline": patch_digest(diff.patch), "baseline_files": baseline_files,
                "role": "reviewer",
                **({"loop_id": info["loop_id"], "card_id": info.get("card_id")}
                   if info.get("loop_id") else {}),
            }
            try:
                run_id = self.start(
                    prompt, _session_meta={"spec_run": meta},
                    _title=f"Review {Path(info['spec']).name}",
                )
                review_id = self._conversation[1].run_id
                source = SessionStore.open(self._sessions_root, session_id)
                try:
                    source_meta = source.read_meta()
                    source_meta["review"] = {"session_id": review_id, "verdict": "running", "follow_ups": [], "not_copied": []}
                    source.write_meta(source_meta)
                finally:
                    source.close()
                return review_id, run_id
            except Exception:
                self._conversation, self._policy, self._active = previous
                raise

    def start_spec_plan(
        self, phase: str, item: int, title: str, phase_name: str,
        phase_plan: str, prompt: str, baseline: list[str],
    ) -> tuple[str, str]:
        with self._lock:
            if len(self._active_by_session) >= 4:
                raise RunActiveError("4 conversations are already running")
            previous = (self._conversation, self._policy, self._active)
            self._conversation = None
            self._active = None
            self._policy = replace(self._base_policy, repo_root=self._repo_root)
            self._policy.allowed_write_scope = [self._repo_root / "specs"]
            self._policy.shell_enabled = False
            self._policy.mode = "allow"
            full_prompt = (
                f"Roadmap phase: {phase_name}\nItem: {title}\n\n"
                f"Phase plan:\n{phase_plan or 'No phase plan is present.'}\n\n"
                f"{prompt}\n\nWrite exactly one spec under specs/{phase}/."
            )
            meta = {"spec_run": {
                "kind": "plan", "phase": phase, "item": item,
                "baseline_specs": baseline, "role": "planner",
            }}
            try:
                run_id = self.start(full_prompt, _session_meta=meta, _title=f"Plan {title}")
                return self._conversation[1].run_id, run_id
            except Exception:
                self._conversation, self._policy, self._active = previous
                raise

    def start_loop(self, goal: str, *, runner_pid: int | None = None) -> dict:
        store = LoopStore(self._repo_root)
        loop = store.create(goal)
        if runner_pid is not None:
            loop["runner_pid"] = runner_pid
            store.save(loop)
        return self._start_loop_planner(loop, "created")

    def _start_loop_planner(
        self, loop: dict, change: str, *, note: str | None = None,
        current_plan: dict | None = None,
    ) -> dict:
        store = LoopStore(self._repo_root)
        folder = store.path(loop["id"], ".")
        revision = ""
        if note is not None:
            revision = (
                f"\n\nRevision note from the person:\n{note}\n\n"
                "Current plan to revise:\n"
                + json.dumps(current_plan or {}, ensure_ascii=False, indent=2)
            )
        prompt = (
            f"Plan this goal: {loop['goal']}\n\n"
            f"Write one Markdown spec per card at {folder / 'specs'}/<id>-<slug>.md. Use as few cards as the goal needs: "
            "one card is right for a small change. Split only where a card can be implemented and reviewed on its own. "
            "Use ids starting at a in order, with at most 8 cards. "
            f"Follow {self._repo_root / 'specs' / 'TEMPLATE.md'} if it exists; otherwise use the sections "
            "Goal, Scope, Context, Contract, Acceptance criteria, Validation, and Report. "
            "The Validation section must hold one fenced `sh` block with one shell command per line, "
            "run from the project root; nothing else in that section is run. "
            "Inspect the project before making claims. Group specs whose Scope paths overlap or that depend "
            "on each other into the same lane, with at most 3 lanes. Write plan.json last at "
            f"{folder / 'plan.json'} as {{\"cards\":[{{\"id\":\"a\",\"title\":\"Short title\","
            "\"spec\":\"specs/a-short-title.md\",\"lane\":1,\"after\":[]}]}}. "
            "A card whose work documents, tests, or builds on another card's change lists that card in `after`, "
            "whether or not they share a lane. A card starts from the work of the cards before it and sees nothing else. "
            "For this loop, that plan.json path is the only permitted exception to the planner instruction "
            "to write under specs/." + revision
        )
        with self._lock:
            if len(self._active_by_session) >= 4:
                raise RunActiveError("4 conversations are already running")
            previous = (self._conversation, self._policy, self._active)
            self._conversation = None
            self._active = None
            self._policy = replace(self._base_policy, repo_root=self._repo_root)
            self._policy.allowed_write_scope = [folder / "specs", folder / "plan.json"]
            self._policy.shell_enabled = False
            self._policy.mode = "allow"
            try:
                self.start(
                    prompt,
                    _session_meta={"spec_run": {
                        "kind": "loop_plan", "loop_id": loop["id"], "role": "planner",
                    }},
                    _title=f"Plan: {loop['goal'][:60]}",
                )
                loop["planner_session_id"] = self._conversation[1].run_id
                store.save(loop)
                self._loop_changed(loop, change)
                return loop
            except Exception:
                if change == "created":
                    self._loop_changed(loop, change)
                self._conversation, self._policy, self._active = previous
                raise

    def _loop_changed(self, loop: dict, change: str, card_id: str = "") -> None:
        self._broker.publish(LoopChanged(
            agent_id="", run_id="", loop_id=loop["id"], state=loop["state"],
            change=change, card_id=card_id, reason=loop.get("reason", ""),
        ))

    def approve_loop_plan(self, loop_id: str) -> dict:
        with self._lock:
            store = LoopStore(self._repo_root)
            loop = store.load(loop_id)
            if loop["state"] != "awaiting_plan":
                raise RunActiveError(loop["state"])
            if not any(card.get("state") != "dropped" for card in loop.get("cards", [])):
                raise ValueError("nothing left to run")
            branch_prefix = f"symphonai/{loop['slug']}"
            integration_admin = self._sessions_root / "loops" / loop_id / "integration"
            branch_created = False
            integration_created = False
            branch = branch_prefix
            try:
                _loop_git(self._repo_root, "rev-parse", "--show-toplevel")
                suffix = 1
                while True:
                    branch = branch_prefix if suffix == 1 else f"{branch_prefix}-{suffix}"
                    result = _loop_git(
                        self._repo_root, "show-ref", "--verify", "--quiet",
                        f"refs/heads/{branch}", check=False,
                    )
                    if result.returncode != 0:
                        break
                    suffix += 1
                _loop_git(self._repo_root, "branch", branch)
                branch_created = True
                loop["base_commit"] = os.fsdecode(
                    _loop_git(self._repo_root, "rev-parse", "HEAD").stdout.strip(),
                )
                integration_root = create_worktree(
                    self._repo_root, integration_admin, start=branch,
                )
                integration_created = True
                _loop_git(integration_root, "switch", branch)
                loop["branch"] = branch
                store.set_state(loop, "running")
            except Exception as exc:
                if integration_created or integration_admin.exists():
                    try:
                        remove_worktree(self._repo_root, integration_admin)
                    except (OSError, WorktreeError):
                        shutil.rmtree(integration_admin, ignore_errors=True)
                        try:
                            _loop_git(self._repo_root, "worktree", "prune")
                        except RuntimeError:
                            pass
                for directory in (integration_admin.parent, integration_admin.parent.parent):
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
                if branch_created:
                    try:
                        _loop_git(self._repo_root, "branch", "-D", branch)
                    except RuntimeError:
                        pass
                try:
                    persisted = store.load(loop_id)
                    if persisted.get("state") != "awaiting_plan" or persisted.get("branch") is not None:
                        persisted["state"] = "awaiting_plan"
                        persisted["branch"] = None
                        store.save(persisted)
                except (KeyError, OSError, ValueError):
                    pass
                raise ValueError(str(exc)) from exc
            self._loop_changed(loop, "state")
            self._advance_loop(loop_id)
            loop = store.load(loop_id)
            return loop

    def drop_loop_cards(self, loop_id: str, card_ids: list[str]) -> dict:
        with self._lock:
            store = LoopStore(self._repo_root)
            loop = store.load(loop_id)
            if loop["state"] != "awaiting_plan":
                raise RunActiveError(loop["state"])
            cards = loop.get("cards", [])
            known = {card.get("id") for card in cards}
            unknown = sorted(set(card_ids) - known)
            if unknown:
                raise ValueError(f"unknown card id: {unknown[0]}")
            dropped = set(card_ids)
            for card in cards:
                if card.get("id") not in dropped and dropped.intersection(card.get("after", [])):
                    raise ValueError(f"kept card {card.get('id')} depends on a dropped card")
            changed = []
            for card in cards:
                if card.get("id") in dropped and card.get("state") != "dropped":
                    card["state"] = "dropped"
                    changed.append(card["id"])
            if changed:
                store.save(loop)
                for card_id in changed:
                    self._loop_changed(loop, "card", card_id)
            return loop

    def revise_loop_plan(self, loop_id: str, note: str) -> dict:
        with self._lock:
            store = LoopStore(self._repo_root)
            loop = store.load(loop_id)
            if loop["state"] != "awaiting_plan":
                raise RunActiveError(loop["state"])
            if len(self._active_by_session) >= 4:
                raise RunActiveError("4 conversations are already running")
            current_plan = {"lanes": loop.get("lanes", []), "cards": loop.get("cards", [])}
            plan_path = store.path(loop_id, "plan.json")
            previous_plan = plan_path.read_bytes() if plan_path.is_file() else None
            plan_path.unlink(missing_ok=True)
            store.set_state(loop, "planning")
            try:
                return self._start_loop_planner(
                    loop, "state", note=note, current_plan=current_plan,
                )
            except Exception:
                loop = store.load(loop_id)
                store.set_state(loop, "awaiting_plan")
                if previous_plan is not None:
                    plan_path.write_bytes(previous_plan)
                self._loop_changed(loop, "state")
                raise

    def cancel_loop(self, loop_id: str) -> dict:
        worktree_ids = set()
        session_ids = set()
        with self._lock:
            store = LoopStore(self._repo_root)
            loop = store.load(loop_id)
            if loop["state"] in ("done", "cancelled"):
                raise RunActiveError(loop["state"])
            if isinstance(loop.get("planner_session_id"), str):
                session_ids.add(loop["planner_session_id"])
            for card in loop.get("cards", []):
                session_id = card.get("session_id")
                if isinstance(session_id, str):
                    session_ids.add(session_id)
                    worktree_ids.add(session_id)
                    try:
                        source = SessionStore.open(self._sessions_root, session_id)
                        try:
                            review_id = source.read_meta().get("review", {}).get("session_id")
                            if isinstance(review_id, str):
                                session_ids.add(review_id)
                        finally:
                            source.close()
                    except (OSError, SessionError, AttributeError):
                        pass
            for session_id in session_ids:
                active = self._active_by_session.get(session_id)
                if active is not None:
                    active.cancel.cancel()
                approvals = self._approvals_by_session.get(session_id)
                if approvals is not None:
                    approvals.cancel_all(reason="loop cancelled")
                goal_check = self._goal_checks_by_session.get(session_id)
                if goal_check is not None:
                    goal_check.cancel.set()
            validation_cancel = self._loop_validation_cancel.get(loop_id)
            if validation_cancel is not None:
                validation_cancel.set()
            process = self._loop_validation_processes.get(loop_id)
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    process.kill()
            store.set_state(loop, "cancelled")
            self._loop_changed(loop, "state")
            active_runs = [self._active_by_session[sid] for sid in session_ids if sid in self._active_by_session]
            goal_thread = self._goal_check_thread
        for active in active_runs:
            if active.thread is not threading.current_thread():
                active.thread.join(timeout=5)
        if goal_thread is not None and goal_thread is not threading.current_thread():
            goal_thread.join(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with self._lock:
                checks_running = any(sid in self._goal_checks_by_session for sid in session_ids)
                validation_running = loop_id in self._loop_validation_cancel
            if not checks_running and not validation_running:
                break
            time.sleep(0.01)
        for session_id in worktree_ids:
            worktree = self._sessions_root / session_id / "worktree"
            if worktree.exists():
                remove_worktree(self._repo_root, worktree)
        integration_admin = self._sessions_root / "loops" / loop_id / "integration"
        if integration_admin.exists():
            remove_worktree(self._repo_root, integration_admin)
        branch = loop.get("branch")
        if isinstance(branch, str):
            _loop_git(self._repo_root, "branch", "-D", branch)
        return store.load(loop_id)

    def integrate(self, loop: dict, card: dict, worktree: Path) -> str:
        loop_id = loop.get("id")
        LoopStore(self._repo_root).path(loop_id, ".")
        integration_root = self._loop_integration_root(loop_id)
        diff = self._loop_card_diff(loop_id, Path(worktree))
        if not diff.patch:
            return ""
        linked = _linked_paths(integration_root.resolve())
        status = _loop_git(
            integration_root, "status", "--porcelain", "--untracked-files=all", "--", ".",
            *(f":(exclude){path}" for path in linked),
        ).stdout
        if status:
            raise RuntimeError("integration worktree has uncommitted changes")
        patch = diff.patch.encode("utf-8", errors="surrogateescape")
        checked = subprocess.run(
            ["git", "apply", "--check", "--binary"], cwd=integration_root,
            input=patch, capture_output=True, check=False,
        )
        if checked.returncode:
            message = checked.stderr.decode("utf-8", errors="replace").strip()
            raise WorktreeApplyConflict(message or "git apply --check failed")
        applied = subprocess.run(
            ["git", "apply", "--binary"], cwd=integration_root,
            input=patch, capture_output=True, check=False,
        )
        if applied.returncode:
            message = applied.stderr.decode("utf-8", errors="replace").strip()
            raise WorktreeApplyConflict(message or "git apply failed")
        worktree_diff(integration_root)
        _loop_git(integration_root, "commit", "-m", f"{card['id']}: {card['title']}")
        return os.fsdecode(_loop_git(integration_root, "rev-parse", "HEAD").stdout.strip())

    @staticmethod
    def _loop_card_diff(loop_id: str, worktree: Path) -> WorktreeDiff:
        worktree_diff(worktree)
        loop_files = f":(exclude).symphonai/loops/{loop_id}/**"
        files = _loop_git(
            worktree, "diff", "--cached", "--name-only", "--relative", "-z", "HEAD",
            "--", ".", loop_files,
        ).stdout
        patch = _loop_git(
            worktree, "diff", "--cached", "--binary", "HEAD", "--", ".", loop_files,
        ).stdout
        return WorktreeDiff(
            patch=os.fsdecode(patch),
            files=tuple(os.fsdecode(item) for item in files.split(b"\0") if item),
        )

    def _loop_integration_root(self, loop_id: str) -> Path:
        LoopStore(self._repo_root).path(loop_id, ".")
        top = Path(os.fsdecode(_loop_git(
            self._repo_root, "rev-parse", "--show-toplevel",
        ).stdout.strip())).resolve()
        relative_root = self._repo_root.resolve().relative_to(top)
        return self._sessions_root / "loops" / loop_id / "integration" / relative_root

    def _loop_card_changed(self, loop: dict, card: dict) -> None:
        LoopStore(self._repo_root).save(loop)
        self._loop_changed(loop, "card", str(card.get("id", "")))

    def _loop_attention(self, loop: dict, reason: str, card: dict | None = None) -> None:
        store = LoopStore(self._repo_root)
        if card is not None:
            card["state"] = "failed"
            self._loop_card_changed(loop, card)
        if loop.get("state") == "running":
            store.set_state(loop, "needs_attention", reason)
        else:
            loop["reason"] = reason
            store.save(loop)
        self._loop_changed(loop, "state")

    def _advance_loop(self, loop_id: str) -> None:
        """Start ready lane cards while conversation slots are available."""
        with self._lock:
            store = LoopStore(self._repo_root)
            loop = store.load(loop_id)
            if loop.get("state") == "validating":
                self._start_loop_validation(loop_id)
                return
            if loop.get("state") != "running":
                return
            integration = self._loop_integration_root(loop_id)
            active_count = len(self._active_by_session)
            lanes = loop.get("lanes", [])
            lane_order = [lane.get("id", lane.get("lane", index + 1)) for index, lane in enumerate(lanes)]
            cards = loop.get("cards", [])
            for card in cards:
                if active_count >= 4:
                    break
                if card.get("state") != "review" or not card.get("session_id"):
                    continue
                try:
                    source = SessionStore.open(self._sessions_root, card["session_id"])
                    try:
                        review = source.read_meta().get("review")
                    finally:
                        source.close()
                    if isinstance(review, dict) and review.get("session_id"):
                        continue
                    self.start_spec_review(card["session_id"])
                    active_count += 1
                except RunActiveError:
                    break
                except Exception as exc:
                    self._loop_attention(loop, f"card {card['id']} review could not start: {exc}", card)
                    return
            for lane_id in lane_order:
                if active_count >= 4:
                    break
                lane_cards = [card for card in cards if card.get("lane") == lane_id]
                if any(card.get("state") in ("running", "review") for card in lane_cards):
                    continue
                card = next((item for item in lane_cards if item.get("state") == "planned"), None)
                if card is None or any(
                    next((dep.get("state") for dep in cards if dep.get("id") == dep_id), None) != "done"
                    for dep_id in card.get("after", [])
                ):
                    continue
                spec_value = card.get("spec")
                try:
                    if not isinstance(spec_value, str):
                        raise ValueError("card has no spec path")
                    spec_path = store.path(loop_id, spec_value)
                    source = spec_path.read_text(encoding="utf-8")
                    from symphonai_host.spec_run import spec_validation
                    validation = spec_validation(source)
                    if not validation:
                        raise ValueError(f"card {card['id']} has no Validation commands")
                    tip = os.fsdecode(_loop_git(integration, "rev-parse", "HEAD").stdout.strip())
                    card_spec_path = spec_path.relative_to(self._repo_root).as_posix()
                    report_path = store.path(
                        loop_id, card.get("report", f"reports/{card['id']}-report.md"),
                    )
                    relative_report = report_path.relative_to(self._repo_root).as_posix()
                    _, session_id = self.start_spec_run({
                        "text": source, "validation": validation,
                        "path": card_spec_path, "report": relative_report,
                        "loop_id": loop_id, "card_id": card["id"], "start_ref": tip,
                    })
                except RunActiveError:
                    break
                except Exception as exc:
                    self._loop_attention(loop, f"card {card.get('id')} could not start: {exc}", card)
                    return
                card["state"] = "running"
                card["session_id"] = session_id
                self._loop_card_changed(loop, card)
                active_count += 1
            non_dropped = [card for card in cards if card.get("state") != "dropped"]
            if non_dropped and all(card.get("state") == "done" for card in non_dropped):
                store.set_state(loop, "validating")
                self._loop_changed(loop, "state")
                self._start_loop_validation(loop_id)

    def _finish_loop_card_run(
        self, session_id: str, state: str, *, error: str = "", goal_reason: str = "",
    ) -> None:
        info = {}
        try:
            source = SessionStore.open(self._sessions_root, session_id)
            try:
                info = source.read_meta().get("spec_run", {})
            finally:
                source.close()
            loop_id, card_id = info.get("loop_id"), info.get("card_id")
            if not loop_id or not card_id:
                return
            with self._lock:
                store = LoopStore(self._repo_root)
                loop = store.load(loop_id)
                if loop.get("state") == "cancelled":
                    return
                card = next(item for item in loop["cards"] if item.get("id") == card_id)
                if state == "finished":
                    card["state"] = "review"
                    self._loop_card_changed(loop, card)
                    self._advance_loop(loop_id)
                    return
                cause = error or goal_reason or state
                self._loop_attention(loop, f"card {card_id} could not finish: {cause}", card)
        except (OSError, KeyError, ValueError, SessionError) as exc:
            if isinstance(info, dict) and info.get("loop_id"):
                with self._lock:
                    loop = LoopStore(self._repo_root).load(info["loop_id"])
                    self._loop_attention(loop, f"card {info.get('card_id')} could not finish: {exc}")

    def _finish_loop_card_review(self, review_info: dict) -> None:
        loop_id, card_id = review_info.get("loop_id"), review_info.get("card_id")
        if not loop_id or not card_id:
            return
        with self._lock:
            store = LoopStore(self._repo_root)
            loop = store.load(loop_id)
            if loop.get("state") == "cancelled":
                return
            card = next(item for item in loop["cards"] if item.get("id") == card_id)
            source = SessionStore.open(self._sessions_root, review_info["of"])
            try:
                result = source.read_meta().get("review", {})
            finally:
                source.close()
            verdict = result.get("verdict")
            if verdict not in ("passed", "follow-ups"):
                error = result.get("error")
                if error:
                    reason = f"card {card_id} review failed: {error}"
                elif verdict == "tree-changed":
                    reason = f"card {card_id} review changed files other than its follow-up specs"
                elif verdict == "no-verdict":
                    reason = f"card {card_id} review gave no verdict"
                else:
                    reason = f"card {card_id} review did not pass"
                self._loop_attention(loop, reason, card)
                return
            try:
                card_worktree = self._sessions_root / review_info["of"] / "worktree"
                commit = self.integrate(loop, card, card_worktree)
            except (OSError, RuntimeError) as exc:
                self._loop_attention(loop, str(exc), card)
                return
            if card_worktree.exists():
                remove_worktree(self._repo_root, card_worktree)
            if commit:
                card["commit"] = commit
            card["state"] = "done"
            self._loop_card_changed(loop, card)
            if verdict == "follow-ups":
                original_id = card.get("follow_up_of") or card_id
                rounds = int(card.get("rounds", 0)) + 1
                if rounds > 2:
                    self._loop_attention(
                        loop, f"card {original_id} still has findings after 2 follow-up rounds",
                    )
                    return
                follow_ups = result.get("follow_ups", [])
                for follow_path in follow_ups:
                    relative = Path(follow_path)
                    prefix = f".symphonai/loops/{loop_id}/"
                    if not follow_path.startswith(prefix) or ".." in relative.parts:
                        continue
                    source_path = self._repo_root / relative
                    if not source_path.is_file():
                        continue
                    suffix = "F" if rounds == 1 else "F2"
                    follow_id = f"{original_id}{suffix}"
                    relative_spec = relative.relative_to(prefix).as_posix()
                    if any(item.get("id") == follow_id for item in loop["cards"]):
                        continue
                    follow = {
                        "id": follow_id, "title": f"Follow up: {card['title']}",
                        "spec": relative_spec, "lane": card["lane"],
                        "after": [card_id], "state": "planned",
                        "follow_up_of": original_id, "rounds": rounds,
                    }
                    loop["cards"].append(follow)
                    self._loop_card_changed(loop, follow)
                    break
            self._advance_loop(loop_id)

    def _start_loop_validation(self, loop_id: str) -> None:
        with self._lock:
            if loop_id in self._loop_validation_cancel or self._closing:
                return
            loop = LoopStore(self._repo_root).load(loop_id)
            if loop.get("state") != "validating":
                return
            cancelled = threading.Event()
            self._loop_validation_cancel[loop_id] = cancelled
            thread = threading.Thread(
                target=self._validate_loop, args=(loop_id, cancelled),
                name=f"symphonai-loop-validation-{loop_id}", daemon=True,
            )
            try:
                thread.start()
            except Exception:
                self._loop_validation_cancel.pop(loop_id, None)
                raise

    def _run_loop_validation_command(
        self, loop_id: str, command: str, cwd: Path, cancelled: threading.Event,
    ) -> dict:
        from symphonai_api.tools.shell import (
            SANDBOX_EXEC, _bwrap_command, _sandbox_available, _seatbelt_profile,
        )

        policy = replace(self._base_policy, repo_root=cwd)
        policy.mode = "allow"
        argv = [shell_executable(), "-c", command]
        available = _sandbox_available()
        if policy.shell_sandbox and not available:
            return {"command": command, "exit": None, "output": "sandbox requested but unavailable on this platform"}
        if available:
            argv = (
                _bwrap_command(policy, argv, cwd)
                if sys.platform == "linux"
                else [SANDBOX_EXEC, "-p", _seatbelt_profile(policy), *argv]
            )
        try:
            with tempfile.TemporaryFile() as output_file:
                process = subprocess.Popen(
                    argv, cwd=cwd, env=shell_environment(cwd, for_agent=True),
                    stdin=subprocess.DEVNULL, stdout=output_file, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                with self._lock:
                    self._loop_validation_processes[loop_id] = process
                deadline = time.monotonic() + 600
                timed_out = False
                while process.poll() is None:
                    if cancelled.is_set():
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except (OSError, ProcessLookupError):
                            process.kill()
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except (OSError, ProcessLookupError):
                            process.kill()
                        break
                    time.sleep(min(0.05, remaining))
                process.wait()
                output_file.seek(max(0, os.fstat(output_file.fileno()).st_size - 4000))
                output = output_file.read()[-4000:].decode("utf-8", errors="replace")
                code = None if timed_out or cancelled.is_set() else process.returncode
                return {"command": command, "exit": code, "output": output}
        except OSError as exc:
            return {"command": command, "exit": None, "output": str(exc)[-4000:]}
        finally:
            with self._lock:
                self._loop_validation_processes.pop(loop_id, None)

    def _validate_loop(self, loop_id: str, cancelled: threading.Event) -> None:
        from symphonai_host.spec_run import spec_validation

        store = LoopStore(self._repo_root)
        try:
            loop = store.load(loop_id)
            if loop.get("state") != "validating":
                return
            commands = []
            seen = set()
            for card in loop.get("cards", []):
                if card.get("state") != "done":
                    continue
                spec = store.path(loop_id, card["spec"]).read_text(encoding="utf-8")
                for line in spec_validation(spec):
                    if line not in seen:
                        commands.append(line)
                        seen.add(line)
            integration = self._loop_integration_root(loop_id)
            validation = [
                self._run_loop_validation_command(loop_id, command, integration, cancelled)
                for command in commands
            ]
            if cancelled.is_set():
                return
            history = _loop_git(integration, "log", "--format=%H%x09%s").stdout.decode(errors="replace").splitlines()
            commit_by_subject = {}
            for line in history:
                commit_id, separator, subject = line.partition("\t")
                if separator:
                    commit_by_subject[subject] = commit_id
            done_cards = [card for card in loop.get("cards", []) if card.get("state") == "done"]
            commits = []
            for card in done_cards:
                commit_id = card.get("commit") or commit_by_subject.get(f"{card['id']}: {card['title']}")
                if commit_id:
                    commits.append({"card": card["id"], "commit": commit_id, "title": card["title"]})
            base_commit = loop.get("base_commit")
            if not isinstance(base_commit, str) or not base_commit:
                history_positions = {entry.partition("\t")[0]: index for index, entry in enumerate(history)}
                oldest = max(
                    (entry["commit"] for entry in commits),
                    key=lambda commit_id: history_positions.get(commit_id, -1),
                    default=None,
                )
                if oldest:
                    base_commit = os.fsdecode(_loop_git(integration, "rev-parse", f"{oldest}^").stdout.strip())
                else:
                    base_commit = os.fsdecode(_loop_git(integration, "rev-parse", "HEAD").stdout.strip())
            head = os.fsdecode(_loop_git(integration, "rev-parse", "HEAD").stdout.strip())
            files = _loop_git(integration, "diff", "--name-only", f"{base_commit}..{head}").stdout
            summary = {
                "commits": commits,
                "files": sorted(item for item in os.fsdecode(files).splitlines() if item),
                "validation": validation,
            }
            with self._lock:
                loop = store.load(loop_id)
                if loop.get("state") != "validating" or cancelled.is_set():
                    return
                loop["summary"] = summary
                failed = next((item for item in validation if item["exit"] != 0), None)
                if failed is None:
                    store.set_state(loop, "awaiting_commit")
                else:
                    store.set_state(loop, "needs_attention", f"validation failed: {failed['command']}")
                self._loop_changed(loop, "state")
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            with self._lock:
                try:
                    loop = store.load(loop_id)
                    if loop.get("state") == "validating":
                        store.set_state(loop, "needs_attention", f"validation failed: {exc}")
                        self._loop_changed(loop, "state")
                except (KeyError, ValueError, OSError):
                    pass
        finally:
            with self._lock:
                self._loop_validation_cancel.pop(loop_id, None)

    def commit_loop(self, loop_id: str, action: str) -> dict:
        with self._lock:
            store = LoopStore(self._repo_root)
            loop = store.load(loop_id)
            if loop.get("state") != "awaiting_commit":
                raise RunActiveError(loop.get("state", "unknown"))
            if action not in ("approve", "decline"):
                raise ValueError("loop commit action must be approve or decline")
            integration_admin = self._sessions_root / "loops" / loop_id / "integration"
            branch = loop.get("branch")
            try:
                if integration_admin.exists():
                    remove_worktree(self._repo_root, integration_admin)
                for card in loop.get("cards", []):
                    session_id = card.get("session_id")
                    if isinstance(session_id, str):
                        card_worktree = self._sessions_root / session_id / "worktree"
                        if card_worktree.exists():
                            remove_worktree(self._repo_root, card_worktree)
                if action == "decline" and isinstance(branch, str):
                    _loop_git(self._repo_root, "branch", "-D", branch)
            except (OSError, WorktreeError, RuntimeError) as exc:
                raise ValueError(str(exc)) from exc
            if action == "approve":
                store.set_state(loop, "done")
            else:
                store.set_state(loop, "cancelled", "declined")
            self._loop_changed(loop, "state")
            return loop

    def resume_loop(self, loop_id: str) -> dict:
        with self._lock:
            store = LoopStore(self._repo_root)
            loop = store.load(loop_id)
            if loop.get("state") != "needs_attention":
                raise RunActiveError(loop.get("state", "unknown"))
            for card in loop.get("cards", []):
                if card.get("state") != "failed":
                    continue
                session_id = card.get("session_id")
                if isinstance(session_id, str):
                    worktree = self._sessions_root / session_id / "worktree"
                    if worktree.exists():
                        remove_worktree(self._repo_root, worktree)
                card.update(state="planned", session_id=None)
                self._loop_card_changed(loop, card)
            if str(loop.get("reason", "")).startswith("validation failed:"):
                store.set_state(loop, "validating", "")
                self._loop_changed(loop, "state")
                self._start_loop_validation(loop_id)
            else:
                store.set_state(loop, "running", "")
                self._loop_changed(loop, "state")
                self._advance_loop(loop_id)
            return store.load(loop_id)

    def loop_card_details(self, loop_id: str, card_id: str) -> dict:
        store = LoopStore(self._repo_root)
        loop = store.load(loop_id)
        card = next((item for item in loop.get("cards", []) if item.get("id") == card_id), None)
        if card is None:
            raise KeyError(card_id)

        def read_text(relative: str) -> str:
            try:
                path = store.path(loop_id, relative)
                with path.open("rb") as source:
                    return source.read(MAX_READ_BYTES).decode("utf-8", errors="replace")
            except (OSError, ValueError, TypeError):
                return ""

        spec = read_text(card.get("spec", ""))
        report = read_text(card.get("report", ""))
        diff_bytes = b""
        commit = card.get("commit")
        if not isinstance(commit, str):
            commit = next((
                item.get("commit") for item in loop.get("summary", {}).get("commits", [])
                if item.get("card") == card_id
            ), None)
        branch = loop.get("branch")
        valid_branch = isinstance(branch, str) and re.fullmatch(r"symphonai/[a-z0-9-]+", branch)
        if (isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40,64}", commit)
                and valid_branch):
            ancestry = _loop_git(
                self._repo_root, "merge-base", "--is-ancestor", commit, branch, check=False,
            )
            if ancestry.returncode == 0:
                diff_bytes = _loop_git(
                    self._repo_root, "show", "--format=", "--binary", commit,
                ).stdout
        elif isinstance(card.get("session_id"), str):
            worktree = self._sessions_root / card["session_id"] / "worktree"
            if worktree.is_dir():
                try:
                    diff_bytes = worktree_diff(worktree).patch.encode("utf-8", errors="surrogateescape")
                except (OSError, RuntimeError):
                    diff_bytes = b""
        truncated = len(diff_bytes) > MAX_READ_BYTES
        diff = diff_bytes[:MAX_READ_BYTES].decode("utf-8", errors="replace")

        review_meta = {}
        session_id = card.get("session_id") if isinstance(card.get("session_id"), str) else None
        if session_id is not None:
            try:
                source = SessionStore.open(self._sessions_root, session_id)
                try:
                    review_meta = source.read_meta().get("review", {})
                finally:
                    source.close()
            except (OSError, SessionError, AttributeError):
                review_meta = {}
        verdict = review_meta.get("verdict", "") if isinstance(review_meta, dict) else ""
        if verdict not in ("passed", "follow-ups", "running"):
            verdict = ""
        review_text = ""
        review_id = review_meta.get("session_id") if isinstance(review_meta, dict) else None
        if isinstance(review_id, str) and verdict != "running":
            try:
                review_store = SessionStore.open(self._sessions_root, review_id)
                try:
                    review_run = load_run(review_store)
                    review_text = next((
                        message.text for message in reversed(review_run.messages)
                        if message.role == Role.ASSISTANT and message.text.strip()
                    ), "")
                finally:
                    review_store.close()
            except (OSError, SessionError):
                review_text = ""
        follow_up_ids = [
            item["id"] for item in loop.get("cards", [])
            if card_id in item.get("after", []) and item.get("follow_up_of")
        ]
        return {
            "card": card,
            "spec": spec,
            "report": report,
            "diff": diff,
            "diff_truncated": truncated,
            "review": {"verdict": verdict, "text": review_text, "follow_ups": follow_up_ids},
            "session_id": session_id,
        }

    def recover_loops(self) -> None:
        store = LoopStore(self._repo_root)
        for loop in store.list():
            with self._lock:
                if loop.get("state") == "validating":
                    self._start_loop_validation(loop["id"])
                    continue
                if loop.get("state") != "running":
                    continue
                runner_pid = loop.get("runner_pid")
                if type(runner_pid) is int and runner_pid > 0:
                    try:
                        os.kill(runner_pid, 0)
                    except ProcessLookupError:
                        pass
                    except OSError:
                        continue
                    else:
                        continue
                changed = False
                changed_cards = []
                for card in loop.get("cards", []):
                    if card.get("state") not in ("running", "review"):
                        continue
                    session_id = card.get("session_id")
                    if isinstance(session_id, str) and session_id in self._active_by_session:
                        continue
                    if card.get("state") == "review" and isinstance(session_id, str):
                        try:
                            source = SessionStore.open(self._sessions_root, session_id)
                            try:
                                review_id = source.read_meta().get("review", {}).get("session_id")
                            finally:
                                source.close()
                        except (OSError, SessionError, AttributeError):
                            review_id = None
                        if isinstance(review_id, str) and review_id in self._active_by_session:
                            continue
                    if isinstance(session_id, str):
                        worktree = self._sessions_root / session_id / "worktree"
                        if worktree.exists():
                            remove_worktree(self._repo_root, worktree)
                    card.update(state="planned", session_id=None)
                    changed_cards.append(card["id"])
                    changed = True
                if changed:
                    store.save(loop)
                    for card_id in changed_cards:
                        self._loop_changed(loop, "card", card_id)
                self._advance_loop(loop["id"])

    def _finish_loop_plan(self, session_id: str, terminal_event: RunFinished | RunFailed | None) -> None:
        session = SessionStore.open(self._sessions_root, session_id)
        try:
            meta = session.read_meta()
            info = meta.get("spec_run")
        finally:
            session.close()
        if not isinstance(info, dict) or info.get("kind") != "loop_plan":
            return
        loop_id = info.get("loop_id")
        if not isinstance(loop_id, str):
            return
        store = LoopStore(self._repo_root)
        try:
            loop = store.load(loop_id)
        except (KeyError, ValueError):
            return
        if loop.get("state") != "planning":
            return
        reason = ""
        if isinstance(terminal_event, RunFailed):
            reason = f"planner failed: {terminal_event.error}"
        elif not isinstance(terminal_event, RunFinished) or terminal_event.stopped_reason != "final_response":
            stopped_reason = terminal_event.stopped_reason if isinstance(terminal_event, RunFinished) else "unknown"
            reason = f"planner stopped: {stopped_reason}"
        else:
            plan_path = store.path(loop_id, "plan.json")
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                reason = "plan.json is missing"
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                reason = f"plan.json is invalid: {exc}"
            else:
                reason = self._loop_plan_problem(store, loop_id, plan)
                if not reason:
                    cards = plan["cards"]
                    loop["cards"] = [
                        {
                            "id": card["id"],
                            "title": card["title"],
                            "spec": card["spec"],
                            "report": "reports/" + card["id"] + "-" + (re.sub(
                                r"[^a-z0-9]+", "-", card["title"].lower(),
                            ).strip("-")[:40].rstrip("-") or "spec") + "-report.md",
                            "lane": card["lane"],
                            "after": card["after"],
                            "state": "planned",
                            "rounds": 0,
                            "follow_up_of": None,
                            "session_id": None,
                        }
                        for card in cards
                    ]
                    lane_ids = sorted({card["lane"] for card in cards})
                    loop["lanes"] = [
                        {"id": lane_id, "cards": [card["id"] for card in cards if card["lane"] == lane_id]}
                        for lane_id in lane_ids
                    ]
                    store.set_state(loop, "awaiting_plan")
                    self._loop_changed(loop, "planned")
                    return
        loop["reason"] = reason
        store.set_state(loop, "needs_attention", reason)
        self._loop_changed(loop, "state")

    @staticmethod
    def _loop_plan_problem(store: LoopStore, loop_id: str, plan: object) -> str:
        if not isinstance(plan, dict) or not isinstance(plan.get("cards"), list):
            return "plan.json must contain a cards list"
        cards = plan["cards"]
        if not 1 <= len(cards) <= 8:
            return f"plan must contain 1 to 8 cards; found {len(cards)}"
        ids = []
        for card in cards:
            if not isinstance(card, dict):
                return "plan card must be an object"
            card_id = card.get("id")
            if not isinstance(card_id, str) or card_id not in set("abcdefgh"):
                return f"invalid card id: {card_id!r}"
            if card_id in ids:
                return f"duplicate card id: {card_id}"
            ids.append(card_id)
            if not isinstance(card.get("title"), str) or not card["title"].strip():
                return f"card {card_id} has no title"
            if not isinstance(card.get("spec"), str):
                return f"card {card_id} has an invalid spec path"
            if type(card.get("lane")) is not int or not 1 <= card["lane"] <= 3:
                return f"card {card_id} has an invalid lane"
            if not isinstance(card.get("after"), list) or not all(
                isinstance(item, str) for item in card["after"]
            ):
                return f"card {card_id} has an invalid after list"
        known_ids = set(ids)
        graph = {card["id"]: card["after"] for card in cards}
        for card in cards:
            card_id = card["id"]
            if any(parent not in known_ids for parent in card["after"]):
                return f"card {card_id} has an unknown after id"
            try:
                path = store.path(loop_id, card["spec"])
            except (KeyError, ValueError, OSError):
                return f"card {card_id} spec path escapes the loop folder"
            specs_root = store.path(loop_id, "specs")
            if not path.is_relative_to(specs_root) or not path.is_file():
                return f"card {card_id} spec file is missing or outside loop specs"
            from symphonai_host.spec_run import spec_validation
            if not spec_validation(path.read_text(encoding="utf-8")):
                return f"card {card_id} has no Validation commands"
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(card_id: str) -> bool:
            if card_id in visiting:
                return False
            if card_id in visited:
                return True
            visiting.add(card_id)
            if not all(visit(parent) for parent in graph[card_id]):
                return False
            visiting.remove(card_id)
            visited.add(card_id)
            return True

        if not all(visit(card_id) for card_id in ids):
            return "card dependencies contain a cycle"
        return ""

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
                self._finish_spec_run(session_id, status)
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
            self._refresh_sleep_prevention()
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

    def _refresh_sleep_prevention(self) -> None:
        global _sleep_keeper_process
        active = (
            not self._sleep_keeper_released
            and bool(self._active_by_session or self._goal_checks_by_session)
        )
        supported = sys.platform == "darwin" and os.path.isfile("/usr/bin/caffeinate")
        with _sleep_keeper_lock:
            if active and supported:
                _sleep_keeper_holders.add(id(self))
            else:
                _sleep_keeper_holders.discard(id(self))
            if not supported:
                return
            if _sleep_keeper_holders:
                if _sleep_keeper_process is None or _sleep_keeper_process.poll() is not None:
                    try:
                        _sleep_keeper_process = subprocess.Popen(
                            ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except OSError:
                        _sleep_keeper_process = None
            elif _sleep_keeper_process is not None:
                process = _sleep_keeper_process
                _sleep_keeper_process = None
                try:
                    process.terminate()
                    process.wait(timeout=1)
                except (OSError, subprocess.SubprocessError):
                    try:
                        process.kill()
                    except OSError:
                        pass

    def _release_sleep_prevention(self) -> None:
        self._sleep_keeper_released = True
        self._refresh_sleep_prevention()

    def _start_for_session(self, session_id: str, prompt: str) -> str:
        with self._lock:
            conversation = self._open_conversations.get(session_id)
            if conversation is None:
                raise SessionError(f"session {session_id!r} is not open")
            previous = (
                self._conversation, self._policy, self.approvals, self._active,
                self._goal, self._goal_session_id, self._goal_check,
                self._goal_check_thread,
            )
            self._conversation = conversation
            self._policy = self._policy_by_session[session_id]
            self.approvals = self._approvals_by_session[session_id]
            self._active = self._active_by_session.get(session_id)
            self._goal = self._goals_by_session.get(session_id)
            self._goal_session_id = session_id if self._goal is not None else None
            self._goal_check = self._goal_checks_by_session.get(session_id)
            try:
                return self.start(prompt, _goal_round=True)
            finally:
                (
                    self._conversation, self._policy, self.approvals, self._active,
                    self._goal, self._goal_session_id, self._goal_check,
                    self._goal_check_thread,
                ) = previous

    def _publish_session(self, event: Event, session_id: str) -> None:
        self._remember_event_session(event, session_id)
        self._broker.publish(event)

    @staticmethod
    def _checkpoint_replay_data(
        records: list[dict], subagent_records: list[list[dict]] | None = None,
    ) -> tuple[dict[str, str], dict[str, list[dict[str, str | int]]]]:
        keys: dict[str, str] = {}
        root_run_keys: dict[str, str] = {}
        edits: dict[str, list[dict[str, str | int]]] = {}
        pending_key: str | None = None
        current_root_run: str | None = None
        for record in records:
            if record.get("type") == "run_started":
                data = record.get("data")
                if isinstance(data, dict) and data.get("parent_run_id") is None:
                    current_root_run = record.get("run_id")
            elif record.get("type") == "checkpoint":
                key = record.get("data", {}).get("key")
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
    def _replay_tool_call(call, result=None) -> dict:
        replay = {"id": call.id, "name": call.name}
        if call.name == "dispatch_subagent":
            for key in ("subagent_name", "task"):
                value = call.arguments.get(key)
                if isinstance(value, str):
                    replay[key] = value
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
            "write_file", "edit_file", "multi_edit_file",
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
    def _replay_tool_calls(message: Message, results: Mapping[str, object]) -> list[dict]:
        return [
            HostRun._replay_tool_call(call, results.get(call.id))
            for call in message.tool_calls
        ]

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
            for broker in self._approvals_by_session.values():
                for approval in broker.pending():
                    if approval.session_id is not None:
                        activity[approval.session_id] = "waiting"
            for session_id in self._goal_checks_by_session:
                activity[session_id] = "working"
            return activity

    def has_work_in_flight(self) -> bool:
        with self._lock:
            conversational_work = bool(
                self._active_by_session
                or self._goal_checks_by_session
                or self._loop_validation_cancel
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
        return any(broker.resolve(approval_id, **decision) for broker in brokers)

    def start(
        self,
        prompt: str,
        *,
        attachments: tuple[ImageBlock | DocumentBlock, ...] = (),
        _goal_round: bool = False,
        _new_goal: Goal | None = None,
        _session_meta: dict | None = None,
        _title: str | None = None,
    ) -> str:
        with self._lock:
            if self._user_command is not None:
                raise CommandActiveError("wait for the command to finish")
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            if len(self._active_by_session) >= 4:
                raise RunActiveError("4 conversations are already running")
            run_id = new_id("run")
            cancel = CancellationToken()
            if self._conversation is None:
                if self._provider is None and self._provider_factory is not None:
                    self._provider = self._provider_factory(None, None, None)
                    if self._provider is not None:
                        self._provider_choice = {"name": self._provider.name}
                if self._provider is None:
                    raise ProviderSelectionError("no configured provider; add an API key in Settings")
                self._policy = replace(self._policy)
                spec_info = None if not _session_meta else _session_meta.get("spec_run")
                if isinstance(spec_info, dict) and spec_info.get("kind") == "implement":
                    worktree_admin = self._sessions_root / run_id / "worktree"
                    start_ref = spec_info.get("start_ref", "HEAD")
                    if not isinstance(start_ref, str) or not start_ref:
                        raise ValueError("spec run start_ref must be a non-blank string")
                    worktree_root = create_worktree(self._repo_root, worktree_admin, start=start_ref)
                    self._policy = self._policy.rerooted(worktree_root)
                    self._policy.mode = "allow"
                self.approvals = ApprovalBroker(
                    self._publish_approval_callback,
                    timeout=self._approval_timeout,
                    session_id=run_id,
                )
                self._policy.approval_callback = self.approvals.callback
                session = SessionStore(
                    self._sessions_root,
                    run_id,
                    repo_root=self._repo_root if _session_meta and "spec_run" in _session_meta else self._policy.repo_root,
                    events=fan_out(
                        lambda event, sid=run_id: self._publish_session(event, sid),
                        self._hooks,
                    ),
                )
                if _session_meta:
                    meta = session.read_meta()
                    meta.update(_session_meta)
                    session.write_meta(meta)
                try:
                    leader = self._new_leader(session)
                except Exception:
                    session.close()
                    raise
                seeded = []
                if self._system_prompt:
                    seeded.append(Message(role=Role.SYSTEM, content=self._system_prompt))
                instruction_policy = self._policy
                if _session_meta and "spec_run" in _session_meta:
                    instruction_policy = replace(self._policy, repo_root=self._repo_root)
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
                            repo_root=self._policy.repo_root,
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
                meta = session.read_meta()
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
                session.write_meta(meta)
                self._conversation = (leader, session)
                self._open_conversations[session.run_id] = self._conversation
                self._policy_by_session[session.run_id] = self._policy
                self._approvals_by_session[session.run_id] = self.approvals
                self._session_by_root_agent[leader.agent_ref.agent_id] = session.run_id
                self._context_report = None
                self._usage_by_agent.clear()
                self._context_by_session[session.run_id] = None
                self._usage_by_session[session.run_id] = {}
            else:
                leader, _ = self._conversation
            session_id = self._conversation[1].run_id
            chat_message: ContentInput = prompt
            if attachments:
                chat_message = ([TextBlock(prompt)] if prompt else []) + list(attachments)
            previous_rounds = None
            if _new_goal is not None:
                _new_goal.rounds = 1
                self._goal = _new_goal
                self._goal_session_id = session_id
                self._goals_by_session[session_id] = _new_goal
                self._save_goal(session_id, _new_goal)
                self._goal_event("set", _new_goal, agent_id=leader.agent_ref.agent_id)
            elif _goal_round and self._goal is not None and self._goal_session_id == session_id:
                previous_rounds = self._goal.rounds
                self._goal.rounds += 1
                self._goals_by_session[session_id] = self._goal
                self._save_goal(session_id, self._goal)
            elif not _goal_round:
                context = self._goal_checks_by_session.get(session_id)
                if context is not None:
                    context.interrupted = "interrupted"
            is_goal_round = _goal_round or _new_goal is not None
            if self._pending_user_context:
                for context in self._pending_user_context:
                    leader.add_context(context)
                self._pending_user_context.clear()
            thread = threading.Thread(
                target=self._run,
                args=(run_id, leader, chat_message, cancel, is_goal_round, session_id),
                name=f"symphonai-host-{run_id}",
                daemon=True,
            )
            self._active = _ActiveRun(
                run_id, cancel, thread, leader.agent_ref.agent_id, goal_round=is_goal_round,
            )
            self._active_by_session[session_id] = self._active
            self._refresh_sleep_prevention()
            try:
                thread.start()
            except Exception:
                self._active = None
                self._active_by_session.pop(session_id, None)
                self._refresh_sleep_prevention()
                if _new_goal is not None:
                    self._save_goal(session_id, None)
                    self._goals_by_session.pop(session_id, None)
                    self._goal = None
                    self._goal_session_id = None
                    self._goal_event("clear", None, agent_id=leader.agent_ref.agent_id, session_id=session_id)
                elif previous_rounds is not None and self._goal is not None:
                    self._goal.rounds = previous_rounds
                    self._save_goal(session_id, self._goal)
                raise
            return run_id

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

    def _new_leader(self, session: SessionStore, *, jobs: JobManager | None = None) -> Leader:
        if jobs is None:
            jobs = JobManager(
                self._policy.shell_output_limit_chars,
                owner="agent",
                on_change=lambda job: self._job_changed(job, session.run_id),
            )
        checkpoints = CheckpointStore(
            session.directory / "checkpoints",
            self._policy.repo_root,
        )
        result_store = ToolResultStore(
            directory=session.tool_results_directory,
            fallback_directories=tool_result_search_path(session),
        )
        if (
            self._restricted_tool_names is not None
            and "read_tool_result" not in self._restricted_tool_names
        ):
            result_store = None
        skills = None if self._extensions is None else self._extensions.skills
        roster = builtin_subagent_specs(
            self._provider,
            self._policy,
            self._search_backend,
            skills=skills,
            lsp=self._lsp,
            jobs=jobs,
            model_override=self._model,
            use_routine_model=self._model is None,
        )
        if self._extensions is not None:
            roster.update(self._extensions.agents)
        if self._restricted_tool_names is not None:
            available_extra = set(self._mcp_tools or {})
            restricted_roster = {}
            for name, spec in roster.items():
                if name == "leader":
                    restricted_roster[name] = spec
                    continue
                tool_names = (
                    tuple(
                        tool for tool in standard_tool_registry(
                            result_store=(result_store if "read_tool_result" in self._restricted_tool_names else None),
                            search_backend=self._search_backend,
                            skills=skills,
                            lsp=self._lsp,
                            jobs=jobs,
                        )
                        if tool in self._restricted_tool_names
                    ) if spec.tool_names is None
                    else tuple(tool for tool in spec.tool_names if tool in self._restricted_tool_names)
                )
                mcp_names = (
                    tuple(tool for tool in available_extra if tool in self._restricted_tool_names)
                    if spec.mcp_tool_names is None
                    else tuple(tool for tool in spec.mcp_tool_names if tool in self._restricted_tool_names)
                )
                if tool_names == () and not mcp_names:
                    continue
                memory = spec.memory
                if not {"remember", "forget_memory"}.issubset(self._restricted_tool_names):
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
        meta = session.read_meta()
        role = meta.get("spec_run", {}).get("role") if isinstance(meta.get("spec_run"), dict) else None
        if isinstance(role, str) and role in roster:
            roster["leader"] = roster[role]
        defined_leader = roster.get("leader")
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
                provider._symphonai_fallback_models = tuple(
                    () if self._extensions is None else
                    self._extensions.config.get(f"fallback.{provider.name}", ())
                )
            return provider

        configure_fallback(leader_provider)
        conversation_approvals = self.approvals
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
        approval_broker = self.approvals

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
                repo_root=str(self._policy.repo_root),
                base_prompt=self._base_prompt,
                subagent_base_prompt=self._subagent_base_prompt,
                max_leader_turns=self._max_turns,
                leader_budget=self._leader_budget,
                chat_token_budget=self._chat_token_budget,
                chat_recent_turns=self._chat_recent_turns,
                permission_mode=self._policy.mode,
                approval_callback=self.approvals.callback,
                events=lambda event: self._publish_active(event),
                extensions=self._extensions,
                stream=True,
                result_store=result_store,
                search_backend=self._search_backend,
                extra_tools=self._mcp_tools,
                leader_policy=self._policy,
                leader_model=leader_model,
                leader_effort=self._effort,
                model_summary=True,
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
                leader_tools=goal_tools(
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
        if self._restricted_tool_names is not None:
            selected = set(self._restricted_tool_names)
            leader._agent._tools = {
                name: tool for name, tool in leader._agent._tools.items()
                if name in selected
            }
            leader._agent._tool_schemas = [
                schema for schema in leader._agent._tool_schemas
                if _schema_tool_name(schema) in selected
            ]
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

    def open_session(self, run_id: str) -> dict:
        """Load and replay a finished transcript without ever rewriting it."""
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
                metadata = reader.read_meta()
                checkpoint_keys, replay_edits = self._checkpoint_replay_data(
                    records, subagent_records
                )
                replay_turns = self._turn_replay_data(records)
                undone_keys = metadata.get("undone_checkpoint_keys", [])
                if not isinstance(undone_keys, list):
                    undone_keys = []
                choice = metadata.get("provider_choice")
                provider_state_reset = metadata.get("provider_state_reset") is True
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
                for message, record_id in zip(loaded.messages, loaded.record_ids):
                    checkpoint_key = checkpoint_keys.get(record_id)
                    self._broker.publish(ForkableHistoryMessage(
                        role=message.role.value,
                        text=message.text,
                        tool_calls=self._replay_tool_calls(message, tool_results),
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
                self._close_idle_conversations_locked()
                return {
                    "run_id": loaded.run_id, "state": diagnosis.state.value,
                    "replayed": len(loaded.record_ids), "repaired_ids": repaired_ids,
                    "dropped_bytes": loaded.dropped_bytes,
                }
            previous_provider = (
                self._provider,
                self._model,
                self._effort,
                self._provider_choice,
            )
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
                self._policy = replace(self._policy)
                self._policy.mode = self._starting_mode
                self.approvals = ApprovalBroker(
                    self._publish_approval_callback,
                    timeout=self._approval_timeout,
                    session_id=run_id,
                )
                self._policy.approval_callback = self.approvals.callback
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
                    metadata["goal"] = goal.payload()
                    store.write_meta(metadata)
            except Exception:
                self._provider, self._model, self._effort, self._provider_choice = previous_provider
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
            self._usage_by_agent = {}
            self._context_by_session[run_id] = None
            self._usage_by_session[run_id] = {}
            self._plans_by_session.pop(run_id, None)
            if saved_plan is not None:
                self._plans_by_session[run_id] = saved_plan
            if goal is not None:
                self._goals_by_session[run_id] = goal
            goal_agent_id = leader.agent_ref.agent_id
            self._close_idle_conversations_locked()
        if reopened_goal:
            self._goal_event("pause", goal, agent_id=goal_agent_id)
        for message, record_id in zip(loaded.messages, loaded.record_ids):
            checkpoint_key = checkpoint_keys.get(record_id)
            self._broker.publish(ForkableHistoryMessage(
                role=message.role.value,
                text=message.text,
                tool_calls=self._replay_tool_calls(message, tool_results),
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
        return {
            "run_id": loaded.run_id,
            "state": diagnosis.state.value,
            "replayed": len(loaded.record_ids),
            "repaired_ids": repaired_ids,
            "dropped_bytes": loaded.dropped_bytes,
        }

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
            replay_turns = self._turn_replay_data(records)
            messages = []
            for message, record_id in zip(loaded.messages, loaded.record_ids):
                turn = replay_turns.get(record_id, {})
                messages.append(HistoryMessage(
                    role=message.role.value,
                    text=message.text,
                    tool_calls=self._replay_tool_calls(message, tool_results),
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
                    meta = destination.read_meta()
                    meta["title"] = source_meta.get("title")
                    if "provider_choice" in source_meta:
                        meta["provider_choice"] = source_meta["provider_choice"]
                    destination.write_meta(meta)
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
            if session_id is not None and session_id in self._active_by_session:
                self._conversation = None

                self._active = None
                self._goal = None
                self._goal_session_id = None
                self._goal_check = None
                self._goal_check_thread = None
                self._policy = replace(self._policy)
                self._policy.mode = self._starting_mode
                self._policy.approval_callback = self.approvals.callback
                self._context_report = None
                self._usage_by_agent.clear()
                return
            context = (
                None if session_id is None else self._goal_checks_by_session.get(session_id)
            )
            if context is not None:
                context.interrupted = "interrupted"
            self._goal = None
            self._goal_session_id = None
            self._conversation = None
            self._active = None
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

    def close(self) -> None:
        self._closing = True
        self._release_sleep_prevention()
        with self._lock:
            active_runs = tuple(self._active_by_session.values())
            approval_brokers = tuple(dict.fromkeys(self._approvals_by_session.values()))
            validations = tuple(self._loop_validation_cancel.values())
            validation_processes = tuple(self._loop_validation_processes.values())
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
            goal_check_thread = self._goal_check_thread
        for active_run in active:
            active_run.thread.join(timeout=2)
        if goal_check_thread is not None and goal_check_thread is not threading.current_thread():
            goal_check_thread.join(timeout=2)
        with self._lock:
            conversations = tuple(self._open_conversations.values())
            self._open_conversations.clear()
        if self._lsp is not None:
            self._lsp.close()
        for leader, session in conversations:
            leader.close()
            session.close()
        self._person_jobs.close()

    def stop(self) -> None:
        with self._lock:
            active = self._active
            command = self._user_command
            if command is not None:
                command.stop_requested = True
                process = command.process
            else:
                process = None
            session_id = None if self._conversation is None else self._conversation[1].run_id
            goal_check = None if session_id is None else self._goal_checks_by_session.get(session_id)
        if active is not None:
            active.cancel.cancel()
        if process is not None:
            self._kill_user_command(process)
        if goal_check is not None:
            goal_check.interrupted = "cancelled"
            goal_check.cancel.set()
        self.approvals.cancel_all(reason="stopped")

    def control_agent(self, agent_id: str, action: str, text: str | None = None) -> dict:
        """Control one agent in the active conversation run."""
        with self._lock:
            active = self._active
            conversation = self._conversation
            if active is None:
                raise RunActiveError("none")
            if conversation is None:
                raise AgentControlError(f"agent {agent_id!r} is not running", status=404)
            leader = conversation[0]
            if agent_id == active.root_agent_id and action == "stop":
                self.stop()
                return {"agent_id": agent_id, "state": "stopping"}
        try:
            state = leader.control_agent(agent_id, action, text)
        except AgentControlError:
            raise
        return {"agent_id": agent_id, "state": state}

    def steer(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("steer text must be non-blank")
        with self._lock:
            active = self._active
            conversation = self._conversation
            if active is None or conversation is None:
                raise RunActiveError("no turn is running")
            leader = conversation[0]
            agent_id = active.root_agent_id
        try:
            leader.control_agent(agent_id, "redirect", text)
        except AgentControlError as exc:
            raise RunActiveError("no turn is running") from exc

    def run_user_command(self, command: str) -> dict:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be non-blank")
        with self._lock:
            if self._active is not None:
                raise RunActiveError("wait for the turn to finish")
            if self._user_command is not None:
                raise CommandActiveError("a command is already running")
            conversation = self._conversation
            repo_root = self._repo_root if conversation is None else self._policy.repo_root
            running = _UserCommand()
            self._user_command = running
        process = None
        try:
            process = subprocess.Popen(
                [shell_executable(), "-c", command],
                cwd=repo_root,
                env=shell_environment(repo_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,
            )
            with self._lock:
                running.process = process
                should_stop = running.stop_requested
            if should_stop:
                self._kill_user_command(process)
            deadline = time.monotonic() + 120
            timed_out = False
            stopped = False
            while True:
                with self._lock:
                    stopped = running.stop_requested
                if stopped:
                    self._kill_user_command(process)
                    try:
                        output, _ = process.communicate(timeout=1)
                    except subprocess.TimeoutExpired:
                        output = ""
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    self._kill_user_command(process)
                    try:
                        output, _ = process.communicate(timeout=1)
                    except subprocess.TimeoutExpired:
                        output = ""
                    break
                try:
                    output, _ = process.communicate(timeout=min(0.05, remaining))
                    with self._lock:
                        stopped = running.stop_requested
                    break
                except subprocess.TimeoutExpired:
                    continue
            output = output or ""
            exit_code = None if timed_out or stopped else process.returncode
            limit = self._policy.shell_output_limit_chars
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
                    self._user_command = None

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
        """Force compact the current conversation and publish its new usage."""
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            if self._conversation is None:
                raise NoConversationError("no conversation to compact")
            leader = self._conversation[0]
            result, summary_usage = leader.force_compact_chat(instructions)
            agent_id = leader.agent_ref.agent_id
            name, current_usage = self._usage_by_agent.get(
                agent_id, (leader.agent_ref.name, {})
            )
            self._usage_by_agent[agent_id] = (
                name,
                self._merge_usage(current_usage, summary_usage),
            )
            self._context_report = account_context(
                leader._chat_messages,
                budget=leader.chat_token_budget,
            )
            session_id = self._conversation[1].run_id
            self._context_by_session[session_id] = self._context_report
            self._usage_by_session[session_id] = dict(self._usage_by_agent)
            return result

    def conversation_stats(self) -> dict | None:
        with self._lock:
            conversation = self._conversation
            if conversation is None:
                return None
            session_id = conversation[1].run_id
            graph = conversation[0].run_graph()
            leader = conversation[0]
            active = self._active_by_session.get(session_id)
            active_root_agent_id = active.root_agent_id if active is not None else None
            active_runtime_run_id = active.runtime_run_id if active is not None else None
            selection = {
                "provider": leader._config.leader_provider.name,
                "model": leader._leader_spec.model.model,
                "effort": leader._leader_spec.model.effort,
            }
            started_at = {}
            try:
                paths = sorted(conversation[1].directory.glob("*.jsonl"))
            except OSError:
                paths = []
            for path in paths:
                try:
                    records, _ = read_records(path)
                except Exception:
                    continue
                for record in records:
                    if (
                        record.get("type") == "run_started"
                        and isinstance(record.get("ts"), str)
                    ):
                        started_at.setdefault(record.get("run_id"), record["ts"])
            report = self._context_by_session.get(session_id, self._context_report)
            policy = self._policy_by_session.get(session_id, self._policy)
            mode = policy.mode
            goal_payload = (
                self._goal.payload()
                if self._goal is not None and self._goal_session_id == conversation[1].run_id
                else None
            )
            usage_by_agent = {
                agent_id: (name, dict(by_model))
                for agent_id, (name, by_model) in self._usage_by_session.get(
                    session_id, self._usage_by_agent,
                ).items()
            }

        def usage_fields(by_model: Mapping[str, UsageTotals]) -> dict:
            totals = UsageTotals()
            for usage in by_model.values():
                totals = totals.merged(usage)
            fields = {
                "input_tokens": totals.input_tokens,
                "output_tokens": totals.output_tokens,
                "calls": totals.calls,
                "total_tokens": totals.total_tokens,
                "cache_read_tokens": totals.cache_read_tokens,
                "cache_write_tokens": totals.cache_write_tokens,
            }
            cost = total_cost(by_model, self._price_table)
            if cost is not None and self._price_table is not None:
                fields["cost"] = {
                    "amount": str(cost),
                    "currency": self._price_table.currency,
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
            if has_usage and report is not None:
                folded.update(usage_fields(by_model))
            folded_agents.append(folded)
        agents = folded_agents

        result = {
            "session_id": conversation[1].run_id,
            "agents": agents,
            "mode": mode,
            "goal": goal_payload,
            "plan": self._plans_by_session.get(conversation[1].run_id),
            "instruction_warnings": self._instruction_warnings_by_session.get(
                conversation[1].run_id, []
            ),
            **selection,
        }
        if report is None:
            return result
        all_models: dict[str, UsageTotals] = {}
        for _, by_model in usage_by_agent.values():
            all_models = self._merge_usage(all_models, by_model)
        result.update({
            "context": {
                "used_tokens": report.total_tokens,
                "budget_tokens": report.budget,
                "remaining_tokens": report.remaining_tokens,
                "window_tokens": context_window_for_model(
                    leader._config.leader_provider.wire_format,
                    leader._leader_spec.model.model,
                ),
                "by_source": {
                    source.value: tokens for source, tokens in report.by_source().items()
                },
            },
            "usage": usage_fields(all_models),
        })
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

    def commit_spec(self, session_id: str, message: str) -> dict:
        if not message.strip():
            raise ValueError("commit message must not be blank")
        source = SessionStore.open(self._sessions_root, session_id)
        try:
            meta = source.read_meta()
            info = meta.get("spec_run")
            review = meta.get("review")
        finally:
            source.close()
        if not isinstance(info, dict) or info.get("kind") != "implement":
            raise KeyError(session_id)
        if not isinstance(review, dict) or review.get("verdict") not in ("passed", "follow-ups"):
            raise RunActiveError("spec run has no passing review")
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=self._repo_root,
            capture_output=True, check=False,
        )
        if staged.returncode != 0:
            raise WorktreeApplyConflict("the main tree has staged changes")
        worktree = self._sessions_root / session_id / "worktree"
        diff = worktree_diff(worktree)
        if not diff.files:
            return {"commit": "", "paths": []}
        dirty = []
        for path in diff.files:
            unstaged = subprocess.run(
                ["git", "diff", "--quiet", "--", path], cwd=self._repo_root,
                capture_output=True, check=False,
            )
            main_status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all", "--", path],
                cwd=self._repo_root, capture_output=True, check=False,
            )
            worktree_status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all", "--", path],
                cwd=worktree, capture_output=True, check=False,
            )
            main_untracked = main_status.stdout.decode("utf-8", errors="replace").splitlines()
            worktree_untracked = worktree_status.stdout.decode("utf-8", errors="replace").splitlines()
            adds_path = any(line.startswith(("?? ", "A ", " A")) for line in worktree_untracked)
            conflicts_with_untracked = adds_path and any(line.startswith("?? ") for line in main_untracked)
            if unstaged.returncode != 0 or conflicts_with_untracked:
                dirty.append(path)
        if dirty:
            raise WorktreeApplyConflict(f"uncommitted changes in: {', '.join(dirty)}")
        checkpoints = CheckpointStore(
            self._sessions_root / str(review.get("session_id")) / "checkpoints",
            self._repo_root,
        )
        applied = self.apply_worktree(
            f"spec {session_id}", directory=worktree,
            checkpoints_override=checkpoints,
        )
        paths = applied["applied"]
        added = subprocess.run(
            ["git", "add", "--", *paths], cwd=self._repo_root,
            capture_output=True, check=False,
        )
        if added.returncode:
            raise WorktreeApplyConflict(added.stderr.decode("utf-8", errors="replace").strip())
        committed = subprocess.run(
            ["git", "commit", "-m", message], cwd=self._repo_root,
            capture_output=True, check=False,
        )
        if committed.returncode:
            subprocess.run(["git", "reset", "--", *paths], cwd=self._repo_root, capture_output=True, check=False)
            detail = (committed.stderr or committed.stdout).decode("utf-8", errors="replace").strip()
            raise WorktreeApplyConflict(detail or "git commit failed")
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self._repo_root,
            capture_output=True, check=True,
        )
        sha = os.fsdecode(sha_result.stdout.strip())
        source = SessionStore.open(self._sessions_root, session_id)
        try:
            meta = source.read_meta()
            meta["committed"] = {"sha": sha, "message": message}
            source.write_meta(meta)
        finally:
            source.close()
        try:
            from symphonai_host.spec_run import mark_roadmap_spec_done
            mark_roadmap_spec_done(self._repo_root, str(info["spec"]))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return {"commit": sha, "paths": paths}

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
                metadata = session.read_meta()
                undone_keys = metadata.get("undone_checkpoint_keys", [])
                if not isinstance(undone_keys, list):
                    undone_keys = []
                first_index = checkpoints.keys().index(key)
                for reverted_key in checkpoints.keys()[first_index:]:
                    if reverted_key not in undone_keys:
                        undone_keys.append(reverted_key)
                metadata["undone_checkpoint_keys"] = undone_keys
                session.write_meta(metadata)
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
                self._refresh_sleep_prevention()
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
            self._refresh_sleep_prevention()
            self._goal_event("check", goal, run_id, context.agent_id)
            if goal.phase in ("complete", "blocked", "paused"):
                self._finish_spec_run(context.session_id, goal.phase)
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

    def _finish_spec_run(self, session_id: str, phase: str, *, error: str = "") -> None:
        goal_reason = ""
        try:
            store = SessionStore.open(self._sessions_root, session_id)
            try:
                meta = store.read_meta()
                info = meta.get("spec_run")
                if not isinstance(info, dict) or info.get("kind") != "implement":
                    return
                goal = meta.get("goal")
                if isinstance(goal, dict) and isinstance(goal.get("reason"), str):
                    goal_reason = goal["reason"]
                root = self._sessions_root / session_id / "worktree"
                report = info.get("report")
                copied = False
                if isinstance(report, str):
                    source = root / report
                    ignored = subprocess.run(
                        ["git", "check-ignore", "-q", "--", report], cwd=self._repo_root,
                        capture_output=True, check=False,
                    ).returncode == 0
                    if source.is_file() and ignored and ".env" not in Path(report).parts:
                        target = self._repo_root / report
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                        copied = True
                info.update({
                    "state": {"complete": "finished", "blocked": "blocked", "paused": "stopped"}.get(phase, "stopped"),
                    "report_copied": copied,
                })
                meta["spec_run"] = info
                store.write_meta(meta)
            finally:
                store.close()
        except (OSError, SessionError, ValueError):
            return
        self._finish_loop_card_run(
            session_id,
            {"complete": "finished", "blocked": "blocked", "paused": "stopped"}.get(phase, "stopped"),
            error=error, goal_reason=goal_reason,
        )

    def _finish_spec_review(self, session_id: str, leader: Leader, terminal_event: RunFinished | RunFailed | None) -> None:
        from symphonai_host.spec_run import patch_digest, review_verdict

        review_store = SessionStore.open(self._sessions_root, session_id)
        try:
            review_meta = review_store.read_meta()
            review_info = review_meta.get("spec_run")
        finally:
            review_store.close()
        if not isinstance(review_info, dict) or review_info.get("kind") != "review":
            return
        source_id = review_info.get("of")
        if not isinstance(source_id, str):
            return
        source = SessionStore.open(self._sessions_root, source_id)
        try:
            source_meta = source.read_meta()
            original = source_meta.get("spec_run")
            if not isinstance(original, dict):
                return
            worktree = Path(review_info.get("worktree_path", ""))
            current = worktree_diff(worktree)
            answer = next((message.text for message in reversed(leader._chat_messages) if message.role == Role.ASSISTANT and message.text.strip()), "")
            verdict, follow_ups = review_verdict(answer)
            baseline_files = review_info.get("baseline_files", {})
            unchanged = isinstance(baseline_files, dict) and all(
                (
                    (worktree / name).is_file()
                    and hashlib.sha256((worktree / name).read_bytes()).hexdigest() == digest
                    if digest is not None else not (worktree / name).exists()
                )
                for name, digest in baseline_files.items()
            )
            allowed_additions = set(follow_ups) if verdict == "follow-ups" else set()
            additions = set(current.files) - set(baseline_files if isinstance(baseline_files, dict) else {})
            if review_info.get("loop_id"):
                loop_follow_ups = {
                    value for value in follow_ups
                    if (worktree / value).is_file()
                }
                unchanged = unchanged and loop_follow_ups == set(follow_ups)
            unchanged = unchanged and additions <= allowed_additions
            if not unchanged:
                verdict = "tree-changed"
                follow_ups = []
            copied: list[str] = []
            not_copied: list[str] = []
            spec_directory = Path(str(original.get("spec", ""))).parent
            if verdict == "follow-ups":
                for value in follow_ups:
                    relative = Path(value)
                    if relative.is_absolute() or ".." in relative.parts or not relative.as_posix().startswith(spec_directory.as_posix() + "/"):
                        not_copied.append(value)
                        continue
                    source_file = worktree / relative
                    target_file = self._repo_root / relative
                    if not source_file.is_file() or target_file.exists():
                        not_copied.append(value)
                        continue
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, target_file)
                    copied.append(value)
            review = {
                "session_id": session_id, "verdict": verdict,
                "follow_ups": copied, "not_copied": not_copied,
            }
            if isinstance(terminal_event, RunFinished):
                review["stopped_reason"] = terminal_event.stopped_reason
            elif isinstance(terminal_event, RunFailed):
                review["error"] = terminal_event.error
            source_meta["review"] = review
            source.write_meta(source_meta)
        finally:
            source.close()
        if isinstance(review_info, dict):
            self._finish_loop_card_review(review_info)

    def _finish_spec_plan(self, session_id: str, terminal_event: RunFinished | RunFailed | None) -> None:
        from symphonai_host.spec_run import bind_roadmap_item

        store = SessionStore.open(self._sessions_root, session_id)
        try:
            meta = store.read_meta()
            info = meta.get("spec_run")
            if not isinstance(info, dict) or info.get("kind") != "plan":
                return
            phase = str(info.get("phase", ""))
            baseline = set(info.get("baseline_specs", []))
            phase_root = self._repo_root / "specs" / phase
            created = sorted(
                path.relative_to(self._repo_root).as_posix()
                for path in phase_root.rglob("*.md")
                if path.is_file()
                and path.relative_to(self._repo_root).as_posix() not in baseline
                and not path.name.endswith("-PLAN.md")
                and "specs/report/" not in path.relative_to(self._repo_root).as_posix()
            ) if phase_root.is_dir() else []
            bound = created if len(created) == 1 else []
            if bound:
                try:
                    bind_roadmap_item(self._repo_root, phase, int(info["item"]), bound[0])
                except (OSError, ValueError, KeyError, TypeError):
                    bound = []
            info.update({"created": created, "bound": bound[0] if bound else None})
            if isinstance(terminal_event, RunFinished) and terminal_event.stopped_reason == "final_response":
                info["state"] = "finished"
                info["stopped_reason"] = terminal_event.stopped_reason
            elif isinstance(terminal_event, RunFailed):
                info["state"] = "failed"
                info["error"] = terminal_event.error
            else:
                info["state"] = "stopped"
                if isinstance(terminal_event, RunFinished):
                    info["stopped_reason"] = terminal_event.stopped_reason
            meta["spec_run"] = info
            store.write_meta(meta)
        finally:
            store.close()

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
                if (
                    active_to_finish is not None
                    and self._active_by_session.get(session_id) is active_to_finish
                ):
                    self._active_by_session.pop(session_id, None)
                    self._refresh_sleep_prevention()
                    if self._conversation is not None and self._conversation[1].run_id == session_id:
                        self._active = None
                    if terminal_event is not None:
                        self._publish(run_id, terminal_event, session_id=session_id)
                    for event in active_to_finish.deferred_memory_events:
                        event_run_id = getattr(event, "run_id", run_id)
                        self._publish(event_run_id, event, session_id=session_id)
                    active_to_finish.deferred_memory_events.clear()
                if self._closing and self._conversation is not None and self._conversation[1].run_id == session_id:
                    self._conversation = None
                    self._active = None
        if not goal_round:
            self._finish_spec_review(session_id, leader, terminal_event)
            self._finish_spec_plan(session_id, terminal_event)
            self._finish_loop_plan(session_id, terminal_event)
            with self._lock:
                self._close_idle_conversations_locked()
            if failure is not None:
                raise failure
            return
        next_prompt = None
        with self._lock:
            goal = self._goals_by_session.get(session_id)
            if goal is None or session_id is None or goal.phase != "active":
                self._close_idle_conversations_locked()
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
                    self._refresh_sleep_prevention()
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
                self._finish_spec_run(
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
            raise failure
        with self._lock:
            self._close_idle_conversations_locked()
