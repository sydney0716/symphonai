"""Threaded conversation runs owned by a SymphonAI host process."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
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
from symphonai_api.events import Event, RunFailed, RunFinished, RunStarted, SubagentSpawned, fan_out
from symphonai_api.extensions import Extensions
from symphonai_api.environment import capture_environment
from symphonai_api.identity import new_id
from symphonai_api.instructions import load_instructions
from symphonai_api.lsp import LspManager
from symphonai_api.leader import (
    AgentControlError, Leader, LeaderConfig, LeaderRunResult,
    builtin_subagent_specs,
)
from symphonai_api.model_table import context_window_for_model
from symphonai_api.models import ContentInput, DocumentBlock, ImageBlock, Message, Role, TextBlock
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import ModelProvider
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
from symphonai_api.worktree import remove_worktree, worktree_diff
from symphonai_api.worktree import create_worktree
from symphonai_api.web_search import HttpJsonSearchBackend, search_endpoint
from symphonai_host.broker import EventBroker
from symphonai_host.approvals import ApprovalBroker, PendingApproval
from symphonai_host.goal import Goal, GoalCheck, GoalChanged, goal_tools, run_check
from symphonai_host.protocol import HistoryMessage


class RunActiveError(RuntimeError):
    """A client attempted to start a second run while one is active."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            run_id if run_id == "4 conversations are already running"
            else f"run already active: {run_id}"
        )
        self.run_id = run_id


class WorktreeApplyConflict(RuntimeError):
    """A worktree patch no longer applies to the main repository."""


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


class HostRun:
    """Run one prompt at a time in a persistent Leader conversation."""

    def __init__(
        self,
        provider: ModelProvider | None,
        policy: PermissionPolicy,
        broker: EventBroker,
        *,
        system_prompt: str | None = None,
        working_dir: Path | None = None,
        max_turns: int | None = None,
        model: str | None = None,
        provider_factory: Callable[[str | None, str | None, str | None], ModelProvider | None] | None = None,
        publish_approval=None,
        approval_timeout: float = 300.0,
        sessions_root: Path | None = None,
        memory_root: Path | None = None,
        extensions: Extensions | None = None,
        mcp_tools: Mapping[str, LocalTool] | None = None,
        price_table: PriceTable | None = None,
        chat_token_budget: int | None = None,
        chat_recent_turns: int = DEFAULT_RECENT_TURNS,
        lsp: LspManager | None = None,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._repo_root = policy.repo_root
        self._base_policy = replace(policy)
        self._extensions = extensions
        permitted = self.permitted_modes()
        if not permitted:
            raise ModeSelectionError("agents.ceiling.modes must permit at least one mode")
        self._starting_mode = "ask" if "ask" in permitted else permitted[0]
        self._policy.mode = self._starting_mode
        self._broker = broker
        self._system_prompt = system_prompt
        if working_dir is None:
            current = Path.cwd().resolve()
            self._working_dir = current if current.is_relative_to(policy.repo_root) else policy.repo_root
        else:
            self._working_dir = Path(working_dir)
        self._max_turns = max_turns
        self._model = model
        self._effort: str | None = None
        self._provider_factory = provider_factory
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
            else extensions.hook_runner(cwd=policy.repo_root)
        )
        self._mcp_tools = mcp_tools
        self._lsp = lsp
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
        self._active_by_session: dict[str, _ActiveRun] = {}
        self._open_conversations: dict[str, tuple[Leader, SessionStore]] = {}
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
                    replacement = self._new_leader(session)
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
            check = () if not spec["validation"] else (
                "/bin/sh", "-c", "set -e\n" + "\n".join(spec["validation"]),
            )
            goal = Goal(objective, check, max_rounds=5)
            try:
                run_id = self.start(
                    objective, _new_goal=goal,
                    _session_meta={"spec_run": {
                        "spec": spec["path"], "report": spec["report"],
                        "kind": "implement", "worktree": "worktree", "role": "implementer",
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
            spec = parse_spec(self._repo_root / info["spec"], self._repo_root)
            report_path = self._repo_root / info["report"]
            if not report_path.is_file():
                report_path = worktree / info["report"]
            report = report_path.read_text(encoding="utf-8") if report_path.is_file() else "No report was written."
            patch = diff.patch[:MAX_READ_BYTES]
            prompt = (
                f"Review this spec:\n\n{spec['text']}\n\nReport:\n\n{report}\n\n"
                f"Patch:\n\n{patch}\n\nReview the implementation and report. You may write follow-up specs under {Path(info['spec']).parent}/. "
                "End the final answer with a last line `Verdict: pass` or `Verdict: follow-ups: <path>, <path>`."
            )
            previous = (self._conversation, self._policy, self._active)
            self._conversation = None
            self._active = None
            self._policy = replace(self._base_policy, repo_root=worktree)
            self._policy.allowed_write_scope = [worktree / "specs"]
            self._policy.mode = "allow"
            meta = {
                "spec": info["spec"], "report": info["report"], "kind": "review",
                "of": session_id, "worktree_path": str(worktree),
                "baseline": patch_digest(diff.patch), "baseline_files": baseline_files,
                "role": "reviewer",
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
                    worktree_root = create_worktree(self._repo_root, worktree_admin)
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
            try:
                thread.start()
            except Exception:
                self._active = None
                self._active_by_session.pop(session_id, None)
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

    def _memory_for(self, roster: Mapping[str, AgentSpec]) -> AgentMemory | None:
        if not any(spec.memory.enabled for spec in roster.values()):
            return None
        if not self._memory_open_attempted:
            self._memory_open_attempted = True
            try:
                self._memory = AgentMemory(self._memory_root)
            except OSError:
                self._memory = None
        return self._memory

    def _close_idle_conversations_locked(self) -> None:
        current_id = None if self._conversation is None else self._conversation[1].run_id
        for session_id, conversation in tuple(self._open_conversations.items()):
            if (
                session_id == current_id
                or session_id in self._active_by_session
                or session_id in self._goal_checks_by_session
            ):
                continue
            conversation[1].close()
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

    def _new_leader(self, session: SessionStore) -> Leader:
        checkpoints = CheckpointStore(
            session.directory / "checkpoints",
            self._policy.repo_root,
        )
        result_store = ToolResultStore(
            directory=session.tool_results_directory,
            fallback_directories=tool_result_search_path(session),
        )
        skills = None if self._extensions is None else self._extensions.skills
        roster = builtin_subagent_specs(
            self._provider,
            self._policy,
            self._search_backend,
            skills=skills,
            lsp=self._lsp,
        )
        if self._extensions is not None:
            roster.update(self._extensions.agents)
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
        if defined_leader is not None:
            selector = defined_leader.model
            if selector.provider != self._provider.name:
                if self._provider_factory is None:
                    raise ProviderSelectionError(f"leader provider {selector.provider!r} is unavailable")
                leader_provider = self._provider_factory(selector.provider, selector.model, None)
                if leader_provider is None:
                    raise ProviderSelectionError(f"leader provider {selector.provider!r} is unavailable")
                leader_model = selector.model
            elif selector.model is not None:
                leader_model = selector.model
        return Leader(
            LeaderConfig(
                leader_provider=leader_provider,
                subagent_provider=self._provider,
                repo_root=str(self._policy.repo_root),
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
                memory=self._memory_for(roster),
                checkpoints=checkpoints,
                lsp=self._lsp,
                leader_tools=goal_tools(
                    lambda: self._goal_for_session(session.run_id),
                    lambda status, message: self._update_goal_for_session(
                        session.run_id, status, message,
                    ),
                ),
            ),
            session=session,
        )

    def _publish_active(self, event: Event) -> None:
        with self._lock:
            session_id = self._session_by_root_agent.get(event.agent_id)
            if isinstance(event, SubagentSpawned) and session_id is not None:
                self._session_by_root_agent[event.subagent_agent_id] = session_id
            if session_id is None and len(self._active_by_session) == 1:
                session_id = next(iter(self._active_by_session))
            active = self._active_by_session.get(session_id) if session_id is not None else None
            run_id = None if active is None else active.run_id
            if (
                active is not None
                and event.agent_id == active.root_agent_id
                and isinstance(event, (RunFinished, RunFailed))
            ):
                active.terminal_event = event
                return
        if run_id is not None:
            self._publish(run_id, event, session_id=session_id)

    def open_session(self, run_id: str) -> dict:
        """Load and replay a finished transcript without ever rewriting it."""
        with self._lock:
            reader = SessionStore.open(self._sessions_root, run_id)
            try:
                loaded, diagnosis, repaired_ids = load_run_for_resume(reader)
                metadata = reader.read_meta()
                choice = metadata.get("provider_choice")
                provider_state_reset = metadata.get("provider_state_reset") is True
                goal = Goal.from_payload(metadata.get("goal"))
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
                if self._goal is not None:
                    self._goals_by_session[run_id] = self._goal
                for message, record_id in zip(loaded.messages, loaded.record_ids):
                    self._broker.publish(ForkableHistoryMessage(
                        role=message.role.value,
                        text=message.text,
                        tool_calls=[{"id": call.id, "name": call.name} for call in message.tool_calls],
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
                        session_id=run_id,
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
            if goal is not None:
                self._goals_by_session[run_id] = goal
            self._context_report = None
            self._usage_by_agent.clear()
            goal_agent_id = leader.agent_ref.agent_id
            self._close_idle_conversations_locked()
        if reopened_goal:
            self._goal_event("pause", goal, agent_id=goal_agent_id)
        for message, record_id in zip(loaded.messages, loaded.record_ids):
            self._broker.publish(ForkableHistoryMessage(
                role=message.role.value,
                text=message.text,
                tool_calls=[{"id": call.id, "name": call.name} for call in message.tool_calls],
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
                session_id=run_id,
            ))
        return {
            "run_id": loaded.run_id,
            "state": diagnosis.state.value,
            "replayed": len(loaded.record_ids),
            "repaired_ids": repaired_ids,
            "dropped_bytes": loaded.dropped_bytes,
        }

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
            conversation[1].close()
            self._open_conversations.pop(conversation[1].run_id, None)
            self._policy_by_session.pop(conversation[1].run_id, None)
            self._approvals_by_session.pop(conversation[1].run_id, None)

    def close(self) -> None:
        self._closing = True
        with self._lock:
            active_runs = tuple(self._active_by_session.values())
            approval_brokers = tuple(dict.fromkeys(self._approvals_by_session.values()))
        for active_run in active_runs:
            active_run.cancel.cancel()
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
        for _, session in conversations:
            session.close()

    def stop(self) -> None:
        with self._lock:
            active = self._active
            session_id = None if self._conversation is None else self._conversation[1].run_id
            goal_check = None if session_id is None else self._goal_checks_by_session.get(session_id)
        if active is not None:
            active.cancel.cancel()
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

    @staticmethod
    def _merge_usage(
        current: dict[str, UsageTotals], incoming: Mapping[str, UsageTotals]
    ) -> dict[str, UsageTotals]:
        merged = dict(current)
        for model, totals in incoming.items():
            merged[model] = merged.get(model, UsageTotals()).merged(totals)
        return merged

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
        seen = set()

        def run_order(node):  # noqa: ANN001
            return node.run_id not in started_at, started_at.get(node.run_id, "")

        def add_node(node, parent_agent_id: str | None) -> None:  # noqa: ANN001
            if node.agent_id not in seen:
                seen.add(node.agent_id)
                name, by_model = usage_by_agent.get(node.agent_id, (node.agent_name, None))
                agent = {
                    "agent_id": node.agent_id,
                    "name": name,
                    "parent_agent_id": parent_agent_id if parent_agent_id != node.agent_id else None,
                }
                if by_model is not None and report is not None:
                    agent.update(usage_fields(by_model))
                agents.append(agent)
            for child in sorted(node.children, key=run_order):
                add_node(child, node.agent_id)

        for root in sorted(graph, key=run_order):
            add_node(root, None)
        for agent_id, (name, by_model) in usage_by_agent.items():
            if agent_id not in seen:
                agents.append({
                    "agent_id": agent_id,
                    "name": name,
                    "parent_agent_id": None,
                    **(usage_fields(by_model) if report is not None else {}),
                })

        result = {
            "session_id": conversation[1].run_id,
            "agents": agents,
            "mode": mode,
            "goal": goal_payload,
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
            leader, _ = self._conversation
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
            return {"reverted": [entry.path for entry in targets]}

    def _publish(self, host_run_id: str, event: Event, *, session_id: str | None = None) -> None:
        if session_id is None:
            with self._lock:
                session_id = next((
                    sid for sid, active in self._active_by_session.items()
                    if active.run_id == host_run_id
                ), None)
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

        result = run_check(
            context.goal.check,
            self._policy_by_session[context.session_id].repo_root,
            cancelled=context.cancel,
            set_process=set_process,
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

    def _finish_spec_run(self, session_id: str, phase: str) -> None:
        try:
            store = SessionStore.open(self._sessions_root, session_id)
            try:
                meta = store.read_meta()
                info = meta.get("spec_run")
                if not isinstance(info, dict) or info.get("kind") != "implement":
                    return
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
                    self._active_by_session.pop(session_id, None)
                    if self._conversation is not None and self._conversation[1].run_id == session_id:
                        self._active = None
                if self._closing and self._conversation is not None and self._conversation[1].run_id == session_id:
                    self._conversation = None
                    self._active = None
            if terminal_event is not None:
                self._publish(run_id, terminal_event, session_id=session_id)
        if not goal_round:
            self._finish_spec_review(session_id, leader, terminal_event)
            self._finish_spec_plan(session_id, terminal_event)
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
                self._finish_spec_run(session_id, "paused")
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
