"""Threaded conversation runs owned by a SymphonAI host process."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from symphonai_api.agent_loop import DEFAULT_MAX_TURNS
from symphonai_api.cancellation import CancellationToken
from symphonai_api.compaction import DEFAULT_CONTEXT_TOKEN_BUDGET, DEFAULT_RECENT_TURNS
from symphonai_api.context_report import ContextReport, account_context
from symphonai_api.cost import PriceTable, UsageTotals, total_cost
from symphonai_api.events import Event, RunFailed, RunFinished, RunStarted, fan_out
from symphonai_api.extensions import Extensions
from symphonai_api.identity import new_id
from symphonai_api.instructions import load_instructions
from symphonai_api.leader import Leader, LeaderConfig, LeaderRunResult
from symphonai_api.models import Message, Role
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import ModelProvider
from symphonai_api.session import (
    SessionStore,
    default_sessions_root,
    load_run_for_resume,
    tool_result_search_path,
)
from symphonai_api.tool_results import ToolResultStore
from symphonai_api.tools.base import LocalTool
from symphonai_host.broker import EventBroker
from symphonai_host.approvals import ApprovalBroker, PendingApproval
from symphonai_host.protocol import HistoryMessage


class RunActiveError(RuntimeError):
    """A client attempted to start a second run while one is active."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run already active: {run_id}")
        self.run_id = run_id


class ProviderSelectionError(ValueError):
    """A conversation cannot start with the requested provider."""


CONVERSATION_TITLE_LIMIT = 80


def _conversation_title(prompt: str) -> str:
    return " ".join(prompt.split())[:CONVERSATION_TITLE_LIMIT]


@dataclass
class _ActiveRun:
    run_id: str
    cancel: CancellationToken
    thread: threading.Thread
    root_agent_id: str
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
        max_turns: int = DEFAULT_MAX_TURNS,
        model: str | None = None,
        provider_factory: Callable[[str | None, str | None, str | None], ModelProvider | None] | None = None,
        publish_approval=None,
        approval_timeout: float = 300.0,
        sessions_root: Path | None = None,
        extensions: Extensions | None = None,
        mcp_tools: Mapping[str, LocalTool] | None = None,
        price_table: PriceTable | None = None,
        chat_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
        chat_recent_turns: int = DEFAULT_RECENT_TURNS,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._broker = broker
        self._system_prompt = system_prompt
        if working_dir is None:
            current = Path.cwd().resolve()
            self._working_dir = current if current.is_relative_to(policy.repo_root) else policy.repo_root
        else:
            self._working_dir = Path(working_dir)
        self._max_turns = max_turns
        self._model = model
        self._provider_factory = provider_factory
        self._provider_choice = (
            {"name": provider.name, "model": model, "base_url": getattr(provider, "base_url", None)}
            if provider is not None and provider.name in ("anthropic", "gemini", "openai")
            else None
        )
        self._extensions = extensions
        self._hooks = (
            None
            if extensions is None
            else extensions.hook_runner(cwd=policy.repo_root)
        )
        self._mcp_tools = mcp_tools
        self._price_table = price_table
        self._chat_token_budget = chat_token_budget
        self._chat_recent_turns = chat_recent_turns
        self._active: _ActiveRun | None = None
        self._conversation: tuple[Leader, SessionStore] | None = None
        self._context_report: ContextReport | None = None
        self._usage_by_agent: dict[str, tuple[str, dict[str, UsageTotals]]] = {}
        self._closing = False
        self._sessions_root = default_sessions_root() if sessions_root is None else Path(sessions_root)
        self._lock = threading.Lock()
        self.approvals = ApprovalBroker(publish_approval or (lambda _: False), timeout=approval_timeout)
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

    def select_provider(self, provider: ModelProvider, model: str | None = None, choice: dict | None = None) -> None:
        with self._lock:
            self._provider = provider
            self._model = model
            self._provider_choice = choice

    def start(self, prompt: str) -> str:
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            run_id = new_id("run")
            cancel = CancellationToken()
            if self._conversation is None:
                if self._provider is None and self._provider_factory is not None:
                    self._provider = self._provider_factory(None, None, None)
                    if self._provider is not None:
                        self._provider_choice = {"name": self._provider.name}
                if self._provider is None:
                    raise ProviderSelectionError("no configured provider; add an API key in Settings")
                session = SessionStore(
                    self._sessions_root,
                    run_id,
                    repo_root=self._policy.repo_root,
                    events=fan_out(self._broker.publish, self._hooks),
                )
                try:
                    leader = self._new_leader(session)
                except Exception:
                    session.close()
                    raise
                seeded = []
                if self._system_prompt:
                    seeded.append(Message(role=Role.SYSTEM, content=self._system_prompt))
                instructions = load_instructions(self._policy, working_dir=self._working_dir)
                for warning in instructions.warnings:
                    print(f"instruction warning: {warning}", file=sys.stderr)
                rendered = instructions.render()
                if rendered:
                    seeded.append(Message(role=Role.SYSTEM, content=rendered))
                if seeded:
                    leader.seed_chat(seeded)
                meta = session.read_meta()
                meta["title"] = _conversation_title(prompt)
                if self._provider_choice is not None:
                    meta["provider_choice"] = self._provider_choice
                session.write_meta(meta)
                self._conversation = (leader, session)
                self._context_report = None
                self._usage_by_agent.clear()
            else:
                leader, _ = self._conversation
            thread = threading.Thread(
                target=self._run,
                args=(run_id, leader, prompt, cancel),
                name=f"symphonai-host-{run_id}",
                daemon=True,
            )
            self._active = _ActiveRun(run_id, cancel, thread, leader.agent_ref.agent_id)
            thread.start()
            return run_id

    def _new_leader(self, session: SessionStore) -> Leader:
        result_store = ToolResultStore(
            directory=session.tool_results_directory,
            fallback_directories=tool_result_search_path(session),
        )
        return Leader(
            LeaderConfig(
                leader_provider=self._provider,
                subagent_provider=self._provider,
                repo_root=str(self._policy.repo_root),
                max_leader_turns=self._max_turns,
                chat_token_budget=self._chat_token_budget,
                chat_recent_turns=self._chat_recent_turns,
                permission_mode=self._policy.mode,
                approval_callback=self.approvals.callback,
                events=lambda event: self._publish_active(event),
                extensions=self._extensions,
                stream=True,
                result_store=result_store,
                extra_tools=self._mcp_tools,
                leader_policy=self._policy,
                leader_model=self._model,
                hook_runner=self._hooks,
            ),
            session=session,
        )

    def _publish_active(self, event: Event) -> None:
        with self._lock:
            active = self._active
            run_id = None if active is None else active.run_id
            if (
                active is not None
                and event.agent_id == active.root_agent_id
                and isinstance(event, (RunFinished, RunFailed))
            ):
                active.terminal_event = event
                return
        if run_id is not None:
            self._publish(run_id, event)

    def open_session(self, run_id: str) -> dict:
        """Load and replay a finished transcript without ever rewriting it."""
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            reader = SessionStore.open(self._sessions_root, run_id)
            loaded, diagnosis, repaired_ids = load_run_for_resume(reader)
            choice = reader.read_meta().get("provider_choice")
            reader.close()
            if isinstance(choice, dict) and self._provider_factory is not None:
                provider = self._provider_factory(choice.get("name"), choice.get("model"), choice.get("base_url"))
                if provider is None:
                    raise ProviderSelectionError("session provider is unavailable")
                self._provider = provider
                self._model = choice.get("model")
                self._provider_choice = choice
            elif self._provider is None and self._provider_factory is not None:
                self._provider = self._provider_factory(None, None, None)
                if self._provider is None:
                    raise ProviderSelectionError("no configured provider; add an API key in Settings")
            if self._conversation is not None:
                self._conversation[1].close()
            store = SessionStore.open(
                self._sessions_root,
                run_id,
                events=fan_out(self._broker.publish, self._hooks),
            )
            leader = self._new_leader(store)
            leader.seed_chat(loaded.messages, persisted=True)
            self._conversation = (leader, store)
            self._context_report = None
            self._usage_by_agent.clear()
        for message in loaded.messages:
            self._broker.publish(HistoryMessage(
                role=message.role.value,
                text=message.text,
                tool_calls=[{"id": call.id, "name": call.name} for call in message.tool_calls],
                turn_id=message.turn_id,
            ))
        return {
            "run_id": loaded.run_id,
            "state": diagnosis.state.value,
            "replayed": len(loaded.messages),
            "repaired_ids": repaired_ids,
            "dropped_bytes": loaded.dropped_bytes,
        }

    def end_conversation(self) -> None:
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            conversation = self._conversation
            self._conversation = None
            self._context_report = None
            self._usage_by_agent.clear()
        if conversation is not None:
            conversation[1].close()

    def close(self) -> None:
        self._closing = True
        self.stop()
        with self._lock:
            active = self._active
        if active is not None:
            active.thread.join(timeout=2)
        if not self.active:
            self.end_conversation()

    def stop(self) -> None:
        with self._lock:
            active = self._active
        if active is not None:
            active.cancel.cancel()
        self.approvals.cancel_all(reason="stopped")

    @staticmethod
    def _merge_usage(
        current: dict[str, UsageTotals], incoming: Mapping[str, UsageTotals]
    ) -> dict[str, UsageTotals]:
        merged = dict(current)
        for model, totals in incoming.items():
            merged[model] = merged.get(model, UsageTotals()).merged(totals)
        return merged

    def _record_result(self, leader: Leader, result: LeaderRunResult) -> None:
        root_id = result.agent.agent_id
        root_current = self._usage_by_agent.get(root_id, (result.agent.name, {}))[1]
        self._usage_by_agent[root_id] = (
            result.agent.name,
            self._merge_usage(root_current, result.usage_by_agent.get(root_id, {})),
        )
        for name, record in result.subagents.items():
            self._usage_by_agent[record.agent_ref.agent_id] = (
                name,
                dict(record.usage_by_model),
            )
        self._context_report = account_context(
            leader._chat_messages,
            budget=self._chat_token_budget,
        )

    def conversation_stats(self) -> dict | None:
        with self._lock:
            report = self._context_report
            usage_by_agent = {
                agent_id: (name, dict(by_model))
                for agent_id, (name, by_model) in self._usage_by_agent.items()
            }
        if report is None:
            return None

        def usage_fields(by_model: Mapping[str, UsageTotals]) -> dict:
            totals = UsageTotals()
            for usage in by_model.values():
                totals = totals.merged(usage)
            fields = {
                "input_tokens": totals.input_tokens,
                "output_tokens": totals.output_tokens,
                "calls": totals.calls,
                "total_tokens": totals.total_tokens,
            }
            cost = total_cost(by_model, self._price_table)
            if cost is not None and self._price_table is not None:
                fields["cost"] = {
                    "amount": str(cost),
                    "currency": self._price_table.currency,
                }
            return fields

        all_models: dict[str, UsageTotals] = {}
        agents = []
        for agent_id, (name, by_model) in usage_by_agent.items():
            all_models = self._merge_usage(all_models, by_model)
            agents.append({
                "agent_id": agent_id,
                "name": name,
                **usage_fields(by_model),
            })
        return {
            "context": {
                "used_tokens": report.total_tokens,
                "budget_tokens": report.budget,
                "remaining_tokens": report.remaining_tokens,
                "by_source": {
                    source.value: tokens for source, tokens in report.by_source().items()
                },
            },
            "usage": usage_fields(all_models),
            "agents": agents,
        }

    def _publish(self, host_run_id: str, event: Event) -> None:
        if isinstance(event, RunStarted):
            try:
                with self._lock:
                    active = self._active
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
        self._broker.publish(event)

    def _run(self, run_id: str, leader: Leader, prompt: str, cancel: CancellationToken) -> None:
        terminal_event = None
        try:
            result = leader.chat(prompt, cancel=cancel)
            with self._lock:
                if self._conversation is not None and self._conversation[0] is leader:
                    self._record_result(leader, result)
        finally:
            conversation = None
            with self._lock:
                if self._active is not None and self._active.run_id == run_id:
                    terminal_event = self._active.terminal_event
                    self._active = None
                if self._closing:
                    conversation = self._conversation
                    self._conversation = None
            if conversation is not None:
                conversation[1].close()
            if terminal_event is not None:
                self._publish(run_id, terminal_event)
