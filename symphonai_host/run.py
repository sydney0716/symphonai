"""Threaded conversation runs owned by a SymphonAI host process."""

from __future__ import annotations

import shutil
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from symphonai_api.agent_loop import DEFAULT_MAX_TURNS
from symphonai_api.agent_memory import AgentMemory
from symphonai_api.agent_spec import AgentSpec
from symphonai_api.budgets import RunBudget
from symphonai_api.cancellation import CancellationToken
from symphonai_api.compaction import DEFAULT_RECENT_TURNS
from symphonai_api.config import resolve_run_budgets
from symphonai_api.context_report import ContextReport, account_context
from symphonai_api.cost import PriceTable, UsageTotals, total_cost
from symphonai_api.events import Event, RunFailed, RunFinished, RunStarted, fan_out
from symphonai_api.extensions import Extensions
from symphonai_api.identity import new_id
from symphonai_api.instructions import load_instructions
from symphonai_api.leader import (
    DEFAULT_SUBAGENT_MAX_TURNS, Leader, LeaderConfig, LeaderRunResult,
    builtin_subagent_specs,
)
from symphonai_api.models import Message, Role
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
from symphonai_api.tool_results import ToolResultStore
from symphonai_api.tools.base import LocalTool
from symphonai_api.web_search import HttpJsonSearchBackend, search_endpoint
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


class ModeSelectionError(ValueError):
    """The requested permission mode is not available to this host."""


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
            min(existing.max_turns, ceiling.max_turns)
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
        memory_root: Path | None = None,
        extensions: Extensions | None = None,
        mcp_tools: Mapping[str, LocalTool] | None = None,
        price_table: PriceTable | None = None,
        chat_token_budget: int | None = None,
        chat_recent_turns: int = DEFAULT_RECENT_TURNS,
    ) -> None:
        self._provider = provider
        self._policy = policy
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
                subagent_max_turns=DEFAULT_SUBAGENT_MAX_TURNS,
                price_table=price_table,
            )
        self._chat_token_budget = chat_token_budget
        self._chat_recent_turns = chat_recent_turns
        self._active: _ActiveRun | None = None
        self._conversation: tuple[Leader, SessionStore] | None = None
        self._context_report: ContextReport | None = None
        self._usage_by_agent: dict[str, tuple[str, dict[str, UsageTotals]]] = {}
        self._closing = False
        self._sessions_root = default_sessions_root() if sessions_root is None else Path(sessions_root)
        self._memory_root = default_memory_root() if memory_root is None else Path(memory_root)
        self._memory: AgentMemory | None = None
        self._memory_open_attempted = False
        self._lock = threading.RLock()
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

    def _new_leader(self, session: SessionStore) -> Leader:
        result_store = ToolResultStore(
            directory=session.tool_results_directory,
            fallback_directories=tool_result_search_path(session),
        )
        roster = builtin_subagent_specs(self._provider, self._policy, self._search_backend)
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
        defined_leader = roster.get("leader")
        if (
            defined_leader is not None
            and self._search_backend is None
            and "web_search" in (defined_leader.tool_names or ())
        ):
            raise ProviderSelectionError("leader cannot use web_search: search is not configured")
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
                subagent_specs=roster,
                subagent_budget=self._subagent_budget,
                hook_runner=self._hooks,
                memory=self._memory_for(roster),
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
            try:
                loaded, diagnosis, repaired_ids = load_run_for_resume(reader)
                metadata = reader.read_meta()
                choice = metadata.get("provider_choice")
                provider_state_reset = metadata.get("provider_state_reset") is True
            finally:
                reader.close()
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
                events=fan_out(self._broker.publish, self._hooks),
            )
            try:
                self._provider, self._model, self._effort, self._provider_choice = (
                    provider,
                    model,
                    effort,
                    provider_choice,
                )
                self._policy.mode = self._starting_mode
                leader = self._new_leader(store)
                messages = (
                    _without_vendor_state(loaded.messages)
                    if provider_state_reset
                    else loaded.messages
                )
                leader.seed_chat(messages, persisted=True)
            except Exception:
                self._provider, self._model, self._effort, self._provider_choice = previous_provider
                self._policy.mode = previous_mode
                store.close()
                raise
            if self._conversation is not None:
                self._conversation[1].close()
            self._conversation = (leader, store)
            self._context_report = None
            self._usage_by_agent.clear()
        for message, record_id in zip(loaded.messages, loaded.record_ids, strict=True):
            self._broker.publish(ForkableHistoryMessage(
                role=message.role.value,
                text=message.text,
                tool_calls=[{"id": call.id, "name": call.name} for call in message.tool_calls],
                turn_id=message.turn_id,
                record_id=record_id,
            ))
        return {
            "run_id": loaded.run_id,
            "state": diagnosis.state.value,
            "replayed": len(loaded.messages),
            "repaired_ids": repaired_ids,
            "dropped_bytes": loaded.dropped_bytes,
        }

    def fork_session(self, run_id: str, record_id: str) -> dict:
        """Copy a message prefix, then reopen the descendant conversation."""
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            source = SessionStore.open(self._sessions_root, run_id)
            try:
                loaded = load_run(source)
                if record_id not in loaded.record_ids:
                    raise SessionError(f"run {run_id!r} has no current message record {record_id!r}")
                source_meta = source.read_meta()
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
                except Exception:
                    destination.close()
                    shutil.rmtree(destination.directory)
                    raise
                else:
                    destination.close()
            finally:
                source.close()
            try:
                return self.open_session(new_run_id)
            except Exception:
                shutil.rmtree(destination.directory)
                raise

    def end_conversation(self) -> None:
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            conversation = self._conversation
            self._conversation = None
            self._policy.mode = self._starting_mode
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
            budget=leader.chat_token_budget,
        )

    def conversation_stats(self) -> dict | None:
        with self._lock:
            conversation = self._conversation
            if conversation is None:
                return None
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
            report = self._context_report
            mode = self._policy.mode
            usage_by_agent = {
                agent_id: (name, dict(by_model))
                for agent_id, (name, by_model) in self._usage_by_agent.items()
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

        result = {"agents": agents, "mode": mode, **selection}
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
                "by_source": {
                    source.value: tokens for source, tokens in report.by_source().items()
                },
            },
            "usage": usage_fields(all_models),
        })
        return result

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
