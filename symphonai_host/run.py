"""One threaded runtime run owned by a SymphonAI host process."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from symphonai_api.agent_loop import DEFAULT_MAX_TURNS, ApiAgent
from symphonai_api.cancellation import CancellationToken
from symphonai_api.events import Event, RunStarted
from symphonai_api.identity import AgentRef, new_agent_ref, new_id
from symphonai_api.models import Message, Role
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import ModelProvider
from symphonai_api.runner import standard_tool_registry
from symphonai_api.session import (
    LoadedRun,
    RunDiagnosis,
    SessionStore,
    default_sessions_root,
    load_run_for_resume,
    tool_result_search_path,
)
from symphonai_api.tool_schema import tool_registry_schemas
from symphonai_api.tool_results import ToolResultStore
from symphonai_host.broker import EventBroker
from symphonai_host.approvals import ApprovalBroker, PendingApproval
from symphonai_host.protocol import HistoryMessage


class RunActiveError(RuntimeError):
    """A client attempted to start a second run while one is active."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run already active: {run_id}")
        self.run_id = run_id


@dataclass
class _ActiveRun:
    run_id: str
    cancel: CancellationToken
    thread: threading.Thread
    root_agent_id: str
    runtime_run_id: str | None = None


class HostRun:
    """Build and run one ApiAgent at a time without changing the runtime API."""

    def __init__(
        self,
        provider: ModelProvider,
        policy: PermissionPolicy,
        broker: EventBroker,
        *,
        system_prompt: str | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        model: str | None = None,
        publish_approval=None,
        approval_timeout: float = 300.0,
        sessions_root: Path | None = None,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._broker = broker
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._model = model
        self._active: _ActiveRun | None = None
        self._opened: tuple[SessionStore, LoadedRun, RunDiagnosis, list[str]] | None = None
        self._sessions_root = default_sessions_root() if sessions_root is None else Path(sessions_root)
        self._lock = threading.Lock()
        self.approvals = ApprovalBroker(publish_approval or (lambda _: False), timeout=approval_timeout)
        self._policy.approval_callback = self.approvals.callback

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

    def start(self, prompt: str) -> str:
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            run_id = new_id("run")
            cancel = CancellationToken()
            agent_ref = new_agent_ref("agent")
            session = SessionStore(self._sessions_root, run_id)
            opened = self._opened
            self._opened = None
            if opened is None:
                messages = [Message(role=Role.USER, content=prompt)]
                if self._system_prompt:
                    messages.insert(0, Message(role=Role.SYSTEM, content=self._system_prompt))
                parent_run_id = None
                fallback_directories: tuple = ()
            else:
                store, loaded, _, _ = opened
                messages = [*loaded.messages, Message(role=Role.USER, content=prompt)]
                parent_run_id = loaded.run_id
                session.set_parent_session(store.run_id)
                fallback_directories = tool_result_search_path(store)
            thread = threading.Thread(
                target=self._run,
                args=(run_id, agent_ref, messages, parent_run_id, session, fallback_directories, cancel),
                name=f"symphonai-host-{run_id}",
                daemon=True,
            )
            self._active = _ActiveRun(run_id, cancel, thread, agent_ref.agent_id)
            thread.start()
            return run_id

    def open_session(self, run_id: str) -> dict:
        """Load and replay a finished transcript without ever rewriting it."""
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            store = SessionStore.open(self._sessions_root, run_id)
            loaded, diagnosis, repaired_ids = load_run_for_resume(store)
            self._opened = (store, loaded, diagnosis, repaired_ids)
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

    def stop(self) -> None:
        with self._lock:
            active = self._active
        if active is not None:
            active.cancel.cancel()
        self.approvals.cancel_all(reason="stopped")

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

    def _run(
        self,
        run_id: str,
        agent_ref: AgentRef,
        messages: list[Message],
        parent_run_id: str | None,
        session: SessionStore,
        fallback_directories: tuple,
        cancel: CancellationToken,
    ) -> None:
        result_store = ToolResultStore(
            directory=session.tool_results_directory,
            fallback_directories=fallback_directories,
        )
        tools = standard_tool_registry(result_store=result_store)
        agent = ApiAgent(
            provider=self._provider,
            tools=tools,
            policy=self._policy,
            max_turns=self._max_turns,
            tool_schemas=tool_registry_schemas(tools, self._provider.wire_format),
            agent_ref=agent_ref,
            events=lambda event: self._publish(run_id, event),
            result_store=result_store,
            transcript=session.writer_for(agent_ref.agent_id, is_root=True),
        )
        try:
            agent.run(messages, model=self._model, parent_run_id=parent_run_id, cancel=cancel)
        finally:
            session.close()
            with self._lock:
                if self._active is not None and self._active.run_id == run_id:
                    self._active = None
