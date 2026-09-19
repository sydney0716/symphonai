"""Threaded conversation runs owned by a SymphonAI host process."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from symphonai_api.agent_loop import DEFAULT_MAX_TURNS
from symphonai_api.cancellation import CancellationToken
from symphonai_api.events import Event, RunStarted, fan_out
from symphonai_api.extensions import Extensions
from symphonai_api.identity import new_id
from symphonai_api.leader import Leader, LeaderConfig
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


class HostRun:
    """Run one prompt at a time in a persistent Leader conversation."""

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
        extensions: Extensions | None = None,
        mcp_tools: Mapping[str, LocalTool] | None = None,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._broker = broker
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._model = model
        self._extensions = extensions
        self._hooks = (
            None
            if extensions is None
            else extensions.hook_runner(cwd=policy.repo_root)
        )
        self._mcp_tools = mcp_tools
        self._active: _ActiveRun | None = None
        self._conversation: tuple[Leader, SessionStore] | None = None
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

    def start(self, prompt: str) -> str:
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            run_id = new_id("run")
            cancel = CancellationToken()
            if self._conversation is None:
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
                if self._system_prompt:
                    leader.seed_chat([Message(role=Role.SYSTEM, content=self._system_prompt)])
                meta = session.read_meta()
                meta["title"] = _conversation_title(prompt)
                session.write_meta(meta)
                self._conversation = (leader, session)
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
        if run_id is not None:
            self._publish(run_id, event)

    def open_session(self, run_id: str) -> dict:
        """Load and replay a finished transcript without ever rewriting it."""
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            reader = SessionStore.open(self._sessions_root, run_id)
            loaded, diagnosis, repaired_ids = load_run_for_resume(reader)
            reader.close()
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
        try:
            leader.chat(prompt, cancel=cancel)
        finally:
            conversation = None
            with self._lock:
                if self._active is not None and self._active.run_id == run_id:
                    self._active = None
                if self._closing:
                    conversation = self._conversation
                    self._conversation = None
            if conversation is not None:
                conversation[1].close()
