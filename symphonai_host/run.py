"""One threaded runtime run owned by a SymphonAI host process."""

from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass

from symphonai_api.agent_loop import DEFAULT_MAX_TURNS, ApiAgent
from symphonai_api.cancellation import CancellationToken
from symphonai_api.events import Event
from symphonai_api.identity import new_agent_ref, new_id
from symphonai_api.models import Message, Role
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import ModelProvider
from symphonai_api.runner import standard_tool_registry
from symphonai_api.tool_schema import tool_registry_schemas
from symphonai_host.broker import EventBroker


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
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._broker = broker
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._model = model
        self._active: _ActiveRun | None = None
        self._lock = threading.Lock()

    @property
    def active_run_id(self) -> str | None:
        with self._lock:
            return None if self._active is None else self._active.run_id

    @property
    def active(self) -> bool:
        return self.active_run_id is not None

    def start(self, prompt: str) -> str:
        with self._lock:
            if self._active is not None:
                raise RunActiveError(self._active.run_id)
            run_id = new_id("run")
            cancel = CancellationToken()
            thread = threading.Thread(
                target=self._run,
                args=(run_id, prompt, cancel),
                name=f"symphonai-host-{run_id}",
                daemon=True,
            )
            self._active = _ActiveRun(run_id, cancel, thread)
            thread.start()
            return run_id

    def stop(self) -> None:
        with self._lock:
            active = self._active
        if active is not None:
            active.cancel.cancel()

    def _publish(self, host_run_id: str, event: Event) -> None:
        # ApiAgent allocates its own internal run id. The host owns the wire run,
        # so clients can safely use the id returned by /prompt for every event.
        self._broker.publish(dataclasses.replace(event, run_id=host_run_id))

    def _run(self, run_id: str, prompt: str, cancel: CancellationToken) -> None:
        agent_ref = new_agent_ref("agent")
        tools = standard_tool_registry()
        agent = ApiAgent(
            provider=self._provider,
            tools=tools,
            policy=self._policy,
            max_turns=self._max_turns,
            tool_schemas=tool_registry_schemas(tools, self._provider.wire_format),
            agent_ref=agent_ref,
            events=lambda event: self._publish(run_id, event),
        )
        messages = [Message(role=Role.USER, content=prompt)]
        if self._system_prompt:
            messages.insert(0, Message(role=Role.SYSTEM, content=self._system_prompt))
        try:
            agent.run(messages, model=self._model, cancel=cancel)
        finally:
            with self._lock:
                if self._active is not None and self._active.run_id == run_id:
                    self._active = None
