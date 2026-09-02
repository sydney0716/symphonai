"""Typed, ephemeral observation events for SymphonAI runtime runs."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from symphonai_api.cancellation import OperationCancelled
from symphonai_api.identity import SCHEMA_VERSION


@dataclass(frozen=True)
class Event:
    """Base for every runtime event.

    Events are ephemeral observation, not the transcript. They are never
    persisted as conversation, never sent to a provider, and dropping every
    one of them must not change what a run computes. Phase 04 persists
    ``Message`` records; this channel is for live consumers.
    """

    agent_id: str
    run_id: str
    turn_id: str | None = None
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class RunStarted(Event):
    agent_name: str = ""


@dataclass(frozen=True)
class RunFinished(Event):
    agent_name: str = ""
    stopped_reason: str = ""


@dataclass(frozen=True)
class RunFailed(Event):
    agent_name: str = ""
    error: str = ""


@dataclass(frozen=True)
class TurnStarted(Event):
    index: int = 0


@dataclass(frozen=True)
class TurnFinished(Event):
    index: int = 0


@dataclass(frozen=True)
class AssistantTextDelta(Event):
    text: str = ""


@dataclass(frozen=True)
class ToolCallStarted(Event):
    tool_name: str = ""
    tool_call_id: str = ""


@dataclass(frozen=True)
class ToolCallFinished(Event):
    tool_name: str = ""
    tool_call_id: str = ""
    ok: bool = False


@dataclass(frozen=True)
class SubagentSpawned(Event):
    subagent_name: str = ""
    subagent_agent_id: str = ""


@dataclass(frozen=True)
class CompactionApplied(Event):
    before_tokens: int = 0
    after_tokens: int = 0
    dropped_messages: int = 0


EventSink = Callable[[Event], None]
"""A thread-safe consumer of runtime events."""


def emit(sink: EventSink | None, event: Event) -> None:
    """Deliver one event, doing nothing when there is no sink."""
    if sink is None:
        return
    try:
        sink(event)
    except OperationCancelled:
        raise
    except Exception:
        pass


class CollectingSink:
    """An event sink that records events in order for tests and debugging."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._lock = threading.Lock()

    def __call__(self, event: Event) -> None:
        with self._lock:
            self.events.append(event)

    def of_type(self, event_type: type) -> list[Event]:
        with self._lock:
            return [event for event in self.events if isinstance(event, event_type)]
