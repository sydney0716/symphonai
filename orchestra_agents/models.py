"""Provider-agnostic data model for the Orchestra SubagentProvider contract.

These types describe how the Orchestra leader talks to any CLI- or
API-backed coding agent (Claude Code, Codex, Gemini, future providers)
through one shared shape, independent of any vendor-specific terms.

Field naming note: this module uses `session_id`, never a vendor-specific
name like `thread_id`. `session_id` is the provider-agnostic conversation
handle a leader stores to resume a subagent later. It is distinct from the
`thread_id` field already used by the existing Orchestra task/worker schema
in `orchestra_mcp/schema.py`, which stays as-is for Codex worker records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    """Kinds of events a provider may emit while running or resuming a task."""

    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    DONE = "done"


@dataclass(frozen=True)
class AgentProfile:
    """A configured, addressable agent the leader can dispatch work to.

    This is the provider-agnostic identity of a subagent (e.g. "claude-default",
    "codex-sandboxed"), distinct from `provider`, the name of the
    `SubagentProvider` implementation that knows how to talk to a specific
    vendor CLI or API.
    """

    name: str
    provider: str
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTask:
    """A unit of work the leader hands to a subagent provider."""

    task_id: str
    prompt: str
    cwd: str | None = None
    scope: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentEvent:
    """One event emitted while a subagent runs a task or a follow-up.

    `session_id` is the provider-agnostic handle used to resume this run
    later, whatever a given vendor CLI calls it internally.
    """

    session_id: str
    kind: EventKind
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCapability:
    """A single capability a provider is believed or confirmed to support.

    Instances built from static research (see `orchestra_agents.probes`) are
    candidates, not confirmed runtime facts, until a probe result backs them.
    `confirmed` distinguishes the two states; see `ValidationResult` for the
    outcome of an actual probe run.
    """

    name: str
    supported: bool
    detail: str = ""
    confirmed: bool = False


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of validating a capability, probe, or run."""

    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class AgentRunResult:
    """The terminal result of a `SubagentProvider.spawn()` or `.resume()` call.

    `session_id` is always present so the leader can persist it and later
    call `resume(session_id, follow_up)` against the same subagent
    conversation without knowing any vendor-specific handle name.
    """

    session_id: str
    ok: bool
    events: list[AgentEvent] = field(default_factory=list)
    final_message: str = ""
    error: str | None = None
