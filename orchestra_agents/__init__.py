"""Provider-agnostic contract for dispatching work to subagent CLIs/APIs.

See `docs/orchestra-agent-adapters.md` for the full design writeup.
"""

from __future__ import annotations

from orchestra_agents.models import (
    AgentEvent,
    AgentProfile,
    AgentRunResult,
    AgentTask,
    EventKind,
    ProviderCapability,
    ValidationResult,
)

__all__ = [
    "AgentEvent",
    "AgentProfile",
    "AgentRunResult",
    "AgentTask",
    "EventKind",
    "ProviderCapability",
    "ValidationResult",
]
