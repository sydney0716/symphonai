"""The SubagentProvider contract every agent backend implements.

A `SubagentProvider` is anything the Orchestra leader can hand an
`AgentTask` to and later resume by `session_id` -- a wrapped CLI subprocess
today, a raw vendor API client later. This module defines only the
contract: it never shells out, never calls a network API, and never
imports a concrete adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from orchestra_agents.models import AgentRunResult, AgentTask


class SubagentProvider(ABC):
    """Abstract contract for dispatching and resuming subagent work.

    Implementations live under `orchestra_agents.adapters` (e.g.
    `FakeProvider` today; real CLI-backed adapters are future work -- see
    `docs/orchestra-agent-adapters.md`).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used to register/look up this provider, e.g. "fake"."""

    @abstractmethod
    def spawn(self, task: AgentTask) -> AgentRunResult:
        """Start a new subagent run for `task` and return its terminal result.

        The returned `AgentRunResult.session_id` is the handle a caller must
        pass back into `resume()` to continue this same conversation.
        """

    @abstractmethod
    def resume(self, session_id: str, follow_up: str) -> AgentRunResult:
        """Continue an existing subagent run identified by `session_id`.

        `follow_up` is the next instruction for the subagent. The returned
        `AgentRunResult` carries a `session_id` a caller can keep resuming
        with (the same one, or an updated one if the provider forks/rotates
        sessions on resume).
        """
