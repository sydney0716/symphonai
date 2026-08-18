"""A fully in-memory SubagentProvider for tests and local development.

`FakeProvider` never shells out, never touches the network, and never reads
the filesystem beyond its own in-memory state. Given the same sequence of
calls, it always produces the same `session_id` and events -- no randomness,
no wall-clock dependence -- so callers can exercise the `SubagentProvider`
contract deterministically without a real vendor CLI or API key.
"""

from __future__ import annotations

from orchestra_agents.base import SubagentProvider
from orchestra_agents.models import AgentEvent, AgentRunResult, AgentTask, EventKind


def _derive_session_id(task_id: str) -> str:
    return f"fake-session-{task_id}"


class FakeProvider(SubagentProvider):
    """Deterministic in-memory provider that echoes each turn's prompt back."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return "fake"

    def spawn(self, task: AgentTask) -> AgentRunResult:
        session_id = _derive_session_id(task.task_id)
        history = [task.prompt]
        self._sessions[session_id] = history
        return self._result_for(session_id, history)

    def resume(self, session_id: str, follow_up: str) -> AgentRunResult:
        history = self._sessions.get(session_id)
        if history is None:
            return AgentRunResult(
                session_id=session_id,
                ok=False,
                events=[
                    AgentEvent(
                        session_id=session_id,
                        kind=EventKind.ERROR,
                        data={"reason": "unknown session_id"},
                    )
                ],
                final_message="",
                error=f"unknown session_id: {session_id!r}",
            )
        history.append(follow_up)
        return self._result_for(session_id, history)

    def _result_for(self, session_id: str, history: list[str]) -> AgentRunResult:
        events = [
            AgentEvent(
                session_id=session_id,
                kind=EventKind.MESSAGE,
                data={"text": text, "turn": turn},
            )
            for turn, text in enumerate(history)
        ]
        events.append(
            AgentEvent(
                session_id=session_id,
                kind=EventKind.DONE,
                data={"turn_count": len(history)},
            )
        )
        final_message = f"fake response to: {history[-1]}"
        return AgentRunResult(
            session_id=session_id,
            ok=True,
            events=events,
            final_message=final_message,
        )
