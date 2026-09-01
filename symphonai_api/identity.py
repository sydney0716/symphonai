"""Canonical identities for persisted SymphonAI runtime records."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

SCHEMA_VERSION = 1


def new_id(prefix: str) -> str:
    """A unique id, prefixed for legibility in logs: ``run_3f2a...``."""
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class AgentRef:
    """Identity of one agent, and who created it."""

    agent_id: str
    name: str
    parent_agent_id: str | None = None
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class RunRef:
    """One execution of one agent. ``agent_id`` is the owner."""

    run_id: str
    agent_id: str
    parent_run_id: str | None = None
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class TurnRef:
    """One model call within a run. ``run_id`` is the owner."""

    turn_id: str
    run_id: str
    index: int
    schema_version: int = SCHEMA_VERSION


def new_agent_ref(name: str, parent_agent_id: str | None = None) -> AgentRef:
    return AgentRef(agent_id=new_id("agent"), name=name, parent_agent_id=parent_agent_id)


def new_run_ref(agent_id: str, parent_run_id: str | None = None) -> RunRef:
    return RunRef(run_id=new_id("run"), agent_id=agent_id, parent_run_id=parent_run_id)


def new_turn_ref(run_id: str, index: int) -> TurnRef:
    return TurnRef(turn_id=new_id("turn"), run_id=run_id, index=index)
