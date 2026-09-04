"""Live AgentSpec execution bookkeeping and transcript-derived run graphs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from symphonai_api.agent_spec import AgentSpec
from symphonai_api.cancellation import CancellationToken
from symphonai_api.cost import UsageTotals
from symphonai_api.identity import AgentRef, RunRef, new_agent_ref, new_run_ref
from symphonai_api.session import SessionStore, read_records


class RunPhase(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentRun:
    spec: AgentSpec
    agent: AgentRef
    run: RunRef
    phase: RunPhase = RunPhase.PENDING
    started_monotonic: float | None = None
    finished_monotonic: float | None = None
    last_heartbeat_monotonic: float | None = None
    turns_used: int = 0
    usage_by_model: dict[str, UsageTotals] = field(default_factory=dict)
    stopped_reason: str | None = None
    error: str | None = None
    token: CancellationToken | None = None

    @property
    def workspace_prefix(self) -> str | None:
        return self.spec.isolation.workspace_prefix

    @property
    def quiet_seconds(self) -> float | None:
        if self.phase is not RunPhase.RUNNING or self.last_heartbeat_monotonic is None:
            return None
        return time.monotonic() - self.last_heartbeat_monotonic

    def _require(self, method: str, *phases: RunPhase) -> None:
        if self.phase not in phases:
            expected = " or ".join(phase.value for phase in phases)
            raise ValueError(f"{method} requires {expected}; current phase is {self.phase.value}")

    def start(self, token: CancellationToken) -> None:
        self._require("start", RunPhase.PENDING)
        now = time.monotonic()
        self.phase = RunPhase.RUNNING
        self.started_monotonic = now
        self.last_heartbeat_monotonic = now
        self.token = token

    def heartbeat(self) -> None:
        self._require("heartbeat", RunPhase.RUNNING)
        self.last_heartbeat_monotonic = time.monotonic()

    def finish(self, result: object) -> None:
        self._require("finish", RunPhase.RUNNING)
        try:
            self.turns_used = result.turns_used
            self.usage_by_model = dict(result.usage_by_model)
            self.stopped_reason = result.stopped_reason
            self.finished_monotonic = time.monotonic()
            self.phase = RunPhase.FINISHED
        finally:
            if self.token is not None:
                self.token.close()

    def fail(self, error: str) -> None:
        self._require("fail", RunPhase.RUNNING)
        try:
            self.error = error
            self.finished_monotonic = time.monotonic()
            self.phase = RunPhase.FAILED
        finally:
            if self.token is not None:
                self.token.close()

    def cancel(self) -> None:
        self._require("cancel", RunPhase.PENDING, RunPhase.RUNNING)
        try:
            if self.token is not None:
                self.token.cancel()
            self.finished_monotonic = time.monotonic()
            self.phase = RunPhase.CANCELLED
        finally:
            if self.token is not None:
                self.token.close()


def new_agent_run(spec: AgentSpec, *, parent: AgentRun | None = None) -> AgentRun:
    agent = new_agent_ref(spec.name, parent.agent.agent_id if parent else None)
    run = new_run_ref(agent.agent_id, parent.run.run_id if parent else None)
    return AgentRun(spec, agent, run)


@dataclass(frozen=True)
class RunNode:
    run_id: str
    agent_id: str
    agent_name: str
    parent_run_id: str | None
    stopped_reason: str | None
    failed: bool
    children: tuple["RunNode", ...] = ()


def read_run_graph(store: SessionStore) -> tuple[RunNode, ...]:
    entries: dict[str, dict] = {}
    order: list[str] = []
    try:
        paths = sorted(store.directory.glob("*.jsonl"))
    except Exception:
        return ()
    for path in paths:
        try:
            records, _ = read_records(path)
        except Exception:
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            run_id, agent_id, data = record.get("run_id"), record.get("agent_id"), record.get("data")
            if not isinstance(run_id, str) or not isinstance(agent_id, str) or not isinstance(data, dict):
                continue
            if record.get("type") == "run_started" and run_id not in entries:
                entries[run_id] = {"agent": agent_id, "name": data.get("agent_name", ""), "parent": data.get("parent_run_id"), "reason": None, "failed": False}
                order.append(run_id)
            elif run_id in entries and record.get("type") == "run_finished":
                entries[run_id]["reason"] = data.get("stopped_reason")
            elif run_id in entries and record.get("type") == "run_failed":
                entries[run_id]["failed"] = True
    children = {run_id: [] for run_id in order}
    roots: list[str] = []
    for run_id in order:
        parent = entries[run_id]["parent"]
        if parent in children and parent != run_id:
            children[parent].append(run_id)
        else:
            roots.append(run_id)

    reachable: set[str] = set()

    def mark(run_id: str, seen: set[str]) -> None:
        if run_id in seen:
            return
        reachable.add(run_id)
        for child in children[run_id]:
            mark(child, seen | {run_id})

    for run_id in roots:
        mark(run_id, set())
    for run_id in order:
        if run_id not in reachable:
            roots.append(run_id)
            mark(run_id, set())
    def build(run_id: str, seen: set[str]) -> RunNode:
        item = entries[run_id]
        nested = tuple(
            build(child, seen | {run_id})
            for child in children[run_id]
            if child not in seen
        )
        return RunNode(run_id, item["agent"], item["name"], item["parent"], item["reason"], item["failed"], nested)
    return tuple(build(run_id, set()) for run_id in roots)
