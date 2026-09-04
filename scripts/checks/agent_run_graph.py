"""Checks for AgentRun lifecycle bookkeeping."""
from __future__ import annotations
import tempfile
from pathlib import Path
from symphonai_api.agent_run import RunPhase, new_agent_run
from symphonai_api.agent_run import read_run_graph
from symphonai_api.agent_spec import AgentSpec, ModelSelector
from symphonai_api.cancellation import CancellationToken
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.session import SessionStore
from scripts.checks.harness import check, fail

def _run(root: Path):
    return new_agent_run(AgentSpec("worker", "", ModelSelector("fake"), PermissionPolicy(repo_root=root)))

@check("agent_run.identity_and_parenting")
def identity_and_parenting() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        parent = _run(Path(temporary)); child = new_agent_run(parent.spec, parent=parent)
        if parent.phase is not RunPhase.PENDING or parent.token is not None or parent.run.parent_run_id is not None: fail("root run defaults changed")
        if child.agent.parent_agent_id != parent.agent.agent_id or child.run.parent_run_id != parent.run.run_id: fail("child refs were not parented")

@check("agent_run.lifecycle_transitions")
def lifecycle_transitions() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run = _run(Path(temporary)); run.start(CancellationToken()); started = run.started_monotonic; run.heartbeat()
        if run.phase is not RunPhase.RUNNING or run.last_heartbeat_monotonic < started: fail("heartbeat lifecycle failed")
        run.fail("broken")
        try: run.start(CancellationToken())
        except ValueError: return
        fail("terminal run restarted")

@check("agent_run.result_and_cancellation")
def result_and_cancellation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run = _run(Path(temporary)); token = CancellationToken(); run.start(token); run.cancel()
        if run.phase is not RunPhase.CANCELLED or not token.cancelled or run.quiet_seconds is not None: fail("cancellation state failed")

@check("agent_run.token_closed_in_finally")
def token_closed_in_finally() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        parent = CancellationToken(); child = parent.child(); run = _run(Path(temporary)); run.start(child)
        class BadResult:
            @property
            def turns_used(self): raise RuntimeError("copy failed")
        try: run.finish(BadResult())
        except RuntimeError: pass
        if parent._callbacks: fail("failed finish retained child listener")

@check("agent_run.parent_close_does_not_detach_children")
def parent_close_does_not_detach_children() -> None:
    parent = CancellationToken(); child = parent.child(); parent.close(); parent.cancel()
    if not child.cancelled: fail("parent close detached child")


def _append_started(writer, run_id, agent_id, parent=None):
    writer.append("run_started", run_id=run_id, agent_id=agent_id, turn_id=None, data={"agent_name": agent_id, "parent_run_id": parent, "model": "fake"})


@check("agent_run.graph_from_transcripts")
def graph_from_transcripts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = SessionStore(Path(temporary), "session")
        _append_started(store.writer_for("root", is_root=True), "root", "root")
        _append_started(store.writer_for("child"), "child", "child", "root")
        graph = read_run_graph(store)
        if len(graph) != 1 or [node.run_id for node in graph[0].children] != ["child"]:
            fail(f"graph missed agent transcript: {graph!r}")


@check("agent_run.graph_covers_every_run")
def graph_covers_every_run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = SessionStore(Path(temporary), "session")
        writer = store.writer_for("a", is_root=True)
        _append_started(writer, "x", "x", "y")
        _append_started(writer, "y", "y", "x")
        _append_started(writer, "z", "z", "x")
        def flatten(nodes):
            flattened = []
            for node in nodes:
                flattened.append(node.run_id)
                flattened.extend(flatten(node.children))
            return flattened
        if sorted(flatten(read_run_graph(store))) != ["x", "y", "z"]:
            fail("cycle graph did not retain every run")
