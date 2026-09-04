"""Checks for AgentRun lifecycle bookkeeping."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from symphonai_api.agent_loop import AgentRunResult
from symphonai_api.agent_run import RunPhase, new_agent_run, read_run_graph
from symphonai_api.agent_spec import AgentSpec, Isolation, ModelSelector
from symphonai_api.cancellation import CancelReason, CancellationToken
from symphonai_api.cost import UsageTotals
from symphonai_api.models import Message, ModelResponse, Role
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.session import SessionStore, TranscriptWriter
from scripts.checks.agent_spec import FORBIDDEN_IMPORTS, _forbidden_imports
from scripts.checks.harness import check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(root: Path, *, isolation: Isolation | None = None):
    return new_agent_run(
        AgentSpec(
            "worker",
            "",
            ModelSelector("fake"),
            PermissionPolicy(repo_root=root),
            isolation=isolation or Isolation(),
        )
    )


def _result(
    run,
    stopped_reason: str = "final_response",
    usage_by_model: dict[str, UsageTotals] | None = None,
) -> AgentRunResult:
    return AgentRunResult(
        final_response=ModelResponse(Message(Role.ASSISTANT, "done")),
        messages=[],
        turns_used=3,
        stopped_reason=stopped_reason,
        run=run.run,
        agent=run.agent,
        usage_by_model=usage_by_model or {},
    )


def _assert_transition_error(
    action,
    method: str,
    current: RunPhase,
    required: RunPhase,
) -> None:
    try:
        action()
    except ValueError as exc:
        message = str(exc)
    else:
        fail(f"{method} was allowed from {current.value}")
    for expected in (method, current.value, required.value):
        if expected not in message:
            fail(f"{method} error omitted {expected!r}: {message!r}")


@check("agent_run.identity_and_parenting")
def identity_and_parenting() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = _run(root, isolation=Isolation(workspace_prefix="workers/one"))
        child = new_agent_run(parent.spec, parent=parent)
        default = _run(root)
        if parent.phase is not RunPhase.PENDING:
            fail("root run did not begin pending")
        if parent.token is not None:
            fail("root run unexpectedly has a token")
        if parent.run.parent_run_id is not None:
            fail("root run unexpectedly has a parent")
        if child.agent.parent_agent_id != parent.agent.agent_id:
            fail("child agent ref was not parented")
        if child.run.parent_run_id != parent.run.run_id:
            fail("child run ref was not parented")
        if parent.workspace_prefix != "workers/one":
            fail("workspace prefix did not come from isolation")
        if default.workspace_prefix is not None:
            fail("default workspace prefix was not None")


@check("agent_run.lifecycle_transitions")
def lifecycle_transitions() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        pending = _run(root)
        _assert_transition_error(
            pending.heartbeat,
            "heartbeat",
            RunPhase.PENDING,
            RunPhase.RUNNING,
        )
        _assert_transition_error(
            lambda: pending.fail("broken"),
            "fail",
            RunPhase.PENDING,
            RunPhase.RUNNING,
        )

        finished = _run(root)
        finished.start(CancellationToken())
        finished.heartbeat()
        if finished.phase is not RunPhase.RUNNING:
            fail("heartbeat did not retain the running phase")
        finished.finish(_result(finished))
        _assert_transition_error(
            lambda: finished.finish(_result(finished)),
            "finish",
            RunPhase.FINISHED,
            RunPhase.RUNNING,
        )
        _assert_transition_error(
            lambda: finished.start(CancellationToken()),
            "start",
            RunPhase.FINISHED,
            RunPhase.PENDING,
        )

        failed = _run(root)
        failed.start(CancellationToken())
        failed.fail("broken")
        _assert_transition_error(
            lambda: failed.start(CancellationToken()),
            "start",
            RunPhase.FAILED,
            RunPhase.PENDING,
        )

        cancelled = _run(root)
        cancelled.cancel()
        _assert_transition_error(
            lambda: cancelled.start(CancellationToken()),
            "start",
            RunPhase.CANCELLED,
            RunPhase.PENDING,
        )


@check("agent_run.result_and_cancellation")
def result_and_cancellation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for stopped_reason in (
            "final_response",
            "max_turns",
            "cancelled",
            "budget_tokens",
        ):
            run = _run(root)
            usage_by_model = {"fake": UsageTotals(2, 3, 1)}
            result = _result(run, stopped_reason, usage_by_model)
            run.start(CancellationToken())
            if not isinstance(run.quiet_seconds, float) or run.quiet_seconds < 0:
                fail("running quiet_seconds was not a non-negative float")
            run.finish(result)
            if run.turns_used != 3:
                fail("finish did not copy turns_used")
            if run.usage_by_model != usage_by_model:
                fail("finish did not copy usage_by_model")
            if run.stopped_reason != stopped_reason:
                fail("finish did not copy stopped_reason")
            if run.quiet_seconds is not None:
                fail("terminal run still reported quiet_seconds")
            usage_by_model["later"] = UsageTotals(5, 8, 1)
            if "later" in run.usage_by_model:
                fail("run retained the result usage mapping")

        no_token = _run(root)
        no_token.cancel()
        if no_token.phase is not RunPhase.CANCELLED:
            fail("cancelling a tokenless run did not reach CANCELLED")

        token = CancellationToken()
        token.cancel()
        already_cancelled = _run(root)
        already_cancelled.start(token)
        already_cancelled.cancel()
        if token.reason is not CancelReason.EXPLICIT:
            fail("run cancellation overwrote the token's first cause")


@check("agent_run.token_closed_in_finally")
def token_closed_in_finally() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = CancellationToken()
        child = parent.child()
        bad_result = _run(root)
        bad_result.start(child)

        class BadResult:
            @property
            def turns_used(self):
                raise RuntimeError("copy failed")

        try:
            bad_result.finish(BadResult())
        except RuntimeError:
            pass
        else:
            fail("bad finish result did not raise")
        if parent._callbacks:
            fail("failed finish retained child listener")

        for terminal in ("finish", "fail", "cancel"):
            parent = CancellationToken()
            callbacks_before_child = len(parent._callbacks)
            child = parent.child()
            run = _run(root)
            run.start(child)
            if terminal == "finish":
                run.finish(_result(run))
            elif terminal == "fail":
                run.fail("broken")
            else:
                run.cancel()
            if len(parent._callbacks) != callbacks_before_child:
                fail(f"{terminal} retained a child cancellation listener")


@check("agent_run.parent_close_does_not_detach_children")
def parent_close_does_not_detach_children() -> None:
    parent = CancellationToken()
    child = parent.child()
    parent.close()
    parent.cancel()
    if not child.cancelled:
        fail("parent close detached child")
    if child.reason is not CancelReason.PARENT:
        fail("parent cancellation did not retain its cause")


def _append_started(
    writer: TranscriptWriter,
    run_id: str,
    agent_id: str,
    parent: str | None = None,
) -> None:
    writer.append(
        "run_started",
        run_id=run_id,
        agent_id=agent_id,
        turn_id=None,
        data={"agent_name": agent_id, "parent_run_id": parent, "model": "fake"},
    )


def _append_raw_record(writer: TranscriptWriter, record: dict) -> None:
    writer.close()
    with writer.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _node_count(nodes) -> int:
    return sum(1 + _node_count(node.children) for node in nodes)


@check("agent_run.graph_from_transcripts")
def graph_from_transcripts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = SessionStore(Path(temporary), "session")
        _append_started(store.writer_for("root", is_root=True), "root", "root")
        _append_started(store.writer_for("child"), "child", "child", "root")
        graph = read_run_graph(store)
        if len(graph) != 1:
            fail(f"graph had the wrong root count: {graph!r}")
        if [node.run_id for node in graph[0].children] != ["child"]:
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


@check("agent_run.graph_is_total")
def graph_is_total() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        def empty(store: SessionStore) -> None:
            return

        def duplicate_started(store: SessionStore) -> None:
            writer = store.writer_for("root", is_root=True)
            _append_started(writer, "one", "one")
            _append_started(writer, "one", "different")

        def missing_run_id(store: SessionStore) -> None:
            _append_raw_record(
                store.writer_for("missing-run-id"),
                {"type": "run_started", "agent_id": "agent", "data": {}},
            )

        def missing_agent_id(store: SessionStore) -> None:
            _append_raw_record(
                store.writer_for("missing-agent-id"),
                {"type": "run_started", "run_id": "run", "data": {}},
            )

        def missing_data(store: SessionStore) -> None:
            _append_raw_record(
                store.writer_for("missing-data"),
                {"type": "run_started", "run_id": "run", "agent_id": "agent"},
            )

        def truncated_final_line(store: SessionStore) -> None:
            writer = store.writer_for("truncated", is_root=True)
            _append_started(writer, "valid", "valid")
            writer.close()
            with writer.path.open("a", encoding="utf-8") as handle:
                handle.write('{"type":"run_started"')

        cases = (
            ("empty directory", empty, 0),
            ("duplicate run_started", duplicate_started, 1),
            ("missing run_id", missing_run_id, 0),
            ("missing agent_id", missing_agent_id, 0),
            ("missing data", missing_data, 0),
            ("truncated final line", truncated_final_line, 1),
        )
        for index, (defect, populate, expected_nodes) in enumerate(cases):
            store = SessionStore(root, f"case-{index}")
            populate(store)
            actual_nodes = _node_count(read_run_graph(store))
            if actual_nodes != expected_nodes:
                fail(
                    f"{defect} produced {actual_nodes} nodes; "
                    f"expected {expected_nodes}"
                )


@check("agent_run.no_runtime_imports")
def no_runtime_imports() -> None:
    source = (REPO_ROOT / "symphonai_api/agent_run.py").read_text()
    found = _forbidden_imports(source)
    if found:
        fail(f"agent_run imports runtime wiring: {found!r}")
    probes = [
        ("from symphonai_api.runner import z\n", True),
        ("from symphonai_api.leader import z\n", True),
        ("from symphonai_api.agent_loop import z\n", True),
        ("from symphonai_api.providers.openai import z\n", True),
        ("import symphonai_api.runner\n", True),
        ("from . import runner\n", True),
        ("from symphonai_api import runner\n", True),
        ("from symphonai_api import leader, budgets\n", True),
        ("import symphonai_api.budgets\n", False),
        ("from symphonai_api.budgets import RunBudget\n", False),
    ]
    if FORBIDDEN_IMPORTS != {
        "agent_loop",
        "leader",
        "runner",
        "provider_catalog",
        "providers",
    }:
        fail("shared forbidden import set changed")
    for line, expected in probes:
        if bool(_forbidden_imports(line)) != expected:
            fail(f"import inspection got {line.strip()!r} wrong")
