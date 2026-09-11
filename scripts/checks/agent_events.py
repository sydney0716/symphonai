"""Workspace-backed checks for agent events."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
import unittest.mock as mock

from symphonai_api.agent_loop import ApiAgent
from symphonai_api.agent_run import RunPhase
from symphonai_api.cancellation import OperationCancelled
from symphonai_api.events import (
    CollectingSink,
    Event,
    PermissionDenied,
    PermissionRequested,
    PromptSubmitted,
    RunFailed,
    RunFinished,
    RunStarted,
    SessionEnded,
    SessionStarted,
    SubagentSpawned,
    SubagentStopped,
    ToolCallFailed,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
    TurnStarted,
    fan_out,
)
from symphonai_api.identity import SCHEMA_VERSION, TurnRef
from symphonai_api.leader import Leader, LeaderConfig
from symphonai_api.models import (
    Message,
    ModelResponse,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import ModelProvider, ProviderError
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.runner import standard_tool_registry
from symphonai_api.session import SessionStore
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


class _RaisingProvider(ModelProvider):
    @property
    def name(self) -> str:
        return "raising"

    @property
    def wire_format(self) -> int:
        return 4

    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        raise ProviderError("event test failure")


@check("events.fan_out")
def check_fan_out() -> None:
    event = Event(agent_id="agent", run_id="run")
    if fan_out(None, None) is not None:
        fail("all-None fan_out did not preserve the absence of a sink")

    single = CollectingSink()
    if fan_out(None, single, None) is not single:
        fail("fan_out wrapped a single sink")

    calls: list[str] = []

    def raising(_: Event) -> None:
        calls.append("raising")
        raise RuntimeError("observer failed")

    def second(_: Event) -> None:
        calls.append("second")

    def third(_: Event) -> None:
        calls.append("third")

    combined = fan_out(raising, second, third)
    if combined is None:
        fail("fan_out dropped three live sinks")
    combined(event)
    if calls != ["raising", "second", "third"]:
        fail(f"fan_out did not isolate sinks in argument order: {calls!r}")


@check("events.final_identity")
def check_events_final_identity() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        final_events = CollectingSink()
        final_event_result = ApiAgent(
            FakeModelProvider(
                responses=[ModelResponse(Message(Role.ASSISTANT, "event answer"))]
            ),
            {},
            policy,
            events=final_events,
        ).run([Message(Role.USER, "event test")])
        if len(final_events.of_type(RunStarted)) != 1:
            fail(f"final run did not emit one RunStarted: {final_events.events!r}")
        final_terminals = final_events.of_type(RunFinished) + final_events.of_type(RunFailed)
        if len(final_terminals) != 1 or final_terminals[0].stopped_reason != "final_response":
            fail(f"final run terminal events are invalid: {final_events.events!r}")
        if any(
            event.agent_id != final_event_result.agent.agent_id
            or event.run_id != final_event_result.run.run_id
            for event in final_events.events
        ):
            fail(f"event identity does not match the agent result: {final_events.events!r}")
        turn_ids = {
            message.turn_id
            for message in final_event_result.messages
            if message.turn_id is not None
        }
        if any(
            event.turn_id is not None and event.turn_id not in turn_ids
            for event in final_events.events
        ):
            fail(f"event turn identity does not match the transcript: {final_events.events!r}")
        if len(final_events.of_type(TurnStarted)) != 1 or len(final_events.of_type(TurnFinished)) != 1:
            fail(f"completed final turn was not bracketed: {final_events.events!r}")

@check("events.tool_bracketing")
def check_events_tool_bracketing() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools

        def run_call(tool_call: ToolCall) -> tuple[object, CollectingSink]:
            recorded = CollectingSink()
            result = ApiAgent(
                FakeModelProvider(
                    responses=[
                        ModelResponse(
                            Message(Role.ASSISTANT, tool_calls=[tool_call])
                        ),
                        ModelResponse(Message(Role.ASSISTANT, "done")),
                    ]
                ),
                tools,
                policy,
                events=recorded,
            ).run([Message(Role.USER, "tool event")])
            return result, recorded

        tool_event_result, tool_events = run_call(
            ToolCall(
                id="event-read",
                name="read_file",
                arguments={"path": "existing.txt"},
            )
        )
        tool_started = tool_events.of_type(ToolCallStarted)
        tool_finished = tool_events.of_type(ToolCallFinished)
        default_result_fields = ("", "", 0, 0, "", False)
        if (
            len(tool_started) != 1
            or len(tool_finished) != 1
            or tool_started[0].tool_call_id != "event-read"
            or tool_started[0].target != "existing.txt"
            or tool_finished[0].tool_call_id != "event-read"
            or not tool_finished[0].ok
            or (
                tool_finished[0].result_kind,
                tool_finished[0].result_path,
                tool_finished[0].lines_added,
                tool_finished[0].lines_removed,
                tool_finished[0].diff,
                tool_finished[0].truncated,
            )
            != default_result_fields
        ):
            fail(f"successful tool events did not bracket execution: {tool_events.events!r}")
        if tool_event_result.stopped_reason != "final_response":
            fail(f"event-producing tool run changed its result: {tool_event_result!r}")

        edit_result, edit_events = run_call(
            ToolCall(
                id="event-edit",
                name="edit_file",
                arguments={
                    "path": "existing.txt",
                    "old_string": "hello",
                    "new_string": "goodbye",
                },
            )
        )
        edit_started = edit_events.of_type(ToolCallStarted)
        edit_finished = edit_events.of_type(ToolCallFinished)
        edit_messages = [
            message.tool_result
            for message in edit_result.messages
            if message.tool_result is not None
        ]
        if (
            len(edit_started) != 1
            or edit_started[0].target != "existing.txt"
            or len(edit_finished) != 1
            or edit_finished[0].result_kind != "file_diff"
            or edit_finished[0].result_path != "existing.txt"
            or edit_finished[0].lines_added != 1
            or edit_finished[0].lines_removed != 1
            or edit_finished[0].truncated
            or len(edit_messages) != 1
            or edit_finished[0].diff != edit_messages[0].content
            or "-hello from disk" not in edit_finished[0].diff
            or "+goodbye from disk" not in edit_finished[0].diff
        ):
            fail(f"edit result did not reach its finish event: {edit_events.events!r}")

        failed_payload = {
            "kind": "file_diff",
            "path": "decoy.txt",
            "lines_added": 8,
            "lines_removed": 5,
            "truncated": True,
        }
        with mock.patch.object(
            tools["edit_file"],
            "_execute",
            return_value=ToolResult(
                tool_call_id="event-failed-edit",
                ok=False,
                content="decoy diff",
                error="edit failed",
                payload=failed_payload,
            ),
        ):
            _, failed_edit_events = run_call(
                ToolCall(
                    id="event-failed-edit",
                    name="edit_file",
                    arguments={
                        "path": "existing.txt",
                        "old_string": "goodbye",
                        "new_string": "hello",
                    },
                )
            )
        failed_finished = failed_edit_events.of_type(ToolCallFinished)
        if len(failed_finished) != 1 or (
            failed_finished[0].result_kind,
            failed_finished[0].result_path,
            failed_finished[0].lines_added,
            failed_finished[0].lines_removed,
            failed_finished[0].diff,
            failed_finished[0].truncated,
        ) != default_result_fields:
            fail(f"failed edit leaked a diff summary: {failed_edit_events.events!r}")

        with mock.patch.object(
            tools["read_file"],
            "_execute",
            return_value=ToolResult(
                tool_call_id="event-non-diff",
                ok=True,
                content="ordinary result",
                payload={
                    "kind": "search_hits",
                    "path": "decoy.txt",
                    "lines_added": 3,
                    "lines_removed": 2,
                    "truncated": True,
                },
            ),
        ):
            _, non_diff_events = run_call(
                ToolCall(
                    id="event-non-diff",
                    name="read_file",
                    arguments={"path": "existing.txt"},
                )
            )
        non_diff_finished = non_diff_events.of_type(ToolCallFinished)
        if len(non_diff_finished) != 1 or (
            non_diff_finished[0].result_kind,
            non_diff_finished[0].result_path,
            non_diff_finished[0].lines_added,
            non_diff_finished[0].lines_removed,
            non_diff_finished[0].diff,
            non_diff_finished[0].truncated,
        ) != default_result_fields:
            fail(f"non-diff result leaked a diff summary: {non_diff_events.events!r}")

        unknown_events = CollectingSink()
        unknown_result = ApiAgent(
            FakeModelProvider(
                responses=[
                    ModelResponse(
                        Message(
                            Role.ASSISTANT,
                            tool_calls=[ToolCall(id="event-unknown", name="missing")],
                        )
                    )
                ]
            ),
            {},
            policy,
            max_turns=1,
            events=unknown_events,
        ).run([Message(Role.USER, "unknown")])
        unknown_finished = unknown_events.of_type(ToolCallFinished)
        if len(unknown_finished) != 1 or unknown_finished[0].ok:
            fail(f"unknown-tool completion event was not ok=False: {unknown_events.events!r}")
        unknown_terminals = unknown_events.of_type(RunFinished) + unknown_events.of_type(RunFailed)
        if (
            unknown_result.stopped_reason != "max_turns"
            or len(unknown_terminals) != 1
            or unknown_terminals[0].stopped_reason != "max_turns"
        ):
            fail(f"max-turn event terminal is invalid: {unknown_events.events!r}")

@check("events.provider_failure")
def check_events_provider_failure() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        failed_events = CollectingSink()
        try:
            ApiAgent(_RaisingProvider(), {}, policy, events=failed_events).run(
                [Message(Role.USER, "fail")]
            )
        except ProviderError:
            pass
        else:
            fail("raising provider did not propagate ProviderError")
        failed_terminals = failed_events.of_type(RunFinished) + failed_events.of_type(RunFailed)
        if len(failed_events.of_type(RunStarted)) != 1 or len(failed_terminals) != 1:
            fail(f"failed run terminal cardinality is invalid: {failed_events.events!r}")
        if not isinstance(failed_terminals[0], RunFailed):
            fail(f"raising provider did not emit RunFailed: {failed_events.events!r}")

@check("events.sink_isolation")
def check_events_sink_isolation() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        def _broken_sink(event) -> None:  # noqa: ANN001
            raise RuntimeError("broken observer")

        broken_sink_result = ApiAgent(
            FakeModelProvider(
                responses=[ModelResponse(Message(Role.ASSISTANT, "still works"))]
            ),
            {},
            policy,
            events=_broken_sink,
        ).run([Message(Role.USER, "ignore observer")])
        if broken_sink_result.stopped_reason != "final_response":
            fail(f"raising event sink broke the run: {broken_sink_result!r}")

        def _cancelling_sink(event) -> None:  # noqa: ANN001
            raise OperationCancelled

        sink_cancel_result = ApiAgent(
            FakeModelProvider(), {}, policy, events=_cancelling_sink
        ).run([Message(Role.USER, "cancel from sink")])
        if sink_cancel_result.stopped_reason != "cancelled":
            fail(f"OperationCancelled from sink was swallowed: {sink_cancel_result!r}")

@check("events.stream_optional")
def check_events_stream_optional() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        inert_response = ModelResponse(Message(Role.ASSISTANT, "same result"))

        def _fixed_turn(run_id: str, index: int) -> TurnRef:
            return TurnRef(turn_id=f"turn_fixed_{index}", run_id=run_id, index=index)

        with mock.patch("symphonai_api.agent_loop.new_turn_ref", side_effect=_fixed_turn):
            without_events = ApiAgent(
                FakeModelProvider([inert_response]), {}, policy
            ).run([Message(Role.USER, "same input")])
            with_events = ApiAgent(
                FakeModelProvider([inert_response]), {}, policy, events=CollectingSink()
            ).run([Message(Role.USER, "same input")])
        if (
            without_events.messages != with_events.messages
            or without_events.stopped_reason != with_events.stopped_reason
        ):
            fail("collecting events changed the agent result")


@check("events.new_types_and_schema")
def new_types_and_schema() -> None:
    event_types = (
        PromptSubmitted,
        ToolCallFailed,
        PermissionRequested,
        PermissionDenied,
        SessionStarted,
        SessionEnded,
        SubagentStopped,
    )
    if SCHEMA_VERSION != 1:
        fail(f"event additions changed SCHEMA_VERSION: {SCHEMA_VERSION}")
    base_names = {"agent_id", "run_id", "turn_id", "schema_version"}
    for event_type in event_types:
        event = event_type(agent_id="agent", run_id="run", turn_id="turn")
        if not isinstance(event, Event):
            fail(f"{event_type.__name__} is not an Event")
        if not base_names.issubset({item.name for item in fields(event_type)}):
            fail(f"{event_type.__name__} omitted base event fields")
        if event.schema_version != 1:
            fail(f"{event_type.__name__} reported schema {event.schema_version}")
        try:
            event.run_id = "changed"  # type: ignore[misc]
        except FrozenInstanceError:
            pass
        else:
            fail(f"{event_type.__name__} was mutable")


@check("events.prompt_submitted")
def prompt_submitted() -> None:
    with workspace() as ws:
        sink = CollectingSink()
        messages = [
            Message(Role.SYSTEM, "system"),
            Message(Role.USER, "earlier"),
            Message(Role.ASSISTANT, "answer"),
            Message(Role.USER, "submitted now"),
        ]
        ApiAgent(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            {},
            ws.policy,
            events=sink,
        ).run(messages)
        submitted = sink.of_type(PromptSubmitted)
        if (
            len(submitted) != 1
            or submitted[0].text != "submitted now"
            or submitted[0].message_count != 3
        ):
            fail(f"prompt event payload differed: {sink.events!r}")
        first_turn = next(
            index
            for index, event in enumerate(sink.events)
            if isinstance(event, TurnStarted)
        )
        if sink.events.index(submitted[0]) >= first_turn:
            fail(f"prompt event did not precede the first turn: {sink.events!r}")


@check("events.tool_failure_is_additional")
def tool_failure_is_additional() -> None:
    with workspace() as ws:
        sink = CollectingSink()
        result = ApiAgent(
            FakeModelProvider(
                [
                    ModelResponse(
                        Message(
                            Role.ASSISTANT,
                            tool_calls=[
                                ToolCall(
                                    id="successful-read",
                                    name="read_file",
                                    arguments={"path": "existing.txt"},
                                ),
                                ToolCall(id="failed-missing", name="missing"),
                            ],
                        )
                    ),
                    ModelResponse(Message(Role.ASSISTANT, "done")),
                ]
            ),
            standard_tool_registry(),
            ws.policy,
            events=sink,
        ).run([Message(Role.USER, "use tools")])
        if result.stopped_reason != "final_response":
            fail(f"tool event run changed result: {result!r}")
        finished = {
            event.tool_call_id: event
            for event in sink.of_type(ToolCallFinished)
        }
        failed = sink.of_type(ToolCallFailed)
        if set(finished) != {"successful-read", "failed-missing"}:
            fail(f"tool completions were replaced or duplicated: {sink.events!r}")
        if not finished["successful-read"].ok or finished["failed-missing"].ok:
            fail(f"tool completion status changed: {finished!r}")
        if len(failed) != 1 or failed[0].tool_call_id != "failed-missing":
            fail(f"tool failure event cardinality differed: {sink.events!r}")
        finished_index = sink.events.index(finished["failed-missing"])
        if sink.events[finished_index + 1] is not failed[0]:
            fail(f"ToolCallFailed did not immediately follow failure finish: {sink.events!r}")
        if any(event.tool_call_id == "successful-read" for event in failed):
            fail("a successful tool emitted ToolCallFailed")


@check("events.session_lifecycle")
def session_lifecycle() -> None:
    with workspace() as ws:
        sink = CollectingSink()
        store = SessionStore(ws.root / "sessions", "session-events", events=sink)
        if sink.of_type(SessionEnded):
            fail("SessionEnded fired during construction")
        started = sink.of_type(SessionStarted)
        if (
            len(started) != 1
            or started[0].session_run_id != store.run_id
            or started[0].run_id != store.run_id
        ):
            fail(f"SessionStarted payload differed: {sink.events!r}")
        store.close()
        store.close()
        ended = sink.of_type(SessionEnded)
        if len(ended) != 1 or ended[0].session_run_id != store.run_id:
            fail(f"SessionEnded cardinality or payload differed: {sink.events!r}")

        reopened_sink = CollectingSink()
        reopened = SessionStore.open(
            ws.root / "sessions",
            store.run_id,
            events=reopened_sink,
        )
        reopened.close()
        if (
            len(reopened_sink.of_type(SessionStarted)) != 1
            or len(reopened_sink.of_type(SessionEnded)) != 1
        ):
            fail(f"opened session lifecycle differed: {reopened_sink.events!r}")


def _one_subagent_leader(root: Path, events) -> Leader:
    return Leader(
        LeaderConfig(
            leader_provider=FakeModelProvider(
                [
                    ModelResponse(
                        Message(
                            Role.ASSISTANT,
                            tool_calls=[
                                ToolCall(
                                    id="dispatch-one",
                                    name="dispatch_subagent",
                                    arguments={
                                        "subagent_name": "worker",
                                        "task": "work",
                                    },
                                )
                            ],
                        )
                    ),
                    ModelResponse(Message(Role.ASSISTANT, "leader done")),
                ]
            ),
            subagent_provider=FakeModelProvider(
                [ModelResponse(Message(Role.ASSISTANT, "worker done"))]
            ),
            repo_root=str(root),
            events=events,
        )
    )


@check("events.subagent_stopped")
def subagent_stopped() -> None:
    with workspace() as ws:
        sink = CollectingSink()
        result = _one_subagent_leader(ws.root, sink).run("delegate")
        spawned = sink.of_type(SubagentSpawned)
        stopped = sink.of_type(SubagentStopped)
        if len(spawned) != 1 or len(stopped) != 1:
            fail(f"subagent boundary events differed: {sink.events!r}")
        if (
            spawned[0].agent_id != stopped[0].agent_id
            or spawned[0].run_id != stopped[0].run_id
            or spawned[0].turn_id != stopped[0].turn_id
            or spawned[0].subagent_name != stopped[0].subagent_name
            or spawned[0].subagent_agent_id != stopped[0].subagent_agent_id
        ):
            fail(f"SubagentStopped did not mirror SubagentSpawned: {sink.events!r}")
        child_finished = next(
            index
            for index, event in enumerate(sink.events)
            if isinstance(event, RunFinished)
            and event.agent_id == stopped[0].subagent_agent_id
        )
        if sink.events.index(stopped[0]) <= child_finished:
            fail(f"SubagentStopped preceded the child terminal: {sink.events!r}")
        if result.subagents["worker"].runs[0].phase is not RunPhase.FINISHED:
            fail("SubagentStopped fired without a finished child run")


@check("events.new_sink_isolation")
def new_sink_isolation() -> None:
    with workspace() as ws:
        seen: list[type[Event]] = []

        def raising_sink(event: Event) -> None:
            seen.append(type(event))
            raise RuntimeError("observer failed")

        tool_result = ApiAgent(
            FakeModelProvider(
                [
                    ModelResponse(
                        Message(
                            Role.ASSISTANT,
                            tool_calls=[ToolCall(id="failed", name="missing")],
                        )
                    ),
                    ModelResponse(Message(Role.ASSISTANT, "done")),
                ]
            ),
            {},
            ws.policy,
            events=raising_sink,
        ).run([Message(Role.USER, "fail a tool")])
        if tool_result.stopped_reason != "final_response":
            fail("raising sink changed the tool-failure run")

        policy = PermissionPolicy(
            ws.root,
            mode="prompt",
            approval_callback=lambda request: False,
        )
        policy.attach_event_sink(
            raising_sink,
            agent_id="permission-agent",
            run_id="permission-run",
        )
        if policy.check_write("new.txt").allowed:
            fail("raising sink changed a permission denial")

        store = SessionStore(
            ws.root / "raising-sessions",
            "raising-session",
            events=raising_sink,
        )
        store.close()

        leader_result = _one_subagent_leader(ws.root, raising_sink).run("delegate")
        if leader_result.stopped_reason != "final_response":
            fail("raising sink changed a subagent run")
        expected = {
            PromptSubmitted,
            ToolCallFailed,
            PermissionRequested,
            PermissionDenied,
            SessionStarted,
            SessionEnded,
            SubagentStopped,
        }
        if not expected.issubset(seen):
            missing = sorted(item.__name__ for item in expected - set(seen))
            fail(f"raising sink did not observe every new event: {missing!r}")


def _message_snapshot(message: Message):
    tool_result = message.tool_result
    return (
        message.role.value,
        message.text,
        tuple((call.id, call.name) for call in message.tool_calls),
        (
            None
            if tool_result is None
            else (tool_result.tool_call_id, tool_result.ok, tool_result.error)
        ),
    )


_PRE_10B_COMMIT = "ff3163a"
_FROZEN_PRE_10B_OUTCOME = {
    "leader_messages": (
        ("user", "goal", (), None),
        ("assistant", "", (("dispatch-main", "dispatch_subagent"),), None),
        ("tool", "", (), ("dispatch-main", True, None)),
        ("assistant", "leader done", (), None),
    ),
    "pool": {
        "worker": {
            "messages": (
                ("user", "inspect", (), None),
                (
                    "assistant",
                    "",
                    (
                        ("successful-read", "read_file"),
                        ("failed-missing", "missing"),
                        ("denied-write", "write_file"),
                    ),
                    None,
                ),
                ("tool", "", (), ("successful-read", True, None)),
                (
                    "tool",
                    "",
                    (),
                    ("failed-missing", False, "unknown tool: 'missing'"),
                ),
                (
                    "tool",
                    "",
                    (),
                    (
                        "denied-write",
                        False,
                        "path is outside the explicit allowed write scope: "
                        "'denied.txt'",
                    ),
                ),
                ("assistant", "child done", (), None),
            ),
            "turns": 2,
            "phases": ("finished",),
            "breaker": 0,
        }
    },
    "stop": "final_response",
    "usage": {
        "leader": {"unknown": (4, 6)},
        "worker": {"unknown": (12, 14)},
    },
    "session_stop": "final_response",
}


def _scripted_leader_outcome(root: Path, events, session_id: str):
    session = SessionStore(root / "scripted-sessions", session_id, events=events)
    leader = Leader(
        LeaderConfig(
            leader_provider=FakeModelProvider(
                [
                    ModelResponse(
                        Message(
                            Role.ASSISTANT,
                            tool_calls=[
                                ToolCall(
                                    id="dispatch-main",
                                    name="dispatch_subagent",
                                    arguments={
                                        "subagent_name": "worker",
                                        "task": "inspect",
                                    },
                                )
                            ],
                        ),
                        usage=Usage(1, 2),
                    ),
                    ModelResponse(
                        Message(Role.ASSISTANT, "leader done"),
                        usage=Usage(3, 4),
                    ),
                ]
            ),
            subagent_provider=FakeModelProvider(
                [
                    ModelResponse(
                        Message(
                            Role.ASSISTANT,
                            tool_calls=[
                                ToolCall(
                                    id="successful-read",
                                    name="read_file",
                                    arguments={"path": "existing.txt"},
                                ),
                                ToolCall(id="failed-missing", name="missing"),
                                ToolCall(
                                    id="denied-write",
                                    name="write_file",
                                    arguments={"path": "denied.txt", "content": "x"},
                                ),
                            ],
                        ),
                        usage=Usage(5, 6),
                    ),
                    ModelResponse(
                        Message(Role.ASSISTANT, "child done"),
                        usage=Usage(7, 8),
                    ),
                ]
            ),
            repo_root=str(root),
            events=events,
        ),
        session=session,
    )
    result = leader.run("goal")
    session_stop = session.read_meta()["stopped_reason"]
    session.close()
    worker = result.subagents["worker"]
    summary = {
        "leader_messages": tuple(_message_snapshot(item) for item in result.leader_messages),
        "pool": {
            "worker": {
                "messages": tuple(_message_snapshot(item) for item in worker.messages),
                "turns": worker.turns_used,
                "phases": tuple(run.phase.value for run in worker.runs),
                "breaker": worker.breaker.consecutive_failures,
            }
        },
        "stop": result.stopped_reason,
        "usage": {
            "leader": {
                model: (usage.input_tokens, usage.output_tokens)
                for model, usage in result.usage_by_agent[result.agent.agent_id].items()
            },
            "worker": {
                model: (usage.input_tokens, usage.output_tokens)
                for model, usage in result.usage_by_agent[worker.agent_ref.agent_id].items()
            },
        },
        "session_stop": session_stop,
    }
    return result, summary


@check("events.dropping_all_changes_nothing")
def dropping_all_changes_nothing() -> None:
    with workspace() as ws:
        def fixed_turn(run_id: str, index: int) -> TurnRef:
            return TurnRef(turn_id=f"turn-fixed-{index}", run_id=run_id, index=index)

        sink = CollectingSink()
        with mock.patch(
            "symphonai_api.agent_loop.new_turn_ref",
            side_effect=fixed_turn,
        ):
            without_events, without_summary = _scripted_leader_outcome(
                ws.root,
                None,
                "without-events",
            )
            with_events, with_summary = _scripted_leader_outcome(
                ws.root,
                sink,
                "with-events",
            )
        without_worker = without_events.subagents["worker"]
        with_worker = with_events.subagents["worker"]
        if (
            without_events.leader_messages != with_events.leader_messages
            or without_events.stopped_reason != with_events.stopped_reason
            or without_worker.messages != with_worker.messages
            or without_worker.turns_used != with_worker.turns_used
            or without_summary["usage"] != with_summary["usage"]
            or without_summary["pool"] != with_summary["pool"]
        ):
            fail("dropping all events changed messages, stop, usage, or pool state")
        if without_summary != _FROZEN_PRE_10B_OUTCOME:
            fail(
                f"outcome changed from {_PRE_10B_COMMIT}: "
                f"{without_summary!r}"
            )
        denied = [
            event
            for event in sink.of_type(PermissionDenied)
            if event.tool_call_id == "denied-write"
        ]
        if len(denied) != 1 or denied[0].tool_name != "write_file":
            fail(f"scripted permission denial event differed: {sink.events!r}")
