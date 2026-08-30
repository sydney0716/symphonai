"""Workspace-backed checks for agent events."""

from __future__ import annotations

import unittest.mock as mock
from orchestra_api.agent_loop import ApiAgent
from orchestra_api.cancellation import OperationCancelled
from orchestra_api.events import CollectingSink, RunFailed, RunFinished, RunStarted, ToolCallFinished, ToolCallStarted, TurnFinished, TurnStarted
from orchestra_api.identity import TurnRef
from orchestra_api.models import Message, ModelResponse, Role, ToolCall
from orchestra_api.providers.base import ModelProvider, ProviderError
from orchestra_api.providers.fake import FakeModelProvider
from orchestra_api.runner import standard_tool_registry
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
        tool_events = CollectingSink()
        tool_event_result = ApiAgent(
            FakeModelProvider(
                responses=[
                    ModelResponse(
                        Message(
                            Role.ASSISTANT,
                            tool_calls=[
                                ToolCall(
                                    id="event-read",
                                    name="read_file",
                                    arguments={"path": "existing.txt"},
                                )
                            ],
                        )
                    ),
                    ModelResponse(Message(Role.ASSISTANT, "done")),
                ]
            ),
            standard_tool_registry(),
            policy,
            events=tool_events,
        ).run([Message(Role.USER, "read")])
        tool_started = tool_events.of_type(ToolCallStarted)
        tool_finished = tool_events.of_type(ToolCallFinished)
        if (
            len(tool_started) != 1
            or len(tool_finished) != 1
            or tool_started[0].tool_call_id != "event-read"
            or tool_finished[0].tool_call_id != "event-read"
            or not tool_finished[0].ok
        ):
            fail(f"successful tool events did not bracket execution: {tool_events.events!r}")
        if tool_event_result.stopped_reason != "final_response":
            fail(f"event-producing tool run changed its result: {tool_event_result!r}")

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

        with mock.patch("orchestra_api.agent_loop.new_turn_ref", side_effect=_fixed_turn):
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
