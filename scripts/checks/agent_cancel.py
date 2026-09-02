"""Workspace-backed checks for agent cancel."""

from __future__ import annotations

import json
import os
import time
import unittest.mock as mock
from symphonai_api import ToolEffect, ToolMetadata
from symphonai_api.agent_loop import ApiAgent
from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.events import CollectingSink, RunFailed, RunFinished, RunStarted
from symphonai_api.identity import AgentRef, RunRef, TurnRef
from symphonai_api.models import (
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    ToolCall,
    ToolResult,
)
from symphonai_api.providers.base import ModelProvider, ProviderError
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.providers.openai_provider import API_KEY_ENV_VAR, OpenAIProvider
from symphonai_api.providers.openai_provider import _build_request_body as _build_openai_body
from symphonai_api.repair import unanswered_tool_call_ids
from symphonai_api.serialization import message_to_json
from symphonai_api.tools.base import LocalTool
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


def _assert_openai_tool_calls_answered(messages: list[Message], context: str) -> None:
    wire_messages = _build_openai_body(ModelRequest(messages=messages), "test-model")["messages"]
    for index, message in enumerate(wire_messages):
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            continue
        expected_ids = [tool_call["id"] for tool_call in tool_calls]
        actual_ids = [
            candidate.get("tool_call_id")
            for candidate in wire_messages[index + 1 : index + 1 + len(expected_ids)]
            if candidate.get("role") == "tool"
        ]
        if actual_ids != expected_ids:
            fail(
                f"{context} left unanswered OpenAI tool calls: "
                f"expected={expected_ids!r}, actual={actual_ids!r}, body={wire_messages!r}"
            )

class _CancellingTool(LocalTool):
    @property
    def name(self) -> str:
        return "cancel_work"

    @property
    def description(self) -> str:
        return "Cancel the active test turn."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.DESTRUCTIVE,
            concurrency_safe=False,
            paths=None,
        )

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        assert cancel is not None
        cancel.cancel()
        raise OperationCancelled

@check("cancel.pre_cancelled_agent")
def check_cancel_pre_cancelled_agent() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        pre_cancel = CancellationToken()
        pre_cancel_events = CollectingSink()
        pre_cancel.cancel()
        pre_cancel_provider = FakeModelProvider()
        pre_cancel_result = ApiAgent(pre_cancel_provider, {}, policy).run(
            [Message(role=Role.USER, content="stop")],
            cancel=pre_cancel,
            events=pre_cancel_events,
        )
        if pre_cancel_result.stopped_reason != "cancelled":
            fail(f"pre-cancelled run did not return cancelled: {pre_cancel_result!r}")
        if pre_cancel_provider.call_count != 0:
            fail(f"pre-cancelled run called its provider {pre_cancel_provider.call_count} times")
        pre_cancel_terminals = pre_cancel_events.of_type(RunFinished) + pre_cancel_events.of_type(RunFailed)
        if (
            len(pre_cancel_events.of_type(RunStarted)) != 1
            or len(pre_cancel_terminals) != 1
            or pre_cancel_terminals[0].stopped_reason != "cancelled"
        ):
            fail(f"cancelled run terminal cardinality is invalid: {pre_cancel_events.events!r}")

@check("cancel.tool_repair")
def check_cancel_tool_repair() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        during_tool_token = CancellationToken()
        cancelling_call = ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(id="cancel-1", name="cancel_work"),
                    ToolCall(id="cancel-2", name="cancel_work"),
                ],
            )
        )
        during_tool_result = ApiAgent(
            FakeModelProvider(responses=[cancelling_call]),
            {"cancel_work": _CancellingTool()},
            policy,
        ).run([Message(role=Role.USER, content="cancel in tool")], cancel=during_tool_token)
        if during_tool_result.stopped_reason != "cancelled":
            fail(f"tool cancellation escaped the agent boundary: {during_tool_result!r}")
        if not any(message.role == Role.ASSISTANT for message in during_tool_result.messages):
            fail(f"tool cancellation discarded the assistant message: {during_tool_result!r}")
        returned_calls = [
            tool_call
            for message in during_tool_result.messages
            for tool_call in message.tool_calls
        ]
        returned_results = {
            message.tool_result.tool_call_id: message.tool_result
            for message in during_tool_result.messages
            if message.tool_result is not None
        }
        if {tool_call.id for tool_call in returned_calls} != set(returned_results):
            fail(f"tool cancellation left an unanswered call: {during_tool_result.messages!r}")
        repaired = returned_results["cancel-1"]
        if repaired.ok or not repaired.cancelled:
            fail(f"cancelled tool result has the wrong flags: {repaired!r}")
        if returned_results["cancel-2"].ok or not returned_results["cancel-2"].cancelled:
            fail(f"unstarted tool call has the wrong cancellation result: {returned_results['cancel-2']!r}")
        next_turn_messages = [
            *during_tool_result.messages,
            Message(role=Role.USER, content="continue after cancellation"),
        ]
        _assert_openai_tool_calls_answered(next_turn_messages, "cancelled agent transcript")

@check("cancel.http_read_recheck")
def check_cancel_http_read_recheck() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        # -- regression: a real provider's run_task() request must include
        # schemas for all eight standard tools --
        captured: dict = {}

        class _FakeHttpResponse:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def read(self) -> bytes:
                return self._body

            def __enter__(self) -> "_FakeHttpResponse":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        delayed_token = CancellationToken()

        class _DelayedCancellingResponse(_FakeHttpResponse):
            def read(self) -> bytes:
                time.sleep(0.01)
                delayed_token.cancel()
                return super().read()

        delayed_payload = json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "late answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
        ).encode("utf-8")
        with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: "delayed-cancel-test-key"}):
            with mock.patch(
                "urllib.request.urlopen",
                return_value=_DelayedCancellingResponse(delayed_payload),
            ):
                try:
                    OpenAIProvider().create_response(
                        ModelRequest(messages=[Message(role=Role.USER, content="wait")]),
                        cancel=delayed_token,
                    )
                except ProviderError as exc:
                    fail(f"transport cancellation was wrapped as ProviderError: {exc!r}")
                except OperationCancelled:
                    pass
                else:
                    fail("successful HTTP body bypassed cancellation after response.read()")

@check("cancel.late_response_retained")
def check_cancel_late_response_retained() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        late_agent_token = CancellationToken()

        class _LateCancellingProvider(ModelProvider):
            @property
            def name(self) -> str:
                return "late-cancelling"

            @property
            def wire_format(self) -> int:
                return 4

            def create_response(
                self,
                request: ModelRequest,
                *,
                cancel: CancellationToken | None = None,
            ) -> ModelResponse:
                assert cancel is not None
                time.sleep(0.01)
                cancel.cancel()
                return ModelResponse(Message(Role.ASSISTANT, "late answer"))

        late_agent_result = ApiAgent(
            _LateCancellingProvider(), {}, policy
        ).run(
            [Message(role=Role.USER, content="wait")],
            cancel=late_agent_token,
        )
        if late_agent_result.stopped_reason != "cancelled":
            fail(f"late successful response bypassed agent cancellation: {late_agent_result!r}")
        if not any(
            message.role == Role.ASSISTANT and message.text == "late answer"
            for message in late_agent_result.messages
        ):
            fail(f"late assistant response was not retained: {late_agent_result.messages!r}")


@check("agent_cancel.unanswered_ids_last_assistant_only")
def check_unanswered_ids_last_assistant_only() -> None:
    earlier = Message(
        role=Role.ASSISTANT,
        tool_calls=[ToolCall(id="earlier", name="first")],
    )
    latest = Message(
        role=Role.ASSISTANT,
        tool_calls=[
            ToolCall(id="latest-1", name="second"),
            ToolCall(id="latest-2", name="second"),
            ToolCall(id="latest-3", name="second"),
        ],
    )
    answered = Message(
        role=Role.TOOL,
        tool_result=ToolResult(tool_call_id="latest-2", ok=True, content="done"),
    )
    if unanswered_tool_call_ids([Message(role=Role.USER, content="plain")]) != []:
        fail("conversation without tool calls reported unanswered ids")
    if unanswered_tool_call_ids([latest, answered]) != ["latest-1", "latest-3"]:
        fail("unanswered ids lost the latest assistant's tool-call order")
    fully_answered = [
        latest,
        Message(role=Role.TOOL, tool_result=ToolResult("latest-1", True)),
        answered,
        Message(role=Role.TOOL, tool_result=ToolResult("latest-3", True)),
    ]
    if unanswered_tool_call_ids(fully_answered) != []:
        fail("fully answered assistant message still reported ids")
    if unanswered_tool_call_ids([earlier, latest, answered]) != [
        "latest-1",
        "latest-3",
    ]:
        fail("an earlier unanswered turn was selected instead of the latest one")


# Captured from commit 210ebf6 -- the last tree before the cancellation repair
# moved into repair.py -- by running the scenario below against a `git archive`
# extraction of it. Frozen rather than recomputed: comparing against HEAD would
# compare this refactor with itself the moment it is committed, and would need a
# git checkout to run at all, which the published snapshot does not have.
_CANCELLED_MESSAGES_BEFORE_THE_REFACTOR = """
[
  {"role": "user", "schema_version": 1, "turn_id": null, "tool_calls": [],
   "tool_result": null,
   "content": [{"kind": "text", "schema_version": 1, "text": "cancel"}]},
  {"role": "assistant", "schema_version": 1, "turn_id": "turn-fixed",
   "content": [], "tool_result": null,
   "tool_calls": [
     {"id": "cancel-1", "name": "cancel_work", "arguments": {},
      "provider_metadata": {}, "schema_version": 1, "vendor_id": null},
     {"id": "cancel-2", "name": "cancel_work", "arguments": {},
      "provider_metadata": {}, "schema_version": 1, "vendor_id": null}]},
  {"role": "tool", "schema_version": 1, "turn_id": "turn-fixed",
   "content": [], "tool_calls": [],
   "tool_result": {"tool_call_id": "cancel-1", "ok": false, "content": "",
                   "error": "cancelled before this tool completed",
                   "cancelled": true, "offloaded": null, "payload": null,
                   "schema_version": 1}},
  {"role": "tool", "schema_version": 1, "turn_id": "turn-fixed",
   "content": [], "tool_calls": [],
   "tool_result": {"tool_call_id": "cancel-2", "ok": false, "content": "",
                   "error": "cancelled before this tool completed",
                   "cancelled": true, "offloaded": null, "payload": null,
                   "schema_version": 1}}
]
"""


@check("agent_cancel.cancelled_messages_unchanged_by_refactor")
def check_cancelled_messages_unchanged_by_refactor() -> None:
    response = ModelResponse(
        message=Message(
            role=Role.ASSISTANT,
            tool_calls=[
                ToolCall(id="cancel-1", name="cancel_work"),
                ToolCall(id="cancel-2", name="cancel_work"),
            ],
        )
    )
    with workspace() as ws:
        # Fixed ids: the run and turn ids are the only nondeterminism in the
        # messages, and the old tree stamped the repairs with the turn id too.
        with mock.patch(
            "symphonai_api.agent_loop.new_run_ref",
            return_value=RunRef("run-fixed", "agent-fixed"),
        ), mock.patch(
            "symphonai_api.agent_loop.new_turn_ref",
            return_value=TurnRef("turn-fixed", "run-fixed", 1),
        ):
            result = ApiAgent(
                FakeModelProvider([response]),
                {"cancel_work": _CancellingTool()},
                ws.policy,
                agent_ref=AgentRef("agent-fixed", "agent"),
            ).run(
                [Message(role=Role.USER, content="cancel")],
                cancel=CancellationToken(),
            )
    produced = [message_to_json(message) for message in result.messages]
    if produced != json.loads(_CANCELLED_MESSAGES_BEFORE_THE_REFACTOR):
        fail(f"refactored cancellation messages differ from 210ebf6: {produced!r}")
