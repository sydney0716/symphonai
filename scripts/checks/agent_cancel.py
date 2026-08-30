"""Workspace-backed checks for agent cancel."""

from __future__ import annotations

import json
import os
import time
import unittest.mock as mock
from orchestra_api import ToolEffect, ToolMetadata
from orchestra_api.agent_loop import ApiAgent
from orchestra_api.cancellation import CancellationToken, OperationCancelled
from orchestra_api.events import CollectingSink, RunFailed, RunFinished, RunStarted
from orchestra_api.models import Message, ModelRequest, ModelResponse, Role, ToolCall
from orchestra_api.providers.base import ModelProvider, ProviderError
from orchestra_api.providers.fake import FakeModelProvider
from orchestra_api.providers.openai_provider import API_KEY_ENV_VAR, OpenAIProvider
from orchestra_api.providers.openai_provider import _build_request_body as _build_openai_body
from orchestra_api.tools.base import LocalTool
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
