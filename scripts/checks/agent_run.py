"""Workspace-backed checks for agent run."""

from __future__ import annotations

import unittest.mock as mock

from symphonai_api import ToolEffect, ToolMetadata
from symphonai_api.agent_loop import ApiAgent
from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.events import CollectingSink, ToolCallFailed
from symphonai_api.models import Message, ModelResponse, Role, ToolCall, ToolResult
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.runner import run_task, standard_tool_registry
from symphonai_api.session import TranscriptWriter, read_records
from symphonai_api.tools.base import LocalTool
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


class _ToolContractStub(LocalTool):
    @property
    def name(self) -> str:
        return "contract_stub"

    @property
    def description(self) -> str:
        return "Exercise the LocalTool contract."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

class _ValidationStageTool(_ToolContractStub):
    def __init__(self) -> None:
        self.executed = False

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(),
        )

    def validate(self, arguments: dict) -> str | None:
        return "scripted validation failure"

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        self.executed = True
        return ToolResult(tool_call_id=tool_call.id, ok=True)

@check("agent.full_run")
def check_agent_full_run() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        # -- full agent run: tool-call turn (read_file) then final answer --
        tool_turn = ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "existing.txt"})],
            )
        )
        final_turn = ModelResponse(message=Message(role=Role.ASSISTANT, content="task complete"))
        provider = FakeModelProvider(responses=[tool_turn, final_turn])

        result = run_task(provider, policy, "read existing.txt then finish")
        if result.stopped_reason != "final_response":
            fail(f"expected stopped_reason='final_response', got {result.stopped_reason!r}")
        if result.final_response.message.text != "task complete":
            fail("final response content mismatch")
        tool_messages = [m for m in result.messages if m.role == Role.TOOL]
        if not tool_messages or not tool_messages[0].tool_result.ok:
            fail("expected the read_file tool call to succeed")
        if tool_messages[0].tool_result.content != "1\thello from disk":
            fail("read_file returned unexpected content")
        appended = [message for message in result.messages if message.role in (Role.ASSISTANT, Role.TOOL)]
        if any(message.turn_id is None for message in appended):
            fail(f"agent-appended messages were not stamped with turn ids: {appended!r}")
        if appended[0].turn_id != appended[1].turn_id:
            fail(f"assistant tool call and result must share a turn id: {appended!r}")
        if result.run.agent_id != result.agent.agent_id:
            fail(f"agent run owner link is inconsistent: {result!r}")
        if result.final_response.message.turn_id != result.messages[-1].turn_id:
            fail(
                "final_response carries a different turn identity than the same "
                f"message in the transcript: {result.final_response.message!r}"
            )
        if result.final_response.message.turn_id is None:
            fail("final_response.message was returned without a turn id")

@check("agent.base_validation")
def check_agent_base_validation() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        # -- allow path: read/list/write inside repo_root + allowed_write_scope --
        tools = standard_tool_registry()

        validation_tool = _ValidationStageTool()
        validation_result = validation_tool.execute(
            ToolCall(id="validation-stage", name=validation_tool.name),
            policy,
        )
        if (
            validation_result.ok
            or validation_result.error != "scripted validation failure"
            or validation_tool.executed
        ):
            fail(
                "validation did not short-circuit before permitted work: "
                f"result={validation_result!r}, executed={validation_tool.executed!r}"
            )

        invalid_shell = tools["run_shell"].execute(
            ToolCall(
                id="invalid-shell-arguments",
                name="run_shell",
                arguments={"argv": ["ls", 3]},
            ),
            policy,
        )
        expected_shell_error = (
            "missing or invalid required argument: argv "
            "(must be a non-empty list of strings)"
        )
        if invalid_shell.ok or invalid_shell.error != expected_shell_error:
            fail(f"run_shell validation message changed: {invalid_shell!r}")

        missing_read_path = tools["read_file"].execute(
            ToolCall(id="missing-read-path", name="read_file"),
            policy,
        )
        if missing_read_path.ok or missing_read_path.error != "missing required argument: path":
            fail(f"read_file validation message changed: {missing_read_path!r}")

        missing_write_content = tools["write_file"].execute(
            ToolCall(
                id="missing-write-content",
                name="write_file",
                arguments={"path": "missing-content.txt"},
            ),
            policy,
        )
        if (
            missing_write_content.ok
            or missing_write_content.error != "missing required argument: content"
        ):
            fail(f"write_file validation message changed: {missing_write_content!r}")

        empty_write = tools["write_file"].execute(
            ToolCall(
                id="empty-write-content",
                name="write_file",
                arguments={"path": "empty-content.txt", "content": ""},
            ),
            policy,
        )
        if not empty_write.ok or (root / "empty-content.txt").read_text() != "":
            fail(f"write_file rejected valid empty content: {empty_write!r}")

        default_list = tools["list_files"].execute(
            ToolCall(id="default-list-path", name="list_files"),
            policy,
        )
        if not default_list.ok or "existing.txt" not in default_list.content:
            fail(f"list_files no longer defaults to the repo root: {default_list!r}")

        invalid_cancel = CancellationToken()
        invalid_cancel.cancel()
        try:
            tools["run_shell"].execute(
                ToolCall(
                    id="cancel-before-validation",
                    name="run_shell",
                    arguments={"argv": ["ls", 3]},
                ),
                policy,
                cancel=invalid_cancel,
            )
        except OperationCancelled:
            pass
        else:
            fail("LocalTool validated invalid arguments before checking cancellation")


@check("agent.repeated_failure_is_named")
def check_repeated_failure_is_named() -> None:
    error = "scripted validation failure"
    warning = (
        " This has now failed {count} times in a row with the same error. "
        "Do not send it again; change approach, or tell the person what is blocking you."
    )
    cases = [
        ("repeated", [("first", error)] * 4,
         [error, error, error + warning.format(count=3), error + warning.format(count=4)]),
        ("success resets", [("first", error)] * 2 + [("first", None), ("first", error)],
         [error, error, None, error]),
        ("different error resets", [("first", error), ("first", "other error"), ("first", error)],
         [error, "other error", error]),
        ("alternating tools", [("first", error), ("second", error)] * 3,
         [error] * 6),
    ]
    with workspace() as ws:
        for case_index, (label, outcomes, expected) in enumerate(cases):
            calls = [
                ToolCall(id=f"call-{index}", name=name, arguments={"attempt": index})
                for index, (name, _) in enumerate(outcomes)
            ]
            provider = FakeModelProvider([
                *[
                    ModelResponse(Message(Role.ASSISTANT, tool_calls=[call]))
                    for call in calls
                ],
                ModelResponse(Message(Role.ASSISTANT, content="done")),
            ])
            sink = CollectingSink()
            path = ws.root / f"repeated-failure-{case_index}.jsonl"
            writer = TranscriptWriter(path)
            tool = _ValidationStageTool()
            agent = ApiAgent(
                provider, {"first": tool, "second": tool}, ws.policy,
                max_turns=len(calls) + 1, events=sink, transcript=writer,
            )
            with mock.patch.object(tool, "validate", side_effect=[error for _, error in outcomes]), \
                    mock.patch.object(provider, "create_response", wraps=provider.create_response) as create:
                try:
                    result = agent.run([Message(Role.USER, "exercise failures")])
                finally:
                    writer.close()
            tool_results = [
                message.tool_result for message in result.messages if message.role == Role.TOOL
            ]
            actual = [item.error for item in tool_results]
            if actual != expected:
                fail(f"{label}: expected errors {expected!r}, got {actual!r}")
            if [item.ok for item in tool_results] != [error is None for _, error in outcomes]:
                fail(f"{label}: failure annotation changed tool status")
            if result.stopped_reason != "final_response" or provider.call_count != len(calls) + 1:
                fail(f"{label}: repeated failures stopped the run")
            for index, request_call in enumerate(create.call_args_list[1:]):
                received = [
                    message.tool_result.error
                    for message in request_call.args[0].messages if message.role == Role.TOOL
                ]
                if received != expected[:index + 1]:
                    fail(f"{label}: model received different errors: {received!r}")
            records, dropped = read_records(path)
            persisted = [
                record["data"]["tool_result"]["error"] for record in records
                if record["type"] == "message" and record["data"]["role"] == "tool"
            ]
            if dropped or persisted != expected:
                fail(f"{label}: persisted errors differed: {persisted!r}")
            event_errors = [event.error for event in sink.of_type(ToolCallFailed)]
            if event_errors != [error for error in expected if error is not None]:
                fail(f"{label}: failure event errors differed: {event_errors!r}")
