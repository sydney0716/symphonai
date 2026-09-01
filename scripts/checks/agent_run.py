"""Workspace-backed checks for agent run."""

from __future__ import annotations

from symphonai_api import ToolEffect, ToolMetadata
from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.models import Message, ModelResponse, Role, ToolCall, ToolResult
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.runner import run_task, standard_tool_registry
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
