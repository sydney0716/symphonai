"""The ApiAgent loop: call the model, execute any tool calls, repeat.

On each turn: call `provider.create_response()`. If the response carries
one or more `ToolCall`s, execute each via the tool registry (gated by
`PermissionPolicy`), append the resulting `ToolResult`(s) as tool-role
`Message`s, and loop. Otherwise the response is final. The loop always
terminates -- either on a final response or once `max_turns` is reached --
and never raises just because `max_turns` was hit.
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestra_api.models import Message, ModelRequest, ModelResponse, Role, ToolResult
from orchestra_api.permissions import PermissionPolicy
from orchestra_api.providers.base import ModelProvider
from orchestra_api.tools.base import LocalTool

DEFAULT_MAX_TURNS = 10


@dataclass
class AgentRunResult:
    """The outcome of an `ApiAgent.run()` call."""

    final_response: ModelResponse
    messages: list[Message]
    turns_used: int
    stopped_reason: str  # "final_response" or "max_turns"


class ApiAgent:
    """Runs the call-model / execute-tools loop against a `ModelProvider`."""

    def __init__(
        self,
        provider: ModelProvider,
        tools: dict[str, LocalTool],
        policy: PermissionPolicy,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self._provider = provider
        self._tools = tools
        self._policy = policy
        self._max_turns = max_turns

    def run(self, messages: list[Message], *, model: str | None = None) -> AgentRunResult:
        conversation = list(messages)
        response: ModelResponse | None = None
        for turn in range(1, self._max_turns + 1):
            request = ModelRequest(messages=list(conversation), model=model)
            response = self._provider.create_response(request)
            conversation.append(response.message)
            if not response.has_tool_calls:
                return AgentRunResult(
                    final_response=response,
                    messages=conversation,
                    turns_used=turn,
                    stopped_reason="final_response",
                )
            for tool_call in response.message.tool_calls:
                tool = self._tools.get(tool_call.name)
                if tool is None:
                    result = ToolResult(
                        tool_call_id=tool_call.id,
                        ok=False,
                        error=f"unknown tool: {tool_call.name!r}",
                    )
                else:
                    result = tool.execute(tool_call, self._policy)
                conversation.append(Message(role=Role.TOOL, tool_result=result))
        assert response is not None  # loop runs at least once since max_turns >= 1
        return AgentRunResult(
            final_response=response,
            messages=conversation,
            turns_used=self._max_turns,
            stopped_reason="max_turns",
        )
