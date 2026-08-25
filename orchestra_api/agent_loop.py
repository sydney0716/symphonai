"""The ApiAgent loop: call the model, execute any tool calls, repeat.

On each turn: call `provider.create_response()`. If the response carries
one or more `ToolCall`s, execute each via the tool registry (gated by
`PermissionPolicy`), append the resulting `ToolResult`(s) as tool-role
`Message`s, and loop. Otherwise the response is final. The loop always
terminates -- either on a final response or once `max_turns` is reached --
and never raises just because `max_turns` was hit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from orchestra_api.cancellation import CancellationToken, OperationCancelled
from orchestra_api.identity import AgentRef, RunRef, new_agent_ref, new_run_ref, new_turn_ref
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
    stopped_reason: str  # "final_response", "max_turns", or "cancelled"
    run: RunRef
    agent: AgentRef


class ApiAgent:
    """Runs the call-model / execute-tools loop against a `ModelProvider`."""

    def __init__(
        self,
        provider: ModelProvider,
        tools: dict[str, LocalTool],
        policy: PermissionPolicy,
        max_turns: int = DEFAULT_MAX_TURNS,
        tool_schemas: list[dict] | None = None,
        *,
        agent_ref: AgentRef | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self._provider = provider
        self._tools = tools
        self._policy = policy
        self._max_turns = max_turns
        self._agent_ref = agent_ref or new_agent_ref("agent")
        # Schemas actually sent to the model so it knows these tools exist.
        # `tools` above is only the *execution* registry, keyed by name --
        # without this, a real provider is never told any tool exists and
        # can never call one, even if `tools` would happily execute it.
        self._tool_schemas = tool_schemas or []

    @property
    def agent_ref(self) -> AgentRef:
        return self._agent_ref

    def run(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        parent_run_id: str | None = None,
        cancel: CancellationToken | None = None,
    ) -> AgentRunResult:
        run_ref = new_run_ref(self._agent_ref.agent_id, parent_run_id)
        conversation = list(messages)
        response: ModelResponse | None = None
        for turn in range(1, self._max_turns + 1):
            turn_ref = new_turn_ref(run_ref.run_id, turn)
            try:
                if cancel is not None:
                    cancel.raise_if_cancelled()
                request = ModelRequest(
                    messages=list(conversation), model=model, tools=self._tool_schemas
                )
                if cancel is None:
                    response = self._provider.create_response(request)
                else:
                    response = self._provider.create_response(request, cancel=cancel)
                # Stamp once and reuse, so the message in `conversation` and
                # `final_response` carry the same turn identity.
                response = replace(
                    response, message=replace(response.message, turn_id=turn_ref.turn_id)
                )
                conversation.append(response.message)
                if not response.has_tool_calls:
                    return AgentRunResult(
                        final_response=response,
                        messages=conversation,
                        turns_used=turn,
                        stopped_reason="final_response",
                        run=run_ref,
                        agent=self._agent_ref,
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
                        result = tool.execute(tool_call, self._policy, cancel=cancel)
                    conversation.append(
                        Message(role=Role.TOOL, tool_result=result, turn_id=turn_ref.turn_id)
                    )
            except OperationCancelled:
                cancelled_response = response or ModelResponse(
                    message=Message(role=Role.ASSISTANT), stop_reason="cancelled"
                )
                return AgentRunResult(
                    final_response=cancelled_response,
                    messages=conversation,
                    turns_used=turn,
                    stopped_reason="cancelled",
                    run=run_ref,
                    agent=self._agent_ref,
                )
        assert response is not None  # loop runs at least once since max_turns >= 1
        return AgentRunResult(
            final_response=response,
            messages=conversation,
            turns_used=self._max_turns,
            stopped_reason="max_turns",
            run=run_ref,
            agent=self._agent_ref,
        )
