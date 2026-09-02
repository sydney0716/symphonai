"""Conversation repair shared by live cancellation and persisted recovery."""

from __future__ import annotations

from collections.abc import Sequence

from symphonai_api.models import Message, Role, ToolResult


def unanswered_tool_call_ids(messages: Sequence[Message]) -> list[str]:
    """Return unanswered ids from the last assistant message with tool calls."""

    assistant_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].role == Role.ASSISTANT
            and messages[index].tool_calls
        ),
        None,
    )
    if assistant_index is None:
        return []
    answered_ids = {
        message.tool_result.tool_call_id
        for message in messages[assistant_index + 1 :]
        if message.tool_result is not None
    }
    return [
        tool_call.id
        for tool_call in messages[assistant_index].tool_calls
        if tool_call.id not in answered_ids
    ]


def repair_unanswered_tool_calls(
    messages: list[Message],
    *,
    error: str,
    cancelled: bool,
    turn_id: str | None = None,
) -> list[str]:
    """Append one failed tool-role message per unanswered id, in place."""

    unanswered = unanswered_tool_call_ids(messages)
    for tool_call_id in unanswered:
        messages.append(
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(
                    tool_call_id=tool_call_id,
                    ok=False,
                    cancelled=cancelled,
                    error=error,
                ),
                turn_id=turn_id,
            )
        )
    return unanswered
