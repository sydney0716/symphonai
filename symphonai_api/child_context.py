"""Pure child-run context seeding."""

from __future__ import annotations

from collections.abc import Sequence

from symphonai_api.agent_spec import AgentSpec, ContextInheritance
from symphonai_api.compaction import recent_window_start
from symphonai_api.models import Message, Role


def tail_start(messages: Sequence[Message], turns: int) -> int:
    """Move the recent-turn boundary back only far enough to retain tool groups."""

    start = recent_window_start(messages, turns)
    while start > 0:
        issued: set[str] = set()
        orphan_ids: set[str] = set()
        for message in messages[start:]:
            if message.role is Role.ASSISTANT:
                issued.update(call.id for call in message.tool_calls)
            elif message.role is Role.TOOL and message.tool_result is not None:
                if message.tool_result.tool_call_id not in issued:
                    orphan_ids.add(message.tool_result.tool_call_id)
        if not orphan_ids:
            return start
        issuers = [
            index
            for index, message in enumerate(messages[:start])
            if message.role is Role.ASSISTANT
            and any(call.id in orphan_ids for call in message.tool_calls)
        ]
        if not issuers:
            return start
        start = min(issuers)
    return start


def seed_messages(
    spec: AgentSpec,
    task: str,
    *,
    parent_messages: Sequence[Message] = (),
) -> list[Message]:
    """Seed a child with its task and optional whole-turn parent context.

    Tail inheritance includes the last requested turns, extending backward only
    when necessary to keep tool results with the assistant calls that issued them.
    """

    messages: list[Message] = []
    if spec.prompt.strip():
        messages.append(Message(role=Role.SYSTEM, content=spec.prompt))
    if spec.isolation.inherit is ContextInheritance.ALL:
        inherited = parent_messages
    elif spec.isolation.inherit is ContextInheritance.TAIL:
        inherited = parent_messages[tail_start(parent_messages, spec.isolation.inherit_tail):]
    else:
        inherited = ()
    messages.extend(message for message in inherited if message.role is not Role.SYSTEM)
    messages.append(Message(role=Role.USER, content=task))
    return messages
