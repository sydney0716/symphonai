"""Pure conversation compaction helpers for long-running chat sessions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from orchestra_api.models import Message, Role, ToolCall, ToolResult

DEFAULT_CONTEXT_TOKEN_BUDGET = 16_000
DEFAULT_RECENT_TURNS = 4


class ContextCompactionError(ValueError):
    """Raised when required preserved context cannot fit under the budget."""


@dataclass(frozen=True)
class CompactionResult:
    """Result of applying the conversation compaction policy."""

    messages: list[Message]
    before_tokens: int
    after_tokens: int
    budget: int
    changed: bool
    dropped_messages: int = 0
    summary_messages: int = 0
    recent_turns: int = DEFAULT_RECENT_TURNS

    @property
    def reclaimed_tokens(self) -> int:
        return max(0, self.before_tokens - self.after_tokens)


def estimate_text_tokens(text: str) -> int:
    """Estimate token cost from text length.

    This is a rough heuristic, not a real tokenizer and not provider-accurate.
    It uses about four characters per token plus small per-message overheads
    elsewhere in this module so callers can fail early before a request is
    obviously too large.
    """

    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimate_message_tokens(message: Message) -> int:
    """Estimate the token cost of one message.

    This is an approximation for budgeting only. It does not model provider
    tokenizers, hidden protocol overhead, or vendor-specific tool-call formats.
    """

    parts = [message.role.value, message.text]
    if message.tool_calls:
        parts.extend(_tool_call_text(call) for call in message.tool_calls)
    if message.tool_result is not None:
        parts.append(_tool_result_text(message.tool_result))
    return 4 + estimate_text_tokens("\n".join(part for part in parts if part))


def estimate_messages_tokens(messages: list[Message]) -> int:
    """Estimate total token cost for a conversation.

    This is a budget heuristic, not an exact tokenizer count. It should be
    treated as an early warning guardrail, not as a provider billing or context
    limit authority.
    """

    return sum(estimate_message_tokens(message) for message in messages)


def compact_messages_for_budget(
    messages: list[Message],
    *,
    budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    recent_turns: int = DEFAULT_RECENT_TURNS,
) -> CompactionResult:
    """Compact a conversation if its estimated token cost exceeds `budget`.

    The compactor preserves all system messages before the recent window, the
    earliest user goal, and the most recent `recent_turns` user turns. Older
    middle messages are replaced by a small deterministic summary when there
    is room. If the preserved context alone cannot fit, a clear
    `ContextCompactionError` is raised before a provider request is made.
    """

    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if recent_turns < 1:
        raise ValueError(f"recent_turns must be >= 1, got {recent_turns}")

    original = list(messages)
    before_tokens = estimate_messages_tokens(original)
    if before_tokens <= budget:
        return CompactionResult(
            messages=original,
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            budget=budget,
            changed=False,
            recent_turns=recent_turns,
        )

    recent_start = _recent_window_start(original, recent_turns)
    prefix_indices = _preserved_prefix_indices(original, recent_start)
    recent_indices = set(range(recent_start, len(original)))
    dropped_indices = [
        index
        for index in range(len(original))
        if index not in prefix_indices and index not in recent_indices
    ]
    if not dropped_indices:
        raise _impossible_error(before_tokens, budget, recent_turns)

    prefix = [original[index] for index in sorted(prefix_indices)]
    recent = [original[index] for index in range(recent_start, len(original))]
    dropped = [original[index] for index in dropped_indices]

    for max_summary_chars in (800, 320, 120, 0):
        compacted = list(prefix)
        summary_messages = 0
        if max_summary_chars > 0:
            compacted.append(
                Message(
                    role=Role.SYSTEM,
                    content=_summarize_dropped_messages(
                        dropped,
                        max_chars=max_summary_chars,
                    ),
                )
            )
            summary_messages = 1
        compacted.extend(recent)
        after_tokens = estimate_messages_tokens(compacted)
        if after_tokens <= budget:
            return CompactionResult(
                messages=compacted,
                before_tokens=before_tokens,
                after_tokens=after_tokens,
                budget=budget,
                changed=True,
                dropped_messages=len(dropped),
                summary_messages=summary_messages,
                recent_turns=recent_turns,
            )

    raise _impossible_error(before_tokens, budget, recent_turns)


def describe_compaction(result: CompactionResult) -> str:
    """Return a short human-readable description of a compaction result."""

    if not result.changed:
        return (
            "No compaction needed: "
            f"estimated {result.before_tokens} tokens within budget {result.budget}."
        )
    summary_part = (
        f"inserted {result.summary_messages} summary message"
        if result.summary_messages
        else "no summary fit"
    )
    return (
        "Compacted conversation: "
        f"dropped {result.dropped_messages} older messages, {summary_part}, "
        f"reclaimed about {result.reclaimed_tokens} tokens "
        f"({result.before_tokens} -> {result.after_tokens}; budget {result.budget})."
    )


def _tool_call_text(tool_call: ToolCall) -> str:
    try:
        arguments = json.dumps(tool_call.arguments, sort_keys=True)
    except TypeError:
        arguments = repr(tool_call.arguments)
    return f"tool_call {tool_call.name} {tool_call.id} {arguments}"


def _tool_result_text(tool_result: ToolResult) -> str:
    fields = [tool_result.tool_call_id, "ok" if tool_result.ok else "error"]
    if tool_result.content:
        fields.append(tool_result.content)
    if tool_result.error:
        fields.append(tool_result.error)
    return " ".join(fields)


def _recent_window_start(messages: list[Message], recent_turns: int) -> int:
    user_indices = [index for index, message in enumerate(messages) if message.role == Role.USER]
    if not user_indices:
        return max(0, len(messages) - recent_turns)
    if len(user_indices) <= recent_turns:
        return 0
    return user_indices[-recent_turns]


def _preserved_prefix_indices(messages: list[Message], recent_start: int) -> set[int]:
    prefix_indices = {
        index
        for index, message in enumerate(messages[:recent_start])
        if message.role == Role.SYSTEM
    }
    first_user_index = next(
        (index for index, message in enumerate(messages) if message.role == Role.USER),
        None,
    )
    if first_user_index is not None and first_user_index < recent_start:
        prefix_indices.add(first_user_index)
    return prefix_indices


def _summarize_dropped_messages(messages: list[Message], *, max_chars: int) -> str:
    role_counts: dict[str, int] = {}
    for message in messages:
        role_counts[message.role.value] = role_counts.get(message.role.value, 0) + 1
    counts = ", ".join(f"{role}={count}" for role, count in sorted(role_counts.items()))
    text = (
        "Earlier conversation compacted before this request. "
        f"Removed {len(messages)} messages"
        f"{f' ({counts})' if counts else ''}. "
        "System instructions, the first user goal, and recent turns are preserved."
    )
    excerpts = _message_excerpts(messages, limit=3)
    if excerpts:
        text += " Old excerpts: " + " | ".join(excerpts)
    return _truncate(text, max_chars)


def _message_excerpts(messages: list[Message], *, limit: int) -> list[str]:
    excerpts: list[str] = []
    for message in messages:
        snippet = _compact_whitespace(message.text)
        if not snippet and message.tool_calls:
            names = ", ".join(call.name for call in message.tool_calls)
            snippet = f"tool calls: {names}"
        if not snippet and message.tool_result is not None:
            result = message.tool_result
            snippet = result.content or result.error or f"tool result {result.tool_call_id}"
        if snippet:
            excerpts.append(f"{message.role.value}: {_truncate(snippet, 120)}")
        if len(excerpts) >= limit:
            break
    return excerpts


def _compact_whitespace(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = "..."
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix


def _impossible_error(before_tokens: int, budget: int, recent_turns: int) -> ContextCompactionError:
    return ContextCompactionError(
        "conversation is too large to fit the context budget after preserving "
        "system messages, the first user goal, and the recent turn window "
        f"(estimated {before_tokens} tokens, budget {budget}, "
        f"recent_turns {recent_turns}). Increase the budget, reduce the latest "
        "message size, or configure fewer preserved recent turns."
    )
