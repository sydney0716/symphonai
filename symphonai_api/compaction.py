"""Pure conversation compaction helpers for long-running chat sessions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace

from symphonai_api.cancellation import CancellationToken
from symphonai_api.models import DocumentBlock, ImageBlock, Message, Role, TextBlock, ToolCall, ToolResult

DEFAULT_CONTEXT_TOKEN_BUDGET = 16_000
DEFAULT_RECENT_TURNS = 4
ATTACHMENT_BYTES_PER_TOKEN = 750
COMPACTABLE_TOOL_NAMES = frozenset(
    {
        "read_file",
        "list_files",
        "glob",
        "grep",
        "run_shell",
        "read_tool_result",
    }
)
CLEARED_CONTENT_MARKER = "[old tool result content cleared]"


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
    cleared_tool_results: int = 0
    recent_turns: int = DEFAULT_RECENT_TURNS

    @property
    def reclaimed_tokens(self) -> int:
        return max(0, self.before_tokens - self.after_tokens)


@dataclass(frozen=True)
class MicrocompactionResult:
    """Result of clearing old, re-derivable tool-result content."""

    messages: list[Message]
    before_tokens: int
    after_tokens: int
    budget: int
    changed: bool
    cleared_tool_results: int = 0
    recent_turns: int = DEFAULT_RECENT_TURNS


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


def estimate_attachment_tokens(block: ImageBlock | DocumentBlock) -> int:
    """Rough token cost of one attachment, for budgeting only.

    Vendors price an image from its pixel dimensions -- Anthropic bills about
    `width * height / 750` tokens -- and we deliberately do not decode the
    image to find them. Scaling off the decoded byte count instead lands
    within an order of magnitude, which is all `compact_messages_for_budget`
    needs to decide whether a conversation still fits.
    """
    decoded = len(block.data) * 3 // 4
    return max(1, math.ceil(decoded / ATTACHMENT_BYTES_PER_TOKEN))


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
    attachment_tokens = sum(
        estimate_attachment_tokens(block)
        for block in message.content
        if isinstance(block, (ImageBlock, DocumentBlock))
    )
    return 4 + estimate_text_tokens("\n".join(part for part in parts if part)) + attachment_tokens


def estimate_messages_tokens(messages: list[Message]) -> int:
    """Estimate total token cost for a conversation.

    This is a budget heuristic, not an exact tokenizer count. It should be
    treated as an early warning guardrail, not as a provider billing or context
    limit authority.
    """

    return sum(estimate_message_tokens(message) for message in messages)


def microcompact_messages(
    messages: list[Message],
    *,
    budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    recent_turns: int = DEFAULT_RECENT_TURNS,
    cancel: CancellationToken | None = None,
) -> MicrocompactionResult:
    """Clear old re-derivable tool results until the conversation fits."""

    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if recent_turns < 1:
        raise ValueError(f"recent_turns must be >= 1, got {recent_turns}")
    if cancel is not None:
        cancel.raise_if_cancelled()

    original = list(messages)
    before_tokens = estimate_messages_tokens(original)
    if before_tokens <= budget:
        return MicrocompactionResult(
            messages=original,
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            budget=budget,
            changed=False,
            recent_turns=recent_turns,
        )

    recent_start = _recent_window_start(original, recent_turns)
    tool_names: dict[str, str] = {}
    clearable_indices: list[int] = []
    for index, message in enumerate(original):
        if message.role == Role.ASSISTANT:
            for tool_call in message.tool_calls:
                tool_names[tool_call.id] = tool_call.name
        if index >= recent_start or message.role != Role.TOOL:
            continue
        tool_result = message.tool_result
        if (
            tool_result is not None
            and tool_names.get(tool_result.tool_call_id) in COMPACTABLE_TOOL_NAMES
            and tool_result.content
            and not _is_cleared_tool_result_content(tool_result.content)
        ):
            clearable_indices.append(index)

    compacted = list(original)
    after_tokens = before_tokens
    cleared_tool_results = 0
    for index in clearable_indices:
        message = compacted[index]
        tool_result = message.tool_result
        assert tool_result is not None
        cleared_result = replace(
            tool_result,
            content=_cleared_tool_result_content(tool_result),
        )
        cleared_message = replace(message, tool_result=cleared_result)
        # The total is a plain sum over messages, so subtracting one message's
        # change is exactly a re-estimate of the whole conversation -- and stays
        # linear in the number cleared instead of re-walking every message each
        # time.
        after_tokens -= estimate_message_tokens(message) - estimate_message_tokens(
            cleared_message
        )
        compacted[index] = cleared_message
        cleared_tool_results += 1
        if after_tokens <= budget:
            break

    return MicrocompactionResult(
        messages=compacted,
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        budget=budget,
        changed=cleared_tool_results > 0,
        cleared_tool_results=cleared_tool_results,
        recent_turns=recent_turns,
    )


def compact_messages_for_budget(
    messages: list[Message],
    *,
    budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    recent_turns: int = DEFAULT_RECENT_TURNS,
    cancel: CancellationToken | None = None,
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
    if cancel is not None:
        cancel.raise_if_cancelled()

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

    microcompacted = microcompact_messages(
        original,
        budget=budget,
        recent_turns=recent_turns,
        cancel=cancel,
    )
    if microcompacted.after_tokens <= budget:
        return CompactionResult(
            messages=microcompacted.messages,
            before_tokens=before_tokens,
            after_tokens=microcompacted.after_tokens,
            budget=budget,
            changed=True,
            cleared_tool_results=microcompacted.cleared_tool_results,
            recent_turns=recent_turns,
        )

    working = microcompacted.messages
    recent_start = _recent_window_start(working, recent_turns)
    prefix_indices = _preserved_prefix_indices(working, recent_start)
    recent_indices = set(range(recent_start, len(working)))
    dropped_indices = [
        index
        for index in range(len(working))
        if index not in prefix_indices and index not in recent_indices
    ]
    if not dropped_indices:
        raise _impossible_error(before_tokens, budget, recent_turns)

    prefix = [working[index] for index in sorted(prefix_indices)]
    recent = [working[index] for index in range(recent_start, len(working))]
    # Summarize from the *uncleared* messages. Clearing preserves list
    # positions, so the indices still line up, and the summary is the only
    # surviving trace of what was dropped -- excerpting a clearing marker
    # would spend that trace describing bookkeeping instead of content.
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
                cleared_tool_results=microcompacted.cleared_tool_results,
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
    if result.cleared_tool_results and not result.dropped_messages:
        summary_part = "no summary needed"
    else:
        summary_part = (
            f"inserted {result.summary_messages} summary message"
            if result.summary_messages
            else "no summary fit"
        )
    cleared_part = (
        f"cleared {result.cleared_tool_results} old tool "
        f"result{'s' if result.cleared_tool_results != 1 else ''}, "
        if result.cleared_tool_results
        else ""
    )
    return (
        "Compacted conversation: "
        f"{cleared_part}dropped {result.dropped_messages} older messages, {summary_part}, "
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


def _cleared_tool_result_content(tool_result: ToolResult) -> str:
    if tool_result.offloaded is None:
        return CLEARED_CONTENT_MARKER
    offloaded = tool_result.offloaded
    return (
        "[old tool result content cleared; read_tool_result id "
        f'"{offloaded.id}" still holds {offloaded.characters} characters]'
    )


def _is_cleared_tool_result_content(content: str) -> bool:
    return content == CLEARED_CONTENT_MARKER or content.startswith(
        CLEARED_CONTENT_MARKER[:-1] + ";"
    )


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
            content = (
                "" if _is_cleared_tool_result_content(result.content) else result.content
            )
            snippet = content or result.error or f"tool result {result.tool_call_id}"
        if not snippet:
            attachments = [
                block for block in message.content if not isinstance(block, TextBlock)
            ]
            if attachments:
                count = len(attachments)
                types = ", ".join(block.media_type for block in attachments)
                snippet = f"[{count} attachment{'' if count == 1 else 's'}: {types}]"
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
