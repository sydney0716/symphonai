"""Pure conversation compaction helpers for long-running chat sessions."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.model_table import compact_at_for_model, context_window_for_model
from symphonai_api.tools.metadata import changed_paths
from symphonai_api.models import DocumentBlock, ImageBlock, Message, Role, TextBlock, ToolCall, ToolResult
from symphonai_api.providers.base import tool_result_text

_RESTORE_PREFIX = "Restored after compaction:\n"


def is_restore_message(message: Message) -> bool:
    return message.role == Role.SYSTEM and message.text.startswith(_RESTORE_PREFIX)


def restored_state(messages) -> dict:
    state = {}
    for message in messages:
        if is_restore_message(message):
            try:
                value = json.loads(message.text[len(_RESTORE_PREFIX):])
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                state = value
    return state


DEFAULT_CONTEXT_TOKEN_BUDGET = 16_000
DEFAULT_CONTEXT_WINDOW = 128_000
REPLY_RESERVE_TOKENS = 32_000
COMPACTION_BUFFER_TOKENS = 13_000
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
MODEL_SUMMARY_PROMPT = """You are summarizing the earlier part of a coding session so the work can continue without it. Respond with text only; do not call tools. Write these sections:
1. Requests and intent: every explicit request the user made, in detail.
2. Key technical concepts.
3. Files and code: each file examined, changed or created, the snippets that matter, and why.
4. Errors and fixes, including any feedback the user gave.
5. Problem solving: what was solved and what is still being investigated.
6. User messages: every user message, as close to verbatim as you can.
7. Pending tasks.
8. Current work: precisely what was in progress where this part ends.
Be specific: names, paths, values."""


def budget_for_model(
    wire_format: int, model: str | None, limit: int | None = None,
) -> int:
    """Derive the compaction budget from a known model window or the default."""
    context_window = context_window_for_model(wire_format, model)
    if context_window is None:
        context_window = DEFAULT_CONTEXT_WINDOW
    budget = context_window - REPLY_RESERVE_TOKENS - COMPACTION_BUFFER_TOKENS
    model_limit = compact_at_for_model(wire_format, model)
    if model_limit is not None:
        budget = min(budget, model_limit)
    if limit is not None:
        budget = min(budget, limit)
    return budget


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


MicrocompactionResult = CompactionResult

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
    assistant_turn_window: bool | None = None,
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

    recent_start = recent_window_start(
        original, recent_turns, assistant_turn_window=assistant_turn_window
    )
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
    assistant_turn_window: bool | None = None,
    cancel: CancellationToken | None = None,
    summarize: Callable[[list[Message]], str] | None = None,
    force: bool = False,
    recent_files: list[tuple[str, str]] | None = None,
) -> CompactionResult:
    """Compact a conversation if it exceeds `budget`, or when `force` is true.

    The compactor preserves system messages, the first user goal, and the
    recent window. When user turns are fewer than `recent_turns`, that window
    counts assistant turns with their following tool results and preserves all
    user messages. Older middle messages are replaced by a summary when there
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
    compacted = original
    after_tokens = before_tokens
    cleared_tool_results = 0
    dropped_messages = 0
    summary_messages = 0
    if before_tokens > budget or force:
        if not force:
            microcompacted = microcompact_messages(
                original, budget=budget, recent_turns=recent_turns,
                assistant_turn_window=assistant_turn_window, cancel=cancel,
            )
            compacted = microcompacted.messages
            after_tokens = microcompacted.after_tokens
            cleared_tool_results = microcompacted.cleared_tool_results
        if force or after_tokens > budget:
            assistant_window = (
                sum(message.role == Role.USER for message in compacted) < recent_turns
                if assistant_turn_window is None else assistant_turn_window
            )
            recent_start = recent_window_start(
                compacted, recent_turns, assistant_turn_window=assistant_window,
            )
            prefix_indices = _preserved_prefix_indices(
                compacted, recent_start, assistant_window,
            )
            recent_indices = set(range(recent_start, len(compacted)))
            dropped_indices = [
                index for index in range(len(compacted))
                if index not in prefix_indices and index not in recent_indices
            ]
            if not dropped_indices:
                if not force:
                    raise _impossible_error(before_tokens, budget, recent_turns)
            else:
                prefix = [compacted[index] for index in sorted(prefix_indices)]
                recent = [compacted[index] for index in range(recent_start, len(compacted))]
                # Summaries retain the original content, before microcompaction clears it.
                dropped = [original[index] for index in dropped_indices]
                # Replace the prior restoration rather than accumulating stale file bodies.
                restored = restored_state(original)
                prefix = [message for message in prefix if not is_restore_message(message)]
                plan = restored.get("plan")
                for message in original:
                    for call in message.tool_calls:
                        if call.name == "update_plan":
                            plan = call.arguments
                successful = {message.tool_result.tool_call_id for message in original
                              if message.tool_result is not None and message.tool_result.ok}
                paths = list(restored.get("files_changed", []))
                for message in dropped:
                    for call in message.tool_calls:
                        if call.id in successful:
                            paths.extend(path for path in changed_paths(call.name, call.arguments) if path not in paths)
                restore = {}
                if plan is not None:
                    restore["plan"] = plan
                if paths:
                    restore["files_changed"] = paths[:30]

                def restore_message(state):
                    return Message(Role.SYSTEM, _RESTORE_PREFIX
                                   + json.dumps(state, ensure_ascii=False, indent=2))

                mandatory = [restore_message(restore)] if restore else []
                selected = None
                if summarize is not None:
                    try:
                        summary_text = summarize(dropped)
                    except OperationCancelled:
                        raise
                    except Exception:
                        summary_text = ""
                    if isinstance(summary_text, str) and summary_text.strip():
                        candidate = [
                            *prefix,
                            Message(
                                role=Role.SYSTEM,
                                content=(
                                    "Summary of the earlier conversation, written when it was compacted:\n\n"
                                    + summary_text
                                ),
                            ),
                            *mandatory,
                            *recent,
                        ]
                        if estimate_messages_tokens(candidate) <= budget:
                            selected = candidate
                            summary_messages = 1
                if selected is None:
                    for max_summary_chars in (800, 320, 120, 0):
                        candidate = list(prefix)
                        if max_summary_chars > 0 or mandatory or recent_files:
                            candidate.append(Message(
                                role=Role.SYSTEM,
                                content=_summarize_dropped_messages(dropped, max_chars=max_summary_chars or 40),
                            ))
                        candidate.extend(mandatory)
                        candidate.extend(recent)
                        if estimate_messages_tokens(candidate) <= budget:
                            selected = candidate
                            summary_messages = int(max_summary_chars > 0 or bool(mandatory) or bool(recent_files))
                            break
                if selected is None:
                    raise _impossible_error(before_tokens, budget, recent_turns)
                # Disk contents are supplied by the loop, newest read first.
                full_reads = {call.id: call.arguments.get("path") for message in recent
                              for call in message.tool_calls if call.name == "read_file"}
                for path, content in (recent_files or [])[:5]:
                    rendered = "\n".join(f"{index}\t{line}" for index, line in enumerate(content.splitlines(), 1))
                    if any(message.tool_result is not None and message.tool_result.ok
                           and full_reads.get(message.tool_result.tool_call_id) == path
                           and message.tool_result.content in (content, rendered)
                           for message in recent):
                        continue
                    if estimate_text_tokens(content) > 5000:
                        content = content[:20000] + "\n[truncated at 5,000 estimated tokens]"
                    state = {**restore, "recent_files": [*restore.get("recent_files", []),
                                                        {"path": path, "content": content}]}
                    insertion = len(prefix) + summary_messages
                    candidate = list(selected)
                    if restore:
                        candidate[insertion] = restore_message(state)
                    else:
                        candidate.insert(insertion, restore_message(state))
                    if estimate_messages_tokens(candidate) <= budget:
                        selected = candidate
                        restore = state
                compacted = selected
                after_tokens = estimate_messages_tokens(compacted)
                dropped_messages = len(dropped)

    return CompactionResult(
        messages=compacted,
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        budget=budget,
        changed=cleared_tool_results > 0 or dropped_messages > 0,
        dropped_messages=dropped_messages,
        summary_messages=summary_messages,
        cleared_tool_results=cleared_tool_results,
        recent_turns=recent_turns,
    )


def render_dropped_messages(messages: list[Message]) -> str:
    transcript: list[str] = []
    for index, message in enumerate(messages, start=1):
        lines = [f"[{message.role.value.upper()} message {index}]"]
        if message.text:
            lines.append(message.text)
        for call in message.tool_calls:
            arguments = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
            lines.append(f"Tool call: {call.name}\nArguments: {arguments}")
        if message.tool_result is not None:
            result = message.tool_result
            label = "Tool result" if result.ok else "Tool error"
            content = tool_result_text(result)
            lines.append(f"{label}:\n{content}")
        for block in message.content:
            if isinstance(block, (ImageBlock, DocumentBlock)):
                lines.append(f"[attachment: {block.media_type}]")
        transcript.append("\n".join(lines))
    return "\n\n".join(transcript)




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


def recent_window_start(
    messages: list[Message],
    recent_turns: int,
    *,
    assistant_turn_window: bool | None = None,
) -> int:
    user_indices = [index for index, message in enumerate(messages) if message.role == Role.USER]
    use_assistant_window = (
        len(user_indices) < recent_turns
        if assistant_turn_window is None
        else assistant_turn_window
    )
    if not use_assistant_window:
        return user_indices[-recent_turns] if user_indices else max(0, len(messages) - recent_turns)
    assistant_indices = [
        index for index, message in enumerate(messages)
        if message.role == Role.ASSISTANT
    ]
    if len(assistant_indices) <= recent_turns:
        return 0
    return assistant_indices[-recent_turns]


def _preserved_prefix_indices(
    messages: list[Message], recent_start: int, assistant_turn_window: bool
) -> set[int]:
    prefix_indices = {
        index
        for index, message in enumerate(messages[:recent_start])
        if message.role == Role.SYSTEM
    }
    user_indices = [
        index for index, message in enumerate(messages) if message.role == Role.USER
    ]
    if assistant_turn_window:
        prefix_indices.update(index for index in user_indices if index < recent_start)
    elif user_indices and user_indices[0] < recent_start:
        prefix_indices.add(user_indices[0])
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
