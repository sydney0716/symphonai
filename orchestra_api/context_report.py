"""Pure context-window accounting for conversations callers already hold.

This module deliberately has no agent-loop or event wiring. Phase 18's app is
the intended consumer; until then, callers can request the data directly
without building a presentation interface into the runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from orchestra_api.compaction import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_text_tokens,
)
from orchestra_api.instructions import InstructionSet, LoadedInstruction
from orchestra_api.models import Message, Role


class ContextSource(str, Enum):
    SYSTEM_PROMPT = "system_prompt"
    INSTRUCTIONS = "instructions"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


@dataclass(frozen=True)
class ContextEntry:
    source: ContextSource
    label: str
    tokens: int
    characters: int
    turn_id: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ContextReport:
    entries: tuple[ContextEntry, ...]
    total_tokens: int
    unattributed_tokens: int
    budget: int
    warnings: tuple[str, ...] = ()

    @property
    def remaining_tokens(self) -> int:
        return self.budget - self.total_tokens

    def by_source(self) -> dict[ContextSource, int]:
        subtotals: dict[ContextSource, int] = {}
        for entry in self.entries:
            subtotals[entry.source] = subtotals.get(entry.source, 0) + entry.tokens
        return subtotals

    def render(self) -> str:
        headers = ("source", "label", "tokens", "characters", "turn_id", "detail")
        rows = [
            (
                entry.source.value,
                entry.label,
                str(entry.tokens),
                str(entry.characters),
                entry.turn_id or "",
                entry.detail or "",
            )
            for entry in self.entries
        ]
        widths = [
            max(len(headers[index]), *(len(row[index]) for row in rows))
            if rows
            else len(headers[index])
            for index in range(len(headers))
        ]

        def format_row(row: tuple[str, ...]) -> str:
            return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()

        lines = [format_row(headers), format_row(tuple("-" * width for width in widths))]
        lines.extend(format_row(row) for row in rows)
        lines.extend(
            (
                "",
                f"total_tokens: {self.total_tokens}",
                f"unattributed_tokens: {self.unattributed_tokens}",
                f"budget: {self.budget}",
                f"remaining_tokens: {self.remaining_tokens}",
                "",
                "by_source:",
            )
        )
        lines.extend(
            f"  {source.value}: {tokens}" for source, tokens in self.by_source().items()
        )
        if self.warnings:
            lines.extend(("", "warnings:"))
            lines.extend(f"  {warning}" for warning in self.warnings)
        return "\n".join(lines)


def account_context(
    messages: Sequence[Message],
    *,
    budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    instructions: InstructionSet | None = None,
) -> ContextReport:
    """Describe the estimated context cost using compaction's exact total."""

    conversation = list(messages)
    total_tokens = estimate_messages_tokens(conversation)
    entries: list[ContextEntry] = []
    warnings: list[str] = []
    tool_names: dict[str, str] = {}

    split_first = False
    if instructions is not None:
        rendered_instructions = instructions.render()
        if rendered_instructions:
            split_first = bool(
                conversation
                and conversation[0].role == Role.SYSTEM
                and conversation[0].text.startswith(rendered_instructions)
            )
            if split_first:
                entries.extend(_instruction_entries(instructions, conversation[0].turn_id))
                remainder = conversation[0].text[len(rendered_instructions) :]
                if remainder.startswith("\n\n"):
                    remainder = remainder[2:]
                if remainder:
                    entries.append(
                        ContextEntry(
                            source=ContextSource.SYSTEM_PROMPT,
                            label="system prompt",
                            tokens=estimate_text_tokens(remainder),
                            characters=len(remainder),
                            turn_id=conversation[0].turn_id,
                        )
                    )
            else:
                warnings.append("instruction set does not match the first system message")

    for index, message in enumerate(conversation):
        if index == 0 and split_first:
            _remember_tool_calls(message, tool_names)
            continue
        entries.append(_message_entry(message, tool_names, warnings))
        _remember_tool_calls(message, tool_names)

    attributed_tokens = sum(entry.tokens for entry in entries)
    return ContextReport(
        entries=tuple(entries),
        total_tokens=total_tokens,
        unattributed_tokens=total_tokens - attributed_tokens,
        budget=budget,
        warnings=tuple(warnings),
    )


def _instruction_entries(
    instructions: InstructionSet,
    turn_id: str | None,
) -> list[ContextEntry]:
    entries: list[ContextEntry] = []
    for instruction in instructions.entries:
        rendered = instructions.render_entry(instruction)
        if not rendered:
            continue
        entries.append(
            ContextEntry(
                source=ContextSource.INSTRUCTIONS,
                label=_instruction_label(instruction, instructions),
                tokens=estimate_text_tokens(rendered),
                characters=len(rendered),
                turn_id=turn_id,
                detail=_instruction_detail(instruction, instructions),
            )
        )
    return entries


def _message_entry(
    message: Message,
    tool_names: dict[str, str],
    warnings: list[str],
) -> ContextEntry:
    source = {
        Role.SYSTEM: ContextSource.SYSTEM_PROMPT,
        Role.USER: ContextSource.USER,
        Role.ASSISTANT: ContextSource.ASSISTANT,
        Role.TOOL: ContextSource.TOOL_RESULT,
    }[message.role]
    label = {
        Role.SYSTEM: "system prompt",
        Role.USER: "user message",
        Role.ASSISTANT: "assistant message",
        Role.TOOL: "tool message",
    }[message.role]
    detail: str | None = None
    characters = len(message.text)

    if message.role == Role.ASSISTANT and message.tool_calls:
        detail = f"calls {', '.join(call.name for call in message.tool_calls)}"
    elif message.role == Role.TOOL and message.tool_result is not None:
        result = message.tool_result
        matched_tool = result.tool_call_id in tool_names
        label = tool_names.get(result.tool_call_id, "unknown tool")
        if not matched_tool:
            warnings.append(f"orphan tool result for call id {result.tool_call_id}")
        if result.cancelled:
            status = "cancelled"
        elif result.ok:
            status = "ok"
        else:
            status = "error"
        detail = f"call id {result.tool_call_id}, {status}"
        characters = len(result.content or result.error or "")

    return ContextEntry(
        source=source,
        label=label,
        tokens=estimate_message_tokens(message),
        characters=characters,
        turn_id=message.turn_id,
        detail=detail,
    )


def _remember_tool_calls(message: Message, tool_names: dict[str, str]) -> None:
    if message.role == Role.ASSISTANT:
        for call in message.tool_calls:
            tool_names[call.id] = call.name


def _instruction_label(
    instruction: LoadedInstruction,
    instructions: InstructionSet,
) -> str:
    if instruction.path is None:
        return "agent instructions"
    return instructions.display_path(instruction.path)


def _instruction_detail(
    instruction: LoadedInstruction,
    instructions: InstructionSet,
) -> str:
    detail = instruction.scope.value
    if instruction.parent is not None:
        detail += f", included by {instructions.display_path(instruction.parent)}"
    return detail
