"""Checks for context-window accounting and provenance."""

from __future__ import annotations

import copy

from orchestra_api.compaction import (
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_text_tokens,
)
from orchestra_api.context_report import ContextSource, account_context
from orchestra_api.instructions import load_instructions
from orchestra_api.models import ImageBlock, Message, Role, ToolCall, ToolResult
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


@check("context.reconciles_unsplit")
def check_reconciles_unsplit() -> None:
    messages = [
        Message(role=Role.SYSTEM, content="system prompt", turn_id="turn-system"),
        Message(role=Role.USER, content="question", turn_id="turn-user"),
        Message(role=Role.ASSISTANT, content="answer", turn_id="turn-assistant"),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="missing", ok=True, content="result"),
            turn_id="turn-tool",
        ),
    ]
    report = account_context(messages)
    direct_total = estimate_messages_tokens(messages)
    entry_total = sum(entry.tokens for entry in report.entries)
    if report.total_tokens != direct_total:
        fail(f"report total {report.total_tokens} did not match direct total {direct_total}")
    if entry_total + report.unattributed_tokens != report.total_tokens:
        fail(f"context entries did not reconcile: {report!r}")
    if len(report.entries) != len(messages) or report.unattributed_tokens != 0:
        fail(f"unsplit messages were not represented one-for-one: {report!r}")
    if [entry.turn_id for entry in report.entries] != [message.turn_id for message in messages]:
        fail(f"turn ids were not copied: {report.entries!r}")


@check("context.splits_instructions")
def check_splits_instructions() -> None:
    with workspace() as ws:
        docs = ws.root / "docs"
        docs.mkdir()
        (ws.root / "CLAUDE.md").write_text("project rule\n@docs/style.md")
        (docs / "style.md").write_text("included style")
        loaded = load_instructions(
            ws.policy,
            user_home=ws.root / "missing-user-home",
            agent_instructions="agent rule",
        )
        caller_prompt = "caller system prompt"
        messages = [
            Message(
                role=Role.SYSTEM,
                content=f"{loaded.render()}\n\n{caller_prompt}",
                turn_id="system-turn",
            ),
            Message(role=Role.USER, content="hello"),
        ]
        report = account_context(messages, instructions=loaded)

    system_entries = [
        entry
        for entry in report.entries
        if entry.source in (ContextSource.INSTRUCTIONS, ContextSource.SYSTEM_PROMPT)
    ]
    non_empty_instructions = [entry for entry in loaded.entries if entry.text]
    if len(system_entries) != len(non_empty_instructions) + 1:
        fail(f"system message did not split into its real instruction parts: {report!r}")
    if [entry.label for entry in system_entries] != [
        "CLAUDE.md",
        "docs/style.md",
        "agent instructions",
        "system prompt",
    ]:
        fail(f"split labels lost instruction provenance: {system_entries!r}")
    if system_entries[1].detail != "project, included by CLAUDE.md":
        fail(f"included instruction did not name its parent: {system_entries[1]!r}")
    if system_entries[-1].characters != len(caller_prompt):
        fail(f"caller prompt remainder was not isolated: {system_entries[-1]!r}")
    split_tokens = sum(entry.tokens for entry in system_entries)
    measured_difference = estimate_message_tokens(messages[0]) - split_tokens
    if measured_difference <= 0:
        fail(f"split overhead was not reported as unattributed: {report!r}")
    if sum(entry.tokens for entry in report.entries) + report.unattributed_tokens != report.total_tokens:
        fail(f"split report did not reconcile: {report!r}")


@check("context.tool_provenance")
def check_tool_provenance() -> None:
    messages = [
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="call-1", name="read_file", arguments={"path": "a"})],
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="call-1", ok=True, content="file text"),
        ),
    ]
    report = account_context(messages)
    tool_entry = report.entries[1]
    if tool_entry.source != ContextSource.TOOL_RESULT or tool_entry.label != "read_file":
        fail(f"tool result did not recover its earlier call name: {tool_entry!r}")
    if tool_entry.detail != "call id call-1, ok":
        fail(f"tool result status detail was wrong: {tool_entry!r}")
    if report.warnings:
        fail(f"matched tool result unexpectedly warned: {report.warnings!r}")


@check("context.orphan_tool_warning")
def check_orphan_tool_warning() -> None:
    report = account_context(
        [
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(
                    tool_call_id="orphan-7",
                    ok=False,
                    error="not found",
                ),
            )
        ]
    )
    if report.entries[0].label != "unknown tool":
        fail(f"orphan tool received a fictional name: {report.entries[0]!r}")
    if report.entries[0].detail != "call id orphan-7, error":
        fail(f"orphan status detail was wrong: {report.entries[0]!r}")
    if len(report.warnings) != 1 or "orphan-7" not in report.warnings[0]:
        fail(f"orphan warning did not name the call id: {report.warnings!r}")


@check("context.mismatched_instructions")
def check_mismatched_instructions() -> None:
    with workspace() as ws:
        (ws.root / "AGENTS.md").write_text("loaded rule")
        loaded = load_instructions(ws.policy, user_home=ws.root / "missing-user-home")
        messages = [Message(role=Role.SYSTEM, content="different system prompt")]
        report = account_context(messages, instructions=loaded)
    if len(report.entries) != 1 or report.entries[0].source != ContextSource.SYSTEM_PROMPT:
        fail(f"mismatched instructions split the system message: {report!r}")
    if report.entries[0].tokens != estimate_message_tokens(messages[0]):
        fail(f"mismatched system message lost its full estimate: {report.entries[0]!r}")
    if report.unattributed_tokens != 0:
        fail(f"unsplit mismatched message left unattributed tokens: {report!r}")
    if len(report.warnings) != 1 or "does not match" not in report.warnings[0]:
        fail(f"mismatched instructions did not warn: {report.warnings!r}")


@check("context.subtotals_and_budget")
def check_subtotals_and_budget() -> None:
    report = account_context(
        [
            Message(role=Role.USER, content="large request " * 20),
            Message(role=Role.ASSISTANT, content="large answer " * 20),
        ],
        budget=10,
    )
    subtotals = report.by_source()
    if sum(subtotals.values()) != sum(entry.tokens for entry in report.entries):
        fail(f"source subtotals did not reconcile with entries: {subtotals!r}")
    if ContextSource.SYSTEM_PROMPT in subtotals or ContextSource.TOOL_RESULT in subtotals:
        fail(f"absent sources appeared with zero totals: {subtotals!r}")
    if report.remaining_tokens != report.budget - report.total_tokens:
        fail(f"remaining tokens did not subtract the total: {report!r}")
    if report.remaining_tokens >= 0:
        fail(f"over-budget report clamped or hid its overage: {report!r}")


@check("context.assistant_attachments_and_immutability")
def check_assistant_attachments_and_immutability() -> None:
    attached = Message(
        role=Role.ASSISTANT,
        content=["analysis", ImageBlock(data="a" * 4_000, media_type="image/png")],
        tool_calls=[
            ToolCall(id="read-1", name="read_file"),
            ToolCall(id="grep-1", name="grep"),
        ],
        turn_id="attachment-turn",
    )
    messages = [Message(role=Role.USER, content="inspect"), attached]
    before_list = copy.deepcopy(messages)
    before_messages = tuple(copy.deepcopy(message) for message in messages)
    report = account_context(messages)
    entry = report.entries[1]
    if entry.detail != "calls read_file, grep":
        fail(f"assistant tool calls were not named in detail: {entry!r}")
    if entry.tokens != estimate_message_tokens(attached):
        fail(f"attachment message did not retain its full estimate: {entry!r}")
    without_attachment = Message(
        role=Role.ASSISTANT,
        content="analysis",
        tool_calls=attached.tool_calls,
        turn_id=attached.turn_id,
    )
    if entry.tokens <= estimate_message_tokens(without_attachment):
        fail(f"attachment tokens were not included: {entry!r}")
    if messages != before_list:
        fail(f"accounting mutated the message list: before={before_list!r}, after={messages!r}")
    for index, (before, after) in enumerate(zip(before_messages, messages, strict=True)):
        if after != before:
            fail(f"accounting mutated message {index}: before={before!r}, after={after!r}")


@check("context.render_plain_text")
def check_render_plain_text() -> None:
    message = Message(
        role=Role.TOOL,
        tool_result=ToolResult(tool_call_id="render-orphan", ok=False, cancelled=True),
    )
    report = account_context([message], budget=3)
    rendered = report.render()
    required = (
        f"total_tokens: {report.total_tokens}",
        f"unattributed_tokens: {report.unattributed_tokens}",
        "budget: 3",
        f"remaining_tokens: {report.remaining_tokens}",
        report.warnings[0],
    )
    if not all(value in rendered for value in required):
        fail(f"render omitted totals, budget, or warnings: {rendered!r}")
    if "<table" in rendered or "</" in rendered:
        fail(f"render returned HTML instead of plain text: {rendered!r}")
    if len(rendered.splitlines()) < len(report.entries) + 8:
        fail(f"render did not include one row per entry and totals: {rendered!r}")


@check("context.empty_instruction_set")
def check_empty_instruction_set() -> None:
    with workspace() as ws:
        loaded = load_instructions(ws.policy, user_home=ws.root / "missing-user-home")
        if loaded.render() != "":
            fail(f"empty instruction set rendered content: {loaded.render()!r}")
        caller_prompt = "caller system prompt only"
        composed = "\n\n".join(part for part in (loaded.render(), caller_prompt) if part)
        messages = [Message(role=Role.SYSTEM, content=composed)]
        report = account_context(messages, instructions=loaded)

    if composed != caller_prompt:
        fail(f"empty instructions introduced a separator: {composed!r}")
    if len(report.entries) != 1 or report.entries[0].source != ContextSource.SYSTEM_PROMPT:
        fail(f"empty instructions did not leave one system entry: {report!r}")
    if report.entries[0].tokens != estimate_message_tokens(messages[0]):
        fail(f"empty instructions lost the full message estimate: {report.entries[0]!r}")
    if report.unattributed_tokens != 0 or report.warnings:
        fail(f"empty instructions split or warned: {report!r}")


@check("context.split_covers_rendered_text")
def check_split_covers_rendered_text() -> None:
    with workspace() as ws:
        docs = ws.root / "docs"
        docs.mkdir()
        (ws.root / "CLAUDE.md").write_text("project rule\n@docs/style.md")
        (docs / "style.md").write_text("included style")
        loaded = load_instructions(
            ws.policy,
            user_home=ws.root / "missing-user-home",
            agent_instructions="agent rule",
        )
        rendered = loaded.render()
        messages = [
            Message(role=Role.SYSTEM, content=f"{rendered}\n\ncaller prompt"),
            Message(role=Role.USER, content="hello"),
        ]
        report = account_context(messages, instructions=loaded)

    instruction_rows = [
        entry for entry in report.entries if entry.source == ContextSource.INSTRUCTIONS
    ]
    covered_characters = sum(entry.characters for entry in instruction_rows) + 2 * (
        len(instruction_rows) - 1
    )
    if covered_characters != len(rendered):
        fail(
            "instruction rows did not cover the rendered prefix: "
            f"covered {covered_characters}, rendered {len(rendered)}"
        )
    if sum(entry.tokens for entry in report.entries) + report.unattributed_tokens != report.total_tokens:
        fail(f"covered split did not reconcile: {report!r}")
    if report.total_tokens != estimate_messages_tokens(messages):
        fail(f"covered split total did not use the compaction estimator: {report!r}")


@check("context.tool_message_without_result")
def check_tool_message_without_result() -> None:
    report = account_context(
        [
            Message(role=Role.TOOL, content="tool protocol note"),
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(tool_call_id="still-orphan", ok=False, error="missing"),
            ),
        ]
    )
    plain_tool, orphan = report.entries
    if plain_tool.label != "tool message" or plain_tool.detail is not None:
        fail(f"result-less tool message claimed a tool result: {plain_tool!r}")
    if orphan.label != "unknown tool" or orphan.detail != "call id still-orphan, error":
        fail(f"orphan tool-result behavior changed: {orphan!r}")
    if len(report.warnings) != 1 or "still-orphan" not in report.warnings[0]:
        fail(f"tool-message warnings were wrong: {report.warnings!r}")
