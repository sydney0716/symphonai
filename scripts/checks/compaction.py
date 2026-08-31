"""Fixture-free checks for compaction."""

from __future__ import annotations

from dataclasses import replace

from orchestra_api.cancellation import CancellationToken, OperationCancelled
from orchestra_api.compaction import (
    CLEARED_CONTENT_MARKER,
    COMPACTABLE_TOOL_NAMES,
    ContextCompactionError,
    compact_messages_for_budget,
    describe_compaction,
    estimate_messages_tokens,
    microcompact_messages,
)
from orchestra_api.models import ImageBlock, Message, Role, ToolCall, ToolResult
from orchestra_api.tool_results import ToolResultStore, offload_tool_result
from scripts.checks.harness import check, fail


def _single_clearable_conversation() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="system prompt"),
        Message(role=Role.USER, content="first goal"),
        Message(
            role=Role.ASSISTANT,
            content="I will read the old result.",
            tool_calls=[ToolCall(id="old-read", name="read_file")],
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(
                tool_call_id="old-read",
                ok=True,
                content="old re-derivable content " * 120,
            ),
        ),
        Message(role=Role.USER, content="latest request"),
    ]


@check("compaction.cancellation_at_entry")
def check_compaction_cancellation_at_entry() -> None:
    compact_cancel = CancellationToken()
    compact_cancel.cancel()
    try:
        compact_messages_for_budget(
            [Message(role=Role.USER, content="cancel compaction")],
            cancel=compact_cancel,
        )
    except OperationCancelled:
        pass
    else:
        fail("pre-cancelled compaction did not raise OperationCancelled")

@check("compaction.under_budget_unchanged")
def check_compaction_under_budget_unchanged() -> None:
    # -- context compaction: under budget leaves messages untouched --
    compact_under_messages = [
        Message(role=Role.SYSTEM, content="stay concise"),
        Message(role=Role.USER, content="first goal"),
        Message(role=Role.ASSISTANT, content="short answer"),
    ]
    compact_under = compact_messages_for_budget(compact_under_messages, budget=1_000)
    if compact_under.changed or compact_under.messages != compact_under_messages:
        fail(f"under-budget compaction should leave messages untouched, got {compact_under}")

@check("compaction.preserves_required_context")
def check_compaction_preserves_required_context() -> None:
    # -- context compaction: over budget drops old middle while staying coherent --
    compact_over_messages = [
        Message(role=Role.SYSTEM, content="system prompt must stay"),
        Message(role=Role.USER, content="earliest user goal must stay"),
        Message(role=Role.ASSISTANT, content="old assistant detail " * 120),
        Message(role=Role.USER, content="old follow-up " * 120),
        Message(role=Role.ASSISTANT, content="old tool analysis " * 120),
        Message(role=Role.USER, content="latest user request must stay"),
    ]
    compact_over = compact_messages_for_budget(
        compact_over_messages,
        budget=140,
        recent_turns=1,
    )
    compacted_contents = [message.text for message in compact_over.messages]
    if not all(
        isinstance(message, Message) and isinstance(message.content, tuple)
        for message in compact_over.messages
    ):
        fail(f"compaction returned non-canonical message shapes: {compact_over.messages!r}")
    if not compact_over.changed or compact_over.dropped_messages < 1:
        fail(f"expected over-budget conversation to compact, got {compact_over}")
    if compact_over.after_tokens > compact_over.budget:
        fail(f"compacted conversation still exceeds budget: {compact_over}")
    if "system prompt must stay" not in compacted_contents:
        fail("compaction did not preserve the system prompt")
    if "earliest user goal must stay" not in compacted_contents:
        fail("compaction did not preserve the earliest user goal")
    if "latest user request must stay" not in compacted_contents:
        fail("compaction did not preserve the latest user turn")
    if not any("Earlier conversation compacted" in content for content in compacted_contents):
        fail(f"compaction did not insert a useful summary: {compacted_contents!r}")

@check("compaction.impossible_budget_fails")
def check_compaction_impossible_budget_fails() -> None:
    # -- context compaction: impossible budget raises a clear local error --
    impossible_messages = [
        Message(role=Role.SYSTEM, content="system prompt"),
        Message(role=Role.USER, content="x" * 4_000),
    ]
    try:
        compact_messages_for_budget(impossible_messages, budget=100, recent_turns=1)
    except ContextCompactionError as exc:
        message = str(exc)
        if "Increase the budget" not in message or "recent_turns" not in message:
            fail(f"compaction error was not actionable: {message!r}")
    else:
        fail("expected impossible compaction to raise ContextCompactionError")


@check("compaction.microcompaction_clears_re_derivable")
def check_microcompaction_clears_re_derivable() -> None:
    if COMPACTABLE_TOOL_NAMES != frozenset(
        {"read_file", "list_files", "glob", "grep", "run_shell", "read_tool_result"}
    ):
        fail(f"compactable tool set changed: {COMPACTABLE_TOOL_NAMES!r}")
    assistant = Message(
        role=Role.ASSISTANT,
        content=[
            "assistant prose stays exact",
            ImageBlock(data="aW1hZ2U=", media_type="image/png"),
        ],
        tool_calls=[
            ToolCall(id="old-read", name="read_file"),
            ToolCall(id="old-grep", name="grep"),
            ToolCall(id="old-shell", name="run_shell"),
        ],
        turn_id="assistant-turn",
    )
    messages = [
        Message(role=Role.USER, content="original user goal", turn_id="first-user"),
        assistant,
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="old-read", ok=True, content="r" * 600),
            turn_id="read-turn",
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="old-grep", ok=True, content="g" * 600),
            turn_id="grep-turn",
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="old-shell", ok=True, content="s" * 600),
            turn_id="shell-turn",
        ),
        Message(role=Role.USER, content="latest request", turn_id="latest-user"),
    ]
    prose_before = [
        message for message in messages if message.role in (Role.USER, Role.ASSISTANT)
    ]
    before_tokens = estimate_messages_tokens(messages)
    result = microcompact_messages(messages, budget=before_tokens - 1, recent_turns=1)
    prose_after = [
        message for message in result.messages if message.role in (Role.USER, Role.ASSISTANT)
    ]
    if prose_after != prose_before:
        fail(f"microcompaction changed user or assistant prose: {result.messages!r}")
    if result.cleared_tool_results != 1 or not result.changed:
        fail(f"microcompaction did not stop after the oldest sufficient result: {result!r}")
    if result.messages[2].tool_result.content != CLEARED_CONTENT_MARKER:
        fail(f"oldest re-derivable result was not cleared: {result.messages[2]!r}")
    if result.messages[3:] != messages[3:]:
        fail(f"microcompaction cleared results after reaching the budget: {result.messages!r}")
    if result.after_tokens > result.budget or result.before_tokens != before_tokens:
        fail(f"microcompaction token accounting was wrong: {result!r}")
    if result.after_tokens != estimate_messages_tokens(result.messages):
        fail(
            "microcompaction's running total drifted from a direct estimate: "
            f"{result.after_tokens} vs {estimate_messages_tokens(result.messages)}"
        )


@check("compaction.microcompaction_spares_mutations")
def check_microcompaction_spares_mutations() -> None:
    messages = [
        Message(role=Role.USER, content="first goal"),
        Message(
            role=Role.ASSISTANT,
            tool_calls=[
                ToolCall(id="edit", name="edit_file"),
                ToolCall(id="write", name="write_file"),
                ToolCall(id="multi", name="multi_edit_file"),
            ],
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="edit", ok=True, content="e" * 600),
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="write", ok=True, content="w" * 600),
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="multi", ok=True, content="m" * 600),
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="orphan", ok=True, content="o" * 600),
        ),
        Message(role=Role.USER, content="latest request"),
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="recent-read", name="read_file")],
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(
                tool_call_id="recent-read",
                ok=True,
                content="recent" * 100,
            ),
        ),
    ]
    result = microcompact_messages(messages, budget=1, recent_turns=1)
    if result.changed or result.cleared_tool_results != 0:
        fail(f"microcompaction cleared a mutation, recent result, or orphan: {result!r}")
    if result.messages != messages:
        fail(f"spared tool results changed despite no eligible content: {result.messages!r}")


@check("compaction.microcompaction_preserves_handles")
def check_microcompaction_preserves_handles() -> None:
    payload = {"kind": "file_diff", "hunks": [{"old_start": 1, "new_start": 1}]}
    full_content = "stored line\n" * 500
    offloaded = offload_tool_result(
        ToolResult(
            tool_call_id="stored-read",
            ok=True,
            content=full_content,
            error="retained diagnostic",
            payload=payload,
        ),
        tool_name="read_file",
        store=ToolResultStore(),
    )
    if offloaded.offloaded is None:
        fail("offload fixture did not produce a handle")
    tool_message = Message(
        role=Role.TOOL,
        tool_result=offloaded,
        turn_id="stored-turn",
    )
    messages = [
        Message(role=Role.USER, content="first goal"),
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="stored-read", name="read_file")],
        ),
        tool_message,
        Message(role=Role.USER, content="latest request"),
    ]
    before_messages = list(messages)
    before_tokens = estimate_messages_tokens(messages)
    result = microcompact_messages(messages, budget=before_tokens - 1, recent_turns=1)
    handle = offloaded.offloaded
    expected_content = (
        "[old tool result content cleared; read_tool_result id "
        f'"{handle.id}" still holds {handle.characters} characters]'
    )
    cleared_message = result.messages[2]
    expected_result = replace(offloaded, content=expected_content)
    if cleared_message != replace(tool_message, tool_result=expected_result):
        fail(f"clearing changed offload metadata or another message field: {cleared_message!r}")
    if handle.id not in expected_result.content or str(handle.characters) not in expected_result.content:
        fail(f"cleared offload marker stranded its stored handle: {expected_result.content!r}")
    if expected_result.payload is not payload:
        fail("clearing copied or dropped the tool-owned payload")
    if messages != before_messages or messages[2] != tool_message:
        fail("microcompaction mutated its input list or message")


@check("compaction.microcompaction_is_idempotent")
def check_microcompaction_is_idempotent() -> None:
    under_budget = [
        Message(role=Role.USER, content="short request"),
        Message(role=Role.ASSISTANT, content="short answer"),
    ]
    unchanged = microcompact_messages(under_budget, budget=1_000)
    if unchanged.changed or unchanged.cleared_tool_results or unchanged.messages != under_budget:
        fail(f"within-budget conversation changed: {unchanged!r}")

    messages = _single_clearable_conversation()
    budget = 1
    first = microcompact_messages(messages, budget=budget, recent_turns=1)
    second = microcompact_messages(first.messages, budget=budget, recent_turns=1)
    if first.cleared_tool_results != 1:
        fail(f"idempotence fixture did not clear once: {first!r}")
    if second.changed or second.cleared_tool_results != 0 or second.messages != first.messages:
        fail(f"second microcompaction was not a no-op: {second!r}")

    # A conversation that was cleared earlier is the input a later compaction
    # sees, so its markers must never become a summary excerpt.
    cleared_earlier = [
        Message(role=Role.SYSTEM, content="system prompt"),
        Message(role=Role.USER, content="first goal"),
    ]
    for index in range(3):
        cleared_earlier.extend(
            (
                Message(
                    role=Role.ASSISTANT,
                    tool_calls=[ToolCall(id=f"c{index}", name="read_file")],
                ),
                Message(
                    role=Role.TOOL,
                    tool_result=ToolResult(
                        tool_call_id=f"c{index}",
                        ok=True,
                        content=f"body{index} " * 150,
                    ),
                ),
                Message(role=Role.USER, content=f"follow up {index} " * 40),
            )
        )
    cleared_earlier.append(Message(role=Role.USER, content="latest request"))
    cleared_only = compact_messages_for_budget(cleared_earlier, budget=560, recent_turns=2)
    if cleared_only.dropped_messages or cleared_only.cleared_tool_results != 3:
        fail(f"two-pass fixture did not clear without dropping: {cleared_only!r}")
    dropped_after = compact_messages_for_budget(
        cleared_only.messages, budget=200, recent_turns=1
    )
    summaries = [
        message.text
        for message in dropped_after.messages
        if message.role == Role.SYSTEM and "compacted" in message.text
    ]
    if len(summaries) != 1 or CLEARED_CONTENT_MARKER in summaries[0]:
        fail(f"an already-cleared result became a summary excerpt: {summaries!r}")
    if "tool result c0" not in summaries[0]:
        fail(f"a cleared excerpt did not fall through to its call id: {summaries[0]!r}")


@check("compaction.compaction_prefers_clearing")
def check_compaction_prefers_clearing() -> None:
    messages = _single_clearable_conversation()
    before_tokens = estimate_messages_tokens(messages)
    result = compact_messages_for_budget(messages, budget=before_tokens - 1, recent_turns=1)
    if (
        not result.changed
        or result.cleared_tool_results != 1
        or result.dropped_messages != 0
        or result.summary_messages != 0
        or len(result.messages) != len(messages)
    ):
        fail(f"full compaction did not prefer clearing alone: {result!r}")
    if result.before_tokens != before_tokens or result.after_tokens > result.budget:
        fail(f"clearing-only compaction token accounting was wrong: {result!r}")
    if "cleared 1 old tool result" not in describe_compaction(result):
        fail(f"compaction description omitted clearing: {describe_compaction(result)!r}")


@check("compaction.compaction_clears_then_drops")
def check_compaction_clears_then_drops() -> None:
    messages = [
        Message(role=Role.SYSTEM, content="system prompt"),
        Message(role=Role.USER, content="first goal"),
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="old-read", name="read_file")],
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="old-read", ok=True, content="r" * 1_200),
        ),
        Message(role=Role.ASSISTANT, content="old assistant prose " * 150),
        Message(role=Role.USER, content="old follow-up " * 100),
        Message(role=Role.ASSISTANT, content="more old prose " * 100),
        Message(role=Role.USER, content="latest request"),
    ]
    before_tokens = estimate_messages_tokens(messages)
    result = compact_messages_for_budget(messages, budget=140, recent_turns=1)
    if (
        not result.changed
        or result.cleared_tool_results != 1
        or result.dropped_messages < 1
        or result.summary_messages != 1
    ):
        fail(f"full compaction did not clear then summarize dropped messages: {result!r}")
    if result.before_tokens != before_tokens:
        fail(f"full compaction lost the original before estimate: {result!r}")
    if result.reclaimed_tokens != before_tokens - result.after_tokens:
        fail(f"reclaimed tokens did not include clearing plus dropping: {result!r}")
    if result.after_tokens > result.budget:
        fail(f"clear-then-drop compaction remained over budget: {result!r}")
    summaries = [
        message.text
        for message in result.messages
        if message.role == Role.SYSTEM and "compacted" in message.text
    ]
    if len(summaries) != 1:
        fail(f"expected exactly one summary message: {result.messages!r}")
    if CLEARED_CONTENT_MARKER in summaries[0]:
        fail(f"the summary spent an excerpt describing the clearing: {summaries[0]!r}")
    if "r" * 50 not in summaries[0]:
        fail(f"the summary lost the dropped tool result's real content: {summaries[0]!r}")
