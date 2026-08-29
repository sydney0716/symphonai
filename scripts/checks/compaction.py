"""Fixture-free checks for compaction."""

from __future__ import annotations

from orchestra_api.cancellation import CancellationToken, OperationCancelled
from orchestra_api.compaction import ContextCompactionError, compact_messages_for_budget
from orchestra_api.models import Message, Role
from scripts.checks.harness import check, fail


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
