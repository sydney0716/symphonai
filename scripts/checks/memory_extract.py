"""Checks for background durable-memory extraction."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

from symphonai_api.agent_memory import AgentMemory, MemorySettings
from symphonai_api.agent_spec import AgentSpec, ModelSelector
from symphonai_api.cancellation import CancellationToken
from symphonai_api.events import CollectingSink, ToolCallFailed
from symphonai_api.memory_extract import (
    MemoryExtractor,
    _cursor_digest,
    _messages_after_cursor,
    evidence_rows,
    redact_secrets,
    serialize_tiered_input,
)
from symphonai_api.models import ImageBlock, Message, ModelRequest, ModelResponse, Role, ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.leader import Leader, LeaderConfig
from scripts.checks.harness import check, fail


class _Provider(FakeModelProvider):
    model = "fake-model"

    def __init__(self, responses=(), entered=None, release=None):
        super().__init__(list(responses) or [ModelResponse(Message(Role.ASSISTANT, "done"))])
        self.responses = list(responses) or [ModelResponse(Message(Role.ASSISTANT, "done"))]
        self.requests = []
        self.entered = entered
        self.release = release
        self.block_after = 1

    def create_response(self, request: ModelRequest, *, cancel=None) -> ModelResponse:
        self.requests.append(request)
        if self.entered is not None and len(self.requests) >= self.block_after:
            self.entered.set()
        if self.release is not None and len(self.requests) >= self.block_after:
            self.release.wait(2)
        return self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]


def _fixture(root: Path, provider: ModelProvider, events=None, usage=None):
    repo = root / "repo"
    repo.mkdir(exist_ok=True)
    policy = PermissionPolicy(repo_root=repo, mode="allow")
    store = AgentMemory(root / "memory", repo)
    spec = AgentSpec(
        name="leader", prompt="", model=ModelSelector("fake", "fake-model"),
        policy_ceiling=policy, memory=MemorySettings(enabled=True),
    )
    extractor = MemoryExtractor(
        store, spec, policy, agent_id="leader-id", events=events,
        usage_callback=usage,
    )
    return store, extractor, policy


def _context(provider, messages, *, schemas=(), memory_provider=None, memory_model=None):
    return (
        list(messages), provider, "fake-model", "medium", list(schemas),
        memory_provider, memory_model,
    )


@check("memory_extract.evidence_tiers_redaction_and_budget")
def evidence_tiers_redaction_and_budget() -> None:
    messages = [
        Message(Role.SYSTEM, "never include system prompts"),
        Message(Role.USER, "Use sk-" + "A" * 24 + " and password = hunter2hunter2."),
        Message(Role.ASSISTANT, "The durable preference is concise answers."),
        Message(Role.TOOL, tool_result=ToolResult("tool-1", True, "result " * 3000)),
        Message(Role.TOOL, tool_result=ToolResult("tool-2", True, "another " * 3000)),
        Message(Role.TOOL, tool_result=ToolResult("tool-3", True, "third " * 3000)),
    ]
    rows = evidence_rows(messages)
    rendered = serialize_tiered_input(messages, token_budget=30)
    if len(rows) != 5 or "system prompts" in rendered:
        fail(f"system evidence was retained: {rows!r} {rendered!r}")
    if "[REDACTED_SECRET]" not in "\n".join(row.text for row in rows):
        fail("API key shaped evidence was not redacted")
    user_row = next(row.text for row in rows if row.tier == 0)
    if "password = [REDACTED_SECRET]" not in user_row:
        fail(f"password shaped evidence was not redacted: {user_row!r}")
    if "messages omitted" not in rendered:
        fail("budget omission was not marked")
    if "human user:" not in rendered or rendered.count("messages omitted") != 1:
        fail(f"priority selection or omission grouping was wrong: {rendered!r}")
    if rows[-1].text.find("tool output truncated") < 0 or len(rows[-1].text.encode()) > 10_000:
        fail("tool result was not capped near 2,000 tokens and 10,000 bytes")
    if redact_secrets("Bearer " + "B" * 20).find("[REDACTED_SECRET]") < 0:
        fail("bearer token was not redacted")


@check("memory_extract.attachments_are_placeholders")
def attachments_are_placeholders() -> None:
    message = Message(Role.USER, ["question", ImageBlock("AA==", "image/png")])
    text = serialize_tiered_input([message])
    if "[image omitted]" not in text or "AA==" in text:
        fail(f"image payload leaked into evidence: {text!r}")


@check("memory_extract.fork_uses_original_history_and_schema")
def fork_uses_original_history_and_schema() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        extractor_entered, extractor_release = threading.Event(), threading.Event()
        provider = _Provider([
            ModelResponse(Message(Role.ASSISTANT, "I prefer tabs.")),
            ModelResponse(Message(Role.ASSISTANT, "No durable memory.")),
        ], entered=extractor_entered, release=extractor_release)
        provider.block_after = 2
        reported = threading.Event()
        leader = Leader(LeaderConfig(
            leader_provider=provider,
            subagent_provider=provider,
            repo_root=str(root),
            memory=AgentMemory(root / "memory", root),
            leader_effort="medium",
            memory_usage_callback=lambda _model, _usage: reported.set(),
        ))
        try:
            result = leader.chat("I prefer tabs.")
            if not extractor_entered.wait(1):
                fail("leader did not start its forked extraction")
            if len(provider.requests) != 2 or extractor_release.is_set():
                fail("chat waited for the slow background extraction")
            leader_request, extraction_request = provider.requests
            if extraction_request.messages[:-1] != result.leader_messages:
                fail(f"fork did not use the leader's exact pre-compaction messages: {extraction_request.messages[:-1]!r} != {result.leader_messages!r}")
            if extraction_request.tools != leader_request.tools:
                fail("fork did not use the leader's exact tool schemas")
            if (extraction_request.model, extraction_request.effort) != (
                leader_request.model, leader_request.effort
            ):
                fail("fork changed the leader model or effort")
            if extraction_request.messages[-1].role is not Role.USER or "Existing memories" not in extraction_request.messages[-1].text:
                fail("fork instruction was not appended as one user message")
            if "Minimum signal gate" in result.leader_messages[-1].text:
                fail("fork instruction leaked into the leader transcript")
            extractor_release.set()
            if not reported.wait(1):
                fail("background usage was not reported")
        finally:
            extractor_release.set()
            leader.close()


@check("memory_extract.separate_model_gets_only_evidence_and_memory_tools")
def separate_model_gets_only_evidence_and_memory_tools() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        leader = _Provider()
        target = _Provider()
        _, extractor, _ = _fixture(Path(temporary), target)
        messages = [Message(Role.SYSTEM, "secret system"), Message(Role.USER, "I prefer concise answers")]
        extractor._extract(
            _context(leader, messages, memory_provider=target, memory_model="separate"),
            None, CancellationToken(),
        )
        request = target.requests[0]
        names = {tool["name"] for tool in request.tools}
        if names != {"remember", "forget"}:
            fail(f"separate extractor had tools {names!r}")
        if len(request.messages) != 2 or "secret system" in request.messages[1].text:
            fail("separate extractor received system or raw transcript messages")
        if "concise answers" not in request.messages[1].text or request.effort is not None:
            fail(f"separate request omitted evidence or assigned unsupported effort: {request!r}")
        extractor._extract(
            _context(leader, messages, memory_provider=target, memory_model="fake-model"),
            None, CancellationToken(),
        )
        if leader.requests[-1].messages[:-1] != messages:
            fail("a configured model matching the leader did not switch back to a fork")
        extractor.close()


@check("memory_extract.only_memory_tools_execute_and_save")
def only_memory_tools_execute_and_save() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        responses = [
            ModelResponse(Message(Role.ASSISTANT, tool_calls=[
                ToolCall("save", "remember", {"text": "Prefer concise answers."}),
                ToolCall("refused", "read_file", {"path": "secret"}),
            ])),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ]
        provider = _Provider(responses)
        events = CollectingSink()
        store, extractor, policy = _fixture(Path(temporary), provider, events)
        messages = [Message(Role.USER, "I prefer concise answers")]
        if not extractor._extract(_context(provider, messages), None, CancellationToken()):
            fail("tool extraction did not finish")
        entries = store.read()
        if [entry.text for entry in entries] != ["Prefer concise answers."] or entries[0].agent_name != "leader":
            fail("remember tool call did not persist memory")
        failures = [event for event in events.events if isinstance(event, ToolCallFailed)]
        if len(failures) != 1 or failures[0].tool_name != "read_file":
            fail(f"non-memory tool call was not refused: {events.events!r}")
        extractor.close()


@check("memory_extract.cursor_skips_memory_turn_and_recovers_missing_anchor")
def cursor_skips_memory_turn_and_recovers_missing_anchor() -> None:
    anchor = Message(Role.USER, "prior")
    later = Message(Role.USER, "new")
    if _messages_after_cursor([anchor, later], (1, _cursor_digest(anchor))) != [later]:
        fail("cursor did not select only messages after its anchor")
    if _messages_after_cursor([later], (1, "missing-anchor")) != [later]:
        fail("missing anchor did not fall back to all available messages")
    with tempfile.TemporaryDirectory() as temporary:
        provider = _Provider()
        _, extractor, _ = _fixture(Path(temporary), provider)
        memory_call = Message(Role.ASSISTANT, tool_calls=[ToolCall("m", "remember", {"text": "x"})])
        if not extractor._extract(_context(provider, [memory_call]), None, CancellationToken()):
            fail("memory turn skip failed")
        if provider.requests:
            fail("memory tool turn triggered a redundant extraction")
        extractor.close()


@check("memory_extract.cursor_uses_position_for_identical_replies")
def cursor_uses_position_for_identical_replies() -> None:
    repeated = Message(Role.ASSISTANT, "same reply")
    middle = Message(Role.USER, "intervening turn")
    messages = [repeated, middle, repeated]
    selected = _messages_after_cursor(messages, (2, _cursor_digest(repeated)))
    if selected != messages:
        fail(f"duplicate text at another position moved the cursor: {selected!r}")


@check("memory_extract.unknown_model_has_no_effort")
def unknown_model_has_no_effort() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        leader = _Provider()
        extractor_provider = _Provider()
        _, extractor, _ = _fixture(Path(temporary), leader)
        extractor._extract(
            _context(
                leader, [Message(Role.USER, "remember this")],
                memory_provider=extractor_provider, memory_model="unlisted-model",
            ),
            None,
            CancellationToken(),
        )
        if len(extractor_provider.requests) != 1 or extractor_provider.requests[0].effort is not None:
            fail(f"unknown model received an unsupported effort: {extractor_provider.requests!r}")
        extractor.close()


@check("memory_extract.failed_run_keeps_cursor")
def failed_run_keeps_cursor() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        provider = _Provider()
        _, extractor, _ = _fixture(Path(temporary), provider)
        anchor = Message(Role.USER, "anchor")
        extractor.set_cursor([anchor])
        original_cursor = extractor._cursor
        provider.create_response = lambda request, cancel=None: (_ for _ in ()).throw(RuntimeError("offline"))
        context = _context(provider, [anchor, Message(Role.USER, "new")])
        extractor.submit(context[0], leader_provider=provider, leader_model="fake-model", leader_effort=None, leader_tool_schemas=[])
        deadline = time.monotonic() + 1
        while extractor._thread is not None and extractor._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.005)
        if extractor._cursor != original_cursor:
            fail("failed provider call advanced the cursor")
        extractor.close()


@check("memory_extract.coalesces_overlapping_turns")
def coalesces_overlapping_turns() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        entered, release = threading.Event(), threading.Event()
        provider = _Provider(entered=entered, release=release)
        _, extractor, _ = _fixture(Path(temporary), provider)
        first = [Message(Role.USER, "first")]
        second = [*first, Message(Role.ASSISTANT, "reply"), Message(Role.USER, "second")]
        third = [*second, Message(Role.ASSISTANT, "reply 2"), Message(Role.USER, "third")]
        fourth = [*third, Message(Role.ASSISTANT, "reply 3"), Message(Role.USER, "fourth")]
        extractor.submit(first, leader_provider=provider, leader_model="fake-model", leader_effort=None, leader_tool_schemas=[])
        if not entered.wait(1):
            fail("first background extraction did not start")
        extractor.submit(second, leader_provider=provider, leader_model="fake-model", leader_effort=None, leader_tool_schemas=[])
        extractor.submit(third, leader_provider=provider, leader_model="fake-model", leader_effort=None, leader_tool_schemas=[])
        extractor.submit(fourth, leader_provider=provider, leader_model="fake-model", leader_effort=None, leader_tool_schemas=[])
        release.set()
        deadline = time.monotonic() + 2
        while (extractor._thread is not None and extractor._thread.is_alive()) and time.monotonic() < deadline:
            time.sleep(0.005)
        if len(provider.requests) != 2 or not any(
            "fourth" in message.text for message in provider.requests[1].messages
        ):
            fail("overlapping turns were dropped instead of coalesced")
        extractor.close()


@check("memory_extract.usage_is_counted_for_background_calls")
def usage_is_counted_for_background_calls() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        usage = []
        provider = _Provider()
        _, extractor, _ = _fixture(Path(temporary), provider, usage=lambda model, totals: usage.append((model, totals)))
        extractor._extract(
            _context(provider, [Message(Role.USER, "question")]), None, CancellationToken()
        )
        if len(usage) != 1 or usage[0][1].calls != 1:
            fail(f"background call usage was not reported: {usage!r}")
        extractor.close()

