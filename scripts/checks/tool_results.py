"""Checks for bounded in-process tool-result offload and retrieval."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

from orchestra_api.agent_loop import ApiAgent
from orchestra_api.compaction import estimate_message_tokens, estimate_messages_tokens
from orchestra_api.context_report import ContextSource, account_context
from orchestra_api.models import Message, ModelResponse, Role, ToolCall, ToolResult
from orchestra_api.providers.fake import FakeModelProvider
from orchestra_api.runner import standard_tool_registry
from orchestra_api.tool_results import (
    MAX_RESULT_SLICE_CHARS,
    MAX_STORE_CHARS,
    MAX_STORE_ENTRIES,
    MIN_PREVIEW_CHARS,
    OFFLOAD_THRESHOLD_CHARS,
    PREVIEW_CHARS,
    ToolResultStore,
    offload_tool_result,
)
from orchestra_api.tools.base import LocalTool
from orchestra_api.tools.metadata import ResultHint, ToolEffect, ToolMetadata, safe_metadata
from orchestra_api.tools.stored_result import ReadToolResultTool
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


class _LargeResultTool(LocalTool):
    def __init__(self, content: str) -> None:
        self._content = content

    @property
    def name(self) -> str:
        return "large_result"

    @property
    def description(self) -> str:
        return "Return a scripted large result."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(),
        )

    def _execute(self, tool_call, policy, cancel=None) -> ToolResult:
        return ToolResult(tool_call_id=tool_call.id, ok=True, content=self._content)


def _read_result(
    tool: ReadToolResultTool,
    policy,
    result_id: str,
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> ToolResult:
    arguments: dict[str, object] = {"id": result_id}
    if offset is not None:
        arguments["offset"] = offset
    if limit is not None:
        arguments["limit"] = limit
    return tool.execute(
        ToolCall(id="read-stored", name=tool.name, arguments=arguments),
        policy,
    )


@check("results.threshold_and_preview")
def check_threshold_and_preview() -> None:
    store = ToolResultStore()
    threshold_result = ToolResult(
        tool_call_id="threshold",
        ok=True,
        content="x" * OFFLOAD_THRESHOLD_CHARS,
    )
    unchanged = offload_tool_result(threshold_result, tool_name="read_file", store=store)
    if unchanged is not threshold_result or len(store) != 0:
        fail(f"threshold-sized result was offloaded: {unchanged!r}")

    newline_content = "line\n" * 1_001
    original = ToolResult(tool_call_id="newline", ok=True, content=newline_content)
    offloaded = offload_tool_result(original, tool_name="read_file", store=store)
    if offloaded.offloaded is None:
        fail("large newline result was not offloaded")
    expected_preview = newline_content[:PREVIEW_CHARS]
    expected_preview = expected_preview[: expected_preview.rfind("\n")]
    marker = offloaded.content[len(expected_preview) + 1 :]
    record = offloaded.offloaded
    required_marker_parts = (
        str(len(newline_content)),
        record.id,
        str(len(expected_preview)),
        "Call read_tool_result",
    )
    if not offloaded.content.startswith(f"{expected_preview}\n") or not all(
        part in marker for part in required_marker_parts
    ):
        fail(f"newline preview or marker was wrong: {offloaded.content!r}")
    if (
        record.characters != len(newline_content)
        or record.preview_characters != len(expected_preview)
        or record.tool_name != "read_file"
    ):
        fail(f"offload metadata was wrong: {record!r}")
    stored = store.get(record.id)
    if stored is None or stored.content != newline_content:
        fail("stored content was not retrievable byte-for-byte")

    hard_content = "z" * (OFFLOAD_THRESHOLD_CHARS + 1)
    hard = offload_tool_result(
        ToolResult(tool_call_id="hard", ok=True, content=hard_content),
        tool_name="grep",
        store=store,
    )
    if hard.offloaded is None or hard.offloaded.preview_characters != PREVIEW_CHARS:
        fail(f"no-newline preview did not use the hard slice: {hard!r}")
    if not hard.content.startswith(f"{'z' * PREVIEW_CHARS}\n[tool result offloaded:"):
        fail(f"hard-slice preview content was wrong: {hard.content!r}")

    early_newline_content = "\n" + "q" * 100_000
    early = offload_tool_result(
        ToolResult(tool_call_id="early", ok=True, content=early_newline_content),
        tool_name="read_file",
        store=store,
    )
    if early.offloaded is None or early.offloaded.preview_characters != PREVIEW_CHARS:
        fail(f"a leading newline cut the preview away: {early!r}")
    boundary_content = "b" * MIN_PREVIEW_CHARS + "\n" + "b" * 100_000
    boundary = offload_tool_result(
        ToolResult(tool_call_id="boundary", ok=True, content=boundary_content),
        tool_name="read_file",
        store=store,
    )
    if boundary.offloaded is None or boundary.offloaded.preview_characters != MIN_PREVIEW_CHARS:
        fail(f"a newline at the preview floor was not used as the cut: {boundary!r}")


@check("results.preserves_other_fields")
def check_preserves_other_fields() -> None:
    payload = {
        "kind": "file_diff",
        "path": "example.py",
        "hunks": [{"old_start": 1, "new_start": 1}],
    }
    original = ToolResult(
        tool_call_id="edit-call",
        ok=True,
        content="diff line\n" * 500,
        error=None,
        cancelled=False,
        payload=payload,
    )
    offloaded = offload_tool_result(
        original,
        tool_name="edit_file",
        store=ToolResultStore(),
    )
    if offloaded.offloaded is None or offloaded.content == original.content:
        fail(f"large diff was not replaced by a preview: {offloaded!r}")
    if (
        offloaded.tool_call_id != original.tool_call_id
        or offloaded.ok != original.ok
        or offloaded.error != original.error
        or offloaded.cancelled != original.cancelled
        or offloaded.payload is not payload
        or offloaded.schema_version != original.schema_version
    ):
        fail(f"offload changed a non-content field: {offloaded!r}")


@check("results.failures_not_offloaded")
def check_failures_not_offloaded() -> None:
    store = ToolResultStore()
    error = ToolResult(
        tool_call_id="error",
        ok=False,
        content="e" * (OFFLOAD_THRESHOLD_CHARS + 1),
        error="failed",
    )
    cancelled = ToolResult(
        tool_call_id="cancelled",
        ok=True,
        content="c" * (OFFLOAD_THRESHOLD_CHARS + 1),
        cancelled=True,
    )
    if offload_tool_result(error, tool_name="grep", store=store) is not error:
        fail("failed result was offloaded")
    if offload_tool_result(cancelled, tool_name="grep", store=store) is not cancelled:
        fail("cancelled result was offloaded")
    if len(store) != 0:
        fail(f"failed or cancelled content reached the store: {len(store)} entries")


@check("results.content_addressing")
def check_content_addressing() -> None:
    store = ToolResultStore()
    first = store.store(tool_name="read_file", tool_call_id="first", content="same")
    duplicate = store.store(tool_name="grep", tool_call_id="second", content="same")
    different = store.store(tool_name="read_file", tool_call_id="third", content="different")
    expected_id = "res_" + hashlib.sha256(b"same").hexdigest()[:12]
    if first.id != expected_id or duplicate.id != first.id or len(store) != 2:
        fail(f"identical content was not deduplicated by hash: {first!r}, {duplicate!r}")
    if duplicate.tool_call_id != "first" or duplicate.tool_name != "read_file":
        fail(f"duplicate content replaced first provenance: {duplicate!r}")
    if different.id == first.id:
        fail("different content received the same id")


@check("results.slice_and_limits")
def check_slice_and_limits() -> None:
    content = "0123456789" * (MAX_RESULT_SLICE_CHARS // 10 + 10)
    store = ToolResultStore()
    stored = store.store(tool_name="grep", tool_call_id="source", content=content)
    tool = ReadToolResultTool(store)
    with workspace() as ws:
        sliced = _read_result(tool, ws.policy, stored.id, offset=3, limit=5)
        expected_marker = (
            f"[{len(content) - 8} characters remain; call read_tool_result again "
            "with offset=8]"
        )
        if sliced.content != f"{content[3:8]}\n{expected_marker}":
            fail(f"requested stored-result slice was wrong: {sliced!r}")
        if sliced.payload != {
            "kind": "stored_result_slice",
            "id": stored.id,
            "offset": 3,
            "characters": 5,
            "total_characters": len(content),
            "more_follows": True,
        }:
            fail(f"slice payload was wrong: {sliced.payload!r}")

        clamped = _read_result(
            tool,
            ws.policy,
            stored.id,
            limit=MAX_RESULT_SLICE_CHARS + 999,
        )
        if clamped.payload["characters"] != MAX_RESULT_SLICE_CHARS:
            fail(f"slice limit was not clamped: {clamped.payload!r}")
        if not clamped.payload["more_follows"]:
            fail(f"clamped slice omitted its continuation: {clamped!r}")

        past_end = _read_result(tool, ws.policy, stored.id, offset=len(content) + 50)
        if past_end.content != "" or past_end.payload["more_follows"]:
            fail(f"past-end slice was not an empty success: {past_end!r}")


@check("results.missing_id")
def check_missing_id() -> None:
    store = ToolResultStore()
    tool = ReadToolResultTool(store)
    first = store.store(tool_name="grep", tool_call_id="oldest", content="oldest")
    for index in range(MAX_STORE_ENTRIES):
        store.store(tool_name="grep", tool_call_id=str(index), content=f"content-{index}")
    with workspace() as ws:
        never_stored = _read_result(tool, ws.policy, "res_missing")
        evicted = _read_result(tool, ws.policy, first.id)
    for result_id, result in (("res_missing", never_stored), (first.id, evicted)):
        if result.ok or result.error is None or result_id not in result.error:
            fail(f"missing id did not return a named failure: {result!r}")


@check("results.tool_metadata")
def check_tool_metadata() -> None:
    tool = ReadToolResultTool(ToolResultStore())
    metadata = safe_metadata(tool, {"id": "res_any"})
    if metadata != ToolMetadata(
        effect=ToolEffect.READ_ONLY,
        concurrency_safe=True,
        paths=(),
        result_hint=ResultHint.TEXT,
    ):
        fail(f"stored-result metadata was wrong: {metadata!r}")
    invalid_arguments = (
        {},
        {"id": 3},
        {"id": "res_any", "offset": True},
        {"id": "res_any", "offset": -1},
        {"id": "res_any", "limit": "5"},
        {"id": "res_any", "limit": -1},
    )
    if any(tool.validate(arguments) is None for arguments in invalid_arguments):
        fail("stored-result validation accepted a malformed argument shape")


@check("results.concurrent_stores")
def check_concurrent_stores() -> None:
    store = ToolResultStore()
    contents = [f"parallel-content-{index}" for index in range(32)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                store.store,
                tool_name="grep",
                tool_call_id=f"call-{index}",
                content=content,
            )
            for index, content in enumerate(contents)
        ]
        stored_results = [future.result() for future in futures]
    if len(store) != len(contents):
        fail(f"parallel stores lost entries: expected {len(contents)}, got {len(store)}")
    for expected, stored in zip(contents, stored_results, strict=True):
        resolved = store.get(stored.id)
        if resolved is None or resolved.content != expected:
            fail(f"parallel store result was not retrievable: {stored!r}")


@check("results.bounds")
def check_bounds() -> None:
    store = ToolResultStore()
    stored = [
        store.store(tool_name="grep", tool_call_id=str(index), content=f"bound-{index}")
        for index in range(MAX_STORE_ENTRIES + 1)
    ]
    if len(store) != MAX_STORE_ENTRIES:
        fail(f"store entry bound was not enforced: {len(store)}")
    if store.get(stored[0].id) is not None:
        fail("oldest stored result survived past the entry bound")
    if store.get(stored[-1].id) is None:
        fail("newest stored result was evicted instead of the oldest")

    oversized_store = ToolResultStore()
    oversized_store.store(tool_name="grep", tool_call_id="small", content="small entry")
    oversized = oversized_store.store(
        tool_name="grep",
        tool_call_id="oversized",
        content="h" * (MAX_STORE_CHARS + 1),
    )
    if oversized_store.get(oversized.id) is None:
        fail("an entry larger than the character bound was evicted by its own insertion")
    if len(oversized_store) != 1:
        fail(f"the character bound did not evict everything else: {len(oversized_store)}")


@check("results.agent_loop_offload")
def check_agent_loop_offload() -> None:
    large_content = "agent result\n" * 400
    tool = _LargeResultTool(large_content)
    tool_turn = ModelResponse(
        message=Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="large-call", name=tool.name)],
        )
    )
    final_turn = ModelResponse(message=Message(role=Role.ASSISTANT, content="done"))
    with workspace() as ws:
        store = ToolResultStore()
        with_store = ApiAgent(
            FakeModelProvider(responses=[tool_turn, final_turn]),
            {tool.name: tool},
            ws.policy,
            result_store=store,
        ).run([Message(role=Role.USER, content="run it")])
        without_store = ApiAgent(
            FakeModelProvider(responses=[tool_turn, final_turn]),
            {tool.name: tool},
            ws.policy,
        ).run([Message(role=Role.USER, content="run it")])

    preview_result = next(
        message.tool_result for message in with_store.messages if message.role == Role.TOOL
    )
    whole_result = next(
        message.tool_result for message in without_store.messages if message.role == Role.TOOL
    )
    if preview_result.offloaded is None or preview_result.content == large_content:
        fail(f"agent store did not offload the tool result: {preview_result!r}")
    stored = store.get(preview_result.offloaded.id)
    if stored is None or stored.content != large_content:
        fail("agent-offloaded content was not retained in its store")
    expected_whole = ToolResult(tool_call_id="large-call", ok=True, content=large_content)
    if whole_result != expected_whole:
        fail(f"agent without a store changed today's transcript shape: {whole_result!r}")


@check("results.registry_opt_in")
def check_registry_opt_in() -> None:
    existing_names = [
        "read_file",
        "write_file",
        "edit_file",
        "multi_edit_file",
        "list_files",
        "glob",
        "grep",
        "run_shell",
        "web_fetch",
    ]
    default = standard_tool_registry()
    store = ToolResultStore()
    opted_in = standard_tool_registry(result_store=store)
    selected = standard_tool_registry(["read_tool_result"], result_store=store)
    if list(default) != existing_names:
        fail(f"default registry changed from its nine tools: {list(default)!r}")
    if list(opted_in) != [*existing_names, "read_tool_result"]:
        fail(f"stored-result registry did not append the tenth tool: {list(opted_in)!r}")
    if list(selected) != ["read_tool_result"]:
        fail(f"stored-result tool was not selectable by name: {list(selected)!r}")


@check("results.context_report_counts_preview")
def check_context_report_counts_preview() -> None:
    original_content = "context result\n" * 400
    store = ToolResultStore()
    offloaded = offload_tool_result(
        ToolResult(tool_call_id="context-call", ok=True, content=original_content),
        tool_name="grep",
        store=store,
    )
    message = Message(role=Role.TOOL, tool_result=offloaded)
    report = account_context([message])
    if report.total_tokens != estimate_messages_tokens([message]):
        fail(f"context report total drifted from compaction: {report!r}")
    if len(report.entries) != 1 or report.entries[0].source != ContextSource.TOOL_RESULT:
        fail(f"offloaded bytes gained a context source or entry: {report!r}")
    if report.entries[0].tokens != estimate_message_tokens(message):
        fail(f"tool row did not count the preview message exactly: {report.entries[0]!r}")
    if report.entries[0].characters == len(original_content):
        fail(f"tool row counted stored bytes as inline characters: {report.entries[0]!r}")
    original_length = str(len(original_content))
    if any(
        original_length in value
        for entry in report.entries
        for value in (entry.label, entry.detail or "")
    ):
        fail(f"context entry mentioned the off-store length: {report.entries!r}")
