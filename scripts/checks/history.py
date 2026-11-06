"""Core persistence behaviour through sessions, agents and provider requests."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace

from symphonai_api.agent_loop import ApiAgent
from symphonai_api.compaction import compact_messages_for_budget
from symphonai_api.identity import new_agent_ref
from symphonai_api.models import (
    DocumentBlock, ImageBlock, Message, ModelRequest, ModelResponse, Role,
    ToolCall, ToolResult,
)
from symphonai_api.providers.anthropic_provider import AnthropicProvider
from symphonai_api.providers.gemini_provider import GeminiProvider
from symphonai_api.providers.openai_provider import OpenAIProvider
from symphonai_api.runner import resume_task, run_task
from symphonai_api.serialization import message_to_json
from symphonai_api.session import (
    SessionStore, fork_run, load_run, load_run_for_resume, read_records, tool_result_search_path,
)
from symphonai_api.tool_results import ToolResultStore
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolMetadata
from symphonai_api.tools.stored_result import ReadToolResultTool
from symphonai_host.broker import EventBroker
from symphonai_host.run import HostRun
from scripts.checks.harness import check
from scripts.checks.workspace import Script, answer, exchange, handoff_spec, workspace


def run_agent(ws, store, provider, messages, **options):
    agent = new_agent_ref("history")
    return ApiAgent(provider, ws.tools, ws.policy, agent_ref=agent,
                    transcript=store.writer_for(agent.agent_id, is_root=True),
                    **options).run(messages)


def assert_complete(messages):
    pending = set()
    for message in messages:
        assert message.content or message.tool_calls or message.tool_result, "empty message"
        for call in message.tool_calls:
            assert call.id not in pending, "repeated unanswered call"
            pending.add(call.id)
        if message.tool_result is not None:
            assert message.tool_result.tool_call_id in pending, "orphaned tool result"
            pending.remove(message.tool_result.tool_call_id)
    assert not pending, f"unanswered calls: {pending}"


def provider_body(provider, messages, response):
    with exchange(response) as (bodies, _, _):
        provider.create_response(ModelRequest(messages=messages))
    assert len(bodies) == 1
    return bodies[0]


@check("history.finished_run_reloads_whole")
def finished_run_reloads_whole():
    with workspace() as ws, closing(SessionStore(ws.root / "sessions", "whole")) as store:
        messages = [Message(Role.SYSTEM, "Keep fixture instructions"), Message(Role.USER, [
            "Inspect the attached examples", ImageBlock("aGk=", "image/png"),
            DocumentBlock("cGRm", filename="example.pdf"),
        ])]
        call = ToolCall("read-existing", "read_file", {"path": "existing.txt"})
        result = run_agent(ws, store, Script([answer("", [call]), answer()]), messages)
        loaded = load_run(store)
        assert loaded.messages == result.messages
        assert any(message.tool_calls == [call] for message in loaded.messages)
        assert any(message.tool_result is not None and message.tool_result.ok for message in loaded.messages)
        assert loaded.messages[1].content == messages[1].content
        assert_complete(loaded.messages)


def interrupted(store):
    writer = store.writer_for("crashed-agent", is_root=True)
    writer.append("run_started", run_id="crashed-run", agent_id="crashed-agent", turn_id=None,
                  data={"agent_name": "history", "parent_run_id": None})
    writer.append("turn_started", run_id="crashed-run", agent_id="crashed-agent", turn_id="turn-1", data={})
    messages = [Message(Role.USER, "Inspect two paths"), Message(Role.ASSISTANT, "", tool_calls=[
        ToolCall("first-call", "read_file", {"path": "existing.txt"}),
        ToolCall("second-call", "read_file", {"path": "missing.txt"}),
    ])]
    for message in messages:
        writer.append("message", run_id="crashed-run", agent_id="crashed-agent", turn_id="turn-1",
                      data=message_to_json(message))
    loaded, diagnosis, repaired = load_run_for_resume(store)
    assert set(repaired) == {"first-call", "second-call"}
    assert set(diagnosis.unanswered_tool_call_ids) == set(repaired)
    assert_complete(loaded.messages)
    return loaded.messages


@check("history.interrupted_tools_resend_to_anthropic")
def interrupted_tools_resend_to_anthropic():
    with workspace() as ws, closing(SessionStore(ws.root / "sessions", "interrupted")) as store:
        body = provider_body(AnthropicProvider(), interrupted(store), {
            "content": [{"type": "text", "text": "continued"}], "stop_reason": "end_turn",
        })
        calls, results = [], []
        for message in body["messages"]:
            assert message["content"]
            for block in message["content"] if isinstance(message["content"], list) else []:
                if block["type"] == "tool_use":
                    calls.append(block["id"])
                if block["type"] == "tool_result":
                    results.append(block["tool_use_id"])
        assert len(calls) == 2 and set(calls) == set(results)


@check("history.interrupted_tools_resend_to_openai")
def interrupted_tools_resend_to_openai():
    with workspace() as ws, closing(SessionStore(ws.root / "sessions", "interrupted")) as store:
        body = provider_body(OpenAIProvider(), interrupted(store), {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "continued"}]}],
        })
        calls = [item["call_id"] for item in body["input"] if item.get("type") == "function_call"]
        results = [item["call_id"] for item in body["input"] if item.get("type") == "function_call_output"]
        assert len(calls) == 2 and set(calls) == set(results)
        assert all(item.get("content") for item in body["input"] if "role" in item)


@check("history.interrupted_tools_resend_to_gemini")
def interrupted_tools_resend_to_gemini():
    with workspace() as ws, closing(SessionStore(ws.root / "sessions", "interrupted")) as store:
        body = provider_body(GeminiProvider(), interrupted(store), {
            "candidates": [{"content": {"parts": [{"text": "continued"}]}, "finishReason": "STOP"}],
        })
        calls, results = [], []
        for message in body["contents"]:
            assert message["parts"]
            for part in message["parts"]:
                if "functionCall" in part:
                    calls.append(part["functionCall"]["name"])
                if "functionResponse" in part:
                    results.append(part["functionResponse"]["name"])
        assert len(calls) == 2 and calls == results


def long_conversation():
    return [
        Message(Role.SYSTEM, "Retain system instructions"), Message(Role.USER, "The original goal"),
        Message(Role.ASSISTANT, "old reasoning " * 1000), Message(Role.USER, "old followup " * 1000),
        Message(Role.ASSISTANT, "", tool_calls=[ToolCall("read", "read_file", {"path": "existing.txt"})]),
        Message(Role.TOOL, tool_result=ToolResult("read", ok=True, content="old file " * 1000)),
        Message(Role.ASSISTANT, "old conclusion " * 1000), Message(Role.USER, "The recent request"),
        Message(Role.ASSISTANT, "", tool_calls=[ToolCall("recent", "read_file", {"path": "existing.txt"})]),
        Message(Role.TOOL, tool_result=ToolResult("recent", ok=True, content="recent result")),
    ]


@check("history.compaction_keeps_tool_pairs")
def compaction_keeps_tool_pairs():
    compacted = compact_messages_for_budget(long_conversation(), budget=600, recent_turns=1)
    assert compacted.changed and compacted.dropped_messages > 0
    assert_complete(compacted.messages)
    assert any(call.id == "recent" for message in compacted.messages for call in message.tool_calls)


@check("history.compaction_keeps_required_context")
def compaction_keeps_required_context():
    messages = long_conversation()
    compacted = compact_messages_for_budget(messages, budget=600, recent_turns=1)
    assert compacted.changed and compacted.dropped_messages > 0
    assert messages[0] in compacted.messages and messages[1] in compacted.messages
    assert compacted.messages[-3:] == messages[-3:]


@check("history.compacted_run_reloads_live_conversation")
def compacted_run_reloads_live_conversation():
    with workspace() as ws, closing(SessionStore(ws.root / "sessions", "compacted")) as store:
        result = run_agent(ws, store, Script([answer()]), long_conversation(),
                           context_budget=lambda model: 600, context_recent_turns=1)
        records, _ = read_records(store.directory / "run.jsonl")
        assert any(record["type"] == "compaction" for record in records)
        assert load_run(store).messages == result.messages
        assert_complete(result.messages)


@check("history.fork_is_an_unchanged_prefix")
def fork_is_an_unchanged_prefix():
    with workspace() as ws, closing(SessionStore(ws.root / "sessions", "source")) as source, \
            closing(SessionStore(ws.root / "sessions", "fork")) as fork:
        run_task(Script([answer("first")]), ws.policy, "first goal", session=source)
        loaded = load_run(source)
        original = (source.directory / "run.jsonl").read_bytes()
        forked = fork_run(source, through_record_id=loaded.record_ids[0], new_store=fork)
        assert forked.messages == loaded.messages[:1]
        assert load_run(fork).messages == loaded.messages[:1]
        assert (source.directory / "run.jsonl").read_bytes() == original


@check("history.metadata_updates_survive_other_store_threads")
def metadata_updates_survive_other_store_threads():
    with workspace() as ws, closing(SessionStore(ws.root / "sessions", "shared")) as first, \
            closing(SessionStore.open(ws.root / "sessions", "shared")) as second:
        ready = threading.Barrier(2)

        def update(store, key):
            ready.wait(timeout=5)

            def change(meta):
                # Yield inside the read/modify/write boundary to exercise overlap.
                time.sleep(0.01)
                meta[key] = key
            store.update_meta(change)

        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(update, first, "one"), workers.submit(update, second, "two")]
            for future in futures:
                future.result(timeout=5)
        meta = first.read_meta()
        assert meta["one"] == "one" and meta["two"] == "two"


@check("history.refused_turn_leaves_a_sendable_conversation")
def refused_turn_leaves_a_sendable_conversation():
    with workspace() as ws, closing(SessionStore(ws.root / "sessions", "refused")) as store, \
            closing(SessionStore(ws.root / "sessions", "continued")) as continued:
        refused = ModelResponse(Message(Role.ASSISTANT, ""), stop_reason="refusal", refusal_category="policy")
        result = run_task(Script([refused]), ws.policy, "first goal", session=store)
        assert all(message.role != Role.ASSISTANT or message.content or message.tool_calls for message in result.messages)
        provider = Script([answer()])
        resume_task(provider, ws.policy, "try a different approach", store=store, new_store=continued)
        assert_complete(provider.requests[0].messages)
        assert provider.requests[0].messages[-1].text == "try a different approach"


@check("history.truncated_last_record_is_dropped")
def truncated_last_record_is_dropped():
    with workspace() as ws, closing(SessionStore(ws.root / "sessions", "truncated")) as store:
        result = run_task(Script([answer()]), ws.policy, "fixture", session=store)
        partial = b'{"schema_version":1,"type":"message","data":'
        with (store.directory / "run.jsonl").open("ab") as handle:
            handle.write(partial)
        loaded = load_run(store)
        assert loaded.messages == result.messages and loaded.dropped_bytes == len(partial)


class LargeResult(LocalTool):
    name = "large_result"
    description = "Return a large fixture result."
    parameters = {"type": "object", "properties": {}}

    def metadata(self, arguments):
        return ToolMetadata(concurrency_safe=True)

    def _execute(self, tool_call, policy, cancel=None):
        return ToolResult(tool_call.id, ok=True, content="0123456789" * 1000)


@check("history.offloaded_result_survives_resume_and_fork")
def offloaded_result_survives_resume_and_fork():
    with workspace() as ws, closing(SessionStore(ws.root / "sessions", "offloaded")) as source, \
            closing(SessionStore(ws.root / "sessions", "fork")) as fork, \
            closing(SessionStore(ws.root / "sessions", "resumed")) as resumed:
        result = run_task(Script([answer("", [ToolCall("large", "large_result", {})]), answer()]),
                          ws.policy, "fixture", session=source, offload_tool_results=True,
                          mcp_tools={"large_result": LargeResult()})
        offloaded = next(message.tool_result.offloaded for message in result.messages
                         if message.tool_result is not None and message.tool_result.offloaded is not None)
        assert offloaded.characters == 10000
        loaded = load_run(source)
        fork_run(source, through_record_id=loaded.record_ids[-1], new_store=fork)
        source.close()  # No live in-memory result store is available to either path.
        with closing(SessionStore.open(ws.root / "sessions", "offloaded")) as reopened:
            provider = Script([answer("", [ToolCall("read-large", "read_tool_result", {"id": offloaded.id, "offset": 9000, "limit": 1000})]), answer()])
            result = resume_task(provider, ws.policy, "Read the stored result", store=reopened,
                                 new_store=resumed, offload_tool_results=True)
        read = next(message.tool_result for message in result.messages
                    if message.tool_result is not None and message.tool_result.tool_call_id == "read-large")
        assert read.ok and read.content == "0123456789" * 100
        fork_results = ToolResultStore(directory=fork.tool_results_directory,
                                       fallback_directories=tool_result_search_path(fork))
        read = ReadToolResultTool(fork_results).execute(ToolCall("fork-read", "read_tool_result", {"id": offloaded.id}), ws.policy)
        assert read.ok and read.content == "0123456789" * 1000


@check("history.reverting_a_turn_restores_only_its_files")
def reverting_a_turn_restores_only_its_files():
    with workspace() as ws:
        untouched = ws.root / "unrelated.txt"
        untouched.write_text("unrelated")
        provider = Script([
            answer("", [ToolCall("dispatch", "dispatch_subagent", {
                "subagent_name": "worker", "task": handoff_spec("Update existing.txt and create created.txt"),
            })]),
            answer("", [ToolCall("read-before-write", "read_file", {"path": "existing.txt"})]),
            answer("", [ToolCall("change", "write_file", {"path": "existing.txt", "content": "changed"}),
                         ToolCall("create", "write_file", {"path": "created.txt", "content": "created"})]),
            answer(), answer(),
        ])
        host = HostRun(provider, replace(ws.policy, shell_enabled=True, shell_allowlist=[()]), EventBroker(), sessions_root=ws.root / "sessions",
                       memory_root=ws.root / "memory", memory_extraction_enabled=False)
        try:
            host.select_mode("allow")
            host.start("Update the fixture files")
            deadline = time.monotonic() + 5
            while host.active_run_id is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert host.active_run_id is None, "fixture did not finish"
            assert (ws.root / "existing.txt").read_text() == "changed"
            assert (ws.root / "created.txt").read_text() == "created"
            turns = host.changes()["turns"]
            assert turns
            host.revert_changes(key=turns[0]["key"])
            assert (ws.root / "existing.txt").read_text() == "hello from disk"
            assert not (ws.root / "created.txt").exists()
            assert untouched.read_text() == "unrelated"
        finally:
            host.close()


@check("history.memory_consolidation_preserves_unnamed_entries_and_rejects_invalid_results")
def memory_consolidation_preserves_unnamed_entries_and_rejects_invalid_results():
    import json
    from symphonai_api.agent_memory import AgentMemory
    from symphonai_api.memory_consolidate import apply_result

    for scope in ("project", "global"):
        with workspace() as ws:
            store = AgentMemory(ws.outside / "memory", ws.root)
            original = tuple(store.write(text, agent_name="fixture", run_id="fixture", scope=scope)
                             for text in ("Use pytest", "Run pytest", "Use the retired command",
                                          "Keep this exact fact", "Keep this other fact"))
            merge = {"ids": [original[0].id, original[1].id], "text": "Run pytest"}
            for invalid in (
                {"merge": [merge], "remove": [{"id": "missing", "reason": "outdated"}]},
                {"merge": [merge], "remove": [], "entries": []},
                {"merge": [], "remove": [{"reason": "remove everything"}]},
                {"merge": [merge], "remove": [{"id": original[0].id, "reason": "duplicate"}]},
                {"merge": [{"ids": merge["ids"], "text": "x" * 501}], "remove": []},
            ):
                try:
                    apply_result(store, scope, original, json.dumps(invalid))
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid consolidation was accepted")
                assert store.read_status(scope) == (original, False)
                assert store.last_consolidation(scope) is None and not store.has_previous(scope)
            added = store.write("Added while the model was working", agent_name="fixture", run_id="later", scope=scope)
            apply_result(store, scope, original, json.dumps({
                "merge": [merge], "remove": [{"id": original[2].id, "reason": "retired command"}],
            }))
            remaining, unreadable = store.read_status(scope)
            assert not unreadable
            assert tuple(entry for entry in remaining if entry.id in {original[3].id, original[4].id, added.id}) == (*original[3:], added)
            assert len(remaining) == 4 and remaining[-1].text == "Run pytest"
            store.restore_previous(scope)
            assert store.read_status(scope) == ((*original, added), False)
