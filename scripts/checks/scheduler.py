"""Checks for tool-call partitioning and parallel execution."""

from __future__ import annotations

import threading
import time
from collections import Counter
from types import SimpleNamespace

from orchestra_api.agent_loop import ApiAgent
from orchestra_api.cancellation import CancellationToken, OperationCancelled
from orchestra_api.events import (
    CollectingSink,
    ToolCallFinished,
    ToolCallStarted,
)
from orchestra_api.models import Message, ModelResponse, Role, ToolCall, ToolResult
from orchestra_api.permissions import PermissionPolicy
from orchestra_api.providers.fake import FakeModelProvider
from orchestra_api.scheduler import partition_tool_calls
from orchestra_api.tools.base import LocalTool
from orchestra_api.tools.metadata import ToolEffect, ToolMetadata
from orchestra_api.tools.read_ledger import ReadLedger
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


LEGACY_SINGLETON_EVENT_SEQUENCE = [
    ("RunStarted", None, None),
    ("TurnStarted", None, None),
    ("ToolCallStarted", "single", None),
    ("ToolCallFinished", "single", True),
    ("TurnFinished", None, None),
    ("RunFinished", None, None),
]


def _batch_ids(batches: list[list[ToolCall]]) -> list[list[str]]:
    return [[tool_call.id for tool_call in batch] for batch in batches]


def _tool_turn(*tool_calls: ToolCall) -> ModelResponse:
    return ModelResponse(Message(Role.ASSISTANT, tool_calls=list(tool_calls)))


def _tool_results(messages: list[Message]) -> list[ToolResult]:
    return [
        message.tool_result
        for message in messages
        if message.tool_result is not None
    ]


class _SafeTool(LocalTool):
    def __init__(self, name: str = "safe") -> None:
        self._name = name
        self.completion_order: list[str] = []
        self._completion_lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Return the call's test value after an optional delay."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(),
        )

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        delay = tool_call.arguments.get("delay", 0)
        if delay:
            time.sleep(delay)
        error = tool_call.arguments.get("error")
        if error:
            raise RuntimeError(error)
        with self._completion_lock:
            self.completion_order.append(tool_call.id)
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=True,
            content=tool_call.arguments.get("value", tool_call.id),
        )


class _MetadataFailureTool(_SafeTool):
    def __init__(self, *, cancelled: bool = False) -> None:
        super().__init__("metadata_failure")
        self._cancelled = cancelled

    def metadata(self, arguments: dict) -> ToolMetadata:
        if self._cancelled:
            raise OperationCancelled
        raise ValueError("metadata could not parse arguments")


class _ThreadIdentityTool(_SafeTool):
    def __init__(self) -> None:
        super().__init__("singleton_thread")

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=True,
            content=threading.current_thread().name,
        )


class _CancellingBatchTool(_SafeTool):
    def __init__(self) -> None:
        super().__init__("cancel_batch")
        self._first_finished = threading.Event()
        self._third_started = threading.Event()
        self._cancelled = threading.Event()

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        assert cancel is not None
        sequence = tool_call.arguments["sequence"]
        if sequence == 1:
            self._first_finished.set()
            return ToolResult(tool_call_id=tool_call.id, ok=True, content="first-real")
        if sequence == 2:
            if not self._first_finished.wait(timeout=1):
                raise RuntimeError("first call did not finish")
            if not self._third_started.wait(timeout=1):
                raise RuntimeError("third call did not start in parallel")
            cancel.cancel()
            self._cancelled.set()
            raise OperationCancelled
        self._third_started.set()
        if not self._cancelled.wait(timeout=1):
            raise RuntimeError("second call did not cancel")
        raise OperationCancelled


class _CoordinatedLedger:
    def __init__(self) -> None:
        self._ledger = ReadLedger()
        self._record_barrier = threading.Barrier(2)

    def record(self, resolved, *, full: bool, content: str | None) -> None:
        self._record_barrier.wait(timeout=1)
        self._ledger.record(resolved, full=full, content=content)

    def check(self, resolved) -> str | None:
        return self._ledger.check(resolved)


class _ProbeLock:
    def __init__(self) -> None:
        self.held = False
        self.entries = 0

    def __enter__(self) -> None:
        if self.held:
            raise RuntimeError("probe lock was entered recursively")
        self.held = True
        self.entries += 1

    def __exit__(self, *exc: object) -> None:
        self.held = False


class _ProbePath:
    def __init__(self, lock: _ProbeLock) -> None:
        self.lock = lock
        self.mtime_ns = 1
        self.content = "same"

    def stat(self) -> SimpleNamespace:
        if not self.lock.held:
            raise RuntimeError("ledger called stat() without holding its lock")
        return SimpleNamespace(st_mtime_ns=self.mtime_ns)

    def read_text(self, *, encoding: str) -> str:
        if not self.lock.held:
            raise RuntimeError("ledger called read_text() without holding its lock")
        return self.content


@check("scheduler.partition_barriers")
def check_partition_barriers() -> None:
    with workspace() as ws:
        read_one = ToolCall(id="read-1", name="read_file", arguments={"path": "a"})
        read_two = ToolCall(id="read-2", name="read_file", arguments={"path": "b"})
        write = ToolCall(
            id="write",
            name="write_file",
            arguments={"path": "a", "content": "changed"},
        )
        glob = ToolCall(id="glob", name="glob", arguments={"pattern": "*.py"})
        grep = ToolCall(id="grep", name="grep", arguments={"pattern": "needle"})
        mixed = partition_tool_calls(
            [read_one, read_two, write, glob, grep], ws.tools
        )
        if _batch_ids(mixed) != [["read-1", "read-2"], ["write"], ["glob", "grep"]]:
            fail(f"safe runs and mutation barriers partitioned incorrectly: {mixed!r}")

        split = partition_tool_calls([glob, write, grep], ws.tools)
        if _batch_ids(split) != [["glob"], ["write"], ["grep"]]:
            fail(f"one unsafe call did not split safe neighbours: {split!r}")

        unknown = ToolCall(id="unknown", name="not_registered")
        unknown_split = partition_tool_calls([glob, unknown, grep], ws.tools)
        if _batch_ids(unknown_split) != [["glob"], ["unknown"], ["grep"]]:
            fail(f"an unknown tool did not become its own barrier: {unknown_split!r}")


@check("scheduler.metadata_fail_closed")
def check_metadata_fail_closed() -> None:
    safe_tool = _SafeTool()
    metadata_failure = _MetadataFailureTool()
    calls = [
        ToolCall(id="safe-before", name="safe"),
        ToolCall(id="bad-metadata", name="metadata_failure"),
        ToolCall(id="safe-after", name="safe"),
    ]
    batches = partition_tool_calls(
        calls,
        {safe_tool.name: safe_tool, metadata_failure.name: metadata_failure},
    )
    if _batch_ids(batches) != [["safe-before"], ["bad-metadata"], ["safe-after"]]:
        fail(f"raising metadata did not fail closed as a barrier: {batches!r}")

    cancelled_metadata = _MetadataFailureTool(cancelled=True)
    try:
        partition_tool_calls(
            [ToolCall(id="cancel", name=cancelled_metadata.name)],
            {cancelled_metadata.name: cancelled_metadata},
        )
    except OperationCancelled:
        pass
    else:
        fail("metadata cancellation did not propagate out of the scheduler")


@check("scheduler.classification")
def check_scheduler_classification() -> None:
    with workspace() as ws:
        glob = ToolCall(id="glob", name="glob", arguments={"pattern": "*.py"})
        grep = ToolCall(id="grep", name="grep", arguments={"pattern": "needle"})
        ls = ToolCall(id="ls", name="run_shell", arguments={"argv": ["ls"]})
        git_status = ToolCall(
            id="git-status",
            name="run_shell",
            arguments={"argv": ["git", "status"]},
        )
        safe_shell = partition_tool_calls([glob, ls], ws.tools)
        if _batch_ids(safe_shell) != [["glob", "ls"]]:
            fail(f"parallel-safe shell classification was lost: {safe_shell!r}")
        serial_git = partition_tool_calls([glob, git_status, grep], ws.tools)
        if _batch_ids(serial_git) != [["glob"], ["git-status"], ["grep"]]:
            fail(f"serial Git classification was lost: {serial_git!r}")


@check("scheduler.parallel_reads")
def check_parallel_reads() -> None:
    with workspace() as ws:
        (ws.root / "second.txt").write_text("second from disk")
        coordinated_ledger = _CoordinatedLedger()
        ws.tools["read_file"]._ledger = coordinated_ledger
        ws.tools["write_file"]._ledger = coordinated_ledger
        provider = FakeModelProvider(
            [
                _tool_turn(
                    ToolCall(
                        id="read-existing",
                        name="read_file",
                        arguments={"path": "existing.txt"},
                    ),
                    ToolCall(
                        id="read-second",
                        name="read_file",
                        arguments={"path": "second.txt"},
                    ),
                ),
                ModelResponse(Message(Role.ASSISTANT, "done")),
            ]
        )
        result = ApiAgent(provider, ws.tools, ws.policy).run(
            [Message(Role.USER, "read both")]
        )
        read_results = {item.tool_call_id: item for item in _tool_results(result.messages)}
        if (
            not read_results["read-existing"].ok
            or "hello from disk" not in read_results["read-existing"].content
            or not read_results["read-second"].ok
            or "second from disk" not in read_results["read-second"].content
        ):
            fail(f"parallel reads returned incorrect content: {read_results!r}")

        ws.tools["write_file"]._ledger = coordinated_ledger._ledger
        writes = [
            ws.tools["write_file"].execute(
                ToolCall(
                    id="write-existing",
                    name="write_file",
                    arguments={"path": "existing.txt", "content": "first changed"},
                ),
                ws.policy,
            ),
            ws.tools["write_file"].execute(
                ToolCall(
                    id="write-second",
                    name="write_file",
                    arguments={"path": "second.txt", "content": "second changed"},
                ),
                ws.policy,
            ),
        ]
        if not all(item.ok for item in writes):
            fail(f"parallel reads were not both recorded in the ledger: {writes!r}")


@check("scheduler.result_order")
def check_result_order() -> None:
    with workspace() as ws:
        tool = _SafeTool("timed")
        result = ApiAgent(
            FakeModelProvider(
                [
                    _tool_turn(
                        ToolCall(
                            id="first",
                            name=tool.name,
                            arguments={"value": "first-value", "delay": 0.05},
                        ),
                        ToolCall(
                            id="second",
                            name=tool.name,
                            arguments={"value": "second-value"},
                        ),
                    )
                ]
            ),
            {tool.name: tool},
            ws.policy,
            max_turns=1,
        ).run([Message(Role.USER, "invert completion")])
        ordered_results = _tool_results(result.messages)
        if tool.completion_order != ["second", "first"]:
            fail(f"test did not force completion-order inversion: {tool.completion_order!r}")
        if [item.tool_call_id for item in ordered_results] != ["first", "second"]:
            fail(f"conversation followed completion order: {ordered_results!r}")
        if [item.content for item in ordered_results] != ["first-value", "second-value"]:
            fail(f"ordered results carried the wrong content: {ordered_results!r}")


@check("scheduler.event_order")
def check_event_order() -> None:
    with workspace() as ws:
        tool = _SafeTool("event_safe")
        events = CollectingSink()
        ApiAgent(
            FakeModelProvider(
                [
                    _tool_turn(
                        ToolCall(
                            id="event-first",
                            name=tool.name,
                            arguments={"delay": 0.05},
                        ),
                        ToolCall(id="event-second", name=tool.name),
                    )
                ]
            ),
            {tool.name: tool},
            ws.policy,
            max_turns=1,
            events=events,
        ).run([Message(Role.USER, "events")])
        tool_events = [
            (type(event).__name__, event.tool_call_id)
            for event in events.events
            if isinstance(event, (ToolCallStarted, ToolCallFinished))
        ]
        expected = [
            ("ToolCallStarted", "event-first"),
            ("ToolCallStarted", "event-second"),
            ("ToolCallFinished", "event-first"),
            ("ToolCallFinished", "event-second"),
        ]
        if tool_events != expected:
            fail(f"parallel tool events were not deterministically ordered: {tool_events!r}")


@check("scheduler.singleton_compatibility")
def check_singleton_compatibility() -> None:
    with workspace() as ws:
        events = CollectingSink()
        result = ApiAgent(
            FakeModelProvider(
                [
                    _tool_turn(
                        ToolCall(
                            id="single",
                            name="read_file",
                            arguments={"path": "existing.txt"},
                        )
                    )
                ]
            ),
            ws.tools,
            ws.policy,
            max_turns=1,
            events=events,
        ).run([Message(Role.USER, "read")])
        actual_events = [
            (
                type(event).__name__,
                getattr(event, "tool_call_id", None),
                getattr(event, "ok", None),
            )
            for event in events.events
        ]
        if actual_events != LEGACY_SINGLETON_EVENT_SEQUENCE:
            fail(f"singleton event sequence changed from HEAD: {actual_events!r}")
        message_sequence = [
            (
                message.role.value,
                message.tool_result.tool_call_id if message.tool_result else None,
                message.tool_result.ok if message.tool_result else None,
            )
            for message in result.messages
        ]
        expected_messages = [
            ("user", None, None),
            ("assistant", None, None),
            ("tool", "single", True),
        ]
        if message_sequence != expected_messages:
            fail(f"singleton message sequence changed from HEAD: {message_sequence!r}")
        tool_message = result.messages[-1]
        if (
            "hello from disk" not in tool_message.tool_result.content
            or tool_message.turn_id != result.messages[-2].turn_id
        ):
            fail(f"singleton tool message payload changed from HEAD: {tool_message!r}")

        thread_tool = _ThreadIdentityTool()
        thread_result = ApiAgent(
            FakeModelProvider(
                [_tool_turn(ToolCall(id="thread", name=thread_tool.name))]
            ),
            {thread_tool.name: thread_tool},
            ws.policy,
            max_turns=1,
        ).run([Message(Role.USER, "singleton thread")])
        if _tool_results(thread_result.messages)[0].content != "MainThread":
            fail("a singleton tool call was moved onto an executor thread")


@check("scheduler.exception_isolation")
def check_exception_isolation() -> None:
    with workspace() as ws:
        tool = _SafeTool("raising_safe")
        result = ApiAgent(
            FakeModelProvider(
                [
                    _tool_turn(
                        ToolCall(id="mate-one", name=tool.name, arguments={"value": "one"}),
                        ToolCall(
                            id="broken",
                            name=tool.name,
                            arguments={"error": "batch boom"},
                        ),
                        ToolCall(id="mate-two", name=tool.name, arguments={"value": "two"}),
                    )
                ]
            ),
            {tool.name: tool},
            ws.policy,
            max_turns=1,
        ).run([Message(Role.USER, "isolate failure")])
        results = {item.tool_call_id: item for item in _tool_results(result.messages)}
        if not results["mate-one"].ok or results["mate-one"].content != "one":
            fail(f"first batch-mate result was lost: {results!r}")
        if results["broken"].ok or results["broken"].error != "RuntimeError: batch boom":
            fail(f"raising tool did not become an error result: {results!r}")
        if not results["mate-two"].ok or results["mate-two"].content != "two":
            fail(f"second batch-mate result was lost: {results!r}")


@check("scheduler.cancellation_repair")
def check_cancellation_repair() -> None:
    with workspace() as ws:
        token = CancellationToken()
        tool = _CancellingBatchTool()
        result = ApiAgent(
            FakeModelProvider(
                [
                    _tool_turn(
                        ToolCall(
                            id="cancel-first",
                            name=tool.name,
                            arguments={"sequence": 1},
                        ),
                        ToolCall(
                            id="cancel-second",
                            name=tool.name,
                            arguments={"sequence": 2},
                        ),
                        ToolCall(
                            id="cancel-third",
                            name=tool.name,
                            arguments={"sequence": 3},
                        ),
                    )
                ]
            ),
            {tool.name: tool},
            ws.policy,
        ).run([Message(Role.USER, "cancel batch")], cancel=token)
        returned = _tool_results(result.messages)
        counts = Counter(item.tool_call_id for item in returned)
        by_id = {item.tool_call_id: item for item in returned}
        if result.stopped_reason != "cancelled":
            fail(f"batch cancellation escaped the turn boundary: {result!r}")
        if counts != Counter(
            {"cancel-first": 1, "cancel-second": 1, "cancel-third": 1}
        ):
            fail(f"cancelled batch answered ids incorrectly: {returned!r}")
        if not by_id["cancel-first"].ok or by_id["cancel-first"].content != "first-real":
            fail(f"completed result was replaced during repair: {returned!r}")
        if not by_id["cancel-second"].cancelled or not by_id["cancel-third"].cancelled:
            fail(f"unanswered cancelled calls were not repaired: {returned!r}")


@check("scheduler.ledger_locking")
def check_ledger_locking() -> None:
    probe_lock = _ProbeLock()
    probe_path = _ProbePath(probe_lock)
    probe_ledger = ReadLedger()
    probe_ledger._lock = probe_lock
    probe_ledger.record(probe_path, full=True, content="same")
    probe_path.mtime_ns = 2
    if probe_ledger.check(probe_path) is not None:
        fail("unchanged probe content was unexpectedly stale")
    if probe_lock.held or probe_lock.entries != 2:
        fail(f"ledger did not lock record and check exactly once: {probe_lock.__dict__!r}")
    if probe_ledger._records[probe_path].mtime_ns != 2:
        fail("ledger did not refresh unchanged content while holding the lock")
    if "concurrency-safe" not in (ReadLedger.__doc__ or "") or "lock" not in (
        ReadLedger.__doc__ or ""
    ):
        fail(f"ReadLedger docstring does not explain its lock: {ReadLedger.__doc__!r}")

    with workspace() as ws:
        ledger = ReadLedger()
        paths = []
        for index in range(200):
            path = ws.root / f"ledger-{index}.txt"
            path.write_text(str(index))
            paths.append(path)
        errors: list[Exception] = []
        errors_lock = threading.Lock()

        def record_paths(assigned_paths) -> None:
            try:
                for path in assigned_paths:
                    ledger.record(path, full=True, content=path.read_text())
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=record_paths, args=(paths[index::8],))
            for index in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in threads):
            fail("concurrent ledger writers did not finish")
        if errors:
            fail(f"concurrent ledger writers raised: {errors!r}")
        if len(ledger._records) != 200:
            fail(f"concurrent ledger writes left {len(ledger._records)} of 200 entries")
