"""Checks for bounded read-ledger records and cached range reads."""

from __future__ import annotations

import os
import unittest.mock as mock
from pathlib import Path
from types import SimpleNamespace

from orchestra_api.models import ToolCall
from orchestra_api.runner import standard_tool_registry
import orchestra_api.tools.filesystem as filesystem
import orchestra_api.tools.read_ledger as read_ledger
from orchestra_api.tools.filesystem import MAX_READ_LINES
from orchestra_api.tools.read_ledger import ReadLedger
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


NOT_READ_ERROR = "file has not been read yet; read it with read_file before editing it"
CHANGED_ERROR = "file has changed since it was read; read it again before editing it"
PARTIAL_VIEW_ERROR = (
    "only a processed view of this file has been read; "
    "read it with read_file before editing it"
)


class _FakePath:
    def __init__(self, label: str, *, mtime_ns: int = 1, content: str = "") -> None:
        self.label = label
        self.mtime_ns = mtime_ns
        self.content = content
        self.raise_on_stat = False

    def stat(self) -> SimpleNamespace:
        if self.raise_on_stat:
            raise AssertionError("partial-view check touched the filesystem")
        return SimpleNamespace(st_mtime_ns=self.mtime_ns)

    def read_text(self, *, encoding: str) -> str:
        if encoding != "utf-8":
            raise AssertionError(f"unexpected encoding: {encoding!r}")
        return self.content


class _ProbeLock:
    def __init__(self) -> None:
        self.held = False
        self.entries = 0

    def __enter__(self) -> None:
        if self.held:
            raise AssertionError("probe lock re-entered")
        self.held = True
        self.entries += 1

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        if not self.held:
            raise AssertionError("probe lock exited while unlocked")
        self.held = False


class _ProbePath(_FakePath):
    def __init__(self, lock: _ProbeLock) -> None:
        super().__init__("probe", content="same")
        self.lock = lock

    def stat(self) -> SimpleNamespace:
        if not self.lock.held:
            raise AssertionError("stat ran outside the ledger lock")
        return super().stat()


@check("ledger.entry_lru_bound")
def check_entry_lru_bound() -> None:
    ledger = ReadLedger()
    paths = [_FakePath(str(index)) for index in range(5)]
    with mock.patch.object(read_ledger, "MAX_LEDGER_ENTRIES", 3), mock.patch.object(
        read_ledger, "MAX_LEDGER_BYTES", 1_000
    ):
        for path in paths:
            ledger.record(path, full=True, content=path.label)
    if list(ledger._records) != paths[-3:]:
        fail(f"entry LRU bound kept the wrong records: {list(ledger._records)!r}")


@check("ledger.content_lru_bound")
def check_content_lru_bound() -> None:
    ledger = ReadLedger()
    first = _FakePath("first")
    second = _FakePath("second")
    with mock.patch.object(read_ledger, "MAX_LEDGER_ENTRIES", 10), mock.patch.object(
        read_ledger, "MAX_LEDGER_BYTES", 5
    ):
        ledger.record(first, full=True, content="aaa")
        ledger.record(second, full=True, content="bbb")
    if len(ledger._records) != 2:
        fail(f"content pressure dropped a record: {ledger._records!r}")
    if ledger._records[first].content is not None or ledger._records[second].content != "bbb":
        fail(f"content was not degraded in LRU order: {ledger._records!r}")


@check("ledger.evicted_content_unchanged")
def check_evicted_content_unchanged() -> None:
    ledger = ReadLedger()
    path = _FakePath("unchanged", content="same")
    with mock.patch.object(read_ledger, "MAX_LEDGER_BYTES", 0):
        ledger.record(path, full=True, content="same")
        result = ledger.check(path)
    if ledger._records[path].content is not None or result is not None:
        fail(f"content-evicted unchanged record was refused: {result!r}")


@check("ledger.evicted_content_mtime_bump")
def check_evicted_content_mtime_bump() -> None:
    ledger = ReadLedger()
    path = _FakePath("bumped", content="same")
    with mock.patch.object(read_ledger, "MAX_LEDGER_BYTES", 0):
        ledger.record(path, full=True, content="same")
        path.mtime_ns += 1
        result = ledger.check(path)
    if result != CHANGED_ERROR:
        fail(f"content-evicted newer record was not stale: {result!r}")


@check("ledger.dropped_record_refused")
def check_dropped_record_refused() -> None:
    ledger = ReadLedger()
    first = _FakePath("first")
    second = _FakePath("second")
    with mock.patch.object(read_ledger, "MAX_LEDGER_ENTRIES", 1):
        ledger.record(first, full=True, content="first")
        ledger.record(second, full=True, content="second")
    result = ledger.check(first)
    if result != NOT_READ_ERROR:
        fail(f"dropped record did not use the existing refusal: {result!r}")


@check("ledger.check_refreshes_recency")
def check_check_refreshes_recency() -> None:
    ledger = ReadLedger()
    first = _FakePath("first")
    second = _FakePath("second")
    third = _FakePath("third")
    with mock.patch.object(read_ledger, "MAX_LEDGER_ENTRIES", 2):
        ledger.record(first, full=True, content="first")
        ledger.record(second, full=True, content="second")
        if ledger.check(first) is not None:
            fail("unchanged record failed its recency-refresh check")
        ledger.record(third, full=True, content="third")
    if list(ledger._records) != [first, third]:
        fail(f"successful check did not refresh LRU recency: {list(ledger._records)!r}")


@check("ledger.recorded_ranges")
def check_recorded_ranges() -> None:
    with workspace() as ws:
        ranged_path = ws.root / "ranged.txt"
        full_path = ws.root / "full.txt"
        ranged_path.write_text("one\ntwo\nthree\n")
        full_path.write_text("whole\n")
        read_tool = ws.tools["read_file"]
        ranged = read_tool.execute(
            ToolCall(
                id="range-record",
                name="read_file",
                arguments={"path": ranged_path.name, "offset": 2, "limit": 1},
            ),
            ws.policy,
        )
        full = read_tool.execute(
            ToolCall(
                id="full-record",
                name="read_file",
                arguments={"path": full_path.name},
            ),
            ws.policy,
        )
        ranged_record = read_tool._ledger._records[ranged_path.resolve()]
        full_record = read_tool._ledger._records[full_path.resolve()]
        if not ranged.ok or (ranged_record.offset, ranged_record.limit) != (2, 1):
            fail(f"ranged read recorded the wrong bounds: {ranged_record!r}")
        if not full.ok or (full_record.offset, full_record.limit) != (1, MAX_READ_LINES):
            fail(f"unranged read recorded the wrong defaults: {full_record!r}")


@check("ledger.partial_view_refused")
def check_partial_view_refused() -> None:
    ledger = ReadLedger()
    path = _FakePath("partial")
    ledger.record(path, full=False, content="hidden", partial_view=True)
    if ledger._records[path].content is not None:
        fail(f"partial view retained content: {ledger._records[path]!r}")
    path.raise_on_stat = True
    result = ledger.check(path)
    if result != PARTIAL_VIEW_ERROR:
        fail(f"partial-view refusal changed: {result!r}")


@check("ledger.cached_output_identical")
def check_cached_output_identical() -> None:
    with workspace() as ws:
        path = ws.root / "cached.txt"
        path.write_text("one\ntwo\nthree\nfour\n")
        arguments = {"path": path.name, "offset": 2, "limit": 2}
        read_tool = ws.tools["read_file"]
        first = read_tool.execute(
            ToolCall(id="cache-first", name="read_file", arguments=arguments),
            ws.policy,
        )
        with mock.patch.object(
            Path,
            "open",
            autospec=True,
            side_effect=AssertionError("matching cached read reopened the file"),
        ):
            cached = read_tool.execute(
                ToolCall(id="cache-second", name="read_file", arguments=arguments),
                ws.policy,
            )
        fresh = standard_tool_registry()["read_file"].execute(
            ToolCall(id="cache-fresh", name="read_file", arguments=arguments),
            ws.policy,
        )
        expected = "2\ttwo\n3\tthree\n[lines 2-3; more follow, pass offset=4]"
        if not first.ok or cached.content != fresh.content or cached.content != expected:
            fail(
                "cached output differed from a fresh read: "
                f"first={first!r}, cached={cached!r}, fresh={fresh!r}"
            )


@check("ledger.mtime_change_rereads")
def check_mtime_change_rereads() -> None:
    with workspace() as ws:
        path = ws.root / "mtime.txt"
        path.write_text("old\n")
        read_tool = ws.tools["read_file"]
        read_tool.execute(
            ToolCall(id="mtime-first", name="read_file", arguments={"path": path.name}),
            ws.policy,
        )
        prior_stat = path.stat()
        path.write_text("new\n")
        os.utime(path, ns=(prior_stat.st_atime_ns, prior_stat.st_mtime_ns + 1_000_000_000))
        reread = read_tool.execute(
            ToolCall(id="mtime-second", name="read_file", arguments={"path": path.name}),
            ws.policy,
        )
        if not reread.ok or reread.content != "1\tnew":
            fail(f"newer file was served from stale cache: {reread!r}")


@check("ledger.range_change_rereads")
def check_range_change_rereads() -> None:
    with workspace() as ws:
        path = ws.root / "ranges.txt"
        path.write_text("one\ntwo\nthree\nfour\n")
        read_tool = ws.tools["read_file"]
        read_tool.execute(
            ToolCall(
                id="range-first",
                name="read_file",
                arguments={"path": path.name, "offset": 1, "limit": 2},
            ),
            ws.policy,
        )
        real_open = Path.open
        open_count = [0]

        def counting_open(target, *args, **kwargs):  # noqa: ANN001
            open_count[0] += 1
            return real_open(target, *args, **kwargs)

        with mock.patch.object(Path, "open", autospec=True, side_effect=counting_open):
            changed_offset = read_tool.execute(
                ToolCall(
                    id="range-offset",
                    name="read_file",
                    arguments={"path": path.name, "offset": 2, "limit": 2},
                ),
                ws.policy,
            )
            changed_limit = read_tool.execute(
                ToolCall(
                    id="range-limit",
                    name="read_file",
                    arguments={"path": path.name, "offset": 2, "limit": 1},
                ),
                ws.policy,
            )
        if open_count != [2]:
            fail(f"different read bounds did not reopen twice: {open_count!r}")
        if not changed_offset.content.startswith("2\ttwo\n3\tthree"):
            fail(f"changed offset returned the wrong range: {changed_offset!r}")
        if changed_limit.content != "2\ttwo\n[lines 2-2; more follow, pass offset=3]":
            fail(f"changed limit returned the wrong range: {changed_limit!r}")


@check("ledger.oversize_unranged_still_refused")
def check_oversize_unranged_still_refused() -> None:
    with workspace() as ws:
        path = ws.root / "oversize.txt"
        path.write_text("".join("x" * 19 + "\n" for _ in range(10)))
        read_tool = ws.tools["read_file"]
        with mock.patch.object(filesystem, "MAX_READ_BYTES", 100), mock.patch.object(
            filesystem, "MAX_READ_LINES", 3
        ):
            ranged = read_tool.execute(
                ToolCall(
                    id="oversize-ranged",
                    name="read_file",
                    arguments={"path": path.name, "offset": 1, "limit": 3},
                ),
                ws.policy,
            )
            unranged = read_tool.execute(
                ToolCall(
                    id="oversize-unranged",
                    name="read_file",
                    arguments={"path": path.name},
                ),
                ws.policy,
            )
        if not ranged.ok:
            fail(f"bounded setup read failed: {ranged!r}")
        expected_error = (
            f"file exceeds {100} byte read limit; read part of it with offset and limit"
        )
        if unranged.ok or unranged.error != expected_error:
            fail(f"oversize unranged cache collision was not refused: {unranged!r}")


@check("ledger.messages_unchanged")
def check_messages_unchanged() -> None:
    ledger = ReadLedger()
    unread = _FakePath("unread")
    if ledger.check(unread) != NOT_READ_ERROR:
        fail("not-read ledger message changed")
    changed = _FakePath("changed", content="before")
    ledger.record(changed, full=True, content="before")
    changed.mtime_ns += 1
    changed.content = "after"
    if ledger.check(changed) != CHANGED_ERROR:
        fail("changed-file ledger message changed")


@check("ledger.lock_scope")
def check_lock_scope() -> None:
    ledger = ReadLedger()
    probe_lock = _ProbeLock()
    probe_path = _ProbePath(probe_lock)
    ledger._lock = probe_lock
    real_enforce_bounds = ReadLedger._enforce_bounds
    bound_probes = [0]

    def probed_enforce_bounds(subject: ReadLedger) -> None:
        if not probe_lock.held:
            raise AssertionError("ledger eviction ran outside the lock")
        bound_probes[0] += 1
        real_enforce_bounds(subject)

    with mock.patch.object(
        ReadLedger,
        "_enforce_bounds",
        autospec=True,
        side_effect=probed_enforce_bounds,
    ), mock.patch.object(read_ledger, "MAX_LEDGER_BYTES", 0):
        ledger.record(probe_path, full=True, content="same")
        if ledger.check(probe_path) is not None:
            fail("lock probe record was unexpectedly stale")
    if probe_lock.held or probe_lock.entries != 2 or bound_probes != [2]:
        fail(
            "record/check did not keep stat and eviction under one lock: "
            f"lock={probe_lock.__dict__!r}, bounds={bound_probes!r}"
        )
