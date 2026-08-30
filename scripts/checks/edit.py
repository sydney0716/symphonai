"""Workspace-backed checks for edit."""

from __future__ import annotations

import json
import os
import unittest.mock as mock
from pathlib import Path
from orchestra_api.models import ToolCall, ToolResult
from orchestra_api.runner import standard_tool_registry
import orchestra_api.tools.filesystem as filesystem_tools
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


@check("edit.not_read_refused")
def check_edit_not_read_refused() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        # -- targeted edits, read-before-write staleness, and structured diffs --
        not_read_path = root / "edit-not-read.txt"
        not_read_path.write_text("before\n")
        not_read = tools["edit_file"].execute(
            ToolCall(
                id="edit-not-read",
                name="edit_file",
                arguments={
                    "path": not_read_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        not_read_error = "file has not been read yet; read it with read_file before editing it"
        if not_read.ok or not_read.error != not_read_error:
            fail(f"edit_file no-read refusal changed: {not_read!r}")

@check("edit.stale_refused")
def check_edit_stale_refused() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        stale_path = root / "edit-stale.txt"
        stale_path.write_text("before\n")
        tools["read_file"].execute(
            ToolCall(id="read-stale", name="read_file", arguments={"path": stale_path.name}),
            policy,
        )
        stale_stat = stale_path.stat()
        stale_path.write_text("outside change\n")
        os.utime(
            stale_path,
            ns=(stale_stat.st_atime_ns, stale_stat.st_mtime_ns + 1_000_000_000),
        )
        stale = tools["edit_file"].execute(
            ToolCall(
                id="edit-stale",
                name="edit_file",
                arguments={
                    "path": stale_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        stale_error = "file has changed since it was read; read it again before editing it"
        if stale.ok or stale.error != stale_error:
            fail(f"edit_file stale refusal changed: {stale!r}")

@check("edit.same_content_allowed")
def check_edit_same_content_allowed() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        stale_error = "file has changed since it was read; read it again before editing it"
        same_content_path = root / "edit-same-content.txt"
        same_content_path.write_text("before\n")
        tools["read_file"].execute(
            ToolCall(
                id="read-same-content",
                name="read_file",
                arguments={"path": same_content_path.name},
            ),
            policy,
        )
        same_stat = same_content_path.stat()
        same_content_path.write_text("before\n")
        os.utime(
            same_content_path,
            ns=(same_stat.st_atime_ns, same_stat.st_mtime_ns + 1_000_000_000),
        )
        same_content = tools["edit_file"].execute(
            ToolCall(
                id="edit-same-content",
                name="edit_file",
                arguments={
                    "path": same_content_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        if not same_content.ok or same_content_path.read_text() != "after\n":
            fail(f"identical-content mtime bump was treated as stale: {same_content!r}")

@check("edit.ranged_read_stale")
def check_edit_ranged_read_stale() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        stale_error = "file has changed since it was read; read it again before editing it"
        ranged_path = root / "edit-ranged-stale.txt"
        ranged_path.write_text("before\nafter\n")
        tools["read_file"].execute(
            ToolCall(
                id="read-ranged-stale",
                name="read_file",
                arguments={"path": ranged_path.name, "limit": 1},
            ),
            policy,
        )
        ranged_stat = ranged_path.stat()
        ranged_path.write_text("before\nafter\n")
        os.utime(
            ranged_path,
            ns=(ranged_stat.st_atime_ns, ranged_stat.st_mtime_ns + 1_000_000_000),
        )
        ranged_stale = tools["edit_file"].execute(
            ToolCall(
                id="edit-ranged-stale",
                name="edit_file",
                arguments={
                    "path": ranged_path.name,
                    "old_string": "before",
                    "new_string": "changed",
                },
            ),
            policy,
        )
        if ranged_stale.ok or ranged_stale.error != stale_error:
            fail(f"ranged read incorrectly proved unchanged content: {ranged_stale!r}")

@check("edit.long_unranged_stale")
def check_edit_long_unranged_stale() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        stale_error = "file has changed since it was read; read it again before editing it"
        long_unranged_path = root / "edit-long-unranged-stale.txt"
        long_unranged_path.write_text("one\ntwo\nthree\n")
        with mock.patch.object(filesystem_tools, "MAX_READ_LINES", 2):
            long_unranged_read = tools["read_file"].execute(
                ToolCall(
                    id="read-long-unranged-stale",
                    name="read_file",
                    arguments={"path": long_unranged_path.name},
                ),
                policy,
            )
        if not long_unranged_read.ok or "more follow" not in long_unranged_read.content:
            fail(f"long unranged fixture did not truncate: {long_unranged_read!r}")
        long_unranged_stat = long_unranged_path.stat()
        long_unranged_path.write_text("one\ntwo\nthree\n")
        os.utime(
            long_unranged_path,
            ns=(
                long_unranged_stat.st_atime_ns,
                long_unranged_stat.st_mtime_ns + 1_000_000_000,
            ),
        )
        long_unranged_edit = tools["edit_file"].execute(
            ToolCall(
                id="edit-long-unranged-stale",
                name="edit_file",
                arguments={
                    "path": long_unranged_path.name,
                    "old_string": "one",
                    "new_string": "changed",
                },
            ),
            policy,
        )
        if long_unranged_edit.ok or long_unranged_edit.error != stale_error:
            fail(
                "truncated unranged read incorrectly proved unchanged content: "
                f"{long_unranged_edit!r}"
            )

@check("edit.single_open")
def check_edit_single_open() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        single_open_path = root / "edit-single-open.txt"
        single_open_path.write_text("before\n")
        real_path_open = Path.open
        real_path_read_text = Path.read_text
        open_count = [0]
        read_text_count = [0]

        def _count_path_open(path, *args, **kwargs):  # noqa: ANN001
            open_count[0] += 1
            return real_path_open(path, *args, **kwargs)

        def _count_path_read_text(path, *args, **kwargs):  # noqa: ANN001
            read_text_count[0] += 1
            return real_path_read_text(path, *args, **kwargs)

        with mock.patch.object(
            Path, "open", autospec=True, side_effect=_count_path_open
        ), mock.patch.object(
            Path, "read_text", autospec=True, side_effect=_count_path_read_text
        ):
            single_open_read = tools["read_file"].execute(
                ToolCall(
                    id="read-single-open",
                    name="read_file",
                    arguments={"path": single_open_path.name},
                ),
                policy,
            )
        if (
            not single_open_read.ok
            or open_count != [1]
            or read_text_count != [0]
        ):
            fail(
                "short unranged read did not open exactly once: "
                f"result={single_open_read!r}, opens={open_count!r}, "
                f"read_text={read_text_count!r}"
            )
        single_open_stat = single_open_path.stat()
        single_open_path.write_text("before\n")
        os.utime(
            single_open_path,
            ns=(
                single_open_stat.st_atime_ns,
                single_open_stat.st_mtime_ns + 1_000_000_000,
            ),
        )
        single_open_edit = tools["edit_file"].execute(
            ToolCall(
                id="edit-single-open",
                name="edit_file",
                arguments={
                    "path": single_open_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        if not single_open_edit.ok or single_open_path.read_text() != "after\n":
            fail(f"short full read no longer forgives an identical rewrite: {single_open_edit!r}")

@check("edit.match_count")
def check_edit_match_count() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        match_path = root / "edit-matches.txt"
        match_path.write_text("one one\n")
        tools["read_file"].execute(
            ToolCall(id="read-matches", name="read_file", arguments={"path": match_path.name}),
            policy,
        )
        missing_match = tools["edit_file"].execute(
            ToolCall(
                id="edit-zero-match",
                name="edit_file",
                arguments={
                    "path": match_path.name,
                    "old_string": "absent",
                    "new_string": "new",
                },
            ),
            policy,
        )
        if missing_match.ok or missing_match.error != "old_string not found in edit-matches.txt":
            fail(f"zero-match error changed: {missing_match!r}")
        multi_match = tools["edit_file"].execute(
            ToolCall(
                id="edit-multi-match",
                name="edit_file",
                arguments={
                    "path": match_path.name,
                    "old_string": "one",
                    "new_string": "two",
                },
            ),
            policy,
        )
        expected_multi_match = (
            "old_string matches 2 times in edit-matches.txt; set replace_all=true to "
            "change every match, or add surrounding context to identify one"
        )
        if multi_match.ok or multi_match.error != expected_multi_match:
            fail(f"multi-match error changed: {multi_match!r}")
        replace_all = tools["edit_file"].execute(
            ToolCall(
                id="edit-replace-all",
                name="edit_file",
                arguments={
                    "path": match_path.name,
                    "old_string": "one",
                    "new_string": "two",
                    "replace_all": True,
                },
            ),
            policy,
        )
        if not replace_all.ok or match_path.read_text() != "two two\n":
            fail(f"replace_all did not replace every occurrence: {replace_all!r}")

@check("edit.one_match")
def check_edit_one_match() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        one_match_path = root / "edit-one-match.txt"
        one_match_path.write_text("before keep\n")
        tools["read_file"].execute(
            ToolCall(id="read-one-match", name="read_file", arguments={"path": one_match_path.name}),
            policy,
        )
        one_match = tools["edit_file"].execute(
            ToolCall(
                id="edit-one-match",
                name="edit_file",
                arguments={
                    "path": one_match_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        if not one_match.ok or one_match_path.read_text() != "after keep\n":
            fail(f"default single replacement changed: {one_match!r}")

@check("edit.multi_edit_sequence")
def check_edit_multi_edit_sequence() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        batch_path = root / "multi-edit.txt"
        batch_path.write_text("alpha beta gamma\n")
        tools["read_file"].execute(
            ToolCall(id="read-multi-edit", name="read_file", arguments={"path": batch_path.name}),
            policy,
        )
        ordered_batch = tools["multi_edit_file"].execute(
            ToolCall(
                id="multi-edit-ordered",
                name="multi_edit_file",
                arguments={
                    "path": batch_path.name,
                    "edits": [
                        {"old_string": "alpha", "new_string": "delta"},
                        {"old_string": "delta beta", "new_string": "joined"},
                    ],
                },
            ),
            policy,
        )
        if not ordered_batch.ok or batch_path.read_text() != "joined gamma\n":
            fail(f"multi_edit_file did not apply edits in order: {ordered_batch!r}")
        if (
            ordered_batch.payload is None
            or ordered_batch.payload["kind"] != "file_diff"
            or ordered_batch.payload["path"] != "multi-edit.txt"
            or json.loads(json.dumps(ordered_batch.payload)) != ordered_batch.payload
            or not ordered_batch.content.startswith("--- multi-edit.txt\n+++ multi-edit.txt\n")
        ):
            fail(f"multi_edit_file did not return a structured unified diff: {ordered_batch!r}")
        before_failed_batch = batch_path.read_text()
        failed_batch = tools["multi_edit_file"].execute(
            ToolCall(
                id="multi-edit-atomic",
                name="multi_edit_file",
                arguments={
                    "path": batch_path.name,
                    "edits": [
                        {"old_string": "joined", "new_string": "changed"},
                        {"old_string": "absent", "new_string": "never"},
                    ],
                },
            ),
            policy,
        )
        if failed_batch.ok or failed_batch.error != "edit 2: old_string not found in multi-edit.txt":
            fail(f"multi_edit_file failure message changed: {failed_batch!r}")
        if batch_path.read_text() != before_failed_batch:
            fail("multi_edit_file wrote a partial batch")

        consecutive_second = tools["edit_file"].execute(
            ToolCall(
                id="edit-consecutive",
                name="edit_file",
                arguments={
                    "path": batch_path.name,
                    "old_string": "joined",
                    "new_string": "final",
                },
            ),
            policy,
        )
        if not consecutive_second.ok or batch_path.read_text() != "final gamma\n":
            fail(f"writer did not refresh the ledger after its own write: {consecutive_second!r}")

@check("edit.structured_diff")
def check_edit_structured_diff() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        diff_path = root / "structured-diff.txt"
        diff_path.write_text("alpha\nkeep\nomega\n")
        tools["read_file"].execute(
            ToolCall(id="read-diff", name="read_file", arguments={"path": diff_path.name}),
            policy,
        )
        diff_result = tools["edit_file"].execute(
            ToolCall(
                id="edit-diff",
                name="edit_file",
                arguments={
                    "path": diff_path.name,
                    "old_string": "keep",
                    "new_string": "changed",
                },
            ),
            policy,
        )
        expected_diff = (
            "--- structured-diff.txt\n+++ structured-diff.txt\n@@ -1,3 +1,3 @@\n"
            " alpha\n-keep\n+changed\n omega"
        )
        if diff_result.content != expected_diff:
            fail(f"unified diff rendering changed: {diff_result!r}")
        expected_payload = {
            "kind": "file_diff",
            "path": "structured-diff.txt",
            "lines_added": 1,
            "lines_removed": 1,
            "truncated": False,
            "hunks": [
                {
                    "old_start": 1,
                    "old_lines": 3,
                    "new_start": 1,
                    "new_lines": 3,
                    "lines": [
                        {"op": "context", "text": "alpha"},
                        {"op": "remove", "text": "keep"},
                        {"op": "add", "text": "changed"},
                        {"op": "context", "text": "omega"},
                    ],
                }
            ],
        }
        if diff_result.payload != expected_payload:
            fail(f"structured diff payload changed: {diff_result.payload!r}")
        if json.loads(json.dumps(diff_result.payload)) != diff_result.payload:
            fail(f"structured diff payload was not JSON-stable: {diff_result.payload!r}")
        diff_ops = {
            line["op"]
            for hunk in diff_result.payload["hunks"]
            for line in hunk["lines"]
        }
        if diff_ops != {"context", "add", "remove"}:
            fail(f"structured diff line operations changed: {diff_ops!r}")

@check("edit.truncated_diff")
def check_edit_truncated_diff() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        truncated_path = root / "truncated-diff.txt"
        truncated_path.write_text("a long original line\n")
        tools["read_file"].execute(
            ToolCall(
                id="read-truncated-diff",
                name="read_file",
                arguments={"path": truncated_path.name},
            ),
            policy,
        )
        with mock.patch.object(filesystem_tools, "MAX_READ_BYTES", 10):
            truncated_diff = tools["edit_file"].execute(
                ToolCall(
                    id="edit-truncated-diff",
                    name="edit_file",
                    arguments={
                        "path": truncated_path.name,
                        "old_string": "a long original line",
                        "new_string": "a substantially different line",
                    },
                ),
                policy,
            )
        expected_omitted = (
            "[diff omitted: 1 lines added, 1 removed, over the 10 byte diff limit]"
        )
        if (
            not truncated_diff.ok
            or truncated_diff.content != expected_omitted
            or truncated_diff.payload["hunks"] != []
            or truncated_diff.payload["truncated"] is not True
        ):
            fail(f"oversized diff was not safely omitted: {truncated_diff!r}")

        write_created = tools["write_file"].execute(
            ToolCall(
                id="write-create-unread",
                name="write_file",
                arguments={"path": "write-created.txt", "content": "created"},
            ),
            policy,
        )
        if not write_created.ok or (root / "write-created.txt").read_text() != "created":
            fail(f"write_file did not create an unread path: {write_created!r}")

@check("edit.write_unread_existing")
def check_edit_write_unread_existing() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        not_read_error = "file has not been read yet; read it with read_file before editing it"
        write_unread_path = root / "write-unread-existing.txt"
        write_unread_path.write_text("original")
        write_unread = tools["write_file"].execute(
            ToolCall(
                id="write-unread-existing",
                name="write_file",
                arguments={"path": write_unread_path.name, "content": "replacement"},
            ),
            policy,
        )
        if write_unread.ok or write_unread.error != not_read_error:
            fail(f"write_file overwrote an unread existing file: {write_unread!r}")
        if write_unread_path.read_text() != "original":
            fail("write_file changed an unread existing file despite refusing it")

@check("edit.isolated_ledger")
def check_edit_isolated_ledger() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        not_read_error = "file has not been read yet; read it with read_file before editing it"
        first_registry = standard_tool_registry()
        second_registry = standard_tool_registry()
        isolated_path = root / "isolated-ledger.txt"
        isolated_path.write_text("before\n")
        first_registry["read_file"].execute(
            ToolCall(id="isolated-read", name="read_file", arguments={"path": isolated_path.name}),
            policy,
        )
        isolated_edit = second_registry["edit_file"].execute(
            ToolCall(
                id="isolated-edit",
                name="edit_file",
                arguments={
                    "path": isolated_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        if isolated_edit.ok or isolated_edit.error != not_read_error:
            fail(f"read ledger leaked between registries: {isolated_edit!r}")

@check("edit.narrowed_isolated_ledger")
def check_edit_narrowed_isolated_ledger() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        not_read_error = "file has not been read yet; read it with read_file before editing it"
        narrowed_names = ["read_file", "write_file", "edit_file"]
        first_narrowed_registry = standard_tool_registry(narrowed_names)
        second_narrowed_registry = standard_tool_registry(narrowed_names)
        narrowed_isolated_path = root / "narrowed-isolated-ledger.txt"
        narrowed_isolated_path.write_text("before\n")
        first_narrowed_registry["read_file"].execute(
            ToolCall(
                id="narrowed-isolated-read",
                name="read_file",
                arguments={"path": narrowed_isolated_path.name},
            ),
            policy,
        )
        narrowed_isolated_edit = second_narrowed_registry["edit_file"].execute(
            ToolCall(
                id="narrowed-isolated-edit",
                name="edit_file",
                arguments={
                    "path": narrowed_isolated_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        if narrowed_isolated_edit.ok or narrowed_isolated_edit.error != not_read_error:
            fail(f"read ledger leaked between narrowed registries: {narrowed_isolated_edit!r}")
        if ToolResult(tool_call_id="payload-default", ok=True).payload is not None:
            fail("existing ToolResult callers gained a non-empty payload")
