"""Workspace-backed checks for read file."""

from __future__ import annotations

import unittest.mock as mock
from orchestra_api.models import ToolCall
import orchestra_api.tools.filesystem as filesystem_tools
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


@check("read_file.ranges_and_limits")
def check_read_file_ranges_and_limits() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        read_fixture = root / "read-fixture.txt"
        read_fixture.write_text("a\nb\nc")
        full_read = tools["read_file"].execute(
            ToolCall(id="read-full", name="read_file", arguments={"path": "read-fixture.txt"}),
            policy,
        )
        partial_read = tools["read_file"].execute(
            ToolCall(
                id="read-partial",
                name="read_file",
                arguments={"path": "read-fixture.txt", "offset": 2, "limit": 1},
            ),
            policy,
        )
        end_read = tools["read_file"].execute(
            ToolCall(
                id="read-end",
                name="read_file",
                arguments={"path": "read-fixture.txt", "offset": 3},
            ),
            policy,
        )
        if full_read.content != "1\ta\n2\tb\n3\tc":
            fail(f"full read gained a marker or wrong line numbers: {full_read!r}")
        if partial_read.content != "2\tb\n[lines 2-2; more follow, pass offset=3]":
            fail(f"partial read marker changed: {partial_read!r}")
        if end_read.content != "3\tc\n[lines 3-3; end of file]":
            fail(f"end read marker changed: {end_read!r}")
        past_read = tools["read_file"].execute(
            ToolCall(
                id="read-past-end",
                name="read_file",
                arguments={"path": "read-fixture.txt", "offset": 999999},
            ),
            policy,
        )
        if past_read.content != "[no lines at offset 999999; file has 3 lines]":
            fail(f"past-end read returned an ambiguous result: {past_read!r}")
        (root / "trailing-newline.txt").write_text("a\nb\nc\n")
        trailing_read = tools["read_file"].execute(
            ToolCall(
                id="read-trailing",
                name="read_file",
                arguments={"path": "trailing-newline.txt"},
            ),
            policy,
        )
        if trailing_read.content != "1\ta\n2\tb\n3\tc":
            fail(f"trailing newline produced an empty numbered line: {trailing_read!r}")

        oversized_path = root / "oversized-read.txt"
        oversized_path.write_bytes(b"x\n" * (filesystem_tools.MAX_READ_BYTES // 2 + 1))
        refused = tools["read_file"].execute(
            ToolCall(id="read-byte-cap", name="read_file", arguments={"path": oversized_path.name}),
            policy,
        )
        expected_byte_error = (
            f"file exceeds {filesystem_tools.MAX_READ_BYTES} byte read limit; "
            "read part of it with offset and limit"
        )
        if refused.ok or refused.error != expected_byte_error:
            fail(f"unranged oversized read was not refused: {refused!r}")
        ranged = tools["read_file"].execute(
            ToolCall(
                id="read-byte-range",
                name="read_file",
                arguments={"path": oversized_path.name, "offset": 2, "limit": 1},
            ),
            policy,
        )
        if not ranged.ok or not ranged.content.startswith("2\tx\n[lines 2-2; more follow"):
            fail(f"explicit range of oversized file did not succeed: {ranged!r}")
        (root / "lookahead-byte-cap.txt").write_text("12345\n67890\nabcde\n")
        with mock.patch.object(filesystem_tools, "MAX_READ_BYTES", 12):
            lookahead_allowed = tools["read_file"].execute(
                ToolCall(
                    id="read-lookahead-byte-cap",
                    name="read_file",
                    arguments={"path": "lookahead-byte-cap.txt", "limit": 2},
                ),
                policy,
            )
        if lookahead_allowed.content != (
            "1\t12345\n2\t67890\n[lines 1-2; more follow, pass offset=3]"
        ):
            fail(f"read_file charged its lookahead line to the byte cap: {lookahead_allowed!r}")
        (root / "stream-byte-cap.txt").write_text("12345\n67890\n")
        for arguments in (
            {"path": "stream-byte-cap.txt", "offset": 1, "limit": 0},
            {"path": "stream-byte-cap.txt", "limit": 2000},
        ):
            with mock.patch.object(filesystem_tools, "MAX_READ_BYTES", 10):
                stream_refused = tools["read_file"].execute(
                    ToolCall(
                        id="read-stream-byte-cap",
                        name="read_file",
                        arguments=arguments,
                    ),
                    policy,
                )
            if stream_refused.ok or stream_refused.error != (
                "selected range exceeds 10 byte read limit; narrow it with offset and limit"
            ):
                fail(f"streamed range exceeded its byte accumulator: {stream_refused!r}")
        (root / "single-line-byte-cap.txt").write_text("abcdefghij\n")
        with mock.patch.object(filesystem_tools, "MAX_READ_BYTES", 4):
            single_line_byte_refused = tools["read_file"].execute(
                ToolCall(
                    id="read-single-line-byte-cap",
                    name="read_file",
                    arguments={"path": "single-line-byte-cap.txt", "limit": 1},
                ),
                policy,
            )
        if single_line_byte_refused.ok or single_line_byte_refused.error != (
            "line 1 is 11 characters, over the 4 byte read limit; "
            "search the file with grep instead"
        ):
            fail(f"single-line byte refusal did not name grep: {single_line_byte_refused!r}")
        (root / "later-line-byte-cap.txt").write_text("ab\nabcdefghij\n")
        with mock.patch.object(filesystem_tools, "MAX_READ_BYTES", 4):
            later_line_byte_refused = tools["read_file"].execute(
                ToolCall(
                    id="read-later-line-byte-cap",
                    name="read_file",
                    arguments={"path": "later-line-byte-cap.txt", "offset": 1, "limit": 2},
                ),
                policy,
            )
            narrowed_before_long_line = tools["read_file"].execute(
                ToolCall(
                    id="read-before-later-line-byte-cap",
                    name="read_file",
                    arguments={"path": "later-line-byte-cap.txt", "offset": 1, "limit": 1},
                ),
                policy,
            )
            direct_later_line_refused = tools["read_file"].execute(
                ToolCall(
                    id="read-direct-later-line-byte-cap",
                    name="read_file",
                    arguments={"path": "later-line-byte-cap.txt", "offset": 2, "limit": 1},
                ),
                policy,
            )
        if later_line_byte_refused.ok or later_line_byte_refused.error != (
            "selected range exceeds 4 byte read limit; narrow it with offset and limit"
        ):
            fail(f"later-line byte refusal did not recommend narrowing: {later_line_byte_refused!r}")
        if narrowed_before_long_line.content != (
            "1\tab\n[lines 1-1; more follow, pass offset=2]"
        ):
            fail(f"later-line byte refusal named an ineffective remedy: {narrowed_before_long_line!r}")
        if direct_later_line_refused.ok or direct_later_line_refused.error != (
            "line 2 is 11 characters, over the 4 byte read limit; "
            "search the file with grep instead"
        ):
            fail(f"direct oversized-line read did not name its absolute line: {direct_later_line_refused!r}")
        (root / "token-cap.txt").write_text("abcdefghij" * 8)
        with mock.patch.object(filesystem_tools, "MAX_READ_TOKENS", 2):
            token_refused = tools["read_file"].execute(
                ToolCall(
                    id="read-token-cap",
                    name="read_file",
                    arguments={"path": "token-cap.txt", "offset": 1, "limit": 1},
                ),
                policy,
            )
        if token_refused.ok or token_refused.error != (
            "line 1 is about 21 tokens, over the 2 token limit; "
            "search the file with grep instead"
        ):
            fail(f"single-line token refusal did not name grep: {token_refused!r}")
        (root / "many-lines-token-cap.txt").write_text("a\na\na")
        with mock.patch.object(filesystem_tools, "MAX_READ_TOKENS", 2):
            many_lines_refused = tools["read_file"].execute(
                ToolCall(
                    id="read-many-lines-token-cap",
                    name="read_file",
                    arguments={"path": "many-lines-token-cap.txt", "offset": 1, "limit": 3},
                ),
                policy,
            )
        if many_lines_refused.ok or many_lines_refused.error != (
            "selected range is about 3 tokens, over the 2 token limit; "
            "narrow it with offset and limit"
        ):
            fail(f"multi-line token refusal lost its narrowing remedy: {many_lines_refused!r}")
