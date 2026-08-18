#!/usr/bin/env python3
"""Smoke test for the orchestra_api runtime, using FakeModelProvider only.

Verifies:
  - a full ApiAgent run through runner.run_task(): a tool-call turn
    (read_file) followed by a final-answer turn
  - the allow path: read_file/list_files/write_file all succeed inside a
    temp dir that is both repo_root and the explicit allowed write scope
  - the deny path: write outside the allowed write scope, a
    forbidden-pattern path (.env), a `..` path-traversal attempt, run_shell
    denied by default, and an always-deny command (rm) denied even when
    explicitly allowlisted

No network calls anywhere in this script -- FakeModelProvider is the only
ModelProvider used.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.models import Message, ModelResponse, Role, ToolCall  # noqa: E402
from orchestra_api.permissions import PermissionPolicy  # noqa: E402
from orchestra_api.providers.fake import FakeModelProvider  # noqa: E402
from orchestra_api.runner import run_task, standard_tool_registry  # noqa: E402


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "existing.txt").write_text("hello from disk")
        (root / ".env").write_text("SECRET=do-not-read-me")

        # repo_root and allowed write scope are the same temp dir here.
        policy = PermissionPolicy(repo_root=root, allowed_write_scope=[root])

        # -- full agent run: tool-call turn (read_file) then final answer --
        tool_turn = ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "existing.txt"})],
            )
        )
        final_turn = ModelResponse(message=Message(role=Role.ASSISTANT, content="task complete"))
        provider = FakeModelProvider(responses=[tool_turn, final_turn])

        result = run_task(provider, policy, "read existing.txt then finish")
        if result.stopped_reason != "final_response":
            fail(f"expected stopped_reason='final_response', got {result.stopped_reason!r}")
        if result.final_response.message.content != "task complete":
            fail("final response content mismatch")
        tool_messages = [m for m in result.messages if m.role == Role.TOOL]
        if not tool_messages or not tool_messages[0].tool_result.ok:
            fail("expected the read_file tool call to succeed")
        if tool_messages[0].tool_result.content != "hello from disk":
            fail("read_file returned unexpected content")
        ok("full ApiAgent run (tool-call turn -> final answer) via FakeModelProvider")

        # -- allow path: read/list/write inside repo_root + allowed_write_scope --
        tools = standard_tool_registry()

        r = tools["read_file"].execute(ToolCall(id="a1", name="read_file", arguments={"path": "existing.txt"}), policy)
        if not r.ok:
            fail(f"read_file should be allowed inside repo_root: {r.error}")
        ok("allow: read_file inside repo_root")

        r = tools["list_files"].execute(ToolCall(id="a2", name="list_files", arguments={"path": "."}), policy)
        if not r.ok or "existing.txt" not in r.content:
            fail(f"list_files should be allowed and show existing.txt: {r}")
        ok("allow: list_files inside repo_root")

        r = tools["write_file"].execute(
            ToolCall(id="a3", name="write_file", arguments={"path": "new.txt", "content": "written by smoke test"}),
            policy,
        )
        if not r.ok or not (root / "new.txt").exists():
            fail(f"write_file should be allowed inside allowed_write_scope: {r}")
        ok("allow: write_file inside allowed_write_scope")

        # -- deny path: write outside the allowed write scope --
        no_write_policy = PermissionPolicy(repo_root=root)  # allowed_write_scope defaults to empty
        r = tools["write_file"].execute(
            ToolCall(id="d1", name="write_file", arguments={"path": "should_not_exist.txt", "content": "x"}),
            no_write_policy,
        )
        if r.ok or (root / "should_not_exist.txt").exists():
            fail("write_file should be denied when allowed_write_scope is empty")
        ok("deny: write_file outside the explicit allowed write scope")

        # -- deny path: forbidden pattern (.env) --
        r = tools["read_file"].execute(ToolCall(id="d2", name="read_file", arguments={"path": ".env"}), policy)
        if r.ok:
            fail("read_file should deny a forbidden-pattern path (.env)")
        ok("deny: read_file on a forbidden-pattern path (.env)")

        # -- deny path: .. path traversal --
        r = tools["read_file"].execute(
            ToolCall(id="d3", name="read_file", arguments={"path": "../outside.txt"}), policy
        )
        if r.ok:
            fail("read_file should deny a .. path-traversal attempt")
        ok("deny: read_file on a .. path-traversal attempt")

        # -- deny path: run_shell disabled by default --
        r = tools["run_shell"].execute(
            ToolCall(id="d4", name="run_shell", arguments={"argv": ["echo", "hi"]}), policy
        )
        if r.ok:
            fail("run_shell should be denied by default")
        ok("deny: run_shell disabled by default")

        # -- deny path: always-deny command wins even when explicitly allowlisted --
        risky_policy = PermissionPolicy(
            repo_root=root, shell_enabled=True, shell_allowlist=[("rm",)]
        )
        r = tools["run_shell"].execute(
            ToolCall(id="d5", name="run_shell", arguments={"argv": ["rm", "-rf", "existing.txt"]}), risky_policy
        )
        if r.ok or not (root / "existing.txt").exists():
            fail("run_shell must deny 'rm' even when explicitly allowlisted")
        ok("deny: run_shell always-deny command ('rm') overrides an explicit allowlist")

        print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
