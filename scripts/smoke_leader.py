#!/usr/bin/env python3
"""Smoke test for the Leader / DispatchSubagentTool, using FakeModelProvider only.

Verifies:
  - a full Leader.run(): a dispatch_subagent tool-call turn followed by a
    final answer, stopped_reason == "final_response"
  - dispatch with a new subagent_name creates a subagent
  - dispatch with the same subagent_name reuses it (same object identity,
    message history grows across calls)
  - a different subagent_name creates an isolated second pool entry
  - max_subagents is enforced once the limit is reached
  - the on_status callback fires the expected ordered sequence of
    (label, status) events for a run that dispatches one subagent
  - Leader.chat() persists conversation across calls -- a second chat()
    call's leader_messages include the first call's exchange

No network calls anywhere in this script -- FakeModelProvider is the only
ModelProvider used, for both the leader and every subagent.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.leader import DispatchSubagentTool, Leader, LeaderConfig  # noqa: E402
from orchestra_api.models import Message, ModelResponse, Role, ToolCall  # noqa: E402
from orchestra_api.permissions import PermissionPolicy  # noqa: E402
from orchestra_api.providers.fake import FakeModelProvider  # noqa: E402


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # -- full Leader.run(): dispatch then final answer --
        subagent_provider = FakeModelProvider(
            responses=[ModelResponse(message=Message(role=Role.ASSISTANT, content="the sky is blue"))]
        )
        leader_provider = FakeModelProvider(
            responses=[
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="lc1",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "researcher", "task": "why is the sky blue?"},
                            )
                        ],
                    )
                ),
                ModelResponse(message=Message(role=Role.ASSISTANT, content="Answer: the sky is blue.")),
            ]
        )
        config = LeaderConfig(leader_provider=leader_provider, subagent_provider=subagent_provider, repo_root=str(root))
        leader = Leader(config)
        result = leader.run("why is the sky blue?")

        if result.stopped_reason != "final_response":
            fail(f"expected stopped_reason='final_response', got {result.stopped_reason!r}")
        if "researcher" not in result.subagents:
            fail("expected a 'researcher' subagent in the pool")
        if not result.final_answer:
            fail("expected a non-empty final answer")
        ok(f"Leader.run() dispatched a subagent and reached a final answer: {result.final_answer!r}")

        # -- create / reuse / isolate / max_subagents, exercised directly on the tool --
        policy = PermissionPolicy(repo_root=root)
        pool_provider = FakeModelProvider(
            responses=[ModelResponse(message=Message(role=Role.ASSISTANT, content="sub reply"))]
        )
        tool = DispatchSubagentTool(
            subagent_provider=pool_provider, subagent_policy=policy, max_subagents=2, subagent_max_turns=3
        )

        r1 = tool.execute(
            ToolCall(id="c1", name="dispatch_subagent", arguments={"subagent_name": "worker", "task": "task one"}),
            policy,
        )
        if not r1.ok:
            fail(f"expected first dispatch to succeed: {r1.error}")
        worker_record = tool.pool.get("worker")
        if worker_record is None:
            fail("expected a 'worker' entry in the pool after first dispatch")
        ok("dispatch with a new name creates a subagent")

        r2 = tool.execute(
            ToolCall(id="c2", name="dispatch_subagent", arguments={"subagent_name": "worker", "task": "task two"}),
            policy,
        )
        if not r2.ok:
            fail(f"expected second dispatch to succeed: {r2.error}")
        if tool.pool["worker"] is not worker_record:
            fail("expected reuse to keep the same SubagentRecord object identity")
        if len(worker_record.messages) < 4:
            fail(f"expected message history to grow across both calls, got {len(worker_record.messages)}")
        ok("dispatch with the same name reuses the subagent (same identity, growing history)")

        r3 = tool.execute(
            ToolCall(id="c3", name="dispatch_subagent", arguments={"subagent_name": "helper", "task": "task three"}),
            policy,
        )
        if not r3.ok or tool.pool.get("helper") is worker_record:
            fail("expected a different name to create an isolated second subagent")
        ok("a different name creates an isolated second pool entry")

        r4 = tool.execute(
            ToolCall(id="c4", name="dispatch_subagent", arguments={"subagent_name": "third", "task": "x"}), policy
        )
        if r4.ok or len(tool.pool) != 2:
            fail("expected max_subagents to be enforced once the limit is reached")
        ok("max_subagents is enforced")

        # -- on_status callback fires the expected ordered sequence --
        events: list[tuple[str, str]] = []
        status_subagent_provider = FakeModelProvider(
            responses=[ModelResponse(message=Message(role=Role.ASSISTANT, content="the sky is blue"))]
        )
        status_leader_provider = FakeModelProvider(
            responses=[
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="lc1",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "researcher", "task": "why blue?"},
                            )
                        ],
                    )
                ),
                ModelResponse(message=Message(role=Role.ASSISTANT, content="Answer: blue.")),
            ]
        )
        status_config = LeaderConfig(
            leader_provider=status_leader_provider,
            subagent_provider=status_subagent_provider,
            repo_root=str(root),
            on_status=lambda label, status: events.append((label, status)),
        )
        Leader(status_config).run("why is the sky blue?")
        expected_events = [
            ("leader", "working"),
            ("researcher", "pending"),
            ("researcher", "working"),
            ("researcher", "done"),
            ("leader", "done"),
        ]
        if events != expected_events:
            fail(f"expected status event sequence {expected_events}, got {events}")
        ok(f"on_status fires the expected sequence: {events}")

        # -- Leader.chat() persists conversation across calls --
        chat_provider = FakeModelProvider(
            responses=[
                ModelResponse(message=Message(role=Role.ASSISTANT, content="hi there")),
                ModelResponse(message=Message(role=Role.ASSISTANT, content="yes, I remember you said hello")),
            ]
        )
        chat_leader = Leader(
            LeaderConfig(leader_provider=chat_provider, subagent_provider=FakeModelProvider(), repo_root=str(root))
        )
        first = chat_leader.chat("hello")
        if len(first.leader_messages) != 2:
            fail(f"expected 2 messages after first chat() call, got {len(first.leader_messages)}")
        second = chat_leader.chat("do you remember what I said?")
        if len(second.leader_messages) != 4:
            fail(f"expected 4 messages after second chat() call (history carried forward), got {len(second.leader_messages)}")
        contents = [m.content for m in second.leader_messages]
        if "hello" not in contents:
            fail("expected the first call's user message to still be present in the second call's context")
        ok("Leader.chat() persists conversation across calls")

        print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
