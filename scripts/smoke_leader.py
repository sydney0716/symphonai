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
  - regression check: a real AnthropicProvider used as leader_provider
    actually includes the dispatch_subagent tool definition in its
    outgoing request body (via mocked urllib.request.urlopen) -- guards
    against ApiAgent silently never telling a real model any tool exists
  - regression check: a real subagent's own outgoing request (behind a
    real leader dispatching to it) includes schemas for all four standard
    tools (read_file/write_file/list_files/run_shell)

No real network call is ever made in this script. FakeModelProvider is
used for every check except the one regression check above, which uses a
real AnthropicProvider purely to exercise its real request-building code
against a mocked HTTP layer.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.leader import DispatchSubagentTool, Leader, LeaderConfig  # noqa: E402
from orchestra_api.models import Message, ModelResponse, Role, ToolCall  # noqa: E402
from orchestra_api.permissions import PermissionPolicy  # noqa: E402
from orchestra_api.providers.anthropic_provider import API_KEY_ENV_VAR, AnthropicProvider  # noqa: E402
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

        # -- regression: a real leader provider's outgoing request must
        # actually include the dispatch_subagent tool definition --
        os.environ[API_KEY_ENV_VAR] = "sk-ant-fake-test-key-do-not-use"
        captured: dict = {}

        class _FakeHttpResponse:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def read(self) -> bytes:
                return self._body

            def __enter__(self) -> "_FakeHttpResponse":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        def _fake_urlopen(request, timeout=None):  # noqa: ANN001
            captured["body"] = json.loads(request.data.decode("utf-8"))
            payload = json.dumps(
                {
                    "content": [{"type": "text", "text": "no tool needed"}],
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                    "stop_reason": "end_turn",
                }
            ).encode("utf-8")
            return _FakeHttpResponse(payload)

        real_leader = Leader(
            LeaderConfig(
                leader_provider=AnthropicProvider(),
                subagent_provider=AnthropicProvider(),
                repo_root=str(root),
            )
        )
        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            real_leader.run("hello")
        del os.environ[API_KEY_ENV_VAR]

        sent_tools = captured.get("body", {}).get("tools")
        if not sent_tools or sent_tools[0].get("name") != "dispatch_subagent":
            fail(f"expected outgoing request to include the dispatch_subagent tool, got tools={sent_tools!r}")
        ok("real leader provider's outgoing request includes the dispatch_subagent tool definition")

        # -- regression: a real SUBAGENT's outgoing request must include
        # schemas for all four standard tools (read_file/write_file/
        # list_files/run_shell), not just the leader's own tool --
        os.environ[API_KEY_ENV_VAR] = "sk-ant-fake-test-key-do-not-use"
        subagent_requests: list[dict] = []
        call_count = [0]

        def _fake_urlopen_dispatch(request, timeout=None):  # noqa: ANN001
            call_count[0] += 1
            body = json.loads(request.data.decode("utf-8"))
            if call_count[0] == 1:
                # leader's turn: decide to dispatch to a subagent
                payload = {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "c1",
                            "name": "dispatch_subagent",
                            "input": {"subagent_name": "researcher", "task": "read a.txt"},
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "tool_use",
                }
            elif call_count[0] == 2:
                # the subagent's own turn -- this is the request we care about
                subagent_requests.append(body)
                payload = {
                    "content": [{"type": "text", "text": "done"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "end_turn",
                }
            else:
                # leader's final answer
                payload = {
                    "content": [{"type": "text", "text": "final answer"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "end_turn",
                }
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

        dispatch_leader = Leader(
            LeaderConfig(
                leader_provider=AnthropicProvider(),
                subagent_provider=AnthropicProvider(),
                repo_root=str(root),
            )
        )
        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen_dispatch):
            dispatch_leader.run("please read a.txt via researcher")
        del os.environ[API_KEY_ENV_VAR]

        if not subagent_requests:
            fail("expected the subagent's own request to have been captured")
        sent_tool_names = {t.get("name") for t in subagent_requests[0].get("tools", [])}
        expected_tool_names = {"read_file", "write_file", "list_files", "run_shell"}
        if sent_tool_names != expected_tool_names:
            fail(f"expected subagent request to include {expected_tool_names}, got {sent_tool_names!r}")
        ok("real subagent's outgoing request includes all four standard tool schemas")

        print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
