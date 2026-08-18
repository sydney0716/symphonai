#!/usr/bin/env python3
"""Smoke test for the SubagentProvider contract, using FakeProvider.

Verifies:
  - FakeProvider can be registered in and retrieved from the registry
  - spawn() returns an AgentRunResult with a session_id and MESSAGE + DONE events
  - resume() continues the same session_id and extends the event history
  - resume() on an unknown session_id fails gracefully (ok=False, error set)

No external commands, no network calls -- everything here is in-memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_agents.adapters.fake import FakeProvider  # noqa: E402
from orchestra_agents.models import AgentTask, EventKind  # noqa: E402
from orchestra_agents.registry import (  # noqa: E402
    clear_registry,
    get_provider,
    register_provider,
)


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def main() -> None:
    clear_registry()
    register_provider(FakeProvider())
    ok("registered FakeProvider")

    provider = get_provider("fake")
    if provider.name != "fake":
        fail(f"expected provider.name == 'fake', got {provider.name!r}")
    ok("retrieved FakeProvider from registry by name")

    task = AgentTask(task_id="smoke-1", prompt="say hello")
    result = provider.spawn(task)
    if not result.ok:
        fail(f"spawn() returned ok=False: {result.error}")
    if not result.session_id:
        fail("spawn() returned an empty session_id")
    kinds = [e.kind for e in result.events]
    if EventKind.MESSAGE not in kinds or EventKind.DONE not in kinds:
        fail(f"spawn() events missing MESSAGE/DONE: {kinds}")
    ok(f"spawn() returned session_id={result.session_id!r} with {len(result.events)} events")

    follow_up = provider.resume(result.session_id, "say it again")
    if not follow_up.ok:
        fail(f"resume() returned ok=False: {follow_up.error}")
    if follow_up.session_id != result.session_id:
        fail("resume() changed session_id unexpectedly for FakeProvider")
    if len(follow_up.events) <= len(result.events):
        fail("resume() did not extend the event history")
    ok(f"resume() extended session to {len(follow_up.events)} events")

    unknown = provider.resume("not-a-real-session", "hello?")
    if unknown.ok:
        fail("resume() on an unknown session_id unexpectedly succeeded")
    if not unknown.error:
        fail("resume() on an unknown session_id did not set .error")
    ok("resume() on an unknown session_id failed gracefully as expected")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
