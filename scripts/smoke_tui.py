#!/usr/bin/env python3
"""Headless smoke test for the optional Textual Leader TUI.

Uses FakeModelProvider and a local failing provider only; no real network
request is made by this script.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.models import Message, ModelRequest, ModelResponse, Role, ToolCall  # noqa: E402
from orchestra_api.providers.base import ModelProvider, ProviderError  # noqa: E402
from orchestra_api.providers.fake import FakeModelProvider  # noqa: E402
from orchestra_tui.app import OrchestraTuiApp  # noqa: E402
from textual.widgets import Input  # noqa: E402


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"OK:   {message}")


class FailingProvider(ModelProvider):
    @property
    def name(self) -> str:
        return "failing"

    @property
    def wire_format(self) -> int:
        return 4

    def create_response(self, request: ModelRequest) -> ModelResponse:
        raise ProviderError("scripted provider failure")


async def wait_until(pilot, predicate, label: str, timeout: float = 5.0) -> None:  # noqa: ANN001
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            fail(f"timed out waiting for {label}")
        await pilot.pause(0.05)


async def submit_message(pilot, value: str) -> None:  # noqa: ANN001
    input_widget = pilot.app.query_one("#message-input", Input)
    input_widget.focus()
    input_widget.value = value
    await pilot.press("enter")


async def smoke_success_case(root: Path) -> None:
    subagent_provider = FakeModelProvider(
        responses=[ModelResponse(message=Message(role=Role.ASSISTANT, content="subagent checked it"))]
    )
    leader_provider = FakeModelProvider(
        responses=[
            ModelResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="dispatch-1",
                            name="dispatch_subagent",
                            arguments={"subagent_name": "researcher", "task": "check the answer"},
                        )
                    ],
                )
            ),
            ModelResponse(message=Message(role=Role.ASSISTANT, content="assistant reply from leader")),
        ]
    )
    app = OrchestraTuiApp(
        leader_provider=leader_provider,
        subagent_provider=subagent_provider,
        repo_root=root,
    )
    async with app.run_test(size=(64, 18)) as pilot:
        if not app.is_mounted:
            fail("app did not mount")
        ok("app mounts headlessly")

        await submit_message(pilot, "hello")
        await wait_until(
            pilot,
            lambda: ("leader", "assistant reply from leader") in app.chat_entries,
            "assistant reply in chat log",
        )
        ok("submitting a message produces an assistant reply in the chat log")

        expected_events = {
            ("leader", "working"),
            ("researcher", "pending"),
            ("researcher", "working"),
            ("researcher", "done"),
            ("leader", "done"),
        }
        missing = expected_events.difference(app.status_entries)
        if missing:
            fail(f"missing status events: {sorted(missing)!r}")
        if "researcher: done" not in app.visible_status_text:
            fail(f"status panel did not show researcher done state: {app.visible_status_text!r}")
        ok("on_status events reach the subagent status panel")

        if app.query_one("#message-input", Input).disabled:
            fail("input stayed disabled after successful turn")
        ok("input re-enables after a successful turn")


async def smoke_error_case(root: Path) -> None:
    app = OrchestraTuiApp(
        leader_provider=FailingProvider(),
        subagent_provider=FakeModelProvider(),
        repo_root=root,
    )
    async with app.run_test(size=(50, 12)) as pilot:
        await submit_message(pilot, "fail please")
        await wait_until(
            pilot,
            lambda: any(
                label == "error"
                and "Provider error: scripted provider failure" in message
                for label, message in app.chat_entries
            ),
            "visible provider error line",
        )
        if app.query_one("#message-input", Input).disabled:
            fail("input stayed disabled after provider error")
        ok("provider errors render visibly and leave the app usable")


async def smoke_small_terminal_case(root: Path) -> None:
    app = OrchestraTuiApp(
        leader_provider=FakeModelProvider(),
        subagent_provider=FakeModelProvider(),
        repo_root=root,
    )
    async with app.run_test(size=(24, 8)):
        if not app.is_mounted:
            fail("app did not mount at a small terminal size")
        ok("app mounts at a small terminal size")


async def main_async() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        await smoke_success_case(root)
        await smoke_error_case(root)
        await smoke_small_terminal_case(root)


def main() -> int:
    asyncio.run(main_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
