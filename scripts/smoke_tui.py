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
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.models import Message, ModelRequest, ModelResponse, Role, ToolCall  # noqa: E402
from orchestra_api.providers.base import ModelProvider, ProviderError  # noqa: E402
from orchestra_api.providers.fake import FakeModelProvider  # noqa: E402
from orchestra_tui.app import OrchestraTuiApp  # noqa: E402
from orchestra_tui.picker import ProviderPickerScreen  # noqa: E402
from textual.widgets import Button, Checkbox, Input, Select, Static  # noqa: E402


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
        confirm_real_providers=False,
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


async def wait_for_picker(pilot) -> ProviderPickerScreen:  # noqa: ANN001
    await wait_until(
        pilot,
        lambda: isinstance(pilot.app.screen, ProviderPickerScreen),
        "provider picker screen",
    )
    screen = pilot.app.screen
    if not isinstance(screen, ProviderPickerScreen):
        fail(f"expected picker screen, found {type(screen).__name__}")
    return screen


def set_select(screen: ProviderPickerScreen, widget_id: str, value: str) -> None:
    screen.query_one(f"#{widget_id}", Select).value = value


async def smoke_picker_success_case(root: Path) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_list_models(provider: ModelProvider, *, include_all: bool = False) -> list[str]:
        calls.append((provider.name, include_all))
        if provider.name == "openai":
            return ["gpt-picked-leader", "gpt-other"]
        if provider.name == "anthropic":
            return ["claude-picked-subagent", "claude-other"]
        return [f"{provider.name}-model"]

    app = OrchestraTuiApp(repo_root=root, confirm_real_providers=False)
    with mock.patch("orchestra_tui.picker.list_models", side_effect=fake_list_models):
        async with app.run_test(size=(82, 28)) as pilot:
            screen = await wait_for_picker(pilot)

            set_select(screen, "leader-provider", "openai")
            await wait_until(
                pilot,
                lambda: not screen.query_one("#leader-model", Select).disabled,
                "leader models loaded",
            )
            set_select(screen, "leader-model", "gpt-picked-leader")
            await wait_until(
                pilot,
                lambda: screen.query_one("#leader-manual-model", Input).value == "gpt-picked-leader",
                "leader model copied into manual entry",
            )

            set_select(screen, "subagent-provider", "anthropic")
            await wait_until(
                pilot,
                lambda: not screen.query_one("#subagent-model", Select).disabled,
                "subagent models loaded",
            )
            set_select(screen, "subagent-model", "claude-picked-subagent")
            await wait_until(
                pilot,
                lambda: screen.query_one("#subagent-manual-model", Input).value == "claude-picked-subagent",
                "subagent model copied into manual entry",
            )

            screen.query_one("#start-chat", Button).press()
            await wait_until(
                pilot,
                lambda: not pilot.app.query_one("#message-input", Input).disabled,
                "chat input enabled after picker",
            )

            config = app._leader._config if app._leader is not None else None  # noqa: SLF001
            if config is None:
                fail("picker did not configure a leader")
            if getattr(config.leader_provider, "model", None) != "gpt-picked-leader":
                fail("leader provider did not receive selected model")
            if getattr(config.subagent_provider, "model", None) != "claude-picked-subagent":
                fail("subagent provider did not receive selected model")
            if ("openai", False) not in calls or ("anthropic", False) not in calls:
                fail(f"picker did not list selected providers: {calls!r}")
            ok("picker selects provider/model pairs and hands them to the app")


async def smoke_picker_fallback_case(root: Path) -> None:
    def failing_list_models(provider: ModelProvider, *, include_all: bool = False) -> list[str]:
        raise ProviderError("scripted listing failure")

    app = OrchestraTuiApp(repo_root=root, confirm_real_providers=False)
    with mock.patch("orchestra_tui.picker.list_models", side_effect=failing_list_models):
        async with app.run_test(size=(82, 28)) as pilot:
            screen = await wait_for_picker(pilot)
            set_select(screen, "leader-provider", "openai")
            await wait_until(
                pilot,
                lambda: "Enter a model manually"
                in str(screen.query_one("#leader-model-status", Static).content),
                "leader manual fallback",
            )
            manual = screen.query_one("#leader-manual-model", Input)
            if manual.disabled:
                fail("manual model entry stayed disabled after listing failure")
            if manual.value != "gpt-5.4-mini":
                fail(f"manual model entry was not seeded with provider default: {manual.value!r}")
            if not screen.query_one("#leader-model", Select).disabled:
                fail("model select stayed enabled after listing failure")

            set_select(screen, "subagent-provider", "anthropic")
            await wait_until(
                pilot,
                lambda: "Enter a model manually"
                in str(screen.query_one("#subagent-model-status", Static).content),
                "subagent manual fallback",
            )
            screen.query_one("#start-chat", Button).press()
            await wait_until(
                pilot,
                lambda: not pilot.app.query_one("#message-input", Input).disabled,
                "chat input enabled after fallback picker",
            )
            ok("listing failures offer seeded manual entry and keep the app usable")


async def smoke_picker_include_all_case(root: Path) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_list_models(provider: ModelProvider, *, include_all: bool = False) -> list[str]:
        calls.append((provider.name, include_all))
        if include_all:
            return ["visible-text-model", "hidden-image-model"]
        return ["visible-text-model"]

    app = OrchestraTuiApp(repo_root=root, confirm_real_providers=False)
    with mock.patch("orchestra_tui.picker.list_models", side_effect=fake_list_models):
        async with app.run_test(size=(82, 28)) as pilot:
            screen = await wait_for_picker(pilot)
            set_select(screen, "leader-provider", "openai")
            await wait_until(
                pilot,
                lambda: not screen.query_one("#leader-model", Select).disabled,
                "initial filtered model list",
            )
            screen.query_one("#leader-include-all", Checkbox).value = True
            await wait_until(
                pilot,
                lambda: ("openai", True) in calls,
                "include_all model listing call",
            )
            await wait_until(
                pilot,
                lambda: "Showing 2 openai all models"
                in str(screen.query_one("#leader-model-status", Static).content),
                "all models status",
            )
            set_select(screen, "leader-model", "hidden-image-model")
            await wait_until(
                pilot,
                lambda: screen.query_one("#leader-manual-model", Input).value == "hidden-image-model",
                "include_all-only model selectable",
            )
            ok("include_all toggle reloads and exposes unfiltered models")


async def main_async() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        await smoke_success_case(root)
        await smoke_error_case(root)
        await smoke_small_terminal_case(root)
        await smoke_picker_success_case(root)
        await smoke_picker_fallback_case(root)
        await smoke_picker_include_all_case(root)


def main() -> int:
    asyncio.run(main_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
