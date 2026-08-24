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
from orchestra_tui.app import OrchestraTuiApp, ToolApprovalScreen  # noqa: E402
from orchestra_tui.picker import (  # noqa: E402
    ProviderConfirmationScreen,
    ProviderPickerScreen,
    build_picker_provider,
)
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


async def smoke_terminal_status_case(root: Path) -> None:
    app = OrchestraTuiApp(
        leader_provider=FakeModelProvider(),
        subagent_provider=FakeModelProvider(),
        repo_root=root,
    )
    async with app.run_test(size=(64, 20)):
        app._update_agent_status("leader", "failed")  # noqa: SLF001
        app._update_agent_status("worker-1", "exhausted")  # noqa: SLF001
        panel_text = app.query_one("#status-panel", Static).render()
        if "leader: failed" not in panel_text.plain or "worker-1: exhausted" not in panel_text.plain:
            fail(f"terminal states were not rendered: {panel_text.plain!r}")
        failed_offset = panel_text.plain.index("failed")
        exhausted_offset = panel_text.plain.index("exhausted")
        failed_style = str(panel_text.get_style_at_offset(failed_offset))
        exhausted_style = str(panel_text.get_style_at_offset(exhausted_offset))
        if failed_style == exhausted_style or "red" not in failed_style or "magenta" not in exhausted_style:
            fail(
                "failed and exhausted statuses were not visually distinct: "
                f"failed={failed_style!r}, exhausted={exhausted_style!r}"
            )
        ok("failed and exhausted render as distinct, noticeable terminal states")

        for index in range(1, 6):
            app._update_agent_status(f"worker-{index}", "done")  # noqa: SLF001
        expected = {"leader", *(f"worker-{index}" for index in range(1, 6))}
        visible = {line.partition(":")[0] for line in app.visible_status_text.splitlines()}
        if expected != visible:
            fail(f"status panel did not contain all six agents: {app.visible_status_text!r}")
        if app.query_one("#status-panel", Static).content_region.height < 7:
            fail("status panel content area cannot display its title plus six agents")
        ok("status panel displays the leader plus five subagents")


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


async def smoke_confirmation_before_discovery_case(root: Path) -> None:
    discovery_calls: list[str] = []

    def fake_list_models(provider: ModelProvider, *, include_all: bool = False) -> list[str]:
        discovery_calls.append(provider.name)
        return [getattr(provider, "model", "model")]

    app = OrchestraTuiApp(repo_root=root)
    with mock.patch("orchestra_tui.picker.list_models", side_effect=fake_list_models):
        async with app.run_test(size=(82, 28)) as pilot:
            picker = await wait_for_picker(pilot)
            set_select(picker, "leader-provider", "openai")
            await wait_until(
                pilot,
                lambda: isinstance(pilot.app.screen, ProviderConfirmationScreen),
                "confirmation before discovery",
            )
            if discovery_calls:
                fail(f"model discovery ran before confirmation: {discovery_calls!r}")
            pilot.app.screen.query_one("#confirm-continue", Button).press()
            await wait_until(pilot, lambda: bool(discovery_calls), "discovery after confirmation")
            ok("real-provider confirmation precedes model discovery")


async def smoke_slash_commands_case(root: Path) -> None:
    app = OrchestraTuiApp(
        leader_provider=FakeModelProvider(),
        subagent_provider=FakeModelProvider(),
        repo_root=root,
        chat_token_budget=140,
        chat_recent_turns=1,
    )
    async with app.run_test(size=(72, 18)) as pilot:
        await submit_message(pilot, "/help")
        await wait_until(
            pilot,
            lambda: any(
                label == "system" and "/help" in message and "/compact" in message
                for label, message in app.chat_entries
            ),
            "slash help output",
        )
        ok("/help lists available slash commands")

        await submit_message(pilot, "/does-not-exist")
        await wait_until(
            pilot,
            lambda: any(
                label == "error"
                and "Unknown command '/does-not-exist'" in message
                and "/model" in message
                for label, message in app.chat_entries
            ),
            "unknown slash command output",
        )
        ok("unknown slash commands render a helpful error")

        await submit_message(pilot, "remember this")
        await wait_until(
            pilot,
            lambda: any(label == "leader" for label, _ in app.chat_entries),
            "leader reply before clear",
        )
        leader = app._leader  # noqa: SLF001
        if leader is None or not leader._chat_messages:  # noqa: SLF001
            fail("leader chat state was not populated before /clear")
        clear_subagents = mock.Mock(return_value=2)
        leader.clear_subagents = clear_subagents  # type: ignore[attr-defined]
        await submit_message(pilot, "/clear")
        await wait_until(
            pilot,
            lambda: app.chat_entries == [("system", "Chat cleared; 2 subagents cleared.")],
            "chat cleared report",
        )
        if leader._chat_messages:  # noqa: SLF001
            fail("/clear did not clear the leader conversation state")
        clear_subagents.assert_called_once_with()
        ok("/clear clears chat and subagents and reports the cleared count")

        leader._chat_messages = [  # noqa: SLF001
            Message(role=Role.SYSTEM, content="system prompt must stay"),
            Message(role=Role.USER, content="earliest user goal must stay"),
            Message(role=Role.ASSISTANT, content="old assistant detail " * 120),
            Message(role=Role.USER, content="old follow-up " * 120),
            Message(role=Role.ASSISTANT, content="old tool analysis " * 120),
            Message(role=Role.USER, content="latest user request must stay"),
        ]
        await submit_message(pilot, "/compact")
        await wait_until(
            pilot,
            lambda: any(
                label == "system" and "Compacted conversation" in message
                for label, message in app.chat_entries
            ),
            "manual compaction report",
        )
        if len(leader._chat_messages) >= 6:  # noqa: SLF001
            fail("/compact did not reduce the leader conversation state")
        ok("/compact compacts leader state and reports the result")


async def smoke_model_command_case(root: Path) -> None:
    app = OrchestraTuiApp(
        leader_provider=FakeModelProvider(),
        subagent_provider=FakeModelProvider(),
        repo_root=root,
    )
    async with app.run_test(size=(82, 28)) as pilot:
        await submit_message(pilot, "/model")
        screen = await wait_for_picker(pilot)
        if not isinstance(screen, ProviderPickerScreen):
            fail("/model did not open the provider picker")
        await pilot.press("escape")
        await wait_until(
            pilot,
            lambda: not isinstance(pilot.app.screen, ProviderPickerScreen),
            "picker cancellation back to chat",
        )
        if not app.is_running or app.query_one("#message-input", Input).disabled:
            fail("cancelling /model exited or disabled the existing chat")
        ok("cancelling the /model picker returns to the running chat")

        await submit_message(pilot, "/model")
        screen = await wait_for_picker(pilot)
        set_select(screen, "leader-provider", "openai")
        await wait_until(
            pilot,
            lambda: isinstance(pilot.app.screen, ProviderConfirmationScreen),
            "/model provider confirmation",
        )
        pilot.app.screen.query_one("#confirm-cancel", Button).press()
        await wait_until(
            pilot,
            lambda: not isinstance(
                pilot.app.screen, (ProviderPickerScreen, ProviderConfirmationScreen)
            ),
            "confirmation cancellation back to chat",
        )
        if not app.is_running or app.query_one("#message-input", Input).disabled:
            fail("cancelling /model confirmation exited or disabled the existing chat")
        ok("cancelling /model confirmation returns to the running chat")


async def smoke_model_preselection_and_divider_case(root: Path) -> None:
    leader_provider = build_picker_provider("openai", model="current-leader-model")
    subagent_provider = build_picker_provider("anthropic", model="current-subagent-model")

    def fake_list_models(provider: ModelProvider, *, include_all: bool = False) -> list[str]:
        return [getattr(provider, "model", ""), "replacement-model"]

    app = OrchestraTuiApp(
        leader_provider=leader_provider,
        subagent_provider=subagent_provider,
        repo_root=root,
        confirm_real_providers=False,
    )
    app.chat_entries.append(("leader", "old transcript"))
    with mock.patch("orchestra_tui.picker.list_models", side_effect=fake_list_models):
        async with app.run_test(size=(82, 28)) as pilot:
            await submit_message(pilot, "/model")
            screen = await wait_for_picker(pilot)
            await wait_until(
                pilot,
                lambda: (
                    screen.query_one("#leader-manual-model", Input).value
                    == "current-leader-model"
                    and screen.query_one("#subagent-manual-model", Input).value
                    == "current-subagent-model"
                ),
                "current provider/model preselection",
            )
            if screen.query_one("#leader-provider", Select).value != "openai":
                fail("current leader provider was not preselected")
            if screen.query_one("#subagent-provider", Select).value != "anthropic":
                fail("current subagent provider was not preselected")
            screen.query_one("#start-chat", Button).press()
            await wait_until(
                pilot,
                lambda: any("does not inherit" in message for _, message in app.chat_entries),
                "new model session divider",
            )
            if ("leader", "old transcript") not in app.chat_entries:
                fail("model switch unexpectedly removed the visible prior transcript")
            ok("/model preselects current choices and adds a no-context session divider")


async def smoke_exit_command_case(root: Path) -> None:
    app = OrchestraTuiApp(
        leader_provider=FakeModelProvider(),
        subagent_provider=FakeModelProvider(),
        repo_root=root,
    )
    exit_called: list[bool] = []
    with mock.patch.object(app, "exit", side_effect=lambda *args, **kwargs: exit_called.append(True)):
        async with app.run_test(size=(50, 12)) as pilot:
            await submit_message(pilot, "/exit")
            await wait_until(pilot, lambda: bool(exit_called), "exit command")
            ok("/exit requests a clean app quit")


async def smoke_approval_case(root: Path, *, approve: bool) -> None:
    target = "approved.txt" if approve else "denied.txt"
    subagent_provider = FakeModelProvider(
        responses=[
            ModelResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="write-1",
                            name="write_file",
                            arguments={"path": target, "content": "written from prompt mode"},
                        )
                    ],
                )
            ),
            ModelResponse(message=Message(role=Role.ASSISTANT, content="subagent finished")),
        ]
    )
    leader_provider = FakeModelProvider(
        responses=[
            ModelResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="dispatch-approval",
                            name="dispatch_subagent",
                            arguments={"subagent_name": "writer", "task": "write the file"},
                        )
                    ],
                )
            ),
            ModelResponse(message=Message(role=Role.ASSISTANT, content="leader saw tool result")),
        ]
    )
    app = OrchestraTuiApp(
        leader_provider=leader_provider,
        subagent_provider=subagent_provider,
        repo_root=root,
        permission_mode="prompt",
    )
    async with app.run_test(size=(76, 20)) as pilot:
        await submit_message(pilot, "please write")
        await wait_until(
            pilot,
            lambda: isinstance(pilot.app.screen, ToolApprovalScreen),
            "tool approval screen",
        )
        screen = pilot.app.screen
        if not isinstance(screen, ToolApprovalScreen):
            fail(f"expected ToolApprovalScreen, found {type(screen).__name__}")
        button_id = "#approval-approve" if approve else "#approval-deny"
        screen.query_one(button_id, Button).press()
        await wait_until(
            pilot,
            lambda: ("leader", "leader saw tool result") in app.chat_entries,
            "leader final answer after approval decision",
        )
        file_exists = (root / target).exists()
        if approve and not file_exists:
            fail("approved write_file call did not write the target file")
        if not approve and file_exists:
            fail("denied write_file call still wrote the target file")
        leader = app._leader  # noqa: SLF001
        if leader is None or "writer" not in leader.subagents:
            fail("approval flow did not create the expected writer subagent")
        tool_messages = [
            message
            for message in leader.subagents["writer"].messages
            if message.role == Role.TOOL and message.tool_result is not None
        ]
        if not tool_messages:
            fail("approval flow did not append a normal tool-result message")
        tool_result = tool_messages[0].tool_result
        if tool_result is None:
            fail("tool-result message was missing the ToolResult payload")
        if approve and not tool_result.ok:
            fail(f"approved tool call returned a failing ToolResult: {tool_result}")
        if not approve and (tool_result.ok or "denied by user" not in (tool_result.error or "")):
            fail(f"denied tool call did not return ToolResult(ok=False): {tool_result}")
        label = "approve" if approve else "deny"
        ok(f"prompt permission mode can {label} a side-effectful tool call")


async def main_async() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        await smoke_success_case(root)
        await smoke_error_case(root)
        await smoke_small_terminal_case(root)
        await smoke_terminal_status_case(root)
        await smoke_picker_success_case(root)
        await smoke_picker_fallback_case(root)
        await smoke_picker_include_all_case(root)
        await smoke_confirmation_before_discovery_case(root)
        await smoke_slash_commands_case(root)
        await smoke_model_command_case(root)
        await smoke_model_preselection_and_divider_case(root)
        await smoke_exit_command_case(root)
        await smoke_approval_case(root, approve=True)
        await smoke_approval_case(root, approve=False)


def main() -> int:
    asyncio.run(main_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
