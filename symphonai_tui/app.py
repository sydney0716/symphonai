"""Textual app for chatting with the SymphonAI API Leader."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, RichLog, Static

from symphonai_api.cancellation import CancellationToken
from symphonai_api.compaction import ContextCompactionError, describe_compaction
from symphonai_api.events import (
    CompactionApplied,
    Event,
    RunFailed,
    RunFinished,
    RunStarted,
    SubagentSpawned,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
    TurnStarted,
)
from symphonai_api.leader import Leader, LeaderConfig, LeaderRunResult
from symphonai_api.permissions import (
    DenialReason,
    PermissionDecision,
    PermissionMode,
    ToolApprovalRequest,
)
from symphonai_api.providers.base import ModelProvider, ProviderError
from symphonai_tui.commands import (
    ParsedSlashCommand,
    help_text,
    parse_slash_command,
    unknown_command_text,
)
from symphonai_tui.picker import (
    ProviderConfirmationScreen,
    ProviderPickerResult,
    ProviderPickerScreen,
    provider_summary,
    uses_real_provider,
)

AGENT_STATE_PENDING = "pending"
AGENT_STATE_WORKING = "working"
AGENT_STATE_DONE = "done"
AGENT_STATE_EXHAUSTED = "exhausted"
AGENT_STATE_FAILED = "failed"
AGENT_STATE_CANCELLED = "cancelled"

STATUS_STYLES = {
    AGENT_STATE_PENDING: "yellow",
    AGENT_STATE_WORKING: "bold blue",
    AGENT_STATE_DONE: "green",
    AGENT_STATE_FAILED: "bold red",
    AGENT_STATE_EXHAUSTED: "bold magenta",
    AGENT_STATE_CANCELLED: "bold cyan",
}


@dataclass(eq=False)
class _PendingApproval:
    request: ToolApprovalRequest
    event: threading.Event
    decision: PermissionDecision | None = None


class ToolApprovalScreen(Screen[bool]):
    """Approve or deny one side-effectful local tool call."""

    BINDINGS = [
        ("escape", "cancel", "Deny"),
        ("ctrl+c", "cancel", "Deny"),
        ("ctrl+x", "stop_turn", "Stop turn"),
    ]

    def __init__(self, request: ToolApprovalRequest) -> None:
        super().__init__()
        self._request = request

    def compose(self) -> ComposeResult:
        yield Static("Tool approval", id="approval-title")
        yield Static(f"Operation: {self._request.operation}", id="approval-operation")
        yield Static(f"Target: {self._request.target}", id="approval-target")
        yield Static(self._request.details, id="approval-details")
        yield Button("Approve", id="approval-approve", variant="primary")
        yield Button("Deny", id="approval-deny", variant="error")
        yield Button("Stop turn", id="approval-stop")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#approval-deny", Button).focus()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_stop_turn(self) -> None:
        self.app.action_stop_turn()
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approval-approve":
            self.dismiss(True)
        elif event.button.id == "approval-deny":
            self.dismiss(False)
        elif event.button.id == "approval-stop":
            self.action_stop_turn()


class SymphonAITuiApp(App[None]):
    """A small Textual shell around the synchronous Leader API."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #chat-log {
        height: 1fr;
        min-height: 3;
        min-width: 1;
        border: solid $primary;
        padding: 0 1;
    }

    #status-panel {
        height: 9;
        min-height: 3;
        min-width: 1;
        border: solid $accent;
        padding: 0 1;
    }

    #message-input {
        height: 3;
    }

    #turn-status {
        height: 1;
        min-width: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }

    ProviderPickerScreen, ProviderConfirmationScreen, ToolApprovalScreen {
        layout: vertical;
        padding: 1 2;
    }

    #picker-title, #confirm-title, #approval-title {
        height: 1;
        content-align: center middle;
        text-style: bold;
        color: $accent;
    }

    #picker-help, #picker-error, #confirm-warning,
    #confirm-leader, #confirm-subagent, #approval-operation,
    #approval-target, #approval-details {
        height: auto;
        min-height: 1;
    }

    #picker-error, #confirm-warning {
        color: $warning;
    }

    .picker-role-title {
        height: 1;
        margin-top: 1;
        text-style: bold;
    }

    Select, Checkbox, #leader-manual-model, #subagent-manual-model {
        height: 3;
    }

    #leader-model-status, #subagent-model-status {
        height: auto;
        min-height: 1;
        color: $text-muted;
    }

    #start-chat, #quit-picker, #confirm-continue, #confirm-cancel,
    #approval-approve, #approval-deny, #approval-stop {
        width: 100%;
        margin-top: 1;
    }

    Footer {
        height: 1;
    }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit"), ("escape", "stop_turn", "Stop")]

    def __init__(
        self,
        *,
        leader_provider: ModelProvider | None = None,
        subagent_provider: ModelProvider | None = None,
        repo_root: str | Path,
        max_leader_turns: int | None = None,
        max_subagents: int | None = None,
        subagent_max_turns: int | None = None,
        confirm_real_providers: bool = True,
        permission_mode: PermissionMode | None = None,
        chat_token_budget: int | None = None,
        chat_recent_turns: int | None = None,
    ) -> None:
        super().__init__()
        self._repo_root = str(repo_root)
        self._max_leader_turns = max_leader_turns
        self._max_subagents = max_subagents
        self._subagent_max_turns = subagent_max_turns
        self._confirm_real_providers = confirm_real_providers
        self._permission_mode = self._resolve_permission_mode(permission_mode)
        self._chat_token_budget = chat_token_budget
        self._chat_recent_turns = chat_recent_turns
        self._confirmed_real_providers = False
        self._leader_provider: ModelProvider | None = None
        self._subagent_provider: ModelProvider | None = None
        self._leader: Leader | None = None
        self._turn_in_flight = False
        self._cancel_token: CancellationToken | None = None
        self._agent_status: dict[str, tuple[str, str]] = {}
        self._approval_lock = threading.Lock()
        self._pending_approvals: list[_PendingApproval] = []
        self.chat_entries: list[tuple[str, str]] = []
        self.status_entries: list[tuple[str, str]] = []
        self.visible_status_text = "No agent activity yet."
        if leader_provider is not None and subagent_provider is not None:
            self._configure_leader(leader_provider, subagent_provider)

    def compose(self) -> ComposeResult:
        yield RichLog(
            id="chat-log",
            min_width=1,
            wrap=True,
            markup=False,
            auto_scroll=True,
        )
        yield Static(id="status-panel")
        yield Input(placeholder="Message the leader", id="message-input")
        yield Static("Ready", id="turn-status")
        yield Footer()

    def on_mount(self) -> None:
        self._render_status_panel()
        if self._leader is None:
            self._set_waiting_for_setup("Select providers first")
            self.call_after_refresh(self._open_picker)
            return
        if self._should_confirm_real_providers():
            self._set_waiting_for_setup("Confirm provider usage first")
            self.call_after_refresh(self._open_confirmation)
            return
        self._set_ready()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "message-input":
            return
        prompt = event.value.strip()
        event.input.value = ""
        if not prompt:
            return
        command = parse_slash_command(prompt)
        if command is not None:
            self._handle_slash_command(command)
            return
        if self._leader is None:
            self._set_turn_status("Select providers first")
            return
        if self._turn_in_flight:
            self._set_turn_status("Leader is already working")
            return

        self._append_chat_line("you", prompt, "bold cyan")
        self._set_busy(True)
        # Created here, on the UI thread, rather than inside the worker: the
        # worker may not start for some time, and a Stop pressed in that gap
        # would find no token and be silently dropped.
        self._cancel_token = CancellationToken()
        self._run_leader_turn(prompt)

    def _events_from_worker(self, event: Event) -> None:
        """Consume one runtime event. Called from the leader worker thread."""
        if isinstance(event, SubagentSpawned):
            update = (
                event.subagent_agent_id,
                event.subagent_name,
                AGENT_STATE_PENDING,
            )
        elif isinstance(event, RunStarted):
            update = (event.agent_id, event.agent_name, AGENT_STATE_WORKING)
        elif isinstance(event, RunFinished) and event.stopped_reason == "final_response":
            update = (event.agent_id, event.agent_name, AGENT_STATE_DONE)
        elif isinstance(event, RunFinished) and event.stopped_reason == "max_turns":
            update = (event.agent_id, event.agent_name, AGENT_STATE_EXHAUSTED)
        elif isinstance(event, RunFinished) and event.stopped_reason == "cancelled":
            update = (event.agent_id, event.agent_name, AGENT_STATE_CANCELLED)
        elif isinstance(event, RunFailed):
            update = (event.agent_id, event.agent_name, AGENT_STATE_FAILED)
        elif isinstance(
            event,
            (
                TurnStarted,
                TurnFinished,
                ToolCallStarted,
                ToolCallFinished,
                CompactionApplied,
            ),
        ):
            return
        else:
            raise TypeError(f"unsupported runtime event: {type(event).__name__}")
        self._call_ui_from_worker(self._update_agent_status, *update)

    def _call_ui_from_worker(self, callback, *args) -> None:  # noqa: ANN001
        if not self.is_running:
            return
        try:
            self.call_from_thread(callback, *args)
        except RuntimeError:
            return

    def _resolve_permission_mode(self, permission_mode: PermissionMode | None) -> PermissionMode:
        raw_mode = permission_mode or os.environ.get("SYMPHONAI_TUI_PERMISSION_MODE", "auto")
        mode = raw_mode.strip().lower()
        if mode not in ("auto", "prompt"):
            raise ValueError(
                f"unknown permission mode {raw_mode!r}; expected 'auto' or 'prompt'"
            )
        return mode  # type: ignore[return-value]

    def _append_chat_line(self, label: str, message: str, style: str) -> None:
        self.chat_entries.append((label, message))
        line = Text()
        line.append(f"{label}> ", style=style)
        line.append(message)
        self.query_one("#chat-log", RichLog).write(line)

    def _append_system_line(self, message: str) -> None:
        self._append_chat_line("system", message, "dim")

    def _append_error_line(self, message: str) -> None:
        self.chat_entries.append(("error", message))
        line = Text()
        line.append("error> ", style="bold red")
        line.append(message, style="red")
        self.query_one("#chat-log", RichLog).write(line)

    def _set_busy(self, busy: bool) -> None:
        self._turn_in_flight = busy
        input_widget = self.query_one("#message-input", Input)
        input_widget.disabled = busy
        input_widget.placeholder = "Leader is working..." if busy else "Message the leader"
        self._set_turn_status("Working" if busy else "Ready")
        if not busy:
            input_widget.focus()

    def _set_ready(self) -> None:
        input_widget = self.query_one("#message-input", Input)
        input_widget.disabled = False
        input_widget.placeholder = "Message the leader"
        self._set_turn_status("Ready")
        input_widget.focus()

    def _set_waiting_for_setup(self, message: str) -> None:
        input_widget = self.query_one("#message-input", Input)
        input_widget.disabled = True
        input_widget.placeholder = message
        self._set_turn_status(message)

    def _configure_leader(self, leader_provider: ModelProvider, subagent_provider: ModelProvider) -> None:
        config_kwargs = {
            "leader_provider": leader_provider,
            "subagent_provider": subagent_provider,
            "repo_root": self._repo_root,
            "events": self._events_from_worker,
        }
        if self._max_leader_turns is not None:
            config_kwargs["max_leader_turns"] = self._max_leader_turns
        if self._max_subagents is not None:
            config_kwargs["max_subagents"] = self._max_subagents
        if self._subagent_max_turns is not None:
            config_kwargs["subagent_max_turns"] = self._subagent_max_turns
        config_kwargs["permission_mode"] = self._permission_mode
        config_kwargs["approval_callback"] = self._approval_from_worker
        if self._chat_token_budget is not None:
            config_kwargs["chat_token_budget"] = self._chat_token_budget
        if self._chat_recent_turns is not None:
            config_kwargs["chat_recent_turns"] = self._chat_recent_turns
        self._leader_provider = leader_provider
        self._subagent_provider = subagent_provider
        self._leader = Leader(LeaderConfig(**config_kwargs))

    def _handle_slash_command(self, command: ParsedSlashCommand) -> None:
        if not command.known:
            self._append_error_line(unknown_command_text(command))
            return
        if command.name == "help":
            self._append_system_line(help_text())
            return
        if command.name == "model":
            if self._turn_in_flight:
                self._append_error_line("Cannot change models while the leader is working.")
                return
            self._confirmed_real_providers = False
            self._set_waiting_for_setup("Select providers first")
            self._open_picker(is_model_change=True)
            return
        if command.name == "clear":
            self._clear_chat()
            return
        if command.name == "compact":
            self._compact_chat()
            return
        if command.name == "exit":
            self.action_quit()
            return
        self._append_error_line(unknown_command_text(command))

    def _clear_chat(self) -> None:
        self.chat_entries.clear()
        self.query_one("#chat-log", RichLog).clear()
        cleared_subagents = 0
        if self._leader is not None:
            # clear_chat() clears the pool too and reports the count, so
            # calling clear_subagents() as well would clear it twice.
            cleared_subagents = self._leader.clear_chat()
        noun = "subagent" if cleared_subagents == 1 else "subagents"
        message = f"Chat cleared; {cleared_subagents} {noun} cleared."
        self._append_system_line(message)
        self._set_turn_status(message)

    def _compact_chat(self) -> None:
        if self._leader is None:
            self._append_error_line("Select providers before compacting the conversation.")
            return
        try:
            result = self._leader.compact_chat()
        except ContextCompactionError as exc:
            self._append_error_line(f"Context compaction error: {exc}")
            return
        self._append_system_line(describe_compaction(result))

    def _open_picker(self, *, is_model_change: bool = False) -> None:
        self.push_screen(
            ProviderPickerScreen(
                leader_provider=self._leader_provider,
                subagent_provider=self._subagent_provider,
                confirm_real_providers=(
                    self._confirm_real_providers and not self._confirmed_real_providers
                ),
                on_real_providers_confirmed=self._mark_real_providers_confirmed,
                cancel_label="Cancel" if is_model_change else "Quit",
            ),
            lambda result: self._picker_finished(result, is_model_change=is_model_change),
        )

    def _picker_finished(
        self,
        result: ProviderPickerResult | None,
        *,
        is_model_change: bool,
    ) -> None:
        if result is None:
            if is_model_change:
                self._set_ready()
            else:
                self.exit()
            return
        self._configure_leader(result.leader_provider, result.subagent_provider)
        if is_model_change:
            self._agent_status.clear()
            self._render_status_panel()
            self._append_system_line(
                "--- New model session: the new leader does not inherit the transcript above. ---"
            )
        self._set_turn_status(
            "Selected "
            f"leader={provider_summary(result.leader_provider)}, "
            f"subagent={provider_summary(result.subagent_provider)}"
        )
        if self._should_confirm_real_providers():
            self._open_confirmation()
        else:
            self._set_ready()

    def _mark_real_providers_confirmed(self) -> None:
        self._confirmed_real_providers = True

    def _should_confirm_real_providers(self) -> bool:
        if not self._confirm_real_providers or self._confirmed_real_providers:
            return False
        if self._leader_provider is None or self._subagent_provider is None:
            return False
        return uses_real_provider(self._leader_provider) or uses_real_provider(self._subagent_provider)

    def _open_confirmation(self) -> None:
        if self._leader_provider is None or self._subagent_provider is None:
            return
        self.push_screen(
            ProviderConfirmationScreen(
                leader_provider=self._leader_provider,
                subagent_provider=self._subagent_provider,
            ),
            self._confirmation_finished,
        )

    def _confirmation_finished(self, accepted: bool | None) -> None:
        if not accepted:
            self.exit()
            return
        self._confirmed_real_providers = True
        self._set_ready()

    def _approval_from_worker(self, request: ToolApprovalRequest) -> PermissionDecision:
        waiter = _PendingApproval(request=request, event=threading.Event())
        with self._approval_lock:
            self._pending_approvals.append(waiter)
        try:
            self.call_from_thread(self._open_tool_approval, waiter)
        except RuntimeError:
            self._resolve_approval(
                waiter,
                PermissionDecision.deny(
                    f"{request.operation} approval cancelled because the TUI is not running",
                    denial=DenialReason.APPROVAL_FAILED,
                ),
            )
        while not waiter.event.wait(0.1):
            if not self.is_running:
                self._resolve_approval(
                    waiter,
                    PermissionDecision.deny(
                        f"{request.operation} approval cancelled because the TUI is shutting down",
                        denial=DenialReason.APPROVAL_FAILED,
                    ),
                )
                break
        if waiter.decision is None:
            return PermissionDecision.deny(
                f"{request.operation} approval did not complete",
                denial=DenialReason.APPROVAL_FAILED,
            )
        return waiter.decision

    def _open_tool_approval(self, waiter: _PendingApproval) -> None:
        if not self.is_running:
            self._resolve_approval(
                waiter,
                PermissionDecision.deny(
                    f"{waiter.request.operation} approval cancelled because the TUI is not running",
                    denial=DenialReason.APPROVAL_FAILED,
                ),
            )
            return
        self.push_screen(
            ToolApprovalScreen(waiter.request),
            lambda accepted: self._approval_finished(waiter, accepted),
        )

    def _approval_finished(self, waiter: _PendingApproval, accepted: bool | None) -> None:
        if accepted:
            decision = PermissionDecision.allow()
        else:
            decision = PermissionDecision.deny(
                f"{waiter.request.operation} denied by user",
                denial=DenialReason.DENIED_BY_USER,
            )
        self._resolve_approval(waiter, decision)

    def _resolve_approval(self, waiter: _PendingApproval, decision: PermissionDecision) -> None:
        with self._approval_lock:
            if waiter.decision is not None:
                return
            waiter.decision = decision
            if waiter in self._pending_approvals:
                self._pending_approvals.remove(waiter)
            waiter.event.set()

    def _cancel_pending_approvals(self, reason: str) -> None:
        with self._approval_lock:
            pending = list(self._pending_approvals)
        for waiter in pending:
            self._resolve_approval(
                waiter,
                PermissionDecision.deny(
                    f"{waiter.request.operation} approval cancelled: {reason}",
                    denial=DenialReason.APPROVAL_FAILED,
                ),
            )

    def action_quit(self) -> None:
        self._cancel_pending_approvals("application is quitting")
        self.exit()

    def action_stop_turn(self) -> None:
        if not self._turn_in_flight or self._cancel_token is None:
            return
        self._cancel_token.cancel()
        self._cancel_pending_approvals("turn stopped by user")
        self._set_turn_status("Stopping...")

    def on_unmount(self) -> None:
        self._cancel_pending_approvals("application is shutting down")

    def _set_turn_status(self, message: str) -> None:
        self.query_one("#turn-status", Static).update(message)

    def _update_agent_status(self, agent_id: str, agent_name: str, state: str) -> None:
        self.status_entries.append((agent_name, state))
        self._agent_status[agent_id] = (agent_name, state)
        self._render_status_panel()

    def _render_status_panel(self) -> None:
        text = Text()
        text.append("Agent status\n", style="bold")
        if not self._agent_status:
            text.append("No agent activity yet.", style="dim")
            self.visible_status_text = "No agent activity yet."
            self.query_one("#status-panel", Static).update(text)
            return

        agent_ids = sorted(
            self._agent_status,
            key=lambda value: (
                self._agent_status[value][0] != "leader",
                self._agent_status[value][0].lower(),
            ),
        )
        label_counts: dict[str, int] = {}
        for label, _ in self._agent_status.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        visible_lines: list[str] = []
        for agent_id in agent_ids:
            label, state = self._agent_status[agent_id]
            short_id = agent_id.removeprefix("agent_")[:6]
            display_label = (
                f"{label} ({short_id})" if label_counts[label] > 1 else label
            )
            visible_lines.append(f"{display_label}: {state}")
            text.append(display_label, style="bold" if label == "leader" else "")
            text.append(": ")
            text.append(state, style=STATUS_STYLES.get(state, ""))
            text.append("\n")
        self.visible_status_text = "\n".join(visible_lines)
        self.query_one("#status-panel", Static).update(text)

    @work(thread=True, exclusive=True, group="leader")
    def _run_leader_turn(self, prompt: str) -> None:
        # Leader.chat() is blocking and may perform HTTP requests. Keep it off
        # Textual's event loop so the UI can redraw while a turn is in flight.
        cancel_token = self._cancel_token
        try:
            if self._leader is None:
                raise ProviderError("leader is not configured")
            result = self._leader.chat(prompt, cancel=cancel_token)
        except ContextCompactionError as exc:
            self._call_ui_from_worker(self._finish_turn_error, f"Context compaction error: {exc}")
        except ProviderError as exc:
            self._call_ui_from_worker(self._finish_turn_error, f"Provider error: {exc}")
        except Exception as exc:  # noqa: BLE001
            self._call_ui_from_worker(
                self._finish_turn_error,
                f"Unexpected error ({type(exc).__name__}): {exc}",
            )
        else:
            self._call_ui_from_worker(self._finish_turn_success, result)

    def _finish_turn_success(self, result: LeaderRunResult) -> None:
        if result.final_answer:
            self._append_chat_line("leader", result.final_answer, "bold green")
        if result.stopped_reason == "cancelled":
            self._append_system_line("Turn stopped; answer may be incomplete.")
        elif result.stopped_reason != "final_response":
            self._append_error_line(
                f"Stopped due to {result.stopped_reason}; answer may be incomplete."
            )
        self._cancel_token = None
        self._set_busy(False)

    def _finish_turn_error(self, message: str) -> None:
        self._append_error_line(message)
        self._cancel_token = None
        self._set_busy(False)
