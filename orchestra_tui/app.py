"""Textual app for chatting with the Orchestra API Leader."""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Input, RichLog, Static

from orchestra_api.leader import Leader, LeaderConfig, LeaderRunResult
from orchestra_api.providers.base import ModelProvider, ProviderError

STATUS_STYLES = {
    "pending": "yellow",
    "working": "bold blue",
    "done": "green",
    "failed": "bold red",
}


class OrchestraTuiApp(App[None]):
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
        height: 6;
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

    Footer {
        height: 1;
    }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(
        self,
        *,
        leader_provider: ModelProvider,
        subagent_provider: ModelProvider,
        repo_root: str | Path,
        max_leader_turns: int | None = None,
        max_subagents: int | None = None,
        subagent_max_turns: int | None = None,
    ) -> None:
        super().__init__()
        config_kwargs = {
            "leader_provider": leader_provider,
            "subagent_provider": subagent_provider,
            "repo_root": str(repo_root),
            "on_status": self._status_from_worker,
        }
        if max_leader_turns is not None:
            config_kwargs["max_leader_turns"] = max_leader_turns
        if max_subagents is not None:
            config_kwargs["max_subagents"] = max_subagents
        if subagent_max_turns is not None:
            config_kwargs["subagent_max_turns"] = subagent_max_turns
        self._leader = Leader(LeaderConfig(**config_kwargs))
        self._turn_in_flight = False
        self._agent_status: dict[str, str] = {}
        self.chat_entries: list[tuple[str, str]] = []
        self.status_entries: list[tuple[str, str]] = []
        self.visible_status_text = "No agent activity yet."

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
        self.query_one("#message-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "message-input":
            return
        prompt = event.value.strip()
        event.input.value = ""
        if not prompt:
            return
        if self._turn_in_flight:
            self._set_turn_status("Leader is already working")
            return

        self._append_chat_line("you", prompt, "bold cyan")
        self._set_busy(True)
        self._run_leader_turn(prompt)

    def _status_from_worker(self, agent_name: str, state: str) -> None:
        self._call_ui_from_worker(self._update_agent_status, agent_name, state)

    def _call_ui_from_worker(self, callback, *args) -> None:  # noqa: ANN001
        if not self.is_running:
            return
        try:
            self.call_from_thread(callback, *args)
        except RuntimeError:
            return

    def _append_chat_line(self, label: str, message: str, style: str) -> None:
        self.chat_entries.append((label, message))
        line = Text()
        line.append(f"{label}> ", style=style)
        line.append(message)
        self.query_one("#chat-log", RichLog).write(line)

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

    def _set_turn_status(self, message: str) -> None:
        self.query_one("#turn-status", Static).update(message)

    def _update_agent_status(self, agent_name: str, state: str) -> None:
        self.status_entries.append((agent_name, state))
        self._agent_status[agent_name] = state
        self._render_status_panel()

    def _render_status_panel(self) -> None:
        text = Text()
        text.append("Agent status\n", style="bold")
        if not self._agent_status:
            text.append("No agent activity yet.", style="dim")
            self.visible_status_text = "No agent activity yet."
            self.query_one("#status-panel", Static).update(text)
            return

        labels = sorted(
            self._agent_status,
            key=lambda value: (value != "leader", value.lower()),
        )
        visible_lines: list[str] = []
        for label in labels:
            state = self._agent_status[label]
            visible_lines.append(f"{label}: {state}")
            text.append(label, style="bold" if label == "leader" else "")
            text.append(": ")
            text.append(state, style=STATUS_STYLES.get(state, ""))
            text.append("\n")
        self.visible_status_text = "\n".join(visible_lines)
        self.query_one("#status-panel", Static).update(text)

    @work(thread=True, exclusive=True, group="leader")
    def _run_leader_turn(self, prompt: str) -> None:
        # Leader.chat() is blocking and may perform HTTP requests. Keep it off
        # Textual's event loop so the UI can redraw while a turn is in flight.
        try:
            result = self._leader.chat(prompt)
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
        self._append_chat_line("leader", result.final_answer, "bold green")
        if result.stopped_reason != "final_response":
            self._append_error_line(
                f"Stopped due to {result.stopped_reason}; answer may be incomplete."
            )
        self._set_busy(False)

    def _finish_turn_error(self, message: str) -> None:
        self._append_error_line(message)
        self._set_busy(False)
