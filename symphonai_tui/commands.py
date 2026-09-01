"""Slash command parsing for the SymphonAI Textual TUI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str

    @property
    def token(self) -> str:
        return f"/{self.name}"


@dataclass(frozen=True)
class ParsedSlashCommand:
    raw: str
    name: str
    argument_text: str = ""
    known: bool = True

    @property
    def token(self) -> str:
        return f"/{self.name}" if self.name else self.raw


COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("help", "List available slash commands."),
    SlashCommand("model", "Open the provider/model picker."),
    SlashCommand("clear", "Clear the chat, leader conversation, and subagent pool."),
    SlashCommand("compact", "Compact the leader conversation state now."),
    SlashCommand("exit", "Quit the TUI."),
)

COMMAND_NAMES = {command.name for command in COMMANDS}


def parse_slash_command(text: str) -> ParsedSlashCommand | None:
    """Parse `text` as a slash command, or return None for normal chat."""

    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    token, _, argument_text = stripped.partition(" ")
    name = token[1:].strip().lower()
    return ParsedSlashCommand(
        raw=stripped,
        name=name,
        argument_text=argument_text.strip(),
        known=name in COMMAND_NAMES,
    )


def valid_command_tokens() -> list[str]:
    return [command.token for command in COMMANDS]


def help_text() -> str:
    return "Commands: " + "; ".join(
        f"{command.token} - {command.description}" for command in COMMANDS
    )


def unknown_command_text(command: ParsedSlashCommand) -> str:
    return (
        f"Unknown command {command.token!r}. "
        f"Valid commands: {', '.join(valid_command_tokens())}. "
        "Use /help for details."
    )
