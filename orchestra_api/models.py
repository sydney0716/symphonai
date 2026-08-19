"""Provider-agnostic data model for the Orchestra API agent runtime.

These types describe the shape of a request/response exchange with any
`ModelProvider` and the messages/tool calls exchanged during an `ApiAgent`
run. Fake, OpenAI, Anthropic, Gemini, and OpenAI-compatible providers all
implement the runtime provider interface -- see
`docs/orchestra-api-runtime.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Who sent a given `Message`."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Usage:
    """Token accounting for a single model call.

    Defaults to all zeros; a real provider fills these in from its API
    response. `FakeModelProvider` leaves them at zero.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to invoke a local tool by name.

    `provider_metadata` is opaque vendor passthrough data. Provider-specific
    values stored here must be returned verbatim when required by that
    provider, and must never be inspected, parsed, transformed, re-encoded,
    truncated, normalized, concatenated, or logged by generic runtime code.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """The outcome of executing a `ToolCall` against a `LocalTool`.

    `ok=False` covers both permission denials and execution failures;
    `error` carries a human-readable reason in that case.
    """

    tool_call_id: str
    ok: bool
    content: str = ""
    error: str | None = None


@dataclass(frozen=True)
class Message:
    """One turn in the conversation sent to, or returned from, a `ModelProvider`.

    An assistant `Message` may carry `tool_calls` (the model wants tools
    run); a tool-role `Message` carries the corresponding `tool_result`
    appended back into the conversation by the agent loop.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: ToolResult | None = None


@dataclass(frozen=True)
class ModelRequest:
    """A full request to a `ModelProvider`: the conversation plus available tools."""

    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class ModelResponse:
    """A `ModelProvider`'s reply: a message that may carry tool calls, a final answer, or both."""

    message: Message
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.message.tool_calls)
