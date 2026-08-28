"""Provider-agnostic data model for the Orchestra API agent runtime.

These types describe the shape of a request/response exchange with any
`ModelProvider` and the messages/tool calls exchanged during an `ApiAgent`
run. Fake, OpenAI, Anthropic, Gemini, and OpenAI-compatible providers all
implement the runtime provider interface -- see
`docs/orchestra-api-runtime.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from orchestra_api.identity import SCHEMA_VERSION


@dataclass(frozen=True)
class TextBlock:
    """A run of plain text in a message."""

    text: str
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class ImageBlock:
    """A raster image shown to the model, base64-encoded."""

    data: str
    """Standard base64 of the raw image bytes: no `data:` URI prefix, no
    newlines. Providers add whatever framing their wire format wants."""
    media_type: str
    """One of `content.SUPPORTED_IMAGE_MEDIA_TYPES`."""
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class DocumentBlock:
    """A PDF shown to the model, base64-encoded."""

    data: str
    media_type: str = "application/pdf"
    filename: str | None = None
    """Display name, when one is known. OpenAI's wire format requires a
    filename; the others treat it as optional metadata."""
    schema_version: int = SCHEMA_VERSION


ContentBlock = TextBlock | ImageBlock | DocumentBlock

# Accepted at construction time and normalized by Message.__post_init__.
ContentInput = str | ContentBlock | Sequence[str | ContentBlock] | None


def _normalize_content(value: ContentInput) -> tuple[ContentBlock, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return (TextBlock(text=value),)
    if isinstance(value, (TextBlock, ImageBlock, DocumentBlock)):
        return (value,)
    if isinstance(value, Sequence):
        blocks: list[ContentBlock] = []
        for item in value:
            blocks.extend(_normalize_content(item))
        return tuple(blocks)
    raise TypeError(f"unsupported message content type: {type(value).__name__}")


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
    vendor_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ToolCall.id must be a non-empty string")


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
    cancelled: bool = False
    payload: dict | None = None
    """Structured result for callers that can use one, JSON-shaped and
    JSON-serializable. `content` stays the human- and model-readable form; a
    consumer that does not understand `payload` loses nothing. `ToolMetadata.
    result_hint` names which shape to expect."""
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class Message:
    """One turn in the conversation sent to, or returned from, a `ModelProvider`.

    An assistant `Message` may carry `tool_calls` (the model wants tools
    run); a tool-role `Message` carries the corresponding `tool_result`
    appended back into the conversation by the agent loop.
    """

    role: Role
    content: ContentInput = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: ToolResult | None = None
    turn_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", _normalize_content(self.content))

    @property
    def text(self) -> str:
        """All text blocks concatenated. The str view every caller wants."""
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))


def has_attachments(message: Message) -> bool:
    """True when `message.content` holds any block that is not a `TextBlock`.

    Providers branch on this so that a text-only message keeps producing the
    exact wire shape it produced before attachments existed.
    """
    return any(not isinstance(block, TextBlock) for block in message.content)


def wire_tool_call_ids(messages: Sequence[Message]) -> dict[str, str]:
    """Map canonical ToolCall.id to the id to send back on the wire.

    Providers echoing a tool result must send the id the vendor gave them,
    not our canonical one. Falls back to the canonical id when the vendor
    supplied none.
    """
    return {
        tool_call.id: tool_call.vendor_id or tool_call.id
        for message in messages
        for tool_call in message.tool_calls
    }


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
