"""Anthropic-backed ModelProvider: calls the real Anthropic Messages API.

Maps symphonai_api's provider-agnostic Message/ToolCall/ToolResult shape
into and out of Anthropic's Messages API wire format
(https://docs.anthropic.com/en/api/messages), using only the standard
library (`urllib.request`) -- no new dependency.

`ModelRequest.tools` is passed through unmodified into the request's
`tools` field, so callers must already supply tool definitions in
Anthropic's native `{"name", "description", "input_schema"}` shape. The
standard runtime call sites prepare that shape via
`symphonai_api.tool_schema`.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator

from symphonai_api.cancellation import CancellationToken
from symphonai_api.identity import new_id
from symphonai_api.model_table import resolve_effort
from symphonai_api.models import (
    DocumentBlock,
    ImageBlock,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    TextBlock,
    ToolCall,
    Usage,
    has_attachments,
    reject_system_attachments,
    wire_tool_call_ids,
)
from symphonai_api.providers.base import ModelProvider, ProviderError, parse_json_object, tool_result_text
from symphonai_api.retry import DEFAULT_MAX_ATTEMPTS, read_with_retry, redact_secret
from symphonai_api.streaming import (
    StreamChunk,
    StreamCompleted,
    StreamEventError,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    open_stream_with_retry,
    retry_stream_chunks,
    sse_events,
)

API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_MODEL = "claude-opus-5-5"
ROUTINE_MODEL = "claude-sonnet-5-5"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 32000
DEFAULT_STREAM_MAX_TOKENS = 32000
_ANTHROPIC_CONTENT_KEY = "anthropic_content"
_MAX_TOKENS_LIMIT_PATTERNS = (
    re.compile(r"\bmax_tokens\b.*?>\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bmax_tokens\b.*?\b(?:limit|maximum)\D{0,24}(\d+)", re.IGNORECASE),
    re.compile(r"\b(?:limit|maximum)(?:\s+output)?\s+tokens?\s*(?:is|of|:|=)?\s*(\d+)", re.IGNORECASE),
    re.compile(r"\blimit\s+(?:is|of|:|=)\s*(\d+)", re.IGNORECASE),
)
_THINKING_DISABLED: set[tuple[str, str]] = set()
_MAX_TOKENS_LIMITS: dict[tuple[str, str], int] = {}
_ANTHROPIC_CAPABILITIES_LOCK = threading.Lock()


def _model_key(base_url: str, model: str) -> tuple[str, str]:
    return base_url, model


def _cached_max_tokens(key: tuple[str, str]) -> int | None:
    with _ANTHROPIC_CAPABILITIES_LOCK:
        return _MAX_TOKENS_LIMITS.get(key)


def _thinking_disabled(key: tuple[str, str]) -> bool:
    with _ANTHROPIC_CAPABILITIES_LOCK:
        return key in _THINKING_DISABLED


def _max_tokens_limit(error: ProviderError) -> int | None:
    message = str(error)
    if "max_tokens" not in message.casefold():
        return None
    for pattern in _MAX_TOKENS_LIMIT_PATTERNS:
        match = pattern.search(message)
        if match:
            return int(match.group(1))
    return None


def _thinking_parameter_rejected(error: ProviderError) -> bool:
    message = str(error).casefold()
    parameter = "thinking" in message or "display" in message
    rejection = any(
        phrase in message
        for phrase in ("unsupported", "not supported", "unrecognized", "unknown parameter", "invalid parameter")
    )
    return parameter and rejection

def _synthesize_tool_call_id() -> str:
    """Build a fallback canonical id for tool calls that arrive without one.

    Must not be position-based: the same index in a later turn would reuse
    the id, leaving two distinct calls sharing a `tool_call_id`.
    """
    return new_id("call")


def _anthropic_blocks(message: Message) -> list[dict[str, Any]]:
    """Message content as Anthropic content blocks, text runs merged."""
    blocks: list[dict[str, Any]] = []
    text_run = ""
    for block in message.content:
        if isinstance(block, TextBlock):
            text_run += block.text
            continue
        if text_run:
            blocks.append({"type": "text", "text": text_run})
            text_run = ""
        if isinstance(block, ImageBlock):
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.media_type,
                        "data": block.data,
                    },
                }
            )
        elif isinstance(block, DocumentBlock):
            document = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": block.media_type,
                    "data": block.data,
                },
            }
            if block.filename is not None:
                document["title"] = block.filename
            blocks.append(document)
    if text_run:
        blocks.append({"type": "text", "text": text_run})
    return blocks


def _to_anthropic_content(message: Message, id_map: dict[str, str]) -> list[dict[str, Any]] | str:
    """Build the `content` value for one outgoing Anthropic message."""
    if message.role == Role.TOOL:
        result = message.tool_result
        assert result is not None, "tool-role Message must carry a tool_result"
        return [
            {
                "type": "tool_result",
                "tool_use_id": id_map.get(result.tool_call_id, result.tool_call_id),
                "content": tool_result_text(result),
                "is_error": not result.ok,
            }
        ]
    if message.role == Role.ASSISTANT and message.tool_calls:
        stored = message.tool_calls[0].provider_metadata.get(_ANTHROPIC_CONTENT_KEY)
        if isinstance(stored, list) and all(isinstance(block, dict) for block in stored):
            stored_ids = [
                block.get("id")
                for block in stored
                if block.get("type") == "tool_use"
            ]
            call_ids = [call.vendor_id or call.id for call in message.tool_calls]
            if stored_ids == call_ids:
                return stored
        blocks = _anthropic_blocks(message)
        for tool_call in message.tool_calls:
            blocks.append(
                {"type": "tool_use", "id": tool_call.vendor_id or tool_call.id, "name": tool_call.name, "input": tool_call.arguments}
            )
        return blocks
    return message.text if not has_attachments(message) else _anthropic_blocks(message)


def _to_anthropic_role(role: Role) -> str:
    # Anthropic only has "user" and "assistant" roles; a tool result rides
    # on a user-role message (see _to_anthropic_content).
    return "assistant" if role == Role.ASSISTANT else "user"


def _anthropic_messages(
    messages: list[Message], id_map: dict[str, str]
) -> list[dict[str, Any]]:
    wire_messages: list[dict[str, Any]] = []
    for message in messages:
        role = _to_anthropic_role(message.role)
        content = _to_anthropic_content(message, id_map)
        if role == "user" and wire_messages and wire_messages[-1]["role"] == "user":
            previous = wire_messages[-1]["content"]
            previous_blocks = previous if isinstance(previous, list) else [
                {"type": "text", "text": previous}
            ]
            content_blocks = content if isinstance(content, list) else [
                {"type": "text", "text": content}
            ]
            combined = previous_blocks + content_blocks
            wire_messages[-1]["content"] = [
                block for block in combined if block.get("type") == "tool_result"
            ] + [block for block in combined if block.get("type") != "tool_result"]
        else:
            wire_messages.append({"role": role, "content": content})
    return wire_messages


def _usage_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _build_request_body(
    request: ModelRequest,
    model: str,
    default_max_tokens: int,
    *,
    max_tokens_override: int | None = None,
    include_thinking: bool = True,
) -> dict[str, Any]:
    reject_system_attachments(request.messages)
    id_map = wire_tool_call_ids(request.messages)
    system_parts = [m.text for m in request.messages if m.role == Role.SYSTEM and m.text]
    non_system = [m for m in request.messages if m.role != Role.SYSTEM]
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": (
            max_tokens_override
            if max_tokens_override is not None
            else request.max_tokens or default_max_tokens
        ),
        "messages": _anthropic_messages(non_system, id_map),
        "cache_control": {"type": "ephemeral"},
    }
    if system_parts:
        body["system"] = [{
            "type": "text",
            "text": system_parts[0],
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }]
        if len(system_parts) > 1:
            body["system"].append({"type": "text", "text": "\n".join(system_parts[1:])})
    if request.tools:
        body["tools"] = request.tools
    if request.temperature is not None:
        body["temperature"] = request.temperature
    effort = resolve_effort(2, model, request.effort)
    if effort is not None:
        body["output_config"] = {"effort": effort}
        if include_thinking and request.temperature is None:
            body["thinking"] = {"type": "adaptive", "display": "summarized"}
    return body


def _parse_response(data: dict[str, Any]) -> ModelResponse:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    content = data.get("content", [])
    for block in content:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "thinking":
            thinking = block.get("thinking")
            if isinstance(thinking, str):
                reasoning_parts.append(thinking)
        elif block.get("type") == "tool_use":
            vendor_id = block.get("id") or None
            tool_calls.append(
                ToolCall(
                    id=vendor_id or _synthesize_tool_call_id(),
                    name=block["name"],
                    arguments=block.get("input", {}),
                    vendor_id=vendor_id,
                )
            )
    if (
        tool_calls
        and any(
            block.get("type") in {"thinking", "redacted_thinking"}
            for block in content
        )
    ):
        tool_calls[0].provider_metadata[_ANTHROPIC_CONTENT_KEY] = content
    usage_raw = data.get("usage")
    if not isinstance(usage_raw, dict):
        usage_raw = {}
    input_tokens = _usage_count(usage_raw.get("input_tokens"))
    cache_read_tokens = _usage_count(usage_raw.get("cache_read_input_tokens"))
    cache_write_tokens = _usage_count(usage_raw.get("cache_creation_input_tokens"))
    cache_creation = usage_raw.get("cache_creation")
    if not isinstance(cache_creation, dict):
        cache_creation = {}
    message = Message(
        role=Role.ASSISTANT,
        content="".join(text_parts),
        tool_calls=tool_calls,
        reasoning="".join(reasoning_parts),
    )
    return ModelResponse(
        message=message,
        usage=Usage(
            input_tokens=input_tokens + cache_read_tokens + cache_write_tokens,
            output_tokens=_usage_count(usage_raw.get("output_tokens")),
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_write_1h_tokens=_usage_count(cache_creation.get("ephemeral_1h_input_tokens")),
        ),
        stop_reason=data.get("stop_reason") or "end_turn",
    )


def _anthropic_stream_chunks(
    lines: Iterator[bytes], *, api_key: str
) -> Iterator[StreamChunk]:
    """Map Anthropic Messages SSE events to the provider-neutral chunks."""
    input_tokens = 0
    cache_read_tokens = 0
    cache_write_tokens = 0
    cache_write_1h_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"
    content_blocks: dict[int, dict[str, Any]] = {}
    tool_input_fragments: dict[int, list[str]] = {}
    tool_indices: list[int] = []
    for event, payload in sse_events(lines):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Anthropic API stream returned invalid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise ProviderError("Anthropic API stream returned non-object JSON")

        if event == "error":
            error = data.get("error", {})
            message = error.get("message", "Anthropic API stream returned an error") if isinstance(error, dict) else "Anthropic API stream returned an error"
            raise StreamEventError(redact_secret(str(message), api_key), error.get("type") or error.get("code"), "anthropic")
        if event == "message_start":
            message = data.get("message", {})
            usage = message.get("usage", {}) if isinstance(message, dict) else {}
            if isinstance(usage, dict):
                input_tokens = _usage_count(usage.get("input_tokens"))
                cache_read_tokens = _usage_count(
                    usage.get("cache_read_input_tokens")
                )
                cache_write_tokens = _usage_count(
                    usage.get("cache_creation_input_tokens")
                )
                cache_creation = usage.get("cache_creation")
                if isinstance(cache_creation, dict):
                    cache_write_1h_tokens = _usage_count(
                        cache_creation.get("ephemeral_1h_input_tokens")
                    )
            continue
        if event == "content_block_start":
            block = data.get("content_block", {})
            index = data["index"]
            if isinstance(block, dict):
                content_blocks[index] = dict(block)
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_indices.append(index)
                tool_input_fragments[index] = []
                yield ToolCallDelta(
                    index,
                    id=block.get("id"),
                    name=block.get("name"),
                    vendor_id=block.get("id"),
                )
            continue
        if event == "content_block_delta":
            delta = data.get("delta", {})
            if not isinstance(delta, dict):
                continue
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                block = content_blocks.get(data["index"])
                if block is not None:
                    block["text"] = block.get("text", "") + text
                yield TextDelta(text)
            elif delta.get("type") == "thinking_delta":
                block = content_blocks.get(data["index"])
                thinking = delta.get("thinking", "")
                if block is not None:
                    block["thinking"] = block.get("thinking", "") + thinking
                if isinstance(thinking, str) and thinking:
                    yield ReasoningDelta(thinking)
            elif delta.get("type") == "signature_delta":
                block = content_blocks.get(data["index"])
                if block is not None:
                    block["signature"] = block.get("signature", "") + delta.get(
                        "signature", ""
                    )
            elif delta.get("type") == "input_json_delta":
                fragment = delta.get("partial_json", "")
                tool_input_fragments.setdefault(data["index"], []).append(fragment)
                yield ToolCallDelta(
                    data["index"], arguments_fragment=fragment
                )
            continue
        if event == "message_delta":
            delta = data.get("delta", {})
            usage = data.get("usage", {})
            if isinstance(delta, dict) and delta.get("stop_reason") is not None:
                stop_reason = delta["stop_reason"]
            if isinstance(usage, dict):
                output_tokens = _usage_count(usage.get("output_tokens"))
            continue
        if event == "message_stop":
            content = [content_blocks[index] for index in sorted(content_blocks)]
            has_thinking = any(
                block.get("type") in {"thinking", "redacted_thinking"}
                for block in content
            )
            if has_thinking and tool_indices:
                cut_index = (
                    max(tool_indices)
                    if stop_reason in {"max_tokens", "length", "MAX_TOKENS"}
                    else None
                )
                for index in tool_indices:
                    if index == cut_index:
                        content_blocks[index]["input"] = {}
                        continue
                    fragments = "".join(tool_input_fragments.get(index, []))
                    if fragments:
                        try:
                            arguments = json.loads(fragments)
                        except json.JSONDecodeError as exc:
                            name = content_blocks[index].get("name") or "<unknown>"
                            raise ProviderError(
                                f"streamed tool {name!r} at index {index} returned invalid "
                                f"arguments JSON: {exc}"
                            ) from None
                        if not isinstance(arguments, dict):
                            name = content_blocks[index].get("name") or "<unknown>"
                            raise ProviderError(
                                f"streamed tool {name!r} at index {index} returned "
                                "non-object arguments"
                            )
                        content_blocks[index]["input"] = arguments
                first_tool_index = min(tool_indices)
                yield ToolCallDelta(
                    first_tool_index,
                    provider_metadata={_ANTHROPIC_CONTENT_KEY: content},
                )
            yield StreamCompleted(
                ModelResponse(
                    Message(role=Role.ASSISTANT),
                    usage=Usage(
                        input_tokens=input_tokens + cache_read_tokens + cache_write_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        cache_write_1h_tokens=cache_write_1h_tokens,
                    ),
                    stop_reason=stop_reason,
                )
            )
            return


@dataclass
class AnthropicProvider(ModelProvider):
    """Calls the real Anthropic Messages API.

    The API key is never accepted as a constructor argument or stored on
    this object -- it is read from `ANTHROPIC_API_KEY` fresh on every call,
    never logged, and never included in any error message.
    """

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_seconds: float = 600.0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def wire_format(self) -> int:
        return 2

    @staticmethod
    def is_configured() -> bool:
        """Whether ANTHROPIC_API_KEY is set and non-empty. Never reads/logs its value."""
        return bool(os.environ.get(API_KEY_ENV_VAR, "").strip())

    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise ProviderError(f"{API_KEY_ENV_VAR} is not set")

        model = request.model if request.model is not None else self.model
        key = _model_key(self.base_url, model)
        max_tokens_override = (
            _cached_max_tokens(key) if request.max_tokens is None else None
        )
        effort = resolve_effort(2, model, request.effort)
        include_thinking = (
            effort is not None
            and request.temperature is None
            and not _thinking_disabled(key)
        )
        retried_max_tokens = False
        while True:
            body = _build_request_body(
                request,
                model,
                self.max_tokens,
                max_tokens_override=max_tokens_override,
                include_thinking=include_thinking,
            )
            http_request = urllib.request.Request(
                f"{self.base_url}/messages",
                data=json.dumps(body).encode("utf-8"),
                method="POST",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
            )
            try:
                raw = read_with_retry(
                    http_request,
                    timeout=self.timeout_seconds,
                    max_attempts=self.max_attempts,
                    api_key=api_key,
                    operation="Anthropic API",
                    fallback_configured=bool(getattr(self, "_symphonai_fallback_models", ())),
                    cancel=cancel,
                    call_class=request.call_class,
                )
            except ProviderError as exc:
                limit = _max_tokens_limit(exc)
                sent_tokens = body["max_tokens"]
                if (
                    not retried_max_tokens
                    and limit is not None
                    and sent_tokens > limit
                ):
                    with _ANTHROPIC_CAPABILITIES_LOCK:
                        _MAX_TOKENS_LIMITS[key] = limit
                    max_tokens_override = limit
                    retried_max_tokens = True
                    continue
                if include_thinking and _thinking_parameter_rejected(exc):
                    with _ANTHROPIC_CAPABILITIES_LOCK:
                        _THINKING_DISABLED.add(key)
                    include_thinking = False
                    continue
                raise
            return _parse_response(parse_json_object(raw, "Anthropic API"))

    def create_response_stream(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> Iterator[StreamChunk]:
        api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise ProviderError(f"{API_KEY_ENV_VAR} is not set")

        model = request.model if request.model is not None else self.model
        key = _model_key(self.base_url, model)
        max_tokens_override = (
            _cached_max_tokens(key) if request.max_tokens is None else None
        )
        effort = resolve_effort(2, model, request.effort)
        include_thinking = (
            effort is not None
            and request.temperature is None
            and not _thinking_disabled(key)
        )
        retried_max_tokens = False
        while True:
            body = _build_request_body(
                request,
                model,
                DEFAULT_STREAM_MAX_TOKENS,
                max_tokens_override=max_tokens_override,
                include_thinking=include_thinking,
            )
            body["stream"] = True
            http_request = urllib.request.Request(
                f"{self.base_url}/messages",
                data=json.dumps(body).encode("utf-8"),
                method="POST",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
            )

            def attempt(remaining):
                lines = open_stream_with_retry(
                    http_request,
                    timeout=remaining,
                    max_attempts=1,
                    api_key=api_key,
                    operation="Anthropic API",
                    fallback_configured=bool(getattr(self, "_symphonai_fallback_models", ())),
                    cancel=cancel,
                )
                return _anthropic_stream_chunks(lines, api_key=api_key)

            emitted = False
            try:
                for chunk in retry_stream_chunks(
                    attempt,
                    timeout=self.timeout_seconds,
                    max_attempts=self.max_attempts,
                    fallback_configured=bool(getattr(self, "_symphonai_fallback_models", ())),
                    cancel=cancel,
                ):
                    emitted = True
                    yield chunk
                return
            except ProviderError as exc:
                if emitted:
                    raise
                limit = _max_tokens_limit(exc)
                sent_tokens = body["max_tokens"]
                if (
                    not retried_max_tokens
                    and limit is not None
                    and sent_tokens > limit
                ):
                    with _ANTHROPIC_CAPABILITIES_LOCK:
                        _MAX_TOKENS_LIMITS[key] = limit
                    max_tokens_override = limit
                    retried_max_tokens = True
                    continue
                if include_thinking and _thinking_parameter_rejected(exc):
                    with _ANTHROPIC_CAPABILITIES_LOCK:
                        _THINKING_DISABLED.add(key)
                    include_thinking = False
                    continue
                raise
