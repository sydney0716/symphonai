"""Anthropic-backed ModelProvider: calls the real Anthropic Messages API.

Maps orchestra_api's provider-agnostic Message/ToolCall/ToolResult shape
into and out of Anthropic's Messages API wire format
(https://docs.anthropic.com/en/api/messages), using only the standard
library (`urllib.request`) -- no new dependency.

`ModelRequest.tools` is passed through unmodified into the request's
`tools` field, so callers must already supply tool definitions in
Anthropic's native `{"name", "description", "input_schema"}` shape. The
standard runtime call sites prepare that shape via
`orchestra_api.tool_schema`.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any

from orchestra_api.models import Message, ModelRequest, ModelResponse, Role, ToolCall, Usage
from orchestra_api.providers.base import ModelProvider, ProviderError, parse_json_object
from orchestra_api.retry import DEFAULT_MAX_ATTEMPTS, read_with_retry

API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024


def _to_anthropic_content(message: Message) -> list[dict[str, Any]] | str:
    """Build the `content` value for one outgoing Anthropic message."""
    if message.role == Role.TOOL:
        result = message.tool_result
        assert result is not None, "tool-role Message must carry a tool_result"
        return [
            {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": result.content if result.ok else (result.error or ""),
                "is_error": not result.ok,
            }
        ]
    if message.role == Role.ASSISTANT and message.tool_calls:
        blocks: list[dict[str, Any]] = []
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        for tool_call in message.tool_calls:
            blocks.append(
                {"type": "tool_use", "id": tool_call.id, "name": tool_call.name, "input": tool_call.arguments}
            )
        return blocks
    return message.content


def _to_anthropic_role(role: Role) -> str:
    # Anthropic only has "user" and "assistant" roles; a tool result rides
    # on a user-role message (see _to_anthropic_content).
    return "assistant" if role == Role.ASSISTANT else "user"


def _build_request_body(request: ModelRequest, model: str, default_max_tokens: int) -> dict[str, Any]:
    system_parts = [m.content for m in request.messages if m.role == Role.SYSTEM and m.content]
    non_system = [m for m in request.messages if m.role != Role.SYSTEM]
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_tokens or default_max_tokens,
        "messages": [
            {"role": _to_anthropic_role(m.role), "content": _to_anthropic_content(m)} for m in non_system
        ],
    }
    if system_parts:
        body["system"] = "\n".join(system_parts)
    if request.tools:
        body["tools"] = request.tools
    if request.temperature is not None:
        body["temperature"] = request.temperature
    return body


def _parse_response(data: dict[str, Any]) -> ModelResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(id=block["id"], name=block["name"], arguments=block.get("input", {}))
            )
    usage_raw = data.get("usage", {})
    message = Message(role=Role.ASSISTANT, content="".join(text_parts), tool_calls=tool_calls)
    return ModelResponse(
        message=message,
        usage=Usage(
            input_tokens=usage_raw.get("input_tokens", 0),
            output_tokens=usage_raw.get("output_tokens", 0),
        ),
        stop_reason=data.get("stop_reason") or "end_turn",
    )


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
    timeout_seconds: float = 30.0
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

    def create_response(self, request: ModelRequest) -> ModelResponse:
        api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise ProviderError(f"{API_KEY_ENV_VAR} is not set")

        model = request.model if request.model is not None else self.model
        body = _build_request_body(request, model, self.max_tokens)
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
        raw = read_with_retry(
            http_request,
            timeout=self.timeout_seconds,
            max_attempts=self.max_attempts,
            api_key=api_key,
            operation="Anthropic API",
        )
        data = parse_json_object(raw, "Anthropic API")

        return _parse_response(data)
