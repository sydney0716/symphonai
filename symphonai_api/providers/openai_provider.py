"""OpenAI-backed ModelProvider: calls the real OpenAI Chat Completions API.

Maps symphonai_api's provider-agnostic Message/ToolCall/ToolResult shape
into and out of OpenAI's Chat Completions wire format
(https://platform.openai.com/docs/api-reference/chat), using only the
standard library (`urllib.request`) -- no new dependency.

`ModelRequest.tools` is passed through unmodified into the request's
`tools` field, so callers must already supply tool definitions in
OpenAI's native `{"type": "function", "function": {...}}` shape. The
standard runtime call sites prepare that shape via
`symphonai_api.tool_schema`.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator

from symphonai_api.cancellation import CancellationToken
from symphonai_api.identity import new_id
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
from symphonai_api.providers.base import ModelProvider, ProviderError, parse_json_object
from symphonai_api.retry import DEFAULT_MAX_ATTEMPTS, read_with_retry, redact_secret
from symphonai_api.streaming import (
    StreamChunk,
    StreamCompleted,
    TextDelta,
    ToolCallDelta,
    open_stream_with_retry,
    sse_events,
)

API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4-mini"

_ROLE_TO_OPENAI = {
    Role.SYSTEM: "system",
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL: "tool",
}


def _openai_content(message: Message) -> list[dict[str, Any]]:
    """Message content as OpenAI content parts, text runs merged."""
    parts: list[dict[str, Any]] = []
    text_run = ""
    for block in message.content:
        if isinstance(block, TextBlock):
            text_run += block.text
            continue
        if text_run:
            parts.append({"type": "text", "text": text_run})
            text_run = ""
        if isinstance(block, ImageBlock):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.media_type};base64,{block.data}"
                    },
                }
            )
        elif isinstance(block, DocumentBlock):
            parts.append(
                {
                    "type": "file",
                    "file": {
                        "filename": block.filename or "document.pdf",
                        "file_data": f"data:{block.media_type};base64,{block.data}",
                    },
                }
            )
    if text_run:
        parts.append({"type": "text", "text": text_run})
    return parts


def _to_openai_message(message: Message, id_map: dict[str, str]) -> dict[str, Any]:
    if message.role == Role.TOOL:
        result = message.tool_result
        assert result is not None, "tool-role Message must carry a tool_result"
        return {
            "role": "tool",
            "tool_call_id": id_map.get(result.tool_call_id, result.tool_call_id),
            "content": result.content if result.ok else (result.error or ""),
        }
    content = (
        message.text or None
        if not has_attachments(message)
        else _openai_content(message)
    )
    out: dict[str, Any] = {"role": _ROLE_TO_OPENAI[message.role], "content": content}
    if message.role == Role.ASSISTANT and message.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.vendor_id or tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in message.tool_calls
        ]
    return out


def _build_request_body(request: ModelRequest, model: str) -> dict[str, Any]:
    reject_system_attachments(request.messages)
    id_map = wire_tool_call_ids(request.messages)
    body: dict[str, Any] = {
        "model": model,
        "messages": [_to_openai_message(m, id_map) for m in request.messages],
    }
    if request.tools:
        body["tools"] = request.tools
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    return body


def _synthesize_tool_call_id() -> str:
    """Build a fallback canonical id for tool calls that arrive without one.

    Must not be position-based: the same index in a later turn would reuse
    the id, leaving two distinct calls sharing a `tool_call_id`.
    """
    return new_id("call")


def _parse_response(data: dict[str, Any]) -> ModelResponse:
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError("OpenAI API response contained no choices")
    message_raw = choices[0].get("message", {})
    tool_calls: list[ToolCall] = []
    for tc in message_raw.get("tool_calls") or []:
        function = tc.get("function", {})
        name = function.get("name", "")
        vendor_id = tc.get("id") or None
        raw_args = function.get("arguments", "{}")
        try:
            arguments = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            raise ProviderError(f"OpenAI tool call arguments were not valid JSON: {exc}") from None
        tool_calls.append(
            ToolCall(
                id=vendor_id or _synthesize_tool_call_id(),
                name=name,
                arguments=arguments,
                vendor_id=vendor_id,
            )
        )

    usage_raw = data.get("usage", {})
    message = Message(role=Role.ASSISTANT, content=message_raw.get("content") or "", tool_calls=tool_calls)
    return ModelResponse(
        message=message,
        usage=Usage(
            input_tokens=usage_raw.get("prompt_tokens", 0),
            output_tokens=usage_raw.get("completion_tokens", 0),
        ),
        stop_reason=choices[0].get("finish_reason") or "stop",
    )


def _openai_stream_chunks(
    lines: Iterator[bytes], *, operation: str, api_key: str = ""
) -> Iterator[StreamChunk]:
    """Map OpenAI SSE payloads to chunks, ignoring valid usage-only choices."""
    usage = Usage()
    finish_reason = "stop"
    for _, payload in sse_events(lines):
        if payload == "[DONE]":
            yield StreamCompleted(
                ModelResponse(
                    Message(role=Role.ASSISTANT),
                    usage=usage,
                    stop_reason=finish_reason,
                )
            )
            return
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{operation} stream returned invalid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise ProviderError(f"{operation} stream returned non-object JSON")

        # An error payload carries no choices, so without this it would be
        # skipped and the vendor's reason replaced by "stream ended without a
        # completion event".
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or f"{operation} stream returned an error"
            raise ProviderError(redact_secret(str(message), api_key))

        usage_raw = data.get("usage")
        if isinstance(usage_raw, dict):
            usage = Usage(
                input_tokens=usage_raw.get("prompt_tokens", 0),
                output_tokens=usage_raw.get("completion_tokens", 0),
            )
        # Unlike _parse_response, this accepts an empty choices list because
        # include_usage makes the final usage payload legitimately choice-less.
        choices = data.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ProviderError(f"{operation} stream returned invalid choice")
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            continue
        if isinstance(delta.get("content"), str):
            yield TextDelta(delta["content"])
        tool_calls = delta.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise ProviderError(f"{operation} stream returned invalid tool calls")
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict) or not isinstance(tool_call.get("index"), int):
                raise ProviderError(f"{operation} stream tool call omitted its index")
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                function = {}
            arguments = function.get("arguments", "")
            if arguments is None:
                arguments = ""
            if not isinstance(arguments, str):
                raise ProviderError(f"{operation} stream returned invalid tool arguments")
            yield ToolCallDelta(
                tool_call["index"],
                id=tool_call.get("id"),
                name=function.get("name"),
                arguments_fragment=arguments,
                vendor_id=tool_call.get("id"),
            )


@dataclass
class OpenAIProvider(ModelProvider):
    """Calls the real OpenAI Chat Completions API.

    The API key is never accepted as a constructor argument or stored on
    this object -- it is read from `OPENAI_API_KEY` fresh on every call,
    never logged, and never included in any error message.
    """

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 30.0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    @property
    def name(self) -> str:
        return "openai"

    @property
    def wire_format(self) -> int:
        return 1

    @staticmethod
    def is_configured() -> bool:
        """Whether OPENAI_API_KEY is set and non-empty. Never reads/logs its value."""
        return bool(os.environ.get(API_KEY_ENV_VAR, "").strip())

    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise ProviderError(f"{API_KEY_ENV_VAR} is not set")

        model = request.model if request.model is not None else self.model
        body = _build_request_body(request, model)
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
        )
        raw = read_with_retry(
            http_request,
            timeout=self.timeout_seconds,
            max_attempts=self.max_attempts,
            api_key=api_key,
            operation="OpenAI API",
            cancel=cancel,
            call_class=request.call_class,
        )
        data = parse_json_object(raw, "OpenAI API")

        return _parse_response(data)

    def create_response_stream(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> Iterator[StreamChunk]:
        api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise ProviderError(f"{API_KEY_ENV_VAR} is not set")

        model = request.model if request.model is not None else self.model
        body = _build_request_body(request, model)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
        )
        lines = open_stream_with_retry(
            http_request,
            timeout=self.timeout_seconds,
            max_attempts=self.max_attempts,
            api_key=api_key,
            operation="OpenAI API",
            cancel=cancel,
        )
        yield from _openai_stream_chunks(
            lines, operation="OpenAI API", api_key=api_key
        )
