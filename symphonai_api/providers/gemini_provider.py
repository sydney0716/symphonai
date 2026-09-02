"""Gemini-backed ModelProvider: calls the real Generative Language API.

Gemini is the one provider whose wire format differs structurally from the
other two, so this module does more translation than
`anthropic_provider` or `openai_provider`:

- Messages are `contents[]` of `parts[]`, and the assistant role is called
  `"model"`, not `"assistant"`.
- A system prompt is a separate top-level `system_instruction`, not a
  message.
- Tool calls are `functionCall` parts and tool results are
  `functionResponse` parts carried on a **user**-role message; Gemini has
  no dedicated tool role.
- Gemini function calls now usually carry an id. Responses that omit one
  still get a synthesized fallback id so the rest of `symphonai_api`
  (which keys tool results by `tool_call_id`) works unchanged, and we
  resolve the function name back from that id on the way out.
- Tool parameter schemas must be sanitized into Gemini's restricted JSON
  Schema dialect first -- see `symphonai_api.gemini_schema`.

Standard library only (`urllib.request`), no new dependency.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator

from symphonai_api.cancellation import CancellationToken
from symphonai_api.gemini_schema import sanitize_for_gemini
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

API_KEY_ENV_VAR = "GEMINI_API_KEY"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# A rolling alias, not a pinned version, and deliberately so: an earlier
# default (`gemini-2.0-flash`) was retired server-side and every call with it
# began failing with HTTP 404. An alias cannot rot that way. Pin an explicit
# version via `model=` when reproducibility matters more than staying current.
#
# The *lite* alias specifically, because a default should be the cheapest
# thing that actually works. Verified live that flash-lite still drives a
# full multi-turn tool loop (list_files -> read_file -> answer), which is the
# capability this runtime's subagents depend on -- a cheaper model would not
# be worth much if it could not call tools. Step up to `gemini-flash-latest`
# or a pro model via `model=` when a task needs more capability.
DEFAULT_MODEL = "gemini-flash-lite-latest"


def _synthesize_tool_call_id() -> str:
    """Build a fallback canonical id for Gemini functionCalls that omit one.

    Position-based ids (`gemini-{index}-{name}`) collided across turns: the
    same function at the same part index in a later turn produced the same
    id, so two distinct calls in one conversation shared a `tool_call_id`.
    A fresh id per call keeps `ToolResult.tool_call_id` unambiguous.
    """
    return new_id("call")


def _tool_call_names(messages: list[Message]) -> dict[str, str]:
    """Map tool_call_id -> function name, by scanning earlier assistant turns.

    Needed because a `functionResponse` part must name the function it
    answers, but our `ToolResult` only carries the originating call's id.
    """
    names: dict[str, str] = {}
    for message in messages:
        for tool_call in message.tool_calls:
            names[tool_call.id] = tool_call.name
    return names


def _gemini_parts(message: Message) -> list[dict[str, Any]]:
    """Message content as Gemini parts, text runs merged."""
    parts: list[dict[str, Any]] = []
    text_run = ""
    for block in message.content:
        if isinstance(block, TextBlock):
            text_run += block.text
            continue
        if text_run:
            parts.append({"text": text_run})
            text_run = ""
        if isinstance(block, (ImageBlock, DocumentBlock)):
            parts.append(
                {
                    "inlineData": {
                        "mimeType": block.media_type,
                        "data": block.data,
                    }
                }
            )
    if text_run:
        parts.append({"text": text_run})
    return parts


def _build_contents(messages: list[Message]) -> list[dict[str, Any]]:
    names = _tool_call_names(messages)
    contents: list[dict[str, Any]] = []

    for message in messages:
        if message.role == Role.SYSTEM:
            continue  # hoisted into system_instruction by the caller

        if message.role == Role.TOOL:
            result = message.tool_result
            if result is None:
                continue
            payload = (
                {"output": result.content} if result.ok else {"error": result.error or "tool failed"}
            )
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": names.get(result.tool_call_id, result.tool_call_id),
                                "response": payload,
                            }
                        }
                    ],
                }
            )
            continue

        parts: list[dict[str, Any]] = []
        if has_attachments(message):
            parts.extend(_gemini_parts(message))
        elif message.text:
            parts.append({"text": message.text})
        for tool_call in message.tool_calls:
            function_call = {"name": tool_call.name, "args": tool_call.arguments}
            part = {"functionCall": function_call}
            if "thoughtSignature" in tool_call.provider_metadata:
                function_call["id"] = tool_call.vendor_id or tool_call.id
                part["thoughtSignature"] = tool_call.provider_metadata["thoughtSignature"]
            parts.append(part)
        if not parts:
            continue
        contents.append({"role": "model" if message.role == Role.ASSISTANT else "user", "parts": parts})

    return contents


def _build_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap our provider-shaped tool schemas as Gemini functionDeclarations.

    Accepts either the Anthropic-ish `{name, description, input_schema}`
    or OpenAI-ish `{type: function, function: {...}}` shape that
    `tool_schema.to_provider_tool_schema` may hand us, plus Gemini's own
    `{name, description, parameters}`, and normalizes all of them.
    """
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        spec = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = spec.get("name")
        if not name:
            continue
        raw_params = spec.get("parameters") or spec.get("input_schema") or spec.get("properties")
        declarations.append(
            {
                "name": name,
                "description": spec.get("description", ""),
                "parameters": sanitize_for_gemini(raw_params if isinstance(raw_params, dict) else None),
            }
        )
    return [{"functionDeclarations": declarations}] if declarations else []


def _build_request_body(request: ModelRequest) -> dict[str, Any]:
    reject_system_attachments(request.messages)
    body: dict[str, Any] = {"contents": _build_contents(request.messages)}

    system_parts = [m.text for m in request.messages if m.role == Role.SYSTEM and m.text]
    if system_parts:
        body["system_instruction"] = {"parts": [{"text": "\n".join(system_parts)}]}

    if request.tools:
        tools = _build_tools(request.tools)
        if tools:
            body["tools"] = tools

    generation_config: dict[str, Any] = {}
    if request.max_tokens is not None:
        generation_config["maxOutputTokens"] = request.max_tokens
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if generation_config:
        body["generationConfig"] = generation_config

    return body


def _parse_response(data: dict[str, Any]) -> ModelResponse:
    candidates = data.get("candidates") or []
    if not candidates:
        # A prompt blocked by safety filters comes back with no candidates.
        feedback = data.get("promptFeedback", {})
        raise ProviderError(f"Gemini returned no candidates (promptFeedback: {feedback})")

    candidate = candidates[0]
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for part in candidate.get("content", {}).get("parts", []) or []:
        if "text" in part:
            text_parts.append(part["text"] or "")
        function_call = part.get("functionCall")
        if isinstance(function_call, dict) and function_call.get("name"):
            name = function_call["name"]
            provider_metadata = {}
            if "thoughtSignature" in part:
                provider_metadata["thoughtSignature"] = part["thoughtSignature"]
            vendor_id = function_call.get("id") or None
            tool_calls.append(
                ToolCall(
                    id=vendor_id or _synthesize_tool_call_id(),
                    name=name,
                    arguments=function_call.get("args") or {},
                    provider_metadata=provider_metadata,
                    vendor_id=vendor_id,
                )
            )

    usage_raw = data.get("usageMetadata", {})
    return ModelResponse(
        message=Message(role=Role.ASSISTANT, content="".join(text_parts), tool_calls=tool_calls),
        usage=Usage(
            input_tokens=usage_raw.get("promptTokenCount", 0),
            output_tokens=usage_raw.get("candidatesTokenCount", 0),
        ),
        stop_reason=candidate.get("finishReason") or "STOP",
    )


def _gemini_stream_chunks(
    lines: Iterator[bytes], *, api_key: str
) -> Iterator[StreamChunk]:
    """Map Gemini streamGenerateContent SSE payloads to runtime chunks."""
    usage = Usage()
    finish_reason = "STOP"
    function_call_index = 0
    for _, payload in sse_events(lines):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Gemini API stream returned invalid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise ProviderError("Gemini API stream returned non-object JSON")

        feedback = data.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason") is not None:
            reason = redact_secret(str(feedback["blockReason"]), api_key)
            raise ProviderError(f"Gemini prompt was blocked: {reason}")

        usage_raw = data.get("usageMetadata")
        if isinstance(usage_raw, dict):
            usage = Usage(
                input_tokens=usage_raw.get("promptTokenCount", 0),
                output_tokens=usage_raw.get("candidatesTokenCount", 0),
            )
        candidates = data.get("candidates") or []
        if not candidates:
            continue
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise ProviderError("Gemini API stream returned invalid candidate")
        if candidate.get("finishReason") is not None:
            finish_reason = candidate["finishReason"]
        content = candidate.get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(parts, list):
            raise ProviderError("Gemini API stream returned invalid content parts")
        for part in parts:
            if not isinstance(part, dict):
                continue
            if "text" in part:
                yield TextDelta(part["text"] or "")
            function_call = part.get("functionCall")
            if not isinstance(function_call, dict) or not function_call.get("name"):
                continue
            provider_metadata = (
                {"thoughtSignature": part["thoughtSignature"]}
                if "thoughtSignature" in part
                else None
            )
            # The vendor id goes back on the wire as this call's id
            # (`_build_request_body`); dropping it here would answer Gemini
            # with an id it never issued.
            vendor_id = function_call.get("id") or None
            yield ToolCallDelta(
                function_call_index,
                id=vendor_id,
                name=function_call["name"],
                arguments_fragment=json.dumps(function_call.get("args") or {}),
                vendor_id=vendor_id,
                provider_metadata=provider_metadata,
            )
            function_call_index += 1
    yield StreamCompleted(
        ModelResponse(
            Message(role=Role.ASSISTANT), usage=usage, stop_reason=finish_reason
        )
    )


@dataclass
class GeminiProvider(ModelProvider):
    """Calls the real Gemini Generative Language API.

    The API key is never accepted as a constructor argument or stored on
    this object -- it is read from `GEMINI_API_KEY` fresh on every call,
    sent as the `x-goog-api-key` header (never in the URL query string,
    where it could leak into logs), never logged, and never included in
    any error message.
    """

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 30.0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def wire_format(self) -> int:
        return 3

    @staticmethod
    def is_configured() -> bool:
        """Whether GEMINI_API_KEY is set and non-empty. Never reads/logs its value."""
        return bool(os.environ.get(API_KEY_ENV_VAR, "").strip())

    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise ProviderError(f"{API_KEY_ENV_VAR} is not set")

        model = request.model if request.model is not None else self.model
        body = _build_request_body(request)
        http_request = urllib.request.Request(
            f"{self.base_url}/models/{model}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "x-goog-api-key": api_key,
                "content-type": "application/json",
            },
        )
        raw = read_with_retry(
            http_request,
            timeout=self.timeout_seconds,
            max_attempts=self.max_attempts,
            api_key=api_key,
            operation="Gemini API",
            cancel=cancel,
            call_class=request.call_class,
        )
        data = parse_json_object(raw, "Gemini API")

        return _parse_response(data)

    def create_response_stream(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> Iterator[StreamChunk]:
        api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise ProviderError(f"{API_KEY_ENV_VAR} is not set")

        model = request.model if request.model is not None else self.model
        body = _build_request_body(request)
        http_request = urllib.request.Request(
            f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "x-goog-api-key": api_key,
                "content-type": "application/json",
            },
        )
        lines = open_stream_with_retry(
            http_request,
            timeout=self.timeout_seconds,
            max_attempts=self.max_attempts,
            api_key=api_key,
            operation="Gemini API",
            cancel=cancel,
        )
        yield from _gemini_stream_chunks(lines, api_key=api_key)
