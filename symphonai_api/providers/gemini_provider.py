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

from symphonai_api.providers.base import _usage_count, _synthesize_tool_call_id, RememberedRejections

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator

from symphonai_api.cancellation import CancellationToken
from symphonai_api.gemini_schema import sanitize_for_gemini
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
)
from symphonai_api.providers.base import ModelProvider, ProviderError, parse_json_object, tool_result_text
from symphonai_api.retry import DEFAULT_MAX_ATTEMPTS, read_with_retry, redact_secret
from symphonai_api.streaming import (
    StreamChunk,
    StreamCompleted,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    open_stream_with_retry,
    retry_stream_chunks,
    StreamEventError,
    sse_events,
)

API_KEY_ENV_VAR = "GEMINI_API_KEY"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# A rolling alias, not a pinned version, and deliberately so: an earlier
# default (`gemini-2.0-flash`) was retired server-side and every call with it
# began failing with HTTP 404. An alias cannot rot that way. Pin an explicit
# version via `model=` when reproducibility matters more than staying current.
#
# The leader plans and delegates, so use the pro alias by default.
DEFAULT_MODEL = "gemini-pro-latest"
_THOUGHTS_DISABLED = RememberedRejections()


def _thoughts_key(base_url: str, model: str) -> tuple[str, str]:
    return base_url, model




def _thoughts_rejected(error: ProviderError) -> bool:
    return "includethoughts" in str(error).casefold()





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
                {"output": result.content}
                if result.ok
                else {"error": tool_result_text(result) or "tool failed"}
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


def _build_request_body(
    request: ModelRequest, model: str | None = None, *, include_thoughts: bool = True,
) -> dict[str, Any]:
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
    effort = None if model is None else resolve_effort(3, model, request.effort)
    if isinstance(effort, str):
        generation_config["thinkingConfig"] = {"thinkingLevel": effort}
    elif isinstance(effort, int):
        generation_config["thinkingConfig"] = {"thinkingBudget": effort}
    if include_thoughts:
        generation_config.setdefault("thinkingConfig", {})["includeThoughts"] = True
    if generation_config:
        body["generationConfig"] = generation_config

    return body


def _parse_response(data: dict[str, Any]) -> ModelResponse:
    error = data.get("error")
    if isinstance(error, dict):
        detail = error.get("message")
        serialized_error = json.dumps(error, ensure_ascii=False)
        if not isinstance(detail, str):
            detail = serialized_error
        elif "includethoughts" in serialized_error.casefold() and "includethoughts" not in detail.casefold():
            detail = f"{detail} ({serialized_error})"
        raise ProviderError(redact_secret(detail, os.environ.get(API_KEY_ENV_VAR, "").strip()))
    candidates = data.get("candidates") or []
    if not candidates:
        # A prompt blocked by safety filters comes back with no candidates.
        feedback = data.get("promptFeedback", {})
        raise ProviderError(f"Gemini returned no candidates (promptFeedback: {feedback})")

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason")
    refusal_category = None
    if finish_reason is not None:
        if finish_reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
                      "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
                      "ESCALATION", "PUP_LIMITED_DISABLED"}:
            refusal_category = finish_reason.lower()
        elif finish_reason not in {"STOP", "MAX_TOKENS"}:
            raise ProviderError(f"Gemini stopped generating: {finish_reason}")
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for part in candidate.get("content", {}).get("parts", []) or []:
        if "text" in part:
            target = reasoning_parts if part.get("thought") is True else text_parts
            target.append(part["text"] or "")
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

    usage_raw = data.get("usageMetadata")
    if not isinstance(usage_raw, dict):
        usage_raw = {}
    cache_read_tokens = _usage_count(usage_raw.get("cachedContentTokenCount"))
    return ModelResponse(
        message=Message(
            role=Role.ASSISTANT,
            content="".join(text_parts),
            tool_calls=tool_calls,
            reasoning="".join(reasoning_parts),
        ),
        usage=Usage(
            input_tokens=_usage_count(usage_raw.get("promptTokenCount")),
            output_tokens=_usage_count(usage_raw.get("candidatesTokenCount")),
            cache_read_tokens=cache_read_tokens,
        ),
        stop_reason="refusal" if refusal_category else finish_reason or "STOP",
        refusal_category=refusal_category,
    )


def _gemini_stream_chunks(
    lines: Iterator[bytes], *, api_key: str
) -> Iterator[StreamChunk]:
    """Map Gemini streamGenerateContent SSE payloads to runtime chunks."""
    usage = Usage()
    finish_reason = "STOP"
    refusal_category = None
    function_call_index = 0
    for _, payload in sse_events(lines):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Gemini API stream returned invalid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise ProviderError("Gemini API stream returned non-object JSON")

        error = data.get("error")
        if isinstance(error, dict):
            detail = error.get("message")
            serialized_error = json.dumps(error, ensure_ascii=False)
            if not isinstance(detail, str):
                detail = serialized_error
            elif "includethoughts" in serialized_error.casefold() and "includethoughts" not in detail.casefold():
                detail = f"{detail} ({serialized_error})"
            raise StreamEventError(
                redact_secret(detail, api_key),
                code=error.get("status") if isinstance(error.get("status"), str) else None,
                vendor="gemini",
            )

        feedback = data.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason") is not None:
            reason = redact_secret(str(feedback["blockReason"]), api_key)
            raise ProviderError(f"Gemini prompt was blocked: {reason}")

        usage_raw = data.get("usageMetadata")
        if isinstance(usage_raw, dict):
            usage = Usage(
                input_tokens=_usage_count(usage_raw.get("promptTokenCount")),
                output_tokens=_usage_count(usage_raw.get("candidatesTokenCount")),
                cache_read_tokens=_usage_count(
                    usage_raw.get("cachedContentTokenCount")
                ),
            )
        candidates = data.get("candidates") or []
        if not candidates:
            continue
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise ProviderError("Gemini API stream returned invalid candidate")
        if candidate.get("finishReason") is not None:
            finish_reason = candidate["finishReason"]
            if finish_reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
                      "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
                      "ESCALATION", "PUP_LIMITED_DISABLED"}:
                refusal_category = finish_reason.lower()
            elif finish_reason not in {"STOP", "MAX_TOKENS"}:
                raise ProviderError(f"Gemini stopped generating: {finish_reason}")
        content = candidate.get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(parts, list):
            raise ProviderError("Gemini API stream returned invalid content parts")
        for part in parts:
            if not isinstance(part, dict):
                continue
            if "text" in part:
                if part.get("thought") is True:
                    yield ReasoningDelta(part["text"] or "")
                else:
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
            Message(role=Role.ASSISTANT), usage=usage,
            stop_reason="refusal" if refusal_category else finish_reason,
            refusal_category=refusal_category,
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
    timeout_seconds: float = 600.0
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
        key = _thoughts_key(self.base_url, model)
        include_thoughts = not (key in _THOUGHTS_DISABLED)

        def send(include: bool) -> ModelResponse:
            body = _build_request_body(request, model, include_thoughts=include)
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
                fallback_configured=bool(self.fallback_models),
                cancel=cancel,
                call_class=request.call_class,
            )
            return _parse_response(parse_json_object(raw, "Gemini API"))

        try:
            return send(include_thoughts)
        except ProviderError as exc:
            if not include_thoughts or not _THOUGHTS_DISABLED.remember(key, exc, _thoughts_rejected):
                raise
            return send(False)

    def create_response_stream(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> Iterator[StreamChunk]:
        api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise ProviderError(f"{API_KEY_ENV_VAR} is not set")

        model = request.model if request.model is not None else self.model
        key = _thoughts_key(self.base_url, model)
        include_thoughts = not (key in _THOUGHTS_DISABLED)

        def stream(include: bool) -> Iterator[StreamChunk]:
            body = _build_request_body(request, model, include_thoughts=include)
            http_request = urllib.request.Request(
                f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse",
                data=json.dumps(body).encode("utf-8"),
                method="POST",
                headers={
                    "x-goog-api-key": api_key,
                    "content-type": "application/json",
                },
            )
            def attempt(remaining):
                lines = open_stream_with_retry(
                    http_request, timeout=remaining, api_key=api_key,
                    operation="Gemini API", cancel=cancel,
                )
                return _gemini_stream_chunks(lines, api_key=api_key)

            yield from retry_stream_chunks(
                attempt, timeout=self.timeout_seconds, max_attempts=self.max_attempts,
                fallback_configured=bool(self.fallback_models), cancel=cancel,
                call_class=request.call_class, operation="Gemini API",
            )

        if not include_thoughts:
            yield from stream(False)
            return
        emitted = False
        try:
            for chunk in stream(True):
                emitted = True
                yield chunk
        except ProviderError as exc:
            if emitted or not _THOUGHTS_DISABLED.remember(key, exc, _thoughts_rejected):
                raise
            yield from stream(False)
