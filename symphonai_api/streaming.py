"""Provider-neutral response streaming, assembly, and SSE transport helpers."""

from __future__ import annotations

from symphonai_api.retry import retry_decision

import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator

from symphonai_api.call_class import CallClass
from symphonai_api.cancellation import CancellationToken
from symphonai_api.identity import new_id
from symphonai_api.models import ModelResponse, ToolCall
from symphonai_api.providers.base import (
    ContextLengthExceededError,
    ProviderError,
    ProviderUnavailableError,
)
from symphonai_api.retry import context_token_counts, is_context_length_error, is_retryable_status, is_retryable_url_error, redacted_error_detail, wait_before_retry, redact_secret


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments_fragment: str = ""
    vendor_id: str | None = None
    provider_metadata: dict | None = None


@dataclass(frozen=True)
class StreamCompleted:
    response: ModelResponse


class StreamEventError(ProviderError):
    """A vendor error event, retaining its code for retry classification."""

    def __init__(self, message: str, code: str | None = None, vendor: str = "openai") -> None:
        super().__init__(message)
        self.code = code
        self.vendor = vendor


class _StreamUnavailableError(ProviderUnavailableError):
    def __init__(self, message: str, *, status: int | None, retry_after: str | None = None) -> None:
        super().__init__(message, status=status)
        self.retry_after = retry_after


StreamChunk = TextDelta | ReasoningDelta | ToolCallDelta | StreamCompleted


def retry_stream_chunks(
    factory, *, timeout: float, max_attempts: int,
    fallback_configured: bool = False,
    cancel: CancellationToken | None = None,
    call_class: CallClass = CallClass.FOREGROUND,
    operation: str = "stream",
):
    """Retry transient stream failures until answer text has been yielded."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    permanent_codes = {
        "openai": {"invalid_request_error", "context_length_exceeded", "invalid_prompt"},
        "anthropic": {"invalid_request_error", "authentication_error", "permission_error",
                      "not_found_error", "request_too_large"},
        "gemini": {"INVALID_ARGUMENT", "FAILED_PRECONDITION", "PERMISSION_DENIED",
                   "NOT_FOUND", "UNAUTHENTICATED"},
    }
    overload_codes = {
        "anthropic": {"overloaded_error": 529, "rate_limit_error": 429},
        "openai": {"rate_limit_exceeded": 429},
        "gemini": {"RESOURCE_EXHAUSTED": 429},
    }
    deadline = time.monotonic() + timeout
    for attempt in range(1, max_attempts + 1):
        if cancel is not None:
            cancel.raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderError(f"stream exceeded its {timeout}s overall timeout")
        chunks: list[StreamChunk] = []
        output_yielded = False
        try:
            try:
                for chunk in factory(remaining):
                    if isinstance(chunk, (TextDelta, ReasoningDelta)):
                        if not output_yielded:
                            yield from chunks
                            chunks.clear()
                            output_yielded = True
                        yield chunk
                    elif output_yielded:
                        yield chunk
                    else:
                        chunks.append(chunk)
                if not output_yielded:
                    yield from chunks
                return
            except StreamEventError as exc:
                if exc.code in permanent_codes.get(exc.vendor, permanent_codes["anthropic"]):
                    chunks.clear()
                    raise ProviderError(str(exc)) from None
                status = overload_codes.get(exc.vendor, {}).get(exc.code)
                raise _StreamUnavailableError(str(exc), status=status) from None
        except ProviderUnavailableError as exc:
            if output_yielded:
                yield from chunks
                raise ProviderError(str(exc)) from None
            retry_after = getattr(exc, "retry_after", None)
            _, attempt_limit, retry_after = retry_decision(
                exc.status, str(exc), retry_after, attempt, call_class, fallback_configured,
                max_attempts=max_attempts, operation=operation,
            )
            if wait_before_retry(attempt=attempt, max_attempts=attempt_limit,
                                 deadline=deadline, retry_after=retry_after,
                                 cancel=cancel):
                continue
            message = f"rate limit persisted after {attempt} attempts: {exc}" if exc.status == 429 else str(exc)
            raise ProviderUnavailableError(message, status=exc.status) from None
        except ProviderError:
            yield from chunks
            raise
    raise AssertionError("stream retry loop exhausted without returning or raising")


@dataclass
class _PartialToolCall:
    id: str | None = None
    name: str | None = None
    arguments_fragments: list[str] = field(default_factory=list)
    vendor_id: str | None = None
    provider_metadata: dict = field(default_factory=dict)


class StreamAssembler:
    """Accumulate private deltas into one complete model response."""

    def __init__(self) -> None:
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._tool_calls: dict[int, _PartialToolCall] = {}
        self._completed: ModelResponse | None = None

    def add(self, chunk: StreamChunk) -> None:
        if isinstance(chunk, TextDelta):
            self._text.append(chunk.text)
            return
        if isinstance(chunk, ReasoningDelta):
            self._reasoning.append(chunk.text)
            return
        if isinstance(chunk, ToolCallDelta):
            partial = self._tool_calls.setdefault(chunk.index, _PartialToolCall())
            if partial.id is None and chunk.id is not None:
                partial.id = chunk.id
            if partial.name is None and chunk.name is not None:
                partial.name = chunk.name
            if partial.vendor_id is None and chunk.vendor_id is not None:
                partial.vendor_id = chunk.vendor_id
            partial.arguments_fragments.append(chunk.arguments_fragment)
            if chunk.provider_metadata is not None:
                partial.provider_metadata.update(chunk.provider_metadata)
            return
        self._completed = chunk.response

    def finish(self) -> ModelResponse:
        if self._completed is None:
            raise ProviderError("stream ended without a completion event")
        terminal = self._completed
        content = terminal.message.content
        if not content:
            content = "".join(self._text)
        tool_calls = terminal.message.tool_calls
        cut_tool_call_id = terminal.cut_tool_call_id
        if not tool_calls:
            indices = sorted(self._tool_calls)
            output_limited = terminal.stop_reason in {"max_tokens", "length", "MAX_TOKENS"}
            tool_calls = []
            for index in indices:
                partial = self._tool_calls[index]
                if output_limited and index == indices[-1]:
                    tool_call = ToolCall(
                        id=partial.id or new_id("call"),
                        name=partial.name or "",
                        arguments={},
                        provider_metadata=dict(partial.provider_metadata),
                        vendor_id=partial.vendor_id,
                    )
                    cut_tool_call_id = tool_call.id
                else:
                    tool_call = self._finish_tool_call(index, partial)
                tool_calls.append(tool_call)
        message = replace(
            terminal.message,
            content=content,
            tool_calls=tool_calls,
            reasoning=terminal.message.reasoning or "".join(self._reasoning),
        )
        return replace(terminal, message=message, cut_tool_call_id=cut_tool_call_id)

    @staticmethod
    def _finish_tool_call(index: int, partial: _PartialToolCall) -> ToolCall:
        raw_arguments = "".join(partial.arguments_fragments)
        try:
            arguments = {} if raw_arguments == "" else json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            name = partial.name or "<unknown>"
            raise ProviderError(
                f"streamed tool {name!r} at index {index} returned invalid "
                f"arguments JSON: {exc}"
            ) from None
        if not isinstance(arguments, dict):
            name = partial.name or "<unknown>"
            raise ProviderError(
                f"streamed tool {name!r} at index {index} returned non-object arguments"
            )
        return ToolCall(
            id=partial.id or new_id("call"),
            name=partial.name or "",
            arguments=arguments,
            provider_metadata=dict(partial.provider_metadata),
            vendor_id=partial.vendor_id,
        )


def open_stream_with_retry(
    request: urllib.request.Request,
    *,
    timeout: float,
    max_attempts: int = 1,
    api_key: str,
    operation: str,
    fallback_configured: bool = False,
    cancel: CancellationToken | None = None,
) -> Iterator[bytes]:
    """Yield one attempt; legacy retry keywords remain accepted for callers.

    Only retry_stream_chunks applies the retry budget and fallback policy.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    if cancel is not None:
        cancel.raise_if_cancelled()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for line in response:
                if cancel is not None:
                    cancel.raise_if_cancelled()
                yield line
    except urllib.error.HTTPError as exc:
        try:
            raw_detail = exc.read()
            if raw_detail:
                raw_detail.decode("utf-8")
            detail = redacted_error_detail(raw_detail, api_key) if raw_detail else ""
        except Exception:
            raw_detail = b""
            detail = ""
        reason = detail or redact_secret(str(exc), api_key)
        if 400 <= exc.code <= 499 and not is_retryable_status(exc.code) and is_context_length_error(raw_detail):
            actual_tokens, limit_tokens = context_token_counts(raw_detail)
            raise ContextLengthExceededError(
                f"{operation} stream returned HTTP {exc.code}: request exceeded the model context window",
                actual_tokens=actual_tokens, limit_tokens=limit_tokens,
            ) from None
        message = f"{operation} stream returned HTTP {exc.code}: {reason}"
        if is_retryable_status(exc.code):
            raise _StreamUnavailableError(
                message, status=exc.code,
                retry_after=exc.headers.get("Retry-After") if exc.headers else None,
            ) from None
        raise ProviderError(message) from None
    except urllib.error.URLError as exc:
        message = f"{operation} stream request failed: {redact_secret(str(exc.reason), api_key)}"
        if is_retryable_url_error(exc):
            raise ProviderUnavailableError(message, status=None) from None
        raise ProviderError(message) from None
    except (TimeoutError, http.client.IncompleteRead, ConnectionError) as exc:
        raise ProviderUnavailableError(
            f"{operation} stream failed: {redact_secret(str(exc), api_key)}", status=None,
        ) from None


def sse_events(lines: Iterable[bytes]) -> Iterator[tuple[str, str]]:
    """Parse raw server-sent-event lines into event-name and data pairs."""

    event_name = "message"
    data: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data:
                yield event_name, "\n".join(data)
            event_name = "message"
            data = []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event_name = line.removeprefix("event:").removeprefix(" ")
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").removeprefix(" "))
    if data:
        yield event_name, "\n".join(data)
