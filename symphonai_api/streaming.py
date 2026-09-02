"""Provider-neutral response streaming, assembly, and SSE transport helpers."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator

from symphonai_api.cancellation import CancellationToken
from symphonai_api.identity import new_id
from symphonai_api.models import Message, ModelResponse, ToolCall
from symphonai_api.providers.base import ProviderError
from symphonai_api.retry import (
    _is_retryable_status,
    _is_retryable_url_error,
    _wait_before_retry,
    redact_secret,
)


@dataclass(frozen=True)
class TextDelta:
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


StreamChunk = TextDelta | ToolCallDelta | StreamCompleted


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
        self._tool_calls: dict[int, _PartialToolCall] = {}
        self._completed: ModelResponse | None = None

    def add(self, chunk: StreamChunk) -> None:
        if isinstance(chunk, TextDelta):
            self._text.append(chunk.text)
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
        if not tool_calls:
            tool_calls = [
                self._finish_tool_call(index, self._tool_calls[index])
                for index in sorted(self._tool_calls)
            ]
        message = replace(terminal.message, content=content, tool_calls=tool_calls)
        return replace(terminal, message=message)

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
    max_attempts: int,
    api_key: str,
    operation: str,
    cancel: CancellationToken | None = None,
) -> Iterator[bytes]:
    """Yield response lines, retrying only before the first yielded line."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    deadline = time.monotonic() + timeout
    yielded = False
    for attempt in range(1, max_attempts + 1):
        if cancel is not None:
            cancel.raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderError(
                f"{operation} exceeded its {timeout}s overall timeout"
            )
        try:
            with urllib.request.urlopen(request, timeout=remaining) as response:
                for line in response:
                    if cancel is not None:
                        cancel.raise_if_cancelled()
                    yielded = True
                    yield line
                return
        except urllib.error.HTTPError as exc:
            reason = redact_secret(str(exc), api_key)
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if not yielded and _is_retryable_status(exc.code) and _wait_before_retry(
                attempt=attempt,
                max_attempts=max_attempts,
                deadline=deadline,
                retry_after=retry_after,
                cancel=cancel,
            ):
                continue
            raise ProviderError(
                f"{operation} stream returned HTTP {exc.code}: {reason}"
            ) from None
        except urllib.error.URLError as exc:
            reason = redact_secret(str(exc.reason), api_key)
            if not yielded and _is_retryable_url_error(exc) and _wait_before_retry(
                attempt=attempt,
                max_attempts=max_attempts,
                deadline=deadline,
                retry_after=None,
                cancel=cancel,
            ):
                continue
            raise ProviderError(f"{operation} stream request failed: {reason}") from None
        except (TimeoutError, http.client.IncompleteRead, ConnectionError) as exc:
            reason = redact_secret(str(exc), api_key)
            if not yielded and _wait_before_retry(
                attempt=attempt,
                max_attempts=max_attempts,
                deadline=deadline,
                retry_after=None,
                cancel=cancel,
            ):
                continue
            raise ProviderError(f"{operation} stream failed: {reason}") from None
    raise AssertionError("stream retry loop exhausted without returning or raising")


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
