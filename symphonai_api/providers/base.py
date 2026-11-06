"""The ModelProvider contract every API-key backed agent backend implements.

Implementations live under `symphonai_api.providers`: `FakeModelProvider`,
`OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`, and
`OpenAICompatibleProvider`. See `docs/symphonai-api-runtime.md` for the
wire-format split each one uses.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator

from symphonai_api.cancellation import CancellationToken
from symphonai_api.models import ModelRequest, ModelResponse, ToolResult

if TYPE_CHECKING:
    from symphonai_api.streaming import StreamChunk


class ProviderError(Exception):
    """Raised by a real ModelProvider when a call to its vendor API fails.

    Covers both HTTP-level failures (non-2xx responses) and transport-level
    failures (network errors, timeouts). The message must never include the
    API key or any request header -- only vendor-safe diagnostic text.
    """


class ProviderUnavailableError(ProviderError):
    """Raised when a retryable provider failure exhausts its retries."""

    def __init__(self, message: str, *, status: int | None) -> None:
        super().__init__(message)
        self.status = status


class ContextLengthExceededError(ProviderError):
    """Raised when a vendor rejects a request that exceeds its context window."""

    def __init__(
        self,
        message: str,
        *,
        actual_tokens: int | None = None,
        limit_tokens: int | None = None,
        request_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.actual_tokens = actual_tokens
        self.limit_tokens = limit_tokens
        self.request_tokens = request_tokens


def tool_result_text(result: ToolResult) -> str:
    """Return tool output and failure details in the order useful to a model."""
    if result.ok:
        return result.content
    if result.error and result.content:
        return f"{result.error}\n{result.content}"
    return result.error or result.content


def parse_json_object(raw: bytes, operation: str) -> dict[str, Any]:
    """Decode a successful vendor response as a JSON object.

    Real providers expose malformed response bodies through ProviderError,
    never through JSONDecodeError, UnicodeDecodeError, or an object-only
    parser's AttributeError.
    """
    try:
        text = raw.decode("utf-8")
        data = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderError(f"{operation} returned invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ProviderError(f"{operation} returned non-object JSON")
    return data


class ModelProvider(ABC):
    """Abstract contract for calling a language model to get its next turn."""

    model: str | None = None
    base_url: str | None = None
    fallback_models: tuple[str, ...] = ()

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this provider, e.g. "fake", "openai"."""

    @property
    @abstractmethod
    def wire_format(self) -> int:
        """1=openai wire format, 2=anthropic wire format, 3=gemini wire format, 4=other/unclassified."""

    @abstractmethod
    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        """Given the conversation so far, return the model's next turn."""

    def create_response_stream(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> Iterator[StreamChunk]:
        """Return this response as a stream, defaulting to one completion."""

        from symphonai_api.streaming import StreamCompleted

        response = self.create_response(request, cancel=cancel)
        yield StreamCompleted(response)


def _usage_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _synthesize_tool_call_id() -> str:
    from symphonai_api.identity import new_id
    return new_id("call")


class RememberedRejections:
    def __init__(self):
        self._values = set()
        self._lock = threading.Lock()

    def __contains__(self, key):
        with self._lock:
            return key in self._values

    def remember(self, key, error, rejected):
        if not rejected(error):
            return False
        with self._lock:
            self._values.add(key)
        return True


def parse_json_reply(text: str):
    source = text.strip().replace("\r\n", "\n")
    lines = source.split("\n")
    if len(lines) >= 3 and lines[0] in ("```", "```json") and lines[-1] == "```":
        source = "\n".join(lines[1:-1])
    return json.loads(source)


def bounded_call(function, *, timeout: float, cancel=None, name="symphonai-model-call"):
    import queue
    import time
    token = cancel.child(deadline_seconds=timeout) if cancel is not None else CancellationToken(deadline_seconds=timeout)
    completed = queue.Queue(maxsize=1)
    def invoke():
        try:
            completed.put((True, function(token)))
        except Exception as exc:
            completed.put((False, exc))
    deadline = time.monotonic() + timeout
    threading.Thread(target=invoke, name=name, daemon=True).start()
    try:
        while True:
            if cancel is not None:
                cancel.raise_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("model call deadline exceeded")
            try:
                success, value = completed.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if not success:
                raise value
            return value
    finally:
        token.cancel()
        token.close()


def read_json(opener, request, limit: int, *, timeout=10, **json_options):
    """Read one size-bounded JSON object through the caller's guarded opener."""
    with opener.open(request, timeout=timeout) as response:
        if not 200 <= response.status < 300:
            raise ValueError("unsuccessful JSON response")
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("JSON response exceeds size limit")
    try:
        value = json.loads(raw.decode("utf-8"), **json_options)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("invalid JSON response") from None
    if not isinstance(value, dict):
        raise ValueError("invalid JSON response")
    return value
