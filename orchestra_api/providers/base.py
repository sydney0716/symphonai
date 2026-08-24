"""The ModelProvider contract every API-key backed agent backend implements.

Implementations live under `orchestra_api.providers`: `FakeModelProvider`,
`OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`, and
`OpenAICompatibleProvider`. See `docs/orchestra-api-runtime.md` for the
wire-format split each one uses.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from orchestra_api.models import ModelRequest, ModelResponse


class ProviderError(Exception):
    """Raised by a real ModelProvider when a call to its vendor API fails.

    Covers both HTTP-level failures (non-2xx responses) and transport-level
    failures (network errors, timeouts). The message must never include the
    API key or any request header -- only vendor-safe diagnostic text.
    """


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

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this provider, e.g. "fake", "openai"."""

    @property
    @abstractmethod
    def wire_format(self) -> int:
        """1=openai wire format, 2=anthropic wire format, 3=gemini wire format, 4=other/unclassified."""

    @abstractmethod
    def create_response(self, request: ModelRequest) -> ModelResponse:
        """Given the conversation so far, return the model's next turn."""
