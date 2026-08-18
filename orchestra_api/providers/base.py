"""The ModelProvider contract every API-key backed agent backend implements.

Implementations live under `orchestra_api.providers`. `FakeModelProvider`,
`AnthropicProvider`, and `OpenAIProvider` are fully working; `GeminiProvider`
and `OpenAICompatibleProvider` remain interface-conforming placeholders
whose `create_response` raises `NotImplementedError` -- see
`docs/orchestra-api-runtime.md`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from orchestra_api.models import ModelRequest, ModelResponse


class ProviderError(Exception):
    """Raised by a real ModelProvider when a call to its vendor API fails.

    Covers both HTTP-level failures (non-2xx responses) and transport-level
    failures (network errors, timeouts). The message must never include the
    API key or any request header -- only vendor-safe diagnostic text.
    """


class ModelProvider(ABC):
    """Abstract contract for calling a language model to get its next turn."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this provider, e.g. "fake", "openai"."""

    @abstractmethod
    def create_response(self, request: ModelRequest) -> ModelResponse:
        """Given the conversation so far, return the model's next turn."""
