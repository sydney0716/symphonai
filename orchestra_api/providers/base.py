"""The ModelProvider contract every API-key backed agent backend implements.

Implementations live under `orchestra_api.providers`. `FakeModelProvider` is
the only fully working one so far; `OpenAIProvider`, `AnthropicProvider`,
`GeminiProvider`, and `OpenAICompatibleProvider` are interface-conforming
placeholders whose `create_response` raises `NotImplementedError` -- see
`docs/orchestra-api-runtime.md`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from orchestra_api.models import ModelRequest, ModelResponse


class ModelProvider(ABC):
    """Abstract contract for calling a language model to get its next turn."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this provider, e.g. "fake", "openai"."""

    @abstractmethod
    def create_response(self, request: ModelRequest) -> ModelResponse:
        """Given the conversation so far, return the model's next turn."""
