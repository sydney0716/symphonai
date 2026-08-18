"""Interface-conforming placeholder for an Anthropic-backed ModelProvider.

This class satisfies the `ModelProvider` contract's shape -- name, config
fields, env var name -- but does not call the Anthropic API.
`create_response` raises `NotImplementedError`; real HTTP request/response
handling is deferred to a later task. See `docs/orchestra-api-runtime.md`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from orchestra_api.models import ModelRequest, ModelResponse
from orchestra_api.providers.base import ModelProvider

API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_MODEL = "claude-sonnet-4-5"


@dataclass
class AnthropicProvider(ModelProvider):
    """Placeholder Anthropic provider.

    The API key is never accepted as a constructor argument or stored on
    this object -- only its *presence* in the environment is ever checked,
    via `is_configured()`, at call time.
    """

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL

    @property
    def name(self) -> str:
        return "anthropic"

    @staticmethod
    def is_configured() -> bool:
        """Whether ANTHROPIC_API_KEY is set and non-empty. Never reads/logs its value."""
        return bool(os.environ.get(API_KEY_ENV_VAR, "").strip())

    def create_response(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError(
            "AnthropicProvider.create_response is not implemented in this pass. "
            "Real HTTP request/response handling against the Anthropic API is "
            "deferred to a later task -- see docs/orchestra-api-runtime.md."
        )
