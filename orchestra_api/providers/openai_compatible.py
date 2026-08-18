"""Interface-conforming placeholder for any OpenAI-compatible chat API.

Covers self-hosted or third-party endpoints that speak the OpenAI chat-
completions wire shape, e.g. Qwen. Unlike the vendor-specific placeholders,
both the API key env var name and the base URL are caller-supplied, since
"OpenAI-compatible" covers many different endpoints with different env var
conventions. `create_response` raises `NotImplementedError`; real HTTP
request/response handling is deferred to a later task. See
`docs/orchestra-api-runtime.md`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from orchestra_api.models import ModelRequest, ModelResponse
from orchestra_api.providers.base import ModelProvider

DEFAULT_MODEL = "openai-compatible-placeholder"


@dataclass
class OpenAICompatibleProvider(ModelProvider):
    """Placeholder provider for any OpenAI-compatible endpoint (e.g. Qwen).

    `api_key_env_var` and `base_url` are required and caller-supplied
    (there is no single default endpoint or key name for this family). The
    API key is never accepted as a constructor argument or stored on this
    object -- only its *presence* in the environment is ever checked, via
    `is_configured()`, at call time.
    """

    api_key_env_var: str
    base_url: str
    model: str = DEFAULT_MODEL
    provider_label: str = "openai-compatible"

    @property
    def name(self) -> str:
        return self.provider_label

    def is_configured(self) -> bool:
        """Whether this instance's configured env var is set and non-empty."""
        return bool(os.environ.get(self.api_key_env_var, "").strip())

    def create_response(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError(
            f"OpenAICompatibleProvider({self.provider_label!r}).create_response "
            "is not implemented in this pass. Real HTTP request/response "
            "handling is deferred to a later task -- see "
            "docs/orchestra-api-runtime.md."
        )
