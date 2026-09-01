"""ModelProvider for any endpoint speaking the OpenAI chat-completions format.

"OpenAI-compatible" is the ecosystem's term for an endpoint that accepts
OpenAI's `/v1/chat/completions` request/response JSON. It is a **de facto
standard, not a formal spec**: xAI (Grok), Moonshot (Kimi), DeepSeek,
Alibaba (Qwen), Groq, Together, OpenRouter, Ollama and others implement it
so existing OpenAI client code works against them by changing only the
base URL and key. It has nothing to do with OpenAI the vendor.

**Compliance varies, and tool calling is the least consistent part.** Some
compatible endpoints do not support function calling at all, and others
differ in details. Since `orchestra_api` subagents depend on tool calls,
an endpoint can connect fine here and still never invoke a tool -- verify
per endpoint rather than assuming.

Request building and response parsing are reused verbatim from
`openai_provider` rather than duplicated, so the two cannot drift apart.
See `orchestra_api.provider_catalog` for known endpoint presets.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

from orchestra_api.cancellation import CancellationToken
from orchestra_api.models import ModelRequest, ModelResponse
from orchestra_api.providers.base import ModelProvider, ProviderError, parse_json_object
from orchestra_api.providers.openai_provider import _build_request_body, _parse_response
from orchestra_api.retry import DEFAULT_MAX_ATTEMPTS, read_with_retry

DEFAULT_MODEL = "openai-compatible-placeholder"


@dataclass
class OpenAICompatibleProvider(ModelProvider):
    """Calls any OpenAI-chat-completions-compatible endpoint (Grok, Kimi, Qwen, ...).

    `api_key_env_var` and `base_url` are required and caller-supplied --
    this family has no single default endpoint or key-name convention. The
    API key is never accepted as a constructor argument or stored on this
    object; it is read from the named environment variable fresh on every
    call, never logged, and never included in any error message.
    """

    api_key_env_var: str
    base_url: str
    model: str = DEFAULT_MODEL
    provider_label: str = "openai-compatible"
    timeout_seconds: float = 30.0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    @property
    def name(self) -> str:
        return self.provider_label

    @property
    def wire_format(self) -> int:
        return 1

    def is_configured(self) -> bool:
        """Whether this instance's configured env var is set and non-empty."""
        return bool(os.environ.get(self.api_key_env_var, "").strip())

    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        api_key = os.environ.get(self.api_key_env_var, "").strip()
        if not api_key:
            raise ProviderError(f"{self.api_key_env_var} is not set")

        model = request.model if request.model is not None else self.model
        body = _build_request_body(request, model)
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
        )
        raw = read_with_retry(
            http_request,
            timeout=self.timeout_seconds,
            max_attempts=self.max_attempts,
            api_key=api_key,
            operation=f"{self.provider_label} API",
            cancel=cancel,
            call_class=request.call_class,
        )
        data = parse_json_object(raw, f"{self.provider_label} API")

        return _parse_response(data)
