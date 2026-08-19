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
import urllib.error
import urllib.request
from dataclasses import dataclass

from orchestra_api.models import ModelRequest, ModelResponse
from orchestra_api.providers.base import ModelProvider, ProviderError
from orchestra_api.providers.openai_provider import _build_request_body, _parse_response

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

    @property
    def name(self) -> str:
        return self.provider_label

    @property
    def wire_format(self) -> int:
        return 1

    def is_configured(self) -> bool:
        """Whether this instance's configured env var is set and non-empty."""
        return bool(os.environ.get(self.api_key_env_var, "").strip())

    def create_response(self, request: ModelRequest) -> ModelResponse:
        api_key = os.environ.get(self.api_key_env_var, "").strip()
        if not api_key:
            raise ProviderError(f"{self.api_key_env_var} is not set")

        body = _build_request_body(request, self.model)
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(f"{self.provider_label} API returned HTTP {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise ProviderError(f"{self.provider_label} API request failed: {exc.reason}") from None
        except TimeoutError:
            raise ProviderError(
                f"{self.provider_label} API request timed out after {self.timeout_seconds}s"
            ) from None

        return _parse_response(data)
