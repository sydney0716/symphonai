"""Live model listing for configured ModelProvider instances.

Model discovery is keyed by `ModelProvider.wire_format`, not provider name,
so OpenAI-compatible catalog presets use the same `/models` listing path as
the native OpenAI provider. API keys are read from `os.environ` at call time
and sent only in request headers.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from orchestra_api.providers.anthropic_provider import (
    ANTHROPIC_VERSION,
    API_KEY_ENV_VAR as ANTHROPIC_API_KEY_ENV_VAR,
)
from orchestra_api.providers.base import ModelProvider, ProviderError
from orchestra_api.providers.gemini_provider import API_KEY_ENV_VAR as GEMINI_API_KEY_ENV_VAR
from orchestra_api.providers.openai_provider import API_KEY_ENV_VAR as OPENAI_API_KEY_ENV_VAR


def list_models(provider: ModelProvider) -> list[str]:
    """Return model ids currently listed by the provider's API.

    Raises ProviderError for missing configuration, unsupported wire formats,
    transport errors, HTTP errors, or malformed responses. The provider's key
    is never included in the request URL or in this function's error messages.
    """

    wire_format = provider.wire_format
    if wire_format == 1:
        return _list_openai_wire_models(provider)
    if wire_format == 2:
        return _list_anthropic_models(provider)
    if wire_format == 3:
        return _list_gemini_models(provider)
    raise ProviderError(f"{provider.name} does not support model discovery")


def _list_openai_wire_models(provider: ModelProvider) -> list[str]:
    _env_var, api_key = _read_api_key(provider, OPENAI_API_KEY_ENV_VAR)
    data = _get_json(
        provider,
        f"{_base_url(provider)}/models",
        {"Authorization": f"Bearer {api_key}"},
        api_key,
    )
    models = data.get("data")
    if not isinstance(models, list):
        raise ProviderError(f"{provider.name} model listing response did not contain data[]")
    return [model["id"] for model in models if isinstance(model, dict) and isinstance(model.get("id"), str)]


def _list_anthropic_models(provider: ModelProvider) -> list[str]:
    _env_var, api_key = _read_api_key(provider, ANTHROPIC_API_KEY_ENV_VAR)
    data = _get_json(
        provider,
        f"{_base_url(provider)}/models",
        {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        api_key,
    )
    models = data.get("data")
    if not isinstance(models, list):
        raise ProviderError(f"{provider.name} model listing response did not contain data[]")
    return [model["id"] for model in models if isinstance(model, dict) and isinstance(model.get("id"), str)]


def _list_gemini_models(provider: ModelProvider) -> list[str]:
    _env_var, api_key = _read_api_key(provider, GEMINI_API_KEY_ENV_VAR)
    data = _get_json(
        provider,
        f"{_base_url(provider)}/models?pageSize=200",
        {"x-goog-api-key": api_key},
        api_key,
    )
    models = data.get("models")
    if not isinstance(models, list):
        raise ProviderError(f"{provider.name} model listing response did not contain models[]")

    out: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        methods = model.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        name = model.get("name")
        if isinstance(name, str):
            out.append(name.removeprefix("models/"))
    return out


def _read_api_key(provider: ModelProvider, default_env_var: str) -> tuple[str, str]:
    env_var = getattr(provider, "api_key_env_var", default_env_var)
    if not isinstance(env_var, str) or not env_var:
        raise ProviderError(f"{provider.name} does not expose an API key environment variable")
    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        raise ProviderError(f"{env_var} is not set")
    return env_var, api_key


def _base_url(provider: ModelProvider) -> str:
    base_url = getattr(provider, "base_url", None)
    if not isinstance(base_url, str) or not base_url.strip():
        raise ProviderError(f"{provider.name} does not expose a base_url")
    return base_url.rstrip("/")


def _get_json(
    provider: ModelProvider,
    url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any]:
    http_request = urllib.request.Request(url, method="GET", headers=headers)
    timeout = getattr(provider, "timeout_seconds", 30.0)
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = _redact_key(exc.read().decode("utf-8", errors="replace")[:500], api_key)
        raise ProviderError(f"{provider.name} model listing returned HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        reason = _redact_key(str(exc.reason), api_key)
        raise ProviderError(f"{provider.name} model listing request failed: {reason}") from None
    except TimeoutError:
        raise ProviderError(f"{provider.name} model listing timed out after {timeout}s") from None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"{provider.name} model listing returned invalid JSON: {exc.msg}") from None
    if not isinstance(data, dict):
        raise ProviderError(f"{provider.name} model listing returned a non-object JSON response")
    return data


def _redact_key(text: str, api_key: str) -> str:
    if api_key:
        return text.replace(api_key, "[redacted]")
    return text
