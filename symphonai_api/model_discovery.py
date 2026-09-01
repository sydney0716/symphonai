"""Live model listing for configured ModelProvider instances.

Model discovery is keyed by `ModelProvider.wire_format`, not provider name,
so OpenAI-compatible catalog presets use the same `/models` listing path as
the native OpenAI provider. API keys are read from `os.environ` at call time
and sent only in request headers.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import date
from typing import Any

from symphonai_api.providers.anthropic_provider import (
    ANTHROPIC_VERSION,
    API_KEY_ENV_VAR as ANTHROPIC_API_KEY_ENV_VAR,
)
from symphonai_api.providers.base import ModelProvider, ProviderError
from symphonai_api.providers.gemini_provider import API_KEY_ENV_VAR as GEMINI_API_KEY_ENV_VAR
from symphonai_api.providers.openai_provider import API_KEY_ENV_VAR as OPENAI_API_KEY_ENV_VAR
from symphonai_api.retry import CallClass, DEFAULT_MAX_ATTEMPTS, read_with_retry


# Substrings marking a model as something other than a text-generating LLM.
#
# This is an EXCLUSION list, deliberately, rather than an allow-list of known
# chat families. An allow-list silently hides any model family a vendor names
# in a new way, which is the same staleness that retired `gemini-2.0-flash`
# out from under this project. An exclusion list fails the safe direction: an
# unrecognized new model still shows up, and only known non-text modalities
# are hidden.
#
# Kept deliberately conservative -- exclude only what is clearly not a text
# LLM. Notably NOT excluded:
#   "codex"  -- gpt-5-codex and friends are coding models, exactly the target
#   "search" -- gpt-4o-*-search-preview are ordinary chat models with search
#   "omni"   -- gemini-omni-flash-preview is multimodal chat (note that
#               omni-moderation-* is still caught by "moderation")
NON_TEXT_MARKERS: tuple[str, ...] = (
    "tts",
    "whisper",
    "transcribe",
    "audio",
    "realtime",
    "embedding",
    "moderation",
    "dall-e",
    "image",
    "imagen",
    "sora",
    "veo",
    "lyria",
    "nano-banana",
    "robotics",
    "babbage",
    "davinci",
)


def is_probably_text_model(model_id: str) -> bool:
    """Heuristic: does this id look like a text-generating LLM?

    Name-based by necessity -- no vendor publishes a machine-readable "this
    is a chat model" capability. Gemini's `supportedGenerationMethods` is
    identical for its TTS, image, and music models, and OpenAI's `/models`
    carries no capability field at all. Treat this as a display convenience,
    never as a correctness guarantee: `list_models(..., include_all=True)`
    and typing a raw model name both bypass it.
    """
    lowered = model_id.lower()
    return not any(marker in lowered for marker in NON_TEXT_MARKERS)


def list_models(provider: ModelProvider, *, include_all: bool = False) -> list[str]:
    """Return model ids currently listed by the provider's API.

    By default the result is filtered to ids that look like text-generating
    LLMs (see `is_probably_text_model`) and, for OpenAI-wire providers, to
    models whose published `shutdown_date` has not already passed. Pass
    `include_all=True` for the raw, unfiltered listing.

    Raises ProviderError for missing configuration, unsupported wire formats,
    transport errors, HTTP errors, or malformed responses. The provider's key
    is never included in the request URL or in this function's error messages.
    """

    wire_format = provider.wire_format
    if wire_format == 1:
        models = _list_openai_wire_models(provider, include_all=include_all)
    elif wire_format == 2:
        models = _list_anthropic_models(provider)
    elif wire_format == 3:
        models = _list_gemini_models(provider)
    else:
        raise ProviderError(f"{provider.name} does not support model discovery")

    if include_all:
        return models
    return [model_id for model_id in models if is_probably_text_model(model_id)]


def _list_openai_wire_models(provider: ModelProvider, *, include_all: bool = False) -> list[str]:
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

    out: list[str] = []
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        # OpenAI publishes a retirement date per model. It is the only
        # authoritative staleness signal any of these APIs gives us, so drop
        # models whose date has already passed rather than offering a choice
        # that is guaranteed to fail. Future dates are still selectable.
        if not include_all and _shutdown_date_passed(model.get("shutdown_date")):
            continue
        out.append(model["id"])
    return out


def _shutdown_date_passed(raw: Any) -> bool:
    """True only when `raw` is a YYYY-MM-DD date strictly before today."""
    if not isinstance(raw, str) or not raw.strip():
        return False
    try:
        shutdown = date.fromisoformat(raw.strip())
    except ValueError:
        # An unparseable date is not evidence of retirement -- keep the model.
        return False
    return shutdown < date.today()


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
    raw = read_with_retry(
        http_request,
        timeout=timeout,
        max_attempts=getattr(provider, "max_attempts", DEFAULT_MAX_ATTEMPTS),
        api_key=api_key,
        operation=f"{provider.name} model listing",
        call_class=CallClass.BACKGROUND,
    ).decode("utf-8")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"{provider.name} model listing returned invalid JSON: {exc.msg}") from None
    if not isinstance(data, dict):
        raise ProviderError(f"{provider.name} model listing returned a non-object JSON response")
    return data
