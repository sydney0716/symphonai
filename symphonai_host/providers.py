"""Provider construction and persisted host selection."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from symphonai_api.paths import symphonai_home, write_atomic
from symphonai_api.providers.base import ModelProvider
from symphonai_api.providers.anthropic_provider import API_KEY_ENV_VAR as ANTHROPIC_KEY_ENV_VAR, AnthropicProvider
from symphonai_api.providers.gemini_provider import API_KEY_ENV_VAR as GEMINI_KEY_ENV_VAR, GeminiProvider
from symphonai_api.providers.openai_provider import API_KEY_ENV_VAR as OPENAI_KEY_ENV_VAR, OpenAIProvider
from symphonai_api.provider_catalog import CATALOG, build_catalog_provider, catalog_keys
from symphonai_host.run import MissingAPIKeyError, ProviderSelectionError


PROVIDERS = (
    ("anthropic", ANTHROPIC_KEY_ENV_VAR, AnthropicProvider),
    ("gemini", GEMINI_KEY_ENV_VAR, GeminiProvider),
    ("openai", OPENAI_KEY_ENV_VAR, OpenAIProvider),
) + tuple(
    (key, CATALOG[key].api_key_env_var, None)
    for key in catalog_keys()
)


def provider(name: str | None = None, model: str | None = None, base_url: str | None = None) -> ModelProvider | None:
    if name is None:
        name = next((vendor for vendor, key, _ in PROVIDERS if os.environ.get(key, "").strip()), None)
        if name is None:
            return None
    if not isinstance(name, str) or not any(vendor == name for vendor, _, _ in PROVIDERS):
        raise ProviderSelectionError("unknown provider")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ProviderSelectionError("model must be a non-empty string")
    if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
        raise ProviderSelectionError("base_url must be a non-empty string")
    vendor, key, provider_class = next(row for row in PROVIDERS if row[0] == name)
    if not os.environ.get(key, "").strip():
        raise MissingAPIKeyError(f"{vendor} has no API key")
    if name in CATALOG:
        if base_url is not None:
            raise ProviderSelectionError(f"{name} uses its catalog base URL")
        return build_catalog_provider(name, model)
    options = {}
    if model is not None:
        options["model"] = model
    if base_url is not None:
        options["base_url"] = base_url
    return provider_class(**options)



def selection_path() -> Path:
    return symphonai_home() / "selection.json"



def load_provider_selection() -> dict[str, str] | None:
    try:
        with selection_path().open("r", encoding="utf-8") as source:
            choice = json.load(source)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if (
        not isinstance(choice, dict)
        or set(choice) - {"name", "model", "effort", "base_url"}
        or not isinstance(choice.get("name"), str)
        or not choice["name"]
        or any(
            key in choice and choice[key] is not None
            and (not isinstance(choice[key], str) or not choice[key].strip())
            for key in ("model", "effort", "base_url")
        )
    ):
        return None
    return choice



def provider_for_start(provider_factory=provider) -> tuple[ModelProvider | None, dict[str, str] | None]:
    choice = load_provider_selection()
    if choice is None:
        return provider_factory(), None
    try:
        provider = provider_factory(choice["name"], choice.get("model"), choice.get("base_url"))
    except ProviderSelectionError:
        return provider_factory(), None
    return provider, choice



def save_provider_selection(choice: Mapping[str, str]) -> None:
    target = selection_path()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_atomic(target, json.dumps(dict(choice)), mode=0o600, fsync=True)



def app_root() -> Path:
    return Path(__file__).resolve().parent.parent / "symphonai_app"
