"""Presets for known OpenAI-compatible endpoints (Grok, Kimi, Qwen, ...).

Because most vendors outside Anthropic and Google speak OpenAI's
chat-completions wire format, supporting them is a matter of
*configuration*, not new provider code: each entry below is just a base
URL, an environment variable name, and a default model, fed into
`OpenAICompatibleProvider`.

**Everything here is UNVERIFIED candidate configuration.** These base URLs
and model names were written from general knowledge, not probed against a
live account, and vendors change both. Treat an entry the same way
`orchestra_agents/probes.py` treats its CLI flags: a starting point to
check against the vendor's current docs, never a fact to rely on. If a
call fails with a 404 or an unknown-model error, suspect this file first.

**Tool calling is the least consistent part of "OpenAI-compatible."** Some
endpoints here may not support function calling at all, or may differ in
details. `orchestra_api` subagents depend on tool calls, so an endpoint
can connect fine and still never invoke a tool. Verify per endpoint.

Data only: importing this module performs no network call.
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestra_api.providers.openai_compatible import OpenAICompatibleProvider


@dataclass(frozen=True)
class CatalogEntry:
    """One unverified OpenAI-compatible endpoint preset."""

    key: str
    label: str
    base_url: str
    api_key_env_var: str
    default_model: str
    notes: str = ""


CATALOG: dict[str, CatalogEntry] = {
    "grok": CatalogEntry(
        key="grok",
        label="xAI Grok",
        base_url="https://api.x.ai/v1",
        api_key_env_var="XAI_API_KEY",
        default_model="grok-4",
        notes="xAI's OpenAI-compatible surface.",
    ),
    "kimi": CatalogEntry(
        key="kimi",
        label="Moonshot Kimi",
        base_url="https://api.moonshot.cn/v1",
        api_key_env_var="MOONSHOT_API_KEY",
        default_model="moonshot-v1-8k",
        notes="Moonshot AI. There is also a .ai international endpoint; check which your key is for.",
    ),
    "deepseek": CatalogEntry(
        key="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        api_key_env_var="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
    ),
    "qwen": CatalogEntry(
        key="qwen",
        label="Alibaba Qwen (DashScope)",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env_var="DASHSCOPE_API_KEY",
        default_model="qwen-plus",
        notes="DashScope's explicit compatible-mode path. Region-specific hosts also exist.",
    ),
    "ollama": CatalogEntry(
        key="ollama",
        label="Ollama (local)",
        base_url="http://localhost:11434/v1",
        api_key_env_var="OLLAMA_API_KEY",
        default_model="llama3",
        notes=(
            "Local server; usually ignores the key, but one must still be set "
            "non-empty for is_configured() to pass. Tool-calling support depends "
            "on the specific local model."
        ),
    ),
    "openrouter": CatalogEntry(
        key="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env_var="OPENROUTER_API_KEY",
        default_model="openai/gpt-4o-mini",
        notes="Aggregator: the model string selects the upstream vendor.",
    ),
    "groq": CatalogEntry(
        key="groq",
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env_var="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
    ),
    "together": CatalogEntry(
        key="together",
        label="Together AI",
        base_url="https://api.together.xyz/v1",
        api_key_env_var="TOGETHER_API_KEY",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ),
}


def catalog_keys() -> list[str]:
    """Sorted list of available catalog keys."""
    return sorted(CATALOG)


def build_catalog_provider(key: str, model: str | None = None) -> OpenAICompatibleProvider:
    """Build an OpenAICompatibleProvider from a catalog preset.

    `model` overrides the (unverified) default. Raises KeyError for an
    unknown key.
    """
    try:
        entry = CATALOG[key]
    except KeyError as exc:
        raise KeyError(f"unknown catalog provider {key!r}; known: {catalog_keys()}") from exc
    return OpenAICompatibleProvider(
        api_key_env_var=entry.api_key_env_var,
        base_url=entry.base_url,
        model=model or entry.default_model,
        provider_label=entry.key,
    )
