"""Presets for known OpenAI-compatible endpoints (Grok, Kimi, Qwen, ...).

Because most vendors outside Anthropic and Google speak OpenAI's
chat-completions wire format, supporting them is a matter of
*configuration*, not new provider code: each entry below is just a base
URL, an environment variable name, and a default model, fed into
`OpenAICompatibleProvider`.

**Still not live-verified, but no longer pure guesswork.** These entries
were originally written from general knowledge; the `base_url` values have
since been cross-checked against the provider registry of
[`lidge-jun/opencodex`](https://github.com/lidge-jun/opencodex)
(`src/providers/registry.ts`, MIT-licensed), a maintained proxy that ships
against these endpoints. Each entry's `notes` records what that
cross-check found. Nothing here has been probed against a live account --
no key for any of these vendors exists in this project -- so treat an
entry as a
starting point to check against the vendor's current docs, never a fact to
rely on. If a call fails with a 404 or an unknown-model error, suspect
this file first.

Model names rot faster than base URLs, and silently: the Gemini provider's
former `gemini-2.0-flash` default was retired server-side and returned
HTTP 404 on every call. Where a vendor publishes a rolling alias
(`deepseek-chat`), this file prefers it over a pinned version for exactly
that reason.

**Tool calling is the least consistent part of "OpenAI-compatible."** Some
endpoints here may not support function calling at all, or may differ in
details. `symphonai_api` subagents depend on tool calls, so an endpoint
can connect fine and still never invoke a tool. Verify per endpoint.

Data only: importing this module performs no network call.

---
Portions of the `base_url` values below were cross-checked against, and in
places adopted from, `src/providers/registry.ts` in
https://github.com/lidge-jun/opencodex

    Copyright (c) opencodex contributors
    Licensed under the MIT License.

See `docs/symphonai-external-references.md` for why that project is
treated as reusable and the other external reference is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from symphonai_api.providers.openai_compatible import OpenAICompatibleProvider


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
        default_model="grok-4.5",
        notes=(
            "xAI's OpenAI-compatible surface. base_url confirmed identical in "
            "opencodex's registry; default model raised from grok-4 to grok-4.5 "
            "to match what that registry currently ships."
        ),
    ),
    "kimi": CatalogEntry(
        key="kimi",
        label="Moonshot Kimi",
        base_url="https://api.moonshot.cn/v1",
        api_key_env_var="MOONSHOT_API_KEY",
        default_model="moonshot-v1-8k",
        notes=(
            "Moonshot AI. There is also a .ai international endpoint; check which "
            "your key is for. DIVERGENCE: opencodex uses "
            "'https://api.kimi.com/coding/v1' with model 'kimi-k2.7-code'. That "
            "path is specific to Kimi's coding subscription, so it is deliberately "
            "not adopted as the general default here -- a plain Moonshot API key "
            "would likely fail against it. Switch to it if you hold a coding plan."
        ),
    ),
    "deepseek": CatalogEntry(
        key="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        api_key_env_var="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        notes=(
            "opencodex uses the bare 'https://api.deepseek.com'; DeepSeek documents "
            "both that and the '/v1' form as valid for OpenAI compatibility, so the "
            "explicit '/v1' is kept here. 'deepseek-chat' is a rolling alias for "
            "their current chat model and is preferred over opencodex's pinned "
            "'deepseek-v4-flash' so it cannot go stale."
        ),
    ),
    "qwen": CatalogEntry(
        key="qwen",
        label="Alibaba Qwen (DashScope)",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env_var="DASHSCOPE_API_KEY",
        default_model="qwen-plus",
        notes=(
            "DashScope's explicit compatible-mode path. Region-specific hosts also "
            "exist. STILL UNCORROBORATED: no DashScope/Qwen entry appeared in the "
            "part of opencodex's registry that was read, so this one remains at its "
            "original from-general-knowledge confidence."
        ),
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
            "on the specific local model. base_url confirmed identical in "
            "opencodex's registry."
        ),
    ),
    "openrouter": CatalogEntry(
        key="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env_var="OPENROUTER_API_KEY",
        default_model="openai/gpt-4o-mini",
        notes=(
            "Aggregator: the model string selects the upstream vendor. base_url "
            "confirmed identical in opencodex's registry, which sets no default "
            "model of its own -- so the model here is still an unverified guess "
            "and is the most likely thing to be stale."
        ),
    ),
    "groq": CatalogEntry(
        key="groq",
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env_var="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        notes=(
            "base_url confirmed identical in opencodex's registry, which sets no "
            "default model of its own -- the model here remains an unverified guess."
        ),
    ),
    "together": CatalogEntry(
        key="together",
        label="Together AI",
        base_url="https://api.together.xyz/v1",
        api_key_env_var="TOGETHER_API_KEY",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        notes=(
            "STILL UNCORROBORATED: no Together entry appeared in the part of "
            "opencodex's registry that was read, so this one remains at its "
            "original from-general-knowledge confidence."
        ),
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
