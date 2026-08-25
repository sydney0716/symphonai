#!/usr/bin/env python3
"""Manual live-test script for the real API providers.

Default run is dry-run only: it reports whether each provider is
configured (its env var is set) and what it would send, without ever
calling `urllib` or touching the network.

Making one real call requires BOTH `--live` and an explicit
`--provider` naming a native provider (openai/anthropic/gemini) or an
OpenAI-compatible catalog preset. It prints a cost/network warning first.

This script is never run automatically as part of validation --
`scripts/smoke_api_agent.py` (FakeModelProvider-only) is the automated
test, and stays untouched by this one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.models import Message, ModelRequest, Role  # noqa: E402
from orchestra_api.provider_catalog import build_catalog_provider, catalog_keys  # noqa: E402
from orchestra_api.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from orchestra_api.providers.base import ModelProvider, ProviderError  # noqa: E402
from orchestra_api.providers.gemini_provider import GeminiProvider  # noqa: E402
from orchestra_api.providers.openai_provider import OpenAIProvider  # noqa: E402

# Providers with their own wire format and a dedicated implementation.
NATIVE_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}

ALL_PROVIDERS = list(NATIVE_PROVIDERS) + catalog_keys()

DEFAULT_PROMPT = "Reply with exactly one word: pong"


def _build(provider_name: str, model: str | None = None) -> ModelProvider:
    """Build a provider, optionally overriding its default model.

    A model is just a string passed through to the vendor -- no provider
    branches on it, and every model within one vendor shares that vendor's
    wire format. So overriding it here is always safe: `claude-sonnet-5`
    and `claude-haiku-4-5-...` produce byte-identical request *shapes*,
    differing only in the `model` field.
    """
    if provider_name in NATIVE_PROVIDERS:
        cls = NATIVE_PROVIDERS[provider_name]
        return cls(model=model) if model else cls()
    return build_catalog_provider(provider_name, model=model)


def report_configuration() -> None:
    print("Orchestra API live-test (dry-run unless --live is passed)")
    print()
    for name in ALL_PROVIDERS:
        provider = _build(name)
        kind = "native" if name in NATIVE_PROVIDERS else "openai-compatible (unverified preset)"
        print(f"== {name} == [{kind}]")
        print(f"  configured: {provider.is_configured()} (env var checked, value never read/printed)")
        print(f"  model: {provider.model}")
        if name not in NATIVE_PROVIDERS:
            print(f"  base_url: {provider.base_url}  (verify against current vendor docs)")
        print(f"  would send: 1 user message -> {provider.name}.create_response()")
    print()
    print("Dry run only: no network call was made.")
    print(f"Pass --live --provider {{{','.join(ALL_PROVIDERS)}}} to make one real call.")


def run_live(provider_name: str, prompt: str, model: str | None = None) -> int:
    provider = _build(provider_name, model)
    if not provider.is_configured():
        print(f"FAIL: {provider_name} is not configured (its API key env var is not set)")
        return 1
    print(f"WARNING: about to make a REAL network call to {provider_name} ({provider.model}).")
    print("This will consume real API quota/credits on your account.")
    print(f"Prompt: {prompt!r}")
    print()

    request = ModelRequest(messages=[Message(role=Role.USER, content=prompt)])
    try:
        response = provider.create_response(request)
    except ProviderError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"Response: {response.message.text!r}")
    print(f"Usage: input={response.usage.input_tokens} output={response.usage.output_tokens}")
    print(f"Stop reason: {response.stop_reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Make one real network call instead of a dry run. Requires --provider.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(ALL_PROVIDERS),
        default=None,
        help="Which provider to call under --live.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send under --live (default: a short, cheap test prompt).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override the provider's default model (e.g. claude-sonnet-5, "
            "gemini-3.7-flash). Any model offered by that provider works -- all "
            "models within one provider share its wire format. Defaults are only "
            "a starting point and can be retired by the vendor at any time."
        ),
    )
    args = parser.parse_args(argv)

    if not args.live:
        report_configuration()
        return 0

    if args.provider is None:
        print(f"FAIL: --live requires --provider, one of: {', '.join(sorted(ALL_PROVIDERS))}")
        return 2

    return run_live(args.provider, args.prompt, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
