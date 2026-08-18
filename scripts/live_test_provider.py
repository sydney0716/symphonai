#!/usr/bin/env python3
"""Manual live-test script for the real OpenAI/Anthropic providers.

Default run is dry-run only: it reports whether each provider is
configured (its env var is set) and what it would send, without ever
calling `urllib` or touching the network.

Making one real call requires BOTH `--live` and an explicit
`--provider {openai,anthropic}`. It prints a cost/network warning first.

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
from orchestra_api.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from orchestra_api.providers.base import ProviderError  # noqa: E402
from orchestra_api.providers.openai_provider import OpenAIProvider  # noqa: E402

PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}

DEFAULT_PROMPT = "Reply with exactly one word: pong"


def report_configuration() -> None:
    print("Orchestra API live-test (dry-run unless --live is passed)")
    print()
    for name, cls in PROVIDERS.items():
        provider = cls()
        print(f"== {name} ==")
        print(f"  configured: {cls.is_configured()} (env var checked, value never read/printed)")
        print(f"  model: {provider.model}")
        print(f"  would send: 1 user message -> {provider.name}.create_response()")
    print()
    print("Dry run only: no network call was made.")
    print("Pass --live --provider {openai,anthropic} to make one real call.")


def run_live(provider_name: str, prompt: str) -> int:
    cls = PROVIDERS[provider_name]
    if not cls.is_configured():
        print(f"FAIL: {provider_name} is not configured (its API key env var is not set)")
        return 1

    provider = cls()
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

    print(f"Response: {response.message.content!r}")
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
        choices=sorted(PROVIDERS),
        default=None,
        help="Which provider to call under --live.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send under --live (default: a short, cheap test prompt).",
    )
    args = parser.parse_args(argv)

    if not args.live:
        report_configuration()
        return 0

    if args.provider is None:
        print("FAIL: --live requires --provider {openai,anthropic}")
        return 2

    return run_live(args.provider, args.prompt)


if __name__ == "__main__":
    raise SystemExit(main())
