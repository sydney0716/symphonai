#!/usr/bin/env python3
"""Manual live-test script for the real API providers.

Default run is dry-run only: it reports whether each provider is
configured (its env var is set) and what it would send, without ever
calling `urllib` or touching the network. `--probe-efforts` describes or,
with `--live`, runs a bounded OpenAI-wire effort probe.

Making any real call requires BOTH `--live` and an explicit
`--provider` naming a native provider (openai/anthropic/gemini) or an
OpenAI-compatible catalog preset. It prints a cost/network warning first.

This script is never run automatically as part of validation --
`scripts/check.py` runs the automated offline provider checks, and stays
untouched by this one.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from symphonai_api.model_discovery import list_models  # noqa: E402
from symphonai_api.models import Message, ModelRequest, Role  # noqa: E402
from symphonai_api.provider_catalog import build_catalog_provider, catalog_keys  # noqa: E402
from symphonai_api.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from symphonai_api.providers.base import ModelProvider, ProviderError  # noqa: E402
from symphonai_api.providers.gemini_provider import GeminiProvider  # noqa: E402
from symphonai_api.providers.openai_provider import OpenAIProvider  # noqa: E402

# Providers with their own wire format and a dedicated implementation.
NATIVE_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}

ALL_PROVIDERS = list(NATIVE_PROVIDERS) + catalog_keys()

DEFAULT_PROMPT = "Reply with exactly one word: pong"
DEFAULT_PROBE_MODEL_LIMIT = 3
INVALID_PROBE_EFFORT = "__symphonai_invalid_effort_probe__"
_ENUMERATION_PATTERNS = (
    re.compile(
        r"(?:supported|allowed|accepted|valid)\s+(?:values|options|efforts?)\s+"
        r"(?:are|include)\s*:?\s*(?P<values>[^.\n}]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:must be|expected)\s+one\s+of\s*:?\s*(?P<values>[^.\n}]+)",
        re.IGNORECASE,
    ),
)


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
    print("SymphonAI API live-test (dry-run unless --live is passed)")
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
    print("Pass --probe-efforts --provider PROVIDER to preview a bounded effort probe.")


def report_effort_probe(provider_name: str | None, max_models: int) -> None:
    target = provider_name or "an explicit OpenAI-wire provider"
    print("SymphonAI effort probe (dry-run unless --live is passed)")
    print(f"Would list models from {target} and probe at most {max_models} model(s).")
    print(
        f"Each probe would send one user message containing '.' with effort "
        f"{INVALID_PROBE_EFFORT!r}."
    )
    print("Dry run only: no network call was made.")
    print("Pass --live with an explicit --provider to run the probe.")


def _accepted_efforts(error: str) -> tuple[str, ...]:
    for pattern in _ENUMERATION_PATTERNS:
        match = pattern.search(error)
        if match is None:
            continue
        values = match.group("values")
        candidates = re.findall(
            r"['\"]([A-Za-z][A-Za-z0-9_-]*)['\"]",
            values,
        )
        if not candidates:
            bracketed = re.search(r"\[([^]]+)\]", values)
            if bracketed is not None:
                candidates = re.findall(
                    r"[A-Za-z][A-Za-z0-9_-]*",
                    bracketed.group(1),
                )
        return tuple(
            dict.fromkeys(
                value
                for value in candidates
                if value != INVALID_PROBE_EFFORT
            )
        )
    return ()


def run_effort_probe(provider_name: str, max_models: int) -> int:
    provider = _build(provider_name)
    print(
        f"WARNING: about to list models and make up to {max_models} REAL "
        f"validation calls to {provider_name}."
    )
    print("This will consume real API quota/credits on your account.")
    print()
    if provider.wire_format != 1:
        print("FAIL: effort probing currently supports OpenAI-wire providers only")
        return 2
    if not provider.is_configured():
        print(f"FAIL: {provider_name} is not configured (its API key env var is not set)")
        return 1
    try:
        models = list_models(provider)[:max_models]
    except ProviderError:
        print("FAIL: model discovery failed; vendor detail was suppressed")
        return 1

    for model in models:
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=".")],
            model=model,
            effort=INVALID_PROBE_EFFORT,
        )
        try:
            provider.create_response(request)
        except ProviderError as exc:
            efforts = _accepted_efforts(str(exc))
            print(f"{model}: {', '.join(efforts) if efforts else 'unknown'}")
        except ValueError:
            print(f"{model}: unknown")
        else:
            print(f"{model}: unknown")
    return 0


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
    parser.add_argument(
        "--probe-efforts",
        action="store_true",
        help="List models and probe their accepted effort values.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=DEFAULT_PROBE_MODEL_LIMIT,
        help=(
            "Maximum models probed under --probe-efforts "
            f"(default: {DEFAULT_PROBE_MODEL_LIMIT})."
        ),
    )
    args = parser.parse_args(argv)

    if args.max_models < 1:
        parser.error("--max-models must be at least 1")

    if not args.live:
        if args.probe_efforts:
            report_effort_probe(args.provider, args.max_models)
        else:
            report_configuration()
        return 0

    if args.provider is None:
        print(f"FAIL: --live requires --provider, one of: {', '.join(sorted(ALL_PROVIDERS))}")
        return 2

    if args.probe_efforts:
        return run_effort_probe(args.provider, args.max_models)
    return run_live(args.provider, args.prompt, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
