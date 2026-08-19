#!/usr/bin/env python3
"""Terminal chat CLI for the orchestra_api Leader.

Pick a leader provider+model and a subagent provider+model at startup
(one shared provider for all subagents in the session), then chat with
the leader in a loop. Simple status lines (pending/working/done) print
live via Leader's on_status callback as the leader dispatches and
subagents work -- no message content or "thinking" is streamed, just
coarse status.

Plain stdout/print()/input() only -- no TUI framework, no raw terminal
mode, no new dependency. A real provider requires its API key already
exported into the environment (see .env.example); this script never
reads a key's value itself, only checks is_configured(), same as
scripts/live_test_provider.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.leader import Leader, LeaderConfig  # noqa: E402
from orchestra_api.provider_catalog import build_catalog_provider, catalog_keys  # noqa: E402
from orchestra_api.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from orchestra_api.providers.base import ModelProvider  # noqa: E402
from orchestra_api.providers.fake import FakeModelProvider  # noqa: E402
from orchestra_api.providers.gemini_provider import GeminiProvider  # noqa: E402
from orchestra_api.providers.openai_provider import OpenAIProvider  # noqa: E402

# Providers with their own wire format and a dedicated implementation.
NATIVE_PROVIDERS = {
    "fake": FakeModelProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}

# Everything else is an OpenAI-compatible endpoint from the catalog, which
# is configuration rather than new provider code. Catalog entries are
# unverified -- see orchestra_api/provider_catalog.py.
ALL_CHOICES = list(NATIVE_PROVIDERS) + catalog_keys()


def _prompt_choice(label: str) -> str:
    native = ", ".join(NATIVE_PROVIDERS)
    compatible = ", ".join(catalog_keys())
    print(f"{label} provider:")
    print(f"  native:            {native}")
    print(f"  openai-compatible: {compatible}  (unverified presets)")
    while True:
        raw = input(f"{label} provider (default: fake): ").strip().lower()
        if not raw:
            return "fake"
        if raw in ALL_CHOICES:
            return raw
        print(f"  unknown provider {raw!r}, choose one of: {', '.join(ALL_CHOICES)}")


def _prompt_model(provider_name: str, default_model: str) -> str:
    raw = input(f"{provider_name} model (default: {default_model}): ").strip()
    return raw or default_model


def _build_provider(provider_name: str) -> ModelProvider:
    if provider_name in NATIVE_PROVIDERS:
        cls = NATIVE_PROVIDERS[provider_name]
        if cls is FakeModelProvider:
            # Fake always gives the same canned reply -- useful for checking
            # the CLI's plumbing (selection, status lines, loop), not a real
            # chat.
            return FakeModelProvider()
        if not cls.is_configured():
            print(f"  WARNING: {provider_name} is not configured (its API key env var is not set).")
            print("  Calls to it will fail with a clear error until you export the key.")
        model = _prompt_model(provider_name, cls().model)
        return cls(model=model)

    # Catalog (OpenAI-compatible) entry.
    probe = build_catalog_provider(provider_name)
    if not probe.is_configured():
        print(f"  WARNING: {provider_name} is not configured ({probe.api_key_env_var} is not set).")
        print("  Calls to it will fail with a clear error until you export the key.")
    print(f"  note: {provider_name} is an unverified catalog preset ({probe.base_url}).")
    print("  Check the vendor's current docs if calls fail; tool-calling support varies.")
    model = _prompt_model(provider_name, probe.model)
    return build_catalog_provider(provider_name, model=model)


def _print_status(label: str, status: str) -> None:
    print(f"  [{label}] {status}")


def _provider_summary(provider: ModelProvider) -> str:
    model = getattr(provider, "model", None)
    return f"{provider.name} ({model})" if model else provider.name


def main() -> int:
    print("Orchestra API terminal chat")
    print("Pick a provider for the leader and for subagents (one shared provider for all subagents).")
    print()

    leader_choice = _prompt_choice("Leader")
    leader_provider = _build_provider(leader_choice)
    subagent_choice = _prompt_choice("Subagent")
    subagent_provider = _build_provider(subagent_choice)

    real_selected = leader_choice != "fake" or subagent_choice != "fake"
    if real_selected:
        print()
        print("WARNING: a real provider was selected. Chatting will make real network")
        print("calls and consume real API quota/credits on your account for every turn.")
        confirm = input("Continue? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return 0

    config = LeaderConfig(
        leader_provider=leader_provider,
        subagent_provider=subagent_provider,
        repo_root=str(REPO_ROOT),
        on_status=_print_status,
    )
    leader = Leader(config)

    print()
    print(f"Chatting with leader={_provider_summary(leader_provider)}, subagent={_provider_summary(subagent_provider)}.")
    print("Type 'exit' or 'quit' to stop (Ctrl-C also works).")
    print()

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Exiting.")
            return 0

        try:
            result = leader.chat(user_input)
        except Exception as exc:  # a real provider can raise ProviderError, among others
            print(f"  ERROR: {exc}")
            continue

        print(f"leader> {result.final_answer}")
        if result.stopped_reason != "final_response":
            print(f"  (note: stopped due to {result.stopped_reason}, answer may be incomplete)")


if __name__ == "__main__":
    raise SystemExit(main())
