#!/usr/bin/env python3
"""Terminal chat CLI for the orchestra_api Leader.

Pick a leader provider+model and a subagent provider+model at startup
(one shared provider for all subagents in the session), then chat with
the leader in a loop. Simple status lines (pending/working/done) print
live via the typed event channel as the leader dispatches and
subagents work -- no message content or "thinking" is streamed, just
coarse status.

Plain stdout/print()/input() only -- no TUI framework, no raw terminal
mode, no new dependency. Providers require their API keys already exported
into the environment (see .env.example); model discovery reads keys fresh
from the environment and sends them only as request headers.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.events import (  # noqa: E402
    Event,
    RunFailed,
    RunFinished,
    RunStarted,
    SubagentSpawned,
)
from orchestra_api.leader import Leader, LeaderConfig  # noqa: E402
from orchestra_api.model_discovery import list_models  # noqa: E402
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

# Everything else is an OpenAI-compatible endpoint from the catalog, which
# is configuration rather than new provider code. Catalog entries are
# unverified -- see orchestra_api/provider_catalog.py.
NATIVE_CHOICES = list(NATIVE_PROVIDERS)
CATALOG_CHOICES = catalog_keys()
ALL_CHOICES = NATIVE_CHOICES + CATALOG_CHOICES


def _prompt_choice(label: str) -> str:
    print(f"{label} provider:")
    print("  native:")
    for index, provider_name in enumerate(NATIVE_CHOICES, start=1):
        print(f"    {index}. {provider_name}")
    print("  openai-compatible (unverified presets):")
    for index, provider_name in enumerate(CATALOG_CHOICES, start=len(NATIVE_CHOICES) + 1):
        print(f"    {index}. {provider_name}")
    while True:
        raw = input(f"{label} provider number or name: ").strip().lower()
        if not raw:
            print("  enter a number or provider name; no default provider is selected.")
            continue
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(ALL_CHOICES):
                return ALL_CHOICES[index - 1]
            print(f"  provider number must be 1-{len(ALL_CHOICES)}")
            continue
        if raw in ALL_CHOICES:
            return raw
        print(f"  unknown provider {raw!r}, choose one of: {', '.join(ALL_CHOICES)}")


def _prompt_model_free_text(provider_name: str, default_model: str) -> str:
    raw = input(f"{provider_name} model (default: {default_model}): ").strip()
    return raw or default_model


def _prompt_model(provider: ModelProvider, default_model: str) -> str:
    try:
        models = list_models(provider)
        hidden = len(list_models(provider, include_all=True)) - len(models)
    except ProviderError as exc:
        print(f"  Could not list {provider.name} models: {exc}. Enter a model name manually.")
        return _prompt_model_free_text(provider.name, default_model)

    if not models:
        print(f"  Could not list {provider.name} models: provider returned no models. Enter a model name manually.")
        return _prompt_model_free_text(provider.name, default_model)

    showing_all = False
    while True:
        print(f"{provider.name} models:")
        for index, model in enumerate(models, start=1):
            print(f"  {index}. {model}")
        if hidden > 0 and not showing_all:
            # The filter is a name-based heuristic, so always say what it hid
            # and offer a way past it -- a misjudged model must stay reachable.
            print(f"  ({hidden} non-text or retired models hidden -- type 'all' to show them)")

        raw = input(f"{provider.name} model number or name (default: {default_model}): ").strip()
        if not raw:
            return default_model
        if raw.lower() == "all" and not showing_all:
            try:
                models = list_models(provider, include_all=True)
            except ProviderError as exc:
                print(f"  Could not list all {provider.name} models: {exc}.")
                continue
            showing_all = True
            continue
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(models):
                return models[index - 1]
            print(f"  model number must be 1-{len(models)}, or type a model name.")
            continue
        return raw


def _build_provider(provider_name: str) -> ModelProvider:
    if provider_name in NATIVE_PROVIDERS:
        cls = NATIVE_PROVIDERS[provider_name]
        probe = cls()
        model = _prompt_model(probe, probe.model)
        return cls(model=model)

    # Catalog (OpenAI-compatible) entry.
    probe = build_catalog_provider(provider_name)
    print(f"  note: {provider_name} is an unverified catalog preset ({probe.base_url}).")
    print("  Check the vendor's current docs if calls fail; tool-calling support varies.")
    model = _prompt_model(probe, probe.model)
    return build_catalog_provider(provider_name, model=model)


def _print_status(event: Event) -> None:
    if isinstance(event, SubagentSpawned):
        update = (event.subagent_name, "pending")
    elif isinstance(event, RunStarted):
        update = (event.agent_name, "working")
    elif isinstance(event, RunFinished) and event.stopped_reason == "final_response":
        update = (event.agent_name, "done")
    elif isinstance(event, RunFinished) and event.stopped_reason == "max_turns":
        update = (event.agent_name, "exhausted")
    elif isinstance(event, RunFinished) and event.stopped_reason == "cancelled":
        update = (event.agent_name, "cancelled")
    elif isinstance(event, RunFailed):
        update = (event.agent_name, "failed")
    else:
        return
    print(f"  {update[0]}: {update[1]}")


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

    real_selected = True
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
        events=_print_status,
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
