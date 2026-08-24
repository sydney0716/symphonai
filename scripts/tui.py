#!/usr/bin/env python3
"""Thin launcher for the optional Textual Orchestra TUI."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.provider_catalog import build_catalog_provider, catalog_keys  # noqa: E402
from orchestra_api.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from orchestra_api.providers.fake import FakeModelProvider  # noqa: E402
from orchestra_api.providers.gemini_provider import GeminiProvider  # noqa: E402
from orchestra_api.providers.openai_provider import OpenAIProvider  # noqa: E402
from orchestra_api.providers.base import ModelProvider  # noqa: E402

NATIVE_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}
OFFLINE_PROVIDER = "fake"


def _default_provider(role: str) -> str | None:
    return (
        os.environ.get(f"ORCHESTRA_TUI_{role.upper()}_PROVIDER")
        or os.environ.get("ORCHESTRA_TUI_PROVIDER")
    )


def _default_model(role: str) -> str | None:
    return os.environ.get(f"ORCHESTRA_TUI_{role.upper()}_MODEL") or os.environ.get("ORCHESTRA_TUI_MODEL")


def _build_provider(provider_name: str, model: str | None) -> ModelProvider:
    name = provider_name.strip().lower()
    if name == OFFLINE_PROVIDER:
        return FakeModelProvider()
    if name in NATIVE_PROVIDERS:
        provider_cls = NATIVE_PROVIDERS[name]
        return provider_cls(model=model) if model else provider_cls()
    if name in catalog_keys():
        return build_catalog_provider(name, model=model)
    choices = [OFFLINE_PROVIDER, *NATIVE_PROVIDERS, *catalog_keys()]
    raise ValueError(f"unknown provider {provider_name!r}; choose one of: {', '.join(choices)}")


def _is_fully_specified(provider_name: str | None, model: str | None) -> bool:
    if not provider_name:
        return False
    return provider_name.strip().lower() == OFFLINE_PROVIDER or bool(model and model.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Orchestra Leader Textual TUI.")
    parser.add_argument("--leader-provider", default=_default_provider("leader"))
    parser.add_argument("--leader-model", default=_default_model("leader"))
    parser.add_argument("--subagent-provider", default=_default_provider("subagent"))
    parser.add_argument("--subagent-model", default=_default_model("subagent"))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    return parser


def _load_app_class():
    try:
        from orchestra_tui.app import OrchestraTuiApp
    except ModuleNotFoundError as exc:
        if exc.name in {"textual", "rich"}:
            print(
                "The Orchestra TUI requires the optional Textual extra.\n"
                "Install it with: python3 -m pip install -e '.[tui]'",
                file=sys.stderr,
            )
            return None
        raise
    return OrchestraTuiApp


def main() -> int:
    args = _parser().parse_args()
    app_class = _load_app_class()
    if app_class is None:
        return 1

    skip_picker = _is_fully_specified(args.leader_provider, args.leader_model) and _is_fully_specified(
        args.subagent_provider,
        args.subagent_model,
    )
    if skip_picker:
        try:
            leader_provider = _build_provider(args.leader_provider, args.leader_model)
            subagent_provider = _build_provider(args.subagent_provider, args.subagent_model)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        app = app_class(
            leader_provider=leader_provider,
            subagent_provider=subagent_provider,
            repo_root=args.repo_root,
            confirm_real_providers=True,
        )
    else:
        app = app_class(
            repo_root=args.repo_root,
            confirm_real_providers=True,
        )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
