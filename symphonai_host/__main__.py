"""Run the loopback SymphonAI host as ``python -m symphonai_host``."""

from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path

from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.anthropic_provider import AnthropicProvider
from symphonai_api.providers.gemini_provider import GeminiProvider
from symphonai_api.providers.openai_provider import OpenAIProvider
from symphonai_host.server import HostServer


def _provider(name: str, model: str | None):
    providers = {
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
    }
    provider_class = providers[name]
    return provider_class() if model is None else provider_class(model=model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("anthropic", "gemini", "openai"), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-turns", type=int, default=20)
    arguments = parser.parse_args()
    host = HostServer(
        _provider(arguments.provider, arguments.model),
        PermissionPolicy(repo_root=arguments.repo_root),
        max_turns=arguments.max_turns,
    )

    def shutdown(signum, frame) -> None:
        threading.Thread(target=host.close, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    host.print_handshake()
    try:
        host.serve_forever()
    finally:
        host.close()


if __name__ == "__main__":
    main()
