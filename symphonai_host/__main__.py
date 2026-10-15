"""Run the loopback SymphonAI host as ``python -m symphonai_host``."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from symphonai_api.config import ConfigError
from symphonai_api.extensions import load_extensions
from symphonai_api.lsp import LspManager
from symphonai_api.mcp import McpError
from symphonai_api.mcp_pool import McpPool
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.runner import standard_tool_registry
from symphonai_api.roles import LEADER_BASE_PROMPT, SUBAGENT_BASE_PROMPT
from symphonai_api.session import default_sessions_root
from symphonai_host.server import HostServer, _provider, _provider_for_start, symphonai_home
from symphonai_host.run import ModeSelectionError
from symphonai_host.credentials import CredentialError, apply_to_environment, load
from symphonai_host.sessions import DEFAULT_CLEANUP_PERIOD_DAYS, prune_sessions
from symphonai_host.projects import build_project, choose_start_root, record_recent_project
from symphonai_host.broker import EventBroker


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--max-turns", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _arguments(argv)
    try:
        apply_to_environment(os.environ, load())
    except CredentialError as exc:
        print(f"credential error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    host = None
    runtime = None
    try:
        provider, saved_selection = _provider_for_start(_provider)
        broker = EventBroker()
        sessions_root = default_sessions_root()
        host_holder = {}
        start_root = choose_start_root(arguments.repo_root, Path.cwd(), Path.home(), symphonai_home())

        def publish_approval(approval) -> bool:
            current_host = host_holder.get("host")
            return False if current_host is None else current_host._publish_approval(approval)

        if start_root is not None:
            runtime = build_project(
                start_root,
                broker,
                provider,
                publish_approval=publish_approval,
                base_prompt=LEADER_BASE_PROMPT,
                subagent_base_prompt=SUBAGENT_BASE_PROMPT,
                max_turns=arguments.max_turns,
                sessions_root=sessions_root,
                provider_factory=lambda name, model, base_url: _provider(name, model, base_url),
                extensions_loader=load_extensions,
                pool_factory=McpPool,
                lsp_factory=LspManager,
                policy_factory=PermissionPolicy,
            )
            try:
                period = runtime.run.extensions.config.values.get(
                    "sessions.cleanup_period_days", DEFAULT_CLEANUP_PERIOD_DAYS
                )
                prune_sessions(runtime.run.sessions_root, period_days=period, now=datetime.now(timezone.utc))
            except Exception:
                pass
        host = HostServer(
            provider,
            None if runtime is None else runtime.run._base_policy,
            broker=broker,
            base_prompt=LEADER_BASE_PROMPT,
            subagent_base_prompt=SUBAGENT_BASE_PROMPT,
            max_turns=arguments.max_turns,
            extensions=None if runtime is None else runtime.run.extensions,
            mcp_tools=None if runtime is None else runtime.run._mcp_tools,
            lsp=None if runtime is None else runtime.lsp,
            runtime=runtime,
            provider_selection=saved_selection,
            project_factory=lambda root, broker, publish_approval: build_project(
                root,
                broker,
                provider,
                publish_approval=publish_approval,
                base_prompt=LEADER_BASE_PROMPT,
                subagent_base_prompt=SUBAGENT_BASE_PROMPT,
                max_turns=arguments.max_turns,
                sessions_root=sessions_root,
                provider_factory=lambda name, model, base_url: _provider(name, model, base_url),
                extensions_loader=load_extensions,
                pool_factory=McpPool,
                lsp_factory=LspManager,
                policy_factory=PermissionPolicy,
            ),
        )
        host_holder["host"] = host
        if runtime is not None:
            record_recent_project(symphonai_home(), runtime.root)
        if saved_selection is not None and provider is not None:
            host.run.select_provider(
                provider,
                saved_selection.get("model"),
                saved_selection.get("effort"),
                saved_selection,
            )

        def shutdown(signum, frame) -> None:
            threading.Thread(target=host.close, daemon=True).start()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        host.print_handshake()
        host.serve_forever()
    except ModeSelectionError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except McpError as exc:
        print(f"mcp error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    finally:
        try:
            if host is not None:
                host.close()
        finally:
            if runtime is not None:
                runtime.close()


if __name__ == "__main__":
    main()
