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
from symphonai_api.mcp import McpError
from symphonai_api.mcp_pool import McpPool
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.runner import standard_tool_registry
from symphonai_api.session import default_sessions_root
from symphonai_host.server import HostServer, _provider
from symphonai_host.credentials import CredentialError, apply_to_environment, load
from symphonai_host.sessions import DEFAULT_CLEANUP_PERIOD_DAYS, prune_sessions


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--permission-mode",
        choices=("auto", "prompt", "plan", "accept_edits"),
        default="prompt",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-turns", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _arguments(argv)
    try:
        apply_to_environment(os.environ, load())
    except CredentialError as exc:
        print(f"credential error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    try:
        extensions = load_extensions(repo_root=arguments.repo_root)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    try:
        period = extensions.config.values.get(
            "sessions.cleanup_period_days", DEFAULT_CLEANUP_PERIOD_DAYS
        )
        prune_sessions(default_sessions_root(), period_days=period, now=datetime.now(timezone.utc))
    except Exception:
        pass
    pool = McpPool(
        extensions.mcp_servers,
        cwd=arguments.repo_root,
        reserved_names=set(standard_tool_registry()),
    )
    try:
        mcp_tools = pool.start()
    except McpError as exc:
        print(f"mcp error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    host = None
    try:
        host = HostServer(
            _provider(),
            PermissionPolicy(
                repo_root=arguments.repo_root,
                mode=arguments.permission_mode,
            ),
            max_turns=arguments.max_turns,
            extensions=extensions,
            mcp_tools=mcp_tools,
        )

        def shutdown(signum, frame) -> None:
            threading.Thread(target=host.close, daemon=True).start()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        host.print_handshake()
        host.serve_forever()
    finally:
        if host is not None:
            host.close()
        pool.close()


if __name__ == "__main__":
    main()
