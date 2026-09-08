"""Resolve configured runtime extensions once, before a run starts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from symphonai_api.config import CapabilityCeiling, ResolvedConfig, load_config
from symphonai_api.hooks import HookRunner, HookSpec, hooks_from_config
from symphonai_api.mcp import McpServerSpec, mcp_servers_from_config
from symphonai_api.trust import TrustList, trust_from_config


@dataclass(frozen=True)
class Extensions:
    config: ResolvedConfig
    trust: TrustList
    ceiling: CapabilityCeiling
    hooks: tuple[HookSpec, ...]
    mcp_servers: tuple[McpServerSpec, ...]

    def hook_runner(self, *, cwd: Path) -> HookRunner | None:
        """Build a runner over configured hooks, or None when there are none."""
        if not self.hooks:
            return None
        return HookRunner(self.hooks, cwd=cwd)


def load_extensions(
    *,
    repo_root: Path,
    home: Path | None = None,
    session: Mapping[str, object] | None = None,
) -> Extensions:
    """Resolve all configuration-backed extension values atomically."""
    config = load_config(repo_root=repo_root, home=home, session=session)
    trust = trust_from_config(config)
    ceiling = CapabilityCeiling.from_config(config, repo_root=repo_root)
    hooks = hooks_from_config(config, repo_root=repo_root, trust=trust)
    mcp_servers = mcp_servers_from_config(
        config,
        repo_root=repo_root,
        trust=trust,
    )
    return Extensions(config, trust, ceiling, hooks, mcp_servers)
