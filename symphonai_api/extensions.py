"""Resolve configured runtime extensions once, before a run starts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import warnings
from pathlib import Path
from types import MappingProxyType

from symphonai_api.agent_spec import AgentSpec
from symphonai_api.config import (
    CapabilityCeiling, ConfigError, ConfigLayer, Provenance, ResolvedConfig,
    Scope, load_config,
)
from symphonai_api.discovery import Offered, discover
from symphonai_api.hooks import HookRunner, HookSpec, hooks_from_config
from symphonai_api.mcp import McpServerSpec, mcp_servers_from_config
from symphonai_api.lsp import LspServerSpec, lsp_servers_from_config
from symphonai_api.plugins import Plugin, PluginWarning
from symphonai_api.skills import Skill
from symphonai_api.trust import TrustList, trust_from_config


def _skip_config_section(
    config: ResolvedConfig, error: ConfigError,
) -> ResolvedConfig | None:
    message = str(error)
    for layer in config.layers:
        if layer.scope not in {Scope.PROJECT, Scope.PRIVATE} or layer.source is None:
            continue
        prefix = f"{layer.source}:"
        if not message.startswith(prefix):
            continue
        detail = message[len(prefix):].lstrip()
        sections = {key.split(".", 1)[0] for key in layer.values}
        section = next(
            (name for name in sorted(sections, key=len, reverse=True)
             if detail == name or detail.startswith(f"{name}.")
             or detail.startswith(f"{name}[") or detail.startswith(f"{name}:")),
            None,
        )
        if section is None:
            return None

        values: dict[str, object] = {}
        provenance: dict[str, Provenance] = {}
        layers: list[ConfigLayer] = []
        for current in config.layers:
            kept = {
                key: value for key, value in current.values.items()
                if not (
                    current is layer
                    and (key == section or key.startswith(f"{section}."))
                )
            }
            if not kept:
                continue
            layers.append(ConfigLayer(current.scope, current.source, MappingProxyType(kept)))
            for key, value in kept.items():
                values[key] = value
                provenance[key] = Provenance(key, current.scope, current.source)
        if section == "hooks" and "hooks were refused" in message:
            reason = detail.removeprefix("hooks:").strip()
            warning = f"{layer.source}: {reason or 'hooks were refused'}"
        else:
            reason = detail or f"{section} configuration was rejected"
            warning = f"{layer.source}: {reason}; {section} section skipped"
        return ResolvedConfig(
            values=MappingProxyType(values),
            provenance=MappingProxyType(provenance),
            layers=tuple(layers),
            warnings=(*config.warnings, warning),
        )
    return None


@dataclass(frozen=True)
class Extensions:
    config: ResolvedConfig
    trust: TrustList
    ceiling: CapabilityCeiling
    hooks: tuple[HookSpec, ...]
    mcp_servers: tuple[McpServerSpec, ...]
    agents: Mapping[str, AgentSpec]
    skills: Mapping[str, Skill]
    plugins: Mapping[str, Plugin]
    withheld: tuple[Offered, ...]
    lsp_servers: tuple[LspServerSpec, ...] = ()

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
    while True:
        try:
            trust = trust_from_config(config)
            ceiling = CapabilityCeiling.from_config(config, repo_root=repo_root)
            hooks = hooks_from_config(config, repo_root=repo_root, trust=trust)
            mcp_servers = mcp_servers_from_config(
                config,
                repo_root=repo_root,
                trust=trust,
            )
            lsp_servers = lsp_servers_from_config(config, repo_root=repo_root, trust=trust)
            with warnings.catch_warnings(record=True) as plugin_warnings:
                warnings.simplefilter("always", PluginWarning)
                discovered = discover(
                    repo_root=repo_root,
                    home=home,
                    trust=trust,
                    ceiling=ceiling,
                )
            config = replace(config, warnings=(*config.warnings, *(str(item.message) for item in plugin_warnings)))
            return Extensions(
                config,
                trust,
                ceiling,
                hooks,
                mcp_servers,
                discovered.agents,
                discovered.skills,
                discovered.plugins,
                discovered.withheld,
                lsp_servers,
            )
        except ConfigError as exc:
            config = _skip_config_section(config, exc)
            if config is None:
                raise
