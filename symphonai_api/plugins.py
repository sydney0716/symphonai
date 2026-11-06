"""Load directories that bundle existing SymphonAI extension formats."""

from __future__ import annotations

from symphonai_api.config import ConfigSource

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import re
from types import MappingProxyType
import warnings

from symphonai_api import config as config_module
from symphonai_api.agent_file import AgentFileError, load_agent_directory
from symphonai_api.agent_spec import AgentSpec
from symphonai_api.config import ConfigError, Provenance, ResolvedConfig, Scope
from symphonai_api.hooks import HookSpec, hooks_from_config
from symphonai_api.mcp import McpServerSpec, mcp_servers_from_config
from symphonai_api.skills import Skill, SkillError, load_skill_directory


class PluginWarning(UserWarning):
    """A malformed plugin was skipped while loading a project."""


class PluginError(ValueError):
    """A malformed plugin, naming the plugin and the offending member."""


@dataclass(frozen=True)
class Plugin:
    name: str
    version: str
    description: str
    path: Path
    agents: Mapping[str, AgentSpec]
    skills: Mapping[str, Skill]
    hooks: tuple[HookSpec, ...]
    mcp_servers: tuple[McpServerSpec, ...]


_MANIFEST_KEYS = {"name", "version", "description"}
_PLUGIN_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _raise(path: Path, member: str, detail: str) -> None:
    ConfigSource(path, PluginError, location=f"plugin {path.name!r} at {path}").error(member, detail)


def _load_manifest(path: Path) -> Mapping[str, object]:
    values = ConfigSource(path, PluginError, location=f"plugin {path.name!r} at {path}").read_toml(path=path / "plugin.toml", read_key="plugin.toml", parse_key="plugin.toml")
    ConfigSource(path, PluginError, location=f"plugin {path.name!r} at {path}").unknown_keys("", values, _MANIFEST_KEYS)
    return values


def _required_text(
    path: Path,
    manifest: Mapping[str, object],
    key: str,
) -> str:
    return ConfigSource(path, PluginError, location=f"plugin {path.name!r} at {path}").required_text(manifest, key)


def _member_config(path: Path) -> ResolvedConfig:
    source = path / "config.toml"
    values = config_module._read_toml(source)
    if values is None:
        return ResolvedConfig(MappingProxyType({}), MappingProxyType({}))

    # Config owns this schema; the plugin adapter only restores file provenance.
    config_module._validate(source, values)
    flattened = config_module._flatten(values)
    provenance = {
        key: Provenance(key=key, scope=Scope.USER, source=source)
        for key in flattened
    }
    return ResolvedConfig(
        values=MappingProxyType(flattened),
        provenance=MappingProxyType(provenance),
    )


def _member_failure(path: Path, member: str, exc: ValueError) -> None:
    _raise(path, member, str(exc))


def load_plugin(path: Path, *, repo_root: Path, **loader_kwargs) -> Plugin:
    """Load one plugin entirely, reusing every existing member loader."""
    plugin_path = Path(path)
    manifest = _load_manifest(plugin_path)
    name = _required_text(plugin_path, manifest, "name")
    version = _required_text(plugin_path, manifest, "version")
    description = _required_text(plugin_path, manifest, "description")
    if _PLUGIN_NAME_PATTERN.fullmatch(name) is None:
        _raise(
            plugin_path,
            "name",
            f"value {name!r} must match ^[A-Za-z_][A-Za-z0-9_]*$",
        )
    if name != plugin_path.name:
        _raise(
            plugin_path,
            "name",
            f"declared name {name!r} must match directory name {plugin_path.name!r}",
        )

    try:
        bare_agents = load_agent_directory(
            plugin_path / "agents",
            repo_root=repo_root,
            **loader_kwargs,
        )
    except AgentFileError as exc:
        _member_failure(plugin_path, "agents", exc)
    try:
        bare_skills = load_skill_directory(plugin_path / "skills")
    except SkillError as exc:
        _member_failure(plugin_path, "skills", exc)
    try:
        config = _member_config(plugin_path)
    except ConfigError as exc:
        _member_failure(plugin_path, "config.toml", exc)
    try:
        hooks = hooks_from_config(config, repo_root=repo_root)
    except ConfigError as exc:
        _member_failure(plugin_path, "hooks", exc)
    try:
        bare_servers = mcp_servers_from_config(config, repo_root=repo_root)
    except ConfigError as exc:
        _member_failure(plugin_path, "mcp_servers", exc)

    agents = {
        f"{name}/{member_name}": agent
        for member_name, agent in bare_agents.items()
    }
    skills = {
        f"{name}/{member_name}": skill
        for member_name, skill in bare_skills.items()
    }
    mcp_servers = tuple(
        replace(server, name=f"{name}__{server.name}")
        for server in bare_servers
    )
    return Plugin(
        name=name,
        version=version,
        description=description,
        path=plugin_path,
        agents=MappingProxyType(agents),
        skills=MappingProxyType(skills),
        hooks=hooks,
        mcp_servers=mcp_servers,
    )


def load_plugin_directory(
    path: Path,
    *,
    repo_root: Path,
    **loader_kwargs,
) -> dict[str, Plugin]:
    """Load every direct plugin directory, ordered by directory name."""
    directory = Path(path)
    if not directory.exists():
        return {}
    plugins = {}
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        try:
            plugins[child.name] = load_plugin(child, repo_root=repo_root, **loader_kwargs)
        except PluginError as exc:
            warnings.warn(str(exc), PluginWarning, stacklevel=2)
    return plugins
