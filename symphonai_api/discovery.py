"""Discover user and repository extensions without granting implicit trust."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

from symphonai_api.agent_file import AgentFileError, load_agent_directory
from symphonai_api.agent_spec import AgentSpec, ModelSelector
from symphonai_api.budgets import PriceTable
from symphonai_api.config import CapabilityCeiling, Scope
from symphonai_api.plugins import Plugin, PluginError, load_plugin_directory
from symphonai_api.skills import Skill, SkillError, load_skill_directory
from symphonai_api.trust import TrustList


class DiscoveryError(ValueError):
    """Two scopes offering the same name, or a member that failed to load."""


@dataclass(frozen=True)
class Offered:
    """What a scope has, whether or not it was allowed."""

    scope: Scope
    directory: Path
    names: tuple[str, ...]


@dataclass(frozen=True)
class Discovered:
    agents: Mapping[str, AgentSpec]
    skills: Mapping[str, Skill]
    plugins: Mapping[str, Plugin]
    withheld: tuple[Offered, ...]


_T = TypeVar("_T")


def _load(
    loader: Callable[[Path], dict[str, _T]],
    directory: Path,
    *,
    scope: Scope,
    kind: str,
    errors: tuple[type[ValueError], ...],
) -> dict[str, _T]:
    try:
        return loader(directory)
    except errors as exc:
        raise DiscoveryError(
            f"{scope.value} {kind} directory {directory}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _offered_names(directory: Path, kind: str) -> tuple[str, ...]:
    if kind == "agents":
        return tuple(
            path.stem
            for path in sorted(directory.glob("*.toml"))
            if path.is_file()
        )
    if kind == "skills":
        return tuple(
            path.stem
            for path in sorted(directory.glob("*.md"))
            if path.is_file()
        )
    return tuple(
        path.name for path in sorted(directory.iterdir()) if path.is_dir()
    )


def _combine(
    user: Mapping[str, _T],
    project: Mapping[str, _T],
    *,
    kind: str,
    user_directory: Path,
    project_directory: Path,
) -> Mapping[str, _T]:
    duplicate = next((name for name in user if name in project), None)
    if duplicate is not None:
        raise DiscoveryError(
            f"{kind} {duplicate!r} is offered by user scope at "
            f"{user_directory} and project scope at {project_directory}"
        )
    return MappingProxyType({**user, **project})


def discover(
    *,
    repo_root: Path,
    home: Path | None = None,
    trust: TrustList | None = None,
    ceiling: CapabilityCeiling | None = None,
    price_table: PriceTable | None = None,
    default_model: ModelSelector | None = None,
) -> Discovered:
    """Discover two directory scopes, withholding untrusted repository content."""
    root = Path(repo_root)
    user_root = (Path.home() if home is None else Path(home)) / ".symphonai"
    project_root = root / ".symphonai"
    withheld: list[Offered] = []

    def agent_loader(directory: Path) -> dict[str, AgentSpec]:
        return load_agent_directory(
            directory,
            repo_root=root,
            price_table=price_table,
            default_model=default_model,
            ceiling=ceiling,
        )

    def plugin_loader(directory: Path) -> dict[str, Plugin]:
        return load_plugin_directory(
            directory,
            repo_root=root,
            price_table=price_table,
            default_model=default_model,
            ceiling=ceiling,
        )

    specifications = (
        ("agents", agent_loader, (AgentFileError,)),
        ("skills", load_skill_directory, (SkillError,)),
        ("plugins", plugin_loader, (PluginError,)),
    )
    combined: dict[str, Mapping] = {}
    for kind, loader, errors in specifications:
        user_directory = user_root / kind
        project_directory = project_root / kind
        user_members = _load(
            loader,
            user_directory,
            scope=Scope.USER,
            kind=kind,
            errors=errors,
        )
        if trust is not None and trust.allows(root, kind):
            project_members = _load(
                loader,
                project_directory,
                scope=Scope.PROJECT,
                kind=kind,
                errors=errors,
            )
        else:
            project_members = {}
            if project_directory.exists():
                withheld.append(
                    Offered(
                        scope=Scope.PROJECT,
                        directory=project_directory,
                        names=_offered_names(project_directory, kind),
                    )
                )
        combined[kind] = _combine(
            user_members,
            project_members,
            kind=kind,
            user_directory=user_directory,
            project_directory=project_directory,
        )
    return Discovered(
        agents=combined["agents"],
        skills=combined["skills"],
        plugins=combined["plugins"],
        withheld=tuple(withheld),
    )
