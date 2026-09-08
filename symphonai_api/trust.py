"""Owner-controlled trust grants for repository-provided extensions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from symphonai_api.config import ConfigError, ResolvedConfig, Scope


CAPABILITIES: tuple[str, ...] = (
    "agents",
    "hooks",
    "mcp",
    "plugins",
    "skills",
)


@dataclass(frozen=True)
class RepositoryTrust:
    root: Path
    allow: frozenset[str]
    source: Path | None


@dataclass(frozen=True)
class TrustList:
    entries: tuple[RepositoryTrust, ...] = ()

    def allows(self, repo_root: Path, capability: str) -> bool:
        """True when this exact resolved root was granted this capability."""
        if capability not in CAPABILITIES:
            return False
        resolved = Path(repo_root).expanduser().resolve()
        return any(
            entry.root == resolved and capability in entry.allow
            for entry in self.entries
        )


def _config_error(
    source: Path | None,
    index: int | None,
    field: str,
    detail: str,
) -> ConfigError:
    location = str(source) if source is not None else "<session>"
    prefix = "trust.repositories"
    if index is not None:
        prefix += f"[{index}]"
    if field:
        prefix += f".{field}"
    return ConfigError(f"{location}: {prefix}: {detail}")


def trust_from_config(config: ResolvedConfig) -> TrustList:
    """Parse the owner-provided exact-root repository trust list."""
    raw_entries = config.get("trust.repositories", [])
    origin = config.provenance.get("trust.repositories")
    source = origin.source if origin is not None else None
    if origin is not None and origin.scope in (Scope.PROJECT, Scope.PRIVATE):
        raise _config_error(
            source,
            None,
            "",
            "a repository may not grant itself trust; define grants in "
            "~/.symphonai/config.toml",
        )
    if not isinstance(raw_entries, list):
        raise _config_error(source, None, "", "must be an array of tables")

    parsed: list[RepositoryTrust] = []
    root_indices: dict[Path, int] = {}
    valid_names = ", ".join(CAPABILITIES)
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise _config_error(source, index, "", "must be a table")
        unknown = raw_entry.keys() - {"root", "allow"}
        if unknown:
            raise _config_error(
                source,
                index,
                str(sorted(unknown)[0]),
                "unknown key",
            )
        raw_root = raw_entry.get("root")
        if not isinstance(raw_root, str) or not raw_root.strip():
            raise _config_error(source, index, "root", "must be a non-empty string")
        root = Path(raw_root).expanduser().resolve()
        if root in root_indices:
            raise _config_error(
                source,
                None,
                "",
                f"duplicate root {str(root)!r} at indices "
                f"{root_indices[root]} and {index}",
            )
        root_indices[root] = index

        raw_allow = raw_entry.get("allow")
        if not isinstance(raw_allow, list) or not all(
            isinstance(capability, str) for capability in raw_allow
        ):
            raise _config_error(source, index, "allow", "must be an array of strings")
        unknown_capability = next(
            (
                capability
                for capability in raw_allow
                if capability not in CAPABILITIES
            ),
            None,
        )
        if unknown_capability is not None:
            raise _config_error(
                source,
                index,
                "allow",
                f"unknown capability {unknown_capability!r}; valid values: {valid_names}",
            )
        parsed.append(
            RepositoryTrust(
                root=root,
                allow=frozenset(raw_allow),
                source=source,
            )
        )
    return TrustList(tuple(parsed))
