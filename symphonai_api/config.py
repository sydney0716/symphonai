"""Load layered TOML configuration with per-leaf provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
import tomllib

from symphonai_api.permissions import (
    PermissionPolicy,
    _contains_path,
    _intersect_shell_allowlists,
    _intersect_write_scopes,
    _is_prefix,
)


class ConfigError(ValueError):
    """A malformed or invalid config file, naming the file and the key."""


class Scope(str, Enum):
    USER = "user"
    PROJECT = "project"
    PRIVATE = "private"
    SESSION = "session"


@dataclass(frozen=True)
class Provenance:
    key: str
    scope: Scope
    source: Path | None


@dataclass(frozen=True)
class ConfigLayer:
    scope: Scope
    source: Path | None
    values: Mapping[str, object]


@dataclass(frozen=True)
class ResolvedConfig:
    values: Mapping[str, object]
    provenance: Mapping[str, Provenance]
    layers: tuple[ConfigLayer, ...] = ()

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def scope_of(self, key: str) -> Scope | None:
        origin = self.provenance.get(key)
        return origin.scope if origin is not None else None


_CEILING_KEYS = {
    "allowed_write_scope",
    "shell_enabled",
    "shell_allowlist",
    "fetch_enabled",
    "fetch_allowlist",
    "modes",
}
_SECTIONS = ("agents", "hooks", "skills", "mcp", "plugins", "trust")
_VALID_MODES = {"auto", "prompt", "plan", "accept_edits"}


def _raise(source: Path | None, key: str, detail: str) -> None:
    location = str(source) if source is not None else "<session>"
    raise ConfigError(f"{location}: {key}: {detail}")


def _unknown_keys(
    source: Path | None,
    key: str,
    values: Mapping[str, object],
    allowed: set[str],
) -> None:
    unknown = values.keys() - allowed
    if unknown:
        _raise(source, f"{key}.{sorted(unknown)[0]}".lstrip("."), "unknown key")


def _table(
    source: Path | None,
    values: Mapping[str, object],
    key: str,
) -> Mapping[str, object] | None:
    if key not in values:
        return None
    value = values[key]
    if not isinstance(value, Mapping):
        _raise(source, key, "must be a table")
    return value


def _string(source: Path | None, key: str, value: object) -> str:
    if not isinstance(value, str):
        _raise(source, key, "must be a string")
    return value


def _boolean(source: Path | None, key: str, value: object) -> bool:
    if type(value) is not bool:
        _raise(source, key, "must be a boolean")
    return value


def _strings(source: Path | None, key: str, value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _raise(source, key, "must be an array of strings")
    return value


def _validate(source: Path | None, values: Mapping[str, object]) -> None:
    _unknown_keys(source, "", values, set(_SECTIONS))
    agents = _table(source, values, "agents")
    if agents is None:
        return
    _unknown_keys(source, "agents", agents, {"directory", "ceiling"})
    if "directory" in agents:
        _string(source, "agents.directory", agents["directory"])
    ceiling = _table(source, agents, "ceiling")
    if ceiling is None:
        return
    _unknown_keys(source, "agents.ceiling", ceiling, _CEILING_KEYS)
    for key in ("allowed_write_scope", "fetch_allowlist", "modes"):
        if key in ceiling:
            entries = _strings(source, f"agents.ceiling.{key}", ceiling[key])
            if key == "modes":
                invalid = next((mode for mode in entries if mode not in _VALID_MODES), None)
                if invalid is not None:
                    valid = ", ".join(sorted(_VALID_MODES))
                    _raise(
                        source,
                        "agents.ceiling.modes",
                        f"must contain only: {valid}",
                    )
    for key in ("shell_enabled", "fetch_enabled"):
        if key in ceiling:
            _boolean(source, f"agents.ceiling.{key}", ceiling[key])
    if "shell_allowlist" in ceiling:
        value = ceiling["shell_allowlist"]
        full_key = "agents.ceiling.shell_allowlist"
        if not isinstance(value, list):
            _raise(source, full_key, "must be an array of command arrays")
        for command in value:
            _strings(source, full_key, command)


def _read_toml(path: Path) -> Mapping[str, object] | None:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except OSError as exc:
        _raise(path, "file", f"could not be read: {exc}")
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        _raise(path, "toml", f"could not be parsed: {exc}")


def _flatten(values: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in values.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, dotted))
        else:
            flattened[dotted] = value
    return flattened


def load_config(
    *,
    repo_root: Path,
    home: Path | None = None,
    session: Mapping[str, object] | None = None,
) -> ResolvedConfig:
    """Merge user, project, private, and session values by dotted leaf key."""
    config_home = Path.home() if home is None else Path(home)
    root = Path(repo_root)
    sources: tuple[tuple[Scope, Path | None, Mapping[str, object] | None], ...] = (
        (
            Scope.USER,
            config_home / ".symphonai" / "config.toml",
            None,
        ),
        (
            Scope.PROJECT,
            root / ".symphonai" / "config.toml",
            None,
        ),
        (
            Scope.PRIVATE,
            root / ".symphonai" / "config.local.toml",
            None,
        ),
        (Scope.SESSION, None, session),
    )
    resolved: dict[str, object] = {}
    provenance: dict[str, Provenance] = {}
    layers: list[ConfigLayer] = []
    for scope, source, supplied in sources:
        if scope is Scope.SESSION:
            values = supplied
        else:
            if source is None:
                raise AssertionError("file-backed config scope has no source")
            values = _read_toml(source)
        if values is None:
            continue
        if not isinstance(values, Mapping):
            _raise(source, "config", "must be a table")
        _validate(source, values)
        flattened = _flatten(values)
        if flattened:
            layers.append(
                ConfigLayer(
                    scope=scope,
                    source=source,
                    values=MappingProxyType(flattened.copy()),
                )
            )
        for key, value in flattened.items():
            resolved[key] = value
            provenance[key] = Provenance(key=key, scope=scope, source=source)
    return ResolvedConfig(
        values=MappingProxyType(resolved),
        provenance=MappingProxyType(provenance),
        layers=tuple(layers),
    )


@dataclass(frozen=True)
class CapabilityCeiling:
    allowed_write_scope: tuple[Path, ...] | None = None
    shell_enabled: bool | None = None
    shell_allowlist: tuple[tuple[str, ...], ...] | None = None
    fetch_enabled: bool | None = None
    fetch_allowlist: tuple[str, ...] | None = None
    modes: tuple[str, ...] | None = None

    def meet(self, other: "CapabilityCeiling") -> "CapabilityCeiling":
        """Return the tighter of two ceilings, field by field."""

        def both(
            left: bool | None,
            right: bool | None,
        ) -> bool | None:
            if left is None:
                return right
            if right is None:
                return left
            return left and right

        def intersection(
            left: tuple[str, ...] | None,
            right: tuple[str, ...] | None,
        ) -> tuple[str, ...] | None:
            if left is None:
                return right
            if right is None:
                return left
            return tuple(dict.fromkeys(value for value in left if value in right))

        if self.allowed_write_scope is None:
            write_scope = other.allowed_write_scope
        elif other.allowed_write_scope is None:
            write_scope = self.allowed_write_scope
        else:
            write_scope = tuple(
                _intersect_write_scopes(
                    list(self.allowed_write_scope),
                    list(other.allowed_write_scope),
                )
            )

        if self.shell_allowlist is None:
            shell_allowlist = other.shell_allowlist
        elif other.shell_allowlist is None:
            shell_allowlist = self.shell_allowlist
        else:
            shell_allowlist = tuple(
                _intersect_shell_allowlists(
                    list(self.shell_allowlist),
                    list(other.shell_allowlist),
                )
            )

        return CapabilityCeiling(
            allowed_write_scope=write_scope,
            shell_enabled=both(self.shell_enabled, other.shell_enabled),
            shell_allowlist=shell_allowlist,
            fetch_enabled=both(self.fetch_enabled, other.fetch_enabled),
            fetch_allowlist=intersection(
                self.fetch_allowlist,
                other.fetch_allowlist,
            ),
            modes=intersection(self.modes, other.modes),
        )

    @classmethod
    def from_config(
        cls,
        config: ResolvedConfig,
        *,
        repo_root: Path,
    ) -> "CapabilityCeiling":
        prefix = "agents.ceiling."
        result = cls()
        for layer in config.layers:
            if not any(key.startswith(prefix) for key in layer.values):
                continue
            result = result.meet(
                cls._from_layer(
                    layer.values,
                    source=layer.source,
                    repo_root=repo_root,
                )
            )
        return result

    @classmethod
    def _from_layer(
        cls,
        values: Mapping[str, object],
        *,
        source: Path | None,
        repo_root: Path,
    ) -> "CapabilityCeiling":
        prefix = "agents.ceiling."

        def value(key: str) -> object | None:
            return values.get(prefix + key)

        def boolean(key: str) -> bool | None:
            raw = value(key)
            if raw is None:
                return None
            return _boolean(source, prefix + key, raw)

        modes: tuple[str, ...] | None = None
        raw_modes = value("modes")
        if raw_modes is not None:
            parsed_modes = tuple(_strings(source, prefix + "modes", raw_modes))
            invalid = next((mode for mode in parsed_modes if mode not in _VALID_MODES), None)
            if invalid is not None:
                valid = ", ".join(sorted(_VALID_MODES))
                _raise(source, prefix + "modes", f"must contain only: {valid}")
            modes = parsed_modes

        write_scope: tuple[Path, ...] | None = None
        raw_write_scope = value("allowed_write_scope")
        if raw_write_scope is not None:
            entries = _strings(
                source,
                prefix + "allowed_write_scope",
                raw_write_scope,
            )
            root = Path(repo_root)
            write_scope = tuple((root / entry).resolve() for entry in entries)

        shell_allowlist: tuple[tuple[str, ...], ...] | None = None
        raw_shell_allowlist = value("shell_allowlist")
        if raw_shell_allowlist is not None:
            if not isinstance(raw_shell_allowlist, list):
                _raise(
                    source,
                    prefix + "shell_allowlist",
                    "must be an array of command arrays",
                )
            shell_allowlist = tuple(
                tuple(
                    _strings(
                        source,
                        prefix + "shell_allowlist",
                        command,
                    )
                )
                for command in raw_shell_allowlist
            )

        fetch_allowlist: tuple[str, ...] | None = None
        raw_fetch_allowlist = value("fetch_allowlist")
        if raw_fetch_allowlist is not None:
            fetch_allowlist = tuple(
                host.casefold().rstrip(".")
                for host in _strings(
                    source,
                    prefix + "fetch_allowlist",
                    raw_fetch_allowlist,
                )
            )

        return cls(
            allowed_write_scope=write_scope,
            shell_enabled=boolean("shell_enabled"),
            shell_allowlist=shell_allowlist,
            fetch_enabled=boolean("fetch_enabled"),
            fetch_allowlist=fetch_allowlist,
            modes=modes,
        )

    def refuse(self, policy: PermissionPolicy, *, source: Path) -> None:
        """Raise when a policy asks for a capability outside this ceiling."""
        if self.allowed_write_scope is not None:
            for requested in policy.allowed_write_scope:
                if not any(
                    _contains_path(allowed, requested)
                    for allowed in self.allowed_write_scope
                ):
                    _raise(source, "allowed_write_scope", "exceeds capability ceiling")
        if self.shell_enabled is False and policy.shell_enabled:
            _raise(source, "shell_enabled", "exceeds capability ceiling")
        if self.shell_allowlist is not None:
            for command in policy.shell_allowlist:
                if not any(
                    _is_prefix(allowed, command)
                    for allowed in self.shell_allowlist
                ):
                    _raise(source, "shell_allowlist", "exceeds capability ceiling")
        if self.fetch_enabled is False and policy.fetch_enabled:
            _raise(source, "fetch_enabled", "exceeds capability ceiling")
        if self.fetch_allowlist is not None:
            allowed_hosts = {
                host.casefold().rstrip(".") for host in self.fetch_allowlist
            }
            for host in policy.fetch_allowlist:
                if host not in allowed_hosts:
                    _raise(source, "fetch_allowlist", "exceeds capability ceiling")
        if self.modes is not None and policy.mode not in self.modes:
            _raise(source, "modes", "exceeds capability ceiling")
