"""Load layered TOML configuration with per-leaf provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import MappingProxyType
import tomllib

from symphonai_api.budgets import RunBudget
from symphonai_api.cost import PriceTable, load_price_table
from symphonai_api.model_table import model_capabilities, resolve_effort
from symphonai_api.permissions import (
    ALWAYS_DENY_COMMANDS,
    BUDGET_KEYS, RENAMED_MODES, SettingsParser,
    DEFAULT_FORBIDDEN_PATTERNS,
    PermissionPolicy,
    _contains_path,
    _intersect_shell_allowlists,
    _intersect_write_scopes,
    _is_prefix,
)
from symphonai_api.paths import symphonai_home
from symphonai_api.web_search import search_endpoint


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
    warnings: tuple[str, ...] = ()

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
_SECTIONS = ("agents", "hooks", "skills", "mcp", "plugins", "trust", "sessions", "budgets", "search", "models", "fallback", "memory", "sandbox", "lsp", "worktree", "permissions", "roles", "compaction", "always_deny")
_PROVIDER_NAMES = {
    "anthropic", "gemini", "openai", "deepseek", "grok", "groq", "kimi",
    "ollama", "openrouter", "qwen", "together",
}
_BUILTIN_ROLES = {"planner", "implementer", "reviewer", "worker", "explorer"}


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
    return SettingsParser(lambda field, detail: _raise(source, field, detail)).string(key, value)

def _boolean(source: Path | None, key: str, value: object) -> bool:
    return SettingsParser(lambda field, detail: _raise(source, field, detail)).boolean(key, value)

def _strings(source: Path | None, key: str, value: object) -> list[str]:
    return SettingsParser(lambda field, detail: _raise(source, field, detail)).strings(key, value)

def _validate(source: Path | None, values: Mapping[str, object]) -> None:
    _unknown_keys(source, "", values, set(_SECTIONS))
    always_deny = _table(source, values, "always_deny")
    if always_deny is not None:
        _unknown_keys(source, "always_deny", always_deny, {
            "lift", "allow_uncheckable_shell", "allow_discarding_work",
        })
        for key in ("allow_uncheckable_shell", "allow_discarding_work"):
            if key in always_deny:
                _boolean(source, f"always_deny.{key}", always_deny[key])
        if "lift" in always_deny:
            allowed = {" ".join(command) for command in ALWAYS_DENY_COMMANDS}
            for value in _strings(source, "always_deny.lift", always_deny["lift"]):
                if value not in allowed:
                    _raise(source, "always_deny.lift", f"unsupported built-in command {value!r}")
    compaction = _table(source, values, "compaction")
    if compaction is not None:
        _unknown_keys(source, "compaction", compaction, {"limit"})
        if "limit" in compaction and (
            type(compaction["limit"]) is not int or compaction["limit"] <= 0
        ):
            _raise(source, "compaction.limit", "must be a positive integer")
    permissions = _table(source, values, "permissions")
    if permissions is not None:
        _unknown_keys(source, "permissions", permissions, {"shell", "fetch"})
        if "shell" in permissions:
            shell = permissions["shell"]
            if not isinstance(shell, list):
                _raise(source, "permissions.shell", "must be an array of command arrays")
            for prefix in shell:
                entries = _strings(source, "permissions.shell", prefix)
                if not entries or any(not entry.strip() for entry in entries):
                    _raise(source, "permissions.shell", "prefixes must contain non-blank arguments")
        if "fetch" in permissions:
            for host in _strings(source, "permissions.fetch", permissions["fetch"]):
                normalized = host.casefold().rstrip(".")
                if (
                    not normalized
                    or any(char in normalized for char in "/:@?#\\")
                    or any(char.isspace() for char in normalized)
                ):
                    _raise(source, "permissions.fetch", "entries must be host names without a scheme or path")
    memory = _table(source, values, "memory")
    if memory is not None:
        _unknown_keys(source, "memory", memory, {"model", "provider", "extract"})
        if "model" in memory and not _string(source, "memory.model", memory["model"]).strip():
            _raise(source, "memory.model", "must be a non-empty string")
        if "extract" in memory:
            _boolean(source, "memory.extract", memory["extract"])
        if "provider" in memory:
            provider = _string(source, "memory.provider", memory["provider"])
            if provider not in _PROVIDER_NAMES:
                _raise(source, "memory.provider", f"unknown provider {provider!r}")
    roles = _table(source, values, "roles")
    if roles is not None:
        for role, raw in roles.items():
            key = f"roles.{role}"
            if role not in _BUILTIN_ROLES:
                _raise(source, key, "unknown built-in role")
            if not isinstance(raw, Mapping):
                _raise(source, key, "must be a table")
            _unknown_keys(source, key, raw, {"provider", "model", "effort"})
            for required in ("provider", "model"):
                if required not in raw:
                    _raise(source, f"{key}.{required}", "is required")
                if not _string(source, f"{key}.{required}", raw[required]).strip():
                    _raise(source, f"{key}.{required}", "must be a non-empty string")
            provider = raw["provider"]
            if provider not in _PROVIDER_NAMES:
                _raise(source, f"{key}.provider", f"unknown provider {provider!r}")
            if "effort" in raw:
                effort = _string(source, f"{key}.effort", raw["effort"])
                if not effort.strip():
                    _raise(source, f"{key}.effort", "must be a non-empty string")
                capability = next((
                    item for item in model_capabilities()
                    if item.provider == provider and item.model == raw["model"]
                ), None)
                if capability is not None:
                    try:
                        resolve_effort(capability.wire_format, raw["model"], effort)
                    except ValueError as exc:
                        _raise(source, f"{key}.effort", str(exc))
    models = _table(source, values, "models")
    if models is not None:
        _unknown_keys(source, "models", models, _PROVIDER_NAMES)
        for provider in models:
            _strings(source, f"models.{provider}", models[provider])
    fallback = _table(source, values, "fallback")
    if fallback is not None:
        _unknown_keys(source, "fallback", fallback, _PROVIDER_NAMES)
        for provider, configured_models in fallback.items():
            entries = _strings(source, f"fallback.{provider}", configured_models)
            if any(not model.strip() for model in entries):
                _raise(source, f"fallback.{provider}", "model ids must be non-blank strings")
    search = _table(source, values, "search")
    if search is not None:
        _unknown_keys(source, "search", search, {"endpoint"})
        if "endpoint" in search:
            key = _string(source, "search.endpoint", search["endpoint"])
            try:
                search_endpoint(key)
            except KeyError:
                _raise(source, "search.endpoint", f"unknown search endpoint {key!r}")
    sandbox = _table(source, values, "sandbox")
    if sandbox is not None:
        _unknown_keys(source, "sandbox", sandbox, {"shell", "network", "readable"})
        for key in ("shell", "network"):
            if key in sandbox:
                _boolean(source, f"sandbox.{key}", sandbox[key])
        if "readable" in sandbox:
            for pattern in _strings(source, "sandbox.readable", sandbox["readable"]):
                if pattern not in DEFAULT_FORBIDDEN_PATTERNS or pattern == ".git/":
                    _raise(source, "sandbox.readable", f"not an eligible forbidden pattern: {pattern!r}")
    worktree = _table(source, values, "worktree")
    if worktree is not None:
        _unknown_keys(source, "worktree", worktree, {"symlink", "setup"})
        if "symlink" in worktree:
            for entry in _strings(source, "worktree.symlink", worktree["symlink"]):
                path = Path(entry)
                if not entry or path.is_absolute() or ".." in path.parts:
                    _raise(
                        source,
                        "worktree.symlink",
                        f"unsafe path {entry!r}; entries must be relative paths without '..'",
                    )
        if "setup" in worktree and not _string(source, "worktree.setup", worktree["setup"]).strip():
            _raise(source, "worktree.setup", "must be a non-empty string")
    budgets = _table(source, values, "budgets")
    if budgets is not None:
        _unknown_keys(source, "budgets", budgets, {"leader", "subagent", "price_table"})
        if "price_table" in budgets:
            value = _string(source, "budgets.price_table", budgets["price_table"])
            if not value.strip():
                _raise(source, "budgets.price_table", "must be a non-empty path")
        for role in ("leader", "subagent"):
            limits = _table(source, budgets, role)
            if limits is None:
                continue
            prefix = f"budgets.{role}"
            _unknown_keys(source, prefix, limits, BUDGET_KEYS)
            SettingsParser(lambda key, detail: _raise(source, f"{prefix}.{key}", detail)).budget(limits, config=True)
    sessions = _table(source, values, "sessions")
    if sessions is not None:
        _unknown_keys(source, "sessions", sessions, {"cleanup_period_days"})
        if "cleanup_period_days" in sessions:
            period = sessions["cleanup_period_days"]
            if type(period) is not int or period < 0:
                _raise(source, "sessions.cleanup_period_days", "must be a non-negative integer")
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
    parser = SettingsParser(lambda key, detail: _raise(source, f"agents.ceiling.{key}", detail))
    for key, parse in (("allowed_write_scope", parser.scopes),
                       ("fetch_allowlist", parser.hosts), ("modes", parser.modes),
                       ("shell_enabled", parser.boolean), ("fetch_enabled", parser.boolean),
                       ("shell_allowlist", parser.commands)):
        if key in ceiling:
            parse(key, ceiling[key])


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
    root = Path(repo_root)
    sources: tuple[tuple[Scope, Path | None, Mapping[str, object] | None], ...] = (
        (
            Scope.USER,
            symphonai_home(home) / "config.toml",
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
    warnings: list[str] = []
    for scope, source, supplied in sources:
        if scope is Scope.SESSION:
            values = supplied
        else:
            if source is None:
                raise AssertionError("file-backed config scope has no source")
            try:
                values = _read_toml(source)
            except ConfigError as exc:
                if scope not in {Scope.PROJECT, Scope.PRIVATE}:
                    raise
                warnings.append(str(exc))
                continue
        if values is None:
            continue
        if not isinstance(values, Mapping):
            if scope in {Scope.PROJECT, Scope.PRIVATE}:
                warnings.append(f"{source}: config: must be a table; file skipped")
                continue
            _raise(source, "config", "must be a table")
        if scope in {Scope.PROJECT, Scope.PRIVATE}:
            safe_values: dict[str, object] = {}
            for section, value in values.items():
                if section == "permissions":
                    warnings.append(
                        f"{source}: permissions: grants are refused from project and private config scopes; section skipped"
                    )
                    continue
                if section == "roles":
                    warnings.append(
                        f"{source}: roles: role models are accepted only from user config; section skipped"
                    )
                    continue
                if section == "always_deny":
                    warnings.append(
                        f"{source}: always_deny: settings are accepted only from user config; section skipped"
                    )
                    continue
                if section == "sandbox" and isinstance(value, Mapping) and "readable" in value:
                    warnings.append(
                        f"{source}: sandbox.readable: accepted only from user config; key skipped"
                    )
                    value = {key: item for key, item in value.items() if key != "readable"}
                try:
                    _validate(source, {section: value})
                except ConfigError as exc:
                    warnings.append(f"{exc}; section skipped")
                    continue
                safe_values[section] = value
            values = safe_values
        if "permissions" in values and scope is Scope.SESSION:
            _raise(source, "permissions", "permission grants are supported only in user config")
        if "roles" in values and scope is Scope.SESSION:
            _raise(source, "roles", "role models are supported only in user config")
        if "always_deny" in values and scope is Scope.SESSION:
            _raise(source, "always_deny", "rule lifts are supported only in user config")
        if scope is Scope.SESSION and isinstance(values.get("sandbox"), Mapping) and "readable" in values["sandbox"]:
            _raise(source, "sandbox.readable", "supported only in user config")
        if scope not in {Scope.PROJECT, Scope.PRIVATE}:
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
    config = ResolvedConfig(
        values=MappingProxyType(resolved),
        provenance=MappingProxyType(provenance),
        layers=tuple(layers),
        warnings=tuple(warnings),
    )
    setup_origin = config.provenance.get("worktree.setup")
    if setup_origin is not None and setup_origin.scope in {Scope.PROJECT, Scope.PRIVATE}:
        from symphonai_api.trust import trust_from_config

        try:
            trusted = trust_from_config(config).allows(root, "hooks")
        except ConfigError:
            trusted = False
        if not trusted:
            values = dict(config.values)
            values.pop("worktree.setup", None)
            provenance = dict(config.provenance)
            provenance.pop("worktree.setup", None)
            layers = tuple(
                ConfigLayer(
                    scope=layer.scope,
                    source=layer.source,
                    values=MappingProxyType({
                        key: value for key, value in layer.values.items()
                        if key != "worktree.setup"
                    }),
                )
                for layer in config.layers
            )
            warnings.append(
                f"{setup_origin.source}: worktree.setup: ignored unless the repository "
                "is granted hooks under [[trust.repositories]]"
            )
            config = ResolvedConfig(
                values=MappingProxyType(values),
                provenance=MappingProxyType(provenance),
                layers=layers,
                warnings=tuple(warnings),
            )
    return config


def resolve_run_budgets(
    config: ResolvedConfig,
    *,
    repo_root: Path,
    leader_max_turns: int,
    subagent_max_turns: int,
    price_table: PriceTable | None,
) -> tuple[RunBudget | None, RunBudget | None, PriceTable | None]:
    """Resolve configured limits before a conversation can start."""
    table_key = "budgets.price_table"
    if table_key in config.values:
        origin = config.provenance[table_key].source
        path = Path(config.values[table_key]).expanduser()
        if not path.is_absolute():
            path = (origin.parent if origin is not None else Path(repo_root)) / path
        try:
            price_table = load_price_table(path)
        except ValueError as exc:
            _raise(origin, table_key, str(exc))

    def budget_for(role: str, default_turns: int) -> RunBudget | None:
        prefix = f"budgets.{role}."
        keys = {key.removeprefix(prefix) for key in config.values if key.startswith(prefix)}
        if not keys:
            return None
        cost_key = prefix + "max_cost"
        if cost_key in config.values and price_table is None:
            _raise(config.provenance[cost_key].source, cost_key, "requires a price table")
        return RunBudget(
            max_turns=config.get(prefix + "max_turns", default_turns),
            wall_seconds=config.get(prefix + "wall_seconds"),
            max_total_tokens=config.get(prefix + "max_total_tokens"),
            max_cost=Decimal(config.values[cost_key]) if cost_key in config.values else None,
            price_table=price_table,
        )

    return (
        budget_for("leader", leader_max_turns),
        budget_for("subagent", subagent_max_turns),
        price_table,
    )


@dataclass(frozen=True)
class CapabilityCeiling:
    allowed_write_scope: tuple[Path, ...] | None = None
    shell_enabled: bool | None = None
    shell_allowlist: tuple[tuple[str, ...], ...] | None = None
    fetch_enabled: bool | None = None
    fetch_allowlist: tuple[str, ...] | None = None
    modes: tuple[str, ...] | None = None

    def bound(self, policy: PermissionPolicy) -> PermissionPolicy:
        """Return a copy of a conversation policy limited by this ceiling."""
        bounded = replace(policy)
        if self.allowed_write_scope is not None:
            bounded.allowed_write_scope = _intersect_write_scopes(
                policy.allowed_write_scope,
                [Path(path).resolve() for path in self.allowed_write_scope],
            )
        if self.shell_enabled is False:
            bounded.shell_enabled = False
        if self.shell_allowlist is not None:
            bounded.shell_allowlist = _intersect_shell_allowlists(
                policy.shell_allowlist,
                list(self.shell_allowlist),
            )
        if self.fetch_enabled is False:
            bounded.fetch_enabled = False
            bounded.fetch_allowlist = []
        elif self.fetch_allowlist is not None:
            reachable = (
                set(self.fetch_allowlist)
                if policy.fetch_enabled
                else set(policy.fetch_allowlist).intersection(self.fetch_allowlist)
            )
            bounded.fetch_enabled = False
            bounded.fetch_allowlist = [
                host for host in self.fetch_allowlist if host in reachable
            ]
        return bounded

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

        write_scope = value("allowed_write_scope")
        shell_allowlist = value("shell_allowlist")
        fetch_allowlist = value("fetch_allowlist")
        modes = value("modes")
        return cls(
            allowed_write_scope=(None if write_scope is None else tuple(
                (Path(repo_root) / entry).resolve() for entry in write_scope
            )),
            shell_enabled=value("shell_enabled"),
            shell_allowlist=(None if shell_allowlist is None else tuple(map(tuple, shell_allowlist))),
            fetch_enabled=value("fetch_enabled"),
            fetch_allowlist=(None if fetch_allowlist is None else tuple(
                host.casefold().rstrip(".") for host in fetch_allowlist
            )),
            modes=None if modes is None else tuple(modes),
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


def permission_grants(
    config: ResolvedConfig,
    ceiling: CapabilityCeiling | None = None,
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Return user-config grants narrowed to the configured capability ceiling."""
    raw_shell = config.get("permissions.shell", [])
    raw_fetch = config.get("permissions.fetch", [])
    return restrict_permission_grants(
        tuple(tuple(prefix) for prefix in raw_shell),
        tuple(host.casefold().rstrip(".") for host in raw_fetch),
        ceiling,
    )


def restrict_permission_grants(
    shell: tuple[tuple[str, ...], ...],
    fetch: tuple[str, ...],
    ceiling: CapabilityCeiling | None = None,
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Narrow remembered grants to the configured capability ceiling."""
    if ceiling is not None:
        if ceiling.shell_enabled is False:
            shell = ()
        elif ceiling.shell_allowlist is not None:
            narrowed = []
            for grant in shell:
                for allowed in ceiling.shell_allowlist:
                    if _is_prefix(allowed, grant):
                        narrowed.append(grant)
                    elif _is_prefix(grant, allowed):
                        narrowed.append(allowed)
            shell = tuple(dict.fromkeys(narrowed))

    fetch = tuple(dict.fromkeys(host.casefold().rstrip(".") for host in fetch))
    if ceiling is not None:
        if ceiling.fetch_enabled is False:
            fetch = ()
        elif ceiling.fetch_allowlist is not None:
            allowed_hosts = {host.casefold().rstrip(".") for host in ceiling.fetch_allowlist}
            fetch = tuple(host for host in fetch if host in allowed_hosts)
    return shell, fetch
