"""Load declarative agent roles from TOML files."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
import tomllib

from symphonai_api.agent_spec import (
    AgentSpec,
    ContextInheritance,
    Effort,
    IOContract,
    Isolation,
    ModelSelector,
)
from symphonai_api.agent_memory import MAX_ENTRIES, MemorySettings
from symphonai_api.budgets import PriceTable, RunBudget
from symphonai_api.call_class import CallClass
from symphonai_api.permissions import PermissionPolicy


class AgentFileError(ValueError):
    """A malformed or invalid agent file, naming the file and the key."""


_TOP_LEVEL_KEYS = {
    "prompt",
    "tools",
    "deny_tools",
    "memory",
    "model",
    "isolation",
    "budget",
    "policy",
    "io",
    "deadline_seconds",
    "call_class",
    "max_depth",
}
_TABLE_KEYS = {
    "memory": {"enabled", "max_entries"},
    "model": {"provider", "model", "effort"},
    "isolation": {"inherit", "inherit_tail", "workspace_prefix"},
    "budget": {"max_turns", "wall_seconds", "max_total_tokens", "max_cost"},
    "policy": {
        "allowed_write_scope",
        "forbidden_patterns",
        "shell_enabled",
        "shell_allowlist",
        "fetch_enabled",
        "fetch_allowlist",
        "shell_timeout_seconds",
        "shell_output_limit_chars",
        "mode",
    },
    "io": {"input_schema", "output_schema"},
}
_STANDARD_TOOL_NAMES = (
    "read_file",
    "write_file",
    "edit_file",
    "multi_edit_file",
    "list_files",
    "glob",
    "grep",
    "run_shell",
    "web_fetch",
)


def _raise(path: Path, key: str, detail: str) -> None:
    raise AgentFileError(f"{path}: {key}: {detail}")


def _unknown_keys(path: Path, key: str, values: Mapping[str, object]) -> None:
    allowed = _TOP_LEVEL_KEYS if key == "file" else _TABLE_KEYS[key]
    unknown = values.keys() - allowed
    if unknown:
        unknown_key = sorted(unknown)[0]
        detail = (
            "cannot come from an agent file"
            if unknown_key in {"approval_callback", "price_table"}
            else "unknown key"
        )
        _raise(path, unknown_key, detail)


def _table(
    path: Path,
    values: Mapping[str, object],
    key: str,
) -> Mapping[str, object] | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        _raise(path, key, "must be a table")
    _unknown_keys(path, key, value)
    return value


def _read_toml(path: Path) -> dict[str, object]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        _raise(path, "file", f"could not be read: {exc}")
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        _raise(path, "toml", f"could not be parsed: {exc}")


def _string(path: Path, key: str, value: object) -> str:
    if not isinstance(value, str):
        _raise(path, key, "must be a string")
    return value


def _integer(path: Path, key: str, value: object) -> int:
    if type(value) is not int:
        _raise(path, key, "must be an integer")
    return value


def _number(path: Path, key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _raise(path, key, "must be a number")
    return value


def _boolean(path: Path, key: str, value: object) -> bool:
    if type(value) is not bool:
        _raise(path, key, "must be a boolean")
    return value


def _strings(path: Path, key: str, value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _raise(path, key, "must be an array of strings")
    return value


def _tool_names(path: Path, values: Mapping[str, object]) -> tuple[str, ...] | None:
    tools_present = "tools" in values
    deny_present = "deny_tools" in values
    if not tools_present and not deny_present:
        return None
    tools = (
        _strings(path, "tools", values["tools"])
        if tools_present
        else list(_STANDARD_TOOL_NAMES)
    )
    denied = (
        _strings(path, "deny_tools", values["deny_tools"])
        if deny_present
        else []
    )
    if tools_present and not tools:
        _raise(path, "tools", "must not be empty")
    for key, names in (("tools", tools), ("deny_tools", denied)):
        for name in names:
            if name not in _STANDARD_TOOL_NAMES:
                _raise(path, key, f"unknown tool name: {name!r}")
    base = set(tools)
    denied_names = set(denied)
    result = tuple(
        name
        for name in _STANDARD_TOOL_NAMES
        if name in base and name not in denied_names
    )
    if not result:
        key = "tools and deny_tools" if deny_present else "tools"
        _raise(path, key, "resolved tool set was empty")
    return result


def _memory_settings(
    path: Path,
    table: Mapping[str, object] | None,
) -> MemorySettings:
    if table is None:
        return MemorySettings()
    enabled = False
    if "enabled" in table:
        enabled = _boolean(path, "enabled", table["enabled"])
    max_entries = MAX_ENTRIES
    if "max_entries" in table:
        max_entries = _integer(path, "max_entries", table["max_entries"])
    if not 1 <= max_entries <= MAX_ENTRIES:
        _raise(
            path,
            "max_entries",
            f"must be between 1 and the ceiling {MAX_ENTRIES}",
        )
    return MemorySettings(enabled=enabled, max_entries=max_entries)


def _model(
    path: Path,
    table: Mapping[str, object] | None,
    default: ModelSelector | None,
) -> ModelSelector:
    if table is None:
        if default is None:
            _raise(path, "model", "requires a default_model")
        return default
    if "provider" in table:
        provider = _string(path, "provider", table["provider"])
    elif default is not None:
        provider = default.provider
    else:
        _raise(path, "provider", "is required without a default_model")
    model = default.model if default is not None else None
    if "model" in table:
        model = _string(path, "model", table["model"])
    effort = default.effort if default is not None else Effort.DEFAULT
    if "effort" in table:
        raw_effort = _string(path, "effort", table["effort"])
        try:
            effort = Effort(raw_effort)
        except ValueError:
            valid = ", ".join(value.value for value in Effort)
            _raise(path, "effort", f"must be one of: {valid}")
    try:
        return ModelSelector(provider=provider, model=model, effort=effort)
    except ValueError as exc:
        _raise(path, "provider", str(exc))


def _isolation(path: Path, table: Mapping[str, object] | None) -> Isolation:
    if table is None:
        return Isolation()
    inherit = ContextInheritance.FRESH
    if "inherit" in table:
        raw_inherit = _string(path, "inherit", table["inherit"])
        try:
            inherit = ContextInheritance(raw_inherit)
        except ValueError:
            valid = ", ".join(value.value for value in ContextInheritance)
            _raise(path, "inherit", f"must be one of: {valid}")
    inherit_tail = 0
    if "inherit_tail" in table:
        inherit_tail = _integer(path, "inherit_tail", table["inherit_tail"])
    workspace_prefix = None
    if "workspace_prefix" in table:
        workspace_prefix = _string(path, "workspace_prefix", table["workspace_prefix"])
    try:
        return Isolation(
            inherit=inherit,
            inherit_tail=inherit_tail,
            workspace_prefix=workspace_prefix,
        )
    except ValueError as exc:
        key = "workspace_prefix" if "workspace_prefix" in str(exc) else "inherit_tail"
        _raise(path, key, str(exc))


def _budget(
    path: Path,
    table: Mapping[str, object] | None,
    price_table: PriceTable | None,
) -> RunBudget | None:
    if table is None:
        return None
    values: dict[str, object] = {}
    for key in ("max_turns", "max_total_tokens"):
        if key in table:
            values[key] = _integer(path, key, table[key])
    if "wall_seconds" in table:
        values["wall_seconds"] = _number(path, "wall_seconds", table["wall_seconds"])
    if "max_cost" in table:
        raw_cost = table["max_cost"]
        if not isinstance(raw_cost, str):
            _raise(path, "max_cost", "must be quoted as a string")
        if price_table is None:
            _raise(path, "max_cost", "requires a price table supplied to the loader")
        try:
            cost = Decimal(raw_cost)
        except InvalidOperation:
            _raise(path, "max_cost", "must be a decimal string")
        if not cost.is_finite():
            _raise(path, "max_cost", "must be a finite decimal string")
        values["max_cost"] = cost
    if price_table is not None:
        values["price_table"] = price_table
    try:
        return RunBudget(**values)
    except ValueError as exc:
        for key in ("max_turns", "wall_seconds", "max_total_tokens", "max_cost"):
            if key in str(exc):
                _raise(path, key, str(exc))
        _raise(path, "budget", str(exc))


def _policy(
    path: Path,
    table: Mapping[str, object] | None,
    repo_root: Path,
) -> PermissionPolicy:
    if table is None:
        return PermissionPolicy(repo_root=repo_root)
    values: dict[str, object] = {"repo_root": repo_root}
    if "allowed_write_scope" in table:
        scopes = _strings(path, "allowed_write_scope", table["allowed_write_scope"])
        values["allowed_write_scope"] = [
            (repo_root / scope).resolve() for scope in scopes
        ]
    if "forbidden_patterns" in table:
        values["forbidden_patterns"] = tuple(
            _strings(path, "forbidden_patterns", table["forbidden_patterns"])
        )
    for key in ("shell_enabled", "fetch_enabled"):
        if key in table:
            values[key] = _boolean(path, key, table[key])
    if "shell_allowlist" in table:
        allowlist = table["shell_allowlist"]
        if not isinstance(allowlist, list):
            _raise(path, "shell_allowlist", "must be an array of command arrays")
        commands: list[tuple[str, ...]] = []
        for command in allowlist:
            commands.append(tuple(_strings(path, "shell_allowlist", command)))
        values["shell_allowlist"] = commands
    if "fetch_allowlist" in table:
        values["fetch_allowlist"] = _strings(
            path,
            "fetch_allowlist",
            table["fetch_allowlist"],
        )
    if "shell_timeout_seconds" in table:
        values["shell_timeout_seconds"] = _number(
            path,
            "shell_timeout_seconds",
            table["shell_timeout_seconds"],
        )
    if "shell_output_limit_chars" in table:
        values["shell_output_limit_chars"] = _integer(
            path,
            "shell_output_limit_chars",
            table["shell_output_limit_chars"],
        )
    if "mode" in table:
        values["mode"] = _string(path, "mode", table["mode"])
    try:
        return PermissionPolicy(**values)
    except ValueError as exc:
        key = "mode" if "mode" in str(exc) else "policy"
        _raise(path, key, str(exc))


def _io(path: Path, table: Mapping[str, object] | None) -> IOContract:
    if table is None:
        return IOContract()
    values = {
        key: table[key]
        for key in ("input_schema", "output_schema")
        if key in table
    }
    try:
        return IOContract(**values)
    except ValueError as exc:
        key = "input_schema" if "input_schema" in str(exc) else "output_schema"
        _raise(path, key, str(exc))


def load_agent_file(
    path: Path,
    *,
    repo_root: Path,
    price_table: PriceTable | None = None,
    default_model: ModelSelector | None = None,
) -> AgentSpec:
    """Parse one TOML agent file into an AgentSpec."""
    source = Path(path)
    data = _read_toml(source)
    _unknown_keys(source, "file", data)
    if "prompt" not in data:
        _raise(source, "prompt", "is required")
    prompt = _string(source, "prompt", data["prompt"])
    memory_table = _table(source, data, "memory")
    model_table = _table(source, data, "model")
    isolation_table = _table(source, data, "isolation")
    budget_table = _table(source, data, "budget")
    policy_table = _table(source, data, "policy")
    io_table = _table(source, data, "io")
    _memory_settings(source, memory_table)
    values: dict[str, object] = {
        "name": source.stem,
        "prompt": prompt,
        "model": _model(source, model_table, default_model),
        "policy_ceiling": _policy(source, policy_table, Path(repo_root)),
        "tool_names": _tool_names(source, data),
        "budget": _budget(source, budget_table, price_table),
        "isolation": _isolation(source, isolation_table),
        "io": _io(source, io_table),
    }
    if "deadline_seconds" in data:
        values["deadline_seconds"] = _number(
            source,
            "deadline_seconds",
            data["deadline_seconds"],
        )
    if "call_class" in data:
        raw_call_class = _string(source, "call_class", data["call_class"])
        try:
            values["call_class"] = CallClass(raw_call_class)
        except ValueError:
            valid = ", ".join(value.value for value in CallClass)
            _raise(source, "call_class", f"must be one of: {valid}")
    if "max_depth" in data:
        values["max_depth"] = _integer(source, "max_depth", data["max_depth"])
    try:
        return AgentSpec(**values)
    except ValueError as exc:
        for key in ("name", "deadline_seconds", "max_depth"):
            if key in str(exc):
                _raise(source, key, str(exc))
        _raise(source, "agent", str(exc))


def memory_settings(path: Path) -> MemorySettings:
    """Read the opt-in memory settings from one agent TOML file."""
    source = Path(path)
    data = _read_toml(source)
    _unknown_keys(source, "file", data)
    return _memory_settings(source, _table(source, data, "memory"))


def load_agent_directory(
    path: Path,
    *,
    repo_root: Path,
    price_table: PriceTable | None = None,
    default_model: ModelSelector | None = None,
) -> dict[str, AgentSpec]:
    """Load every direct ``*.toml`` child, keyed by filename stem."""
    directory = Path(path)
    if not directory.exists():
        return {}
    return {
        file.stem: load_agent_file(
            file,
            repo_root=repo_root,
            price_table=price_table,
            default_model=default_model,
        )
        for file in sorted(directory.glob("*.toml"))
        if file.is_file()
    }
