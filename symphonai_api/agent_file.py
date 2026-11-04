"""Load declarative agent roles from TOML files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import tomllib

from symphonai_api.tools.metadata import FILE_EDIT_TOOLS, edit_tool_names
from symphonai_api.agent_spec import (
    AgentSpec,
    ContextInheritance,
    IOContract,
    Isolation,
    ModelSelector,
)
from symphonai_api.agent_memory import MAX_ENTRIES, MemorySettings
from symphonai_api.budgets import PriceTable, RunBudget
from symphonai_api.call_class import CallClass
from symphonai_api.config import CapabilityCeiling, ConfigError
from symphonai_api.permissions import BUDGET_KEYS, PermissionPolicy, SettingsParser


class AgentFileError(ValueError):
    """A malformed or invalid agent file, naming the file and the key."""


_TOP_LEVEL_KEYS = {
    "description",
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
}
_TABLE_KEYS = {
    "memory": {"enabled", "max_entries"},
    "model": {"provider", "model", "effort"},
    "isolation": {"inherit", "inherit_tail", "workspace_prefix"},
    "budget": BUDGET_KEYS,
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
    *sorted(FILE_EDIT_TOOLS),
    "list_files",
    "glob",
    "grep",
    "run_shell",
    "web_fetch",
)
_AGENT_TOOL_NAMES = (*_STANDARD_TOOL_NAMES, "web_search", "use_skill", "lsp",
                     "read_tool_result", "job_output", "job_stop")

def _is_mcp_tool_name(name: str) -> bool:
    if not name.startswith("mcp__"):
        return False
    server, separator, tool = name[5:].partition("__")
    return bool(separator and server and tool)


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
    return SettingsParser(lambda field, detail: _raise(path, field, detail)).string(key, value)

def _integer(path: Path, key: str, value: object) -> int:
    return SettingsParser(lambda field, detail: _raise(path, field, detail)).integer(key, value)

def _number(path: Path, key: str, value: object) -> float:
    return SettingsParser(lambda field, detail: _raise(path, field, detail)).number(key, value)

def _boolean(path: Path, key: str, value: object) -> bool:
    return SettingsParser(lambda field, detail: _raise(path, field, detail)).boolean(key, value)

def _strings(path: Path, key: str, value: object) -> list[str]:
    return SettingsParser(lambda field, detail: _raise(path, field, detail)).strings(key, value)

def _tool_names(path: Path, values: Mapping[str, object]) -> tuple[str, ...] | None:
    tools_present = "tools" in values
    deny_present = "deny_tools" in values
    if not tools_present and not deny_present:
        return None
    tools = (
        _strings(path, "tools", values["tools"])
        if tools_present
        else list(_AGENT_TOOL_NAMES)
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
            if name not in _AGENT_TOOL_NAMES and not _is_mcp_tool_name(name):
                _raise(path, key, f"unknown tool name: {name!r}")
    denied_names = set(denied)
    base = edit_tool_names(set(tools) - denied_names)
    result = tuple(
        name
        for name in _AGENT_TOOL_NAMES
        if name in base and name not in denied_names
    )
    has_mcp_tool = any(
        _is_mcp_tool_name(name) and name not in denied_names
        for name in tools
    )
    if not result and not has_mcp_tool:
        key = "tools and deny_tools" if deny_present else "tools"
        _raise(path, key, "resolved tool set was empty")
    return result


def _mcp_tool_selection(path: Path, values: Mapping[str, object]) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    tools_present = "tools" in values
    tools = _strings(path, "tools", values["tools"]) if tools_present else []
    denied = _strings(path, "deny_tools", values["deny_tools"]) if "deny_tools" in values else []
    for key, names in (("tools", tools), ("deny_tools", denied)):
        for name in names:
            if name.startswith("mcp__") and not _is_mcp_tool_name(name):
                _raise(path, key, f"unknown tool name: {name!r}")
    selected = tuple(name for name in tools if _is_mcp_tool_name(name)) if tools_present else None
    denied_selected = tuple(name for name in denied if _is_mcp_tool_name(name))
    return selected, denied_selected


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
    effort = default.effort if default is not None else None
    if "effort" in table:
        effort = _string(path, "effort", table["effort"])
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
        isolation = Isolation(inherit=inherit, inherit_tail=inherit_tail)
    except ValueError as exc:
        _raise(path, "inherit_tail", str(exc))
    try:
        return replace(isolation, workspace_prefix=workspace_prefix)
    except ValueError as exc:
        _raise(path, "workspace_prefix", str(exc))


def _budget(
    path: Path,
    table: Mapping[str, object] | None,
    price_table: PriceTable | None,
) -> RunBudget | None:
    if table is None:
        return None
    values = SettingsParser(lambda key, detail: _raise(path, key, detail)).budget(
        table, priced=price_table is not None,
    )
    if price_table is not None:
        values["price_table"] = price_table
    return RunBudget(**values)


def _policy(
    path: Path,
    table: Mapping[str, object] | None,
    repo_root: Path,
) -> PermissionPolicy:
    if table is None:
        return PermissionPolicy(
            repo_root=repo_root,
            allowed_write_scope=[repo_root],
            shell_enabled=True,
            shell_allowlist=[()],
            sandbox_network=True,
            fetch_enabled=True,
        )
    parser = SettingsParser(lambda key, detail: _raise(path, key, detail))
    values: dict[str, object] = {"repo_root": repo_root}
    if "allowed_write_scope" in table:
        values["allowed_write_scope"] = parser.scopes("allowed_write_scope", table["allowed_write_scope"], repo_root)
    if "forbidden_patterns" in table:
        values["forbidden_patterns"] = tuple(parser.strings("forbidden_patterns", table["forbidden_patterns"]))
    for key in ("shell_enabled", "fetch_enabled"):
        if key in table:
            values[key] = parser.boolean(key, table[key])
            values[key.removesuffix("_enabled") + "_explicitly_disabled"] = values[key] is False
    if "shell_allowlist" in table:
        values["shell_allowlist"] = parser.commands("shell_allowlist", table["shell_allowlist"])
    if "fetch_allowlist" in table:
        values["fetch_allowlist"] = parser.hosts("fetch_allowlist", table["fetch_allowlist"])
    if "shell_timeout_seconds" in table:
        values["shell_timeout_seconds"] = parser.number("shell_timeout_seconds", table["shell_timeout_seconds"])
    if "shell_output_limit_chars" in table:
        values["shell_output_limit_chars"] = parser.integer("shell_output_limit_chars", table["shell_output_limit_chars"])
    if "mode" in table:
        values["mode"] = parser.mode("mode", table["mode"])
    return PermissionPolicy(**values)


def _io(path: Path, table: Mapping[str, object] | None) -> IOContract:
    if table is None:
        return IOContract()
    values = {
        key: table[key]
        for key in ("input_schema", "output_schema")
        if key in table
    }
    for key, value in values.items():
        try:
            IOContract(**{key: value})
        except ValueError as exc:
            _raise(path, key, str(exc))
    return IOContract(**values)


def load_agent_file(
    path: Path,
    *,
    repo_root: Path,
    price_table: PriceTable | None = None,
    default_model: ModelSelector | None = None,
    ceiling: CapabilityCeiling | None = None,
) -> AgentSpec:
    """Parse one TOML agent file into an AgentSpec."""
    source = Path(path)
    data = _read_toml(source)
    _unknown_keys(source, "file", data)
    if "prompt" not in data:
        _raise(source, "prompt", "is required")
    prompt = _string(source, "prompt", data["prompt"])
    description = None
    if "description" in data:
        description = _string(source, "description", data["description"])
        if not description.strip():
            _raise(source, "description", "must be a non-empty string")
    memory_table = _table(source, data, "memory")
    model_table = _table(source, data, "model")
    isolation_table = _table(source, data, "isolation")
    budget_table = _table(source, data, "budget")
    policy_table = _table(source, data, "policy")
    io_table = _table(source, data, "io")
    mcp_tools, denied_mcp_tools = _mcp_tool_selection(source, data)
    values: dict[str, object] = {
        "name": source.stem,
        "prompt": prompt,
        "description": description,
        "model": _model(source, model_table, default_model),
        "policy_ceiling": _policy(source, policy_table, Path(repo_root)),
        "tool_names": _tool_names(source, data),
        "mcp_tool_names": mcp_tools,
        "denied_mcp_tool_names": denied_mcp_tools,
        "budget": _budget(source, budget_table, price_table),
        "isolation": _isolation(source, isolation_table),
        "io": _io(source, io_table),
        "memory": _memory_settings(source, memory_table),
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
    try:
        spec = AgentSpec(**values)
    except ValueError as exc:
        if not values["name"] or values["name"] != values["name"].strip():
            _raise(source, "name", str(exc))
        if values.get("deadline_seconds") is not None and not values["deadline_seconds"] > 0:
            _raise(source, "deadline_seconds", str(exc))
        _raise(source, "agent", str(exc))
    if ceiling is not None and policy_table is not None:
        try:
            ceiling.refuse(spec.policy_ceiling, source=source)
        except ConfigError as exc:
            raise AgentFileError(str(exc)) from None
    return spec


def load_agent_directory(
    path: Path,
    *,
    repo_root: Path,
    price_table: PriceTable | None = None,
    default_model: ModelSelector | None = None,
    ceiling: CapabilityCeiling | None = None,
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
            ceiling=ceiling,
        )
        for file in sorted(directory.glob("*.toml"))
        if file.is_file()
    }
