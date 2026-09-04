"""Immutable agent-role values and a small, non-comprehensive JSON Schema subset.

The policy ceiling remains mutable by design; executions must narrow from it
rather than mutating it. This module deliberately contains no runtime wiring.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType

from symphonai_api.budgets import RunBudget
from symphonai_api.call_class import CallClass
from symphonai_api.identity import SCHEMA_VERSION
from symphonai_api.permissions import PermissionPolicy


def _frozen(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_frozen(item) for item in value)
    return value


@dataclass(frozen=True)
class ModelSelector:
    provider: str
    model: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")


class ContextInheritance(str, Enum):
    FRESH = "fresh"
    ALL = "all"
    TAIL = "tail"


@dataclass(frozen=True)
class Isolation:
    inherit: ContextInheritance = ContextInheritance.FRESH
    inherit_tail: int = 0
    workspace_prefix: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.inherit is ContextInheritance.TAIL and self.inherit_tail < 1:
            raise ValueError(
                f"inherit {self.inherit.value!r} is incompatible with inherit_tail {self.inherit_tail}"
            )
        if self.inherit is not ContextInheritance.TAIL and self.inherit_tail != 0:
            raise ValueError(
                f"inherit {self.inherit.value!r} is incompatible with inherit_tail {self.inherit_tail}"
            )
        if self.workspace_prefix is not None:
            prefix = PurePosixPath(self.workspace_prefix)
            if not self.workspace_prefix or prefix.is_absolute() or ".." in prefix.parts:
                raise ValueError("workspace_prefix must be a non-empty relative path without '..'")


@dataclass(frozen=True)
class IOContract:
    input_schema: Mapping[str, object] | None = None
    output_schema: Mapping[str, object] | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("input_schema", "output_schema"):
            schema = getattr(self, field_name)
            if schema is None:
                continue
            if not isinstance(schema, Mapping) or schema.get("type") != "object":
                raise ValueError(f"{field_name} must be an object schema mapping")
            object.__setattr__(self, field_name, _frozen(schema))


def validate_output(contract: IOContract, text: str) -> tuple[object | None, str | None]:
    """Parse and check output, returning an actionable error instead of raising."""
    if contract.output_schema is None:
        return text, None
    try:
        source = text.strip()
        if source.startswith("```json") and source.endswith("```"):
            source = source[7:-3].strip()
        value = json.loads(source)
        error = _validate(value, contract.output_schema, "output")
        return (value, None) if error is None else (None, error)
    except Exception as exc:
        return None, f"output must be valid JSON matching the contract: {exc}"


def _validate(value: object, schema: Mapping[str, object], path: str) -> str | None:
    expected = schema.get("type")
    valid = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool), "null": value is None,
    }
    if expected in valid and not valid[expected]:
        return f"{path} must be {expected}"
    enum = schema.get("enum")
    if isinstance(enum, (list, tuple)) and value not in enum:
        return f"{path} must be one of the allowed enum values"
    if expected == "object" and isinstance(value, dict):
        required = schema.get("required", ())
        if isinstance(required, (list, tuple)):
            for key in required:
                if isinstance(key, str) and key not in value:
                    return f"{path} is missing required property {key!r}"
        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            for key, child in properties.items():
                if key in value and isinstance(child, Mapping):
                    error = _validate(value[key], child, f"{path}.{key}")
                    if error:
                        return error
    if expected == "array" and isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            error = _validate(item, schema["items"], f"{path}[{index}]")
            if error:
                return error
    return None


@dataclass(frozen=True)
class AgentSpec:
    name: str
    prompt: str
    model: ModelSelector
    policy_ceiling: PermissionPolicy
    tool_names: tuple[str, ...] | None = None
    budget: RunBudget | None = None
    deadline_seconds: float | None = None
    isolation: Isolation = Isolation()
    io: IOContract = IOContract()
    call_class: CallClass = CallClass.BACKGROUND
    max_depth: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("name must be non-empty without surrounding whitespace")
        if self.tool_names is not None:
            if not self.tool_names:
                raise ValueError("tool_names must not be empty")
            if len(set(self.tool_names)) != len(self.tool_names):
                raise ValueError("tool_names must not contain duplicates")
        if self.deadline_seconds is not None and not self.deadline_seconds > 0:
            raise ValueError("deadline_seconds must be positive")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")

    def with_overrides(self, **changes: object) -> "AgentSpec":
        return replace(self, **changes)
