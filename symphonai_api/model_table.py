"""Fail-closed model effort capabilities shipped with SymphonAI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources


@dataclass(frozen=True)
class ModelEffort:
    id: str
    value: str | int


@dataclass(frozen=True)
class ModelCapability:
    provider: str
    wire_format: int
    model: str
    efforts: tuple[ModelEffort, ...]


def _models_from_json(payload: object) -> tuple[ModelCapability, ...]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return ()
    if not isinstance(payload.get("notes"), str) or not payload["notes"]:
        return ()
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return ()
    models: list[ModelCapability] = []
    seen_models: set[tuple[int, str]] = set()
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            return ()
        provider = raw_model.get("provider")
        wire_format = raw_model.get("wire_format")
        model = raw_model.get("id")
        raw_efforts = raw_model.get("efforts")
        if (
            not isinstance(provider, str)
            or not provider
            or not isinstance(wire_format, int)
            or isinstance(wire_format, bool)
            or wire_format not in {1, 2, 3}
            or not isinstance(model, str)
            or not model
            or not isinstance(raw_efforts, list)
            or (wire_format, model) in seen_models
        ):
            return ()
        efforts: list[ModelEffort] = []
        seen_efforts: set[str] = set()
        for raw_effort in raw_efforts:
            if not isinstance(raw_effort, dict):
                return ()
            identifier = raw_effort.get("id")
            value = raw_effort.get("value")
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in seen_efforts
                or not (
                    (isinstance(value, str) and bool(value))
                    or (isinstance(value, int) and not isinstance(value, bool))
                )
            ):
                return ()
            seen_efforts.add(identifier)
            efforts.append(ModelEffort(identifier, value))
        seen_models.add((wire_format, model))
        models.append(ModelCapability(provider, wire_format, model, tuple(efforts)))
    return tuple(models)


@lru_cache(maxsize=1)
def model_capabilities() -> tuple[ModelCapability, ...]:
    """Load the shipped table once, returning no entries on any mismatch."""

    try:
        payload = json.loads(
            resources.files("symphonai_api.data")
            .joinpath("models.json")
            .read_text(encoding="utf-8")
        )
        return _models_from_json(payload)
    except (OSError, TypeError, ValueError, UnicodeError):
        return ()


def _capability_for_model(
    wire_format: int,
    model: str,
    capabilities: tuple[ModelCapability, ...],
) -> ModelCapability | None:
    matching_format = [
        item for item in capabilities if item.wire_format == wire_format
    ]
    exact = next((item for item in matching_format if item.model == model), None)
    if exact is not None:
        return exact

    for item in matching_format:
        prefix = f"{item.model}-"
        if not model.startswith(prefix):
            continue
        suffix = model[len(prefix):]
        if suffix == "preview" or (
            len(suffix) == 8 and suffix.isascii() and suffix.isdigit()
        ):
            return item
    return None


def resolve_effort(
    wire_format: int,
    model: str,
    effort: str | None,
) -> str | int | None:
    """Resolve one model's effort identifier to its provider wire value."""

    if effort is None:
        return None
    if not isinstance(effort, str) or not effort:
        raise ValueError("effort must be a non-empty string or None")
    capability = _capability_for_model(wire_format, model, model_capabilities())
    if capability is None:
        return effort
    for option in capability.efforts:
        if option.id == effort:
            return option.value
    accepted = ", ".join(option.id for option in capability.efforts) or "none"
    raise ValueError(
        f"model {model!r} does not accept effort {effort!r}; "
        f"accepted efforts: {accepted}"
    )
