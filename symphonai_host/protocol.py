"""Versioned, transport-neutral JSON protocol for runtime events."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

from symphonai_api.events import Event


PROTOCOL_VERSION = 1
_FRAME_KINDS = {"event", "reply", "error", "approval_requested"}


class ProtocolError(ValueError):
    """A frame, event, or request does not meet the host protocol."""


@dataclass(frozen=True)
class UnknownEvent:
    """A forward-compatible event record this build cannot interpret."""

    type: str
    data: dict


@dataclass(frozen=True)
class PromptRequest:
    prompt: str


@dataclass(frozen=True)
class ApprovalReply:
    approval_id: str
    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class ApprovalRequested:
    approval_id: str
    operation: str
    target: str
    details: str


@dataclass(frozen=True)
class StopRequest:
    reason: str = ""


def event_type_name(event_class: type) -> str:
    """Return an Event subclass's unchanged class name for the wire."""
    return event_class.__name__


def _event_subclasses(event_class: type[Event]) -> list[type[Event]]:
    subclasses: list[type[Event]] = []
    for subclass in event_class.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(_event_subclasses(subclass))
    return subclasses


def event_registry() -> dict[str, type[Event]]:
    """Derive decodable event types so new runtime events need no table edit."""
    return {event_type_name(event_class): event_class for event_class in _event_subclasses(Event)}


def encode_event(event: Event) -> dict:
    """Encode an event as its flat, complete JSON-shaped record."""
    return {"type": event_type_name(type(event)), **dataclasses.asdict(event)}


def decode_event(data: dict) -> Event | UnknownEvent:
    """Decode a known Event or preserve an unknown event record verbatim."""
    if not isinstance(data, dict):
        raise ProtocolError("event must be an object")
    event_type = data.get("type")
    if not isinstance(event_type, str):
        raise ProtocolError("event type must be a string")
    event_class = event_registry().get(event_type)
    if event_class is None:
        return UnknownEvent(type=event_type, data=data)

    values: dict[str, Any] = {}
    for field in dataclasses.fields(event_class):
        if field.name not in data:
            raise ProtocolError(f"event {event_type} is missing field {field.name}")
        values[field.name] = data[field.name]
    return event_class(**values)


def encode_frame(kind: str, payload: dict) -> str:
    """Encode one protocol frame as a JSON text record."""
    if kind not in _FRAME_KINDS:
        raise ProtocolError(f"unknown frame kind {kind!r}")
    if not isinstance(payload, dict):
        raise ProtocolError("frame payload must be an object")
    return json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "kind": kind, "payload": payload}
    )


def decode_frame(text: str) -> tuple[str, dict]:
    """Decode and validate one protocol frame."""
    try:
        frame = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid protocol frame: {exc}") from None
    if not isinstance(frame, dict):
        raise ProtocolError("protocol frame must be an object")
    version = frame.get("protocol_version")
    if not isinstance(version, int):
        raise ProtocolError("protocol_version must be an integer")
    if version > PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version {version} is newer than supported {PROTOCOL_VERSION}"
        )
    kind = frame.get("kind")
    payload = frame.get("payload")
    if kind not in _FRAME_KINDS:
        raise ProtocolError(f"unknown frame kind {kind!r}")
    if not isinstance(payload, dict):
        raise ProtocolError("frame payload must be an object")
    return kind, payload


def _required(payload: dict, kind: str, field: str, expected_type: type) -> Any:
    if field not in payload:
        raise ProtocolError(f"{kind} request is missing field {field}")
    value = payload[field]
    if type(value) is not expected_type:
        raise ProtocolError(
            f"{kind} request field {field} must be {expected_type.__name__}"
        )
    return value


def _optional_string(payload: dict, kind: str, field: str) -> str:
    if field not in payload:
        return ""
    return _required(payload, kind, field, str)


def decode_request(kind: str, payload: dict) -> PromptRequest | ApprovalReply | StopRequest:
    """Validate and decode a client request independent of its transport."""
    if not isinstance(payload, dict):
        raise ProtocolError(f"{kind} request payload must be an object")
    if kind == "prompt":
        return PromptRequest(prompt=_required(payload, kind, "prompt", str))
    if kind == "approval":
        return ApprovalReply(
            approval_id=_required(payload, kind, "approval_id", str),
            allowed=_required(payload, kind, "allowed", bool),
            reason=_optional_string(payload, kind, "reason"),
        )
    if kind == "stop":
        return StopRequest(reason=_optional_string(payload, kind, "reason"))
    raise ProtocolError(f"unknown request kind {kind!r}")
