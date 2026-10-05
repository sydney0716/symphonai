"""Versioned, transport-neutral JSON protocol for runtime events."""

from __future__ import annotations

import base64
import binascii
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

from symphonai_api.events import Event
from symphonai_api.content import content_block_from_bytes
from symphonai_api.models import DocumentBlock, ImageBlock


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
    attachments: tuple[ImageBlock | DocumentBlock, ...] = ()


@dataclass(frozen=True)
class ApprovalReply:
    approval_id: str
    allowed: bool
    reason: str = ""
    remember: bool = False


@dataclass(frozen=True)
class ApprovalRequested:
    approval_id: str
    operation: str
    target: str
    details: str
    tool_call_id: str = ""
    remember: str = ""
    session_id: str | None = None


@dataclass(frozen=True)
class StopRequest:
    reason: str = ""


@dataclass(frozen=True)
class OpenSessionRequest:
    run_id: str


@dataclass(frozen=True)
class HistoryMessage:
    """A replay record intentionally limited to display-safe message fields."""

    role: str
    text: str
    tool_calls: list[dict[str, str]]
    turn_id: str | None
    attachments: list[dict] = field(default_factory=list, kw_only=True)
    session_id: str | None = field(default=None, kw_only=True)

    def payload(self) -> dict:
        return {
            "type": "HistoryMessage",
            "role": self.role,
            "text": self.text,
            "tool_calls": self.tool_calls,
            "turn_id": self.turn_id,
            "attachments": self.attachments,
            "session_id": self.session_id,
        }


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


def _optional_bool(payload: dict, kind: str, field: str) -> bool:
    if field not in payload:
        return False
    return _required(payload, kind, field, bool)


def decode_request(kind: str, payload: dict) -> PromptRequest | ApprovalReply | StopRequest | OpenSessionRequest:
    """Validate and decode a client request independent of its transport."""
    if not isinstance(payload, dict):
        raise ProtocolError(f"{kind} request payload must be an object")
    if kind == "prompt":
        prompt = _required(payload, kind, "prompt", str)
        serialized = payload.get("attachments", [])
        if not isinstance(serialized, list):
            raise ProtocolError("prompt request field attachments must be a list")
        if len(serialized) > 10:
            raise ProtocolError("attachment 10: at most 10 attachments are allowed")
        attachments = []
        for index, item in enumerate(serialized):
            if not isinstance(item, dict):
                raise ProtocolError(f"attachment {index}: must be an object")
            data = item.get("data")
            if not isinstance(data, str):
                raise ProtocolError(f"attachment {index}: data must be a base64 string")
            filename = item.get("filename")
            if "filename" in item and not isinstance(filename, str):
                raise ProtocolError(f"attachment {index}: filename must be a string")
            try:
                raw = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                raise ProtocolError(f"attachment {index}: invalid base64") from None
            try:
                attachments.append(content_block_from_bytes(raw, filename=filename))
            except ValueError as exc:
                raise ProtocolError(f"attachment {index}: {exc}") from None
        return PromptRequest(prompt=prompt, attachments=tuple(attachments))
    if kind == "approval":
        return ApprovalReply(
            approval_id=_required(payload, kind, "approval_id", str),
            allowed=_required(payload, kind, "allowed", bool),
            reason=_optional_string(payload, kind, "reason"),
            remember=_optional_bool(payload, kind, "remember"),
        )
    if kind == "stop":
        return StopRequest(reason=_optional_string(payload, kind, "reason"))
    if kind == "session/open":
        return OpenSessionRequest(run_id=_required(payload, kind, "run_id", str))
    raise ProtocolError(f"unknown request kind {kind!r}")
