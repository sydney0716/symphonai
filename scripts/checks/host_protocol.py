"""Checks for the transport-neutral SymphonAI host protocol."""

from __future__ import annotations

import dataclasses
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

from symphonai_api.events import Event
from symphonai_host.protocol import (
    PROTOCOL_VERSION,
    ApprovalReply,
    PromptRequest,
    ProtocolError,
    StopRequest,
    UnknownEvent,
    decode_event,
    decode_frame,
    decode_request,
    encode_event,
    encode_frame,
    event_registry,
)
from scripts.checks.harness import check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]


def _event_instance(event_class: type[Event], *, turn_id_none: bool = False) -> Event:
    values: dict[str, object] = {}
    hints = get_type_hints(event_class)
    for field in dataclasses.fields(event_class):
        if field.name == "turn_id":
            values[field.name] = None if turn_id_none else "turn-1"
        elif field.name == "schema_version":
            values[field.name] = 1
        elif hints[field.name] is str:
            values[field.name] = f"{field.name}-value"
        elif hints[field.name] is int:
            values[field.name] = 7
        elif hints[field.name] is bool:
            values[field.name] = True
        else:
            raise AssertionError(f"unsupported event field {event_class.__name__}.{field.name}")
    return event_class(**values)


def _hand_written_payload(event_class: type[Event]) -> dict:
    instance = _event_instance(event_class, turn_id_none=True)
    return {
        "type": event_class.__name__,
        **{field.name: getattr(instance, field.name) for field in dataclasses.fields(event_class)},
    }


@check("host_protocol.encodes_every_event")
def check_encodes_every_event() -> None:
    registry = event_registry()
    if not registry:
        fail("derived event registry was empty")
    for name, event_class in registry.items():
        event = _event_instance(event_class)
        encoded = encode_event(event)
        expected_fields = {field.name for field in dataclasses.fields(event_class)}
        if encoded.get("type") != name or set(encoded) != {"type", *expected_fields}:
            fail(f"{name} was not encoded as a complete flat record: {encoded!r}")
        try:
            json.dumps(encoded)
        except TypeError as exc:
            fail(f"{name} needed a custom JSON encoder: {exc}")


@check("host_protocol.round_trip_events")
def check_round_trip_events() -> None:
    for event_class in event_registry().values():
        event = _event_instance(event_class, turn_id_none=True)
        if decode_event(encode_event(event)) != event:
            fail(f"{event_class.__name__} did not round-trip from an event")
        payload = _hand_written_payload(event_class)
        decoded = decode_event(payload)
        if encode_event(decoded) != payload:
            fail(f"{event_class.__name__} did not round-trip from a payload: {payload!r}")


@check("host_protocol.unknown_type_preserved")
def check_unknown_type_preserved() -> None:
    payload = {"type": "FutureEvent", "agent_id": "agent", "new_field": {"nested": True}}
    decoded = decode_event(payload)
    if not isinstance(decoded, UnknownEvent) or decoded.type != "FutureEvent" or decoded.data != payload:
        fail(f"unknown event was not preserved intact: {decoded!r}")


@check("host_protocol.field_mismatch")
def check_field_mismatch() -> None:
    event_class = event_registry()["RunStarted"]
    payload = _hand_written_payload(event_class)
    payload["future_field"] = "ignored"
    decoded = decode_event(payload)
    if encode_event(decoded) != {key: value for key, value in payload.items() if key != "future_field"}:
        fail(f"known event did not ignore its unknown field: {decoded!r}")
    missing = _hand_written_payload(event_class)
    del missing["agent_name"]
    try:
        decode_event(missing)
    except ProtocolError as exc:
        if "RunStarted" not in str(exc) or "agent_name" not in str(exc):
            fail(f"missing field error did not name type and field: {exc}")
    else:
        fail("known event accepted a missing field")


@check("host_protocol.registry_is_derived")
def check_registry_is_derived() -> None:
    @dataclass(frozen=True)
    class AddedForProtocolCheck(Event):
        marker: str = ""

    payload = {
        "type": "AddedForProtocolCheck",
        "agent_id": "agent",
        "run_id": "run",
        "turn_id": None,
        "schema_version": 1,
        "marker": "derived",
    }
    decoded = decode_event(payload)
    if not isinstance(decoded, AddedForProtocolCheck) or decoded.marker != "derived":
        fail(f"new Event subclass was absent from the derived registry: {decoded!r}")
    del decoded
    del AddedForProtocolCheck
    gc.collect()


@check("host_protocol.frame_version")
def check_frame_version() -> None:
    if decode_frame(encode_frame("event", {"type": "RunStarted"})) != (
        "event",
        {"type": "RunStarted"},
    ):
        fail("valid protocol frame did not round-trip")
    future = json.dumps(
        {"protocol_version": PROTOCOL_VERSION + 1, "kind": "event", "payload": {}}
    )
    try:
        decode_frame(future)
    except ProtocolError as exc:
        if str(PROTOCOL_VERSION + 1) not in str(exc) or str(PROTOCOL_VERSION) not in str(exc):
            fail(f"version error omitted one of the versions: {exc}")
    else:
        fail("newer protocol version was accepted")


@check("host_protocol.request_validation")
def check_request_validation() -> None:
    if decode_request("prompt", {"prompt": "hello"}) != PromptRequest("hello"):
        fail("valid prompt request was not decoded")
    if decode_request("approval", {"approval_id": "approval-1", "allowed": True}) != ApprovalReply("approval-1", True):
        fail("valid approval request was not decoded")
    if decode_request("stop", {}) != StopRequest():
        fail("valid stop request was not decoded")
    cases = (
        ("prompt", {}, "prompt"),
        ("approval", {"approval_id": "approval-1", "allowed": "yes"}, "allowed"),
        ("future", {}, "future"),
    )
    for kind, payload, expected in cases:
        try:
            decode_request(kind, payload)
        except ProtocolError as exc:
            if expected not in str(exc):
                fail(f"{kind} validation error omitted its name: {exc}")
        else:
            fail(f"invalid {kind} request was accepted")


@check("host_protocol.document_covers_registry")
def check_document_covers_registry() -> None:
    gc.collect()
    rows = (REPO_ROOT / "symphonai_host" / "PROTOCOL.md").read_text().splitlines()
    for name, event_class in event_registry().items():
        row = next((line for line in rows if line.startswith(f"| `{name}` |")), None)
        if row is None:
            fail(f"protocol document omitted event {name}")
        for field in dataclasses.fields(event_class):
            if field.name not in row:
                fail(f"protocol document omitted {name}.{field.name}")


@check("host_protocol.import_direction")
def check_import_direction() -> None:
    # Stdlib rather than ripgrep: the suite has to run wherever the package
    # does, and `rg` is a developer's tool, not a dependency this repo has.
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "symphonai_api").rglob("*.py"))
        if "symphonai_host" in path.read_text(encoding="utf-8")
    ]
    if offenders:
        fail(f"runtime imports host code: {offenders}")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    if "dependencies = []" not in pyproject or '"symphonai_host*"' not in pyproject:
        fail("host package changed dependencies or was omitted from package discovery")
