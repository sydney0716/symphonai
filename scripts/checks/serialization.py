"""Registered checks for lossless transcript serialization."""

from __future__ import annotations

import tempfile
from pathlib import Path

from orchestra_api.models import (
    DocumentBlock,
    ImageBlock,
    Message,
    OffloadedResult,
    Role,
    TextBlock,
    ToolCall,
    ToolResult,
)
from orchestra_api.serialization import (
    content_blocks_from_json,
    message_from_json,
    message_to_json,
    tool_result_from_json,
    tool_result_to_json,
)
from orchestra_api.session import TranscriptError, TranscriptWriter
from scripts.checks.harness import check, fail


def _rich_message() -> Message:
    return Message(
        role=Role.ASSISTANT,
        content=(
            TextBlock("hello"),
            ImageBlock(data="aW1hZ2U=", media_type="image/png"),
            DocumentBlock(data="cGRm", filename="notes.pdf"),
        ),
        tool_calls=[
            ToolCall(id="call-1", name="first", arguments={"depth": 2}),
            ToolCall(
                id="call-2",
                name="second",
                arguments={"items": [1, 2]},
                provider_metadata={"thoughtSignature": {"parts": ["a", {"b": 2}]}},
                vendor_id="vendor-call-2",
            ),
        ],
        turn_id="turn-round-trip",
    )


@check("serialization.message_round_trip")
def check_message_round_trip() -> None:
    message = _rich_message()
    restored = message_from_json(message_to_json(message))
    if restored != message:
        fail(f"message round trip changed the value: {restored!r}")


@check("serialization.tool_result_round_trip")
def check_tool_result_round_trip() -> None:
    result = ToolResult(
        tool_call_id="result-call",
        ok=True,
        content="preview",
        payload={"rows": [{"id": 1}]},
        offloaded=OffloadedResult(
            id="stored-1",
            characters=5000,
            preview_characters=800,
            tool_name="read_file",
        ),
    )
    restored = tool_result_from_json(tool_result_to_json(result))
    if restored != result:
        fail(f"tool result round trip changed the value: {restored!r}")


@check("serialization.provider_metadata_verbatim")
def check_provider_metadata_verbatim() -> None:
    message = _rich_message()
    encoded = message_to_json(message)
    expected = message.tool_calls[1].provider_metadata
    if encoded["tool_calls"][1]["provider_metadata"] != expected:
        fail("provider metadata changed during serialization")
    restored = message_from_json(encoded)
    if restored.tool_calls[1].provider_metadata != expected:
        fail("provider metadata changed during deserialization")
    invalid = Message(
        role=Role.ASSISTANT,
        tool_calls=[
            ToolCall(
                id="bad-metadata-call",
                name="broken",
                provider_metadata={"not_json": object()},
            )
        ],
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "serialization.jsonl"
        writer = TranscriptWriter(path)
        writer.append(
            "turn_started",
            run_id="run-serialization",
            agent_id="agent-serialization",
            turn_id="turn-serialization",
            data={"index": 1},
        )
        before = path.read_bytes()
        try:
            writer.append(
                "message",
                run_id="run-serialization",
                agent_id="agent-serialization",
                turn_id="turn-serialization",
                data=message_to_json(invalid),
            )
        except TranscriptError as exc:
            if "bad-metadata-call" not in str(exc):
                fail(f"serialization error did not name the tool call: {exc}")
        else:
            fail("non-JSON provider metadata was silently accepted")
        finally:
            writer.close()
        if path.read_bytes() != before:
            fail("invalid provider metadata wrote a partial message record")


@check("serialization.unknown_kind_rejected")
def check_unknown_kind_rejected() -> None:
    known = content_blocks_from_json(
        [{"kind": "text", "text": "future-safe", "future_field": 42}]
    )
    if known != (TextBlock("future-safe"),):
        fail(f"unknown field changed a known block: {known!r}")
    try:
        content_blocks_from_json([{"kind": "hologram", "data": "x"}])
    except TranscriptError as exc:
        if "hologram" not in str(exc):
            fail(f"unknown-kind error did not name the kind: {exc}")
    else:
        fail("unknown content block kind was accepted")
