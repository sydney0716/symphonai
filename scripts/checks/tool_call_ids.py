"""Fixture-free checks for tool call ids."""

from __future__ import annotations

import json
from orchestra_api.models import Message, ModelRequest, Role, ToolCall, ToolResult
from orchestra_api.providers.anthropic_provider import _build_request_body as _build_anthropic_body
from orchestra_api.providers.anthropic_provider import _parse_response as _parse_anthropic_response
from orchestra_api.providers.gemini_provider import _build_request_body as _build_gemini_body
from orchestra_api.providers.gemini_provider import _parse_response as _parse_gemini_response
from orchestra_api.providers.openai_provider import _build_request_body as _build_openai_body
from orchestra_api.providers.openai_provider import _parse_response as _parse_openai_response
from scripts.checks.harness import check, fail


@check("ids.non_empty_and_vendor_boundary")
def check_ids_non_empty_and_vendor_boundary() -> None:
    try:
        ToolCall(id="", name="x")
    except ValueError as exc:
        if str(exc) != "ToolCall.id must be a non-empty string":
            fail(f"empty ToolCall id raised the wrong error: {exc!r}")
    else:
        fail("ToolCall accepted an empty canonical id")

    malformed_anthropic = _parse_anthropic_response(
        {
            "content": [
                {"type": "tool_use", "id": "", "name": "lookup", "input": {}}
            ]
        }
    ).message.tool_calls[0]
    if not malformed_anthropic.id or malformed_anthropic.vendor_id is not None:
        fail(f"Anthropic missing-id fallback was not canonical-only: {malformed_anthropic!r}")

    real_anthropic_id = "toolu_real_vendor_id"
    normal_anthropic = _parse_anthropic_response(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": real_anthropic_id,
                    "name": "lookup",
                    "input": {"key": "value"},
                }
            ]
        }
    ).message.tool_calls[0]
    normal_anthropic_body = _build_anthropic_body(
        ModelRequest(
            messages=[
                Message(role=Role.ASSISTANT, tool_calls=[normal_anthropic]),
                Message(
                    role=Role.TOOL,
                    tool_result=ToolResult(
                        tool_call_id=normal_anthropic.id,
                        ok=True,
                        content="result",
                    ),
                ),
            ]
        ),
        "test-model",
        100,
    )
    if json.dumps(normal_anthropic_body).count(real_anthropic_id) != 2:
        fail(f"Anthropic real tool-use id did not round-trip unchanged: {normal_anthropic_body!r}")

@check("ids.request_bodies_keep_internals_private")
def check_ids_request_bodies_keep_internals_private() -> None:
    canonical_id = "canonical_internal_id"
    vendor_id = "call_abc"
    wire_messages = [
        Message(role=Role.SYSTEM, content="system", turn_id="turn_internal"),
        Message(role=Role.USER, content="question"),
        Message(
            role=Role.ASSISTANT,
            content="calling",
            tool_calls=[
                ToolCall(
                    id=canonical_id,
                    vendor_id=vendor_id,
                    name="lookup",
                    arguments={"key": "value"},
                    provider_metadata={"thoughtSignature": "opaque-signature"},
                )
            ],
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id=canonical_id, ok=True, content="result"),
        ),
    ]
    wire_request = ModelRequest(messages=wire_messages)
    wire_bodies = [
        _build_openai_body(wire_request, "test-model"),
        _build_anthropic_body(wire_request, "test-model", 100),
        _build_gemini_body(wire_request),
    ]
    for body in wire_bodies:
        serialized = json.dumps(body)
        if vendor_id not in serialized or canonical_id in serialized:
            fail(f"provider body did not preserve the vendor id boundary: {body!r}")
        if "turn_internal" in serialized or "schema_version" in serialized:
            fail(f"internal record fields leaked onto the wire: {body!r}")

    fallback_messages = [
        Message(
            role=Role.ASSISTANT,
            tool_calls=[
                ToolCall(
                    id=canonical_id,
                    name="lookup",
                    provider_metadata={"thoughtSignature": "opaque-signature"},
                )
            ],
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id=canonical_id, ok=True, content="result"),
        ),
    ]
    fallback_request = ModelRequest(messages=fallback_messages)
    fallback_bodies = [
        _build_openai_body(fallback_request, "test-model"),
        _build_anthropic_body(fallback_request, "test-model", 100),
        _build_gemini_body(fallback_request),
    ]
    if not all(canonical_id in json.dumps(body) for body in fallback_bodies):
        fail(f"canonical id fallback did not reach every provider wire body: {fallback_bodies!r}")

@check("ids.openai_missing_vendor_id")
def check_ids_openai_missing_vendor_id() -> None:
    missing_id_response = _parse_openai_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}]
                    }
                }
            ]
        }
    )
    missing_id_call = missing_id_response.message.tool_calls[0]
    if not missing_id_call.id or missing_id_call.vendor_id is not None:
        fail(f"OpenAI missing-id fallback was not canonical-only: {missing_id_call!r}")

@check("ids.synthesized_are_unique")
def check_ids_synthesized_are_unique() -> None:
    # Synthesized ids must not be position-based: the same tool at the same
    # index in a later turn would otherwise reuse the id, leaving two distinct
    # calls in one conversation sharing a tool_call_id.
    two_missing_ids = {
        _parse_openai_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "lookup", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            }
        ).message.tool_calls[0].id
        for _ in range(2)
    }
    gemini_missing_ids = {
        _parse_gemini_response(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"functionCall": {"name": "lookup", "args": {}}}]
                        }
                    }
                ]
            }
        ).message.tool_calls[0].id
        for _ in range(2)
    }
    anthropic_missing_ids = {
        _parse_anthropic_response(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "",
                        "name": "lookup",
                        "input": {},
                    }
                ]
            }
        ).message.tool_calls[0].id
        for _ in range(2)
    }
    if (
        len(two_missing_ids) != 2
        or len(gemini_missing_ids) != 2
        or len(anthropic_missing_ids) != 2
    ):
        fail(
            "synthesized tool-call ids collided across turns: "
            f"openai={two_missing_ids!r}, gemini={gemini_missing_ids!r}, "
            f"anthropic={anthropic_missing_ids!r}"
        )
