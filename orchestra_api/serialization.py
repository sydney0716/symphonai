"""Lossless JSON serialization for transcript messages."""

from __future__ import annotations

import json

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
from orchestra_api.session import TranscriptError


def content_blocks_to_json(blocks: tuple | list) -> list[dict]:
    serialized: list[dict] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            serialized.append(
                {"kind": "text", "text": block.text, "schema_version": block.schema_version}
            )
        elif isinstance(block, ImageBlock):
            serialized.append(
                {
                    "kind": "image",
                    "data": block.data,
                    "media_type": block.media_type,
                    "schema_version": block.schema_version,
                }
            )
        elif isinstance(block, DocumentBlock):
            serialized.append(
                {
                    "kind": "document",
                    "data": block.data,
                    "media_type": block.media_type,
                    "filename": block.filename,
                    "schema_version": block.schema_version,
                }
            )
        else:
            raise TranscriptError(
                f"cannot serialize unknown content block: {type(block).__name__}"
            )
    return serialized


def content_blocks_from_json(data: list[dict]) -> tuple:
    blocks = []
    for item in data:
        kind = item.get("kind")
        if kind == "text":
            blocks.append(
                TextBlock(
                    text=item["text"],
                    schema_version=item.get("schema_version", 1),
                )
            )
        elif kind == "image":
            blocks.append(
                ImageBlock(
                    data=item["data"],
                    media_type=item["media_type"],
                    schema_version=item.get("schema_version", 1),
                )
            )
        elif kind == "document":
            blocks.append(
                DocumentBlock(
                    data=item["data"],
                    media_type=item.get("media_type", "application/pdf"),
                    filename=item.get("filename"),
                    schema_version=item.get("schema_version", 1),
                )
            )
        else:
            raise TranscriptError(f"unknown content block kind: {kind!r}")
    return tuple(blocks)


def tool_call_to_json(tool_call: ToolCall) -> dict:
    serialized = {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
        "provider_metadata": tool_call.provider_metadata,
        "vendor_id": tool_call.vendor_id,
        "schema_version": tool_call.schema_version,
    }
    try:
        json.dumps(serialized, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TranscriptError(
            f"tool call {tool_call.id!r} contains non-JSON-serializable data"
        ) from exc
    return serialized


def tool_call_from_json(data: dict) -> ToolCall:
    return ToolCall(
        id=data["id"],
        name=data["name"],
        arguments=data.get("arguments", {}),
        provider_metadata=data.get("provider_metadata", {}),
        vendor_id=data.get("vendor_id"),
        schema_version=data.get("schema_version", 1),
    )


def tool_result_to_json(tool_result: ToolResult) -> dict:
    offloaded = tool_result.offloaded
    return {
        "tool_call_id": tool_result.tool_call_id,
        "ok": tool_result.ok,
        "content": tool_result.content,
        "error": tool_result.error,
        "cancelled": tool_result.cancelled,
        "payload": tool_result.payload,
        "offloaded": (
            None
            if offloaded is None
            else {
                "id": offloaded.id,
                "characters": offloaded.characters,
                "preview_characters": offloaded.preview_characters,
                "tool_name": offloaded.tool_name,
                "schema_version": offloaded.schema_version,
            }
        ),
        "schema_version": tool_result.schema_version,
    }


def tool_result_from_json(data: dict) -> ToolResult:
    serialized_offloaded = data.get("offloaded")
    offloaded = (
        None
        if serialized_offloaded is None
        else OffloadedResult(
            id=serialized_offloaded["id"],
            characters=serialized_offloaded["characters"],
            preview_characters=serialized_offloaded["preview_characters"],
            tool_name=serialized_offloaded["tool_name"],
            schema_version=serialized_offloaded.get("schema_version", 1),
        )
    )
    return ToolResult(
        tool_call_id=data["tool_call_id"],
        ok=data["ok"],
        content=data.get("content", ""),
        error=data.get("error"),
        cancelled=data.get("cancelled", False),
        payload=data.get("payload"),
        offloaded=offloaded,
        schema_version=data.get("schema_version", 1),
    )


def message_to_json(message: Message) -> dict:
    return {
        "role": message.role.value,
        "content": content_blocks_to_json(message.content),
        "tool_calls": [tool_call_to_json(call) for call in message.tool_calls],
        "tool_result": (
            None
            if message.tool_result is None
            else tool_result_to_json(message.tool_result)
        ),
        "turn_id": message.turn_id,
        "schema_version": message.schema_version,
    }


def message_from_json(data: dict) -> Message:
    result = data.get("tool_result")
    return Message(
        role=Role(data["role"]),
        content=content_blocks_from_json(data.get("content", [])),
        tool_calls=[tool_call_from_json(call) for call in data.get("tool_calls", [])],
        tool_result=None if result is None else tool_result_from_json(result),
        turn_id=data.get("turn_id"),
        schema_version=data.get("schema_version", 1),
    )
