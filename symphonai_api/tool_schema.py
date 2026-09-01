"""Generic per-provider formatting for LocalTool definitions.

`LocalTool.parameters` gives every tool a vendor-neutral JSON-Schema-shaped
description of its own arguments. `to_provider_tool_schema()` formats that,
plus a tool's `name`/`description`, into a provider wire format -- the same
OpenAI/Anthropic shapes already hand-written for the one-off
`dispatch_subagent_tool_schema()` in `symphonai_api.leader`, plus Gemini's
function-declaration schema.

`dispatch_subagent_tool_schema()` is deliberately left as-is, not
refactored to call into this module -- it's already tested and working,
and there's no reason to risk it for the sake of removing a few duplicated
lines.

This module holds only pure formatting functions: no network, no
subprocess, no I/O.
"""

from __future__ import annotations

from symphonai_api.gemini_schema import sanitize_for_gemini
from symphonai_api.tools.base import LocalTool


def to_provider_tool_schema(tool: LocalTool, wire_format: int) -> dict:
    """Format one LocalTool into a provider's native tool-definition shape."""
    if wire_format == 1:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
    if wire_format == 2:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
    if wire_format == 3:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": sanitize_for_gemini(tool.parameters),
        }
    # Other/unclassified providers: keep schemas self-describing for debugging.
    return {
        "name": tool.name,
        "description": tool.description,
        "properties": tool.parameters,
    }


def tool_registry_schemas(tools: dict[str, LocalTool], wire_format: int) -> list[dict]:
    """Format a whole tool registry into a list of provider-shaped schemas."""
    return [to_provider_tool_schema(tool, wire_format) for tool in tools.values()]
