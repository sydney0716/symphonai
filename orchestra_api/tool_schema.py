"""Generic per-provider formatting for LocalTool definitions.

`LocalTool.parameters` gives every tool a vendor-neutral JSON-Schema-shaped
description of its own arguments. `to_provider_tool_schema()` formats that,
plus a tool's `name`/`description`, into a specific provider's native tool
definition shape -- the same OpenAI/Anthropic shapes already hand-written
for the one-off `dispatch_subagent_tool_schema()` in `orchestra_api.leader`,
generalized to work for any `LocalTool`.

`dispatch_subagent_tool_schema()` is deliberately left as-is, not
refactored to call into this module -- it's already tested and working,
and there's no reason to risk it for the sake of removing a few duplicated
lines.

This module holds only pure formatting functions: no network, no
subprocess, no I/O.
"""

from __future__ import annotations

from orchestra_api.tools.base import LocalTool


def to_provider_tool_schema(tool: LocalTool, provider_name: str) -> dict:
    """Format one LocalTool into a specific provider's native tool-definition shape."""
    if provider_name == "openai":
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
    if provider_name == "anthropic":
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
    # Fake and any other provider: content doesn't matter for a real call,
    # but keep it self-describing for debugging.
    return {
        "name": tool.name,
        "description": tool.description,
        "properties": tool.parameters,
    }


def tool_registry_schemas(tools: dict[str, LocalTool], provider_name: str) -> list[dict]:
    """Format a whole tool registry into a list of provider-shaped schemas."""
    return [to_provider_tool_schema(tool, provider_name) for tool in tools.values()]
