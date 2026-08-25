"""API-key based multi-agent runtime: providers, local tools, permissions, agent loop.

See `docs/orchestra-api-runtime.md` for the full design writeup.
"""

from __future__ import annotations

from orchestra_api.identity import (
    SCHEMA_VERSION,
    AgentRef,
    RunRef,
    TurnRef,
    new_agent_ref,
    new_id,
    new_run_ref,
    new_turn_ref,
)
from orchestra_api.models import (
    ContentBlock,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    ToolCall,
    ToolResult,
    TextBlock,
    Usage,
    wire_tool_call_ids,
)

__all__ = [
    "AgentRef",
    "ContentBlock",
    "Message",
    "ModelRequest",
    "ModelResponse",
    "Role",
    "RunRef",
    "SCHEMA_VERSION",
    "TextBlock",
    "ToolCall",
    "ToolResult",
    "TurnRef",
    "Usage",
    "new_agent_ref",
    "new_id",
    "new_run_ref",
    "new_turn_ref",
    "wire_tool_call_ids",
]
