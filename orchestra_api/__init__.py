"""API-key based multi-agent runtime: providers, local tools, permissions, agent loop.

See `docs/orchestra-api-runtime.md` for the full design writeup.
"""

from __future__ import annotations

from orchestra_api.cancellation import CancellationToken, OperationCancelled
from orchestra_api.events import (
    CollectingSink,
    CompactionApplied,
    Event,
    EventSink,
    RunFailed,
    RunFinished,
    RunStarted,
    SubagentSpawned,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
    TurnStarted,
)
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
from orchestra_api.tools.metadata import (
    FAIL_CLOSED,
    InterruptBehavior,
    ResultHint,
    ToolEffect,
    ToolMetadata,
    safe_metadata,
)

__all__ = [
    "AgentRef",
    "CancellationToken",
    "CollectingSink",
    "CompactionApplied",
    "ContentBlock",
    "Event",
    "EventSink",
    "FAIL_CLOSED",
    "InterruptBehavior",
    "Message",
    "ModelRequest",
    "ModelResponse",
    "OperationCancelled",
    "ResultHint",
    "Role",
    "RunFailed",
    "RunFinished",
    "RunRef",
    "RunStarted",
    "SCHEMA_VERSION",
    "SubagentSpawned",
    "TextBlock",
    "ToolCall",
    "ToolCallFinished",
    "ToolCallStarted",
    "ToolEffect",
    "ToolMetadata",
    "ToolResult",
    "TurnFinished",
    "TurnRef",
    "TurnStarted",
    "Usage",
    "new_agent_ref",
    "new_id",
    "new_run_ref",
    "new_turn_ref",
    "safe_metadata",
    "wire_tool_call_ids",
]
