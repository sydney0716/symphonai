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
from orchestra_api.tools.web_fetch import WebFetchTool
from orchestra_api.web import (
    DEFAULT_FETCH_TIMEOUT_SECONDS,
    MAX_FETCH_BYTES,
    MAX_REDIRECTS,
    FetchedPage,
    WebFetchError,
    fetch_url,
)
from orchestra_api.web_domains import preapproved_domains

__all__ = [
    "AgentRef",
    "CancellationToken",
    "CollectingSink",
    "CompactionApplied",
    "ContentBlock",
    "Event",
    "EventSink",
    "FetchedPage",
    "FAIL_CLOSED",
    "InterruptBehavior",
    "MAX_FETCH_BYTES",
    "MAX_REDIRECTS",
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
    "WebFetchError",
    "WebFetchTool",
    "DEFAULT_FETCH_TIMEOUT_SECONDS",
    "fetch_url",
    "new_agent_ref",
    "new_id",
    "new_run_ref",
    "new_turn_ref",
    "preapproved_domains",
    "safe_metadata",
    "wire_tool_call_ids",
]
