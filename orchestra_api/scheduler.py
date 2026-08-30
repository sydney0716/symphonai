"""Ordering policy for local tool calls within one model turn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from orchestra_api.models import ToolCall
from orchestra_api.tools.base import LocalTool
from orchestra_api.tools.metadata import safe_metadata


MAX_TOOL_CONCURRENCY = 8
"""Upper bound on threads for one batch.

A guardrail against a model emitting a pathological number of calls, not a
measured optimum -- the workload is network- and disk-bound and no measurement
exists yet.
"""


def partition_tool_calls(
    tool_calls: Sequence[ToolCall], tools: Mapping[str, LocalTool]
) -> list[list[ToolCall]]:
    """Group calls into batches that may run together, in argument order."""
    batches: list[list[ToolCall]] = []
    safe_batch: list[ToolCall] = []

    for tool_call in tool_calls:
        tool = tools.get(tool_call.name)
        is_safe = (
            tool is not None
            and safe_metadata(tool, tool_call.arguments).concurrency_safe
        )
        if is_safe:
            safe_batch.append(tool_call)
            continue

        if safe_batch:
            batches.append(safe_batch)
            safe_batch = []
        batches.append([tool_call])

    if safe_batch:
        batches.append(safe_batch)
    return batches
