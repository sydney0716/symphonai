"""Wires a ModelProvider, PermissionPolicy, and the standard local tools
into an ApiAgent, and runs a task to completion.

No UI here -- this is the thinnest possible entry point for a caller (a
script, or a future CLI) to run one agent task.
"""

from __future__ import annotations

from collections.abc import Sequence

from orchestra_api.agent_loop import DEFAULT_MAX_TURNS, AgentRunResult, ApiAgent
from orchestra_api.budgets import RunBudget
from orchestra_api.cancellation import CancellationToken
from orchestra_api.instructions import load_instructions
from orchestra_api.models import Message, Role
from orchestra_api.permissions import PermissionPolicy
from orchestra_api.providers.base import ModelProvider
from orchestra_api.tool_schema import tool_registry_schemas
from orchestra_api.tool_results import ToolResultStore
from orchestra_api.tools.base import LocalTool
from orchestra_api.tools.edit import EditFileTool, MultiEditFileTool
from orchestra_api.tools.filesystem import ListFilesTool, ReadFileTool, WriteFileTool
from orchestra_api.tools.read_ledger import ReadLedger
from orchestra_api.tools.search import GlobTool, GrepTool
from orchestra_api.tools.shell import RunShellTool
from orchestra_api.tools.stored_result import ReadToolResultTool


def standard_tool_registry(
    names: Sequence[str] | None = None,
    *,
    ledger: ReadLedger | None = None,
    result_store: ToolResultStore | None = None,
) -> dict[str, LocalTool]:
    """The eight standard tools, plus read_tool_result when a store is supplied.

    `names` selects a subset, returned in canonical registry order, and an
    unknown or empty sequence raises `ValueError`.
    """
    if ledger is None:
        ledger = ReadLedger()
    tools: list[LocalTool] = [
        ReadFileTool(ledger),
        WriteFileTool(ledger),
        EditFileTool(ledger),
        MultiEditFileTool(ledger),
        ListFilesTool(),
        GlobTool(),
        GrepTool(),
        RunShellTool(),
    ]
    if result_store is not None:
        tools.append(ReadToolResultTool(result_store))
    if names is not None:
        if not names:
            raise ValueError("names must not be empty; omit it for the full registry")
        known_names = {tool.name for tool in tools}
        for name in names:
            if name not in known_names:
                raise ValueError(f"unknown tool name: {name!r}")
        requested_names = set(names)
    else:
        requested_names = None
    return {
        tool.name: tool
        for tool in tools
        if requested_names is None or tool.name in requested_names
    }


def run_task(
    provider: ModelProvider,
    policy: PermissionPolicy,
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    model: str | None = None,
    cancel: CancellationToken | None = None,
    include_instructions: bool = False,
    offload_tool_results: bool = False,
    budget: RunBudget | None = None,
) -> AgentRunResult:
    """Run a single task to completion using the standard tool registry.

    Automatic instructions intentionally omit directory scope.
    """
    ledger: ReadLedger | None = None
    instruction_text = ""
    if include_instructions:
        ledger = ReadLedger()
        instruction_text = load_instructions(policy, ledger=ledger).render()

    messages: list[Message] = []
    combined_system_prompt = "\n\n".join(
        part for part in (instruction_text, system_prompt) if part
    )
    if combined_system_prompt:
        messages.append(Message(role=Role.SYSTEM, content=combined_system_prompt))
    messages.append(Message(role=Role.USER, content=prompt))

    result_store = ToolResultStore() if offload_tool_results else None
    tools = standard_tool_registry(ledger=ledger, result_store=result_store)
    agent = ApiAgent(
        provider=provider,
        tools=tools,
        policy=policy,
        max_turns=max_turns,
        tool_schemas=tool_registry_schemas(tools, provider.wire_format),
        result_store=result_store,
        budget=budget,
    )
    return agent.run(messages, model=model, cancel=cancel)
