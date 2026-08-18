"""The Leader: digests a user goal and dispatches/reuses named subagents.

The leader is itself an `ApiAgent`, but with exactly one tool available:
`dispatch_subagent`. Calling it with a new `subagent_name` creates a fresh
subagent (its own `ApiAgent`, backed by the one configured subagent
provider, with the standard local tool registry and a deny-by-default
`PermissionPolicy`); calling it again with the same name continues that
subagent's existing conversation instead of starting over -- that is the
"reuse" behavior.

The leader never chooses which vendor/model backs itself or its
subagents. Both are fixed by `LeaderConfig`, set by the caller, never
decided by a model. See `docs/orchestra-api-runtime.md`.

Subagent pool state lives only in memory for the duration of one
`Leader.run()` call -- there is no cross-run persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from orchestra_api.agent_loop import DEFAULT_MAX_TURNS, ApiAgent
from orchestra_api.models import Message, Role, ToolCall, ToolResult
from orchestra_api.permissions import PermissionPolicy
from orchestra_api.providers.base import ModelProvider
from orchestra_api.runner import standard_tool_registry
from orchestra_api.tools.base import LocalTool

DISPATCH_TOOL_NAME = "dispatch_subagent"
DEFAULT_MAX_SUBAGENTS = 5
DEFAULT_SUBAGENT_MAX_TURNS = 5

# Coarse status states reported via on_status(label, status). Deliberately
# not a fine-grained event stream (no message content, no per-turn detail)
# -- just "is this agent pending, working, done, or failed."
STATUS_PENDING = "pending"
STATUS_WORKING = "working"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

StatusCallback = Callable[[str, str], None]


def _report(on_status: StatusCallback | None, label: str, status: str) -> None:
    if on_status is not None:
        on_status(label, status)

_DISPATCH_DESCRIPTION = (
    "Dispatch a task to a named subagent. If subagent_name has not been "
    "used yet in this run, a new subagent is created. If it has, the same "
    "subagent continues its existing conversation with this new task "
    "instead of starting over -- use the same name to follow up with the "
    "same subagent."
)
_DISPATCH_PROPERTIES = {
    "subagent_name": {
        "type": "string",
        "description": (
            "A short, stable identifier for this subagent, e.g. "
            "'researcher' or 'coder'. Reuse the same name to continue a "
            "conversation with the same subagent."
        ),
    },
    "task": {
        "type": "string",
        "description": "The task or follow-up message to give this subagent.",
    },
}
_DISPATCH_REQUIRED = ["subagent_name", "task"]


def dispatch_subagent_tool_schema(provider_name: str) -> dict:
    """Build the dispatch_subagent tool definition in one provider's native shape.

    This is deliberately narrow -- a hand-written schema for this one tool,
    not a general cross-vendor tool-schema translator (that remains an open
    gap; see docs/orchestra-api-runtime.md).
    """
    if provider_name == "openai":
        return {
            "type": "function",
            "function": {
                "name": DISPATCH_TOOL_NAME,
                "description": _DISPATCH_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": _DISPATCH_PROPERTIES,
                    "required": _DISPATCH_REQUIRED,
                },
            },
        }
    if provider_name == "anthropic":
        return {
            "name": DISPATCH_TOOL_NAME,
            "description": _DISPATCH_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": _DISPATCH_PROPERTIES,
                "required": _DISPATCH_REQUIRED,
            },
        }
    # Fake and any other provider: content doesn't matter for a real call,
    # but keep it self-describing for debugging.
    return {
        "name": DISPATCH_TOOL_NAME,
        "description": _DISPATCH_DESCRIPTION,
        "properties": _DISPATCH_PROPERTIES,
    }


@dataclass
class SubagentRecord:
    """One named subagent's live state within a single leader run."""

    agent: ApiAgent
    messages: list[Message] = field(default_factory=list)
    turns_used: int = 0


class DispatchSubagentTool(LocalTool):
    """The leader's only tool: create-or-reuse a named subagent and run it.

    Note: the `policy` argument `execute()` receives is the *leader's own*
    policy (passed in by the leader's ApiAgent), which this tool
    deliberately ignores -- each subagent runs under its own
    `subagent_policy`, fixed at construction time, independent of whatever
    the leader itself is permitted to touch.
    """

    def __init__(
        self,
        subagent_provider: ModelProvider,
        subagent_policy: PermissionPolicy,
        *,
        max_subagents: int = DEFAULT_MAX_SUBAGENTS,
        subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS,
        on_status: StatusCallback | None = None,
    ) -> None:
        self._subagent_provider = subagent_provider
        self._subagent_policy = subagent_policy
        self._max_subagents = max_subagents
        self._subagent_max_turns = subagent_max_turns
        self._on_status = on_status
        self.pool: dict[str, SubagentRecord] = {}

    @property
    def name(self) -> str:
        return DISPATCH_TOOL_NAME

    @property
    def description(self) -> str:
        return _DISPATCH_DESCRIPTION

    def execute(self, tool_call: ToolCall, policy: PermissionPolicy) -> ToolResult:
        subagent_name = tool_call.arguments.get("subagent_name")
        task = tool_call.arguments.get("task")
        if not subagent_name or not task:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error="missing required argument: subagent_name and/or task",
            )

        record = self.pool.get(subagent_name)
        if record is None:
            if len(self.pool) >= self._max_subagents:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error=(
                        f"max_subagents ({self._max_subagents}) reached; "
                        f"cannot create new subagent {subagent_name!r}"
                    ),
                )
            _report(self._on_status, subagent_name, STATUS_PENDING)
            record = SubagentRecord(
                agent=ApiAgent(
                    provider=self._subagent_provider,
                    tools=standard_tool_registry(),
                    policy=self._subagent_policy,
                    max_turns=self._subagent_max_turns,
                )
            )
            self.pool[subagent_name] = record

        record.messages.append(Message(role=Role.USER, content=task))
        _report(self._on_status, subagent_name, STATUS_WORKING)
        run_result = record.agent.run(record.messages)
        record.messages = run_result.messages
        record.turns_used += run_result.turns_used

        succeeded = run_result.stopped_reason == "final_response"
        _report(self._on_status, subagent_name, STATUS_DONE if succeeded else STATUS_FAILED)

        return ToolResult(
            tool_call_id=tool_call.id,
            ok=succeeded,
            content=run_result.final_response.message.content,
            error=None if succeeded else "subagent reached max_turns without a final answer",
        )


@dataclass
class LeaderConfig:
    """Fixed, user-chosen configuration for a leader run.

    The leader never picks its own or its subagents' provider/model --
    both are fixed here by the caller, once, before the run starts.
    """

    leader_provider: ModelProvider
    subagent_provider: ModelProvider
    repo_root: str
    max_leader_turns: int = DEFAULT_MAX_TURNS
    max_subagents: int = DEFAULT_MAX_SUBAGENTS
    subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS
    on_status: StatusCallback | None = None


@dataclass
class LeaderRunResult:
    """The outcome of `Leader.run()`."""

    final_answer: str
    leader_messages: list[Message]
    stopped_reason: str
    subagents: dict[str, SubagentRecord]


class Leader:
    """Digests a user goal, dispatching/reusing named subagents as needed.

    `run()` is one-shot: each call starts a fresh conversation. `chat()` is
    the multi-turn version: it keeps its own running message list across
    calls, so a later `chat()` call sees everything said in earlier ones.
    Use `run()` for a single task; use `chat()` for an interactive session.
    """

    def __init__(self, config: LeaderConfig) -> None:
        self._config = config
        subagent_policy = PermissionPolicy(repo_root=config.repo_root)
        self._dispatch_tool = DispatchSubagentTool(
            subagent_provider=config.subagent_provider,
            subagent_policy=subagent_policy,
            max_subagents=config.max_subagents,
            subagent_max_turns=config.subagent_max_turns,
            on_status=config.on_status,
        )
        leader_policy = PermissionPolicy(repo_root=config.repo_root)
        self._agent = ApiAgent(
            provider=config.leader_provider,
            tools={DISPATCH_TOOL_NAME: self._dispatch_tool},
            policy=leader_policy,
            max_turns=config.max_leader_turns,
            tool_schemas=[dispatch_subagent_tool_schema(config.leader_provider.name)],
        )
        self._chat_messages: list[Message] = []

    @property
    def subagents(self) -> dict[str, SubagentRecord]:
        return self._dispatch_tool.pool

    def _run_messages(self, messages: list[Message]) -> LeaderRunResult:
        _report(self._config.on_status, "leader", STATUS_WORKING)
        result = self._agent.run(messages)
        _report(self._config.on_status, "leader", STATUS_DONE)
        return LeaderRunResult(
            final_answer=result.final_response.message.content,
            leader_messages=result.messages,
            stopped_reason=result.stopped_reason,
            subagents=self._dispatch_tool.pool,
        )

    def run(self, goal: str, *, system_prompt: str | None = None) -> LeaderRunResult:
        """Run a single, one-shot task. Each call starts a fresh conversation."""
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        messages.append(Message(role=Role.USER, content=goal))
        return self._run_messages(messages)

    def chat(self, message: str) -> LeaderRunResult:
        """Continue an ongoing conversation with the leader.

        Unlike `run()`, this keeps its own running message history across
        calls -- a later `chat()` call includes everything said in earlier
        ones, so the leader (and its view of already-dispatched subagents)
        has full context of the conversation so far.
        """
        self._chat_messages.append(Message(role=Role.USER, content=message))
        result = self._run_messages(self._chat_messages)
        self._chat_messages = result.leader_messages
        return result
