"""Checks for pure child-run context seeding."""

from __future__ import annotations

from itertools import chain, product
import tempfile
from pathlib import Path

from symphonai_api.agent_spec import (
    AgentSpec,
    ContextInheritance,
    Isolation,
    ModelSelector,
)
from symphonai_api.child_context import seed_messages, tail_start
from symphonai_api.compaction import recent_window_start
from symphonai_api.models import Message, Role, ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from scripts.checks.agent_spec import FORBIDDEN_IMPORTS, _forbidden_imports
from scripts.checks.harness import check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]


def _spec(
    root: Path,
    *,
    prompt: str = "",
    inherit: ContextInheritance = ContextInheritance.FRESH,
    inherit_tail: int = 0,
) -> AgentSpec:
    return AgentSpec(
        name="child",
        prompt=prompt,
        model=ModelSelector("fake"),
        policy_ceiling=PermissionPolicy(repo_root=root),
        isolation=Isolation(inherit=inherit, inherit_tail=inherit_tail),
    )


def _assistant(*call_ids: str) -> Message:
    return Message(
        role=Role.ASSISTANT,
        content="calls",
        tool_calls=[ToolCall(call_id, "tool") for call_id in call_ids],
    )


def _tool(call_id: str) -> Message:
    return Message(
        role=Role.TOOL,
        tool_result=ToolResult(tool_call_id=call_id, ok=True, content="result"),
    )


def _issued_tool_call_ids(messages: list[Message]) -> set[str]:
    issued: set[str] = set()
    for message in messages:
        if message.role is Role.ASSISTANT:
            issued.update(call.id for call in message.tool_calls)
    return issued


def _orphan_tool_result_ids(messages: list[Message]) -> set[str]:
    issued: set[str] = set()
    orphan_ids: set[str] = set()
    for message in messages:
        if message.role is Role.ASSISTANT:
            issued.update(call.id for call in message.tool_calls)
        elif message.role is Role.TOOL and message.tool_result is not None:
            if message.tool_result.tool_call_id not in issued:
                orphan_ids.add(message.tool_result.tool_call_id)
    return orphan_ids


def _assert_no_orphan_tools(
    messages: list[Message],
    permitted_orphan_ids: set[str] | None = None,
) -> None:
    permitted_ids = permitted_orphan_ids if permitted_orphan_ids is not None else set()
    unexpected_orphan_ids = _orphan_tool_result_ids(messages) - permitted_ids
    if unexpected_orphan_ids:
        fail(f"orphan tool result: {sorted(unexpected_orphan_ids)!r}")


PARENT_SYSTEM = Message(Role.SYSTEM, "parent system")
WEDGED_USER = [
    Message(Role.USER, "one"),
    _assistant("one"),
    Message(Role.USER, "between"),
    _tool("one"),
]
USERLESS_TOOL_GROUPS = [
    [_assistant("c0"), _tool("c0")],
    [_assistant("c0"), _tool("c0"), Message(Role.ASSISTANT, "reply")],
]
GHOST_PARENTS = [
    [_tool("ghost")],
    [Message(Role.USER, "one"), _tool("ghost")],
]
REGRESSION_CONVERSATIONS = [
    [],
    [Message(Role.USER, "one")],
    [Message(Role.USER, "one"), Message(Role.ASSISTANT, "reply")],
    [Message(Role.USER, "one"), _assistant("a"), _tool("a")],
    [Message(Role.USER, "one"), _assistant("a", "b"), _tool("a"), _tool("b")],
    [Message(Role.USER, "one"), Message(Role.ASSISTANT, "reply"), Message(Role.USER, "two")],
    [Message(Role.USER, "one"), _assistant("a"), _tool("a"), Message(Role.USER, "two"), Message(Role.ASSISTANT, "reply")],
    WEDGED_USER,
    [Message(Role.SYSTEM, "system"), Message(Role.ASSISTANT, "reply")],
    *USERLESS_TOOL_GROUPS,
    *GHOST_PARENTS,
    [PARENT_SYSTEM],
    [PARENT_SYSTEM, Message(Role.USER, "one"), _assistant("one"), _tool("one"), Message(Role.USER, "two"), Message(Role.ASSISTANT, "final")],
]
ALPHABET = [
    Message(Role.USER, "u"),
    Message(Role.SYSTEM, "s"),
    Message(Role.ASSISTANT, "a"),
    _assistant("a"),
    _tool("a"),
    _tool("g"),
]

def _enumerated_conversations() -> list[list[Message]]:
    conversations: list[list[Message]] = []
    for length in range(6):
        conversations.extend(list(messages) for messages in product(ALPHABET, repeat=length))
    return conversations


def _tool_result_ids(messages: list[Message]) -> set[str]:
    return {
        message.tool_result.tool_call_id
        for message in messages
        if message.role is Role.TOOL and message.tool_result is not None
    }


@check("child_context.fresh_is_todays_behaviour")
def fresh_is_todays_behaviour() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = [
            Message(Role.SYSTEM, "parent system"),
            Message(Role.USER, "parent user"),
        ]
        fresh = seed_messages(_spec(root), "task", parent_messages=parent)
        if fresh != [Message(role=Role.USER, content="task")]:
            fail(f"fresh context changed today's shape: {fresh!r}")
        prompted = seed_messages(
            _spec(root, prompt="child system"),
            "task",
            parent_messages=parent,
        )
        if prompted != [
            Message(role=Role.SYSTEM, content="child system"),
            Message(role=Role.USER, content="task"),
        ]:
            fail(f"fresh context inherited parent messages: {prompted!r}")
        whitespace = seed_messages(_spec(root, prompt=" \t"), "task")
        if whitespace != [Message(role=Role.USER, content="task")]:
            fail("whitespace prompt created a system message")


@check("child_context.inherit_all")
def inherit_all() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = [
            Message(Role.SYSTEM, "parent system one"),
            Message(Role.USER, "first"),
            _assistant("call-one"),
            _tool("call-one"),
            Message(Role.SYSTEM, "parent system two"),
            Message(Role.ASSISTANT, "last"),
        ]
        seeded = seed_messages(
            _spec(root, prompt="child system", inherit=ContextInheritance.ALL),
            "child task",
            parent_messages=parent,
        )
        expected = [
            Message(Role.SYSTEM, "child system"),
            parent[1],
            parent[2],
            parent[3],
            parent[5],
            Message(Role.USER, "child task"),
        ]
        if seeded != expected:
            fail(f"all inheritance changed order or retained a system prompt: {seeded!r}")


@check("child_context.inherit_tail")
def inherit_tail() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for messages in chain(REGRESSION_CONVERSATIONS, _enumerated_conversations()):
            for turns in range(1, 5):
                recent_start = recent_window_start(messages, turns)
                start = tail_start(messages, turns)
                if start > recent_start:
                    fail(f"tail_start advanced beyond the recent window: {messages!r}")
                literal_orphan_ids = _orphan_tool_result_ids(messages[recent_start:])
                literal_has_orphan = bool(literal_orphan_ids)
                rescuable_orphan_ids = literal_orphan_ids & _issued_tool_call_ids(
                    messages[:recent_start]
                )
                if literal_has_orphan and rescuable_orphan_ids:
                    if start >= recent_start:
                        fail(f"tail_start did not extend an orphaning window: {messages!r}")
                elif start != recent_start:
                    fail(f"tail_start diverged without a rescuable orphan: {messages!r}")
                _assert_no_orphan_tools(
                    messages[start:],
                    _orphan_tool_result_ids(messages),
                )

        parent = [
            Message(Role.USER, "first"),
            Message(Role.ASSISTANT, "first reply"),
            Message(Role.USER, "second"),
            _assistant("call-two"),
            _tool("call-two"),
        ]
        tail_one = seed_messages(
            _spec(root, inherit=ContextInheritance.TAIL, inherit_tail=1),
            "task",
            parent_messages=parent,
        )
        if tail_one != [*parent[2:], Message(Role.USER, "task")]:
            fail(f"one-turn tail was not the final user turn: {tail_one!r}")
        tail_two = seed_messages(
            _spec(root, inherit=ContextInheritance.TAIL, inherit_tail=2),
            "task",
            parent_messages=parent,
        )
        if tail_two != [*parent, Message(Role.USER, "task")]:
            fail(f"two-turn tail did not retain the full conversation: {tail_two!r}")
        tail_many = seed_messages(
            _spec(root, inherit=ContextInheritance.TAIL, inherit_tail=4),
            "task",
            parent_messages=parent,
        )
        if tail_many != [*parent, Message(Role.USER, "task")]:
            fail("oversized tail did not retain the full conversation")


@check("child_context.purity")
def purity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = [Message(Role.USER, "parent")]
        spec = _spec(root, inherit=ContextInheritance.ALL)
        parent_before = list(parent)
        seeded = seed_messages(spec, "task", parent_messages=parent)
        if parent != parent_before:
            fail("seeding mutated parent_messages")
        if seeded is parent:
            fail("seeding returned the caller's list")
        seeded.append(Message(Role.ASSISTANT, "new"))
        if parent != parent_before:
            fail("appending to seeded messages mutated parent_messages")
        if spec != _spec(root, inherit=ContextInheritance.ALL):
            fail("seeding mutated the AgentSpec")


@check("child_context.never_orphans_a_tool_result")
def never_orphans_a_tool_result() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for parent in chain(REGRESSION_CONVERSATIONS, _enumerated_conversations()):
            parent_systems = [message for message in parent if message.role is Role.SYSTEM]
            parent_orphan_ids = _orphan_tool_result_ids(parent)
            specs = [
                _spec(root),
                _spec(root, inherit=ContextInheritance.ALL),
                *[
                    _spec(root, inherit=ContextInheritance.TAIL, inherit_tail=turns)
                    for turns in range(1, 5)
                ],
            ]
            for spec in specs:
                seeded = seed_messages(spec, "task", parent_messages=parent)
                _assert_no_orphan_tools(seeded, parent_orphan_ids)
                if any(message is parent_system for message in seeded for parent_system in parent_systems):
                    fail("seeded context retained a parent system message")
                if spec.isolation.inherit is ContextInheritance.ALL:
                    inherited = parent
                elif spec.isolation.inherit is ContextInheritance.TAIL:
                    inherited = parent[tail_start(parent, spec.isolation.inherit_tail):]
                else:
                    inherited = []
                expected_orphan_ids = parent_orphan_ids & _tool_result_ids(inherited)
                if not expected_orphan_ids <= _tool_result_ids(seeded):
                    fail("seeding dropped an unrescuable parent tool result")

        if tail_start(WEDGED_USER, 1) >= recent_window_start(WEDGED_USER, 1):
            fail("tail_start did not extend an orphaning tool group backward")


@check("child_context.no_runtime_imports")
def no_runtime_imports() -> None:
    source = (REPO_ROOT / "symphonai_api/child_context.py").read_text()
    original_forbidden = set(FORBIDDEN_IMPORTS)
    FORBIDDEN_IMPORTS.add("agent_run")
    try:
        found = _forbidden_imports(source)
        if found:
            fail(f"child_context imports runtime wiring: {found!r}")
        probes = [
            ("from symphonai_api.agent_loop import ApiAgent\n", True),
            ("from symphonai_api.leader import Leader\n", True),
            ("from symphonai_api.runner import run_task\n", True),
            ("from symphonai_api.provider_catalog import providers\n", True),
            ("from symphonai_api.agent_run import AgentRun\n", True),
            ("from symphonai_api.providers.openai import OpenAIProvider\n", True),
            ("from symphonai_api import agent_run\n", True),
            ("from . import leader\n", True),
        ]
        for line, expected in probes:
            if bool(_forbidden_imports(line)) != expected:
                fail(f"import inspection got {line.strip()!r} wrong")
    finally:
        FORBIDDEN_IMPORTS.clear()
        FORBIDDEN_IMPORTS.update(original_forbidden)
