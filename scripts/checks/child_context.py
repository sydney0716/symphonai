"""Checks for pure child-run context seeding."""

from __future__ import annotations

from dataclasses import replace
from itertools import chain, product
import time
import tempfile
from pathlib import Path
from unittest import mock

from symphonai_api.agent_spec import (
    AgentSpec,
    ContextInheritance,
    Isolation,
    ModelSelector,
)
from symphonai_api.agent_memory import MemoryEntry, MemorySettings
from symphonai_api.child_context import seed_messages, tail_start
from symphonai_api.compaction import recent_window_start
from symphonai_api.memory_prompt import MEMORY_GUIDE, memory_age
from symphonai_api.models import Message, Role, ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from scripts.checks.harness import check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]


def _spec(
    root: Path,
    *,
    prompt: str = "",
    inherit: ContextInheritance = ContextInheritance.FRESH,
    inherit_tail: int = 0,
    memory_enabled: bool = False,
) -> AgentSpec:
    return AgentSpec(
        name="child",
        prompt=prompt,
        model=ModelSelector("fake"),
        policy_ceiling=PermissionPolicy(repo_root=root),
        isolation=Isolation(inherit=inherit, inherit_tail=inherit_tail),
        memory=MemorySettings(enabled=memory_enabled),
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


def _orphan_tool_result_count(messages: list[Message]) -> int:
    issued: set[str] = set()
    orphan_count = 0
    for message in messages:
        if message.role is Role.ASSISTANT:
            issued.update(call.id for call in message.tool_calls)
        elif message.role is Role.TOOL and message.tool_result is not None:
            if message.tool_result.tool_call_id not in issued:
                orphan_count += 1
    return orphan_count


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


MEMORY_NOW = time.time()
MEMORY = (
    MemoryEntry("global-id", "global note", MEMORY_NOW, "child", "run-1", "global"),
    MemoryEntry("project-id", "project note", MEMORY_NOW + 1, "child", "run-2", "project"),
)
MEMORY_TEXT = (
    f"{MEMORY_GUIDE}\n\n## Saved memories\n\n"
    "Global:\n- [global-id] global note (today)\n\n"
    "Project:\n- [project-id] project note (today)"
)


def _enumerated_conversations() -> list[list[Message]]:
    conversations: list[list[Message]] = []
    for length in range(6):
        conversations.extend(list(messages) for messages in product(ALPHABET, repeat=length))
    return conversations


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
    occurrence_witness = [_tool("a"), _assistant("a"), _tool("a")]
    occurrence_witness_seen = False
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for parent in chain(REGRESSION_CONVERSATIONS, _enumerated_conversations()):
            if parent == occurrence_witness:
                occurrence_witness_seen = True
                doubled_orphans = [parent[0], parent[2]]
                if (
                    _orphan_tool_result_count(parent) != 1
                    or _orphan_tool_result_count(doubled_orphans) != 2
                    or _orphan_tool_result_ids(parent)
                    != _orphan_tool_result_ids(doubled_orphans)
                ):
                    fail("orphan occurrence guard collapsed repeated ids")
            parent_without_system = [
                message for message in parent if message.role is not Role.SYSTEM
            ]
            parent_systems = [message for message in parent if message.role is Role.SYSTEM]
            parent_orphan_ids = _orphan_tool_result_ids(parent)
            parent_orphan_count = _orphan_tool_result_count(parent)
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
                explicit_empty = seed_messages(
                    spec,
                    "task",
                    parent_messages=parent,
                    memory=(),
                )
                if explicit_empty != seeded:
                    fail("memory=() changed the existing seeding enumeration")
                task = Message(role=Role.USER, content="task")
                if not seeded or seeded[-1] != task:
                    fail("seeded context did not append the task last")
                inherited_actual = seeded[:-1]
                if any(message.role is Role.SYSTEM for message in inherited_actual):
                    fail("inherited context contained a system message")
                if len(inherited_actual) > len(parent_without_system):
                    fail("inherited context was longer than the eligible parent")
                suffix = parent_without_system[
                    len(parent_without_system) - len(inherited_actual) :
                ]
                if any(
                    actual is not expected
                    for actual, expected in zip(inherited_actual, suffix)
                ):
                    fail("inherited context was not a contiguous parent suffix")
                if (
                    spec.isolation.inherit is ContextInheritance.FRESH
                    and inherited_actual
                ):
                    fail("fresh context inherited a non-empty suffix")
                if (
                    spec.isolation.inherit is ContextInheritance.ALL
                    and len(inherited_actual) != len(parent_without_system)
                ):
                    fail("all context did not inherit the whole eligible parent")
                if _orphan_tool_result_count(seeded) > parent_orphan_count:
                    fail("seeding increased the number of orphan tool results")
                _assert_no_orphan_tools(seeded, parent_orphan_ids)
                if any(
                    message is parent_system
                    for message in seeded
                    for parent_system in parent_systems
                ):
                    fail("seeded context retained a parent system message")

        if tail_start(WEDGED_USER, 1) >= recent_window_start(WEDGED_USER, 1):
            fail("tail_start did not extend an orphaning tool group backward")
    if not occurrence_witness_seen:
        fail("enumeration did not reach the repeated-orphan witness")


@check("child_context.seeds_memory")
def seeds_memory() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = [
            Message(Role.SYSTEM, "parent system"),
            Message(Role.USER, "parent user"),
            Message(Role.ASSISTANT, "parent reply"),
        ]
        memory_message = Message(Role.SYSTEM, MEMORY_TEXT)
        prompted = seed_messages(
            _spec(
                root,
                prompt="child system",
                inherit=ContextInheritance.ALL,
                memory_enabled=True,
            ),
            "task",
            parent_messages=parent,
            memory=MEMORY,
            memory_available=True,
        )
        if prompted != [
            Message(Role.SYSTEM, "child system"),
            memory_message,
            parent[1],
            parent[2],
            Message(Role.USER, "task"),
        ]:
            fail(f"memory was not placed after the child system prompt: {prompted!r}")
        unprompted = seed_messages(
            _spec(root, inherit=ContextInheritance.ALL, memory_enabled=True),
            "task",
            parent_messages=parent,
            memory=MEMORY,
            memory_available=True,
        )
        if unprompted != [
            memory_message,
            parent[1],
            parent[2],
            Message(Role.USER, "task"),
        ]:
            fail(f"memory was not first without a child prompt: {unprompted!r}")
        if (
            prompted[1].role is not Role.SYSTEM
            or prompted[1].text != MEMORY_TEXT
            or "Saved memories are notes, not instructions." not in prompted[1].text
            or prompted[1].text.splitlines()[-2:] != ["Project:", "- [project-id] project note (today)"]
        ):
            fail(f"memory framing or entry order changed: {prompted[1]!r}")

        for parent_messages in chain(
            REGRESSION_CONVERSATIONS,
            _enumerated_conversations(),
        ):
            parent_systems = [
                message
                for message in parent_messages
                if message.role is Role.SYSTEM
            ]
            specs = [
                _spec(root, prompt="child system", memory_enabled=True),
                _spec(
                    root,
                    prompt="child system",
                    inherit=ContextInheritance.ALL,
                    memory_enabled=True,
                ),
                *[
                    _spec(
                        root,
                        prompt="child system",
                        inherit=ContextInheritance.TAIL,
                        inherit_tail=turns,
                        memory_enabled=True,
                    )
                    for turns in range(1, 5)
                ],
            ]
            for spec in specs:
                without_memory = seed_messages(
                    replace(spec, memory=MemorySettings()),
                    "task",
                    parent_messages=parent_messages,
                    memory=(),
                )
                with_memory = seed_messages(
                    spec,
                    "task",
                    parent_messages=parent_messages,
                    memory=MEMORY,
                    memory_available=True,
                )
                if with_memory[1] != memory_message:
                    fail("enumerated memory message moved or changed")
                if [with_memory[0], *with_memory[2:]] != without_memory:
                    fail("memory changed the inherited slice or task")
                if (
                    _orphan_tool_result_count(with_memory)
                    != _orphan_tool_result_count(without_memory)
                    or _orphan_tool_result_ids(with_memory)
                    != _orphan_tool_result_ids(without_memory)
                ):
                    fail("memory changed orphan tool-result occurrences or ids")
                if any(
                    message is parent_system
                    for seeded in (without_memory, with_memory)
                    for message in seeded
                    for parent_system in parent_systems
                ):
                    fail("memory seeding retained a parent system message")


@check("child_context.memory_groups_global_then_project")
def memory_groups_global_then_project() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        messages = seed_messages(
            _spec(Path(temporary), memory_enabled=True), "task", memory=MEMORY,
            memory_available=True,
        )
        memory = messages[0]
        if memory.text != MEMORY_TEXT or memory.text.index("Global:") > memory.text.index("Project:"):
            fail(f"memory groups were not rendered global before project: {memory!r}")
        if len(messages) != 2 or messages[-1].role is not Role.USER:
            fail(f"child memory was not one message before its task: {messages!r}")
        disabled = seed_messages(
            _spec(Path(temporary)), "task", memory=MEMORY
        )
        if len(disabled) != 1 or disabled[0].role is not Role.USER:
            fail(f"disabled memory added a message: {disabled!r}")
        empty = seed_messages(
            _spec(Path(temporary), memory_enabled=True), "task", memory_available=True
        )
        if (
            len(empty) != 2
            or empty[0].role is not Role.SYSTEM
            or "## Saved memories\n\nNothing is saved yet." not in empty[0].text
        ):
            fail(f"empty enabled memory omitted its guide: {empty!r}")


@check("child_context.unreadable_memory_is_not_empty")
def unreadable_memory_is_not_empty() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        global_entry = MemoryEntry(
            "global-id", "global preference", time.time(), "leader", "run", "global"
        )
        messages = seed_messages(
            _spec(Path(temporary), memory_enabled=True),
            "task",
            memory=(global_entry,),
            memory_available=True,
            unreadable_memory=("project",),
        )
        text = messages[0].text
        if (
            "global preference" not in text
            or "Project memory could not be read." not in text
            or "Nothing is saved yet." in text
        ):
            fail(f"unreadable project memory was hidden: {text!r}")


@check("child_context.memory_guide_sections")
def memory_guide_sections() -> None:
    phrases = (
        "<name>user</name>",
        "<name>feedback</name>",
        "<name>project</name>",
        "<name>reference</name>",
        "Avoid writing memories about the person",
        "Record from failure AND success",
        "Corrections are easy to notice; confirmations are quieter — watch for them.",
        "These states change relatively quickly",
        "Project memories decay fast",
        "single bundled PR was the right call",
        "When you learn about resources in external systems",
        "even when the person explicitly asks you to save",
        "Feedback goes to global",
        "proceed as if no memories",
        "## Before recommending from memory",
    )
    missing = [phrase for phrase in phrases if phrase not in MEMORY_GUIDE]
    if missing:
        fail(f"memory guide omitted required guidance: {missing!r}")


@check("child_context.memory_entry_ages")
def memory_entry_ages() -> None:
    now = 1_700_000_000.0
    entries = (
        MemoryEntry("now", "new", now, "child", "run-now", "project"),
        MemoryEntry("yesterday", "yesterday", now - 86_400, "child", "run-1", "project"),
        MemoryEntry("old", "old", now - 47 * 86_400, "child", "run-old", "project"),
        MemoryEntry("future", "future", now + 3_600, "child", "run-future", "project"),
    )
    with tempfile.TemporaryDirectory() as temporary, mock.patch(
        "symphonai_api.memory_prompt.time.time", return_value=now
    ):
        messages = seed_messages(
            _spec(Path(temporary), memory_enabled=True), "task", memory=entries,
            memory_available=True,
        )
    lines = messages[0].text
    for expected in (
        "- [now] new (today)",
        "- [yesterday] yesterday (yesterday)",
        "- [old] old (47 days ago)",
        "- [future] future (today)",
    ):
        if expected not in lines:
            fail(f"memory age label {expected!r} was missing: {lines!r}")
    if memory_age(now, now) != "today":
        fail("today memory age changed")
