"""Registered checks for configured observation and blocking hooks."""

from __future__ import annotations

import ast
import inspect
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import MappingProxyType
from unittest import mock

from symphonai_api.config import ConfigError, Provenance, ResolvedConfig, Scope, load_config
from symphonai_api.events import Event, ToolCallFailed, ToolCallFinished, emit
from symphonai_api.hooks import HookOutcome, HookRunner, HookSpec, hooks_from_config
from symphonai_api.models import Message, ModelResponse, Role, ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.agent_loop import ApiAgent
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_api.trust import RepositoryTrust, TrustList
from scripts.checks.harness import check, fail


_PRE_10C_COMMIT = "f65784f6089dc114f5b7dbab6ab8ecd42c025db2"
_BASE_FIELDS = {
    "agent_id": "agent",
    "run_id": "run",
    "turn_id": "turn",
}


def _script(root: Path, name: str, source: str) -> Path:
    path = root / name
    path.write_text(source, encoding="utf-8")
    return path


def _scope_path(scope: Scope, repo_root: Path, home: Path) -> Path:
    if scope is Scope.USER:
        return home / ".symphonai" / "config.toml"
    if scope is Scope.PROJECT:
        return repo_root / ".symphonai" / "config.toml"
    if scope is Scope.PRIVATE:
        return repo_root / ".symphonai" / "config.local.toml"
    raise ValueError("session scope has no path")


def _write_config(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _resolved(hooks: object, source: Path | None = None) -> ResolvedConfig:
    provenance = {
        "hooks": Provenance("hooks", Scope.SESSION, source),
    }
    return ResolvedConfig(
        values=MappingProxyType({"hooks": hooks}),
        provenance=MappingProxyType(provenance),
    )


class _RecordingTool(LocalTool):
    def __init__(self) -> None:
        self.invocations = 0

    @property
    def name(self) -> str:
        return "recording"

    @property
    def description(self) -> str:
        return "Record whether execution happened."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(ToolEffect.MUTATING, False, ())

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel=None,
    ) -> ToolResult:
        self.invocations += 1
        return ToolResult(tool_call_id=tool_call.id, ok=True, content="ran")


@check("hooks.config_parsing")
def config_parsing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "config.toml"
        specs = hooks_from_config(
            _resolved(
                [
                    {
                        "on": ["ToolCallFailed", "PermissionDenied"],
                        "command": ["./notify"],
                    },
                    {
                        "on": ["PreToolUse"],
                        "command": ["./guard"],
                        "timeout_seconds": 30,
                        "blocking": True,
                    },
                ],
                source,
            ),
            repo_root=root,
        )
        expected = (
            HookSpec(
                ("ToolCallFailed", "PermissionDenied"),
                ("./notify",),
                5.0,
                False,
                source,
            ),
            HookSpec(("PreToolUse",), ("./guard",), 30.0, True, source),
        )
        if specs != expected:
            fail(f"hook config parsed incorrectly: {specs!r}")
        outcome = HookOutcome(True, 0, False, "out", "err")
        try:
            outcome.ok = False  # type: ignore[misc]
        except Exception:
            pass
        else:
            fail("HookOutcome is not frozen")


@check("hooks.load_from_a_real_file")
def load_from_a_real_file() -> None:
    toml = '[[hooks]]\non = ["RunStarted"]\ncommand = ["./notify"]\n'
    raw_hook = {"on": ["RunStarted"], "command": ["./notify"]}
    for scope in Scope:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary) / "repo"
            home = Path(temporary) / "home"
            if scope is Scope.SESSION:
                source = None
                session = {"hooks": [raw_hook]}
            else:
                source = _scope_path(scope, repo_root, home)
                _write_config(source, toml)
                session = None
            resolved = load_config(
                repo_root=repo_root,
                home=home,
                session=session,
            )
            if resolved.scope_of("hooks") is not scope:
                fail(f"{scope.value} hooks had the wrong provenance")
            if scope in (Scope.PROJECT, Scope.PRIVATE):
                try:
                    hooks_from_config(resolved, repo_root=repo_root)
                except ConfigError as exc:
                    message = str(exc)
                    if (
                        str(source) not in message
                        or "inside a repository" not in message
                    ):
                        fail(f"{scope.value} refusal was incomplete: {message!r}")
                    continue
                fail(f"{scope.value} hooks loaded from a real file")
            expected = (
                HookSpec(
                    ("RunStarted",),
                    ("./notify",),
                    source=source,
                ),
            )
            actual = hooks_from_config(resolved, repo_root=repo_root)
            if actual != expected:
                fail(f"{scope.value} hooks parsed incorrectly: {actual!r}")


@check("hooks.config_rejections")
def config_rejections() -> None:
    cases = (
        ({"on": ["UnknownEvent"], "command": ["ok"]}, "on", "UnknownEvent"),
        ({"on": ["RunStarted"], "command": "echo ok"}, "command", None),
        (
            {"on": ["RunStarted"], "command": ["ok"], "timeout_seconds": 0},
            "timeout_seconds",
            None,
        ),
        (
            {"on": ["RunStarted"], "command": ["ok"], "timeout_seconds": 30.1},
            "timeout_seconds",
            None,
        ),
        (
            {"on": ["RunStarted"], "command": ["ok"], "blocking": True},
            "blocking",
            "one-way",
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for index, (raw, field, detail) in enumerate(cases):
            try:
                hooks_from_config(_resolved([raw]), repo_root=root)
            except ConfigError as exc:
                message = str(exc)
                if "hooks[0]" not in message or field not in message:
                    fail(f"case {index} did not name its index and field: {message!r}")
                if detail is not None and detail not in message:
                    fail(f"case {index} omitted {detail!r}: {message!r}")
            else:
                fail(f"invalid hook config case {index} was accepted")


@check("hooks.pre_tool_requires_blocking")
def pre_tool_requires_blocking() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for raw in (
            {"on": ["PreToolUse"], "command": ["guard"]},
            {"on": ["PreToolUse"], "command": ["guard"], "blocking": False},
        ):
            try:
                hooks_from_config(_resolved([raw]), repo_root=root)
            except ConfigError as exc:
                message = str(exc)
                required = ("hooks[0]", "blocking", "veto point", "never run")
                if not all(fragment in message for fragment in required):
                    fail(f"non-blocking PreToolUse error was incomplete: {message!r}")
            else:
                fail("non-blocking PreToolUse hook was accepted")

        accepted = hooks_from_config(
            _resolved(
                [
                    {
                        "on": ["PreToolUse"],
                        "command": ["guard"],
                        "blocking": True,
                    }
                ]
            ),
            repo_root=root,
        )
        if accepted != (HookSpec(("PreToolUse",), ("guard",), blocking=True),):
            fail(f"blocking PreToolUse hook parsed incorrectly: {accepted!r}")

        try:
            hooks_from_config(
                _resolved(
                    [
                        {
                            "on": ["RunStarted"],
                            "command": ["observe"],
                            "blocking": True,
                        }
                    ]
                ),
                repo_root=root,
            )
        except ConfigError as exc:
            if "hooks[0]" not in str(exc) or "blocking" not in str(exc):
                fail(f"blocking channel-event error was incomplete: {exc!r}")
        else:
            fail("blocking channel-event hook was accepted")

        for events in (
            ["PreToolUse", "RunStarted"],
            ["RunStarted", "PreToolUse"],
        ):
            try:
                hooks_from_config(
                    _resolved([{"on": events, "command": ["mixed"]}]),
                    repo_root=root,
                )
            except ConfigError as exc:
                message = str(exc)
                if "hooks[0]" not in message or ".on" not in message:
                    fail(f"mixed PreToolUse error was incomplete: {message!r}")
            else:
                fail(f"mixed PreToolUse hook was accepted: {events!r}")


@check("hooks.project_scope_is_refused")
def project_scope_is_refused() -> None:
    repo_toml = (
        '[[hooks]]\non = ["RunFinished"]\ncommand = ["./repo-hook"]\n'
    )
    user_toml = '[[hooks]]\non = ["RunStarted"]\ncommand = ["./user-hook"]\n'
    for scope in (Scope.PROJECT, Scope.PRIVATE):
        for user_present in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                repo_root = Path(temporary) / "repo"
                home = Path(temporary) / "home"
                source = _scope_path(scope, repo_root, home)
                _write_config(source, repo_toml)
                if user_present:
                    _write_config(
                        _scope_path(Scope.USER, repo_root, home),
                        user_toml,
                    )
                resolved = load_config(repo_root=repo_root, home=home)
                if resolved.scope_of("hooks") is not scope:
                    fail(f"{scope.value} hooks did not win over user hooks")
                try:
                    hooks_from_config(resolved, repo_root=repo_root)
                except ConfigError as exc:
                    message = str(exc)
                    required = (
                        str(source),
                        "inside a repository",
                        "~/.symphonai/config.toml",
                    )
                    if not all(fragment in message for fragment in required):
                        fail(f"{scope.value} hook refusal was incomplete: {message!r}")
                    guidance = message.removeprefix(str(source))
                    if ".symphonai/config.local.toml" in guidance:
                        fail(f"private config was advertised as trusted: {message!r}")
                else:
                    detail = " over user hooks" if user_present else ""
                    fail(f"{scope.value} hooks were honoured{detail}")

    with tempfile.TemporaryDirectory() as temporary:
        repo_root = Path(temporary) / "repo"
        home = Path(temporary) / "home"
        user_path = _scope_path(Scope.USER, repo_root, home)
        _write_config(user_path, user_toml)
        resolved = load_config(repo_root=repo_root, home=home)
        expected = (HookSpec(("RunStarted",), ("./user-hook",), source=user_path),)
        actual = hooks_from_config(resolved, repo_root=repo_root)
        if actual != expected:
            fail(f"user hooks without a repository winner differed: {actual!r}")


@check("hooks.exact_matching_and_sink")
def exact_matching_and_sink() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "events.jsonl"
        writer = _script(
            root,
            "write_event.py",
            "import pathlib, sys\npathlib.Path(sys.argv[1]).open('a').write(sys.stdin.read() + '\\n')\n",
        )
        runner = HookRunner(
            [
                HookSpec(("ToolCallFailed",), (sys.executable, str(writer), str(output))),
                HookSpec(("Event",), (sys.executable, str(writer), str(output))),
            ],
            cwd=root,
        )
        signature = inspect.signature(runner.__call__)
        if list(signature.parameters) != ["event"]:
            fail(f"HookRunner is not an EventSink callable: {signature!s}")
        emit(runner, ToolCallFinished(**_BASE_FIELDS, tool_name="tool", tool_call_id="ok", ok=True))
        emit(
            runner,
            ToolCallFailed(
                **_BASE_FIELDS,
                tool_name="tool",
                tool_call_id="failed",
                error="boom",
            ),
        )
        lines = output.read_text(encoding="utf-8").splitlines()
        payload = json.loads(lines[0]) if len(lines) == 1 else None
        if payload != {
            "type": "ToolCallFailed",
            "agent_id": "agent",
            "run_id": "run",
            "turn_id": "turn",
            "schema_version": 1,
            "tool_name": "tool",
            "tool_call_id": "failed",
            "error": "boom",
        }:
            fail(f"hook matching was not exact by class name: {lines!r}")


@check("hooks.configuration_order")
def configuration_order() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "order.txt"
        writer = _script(
            root,
            "append.py",
            "import pathlib, sys\nwith pathlib.Path(sys.argv[1]).open('a') as f: f.write(sys.argv[2] + '\\n')\n",
        )
        hooks = [
            HookSpec(
                ("ToolCallFailed",),
                (sys.executable, str(writer), str(output), label),
            )
            for label in ("first", "second", "third")
        ]
        emit(
            HookRunner(hooks, cwd=root),
            ToolCallFailed(**_BASE_FIELDS, error="failure"),
        )
        if output.read_text(encoding="utf-8").splitlines() != [
            "first",
            "second",
            "third",
        ]:
            fail("matching hooks did not run in configuration order")


@check("hooks.observation_isolation")
def observation_isolation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        marker = root / "third-ran"
        failing = _script(root, "fail.py", "raise SystemExit(7)\n")
        hanging = _script(root, "hang.py", "import signal\nsignal.pause()\n")
        succeeding = _script(
            root,
            "succeed.py",
            "import pathlib, sys\npathlib.Path(sys.argv[1]).write_text('yes')\n",
        )
        runner = HookRunner(
            [
                HookSpec(("PromptSubmitted",), (sys.executable, str(failing))),
                HookSpec(("PromptSubmitted",), (sys.executable, str(hanging)), 0.1),
                HookSpec(
                    ("PromptSubmitted",),
                    (sys.executable, str(succeeding), str(marker)),
                ),
            ],
            cwd=root,
        )
        result = ApiAgent(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            {},
            PermissionPolicy(root),
            events=runner,
        ).run([Message(Role.USER, "run hooks")])
        if result.stopped_reason != "final_response" or marker.read_text() != "yes":
            fail("one observational hook failure prevented a later hook or changed the run")


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@check("hooks.timeout_kills_process_group")
def timeout_kills_process_group() -> None:
    if not hasattr(os, "killpg"):
        return
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        pid_file = root / "grandchild.pid"
        parent = _script(
            root,
            "tree.py",
            """import pathlib, signal, subprocess, sys
child = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.pause()",
])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.pause()
""",
        )
        refusal = HookRunner(
            [
                HookSpec(
                    ("PreToolUse",),
                    (sys.executable, str(parent), str(pid_file)),
                    0.25,
                    True,
                )
            ],
            cwd=root,
        ).pre_tool("tool", "call")
        if refusal is None or "timed out" not in refusal:
            fail(f"process-tree hook did not time out closed: {refusal!r}")
        if not pid_file.is_file():
            fail("timed-out parent never recorded its grandchild")
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 2.0
        waiter = threading.Event()
        while True:
            if not _process_exists(pid):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                fail("grandchild survived hook timeout")
            if waiter.wait(min(0.02, remaining)):
                fail("private process wait event was unexpectedly set")


@check("hooks.blocking_fail_closed")
def blocking_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        hanging = _script(root, "hang.py", "import signal\nsignal.pause()\n")
        failing = _script(root, "fail.py", "raise SystemExit(9)\n")
        unreadable = _script(
            root,
            "unreadable.py",
            "import sys\nsys.stdout.buffer.write(b'\\xff')\n",
        )
        denying = _script(root, "deny.py", "print('deny: protected by policy')\n")
        permitting = _script(root, "permit.py", "print('allow')\n")
        cases = (
            ((sys.executable, str(hanging)), 0.1, "timed out"),
            ((sys.executable, str(failing)), 1.0, "non-zero"),
            ((str(root / "missing-executable"),), 1.0, "missing executable"),
            ((sys.executable, str(unreadable)), 1.0, "unparseable output"),
            ((sys.executable, str(denying)), 1.0, "protected by policy"),
        )
        for command, timeout, expected in cases:
            refusal = HookRunner(
                [HookSpec(("PreToolUse",), command, timeout, True)],
                cwd=root,
            ).pre_tool("write_file", "call")
            if refusal is None or expected not in refusal:
                fail(f"blocking failure {expected!r} did not refuse: {refusal!r}")
        permitted = HookRunner(
            [
                HookSpec(
                    ("PreToolUse",),
                    (sys.executable, str(permitting)),
                    blocking=True,
                )
            ],
            cwd=root,
        ).pre_tool("read_file", "call")
        if permitted is not None:
            fail(f"zero exit without a deny line refused: {permitted!r}")


@check("hooks.agent_veto")
def agent_veto() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        denying = _script(root, "deny.py", "print('deny: guarded')\n")
        runner = HookRunner(
            [
                HookSpec(
                    ("PreToolUse",),
                    (sys.executable, str(denying)),
                    blocking=True,
                )
            ],
            cwd=root,
        )
        tool = _RecordingTool()
        result = ApiAgent(
            FakeModelProvider(
                [
                    ModelResponse(
                        Message(
                            Role.ASSISTANT,
                            tool_calls=[ToolCall(id="guarded-call", name=tool.name)],
                        )
                    ),
                    ModelResponse(Message(Role.ASSISTANT, "done")),
                ]
            ),
            {tool.name: tool},
            PermissionPolicy(root),
        ).run([Message(Role.USER, "try tool")], hooks=runner)
        denied = next(
            message.tool_result
            for message in result.messages
            if message.tool_result is not None
        )
        if tool.invocations != 0 or denied.ok or denied.error != "guarded":
            fail(f"refused tool still ran or returned the wrong result: {denied!r}")
        source = (Path(__file__).resolve().parents[2] / "symphonai_api/agent_loop.py").read_text()
        if source.count("hooks.pre_tool(") != 1:
            fail("agent loop does not contain exactly one hook veto point")


_FROZEN_NO_HOOKS = (
    "final_response",
    2,
    (
        ("user", "frozen", (), None),
        ("assistant", "", (("execute", "recording"),), None),
        ("tool", "", (), ("execute", True, "ran", None)),
        ("assistant", "done", (), None),
    ),
    1,
    1,
)


def _message_snapshot(message: Message):
    result = message.tool_result
    return (
        message.role.value,
        message.text,
        tuple((call.id, call.name) for call in message.tool_calls),
        None
        if result is None
        else (result.tool_call_id, result.ok, result.content, result.error),
    )


@check("hooks.none_is_unchanged_and_imports")
def none_is_unchanged_and_imports() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        tool = _RecordingTool()
        with mock.patch.object(
            HookRunner,
            "pre_tool",
            side_effect=AssertionError("hooks=None consulted a runner"),
        ):
            result = ApiAgent(
                FakeModelProvider(
                    [
                        ModelResponse(
                            Message(
                                Role.ASSISTANT,
                                tool_calls=[ToolCall(id="execute", name=tool.name)],
                            )
                        ),
                        ModelResponse(Message(Role.ASSISTANT, "done")),
                    ]
                ),
                {tool.name: tool},
                PermissionPolicy(root),
            ).run([Message(Role.USER, "frozen")], hooks=None)
        actual = (
            result.stopped_reason,
            result.turns_used,
            tuple(_message_snapshot(message) for message in result.messages),
            tool.invocations,
            result.final_response.message.schema_version,
        )
        if actual != _FROZEN_NO_HOOKS:
            fail(
                f"hooks=None changed behavior from {_PRE_10C_COMMIT}: "
                f"expected={_FROZEN_NO_HOOKS!r}, actual={actual!r}"
            )

    hooks_path = Path(__file__).resolve().parents[2] / "symphonai_api/hooks.py"
    tree = ast.parse(hooks_path.read_text(encoding="utf-8"))
    forbidden = {
        "agent_loop",
        "leader",
        "runner",
        "agent_run",
        "agent_spec",
        "agent_file",
        "permissions",
        "child_context",
        "provider_catalog",
        "providers",
    }
    imported = {
        node.module.split(".")[1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("symphonai_api.")
    }
    if imported & forbidden:
        fail(f"hooks.py imports forbidden runtime modules: {sorted(imported & forbidden)!r}")
    check_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "time"
        and node.func.attr == "sleep"
        for node in ast.walk(check_tree)
    ):
        fail("hooks checks use time.sleep")


@check("hooks.trust_grants_a_repository")
def trust_grants_a_repository() -> None:
    repo_toml = (
        '[[hooks]]\non = ["RunFinished"]\ncommand = ["./repo-hook"]\n'
    )
    for scope in (Scope.PROJECT, Scope.PRIVATE):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            repo_root = base / "repo"
            home = base / "home"
            source = _scope_path(scope, repo_root, home)
            _write_config(source, repo_toml)
            config = load_config(repo_root=repo_root, home=home)
            grants = (
                None,
                TrustList(),
                TrustList(
                    (
                        RepositoryTrust(
                            (base / "other").resolve(),
                            frozenset(("hooks",)),
                            None,
                        ),
                    )
                ),
                TrustList(
                    (
                        RepositoryTrust(
                            repo_root.resolve(),
                            frozenset(("mcp",)),
                            None,
                        ),
                    )
                ),
            )
            for trust in grants:
                try:
                    hooks_from_config(config, repo_root=repo_root, trust=trust)
                except ConfigError as exc:
                    message = str(exc)
                    required = (
                        str(source),
                        "inside a repository",
                        "~/.symphonai/config.toml",
                        "[[trust.repositories]]",
                    )
                    if not all(fragment in message for fragment in required):
                        fail(f"{scope.value} trust refusal was incomplete: {message!r}")
                else:
                    fail(f"{scope.value} hooks accepted insufficient trust: {trust!r}")

            trust = TrustList(
                (
                    RepositoryTrust(
                        repo_root.resolve(),
                        frozenset(("hooks",)),
                        None,
                    ),
                )
            )
            actual = hooks_from_config(config, repo_root=repo_root, trust=trust)
            expected = (
                HookSpec(
                    ("RunFinished",),
                    ("./repo-hook",),
                    source=source,
                ),
            )
            if actual != expected:
                fail(
                    f"{scope.value} hooks did not parse after exact trust grant: "
                    f"{actual!r}"
                )
