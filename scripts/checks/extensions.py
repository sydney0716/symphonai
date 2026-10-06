"""Checks for resolving configuration into live runtime extensions."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

import symphonai_api.extensions as extensions_module
import symphonai_api.runner as runner_module
from symphonai_api.agent_file import AgentFileError, load_agent_directory
from symphonai_api.agent_loop import ApiAgent
from symphonai_api.config import CapabilityCeiling, ConfigError, Scope
from symphonai_api.discovery import Offered
from symphonai_api.cost import UsageTotals
from symphonai_api.events import CollectingSink
from symphonai_api.extensions import Extensions, load_extensions
from symphonai_api.hooks import HookRunner, HookSpec
from symphonai_api.leader import Leader, LeaderConfig
from symphonai_api.mcp import McpServerSpec
from symphonai_api.models import Message, ModelResponse, Role, ToolCall, Usage
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.runner import run_task
from symphonai_api.trust import RepositoryTrust, TrustList
from scripts.checks.harness import check, fail




def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")






def _hook_extensions(root: Path, hooks: list[dict[str, object]]) -> Extensions:
    home = root / "home"
    return load_extensions(
        repo_root=root,
        home=home,
        session={"hooks": hooks},
    )


def _leader_providers(*, with_child_tool: bool):
    leader_provider = FakeModelProvider(
        [
            ModelResponse(
                Message(
                    Role.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="dispatch",
                            name="dispatch_subagent",
                            arguments={"subagent_name": "worker", "task": "read"},
                        )
                    ],
                )
            ),
            ModelResponse(Message(Role.ASSISTANT, "leader done")),
        ]
    )
    child_responses = (
        [
            ModelResponse(
                Message(
                    Role.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="child-read",
                            name="read_file",
                            arguments={"path": "existing.txt"},
                        )
                    ],
                )
            ),
            ModelResponse(Message(Role.ASSISTANT, "child done")),
        ]
        if with_child_tool
        else [ModelResponse(Message(Role.ASSISTANT, "child done"))]
    )
    return leader_provider, FakeModelProvider(child_responses)


@check("extensions.load_real_config")
def load_real_config() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "repo"
        home = base / "home"
        source = home / ".symphonai" / "config.toml"
        root.mkdir()
        _write(
            source,
            "\n".join(
                (
                    "[[trust.repositories]]",
                    f"root = {json.dumps(str(root.resolve()))}",
                    'allow = ["hooks", "mcp"]',
                    "[agents.ceiling]",
                    'allowed_write_scope = ["work"]',
                    "shell_enabled = false",
                    'modes = ["allow"]',
                    "[[hooks]]",
                    'on = ["RunFinished"]',
                    'command = ["./observe"]',
                    "[[mcp.servers]]",
                    'name = "docs"',
                    'command = ["./server"]',
                    "enabled = true",
                )
            ),
        )
        loaded = load_extensions(repo_root=root, home=home)
        expected_ceiling = CapabilityCeiling(
            allowed_write_scope=((root / "work").resolve(),),
            shell_enabled=False,
            modes=("allow",),
        )
        expected_trust = TrustList(
            (RepositoryTrust(root.resolve(), frozenset(("hooks", "mcp")), source),)
        )
        if loaded.ceiling != expected_ceiling or loaded.trust != expected_trust:
            fail(f"real config did not populate ceiling/trust: {loaded!r}")
        if loaded.hooks != (
            HookSpec(("RunFinished",), ("./observe",), source=source),
        ):
            fail(f"real config did not populate hooks: {loaded.hooks!r}")
        if loaded.mcp_servers != (
            McpServerSpec("docs", ("./server",), enabled=True, source=source),
        ):
            fail(f"real config did not populate MCP servers: {loaded.mcp_servers!r}")
        if loaded.config.provenance["hooks"].source != source:
            fail("resolved config provenance was not retained")

        project = root / ".symphonai" / "config.toml"
        for capability, body in (
            (
                "hooks",
                '[[hooks]]\non = ["RunFinished"]\ncommand = ["./repo-hook"]\n',
            ),
            (
                "mcp",
                '[[mcp.servers]]\nname = "repo"\ncommand = ["./repo-mcp"]\nenabled = true\n',
            ),
        ):
            _write(project, body)
            _write(
                source,
                "[[trust.repositories]]\n"
                f"root = {json.dumps(str(root.resolve()))}\n"
                f'allow = ["{capability}"]\n',
            )
            trusted = load_extensions(repo_root=root, home=home)
            populated = trusted.hooks if capability == "hooks" else trusted.mcp_servers
            if len(populated) != 1:
                fail(f"trusted repository {capability} was not accepted: {trusted!r}")
            _write(source, "")
            try:
                load_extensions(repo_root=root, home=home)
            except ConfigError as exc:
                if str(project) not in str(exc):
                    fail(f"untrusted {capability} refusal lost its source: {exc!r}")
            else:
                fail(f"repository {capability} was accepted without owner trust")


@check("extensions.errors_are_atomic")
def errors_are_atomic() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        targets = (
            "trust_from_config",
            "CapabilityCeiling.from_config",
            "hooks_from_config",
            "mcp_servers_from_config",
        )
        for target in targets:
            source = root / f"{target.replace('.', '-')}.toml"
            error = ConfigError(f"{source}: deliberately malformed")
            if target == "CapabilityCeiling.from_config":
                patcher = mock.patch.object(
                    extensions_module.CapabilityCeiling,
                    "from_config",
                    side_effect=error,
                )
            else:
                patcher = mock.patch.object(
                    extensions_module,
                    target,
                    side_effect=error,
                )
            with patcher:
                try:
                    load_extensions(repo_root=root, home=root / "empty-home")
                except ConfigError as exc:
                    if exc is not error or str(source) not in str(exc):
                        fail(f"{target} ConfigError was altered: {exc!r}")
                else:
                    fail(f"{target} ConfigError was swallowed")


@check("extensions.empty_and_runner")
def empty_and_runner() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        loaded = load_extensions(repo_root=root, home=root / "empty-home")
        if (
            loaded.hooks != ()
            or loaded.mcp_servers != ()
            or loaded.agents
            or loaded.skills
            or loaded.plugins
            or loaded.withheld != ()
            or loaded.ceiling != CapabilityCeiling()
            or loaded.trust != TrustList()
            or loaded.hook_runner(cwd=root) is not None
        ):
            fail(f"empty configuration was not exactly empty: {loaded!r}")
        try:
            loaded.hooks = ()  # type: ignore[misc]
        except FrozenInstanceError:
            pass
        else:
            fail("Extensions is not frozen")

        configured = _hook_extensions(
            root,
            [{"on": ["RunFinished"], "command": [sys.executable, "-c", "pass"]}],
        )
        runner = configured.hook_runner(cwd=root)
        if not isinstance(runner, HookRunner):
            fail("non-empty hooks did not build a HookRunner")


@check("extensions.run_task_default")
def run_task_default() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        captured: dict[str, object] = {}
        real_agent = ApiAgent

        def construct(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            captured.update(kwargs)
            return real_agent(*args, **kwargs)

        provider = FakeModelProvider(
            [ModelResponse(Message(Role.ASSISTANT, "done"), Usage(4, 2))]
        )
        with mock.patch.object(runner_module, "ApiAgent", side_effect=construct):
            result = run_task(
                provider,
                PermissionPolicy(root),
                "frozen",
                model="frozen-model",
                extensions=None,
            )
        if result.stopped_reason != "final_response" or result.turns_used != 1:
            fail(f"default run_task did not complete one turn: {result!r}")
        if captured.get("events", object()) is not None:
            fail("runner-less run_task attached an event sink")


@check("extensions.run_task_hooks")
def run_task_hooks() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "existing.txt").write_text("contents", encoding="utf-8")
        observed = root / "events.jsonl"
        guard_log = root / "guard.jsonl"
        guard = root / "guard.py"
        observer = root / "observer.py"
        _write(
            guard,
            "import json,sys\n"
            "payload=json.load(sys.stdin)\n"
            "with open(sys.argv[1], 'a', encoding='utf-8') as out:\n"
            " out.write(json.dumps(payload)+'\\n')\n"
            "print('deny: no')\n",
        )
        _write(
            observer,
            "import json,sys\n"
            "payload=json.load(sys.stdin)\n"
            "with open(sys.argv[1], 'a', encoding='utf-8') as out:\n"
            " out.write(json.dumps(payload)+'\\n')\n",
        )
        extensions = _hook_extensions(
            root,
            [
                {
                    "on": ["PreToolUse"],
                    "command": [sys.executable, str(guard), str(guard_log)],
                    "blocking": True,
                },
                {
                    "on": ["RunStarted", "ToolCallFinished"],
                    "command": [sys.executable, str(observer), str(observed)],
                },
            ],
        )
        provider = FakeModelProvider(
            [
                ModelResponse(
                    Message(
                        Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="read",
                                name="read_file",
                                arguments={"path": "existing.txt"},
                            )
                        ],
                    )
                ),
                ModelResponse(Message(Role.ASSISTANT, "done")),
            ]
        )
        with mock.patch(
            "symphonai_api.tools.filesystem.ReadFileTool._execute",
            side_effect=AssertionError("blocked tool executed"),
        ):
            result = run_task(
                provider,
                PermissionPolicy(root),
                "read",
                extensions=extensions,
            )
        denied = next(
            message.tool_result
            for message in result.messages
            if message.tool_result is not None
        )
        if denied.ok or denied.error != "no":
            fail(f"blocking hook did not veto the real tool: {denied!r}")
        guard_payloads = [json.loads(line) for line in guard_log.read_text().splitlines()]
        if [item.get("tool_name") for item in guard_payloads] != ["read_file"]:
            fail(f"blocking hook saw the wrong calls: {guard_payloads!r}")
        event_names = [
            json.loads(line).get("type") for line in observed.read_text().splitlines()
        ]
        if "RunStarted" not in event_names or "ToolCallFinished" not in event_names:
            fail(f"observational hook did not receive run events: {event_names!r}")


@check("extensions.ceiling_composes")
def ceiling_composes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        loaded = load_extensions(
            repo_root=root,
            home=root / "home",
            session={"agents": {"ceiling": {"shell_enabled": False}}},
        )
        denied_dir = root / "denied"
        allowed_dir = root / "allowed"
        _write(
            denied_dir / "worker.toml",
            'prompt = "work"\n[model]\nprovider = "fake"\n[policy]\nshell_enabled = true\n',
        )
        try:
            load_agent_directory(
                denied_dir,
                repo_root=root,
                ceiling=loaded.ceiling,
            )
        except AgentFileError as exc:
            if "worker.toml" not in str(exc) or "shell_enabled" not in str(exc):
                fail(f"ceiling refusal lost agent context: {exc!r}")
        else:
            fail("configured ceiling accepted an over-reaching agent")
        _write(
            allowed_dir / "worker.toml",
            'prompt = "work"\n[model]\nprovider = "fake"\n',
        )
        accepted = load_agent_directory(
            allowed_dir,
            repo_root=root,
            ceiling=loaded.ceiling,
        )
        if tuple(accepted) != ("worker",):
            fail(f"configured ceiling rejected a conforming agent: {accepted!r}")


@check("extensions.leader_hooks")
def leader_hooks() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "existing.txt").write_text("contents", encoding="utf-8")
        log = root / "guard.jsonl"
        guard = root / "guard.py"
        _write(
            guard,
            "import json,sys\n"
            "payload=json.load(sys.stdin)\n"
            "with open(sys.argv[1], 'a', encoding='utf-8') as out:\n"
            " out.write(json.dumps(payload)+'\\n')\n"
            "if payload.get('tool_name') == 'read_file': print('deny: child guarded')\n",
        )
        extensions = _hook_extensions(
            root,
            [
                {
                    "on": ["PreToolUse"],
                    "command": [sys.executable, str(guard), str(log)],
                    "blocking": True,
                }
            ],
        )
        leader_provider, child_provider = _leader_providers(with_child_tool=True)
        leader = Leader(
            LeaderConfig(
                leader_provider=leader_provider,
                subagent_provider=child_provider,
                repo_root=str(root),
                extensions=extensions,
            )
        )
        leader.run("delegate")
        record = leader.subagents.get("worker")
        if record is None:
            fail("leader did not dispatch the child")
        denied = next(
            message.tool_result
            for message in record.messages
            if message.tool_result is not None
        )
        if denied.ok or denied.error != "child guarded":
            fail(f"subagent did not inherit the hook veto: {denied!r}")
        names = [
            json.loads(line).get("tool_name") for line in log.read_text().splitlines()
        ]
        if names != ["dispatch_subagent", "read_file"]:
            fail(f"shared runner did not see leader then child tools: {names!r}")
        if leader._hook_runner is not leader._dispatch_tool._hooks:
            fail("leader and subagent did not share one HookRunner")

        observer = root / "observer.py"
        observed = root / "observed.jsonl"
        _write(
            observer,
            "import json,sys\n"
            "payload=json.load(sys.stdin)\n"
            "with open(sys.argv[1], 'a', encoding='utf-8') as out:\n"
            " out.write(json.dumps(payload)+'\\n')\n",
        )
        observing_extensions = _hook_extensions(
            root,
            [
                {
                    "on": ["RunStarted"],
                    "command": [sys.executable, str(observer), str(observed)],
                }
            ],
        )
        event_types: list[list[type]] = []
        for configured in (None, observing_extensions):
            sink = CollectingSink()
            leader_provider, child_provider = _leader_providers(with_child_tool=False)
            Leader(
                LeaderConfig(
                    leader_provider=leader_provider,
                    subagent_provider=child_provider,
                    repo_root=str(root),
                    events=sink,
                    extensions=configured,
                )
            ).run("delegate")
            event_types.append([type(event) for event in sink.events])
        if not event_types[0]:
            fail("leader silenced the caller's event sink")
        if event_types[0] != event_types[1]:
            fail("fanning hooks into the leader changed the caller's event types")
        if not observed.exists() or not any(
            json.loads(line).get("type") == "RunStarted"
            for line in observed.read_text().splitlines()
        ):
            fail("leader events did not reach the observational hook")


@check("extensions.leader_default")
def leader_default_and_imports() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        leader = Leader(
            LeaderConfig(
                leader_provider=FakeModelProvider(
                    [
                        ModelResponse(
                            Message(Role.ASSISTANT, "leader answer"),
                            Usage(3, 1),
                        )
                    ]
                ),
                subagent_provider=FakeModelProvider(),
                repo_root=str(root),
            )
        )
        result = leader.run("frozen leader")
        if result.final_answer != "leader answer" or result.stopped_reason != "final_response":
            fail(f"extension-free leader did not finish: {result!r}")
        if leader._hook_runner is not None or leader._event_sink._events is not None:
            fail("extension-free leader constructed a runner or attached a sink")



@check("extensions.discovery")
def extension_discovery() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "repo"
        home = base / "home"
        root.mkdir()
        user_skills = home / ".symphonai" / "skills"
        project_skills = root / ".symphonai" / "skills"
        _write(
            user_skills / "personal.md",
            "+++\n"
            'name = "personal"\n'
            'description = "Personal skill"\n'
            'when_to_use = "When requested"\n'
            "+++\nbody\n",
        )
        _write(project_skills / "offered.md", "untrusted content is not parsed")
        with mock.patch.object(
            extensions_module,
            "discover",
            wraps=extensions_module.discover,
        ) as discover_spy:
            loaded = load_extensions(repo_root=root, home=home)
        if tuple(loaded.skills) != ("personal",):
            fail(f"Extensions did not carry discovered user skills: {loaded!r}")
        expected_withheld = (
            Offered(Scope.PROJECT, project_skills, ("offered",)),
        )
        if loaded.withheld != expected_withheld:
            fail(f"Extensions did not carry the repository offer: {loaded!r}")
        if "offered" in loaded.skills:
            fail("Extensions exposed an untrusted repository skill")
        if discover_spy.call_count != 1:
            fail("load_extensions did not discover exactly once")
        arguments = discover_spy.call_args.kwargs
        if (
            arguments.get("home") != home
            or arguments.get("trust") != loaded.trust
            or arguments.get("ceiling") != loaded.ceiling
        ):
            fail(f"load_extensions did not forward resolved discovery inputs: {arguments!r}")
