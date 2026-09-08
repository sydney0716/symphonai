"""Checks for the subprocess-backed MCP client."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock

import symphonai_api.mcp as mcp_module
from symphonai_api.config import ConfigError, Scope, load_config
from symphonai_api.events import CollectingSink, PermissionDenied, PermissionRequested
from symphonai_api.mcp import (
    McpClient,
    McpError,
    McpServerSpec,
    McpTool,
    mcp_servers_from_config,
)
from symphonai_api.models import ToolCall
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.runner import standard_tool_registry
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect
from symphonai_api.trust import RepositoryTrust, TrustList
from scripts.checks.agent_spec import _forbidden_imports
from scripts.checks.harness import check, fail


_FAKE_SERVER = r'''import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

mode = sys.argv[1]
tool_name = sys.argv[2]
auxiliary = None if sys.argv[3] == "-" else Path(sys.argv[3])

def send(request_id, result):
    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)

def spawn_descendant(path):
    code = (
        "import os, signal, sys, threading;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "open(sys.argv[1], 'w').write(str(os.getpid()));"
        "threading.Event().wait()"
    )
    subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    while not path.exists():
        threading.Event().wait(0.01)

if mode == "hang_start":
    spawn_descendant(auxiliary)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    threading.Event().wait()

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if "id" not in message:
        continue
    request_id = message["id"]
    if method == "initialize":
        send(request_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1"},
        })
    elif method == "tools/list":
        send(request_id, {"tools": [{
            "name": tool_name,
            "description": "Search the fake corpus.",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }]})
    elif method == "tools/call":
        if mode == "hang_call":
            spawn_descendant(auxiliary)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            threading.Event().wait()
        if mode == "malformed":
            print("{not-json", flush=True)
            continue
        if mode == "unknown_id":
            send(request_id + 100, {"content": []})
            continue
        if mode == "exit_call":
            raise SystemExit(7)
        if auxiliary is not None:
            with auxiliary.open("a", encoding="utf-8") as stream:
                stream.write("call\n")
        arguments = message.get("params", {}).get("arguments", {})
        send(request_id, {"content": [{
            "type": "text",
            "text": "called:" + json.dumps(arguments, sort_keys=True),
        }]})
'''


def _write_fake(directory: Path) -> Path:
    path = directory / "fake_mcp.py"
    path.write_text(_FAKE_SERVER, encoding="utf-8")
    return path


def _spec(
    script: Path,
    *,
    name: str = "docs",
    mode: str = "normal",
    tool: str = "search",
    auxiliary: Path | None = None,
    startup_timeout: float = 1.0,
    call_timeout: float = 1.0,
) -> McpServerSpec:
    return McpServerSpec(
        name=name,
        command=(
            sys.executable,
            str(script),
            mode,
            tool,
            "-" if auxiliary is None else str(auxiliary),
        ),
        enabled=True,
        startup_timeout_seconds=startup_timeout,
        call_timeout_seconds=call_timeout,
        source=script,
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _server_table(*, enabled: bool | None, name: str = "docs") -> str:
    enabled_line = "" if enabled is None else f"enabled = {str(enabled).lower()}\n"
    return (
        "[[mcp.servers]]\n"
        f'name = "{name}"\n'
        'command = ["fake-server", "--stdio"]\n'
        f"{enabled_line}"
    )


def _scope_config(
    temporary: str,
    scope: Scope,
    *,
    enabled: bool,
):
    root = Path(temporary) / "repo"
    home = Path(temporary) / "home"
    content = _server_table(enabled=enabled)
    session = None
    source = None
    if scope is Scope.USER:
        source = home / ".symphonai" / "config.toml"
        _write(source, content)
    elif scope is Scope.PROJECT:
        source = root / ".symphonai" / "config.toml"
        _write(source, content)
    elif scope is Scope.PRIVATE:
        source = root / ".symphonai" / "config.local.toml"
        _write(source, content)
    else:
        session = {
            "mcp": {
                "servers": [
                    {
                        "name": "docs",
                        "command": ["fake-server", "--stdio"],
                        "enabled": enabled,
                    }
                ]
            }
        }
    return load_config(repo_root=root, home=home, session=session), source


def _wait_until(predicate, timeout: float) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        threading.Event().wait(0.01)
    return predicate()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _clean_pid(pid: int) -> None:
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@check("mcp.config_parsing_and_disabled_default")
def config_parsing_and_disabled_default() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        home = Path(temporary) / "home"
        source = home / ".symphonai" / "config.toml"
        _write(source, _server_table(enabled=None))
        config = load_config(repo_root=root, home=home)
        specs = mcp_servers_from_config(config)
        if (
            len(specs) != 1
            or specs[0].source != source
            or specs[0].command != ("fake-server", "--stdio")
        ):
            fail(f"nested MCP config parsed incorrectly: {specs!r}")
        with mock.patch.object(
            mcp_module.subprocess,
            "Popen",
            side_effect=AssertionError("disabled config spawned a process"),
        ) as popen:
            try:
                McpClient(specs[0], cwd=root).start()
            except McpError as exc:
                if "docs" not in str(exc) or "disabled" not in str(exc):
                    fail(f"disabled-server refusal omitted its name: {exc!r}")
            else:
                fail("config without enabled=true started its MCP server")
            if popen.called:
                fail("config without enabled=true spawned a process")

        duplicate = _server_table(enabled=False) + _server_table(
            enabled=False,
            name="docs",
        )
        _write(source, duplicate)
        try:
            mcp_servers_from_config(load_config(repo_root=root, home=home))
        except ConfigError as exc:
            message = str(exc)
            if str(source) not in message or "indices 0 and 1" not in message:
                fail(f"duplicate error omitted source or indices: {message!r}")
        else:
            fail("duplicate MCP server name was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        home = Path(temporary) / "home"
        bad = root / ".symphonai" / "config.toml"
        _write(
            bad,
            '[[mcp_servers]]\nname = "docs"\ncommand = ["fake-server"]\n',
        )
        try:
            load_config(repo_root=root, home=home)
        except ConfigError as exc:
            if str(bad) not in str(exc) or "mcp_servers" not in str(exc):
                fail(f"unnested-key error omitted source or key: {exc!r}")
        else:
            fail("unnested mcp_servers key was accepted")


@check("mcp.owner_scope_controls_enablement")
def owner_scope_controls_enablement() -> None:
    for scope in Scope:
        with tempfile.TemporaryDirectory() as temporary:
            config, source = _scope_config(temporary, scope, enabled=True)
            if scope in (Scope.USER, Scope.SESSION):
                specs = mcp_servers_from_config(config)
                if len(specs) != 1 or not specs[0].enabled or specs[0].source != source:
                    fail(f"owner scope {scope.value} did not enable its server")
            else:
                try:
                    mcp_servers_from_config(config)
                except ConfigError as exc:
                    message = str(exc)
                    if str(source) not in message or "docs" not in message:
                        fail(f"{scope.value} refusal omitted source or server: {message!r}")
                else:
                    fail(f"repository scope {scope.value} enabled a subprocess")

                disabled_config, disabled_source = _scope_config(
                    str(Path(temporary) / "disabled"),
                    scope,
                    enabled=False,
                )
                specs = mcp_servers_from_config(disabled_config)
                if len(specs) != 1 or specs[0].enabled or specs[0].source != disabled_source:
                    fail(f"{scope.value} could not declare a disabled server")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        home = Path(temporary) / "home"
        _write(home / ".symphonai" / "config.toml", _server_table(enabled=True))
        project = root / ".symphonai" / "config.toml"
        _write(project, _server_table(enabled=True))
        config = load_config(repo_root=root, home=home)
        try:
            mcp_servers_from_config(config)
        except ConfigError as exc:
            if str(project) not in str(exc) or "docs" not in str(exc):
                fail(f"winning-project refusal omitted its source or server: {exc!r}")
        else:
            fail("project list bypassed owner scope through a lower user list")


@check("mcp.handshake_tools_and_namespaces")
def handshake_tools_and_namespaces() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        clients = (
            McpClient(_spec(script, name="docs"), cwd=directory),
            McpClient(_spec(script, name="code"), cwd=directory),
        )
        try:
            tools = []
            for client in clients:
                client.start()
                listed = client.list_tools()
                if len(listed) != 1:
                    fail(f"fake server returned wrong tool count: {listed!r}")
                tools.append(listed[0])
            if [tool.name for tool in tools] != [
                "mcp__docs__search",
                "mcp__code__search",
            ]:
                fail(f"MCP tool namespaces differed: {tools!r}")
            if len({tool.name for tool in tools}) != 2:
                fail("same server-side name collided across servers")
            if McpTool.__abstractmethods__:
                fail(f"McpTool left LocalTool members abstract: {McpTool.__abstractmethods__!r}")
            for tool in tools:
                if not isinstance(tool, LocalTool):
                    fail(f"adapted tool is not a LocalTool: {tool!r}")
                if tool.description != "Search the fake corpus.":
                    fail(f"tool description was not adapted: {tool.description!r}")
                if tool.parameters.get("type") != "object":
                    fail(f"tool input schema was not adapted: {tool.parameters!r}")
                metadata = tool.metadata({})
                if (
                    metadata.effect is not ToolEffect.DESTRUCTIVE
                    or metadata.paths is not None
                    or metadata.concurrency_safe
                ):
                    fail(f"opaque MCP metadata was unsafe: {metadata!r}")
        finally:
            for client in clients:
                client.close()


@check("mcp.reserved_names_are_injected")
def reserved_names_are_injected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        namespaced = "mcp__docs__search"
        live_registry = standard_tool_registry()
        reserved = set(live_registry)
        reserved.add(namespaced)
        collision = McpClient(
            _spec(script),
            cwd=directory,
            reserved_names=reserved,
        )
        try:
            collision.start()
            try:
                collision.list_tools()
            except McpError as exc:
                message = str(exc)
                if "docs" not in message or "search" not in message:
                    fail(f"collision error omitted server or tool: {message!r}")
            else:
                fail("injected reserved name did not refuse a collision")
        finally:
            collision.close()

        unrestricted = McpClient(_spec(script), cwd=directory)
        try:
            unrestricted.start()
            tools = unrestricted.list_tools()
            if len(tools) != 1 or tools[0].name != namespaced:
                fail("default empty reserved names refused a tool")
        finally:
            unrestricted.close()


@check("mcp.policy_modes_fail_closed")
def policy_modes_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        call_log = directory / "calls.log"
        client = McpClient(_spec(script, auxiliary=call_log), cwd=directory)
        try:
            client.start()
            tool = client.list_tools()[0]
            outcomes = {}
            for mode in ("auto", "prompt", "plan", "accept_edits"):
                callback = (
                    (lambda request: True)
                    if mode in ("prompt", "accept_edits")
                    else None
                )
                policy = PermissionPolicy(
                    repo_root=directory,
                    mode=mode,
                    approval_callback=callback,
                )
                outcomes[mode] = tool._execute(
                    ToolCall(id=f"call-{mode}", name=tool.name, arguments={"query": mode}),
                    policy,
                )
            if not outcomes["auto"].ok or not outcomes["prompt"].ok:
                fail(f"trusted MCP modes did not proceed: {outcomes!r}")
            if outcomes["plan"].ok or "opaque" not in (outcomes["plan"].error or ""):
                fail(f"plan mode did not refuse opaquely: {outcomes['plan']!r}")
            if not outcomes["accept_edits"].ok:
                fail(f"accept_edits did not use server trust: {outcomes['accept_edits']!r}")

            denied = tool._execute(
                ToolCall(id="call-denied", name=tool.name, arguments={"query": "no"}),
                PermissionPolicy(
                    repo_root=directory,
                    mode="prompt",
                    approval_callback=lambda request: False,
                ),
            )
            if denied.ok or "denied" not in (denied.error or ""):
                fail(f"negative prompt approval was not a failed result: {denied!r}")
            calls = call_log.read_text(encoding="utf-8").splitlines()
            if calls != ["call", "call", "call"]:
                fail(f"refused MCP calls reached the server: {calls!r}")
        finally:
            client.close()


@check("mcp.refusals_are_observed")
def refusals_are_observed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        call_log = directory / "calls.log"
        client = McpClient(_spec(script, auxiliary=call_log), cwd=directory)

        def raises(request):  # noqa: ANN001
            raise RuntimeError("approval broke")

        cases = (
            ("auto", None, True, 0, 0),
            ("prompt", lambda request: True, True, 1, 0),
            ("prompt", lambda request: False, False, 1, 1),
            ("plan", None, False, 0, 1),
            ("accept_edits", lambda request: False, False, 1, 1),
            ("accept_edits", lambda request: True, True, 1, 0),
            ("prompt", raises, False, 1, 1),
            ("accept_edits", lambda request: "invalid", False, 1, 1),
        )
        try:
            client.start()
            tool = client.list_tools()[0]
            for index, (mode, callback, expected_ok, requests, denials) in enumerate(
                cases
            ):
                sink = CollectingSink()
                policy = PermissionPolicy(
                    repo_root=directory,
                    mode=mode,
                    approval_callback=callback,
                )
                policy.attach_event_sink(
                    sink,
                    agent_id="agent-mcp",
                    run_id=f"run-{index}",
                )
                result = tool._execute(
                    ToolCall(
                        id=f"observed-{index}",
                        name=tool.name,
                        arguments={"query": mode},
                    ),
                    policy,
                )
                requested = sink.of_type(PermissionRequested)
                denied = sink.of_type(PermissionDenied)
                if result.ok is not expected_ok:
                    fail(f"MCP permission result differed for case {index}: {result!r}")
                if len(requested) != requests or len(denied) != denials:
                    fail(
                        f"MCP permission event counts differed for case {index}: "
                        f"{sink.events!r}"
                    )
                if denied and denied[0].tool_name != tool.name:
                    fail(f"MCP denial named the wrong tool: {denied[0]!r}")
                if requested and requested[0].tool_name != tool.name:
                    fail(f"MCP request named the wrong tool: {requested[0]!r}")
                if requested and denied and sink.events != [requested[0], denied[0]]:
                    fail(f"MCP request/denial order differed: {sink.events!r}")

            calls = call_log.read_text(encoding="utf-8").splitlines()
            if calls != ["call", "call", "call"]:
                fail(f"refused MCP calls reached the fake server: {calls!r}")
        finally:
            client.close()

    path = Path(__file__).resolve().parents[2] / "symphonai_api/mcp.py"
    source = path.read_text(encoding="utf-8")
    private_policy_references = re.findall(r"policy\._[A-Za-z]", source)
    if private_policy_references:
        fail(f"mcp.py references private policy attributes: {private_policy_references!r}")


@check("mcp.startup_timeout_kills_tree")
def startup_timeout_kills_tree() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        descendant_file = directory / "descendant.pid"
        client = McpClient(
            _spec(
                script,
                mode="hang_start",
                auxiliary=descendant_file,
                startup_timeout=0.15,
            ),
            cwd=directory,
        )
        real_popen = subprocess.Popen
        spawned = []

        def capture(*args, **kwargs):  # noqa: ANN002, ANN003
            proc = real_popen(*args, **kwargs)
            spawned.append(proc)
            return proc

        started = time.monotonic()
        try:
            with mock.patch.object(mcp_module.subprocess, "Popen", side_effect=capture):
                try:
                    client.start()
                except McpError as exc:
                    message = str(exc)
                    if "docs" not in message or "timed out" not in message:
                        fail(f"startup timeout omitted server or timeout: {message!r}")
                else:
                    fail("hung MCP startup completed")
            if time.monotonic() - started >= 1.0:
                fail("MCP startup timeout was not bounded")
            if len(spawned) != 1 or spawned[0].poll() is None:
                fail(f"startup timeout left direct process alive: {spawned!r}")
            if not descendant_file.exists():
                fail("hung startup did not record its descendant")
            descendant = int(descendant_file.read_text(encoding="utf-8"))
            try:
                if not _wait_until(lambda: not _pid_alive(descendant), 1.0):
                    fail(f"startup timeout left descendant {descendant} alive")
            finally:
                _clean_pid(descendant)
        finally:
            client.close()


@check("mcp.call_timeout_kills_tree")
def call_timeout_kills_tree() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        descendant_file = directory / "descendant.pid"
        client = McpClient(
            _spec(
                script,
                mode="hang_call",
                auxiliary=descendant_file,
                call_timeout=0.15,
            ),
            cwd=directory,
        )
        descendant = None
        try:
            client.start()
            proc = client._process
            tool = client.list_tools()[0]
            started = time.monotonic()
            result = tool._execute(
                ToolCall(id="timeout", name=tool.name, arguments={"query": "hang"}),
                PermissionPolicy(repo_root=directory),
            )
            if result.ok or "timed out" not in (result.error or ""):
                fail(f"call timeout was not a failed ToolResult: {result!r}")
            if time.monotonic() - started >= 1.0:
                fail("MCP tool call timeout was not bounded")
            if proc is None or proc.poll() is None:
                fail("call timeout left direct server process alive")
            if not descendant_file.exists():
                fail("hung call did not record its descendant")
            descendant = int(descendant_file.read_text(encoding="utf-8"))
            if not _wait_until(lambda: not _pid_alive(descendant), 1.0):
                fail(f"call timeout left descendant {descendant} alive")
        finally:
            client.close()
            if descendant is not None:
                _clean_pid(descendant)


@check("mcp.protocol_failures_are_bounded")
def protocol_failures_are_bounded() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        cases = (
            ("malformed", "malformed JSON-RPC"),
            ("unknown_id", "unknown response id"),
            ("exit_call", "exited"),
        )
        for mode, expected in cases:
            client = McpClient(_spec(script, mode=mode), cwd=directory)
            started = time.monotonic()
            try:
                client.start()
                tool = client.list_tools()[0]
                result = tool._execute(
                    ToolCall(id=mode, name=tool.name, arguments={"query": mode}),
                    PermissionPolicy(repo_root=directory),
                )
                if result.ok or expected not in (result.error or ""):
                    fail(f"{mode} did not become a descriptive failed result: {result!r}")
                if time.monotonic() - started >= 1.0:
                    fail(f"{mode} protocol failure was not bounded")
            finally:
                client.close()


@check("mcp.close_is_idempotent")
def close_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        client = McpClient(_spec(script), cwd=directory)
        client.start()
        proc = client._process
        client.close()
        client.close()
        if proc is None or proc.poll() is None:
            fail("idempotent close left the server process alive")


@check("mcp.import_boundary")
def import_boundary() -> None:
    path = Path(__file__).resolve().parents[2] / "symphonai_api/mcp.py"
    source = path.read_text(encoding="utf-8")
    forbidden = _forbidden_imports(source)
    expanded = {
        "agent_loop",
        "leader",
        "runner",
        "agent_run",
        "agent_spec",
        "agent_file",
        "child_context",
        "hooks",
        "provider_catalog",
        "providers",
    }
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module] if node.module is not None else []
        else:
            continue
        forbidden.extend(
            module
            for module in modules
            if set(module.split(".")) & expanded
        )
    if forbidden:
        fail(f"mcp.py imports forbidden runtime modules: {sorted(set(forbidden))!r}")


@check("mcp.trust_grants_a_repository")
def trust_grants_a_repository() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repo_root = base / "repo"
        home = base / "home"
        source = repo_root / ".symphonai" / "config.toml"
        _write(source, _server_table(enabled=True))
        config = load_config(repo_root=repo_root, home=home)
        exact_mcp = TrustList(
            (
                RepositoryTrust(
                    repo_root.resolve(),
                    frozenset(("mcp",)),
                    None,
                ),
            )
        )
        cases = (
            (repo_root, None),
            (repo_root, TrustList()),
            (
                repo_root,
                TrustList(
                    (
                        RepositoryTrust(
                            (base / "other").resolve(),
                            frozenset(("mcp",)),
                            None,
                        ),
                    )
                ),
            ),
            (
                repo_root,
                TrustList(
                    (
                        RepositoryTrust(
                            repo_root.resolve(),
                            frozenset(("hooks",)),
                            None,
                        ),
                    )
                ),
            ),
            (None, exact_mcp),
        )
        for candidate_root, trust in cases:
            try:
                mcp_servers_from_config(
                    config,
                    repo_root=candidate_root,
                    trust=trust,
                )
            except ConfigError as exc:
                message = str(exc)
                required = (
                    str(source),
                    "docs",
                    "~/.symphonai/config.toml",
                    "[[trust.repositories]]",
                )
                if not all(fragment in message for fragment in required):
                    fail(f"MCP trust refusal was incomplete: {message!r}")
            else:
                fail(
                    "project MCP server accepted insufficient trust: "
                    f"root={candidate_root!r}, trust={trust!r}"
                )

        actual = mcp_servers_from_config(
            config,
            repo_root=repo_root,
            trust=exact_mcp,
        )
        expected = (
            McpServerSpec(
                name="docs",
                command=("fake-server", "--stdio"),
                enabled=True,
                source=source,
            ),
        )
        if actual != expected:
            fail(f"trusted project MCP server parsed incorrectly: {actual!r}")
