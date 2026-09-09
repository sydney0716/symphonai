"""Checks for owning MCP server lifetimes and handing tools to a run."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from unittest import mock

import symphonai_api.mcp as mcp_module
import symphonai_api.mcp_pool as mcp_pool_module
import symphonai_api.runner as runner_module
from symphonai_api.mcp import McpClient, McpError, McpServerSpec
from symphonai_api.mcp_pool import McpPool
from symphonai_api.models import Message, ModelResponse, Role, ToolCall, ToolResult, Usage
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.runner import run_task, standard_tool_registry
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from scripts.checks.agent_spec import _forbidden_imports
from scripts.checks.harness import check, fail
from scripts.checks.mcp import (
    _FAKE_SERVER,
    _clean_pid,
    _pid_alive,
    _spec,
    _wait_until,
)


_PRE_19D_COMMIT = "3c09e8404accc6b50a9fffcd94cca4a807614aaa"
_MCP_SHA256 = "9b445f094387b2b226dc7c847aa400fd508be0c4408cedbd9d457123f04a32fb"
_FROZEN_RUN_TASK = (
    b'["final_response",1,[["user","frozen",[],null],'
    b'["assistant","done",[],null]],[["frozen-model",4,2,1]]]'
)


def _write_fake(directory: Path) -> Path:
    source = _FAKE_SERVER.replace(
        'if mode == "hang_start":\n',
        'if mode == "tree_normal":\n'
        '    spawn_descendant(auxiliary)\n\n'
        'if mode == "hang_start":\n',
    ).replace(
        '    elif method == "tools/list":\n'
        '        send(request_id, {"tools": [{\n',
        '    elif method == "tools/list":\n'
        '        if mode == "fail_list":\n'
        '            print(json.dumps({"jsonrpc": "2.0", "id": request_id, '
        '"error": {"code": -1, "message": "list failed"}}), flush=True)\n'
        '            continue\n'
        '        send(request_id, {"tools": [{\n',
    )
    if source == _FAKE_SERVER:
        raise AssertionError("fake MCP server template no longer matches")
    path = directory / "fake_mcp_pool.py"
    path.write_text(source, encoding="utf-8")
    return path


def _processes(pool: McpPool) -> list[subprocess.Popen[str]]:
    return [
        client._process
        for client in pool._clients
        if client._process is not None
    ]


def _assert_processes_gone(
    processes: list[subprocess.Popen[str]],
    label: str,
) -> None:
    if any(process.poll() is None for process in processes):
        fail(f"{label} left a direct MCP process alive")


class _NamedTool(LocalTool):
    def __init__(self, name: str, *, content: str = "stub result") -> None:
        self._name = name
        self._content = content

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A test-only tool."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(ToolEffect.READ_ONLY, True, ())

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel=None,  # noqa: ANN001
    ) -> ToolResult:
        return ToolResult(tool_call.id, True, self._content)


def _run_snapshot(result) -> bytes:  # noqa: ANN001
    messages = []
    for message in result.messages:
        tool_result = message.tool_result
        messages.append(
            [
                message.role.value,
                message.text,
                [[call.id, call.name] for call in message.tool_calls],
                None
                if tool_result is None
                else [
                    tool_result.tool_call_id,
                    tool_result.ok,
                    tool_result.content,
                    tool_result.error,
                ],
            ]
        )
    usage = [
        [model, totals.input_tokens, totals.output_tokens, totals.calls]
        for model, totals in sorted(result.usage_by_model.items())
    ]
    return json.dumps(
        [result.stopped_reason, result.turns_used, messages, usage],
        separators=(",", ":"),
    ).encode("utf-8")


@check("mcp_pool.starts_and_orders")
def starts_and_orders() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        pool = McpPool(
            (
                _spec(script, name="docs", tool="search"),
                _spec(script, name="code", tool="lookup"),
            ),
            cwd=directory,
        )
        try:
            tools = pool.start()
            expected = ("mcp__docs__search", "mcp__code__lookup")
            if tuple(tools) != expected or tuple(pool.tools) != expected:
                fail(f"pool tool order was wrong: {tuple(tools)!r}")
            if len(_processes(pool)) != 2:
                fail("pool did not start both enabled servers")
        finally:
            pool.close()


@check("mcp_pool.disabled_not_started")
def disabled_not_started() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        disabled = McpServerSpec("off", ("must-not-run",), enabled=False)
        with mock.patch.object(
            mcp_module.subprocess,
            "Popen",
            side_effect=AssertionError("disabled server spawned"),
        ) as popen:
            with McpPool((disabled,), cwd=directory) as pool:
                if pool.tools:
                    fail(f"disabled server contributed tools: {pool.tools!r}")
        if popen.called:
            fail("disabled server reached Popen")


@check("mcp_pool.start_is_atomic")
def start_is_atomic() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        descendant_file = directory / "descendant.pid"
        healthy = _spec(
            script,
            name="healthy",
            mode="tree_normal",
            auxiliary=descendant_file,
        )
        broken = McpServerSpec(
            "broken",
            (str(directory / "missing-server"),),
            enabled=True,
        )
        pool = McpPool((healthy, broken), cwd=directory)
        captured: list[subprocess.Popen[str]] = []
        real_popen = subprocess.Popen

        def capture(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            process = real_popen(*args, **kwargs)
            captured.append(process)
            return process

        descendant = None
        try:
            with mock.patch.object(
                mcp_module.subprocess,
                "Popen",
                side_effect=capture,
            ):
                try:
                    pool.start()
                except McpError as exc:
                    if "broken" not in str(exc):
                        fail(f"failed-start error omitted server B: {exc!r}")
                else:
                    fail("failed server returned a partial MCP tool set")
            if len(captured) != 1 or captured[0].poll() is None:
                fail(f"failed start left server A alive: {captured!r}")
            if not descendant_file.exists():
                fail("healthy server A did not record its descendant")
            descendant = int(descendant_file.read_text(encoding="utf-8"))
            if not _wait_until(lambda: not _pid_alive(descendant), 1.0):
                fail(f"failed start left server A descendant {descendant} alive")
        finally:
            pool.close()
            if descendant is not None:
                _clean_pid(descendant)

        for mode, tool, detail in (
            ("fail_list", "search", "tools/list"),
            ("normal", "bad.tool", "unusable name"),
        ):
            pool = McpPool(
                (
                    _spec(script, name="healthy"),
                    _spec(script, name="broken", mode=mode, tool=tool),
                ),
                cwd=directory,
            )
            captured = []

            def capture_case(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                process = real_popen(*args, **kwargs)
                captured.append(process)
                return process

            try:
                with mock.patch.object(
                    mcp_module.subprocess,
                    "Popen",
                    side_effect=capture_case,
                ):
                    try:
                        pool.start()
                    except McpError as exc:
                        message = str(exc)
                        if "broken" not in message or detail not in message:
                            fail(f"server B failure lost context: {message!r}")
                    else:
                        fail(f"{detail} returned a partial MCP tool set")
                if len(captured) != 2:
                    fail(f"{detail} did not start both expected servers")
                _assert_processes_gone(captured, detail)
            finally:
                pool.close()


@check("mcp_pool.cross_server_collisions")
def cross_server_collisions() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        reservations: list[set[str]] = []

        def construct(spec, *, cwd, reserved_names=()):  # noqa: ANN001, ANN202
            reservations.append(set(reserved_names))
            return McpClient(spec, cwd=cwd, reserved_names=reserved_names)

        pool = McpPool(
            (
                _spec(script, name="duplicate"),
                _spec(script, name="duplicate"),
            ),
            cwd=directory,
            reserved_names={"caller_tool"},
        )
        try:
            with mock.patch.object(
                mcp_pool_module,
                "McpClient",
                side_effect=construct,
            ):
                try:
                    pool.start()
                except McpError as exc:
                    message = str(exc)
                    required = ("duplicate", "indices", "0", "1", "mcp__duplicate__search")
                    if not all(fragment in message for fragment in required):
                        fail(f"cross-server collision omitted context: {message!r}")
                else:
                    fail("two servers provided the same namespaced tool")
            if len(reservations) != 2:
                fail(f"pool did not construct both clients: {reservations!r}")
            if "mcp__duplicate__search" not in reservations[1]:
                fail(f"first server names were not reserved for the second: {reservations!r}")
        finally:
            pool.close()


@check("mcp_pool.standard_registry_collision")
def standard_registry_collision() -> None:
    live_registry = standard_tool_registry()
    seen_reserved: list[set[str]] = []

    class ReservedClient:
        def __init__(self, spec, *, cwd, reserved_names=()):  # noqa: ANN001
            self.spec = spec
            self.reserved = set(reserved_names)
            seen_reserved.append(self.reserved)

        def start(self) -> None:
            pass

        def list_tools(self) -> tuple[LocalTool, ...]:
            tool = _NamedTool("read_file")
            if tool.name in self.reserved:
                raise McpError(
                    f"MCP server {self.spec.name!r} tool collides with "
                    f"reserved name {tool.name!r}"
                )
            return (tool,)

        def close(self) -> None:
            pass

    spec = McpServerSpec("docs", ("fake",), enabled=True)
    pool = McpPool((spec,), cwd=Path.cwd(), reserved_names=set(live_registry))
    with mock.patch.object(mcp_pool_module, "McpClient", ReservedClient):
        try:
            pool.start()
        except McpError as exc:
            if "docs" not in str(exc) or "read_file" not in str(exc):
                fail(f"standard collision omitted context: {exc!r}")
        else:
            fail("live standard registry key was not reserved")
    if seen_reserved != [set(live_registry)]:
        fail(f"pool did not inject the live standard keys: {seen_reserved!r}")


@check("mcp_pool.close_and_context")
def close_and_context() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        pool = McpPool(
            tuple(_spec(script, name=f"server_{index}") for index in range(3)),
            cwd=directory,
        )
        pool.start()
        clients = list(pool._clients)
        processes = _processes(pool)
        target = clients[1]
        real_close = target.close
        marker = RuntimeError("deliberate close failure")

        def close_and_raise() -> None:
            real_close()
            raise marker

        try:
            with mock.patch.object(target, "close", side_effect=close_and_raise):
                try:
                    pool.close()
                except RuntimeError as exc:
                    if exc is not marker:
                        fail(f"pool altered the first close failure: {exc!r}")
                else:
                    fail("pool swallowed a client close failure")
            _assert_processes_gone(processes, "total close")
            pool.close()
        finally:
            for client in clients:
                try:
                    client.close()
                except RuntimeError:
                    pass

        context_pool = McpPool((_spec(script, name="context"),), cwd=directory)
        body_error = LookupError("body failed")
        process = None
        try:
            try:
                with context_pool:
                    process = _processes(context_pool)[0]
                    raise body_error
            except LookupError as exc:
                if exc is not body_error:
                    fail(f"context manager altered the body exception: {exc!r}")
            else:
                fail("context manager swallowed the body exception")
            if process is None or process.poll() is None:
                fail("context manager body failure left its server alive")
        finally:
            context_pool.close()


@check("mcp_pool.run_task_tools")
def run_task_tools() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        script = _write_fake(directory)
        call_log = directory / "calls.log"
        standard = standard_tool_registry()
        captured: dict[str, object] = {}
        real_agent = runner_module.ApiAgent

        def construct(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            captured.update(kwargs)
            return real_agent(*args, **kwargs)

        with McpPool(
            (_spec(script, auxiliary=call_log),),
            cwd=directory,
            reserved_names=set(standard),
        ) as pool:
            provider = FakeModelProvider(
                [
                    ModelResponse(
                        Message(
                            Role.ASSISTANT,
                            tool_calls=[
                                ToolCall("mcp-call", "mcp__docs__search", {"query": "x"})
                            ],
                        )
                    ),
                    ModelResponse(Message(Role.ASSISTANT, "done")),
                ]
            )
            with mock.patch.object(runner_module, "ApiAgent", side_effect=construct):
                result = run_task(
                    provider,
                    PermissionPolicy(directory),
                    "call MCP",
                    mcp_tools=pool.tools,
                )
        tool_messages = [message for message in result.messages if message.role is Role.TOOL]
        if len(tool_messages) != 1 or not tool_messages[0].tool_result.ok:
            fail(f"agent did not call the MCP tool: {result.messages!r}")
        if call_log.read_text(encoding="utf-8").splitlines() != ["call"]:
            fail("MCP call did not reach the fake server exactly once")
        tools = captured.get("tools")
        schemas = captured.get("tool_schemas")
        if not isinstance(tools, dict) or "mcp__docs__search" not in tools:
            fail(f"run_task did not merge MCP tools: {tools!r}")
        if not isinstance(schemas, list) or not any(
            schema.get("name") == "mcp__docs__search" for schema in schemas
        ):
            fail(f"MCP tool was absent from model schemas: {schemas!r}")

        collision = _NamedTool("read_file")
        standard = standard_tool_registry()
        original = standard["read_file"]
        with mock.patch.object(
            runner_module,
            "standard_tool_registry",
            return_value=standard,
        ):
            try:
                run_task(
                    FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
                    PermissionPolicy(directory),
                    "collision",
                    mcp_tools={"read_file": collision},
                )
            except ValueError as exc:
                if "read_file" not in str(exc):
                    fail(f"run_task collision omitted the tool: {exc!r}")
            else:
                fail("run_task overwrote a standard tool with an MCP tool")
        if standard["read_file"] is not original:
            fail("run_task changed the standard binding before refusing collision")


@check("mcp_pool.default_and_imports")
def default_and_imports() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result = run_task(
            FakeModelProvider(
                [ModelResponse(Message(Role.ASSISTANT, "done"), Usage(4, 2))]
            ),
            PermissionPolicy(root),
            "frozen",
            model="frozen-model",
            mcp_tools=None,
        )
    actual = _run_snapshot(result)
    if actual != _FROZEN_RUN_TASK:
        fail(
            f"mcp_tools=None changed run_task from {_PRE_19D_COMMIT}: "
            f"expected={_FROZEN_RUN_TASK!r}, actual={actual!r}"
        )

    root = Path(__file__).resolve().parents[2]
    pool_path = root / "symphonai_api/mcp_pool.py"
    source = pool_path.read_text(encoding="utf-8")
    forbidden = _forbidden_imports(source)
    expanded = {
        "agent_loop",
        "leader",
        "runner",
        "agent_run",
        "agent_spec",
        "agent_file",
        "child_context",
        "extensions",
        "provider_catalog",
        "providers",
    }
    for node in ast.walk(ast.parse(source)):
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
        fail(f"mcp_pool.py imports forbidden modules: {sorted(set(forbidden))!r}")

    mcp_source = (root / "symphonai_api/mcp.py").read_bytes()
    mcp_digest = hashlib.sha256(mcp_source).hexdigest()
    if mcp_digest != _MCP_SHA256:
        fail(f"mcp.py changed from its pre-19d SHA-256: {mcp_digest}")
