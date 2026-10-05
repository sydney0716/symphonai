"""Checks for the stdio Language Server Protocol client."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time

from symphonai_api.config import ConfigError, load_config
from symphonai_api.lsp import LspClient, LspError, LspManager, LspServerSpec, lsp_servers_from_config
from symphonai_api.trust import RepositoryTrust, TrustList
from symphonai_api.leader import builtin_subagent_specs
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.runner import standard_tool_registry
from symphonai_api.tools.lsp import LspTool
from symphonai_api.models import ToolCall
from scripts.checks.harness import check, fail


def _spec(command: tuple[str, ...], *, suffixes=(".py",), enabled=True) -> LspServerSpec:
    return LspServerSpec("fake", command, "python", tuple(suffixes), enabled)


def _fake_server(root: Path) -> tuple[Path, Path]:
    log = root / "messages.jsonl"
    script = root / "fake_lsp.py"
    script.write_text('''import json, sys

def read():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\\r\\n", b"\\n"):
            break
        key, _, value = line.partition(b":")
        headers[key.lower()] = value.strip()
    return json.loads(sys.stdin.buffer.read(int(headers[b"content-length"])))

def send(value):
    body = json.dumps(value, separators=(",", ":")).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\\r\\n\\r\\n".encode() + body)
    sys.stdout.buffer.flush()

def log(value):
    with open(sys.argv[1], "a", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\\n")

first = read()
log(first)
send({"jsonrpc":"2.0", "id":"workspace", "method":"workspace/configuration", "params":{"items":[{}, {}]}})
send({"jsonrpc":"2.0", "id":"register", "method":"client/registerCapability", "params":{"registrations":[]}})
replies = [read(), read()]
for reply in replies: log(reply)
send({"jsonrpc":"2.0", "id":first["id"], "result":{"capabilities":{"textDocumentSync":1}}})
version = 0
while True:
    message = read()
    if message is None: break
    log(message)
    method = message.get("method")
    if method == "textDocument/didOpen":
        uri = message["params"]["textDocument"]["uri"]
        text = message["params"]["textDocument"]["text"]
        if "crash-diagnostics" in text: sys.exit(0)
        if "no-diagnostics" in text: continue
        if "empty-diagnostics" in text: diagnostics = []
        elif "twenty-five-errors" in text: diagnostics = [{"severity":1,"message":str(i),"range":{"start":{"line":i,"character":0}}} for i in range(25)]
        elif "diagnostic-error" in text: diagnostics = [{"severity":1,"message":"broken type","range":{"start":{"line":4,"character":2}}},{"severity":2,"message":"warning","range":{"start":{"line":0,"character":0}}}]
        elif "warning-only" in text: diagnostics = [{"severity":2,"message":"warning","range":{"start":{"line":0,"character":0}}}]
        else: diagnostics = [{"message":"opened"}]
        send({"jsonrpc":"2.0", "method":"textDocument/publishDiagnostics", "params":{"uri":uri,"diagnostics":diagnostics}})
    elif method == "textDocument/didChange":
        uri = message["params"]["textDocument"]["uri"]
        text = message["params"]["contentChanges"][0]["text"]
        if "crash-diagnostics" in text: sys.exit(0)
        if "no-diagnostics" in text: continue
        if "empty-diagnostics" in text: diagnostics = []
        elif "twenty-five-errors" in text: diagnostics = [{"severity":1,"message":str(i),"range":{"start":{"line":i,"character":0}}} for i in range(25)]
        elif "diagnostic-error" in text: diagnostics = [{"severity":1,"message":"broken type","range":{"start":{"line":4,"character":2}}},{"severity":2,"message":"warning","range":{"start":{"line":0,"character":0}}}]
        elif "warning-only" in text: diagnostics = [{"severity":2,"message":"warning","range":{"start":{"line":0,"character":0}}}]
        else: diagnostics = [{"message":"changed"}]
        send({"jsonrpc":"2.0", "method":"textDocument/publishDiagnostics", "params":{"uri":uri,"diagnostics":diagnostics}})
    elif method in ("textDocument/definition", "textDocument/references"):
        uri = message["params"]["textDocument"]["uri"]
        result = [{"uri":uri,"range":{"start":{"line":2,"character":4}}},
                  {"uri":"file:///usr/lib/fake.pyi","range":{"start":{"line":0,"character":0}}}]
        send({"jsonrpc":"2.0", "id":message["id"], "result":result})
    elif method == "textDocument/hover":
        send({"jsonrpc":"2.0", "id":message["id"], "result":{"contents":{"kind":"markdown","value":"```python\\nx: int\\n```"}}})
    elif method == "textDocument/documentSymbol":
        send({"jsonrpc":"2.0", "id":message["id"], "result":[{"name":"outer","kind":5,"range":{"start":{"line":0}},"children":[{"name":"inner","kind":12,"range":{"start":{"line":1}}}]}]})
    elif method == "workspace/symbol":
        send({"jsonrpc":"2.0", "id":message["id"], "result":[{"name":"workspace_item","kind":12,"location":{"uri":"file:///usr/lib/fake.pyi","range":{"start":{"line":0,"character":0}}}}]})
    elif method == "test/slow":
        continue
    elif method == "test/error":
        send({"jsonrpc":"2.0", "id":message["id"], "error":{"code":-32001,"message":"fixture error"}})
    elif method == "shutdown":
        send({"jsonrpc":"2.0", "id":message["id"], "result":None})
    elif "id" in message:
        send({"jsonrpc":"2.0", "id":message["id"], "result":{"ok":True}})
    elif method == "exit":
        break
''', encoding="utf-8")
    return script, log


@check("lsp.config_parse_and_validation")
def check_config_parse_and_validation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / ".symphonai" / "config.toml"
        source.parent.mkdir()
        source.write_text(
            '[[lsp.servers]]\nname="pyright"\ncommand=["pyright-langserver","--stdio"]\n'
            'language_id="python"\nextensions=[".py", ".pyi"]\nenabled=false\n',
            encoding="utf-8",
        )
        config = load_config(repo_root=root, home=root / "home")
        grant = TrustList((RepositoryTrust(root.resolve(), frozenset({"lsp"}), None),))
        spec, = lsp_servers_from_config(config, repo_root=root, trust=grant)
        if spec != LspServerSpec("pyright", ("pyright-langserver", "--stdio"), "python", (".py", ".pyi"), False, 30.0, 10.0, source):
            fail(f"valid LSP server config parsed incorrectly: {spec!r}")
        cases = (
            ('[[lsp.servers]]\nname="x"\ncommand=["x"]\nlanguage_id="python"\nextensions=[".py"]\nunknown=true\n', "unknown"),
            ('[[lsp.servers]]\nname="x"\ncommand=["x"]\nlanguage_id="python"\nextensions=[".py"]\n[[lsp.servers]]\nname="x"\ncommand=["x"]\nlanguage_id="python"\nextensions=[".rs"]\n', "duplicate"),
            ('[[lsp.servers]]\nname="x"\ncommand=[]\nlanguage_id="python"\nextensions=[".py"]\n', "command"),
            ('[[lsp.servers]]\nname="x"\ncommand=["x"]\nlanguage_id="python"\nextensions=[".py"]\nenabled=true\n[[lsp.servers]]\nname="y"\ncommand=["y"]\nlanguage_id="python"\nextensions=[".py"]\nenabled=true\n', "both claim"),
        )
        for text, expected in cases:
            source.write_text(text, encoding="utf-8")
            config = load_config(repo_root=root, home=root / "home")
            try:
                lsp_servers_from_config(config, repo_root=root, trust=grant)
            except ConfigError as exc:
                if str(source) not in str(exc) or expected not in str(exc):
                    fail(f"LSP config error omitted its source or reason: {exc!r}")
            else:
                fail(f"invalid LSP config was accepted: {expected}")


@check("lsp.project_trust")
def check_project_trust() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        source = root / ".symphonai" / "config.toml"
        source.parent.mkdir()
        source.write_text(
            '[[lsp.servers]]\nname="x"\ncommand=["x"]\nlanguage_id="python"\nextensions=[".py"]\nenabled=true\n',
            encoding="utf-8",
        )
        config = load_config(repo_root=root, home=root / "home")
        try:
            lsp_servers_from_config(config, repo_root=root, trust=TrustList())
        except ConfigError as exc:
            if "lsp trust grant" not in str(exc):
                fail(f"untrusted project LSP error was unclear: {exc!r}")
        else:
            fail("untrusted project LSP server was enabled")
        grant = TrustList((RepositoryTrust(root, frozenset({"lsp"}), None),))
        servers = lsp_servers_from_config(config, repo_root=root, trust=grant)
        if len(servers) != 1 or not servers[0].enabled:
            fail("trusted project LSP server was not enabled")


@check("lsp.lazy_start_server_requests_and_reuse")
def check_lazy_start_server_requests_and_reuse() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script, log = _fake_server(root)
        spec = _spec((os.sys.executable, str(script), str(log)))
        manager = LspManager((spec,), root=root)
        try:
            path = root / "a.py"
            path.write_text("one\n", encoding="utf-8")
            client = manager.client_for(path)
            if client is None or manager.client_for(path) is not client or manager.client_for(root / "a.rs") is not None:
                fail("LSP manager did not start lazily, reuse, or ignore an unsupported suffix")
            entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            replies = [entry for entry in entries if entry.get("id") in ("workspace", "register")]
            if {entry.get("id") for entry in replies} != {"workspace", "register"} or any("result" not in x for x in replies):
                fail(f"client did not answer server requests during initialization: {entries!r}")
        finally:
            manager.close()


@check("lsp.sync_and_diagnostics")
def check_sync_and_diagnostics() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script, log = _fake_server(root)
        manager = LspManager((_spec((os.sys.executable, str(script), str(log))),), root=root)
        try:
            path = root / "doc.py"
            path.write_text("one\n", encoding="utf-8")
            client = manager.client_for(path)
            if client is None:
                fail("fake LSP server did not start")
            first = client.sync(path)
            if client.wait_diagnostics(path, 0, 2) != [{"message": "opened"}]:
                fail("didOpen diagnostics were not recorded")
            current = client.sync(path)
            if current <= first:
                fail("unchanged document sent a new sync version")
            path.write_text("two\n", encoding="utf-8")
            after = client.sync(path)
            diagnostics = client.wait_diagnostics(path, current, 2)
            if after != current or diagnostics != [{"message": "changed"}]:
                fail(f"changed document diagnostics were incorrect: {after!r}, {diagnostics!r}")
            started = time.monotonic()
            if client.wait_diagnostics(path, 100, 0.05) is not None or time.monotonic() - started > 0.5:
                fail("diagnostic wait did not time out promptly")
            entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            changes = [entry for entry in entries if entry.get("method") in ("textDocument/didOpen", "textDocument/didChange")]
            if [entry["method"] for entry in changes] != ["textDocument/didOpen", "textDocument/didChange"] or changes[1]["params"]["textDocument"]["version"] != 2:
                fail(f"LSP document sync versions were wrong: {changes!r}")
        finally:
            manager.close()


@check("lsp.request_timeouts_errors_and_recovery")
def check_request_timeouts_errors_and_recovery() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script, log = _fake_server(root)
        client = LspClient(_spec((os.sys.executable, str(script), str(log))), root=root)
        try:
            try:
                client.request("test/slow", {}, timeout=0.05)
            except LspError as exc:
                if "timed out" not in str(exc):
                    fail(f"request timeout error was unclear: {exc!r}")
            else:
                fail("unanswered request did not time out")
            if client.request("test/ping", {}) != {"ok": True}:
                fail("late response handling prevented a later request")
            try:
                client.request("test/error", {})
            except LspError as exc:
                if "fixture error" not in str(exc):
                    fail(f"server response error was lost: {exc!r}")
            else:
                fail("LSP error response was accepted")
        finally:
            client.close()


@check("lsp.concurrent_requests_match_ids")
def check_concurrent_requests_match_ids() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script, log = _fake_server(root)
        client = LspClient(_spec((os.sys.executable, str(script), str(log))), root=root)
        results = []
        failures = []

        def request() -> None:
            try:
                results.append(client.request("test/ping", {}))
            except Exception as exc:
                failures.append(exc)

        threads = [threading.Thread(target=request) for _ in range(4)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)
            if any(thread.is_alive() for thread in threads) or failures or results != [{"ok": True}] * 4:
                fail(f"concurrent requests were not matched to their replies: {results!r}, {failures!r}")
        finally:
            client.close()


@check("lsp.failed_server_is_not_retried")
def check_failed_server_is_not_retried() -> None:
    spec = _spec(("/no/such/lsp-server",))
    try:
        LspClient(spec, root=Path.cwd())
    except LspError:
        pass
    else:
        fail("a missing LSP executable did not raise LspError")
    manager = LspManager((spec,), root=Path.cwd())
    try:
        try:
            manager.client_for(Path("a.py"))
        except LspError:
            pass
        else:
            fail("first failed server start did not raise LspError")
        if manager.client_for(Path("b.py")) is not None:
            fail("failed LSP server was retried or returned a client")
    finally:
        manager.close()


@check("lsp.close_reaps_process_group")
def check_close_reaps_process_group() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script, log = _fake_server(root)
        manager = LspManager((_spec((os.sys.executable, str(script), str(log))),), root=root)
        client = manager.client_for(root / "a.py")
        if client is None:
            fail("fake LSP server did not start")
        process = client.process
        manager.close()
        if process.poll() is None:
            fail("closing LSP manager left its process alive")


@check("lsp.navigation_tool_positions_and_results")
def check_navigation_tool_positions_and_results() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script, log = _fake_server(root)
        manager = LspManager((_spec((os.sys.executable, str(script), str(log))),), root=root)
        policy = PermissionPolicy(root, mode="allow")
        tool = LspTool(manager)
        path = root / "a.py"
        path.write_text("😀 = 1; xyzw\nsecond\nthird\n", encoding="utf-8")
        try:
            result = tool.execute(ToolCall("def", "lsp", {"operation": "definition", "path": str(path), "line": 3, "column": 5}), policy)
            if not result.ok or "a.py:3:5" not in result.content or "1 outside the readable repository" not in result.content:
                fail(f"definition result or path filtering was wrong: {result!r}")
            result = tool.execute(ToolCall("refs", "lsp", {"operation": "references", "path": str(path), "line": 3, "column": 5}), policy)
            if not result.ok or "a.py:3:5" not in result.content:
                fail(f"references result was not formatted: {result!r}")
            result = tool.execute(ToolCall("emoji", "lsp", {"operation": "hover", "path": str(path), "line": 1, "column": 10}), policy)
            if not result.ok or "```python\nx: int\n```" not in result.content:
                fail(f"hover did not preserve markup or UTF-16 position: {result!r}")
            result = tool.execute(ToolCall("symbols", "lsp", {"operation": "document_symbols", "path": str(path)}), policy)
            if not result.ok or result.content != "Class outer — line 1\n  Function inner — line 2":
                fail(f"document symbols were not rendered recursively: {result!r}")
            result = tool.execute(ToolCall("workspace", "lsp", {"operation": "workspace_symbols", "query": "work"}), policy)
            if not result.ok or "outside the readable repository" not in result.content or not result.content.startswith("No results."):
                fail(f"workspace symbols did not reach/filter server results: {result!r}")
            entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            calls = [entry for entry in entries if entry.get("method") in ("textDocument/definition", "textDocument/hover")]
            if calls[0]["params"]["position"] != {"line": 2, "character": 4} or calls[1]["params"]["position"] != {"line": 0, "character": 10}:
                fail(f"LSP position conversion was wrong: {calls!r}")
            reference = next(entry for entry in entries if entry.get("method") == "textDocument/references")
            workspace_call = next(entry for entry in entries if entry.get("method") == "workspace/symbol")
            if reference["params"]["context"] != {"includeDeclaration": True} or workspace_call["params"]["query"] != "work":
                fail("references or workspace-symbol request parameters were wrong")
        finally:
            manager.close()


@check("lsp.navigation_policy_and_validation")
def check_navigation_policy_and_validation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script, log = _fake_server(root)
        manager = LspManager((_spec((os.sys.executable, str(script), str(log))),), root=root)
        tool = LspTool(manager)
        policy = PermissionPolicy(root, mode="allow")
        try:
            denied = tool.execute(ToolCall("secret", "lsp", {"operation": "hover", "path": ".env", "line": 1, "column": 1}), policy)
            unsupported = tool.execute(ToolCall("rs", "lsp", {"operation": "document_symbols", "path": "a.rs"}), policy)
            if denied.ok or manager.clients or "no language server handles .rs" not in (unsupported.error or ""):
                fail(f"LSP read policy or unsupported suffix behavior was wrong: {denied!r}, {unsupported!r}")
            for args in (
                {"operation": "hover", "path": "a.py", "line": 0, "column": 1},
                {"operation": "unknown", "path": "a.py"},
            ):
                if tool.validate(args) is None:
                    fail(f"LSP accepted invalid arguments: {args!r}")
        finally:
            manager.close()


@check("lsp.optional_registry_wiring")
def check_optional_registry_wiring() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = _spec((os.sys.executable, "unused"))
        enabled = LspManager((spec,), root=root)
        disabled = LspManager((_spec((os.sys.executable, "unused"), enabled=False),), root=root)
        policy = PermissionPolicy(root)
        try:
            roster = builtin_subagent_specs(FakeModelProvider(), policy, lsp=enabled)
            if "lsp" not in standard_tool_registry(lsp=enabled) or any("lsp" not in roster[name].tool_names for name in ("worker", "explorer")):
                fail(f"configured LSP tool was absent from built-in registries: {roster!r}")
            if "lsp" in standard_tool_registry() or "lsp" in standard_tool_registry(lsp=disabled):
                fail("LSP changed standard registries when it was unavailable")
        finally:
            enabled.close()
            disabled.close()


@check("lsp.result_cap_and_metadata")
def check_result_cap_and_metadata() -> None:
    from symphonai_api.tools.lsp import _cap
    from symphonai_api.tools.metadata import ToolEffect

    with tempfile.TemporaryDirectory() as temporary:
        manager = LspManager((_spec((os.sys.executable, "unused")),), root=Path(temporary))
        try:
            tool = LspTool(manager)
            many = _cap([f"item {index}" for index in range(205)])
            if "… 5 more" not in many or many.count("item ") != 200:
                fail("LSP result was not capped at 200 items")
            location = tool.metadata({"operation": "hover", "path": "a.py"})
            workspace = tool.metadata({"operation": "workspace_symbols"})
            if location.effect is not ToolEffect.READ_ONLY or not location.concurrency_safe or location.paths != ("a.py",):
                fail(f"path-based LSP metadata was wrong: {location!r}")
            if workspace.paths != () or not workspace.concurrency_safe:
                fail(f"workspace-symbol metadata was wrong: {workspace!r}")
        finally:
            manager.close()


@check("lsp.diagnostics_after_edit_preserves_diff")
def check_diagnostics_after_edit_preserves_diff() -> None:
    from symphonai_api.tools.read_ledger import ReadLedger
    from symphonai_api.tools.diagnostics import DiagnosticsAfterWrite
    from symphonai_api.tools.edit import EditFileTool
    from symphonai_api.tools.filesystem import ReadFileTool

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script, log = _fake_server(root)
        manager = LspManager((_spec((os.sys.executable, str(script), str(log))),), root=root)
        path = root / "a.py"
        path.write_text("a\nb\nc\nold\nfxx\n", encoding="utf-8")
        ledger = ReadLedger()
        policy = PermissionPolicy(root, allowed_write_scope=[root], mode="allow")
        try:
            ReadFileTool(ledger).execute(ToolCall("read", "read_file", {"path": "a.py"}), policy)
            wrapped = DiagnosticsAfterWrite(EditFileTool(ledger), manager)
            result = wrapped.execute(ToolCall("edit", "edit_file", {"path": "a.py", "old_string": "old", "new_string": "diagnostic-error"}), policy)
            if not result.ok or "Errors reported by fake after this write (1):" not in result.content or "a.py:5:3: broken type" not in result.content or "warning" in result.content:
                fail(f"edit diagnostics were not appended as errors only: {result!r}")
            if not isinstance(result.payload, dict) or result.payload.get("diff") not in (None, result.content.split("\n\nErrors reported by ")[0]):
                fail(f"diagnostic wrapper failed to preserve the original diff: {result.payload!r}")
        finally:
            manager.close()


@check("lsp.diagnostics_timeout_empty_and_suffix")
def check_diagnostics_timeout_empty_and_suffix() -> None:
    from symphonai_api.tools.diagnostics import DiagnosticsAfterWrite
    from symphonai_api.tools.filesystem import WriteFileTool
    from symphonai_api.tools.read_ledger import ReadLedger

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script, log = _fake_server(root)
        manager = LspManager((_spec((os.sys.executable, str(script), str(log))),), root=root)
        wrapped = DiagnosticsAfterWrite(WriteFileTool(ReadLedger()), manager)
        policy = PermissionPolicy(root, allowed_write_scope=[root], mode="allow")
        try:
            plain = wrapped.execute(ToolCall("empty", "write_file", {"path": "a.py", "content": "empty-diagnostics"}), policy)
            if not plain.ok or "Errors reported" in plain.content:
                fail(f"empty diagnostics changed a write result: {plain!r}")
            warning = wrapped.execute(ToolCall("warning", "write_file", {"path": "b.py", "content": "warning-only"}), policy)
            if not warning.ok or "Errors reported" in warning.content:
                fail(f"warning-only diagnostics changed a write result: {warning!r}")
            unsupported = wrapped.execute(ToolCall("rs", "write_file", {"path": "a.rs", "content": "text"}), policy)
            if not unsupported.ok or "Errors reported" in unsupported.content:
                fail(f"unsupported suffix diagnostics changed a write result: {unsupported!r}")
        finally:
            manager.close()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script, log = _fake_server(root)
        manager = LspManager((_spec((os.sys.executable, str(script), str(log))),), root=root)
        tool = DiagnosticsAfterWrite(WriteFileTool(ReadLedger()), manager)
        try:
            started = time.monotonic()
            result = tool.execute(ToolCall("timeout", "write_file", {"path": "slow.py", "content": "no-diagnostics"}), PermissionPolicy(root, allowed_write_scope=[root], mode="allow"))
            if not result.ok or "Errors reported" in result.content or time.monotonic() - started > 3.8:
                fail(f"diagnostic timeout changed or delayed a successful write: {result!r}")
        finally:
            manager.close()


@check("lsp.diagnostics_cap_and_crash")
def check_diagnostics_cap_and_crash() -> None:
    from symphonai_api.tools.diagnostics import DiagnosticsAfterWrite
    from symphonai_api.tools.filesystem import WriteFileTool
    from symphonai_api.tools.read_ledger import ReadLedger

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script, log = _fake_server(root)
        manager = LspManager((_spec((os.sys.executable, str(script), str(log))),), root=root)
        tool = DiagnosticsAfterWrite(WriteFileTool(ReadLedger()), manager)
        policy = PermissionPolicy(root, allowed_write_scope=[root], mode="allow")
        try:
            capped = tool.execute(ToolCall("many", "write_file", {"path": "many.py", "content": "twenty-five-errors"}), policy)
            if not capped.ok or "Errors reported by fake after this write (25):" not in capped.content or "… 5 more" not in capped.content:
                fail(f"diagnostics were not capped at 20: {capped!r}")
        finally:
            manager.close()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script, log = _fake_server(root)
        manager = LspManager((_spec((os.sys.executable, str(script), str(log))),), root=root)
        tool = DiagnosticsAfterWrite(WriteFileTool(ReadLedger()), manager)
        try:
            result = tool.execute(ToolCall("crash", "write_file", {"path": "crash.py", "content": "crash-diagnostics"}), PermissionPolicy(root, allowed_write_scope=[root], mode="allow"))
            if not result.ok or "Errors reported" in result.content:
                fail(f"language server crash changed the successful write: {result!r}")
        finally:
            manager.close()


@check("lsp.diagnostics_registry_opt_in")
def check_diagnostics_registry_opt_in() -> None:
    from symphonai_api.tools.diagnostics import DiagnosticsAfterWrite
    from symphonai_api.tools.edit import EditFileTool
    from symphonai_api.tools.filesystem import WriteFileTool
    from symphonai_api.tools.read_ledger import ReadLedger

    plain = standard_tool_registry()
    if any(isinstance(plain[name], DiagnosticsAfterWrite) for name in ("write_file", "edit_file", "multi_edit_file")):
        fail("write tools were wrapped without an enabled language server")
    with tempfile.TemporaryDirectory() as temporary:
        manager = LspManager((_spec((os.sys.executable, "unused"), suffixes=(".rs",)),), root=Path(temporary))
        try:
            wrapped = standard_tool_registry(lsp=manager)
            if any(not isinstance(wrapped[name], DiagnosticsAfterWrite) for name in ("write_file", "edit_file", "multi_edit_file")):
                fail("enabled language server did not wrap all write tools")
        finally:
            manager.close()
