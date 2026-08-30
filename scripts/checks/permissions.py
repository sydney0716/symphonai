"""Workspace-backed checks for permissions."""

from __future__ import annotations

from orchestra_api.models import ToolCall
from orchestra_api.permissions import PermissionPolicy
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


@check("permissions.read_inside_root")
def check_permissions_read_inside_root() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        r = tools["read_file"].execute(ToolCall(id="a1", name="read_file", arguments={"path": "existing.txt"}), policy)
        if not r.ok:
            fail(f"read_file should be allowed inside repo_root: {r.error}")

@check("permissions.list_inside_root")
def check_permissions_list_inside_root() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        r = tools["list_files"].execute(ToolCall(id="a2", name="list_files", arguments={"path": "."}), policy)
        if not r.ok or "existing.txt" not in r.content:
            fail(f"list_files should be allowed and show existing.txt: {r}")

@check("permissions.write_inside_scope")
def check_permissions_write_inside_scope() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        r = tools["write_file"].execute(
            ToolCall(id="a3", name="write_file", arguments={"path": "new.txt", "content": "written by smoke test"}),
            policy,
        )
        if not r.ok or not (root / "new.txt").exists():
            fail(f"write_file should be allowed inside allowed_write_scope: {r}")

@check("permissions.write_outside_scope_denied")
def check_permissions_write_outside_scope_denied() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        # -- deny path: write outside the allowed write scope --
        no_write_policy = PermissionPolicy(repo_root=root)  # allowed_write_scope defaults to empty
        r = tools["write_file"].execute(
            ToolCall(id="d1", name="write_file", arguments={"path": "should_not_exist.txt", "content": "x"}),
            no_write_policy,
        )
        if r.ok or (root / "should_not_exist.txt").exists():
            fail("write_file should be denied when allowed_write_scope is empty")

@check("permissions.forbidden_read_denied")
def check_permissions_forbidden_read_denied() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        # -- deny path: forbidden pattern (.env) --
        r = tools["read_file"].execute(ToolCall(id="d2", name="read_file", arguments={"path": ".env"}), policy)
        if r.ok:
            fail("read_file should deny a forbidden-pattern path (.env)")

@check("permissions.traversal_denied")
def check_permissions_traversal_denied() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        # -- deny path: .. path traversal --
        r = tools["read_file"].execute(
            ToolCall(id="d3", name="read_file", arguments={"path": "../outside.txt"}), policy
        )
        if r.ok:
            fail("read_file should deny a .. path-traversal attempt")

@check("permissions.shell_disabled")
def check_permissions_shell_disabled() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        # -- deny path: run_shell disabled by default --
        r = tools["run_shell"].execute(
            ToolCall(id="d4", name="run_shell", arguments={"argv": ["echo", "hi"]}), policy
        )
        if r.ok:
            fail("run_shell should be denied by default")

@check("permissions.shell_always_denied")
def check_permissions_shell_always_denied() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        # -- deny path: always-deny command wins even when explicitly allowlisted --
        risky_policy = PermissionPolicy(
            repo_root=root, shell_enabled=True, shell_allowlist=[("rm",)]
        )
        r = tools["run_shell"].execute(
            ToolCall(id="d5", name="run_shell", arguments={"argv": ["rm", "-rf", "existing.txt"]}), risky_policy
        )
        if r.ok or not (root / "existing.txt").exists():
            fail("run_shell must deny 'rm' even when explicitly allowlisted")
