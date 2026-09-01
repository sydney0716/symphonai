"""Workspace-backed checks for permissions."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import fields

from symphonai_api.models import ToolCall
from symphonai_api.permissions import (
    DenialReason,
    PermissionDecision,
    PermissionPolicy,
)
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


@check("permissions.typed_reasons")
def check_permissions_typed_reasons() -> None:
    with workspace() as ws:
        try:
            PermissionDecision.deny("missing typed reason")
        except TypeError:
            pass
        else:
            fail("PermissionDecision.deny accepted a missing denial keyword")
        if PermissionDecision.allow().denial is not None:
            fail("an allowed permission decision carried a denial reason")

        no_write = PermissionPolicy(repo_root=ws.root)
        shell_allowlisted = PermissionPolicy(
            repo_root=ws.root,
            shell_enabled=True,
            shell_allowlist=[("ls",)],
        )
        decisions = [
            (
                no_write.check_read("../outside.txt"),
                "path escapes repo_root: '../outside.txt'",
                DenialReason.OUTSIDE_ROOT,
            ),
            (
                no_write.check_read(".env"),
                "path matches forbidden pattern '.env': '.env'",
                DenialReason.FORBIDDEN_PATTERN,
            ),
            (
                no_write.check_write("outside.txt"),
                "path is outside the explicit allowed write scope: 'outside.txt'",
                DenialReason.OUTSIDE_WRITE_SCOPE,
            ),
            (
                no_write.check_shell([]),
                "empty command",
                DenialReason.EMPTY_COMMAND,
            ),
            (
                no_write.check_shell(["rm", "-rf", "/"]),
                "command matches always-deny rule: 'rm'",
                DenialReason.ALWAYS_DENY,
            ),
            (
                no_write.check_shell(["ls"]),
                "run_shell is disabled by this policy",
                DenialReason.SHELL_DISABLED,
            ),
            (
                shell_allowlisted.check_shell(["echo", "hi"]),
                "command does not match the shell allowlist: ['echo', 'hi']",
                DenialReason.NOT_ALLOWLISTED,
            ),
        ]

        def raising_callback(request) -> bool:
            raise RuntimeError("approval broke")

        approval_decisions = [
            (
                PermissionPolicy(repo_root=ws.root, mode="prompt").check_write(
                    "new.txt"
                ),
                "write_file requires approval, but no approval callback is configured",
                DenialReason.NO_APPROVAL_CALLBACK,
            ),
            (
                PermissionPolicy(
                    repo_root=ws.root,
                    mode="prompt",
                    approval_callback=raising_callback,
                ).check_write("new.txt"),
                "approval callback failed (RuntimeError): approval broke",
                DenialReason.APPROVAL_FAILED,
            ),
            (
                PermissionPolicy(
                    repo_root=ws.root,
                    mode="prompt",
                    approval_callback=lambda request: False,
                ).check_write("new.txt"),
                "write_file denied by user",
                DenialReason.DENIED_BY_USER,
            ),
            (
                PermissionPolicy(
                    repo_root=ws.root,
                    mode="prompt",
                    approval_callback=lambda request: "invalid",
                ).check_write("new.txt"),
                "approval callback returned an invalid decision for write_file",
                DenialReason.INVALID_APPROVAL,
            ),
        ]
        for decision, expected_reason, expected_denial in [
            *decisions,
            *approval_decisions,
        ]:
            if (
                decision.allowed
                or decision.reason != expected_reason
                or decision.denial is not expected_denial
            ):
                fail(
                    "typed permission denial changed: "
                    f"actual={decision!r}, reason={expected_reason!r}, "
                    f"denial={expected_denial!r}"
                )

        encoded = json.dumps(list(DenialReason))
        if json.loads(encoded) != [reason.value for reason in DenialReason]:
            fail(f"DenialReason did not serialize as string values: {encoded!r}")


@check("permissions.named_modes_and_equality")
def check_permissions_named_modes_and_equality() -> None:
    with workspace() as ws:
        PermissionPolicy(repo_root=ws.root, mode="plan")
        PermissionPolicy(repo_root=ws.root, mode="accept_edits")
        try:
            PermissionPolicy(repo_root=ws.root, mode="bogus")
        except ValueError as exc:
            expected = (
                "unknown permission mode 'bogus'; expected "
                "'auto', 'prompt', 'plan', or 'accept_edits'"
            )
            if str(exc) != expected:
                fail(f"unknown-mode error did not name all modes: {exc!r}")
        else:
            fail("an unknown permission mode was accepted")

        first = PermissionPolicy(
            repo_root=ws.root,
            allowed_write_scope=[ws.root],
            mode="prompt",
        )
        second = PermissionPolicy(
            repo_root=ws.root,
            allowed_write_scope=[ws.root],
            mode="prompt",
        )
        if first != second:
            fail("matching PermissionPolicy instances stopped comparing equal")
        if "_approval_lock" in {item.name for item in fields(PermissionPolicy)}:
            fail("the approval lock became a dataclass field")


@check("permissions.accept_edits")
def check_permissions_accept_edits() -> None:
    with workspace() as ws:
        scope = ws.root / "scope"
        scope.mkdir()
        requests = []

        def approve(request) -> bool:
            requests.append(request)
            return True

        policy = PermissionPolicy(
            repo_root=ws.root,
            allowed_write_scope=[scope],
            mode="accept_edits",
            approval_callback=approve,
        )
        write = ws.tools["write_file"].execute(
            ToolCall(
                id="accept-write",
                name="write_file",
                arguments={"path": "scope/new.txt", "content": "accepted"},
            ),
            policy,
        )
        if not write.ok or requests:
            fail(f"accept_edits prompted for an in-scope write: {write!r}, {requests!r}")

        outside = policy.check_write("outside.txt")
        if (
            outside.allowed
            or outside.reason
            != "path is outside the explicit allowed write scope: 'outside.txt'"
            or outside.denial is not DenialReason.OUTSIDE_WRITE_SCOPE
        ):
            fail(f"accept_edits allowed an out-of-scope write: {outside!r}")

        shell = policy.check_shell(["ls"])
        if not shell.allowed or len(requests) != 1 or requests[0].operation != "run_shell":
            fail(f"accept_edits did not prompt for shell: {shell!r}, {requests!r}")


@check("permissions.approval_serialization")
def check_permissions_approval_serialization() -> None:
    with workspace() as ws:
        counter_lock = threading.Lock()
        active = 0
        peak = 0

        def approve(request) -> bool:
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with counter_lock:
                active -= 1
            return True

        policy = PermissionPolicy(
            repo_root=ws.root,
            mode="prompt",
            approval_callback=approve,
        )
        decisions: list[PermissionDecision] = []
        decisions_lock = threading.Lock()

        def ask() -> None:
            decision = policy.check_shell(["ls"])
            with decisions_lock:
                decisions.append(decision)

        threads = [threading.Thread(target=ask) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        if any(thread.is_alive() for thread in threads):
            fail("serialized approval threads did not finish")
        if len(decisions) != 8 or not all(decision.allowed for decision in decisions):
            fail(f"serialized approvals lost decisions: {decisions!r}")
        if peak != 1:
            fail(f"approval callbacks overlapped; observed peak concurrency {peak}")
