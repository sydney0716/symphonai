"""Checks for read-only plan-mode permission enforcement."""

from __future__ import annotations

from orchestra_api.models import ToolCall
from orchestra_api.permissions import DenialReason, PermissionPolicy
from orchestra_api.tools.shell_classify import classify
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


PLAN_MODE_REASON = "plan mode allows reads only; this call would change the world"


@check("plan.path_decisions")
def check_plan_path_decisions() -> None:
    with workspace() as ws:
        policy = PermissionPolicy(
            repo_root=ws.root,
            allowed_write_scope=[ws.root],
            mode="plan",
        )
        read = policy.check_read("existing.txt")
        if not read.allowed or read.denial is not None:
            fail(f"plan mode refused an in-root read: {read!r}")

        write = policy.check_write("existing.txt")
        if (
            write.allowed
            or write.reason != PLAN_MODE_REASON
            or write.denial is not DenialReason.PLAN_MODE
        ):
            fail(f"plan mode did not refuse a writable path: {write!r}")

        outside = policy.check_write("../outside.txt")
        if (
            outside.allowed
            or outside.reason != "path escapes repo_root: '../outside.txt'"
            or outside.denial is not DenialReason.OUTSIDE_ROOT
        ):
            fail(f"outside-root reason did not take precedence in plan mode: {outside!r}")


@check("plan.command_decisions")
def check_plan_command_decisions() -> None:
    with workspace() as ws:
        policy = PermissionPolicy(
            repo_root=ws.root,
            shell_enabled=True,
            shell_allowlist=[("ls",), ("rm",)],
            mode="plan",
        )
        always_denied = policy.check_shell(["rm", "-rf", "/"])
        if (
            always_denied.allowed
            or always_denied.reason != "command matches always-deny rule: 'rm'"
            or always_denied.denial is not DenialReason.ALWAYS_DENY
        ):
            fail(f"always-deny did not take precedence in plan mode: {always_denied!r}")

        ls_metadata = classify(["ls"])
        if ls_metadata is None or not ls_metadata.concurrency_safe:
            fail(f"test precondition changed for ls classification: {ls_metadata!r}")
        plan_denied = policy.check_shell(["ls"])
        if (
            plan_denied.allowed
            or plan_denied.reason != PLAN_MODE_REASON
            or plan_denied.denial is not DenialReason.PLAN_MODE
        ):
            fail(f"safe-classified shell escaped plan mode: {plan_denied!r}")


@check("plan.real_tools")
def check_plan_real_tools() -> None:
    with workspace() as ws:
        policy = PermissionPolicy(
            repo_root=ws.root,
            allowed_write_scope=[ws.root],
            mode="plan",
        )
        read = ws.tools["read_file"].execute(
            ToolCall(
                id="plan-read",
                name="read_file",
                arguments={"path": "existing.txt"},
            ),
            policy,
        )
        if not read.ok or "hello from disk" not in read.content:
            fail(f"real read_file failed in plan mode: {read!r}")

        write = ws.tools["write_file"].execute(
            ToolCall(
                id="plan-write",
                name="write_file",
                arguments={"path": "new.txt", "content": "blocked"},
            ),
            policy,
        )
        if write.ok or write.error != PLAN_MODE_REASON or (ws.root / "new.txt").exists():
            fail(f"real write_file escaped plan mode: {write!r}")

        edit = ws.tools["edit_file"].execute(
            ToolCall(
                id="plan-edit",
                name="edit_file",
                arguments={
                    "path": "existing.txt",
                    "old_string": "hello",
                    "new_string": "goodbye",
                },
            ),
            policy,
        )
        if (
            edit.ok
            or edit.error != PLAN_MODE_REASON
            or (ws.root / "existing.txt").read_text() != "hello from disk"
        ):
            fail(f"real edit_file escaped plan mode: {edit!r}")
