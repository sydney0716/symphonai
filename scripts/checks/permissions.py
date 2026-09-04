"""Workspace-backed checks for permissions."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from dataclasses import fields
from pathlib import Path

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


@check("permissions.narrow_returns_a_new_policy")
def check_narrow_returns_a_new_policy() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        def parent_approval(request) -> bool:
            return True

        def ceiling_approval(request) -> bool:
            return True

        parent = PermissionPolicy(
            repo_root=root,
            allowed_write_scope=[root],
            mode="prompt",
            approval_callback=parent_approval,
        )
        ceiling = PermissionPolicy(
            repo_root=root,
            allowed_write_scope=[root],
            mode="prompt",
            approval_callback=ceiling_approval,
        )
        child = parent.narrowed(ceiling)
        if child is parent or child is ceiling:
            fail("narrowing returned one of its inputs")
        if parent.narrowed(parent) is parent:
            fail("self narrowing returned the parent instead of a fresh policy")
        if child._approval_lock in (parent._approval_lock, ceiling._approval_lock):  # noqa: SLF001
            fail("narrowed policy reused an input approval lock")
        if child.approval_callback is not ceiling_approval:
            fail("narrowed policy did not prefer the ceiling approval callback")

        completed = threading.Event()

        def ask_child() -> None:
            child.check_shell(["ls"])
            completed.set()

        parent._approval_lock.acquire()  # noqa: SLF001
        try:
            worker = threading.Thread(target=ask_child)
            worker.start()
            if not completed.wait(0.5):
                fail("child approval serialized behind the parent lock")
        finally:
            parent._approval_lock.release()  # noqa: SLF001
        worker.join(timeout=0.5)
        if worker.is_alive():
            fail("child approval worker did not finish")

        fallback = PermissionPolicy(
            repo_root=root, mode="prompt", approval_callback=parent_approval
        ).narrowed(PermissionPolicy(repo_root=root, mode="prompt"))
        if fallback.approval_callback is not parent_approval:
            fail("narrowed policy did not retain a parent callback when ceiling lacked one")


@check("permissions.narrow_repo_root")
def check_narrow_repo_root() -> None:
    with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
        root = Path(temporary)
        inside = root / "inside"
        inside.mkdir()
        parent = PermissionPolicy(repo_root=root)
        child = parent.narrowed(PermissionPolicy(repo_root=inside))
        if child.repo_root != inside.resolve():
            fail(f"narrowed root was not the ceiling root: {child.repo_root!r}")
        outside_policy = PermissionPolicy(repo_root=Path(outside))
        try:
            parent.narrowed(outside_policy)
        except ValueError as exc:
            if str(root.resolve()) not in str(exc) or str(Path(outside).resolve()) not in str(exc):
                fail(f"outside-root error did not name both roots: {exc!r}")
        else:
            fail("narrowing accepted an outside repo root")


@check("permissions.narrow_write_scope")
def check_narrow_write_scope() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        shared = root / "shared"
        nested = shared / "nested"
        left = root / "left"
        right = root / "right"
        for path in (nested, left, right):
            path.mkdir(parents=True, exist_ok=True)

        nested_child = PermissionPolicy(
            repo_root=root, allowed_write_scope=[shared]
        ).narrowed(PermissionPolicy(repo_root=root, allowed_write_scope=[nested]))
        if nested_child.allowed_write_scope != [nested.resolve()]:
            fail(f"nested write scopes did not keep the deeper scope: {nested_child.allowed_write_scope!r}")
        if not nested_child.check_write(nested / "ok.txt").allowed:
            fail("the common nested write scope was denied")

        disjoint_child = PermissionPolicy(
            repo_root=root, allowed_write_scope=[left]
        ).narrowed(PermissionPolicy(repo_root=root, allowed_write_scope=[right]))
        if disjoint_child.allowed_write_scope:
            fail(f"disjoint write scopes survived intersection: {disjoint_child.allowed_write_scope!r}")


@check("permissions.narrow_forbidden_union")
def check_narrow_forbidden_union() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent_only = root / "parent-only.secret"
        ceiling_only = root / "ceiling-only.secret"
        parent_only.touch()
        ceiling_only.touch()
        child = PermissionPolicy(
            repo_root=root, forbidden_patterns=("parent-only.secret",)
        ).narrowed(
            PermissionPolicy(repo_root=root, forbidden_patterns=("ceiling-only.secret",))
        )
        if child.forbidden_patterns != ("parent-only.secret", "ceiling-only.secret"):
            fail(f"forbidden patterns were not ordered union: {child.forbidden_patterns!r}")
        if child.check_read(parent_only).allowed or child.check_read(ceiling_only).allowed:
            fail("a pattern forbidden by either policy was readable in the child")


@check("permissions.narrow_shell_and_fetch")
def check_narrow_shell_and_fetch() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = PermissionPolicy(
            repo_root=root,
            shell_enabled=True,
            shell_allowlist=[("git",), ("python", "-m")],
            fetch_enabled=False,
            fetch_allowlist=["parent.example", "shared.example"],
            shell_timeout_seconds=20,
            shell_output_limit_chars=8_000,
        )
        ceiling = PermissionPolicy(
            repo_root=root,
            shell_enabled=True,
            shell_allowlist=[("git", "status"), ("python",)],
            fetch_enabled=False,
            fetch_allowlist=["shared.example", "ceiling.example"],
            shell_timeout_seconds=4,
            shell_output_limit_chars=1_500,
        )
        child = parent.narrowed(ceiling)
        if child.shell_allowlist != [("git", "status"), ("python", "-m")]:
            fail(f"shell allowlist intersection was wrong: {child.shell_allowlist!r}")
        if not child.check_shell(["git", "status"]).allowed or child.check_shell(
            ["git", "log"]
        ).allowed:
            fail("shell intersection did not keep only common argv prefixes")
        if child.fetch_allowlist != ["shared.example"]:
            fail(f"fetch allowlist intersection was wrong: {child.fetch_allowlist!r}")
        if not child.check_fetch("https://shared.example/path").allowed or child.check_fetch(
            "https://parent.example/path"
        ).allowed:
            fail("fetch intersection did not keep only common hosts")
        if child.shell_timeout_seconds != 4 or child.shell_output_limit_chars != 1_500:
            fail("narrowed shell limits were not minima")

        disabled = PermissionPolicy(
            repo_root=root, shell_enabled=True, fetch_enabled=True
        ).narrowed(
            PermissionPolicy(repo_root=root, shell_enabled=False, fetch_enabled=False)
        )
        if disabled.shell_enabled or disabled.fetch_enabled:
            fail("narrowed enabled flags were not intersected")


@check("permissions.narrow_mode")
def check_narrow_mode() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = PermissionPolicy(repo_root=root, mode="auto")
        if parent.narrowed(PermissionPolicy(repo_root=root, mode="auto")).mode != "auto":
            fail("matching mode was not preserved")
        if parent.narrowed(PermissionPolicy(repo_root=root, mode="plan")).mode != "plan":
            fail("plan ceiling did not narrow the mode")
        for ceiling_mode in ("prompt", "accept_edits"):
            try:
                parent.narrowed(PermissionPolicy(repo_root=root, mode=ceiling_mode))
            except ValueError as exc:
                if "auto" not in str(exc) or ceiling_mode not in str(exc):
                    fail(f"incompatible-mode error did not name both modes: {exc!r}")
            else:
                fail(f"incompatible mode {ceiling_mode!r} was accepted")


def _assert_never_widens(
    parent: PermissionPolicy,
    ceiling: PermissionPolicy,
    paths: list[Path],
    urls: list[str],
    argvs: list[list[str]],
) -> None:
    child = parent.narrowed(ceiling)
    for path in paths:
        for check_name in ("check_read", "check_list", "check_write"):
            child_decision = getattr(child, check_name)(path)
            if child_decision.allowed and not (
                getattr(parent, check_name)(path).allowed
                and getattr(ceiling, check_name)(path).allowed
            ):
                fail(f"{check_name} widened for {path!s}")
    for url in urls:
        if child.check_fetch(url).allowed and not (
            parent.check_fetch(url).allowed and ceiling.check_fetch(url).allowed
        ):
            fail(f"check_fetch widened for {url!r}")
    for argv in argvs:
        if child.check_shell(argv).allowed and not (
            parent.check_shell(argv).allowed and ceiling.check_shell(argv).allowed
        ):
            fail(f"check_shell widened for {argv!r}")


@check("permissions.narrow_never_widens")
def check_narrow_never_widens() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = [
            root / name
            for name in (
                "read.txt",
                "parent-only/write.txt",
                "ceiling-only/write.txt",
                "shared/nested/write.txt",
                "shared/other.txt",
                "parent-secret",
                "ceiling-secret",
                ".env",
                "build/output.txt",
                "ordinary/a.txt",
                "ordinary/b.txt",
                "ordinary/c.txt",
            )
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        outside = root.parent / "outside-policy-probe.txt"
        outside.touch()
        paths.append(outside)
        urls = [
            "https://parent.example/path",
            "https://ceiling.example/path",
            "https://shared.example/path",
            "https://unlisted.example/path",
            "ftp://shared.example/path",
        ]
        argvs = [
            ["parent-command"],
            ["ceiling-command"],
            ["shared", "nested"],
            ["shared", "other"],
            ["echo", "no"],
            [],
        ]
        pairs = [
            (
                PermissionPolicy(
                    repo_root=root,
                    allowed_write_scope=[root / "parent-only"],
                    forbidden_patterns=("parent-secret",),
                    shell_enabled=True,
                    shell_allowlist=[("parent-command",), ("shared",)],
                    fetch_enabled=True,
                    fetch_allowlist=["parent.example", "shared.example"],
                ),
                PermissionPolicy(
                    repo_root=root,
                    allowed_write_scope=[root / "ceiling-only"],
                    forbidden_patterns=("ceiling-secret",),
                    shell_enabled=True,
                    shell_allowlist=[("ceiling-command",), ("shared", "nested")],
                    fetch_enabled=True,
                    fetch_allowlist=["ceiling.example", "shared.example"],
                ),
            ),
            (
                PermissionPolicy(
                    repo_root=root,
                    allowed_write_scope=[root / "shared"],
                    forbidden_patterns=("parent-secret",),
                    shell_enabled=True,
                    shell_allowlist=[("shared",)],
                    fetch_enabled=True,
                    fetch_allowlist=["shared.example"],
                ),
                PermissionPolicy(
                    repo_root=root,
                    allowed_write_scope=[root / "shared/nested"],
                    forbidden_patterns=("ceiling-secret",),
                    shell_enabled=True,
                    shell_allowlist=[("shared", "nested")],
                    fetch_enabled=True,
                    fetch_allowlist=["shared.example"],
                ),
            ),
            (
                PermissionPolicy(
                    repo_root=root,
                    allowed_write_scope=[root / "shared/nested"],
                    forbidden_patterns=("parent-secret", "ceiling-secret"),
                    shell_enabled=True,
                    shell_allowlist=[("shared", "nested")],
                    fetch_enabled=True,
                    fetch_allowlist=["shared.example"],
                ),
                PermissionPolicy(
                    repo_root=root,
                    allowed_write_scope=[root / "shared/nested"],
                    forbidden_patterns=("parent-secret", "ceiling-secret"),
                    shell_enabled=True,
                    shell_allowlist=[("shared", "nested")],
                    fetch_enabled=True,
                    fetch_allowlist=["shared.example"],
                ),
            ),
        ]
        for parent, ceiling in pairs:
            _assert_never_widens(parent, ceiling, paths, urls, argvs)
        forbidden_parent = PermissionPolicy(repo_root=root)
        try:
            forbidden_parent.narrowed(PermissionPolicy(repo_root=root / "build"))
        except ValueError:
            pass
        else:
            fail("root narrowing through forbidden ground was accepted")


@check("permissions.narrow_rejects_a_forbidden_root")
def check_narrow_rejects_a_forbidden_root() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = PermissionPolicy(repo_root=root, allowed_write_scope=[root])
        patterns = (".git/", "build/", "dist/", "node_modules/", ".venv/", "__pycache__/", ".ssh/", "*.egg-info/")
        for pattern in patterns:
            name = "anything.egg-info" if pattern == "*.egg-info/" else pattern.rstrip("/")
            ceiling_root = root / name
            ceiling_root.mkdir()
            try:
                parent.narrowed(PermissionPolicy(repo_root=ceiling_root))
            except ValueError as exc:
                if str(ceiling_root) not in str(exc) or pattern not in str(exc):
                    fail(f"forbidden-root error was incomplete: {exc!r}")
            else:
                fail(f"forbidden root was accepted: {ceiling_root!s}")
        nested = root / "a" / "build" / "b"
        nested.mkdir(parents=True)
        try:
            parent.narrowed(PermissionPolicy(repo_root=nested))
        except ValueError as exc:
            if "build/" not in str(exc):
                fail(f"nested root named the wrong pattern: {exc!r}")
        else:
            fail("nested forbidden root was accepted")
        for allowed in (root, root / "src", root / "src" / "nested"):
            allowed.mkdir(parents=True, exist_ok=True)
            if parent.narrowed(PermissionPolicy(repo_root=allowed)).repo_root != allowed.resolve():
                fail(f"allowed root did not narrow: {allowed!s}")
        named_root = root / "build"
        named_root.mkdir(exist_ok=True)
        named = PermissionPolicy(repo_root=named_root)
        if named.narrowed(named).repo_root != named_root.resolve():
            fail("self narrowing rejected a forbidden-named root")


@check("permissions.narrow_is_idempotent_and_composes")
def check_narrow_is_idempotent_and_composes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = [root / "common/nested/file.txt", root / "a/file.txt", root / "b/file.txt"]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        urls = ["https://shared.example/x", "https://parent.example/x"]
        argvs = [["git", "status"], ["git", "log"], ["echo", "x"]]
        parent = PermissionPolicy(
            repo_root=root,
            allowed_write_scope=[root / "a", root / "b", root / "common"],
            forbidden_patterns=("parent-secret",),
            shell_enabled=True,
            shell_allowlist=[("git",), ("echo",)],
            fetch_enabled=True,
            fetch_allowlist=["shared.example", "parent.example"],
        )
        first_ceiling = PermissionPolicy(
            repo_root=root,
            allowed_write_scope=[root / "a", root / "common"],
            forbidden_patterns=("first-secret",),
            shell_enabled=True,
            shell_allowlist=[("git",)],
            fetch_enabled=True,
            fetch_allowlist=["shared.example"],
        )
        second_ceiling = PermissionPolicy(
            repo_root=root,
            allowed_write_scope=[root / "b", root / "common/nested"],
            forbidden_patterns=("second-secret",),
            shell_enabled=True,
            shell_allowlist=[("git", "status")],
            fetch_enabled=True,
            fetch_allowlist=["shared.example"],
        )
        same = parent.narrowed(parent)
        for path in paths:
            for check_name in ("check_read", "check_list", "check_write"):
                if getattr(same, check_name)(path).allowed != getattr(parent, check_name)(path).allowed:
                    fail(f"self narrowing changed {check_name} for {path!s}")
        for url in urls:
            if same.check_fetch(url).allowed != parent.check_fetch(url).allowed:
                fail(f"self narrowing changed fetch permission for {url!r}")
        for argv in argvs:
            if same.check_shell(argv).allowed != parent.check_shell(argv).allowed:
                fail(f"self narrowing changed shell permission for {argv!r}")

        left = parent.narrowed(first_ceiling).narrowed(second_ceiling)
        right = parent.narrowed(second_ceiling).narrowed(first_ceiling)
        for path in paths:
            for check_name in ("check_read", "check_list", "check_write"):
                if getattr(left, check_name)(path).allowed != getattr(right, check_name)(path).allowed:
                    fail(f"narrowing did not compose for {check_name} and {path!s}")
        for url in urls:
            if left.check_fetch(url).allowed != right.check_fetch(url).allowed:
                fail(f"narrowing did not compose for fetch {url!r}")
        for argv in argvs:
            if left.check_shell(argv).allowed != right.check_shell(argv).allowed:
                fail(f"narrowing did not compose for shell {argv!r}")
