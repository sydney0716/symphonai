"""Workspace-backed checks for shell."""

from __future__ import annotations

import os
import socket
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest.mock as mock
from pathlib import Path
from symphonai_api.config import ConfigError, load_config
from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.models import ToolCall
from symphonai_api.permissions import DEFAULT_SHELL_OUTPUT_CHARS, PermissionPolicy
from symphonai_api.tools.shell import RunShellTool, _terminate_process_group
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


def _sandbox_skip() -> bool:
    if sys.platform == "darwin" and os.path.isfile("/usr/bin/sandbox-exec"):
        probe = subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", "(version 1) (allow default)", "/usr/bin/true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            return False
        print(f"SKIP: sandbox-exec could not apply a profile (exit {probe.returncode})")
        return True
    print("SKIP: macOS /usr/bin/sandbox-exec is unavailable")
    return True


def _sandbox_policy(root: Path, *, network: bool = False) -> PermissionPolicy:
    return PermissionPolicy(
        repo_root=root,
        mode="allow",
        shell_enabled=True,
        shell_allowlist=[("touch",), (sys.executable,), ("rg",), ("/usr/bin/grep",)],
        shell_sandbox=True,
        sandbox_network=network,
    )


@check("shell.sandbox_write_confinement")
def check_shell_sandbox_write_confinement() -> None:
    if _sandbox_skip():
        return
    with (
        tempfile.TemporaryDirectory(prefix='symphonai "sandbox ') as directory,
        tempfile.TemporaryDirectory(dir="/private/var/tmp") as outside_directory,
    ):
        root = Path(directory) / 'repo "root'
        root.mkdir()
        policy = _sandbox_policy(root)
        for path in (
            root / "ok",
            Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "symphonai-sandbox-ok",
            Path("/tmp") / "symphonai-sandbox-ok",
        ):
            result = RunShellTool().execute(
                ToolCall(id="sandbox-write", name="run_shell", arguments={"argv": ["touch", str(path)]}),
                policy,
            )
            if not result.ok or not path.exists():
                fail(f"sandbox refused an allowed write to {path}: {result!r}")
            path.unlink()
        denied = Path(outside_directory) / "no"
        result = RunShellTool().execute(
            ToolCall(id="sandbox-outside", name="run_shell", arguments={"argv": ["touch", str(denied)]}),
            policy,
        )
        if result.ok or denied.exists():
            fail(f"sandbox allowed a write outside repo and temp roots: {result!r}")


@check("shell.sandbox_network_policy")
def check_shell_sandbox_network_policy() -> None:
    if _sandbox_skip():
        return
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(2)
        port = listener.getsockname()[1]
        code = f"import socket; socket.create_connection(('127.0.0.1', {port}), timeout=1).close()"
        try:
            for network, allowed in ((False, False), (True, True)):
                result = RunShellTool().execute(
                    ToolCall(id="sandbox-network", name="run_shell", arguments={
                        "argv": [sys.executable, "-c", code],
                    }),
                    _sandbox_policy(root, network=network),
                )
                if result.ok != allowed:
                    fail(f"sandbox network={network} returned the wrong result: {result!r}")
                if allowed:
                    connection, _ = listener.accept()
                    connection.close()
        finally:
            listener.close()


@check("shell.sandbox_unavailable_fails_closed")
def check_shell_sandbox_unavailable_fails_closed() -> None:
    policy = _sandbox_policy(Path.cwd())
    with (
        mock.patch("symphonai_api.tools.shell.sys.platform", "linux"),
        mock.patch("symphonai_api.tools.shell.subprocess.Popen") as popen,
    ):
        result = RunShellTool().execute(
            ToolCall(id="sandbox-unavailable", name="run_shell", arguments={"argv": ["touch", "never"]}),
            policy,
        )
    if result.ok or result.error != "sandbox requested but unavailable on this platform" or popen.called:
        fail(f"unavailable sandbox did not refuse before spawning: {result!r}")


@check("shell.sandbox_opt_in_preserves_plain_argv")
def check_shell_sandbox_opt_in_preserves_plain_argv() -> None:
    argv = ["echo", "plain"]
    process = mock.Mock(returncode=0)
    process.poll.return_value = 0
    process.communicate.return_value = ("", "")
    policy = PermissionPolicy(
        repo_root=Path.cwd(), mode="allow", shell_enabled=True,
        shell_allowlist=[("echo",)],
    )
    with (
        mock.patch("symphonai_api.tools.shell._sandbox_available", return_value=False),
        mock.patch("symphonai_api.tools.shell.subprocess.Popen", return_value=process) as popen,
    ):
        result = RunShellTool().execute(
            ToolCall(id="plain-shell", name="run_shell", arguments={"argv": argv}), policy,
        )
    if not result.ok or popen.call_args.args[0] != argv:
        fail(f"unsandboxed command argv changed: {popen.call_args!r}, {result!r}")


@check("shell.allow_mode_sandboxes_when_available")
def check_allow_mode_sandboxes_when_available() -> None:
    policy = PermissionPolicy(
        repo_root=Path.cwd(), mode="allow", shell_enabled=True,
        shell_allowlist=[()],
    )
    process = mock.Mock(returncode=0)
    process.poll.return_value = 0
    process.communicate.return_value = ("", "")
    with (
        mock.patch("symphonai_api.tools.shell.sys.platform", "darwin"),
        mock.patch("symphonai_api.tools.shell.os.path.isfile", return_value=True),
        mock.patch("symphonai_api.tools.shell.subprocess.run", return_value=subprocess.CompletedProcess([], 0)),
        mock.patch("symphonai_api.tools.shell.subprocess.Popen", return_value=process) as popen,
    ):
        result = RunShellTool().execute(
            ToolCall(id="allow-sandbox", name="run_shell", arguments={"argv": ["echo", "ok"]}),
            policy,
        )
    command = popen.call_args.args[0]
    if not result.ok or command[:2] != ["/usr/bin/sandbox-exec", "-p"] or command[-2:] != ["echo", "ok"]:
        fail(f"allow mode did not sandbox its command: {command!r}, {result!r}")


@check("shell.allow_mode_unavailable_sandbox_falls_back")
def check_allow_mode_unavailable_sandbox_falls_back() -> None:
    argv = ["echo", "plain"]
    process = mock.Mock(returncode=0)
    process.poll.return_value = 0
    process.communicate.return_value = ("", "")
    policy = PermissionPolicy(
        repo_root=Path.cwd(), mode="allow", shell_enabled=True,
        shell_allowlist=[()],
    )
    with (
        mock.patch("symphonai_api.tools.shell.sys.platform", "linux"),
        mock.patch("symphonai_api.tools.shell.subprocess.Popen", return_value=process) as popen,
    ):
        result = RunShellTool().execute(
            ToolCall(id="allow-plain", name="run_shell", arguments={"argv": argv}),
            policy,
        )
    if not result.ok or popen.call_args.args[0] != argv:
        fail(f"allow mode did not run unconfined without a sandbox: {popen.call_args!r}, {result!r}")

    required = PermissionPolicy(
        repo_root=Path.cwd(), mode="allow", shell_enabled=True,
        shell_allowlist=[()], shell_sandbox=True,
    )
    with (
        mock.patch("symphonai_api.tools.shell.sys.platform", "linux"),
        mock.patch("symphonai_api.tools.shell.subprocess.Popen") as strict_popen,
    ):
        refused = RunShellTool().execute(
            ToolCall(id="strict-sandbox", name="run_shell", arguments={"argv": argv}),
            required,
        )
    if refused.ok or refused.error != "sandbox requested but unavailable on this platform" or strict_popen.called:
        fail(f"explicit sandbox setting did not fail closed: {refused!r}")


@check("shell.allow_mode_confinement")
def check_allow_mode_confinement() -> None:
    if _sandbox_skip():
        return
    with (
        tempfile.TemporaryDirectory() as directory,
        tempfile.TemporaryDirectory(dir="/private/var/tmp") as outside_directory,
    ):
        root = Path(directory)
        policy = PermissionPolicy(
            repo_root=root, mode="allow", shell_enabled=True,
            shell_allowlist=[(sys.executable,)],
        )
        code = "from pathlib import Path; Path(__import__('sys').argv[1]).write_text('ok')"
        for target, allowed in ((root / "inside", True), (Path(outside_directory) / "outside", False)):
            result = RunShellTool().execute(
                ToolCall(id="allow-confined", name="run_shell", arguments={
                    "argv": [sys.executable, "-c", code, str(target)],
                }),
                policy,
            )
            if result.ok != allowed or target.exists() != allowed:
                fail(f"allow-mode sandbox write to {target} returned {result!r}")


@check("shell.sandbox_command_exit_is_ordinary")
def check_shell_sandbox_command_exit_is_ordinary() -> None:
    if _sandbox_skip():
        return
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "empty.txt").write_text("present text\n", encoding="utf-8")
        result = RunShellTool().execute(
            ToolCall(id="sandbox-no-match", name="run_shell", arguments={"argv": ["/usr/bin/grep", "absent-pattern", str(Path(directory) / "empty.txt")]}),
            _sandbox_policy(Path(directory)),
        )
    if result.ok or result.error != "exit code 1" or "sandbox" in (result.error or "").casefold():
        fail(f"ordinary exit status was misclassified as a sandbox failure: {result!r}")


@check("config.sandbox_settings")
def check_sandbox_settings() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        project_config = root / ".symphonai" / "config.toml"
        project_config.parent.mkdir()
        project_config.write_text("[sandbox]\nshell = true\nnetwork = false\n", encoding="utf-8")
        resolved = load_config(repo_root=root, home=root / "home")
        if resolved.get("sandbox.shell") is not True or resolved.get("sandbox.network") is not False:
            fail(f"sandbox settings were not flattened: {resolved.values!r}")
        for content, key in (
            ('[sandbox]\nshell = "yes"\n', "sandbox.shell"),
            ("[sandbox]\nother = true\n", "sandbox.other"),
        ):
            project_config.write_text(content, encoding="utf-8")
            try:
                load_config(repo_root=root, home=root / "home")
            except ConfigError as exc:
                if key not in str(exc):
                    fail(f"sandbox validation error omitted {key}: {exc!r}")
            else:
                fail(f"invalid sandbox setting was accepted for {key}")


@check("shell.process_group_fallback")
def check_shell_process_group_fallback() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        shell_token = CancellationToken()
        shell_policy = PermissionPolicy(
            repo_root=root,
            shell_enabled=True,
            shell_allowlist=[(sys.executable,)],
        )
        same_group_proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            child_pgid = os.getpgid(same_group_proc.pid)
            current_pgid = os.getpgid(0)
            if child_pgid != current_pgid:
                fail(
                    "same-group termination test did not exercise the guard: "
                    f"child={child_pgid}, current={current_pgid}"
                )
            with mock.patch("symphonai_api.tools.shell.os.killpg") as killpg_mock:
                _terminate_process_group(same_group_proc)
            if killpg_mock.called:
                fail("same-group termination attempted to signal SymphonAI's process group")
            try:
                same_group_proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                fail("same-group termination did not kill the child process")
        finally:
            if same_group_proc.poll() is None:
                same_group_proc.kill()
                same_group_proc.wait(timeout=1.0)

@check("shell.cancellation_reaps_child")
def check_shell_cancellation_reaps_child() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        shell_token = CancellationToken()
        shell_policy = PermissionPolicy(
            repo_root=root,
            shell_enabled=True,
            shell_allowlist=[(sys.executable,)],
        )
        real_popen = subprocess.Popen
        children: list[subprocess.Popen] = []

        def _capturing_popen(*args, **kwargs):  # noqa: ANN002, ANN003
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child

        shell_timer = threading.Timer(0.05, shell_token.cancel)
        shell_started = time.monotonic()
        shell_timer.start()
        try:
            with mock.patch(
                "symphonai_api.tools.shell.subprocess.Popen",
                side_effect=_capturing_popen,
            ), mock.patch(
                "symphonai_api.tools.shell._sandbox_available", return_value=False,
            ):
                try:
                    RunShellTool().execute(
                        ToolCall(
                            id="cancel-shell",
                            name="run_shell",
                            arguments={
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    "import time; time.sleep(30)",
                                ]
                            },
                        ),
                        shell_policy,
                        cancel=shell_token,
                    )
                except OperationCancelled:
                    pass
                else:
                    fail("cancelled run_shell returned a ToolResult")
        finally:
            shell_timer.cancel()
            shell_timer.join()
        if time.monotonic() - shell_started >= 1.0:
            fail("cancelled run_shell did not return promptly")
        if len(children) != 1 or children[0].poll() is None:
            fail(f"cancelled run_shell left its child running: {children!r}")

@check("shell.cancellation_kills_group")
def check_shell_cancellation_kills_group() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        shell_policy = PermissionPolicy(
            repo_root=root,
            shell_enabled=True,
            shell_allowlist=[(sys.executable,)],
        )
        descendant_pid_path = root / "descendant.pid"
        descendant_token = CancellationToken()
        descendant_script = (
            "import subprocess, sys, time; "
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(30)'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            "open(sys.argv[1], 'w').write(str(child.pid)); "
            "time.sleep(30)"
        )
        descendant_timer = threading.Timer(0.2, descendant_token.cancel)
        descendant_timer.start()
        try:
            try:
                shell_tool_for_tree = RunShellTool()
                shell_tool_for_tree.execute(
                    ToolCall(
                        id="cancel-shell-tree",
                        name="run_shell",
                        arguments={
                            "argv": [
                                sys.executable,
                                "-c",
                                descendant_script,
                                str(descendant_pid_path),
                            ]
                        },
                    ),
                    shell_policy,
                    cancel=descendant_token,
                )
            except OperationCancelled:
                pass
            else:
                fail("cancelled process-tree command returned a ToolResult")
        finally:
            descendant_timer.cancel()
            descendant_timer.join()
        if not descendant_pid_path.exists():
            fail("process-tree command did not record its descendant pid before cancellation")
        descendant_pid = int(descendant_pid_path.read_text())
        descendant_deadline = time.monotonic() + 2.0
        descendant_alive = True
        while time.monotonic() < descendant_deadline:
            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                descendant_alive = False
                break
            time.sleep(0.02)
        if descendant_alive:
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            fail(f"cancelled run_shell left descendant pid {descendant_pid} alive")

@check("shell.cancellation_bounded")
def check_shell_cancellation_bounded() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        shell_policy = PermissionPolicy(
            repo_root=root,
            shell_enabled=True,
            shell_allowlist=[(sys.executable,)],
        )
        inherited_pipe_token = CancellationToken()
        inherited_pipe_script = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)']); "
            "time.sleep(30)"
        )
        inherited_pipe_timer = threading.Timer(0.2, inherited_pipe_token.cancel)
        inherited_pipe_started = time.monotonic()
        inherited_pipe_timer.start()
        try:
            try:
                RunShellTool().execute(
                    ToolCall(
                        id="cancel-shell-inherited-pipe",
                        name="run_shell",
                        arguments={
                            "argv": [sys.executable, "-c", inherited_pipe_script]
                        },
                    ),
                    shell_policy,
                    cancel=inherited_pipe_token,
                )
            except OperationCancelled:
                pass
            else:
                fail("cancelled inherited-pipe command returned a ToolResult")
        finally:
            inherited_pipe_timer.cancel()
            inherited_pipe_timer.join()
        inherited_pipe_elapsed = time.monotonic() - inherited_pipe_started
        if inherited_pipe_elapsed >= 1.5:
            fail(
                "cancelled run_shell blocked on an inherited pipe for "
                f"{inherited_pipe_elapsed:.2f}s"
            )

@check("shell.execution_paths")
def check_shell_execution_paths() -> None:
    with workspace() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        shell_policy = PermissionPolicy(
            repo_root=root,
            shell_enabled=True,
            shell_allowlist=[(sys.executable,)],
        )
        # -- run_shell behaviour unchanged by the subprocess.run -> Popen rewrite.
        # These are the paths the rewrite could silently break; none of them were
        # covered before, so a regression would have shipped green.
        shell_tool = RunShellTool()

        def _shell(argv: list[str], policy: PermissionPolicy = shell_policy) -> ToolResult:
            return shell_tool.execute(
                ToolCall(id="shell-behaviour", name="run_shell", arguments={"argv": argv}),
                policy,
            )

        success = _shell([sys.executable, "-c", "print('to stdout')"])
        if not success.ok or success.content != "to stdout\n" or success.error is not None:
            fail(f"run_shell success path changed: {success!r}")

        merged = _shell(
            [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"]
        )
        if "out" not in merged.content or "err" not in merged.content:
            fail(f"run_shell no longer merges stdout and stderr: {merged!r}")

        failing = _shell([sys.executable, "-c", "print('partial'); raise SystemExit(7)"])
        if failing.ok or failing.error != "exit code 7" or "partial" not in failing.content:
            fail(f"run_shell nonzero-exit path changed: {failing!r}")

        oversized = _shell(
            [
                sys.executable,
                "-c",
                f"print('x' * {DEFAULT_SHELL_OUTPUT_CHARS + 500})",
            ]
        )
        oversized_original_length = DEFAULT_SHELL_OUTPUT_CHARS + 501
        oversized_notice = (
            f"\n[output truncated: {oversized_original_length} chars, over the "
            f"{DEFAULT_SHELL_OUTPUT_CHARS} char limit]"
        )
        if not oversized.content.endswith(oversized_notice):
            fail(f"run_shell truncation notice changed: {oversized.content[-120:]!r}")
        if len(oversized.content) != DEFAULT_SHELL_OUTPUT_CHARS + len(oversized_notice):
            fail(f"run_shell truncated to the wrong length: {len(oversized.content)}")

        variable_argv = [sys.executable, "-c", "print('y' * 2500)"]
        small_output_policy = PermissionPolicy(
            repo_root=root,
            shell_enabled=True,
            shell_allowlist=[(sys.executable,)],
            shell_output_limit_chars=1_000,
        )
        large_output_policy = PermissionPolicy(
            repo_root=root,
            shell_enabled=True,
            shell_allowlist=[(sys.executable,)],
            shell_output_limit_chars=2_000,
        )
        small_output = _shell(variable_argv, small_output_policy)
        large_output = _shell(variable_argv, large_output_policy)
        variable_original_length = 2_501
        small_notice = (
            f"\n[output truncated: {variable_original_length} chars, over the "
            "1000 char limit]"
        )
        large_notice = (
            f"\n[output truncated: {variable_original_length} chars, over the "
            "2000 char limit]"
        )
        if (
            len(small_output.content) != 1_000 + len(small_notice)
            or not small_output.content.endswith(small_notice)
            or len(large_output.content) != 2_000 + len(large_notice)
            or not large_output.content.endswith(large_notice)
            or len(small_output.content) == len(large_output.content)
        ):
            fail(
                "run_shell did not read its output bound from each policy: "
                f"small={len(small_output.content)}, large={len(large_output.content)}"
            )

        timing_out = shell_tool.execute(
            ToolCall(
                id="shell-timeout",
                name="run_shell",
                arguments={"argv": [sys.executable, "-c", "import time; time.sleep(30)"]},
            ),
            PermissionPolicy(
                repo_root=root,
                shell_enabled=True,
                shell_allowlist=[(sys.executable,)],
                shell_timeout_seconds=0.3,
            ),
        )
        if timing_out.ok or not (timing_out.error or "").startswith("error running command:"):
            fail(f"run_shell timeout path changed: {timing_out!r}")


@check("shell.timeout_seconds")
def check_shell_timeout_seconds() -> None:
    with workspace() as ws:
        tool = RunShellTool()
        argv = [sys.executable, "-c", "pass"]

        class TimedOutProcess:
            returncode = None
            pid = 1

            def poll(self):
                return None

            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(argv, timeout)

        cases = (
            ({}, 120.0),
            ({"timeout_seconds": 900}, 600.0),
            ({"timeout_seconds": 5}, 5.0),
        )
        for extra, expected in cases:
            arguments = {"argv": argv, **extra}
            policy = PermissionPolicy(
                repo_root=ws.root,
                shell_enabled=True,
                shell_allowlist=[(sys.executable,)],
            )
            if policy.shell_timeout_seconds != 600.0:
                fail(f"default shell timeout was {policy.shell_timeout_seconds!r}, expected 600")
            with (
                mock.patch("symphonai_api.tools.shell._sandbox_available", return_value=False),
                mock.patch("symphonai_api.tools.shell.subprocess.Popen", return_value=TimedOutProcess()),
                mock.patch("symphonai_api.tools.shell._terminate_process_group"),
                mock.patch("symphonai_api.tools.shell.time.monotonic", side_effect=[0, 0, expected]),
            ):
                result = tool.execute(
                    ToolCall(id="shell-timeout-limit", name="run_shell", arguments=arguments),
                    policy,
                )
            if result.ok or f"after {expected:g}" not in (result.error or ""):
                fail(f"run_shell did not use effective timeout {expected:g}: {result!r}")

        invalid_values = (0, -1, True, "5")
        for value in invalid_values:
            error = tool.validate({"argv": argv, "timeout_seconds": value})
            if not error or "timeout_seconds" not in error:
                fail(f"run_shell accepted invalid timeout_seconds {value!r}")
