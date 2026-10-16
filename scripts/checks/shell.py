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
from symphonai_api.jobs import JobManager
from symphonai_api.models import ToolCall
from symphonai_api.permissions import DEFAULT_SHELL_OUTPUT_CHARS, PermissionPolicy
from symphonai_api.tools.shell import RunShellTool, _terminate_process_group, shell_environment
from symphonai_api.tools.shell_parse import ShellCommand, commands_for_argv, parse_command_line
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


@check("shell.background_job_lifecycle")
def check_background_job_lifecycle() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manager = JobManager()
        policy = PermissionPolicy(
            repo_root=root, mode="allow", shell_enabled=True, shell_allowlist=[()],
        )
        tool = RunShellTool(manager)
        call = lambda identifier, command: ToolCall(
            id=identifier, name="run_shell", arguments={"command": command, "background": True},
        )
        try:
            with mock.patch("symphonai_api.tools.shell._sandbox_available", return_value=False):
                started_at = time.monotonic()
                started = tool.execute(
                    call("job-steps", "for i in 1 2 3; do echo $i; sleep 1; done"), policy,
                )
                if not started.ok or time.monotonic() - started_at >= 0.5:
                    fail(f"background shell call did not return at once: {started!r}")
                job_id = started.content.split()[1].rstrip(".")
                time.sleep(0.5)
                first = manager.read(job_id)
                if first is None or first["status"] != "running" or "1\n" not in first["output"]:
                    fail(f"running job output was not available incrementally: {first!r}")
                time.sleep(3.0)
                second = manager.read(job_id, first["since"])
                if second is None or second["status"] != "exited" or second["exit_code"] != 0:
                    fail(f"background command did not exit successfully: {second!r}")
                if first["output"] + second["output"] != "1\n2\n3\n":
                    fail(f"job output offsets lost or duplicated text: {first!r}, {second!r}")

                stopped = tool.execute(call("job-sleep", "sleep 60"), policy)
                stopped_id = stopped.content.split()[1].rstrip(".")
                process = manager.get(stopped_id).process
                stop_result = manager.stop(stopped_id)
                if stop_result is None or stop_result["status"] != "stopped":
                    fail(f"job_stop did not stop a running job: {stop_result!r}")
                try:
                    os.killpg(os.getpgid(process.pid), 0)
                except ProcessLookupError:
                    pass
                else:
                    fail("stopped job process group is still running")

                large = tool.execute(ToolCall(
                    id="job-large", name="run_shell",
                    arguments={"argv": [sys.executable, "-c", "import sys; sys.stdout.write('x' * 150000)"], "background": True},
                ), policy)
                if not large.ok:
                    fail(f"large background job did not start: {large!r}")
                large_id = large.content.split()[1].rstrip(".")
                _wait_job = lambda: manager.get(large_id).status == "exited"
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not _wait_job():
                    time.sleep(0.01)
                large_output = manager.read(large_id, 0, 100_000)
                job = manager.get(large_id)
                if (
                    job.total_chars != 150000 or len(job.output) != 100000
                    or large_output is None or not large_output["output_dropped"]
                    or len(large_output["output"]) != 100_000
                ):
                    fail(
                        "large job output retention was incorrect: "
                        f"total={job.total_chars}, kept={len(job.output)}, read={len(large_output['output'])}, "
                        f"dropped={large_output['output_dropped']}"
                    )

                running = [manager.start(["sleep", "60"], str(root), shell_environment()) for _ in range(8)]
                ninth = tool.execute(call("job-limit", "sleep 60"), policy)
                if ninth.ok or "8" not in (ninth.error or ""):
                    fail(f"ninth concurrent job was not refused: {ninth!r}")
                for running_id in running:
                    manager.stop(running_id)
        finally:
            manager.close()


@check("shell.background_job_policy")
def check_background_job_policy() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manager = JobManager()
        tool = RunShellTool(manager)
        allow_policy = PermissionPolicy(
            repo_root=root, mode="allow", shell_enabled=True, shell_allowlist=[()],
        )
        with mock.patch("symphonai_api.tools.shell._sandbox_available", return_value=False):
            unavailable = RunShellTool().execute(ToolCall(
                id="background-unavailable", name="run_shell",
                arguments={"command": "echo no manager", "background": True},
            ), allow_policy)
            if unavailable.ok or unavailable.error != "background jobs are not available here":
                fail(f"background shell without a manager was not refused clearly: {unavailable!r}")
            refused = tool.execute(ToolCall(
                id="background-rm", name="run_shell",
                arguments={"command": "rm x", "background": True},
            ), allow_policy)
            if refused.ok or "always-deny" not in (refused.error or "") or manager.list():
                fail(f"background rm bypassed the always-deny policy: {refused!r}")
            if tool.metadata({"command": "echo ok"}) != tool.metadata({"command": "echo ok", "background": True}):
                fail("background shell metadata differed from the foreground call")

            approvals = []
            def approve(request):
                approvals.append((request, len(manager.list())))
                return True
            ask_policy = PermissionPolicy(
                repo_root=root, mode="ask", shell_enabled=True,
                approval_callback=approve,
            )
            asked = tool.execute(ToolCall(
                id="background-approval", name="run_shell",
                arguments={"command": "echo approved", "background": True},
            ), ask_policy)
            if not asked.ok or len(approvals) != 1 or approvals[0][1] != 0 or len(manager.list()) != 1:
                fail(f"background command did not ask before starting: {approvals!r}, {asked!r}")
        manager.close()


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


@check("shell.command_lines_parse")
def check_command_lines_parse() -> None:
    cases = (
        ("git status && git diff --stat", (ShellCommand(("git", "status")), ShellCommand(("git", "diff", "--stat")))),
        ("pytest -x | tail -n 20", (ShellCommand(("pytest", "-x")), ShellCommand(("tail", "-n", "20")))),
        ("python3 -m unittest -v > out.txt 2>&1; echo DONE", (
            ShellCommand(("python3", "-m", "unittest", "-v"), True),
            ShellCommand(("echo", "DONE")),
        )),
        ("cmd 2>/dev/null", (ShellCommand(("cmd",)),)),
        ("FOO=1 make test", (ShellCommand(("make", "test")),)),
        ('echo "a | b" \'c && d\'', (ShellCommand(("echo", "a | b", "c && d")),)),
        ("echo a#b", (ShellCommand(("echo", "a#b")),)),
        ("ls\nrm x", (ShellCommand(("ls",)), ShellCommand(("rm", "x")))),
        ('python3 -c "a\nb"', (ShellCommand(("python3", "-c", "a\nb")),)),
        ('bash -lc "make && make test"', (ShellCommand(("make",)), ShellCommand(("make", "test")))),
        ("cmd 2>&1", (ShellCommand(("cmd",)),)),
        ("cmd > /dev/null", (ShellCommand(("cmd",)),)),
        ("bash -o pipefail -c 'make && make test'", (ShellCommand(("make",)), ShellCommand(("make", "test")))),
        ("timeout 5 bash -c 'pytest > out.txt'", (ShellCommand(("timeout", "5", "pytest"), True),)),
        ("echo '$X'", (ShellCommand(("echo", "$X")),)),
        ("command -v curl", (ShellCommand(("command", "-v", "curl")),)),
        ("find . -name open", (ShellCommand(("find", ".", "-name", "open")),)),
        ("find . -exec wc -l {} +", (ShellCommand(("find", ".", "-exec", "wc", "-l", "{}", "+")),)),
        ("find . -exec rm {} ';'", (ShellCommand(("find", ".", "-exec", "rm", "{}", ";")),)),
        ("git show stash@{0}", (ShellCommand(("git", "show", "stash@{0}")),)),
        ("bash -l -c 'rm x'", (ShellCommand(("rm", "x")),)),
        ("bash -o pipefail -c 'git push'", (ShellCommand(("git", "push")),)),
        ("env sh -c 'rm x'", (ShellCommand(("env", "rm", "x")),)),
        ("if [ -f x ]; then rm x; fi", (
            ShellCommand(("[", "-f", "x", "]")), ShellCommand(("rm", "x")), ShellCommand((":" ,)),
        )),
        ("[ -f setup.py ] && echo yes", (
            ShellCommand(("[", "-f", "setup.py", "]")), ShellCommand(("echo", "yes")),
        )),
        ("[[ -d build ]] || mkdir build", (
            ShellCommand(("[[", "-d", "build", "]]")), ShellCommand(("mkdir", "build")),
        )),
        ("bash --version", (ShellCommand(("bash", "--version")),)),
        ("bash -c 'rm x' --help", (ShellCommand(("rm", "x")),)),
        ("sh -c 'git push' --version", (ShellCommand(("git", "push")),)),
        ("timeout 5 bash -c 'curl x' --help", (ShellCommand(("timeout", "5", "curl", "x")),)),
        ("bash --help -c 'rm x'", (ShellCommand(("bash", "--help", "-c", "rm x")),)),
        ("env bash --version", (ShellCommand(("env", "bash", "--version")),)),
        ("command -v bash", (ShellCommand(("command", "-v", "bash")),)),
        ("find . -name sh", (ShellCommand(("find", ".", "-name", "sh")),)),
        ("for f in a b; do echo $f; done", (
            ShellCommand(("for", "f", "in", "a", "b")), ShellCommand(("echo", "$f")), ShellCommand((":" ,)),
        )),
    )
    for text, expected in cases:
        actual = parse_command_line(text)
        if actual != expected:
            fail(f"command line parsed incorrectly: {text!r}: {actual!r}")
    for text in (
        "echo $(id)", "echo `id`", "diff <(a) <(b)", "(cd x && make)",
        "cat <<EOF", "sleep 5 &", 'echo "open', "one && && two",
        "echo 'rm x' | bash", "bash -s < s.sh", "export X=rm; $X x",
        "exec $X x", "eval 'rm x'", "{rm,x}",
        "echo '$X' && /bin/r?", "{", "}",
    ):
        if parse_command_line(text) is not None:
            fail(f"uncheckable command line was accepted: {text!r}")
    for argv in (
        ["timeout", "60", "bash", "-c", "git push"],
        ["env", "sh", "-c", "rm x"],
        ["bash", "-s"],
    ):
        parsed = commands_for_argv(argv)
        if argv[-1] in {"git push", "rm x"} and parsed is None:
            fail(f"wrapped shell command was not expanded: {argv!r}")
        if argv[-1] == "-s" and parsed is not None:
            fail(f"shell reading its script from stdin was accepted: {argv!r}")
    if commands_for_argv(["echo", "a|b"]) != (ShellCommand(("echo", "a|b")),):
        fail("argv words were incorrectly split on shell operators")


@check("shell.command_argument")
def check_command_argument() -> None:
    tool = RunShellTool()
    policy = PermissionPolicy(
        repo_root=Path.cwd(), mode="allow", shell_enabled=True,
        shell_allowlist=[()],
    )
    result = tool.execute(ToolCall(
        id="command-line", name="run_shell",
        arguments={"command": "printf 'a\\nb\\n' | wc -l"},
    ), policy)
    if not result.ok or result.content.strip() != "2":
        fail(f"shell command line did not execute its pipeline: {result!r}")
    cases = (
        ({"command": ""}, "missing or invalid required argument: command (must be a non-empty string)"),
        ({"argv": ["echo"], "command": "echo"}, "pass either argv or command, not both"),
        ({}, "missing required argument: command or argv"),
    )
    for arguments, expected in cases:
        if tool.validate(arguments) != expected:
            fail(f"command argument validation changed for {arguments!r}: {tool.validate(arguments)!r}")


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
            for network, allowed in ((False, True), (True, True)):
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

@check("shell.secrets_withheld_from_env")
def check_shell_secrets_withheld_from_env() -> None:
    variables = {
        "SYMPHONAI_PROBE_API_KEY": "probe-api-key-value",
        "symphonai_probe_token": "probe-token-value",
        "SYMPHONAI_PROBE_SECRET_X": "probe-secret-value",
        "SYMPHONAI_PROBE_PLAIN": "probe-plain-value",
    }
    policy = PermissionPolicy(
        repo_root=Path.cwd(), mode="allow", shell_enabled=True,
        shell_allowlist=[(sys.executable,)],
    )
    code = (
        "import os; "
        "print('\\n'.join(f'{key}={os.environ.get(key, \"<missing>\")}' "
        "for key in " + repr(list(variables)) + "))"
    )
    with mock.patch.dict(os.environ, variables):
        result = RunShellTool().execute(
            ToolCall(id="shell-env", name="run_shell", arguments={
                "argv": [sys.executable, "-c", code],
            }),
            policy,
        )
    if not result.ok:
        fail(f"environment probe command failed: {result!r}")
    for key in list(variables)[:3]:
        if variables[key] in result.content:
            fail(f"shell child received secret environment value for {key}")
    if variables["SYMPHONAI_PROBE_PLAIN"] not in result.content:
        fail(f"shell child did not receive an ordinary environment variable: {result.content!r}")


@check("shell.stdin_closed")
def check_shell_stdin_closed() -> None:
    code = """import sys
from pathlib import Path
from symphonai_api.models import ToolCall
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.shell import RunShellTool
policy = PermissionPolicy(repo_root=Path.cwd(), mode="allow", shell_enabled=True, shell_allowlist=[(sys.executable,)])
result = RunShellTool().execute(ToolCall(id="shell-stdin", name="run_shell", arguments={"argv": [sys.executable, "-c", "import sys; print(sys.stdin.read(), end='')"]}), policy)
sys.stdout.write(result.content)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], input="leaked-input", text=True,
        capture_output=True, check=False,
    )
    if completed.returncode != 0 or completed.stdout:
        fail(f"shell command inherited its caller's stdin: {completed!r}")


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


@check("shell.sandbox_allows_loopback")
def check_sandbox_allows_loopback() -> None:
    if _sandbox_skip():
        return
    code = """
import http.server, threading, urllib.request, socket
server = http.server.HTTPServer(('127.0.0.1', 0), http.server.BaseHTTPRequestHandler)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
server.RequestHandlerClass = Handler
threading.Thread(target=server.serve_forever, daemon=True).start()
fetch = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
for host in ('127.0.0.1', 'localhost'):
    with fetch(f'http://{host}:{server.server_port}/', timeout=2) as response:
        assert response.status == 200
# Use a documentation address to avoid DNS or contacting a real public service.
resolve = socket.getaddrinfo
socket.getaddrinfo = lambda host, *args, **kwargs: resolve('203.0.113.1' if host == 'example.com' else host, *args, **kwargs)
try:
    fetch('http://example.com/', timeout=2)
except urllib.error.URLError as exc:
    assert isinstance(exc.reason, PermissionError), repr(exc)
else:
    raise AssertionError('external network was allowed')
server.shutdown()
print('loopback only')
"""
    with tempfile.TemporaryDirectory() as directory:
        result = RunShellTool().execute(ToolCall(
            id="loopback", name="run_shell", arguments={"argv": [sys.executable, "-c", code]},
        ), _sandbox_policy(Path(directory)))
    if not result.ok or result.content.strip() != "loopback only":
        fail(f"loopback sandbox failed: {result!r}")


@check("shell.project_venv_on_path")
def check_project_venv_on_path() -> None:
    from symphonai_host.broker import EventBroker
    from symphonai_host.run import HostRun

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        baseline = root / "baseline"
        baseline.mkdir()
        stub = baseline / "python"
        stub.write_text("#!/bin/sh\necho baseline python\n")
        stub.chmod(0o755)
        variables = {"PATH": str(baseline) + os.pathsep + os.defpath, "PYTHONHOME": "original",
                     "PROBE_KEY": "hidden", "PROBE_SECRET": "hidden", "PROBE_TOKEN": "hidden"}
        manager = JobManager()
        host = HostRun(None, PermissionPolicy(root), EventBroker(), sessions_root=root / "sessions")
        policy = PermissionPolicy(root, mode="allow", shell_enabled=True, shell_allowlist=[()])
        tool = RunShellTool(manager)
        try:
            with mock.patch.dict(os.environ, variables), mock.patch(
                "symphonai_api.tools.shell._sandbox_available", return_value=False,
            ):
                original = shell_environment()
                if shell_environment(root) != original:
                    fail("no-venv environment changed")
                plain = tool.execute(ToolCall(id="plain-python", name="run_shell", arguments={"command": "python"}), policy)
                if not plain.ok or plain.content != "baseline python\n":
                    fail(f"no-venv python changed: {plain!r}")
                for name in ("venv", ".venv"):
                    venv = root / name
                    (venv / "bin").mkdir(parents=True)
                    (venv / "bin/python").write_text("#!/bin/sh\necho venv python\n")
                    (venv / "bin/python").chmod(0o755)
                    expected = original if name == "venv" else previous
                    if shell_environment(root) != expected:
                        fail("incomplete virtualenv was activated")
                    (venv / "pyvenv.cfg").touch()
                    environment = shell_environment(root)
                    if (environment["PATH"].split(os.pathsep)[0] != str(venv / "bin")
                        or environment.get("VIRTUAL_ENV") != str(venv) or "PYTHONHOME" in environment
                        or any(key in environment for key in ("PROBE_KEY", "PROBE_SECRET", "PROBE_TOKEN"))):
                        fail("virtualenv environment or secret filtering differed")
                    previous = environment
                result = tool.execute(ToolCall(id="venv-python", name="run_shell", arguments={"command": "python"}), policy)
                if not result.ok or result.content != "venv python\n":
                    fail(f"agent did not use virtualenv: {result!r}")
                background = tool.execute(ToolCall(id="venv-job", name="run_shell", arguments={"command": "python", "background": True}), policy)
                if not background.ok:
                    fail(f"virtualenv background job failed: {background!r}")
                agent_id = background.content.split()[1].rstrip(".")
                person_id = host.start_job("python")
                for jobs, job_id in ((manager, agent_id), (host._person_jobs, person_id)):
                    deadline = time.monotonic() + 5
                    while jobs.get(job_id).status == "running" and time.monotonic() < deadline:
                        time.sleep(0.01)
                    job = jobs.read(job_id)
                    if job["exit_code"] != 0 or job["output"] != "venv python\n":
                        fail(f"background job did not use virtualenv: {job!r}")
                reply = host.run_user_command("python")
                if reply["exit_code"] != 0 or reply["output"] != "venv python\n":
                    fail(f"person command did not use virtualenv: {reply!r}")
        finally:
            manager.close()
            host.close()
