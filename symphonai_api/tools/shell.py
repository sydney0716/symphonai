"""The run_shell local tool: executes an allowlisted argv command.

Every invocation is gated by `PermissionPolicy.check_shell()` before
`subprocess.Popen` is ever called, always with `shell=False` and a timeout.
A denied command returns `ToolResult(ok=False, ...)` without ever touching
`subprocess`.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.environment import project_virtualenv
from symphonai_api.jobs import JobManager
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata, call_target
from symphonai_api.tools.shell_classify import classify
from symphonai_api.tools.shell_parse import commands_for_argv, parse_command_line

CANCEL_POLL_SECONDS = 0.05
CLEANUP_TIMEOUT_SECONDS = 1.0
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
BWRAP_EXEC = "bwrap"


def shell_executable() -> str:
    return "/bin/bash" if os.path.isfile("/bin/bash") else "/bin/sh"


def shell_environment(
    repo_root: str | Path | None = None, *, for_agent: bool = False,
) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not any(part in key.upper() for part in ("KEY", "SECRET", "TOKEN"))
    }
    virtualenv = None if repo_root is None else project_virtualenv(repo_root)
    if virtualenv is not None:
        environment["PATH"] = str(virtualenv / "bin") + os.pathsep + environment.get("PATH", os.defpath)
        environment["VIRTUAL_ENV"] = str(virtualenv)
        environment.pop("PYTHONHOME", None)
    if for_agent:
        environment.update({
            "PAGER": "cat",
            "GIT_PAGER": "cat",
            "GIT_EDITOR": "true",
            "EDITOR": "true",
            "VISUAL": "true",
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "never",
        })
    return environment


def _seatbelt_profile(policy: PermissionPolicy) -> str:
    def quote_path(path: str) -> str:
        return '"' + path.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r") + '"'

    writable = {
        str(policy.repo_root.resolve()),
        str(os.path.realpath(os.environ.get("TMPDIR", tempfile.gettempdir()))),
        str(Path("/private/tmp").resolve()),
        str(Path("/private/var/folders").resolve()),
    }
    rules = ["(version 1)", "(allow default)", "(deny file-write*)"]
    rules.extend(
        f"(allow file-write* (subpath {quote_path(path)}))"
        for path in sorted(writable)
    )
    rules.extend(("(allow file-write* (literal \"/dev/null\"))", "(allow file-write* (literal \"/dev/tty\"))"))
    if not policy.sandbox_network:
        rules.extend((
            "(deny network*)", "(allow network* (local unix))",
            '(allow network-bind (local ip "localhost:*"))',
            '(allow network-inbound (local ip "localhost:*"))',
            '(allow network-outbound (remote ip "localhost:*"))',
        ))
    return "\n".join(rules)


def _terminate_process_group(proc: subprocess.Popen) -> None:
    try:
        if hasattr(os, "killpg"):
            pgid = os.getpgid(proc.pid)
            if pgid != os.getpgid(0):
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        else:
            proc.kill()
    except (OSError, ProcessLookupError, PermissionError):
        pass


@lru_cache(maxsize=1)
def _linux_sandbox_available() -> bool:
    if shutil.which(BWRAP_EXEC) is None:
        return False
    try:
        probe = subprocess.run(
            [
                BWRAP_EXEC, "--ro-bind", "/", "/", "--dev", "/dev",
                "--proc", "/proc", "/bin/true",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _sandbox_available() -> bool:
    if sys.platform == "linux":
        return _linux_sandbox_available()
    if sys.platform != "darwin" or not os.path.isfile(SANDBOX_EXEC):
        return False
    try:
        probe = subprocess.run(
            [SANDBOX_EXEC, "-p", "(version 1) (allow default)", "/usr/bin/true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _bwrap_command(policy: PermissionPolicy, argv: list[str], cwd: Path | None = None) -> list[str]:
    root = str(policy.repo_root.resolve())
    working_directory = root if cwd is None else str(cwd.resolve())
    temporary = os.path.realpath(os.environ.get("TMPDIR", tempfile.gettempdir()))
    command = [
        BWRAP_EXEC,
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--bind", root, root,
        "--bind", temporary, temporary,
        "--bind", "/tmp", "/tmp",
    ]
    if not policy.sandbox_network:
        # Each call gets its own loopback; only server and client in one command can connect.
        command.append("--unshare-net")
    command.extend(("--die-with-parent", "--chdir", working_directory, "--", *argv))
    return command


class RunShellTool(LocalTool):
    """Run an allowlisted command line or argv list and return its output."""

    def __init__(self, jobs: JobManager | None = None) -> None:
        self._jobs = jobs
        self._cwd: Path | None = None

    @property
    def name(self) -> str:
        return "run_shell"

    @property
    def description(self) -> str:
        return (
            "Run an allowlisted shell command line or argv list and return its output. "
            "Timeout defaults to 120 seconds and cannot exceed the policy limit."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Run one program without a shell, as a list of strings, "
                        "e.g. ['git', 'status']."
                    ),
                },
                "command": {
                    "type": "string",
                    "description": (
                        "A shell command line where pipes, &&, ;, || and redirects work."
                    ),
                },
                "timeout_seconds": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Optional timeout in seconds; defaults to 120 and is capped by policy.",
                },
                "background": {
                    "type": "boolean",
                    "description": "Start the command as a background job.",
                },
            },
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        argv = arguments.get("argv")
        commands = (
            parse_command_line(arguments["command"])
            if "command" in arguments and isinstance(arguments["command"], str)
            else commands_for_argv(argv)
            if isinstance(argv, (list, tuple))
            else None
        )
        entries = None if commands is None else [
            classify(command.argv) if not command.writes else None
            for command in commands
        ]
        if not entries or any(entry is None for entry in entries):
            return ToolMetadata(
                effect=ToolEffect.DESTRUCTIVE,
                concurrency_safe=False,
                paths=None,
            )
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=all(entry.concurrency_safe for entry in entries if entry is not None),
            paths=None,
        )

    def validate(self, arguments: dict) -> str | None:
        has_argv = "argv" in arguments
        has_command = "command" in arguments
        if has_argv and has_command:
            return "pass either argv or command, not both"
        if has_argv:
            argv = arguments.get("argv")
            if not isinstance(argv, list) or not argv or not all(
                isinstance(argument, str) for argument in argv
            ):
                return (
                    "missing or invalid required argument: argv "
                    "(must be a non-empty list of strings)"
                )
        elif has_command:
            command = arguments.get("command")
            if not isinstance(command, str) or not command:
                return "missing or invalid required argument: command (must be a non-empty string)"
        else:
            return "missing required argument: command or argv"
        timeout_seconds = arguments.get("timeout_seconds")
        if "timeout_seconds" in arguments and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not timeout_seconds > 0
        ):
            return "invalid optional argument: timeout_seconds (must be a number greater than 0)"
        if "background" in arguments and not isinstance(arguments["background"], bool):
            return "invalid optional argument: background (must be a boolean)"
        return None

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        root = policy.repo_root.resolve()
        cwd = self._cwd or root
        reset_message = ""
        if not cwd.is_dir() or not cwd.is_relative_to(root):
            cwd = root
            reset_message = "Shell working directory reset to the project root."
        self._cwd = cwd
        command_line = tool_call.arguments.get("command")
        if isinstance(command_line, str):
            argv = [shell_executable(), "-c", command_line]
        else:
            argv = tool_call.arguments.get("argv")
        timeout_seconds = min(
            tool_call.arguments.get("timeout_seconds") or 120,
            policy.shell_timeout_seconds,
        )
        decision = policy.check_shell(argv, cwd=cwd)
        if not decision.allowed:
            error = decision.reason + (f"\n{reset_message}" if reset_message else "")
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=error)
        command = argv
        cwd_marker = f"\036SYMPHONAI_CWD_{os.urandom(12).hex()}_"
        if isinstance(command_line, str) and tool_call.arguments.get("background") is not True:
            command = [
                shell_executable(), "-c",
                f"{command_line}\n__symphonai_status=$?; printf '{cwd_marker}%s\\037' \"$PWD\"; exit \"$__symphonai_status\"",
            ]
        sandbox_available = _sandbox_available()
        if policy.shell_sandbox and not sandbox_available:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error="sandbox requested but unavailable on this platform",
            )
        if policy.shell_sandbox or (policy.mode == "allow" and sandbox_available):
            if sys.platform == "linux":
                command = _bwrap_command(policy, command, cwd)
            else:
                command = [SANDBOX_EXEC, "-p", _seatbelt_profile(policy), *command]
        env = shell_environment(policy.repo_root, for_agent=True)
        if tool_call.arguments.get("background") is True:
            if self._jobs is None:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error="background jobs are not available here",
                )
            try:
                job_id = self._jobs.start(
                    command, str(cwd), env,
                    label=call_target("run_shell", tool_call.arguments),
                )
            except RuntimeError as exc:
                return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=True,
                content=f"Started {job_id}. Read its output with job_output and stop it with job_stop.",
            )
        try:
            proc = subprocess.Popen(
                command,
                shell=False,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout_seconds
            while True:
                if cancel is not None and cancel.cancelled:
                    _terminate_process_group(proc)
                    try:
                        proc.communicate(timeout=CLEANUP_TIMEOUT_SECONDS)
                    except subprocess.TimeoutExpired:
                        pass
                    raise OperationCancelled
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate()
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _terminate_process_group(proc)
                    try:
                        proc.communicate(timeout=CLEANUP_TIMEOUT_SECONDS)
                    except subprocess.TimeoutExpired:
                        pass
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                delay = min(CANCEL_POLL_SECONDS, remaining)
                try:
                    stdout, stderr = proc.communicate(timeout=delay)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=f"error running command: {exc}")
        stdout = stdout or ""
        if isinstance(command_line, str):
            marker_at = stdout.rfind(cwd_marker)
            if marker_at >= 0 and stdout.endswith("\x1f"):
                captured = stdout[marker_at + len(cwd_marker):-1]
                stdout = stdout[:marker_at]
                ended = Path(captured).resolve()
                if ended.is_dir() and ended.is_relative_to(root):
                    self._cwd = ended
                else:
                    self._cwd = root
                    reset_message = "Shell working directory reset to the project root."
        if not self._cwd.is_dir() or not self._cwd.is_relative_to(root):
            self._cwd = root
            reset_message = "Shell working directory reset to the project root."
        output = stdout + (stderr or "")
        if self._cwd != root:
            output = f"(in {self._cwd.relative_to(root)})\n" + output
        limit = policy.shell_output_limit_chars
        if len(output) > limit:
            output = (
                output[:limit]
                + f"\n[output truncated: {len(output)} chars, over the {limit} char limit]"
            )
        if reset_message:
            output = (output + "\n" if output else "") + reset_message
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=proc.returncode == 0,
            content=output,
            error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
        )
