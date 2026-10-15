"""The run_shell local tool: executes an allowlisted argv command.

Every invocation is gated by `PermissionPolicy.check_shell()` before
`subprocess.Popen` is ever called, always with `shell=False` and a timeout.
A denied command returns `ToolResult(ok=False, ...)` without ever touching
`subprocess`.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_api.tools.shell_classify import classify
from symphonai_api.tools.shell_parse import commands_for_argv, parse_command_line

CANCEL_POLL_SECONDS = 0.05
CLEANUP_TIMEOUT_SECONDS = 1.0
SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def shell_executable() -> str:
    return "/bin/bash" if os.path.isfile("/bin/bash") else "/bin/sh"


def shell_environment() -> dict[str, str]:
    return {
        key: value for key, value in os.environ.items()
        if not any(part in key.upper() for part in ("KEY", "SECRET", "TOKEN"))
    }


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
        rules.extend(("(deny network*)", "(allow network* (local unix))"))
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


def _sandbox_available() -> bool:
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


class RunShellTool(LocalTool):
    """Run an allowlisted command line or argv list and return its output."""

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
        return None

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        command_line = tool_call.arguments.get("command")
        if isinstance(command_line, str):
            argv = [shell_executable(), "-c", command_line]
        else:
            argv = tool_call.arguments.get("argv")
        timeout_seconds = min(
            tool_call.arguments.get("timeout_seconds") or 120,
            policy.shell_timeout_seconds,
        )
        decision = policy.check_shell(argv)
        if not decision.allowed:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=decision.reason)
        command = argv
        sandbox_available = _sandbox_available()
        if policy.shell_sandbox and not sandbox_available:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error="sandbox requested but unavailable on this platform",
            )
        if policy.shell_sandbox or (policy.mode == "allow" and sandbox_available):
            command = [SANDBOX_EXEC, "-p", _seatbelt_profile(policy), *argv]
        env = shell_environment()
        try:
            proc = subprocess.Popen(
                command,
                shell=False,
                cwd=policy.repo_root,
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
        output = (stdout or "") + (stderr or "")
        limit = policy.shell_output_limit_chars
        if len(output) > limit:
            output = (
                output[:limit]
                + f"\n[output truncated: {len(output)} chars, over the {limit} char limit]"
            )
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=proc.returncode == 0,
            content=output,
            error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
        )
