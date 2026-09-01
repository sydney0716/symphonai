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
import time

from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_api.tools.shell_classify import classify

CANCEL_POLL_SECONDS = 0.05
CLEANUP_TIMEOUT_SECONDS = 1.0


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


class RunShellTool(LocalTool):
    """Run an allowlisted command, given as an argv list, and return its output."""

    @property
    def name(self) -> str:
        return "run_shell"

    @property
    def description(self) -> str:
        return "Run an allowlisted shell command (argv list) and return its output."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Command and arguments as a list of strings, e.g. "
                        "['git', 'status']. Never a single shell string."
                    ),
                },
            },
            "required": ["argv"],
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        argv = arguments.get("argv")
        entry = classify(argv) if isinstance(argv, list) else None
        if entry is None:
            return ToolMetadata(
                effect=ToolEffect.DESTRUCTIVE,
                concurrency_safe=False,
                paths=None,
            )
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=entry.concurrency_safe,
            paths=None,
        )

    def validate(self, arguments: dict) -> str | None:
        argv = arguments.get("argv")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(argument, str) for argument in argv
        ):
            return (
                "missing or invalid required argument: argv "
                "(must be a non-empty list of strings)"
            )
        return None

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        argv = tool_call.arguments.get("argv")
        decision = policy.check_shell(argv)
        if not decision.allowed:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=decision.reason)
        try:
            proc = subprocess.Popen(
                argv,
                shell=False,
                cwd=policy.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            deadline = time.monotonic() + policy.shell_timeout_seconds
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
                    raise subprocess.TimeoutExpired(argv, policy.shell_timeout_seconds)
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
