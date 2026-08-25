"""The run_shell local tool: executes an allowlisted argv command.

Every invocation is gated by `PermissionPolicy.check_shell()` before
`subprocess.Popen` is ever called, always with `shell=False` and a timeout.
A denied command returns `ToolResult(ok=False, ...)` without ever touching
`subprocess`.
"""

from __future__ import annotations

import subprocess
import time

from orchestra_api.cancellation import CancellationToken, OperationCancelled
from orchestra_api.models import ToolCall, ToolResult
from orchestra_api.permissions import PermissionPolicy
from orchestra_api.tools.base import LocalTool

MAX_OUTPUT_CHARS = 20_000
CANCEL_POLL_SECONDS = 0.05


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

    def execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        if cancel is not None:
            cancel.raise_if_cancelled()
        argv = tool_call.arguments.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error="missing or invalid required argument: argv (must be a non-empty list of strings)",
            )
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
            )
            deadline = time.monotonic() + policy.shell_timeout_seconds
            while True:
                if cancel is not None and cancel.cancelled:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    proc.communicate()
                    raise OperationCancelled
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate()
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    proc.kill()
                    proc.communicate()
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
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n...(truncated)"
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=proc.returncode == 0,
            content=output,
            error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
        )
