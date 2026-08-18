"""The run_shell local tool: executes an allowlisted argv command.

Every invocation is gated by `PermissionPolicy.check_shell()` before
`subprocess.run` is ever called, always with `shell=False` and a timeout.
A denied command returns `ToolResult(ok=False, ...)` without ever touching
`subprocess`.
"""

from __future__ import annotations

import subprocess

from orchestra_api.models import ToolCall, ToolResult
from orchestra_api.permissions import PermissionPolicy
from orchestra_api.tools.base import LocalTool

MAX_OUTPUT_CHARS = 20_000


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

    def execute(self, tool_call: ToolCall, policy: PermissionPolicy) -> ToolResult:
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
            proc = subprocess.run(
                argv,
                shell=False,
                cwd=policy.repo_root,
                capture_output=True,
                text=True,
                timeout=policy.shell_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=f"error running command: {exc}")
        output = (proc.stdout or "") + (proc.stderr or "")
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n...(truncated)"
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=proc.returncode == 0,
            content=output,
            error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
        )
