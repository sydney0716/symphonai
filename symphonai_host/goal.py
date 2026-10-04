"""Persistent goal state and cancellable completion checks."""

from __future__ import annotations

import os
import json
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from symphonai_api.events import Event
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata


GOAL_CHECK_TIMEOUT_SECONDS = 600
GOAL_OUTPUT_LIMIT = 4000
GOAL_MESSAGE_LIMIT = 2000


@dataclass
class Goal:
    objective: str
    check: tuple[str, ...]
    phase: str = "active"
    rounds: int = 0
    max_rounds: int = 10
    last_check: dict | None = None
    reason: str = ""

    def payload(self) -> dict:
        return {
            "objective": self.objective,
            "check": list(self.check),
            "phase": self.phase,
            "rounds": self.rounds,
            "max_rounds": self.max_rounds,
            "last_check": self.last_check,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, value: object) -> Goal | None:
        if not isinstance(value, dict):
            return None
        objective, check = value.get("objective"), value.get("check")
        if (
            not isinstance(objective, str)
            or not isinstance(check, list)
            or not all(isinstance(arg, str) for arg in check)
        ):
            return None
        phase = value.get("phase")
        rounds = value.get("rounds")
        max_rounds = value.get("max_rounds")
        if (
            not isinstance(phase, str)
            or phase not in {"active", "paused", "blocked", "complete"}
            or type(rounds) is not int
            or type(max_rounds) is not int
        ):
            return None
        last_check = value.get("last_check")
        if last_check is not None and not isinstance(last_check, dict):
            return None
        reason = value.get("reason", "")
        if not isinstance(reason, str):
            return None
        return cls(
            objective=objective,
            check=tuple(check),
            phase=phase,
            rounds=rounds,
            max_rounds=max_rounds,
            last_check=last_check,
            reason=reason,
        )


@dataclass(frozen=True)
class GoalChanged(Event):
    change: str = ""
    phase: str = ""
    rounds: int = 0
    max_rounds: int = 0
    reason: str = ""
    last_check: dict | None = None


class GetGoalTool(LocalTool):
    def __init__(self, read_goal: Callable[[], dict | None]) -> None:
        self._read_goal = read_goal

    @property
    def name(self) -> str:
        return "get_goal"

    @property
    def description(self) -> str:
        return "Read the current goal and its status."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def validate(self, arguments: dict) -> str | None:
        return None if not arguments else "get_goal takes no arguments"

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(effect=ToolEffect.READ_ONLY, concurrency_safe=True, paths=None)

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel=None,
    ) -> ToolResult:
        goal = self._read_goal()
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=True,
            content="No goal is set." if goal is None else json.dumps(goal, ensure_ascii=False),
        )


class UpdateGoalTool(LocalTool):
    def __init__(self, update: Callable[[str, str], tuple[bool, str]]) -> None:
        self._update = update

    @property
    def name(self) -> str:
        return "update_goal"

    @property
    def description(self) -> str:
        return "Report that the current goal is blocked or complete."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["blocked", "complete"]},
                "message": {
                    "type": "string",
                    "maxLength": GOAL_MESSAGE_LIMIT,
                    "description": "Why the goal is blocked or what was completed.",
                },
            },
            "required": ["status", "message"],
        }

    def validate(self, arguments: dict) -> str | None:
        if set(arguments) != {"status", "message"}:
            return "arguments must contain only status and message"
        status = arguments.get("status")
        if not isinstance(status, str) or status not in {"blocked", "complete"}:
            return 'status must be "blocked" or "complete"'
        message = arguments.get("message")
        if not isinstance(message, str) or not message.strip():
            return "message must be a non-blank string"
        if len(message) > GOAL_MESSAGE_LIMIT:
            return f"message must be at most {GOAL_MESSAGE_LIMIT} characters"
        return None

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(effect=ToolEffect.MUTATING, concurrency_safe=False, paths=None)

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel=None,
    ) -> ToolResult:
        ok, result = self._update(
            tool_call.arguments["status"], tool_call.arguments["message"],
        )
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=ok,
            content=result if ok else "",
            error=None if ok else result,
        )


def goal_tools(
    read_goal: Callable[[], dict | None],
    update: Callable[[str, str], tuple[bool, str]],
) -> dict[str, LocalTool]:
    return {
        "get_goal": GetGoalTool(read_goal),
        "update_goal": UpdateGoalTool(update),
    }


@dataclass
class GoalCheck:
    goal: Goal
    session_id: str
    cancel: threading.Event
    agent_id: str = ""
    interrupted: str = ""
    cleared: bool = False
    process: subprocess.Popen | None = None


def _kill_group(process: subprocess.Popen) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError, PermissionError):
        pass


def run_check(
    argv: tuple[str, ...],
    cwd: Path,
    *,
    cancelled: threading.Event,
    set_process: Callable[[subprocess.Popen | None], None],
    timeout: float | None = None,
) -> dict:
    """Run one configured argv and return its bounded result."""
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            shell=False,
            start_new_session=True,
        )
    except OSError as exc:
        return {"exit": None, "ok": False, "output": str(exc)[-GOAL_OUTPUT_LIMIT:]}
    set_process(process)
    deadline = time.monotonic() + (GOAL_CHECK_TIMEOUT_SECONDS if timeout is None else timeout)
    timed_out = False
    output = ""
    completed = False
    try:
        while True:
            if cancelled.is_set():
                _kill_group(process)
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _kill_group(process)
                break
            try:
                output, _ = process.communicate(timeout=min(0.05, remaining))
                completed = True
                break
            except subprocess.TimeoutExpired:
                continue
        if not completed:
            try:
                output, _ = process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                _kill_group(process)
                output, _ = process.communicate()
    finally:
        set_process(None)
    code = None if timed_out or cancelled.is_set() else process.returncode
    output = (output or "")[-GOAL_OUTPUT_LIMIT:]
    return {"exit": code, "ok": code == 0, "output": output}
