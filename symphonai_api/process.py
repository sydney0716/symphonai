"""Bounded execution of short-lived command process groups."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False
    cancelled: bool = False
    error: OSError | ValueError | None = None


def run_process(
    argv: Sequence[str], cwd: Path, *, env: Mapping[str, str] | None = None,
    input: bytes | None = None, timeout: float, cancel: Callable[[], bool] | None = None,
    output_limit: int | None = 4000, merge_stderr: bool = True,
    on_start: Callable[[subprocess.Popen], None] | None = None,
) -> ProcessResult:
    try:
        process = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        return ProcessResult(None, error=exc)
    if on_start is not None:
        on_start(process)
    deadline = time.monotonic() + timeout
    timed_out = cancelled = False
    drain_deadline = None
    stdout = stderr = b""
    killed = False
    while True:
        now = time.monotonic()
        cancelled = cancelled or (cancel is not None and cancel())
        timed_out = timed_out or now >= deadline
        if process.poll() is not None and drain_deadline is None:
            drain_deadline = now + 2.0
        if (cancelled or timed_out) and not killed:
            stop_process_group(process, process.pid, grace=0, timeout=0.1)
            killed = True
            drain_deadline = min(drain_deadline or now + 2.0, now + 2.0)
        if drain_deadline is not None and now >= drain_deadline:
            stop_process_group(process, process.pid, grace=0, timeout=0.1)
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
            break
        remaining = (drain_deadline if killed else min(deadline, drain_deadline or deadline)) - now
        try:
            stdout, stderr = process.communicate(input=input, timeout=min(0.05, remaining))
            cancelled = cancelled or (cancel is not None and cancel())
            break
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = exc.output or b"", exc.stderr or b""
            input = None
    stdout, stderr = stdout or b"", stderr or b""
    if output_limit is not None:
        stdout = stdout[-output_limit:] if output_limit else b""
        stderr = stderr[-output_limit:] if output_limit else b""
    return ProcessResult(None if timed_out or cancelled else process.returncode,
                         stdout, stderr, timed_out, cancelled)




def stop_process_group(proc, pgid, *, grace: float | None = None, timeout: float = 0.2) -> None:
    """Stop child groups without signalling the host's own group; waits stay bounded."""
    if proc is not None and proc.pid == os.getpid():
        return
    groups = (pgid,) if isinstance(pgid, int) else tuple(pgid)
    own_group = os.getpgrp() if hasattr(os, "getpgrp") else None
    groups = tuple(group for group in groups if group != own_group)
    grace = timeout if grace is None else grace
    def send(number):
        if hasattr(os, "killpg"):
            for group in groups:
                try:
                    os.killpg(group, number)
                except OSError:
                    pass
        if proc is not None and proc.poll() is None and (not groups or not hasattr(os, "killpg")):
            try:
                proc.terminate() if number == signal.SIGTERM else proc.kill()
            except OSError:
                pass
    if grace > 0:
        send(signal.SIGTERM)
        if proc is None:
            time.sleep(grace)
        else:
            try:
                proc.wait(timeout=grace)
            except (OSError, subprocess.TimeoutExpired):
                pass
    send(signal.SIGKILL)
    if proc is not None:
        try:
            proc.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass


def run_check_command(argv, cwd: Path, *, policy=None, sandboxed=False, enforce_policy=False,
                      timeout: float, cancel=None, on_start=None, output_limit=4000,
                      for_agent=True) -> dict:
    """Run validation/check argv with bounded waits and structured exit status."""
    from dataclasses import replace
    from symphonai_api.tools.shell import sandbox_command, shell_environment
    if enforce_policy and policy is not None:
        decision = policy.check_shell(argv, cwd=cwd)
        if not decision.allowed:
            return {"exit": 126, "output": decision.reason}
    if sandboxed and policy is not None:
        try:
            argv = sandbox_command(replace(policy, mode="allow"), list(argv), cwd)
        except ValueError as exc:
            return {"exit": None, "output": str(exc)}
    result = run_process(argv, cwd, env=shell_environment(cwd, for_agent=for_agent),
                         timeout=timeout, cancel=cancel, on_start=on_start, output_limit=None)
    output = str(result.error) if result.error is not None else result.stdout.decode("utf-8", errors="replace")
    if output_limit is not None:
        output = output[-output_limit:] if output_limit else ""
    return {"exit": result.exit_code, "output": output, "timed_out": result.timed_out,
            "cancelled": result.cancelled, "error": result.error}
