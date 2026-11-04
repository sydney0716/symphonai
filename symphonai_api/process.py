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
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                if process.poll() is None:
                    try:
                        process.kill()
                    except OSError:
                        pass
            killed = True
            drain_deadline = min(drain_deadline or now + 2.0, now + 2.0)
        if drain_deadline is not None and now >= drain_deadline:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
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


def stop_process_group(proc: subprocess.Popen, pgid: int, *, timeout: float = 0.2) -> None:
    if proc.poll() is None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
        except (OSError, ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    try:
        if hasattr(os, "killpg"):
            os.killpg(pgid, signal.SIGKILL)
        elif proc.poll() is None:
            proc.kill()
    except (OSError, ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
