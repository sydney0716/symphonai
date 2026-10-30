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
    while True:
        cancelled = cancel is not None and cancel()
        remaining = deadline - time.monotonic()
        timed_out = remaining <= 0
        if cancelled or timed_out:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            stdout, stderr = process.communicate()
            break
        try:
            stdout, stderr = process.communicate(input=input, timeout=min(0.05, remaining))
            cancelled = cancel is not None and cancel()
            break
        except subprocess.TimeoutExpired:
            input = None
    stdout, stderr = stdout or b"", stderr or b""
    if output_limit is not None:
        stdout = stdout[-output_limit:] if output_limit else b""
        stderr = stderr[-output_limit:] if output_limit else b""
    return ProcessResult(None if timed_out or cancelled else process.returncode,
                         stdout, stderr, timed_out, cancelled)
