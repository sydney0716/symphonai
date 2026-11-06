"""Bounded Git command execution shared by runtime and host."""
from __future__ import annotations

import subprocess
from pathlib import Path


class WorktreeError(RuntimeError):
    """A Git operation could not complete."""


def git(cwd: Path, *arguments: str, check: bool = True, timeout: float = 30, **options):
    options.setdefault("capture_output", True)
    try:
        result = subprocess.run(["git", *arguments], cwd=cwd, check=False, timeout=timeout, **options)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorktreeError(str(exc)) from exc
    if check and result.returncode:
        detail = result.stderr
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise WorktreeError((detail or "").strip() or f"git {' '.join(arguments)} failed")
    return result
