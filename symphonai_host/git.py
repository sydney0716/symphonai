"""Read repository roots and NUL-delimited working-tree status."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitEntry:
    path: str
    code: str
    old_path: str | None = None


@dataclass(frozen=True)
class GitStatus:
    root: Path
    entries: tuple[GitEntry, ...]


def read_git_status(root: Path, *, all_untracked: bool = True) -> GitStatus | None:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=root,
            capture_output=True, check=False,
        )
        if top.returncode:
            return None
        repository = Path(os.fsdecode(top.stdout.strip())).resolve()
        command = ["git", "status", "--porcelain=v1", "-z"]
        if all_untracked:
            command.append("--untracked-files=all")
        result = subprocess.run(
            command, cwd=repository, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            capture_output=True, check=False,
        )
    except OSError:
        return None
    if result.returncode:
        return None
    fields = result.stdout.split(b"\0")
    entries = []
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if len(entry) < 4:
            continue
        code = os.fsdecode(entry[:2])
        old_path = None
        if "R" in code or "C" in code:
            if index < len(fields):
                old_path = os.fsdecode(fields[index])
                index += 1
        entries.append(GitEntry(os.fsdecode(entry[3:]), code, old_path))
    return GitStatus(repository, tuple(entries))
