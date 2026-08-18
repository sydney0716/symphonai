"""Git worktree helpers using safe, list-form subprocess calls.

No command in this module is ever run with ``shell=True``: every Git
invocation is a list of argv tokens passed straight to ``subprocess.run``,
so there is no shell to interpret metacharacters in task ids, branch names,
or paths.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PREFERRED_WORKTREE_ROOT = ".orchestra/worktrees"


class GitUtilsError(Exception):
    """Raised for invalid input before any subprocess is even started."""


def _run_git(args: list[str], cwd: Path) -> dict:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        shell=False,
        check=False,
    )
    return {
        "command": ["git", *args],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def worktree_add(
    repo_root: Path,
    task_id: str,
    branch: str,
    path: str | None = None,
    reuse_existing: bool = False,
) -> dict:
    """Create a Git worktree + branch for a task.

    Returns a structured result dict with the actual command run, exit
    code, stdout, and stderr — always, whether or not the operation
    succeeded, so callers get full detail without needing to catch an
    exception for the common "already exists" case.
    """
    if not task_id or not task_id.strip():
        raise GitUtilsError("task_id must not be empty")
    if not branch or not branch.strip():
        raise GitUtilsError("branch must not be empty")

    if path is not None and not path.strip():
        raise GitUtilsError("path must not be empty (omit it to use the default)")

    if path:
        rel_path = path
    else:
        rel_path = f"{PREFERRED_WORKTREE_ROOT}/{task_id}"

    warning = None
    if not rel_path.startswith(PREFERRED_WORKTREE_ROOT + "/") and rel_path != PREFERRED_WORKTREE_ROOT:
        warning = (
            f"path {rel_path!r} is outside the preferred worktree root "
            f"{PREFERRED_WORKTREE_ROOT!r}; proceeding anyway"
        )

    abs_path = (repo_root / rel_path).resolve()

    if abs_path.exists():
        if not reuse_existing:
            raise GitUtilsError(
                f"worktree path already exists: {abs_path} "
                "(pass reuse_existing=true to reuse it)"
            )
        return {
            "command": None,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "path": str(rel_path),
            "branch": branch,
            "reused": True,
            "warning": warning,
        }

    result = _run_git(
        ["worktree", "add", str(abs_path), "-b", branch],
        cwd=repo_root,
    )
    result["path"] = str(rel_path)
    result["branch"] = branch
    result["reused"] = False
    result["warning"] = warning
    return result


def worktree_list(repo_root: Path) -> dict:
    return _run_git(["worktree", "list", "--porcelain"], cwd=repo_root)
