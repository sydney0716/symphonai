"""Capture and render the environment at the start of a conversation."""

from __future__ import annotations

import platform as platform_module
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

MAX_GIT_SNAPSHOT_CHARS = 2_000
_TRUNCATION_MARKER = "... (truncated)"


@dataclass(frozen=True)
class GitSnapshot:
    branch: str
    status: str
    recent_commits: str


def render_environment(
    *,
    working_dir: str | Path,
    repo_root: str | Path,
    platform_text: str,
    date_text: str,
    provider: str,
    model: str | None,
    git_snapshot: GitSnapshot | None,
) -> str:
    """Render a captured environment block from explicit values."""

    model_line = f"- Model: {provider}" + (f" {model}" if model else "")
    lines = [
        "Environment when this conversation started (it does not update):",
        f"- Working directory: {working_dir}",
        f"- Repository root: {repo_root}",
        f"- Platform: {platform_text}",
        f"- Date: {date_text}",
        model_line,
    ]
    rendered = "\n".join(lines)
    if git_snapshot is None:
        return rendered

    status = git_snapshot.status or "(clean)"
    git_part = "\n".join(
        [
            "Git snapshot when this conversation started (it does not update):",
            f"Branch: {git_snapshot.branch}",
            "Status:",
            status,
            "Recent commits:",
            git_snapshot.recent_commits,
        ]
    )
    if len(git_part) > MAX_GIT_SNAPSHOT_CHARS:
        prefix_size = MAX_GIT_SNAPSHOT_CHARS - len(_TRUNCATION_MARKER) - 1
        git_part = git_part[:prefix_size].rstrip("\r\n") + "\n" + _TRUNCATION_MARKER
    return rendered + "\n\n" + git_part


def _capture_git_snapshot(repo_root: Path) -> GitSnapshot | None:
    def run_git(*command: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "--no-optional-locks", *command],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError):
            return None
        return completed.stdout.rstrip("\r\n")

    status = run_git("status", "--short")
    if status is None:
        return None

    branch = run_git("symbolic-ref", "--short", "HEAD")
    if branch is None:
        branch = run_git("rev-parse", "--abbrev-ref", "HEAD") or ""
    recent_commits = run_git("log", "--oneline", "-5") or ""
    return GitSnapshot(branch=branch, status=status, recent_commits=recent_commits)


def capture_environment(
    *,
    working_dir: str | Path,
    repo_root: str | Path,
    provider: str,
    model: str | None,
) -> str:
    """Capture process and repository details for a new conversation."""

    root = Path(repo_root)
    git_snapshot = _capture_git_snapshot(root)
    platform_text = f"{sys.platform} ({platform_module.platform(terse=True)})"
    return render_environment(
        working_dir=working_dir,
        repo_root=repo_root,
        platform_text=platform_text,
        date_text=date.today().isoformat(),
        provider=provider,
        model=model,
        git_snapshot=git_snapshot,
    )
