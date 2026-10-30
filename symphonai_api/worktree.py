"""Git worktrees used to isolate subagent changes."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from symphonai_api.config import load_config


_PROJECT_ROOT_METADATA = "symphonai-project-root"
WORKTREE_SETUP_TIMEOUT_SECONDS = 30 * 60


class WorktreeError(RuntimeError):
    """A git worktree could not be created, inspected, or removed."""


@dataclass(frozen=True)
class WorktreeDiff:
    patch: str
    files: tuple[str, ...]


def _git(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorktreeError(str(exc)) from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise WorktreeError(detail or f"git {' '.join(arguments)} failed")
    return result


def _top_level(repo_root: Path) -> Path:
    repo_root = Path(repo_root).resolve()
    result = _git(repo_root, "rev-parse", "--show-toplevel")
    return Path(os.fsdecode(result.stdout.strip())).resolve()


def create_worktree(repo_root: Path, path: Path, start: str = "HEAD") -> Path:
    """Create a detached checkout and return the matching repository root."""
    repo_root = Path(repo_root).resolve()
    top_level = _top_level(repo_root)
    relative_root = repo_root.relative_to(top_level)
    path = Path(path).resolve()
    if path.exists():
        raise WorktreeError(f"worktree path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _git(top_level, "worktree", "add", "--detach", str(path), start)
    except WorktreeError:
        if path.exists():
            try:
                _git(top_level, "worktree", "remove", "--force", str(path))
            except WorktreeError:
                pass
            try:
                _git(top_level, "worktree", "prune")
            except WorktreeError:
                pass
        raise
    worktree_root = path / relative_root
    git_dir = Path(os.fsdecode(_git(path, "rev-parse", "--git-dir").stdout.strip()))
    if not git_dir.is_absolute():
        git_dir = (path / git_dir).resolve()
    (git_dir / _PROJECT_ROOT_METADATA).write_text(relative_root.as_posix(), encoding="utf-8")
    config = load_config(repo_root=repo_root)
    configured = config.get(
        "worktree.symlink", ["node_modules", ".venv", "venv"],
    )
    for entry in configured if isinstance(configured, list) else ():
        source = repo_root / entry
        target = worktree_root / entry
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=source.is_dir())
    setup = config.get("worktree.setup")
    if isinstance(setup, str):
        from symphonai_api.tools.shell import shell_environment

        from symphonai_api.process import run_process
        result = run_process(
            ["/bin/sh", "-c", setup], worktree_root,
            env=shell_environment(worktree_root, for_agent=True),
            timeout=WORKTREE_SETUP_TIMEOUT_SECONDS, output_limit=None,
        )
        output = result.stdout.decode("utf-8", errors="replace")
        if result.error is not None or result.timed_out or result.exit_code != 0:
            if result.error is not None:
                message = f"worktree setup command {setup!r} could not start: {result.error}"
            elif result.timed_out:
                message = f"worktree setup command {setup!r} timed out after 1800 seconds; output: {output[-4000:]}"
            else:
                message = f"worktree setup command {setup!r} exited with code {result.exit_code}; output: {output[-4000:]}"
            try:
                remove_worktree(repo_root, path)
            except WorktreeError as cleanup_error:
                raise WorktreeError(f"{message}; cleanup failed: {cleanup_error}") from result.error
            raise WorktreeError(message) from result.error
    return worktree_root


def _linked_paths(path: Path) -> tuple[str, ...]:
    worktree_top = _top_level(path)
    common_dir = Path(os.fsdecode(_git(path, "rev-parse", "--git-common-dir").stdout.strip()))
    if not common_dir.is_absolute():
        common_dir = (path / common_dir).resolve()
    project_relative = path.relative_to(worktree_top)
    if not project_relative.parts:
        git_dir = Path(os.fsdecode(_git(path, "rev-parse", "--git-dir").stdout.strip()))
        if not git_dir.is_absolute():
            git_dir = (path / git_dir).resolve()
        metadata = git_dir / _PROJECT_ROOT_METADATA
        if metadata.is_file():
            project_relative = Path(metadata.read_text(encoding="utf-8"))
    project_root = worktree_top / project_relative
    repo_root = common_dir.parent / project_relative
    configured = load_config(repo_root=repo_root).get(
        "worktree.symlink", ["node_modules", ".venv", "venv"],
    )
    entries = configured if isinstance(configured, list) else ()
    return tuple(
        (project_root / entry).relative_to(path).as_posix()
        for entry in entries
        if (project_root / entry).is_symlink()
        and os.readlink(project_root / entry) == os.fspath(repo_root / entry)
    )


def stage_worktree(path: Path) -> None:
    """Stage project changes, excluding the worktree's shared symlinks."""
    path = Path(path).resolve()
    _git(path, "add", "-A")
    linked = _linked_paths(path)
    if linked:
        _git(path, "reset", "--quiet", "HEAD", "--", *linked)


def worktree_diff(path: Path) -> WorktreeDiff:
    """Stage worktree changes in its own index and return paths plus patch."""
    path = Path(path).resolve()
    stage_worktree(path)
    files = _git(
        path,
        "diff",
        "--cached",
        "--name-only",
        "--relative",
        "-z",
        "HEAD",
    ).stdout
    patch = _git(path, "diff", "--cached", "--binary", "HEAD").stdout
    return WorktreeDiff(
        patch=os.fsdecode(patch),
        files=tuple(os.fsdecode(item) for item in files.split(b"\0") if item),
    )


def remove_worktree(repo_root: Path, path: Path) -> None:
    """Force-remove a worktree and prune its stale git metadata."""
    top_level = _top_level(Path(repo_root).resolve())
    path = Path(path).resolve()
    _git(top_level, "worktree", "remove", "--force", str(path))
    _git(top_level, "worktree", "prune")
