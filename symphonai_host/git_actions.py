"""Git status, selected-file commits, and pull-request operations."""

from __future__ import annotations

from symphonai_api.git import git as _git

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from symphonai_host.git import read_git_status

PULL_REQUEST_TIMEOUT_SECONDS = 120
UNTRACKED_DIFF_READ_LIMIT = 1024 * 1024


def pull_request_command(args: list[str], root: Path, name: str, *, git_command: bool = False) -> subprocess.CompletedProcess:
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    environment.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    try:
        if git_command:
            return _git(root, *args, stdin=subprocess.DEVNULL, env=environment,
                        check=False, timeout=PULL_REQUEST_TIMEOUT_SECONDS)
        return subprocess.run(
            args, cwd=root, stdin=subprocess.DEVNULL, env=environment,
            capture_output=True, check=False, timeout=PULL_REQUEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{name} timed out after 120 s") from exc



def git_status(root: Path, *, diff_limit: int = 200) -> dict[str, Any]:
    snapshot = read_git_status(root)
    if snapshot is None:
        return {"repo": False}
    root = snapshot.root
    branch_result = _git(root, 'symbolic-ref', '--quiet', '--short', 'HEAD', capture_output=True, check=False)
    branch = os.fsdecode(branch_result.stdout.strip()) if branch_result.returncode == 0 else None
    records = []
    for entry in snapshot.entries:
        code, path, old_path = entry.code, entry.path, entry.old_path
        if code == "??":
            status = "untracked"
        elif "R" in code or "C" in code:
            status = "renamed"
        elif "D" in code:
            status = "deleted"
        elif "A" in code:
            status = "added"
        else:
            status = "modified"
        record = {
            "path": path,
            "status": status,
            "staged": code != "??" and code[0] != " ",
        }
        if old_path is not None:
            record["old_path"] = old_path
        records.append(record)
    records.sort(key=lambda item: item["path"])
    for record in records[:max(0, diff_limit)]:
        diff, truncated = git_diff_text(root, record)
        record["diff"] = diff[:20_000]
        record["diff_truncated"] = truncated
    for record in records[max(0, diff_limit):]:
        record["diff"] = None
        record["diff_deferred"] = True
    return {"repo": True, "branch": branch, "files": records}



def git_diff_text(root: Path, record: dict[str, Any]) -> tuple[str, bool]:
    path = record["path"]
    if record["status"] == "untracked":
        file_path = root / path
        try:
            file_stat = file_path.lstat()
        except OSError:
            return "Not a regular file; diff omitted.", False
        if stat.S_ISLNK(file_stat.st_mode):
            try:
                target = os.readlink(file_path)
            except OSError:
                return "Not a regular file; diff omitted.", False
            diff = (
                f"diff --git a/{path} b/{path}\n"
                "new file mode 120000\n"
                "--- /dev/null\n"
                f"+++ b/{path}\n"
                "@@ -0,0 +1 @@\n"
                f"+{target}\n"
                "\\ No newline at end of file\n"
            )
            return diff, len(diff) > 20_000
        if not stat.S_ISREG(file_stat.st_mode):
            return "Not a regular file; diff omitted.", False
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            return "Not a regular file; diff omitted.", False
        try:
            descriptor = os.open(file_path, os.O_RDONLY | no_follow)
            with os.fdopen(descriptor, "rb") as stream:
                opened_stat = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened_stat.st_mode):
                    return "Not a regular file; diff omitted.", False
                content = stream.read(UNTRACKED_DIFF_READ_LIMIT)
        except OSError:
            return "Not a regular file; diff omitted.", False
        input_truncated = file_stat.st_size > len(content)
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return "Binary file; diff omitted.", False
        if b"\0" in content:
            return "Binary file; diff omitted.", False
        lines = text.splitlines(keepends=True)
        executable = file_stat.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        diff_lines = [
            f"diff --git a/{path} b/{path}\n",
            f"new file mode {'100755' if executable else '100644'}\n",
            "--- /dev/null\n",
            f"+++ b/{path}\n",
        ]
        if lines:
            count = len(lines)
            diff_lines.append(f"@@ -0,0 +1{f',{count}' if count != 1 else ''} @@\n")
            for line in lines:
                diff_lines.append(f"+{line}")
                if not line.endswith(("\n", "\r")):
                    diff_lines.append("\n\\ No newline at end of file\n")
        diff = "".join(diff_lines)
        return diff, input_truncated or len(diff) > 20_000
    else:
        paths = [path, *([record["old_path"]] if isinstance(record.get("old_path"), str) else [])]
        result = _git(root, 'diff', '--no-ext-diff', '--no-color', 'HEAD', '--', *paths, capture_output=True, check=False)
        diff = os.fsdecode(result.stdout)
        if b"GIT binary patch" in result.stdout or b"Binary files " in result.stdout:
            diff = "Binary file; diff omitted."
    return diff, len(diff) > 20_000



def git_diff(root: Path, path: str) -> dict[str, Any] | None:
    snapshot = read_git_status(root)
    if snapshot is None:
        return None
    repository = snapshot.root
    relative = Path(path)
    if not path or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("path must be inside the repository")
    try:
        (repository / relative).parent.resolve(strict=False).relative_to(repository)
    except ValueError as exc:
        raise ValueError("path must be inside the repository") from exc
    entry = next((item for item in snapshot.entries if item.path == path), None)
    if entry is None:
        return None
    record = {"path": entry.path, "status": "untracked" if entry.code == "??" else "modified"}
    if entry.old_path is not None:
        record["old_path"] = entry.old_path
    diff, truncated = git_diff_text(repository, record)
    return {"path": path, "diff": diff[:20_000], "diff_truncated": truncated}



def git_commit(root: Path, message: str, paths: list[str]) -> dict[str, Any]:
    snapshot = read_git_status(root)
    if snapshot is None:
        raise ValueError("not a git repository")
    root = snapshot.root
    changed = {path for entry in snapshot.entries
               for path in (entry.path, entry.old_path) if path is not None}
    selected = sorted(set(paths))
    for path in selected:
        relative = Path(path)
        if (
            not path or relative.is_absolute() or ".." in relative.parts
            or path == "." or path.startswith(".git/") or path == ".git"
        ):
            raise ValueError(f"path is outside the repository: {path}")
        try:
            (root / relative).parent.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"path is outside the repository: {path}") from exc
        if path not in changed:
            raise ValueError(f"no uncommitted change for path: {path}")

    head = _git(root, 'rev-parse', '--verify', '--quiet', 'HEAD', capture_output=True, check=False)
    has_head = head.returncode == 0
    descriptor, index_name = tempfile.mkstemp(prefix="symphonai-git-index-")
    os.close(descriptor)
    os.unlink(index_name)
    environment = {**os.environ, "GIT_INDEX_FILE": index_name}
    try:
        initialized = _git(root, 'read-tree', 'HEAD' if has_head else '--empty', env=environment, capture_output=True, check=False)
        if initialized.returncode:
            output = (initialized.stderr or initialized.stdout).decode("utf-8", errors="replace")
            raise RuntimeError(output[-4000:] or "could not prepare a temporary Git index")
        added = _git(root, 'add', '--', *selected, env=environment, capture_output=True, check=False)
        if added.returncode:
            output = (added.stderr or added.stdout).decode("utf-8", errors="replace")
            raise RuntimeError(output[-4000:] or "git add failed")
        committed = _git(root, 'commit', '-m', message, env=environment, capture_output=True, check=False)
        if committed.returncode:
            output = committed.stdout + committed.stderr
            raise RuntimeError(output.decode("utf-8", errors="replace")[-4000:] or "git commit failed")
        commit = _git(root, 'rev-parse', 'HEAD', capture_output=True, check=False)
        if commit.returncode:
            output = (commit.stderr or commit.stdout).decode("utf-8", errors="replace")
            raise RuntimeError(output[-4000:] or "could not read the new commit")
        reset = _git(root, 'reset', '--quiet', 'HEAD', *(['--', *selected] if has_head else []), capture_output=True, check=False)
        if reset.returncode:
            output = (reset.stderr or reset.stdout).decode("utf-8", errors="replace")
            raise RuntimeError(output[-4000:] or "could not update the selected paths in the index")
        return {"commit": os.fsdecode(commit.stdout.strip()), "paths": selected}
    finally:
        try:
            os.unlink(index_name)
        except FileNotFoundError:
            pass



def git_pull_request_info(root: Path) -> dict[str, Any]:
    gh = shutil.which("gh")
    if gh is None:
        return {"branch": None, "base": None, "title": "", "ahead": 0,
                "problem": "The GitHub CLI (gh) is not installed."}
    auth = pull_request_command([gh, "auth", "status"], root, "gh auth status")
    if auth.returncode:
        return {"branch": None, "base": None, "title": "", "ahead": 0,
                "problem": "gh is not signed in. Run gh auth login."}
    origin = pull_request_command(['remote', 'get-url', 'origin'], root, 'git remote get-url', git_command=True)
    if origin.returncode:
        return {"branch": None, "base": None, "title": "", "ahead": 0,
                "problem": "This repository has no origin remote."}
    current = pull_request_command(['symbolic-ref', '--quiet', '--short', 'HEAD'], root, 'git symbolic-ref', git_command=True)
    if current.returncode:
        return {"branch": None, "base": None, "title": "", "ahead": 0,
                "problem": "Check out a branch first."}
    branch = os.fsdecode(current.stdout.strip())
    remote_head = pull_request_command(['ls-remote', '--symref', 'origin', 'HEAD'], root, 'git ls-remote', git_command=True)
    base = None
    if remote_head.returncode == 0:
        for line in remote_head.stdout.splitlines():
            if line.startswith(b"ref: refs/heads/") and line.endswith(b"\tHEAD"):
                base = os.fsdecode(line[len(b"ref: refs/heads/"):-len(b"\tHEAD")])
                break
    if not base:
        return {"branch": branch, "base": None, "title": "", "ahead": 0,
                "problem": "Could not determine origin's default branch."}
    title_result = pull_request_command(['log', '-1', '--format=%s', 'HEAD'], root, 'git log', git_command=True)
    title = os.fsdecode(title_result.stdout.rstrip(b"\r\n")) if title_result.returncode == 0 else ""
    if branch == base:
        problem = f"Create a branch first; a pull request comes from a branch other than {base}."
        return {"branch": branch, "base": base, "title": title, "ahead": 0, "problem": problem}
    ahead_result = pull_request_command(['rev-list', '--count', f'origin/{base}..HEAD'], root, 'git rev-list', git_command=True)
    if ahead_result.returncode:
        return {"branch": branch, "base": base, "title": title, "ahead": 0,
                "problem": f"Could not compare {branch} with {base}."}
    ahead = int(ahead_result.stdout.strip() or b"0")
    problem = f"{branch} has no commits that {base} does not." if ahead == 0 else None
    return {"branch": branch, "base": base, "title": title, "ahead": ahead, "problem": problem}



def create_pull_request(root: Path, title: str, body: str) -> dict[str, str]:
    info = git_pull_request_info(root)
    if info["problem"]:
        raise RuntimeError(info["problem"])
    push = pull_request_command(['push', '-u', 'origin', info['branch']], root, 'git push', git_command=True)
    if push.returncode:
        output = (push.stdout + push.stderr).decode("utf-8", errors="replace")
        raise RuntimeError(output[-4000:] or "git push failed")
    gh = shutil.which("gh")
    created = pull_request_command(
        [gh, "pr", "create", "--title", title, "--body", body,
         "--base", info["base"], "--head", info["branch"]],
        root, "gh pr create",
    )
    if created.returncode:
        output = (created.stdout + created.stderr).decode("utf-8", errors="replace")
        raise RuntimeError(output[-4000:] or "gh pr create failed")
    url = os.fsdecode(created.stdout.strip()).splitlines()[-1:]
    if not url:
        raise RuntimeError("gh pr create did not return a pull request URL")
    return {"url": url[0]}
