"""Orchestra v1 MCP server.

Manages Orchestra task/worker/run state and Git worktree creation. It does
NOT dispatch work to Codex or any other implementation agent — that stays a
manual step for the orchestrator (see docs/orchestra-mcp-server.md for why).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import git_utils, state

app = FastMCP("orchestra")


def _load(state_path: str | None, repo_root: str | None) -> tuple[dict, Path]:
    root = state.resolve_repo_root(repo_root)
    path = state.resolve_state_path(state_path, str(root))
    data = state.load_state(path)
    return data, path


@app.tool()
def task_list(
    status: str | None = None,
    epic: str | None = None,
    state_path: str | None = None,
    repo_root: str | None = None,
) -> dict:
    """List tasks, optionally filtered by status or epic."""
    data, _ = _load(state_path, repo_root)
    tasks = state.task_list(data, status=status, epic=epic)
    return {"tasks": tasks, "count": len(tasks)}


@app.tool()
def task_create(
    id: str,
    title: str,
    worker: str,
    depends_on: list[str],
    scope: list[str],
    acceptance_criteria: list[str],
    validation_commands: list[str],
    provider: str | None = None,
    worktree: str | None = None,
    branch: str | None = None,
    state_path: str | None = None,
    repo_root: str | None = None,
) -> dict:
    """Create a new task record. Fails if the id already exists."""
    data, path = _load(state_path, repo_root)
    task: dict[str, Any] = {
        "id": id,
        "title": title,
        "worker": worker,
        "depends_on": depends_on,
        "scope": scope,
        "acceptance_criteria": acceptance_criteria,
        "validation_commands": validation_commands,
    }
    if provider is not None:
        task["provider"] = provider
    if worktree is not None:
        task["worktree"] = worktree
    if branch is not None:
        task["branch"] = branch

    record = state.task_create(data, task)
    state.atomic_write_state(path, data)
    return {"task": record}


@app.tool()
def task_update(
    id: str,
    updates: dict,
    state_path: str | None = None,
    repo_root: str | None = None,
) -> dict:
    """Update an existing task's status or metadata. Never creates a task."""
    data, path = _load(state_path, repo_root)
    record = state.task_update(data, id, updates)
    state.atomic_write_state(path, data)
    return {"task": record}


@app.tool()
def worker_register(
    id: str,
    provider: str,
    task_id: str,
    thread_id: str,
    status: str | None = None,
    worktree: str | None = None,
    state_path: str | None = None,
    repo_root: str | None = None,
) -> dict:
    """Add or update a workers[] entry."""
    data, path = _load(state_path, repo_root)
    worker: dict[str, Any] = {
        "id": id,
        "provider": provider,
        "task_id": task_id,
        "thread_id": thread_id,
    }
    if status is not None:
        worker["status"] = status
    if worktree is not None:
        worker["worktree"] = worktree

    record = state.worker_register(data, worker)
    state.atomic_write_state(path, data)
    return {"worker": record}


@app.tool()
def worker_update(
    id: str,
    updates: dict,
    state_path: str | None = None,
    repo_root: str | None = None,
) -> dict:
    """Update an existing worker's status or metadata. Never creates a worker."""
    data, path = _load(state_path, repo_root)
    record = state.worker_update(data, id, updates)
    state.atomic_write_state(path, data)
    return {"worker": record}


@app.tool()
def validation_record(
    task_id: str,
    command: str,
    status: str,
    exit_code: int | None = None,
    summary: str | None = None,
    output: str | None = None,
    executed_by: str | None = None,
    state_path: str | None = None,
    repo_root: str | None = None,
) -> dict:
    """Append a structured validation result to a task."""
    data, path = _load(state_path, repo_root)
    entry = state.validation_record(
        data,
        task_id=task_id,
        command=command,
        status=status,
        exit_code=exit_code,
        summary=summary,
        output=output,
        executed_by=executed_by,
    )
    state.atomic_write_state(path, data)
    return {"validation_result": entry}


@app.tool()
def run_record(
    id: str,
    updates: dict | None = None,
    state_path: str | None = None,
    repo_root: str | None = None,
) -> dict:
    """Create or update a runs[] entry."""
    data, path = _load(state_path, repo_root)
    record = state.run_record(data, id, updates or {})
    state.atomic_write_state(path, data)
    return {"run": record}


@app.tool()
def worktree_create(
    task_id: str,
    branch: str,
    path: str | None = None,
    reuse_existing: bool = False,
    record_in_task: bool = False,
    state_path: str | None = None,
    repo_root: str | None = None,
) -> dict:
    """Create a Git worktree + branch for a task.

    Refuses if the target path already exists unless reuse_existing=true.
    If record_in_task is true, writes the resulting worktree/branch back
    onto the task record in state.
    """
    root = state.resolve_repo_root(repo_root)
    result = git_utils.worktree_add(
        repo_root=root,
        task_id=task_id,
        branch=branch,
        path=path,
        reuse_existing=reuse_existing,
    )

    if record_in_task and result.get("exit_code") == 0:
        data, resolved_path = _load(state_path, repo_root)
        try:
            state.task_update(
                data,
                task_id,
                {"worktree": result["path"], "branch": result["branch"]},
            )
            state.atomic_write_state(resolved_path, data)
            result["recorded_in_task"] = True
        except state.StateError as exc:
            result["recorded_in_task"] = False
            result["record_error"] = str(exc)

    return result


@app.tool()
def state_validate(
    state_path: str | None = None,
    repo_root: str | None = None,
) -> dict:
    """Validate the Orchestra state file and return a structured report."""
    root = state.resolve_repo_root(repo_root)
    path = state.resolve_state_path(state_path, str(root))
    try:
        data = state.load_state(path)
    except state.StateError as exc:
        return {
            "ok": False,
            "checks": [{"name": "valid_json_shape", "ok": False, "errors": [str(exc)]}],
        }
    report = state.state_validate(data)
    report["state_path"] = str(path)
    return report


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
