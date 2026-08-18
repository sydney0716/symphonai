"""Canonical Orchestra state schema: allowed values and field validation."""

from __future__ import annotations

ALLOWED_TASK_STATUSES = frozenset(
    {
        "CREATED",
        "READY",
        "RUNNING",
        "REVIEW",
        "BLOCKED",
        "RETRY",
        "COMPLETE",
        "FAILED",
        "CANCELLED",
    }
)

CANONICAL_THREAD_ID_FIELD = "thread_id"
FORBIDDEN_THREAD_ID_FIELD = "threadId"

REQUIRED_TASK_FIELDS = (
    "id",
    "title",
    "worker",
    "depends_on",
    "scope",
    "acceptance_criteria",
    "validation_commands",
)

OPTIONAL_TASK_FIELDS = (
    "provider",
    "worktree",
    "branch",
    "status",
    "validation_results",
    "followups",
    "note",
    "forbidden_scope",
    "preflight_checks",
    "sources",
    "depends_on",
)

REQUIRED_WORKER_FIELDS = (
    "id",
    "provider",
    "task_id",
    "thread_id",
)

OPTIONAL_WORKER_FIELDS = (
    "status",
    "worktree",
    "branch",
)

TOP_LEVEL_KEYS = ("version", "active_epic", "tasks", "workers", "runs")


def validate_task_status(status: str) -> list[str]:
    """Return a list of error strings; empty list means valid."""
    errors: list[str] = []
    if status not in ALLOWED_TASK_STATUSES:
        errors.append(
            f"invalid task status {status!r}; must be one of {sorted(ALLOWED_TASK_STATUSES)}"
        )
    return errors


def validate_task_record(task: dict) -> list[str]:
    """Validate a single task dict against the canonical schema."""
    errors: list[str] = []
    for field in REQUIRED_TASK_FIELDS:
        if field not in task:
            errors.append(f"task {task.get('id', '<unknown>')!r} missing required field {field!r}")
    if "status" in task:
        errors.extend(validate_task_status(task["status"]))
    if FORBIDDEN_THREAD_ID_FIELD in task:
        errors.append(
            f"task {task.get('id', '<unknown>')!r} uses forbidden field "
            f"{FORBIDDEN_THREAD_ID_FIELD!r}; use {CANONICAL_THREAD_ID_FIELD!r} instead"
        )
    return errors


def validate_worker_record(worker: dict) -> list[str]:
    """Validate a single worker dict against the canonical schema."""
    errors: list[str] = []
    for field in REQUIRED_WORKER_FIELDS:
        if field not in worker:
            errors.append(
                f"worker {worker.get('id', '<unknown>')!r} missing required field {field!r}"
            )
    if FORBIDDEN_THREAD_ID_FIELD in worker:
        errors.append(
            f"worker {worker.get('id', '<unknown>')!r} uses forbidden field "
            f"{FORBIDDEN_THREAD_ID_FIELD!r}; use {CANONICAL_THREAD_ID_FIELD!r} instead"
        )
    return errors


def default_state() -> dict:
    return {
        "version": 1,
        "active_epic": None,
        "tasks": [],
        "workers": [],
        "runs": [],
    }
