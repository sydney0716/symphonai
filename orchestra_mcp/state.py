"""Safe, configurable load/validate/write layer for Orchestra task state.

State path resolution never hardcodes a single location:

- an explicit ``state_path`` argument always wins;
- otherwise the ``ORCHESTRA_STATE_PATH`` environment variable is used;
- otherwise ``<repo_root>/.orchestra/tasks.json`` is used.

Repo root resolution is analogous via ``ORCHESTRA_REPO_ROOT`` / the current
working directory.

Every write goes through :func:`atomic_write_state`, which writes to a
temporary file in the same directory and then calls ``os.replace`` so a
crash mid-write can never leave ``tasks.json`` truncated or corrupted.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from . import schema

DEFAULT_STATE_RELATIVE_PATH = ".orchestra/tasks.json"

ENV_STATE_PATH = "ORCHESTRA_STATE_PATH"
ENV_REPO_ROOT = "ORCHESTRA_REPO_ROOT"


class StateError(Exception):
    """Raised for state load/validation/write failures that callers must handle."""


def resolve_repo_root(repo_root: str | os.PathLike | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    env_value = os.environ.get(ENV_REPO_ROOT)
    if env_value:
        return Path(env_value).resolve()
    return Path.cwd().resolve()


def resolve_state_path(
    state_path: str | os.PathLike | None = None,
    repo_root: str | os.PathLike | None = None,
) -> Path:
    if state_path is not None:
        return Path(state_path).resolve()
    env_value = os.environ.get(ENV_STATE_PATH)
    if env_value:
        return Path(env_value).resolve()
    root = resolve_repo_root(repo_root)
    return (root / DEFAULT_STATE_RELATIVE_PATH).resolve()


def load_state(path: str | os.PathLike, create_if_missing: bool = False) -> dict:
    p = Path(path)
    if not p.exists():
        if create_if_missing:
            return schema.default_state()
        raise StateError(f"state file does not exist: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise StateError(f"failed to read state file {p}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StateError(f"state file {p} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise StateError(f"state file {p} must contain a JSON object at the top level")
    return data


def atomic_write_state(path: str | os.PathLike, state: dict) -> None:
    """Write ``state`` to ``path`` atomically.

    Writes to a sibling temp file first, flushes + fsyncs it, then uses
    ``os.replace`` (atomic on POSIX and Windows) to publish it. The
    original file is never opened for writing directly, so a crash
    mid-write cannot corrupt it.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(state, indent=2, ensure_ascii=False, sort_keys=False)
    if not payload.endswith("\n"):
        payload += "\n"

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, p)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _find_index(items: list[dict], key: str, value: Any) -> int | None:
    for i, item in enumerate(items):
        if item.get(key) == value:
            return i
    return None


# --- task operations ---------------------------------------------------


def task_list(
    state: dict,
    status: str | None = None,
    epic: str | None = None,
) -> list[dict]:
    tasks = state.get("tasks", [])
    result = tasks
    if status is not None:
        result = [t for t in result if t.get("status") == status]
    if epic is not None:
        result = [t for t in result if t.get("epic") == epic or state.get("active_epic") == epic]
    return result


def task_create(state: dict, task: dict) -> dict:
    tasks = state.setdefault("tasks", [])
    task_id = task.get("id")
    if not task_id:
        raise StateError("task_create requires a non-empty 'id'")
    if _find_index(tasks, "id", task_id) is not None:
        raise StateError(f"task {task_id!r} already exists; use task_update instead")

    missing = [f for f in schema.REQUIRED_TASK_FIELDS if f not in task]
    if missing:
        raise StateError(f"task_create missing required fields: {missing}")

    record = dict(task)
    record.setdefault("status", "CREATED")
    errors = schema.validate_task_status(record["status"])
    if errors:
        raise StateError("; ".join(errors))
    if schema.FORBIDDEN_THREAD_ID_FIELD in record:
        raise StateError(
            f"task {task_id!r} must not set {schema.FORBIDDEN_THREAD_ID_FIELD!r}; "
            f"use {schema.CANONICAL_THREAD_ID_FIELD!r}"
        )

    tasks.append(record)
    return record


def task_update(state: dict, task_id: str, updates: dict) -> dict:
    tasks = state.get("tasks", [])
    idx = _find_index(tasks, "id", task_id)
    if idx is None:
        raise StateError(f"task {task_id!r} does not exist; task_update will not create it")

    if "status" in updates:
        errors = schema.validate_task_status(updates["status"])
        if errors:
            raise StateError("; ".join(errors))
    if schema.FORBIDDEN_THREAD_ID_FIELD in updates:
        raise StateError(
            f"task {task_id!r} update must not set {schema.FORBIDDEN_THREAD_ID_FIELD!r}; "
            f"use {schema.CANONICAL_THREAD_ID_FIELD!r}"
        )

    tasks[idx].update(updates)
    return tasks[idx]


# --- worker operations ---------------------------------------------------


def worker_register(state: dict, worker: dict) -> dict:
    workers = state.setdefault("workers", [])
    worker_id = worker.get("id")
    if not worker_id:
        raise StateError("worker_register requires a non-empty 'id'")

    missing = [f for f in schema.REQUIRED_WORKER_FIELDS if f not in worker]
    if missing:
        raise StateError(f"worker_register missing required fields: {missing}")
    if schema.FORBIDDEN_THREAD_ID_FIELD in worker:
        raise StateError(
            f"worker {worker_id!r} must not set {schema.FORBIDDEN_THREAD_ID_FIELD!r}; "
            f"use {schema.CANONICAL_THREAD_ID_FIELD!r}"
        )

    task_id = worker.get("task_id")
    if task_id and _find_index(state.get("tasks", []), "id", task_id) is None:
        raise StateError(f"worker {worker_id!r} references unknown task_id {task_id!r}")

    idx = _find_index(workers, "id", worker_id)
    record = dict(worker)
    if idx is None:
        workers.append(record)
    else:
        workers[idx].update(record)
        record = workers[idx]
    return record


def worker_delete(state: dict, worker_id: str, *, force: bool = False) -> dict:
    """Remove a workers[] entry by id and return the removed record.

    Exists so stale worker records can be retired through the tools rather
    than by hand-editing the state file. Deliberately id-based and one at a
    time: there is no bulk or predicate delete, since losing worker records
    silently would make orchestration history untrustworthy.

    Refuses by default when a task still points at this worker's thread,
    because deleting it would leave that task's `thread_id` dangling and
    make `state_validate` fail -- the tool must not be able to write state
    it knows is invalid. Clear the task's `thread_id` first, or pass
    `force=True` to accept the inconsistency deliberately.
    """
    workers = state.get("workers", [])
    if not isinstance(workers, list):
        raise StateError("'workers' is not a list; refusing to delete from malformed state")
    idx = _find_index(workers, "id", worker_id)
    if idx is None:
        raise StateError(f"worker {worker_id!r} does not exist; nothing to delete")

    record = workers[idx]
    task_id = record.get("task_id")
    if not force and task_id:
        others = [
            w
            for i, w in enumerate(workers)
            if i != idx and isinstance(w, dict) and w.get("task_id") == task_id
        ]
        if not others:
            for task in state.get("tasks", []):
                if isinstance(task, dict) and task.get("id") == task_id and task.get("thread_id"):
                    raise StateError(
                        f"worker {worker_id!r} is the only worker for task {task_id!r}, "
                        f"which still has a thread_id; deleting it would leave that "
                        f"reference dangling. Clear the task's thread_id first, or pass "
                        f"force=True."
                    )
    return workers.pop(idx)


def worker_delete_malformed(state: dict) -> list[dict]:
    """Remove workers[] entries that lack the required fields, returning them.

    Legacy records predating the canonical worker schema (for example ones
    keyed on 'worker' instead of 'id') cannot be addressed by
    `worker_delete`, which needs an id. This is the one escape hatch for
    them, and it only ever removes records that are already invalid -- a
    well-formed record is never touched.
    """
    workers = state.get("workers", [])
    # Fail closed on structural invalidity. If `workers` were a mapping we
    # would otherwise iterate its KEYS, judge each string "malformed",
    # report them as removed, and replace the whole mapping with [] --
    # silently destroying every record. Refusing is always recoverable;
    # deleting is not.
    if not isinstance(workers, list):
        raise StateError(
            f"'workers' must be a list to prune malformed records, got {type(workers).__name__}"
        )
    if not all(isinstance(w, dict) for w in workers):
        raise StateError("'workers' contains non-dict entries; refusing to prune malformed records")

    removed: list[dict] = []
    kept: list[dict] = []
    for worker in workers:
        if all(field in worker for field in schema.REQUIRED_WORKER_FIELDS):
            kept.append(worker)
        else:
            removed.append(worker)
    state["workers"] = kept
    return removed


def worker_update(state: dict, worker_id: str, updates: dict) -> dict:
    workers = state.get("workers", [])
    idx = _find_index(workers, "id", worker_id)
    if idx is None:
        raise StateError(f"worker {worker_id!r} does not exist; worker_update will not create it")
    if schema.FORBIDDEN_THREAD_ID_FIELD in updates:
        raise StateError(
            f"worker {worker_id!r} update must not set {schema.FORBIDDEN_THREAD_ID_FIELD!r}; "
            f"use {schema.CANONICAL_THREAD_ID_FIELD!r}"
        )
    workers[idx].update(updates)
    return workers[idx]


# --- validation record operations ---------------------------------------


def validation_record(
    state: dict,
    task_id: str,
    command: str,
    status: str,
    exit_code: int | None = None,
    summary: str | None = None,
    output: str | None = None,
    executed_by: str | None = None,
) -> dict:
    tasks = state.get("tasks", [])
    idx = _find_index(tasks, "id", task_id)
    if idx is None:
        raise StateError(f"task {task_id!r} does not exist; cannot record validation")

    entry: dict[str, Any] = {"command": command, "status": status}
    if exit_code is not None:
        entry["exit_code"] = exit_code
    if summary is not None:
        entry["summary"] = summary
    if output is not None:
        entry["output"] = output
    if executed_by is not None:
        entry["executed_by"] = executed_by

    results = tasks[idx].setdefault("validation_results", [])
    if not isinstance(results, list):
        raise StateError(
            f"task {task_id!r} has a non-list 'validation_results' field; "
            "cannot append a structured entry"
        )
    results.append(entry)
    return entry


# --- run operations -------------------------------------------------------


def run_record(state: dict, run_id: str, updates: dict) -> dict:
    runs = state.setdefault("runs", [])
    idx = _find_index(runs, "id", run_id)
    if idx is None:
        record = {"id": run_id, **updates}
        runs.append(record)
        return record
    runs[idx].update(updates)
    return runs[idx]


# --- structured validation -------------------------------------------------


def _worker_identity(worker: dict) -> Any:
    """Legacy-tolerant worker identity: prefer 'id', fall back to 'task_id'.

    Orchestra v0 state predates the 'id' field on workers[] entries and used
    task_id as the de-facto unique key. This lets state_validate check
    real v0 data without requiring it to be rewritten.
    """
    return worker.get("id", worker.get("task_id"))


def state_validate(state: dict) -> dict:
    """Run the full structured validation suite and return a report.

    The report always has the shape::

        {
            "ok": bool,
            "checks": [
                {"name": str, "ok": bool, "errors": [str, ...]},
                ...
            ],
        }

    No check is ever skipped or silently passed.
    """
    checks: list[dict] = []

    def add_check(name: str, errors: list[str]) -> None:
        checks.append({"name": name, "ok": len(errors) == 0, "errors": errors})

    # 1. valid JSON is implied by having a `state` dict at all (caller used
    # load_state, which raises StateError on invalid JSON). We still assert
    # top-level shape here.
    shape_errors: list[str] = []
    for key in schema.TOP_LEVEL_KEYS:
        if key not in state:
            shape_errors.append(f"missing top-level key {key!r}")
    if "tasks" in state and not isinstance(state["tasks"], list):
        shape_errors.append("'tasks' must be a list")
    if "workers" in state and not isinstance(state["workers"], list):
        shape_errors.append("'workers' must be a list")
    if "runs" in state and not isinstance(state["runs"], list):
        shape_errors.append("'runs' must be a list")
    add_check("valid_json_shape", shape_errors)

    tasks = state.get("tasks", []) if isinstance(state.get("tasks"), list) else []
    workers = state.get("workers", []) if isinstance(state.get("workers"), list) else []

    # 2. all task ids unique
    task_ids = [t.get("id") for t in tasks]
    dupes = sorted({tid for tid in task_ids if task_ids.count(tid) > 1 and tid is not None})
    errors = [f"duplicate task id: {tid!r}" for tid in dupes]
    errors += [f"task at index {i} is missing an 'id'" for i, tid in enumerate(task_ids) if tid is None]
    add_check("task_ids_unique", errors)

    # 3. all worker ids unique (legacy-tolerant identity)
    worker_ids = [_worker_identity(w) for w in workers]
    dupes = sorted({wid for wid in worker_ids if worker_ids.count(wid) > 1 and wid is not None})
    errors = [f"duplicate worker id: {wid!r}" for wid in dupes]
    errors += [
        f"worker at index {i} has no 'id' or 'task_id' to identify it"
        for i, wid in enumerate(worker_ids)
        if wid is None
    ]
    add_check("worker_ids_unique", errors)

    # 4. every task with thread_id has a matching workers[] entry
    errors = []
    known_worker_task_ids = {w.get("task_id") for w in workers if w.get("task_id")}
    for t in tasks:
        if t.get("thread_id") and t.get("id") not in known_worker_task_ids:
            errors.append(
                f"task {t.get('id')!r} has thread_id {t.get('thread_id')!r} "
                "but no matching workers[] entry (by task_id)"
            )
    add_check("thread_id_tasks_have_worker_entry", errors)

    # 5. no threadId fields remain anywhere in tasks/workers (including followups)
    errors = []

    def scan_for_forbidden(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            if schema.FORBIDDEN_THREAD_ID_FIELD in obj:
                errors.append(f"forbidden field 'threadId' found at {path}")
            for k, v in obj.items():
                scan_for_forbidden(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan_for_forbidden(v, f"{path}[{i}]")

    scan_for_forbidden(tasks, "tasks")
    scan_for_forbidden(workers, "workers")
    add_check("no_legacy_threadId_field", errors)

    # 6. task statuses are from the allowed set
    errors = []
    for t in tasks:
        if "status" not in t:
            errors.append(f"task {t.get('id', '<unknown>')!r} has no 'status'")
            continue
        errors.extend(schema.validate_task_status(t["status"]))
    add_check("task_statuses_allowed", errors)

    # 7. worker task_id references an existing task
    errors = []
    task_id_set = set(task_ids)
    for w in workers:
        tid = w.get("task_id")
        if tid is None:
            errors.append(f"worker {_worker_identity(w)!r} has no 'task_id'")
        elif tid not in task_id_set:
            errors.append(f"worker {_worker_identity(w)!r} references unknown task_id {tid!r}")
    add_check("worker_task_id_references_existing_task", errors)

    # 8. records conform to the canonical schema.
    #
    # schema.validate_task_record / validate_worker_record existed but were
    # never called from anywhere, so a worker record missing its required
    # 'id' and 'provider' could sit in state while state_validate still
    # reported ok. The checks above are deliberately legacy-tolerant
    # (_worker_identity falls back to task_id); this one is strict, so the
    # two together distinguish "identifiable" from "actually conformant".
    errors = []
    for t in tasks:
        errors.extend(schema.validate_task_record(t))
    for w in workers:
        errors.extend(schema.validate_worker_record(w))
    add_check("records_match_canonical_schema", errors)

    overall_ok = all(c["ok"] for c in checks)
    return {"ok": overall_ok, "checks": checks}
