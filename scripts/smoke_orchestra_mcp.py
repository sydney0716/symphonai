#!/usr/bin/env python3
"""Smoke test for orchestra_mcp's state layer.

Verifies:
  - the existing .orchestra/tasks.json can be loaded (read-only)
  - state_validate passes on the current production file (read-only)
  - a temporary task can be created in a TEMP COPY of the state
  - a fake worker with a fake thread_id can be registered
  - a validation result with executed_by can be recorded
  - production state is never written to

This script never calls atomic_write_state on the real .orchestra/tasks.json.
It copies it into a temp directory first and operates only on that copy.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_mcp import state  # noqa: E402


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def main() -> None:
    prod_path = REPO_ROOT / ".orchestra" / "tasks.json"
    prod_mtime_before = prod_path.stat().st_mtime
    prod_bytes_before = prod_path.read_bytes()

    # 1. existing production tasks.json can be loaded (read-only)
    try:
        prod_state = state.load_state(prod_path)
    except state.StateError as exc:
        fail(f"could not load production state: {exc}")
    ok(f"loaded production state from {prod_path}")

    # 2. state_validate passes on the current production file (read-only)
    report = state.state_validate(prod_state)
    for check in report["checks"]:
        mark = "pass" if check["ok"] else "FAIL"
        print(f"       [{mark}] {check['name']}" + (f": {check['errors']}" if check["errors"] else ""))
    if not report["ok"]:
        fail("state_validate did not pass on production .orchestra/tasks.json")
    ok("state_validate passed on production .orchestra/tasks.json")

    # Everything from here on operates on a TEMP COPY only.
    with tempfile.TemporaryDirectory(prefix="orchestra_mcp_smoke_") as tmpdir:
        tmp_state_path = Path(tmpdir) / "tasks.json"
        shutil.copy2(prod_path, tmp_state_path)
        ok(f"copied production state to temp file {tmp_state_path}")

        tmp_state = state.load_state(tmp_state_path)

        # 3. a temporary task can be created in the temp state file
        task = state.task_create(
            tmp_state,
            {
                "id": "smoke-temp-task",
                "title": "Smoke test temp task",
                "worker": "claude",
                "depends_on": [],
                "scope": ["scripts/smoke_orchestra_mcp.py"],
                "acceptance_criteria": ["smoke test passes"],
                "validation_commands": ["true"],
            },
        )
        if task["status"] != "CREATED":
            fail(f"expected default status CREATED, got {task['status']!r}")
        ok("created temp task 'smoke-temp-task' with default status CREATED")

        # 4. a fake worker with a fake thread_id can be registered
        worker = state.worker_register(
            tmp_state,
            {
                "id": "smoke-temp-worker",
                "provider": "fake-provider",
                "task_id": "smoke-temp-task",
                "thread_id": "fake-thread-id-000",
            },
        )
        if worker["thread_id"] != "fake-thread-id-000":
            fail("worker thread_id was not stored correctly")
        ok("registered fake worker 'smoke-temp-worker' with fake thread_id")

        # 5. a validation result with executed_by can be recorded
        entry = state.validation_record(
            tmp_state,
            task_id="smoke-temp-task",
            command="true",
            status="passed",
            exit_code=0,
            summary="smoke test validation entry",
            executed_by="smoke_script",
        )
        if entry.get("executed_by") != "smoke_script":
            fail("validation_record did not preserve executed_by")
        ok("recorded validation result with executed_by='smoke_script'")

        # Write the temp state atomically and confirm it round-trips and
        # still validates cleanly. This is the ONLY atomic_write_state call
        # in this script, and it targets the temp file, never production.
        state.atomic_write_state(tmp_state_path, tmp_state)
        reloaded = state.load_state(tmp_state_path)
        reloaded_report = state.state_validate(reloaded)
        if not reloaded_report["ok"]:
            fail(f"temp state failed validation after round-trip: {reloaded_report}")
        ok("temp state round-tripped through atomic_write_state and re-validated cleanly")

        found_task = any(t.get("id") == "smoke-temp-task" for t in reloaded["tasks"])
        found_worker = any(w.get("id") == "smoke-temp-worker" for w in reloaded["workers"])
        if not (found_task and found_worker):
            fail("temp task or worker missing after round-trip")
        ok("temp task and worker both present after round-trip")

    # 6. no existing production state was overwritten during the test
    prod_mtime_after = prod_path.stat().st_mtime
    prod_bytes_after = prod_path.read_bytes()
    if prod_mtime_after != prod_mtime_before or prod_bytes_after != prod_bytes_before:
        fail("production .orchestra/tasks.json was modified during the smoke test!")
    ok("production .orchestra/tasks.json is byte-for-byte unchanged")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
