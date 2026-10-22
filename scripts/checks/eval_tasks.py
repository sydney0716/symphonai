"""Offline soundness checks for the model evaluation task fixtures."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from scripts.checks.harness import check, fail, ok
from scripts.eval import _build_row

TASKS_ROOT = Path(__file__).resolve().parents[1] / "eval" / "tasks"


def _copy_hidden(task: Path, project: Path) -> None:
    tests = project / "tests"
    for source in (task / "hidden").rglob("*"):
        if source.is_file():
            target = tests / source.relative_to(task / "hidden")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _unittest(project: Path) -> bool:
    result = subprocess.run(
        ["python3", "-m", "unittest"], cwd=project,
        capture_output=True, text=True, check=False,
    )
    return result.returncode == 0


@check("eval_tasks.tasks_are_sound")
def tasks_are_sound() -> None:
    populated = _build_row(
        "fixture", "awaiting_commit", "", "passed", "passed",
        {"total": {
            "input_tokens": 5, "output_tokens": 2, "calls": 3,
            "cache_read_tokens": 1, "cache_write_tokens": 0,
        }},
        0.0,
    )
    if populated["total_tokens"] != 7 or populated["calls"] != 3:
        fail("usage.total counts were not reported as 7 tokens and 3 calls")
    if populated["usage"] != {
        "input_tokens": 5, "output_tokens": 2, "calls": 3,
        "cache_read_tokens": 1, "cache_write_tokens": 0,
    }:
        fail("row did not preserve the complete usage.total object")
    ok("usage.total input/output tokens sum to 7 and calls report 3")

    missing = _build_row("fixture", "error", "", "unavailable", "not run", None, 0.0)
    if missing["total_tokens"] != "unavailable" or missing["calls"] != "unavailable":
        fail("missing usage did not report both counts as unavailable")
    ok("missing usage reports tokens and calls as unavailable")

    tasks = sorted(path for path in TASKS_ROOT.iterdir() if path.is_dir())
    if len(tasks) != 3:
        fail(f"expected three evaluation tasks, found {len(tasks)}")

    for task in tasks:
        project_source = task / "project"
        hidden_source = task / "hidden"
        patch = task / "solution.patch"
        if not project_source.is_dir() or not hidden_source.is_dir() or not patch.is_file():
            fail(f"{task.name} is missing project/, hidden/, or solution.patch")
        with tempfile.TemporaryDirectory(prefix="symphonai-eval-soundness-") as temporary:
            project = Path(temporary) / "project"
            shutil.copytree(project_source, project)
            _copy_hidden(task, project)
            hidden_fails = not _unittest(project)
            if not hidden_fails:
                fail(f"{task.name}: hidden tests passed on the unmodified project")
            if task.name == "off_by_one":
                visible_project = Path(temporary) / "visible"
                shutil.copytree(project_source, visible_project)
                if _unittest(visible_project):
                    fail("off_by_one: visible test passed on the unmodified project")

            shutil.rmtree(project)
            shutil.copytree(project_source, project)
            applied = subprocess.run(
                ["git", "apply", str(patch)], cwd=project,
                capture_output=True, text=True, check=False,
            )
            if applied.returncode:
                fail(f"{task.name}: solution.patch did not apply: {applied.stderr.strip()}")
            _copy_hidden(task, project)
            if not _unittest(project):
                fail(f"{task.name}: visible and hidden tests failed with solution.patch")
