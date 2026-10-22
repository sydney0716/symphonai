#!/usr/bin/env python3
"""Run the isolated, billed model task evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = Path(__file__).resolve().parent / "eval" / "tasks"
sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    task_names = sorted(path.name for path in TASKS_ROOT.iterdir() if path.is_dir())
    parser = argparse.ArgumentParser(description="Evaluate a vendor model on fixed SymphonAI loop tasks.")
    parser.add_argument("--provider", required=True, help="provider name, such as openai or anthropic")
    parser.add_argument("--model", required=True, help="vendor model name")
    parser.add_argument("--task", choices=task_names, help="run one task instead of all three")
    parser.add_argument("--out", type=Path, help="write the result rows as JSON to this path")
    return parser


def _api_key_environment(provider: str) -> str | None:
    from symphonai_host.server import PROVIDERS

    return next((variable for name, variable, _ in PROVIDERS if name == provider), None)


def _git(project: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments], cwd=project, capture_output=True, text=True, check=False,
    )


def _init_project(source: Path, project: Path) -> None:
    shutil.copytree(source, project)
    for arguments in (("init", "--quiet"), ("config", "user.email", "eval@example.test"),
                      ("config", "user.name", "SymphonAI Eval"), ("add", "."),
                      ("commit", "--quiet", "-m", "evaluation task")):
        result = _git(project, *arguments)
        if result.returncode:
            raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")


def _copy_hidden(task: Path, project: Path) -> None:
    tests = project / "tests"
    for source in (task / "hidden").rglob("*"):
        if source.is_file():
            target = tests / source.relative_to(task / "hidden")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _run_tests(project: Path) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest"], cwd=project,
        capture_output=True, text=True, check=False,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def _build_row(
    task: str,
    state: str,
    reason: str,
    visible_validation: str,
    hidden_tests: str,
    usage: dict | None,
    wall_time_seconds: float,
) -> dict:
    total = usage.get("total") if isinstance(usage, dict) else None
    total = total if isinstance(total, dict) else None
    input_tokens = total.get("input_tokens") if total is not None else None
    output_tokens = total.get("output_tokens") if total is not None else None
    total_tokens = (
        input_tokens + output_tokens
        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
        else "unavailable"
    )
    return {
        "task": task,
        "state": state,
        "reason": reason,
        "visible_validation": visible_validation,
        "hidden_tests": hidden_tests,
        "total_tokens": total_tokens,
        "calls": total.get("calls", "unavailable") if total is not None else "unavailable",
        "usage": total,
        "wall_time_seconds": wall_time_seconds,
    }


def _loop_command(project: Path, provider: str, model: str, goal: str) -> list[str]:
    executable = shutil.which("symphonai")
    command = [executable] if executable else [sys.executable, "-m", "symphonai_host.headless"]
    return [*command, "loop", "--repo-root", str(project), "--provider", provider,
            "--model", model, "--json", goal]


def _run_task(task: Path, provider: str, model: str) -> dict:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="symphonai-eval-") as temporary:
        root = Path(temporary)
        project = root / "project"
        home = root / "home"
        sessions = root / "sessions"
        _init_project(task / "project", project)
        home.mkdir()
        sessions.mkdir()
        environment = os.environ.copy()
        environment["SYMPHONAI_HOME"] = str(home)
        environment["SYMPHONAI_SESSIONS_DIR"] = str(sessions)
        goal = (task / "goal.txt").read_text(encoding="utf-8").strip()
        result = subprocess.run(
            _loop_command(project, provider, model, goal), cwd=REPO_ROOT,
            env=environment, capture_output=True, text=True, check=False,
        )
        wall_time = round(time.monotonic() - started, 2)
        try:
            data = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            data = {}
        state = data.get("state", "error" if result.returncode else "unknown")
        reason = data.get("reason") or (result.stderr.strip() if result.returncode else "")
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        validation = summary.get("validation", [])
        visible = "; ".join(
            f"{item.get('command', 'validation')}: exit {item.get('exit', 'unknown')}"
            for item in validation if isinstance(item, dict)
        ) or "unavailable"
        hidden = "not run"
        if state == "awaiting_commit" and data.get("branch"):
            checkout = _git(project, "checkout", "--quiet", "--detach", data["branch"])
            if checkout.returncode:
                hidden = f"failed to check out loop branch: {checkout.stderr.strip()}"
            else:
                visible_ok, visible_output = _run_tests(project)
                visible = "passed" if visible_ok else f"failed (exit; {visible_output[-240:]})"
                _copy_hidden(task, project)
                hidden_ok, hidden_output = _run_tests(project)
                hidden = "passed" if hidden_ok else f"failed ({hidden_output[-240:]})"
        return _build_row(
            task.name, state, reason, visible, hidden, data.get("usage"), wall_time,
        )


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    key_name = _api_key_environment(arguments.provider)
    if key_name is None:
        print(f"unknown provider: {arguments.provider}", file=sys.stderr)
        return 2
    if not os.environ.get(key_name):
        print(f"no API key found; set {key_name} in the environment", file=sys.stderr)
        return 2

    tasks = [TASKS_ROOT / arguments.task] if arguments.task else sorted(
        path for path in TASKS_ROOT.iterdir() if path.is_dir()
    )
    print(f"This eval makes real, billed API calls; tasks to run: {len(tasks)}.", flush=True)
    print("task | final state | reason | visible validation | hidden tests | total tokens | calls | wall time (s)")
    rows = []
    for task in tasks:
        row = _run_task(task, arguments.provider, arguments.model)
        rows.append(row)
        print(" | ".join(str(row[key]) for key in (
            "task", "state", "reason", "visible_validation", "hidden_tests",
            "total_tokens", "calls", "wall_time_seconds",
        )), flush=True)
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
