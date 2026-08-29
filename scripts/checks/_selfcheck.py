"""Executable tests for the check harness."""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.checks.harness import check, fail, ok, run  # noqa: E402


@check("selfcheck.pass")
def passing_check() -> None:
    ok("deliberate pass")


@check("selfcheck.fail")
def failing_check() -> None:
    fail("deliberate")


@check("selfcheck.error")
def crashing_check() -> None:
    1 / 0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def invoke_check(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/check.py", *arguments],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        return_code = run()
    lines = output.getvalue().splitlines()
    require(return_code == 1, f"selfcheck run returned {return_code}")
    require(lines[0] == "PASS  selfcheck.pass", f"unexpected pass line: {lines!r}")
    require(lines[1] == "  OK:   deliberate pass", f"unexpected ok line: {lines!r}")
    require(
        lines[2] == "FAIL  selfcheck.fail: deliberate",
        f"unexpected failure line: {lines!r}",
    )
    require(
        lines[3] == "FAIL  selfcheck.error: ZeroDivisionError: division by zero",
        f"unexpected exception line: {lines!r}",
    )
    require(
        lines[-1] == "1 passed, 2 failed, 3 selected of 3 registered",
        f"unexpected selfcheck summary: {lines!r}",
    )

    try:
        check("selfcheck.pass")(lambda: None)
    except ValueError:
        pass
    else:
        raise RuntimeError("duplicate check name was accepted")

    expected_names = [
        "shell.classify_table",
        "shell.metadata_contract",
        "shell.metadata_fails_closed",
        "shell.permission_not_granted",
        "policy.output_limit_clamp",
        "registry.subsets",
    ]
    full_run = invoke_check()
    require(full_run.returncode == 0, f"full run failed: {full_run.stdout!r}")
    require(
        full_run.stdout.splitlines()[-1]
        == "6 passed, 0 failed, 6 selected of 6 registered",
        f"unexpected full-run summary: {full_run.stdout!r}",
    )

    listed = invoke_check("--list")
    require(listed.returncode == 0, f"list failed: {listed.stderr!r}")
    require(listed.stdout.splitlines() == expected_names, f"unexpected list: {listed.stdout!r}")

    for selector in ("shell", "SHELL"):
        selected = invoke_check("--only", selector)
        require(selected.returncode == 0, f"selector {selector!r} failed")
        selected_lines = selected.stdout.splitlines()
        require(
            selected_lines[-1] == "4 passed, 0 failed, 4 selected of 6 registered",
            f"unexpected selector summary: {selected.stdout!r}",
        )
        require(
            [line.removeprefix("PASS  ") for line in selected_lines if line.startswith("PASS  ")]
            == expected_names[:4],
            f"unexpected selected checks: {selected.stdout!r}",
        )

    missing = invoke_check("--only", "nosuchthing")
    require(missing.returncode == 1, "missing selector exited successfully")
    require(
        missing.stdout == "no check matches 'nosuchthing'\n",
        f"unexpected missing-selector output: {missing.stdout!r}",
    )
    print("harness selfcheck passed")


if __name__ == "__main__":
    main()
