"""Executable tests for the check harness."""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
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


def _copy_harness(root: Path) -> None:
    checks = root / "scripts" / "checks"
    checks.mkdir(parents=True)
    (root / "scripts" / "__init__.py").touch()
    (checks / "__init__.py").touch()
    shutil.copy2(REPO_ROOT / "scripts" / "checks" / "harness.py", checks)


def _run_fixture(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.check"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_skip_fixture(root: Path, repository_checks: list[str]) -> None:
    _copy_harness(root)
    definitions = []
    for index, name in enumerate(repository_checks):
        definitions.append(
            f'@check({name!r}, needs_repository=True)\n'
            f"def repository_check_{index}() -> None:\n"
            '    fail("repository-only check executed")'
        )
    source = (
        "from scripts.checks.harness import check, fail, run\n\n"
        + "\n\n".join(definitions)
        + "\n\nraise SystemExit(run())\n"
    )
    (root / "scripts" / "check.py").write_text(source, encoding="utf-8")


def _write_repository_fixture(root: Path, check_name: str) -> None:
    _copy_harness(root)
    docs = root / "docs"
    docs.mkdir()
    (docs / "roadmap.json").touch()
    (root / "scripts" / "check.py").write_text(
        "from scripts.checks.harness import check, fail, run\n\n"
        f"@check({check_name!r}, needs_repository=True)\n"
        "def broken_repository_check() -> None:\n"
        '    fail("deliberately broken repository check")\n\n'
        f"raise SystemExit(run({check_name!r}))\n",
        encoding="utf-8",
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
        lines[-1] == "1 passed, 2 failed, 0 skipped, 3 selected of 3 registered",
        f"unexpected selfcheck summary: {lines!r}",
    )

    try:
        check("selfcheck.pass")(lambda: None)
    except ValueError:
        pass
    else:
        raise RuntimeError("duplicate check name was accepted")

    marked = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import scripts.check; "
                "from scripts.checks.harness import repository_names; "
                "print('\\n'.join(repository_names()))"
            ),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(marked.returncode == 0, f"repository check listing failed: {marked.stderr!r}")
    repository_checks = marked.stdout.splitlines()
    require(bool(repository_checks), "repository-dependent check registry was empty")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        no_git = root / "no-git"
        _write_skip_fixture(no_git, repository_checks)
        skipped = _run_fixture(no_git)
        expected_skip_lines = [
            f"SKIP  {name}: needs the working repository"
            for name in repository_checks
        ]
        require(skipped.returncode == 0, f"skip-only run failed: {skipped.stdout!r}")
        require(
            skipped.stdout.splitlines()
            == [
                *expected_skip_lines,
                f"0 passed, 0 failed, {len(repository_checks)} skipped, "
                f"{len(repository_checks)} selected of {len(repository_checks)} registered",
            ],
            f"no-git skips were wrong: {skipped.stdout!r}",
        )

        working = root / "working"
        broken_name = repository_checks[0]
        _write_repository_fixture(working, broken_name)
        present = _run_fixture(working)
        require(
            present.returncode == 1
            and f"FAIL  {broken_name}: deliberately broken repository check" in present.stdout
            and "SKIP  " not in present.stdout,
            f"broken repository check was skipped or not reported: {present.stdout!r}",
        )

    bytecode_cache = Path(__file__).parent / "__pycache__"
    shutil.rmtree(bytecode_cache, ignore_errors=True)
    direct_import = subprocess.run(
        [sys.executable, "-c", "from scripts.checks import leader"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(
        direct_import.returncode == 0,
        f"direct leader import failed: {direct_import.stderr!r}",
    )
    cached_modules = list(bytecode_cache.glob("*.pyc"))
    require(
        all(path.name.startswith("__init__.cpython-") for path in cached_modules),
        f"direct leader import cached check modules: {cached_modules!r}",
    )
    shutil.rmtree(bytecode_cache, ignore_errors=True)

    listed = invoke_check("--list")
    require(listed.returncode == 0, f"list failed: {listed.stderr!r}")
    require(not bytecode_cache.exists(), "check.py created scripts/checks/__pycache__")
    expected_names = listed.stdout.splitlines()
    require(bool(expected_names), "check registry was empty")
    require(
        set(repository_checks).issubset(expected_names),
        f"repository-dependent checks were missing from --list: {repository_checks!r}",
    )

    # --only selects by substring, so a check name that is a substring of
    # another makes that name un-selectable on its own. The per-name assertion
    # below used to depend on this holding by accident; state it instead.
    collisions = [
        (shorter, longer)
        for shorter in expected_names
        for longer in expected_names
        if shorter != longer and shorter in longer
    ]
    require(
        not collisions,
        f"check names collide under --only substring selection: {collisions!r}",
    )

    target_schema = invoke_check("--list", "--only", "tools.target_keys_match_schemas")
    require(
        target_schema.returncode == 0
        and target_schema.stdout.splitlines() == ["tools.target_keys_match_schemas"],
        f"target schema selector did not match exactly: {target_schema.stdout!r}",
    )

    listed_retry = invoke_check("--list", "--only", "retry")
    require(listed_retry.returncode == 0, f"filtered list failed: {listed_retry.stderr!r}")
    require(
        listed_retry.stdout.splitlines()
        == [name for name in expected_names if "retry" in name.casefold()],
        f"unexpected filtered list: {listed_retry.stdout!r}",
    )

    listed_breakers = invoke_check("--list", "--only", "breaker")
    require(
        listed_breakers.returncode == 0,
        f"filtered breaker list failed: {listed_breakers.stderr!r}",
    )
    require(
        listed_breakers.stdout.splitlines()
        == [name for name in expected_names if "breaker" in name.casefold()],
        f"unexpected filtered breaker list: {listed_breakers.stdout!r}",
    )

    shell_selected = [name for name in expected_names if "shell" in name.casefold()]
    for selector in ("shell", "SHELL"):
        selected = invoke_check("--list", "--only", selector)
        require(selected.returncode == 0, f"selector {selector!r} failed")
        require(
            selected.stdout.splitlines() == shell_selected,
            f"selector {selector!r} returned unexpected names: {selected.stdout!r}",
        )

    missing = invoke_check("--only", "nosuchthing")
    require(missing.returncode == 1, "missing selector exited successfully")
    require(
        missing.stdout == "no check matches 'nosuchthing'\n",
        f"unexpected missing-selector output: {missing.stdout!r}",
    )

    mixed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts.checks import _selfcheck, shell_and_registry; "
                "from scripts.checks.harness import run; "
                "raise SystemExit(run())"
            ),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(mixed.returncode == 1, "mixed selfcheck registry exited successfully")
    require(
        mixed.stdout == "refusing to mix selfcheck fixtures with real checks\n",
        f"unexpected mixed-registry output: {mixed.stdout!r}",
    )

    print("harness selfcheck passed")


if __name__ == "__main__":
    main()
