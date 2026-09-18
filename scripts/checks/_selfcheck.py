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


EXPECTED_REPOSITORY_CHECKS = (
    "app.roadmap",
    "app.spec_view",
    "app.real_roadmap",
    "packaging.bundle_input",
    "packaging.page_tracked",
    "roadmap_data.schema",
    "roadmap_data.spec_bindings",
    "plugins.manifest_validation",
)


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


def _write_skip_fixture(root: Path) -> None:
    _copy_harness(root)
    definitions = []
    for index, name in enumerate(EXPECTED_REPOSITORY_CHECKS):
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


def _write_repository_fixture(root: Path) -> Path:
    _copy_harness(root)
    shutil.copy2(
        REPO_ROOT / "scripts" / "checks" / "roadmap_data.py",
        root / "scripts" / "checks" / "roadmap_data.py",
    )
    docs = root / "docs"
    docs.mkdir()
    for name in ("roadmap.json", "roadmap.schema.json"):
        shutil.copy2(REPO_ROOT / "docs" / name, docs / name)
    (root / ".git").mkdir()
    (root / "scripts" / "check.py").write_text(
        "from scripts.checks import roadmap_data\n"
        "from scripts.checks.harness import run\n"
        'raise SystemExit(run("roadmap_data.schema"))\n',
        encoding="utf-8",
    )
    return docs / "roadmap.schema.json"


def _publish_verification_block() -> str:
    source = (REPO_ROOT / "publish.sh").read_text(encoding="utf-8")
    begin_marker = "# BEGIN published snapshot verification"
    end_marker = "# END published snapshot verification"
    try:
        begin = source.index(begin_marker) + len(begin_marker)
        end = source.index(end_marker, begin)
    except ValueError:
        raise RuntimeError("publish.sh omitted the snapshot verification block") from None
    return source[begin:end].strip()


def _write_publish_check(root: Path, skipped: tuple[str, ...], exit_code: int) -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    source = ["import sys"]
    source.extend(
        f'print("SKIP  {name}: needs the working repository")'
        for name in skipped
    )
    source.extend(
        (
            'print("650 passed, 0 failed, 7 skipped, 657 selected of 657 registered")',
            f"raise SystemExit({exit_code})",
        )
    )
    (scripts / "check.py").write_text("\n".join(source) + "\n", encoding="utf-8")


def _run_publish_verification(root: Path) -> subprocess.CompletedProcess[str]:
    command = f"set -eu\nwork=$1\n{_publish_verification_block()}"
    return subprocess.run(
        ["sh", "-c", command, "publish-selfcheck", str(root)],
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
        lines[-1] == "1 passed, 2 failed, 0 skipped, 3 selected of 3 registered",
        f"unexpected selfcheck summary: {lines!r}",
    )

    try:
        check("selfcheck.pass")(lambda: None)
    except ValueError:
        pass
    else:
        raise RuntimeError("duplicate check name was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        no_git = root / "no-git"
        _write_skip_fixture(no_git)
        skipped = _run_fixture(no_git)
        expected_skip_lines = [
            f"SKIP  {name}: needs the working repository"
            for name in EXPECTED_REPOSITORY_CHECKS
        ]
        require(skipped.returncode == 0, f"skip-only run failed: {skipped.stdout!r}")
        require(
            skipped.stdout.splitlines()
            == [
                *expected_skip_lines,
                # Derived, not copied: a count typed beside a list it describes
                # goes stale the moment the list grows, which is how this
                # assertion came to say seven of an eight-name set.
                f"0 passed, 0 failed, {len(EXPECTED_REPOSITORY_CHECKS)} skipped, "
                f"{len(EXPECTED_REPOSITORY_CHECKS)} selected of "
                f"{len(EXPECTED_REPOSITORY_CHECKS)} registered",
            ],
            f"no-git skips were wrong: {skipped.stdout!r}",
        )

        working = root / "working"
        schema = _write_repository_fixture(working)
        present = _run_fixture(working)
        require(
            present.returncode == 0 and "SKIP  " not in present.stdout,
            f"working-repository check was skipped or failed: {present.stdout!r}",
        )
        schema.rename(schema.with_suffix(".hidden"))
        hidden = _run_fixture(working)
        require(hidden.returncode == 1, "broken repository check exited successfully")
        require(
            "FAIL  roadmap_data.schema:" in hidden.stdout,
            f"broken repository check did not fail normally: {hidden.stdout!r}",
        )

        publish_root = root / "publish"
        _write_publish_check(publish_root, tuple(sorted(EXPECTED_REPOSITORY_CHECKS)), 1)
        broken = _run_publish_verification(publish_root)
        require(broken.returncode != 0, "broken published snapshot was accepted")
        require(
            "REFUSING: the published snapshot fails its own checks; nothing was pushed"
            in broken.stdout,
            f"broken snapshot omitted the refusal: {broken.stdout!r}",
        )

        exact_root = root / "exact"
        expected_sorted = tuple(sorted(EXPECTED_REPOSITORY_CHECKS))
        _write_publish_check(exact_root, expected_sorted, 0)
        exact = _run_publish_verification(exact_root)
        require(exact.returncode == 0, f"exact publish skips were refused: {exact.stdout!r}")

        for label, names in (
            ("missing", expected_sorted[:-1]),
            ("extra", (*expected_sorted, "unexpected.eighth")),
        ):
            mismatch_root = root / label
            _write_publish_check(mismatch_root, names, 0)
            mismatch = _run_publish_verification(mismatch_root)
            require(mismatch.returncode != 0, f"{label} publish skips were accepted")
            require(
                "REFUSING: the published snapshot skipped an unexpected set of checks; "
                "nothing was pushed" in mismatch.stdout,
                f"{label} skip mismatch omitted the refusal: {mismatch.stdout!r}",
            )

    # The registry's names come from the registry. A hand-maintained copy of
    # them forced an edit in 61 reports, produced two false failures when it
    # went stale, and never caught a defect -- and the isolation loop below
    # proved registration order cannot affect any outcome, so the order it
    # pinned guarded nothing. `--list` is the source; what is asserted is that
    # the CLI agrees with the registry it prints, not that someone retyped it.
    listed = invoke_check("--list")
    require(listed.returncode == 0, f"list failed: {listed.stderr!r}")
    expected_names = listed.stdout.splitlines()
    require(bool(expected_names), "check registry was empty")

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
    require(
        marked.stdout.splitlines() == list(EXPECTED_REPOSITORY_CHECKS),
        f"repository-dependent check set was wrong: {marked.stdout!r}",
    )

    full_run = invoke_check()
    require(full_run.returncode == 0, f"full run failed: {full_run.stdout!r}")
    require(
        full_run.stdout.splitlines()[-1]
        == (
            f"{len(expected_names)} passed, 0 failed, 0 skipped, "
            f"{len(expected_names)} selected of {len(expected_names)} registered"
        ),
        f"unexpected full-run summary: {full_run.stdout!r}",
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

    target_schema = invoke_check("--only", "tools.target_keys_match_schemas")
    require(
        "  OK:   10 target keys match declared schemas\n" in target_schema.stdout,
        f"target schema check did not exercise all ten rows: {target_schema.stdout!r}",
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

    shell_selected = [name for name in expected_names if "shell" in name.lower()]
    for selector in ("shell", "SHELL"):
        selected = invoke_check("--only", selector)
        require(selected.returncode == 0, f"selector {selector!r} failed")
        selected_lines = selected.stdout.splitlines()
        require(
            selected_lines[-1]
            == (
                f"{len(shell_selected)} passed, 0 failed, 0 skipped, "
                f"{len(shell_selected)} selected of {len(expected_names)} registered"
            ),
            f"unexpected selector summary: {selected.stdout!r}",
        )
        require(
            [
                line.removeprefix("PASS  ")
                for line in selected_lines
                if line.startswith("PASS  ")
            ]
            == shell_selected,
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
    bytecode_run = invoke_check("--only", "content.message_normalization")
    require(bytecode_run.returncode == 0, f"bytecode check run failed: {bytecode_run.stdout!r}")
    require(not bytecode_cache.exists(), "check.py created scripts/checks/__pycache__")
    print("harness selfcheck passed")


if __name__ == "__main__":
    main()
