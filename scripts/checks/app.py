"""Checks for the dependency-free JavaScript host-boundary client."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

from scripts.checks.harness import REGISTRY, CheckFailed, check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_MINIMUM = 18
NODE_TESTS = {
    "app.protocol": "protocol.test.js",
    "app.transcript": "transcript.test.js",
    "app.client": "client.test.js",
    "app.init_flow": "init_flow.test.js",
    "app.keys": "keys.test.js",
    "app.turn": "turn.test.js",
    "app.roadmap": "roadmap.test.js",
    "app.spec_view": "spec_view.test.js",
    "app.approvals": "approvals.test.js",
    "app.page": "app.test.js",
}
APP_SOURCE_EXTENSIONS = frozenset({".js", ".mjs", ".json", ".html", ".css"})


def _node() -> str:
    executable = shutil.which("node")
    if executable is None:
        fail("Node 18 or later is required to run symphonai_app checks")
    version = subprocess.run(
        [executable, "--version"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        major = int(version.stdout.strip().removeprefix("v").split(".", 1)[0])
    except ValueError:
        fail(f"could not determine the Node version: {version.stdout.strip()!r}")
    if version.returncode != 0 or major < NODE_MINIMUM:
        fail("Node 18 or later is required to run symphonai_app checks")
    return executable


def _run_node_test(filename: str, *, environment=None) -> None:  # noqa: ANN001
    result = subprocess.run(
        [
            _node(),
            "--experimental-default-type=module",
            "--test",
            f"symphonai_app/test/{filename}",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        fail(f"node --test {filename} failed:\n{output}")


@check("app.node_required")
def node_required() -> None:
    with mock.patch.object(shutil, "which", return_value=None):
        try:
            _node()
        except CheckFailed as exc:
            if "Node 18 or later" not in str(exc):
                fail(f"missing-node failure omitted the required version: {exc}")
        else:
            fail("app checks reported success when node was absent")


@check("app.protocol")
def protocol() -> None:
    _run_node_test("protocol.test.js")


@check("app.transcript")
def transcript() -> None:
    _run_node_test("transcript.test.js")


@check("app.client")
def client() -> None:
    _run_node_test("client.test.js")


@check("app.init_flow")
def init_flow() -> None:
    _run_node_test("init_flow.test.js")


@check("app.keys")
def keys() -> None:
    environment = dict(os.environ)
    environment["SYMPHONAI_KEYS_PATH"] = str(
        REPO_ROOT / "symphonai_app" / "keys.default.json"
    )
    _run_node_test("keys.test.js", environment=environment)


@check("app.turn")
def turn() -> None:
    _run_node_test("turn.test.js")


@check("app.roadmap")
def roadmap() -> None:
    environment = dict(os.environ)
    environment["SYMPHONAI_ROADMAP_PATH"] = str(REPO_ROOT / "docs" / "roadmap.json")
    _run_node_test("roadmap.test.js", environment=environment)


@check("app.spec_view")
def spec_view() -> None:
    _run_node_test("spec_view.test.js")


@check("app.approvals")
def approvals() -> None:
    _run_node_test("approvals.test.js")


@check("app.page")
def page() -> None:
    _run_node_test("app.test.js")


@check("app.test_registration")
def test_registration() -> None:
    test_root = REPO_ROOT / "symphonai_app" / "test"
    present = {path.name for path in test_root.glob("*.test.js")}
    named = set(NODE_TESTS.values())
    unregistered = sorted(present - named)
    missing = sorted(named - present)
    missing_checks = sorted(set(NODE_TESTS) - set(REGISTRY))
    if unregistered:
        fail(f"JavaScript test files have no registered app check: {unregistered!r}")
    if missing:
        fail(f"registered JavaScript test files are missing: {missing!r}")
    if missing_checks:
        fail(f"JavaScript test mappings have no registered check: {missing_checks!r}")


@check("app.import_boundary")
def import_boundary() -> None:
    def scan(app_root: Path) -> None:
        python_files = sorted(
            path.relative_to(app_root).as_posix()
            for path in app_root.rglob("*.py")
        )
        if python_files:
            fail(f"symphonai_app contains Python files: {python_files!r}")
        forbidden = ("symphonai_api", "symphonai_host", "symphonai_tui")
        references = []
        source_files = sorted(
            path
            for path in app_root.rglob("*")
            if path.is_file() and path.suffix in APP_SOURCE_EXTENSIONS
        )
        for path in source_files:
            relative = path.relative_to(app_root).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                fail(f"symphonai_app source is not UTF-8: {relative}")
            for name in forbidden:
                if name in text:
                    references.append((relative, name))
        if references:
            fail(f"symphonai_app crosses the Python boundary: {references!r}")

    def require_pyproject_exclusion(text: str) -> None:
        if 'exclude = ["symphonai_app*"]' not in text:
            fail("pyproject.toml does not exclude symphonai_app from Python packages")

    def expect_failure(operation, expected: tuple[str, ...], label: str) -> None:  # noqa: ANN001
        try:
            operation()
        except Exception as exc:
            if not isinstance(exc, CheckFailed):
                fail(f"{label} raised {type(exc).__name__} instead of CheckFailed")
            message = str(exc)
            if any(part not in message for part in expected):
                fail(f"{label} did not name its finding: {message!r}")
        else:
            fail(f"{label} was not detected")

    app_root = REPO_ROOT / "symphonai_app"
    scan(app_root)
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    require_pyproject_exclusion(pyproject)

    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary)
        (fixture / "src").mkdir()
        (fixture / ".DS_Store").write_bytes(b"\x00\x8c\xff")
        (fixture / "src" / ".DS_Store").write_bytes(b"\x00\x8c\xff")
        (fixture / "src" / "safe.js").write_text("export {};", encoding="utf-8")
        scan(fixture)

    forbidden = ("symphonai_api", "symphonai_host", "symphonai_tui")
    expected_extensions = (".css", ".html", ".js", ".json", ".mjs")
    for extension in expected_extensions:
        for name in forbidden:
            with tempfile.TemporaryDirectory() as temporary:
                fixture = Path(temporary)
                source = fixture / f"reference{extension}"
                source.write_text(name, encoding="utf-8")
                expect_failure(
                    lambda: scan(fixture),
                    (source.name, name),
                    f"{extension} reference to {name}",
                )

    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary)
        source = fixture / "broken.json"
        source.write_bytes(b"\xff")
        expect_failure(
            lambda: scan(fixture),
            (source.name, "not UTF-8"),
            "binary allowed source",
        )

    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary)
        nested = fixture / "nested"
        nested.mkdir()
        source = nested / "runtime.py"
        source.write_text("", encoding="utf-8")
        expect_failure(
            lambda: scan(fixture),
            ("nested/runtime.py", "Python files"),
            "nested Python source",
        )

    expect_failure(
        lambda: require_pyproject_exclusion(""),
        ("pyproject.toml", "exclude"),
        "missing Python package exclusion",
    )
