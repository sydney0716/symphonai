"""Checks for the Tauri shell and its packaged host sidecar."""

from __future__ import annotations

import json
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

from scripts.checks.harness import check, fail


ROOT = Path(__file__).resolve().parents[2]
TAURI_ROOT = ROOT / "packaging" / "tauri"
RUST_MAIN = TAURI_ROOT / "src" / "main.rs"
SIDECAR_NAME = "SymphonAI-host-aarch64-apple-darwin"


class _PageReferences(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del tag
        self.references.extend(
            value
            for name, value in attrs
            if name in {"href", "src"} and value is not None
        )


def _page_asset_failures(app_root: Path, source: str) -> list[str]:
    parser = _PageReferences()
    parser.feed(source)
    failures = []
    for reference in parser.references:
        if reference.startswith("/"):
            failures.append(f"absolute page reference: {reference!r}")
            continue
        if not (app_root / reference).is_file():
            failures.append(f"missing page asset: {reference!r}")
    return failures


def _run_git(*arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        fail(f"could not run git to inspect page assets: {exc}")


def _page_paths() -> list[Path]:
    app_root = ROOT / "symphonai_app"
    index = app_root / "index.html"
    parser = _PageReferences()
    parser.feed(index.read_text(encoding="utf-8"))
    return [index, *(app_root / reference for reference in parser.references)]


def _rust() -> str:
    return RUST_MAIN.read_text(encoding="utf-8")


@check("packaging.tauri_config")
def check_tauri_config() -> None:
    config = json.loads((TAURI_ROOT / "tauri.conf.json").read_text(encoding="utf-8"))
    if config.get("identifier") != "com.symphonai.app":
        fail(f"Tauri identifier was wrong: {config.get('identifier')!r}")
    if config.get("build", {}).get("frontendDist") != "../../symphonai_app":
        fail("Tauri shell did not package the shared browser application")
    if config.get("app", {}).get("windows") != []:
        fail("Tauri created a window before receiving the host handshake")
    resources = config.get("bundle", {}).get("resources", {})
    source = f"../../dist/{SIDECAR_NAME}"
    if resources != {source: SIDECAR_NAME}:
        fail(f"Tauri did not package the complete target sidecar directory: {resources!r}")


@check("packaging.sidecar_handshake")
def check_sidecar_handshake() -> None:
    source = _rust()
    required = (
        "Command::new(path)",
        ".stdout(Stdio::piped())",
        ".stderr(Stdio::piped())",
        ".read_line(&mut line)",
        "host handshake was not valid JSON",
        "host handshake had an invalid port",
        "host handshake had an invalid token",
        "format!(\"host exited before handshake: {stderr_text}\")",
    )
    missing = [fragment for fragment in required if fragment not in source]
    if missing:
        fail(f"sidecar handshake omitted required behaviour: {missing!r}")
    if source.count(".read_line(&mut line)") != 1:
        fail("shell did not read exactly one host handshake line")


@check("packaging.sidecar_lifecycle")
def check_sidecar_lifecycle() -> None:
    source = _rust()
    required = (
        "libc::SIGTERM",
        "SHUTDOWN_GRACE",
        "child.kill()",
        "WindowEvent::CloseRequested",
        "RunEvent::ExitRequested",
        "ctrlc::set_handler",
    )
    missing = [fragment for fragment in required if fragment not in source]
    if missing:
        fail(f"sidecar lifecycle omitted required behaviour: {missing!r}")
    term = source.index("libc::SIGTERM")
    wait = source.index("let deadline")
    kill = source.index("child.kill()")
    if not term < wait < kill:
        fail("sidecar was not given a graceful stop before SIGKILL")


@check("packaging.shell_security")
def check_shell_security() -> None:
    source = _rust()
    forbidden = (".arg(", "?token=", "println!", "eprintln!", "dbg!")
    found = [fragment for fragment in forbidden if fragment in source]
    if found:
        fail(f"shell exposed data through argv, URL, or logging: {found!r}")
    required = (
        'WebviewUrl::App("index.html".into())',
        "__symphonaiShell",
        "serde_json::to_string(&handshake)",
    )
    missing = [fragment for fragment in required if fragment not in source]
    if missing:
        fail(f"shell bridge was incomplete: {missing!r}")


@check("packaging.app_seam")
def check_app_seam() -> None:
    source_root = ROOT / "symphonai_app" / "src"
    references = []
    for path in sorted(source_root.glob("*.js")):
        if path.name == "host_handle.js":
            continue
        text = path.read_text(encoding="utf-8")
        for marker in ("__TAURI__", "__TAURI_INTERNALS__"):
            if marker in text:
                references.append((path.name, marker))
    if references:
        fail(f"application source bypassed the host seam: {references!r}")
    app_source = (source_root / "app.js").read_text(encoding="utf-8")
    if 'const handshake = resolveHost(global);' not in app_source:
        fail("app.js did not resolve its handshake through host_handle.js")
    if "const handshake = global.__symphonai" in app_source:
        fail("app.js still read the page handshake directly")


@check("packaging.bundle_input")
def check_bundle_input() -> None:
    bundle = ROOT / "dist" / SIDECAR_NAME
    executable = bundle / SIDECAR_NAME
    internal = bundle / "_internal"
    if not executable.is_file() or not internal.is_dir():
        fail(f"existing onedir sidecar was incomplete: {bundle}")
    if not any(path.is_file() for path in internal.rglob("*")):
        fail("sidecar _internal directory was empty")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required = ("scripts/build_host.py", "cargo tauri build", "_internal/")
    missing = [fragment for fragment in required if fragment not in readme]
    if missing:
        fail(f"README omitted Tauri build instructions: {missing!r}")


@check("packaging.page_assets")
def check_page_assets() -> None:
    app_root = ROOT / "symphonai_app"
    source = (app_root / "index.html").read_text(encoding="utf-8")
    failures = _page_asset_failures(app_root, source)
    if failures:
        fail(f"page asset references were invalid: {failures!r}")

    with tempfile.TemporaryDirectory() as temporary:
        fixture_root = Path(temporary)
        existing = fixture_root / "existing.css"
        existing.touch()
        absolute = _page_asset_failures(
            fixture_root,
            f'<link href="{existing}">',
        )
        if not any("absolute page reference" in item for item in absolute):
            fail("page asset validation accepted an absolute reference")
        missing = _page_asset_failures(
            fixture_root,
            '<script src="missing.js"></script>',
        )
        if not any("missing page asset" in item for item in missing):
            fail("page asset validation accepted a missing reference")


@check("packaging.page_tracked")
def check_page_tracked() -> None:
    for path in _page_paths():
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            fail(f"page asset resolved outside the repository: {path}")

        tracked = _run_git("ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode == 1:
            fail(f"page asset is not tracked: {relative}")
        if tracked.returncode != 0:
            detail = tracked.stderr.strip() or f"exit code {tracked.returncode}"
            fail(f"git could not inspect tracked page assets: {detail}")

        ignored = _run_git("check-ignore", "--no-index", "--quiet", "--", relative)
        if ignored.returncode == 0:
            fail(f"page asset is ignored: {relative}")
        if ignored.returncode != 1:
            detail = ignored.stderr.strip() or f"exit code {ignored.returncode}"
            fail(f"git could not inspect ignored page assets: {detail}")
