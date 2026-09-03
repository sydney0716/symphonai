#!/usr/bin/env python3
"""Build the loopback host as a Tauri-named PyInstaller sidecar."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BINARY_BYTES = 40 * 1_000_000
# A clean macOS arm64 onedir build measured 7.45s cold and about 3s warm.
# This is a regression gate, not an unachieved product-performance target.
MAX_START_SECONDS = 10.0
START_TIME_RESOLUTION_SECONDS = 0.1


def target_triple() -> str:
    """Return Rust's target spelling, or the equivalent Tauri fallback."""
    try:
        rustc = subprocess.run(
            ["rustc", "-vV"], text=True, capture_output=True, check=False
        )
    except FileNotFoundError:
        rustc = None
    if rustc is not None:
        for line in rustc.stdout.splitlines():
            if line.startswith("host: "):
                return line.removeprefix("host: ")

    machine = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "arm64": "aarch64",
    }.get(platform.machine().lower(), platform.machine().lower())
    triples = {
        "darwin": f"{machine}-apple-darwin",
        "linux": f"{machine}-unknown-linux-gnu",
        "win32": f"{machine}-pc-windows-msvc",
    }
    try:
        return triples[sys.platform]
    except KeyError as exc:
        raise SystemExit(
            f"cannot derive a Tauri target triple for platform {sys.platform!r}"
        ) from exc


def verify_handshake_without_python(binary: Path) -> float:
    """Prove the frozen executable starts independently of PATH's Python."""
    environment = os.environ.copy()
    environment["PATH"] = ""
    started = time.monotonic()
    process = subprocess.Popen(
        [str(binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        if process.stdout is None:
            raise RuntimeError("sidecar did not expose stdout")
        line: list[str] = []
        ready = threading.Event()

        def read_handshake() -> None:
            line.append(process.stdout.readline())
            ready.set()

        threading.Thread(target=read_handshake, daemon=True).start()
        if not ready.wait(MAX_START_SECONDS + START_TIME_RESOLUTION_SECONDS / 2):
            elapsed = time.monotonic() - started
            raise RuntimeError(
                f"host binary start time {elapsed:.2f}s exceeds {MAX_START_SECONDS:.1f}s"
            )
        handshake = json.loads(line[0])
        if not isinstance(handshake.get("port"), int) or not isinstance(
            handshake.get("token"), str
        ):
            raise RuntimeError(f"sidecar emitted an invalid handshake: {line[0]!r}")
        return round(time.monotonic() - started, 1)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-smoke", action="store_true")
    arguments = parser.parse_args()

    triple = target_triple()
    binary_name = f"SymphonAI-host-{triple}"
    output = ROOT / "dist" / binary_name
    if output.is_file():
        # 17o deliberately changes one-file output into an onedir bundle.
        # A previous generated file cannot occupy the required directory.
        output.unlink()
    environment = os.environ.copy()
    environment["SYMPHONAI_HOST_BINARY_NAME"] = binary_name
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(ROOT / "dist"),
            "--workpath",
            str(ROOT / "build"),
            str(ROOT / "packaging" / "SymphonAI-host.spec"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )

    binary = output / binary_name if output.is_dir() else output
    size_bytes = (
        sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if output.is_dir()
        else binary.stat().st_size
    )
    size_mb = size_bytes / 1_000_000
    print(f"{binary} ({size_mb:.1f} MB)")
    if size_bytes > MAX_BINARY_BYTES:
        raise SystemExit("host binary exceeds 40 MB")

    # PyInstaller uses the spec stem for its work directory even when EXE has
    # a target-suffixed output name. Check its generated reports rather than
    # trusting that `excludes` had the intended effect.
    report_directory = ROOT / "build" / "SymphonAI-host"
    warn = report_directory / "warn-SymphonAI-host.txt"
    xref = report_directory / "xref-SymphonAI-host.html"
    for report in (warn, xref):
        if not report.exists():
            raise SystemExit(f"PyInstaller did not produce a dependency report: {report}")
    report_text = "\n".join(
        report.read_text(encoding="utf-8", errors="replace") for report in (warn, xref)
    )
    for excluded in ("symphonai_tui", "textual"):
        if excluded in report_text:
            raise SystemExit(f"PyInstaller dependency report includes excluded module {excluded}")

    start_seconds = verify_handshake_without_python(binary)
    print(f"OK:   sidecar emitted its first handshake line with PATH cleared in {start_seconds:.2f}s")
    if start_seconds > MAX_START_SECONDS:
        raise SystemExit(
            f"host binary start time {start_seconds:.2f}s exceeds {MAX_START_SECONDS:.1f}s"
        )
    if not arguments.skip_smoke:
        subprocess.run(
            [sys.executable, "scripts/smoke_host.py", "--binary", str(binary)],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
