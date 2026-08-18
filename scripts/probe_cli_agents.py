#!/usr/bin/env python3
"""Dry-run (by default) capability probe for the claude/codex/gemini CLIs.

Default behavior only checks whether each CLI binary is on PATH and prints
the CANDIDATE flags recorded in `orchestra_agents.probes` -- it never
invokes any of the CLIs. Pass `--live` to additionally run each found
binary's `--version`/`--help` candidates (nothing else, ever) and report
whether they succeeded, as a first step toward telling candidate flags
apart from confirmed ones.

This script must never run a real agent task against claude/codex/gemini,
with or without --live, and never runs anything destructive. See
docs/orchestra-agent-adapters.md for the full design.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_agents.probes import ALL_PROBES, CliProbeSpec  # noqa: E402


def check_binary(spec: CliProbeSpec) -> str | None:
    """Return the resolved path to spec.binary if it's on PATH, else None."""
    return shutil.which(spec.binary)


def run_live_flag_check(binary_path: str, flag: str, *, timeout: float = 5.0) -> tuple[bool, str]:
    """Run `binary_path <flag>` and report success plus a short summary.

    Only ever called for entries in version_flag_candidates/
    help_flag_candidates. Never called with a task prompt and never in a
    way that could mutate any state.
    """
    args = [binary_path, *flag.split()]
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"error running {args}: {exc}"
    ok = proc.returncode == 0
    output_lines = (proc.stdout or proc.stderr or "").strip().splitlines()
    first_line = output_lines[0] if output_lines else ""
    return ok, f"exit={proc.returncode} first_line={first_line!r}"


def _print_candidates(label: str, flags: tuple[str, ...]) -> None:
    print(f"  candidate {label} flags:", ", ".join(flags) if flags else "(none recorded)")


def report_spec(spec: CliProbeSpec, *, live: bool) -> None:
    print(f"== {spec.binary} ==")
    binary_path = check_binary(spec)
    print(f"  binary: {binary_path if binary_path else 'NOT FOUND on PATH'}")

    _print_candidates("non_interactive", spec.non_interactive_flag_candidates)
    _print_candidates("structured_output", spec.structured_output_flag_candidates)
    _print_candidates("resume", spec.resume_flag_candidates)
    _print_candidates("cwd", spec.cwd_flag_candidates)
    _print_candidates("sandbox_permission", spec.sandbox_permission_flag_candidates)
    print(f"  notes: {spec.notes}")

    if not live:
        print("  live check: skipped (pass --live to run --version/--help only)")
        return

    if binary_path is None:
        print("  live check: skipped (binary not found)")
        return

    for flag in (*spec.version_flag_candidates, *spec.help_flag_candidates):
        ok, detail = run_live_flag_check(binary_path, flag)
        print(f"  live [{flag}]: {'OK' if ok else 'FAIL'} ({detail})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Also run each found binary's --version/--help candidates. "
            "Still never runs a real agent task. Default is dry-run only "
            "(no subprocess calls at all)."
        ),
    )
    args = parser.parse_args(argv)

    print("Orchestra CLI capability probe")
    print("All flags below are CANDIDATES from orchestra_agents.probes, not confirmed facts.")
    print(f"Mode: {'live (--version/--help only)' if args.live else 'dry-run (no subprocess calls)'}")
    print()

    for spec in ALL_PROBES:
        report_spec(spec, live=args.live)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
