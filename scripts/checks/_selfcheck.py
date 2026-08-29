"""Executable tests for the check harness."""

from __future__ import annotations

import contextlib
import io
import shutil
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
        "content.message_normalization",
        "content.attachment_construction",
        "content.system_attachments_refused",
        "content.provider_encoding",
        "tools.localtool_contract",
        "tools.metadata_contract",
        "tools.metadata_absent_from_schemas",
        "ids.non_empty_and_vendor_boundary",
        "ids.request_bodies_keep_internals_private",
        "ids.openai_missing_vendor_id",
        "ids.synthesized_are_unique",
        "retry.backoff_wakes_promptly",
        "retry.cancel_none_uses_time_sleep",
        "retry.http_503_succeeds",
        "retry.timeout_error_succeeds",
        "retry.transient_urlerror_succeeds",
        "retry.certificate_urlerror_fails_immediately",
        "retry.truncated_body_succeeds",
        "retry.permanent_http_statuses_fail_immediately",
        "retry.overall_deadline",
        "retry.http_400_fails_immediately",
        "retry.exhausted_transient_failures",
        "retry.numeric_retry_after",
        "retry.http_date_retry_after",
        "retry.malformed_retry_after",
        "retry.retry_after_capped",
        "retry.keys_redacted",
        "retry.key_prefix_boundary_redacted",
        "discovery.openai_models",
        "discovery.anthropic_models",
        "discovery.gemini_models",
        "discovery.text_model_filter",
        "discovery.shutdown_date_filter",
        "providers.model_overrides",
        "providers.malformed_json",
        "compaction.cancellation_at_entry",
        "compaction.under_budget_unchanged",
        "compaction.preserves_required_context",
        "compaction.impossible_budget_fails",
    ]
    full_run = invoke_check()
    require(full_run.returncode == 0, f"full run failed: {full_run.stdout!r}")
    require(
        full_run.stdout.splitlines()[-1]
        == "45 passed, 0 failed, 45 selected of 45 registered",
        f"unexpected full-run summary: {full_run.stdout!r}",
    )

    listed = invoke_check("--list")
    require(listed.returncode == 0, f"list failed: {listed.stderr!r}")
    require(listed.stdout.splitlines() == expected_names, f"unexpected list: {listed.stdout!r}")

    listed_retry = invoke_check("--list", "--only", "retry")
    require(listed_retry.returncode == 0, f"filtered list failed: {listed_retry.stderr!r}")
    require(
        listed_retry.stdout.splitlines() == expected_names[17:34],
        f"unexpected filtered list: {listed_retry.stdout!r}",
    )

    for selector in ("shell", "SHELL"):
        selected = invoke_check("--only", selector)
        require(selected.returncode == 0, f"selector {selector!r} failed")
        selected_lines = selected.stdout.splitlines()
        require(
            selected_lines[-1] == "4 passed, 0 failed, 4 selected of 45 registered",
            f"unexpected selector summary: {selected.stdout!r}",
        )
        require(
            [line.removeprefix("PASS  ") for line in selected_lines if line.startswith("PASS  ")]
            == [
                "shell.classify_table",
                "shell.metadata_contract",
                "shell.metadata_fails_closed",
                "shell.permission_not_granted",
            ],
            f"unexpected selected checks: {selected.stdout!r}",
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
    bytecode_run = invoke_check("--only", "content.message_normalization")
    require(bytecode_run.returncode == 0, f"bytecode check run failed: {bytecode_run.stdout!r}")
    require(not bytecode_cache.exists(), "check.py created scripts/checks/__pycache__")
    print("harness selfcheck passed")


if __name__ == "__main__":
    main()
