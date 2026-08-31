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
        "compaction.microcompaction_clears_re_derivable",
        "compaction.microcompaction_spares_mutations",
        "compaction.microcompaction_preserves_handles",
        "compaction.microcompaction_is_idempotent",
        "compaction.compaction_prefers_clearing",
        "compaction.compaction_clears_then_drops",
        "context.reconciles_unsplit",
        "context.splits_instructions",
        "context.tool_provenance",
        "context.orphan_tool_warning",
        "context.mismatched_instructions",
        "context.subtotals_and_budget",
        "context.assistant_attachments_and_immutability",
        "context.render_plain_text",
        "context.empty_instruction_set",
        "context.split_covers_rendered_text",
        "context.tool_message_without_result",
        "results.threshold_and_preview",
        "results.preserves_other_fields",
        "results.failures_not_offloaded",
        "results.content_addressing",
        "results.slice_and_limits",
        "results.missing_id",
        "results.tool_metadata",
        "results.concurrent_stores",
        "results.bounds",
        "results.agent_loop_offload",
        "results.registry_opt_in",
        "results.context_report_counts_preview",
        "events.final_identity",
        "events.tool_bracketing",
        "events.provider_failure",
        "events.sink_isolation",
        "events.stream_optional",
        "cancel.pre_cancelled_agent",
        "cancel.tool_repair",
        "cancel.http_read_recheck",
        "cancel.late_response_retained",
        "agent.full_run",
        "agent.base_validation",
        "search.permission_gates",
        "search.ordering_and_pagination",
        "search.grep_modes",
        "search.cancellation",
        "read_file.ranges_and_limits",
        "ledger.entry_lru_bound",
        "ledger.content_lru_bound",
        "ledger.evicted_content_unchanged",
        "ledger.evicted_content_mtime_bump",
        "ledger.dropped_record_refused",
        "ledger.check_refreshes_recency",
        "ledger.recorded_ranges",
        "ledger.partial_view_refused",
        "ledger.cached_output_identical",
        "ledger.mtime_change_rereads",
        "ledger.range_change_rereads",
        "ledger.oversize_unranged_still_refused",
        "ledger.messages_unchanged",
        "ledger.lock_scope",
        "instructions.scope_order",
        "instructions.outside_working_dir",
        "instructions.include_provenance",
        "instructions.depth_cap",
        "instructions.cycle_detection",
        "instructions.forbidden_include",
        "instructions.processing_and_size_warning",
        "instructions.ledger_partial_view",
        "instructions.run_task_wiring",
        "instructions.directive_boundaries",
        "instructions.fence_run_length",
        "instructions.user_scope_include",
        "instructions.user_scope_guards",
        "instructions.user_scope_relative_denylist",
        "instructions.tab_indented_content",
        "instructions.symlinked_user_home",
        "instructions.whitespace_only_entry",
        "edit.not_read_refused",
        "edit.stale_refused",
        "edit.same_content_allowed",
        "edit.ranged_read_stale",
        "edit.long_unranged_stale",
        "edit.single_open",
        "edit.match_count",
        "edit.one_match",
        "edit.multi_edit_sequence",
        "edit.structured_diff",
        "edit.truncated_diff",
        "edit.write_unread_existing",
        "edit.isolated_ledger",
        "edit.narrowed_isolated_ledger",
        "permissions.read_inside_root",
        "permissions.list_inside_root",
        "permissions.write_inside_scope",
        "permissions.write_outside_scope_denied",
        "permissions.forbidden_read_denied",
        "permissions.traversal_denied",
        "permissions.shell_disabled",
        "permissions.shell_always_denied",
        "permissions.typed_reasons",
        "permissions.named_modes_and_equality",
        "permissions.accept_edits",
        "permissions.approval_serialization",
        "shell.process_group_fallback",
        "shell.cancellation_reaps_child",
        "shell.cancellation_kills_group",
        "shell.cancellation_bounded",
        "shell.execution_paths",
        "providers_live.openai_tools_and_model_override",
        "providers_live.gemini_tools_and_model_override",
        "providers_live.gemini_thought_signature",
        "leader.compatibility_names_removed",
        "leader.run_dispatches_subagent",
        "leader.compaction_identity",
        "leader.cancellation_transcript",
        "leader.chat_cancellation",
        "leader.standalone_dispatch",
        "leader.dispatch_metadata",
        "leader.dispatch_pool",
        "leader.typed_event_lifecycle",
        "leader.event_sink_isolation",
        "leader.leader_failure_events",
        "leader.subagent_failure_events",
        "leader.leader_max_turns",
        "leader.subagent_max_turns",
        "leader.fresh_run_subagents",
        "leader.pool_reset",
        "leader.chat_history",
        "leader.anthropic_leader_tool_schema",
        "leader.anthropic_subagent_tool_schemas",
        "leader.subagent_tool_subsets",
        "leader.gemini_dispatch_schema",
        "leader.openai_compatible_tool_schemas",
        "scheduler.partition_barriers",
        "scheduler.metadata_fail_closed",
        "scheduler.classification",
        "scheduler.parallel_reads",
        "scheduler.result_order",
        "scheduler.event_order",
        "scheduler.singleton_compatibility",
        "scheduler.exception_isolation",
        "scheduler.cancellation_repair",
        "scheduler.ledger_locking",
        "plan.path_decisions",
        "plan.command_decisions",
        "plan.real_tools",
    ]
    full_run = invoke_check()
    require(full_run.returncode == 0, f"full run failed: {full_run.stdout!r}")
    require(
        full_run.stdout.splitlines()[-1]
        == "190 passed, 0 failed, 190 selected of 190 registered",
        f"unexpected full-run summary: {full_run.stdout!r}",
    )

    listed = invoke_check("--list")
    require(listed.returncode == 0, f"list failed: {listed.stderr!r}")
    require(listed.stdout.splitlines() == expected_names, f"unexpected list: {listed.stdout!r}")

    for name in expected_names:
        selected_alone = invoke_check("--only", name)
        require(
            selected_alone.returncode == 0,
            f"standalone check {name!r} failed: {selected_alone.stdout!r}",
        )
        selected_alone_lines = selected_alone.stdout.splitlines()
        require(
            selected_alone_lines[0] == f"PASS  {name}",
            f"standalone check selected the wrong name: {selected_alone.stdout!r}",
        )
        require(
            selected_alone_lines[-1]
            == "1 passed, 0 failed, 1 selected of 190 registered",
            f"standalone check selected more than one entry: {selected_alone.stdout!r}",
        )

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
            selected_lines[-1] == "11 passed, 0 failed, 11 selected of 190 registered",
            f"unexpected selector summary: {selected.stdout!r}",
        )
        require(
            [line.removeprefix("PASS  ") for line in selected_lines if line.startswith("PASS  ")]
            == [
                "shell.classify_table",
                "shell.metadata_contract",
                "shell.metadata_fails_closed",
                "shell.permission_not_granted",
                "permissions.shell_disabled",
                "permissions.shell_always_denied",
                "shell.process_group_fallback",
                "shell.cancellation_reaps_child",
                "shell.cancellation_kills_group",
                "shell.cancellation_bounded",
                "shell.execution_paths",
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
