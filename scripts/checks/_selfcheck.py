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
        "retry.overload_foreground_retries",
        "retry.overload_background_bails",
        "retry.background_still_retries_transient",
        "retry.call_class_defaults_foreground",
        "retry.providers_forward_call_class",
        "retry.leader_subagents_are_background",
        "breaker.opens_on_consecutive_failures",
        "breaker.success_resets",
        "breaker.concurrent_failures_counted",
        "breaker.leader_stops_automatic_compaction",
        "breaker.manual_compaction_still_runs",
        "breaker.subagent_refused_after_repeated_failure",
        "breaker.stopped_repairs_reported",
        "discovery.openai_models",
        "discovery.anthropic_models",
        "discovery.gemini_models",
        "discovery.text_model_filter",
        "discovery.shutdown_date_filter",
        "providers.model_overrides",
        "providers.malformed_json",
        "streaming.default_yields_one_completion",
        "streaming.text_accumulates",
        "streaming.tool_fragments_by_index",
        "streaming.arguments_parsed_once",
        "streaming.synthesized_ids_unique",
        "streaming.no_completion_raises",
        "streaming.terminal_response_wins",
        "streaming.loop_matches_non_streaming",
        "streaming.deltas_emitted",
        "streaming.dropped_events_change_nothing",
        "streaming.retry_before_first_line_only",
        "streaming.cancel_mid_stream",
        "streaming.sse_parsing",
        "streaming.anthropic_matches_non_streaming",
        "streaming.openai_matches_non_streaming",
        "streaming.compatible_shares_openai_mapping",
        "streaming.anthropic_partial_json",
        "streaming.openai_parallel_tool_calls",
        "streaming.anthropic_error_event",
        "streaming.openai_empty_choices_ignored",
        "streaming.compatible_without_stream_usage",
        "streaming.truncated_stream_raises",
        "streaming.stream_flag_present",
        "streaming.non_streaming_path_unchanged",
        "streaming.openai_error_event",
        "streaming.gemini_matches_non_streaming",
        "streaming.gemini_thought_signature_round_trip",
        "streaming.gemini_no_signature_no_key",
        "streaming.gemini_two_calls",
        "streaming.gemini_block_reason",
        "streaming.gemini_truncated_stream",
        "streaming.gemini_url_has_no_key",
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
        "tool_results.memory_only_touches_no_disk",
        "tool_results.disk_write_and_mode",
        "tool_results.disk_write_atomic",
        "tool_results.duplicate_store_skips_write",
        "tool_results.cold_store_reads_disk",
        "tool_results.cold_read_preserves_newlines",
        "tool_results.disk_hit_readmits_to_memory",
        "tool_results.handle_pattern_rejected",
        "tool_results.unreadable_file_is_none",
        "tool_results.eviction_keeps_file",
        "tool_results.prune_bounds_directory",
        "tool_results.write_failure_raises",
        "tool_results.tool_resolves_disk_handle",
        "tool_results.resume_fallback_directory",
        "tool_results.resume_chain_resolves_ancestor_handle",
        "serialization.message_round_trip",
        "serialization.tool_result_round_trip",
        "serialization.provider_metadata_verbatim",
        "serialization.unknown_kind_rejected",
        "session.no_transcript_no_files",
        "session.record_sequence",
        "session.tool_records",
        "session.cancellation_record",
        "session.failure_record",
        "session.truncated_tail_recovers",
        "session.corrupt_middle_raises",
        "session.subagent_transcript_separate",
        "session.meta_atomic_replace",
        "session.request_record_has_no_secret",
        "session.directory_outside_repo_root",
        "session.sessions_root_env_override",
        "session.load_round_trip",
        "session.load_ignores_unknown_record_type",
        "session.resume_continues_conversation",
        "session.resume_is_a_new_run",
        "session.fork_prefix_only",
        "session.fork_rejects_non_message",
        "session.fork_rejects_unknown_record",
        "session.fork_rejects_unanswered_tool_calls",
        "session.fork_leaves_original_untouched",
        "session.resume_after_truncated_tail",
        "session.schema_version_from_the_future",
        "session.seed_messages_persisted",
        "session.resume_round_trips_a_real_run",
        "session.chat_persists_each_turn_once",
        "session.resumed_run_transcript_is_self_contained",
        "session.construct_does_not_clobber_meta",
        "session.open_preserves_meta",
        "session.open_missing_run_raises",
        "session.compacted_chat_persists_every_message",
        "session.conversation_rewritten_record_shape",
        "session.load_run_honours_a_rewrite",
        "session.rewrite_prefix_beyond_messages_raises",
        "session.resume_a_compacted_chat",
        "session.one_shot_after_chat_repersists",
        "session.diagnose_completed",
        "session.diagnose_cancelled",
        "session.diagnose_failed",
        "session.diagnose_crashed",
        "session.turn_states",
        "session.diagnose_does_not_mutate",
        "session.repair_on_resume",
        "session.repair_leaves_transcript_untouched",
        "session.diagnose_last_run_of_many",
        "session.loaded_run_id_is_the_last_run",
        "session.diagnose_a_compacted_transcript",
        "session.search_path_walks_ancestry",
        "session.search_path_creates_nothing",
        "cost.usage_totals_merge",
        "cost.run_accumulates_usage",
        "cost.cancelled_run_reports_usage",
        "cost.price_table_loads_example",
        "cost.price_table_rejects_malformed",
        "cost.unknown_model_costs_nothing_known",
        "cost.leader_usage_per_agent",
        "budget.rejects_invalid_construction",
        "budget.none_is_todays_behaviour",
        "budget.wall_time_stops_before_provider_call",
        "budget.token_and_cost_stops",
        "budget.reason_precedence",
        "budget.stop_is_normal_and_reported",
        "budget.cancellation_wins",
        "budget.stop_answers_every_tool_call",
        "budget.subagents_have_their_own",
        "budget.run_task_forwards",
        "events.final_identity",
        "events.tool_bracketing",
        "events.provider_failure",
        "events.sink_isolation",
        "events.stream_optional",
        "host_protocol.encodes_every_event",
        "host_protocol.round_trip_events",
        "host_protocol.unknown_type_preserved",
        "host_protocol.field_mismatch",
        "host_protocol.registry_is_derived",
        "host_protocol.frame_version",
        "host_protocol.request_validation",
        "host_protocol.document_covers_registry",
        "host_protocol.import_direction",
        "host_server.handshake_line",
        "host_server.auth_required",
        "host_server.event_stream_delivers",
        "host_server.two_subscribers",
        "host_server.slow_subscriber_drops_oldest",
        "host_server.subscriber_disconnect",
        "host_server.prompt_starts_run",
        "host_server.second_prompt_conflicts",
        "host_server.stop_cancels",
        "host_server.bad_request_and_unknown_path",
        "host_server.keepalive",
        "host_server.api_untouched",
        "host_server.runtime_run_id_preserved",
        "host_server.subagent_run_ids_distinct",
        "host_approvals.request_published",
        "host_approvals.allow_resumes",
        "host_approvals.deny_blocks_call",
        "host_approvals.unknown_id_404",
        "host_approvals.timeout_denies",
        "host_approvals.no_subscriber_denies_fast",
        "host_approvals.stop_unparks",
        "host_approvals.callback_never_raises",
        "host_approvals.permissions_untouched",
        "host_approvals.pending_listing",
        "host_approvals.pending_endpoint",
        "host_approvals.round_trip_over_http",
        "host_approvals.survives_a_dropped_frame",
        "host_approvals.no_subscriber_over_http",
        "host_approvals.stop_unparks_over_http",
        "cancel.pre_cancelled_agent",
        "cancel.tool_repair",
        "cancel.http_read_recheck",
        "cancel.late_response_retained",
        "agent_cancel.unanswered_ids_last_assistant_only",
        "agent_cancel.cancelled_messages_unchanged_by_refactor",
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
        "web_fetch.get_only_schema",
        "web_fetch.preapproved_no_prompt",
        "web_fetch.unapproved_auto_denied",
        "web_fetch.unapproved_prompt_asks",
        "web_fetch.scheme_denied",
        "web_fetch.blocked_hosts",
        "web_fetch.redirect_offsite_refused",
        "web_fetch.redirect_onsite_followed",
        "web_fetch.redirect_limit",
        "web_fetch.size_cap_refuses",
        "web_fetch.content_type_refused",
        "web_fetch.html_to_text",
        "web_fetch.domain_table_missing_is_empty",
        "web_fetch.subdomain_not_inherited",
        "web_fetch.plan_mode_allows_fetch",
        "web_fetch.metadata_contract",
        "web_fetch.registry_registration",
        "web_search.absent_without_backend",
        "web_search.registered_with_backend",
        "web_search.key_from_env_not_url",
        "web_search.parses_results",
        "web_search.limit_clamped",
        "web_search.background_call_class",
        "web_search.contacts_only_endpoint",
        "web_search.metadata_contract",
        "web_search.error_carries_no_secret",
        "web_search.cancel_propagates",
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
        == "387 passed, 0 failed, 387 selected of 387 registered",
        f"unexpected full-run summary: {full_run.stdout!r}",
    )

    listed = invoke_check("--list")
    require(listed.returncode == 0, f"list failed: {listed.stderr!r}")
    require(listed.stdout.splitlines() == expected_names, f"unexpected list: {listed.stdout!r}")

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

    for name in expected_names:
        selected_alone = invoke_check("--only", name)
        require(
            selected_alone.returncode == 0,
            f"standalone check {name!r} failed: {selected_alone.stdout!r}",
        )
        selected_alone_lines = selected_alone.stdout.splitlines()
        expected_selected_names = [
            candidate
            for candidate in expected_names
            if name.casefold() in candidate.casefold()
        ]
        require(
            [
                line.removeprefix("PASS  ")
                for line in selected_alone_lines
                if line.startswith("PASS  ")
            ]
            == expected_selected_names,
            f"standalone check selected the wrong name: {selected_alone.stdout!r}",
        )
        selected_count = len(expected_selected_names)
        require(
            selected_alone_lines[-1]
            == (
                f"{selected_count} passed, 0 failed, {selected_count} selected "
                "of 387 registered"
            ),
            f"standalone check selected an unexpected count: {selected_alone.stdout!r}",
        )

    listed_retry = invoke_check("--list", "--only", "retry")
    require(listed_retry.returncode == 0, f"filtered list failed: {listed_retry.stderr!r}")
    require(
        listed_retry.stdout.splitlines()
        == expected_names[17:40] + [expected_names[64]],
        f"unexpected filtered list: {listed_retry.stdout!r}",
    )

    listed_breakers = invoke_check("--list", "--only", "breaker")
    require(
        listed_breakers.returncode == 0,
        f"filtered breaker list failed: {listed_breakers.stderr!r}",
    )
    require(
        listed_breakers.stdout.splitlines() == expected_names[40:47],
        f"unexpected filtered breaker list: {listed_breakers.stdout!r}",
    )

    for selector in ("shell", "SHELL"):
        selected = invoke_check("--only", selector)
        require(selected.returncode == 0, f"selector {selector!r} failed")
        selected_lines = selected.stdout.splitlines()
        require(
            selected_lines[-1] == "11 passed, 0 failed, 11 selected of 387 registered",
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
