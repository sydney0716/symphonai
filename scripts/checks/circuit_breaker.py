"""Registered checks for consecutive-failure circuit breakers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import unittest.mock as mock

from orchestra_api.budgets import RunBudget
from orchestra_api.cancellation import OperationCancelled
from orchestra_api.circuit_breaker import CircuitOpen, ConsecutiveFailureBreaker
from orchestra_api.compaction import CompactionResult, ContextCompactionError
from orchestra_api.leader import DispatchSubagentTool, Leader, LeaderConfig
from orchestra_api.models import Message, ModelResponse, Role, ToolCall, Usage
from orchestra_api.providers.fake import FakeModelProvider
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


def _final_response(content: str = "done") -> ModelResponse:
    return ModelResponse(Message(Role.ASSISTANT, content))


def _unfinished_response(call_id: str = "unfinished") -> ModelResponse:
    return ModelResponse(
        Message(
            Role.ASSISTANT,
            tool_calls=[ToolCall(id=call_id, name="missing")],
        )
    )


def _unchanged_compaction(messages: list[Message], budget: int) -> CompactionResult:
    return CompactionResult(
        messages=list(messages),
        before_tokens=0,
        after_tokens=0,
        budget=budget,
        changed=False,
    )


def _dispatch(name: str, index: int = 0) -> ToolCall:
    return ToolCall(
        id=f"dispatch-{name}-{index}",
        name="dispatch_subagent",
        arguments={"subagent_name": name, "task": f"task {index}"},
    )


@check("breaker.opens_on_consecutive_failures")
def check_opens_on_consecutive_failures() -> None:
    breaker = ConsecutiveFailureBreaker("automatic compaction", max_consecutive_failures=2)
    breaker.raise_if_open()
    if breaker.record_failure() != 1 or breaker.is_open:
        fail("breaker opened before its configured failure threshold")
    if breaker.record_failure() != 2 or not breaker.is_open:
        fail("breaker did not open at its configured failure threshold")
    try:
        breaker.raise_if_open()
    except CircuitOpen as exc:
        if str(exc) != "automatic compaction gave up after 2 consecutive failures":
            fail(f"open breaker error was not descriptive: {exc!r}")
    else:
        fail("raise_if_open did not raise for an open breaker")

    try:
        ConsecutiveFailureBreaker("invalid", max_consecutive_failures=0)
    except ValueError as exc:
        if "max_consecutive_failures" not in str(exc):
            fail(f"invalid threshold error did not name the option: {exc!r}")
    else:
        fail("breaker accepted max_consecutive_failures=0")


@check("breaker.success_resets")
def check_success_resets() -> None:
    breaker = ConsecutiveFailureBreaker("repair", max_consecutive_failures=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    if breaker.consecutive_failures != 0 or breaker.is_open:
        fail("success did not reset the consecutive failure count")
    breaker.record_failure()
    breaker.record_failure()
    if breaker.consecutive_failures != 2 or breaker.is_open:
        fail("non-consecutive failures opened the breaker")


@check("breaker.concurrent_failures_counted")
def check_concurrent_failures_counted() -> None:
    breaker = ConsecutiveFailureBreaker("parallel repair", max_consecutive_failures=10_000)
    failure_count = 2_000
    with ThreadPoolExecutor(max_workers=8) as executor:
        counts = list(executor.map(lambda _: breaker.record_failure(), range(failure_count)))
    if breaker.consecutive_failures != failure_count:
        fail(
            "concurrent breaker updates lost failures: "
            f"expected={failure_count}, actual={breaker.consecutive_failures}"
        )
    if sorted(counts) != list(range(1, failure_count + 1)):
        fail("record_failure did not return every atomic increment")


@check("breaker.leader_stops_automatic_compaction")
def check_leader_stops_automatic_compaction() -> None:
    calls = 0

    def failing_compactor(
        messages: list[Message], *, budget: int, recent_turns: int, cancel: object
    ) -> CompactionResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OperationCancelled
        raise ContextCompactionError("preserved context exceeds budget")

    with workspace() as ws:
        provider = FakeModelProvider([_final_response()])
        leader = Leader(
            LeaderConfig(
                leader_provider=provider,
                subagent_provider=FakeModelProvider(),
                repo_root=str(ws.root),
                max_consecutive_compaction_failures=3,
            )
        )
        with mock.patch(
            "orchestra_api.leader.compact_messages_for_budget",
            side_effect=failing_compactor,
        ):
            first = leader.chat("first")
            if leader._automatic_compaction_breaker.consecutive_failures != 1:
                fail("cancelled compaction was counted as a failure")
            second = leader.chat("second")
            calls_at_open = calls
            third = leader.chat("third")

    if any(result.stopped_reason != "final_response" for result in (first, second, third)):
        fail("automatic compaction failure prevented a provider-backed chat turn")
    if provider.call_count != 3:
        fail(f"chat turns did not continue after compaction failures: {provider.call_count}")
    if calls_at_open != 4 or calls != calls_at_open:
        fail(
            "open breaker did not skip later compactor calls: "
            f"at_open={calls_at_open}, final={calls}"
        )


@check("breaker.manual_compaction_still_runs")
def check_manual_compaction_still_runs() -> None:
    calls = 0
    should_fail = True

    def controlled_compactor(
        messages: list[Message], *, budget: int, recent_turns: int, cancel: object
    ) -> CompactionResult:
        nonlocal calls
        calls += 1
        if should_fail:
            raise ContextCompactionError("preserved context exceeds budget")
        return _unchanged_compaction(messages, budget)

    with workspace() as ws:
        leader = Leader(
            LeaderConfig(
                leader_provider=FakeModelProvider([_final_response()]),
                subagent_provider=FakeModelProvider(),
                repo_root=str(ws.root),
                max_consecutive_compaction_failures=1,
            )
        )
        with mock.patch(
            "orchestra_api.leader.compact_messages_for_budget",
            side_effect=controlled_compactor,
        ):
            leader.chat("open the breaker")
            if calls != 1 or not leader._automatic_compaction_breaker.is_open:
                fail("test setup did not open automatic compaction breaker")

            try:
                leader.compact_chat()
            except ContextCompactionError:
                pass
            else:
                fail("manual compaction failure did not propagate")
            if (
                calls != 2
                or leader._automatic_compaction_breaker.consecutive_failures != 1
            ):
                fail("manual compaction failure changed the automatic breaker")

            should_fail = False
            leader.compact_chat()
            if calls != 3 or leader._automatic_compaction_breaker.is_open:
                fail("manual compaction did not run and close the open breaker")
            leader.chat("automatic compaction resumes")
            if calls != 5:
                fail("automatic compaction did not resume after manual success")

            should_fail = True
            leader.chat("open it again")
            calls_before_clear = calls
            leader.clear_chat()
            should_fail = False
            leader.chat("automatic compaction resumes after clear")
            if calls != calls_before_clear + 2:
                fail("clear_chat did not reopen automatic compaction")


@check("breaker.subagent_refused_after_repeated_failure")
def check_subagent_refused_after_repeated_failure() -> None:
    with workspace() as ws:
        provider = FakeModelProvider([_unfinished_response()])
        tool = DispatchSubagentTool(
            provider,
            ws.policy,
            subagent_max_turns=1,
            max_consecutive_subagent_failures=3,
        )
        failures = [tool.execute(_dispatch("researcher", index), ws.policy) for index in range(3)]
        if any(result.ok or result.error != "subagent reached max_turns without a final answer" for result in failures):
            fail(f"pre-threshold dispatch errors changed: {failures!r}")
        turns_at_open = tool.pool["researcher"].turns_used
        refused = tool.execute(_dispatch("researcher", 3), ws.policy)
        if (
            refused.ok
            or refused.error
            != "subagent 'researcher' failed 3 times in a row; not dispatching again this run"
            or tool.pool["researcher"].turns_used != turns_at_open
            or provider.call_count != 3
        ):
            fail(f"open subagent breaker did not refuse without running: {refused!r}")

        other = tool.execute(_dispatch("reviewer"), ws.policy)
        if other.error == refused.error or tool.pool["reviewer"].turns_used != 1:
            fail(f"one subagent breaker affected another subagent: {other!r}")

        reset_tool = DispatchSubagentTool(
            FakeModelProvider(
                [
                    _unfinished_response("reset-fail-1"),
                    _final_response("recovered"),
                    _unfinished_response("reset-fail-2"),
                ]
            ),
            ws.policy,
            subagent_max_turns=1,
            max_consecutive_subagent_failures=2,
        )
        reset_results = [
            reset_tool.execute(_dispatch("recovering", index), ws.policy)
            for index in range(3)
        ]
        reset_breaker = reset_tool.pool["recovering"].breaker
        if [result.ok for result in reset_results] != [False, True, False]:
            fail(f"scripted subagent recovery did not occur: {reset_results!r}")
        if reset_breaker.consecutive_failures != 1 or reset_breaker.is_open:
            fail("successful subagent dispatch did not reset its breaker")

        budget_tool = DispatchSubagentTool(
            FakeModelProvider(
                [
                    ModelResponse(
                        _unfinished_response("budget-stop").message,
                        usage=Usage(input_tokens=2),
                    )
                ]
            ),
            ws.policy,
            subagent_max_turns=2,
            subagent_budget=RunBudget(max_total_tokens=1),
        )
        budget_result = budget_tool.execute(_dispatch("budgeted"), ws.policy)
        if (
            budget_result.ok
            or "budget_tokens" not in (budget_result.error or "")
            or "max_turns" in (budget_result.error or "")
        ):
            fail(f"budget stop was mislabeled as max_turns: {budget_result!r}")


@check("breaker.stopped_repairs_reported")
def check_stopped_repairs_reported() -> None:
    with workspace() as ws:
        healthy = Leader(
            LeaderConfig(
                leader_provider=FakeModelProvider([_final_response()]),
                subagent_provider=FakeModelProvider(),
                repo_root=str(ws.root),
            )
        ).run("healthy")
        if healthy.stopped_repairs != ():
            fail(f"healthy run reported stopped repairs: {healthy.stopped_repairs!r}")

        leader_provider = FakeModelProvider(
            [
                ModelResponse(
                    Message(
                        Role.ASSISTANT,
                        tool_calls=[_dispatch("zeta"), _dispatch("alpha")],
                    )
                ),
                _final_response("dispatches handled"),
                _final_response("chat handled"),
            ]
        )
        leader = Leader(
            LeaderConfig(
                leader_provider=leader_provider,
                subagent_provider=FakeModelProvider([_unfinished_response()]),
                repo_root=str(ws.root),
                subagent_max_turns=1,
                max_consecutive_compaction_failures=1,
                max_consecutive_subagent_failures=1,
            )
        )
        dispatch_result = leader.run("open subagent breakers")
        if dispatch_result.stopped_repairs != ("subagent alpha", "subagent zeta"):
            fail(
                "run did not report sorted open subagent breakers: "
                f"{dispatch_result.stopped_repairs!r}"
            )

        compaction_calls = 0

        def fail_post_run(
            messages: list[Message], *, budget: int, recent_turns: int, cancel: object
        ) -> CompactionResult:
            nonlocal compaction_calls
            compaction_calls += 1
            if compaction_calls == 2:
                raise ContextCompactionError("post-run failure")
            return _unchanged_compaction(messages, budget)

        with mock.patch(
            "orchestra_api.leader.compact_messages_for_budget",
            side_effect=fail_post_run,
        ):
            chat_result = leader.chat("open compaction after the model run")
        expected = ("automatic compaction", "subagent alpha", "subagent zeta")
        if chat_result.stopped_repairs != expected:
            fail(
                "chat did not refresh stopped repairs after post-run compaction: "
                f"{chat_result.stopped_repairs!r}"
            )
