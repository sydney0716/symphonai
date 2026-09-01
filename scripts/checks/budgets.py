"""Registered checks for per-run turn, wall-time, token, and cost budgets."""

from __future__ import annotations

import unittest.mock as mock
from decimal import Decimal

from symphonai_api.agent_loop import ApiAgent
from symphonai_api.budgets import BudgetState, RunBudget
from symphonai_api.cancellation import CancellationToken
from symphonai_api.cost import ModelPrice, PriceTable, UsageTotals
from symphonai_api.events import CollectingSink, RunFailed, RunFinished
from symphonai_api.identity import AgentRef, RunRef, TurnRef
from symphonai_api.leader import Leader, LeaderConfig
from symphonai_api.models import Message, ModelResponse, Role, ToolCall, Usage
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.runner import run_task
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


def _response(
    content: str = "",
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    tool_call: ToolCall | None = None,
) -> ModelResponse:
    return ModelResponse(
        message=Message(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=[] if tool_call is None else [tool_call],
        ),
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _price_table() -> PriceTable:
    return PriceTable(
        prices={"priced-model": ModelPrice(Decimal("100"), Decimal("200"))},
        currency="USD",
    )


@check("budget.rejects_invalid_construction")
def check_rejects_invalid_construction() -> None:
    cases = [
        ({"max_turns": 0}, "max_turns"),
        ({"wall_seconds": 0}, "wall_seconds"),
        ({"max_total_tokens": 0}, "max_total_tokens"),
        ({"max_cost": Decimal("-0.01"), "price_table": _price_table()}, "max_cost"),
        ({"max_cost": Decimal("1")}, "max_cost"),
    ]
    for arguments, field_name in cases:
        try:
            RunBudget(**arguments)
        except ValueError as exc:
            if field_name not in str(exc):
                fail(f"invalid {field_name} error did not name its field: {exc!r}")
        else:
            fail(f"invalid {field_name} budget was accepted: {arguments!r}")


@check("budget.none_is_todays_behaviour")
def check_none_is_todays_behaviour() -> None:
    responses = [
        _response(tool_call=ToolCall(id="unknown", name="missing")),
        _response("done", input_tokens=2, output_tokens=3),
    ]
    agent_ref = AgentRef(agent_id="agent_fixed", name="agent")
    run_ref = RunRef(run_id="run_fixed", agent_id=agent_ref.agent_id)

    def run(*, explicit_none: bool):
        with workspace() as ws:
            arguments = {"budget": None} if explicit_none else {}
            provider = FakeModelProvider(responses)
            agent = ApiAgent(provider, {}, ws.policy, agent_ref=agent_ref, **arguments)
            with mock.patch("symphonai_api.agent_loop.new_run_ref", return_value=run_ref):
                with mock.patch(
                    "symphonai_api.agent_loop.new_turn_ref",
                    side_effect=lambda run_id, index: TurnRef(
                        turn_id=f"turn_{index}", run_id=run_id, index=index
                    ),
                ):
                    return agent.run([Message(role=Role.USER, content="same")])

    omitted = run(explicit_none=False)
    explicit = run(explicit_none=True)
    if explicit != omitted:
        fail(f"budget=None changed the full AgentRunResult: {omitted!r} != {explicit!r}")


@check("budget.wall_time_stops_before_provider_call")
def check_wall_time_stops_before_provider_call() -> None:
    with workspace() as ws:
        provider = FakeModelProvider(
            [
                _response(tool_call=ToolCall(id="first", name="missing")),
                _response("provider must not reach this"),
            ]
        )
        agent = ApiAgent(
            provider,
            {},
            ws.policy,
            budget=RunBudget(max_turns=3, wall_seconds=1),
        )
        with mock.patch("symphonai_api.agent_loop.time.monotonic", return_value=0.0):
            with mock.patch(
                "symphonai_api.budgets.time.monotonic",
                side_effect=[0.0, 0.0, 2.0],
            ):
                result = agent.run([Message(role=Role.USER, content="work")])

    if result.stopped_reason != "budget_wall_time":
        fail(f"wall budget returned the wrong reason: {result!r}")
    if provider.call_count != 1 or result.turns_used != 1:
        fail(f"wall budget did not prevent provider call two: {provider.call_count!r}")


@check("budget.token_and_cost_stops")
def check_token_and_cost_stops() -> None:
    with workspace() as ws:
        token_result = ApiAgent(
            FakeModelProvider(
                [
                    _response(
                        input_tokens=7,
                        output_tokens=4,
                        tool_call=ToolCall(id="tokens", name="missing"),
                    )
                ]
            ),
            {},
            ws.policy,
            budget=RunBudget(max_total_tokens=10),
        ).run([Message(role=Role.USER, content="tokens")], model="token-model")
        if token_result.stopped_reason != "budget_tokens":
            fail(f"token budget did not stop the run: {token_result!r}")
        if token_result.usage_by_model != {"token-model": UsageTotals(7, 4, 1)}:
            fail(f"token budget discarded usage: {token_result.usage_by_model!r}")

        cost_provider = FakeModelProvider(
            [
                _response(
                    input_tokens=20,
                    tool_call=ToolCall(id="cost", name="missing"),
                )
            ]
        )
        cost_provider.model = "priced-model"
        cost_result = ApiAgent(
            cost_provider,
            {},
            ws.policy,
            budget=RunBudget(max_cost=Decimal("0.001"), price_table=_price_table()),
        ).run([Message(role=Role.USER, content="cost")])
        if cost_result.stopped_reason != "budget_cost":
            fail(f"priced cost budget did not stop the run: {cost_result!r}")

        unpriced_provider = FakeModelProvider(
            [
                _response(
                    input_tokens=10_000_000,
                    output_tokens=10_000_000,
                    tool_call=ToolCall(id="unpriced", name="missing"),
                ),
                _response("finished despite unknown price"),
            ]
        )
        unpriced_provider.model = "unpriced-model"
        unpriced_result = ApiAgent(
            unpriced_provider,
            {},
            ws.policy,
            budget=RunBudget(max_cost=Decimal("0.000001"), price_table=_price_table()),
        ).run([Message(role=Role.USER, content="unknown cost")])
        if unpriced_result.stopped_reason != "final_response":
            fail(f"unpriced model incorrectly tripped the cost budget: {unpriced_result!r}")
        if unpriced_provider.call_count != 2:
            fail("unpriced model did not continue to its final response")
        unknown_cost_state = BudgetState(
            started_monotonic=0.0,
            usage_by_model={"unpriced-model": UsageTotals(10_000_000, 10_000_000, 1)},
        )
        if unknown_cost_state.exceeded(
            RunBudget(max_cost=Decimal(0), price_table=_price_table())
        ) is not None:
            fail("unknown cost was treated as a known zero cost")


@check("budget.reason_precedence")
def check_reason_precedence() -> None:
    usage = {"priced-model": UsageTotals(100, 100, 1)}
    state = BudgetState(started_monotonic=0.0, usage_by_model=usage)
    all_limits = RunBudget(
        wall_seconds=1,
        max_total_tokens=1,
        max_cost=Decimal(0),
        price_table=_price_table(),
    )
    with mock.patch("symphonai_api.budgets.time.monotonic", return_value=2.0):
        if state.exceeded(all_limits) != "budget_wall_time":
            fail("wall time did not take precedence over token and cost ceilings")

    token_and_cost = RunBudget(
        wall_seconds=10,
        max_total_tokens=1,
        max_cost=Decimal(0),
        price_table=_price_table(),
    )
    with mock.patch("symphonai_api.budgets.time.monotonic", return_value=0.0):
        if state.exceeded(token_and_cost) != "budget_tokens":
            fail("tokens did not take precedence over the cost ceiling")


@check("budget.stop_is_normal_and_reported")
def check_stop_is_normal_and_reported() -> None:
    with workspace() as ws:
        events = CollectingSink()
        result = ApiAgent(
            FakeModelProvider(
                [
                    _response(
                        input_tokens=2,
                        tool_call=ToolCall(id="stop", name="missing"),
                    )
                ]
            ),
            {},
            ws.policy,
            events=events,
            budget=RunBudget(max_total_tokens=1),
        ).run([Message(role=Role.USER, content="stop normally")])
        if result.stopped_reason != "budget_tokens" or len(result.messages) != 3:
            fail(f"budget stop did not return its produced conversation: {result!r}")
        finished = events.of_type(RunFinished)
        if len(finished) != 1 or finished[0].stopped_reason != result.stopped_reason:
            fail(f"budget stop was not reported by RunFinished: {events.events!r}")
        if events.of_type(RunFailed):
            fail(f"normal budget stop emitted RunFailed: {events.events!r}")

        max_turns_result = ApiAgent(
            FakeModelProvider(
                [_response(tool_call=ToolCall(id="limit", name="missing"))]
            ),
            {},
            ws.policy,
            max_turns=5,
            budget=RunBudget(max_turns=1),
        ).run([Message(role=Role.USER, content="one turn")])
        if max_turns_result.stopped_reason != "max_turns" or max_turns_result.turns_used != 1:
            fail(f"budget max_turns did not supersede the constructor: {max_turns_result!r}")


@check("budget.cancellation_wins")
def check_cancellation_wins() -> None:
    """Cancellation outranks a budget at the top of a turn.

    The ordering is only observable on the first turn, and only with a wall
    budget. A token or cost budget cannot reach the top-of-turn check at all:
    usage changes only at a provider call, so the previous turn's post-batch
    check sees the same breach and fires first. Written any other way this
    check passes with the two checks swapped, which is how it was first
    written.
    """
    token = CancellationToken()
    token.cancel()
    clock = iter([0.0])
    with workspace() as ws, mock.patch(
        "time.monotonic", lambda: next(clock, 10_000.0)
    ):
        result = ApiAgent(
            FakeModelProvider([_response("never reached", input_tokens=10)]),
            {},
            ws.policy,
            budget=RunBudget(wall_seconds=1.0),
        ).run([Message(role=Role.USER, content="both are live")], cancel=token)

    if result.stopped_reason != "cancelled":
        fail(f"budget reason overrode explicit cancellation: {result.stopped_reason!r}")
    if result.usage_by_model:
        fail(f"cancelled before the provider call, yet usage was recorded: {result.usage_by_model!r}")


@check("budget.stop_answers_every_tool_call")
def check_stop_answers_every_tool_call() -> None:
    """A budget stop never returns an unanswered tool call.

    The stop is checked once every batch of a turn has been answered. Checking
    between batches would return a conversation carrying a tool call with no
    result, which every provider rejects — and unlike the cancellation path
    there is no repair here to fix it up.
    """
    calls = [ToolCall(id="a", name="absent-one"), ToolCall(id="b", name="absent-two")]
    response = ModelResponse(
        message=Message(role=Role.ASSISTANT, content="two calls", tool_calls=calls),
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    clock = iter([0.0, 0.0])
    with workspace() as ws, mock.patch(
        "time.monotonic", lambda: next(clock, 10_000.0)
    ):
        result = ApiAgent(
            FakeModelProvider([response]),
            {},
            ws.policy,
            budget=RunBudget(wall_seconds=1.0),
        ).run([Message(role=Role.USER, content="go")])

    if result.stopped_reason != "budget_wall_time":
        fail(f"scripted clock did not trip the wall budget: {result.stopped_reason!r}")
    answered = {
        message.tool_result.tool_call_id
        for message in result.messages
        if message.tool_result is not None
    }
    asked = {call.id for message in result.messages for call in message.tool_calls}
    if asked - answered:
        fail(f"budget stop left tool calls unanswered: {sorted(asked - answered)}")


@check("budget.subagents_have_their_own")
def check_subagents_have_their_own() -> None:
    with workspace() as ws:
        leader_provider = FakeModelProvider(
            [
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="dispatch-a",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "a", "task": "exhaust"},
                            ),
                            ToolCall(
                                id="dispatch-b",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "b", "task": "finish"},
                            ),
                        ],
                    )
                ),
                _response("leader finished"),
            ]
        )
        subagent_provider = FakeModelProvider(
            [
                _response(
                    input_tokens=11,
                    tool_call=ToolCall(id="exhaust", name="missing"),
                ),
                _response("second agent finished", input_tokens=1),
            ]
        )
        subagent_provider.model = "subagent-model"
        result = Leader(
            LeaderConfig(
                leader_provider=leader_provider,
                subagent_provider=subagent_provider,
                repo_root=str(ws.root),
                subagent_budget=RunBudget(max_total_tokens=10),
            )
        ).run("dispatch both")

    if result.stopped_reason != "final_response" or subagent_provider.call_count != 2:
        fail(f"one exhausted subagent stopped later dispatches: {result!r}")
    if set(result.subagents) != {"a", "b"}:
        fail(f"both independently budgeted subagents were not created: {result.subagents!r}")
    usage_a = result.subagents["a"].usage_by_model
    usage_b = result.subagents["b"].usage_by_model
    if usage_a != {"subagent-model": UsageTotals(11, 0, 1)}:
        fail(f"first subagent's usage is wrong: {usage_a!r}")
    if usage_b != {"subagent-model": UsageTotals(1, 0, 1)}:
        fail(f"second subagent inherited the first one's drawdown: {usage_b!r}")


@check("budget.run_task_forwards")
def check_run_task_forwards() -> None:
    responses = [
        _response(
            input_tokens=2,
            tool_call=ToolCall(id="forward", name="missing"),
        ),
        _response("done"),
    ]
    with workspace() as ws:
        budget_result = run_task(
            FakeModelProvider(responses),
            ws.policy,
            "budgeted",
            budget=RunBudget(max_total_tokens=1),
        )
        ordinary_result = run_task(
            FakeModelProvider(responses),
            ws.policy,
            "ordinary",
        )
    if budget_result.stopped_reason != "budget_tokens":
        fail(f"run_task did not forward its budget: {budget_result!r}")
    if ordinary_result.stopped_reason != "final_response":
        fail(f"run_task without a budget changed behavior: {ordinary_result!r}")
