"""Registered checks for runtime usage aggregation and model pricing."""

from __future__ import annotations

import json
import tempfile
from dataclasses import fields
from decimal import Decimal
from pathlib import Path

from symphonai_api.agent_loop import ApiAgent
from symphonai_api.cancellation import CancellationToken
from symphonai_api.cost import (
    ModelPrice,
    PriceTable,
    UsageTotals,
    load_price_table,
    total_cost,
)
from symphonai_api.events import Event
from symphonai_api.leader import Leader, LeaderConfig
from symphonai_api.models import Message, ModelRequest, ModelResponse, Role, ToolCall, Usage
from symphonai_api.providers.fake import FakeModelProvider
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


REPO_ROOT = Path(__file__).resolve().parents[2]


def _response(
    content: str = "",
    *,
    input_tokens: int,
    output_tokens: int,
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


@check("cost.usage_totals_merge")
def check_usage_totals_merge() -> None:
    left = UsageTotals(input_tokens=10, output_tokens=20, calls=1)
    right = UsageTotals(input_tokens=3, output_tokens=4, calls=2)
    expected = UsageTotals(input_tokens=13, output_tokens=24, calls=3)
    if left.merged(right) != expected or right.merged(left) != expected:
        fail("usage merge is not commutative across all counters")
    if left != UsageTotals(10, 20, 1) or right != UsageTotals(3, 4, 2):
        fail("usage merge mutated an operand")
    if UsageTotals.from_usage(Usage(7, 9)) != UsageTotals(7, 9, 1):
        fail("from_usage did not count exactly one provider call")
    if expected.total_tokens != 37:
        fail(f"total_tokens ignored a token counter: {expected!r}")

    expected_event_fields = {
        "RunStarted": {"agent_id", "run_id", "turn_id", "schema_version", "agent_name"},
        "RunFinished": {
            "agent_id", "run_id", "turn_id", "schema_version", "agent_name", "stopped_reason",
        },
        "RunFailed": {"agent_id", "run_id", "turn_id", "schema_version", "agent_name", "error"},
        "TurnStarted": {"agent_id", "run_id", "turn_id", "schema_version", "index"},
        "TurnFinished": {"agent_id", "run_id", "turn_id", "schema_version", "index"},
        "AssistantTextDelta": {
            "agent_id", "run_id", "turn_id", "schema_version", "text",
        },
        "ToolCallStarted": {
            "agent_id", "run_id", "turn_id", "schema_version", "tool_name", "tool_call_id",
        },
        "ToolCallFinished": {
            "agent_id", "run_id", "turn_id", "schema_version", "tool_name", "tool_call_id", "ok",
        },
        "SubagentSpawned": {
            "agent_id", "run_id", "turn_id", "schema_version", "subagent_name", "subagent_agent_id",
        },
        "CompactionApplied": {
            "agent_id", "run_id", "turn_id", "schema_version", "before_tokens", "after_tokens",
            "dropped_messages",
        },
    }
    actual_event_fields = {
        event_type.__name__: {item.name for item in fields(event_type)}
        for event_type in Event.__subclasses__()
    }
    if actual_event_fields != expected_event_fields:
        fail(f"event schema changed during usage tracking: {actual_event_fields!r}")


@check("cost.run_accumulates_usage")
def check_run_accumulates_usage() -> None:
    with workspace() as ws:
        responses = [
            _response(
                input_tokens=10 * turn,
                output_tokens=turn,
                tool_call=ToolCall(id=f"call-{turn}", name="missing"),
            )
            for turn in range(1, 4)
        ]
        result = ApiAgent(
            FakeModelProvider(responses), {}, ws.policy, max_turns=3
        ).run(
            [Message(role=Role.USER, content="use all three turns")],
            model="requested-model",
        )
    expected = {"requested-model": UsageTotals(60, 6, 3)}
    if result.stopped_reason != "max_turns" or result.turns_used != 3:
        fail(f"scripted run did not reach max_turns: {result!r}")
    if result.usage_by_model != expected:
        fail(f"run usage was not accumulated by requested model: {result.usage_by_model!r}")


@check("cost.cancelled_run_reports_usage")
def check_cancelled_run_reports_usage() -> None:
    with workspace() as ws:
        token = CancellationToken()

        class _LateCancellingFakeProvider(FakeModelProvider):
            def create_response(
                self,
                request: ModelRequest,
                *,
                cancel: CancellationToken | None = None,
            ) -> ModelResponse:
                response = super().create_response(request, cancel=cancel)
                assert cancel is not None
                cancel.cancel()
                return response

        provider = _LateCancellingFakeProvider(
            [_response("late", input_tokens=12, output_tokens=5)]
        )
        provider.model = "cancel-model"
        result = ApiAgent(provider, {}, ws.policy).run(
            [Message(role=Role.USER, content="cancel after billing")], cancel=token
        )
    if result.stopped_reason != "cancelled":
        fail(f"late-cancelled run did not report cancellation: {result!r}")
    if result.usage_by_model != {"cancel-model": UsageTotals(12, 5, 1)}:
        fail(f"cancelled run discarded billed usage: {result.usage_by_model!r}")


@check("cost.price_table_loads_example")
def check_price_table_loads_example() -> None:
    table = load_price_table(REPO_ROOT / "configs/model-prices.example.json")
    if table.currency != "USD" or set(table.prices) != {"example-model-name"}:
        fail(f"example price table loaded with the wrong shape: {table!r}")
    price = table.prices["example-model-name"]
    if price.input_per_million != Decimal("0.00") or price.output_per_million != Decimal("0.00"):
        fail(f"example rates did not load exactly: {price!r}")
    if not isinstance(price.input_per_million, Decimal) or not isinstance(
        price.output_per_million, Decimal
    ):
        fail(f"example rates are not Decimal values: {price!r}")


@check("cost.price_table_rejects_malformed")
def check_price_table_rejects_malformed() -> None:
    def assert_rejected(path: Path, expected: str) -> None:
        try:
            load_price_table(path)
        except ValueError as exc:
            if expected not in str(exc):
                fail(f"malformed price table error did not name {expected!r}: {exc!r}")
        else:
            fail(f"malformed price table was accepted; expected {expected!r}")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assert_rejected(root / "missing.json", "could not be read")

        cases: list[tuple[object, str]] = [
            ("not-json", "not valid JSON"),
            ("not-utf8", "not valid JSON"),
            ([], "JSON object"),
            ({"currency": "USD", "models": {}}, "version"),
            ({"version": 2, "currency": "USD", "models": {}}, "version"),
            ({"version": True, "currency": "USD", "models": {}}, "version"),
            ({"version": 1, "models": {}}, "currency"),
            ({"version": 1, "currency": "USD"}, "models"),
            (
                {"version": 1, "currency": "USD", "models": {"m": []}},
                "entry",
            ),
            (
                {
                    "version": 1,
                    "currency": "USD",
                    "models": {"m": {"input": "bad", "output": "1"}},
                },
                "non-negative number",
            ),
            (
                {
                    "version": 1,
                    "currency": "USD",
                    "models": {"m": {"input": "1", "output": -1}},
                },
                "non-negative number",
            ),
        ]
        for index, (data, expected) in enumerate(cases):
            path = root / f"case-{index}.json"
            if data == "not-json":
                path.write_text("{", encoding="utf-8")
            elif data == "not-utf8":
                path.write_bytes(b"\xff")
            else:
                path.write_text(json.dumps(data), encoding="utf-8")
            assert_rejected(path, expected)


@check("cost.unknown_model_costs_nothing_known")
def check_unknown_model_costs_nothing_known() -> None:
    table = PriceTable(
        prices={
            "known": ModelPrice(Decimal("0.10"), Decimal("0.20")),
            "other": ModelPrice(Decimal("0.30"), Decimal("0.40")),
        },
        currency="USD",
    )
    known_usage = UsageTotals(2_000_000, 3_000_000, 7)
    if table.cost("known", known_usage) != Decimal("0.80"):
        fail(f"known model cost lost Decimal precision: {table.cost('known', known_usage)!r}")
    if table.cost("unknown", known_usage) is not None:
        fail("unknown model was presented as having a known zero cost")
    if total_cost({"known": known_usage, "unknown": UsageTotals(1, 1, 1)}, table) is not None:
        fail("total_cost presented a partial known-model sum as a total")
    expected_total = Decimal("1.50")
    actual_total = total_cost(
        {"known": known_usage, "other": UsageTotals(1_000_000, 1_000_000, 1)},
        table,
    )
    if actual_total != expected_total or not isinstance(actual_total, Decimal):
        fail(f"fully priced total was not an exact Decimal sum: {actual_total!r}")
    if total_cost({"known": known_usage}, None) is not None:
        fail("a missing price table was presented as a zero cost")


@check("cost.leader_usage_per_agent")
def check_leader_usage_per_agent() -> None:
    with workspace() as ws:
        leader_provider = FakeModelProvider(
            [
                _response(
                    input_tokens=10,
                    output_tokens=1,
                    tool_call=ToolCall(
                        id="dispatch-1",
                        name="dispatch_subagent",
                        arguments={"subagent_name": "researcher", "task": "first"},
                    ),
                ),
                _response(
                    input_tokens=20,
                    output_tokens=2,
                    tool_call=ToolCall(
                        id="dispatch-2",
                        name="dispatch_subagent",
                        arguments={"subagent_name": "researcher", "task": "second"},
                    ),
                ),
                _response("done", input_tokens=30, output_tokens=3),
            ]
        )
        leader_provider.model = "leader-model"
        subagent_provider = FakeModelProvider(
            [
                _response("first result", input_tokens=4, output_tokens=5),
                _response("second result", input_tokens=6, output_tokens=7),
            ]
        )
        subagent_provider.model = "subagent-model"
        result = Leader(
            LeaderConfig(
                leader_provider=leader_provider,
                subagent_provider=subagent_provider,
                repo_root=str(ws.root),
            )
        ).run("delegate twice")

    if len(result.usage_by_agent) != 2:
        fail(f"leader usage does not contain exactly both agents: {result.usage_by_agent!r}")
    leader_usage = result.usage_by_agent.get(result.agent.agent_id)
    if leader_usage != {"leader-model": UsageTotals(60, 6, 3)}:
        fail(f"leader's own usage is wrong: {leader_usage!r}")
    record = result.subagents["researcher"]
    child_usage = result.usage_by_agent.get(record.agent_ref.agent_id)
    if child_usage != {"subagent-model": UsageTotals(10, 12, 2)}:
        fail(f"reused subagent usage was not merged: {child_usage!r}")
    if record.usage_by_model != child_usage:
        fail(f"subagent record and leader result disagree: {record.usage_by_model!r}")
