"""Registered checks for runtime usage aggregation and model pricing."""

from __future__ import annotations

import json
import tempfile
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
from symphonai_api.leader import Leader, LeaderConfig
from symphonai_api.models import Message, ModelRequest, ModelResponse, Role, ToolCall, Usage
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.permissions import PermissionPolicy
from scripts.checks.harness import check, fail
from scripts.checks.workspace import handoff_spec
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


@check("cost.cache_write_rates")
def check_cache_write_rates() -> None:
    totals = UsageTotals.from_usage(Usage(10000, 0, 6000, 3000))
    price = ModelPrice(
        Decimal("0.10"), Decimal("0"), Decimal("0.01"), Decimal("0.125")
    )
    actual = PriceTable({"gpt-6-luna": price}, "USD").cost("gpt-6-luna", totals)
    expected = (Decimal(1000) * Decimal("0.10") + Decimal(6000) * Decimal("0.01")
                + Decimal(3000) * Decimal("0.125")) / Decimal(1_000_000)
    if actual != expected:
        fail(f"cache writes were not priced at their own rate: {actual!r}")


@check("cost.cache_write_1h_rates")
def check_cache_write_1h_rates() -> None:
    totals = UsageTotals(
        input_tokens=10_000,
        calls=1,
        cache_write_tokens=10_000,
        cache_write_1h_tokens=8_000,
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        priced_path = root / "priced.json"
        priced_path.write_text(json.dumps({"version": 1, "currency": "USD", "models": {
            "claude-opus": {"input": 4, "output": 20, "cache_write": 5, "cache_write_1h": 8},
        }}), encoding="utf-8")
        fallback_path = root / "fallback.json"
        fallback_path.write_text(json.dumps({"version": 1, "currency": "USD", "models": {
            "claude-opus": {"input": 4, "output": 20, "cache_write": 5},
        }}), encoding="utf-8")
        priced = load_price_table(priced_path)
        fallback = load_price_table(fallback_path)
    actual = priced.cost("claude-opus", totals)
    expected = (Decimal(8_000) * Decimal(8) + Decimal(2_000) * Decimal(5)) / Decimal(1_000_000)
    if actual != expected or priced.cache_write_1h_is_estimated("claude-opus", totals):
        fail(f"one-hour and five-minute writes were not priced separately: {actual!r}")
    if fallback.cost("claude-opus", totals) != Decimal("0.05"):
        fail("missing one-hour price did not fall back to the cache-write rate")
    if not fallback.cache_write_1h_is_estimated("claude-opus", totals):
        fail("fallback one-hour write cost was not marked estimated")


@check("cost.long_context_tier")
def check_long_context_tier() -> None:
    usage = UsageTotals(400000, 20, 2, long_context_input_tokens=300000,
                        long_context_output_tokens=10)
    base = ModelPrice(Decimal("1"), Decimal("2"))
    long = ModelPrice(Decimal("3"), Decimal("4"))
    table = PriceTable({"gpt-6-luna": ModelPrice(
        base.input_per_million, base.output_per_million, long_context=long
    )}, "USD")
    expected = (Decimal(100000) + Decimal(300000) * 3 + Decimal(10) * 2
                + Decimal(10) * 4) / Decimal(1_000_000)
    if table.cost("gpt-6-luna", usage) != expected:
        fail("long-context usage was not split and priced across both tiers")


@check("cost.long_context_missing_tier_is_estimated")
def check_long_context_missing_tier_is_estimated() -> None:
    usage = UsageTotals(300000, 10, 1, long_context_input_tokens=300000,
                        long_context_output_tokens=10)
    table = PriceTable({"gpt-6-luna": ModelPrice(Decimal("1"), Decimal("2"))}, "USD")
    if table.cost("gpt-6-luna", usage) != Decimal("0.30002"):
        fail("missing long-context tier did not fall back to base rates")
    if not table.long_context_is_estimated("gpt-6-luna", usage):
        fail("base-rate long-context cost was not marked as estimated")



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
    if price.cache_read_per_million != Decimal("0.00") or price.cache_write_per_million != Decimal("0.00"):
        fail(f"example cache rates did not load exactly: {price!r}")
    if price.cache_write_1h_per_million != Decimal("0.00"):
        fail(f"example one-hour cache-write rate did not load exactly: {price!r}")
    if not isinstance(price.input_per_million, Decimal) or not isinstance(
        price.output_per_million, Decimal
    ) or not isinstance(price.cache_read_per_million, Decimal) or not isinstance(
        price.cache_write_per_million, Decimal
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


@check("cost.cache_rates_adjust_cost")
def check_cache_rates_adjust_cost() -> None:
    totals = UsageTotals(
        input_tokens=1050,
        output_tokens=20,
        calls=1,
        cache_read_tokens=900,
        cache_write_tokens=50,
    )
    discounted = PriceTable(
        prices={
            "model": ModelPrice(
                Decimal("4"),
                Decimal("20"),
                Decimal("0.40"),
                Decimal("5.00"),
            )
        },
        currency="USD",
    )
    if discounted.cost("model", totals) != Decimal("0.00141"):
        fail(f"cached input was not priced at cache rates: {discounted.cost('model', totals)!r}")

    regular = PriceTable(
        prices={"model": ModelPrice(Decimal("4"), Decimal("20"))},
        currency="USD",
    )
    if regular.cost("model", totals) != Decimal("0.0046"):
        fail(f"missing cache rates changed legacy cost: {regular.cost('model', totals)!r}")


@check("cost.cache_rate_validation")
def check_cache_rate_validation() -> None:
    cases = {
        "cache_read": "bad",
        "cache_write": -1,
        "cache_read_bool": True,
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cache-rates.json"
        for field, value in cases.items():
            price_field = "cache_read" if field == "cache_read_bool" else field
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "currency": "USD",
                        "models": {
                            "model": {
                                "input": "4",
                                "output": "20",
                                price_field: value,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            try:
                load_price_table(path)
            except ValueError as exc:
                if "non-negative number" not in str(exc):
                    fail(f"invalid cache rate {field!r} produced the wrong error: {exc!r}")
            else:
                fail(f"invalid cache rate {field!r} was accepted")


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
                        arguments={"subagent_name": "researcher", "task": handoff_spec("first")},
                    ),
                ),
                _response(
                    input_tokens=20,
                    output_tokens=2,
                    tool_call=ToolCall(
                        id="dispatch-2",
                        name="dispatch_subagent",
                        arguments={"subagent_name": "researcher", "task": handoff_spec("second")},
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
                leader_policy=PermissionPolicy(
                    ws.root, shell_enabled=True, shell_allowlist=[()],
                ),
                subagent_tool_names=("read_file",),
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


@check("cost.anthropic_answering_model")
def check_anthropic_answering_model() -> None:
    from symphonai_api.events import CollectingSink, ModelFallback
    from symphonai_api.providers.anthropic_provider import AnthropicProvider
    from symphonai_api.session import TranscriptWriter, read_records
    from unittest import mock
    import io
    import os
    for streamed in (False, True):
        with workspace() as ws:
            payload = {"model": "claude-opus-4-8", "content": [{"type": "text", "text": "done"}],
                       "usage": {"input_tokens": 10, "output_tokens": 2}}
            def respond(request, timeout=None):
                if streamed:
                    return io.BytesIO((
                        "event: message_start\ndata: " + json.dumps({"message": payload}) + "\n\n"
                        + 'event: content_block_start\ndata: {"index":0,"content_block":{"type":"text","text":""}}\n\n'
                        + 'event: content_block_delta\ndata: {"index":0,"delta":{"type":"text_delta","text":"done"}}\n\n'
                        + 'event: message_delta\ndata: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n'
                        + 'event: message_stop\ndata: {}\n\n').encode())
                return io.BytesIO(json.dumps(payload).encode())
            events = CollectingSink()
            writer = TranscriptWriter(ws.root / "usage.jsonl")
            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake"}), mock.patch(
                "urllib.request.urlopen", side_effect=respond
            ):
                agent = ApiAgent(AnthropicProvider(model="claude-opus-5-5"), {}, ws.policy,
                                 events=events, stream=streamed, transcript=writer)
                first = agent.run([Message(Role.USER, "first")])
                first = agent.run([*first.messages, Message(Role.USER, "sticky prompt")])
                # Two calls in one agent run exercise sticky fallback suppression.
                payload["content"] = [{"type": "tool_use", "id": "missing", "name": "missing", "input": {}}]
                if not streamed:
                    count = 0
                    def two_calls(request, timeout=None):
                        nonlocal count
                        count += 1
                        value = dict(payload)
                        if count > 1:
                            value["content"] = [{"type": "text", "text": "done"}]
                        return io.BytesIO(json.dumps(value).encode())
                    events.events.clear()
                    agent = ApiAgent(AnthropicProvider(model="claude-opus-5-5"), {}, ws.policy,
                                     events=events, transcript=writer)
                    with mock.patch("urllib.request.urlopen", side_effect=two_calls):
                        first = agent.run([Message(Role.USER, "sticky")])
            writer.close()
            records, _ = read_records(ws.root / "usage.jsonl")
            if set(first.usage_by_model) != {"claude-opus-4-8"}:
                fail(f"usage charged to requested model: {first.usage_by_model!r}")
            if any(r["data"]["model"] != "claude-opus-4-8" for r in records if r["type"] == "usage"):
                fail("usage transcript used requested model")
            prices = PriceTable({
                "claude-opus-4-8": ModelPrice(Decimal("1"), Decimal("2")),
                "claude-opus-5-5": ModelPrice(Decimal("100"), Decimal("200")),
            }, "USD")
            expected_calls = 1 if streamed else 2
            if total_cost(first.usage_by_model, prices) != Decimal("0.000014") * expected_calls:
                fail("fallback call was not priced at the answering model's rate")
            fallback = events.of_type(ModelFallback)
            if len(fallback) != 1 or fallback[0].reason != "refusal" or fallback[0].from_model != "claude-opus-5-5":
                fail(f"fallback event was missing or repeated: {fallback!r}")

    for streamed in (False, True):
        with workspace() as ws:
            events = CollectingSink()
            writer = TranscriptWriter(ws.root / "dated.jsonl")
            payload = {"model": "claude-haiku-4-5-20251001",
                       "content": [{"type": "text", "text": "done"}],
                       "usage": {"input_tokens": 10, "output_tokens": 2}}
            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake"}), mock.patch(
                "urllib.request.urlopen", side_effect=respond,
            ):
                result = ApiAgent(AnthropicProvider(model="claude-haiku-4-5"), {}, ws.policy,
                                  events=events, stream=streamed, transcript=writer).run([
                                      Message(Role.USER, "alias request"),
                                  ])
            writer.close()
            records, _ = read_records(ws.root / "dated.jsonl")
            if result.usage_by_model != {"claude-haiku-4-5": UsageTotals(10, 2, 1)}:
                fail(f"dated alias was treated as another model: {result.usage_by_model!r}")
            if [r["data"]["model"] for r in records if r["type"] == "usage"] != ["claude-haiku-4-5"]:
                fail("dated alias usage record did not retain requested model")
            prices = PriceTable({"claude-haiku-4-5": ModelPrice(Decimal("1"), Decimal("2"))}, "USD")
            if total_cost(result.usage_by_model, prices) != Decimal("0.000014"):
                fail("dated alias cost did not use requested model price")
            if events.of_type(ModelFallback):
                fail("dated alias emitted a refusal fallback event")
