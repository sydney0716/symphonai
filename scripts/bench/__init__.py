"""Terminal-Bench pilot helpers shared by its agent and offline checks."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import fields
from decimal import Decimal
from pathlib import Path
from typing import Any

from symphonai_api.config import load_config
from symphonai_api.cost import PriceTable, UsageTotals, total_cost
from symphonai_api.permissions import ALWAYS_DENY_COMMANDS


def write_user_config(home: Path, roles: dict[str, dict[str, str]]) -> Path:
    """Write the task's isolated user config for SymphonAI."""
    lines = [
        "[always_deny]",
        f"lift = {json.dumps([' '.join(command) for command in ALWAYS_DENY_COMMANDS])}",
        "allow_uncheckable_shell = true",
        "allow_discarding_work = true",
        "",
        "[memory]",
        "extract = false",
        "",
    ]
    for role, model in roles.items():
        lines.extend((f"[roles.{role}]", f'provider = {json.dumps(model["provider"])}',
                      f'model = {json.dumps(model["model"])}'))
        if model.get("effort"):
            lines.append(f'effort = {json.dumps(model["effort"])}')
        lines.append("")
    path = home / ".symphonai" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def verify_user_config(
    home: Path, repo_root: Path, roles: dict[str, dict[str, str]],
) -> None:
    write_user_config(home, roles)
    config = load_config(repo_root=repo_root, home=home)
    for role, model in roles.items():
        if (config.get(f"roles.{role}.provider") != model["provider"]
                or config.get(f"roles.{role}.model") != model["model"]
                or (model.get("effort") and config.get(f"roles.{role}.effort") != model["effort"])):
            raise ValueError(f"role config did not load as provided: {role}")
    if config.get("memory.extract") is not False:
        raise ValueError("memory extraction was not disabled")
    expected_lifts = [" ".join(command) for command in ALWAYS_DENY_COMMANDS]
    if config.get("always_deny.lift") != expected_lifts:
        raise ValueError("always-deny lifts were not loaded as provided")
    if (config.get("always_deny.allow_uncheckable_shell") is not True
            or config.get("always_deny.allow_discarding_work") is not True):
        raise ValueError("always-deny shell and discard overrides were not loaded as provided")


def usage_summary(
    records: list[dict[str, Any]], prices: PriceTable | None = None,
) -> dict[str, Any]:
    """Aggregate session usage by model and agent and project Harbor totals."""
    models: dict[str, UsageTotals] = defaultdict(UsageTotals)
    agents: dict[str, UsageTotals] = defaultdict(UsageTotals)
    largest_request = 0
    for record in records:
        data = record.get("data") if record.get("type") == "usage" else None
        if not isinstance(data, dict) or not isinstance(data.get("model"), str):
            continue
        values = {field.name: data.get(field.name, 0) for field in fields(UsageTotals)}
        if any(type(value) is not int or value < 0 for value in values.values()):
            continue
        usage = UsageTotals(**values)
        model = data["model"]
        agent = data.get("agent_name")
        agent = agent if isinstance(agent, str) and agent else str(record.get("agent_id") or "leader")
        models[model] = models[model].merged(usage)
        agents[agent] = agents[agent].merged(usage)
        largest_request = max(
            largest_request,
            usage.input_tokens,
        )

    summary: dict[str, Any] = {
        "models": {name: _usage_dict(value) for name, value in sorted(models.items())},
        "agents": {name: _usage_dict(value) for name, value in sorted(agents.items())},
        "largest_single_request": largest_request,
        "context": {
            "n_input_tokens": sum(
                value.input_tokens for value in models.values()
            ),
            "n_cache_tokens": sum(value.cache_read_tokens for value in models.values()),
            "n_output_tokens": sum(value.output_tokens for value in models.values()),
        },
    }
    if prices is not None:
        costs = {
            model: total_cost({model: usage}, prices)
            for model, usage in models.items()
        }
        complete = all(value is not None for value in costs.values())
        summary["cost"] = {
            "currency": prices.currency,
            "models": {key: str(value) for key, value in costs.items() if value is not None},
            "total": str(sum((value for value in costs.values() if value is not None), Decimal(0)))
            if complete else None,
        }
    return summary


def populate_agent_context(
    context: Any, summary: dict[str, Any], prices: PriceTable | None = None,
) -> None:
    """Fill Harbor's AgentContext fields from the summary and optional prices."""
    context.n_input_tokens = summary["context"]["n_input_tokens"]
    context.n_cache_tokens = summary["context"]["n_cache_tokens"]
    context.n_output_tokens = summary["context"]["n_output_tokens"]
    context.model_usage = {
        model: {
            "n_input_tokens": row["input"],
            "n_cache_tokens": row["cache_read"],
            "n_output_tokens": row["output"],
            "cost_usd": (float(summary["cost"]["models"][model])
                         if prices is not None and model in summary["cost"]["models"] else None),
        }
        for model, row in summary["models"].items()
    }
    context.cost_usd = (
        float(summary["cost"]["total"])
        if prices is not None and summary.get("cost", {}).get("total") is not None else None
    )


def usage_table_data(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the token breakdowns shown in the pilot table."""
    return {
        "models": summary.get("models", {}),
        "agents": summary.get("agents", {}),
        "largest_single_request": summary.get("largest_single_request", 0),
        "cost": summary.get("cost"),
    }


def price_agent_usage(
    model_usage: dict[str, Any] | None,
    prices: PriceTable,
    *,
    symphonai_usage: dict[str, Any] | None = None,
    trajectory_path: Path | None = None,
    claude_stream_path: Path | None = None,
    cache_write_1h_rates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Price Harbor or SymphonAI usage from one normalized token breakdown."""
    harbor_usage = model_usage if isinstance(model_usage, dict) else {}
    symphony_models = (symphonai_usage or {}).get("models", {})
    models = symphony_models if isinstance(symphony_models, dict) and symphony_models else harbor_usage
    cache_writes: dict[str, int] = {}
    if trajectory_path is not None:
        try:
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            trajectory = {}
        final = trajectory.get("final_metrics", {}) if isinstance(trajectory, dict) else {}
        final_extra = final.get("extra", {}) if isinstance(final, dict) else {}
        steps = trajectory.get("steps", []) if isinstance(trajectory, dict) else []
        for step in steps if isinstance(steps, list) else []:
            if not isinstance(step, dict):
                continue
            metrics = step.get("metrics", {})
            extra = metrics.get("extra", {}) if isinstance(metrics, dict) else {}
            model = metrics.get("model_name") if isinstance(metrics, dict) else None
            count = extra.get("cache_creation_input_tokens", extra.get("cache_write_input_tokens"))
            if isinstance(model, str) and type(count) is int and count >= 0:
                cache_writes[model] = cache_writes.get(model, 0) + count
        if isinstance(final_extra, dict):
            count = final_extra.get("total_cache_creation_input_tokens",
                                    final_extra.get("total_cache_write_input_tokens"))
            if len(models) == 1 and type(count) is int and count >= 0:
                cache_writes[next(iter(models))] = count

    cache_write_splits: dict[str, dict[str, int]] = {}
    if claude_stream_path is not None:
        try:
            with claude_stream_path.open(encoding="utf-8") as stream:
                seen_messages: set[str] = set()
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    message = event.get("message") if isinstance(event, dict) else None
                    if not isinstance(message, dict):
                        continue
                    message_id = message.get("id")
                    model = message.get("model")
                    usage = message.get("usage")
                    if (not isinstance(message_id, str) or message_id in seen_messages
                            or not isinstance(model, str) or not isinstance(usage, dict)):
                        continue
                    creation = usage.get("cache_creation")
                    if not isinstance(creation, dict):
                        continue
                    five_minute = creation.get("ephemeral_5m_input_tokens", 0)
                    one_hour = creation.get("ephemeral_1h_input_tokens", 0)
                    if (type(five_minute) is not int or five_minute < 0
                            or type(one_hour) is not int or one_hour < 0):
                        continue
                    seen_messages.add(message_id)
                    split = cache_write_splits.setdefault(model, {"5m": 0, "1h": 0})
                    split["5m"] += five_minute
                    split["1h"] += one_hour
        except OSError:
            pass

    normalized: dict[str, dict[str, int]] = {}
    costs: dict[str, str] = {}
    unknown: list[str] = []
    estimated = False
    for model, values in models.items():
        if not isinstance(values, dict):
            continue
        symphony = "input" in values
        input_tokens = values.get("input", 0) if symphony else values.get("n_input_tokens", 0)
        output_tokens = values.get("output", 0) if symphony else values.get("n_output_tokens", 0)
        cache_read = values.get("cache_read", 0) if symphony else values.get("n_cache_tokens", 0)
        cache_write = values.get("cache_write", 0) if symphony else cache_writes.get(model, 0)
        cache_write_1h = values.get("cache_write_1h", 0) if symphony else 0
        long_input = values.get("long_context_input", 0) if symphony else 0
        long_output = values.get("long_context_output", 0) if symphony else 0
        long_cache_read = values.get("long_context_cache_read", 0) if symphony else 0
        long_cache_write = values.get("long_context_cache_write", 0) if symphony else 0
        if not all(type(value) is int and value >= 0 for value in (
            input_tokens, output_tokens, cache_read, cache_write, long_input,
            long_output, long_cache_read, long_cache_write, cache_write_1h,
        )):
            continue
        # Both APIs report input as the whole prompt, including cached tokens.
        total_input = input_tokens
        normalized[model] = {
            "input": total_input, "uncached_input": max(0, total_input - cache_read - cache_write),
            "cache_read": cache_read, "cache_write": cache_write,
            "cache_write_1h": cache_write_1h, "output": output_tokens,
        }
        if long_input or long_output or long_cache_read or long_cache_write:
            normalized[model].update({
                "long_context_input": long_input,
                "long_context_output": long_output,
                "long_context_cache_read": long_cache_read,
                "long_context_cache_write": long_cache_write,
            })
        split = cache_write_splits.get(model)
        if split is None:
            split = cache_write_splits.get(model.rsplit("/", 1)[-1])
        if split is not None:
            cache_write = split["5m"] + split["1h"]
            normalized[model]["cache_write"] = cache_write
            normalized[model]["uncached_input"] = max(
                0, total_input - cache_read - cache_write,
            )
            normalized[model]["cache_write_5m"] = split["5m"]
            normalized[model]["cache_write_1h"] = split["1h"]
            cache_write_1h = split["1h"]
        if not symphony and model not in cache_writes and split is None:
            estimated = True
        totals = UsageTotals(
            total_input, output_tokens, 0, cache_read, cache_write, long_input,
            long_output, long_cache_read, long_cache_write, cache_write_1h,
        )
        price_model = model if model in prices.prices else model.rsplit("/", 1)[-1]
        if (cache_write_1h and price_model in prices.prices
                and prices.cache_write_1h_is_estimated(price_model, totals)):
            estimated = True
        if split is not None and price_model in prices.prices:
            price = prices.prices[price_model]
            one_hour_rate = (cache_write_1h_rates or {}).get(price_model)
            one_hour_rate = Decimal(str(one_hour_rate)) if one_hour_rate is not None else None
            if split["1h"] and one_hour_rate is None:
                estimated = True
            base_totals = UsageTotals(
                max(0, totals.input_tokens - split["5m"] - split["1h"]),
                totals.output_tokens, totals.calls,
                totals.cache_read_tokens, 0, totals.long_context_input_tokens,
                totals.long_context_output_tokens, totals.long_context_cache_read_tokens,
                0,
            )
            base_cost = prices.cost(price_model, base_totals)
            write_5m_rate = (price.cache_write_per_million
                             if price.cache_write_per_million is not None
                             else price.input_per_million)
            write_1h_rate = one_hour_rate if one_hour_rate is not None else write_5m_rate
            cost = (base_cost + Decimal(split["5m"]) * write_5m_rate / Decimal(1_000_000)
                    + Decimal(split["1h"]) * write_1h_rate / Decimal(1_000_000)
                    if base_cost is not None else None)
        else:
            cost = prices.cost(price_model, totals)
        if cost is None:
            unknown.append(model)
        else:
            costs[model] = str(cost)
    total = None if unknown or not normalized else str(sum((Decimal(value) for value in costs.values()), Decimal(0)))
    return {
        "currency": prices.currency, "models": normalized, "model_costs": costs,
        "total": total, "unknown_models": sorted(unknown), "estimated": estimated,
    }


def read_usage_files(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in root.rglob("*.jsonl"):
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except OSError:
            continue
    return records


def read_leader_shell_changes(root: Path) -> list[str]:
    paths: list[str] = []
    for record in read_usage_files(root):
        if record.get("type") != "leader_shell_changes":
            continue
        data = record.get("data")
        changed = data.get("changed_existing_paths") if isinstance(data, dict) else None
        if isinstance(changed, list):
            paths.extend(path for path in changed if isinstance(path, str))
    return list(dict.fromkeys(paths))


def _usage_dict(value: UsageTotals) -> dict[str, int]:
    result = {
        "input": value.input_tokens,
        "output": value.output_tokens,
        "cache_read": value.cache_read_tokens,
        "cache_write": value.cache_write_tokens,
        "cache_write_1h": value.cache_write_1h_tokens,
        "calls": value.calls,
    }
    long_context = {
        "long_context_input": value.long_context_input_tokens,
        "long_context_output": value.long_context_output_tokens,
        "long_context_cache_read": value.long_context_cache_read_tokens,
        "long_context_cache_write": value.long_context_cache_write_tokens,
    }
    if any(long_context.values()):
        result.update(long_context)
    return result
