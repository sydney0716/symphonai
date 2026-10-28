"""Terminal-Bench pilot helpers shared by its agent and offline checks."""

from __future__ import annotations

import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from symphonai_api.config import load_config
from symphonai_api.cost import PriceTable, UsageTotals, total_cost


def write_user_config(home: Path, roles: dict[str, dict[str, str]]) -> Path:
    """Write the task's isolated user config for SymphonAI."""
    lines = ["[memory]", "extract = false", ""]
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
        values = [data.get(key, 0) for key in (
            "input_tokens", "output_tokens", "calls", "cache_read_tokens", "cache_write_tokens",
            "long_context_input_tokens", "long_context_output_tokens",
            "long_context_cache_read_tokens", "long_context_cache_write_tokens",
        )]
        if any(type(value) is not int or value < 0 for value in values):
            continue
        usage = UsageTotals(*values)
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
        long_input = values.get("long_context_input", 0) if symphony else 0
        long_output = values.get("long_context_output", 0) if symphony else 0
        long_cache_read = values.get("long_context_cache_read", 0) if symphony else 0
        long_cache_write = values.get("long_context_cache_write", 0) if symphony else 0
        if not all(type(value) is int and value >= 0 for value in (
            input_tokens, output_tokens, cache_read, cache_write, long_input,
            long_output, long_cache_read, long_cache_write,
        )):
            continue
        # Both APIs report input as the whole prompt, including cached tokens.
        total_input = input_tokens
        normalized[model] = {
            "input": total_input, "uncached_input": max(0, total_input - cache_read - cache_write),
            "cache_read": cache_read, "cache_write": cache_write, "output": output_tokens,
        }
        if long_input or long_output or long_cache_read or long_cache_write:
            normalized[model].update({
                "long_context_input": long_input,
                "long_context_output": long_output,
                "long_context_cache_read": long_cache_read,
                "long_context_cache_write": long_cache_write,
            })
        if not symphony and model not in cache_writes:
            estimated = True
        totals = UsageTotals(
            total_input, output_tokens, 0, cache_read, cache_write, long_input,
            long_output, long_cache_read, long_cache_write,
        )
        cost = prices.cost(model, totals)
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


def _usage_dict(value: UsageTotals) -> dict[str, int]:
    result = {
        "input": value.input_tokens,
        "output": value.output_tokens,
        "cache_read": value.cache_read_tokens,
        "cache_write": value.cache_write_tokens,
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
