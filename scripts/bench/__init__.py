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
            "input_tokens", "output_tokens", "calls", "cache_read_tokens", "cache_write_tokens"
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
            usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens,
        )

    summary: dict[str, Any] = {
        "models": {name: _usage_dict(value) for name, value in sorted(models.items())},
        "agents": {name: _usage_dict(value) for name, value in sorted(agents.items())},
        "largest_single_request": largest_request,
        "context": {
            "n_input_tokens": sum(
                value.input_tokens + value.cache_read_tokens + value.cache_write_tokens
                for value in models.values()
            ),
            "n_cache_tokens": sum(
                value.cache_read_tokens + value.cache_write_tokens for value in models.values()
            ),
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
            "n_input_tokens": row["input"] + row["cache_read"] + row["cache_write"],
            "n_cache_tokens": row["cache_read"] + row["cache_write"],
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
    return {
        "input": value.input_tokens,
        "output": value.output_tokens,
        "cache_read": value.cache_read_tokens,
        "cache_write": value.cache_write_tokens,
        "calls": value.calls,
    }
