"""Per-run turn, wall-time, token, and cost ceilings.

Wall time is checked at the agent loop's existing decision points. It does not
interrupt an in-flight provider request: providers already bound each request,
and interrupting one while preserving a distinct budget stop reason requires a
child cancellation token. That parent/child cancellation design belongs to
phase 07, so this module deliberately creates no deadline thread.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal

from symphonai_api.cost import PriceTable, UsageTotals, total_cost

DEFAULT_MAX_TURNS = 10
BUDGET_STOPPED_REASONS = (
    "budget_wall_time",
    "budget_tokens",
    "budget_cost",
)


@dataclass(frozen=True)
class RunBudget:
    max_turns: int = DEFAULT_MAX_TURNS
    wall_seconds: float | None = None
    max_total_tokens: int | None = None
    max_cost: Decimal | None = None
    price_table: PriceTable | None = None

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if self.wall_seconds is not None and self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be positive")
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if self.max_cost is not None and self.max_cost < 0:
            raise ValueError("max_cost must be non-negative")
        if self.max_cost is not None and self.price_table is None:
            raise ValueError("max_cost requires price_table")

    def lowered(self, **changes: object) -> "RunBudget":
        """Return a copy whose limits are never more permissive than these."""
        if (
            changes.get("price_table", self.price_table) is None
            and changes.get("max_cost", self.max_cost) is not None
        ):
            raise ValueError("price_table cannot be None while max_cost is set")
        for field_name in (
            "max_turns",
            "wall_seconds",
            "max_total_tokens",
            "max_cost",
        ):
            if field_name not in changes:
                continue
            old_value = getattr(self, field_name)
            new_value = changes[field_name]
            if old_value is not None and (
                new_value is None or new_value > old_value
            ):
                raise ValueError(
                    f"{field_name} cannot increase from {old_value!r} to {new_value!r}"
                )
        result = replace(self, **changes)
        return result


@dataclass(frozen=True)
class BudgetState:
    """What has been spent so far, and what stopped the run if anything did."""

    started_monotonic: float
    usage_by_model: Mapping[str, UsageTotals]

    def exceeded(self, budget: RunBudget) -> str | None:
        if (
            budget.wall_seconds is not None
            and time.monotonic() - self.started_monotonic >= budget.wall_seconds
        ):
            return "budget_wall_time"

        total_tokens = sum(usage.total_tokens for usage in self.usage_by_model.values())
        if (
            budget.max_total_tokens is not None
            and total_tokens >= budget.max_total_tokens
        ):
            return "budget_tokens"

        if budget.max_cost is not None:
            cost = total_cost(self.usage_by_model, budget.price_table)
            if cost is not None and cost >= budget.max_cost:
                return "budget_cost"
        return None
