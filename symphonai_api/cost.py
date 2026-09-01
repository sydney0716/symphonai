"""Token-usage aggregation and user-supplied model pricing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from symphonai_api.models import Usage


@dataclass(frozen=True)
class UsageTotals:
    """Token usage accumulated across one or more provider calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def merged(self, other: "UsageTotals") -> "UsageTotals":
        return UsageTotals(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            calls=self.calls + other.calls,
        )

    @classmethod
    def from_usage(cls, usage: Usage) -> "UsageTotals":
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            calls=1,
        )


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    output_per_million: Decimal


@dataclass(frozen=True)
class PriceTable:
    prices: Mapping[str, ModelPrice]
    currency: str

    def cost(self, model: str, totals: UsageTotals) -> Decimal | None:
        price = self.prices.get(model)
        if price is None:
            return None
        million = Decimal(1_000_000)
        return (
            Decimal(totals.input_tokens) * price.input_per_million
            + Decimal(totals.output_tokens) * price.output_per_million
        ) / million


def _rate(value: object, *, model: str, kind: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(
            f"price table rate {kind!r} for model {model!r} must be a non-negative number"
        )
    try:
        rate = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(
            f"price table rate {kind!r} for model {model!r} must be a non-negative number"
        ) from None
    if not rate.is_finite() or rate < 0:
        raise ValueError(
            f"price table rate {kind!r} for model {model!r} must be a non-negative number"
        )
    return rate


def load_price_table(path: str | Path) -> PriceTable:
    """Load and validate a version-1 JSON model-price table."""

    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"price table could not be read: {exc}") from None
    except (UnicodeError, json.JSONDecodeError) as exc:
        detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
        raise ValueError(f"price table is not valid JSON: {detail}") from None

    if not isinstance(data, dict):
        raise ValueError("price table must be a JSON object")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("price table version must be 1")
    currency = data.get("currency")
    if not isinstance(currency, str) or not currency:
        raise ValueError("price table currency must be a non-empty string")
    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError("price table models must be an object")

    prices: dict[str, ModelPrice] = {}
    for model, entry in models.items():
        if not isinstance(entry, dict):
            raise ValueError(f"price table entry for model {model!r} must be an object")
        prices[model] = ModelPrice(
            input_per_million=_rate(entry.get("input"), model=model, kind="input"),
            output_per_million=_rate(entry.get("output"), model=model, kind="output"),
        )
    return PriceTable(prices=prices, currency=currency)


def total_cost(
    usage_by_model: Mapping[str, UsageTotals],
    table: PriceTable | None,
) -> Decimal | None:
    """Return the exact complete cost, or ``None`` if any price is unknown."""

    if table is None:
        return None
    total = Decimal(0)
    for model, totals in usage_by_model.items():
        model_cost = table.cost(model, totals)
        if model_cost is None:
            return None
        total += model_cost
    return total
