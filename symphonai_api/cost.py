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
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    long_context_input_tokens: int = 0
    long_context_output_tokens: int = 0
    long_context_cache_read_tokens: int = 0
    long_context_cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def merged(self, other: "UsageTotals") -> "UsageTotals":
        return UsageTotals(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            calls=self.calls + other.calls,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            long_context_input_tokens=self.long_context_input_tokens + other.long_context_input_tokens,
            long_context_output_tokens=self.long_context_output_tokens + other.long_context_output_tokens,
            long_context_cache_read_tokens=self.long_context_cache_read_tokens + other.long_context_cache_read_tokens,
            long_context_cache_write_tokens=self.long_context_cache_write_tokens + other.long_context_cache_write_tokens,
        )

    @classmethod
    def from_usage(cls, usage: Usage, *, long_context: bool = False) -> "UsageTotals":
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            calls=1,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            long_context_input_tokens=usage.input_tokens if long_context else 0,
            long_context_output_tokens=usage.output_tokens if long_context else 0,
            long_context_cache_read_tokens=usage.cache_read_tokens if long_context else 0,
            long_context_cache_write_tokens=usage.cache_write_tokens if long_context else 0,
        )


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    output_per_million: Decimal
    cache_read_per_million: Decimal | None = None
    cache_write_per_million: Decimal | None = None
    long_context: "ModelPrice | None" = None


@dataclass(frozen=True)
class PriceTable:
    prices: Mapping[str, ModelPrice]
    currency: str

    def cost(self, model: str, totals: UsageTotals) -> Decimal | None:
        price = self.prices.get(model)
        if price is None:
            return None
        million = Decimal(1_000_000)
        def bucket_cost(rates: ModelPrice, input_tokens: int, output_tokens: int,
                        cache_read_tokens: int, cache_write_tokens: int) -> Decimal:
            uncached = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
            read_rate = rates.cache_read_per_million if rates.cache_read_per_million is not None else rates.input_per_million
            write_rate = rates.cache_write_per_million if rates.cache_write_per_million is not None else rates.input_per_million
            return (Decimal(uncached) * rates.input_per_million
                    + Decimal(cache_read_tokens) * read_rate
                    + Decimal(cache_write_tokens) * write_rate
                    + Decimal(output_tokens) * rates.output_per_million)

        long = totals.long_context_input_tokens > 0
        if long and price.long_context is not None:
            base_cost = bucket_cost(
                price, totals.input_tokens - totals.long_context_input_tokens,
                totals.output_tokens - totals.long_context_output_tokens,
                totals.cache_read_tokens - totals.long_context_cache_read_tokens,
                totals.cache_write_tokens - totals.long_context_cache_write_tokens,
            )
            long_cost = bucket_cost(
                price.long_context, totals.long_context_input_tokens,
                totals.long_context_output_tokens, totals.long_context_cache_read_tokens,
                totals.long_context_cache_write_tokens,
            )
            return (base_cost + long_cost) / million
        return bucket_cost(price, totals.input_tokens, totals.output_tokens,
                           totals.cache_read_tokens, totals.cache_write_tokens) / million

    def long_context_is_estimated(self, model: str, totals: UsageTotals) -> bool:
        price = self.prices.get(model)
        return totals.long_context_input_tokens > 0 and price is not None and price.long_context is None


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
        long_context = entry.get("long_context")
        if long_context is not None and not isinstance(long_context, dict):
            raise ValueError(f"price table long_context for model {model!r} must be an object")
        def parse_rates(source: dict, *, prefix: str = "") -> ModelPrice:
            return ModelPrice(
                input_per_million=_rate(source.get("input"), model=model, kind=f"{prefix}input"),
                output_per_million=_rate(source.get("output"), model=model, kind=f"{prefix}output"),
                cache_read_per_million=_rate(source["cache_read"], model=model, kind=f"{prefix}cache_read") if "cache_read" in source else None,
                cache_write_per_million=_rate(source["cache_write"], model=model, kind=f"{prefix}cache_write") if "cache_write" in source else None,
            )
        base = parse_rates(entry)
        prices[model] = ModelPrice(
            input_per_million=base.input_per_million, output_per_million=base.output_per_million,
            cache_read_per_million=base.cache_read_per_million,
            cache_write_per_million=base.cache_write_per_million,
            long_context=parse_rates(long_context, prefix="long_context.") if long_context is not None else None,
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
