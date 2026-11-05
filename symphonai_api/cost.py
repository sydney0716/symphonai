"""Token-usage aggregation and layered model pricing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from importlib import resources
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
    cache_write_1h_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def merged(self, other: "UsageTotals") -> "UsageTotals":
        return type(self)(**{
            field.name: getattr(self, field.name) + getattr(other, field.name)
            for field in fields(self)
        })

    @classmethod
    def from_usage(cls, usage: Usage, *, long_context: bool = False) -> "UsageTotals":
        values = {}
        for field in fields(cls):
            name = field.name
            if name == "calls":
                values[name] = 1
            elif name.startswith("long_context_"):
                values[name] = getattr(usage, name.removeprefix("long_context_"), 0) if long_context else 0
            else:
                values[name] = getattr(usage, name, 0)
        return cls(**values)

    def to_record(self) -> dict[str, int]:
        """Return the usage counters written to transcripts and stats replies."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    output_per_million: Decimal
    cache_read_per_million: Decimal | None = None
    cache_write_per_million: Decimal | None = None
    long_context: "ModelPrice | None" = None
    cache_write_1h_per_million: Decimal | None = None


@dataclass(frozen=True)
class PriceTable:
    prices: Mapping[str, ModelPrice]
    currency: str

    def price_for_model(self, model: str) -> ModelPrice | None:
        price = self.prices.get(model)
        if price is not None:
            return price
        for name, price in self.prices.items():
            prefix = f"{name}-"
            if model.startswith(prefix):
                suffix = model[len(prefix):]
                if suffix == "preview" or (len(suffix) == 8 and suffix.isascii() and suffix.isdigit()):
                    return price
        return None

    def cost(self, model: str, totals: UsageTotals) -> Decimal | None:
        price = self.price_for_model(model)
        if price is None:
            return None
        million = Decimal(1_000_000)
        def bucket_cost(rates: ModelPrice, input_tokens: int, output_tokens: int,
                        cache_read_tokens: int, cache_write_tokens: int,
                        cache_write_1h_tokens: int = 0) -> Decimal:
            uncached = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
            read_rate = rates.cache_read_per_million if rates.cache_read_per_million is not None else rates.input_per_million
            write_rate = rates.cache_write_per_million if rates.cache_write_per_million is not None else rates.input_per_million
            one_hour_tokens = min(cache_write_tokens, cache_write_1h_tokens)
            one_hour_rate = rates.cache_write_1h_per_million if rates.cache_write_1h_per_million is not None else write_rate
            return (Decimal(uncached) * rates.input_per_million
                    + Decimal(cache_read_tokens) * read_rate
                    + Decimal(cache_write_tokens - one_hour_tokens) * write_rate
                    + Decimal(one_hour_tokens) * one_hour_rate
                    + Decimal(output_tokens) * rates.output_per_million)

        long = totals.long_context_input_tokens > 0
        if long and price.long_context is not None:
            long_write_1h_tokens = min(
                totals.cache_write_1h_tokens, totals.long_context_cache_write_tokens,
            )
            base_cost = bucket_cost(
                price, totals.input_tokens - totals.long_context_input_tokens,
                totals.output_tokens - totals.long_context_output_tokens,
                totals.cache_read_tokens - totals.long_context_cache_read_tokens,
                totals.cache_write_tokens - totals.long_context_cache_write_tokens,
                totals.cache_write_1h_tokens - long_write_1h_tokens,
            )
            long_cost = bucket_cost(
                price.long_context, totals.long_context_input_tokens,
                totals.long_context_output_tokens, totals.long_context_cache_read_tokens,
                totals.long_context_cache_write_tokens,
                long_write_1h_tokens,
            )
            return (base_cost + long_cost) / million
        return bucket_cost(price, totals.input_tokens, totals.output_tokens,
                           totals.cache_read_tokens, totals.cache_write_tokens,
                           totals.cache_write_1h_tokens) / million


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

    return _price_table_from_json(data)


@lru_cache(maxsize=1)
def shipped_price_table() -> PriceTable:
    data = json.loads(resources.files("symphonai_api.data").joinpath("model_prices.json").read_text(encoding="utf-8"))
    return _price_table_from_json(data)


def _price_table_from_json(data: object) -> PriceTable:
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
                cache_write_1h_per_million=_rate(source["cache_write_1h"], model=model, kind=f"{prefix}cache_write_1h") if "cache_write_1h" in source else None,
            )
        base = parse_rates(entry)
        prices[model] = ModelPrice(
            input_per_million=base.input_per_million, output_per_million=base.output_per_million,
            cache_read_per_million=base.cache_read_per_million,
            cache_write_per_million=base.cache_write_per_million,
            long_context=parse_rates(long_context, prefix="long_context.") if long_context is not None else None,
            cache_write_1h_per_million=base.cache_write_1h_per_million,
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
