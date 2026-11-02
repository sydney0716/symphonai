"""Saved token totals and host-only vendor cost and balance reports."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from urllib.parse import urlencode

from symphonai_api.cost import UsageTotals, total_cost
from symphonai_api.secrets import mask_secrets
from symphonai_api.session import TranscriptError, read_records
from symphonai_api.web import RedirectBlocker, WebFetchError


COST_URLS = {
    "anthropic": "https://api.anthropic.com/v1/organizations/analytics/cost",
    "openai": "https://api.openai.com/v1/organization/costs",
}
BALANCES = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/user/balance"),
    "kimi": ("MOONSHOT_API_KEY", "https://api.moonshot.ai/v1/users/me/balance"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/credits"),
}


def _number(value):
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("invalid amount")
    return number


def _money(value):
    amount = _number(value["amount"])
    return {**value, "amount": format(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f"),
            "exact_amount": str(amount)}


def _get(url, headers, params=None):
    if params:
        url += "?" + urlencode(params, doseq=True)
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.build_opener(RedirectBlocker()).open(request, timeout=10) as response:
        if not 200 <= response.status < 300:
            raise ValueError("unsuccessful vendor response")
        value = json.loads(response.read(), parse_float=Decimal)
    if not isinstance(value, dict):
        raise ValueError("invalid vendor response")
    return value


def _pages(url, headers, params):
    seen = set()
    while True:
        value = _get(url, headers, params)
        yield value
        if not value.get("has_more"):
            return
        page = value.get("next_page")
        if not isinstance(page, str) or not page or page in seen:
            raise ValueError("invalid pagination")
        seen.add(page)
        params = {**params, "page": page}


def _cost(provider, key, start, end):
    headers = {"Authorization": "Bearer " + key}
    total = Decimal(0)
    currency = None
    refreshed = []
    cursor = start
    if provider == "anthropic":
        headers.update({"anthropic-version": "2023-06-01",
                        "anthropic-beta": "analytics-usage-cost-2026-08-24"})
        cursor = max(start, datetime(2026, 1, 1, tzinfo=timezone.utc), end - timedelta(days=365))
    while cursor < end:
        until = min(cursor + timedelta(days=31), end) if provider == "anthropic" else end
        params = ({"starting_at": cursor.isoformat(), "ending_at": until.isoformat(),
                   "time_bucket_width": "1d", "group_by[]": "model"}
                  if provider == "anthropic" else
                  {"start_time": int(cursor.timestamp()), "end_time": int(until.timestamp()),
                   "bucket_width": "1d", "limit": (until.date() - cursor.date()).days + 1,
                   "group_by": "line_item"})
        for page in _pages(COST_URLS[provider], headers, params):
            if isinstance(page.get("data_refreshed_at"), str):
                refreshed.append(page["data_refreshed_at"])
            for bucket in page["data"]:
                amounts = ([bucket["cost"]] if provider == "anthropic" else
                           [item["amount"] for item in bucket["results"]])
                for amount in amounts:
                    unit = amount["currency"].upper()
                    if currency is not None and unit != currency:
                        raise ValueError("mixed currencies")
                    currency = unit
                    total += (_number(amount["amount"]) / 100 if provider == "anthropic"
                              else _number(amount["value"]))
        cursor = until
    result = {"amount": str(total), "currency": currency or "USD", "source": "vendor",
              "scope": "organization"}
    if refreshed:
        result["data_refreshed_at"] = min(refreshed)
    return result


def _balance(provider, key):
    value = _get(BALANCES[provider][1], {"Authorization": "Bearer " + key})
    if provider == "deepseek":
        if value.get("is_available") is not True:
            raise ValueError("balance unavailable")
        return [{"amount": str(_number(item["total_balance"])), "currency": item["currency"]}
                for item in value["balance_infos"]]
    data = value["data"]
    if provider == "kimi":
        if value.get("code") != 0 or value.get("status") is not True:
            raise ValueError("balance unavailable")
        return [{"amount": str(_number(data["available_balance"])), "currency": None}]
    return [{"amount": str(_number(data["total_credits"]) - _number(data["total_usage"])),
             "currency": "USD"}]


def _local(root, start, end):
    buckets = defaultdict(UsageTotals)
    warnings = []
    root = Path(root)
    if not root.is_dir():
        return buckets, warnings
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name == "loops":
            continue
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            project = meta.get("repo_root", "")
            choice = meta.get("provider_choice") or {}
            provider = choice.get("name", "unknown")
            if meta.get("provider_state_reset"):
                provider = "unknown"
                warnings.append("Calls without a saved provider in switched-provider sessions are grouped as unknown.")
        except (OSError, ValueError, TypeError, AttributeError):
            warnings.append("Some session metadata could not be read.")
            continue
        for path in (directory / "run.jsonl", *sorted(directory.glob("agent-*.jsonl"))):
            try:
                records, _ = read_records(path)
            except (OSError, TranscriptError, ValueError):
                warnings.append("Some session usage could not be read.")
                continue
            starts = 0
            for record in records:
                if record.get("type") == "run_started":
                    starts += 1
                if record.get("type") != "usage":
                    continue
                # Fork prefixes copy old usage; their first real run starts later.
                if path.name == "run.jsonl" and meta.get("parent_session_id") and starts < 2:
                    continue
                try:
                    at = datetime.fromisoformat(record["ts"])
                    if at.tzinfo is None:
                        at = at.replace(tzinfo=timezone.utc)
                    if not start <= at < end:
                        continue
                    data = record["data"]
                    model = data["model"]
                    vendor = data.get("provider", provider)
                    if not all(isinstance(item, str) for item in (model, vendor, project)):
                        raise ValueError("invalid usage identity")
                    values = {field.name: data.get(field.name, 0) for field in fields(UsageTotals)}
                    if any(type(value) is not int or value < 0 for value in values.values()):
                        raise ValueError("invalid usage counters")
                    key = (at.astimezone(timezone.utc).date().isoformat(), vendor, model, project)
                    buckets[key] = buckets[key].merged(UsageTotals(**values))
                except (KeyError, TypeError, ValueError):
                    warnings.append("Some saved usage records were invalid.")
    return buckets, warnings


class UsageReport:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def _vendor(self, provider, kind, key, start, end):
        period = (start.date(), end.date()) if kind == "cost" else None
        identity = (provider, kind, period, hashlib.sha256(key.encode()).digest())
        with self._lock:
            now = time.monotonic()
            self._cache = {item: entry for item, entry in self._cache.items() if entry[0] > now}
            cached = self._cache.get(identity)
            if cached is not None:
                return cached[1]
            try:
                result = {kind: _cost(provider, key, start, end) if kind == "cost" else _balance(provider, key)}
            except urllib.error.HTTPError as exc:
                result = {"error": f"Vendor {kind} request returned HTTP {exc.code}."}
            except (OSError, http.client.HTTPException, ValueError, TypeError, KeyError, AttributeError, InvalidOperation, WebFetchError):
                result = {"error": f"Vendor {kind} report is unavailable."}
            self._cache[identity] = (time.monotonic() + 900, result)
            return result

    def read(self, root, *, days=30, price_table=None, now=None):
        if type(days) is not int or not 1 <= days <= 90:
            raise ValueError("days must be an integer from 1 to 90")
        end = now or datetime.now(timezone.utc)
        start = end.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
        buckets, warnings = _local(root, start, end)
        providers = {}
        projects = defaultdict(UsageTotals)
        daily = defaultdict(UsageTotals)
        tokens = []
        for (date, provider, model, project), totals in sorted(buckets.items()):
            row = providers.setdefault(provider, {"provider": provider, "totals": UsageTotals(), "models": {}})
            row["totals"] = row["totals"].merged(totals)
            row["models"][model] = row["models"].get(model, UsageTotals()).merged(totals)
            projects[project] = projects[project].merged(totals)
            daily[date] = daily[date].merged(totals)
            tokens.append({"date": date, "provider": provider, "model": model, "project": project, **totals.to_record()})
        for provider in ("anthropic", "openai"):
            key = os.environ.get(provider.upper() + "_ADMIN_KEY", "").strip()
            if key:
                row = providers.setdefault(provider, {"provider": provider, "totals": UsageTotals(), "models": {}})
                row.update(self._vendor(provider, "cost", key, start, end))
        for provider, (variable, _) in BALANCES.items():
            key = os.environ.get(variable, "").strip()
            if key:
                row = providers.setdefault(provider, {"provider": provider, "totals": UsageTotals(), "models": {}})
                row.update(self._vendor(provider, "balance", key, start, end))
        for row in providers.values():
            if "cost" not in row:
                estimate = total_cost(row["models"], price_table) if row["models"] else None
                row["cost"] = ({"amount": str(estimate), "currency": price_table.currency, "source": "estimate",
                                "scope": "saved_calls"} if estimate is not None else None)
            row.update(row.pop("totals").to_record())
            row["models"] = [{"model": model, **usage.to_record()} for model, usage in sorted(row["models"].items())]
            row.setdefault("balance", None)
            row.setdefault("error", None)
            if row["cost"] is not None:
                row["cost"] = _money(row["cost"])
            if row["balance"] is not None:
                row["balance"] = [_money(value) if value.get("currency") else value for value in row["balance"]]
        result = {"days": days, "starting_at": start.isoformat(), "ending_at": end.isoformat(),
                  "tokens": tokens, "providers": sorted(providers.values(), key=lambda row: row["provider"]),
                  "projects": [{"repo_root": project, **totals.to_record()} for project, totals in sorted(projects.items())],
                  "daily": [{"date": (start + timedelta(days=index)).date().isoformat(),
                             **daily[(start + timedelta(days=index)).date().isoformat()].to_record()}
                            for index in range(days)], "warnings": sorted(set(warnings))}
        return json.loads(mask_secrets(json.dumps(result)))
