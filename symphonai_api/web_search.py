"""Configurable HTTP JSON web search.

The shipped presets have not been probed against a live account. Treat them as
starting points to compare with current vendor documentation. A 404 or an
unknown-parameter error should make the reader suspect
``data/search_backends.json`` first.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from symphonai_api.call_class import CallClass
from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.identity import SCHEMA_VERSION
from symphonai_api.providers.base import ProviderError, parse_json_object
from symphonai_api.retry import DEFAULT_MAX_ATTEMPTS, read_with_retry


DEFAULT_SEARCH_TIMEOUT_SECONDS = 15.0
DEFAULT_SEARCH_LIMIT = 5
MIN_SEARCH_LIMIT = 1
MAX_SEARCH_LIMIT = 10


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    schema_version: int = SCHEMA_VERSION


class SearchBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Stable backend name for diagnostics and configuration."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        limit: int,
        cancel: CancellationToken | None = None,
    ) -> list[SearchHit]:
        """Search for ``query`` and return at most ``limit`` hits."""


class SearchBackendError(Exception):
    """A search failed. Never carries the API key or a request header."""


@dataclass(frozen=True)
class SearchEndpoint:
    key: str
    label: str
    url: str
    api_key_env_var: str
    query_parameter: str
    auth_header: str
    results_path: tuple[str, ...]
    title_field: str
    url_field: str
    snippet_field: str
    notes: str


def _endpoint_from_json(value: object) -> SearchEndpoint | None:
    if not isinstance(value, dict):
        return None
    string_fields = (
        "key",
        "label",
        "url",
        "api_key_env_var",
        "query_parameter",
        "auth_header",
        "title_field",
        "url_field",
        "snippet_field",
        "notes",
    )
    if any(not isinstance(value.get(field), str) or not value[field] for field in string_fields):
        return None
    path = value.get("results_path")
    if not isinstance(path, list) or not all(isinstance(part, str) and part for part in path):
        return None
    return SearchEndpoint(
        **{field: value[field] for field in string_fields},
        results_path=tuple(path),
    )


@lru_cache(maxsize=1)
def search_endpoints() -> tuple[SearchEndpoint, ...]:
    """Read the shipped unverified search endpoint presets once, fail-closed."""

    try:
        payload = json.loads(
            resources.files("symphonai_api.data")
            .joinpath("search_backends.json")
            .read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return ()
        raw_backends = payload.get("backends")
        if not isinstance(raw_backends, list):
            return ()
        endpoints = tuple(_endpoint_from_json(item) for item in raw_backends)
        if any(endpoint is None for endpoint in endpoints):
            return ()
        return tuple(endpoint for endpoint in endpoints if endpoint is not None)
    except (OSError, TypeError, ValueError, UnicodeError):
        return ()


def search_endpoint(key: str) -> SearchEndpoint:
    """Return one shipped preset by key, raising for an unknown key."""

    for endpoint in search_endpoints():
        if endpoint.key == key:
            return endpoint
    known = sorted(endpoint.key for endpoint in search_endpoints())
    raise KeyError(f"unknown search backend {key!r}; known: {known}")


class HttpJsonSearchBackend(SearchBackend):
    def __init__(
        self,
        endpoint: SearchEndpoint,
        *,
        timeout: float = DEFAULT_SEARCH_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_attempts = max_attempts

    @property
    def name(self) -> str:
        return self.endpoint.key

    def search(
        self,
        query: str,
        *,
        limit: int,
        cancel: CancellationToken | None = None,
    ) -> list[SearchHit]:
        api_key = os.environ.get(self.endpoint.api_key_env_var, "")
        if not api_key:
            raise SearchBackendError(
                f"missing search API key environment variable {self.endpoint.api_key_env_var}"
            )
        bounded_limit = max(MIN_SEARCH_LIMIT, min(MAX_SEARCH_LIMIT, limit))
        query_string = urllib.parse.urlencode(
            {self.endpoint.query_parameter: query}
        )
        request = urllib.request.Request(
            f"{self.endpoint.url}?{query_string}",
            headers={self.endpoint.auth_header: api_key},
            method="GET",
        )
        try:
            raw = read_with_retry(
                request,
                timeout=self.timeout,
                max_attempts=self.max_attempts,
                api_key=api_key,
                operation=f"{self.endpoint.label} search",
                cancel=cancel,
                call_class=CallClass.BACKGROUND,
            )
            payload = parse_json_object(raw, f"{self.endpoint.label} search")
        except OperationCancelled:
            raise
        except ProviderError as exc:
            raise SearchBackendError(str(exc)) from None

        result_values: object = payload
        for part in self.endpoint.results_path:
            if not isinstance(result_values, dict):
                result_values = []
                break
            result_values = result_values.get(part, [])
        if not isinstance(result_values, list):
            return []
        return [
            SearchHit(
                title=_string_field(item, self.endpoint.title_field),
                url=_string_field(item, self.endpoint.url_field),
                snippet=_string_field(item, self.endpoint.snippet_field),
            )
            for item in result_values[:bounded_limit]
        ]


def _string_field(value: object, field: str) -> str:
    if not isinstance(value, dict):
        return ""
    candidate = value.get(field, "")
    return candidate if isinstance(candidate, str) else ""
