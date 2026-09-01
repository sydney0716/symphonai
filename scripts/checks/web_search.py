"""Registered checks for conditionally configured web search."""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import unittest.mock as mock
from contextlib import contextmanager
from typing import Iterator

from symphonai_api.call_class import CallClass
from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.models import ToolCall
from symphonai_api.runner import standard_tool_registry
from symphonai_api.tools.metadata import ResultHint, ToolEffect, ToolMetadata
from symphonai_api.tools.web_search import WebSearchTool
from symphonai_api.web_search import (
    HttpJsonSearchBackend,
    SearchBackend,
    SearchBackendError,
    SearchEndpoint,
    SearchHit,
    search_endpoints,
)
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


KEY_ENV_VAR = "SYMPHONAI_WEB_SEARCH_TEST_KEY"


def _endpoint(**overrides: object) -> SearchEndpoint:
    values: dict[str, object] = {
        "key": "test",
        "label": "Test Search",
        "url": "https://search.example.test/v1/search",
        "api_key_env_var": KEY_ENV_VAR,
        "query_parameter": "q",
        "auth_header": "X-Test-Key",
        "results_path": ("outer", "items"),
        "title_field": "title",
        "url_field": "link",
        "snippet_field": "summary",
        "notes": "UNVERIFIED test endpoint",
    }
    values.update(overrides)
    return SearchEndpoint(**values)


class _FakeBackend(SearchBackend):
    def __init__(
        self,
        hits: list[SearchHit] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.hits = list(hits or [])
        self.error = error
        self.calls: list[tuple[str, int, CancellationToken | None]] = []

    @property
    def name(self) -> str:
        return "fake-search"

    def search(
        self,
        query: str,
        *,
        limit: int,
        cancel: CancellationToken | None = None,
    ) -> list[SearchHit]:
        self.calls.append((query, limit, cancel))
        if self.error is not None:
            raise self.error
        return self.hits[:limit]


class _Response:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@contextmanager
def _environment_value(value: str | None) -> Iterator[None]:
    previous = os.environ.get(KEY_ENV_VAR)
    try:
        if value is None:
            os.environ.pop(KEY_ENV_VAR, None)
        else:
            os.environ[KEY_ENV_VAR] = value
        yield
    finally:
        if previous is None:
            os.environ.pop(KEY_ENV_VAR, None)
        else:
            os.environ[KEY_ENV_VAR] = previous


def _call(query: str, *, limit: int | None = None, index: int = 0) -> ToolCall:
    arguments: dict = {"query": query}
    if limit is not None:
        arguments["limit"] = limit
    return ToolCall(id=f"search-{index}", name="web_search", arguments=arguments)


@check("web_search.absent_without_backend")
def check_absent_without_backend() -> None:
    registry = standard_tool_registry()
    if "web_search" in registry:
        fail(f"web_search appeared without a backend: {list(registry)!r}")
    try:
        standard_tool_registry(["web_search"])
    except ValueError as exc:
        if "unknown tool name" not in str(exc):
            fail(f"absent search name failed unclearly: {exc!r}")
    else:
        fail("web_search was selectable without a backend")


@check("web_search.registered_with_backend")
def check_registered_with_backend() -> None:
    backend = _FakeBackend()
    registry = standard_tool_registry(search_backend=backend)
    selected = standard_tool_registry(
        ["web_search"], search_backend=backend
    )
    if list(registry)[-2:] != ["web_fetch", "web_search"]:
        fail(f"configured search tool order changed: {list(registry)!r}")
    if list(selected) != ["web_search"] or not isinstance(
        selected["web_search"], WebSearchTool
    ):
        fail(f"configured web_search selection failed: {selected!r}")
    endpoints = search_endpoints()
    if len(endpoints) != 2 or any("unverified" not in item.notes.casefold() for item in endpoints):
        fail(f"shipped search presets are missing or not marked unverified: {endpoints!r}")


@check("web_search.key_from_env_not_url")
def check_key_from_env_not_url() -> None:
    backend = HttpJsonSearchBackend(_endpoint())
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, *, timeout: float) -> _Response:
        requests.append(request)
        return _Response({"outer": {"items": []}})

    with mock.patch("urllib.request.urlopen", side_effect=urlopen):
        with _environment_value("first-secret"):
            backend.search("alpha beta", limit=5)
        with _environment_value("second-secret"):
            backend.search("gamma", limit=5)
        with _environment_value(None):
            try:
                backend.search("missing", limit=5)
            except SearchBackendError as exc:
                if KEY_ENV_VAR not in str(exc) or "secret" in str(exc):
                    fail(f"missing-key error leaked or omitted context: {exc!r}")
            else:
                fail("missing search key was accepted")
    if len(requests) != 2:
        fail(f"environment-backed searches issued the wrong calls: {requests!r}")
    for request, expected_key in zip(requests, ("first-secret", "second-secret"), strict=True):
        if expected_key in request.full_url or request.get_header("X-test-key") != expected_key:
            fail(f"search key was not header-only and fresh: {request.full_url!r}")
    if any(value in backend.__dict__.values() for value in ("first-secret", "second-secret")):
        fail("backend retained an API key on the object")


@check("web_search.parses_results")
def check_parses_results() -> None:
    payload = {
        "outer": {
            "items": [
                {"title": "First", "link": "https://one.test", "summary": "One"},
                {"title": "Second"},
                "malformed",
            ]
        }
    }
    backend = HttpJsonSearchBackend(_endpoint())
    with workspace() as ws, _environment_value("parse-secret"), mock.patch(
        "urllib.request.urlopen", return_value=_Response(payload)
    ):
        result = WebSearchTool(backend).execute(_call("parse"), ws.policy)
    expected_hits = [
        {"title": "First", "url": "https://one.test", "snippet": "One"},
        {"title": "Second", "url": "", "snippet": ""},
        {"title": "", "url": "", "snippet": ""},
    ]
    if not result.ok or result.payload != {"hits": expected_hits}:
        fail(f"search result payload was not resilient: {result!r}")
    if "1. First — https://one.test\n   One" not in result.content or "2. Second — " not in result.content:
        fail(f"search result text lost structured information: {result.content!r}")


@check("web_search.limit_clamped")
def check_limit_clamped() -> None:
    backend = _FakeBackend([SearchHit(str(index), str(index)) for index in range(12)])
    tool = WebSearchTool(backend)
    with workspace() as ws:
        low = tool.execute(_call("low", limit=0, index=1), ws.policy)
        high = tool.execute(_call("high", limit=99, index=2), ws.policy)
        default = tool.execute(_call("default", index=3), ws.policy)
    if [call[1] for call in backend.calls] != [1, 10, 5]:
        fail(f"search limits were not clamped/defaulted: {backend.calls!r}")
    if len(low.payload["hits"]) != 1 or len(high.payload["hits"]) != 10 or len(default.payload["hits"]) != 5:
        fail("rendered search payload did not honor bounded limits")


@check("web_search.background_call_class")
def check_background_call_class() -> None:
    backend = HttpJsonSearchBackend(_endpoint())
    with _environment_value("background-secret"), mock.patch(
        "symphonai_api.web_search.read_with_retry",
        return_value=b'{"outer":{"items":[]}}',
    ) as reader:
        backend.search("background", limit=5)
    if reader.call_args.kwargs.get("call_class") is not CallClass.BACKGROUND:
        fail(f"search did not use background retry policy: {reader.call_args!r}")


@check("web_search.contacts_only_endpoint")
def check_contacts_only_endpoint() -> None:
    backend = HttpJsonSearchBackend(_endpoint())
    requests: list[urllib.request.Request] = []
    payload = {
        "outer": {
            "items": [
                {"title": "Result", "link": "https://result.example/never-fetch", "summary": ""}
            ]
        }
    }

    def urlopen(request: urllib.request.Request, *, timeout: float) -> _Response:
        requests.append(request)
        return _Response(payload)

    with _environment_value("endpoint-secret"), mock.patch(
        "urllib.request.urlopen", side_effect=urlopen
    ):
        hits = backend.search("one request", limit=5)
    hosts = [urllib.parse.urlsplit(request.full_url).hostname for request in requests]
    if len(requests) != 1 or hosts != ["search.example.test"]:
        fail(f"search contacted something besides its endpoint: {requests!r}")
    if hits[0].url != "https://result.example/never-fetch":
        fail(f"search result URL was not returned untouched: {hits!r}")


@check("web_search.metadata_contract")
def check_metadata_contract() -> None:
    expected = ToolMetadata(
        effect=ToolEffect.READ_ONLY,
        concurrency_safe=True,
        paths=(),
        result_hint=ResultHint.TEXT,
    )
    if WebSearchTool(_FakeBackend()).metadata({"query": "x"}) != expected:
        fail("web_search metadata contract changed")


@check("web_search.error_carries_no_secret")
def check_error_carries_no_secret() -> None:
    key = "SEARCH_KEY_MUST_NOT_LEAK"
    backend = HttpJsonSearchBackend(_endpoint(), max_attempts=1)
    error = urllib.error.HTTPError(
        _endpoint().url,
        400,
        "Bad Request",
        {},
        io.BytesIO(f"vendor echoed {key}".encode("utf-8")),
    )
    with workspace() as ws, _environment_value(key), mock.patch(
        "urllib.request.urlopen", side_effect=error
    ):
        result = WebSearchTool(backend).execute(_call("fail"), ws.policy)
    if result.ok or key in (result.error or "") or "[redacted]" not in (result.error or ""):
        fail(f"search error exposed its request secret: {result!r}")


@check("web_search.cancel_propagates")
def check_cancellation_propagates() -> None:
    token = CancellationToken()
    token.cancel()
    backend = HttpJsonSearchBackend(_endpoint())
    with workspace() as ws, _environment_value("cancel-secret"), mock.patch(
        "urllib.request.urlopen"
    ) as urlopen:
        try:
            WebSearchTool(backend).execute(_call("cancel"), ws.policy, cancel=token)
        except OperationCancelled:
            pass
        else:
            fail("web_search converted cancellation into a failed result")
    if urlopen.called:
        fail("pre-cancelled web_search issued HTTP")
