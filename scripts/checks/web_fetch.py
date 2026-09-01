"""Registered checks for GET-only, permission-gated web fetching."""

from __future__ import annotations

import urllib.request
import unittest.mock as mock
from dataclasses import dataclass
from email.message import Message as HeaderMessage

from symphonai_api.permissions import DenialReason, PermissionPolicy
from symphonai_api.runner import standard_tool_registry
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_api.tools.web_fetch import WebFetchTool
from symphonai_api.web import MAX_FETCH_BYTES, MAX_REDIRECTS, WebFetchError, fetch_url
from symphonai_api.web_domains import preapproved_domains
from symphonai_api.models import ToolCall
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


@dataclass(frozen=True)
class _Redirect:
    location: str


class _Response:
    def __init__(
        self,
        body: bytes = b"ok",
        *,
        status: int = 200,
        content_type: str = "text/plain; charset=utf-8",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._body = body
        self.status = status
        self.headers = HeaderMessage()
        self.headers["Content-Type"] = content_type
        for name, value in (extra_headers or {}).items():
            self.headers[name] = value
        self.read_limits: list[int] = []

    def read(self, limit: int = -1) -> bytes:
        self.read_limits.append(limit)
        return self._body if limit < 0 else self._body[:limit]

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _RoutingOpener:
    def __init__(self, handler: object, routes: dict[str, _Response | _Redirect]) -> None:
        self.handler = handler
        self.routes = routes
        self.requests: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, *, timeout: float) -> _Response:
        self.requests.append(request)
        outcome = self.routes.get(request.full_url)
        if outcome is None:
            raise RuntimeError(f"unmocked network request: {request.full_url}")
        if isinstance(outcome, _Response):
            return outcome
        headers = HeaderMessage()
        headers["Location"] = outcome.location
        redirected = self.handler.redirect_request(
            request, None, 302, "Found", headers, outcome.location
        )
        if redirected is None:
            raise RuntimeError("redirect handler neither refused nor returned a request")
        return self.open(redirected, timeout=timeout)


def _patched_opener(routes: dict[str, _Response | _Redirect]):
    opened: list[_RoutingOpener] = []

    def build(handler: object) -> _RoutingOpener:
        opener = _RoutingOpener(handler, routes)
        opened.append(opener)
        return opener

    return mock.patch("symphonai_api.web.urllib.request.build_opener", side_effect=build), opened


def _call(url: str, index: int = 0) -> ToolCall:
    return ToolCall(
        id=f"fetch-{index}", name="web_fetch", arguments={"url": url}
    )


@check("web_fetch.get_only_schema")
def check_get_only_schema() -> None:
    url = "https://docs.python.org/3/"
    patcher, opened = _patched_opener({url: _Response()})
    with workspace() as ws, patcher:
        result = WebFetchTool().execute(_call(url), ws.policy)
    schema = WebFetchTool().parameters
    if not result.ok or set(schema["properties"]) != {"url"} or schema["required"] != ["url"]:
        fail(f"web_fetch schema is not URL-only: {schema!r}, result={result!r}")
    request = opened[0].requests[0]
    if request.get_method() != "GET" or request.data is not None:
        fail(f"web_fetch did not issue a bodyless GET: {request!r}")


@check("web_fetch.preapproved_no_prompt")
def check_preapproved_no_prompt() -> None:
    url = "https://DOCS.PYTHON.ORG./3/"
    requests: list[object] = []
    routes = {url: _Response()}
    patcher, _ = _patched_opener(routes)
    with workspace() as ws, patcher:
        for mode in ("auto", "prompt"):
            policy = PermissionPolicy(
                repo_root=ws.root,
                mode=mode,
                approval_callback=lambda request: requests.append(request) or False,
            )
            if not WebFetchTool().execute(_call(url), policy).ok:
                fail(f"preapproved domain was denied in {mode} mode")
    if requests:
        fail(f"preapproved domain prompted for approval: {requests!r}")


@check("web_fetch.unapproved_auto_denied")
def check_unapproved_auto_denied() -> None:
    url = "https://example.com/private"
    patcher, opened = _patched_opener({url: _Response()})
    with workspace() as ws, patcher:
        policy = PermissionPolicy(repo_root=ws.root)
        decision = policy.check_fetch(url)
        result = WebFetchTool().execute(_call(url), policy)
    if decision.denial is not DenialReason.DOMAIN_NOT_APPROVED or result.ok:
        fail(f"unapproved auto fetch was not denied: {decision!r}, {result!r}")
    if opened[0].requests:
        fail(f"denied fetch issued HTTP: {opened[0].requests!r}")


@check("web_fetch.unapproved_prompt_asks")
def check_unapproved_prompt_asks() -> None:
    url = "https://example.com/page"
    seen: list[object] = []
    patcher, opened = _patched_opener({url: _Response()})
    with workspace() as ws, patcher:
        denied = PermissionPolicy(
            repo_root=ws.root,
            mode="prompt",
            approval_callback=lambda request: seen.append(request) or False,
        )
        denied_result = WebFetchTool().execute(_call(url, 1), denied)
        allowed = PermissionPolicy(
            repo_root=ws.root,
            mode="prompt",
            approval_callback=lambda request: seen.append(request) or True,
        )
        allowed_result = WebFetchTool().execute(_call(url, 2), allowed)
    if denied_result.ok or not allowed_result.ok or len(seen) != 2:
        fail(f"prompt answers were not honored: {denied_result!r}, {allowed_result!r}, {seen!r}")
    if any(getattr(request, "operation", None) != "web_fetch" for request in seen):
        fail(f"approval request used the wrong operation: {seen!r}")
    if len(opened[0].requests) != 0 or len(opened[1].requests) != 1:
        fail("prompt denial issued HTTP or approval failed to issue HTTP")


@check("web_fetch.scheme_denied")
def check_scheme_denied() -> None:
    with workspace() as ws:
        for url in ("file:///tmp/a", "ftp://example.com/a", "data:text/plain,x"):
            decision = PermissionPolicy(repo_root=ws.root).check_fetch(url)
            if decision.denial is not DenialReason.UNSUPPORTED_SCHEME:
                fail(f"unsupported scheme was not denied: {url!r}, {decision!r}")


@check("web_fetch.blocked_hosts")
def check_blocked_hosts() -> None:
    urls = (
        "http://localhost/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://10.0.0.1/",
        "http://169.254.169.254/",
    )
    with workspace() as ws:
        policy = PermissionPolicy(repo_root=ws.root, fetch_enabled=True)
        for url in urls:
            if policy.check_fetch(url).denial is not DenialReason.BLOCKED_HOST:
                fail(f"blocked host was allowed: {url!r}")


@check("web_fetch.redirect_offsite_refused")
def check_redirect_offsite_refused() -> None:
    start = "https://docs.python.org/start"
    target = "https://example.com/secret?token=must-not-leak"
    patcher, opened = _patched_opener(
        {start: _Redirect(target), target: _Response(b"must not fetch")}
    )
    with workspace() as ws, patcher:
        result = WebFetchTool().execute(_call(start), ws.policy)
    if result.ok or "example.com" not in (result.error or "") or "token=" in (result.error or ""):
        fail(f"offsite redirect denial was unsafe or unclear: {result!r}")
    if [request.full_url for request in opened[0].requests] != [start]:
        fail(f"offsite redirect was followed: {opened[0].requests!r}")


@check("web_fetch.redirect_onsite_followed")
def check_redirect_onsite_followed() -> None:
    start = "https://docs.python.org/start"
    target = "https://peps.python.org/pep-0008/"
    patcher, opened = _patched_opener({start: _Redirect(target), target: _Response(b"PEP 8")})
    with workspace() as ws, patcher:
        page = fetch_url(start, ws.policy)
    if page.url != target or page.text != "PEP 8":
        fail(f"approved redirect did not return its final page: {page!r}")
    if len(opened[0].requests) != 2:
        fail(f"approved redirect request count changed: {opened[0].requests!r}")


@check("web_fetch.redirect_limit")
def check_redirect_limit() -> None:
    routes: dict[str, _Response | _Redirect] = {}
    for index in range(MAX_REDIRECTS + 1):
        current = f"https://docs.python.org/{index}"
        routes[current] = _Redirect(f"https://docs.python.org/{index + 1}")
    patcher, _ = _patched_opener(routes)
    with workspace() as ws, patcher:
        try:
            fetch_url("https://docs.python.org/0", ws.policy)
        except WebFetchError as exc:
            if str(MAX_REDIRECTS) not in str(exc):
                fail(f"redirect-limit error did not name the cap: {exc!r}")
        else:
            fail("redirect chain beyond the cap succeeded")


@check("web_fetch.size_cap_refuses")
def check_size_cap_refuses() -> None:
    url = "https://docs.python.org/large"
    response = _Response(b"x" * (MAX_FETCH_BYTES + 1))
    patcher, _ = _patched_opener({url: response})
    with workspace() as ws, patcher:
        result = WebFetchTool().execute(_call(url), ws.policy)
    if result.ok or str(MAX_FETCH_BYTES) not in (result.error or "") or result.content:
        fail(f"oversized response was not wholly refused: {result!r}")
    if response.read_limits != [MAX_FETCH_BYTES + 1]:
        fail(f"response body was not read with a strict bound: {response.read_limits!r}")


@check("web_fetch.content_type_refused")
def check_content_type_refused() -> None:
    secret = "HEADER_SECRET_MUST_NOT_LEAK"
    with workspace() as ws:
        for index, content_type in enumerate(("application/octet-stream", "image/png")):
            url = f"https://docs.python.org/binary/{index}"
            patcher, _ = _patched_opener(
                {url: _Response(b"binary", content_type=content_type, extra_headers={"X-Secret": secret})}
            )
            with patcher:
                result = WebFetchTool().execute(_call(url, index), ws.policy)
            if result.ok or content_type not in (result.error or "") or secret in (result.error or ""):
                fail(f"non-text response was not safely refused: {result!r}")


@check("web_fetch.html_to_text")
def check_html_to_text() -> None:
    url = "https://docs.python.org/html"
    html = b"<html><head><title>drop</title></head><style>bad{}</style><p>A &amp; B</p><script>secret()</script><div>Next</div></html>"
    patcher, _ = _patched_opener({url: _Response(html, content_type="text/html")})
    with workspace() as ws, patcher:
        page = fetch_url(url, ws.policy)
    if "A & B" not in page.text or "Next" not in page.text:
        fail(f"HTML text or entities were lost: {page.text!r}")
    if any(hidden in page.text for hidden in ("drop", "bad", "secret")):
        fail(f"excluded HTML content leaked into text: {page.text!r}")


@check("web_fetch.domain_table_missing_is_empty")
def check_domain_table_missing_is_empty() -> None:
    class _Resource:
        def __init__(self, content: str | None) -> None:
            self.content = content

        def joinpath(self, name: str) -> "_Resource":
            return self

        def read_text(self, *, encoding: str) -> str:
            if self.content is None:
                raise FileNotFoundError
            return self.content

    try:
        for content in (None, "{broken"):
            preapproved_domains.cache_clear()
            with mock.patch("symphonai_api.web_domains.resources.files", return_value=_Resource(content)):
                if preapproved_domains() != ():
                    fail(f"missing or malformed domain table was not fail-closed: {content!r}")
    finally:
        preapproved_domains.cache_clear()
    if "docs.python.org" not in preapproved_domains():
        fail("shipped domain table did not load after fail-closed cases")


@check("web_fetch.subdomain_not_inherited")
def check_subdomain_not_inherited() -> None:
    with workspace() as ws, mock.patch(
        "symphonai_api.permissions.preapproved_domains",
        return_value=("docs.python.org",),
    ):
        policy = PermissionPolicy(repo_root=ws.root)
        exact = policy.check_fetch("https://DOCS.PYTHON.ORG./")
        child = policy.check_fetch("https://evil.docs.python.org/")
        parent = policy.check_fetch("https://python.org/")
    if not exact.allowed:
        fail("exact normalized preapproved domain was denied")
    if any(decision.denial is not DenialReason.DOMAIN_NOT_APPROVED for decision in (child, parent)):
        fail(f"domain approval leaked by suffix: child={child!r}, parent={parent!r}")


@check("web_fetch.plan_mode_allows_fetch")
def check_plan_mode_allows_fetch() -> None:
    url = "https://docs.python.org/plan"
    patcher, _ = _patched_opener({url: _Response()})
    with workspace() as ws, patcher:
        result = WebFetchTool().execute(
            _call(url), PermissionPolicy(repo_root=ws.root, mode="plan")
        )
    if not result.ok:
        fail(f"plan mode denied a preapproved fetch: {result!r}")


@check("web_fetch.metadata_contract")
def check_metadata_contract() -> None:
    expected = ToolMetadata(
        effect=ToolEffect.READ_ONLY,
        concurrency_safe=True,
        paths=(),
    )
    if WebFetchTool().metadata({"url": "https://example.com"}) != expected:
        fail("web_fetch metadata did not declare read-only, parallel, pathless work")


@check("web_fetch.registry_registration")
def check_registry_registration() -> None:
    registry = standard_tool_registry()
    selected = standard_tool_registry(["web_fetch"])
    if "web_fetch" not in registry or list(registry)[-1] != "web_fetch":
        fail(f"web_fetch was not appended to the registry: {list(registry)!r}")
    if list(selected) != ["web_fetch"] or not isinstance(selected["web_fetch"], WebFetchTool):
        fail(f"web_fetch registry selection failed: {selected!r}")
