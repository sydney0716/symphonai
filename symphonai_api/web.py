"""Permission-gated outbound HTTP GET and lightweight HTML extraction."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.message import Message as HeaderMessage
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from symphonai_api.cancellation import CancellationToken
from symphonai_api.permissions import PermissionPolicy


MAX_FETCH_BYTES = 5_000_000
MAX_REDIRECTS = 5
DEFAULT_FETCH_TIMEOUT_SECONDS = 15.0
# Split by how each entry is matched. A single tuple sliced by position --
# which is how this started -- makes the order load-bearing without saying so,
# and appending one entry in the wrong group silently changes what counts as
# textual.
TEXTUAL_CONTENT_TYPE_PREFIXES = ("text/",)
TEXTUAL_CONTENT_TYPES_EXACT = (
    "application/json",
    "application/xml",
    "application/xhtml+xml",
)
TEXTUAL_CONTENT_TYPE_SUFFIXES = ("+json", "+xml")
TEXTUAL_CONTENT_TYPES = (
    *TEXTUAL_CONTENT_TYPE_PREFIXES,
    *TEXTUAL_CONTENT_TYPES_EXACT,
    *TEXTUAL_CONTENT_TYPE_SUFFIXES,
)

_USER_AGENT = "SymphonAI-web-fetch/0.1"
_ACCEPT = "text/html, text/plain, application/json, application/xml, application/xhtml+xml"
_DROP_TAGS = frozenset({"script", "style", "head", "noscript", "template"})
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)


class WebFetchError(Exception):
    """A fetch failed. Never carries a header value or a query string."""


class _RedirectBlocked(WebFetchError):
    def __init__(self, location: str, status: int) -> None:
        super().__init__("redirect requires a fresh permission decision")
        self.location = location
        self.status = status


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        raise _RedirectBlocked(newurl, code)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._drop_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag in _DROP_TAGS:
            self._drop_depth += 1
        elif self._drop_depth == 0 and tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if self._drop_depth == 0 and tag.casefold() in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in _DROP_TAGS:
            self._drop_depth = max(0, self._drop_depth - 1)
        elif self._drop_depth == 0 and tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._drop_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        text = "".join(self._chunks).replace("\r", "\n")
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return re.sub(r"\n+", "\n", text).strip()


@dataclass(frozen=True)
class FetchedPage:
    url: str
    status: int
    content_type: str
    text: str
    truncated: bool


def _header_value(response: object, name: str) -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    value = headers.get(name, "")
    return value if isinstance(value, str) else str(value)


def _content_type_and_charset(response: object) -> tuple[str, str]:
    raw_content_type = _header_value(response, "Content-Type")
    header = HeaderMessage()
    header["Content-Type"] = raw_content_type
    content_type = header.get_content_type().casefold()
    charset = header.get_content_charset() or "utf-8"
    return content_type, charset


def _is_textual(content_type: str) -> bool:
    return (
        content_type.startswith(TEXTUAL_CONTENT_TYPE_PREFIXES)
        or content_type in TEXTUAL_CONTENT_TYPES_EXACT
        or content_type.endswith(TEXTUAL_CONTENT_TYPE_SUFFIXES)
    )


def _extract_html(text: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(text)
    parser.close()
    return parser.text()


def fetch_url(
    url: str,
    policy: PermissionPolicy,
    *,
    cancel: CancellationToken | None = None,
    timeout: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
) -> FetchedPage:
    """Fetch one permission-approved textual page with bounded HTTP GETs."""

    current_url = url
    redirects = 0
    opener = urllib.request.build_opener(_NoRedirectHandler())

    while True:
        if cancel is not None:
            cancel.raise_if_cancelled()
        decision = policy.check_fetch(current_url)
        if not decision.allowed:
            raise WebFetchError(decision.reason)

        request = urllib.request.Request(
            current_url,
            headers={"User-Agent": _USER_AGENT, "Accept": _ACCEPT},
            method="GET",
        )
        host = (urlsplit(current_url).hostname or "unknown host").casefold().rstrip(".")
        try:
            response_context = opener.open(request, timeout=timeout)
            with response_context as response:
                content_type, charset = _content_type_and_charset(response)
                if not _is_textual(content_type):
                    raise WebFetchError(
                        f"GET from {host} refused non-textual content type {content_type}"
                    )
                body = response.read(MAX_FETCH_BYTES + 1)
                content_length = _header_value(response, "Content-Length")
                try:
                    declared_length = int(content_length)
                except ValueError:
                    declared_length = None
                if len(body) > MAX_FETCH_BYTES or (
                    declared_length is not None and declared_length > MAX_FETCH_BYTES
                ):
                    detail = (
                        f"; Content-Length {content_length}"
                        if content_length
                        else ""
                    )
                    raise WebFetchError(
                        f"GET from {host} exceeds the {MAX_FETCH_BYTES} byte cap{detail}"
                    )
                if cancel is not None:
                    cancel.raise_if_cancelled()
                text = body.decode(charset, errors="replace")
                if content_type in ("text/html", "application/xhtml+xml"):
                    text = _extract_html(text)
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                return FetchedPage(
                    url=current_url,
                    status=int(status),
                    content_type=content_type,
                    text=text,
                    truncated=False,
                )
        except _RedirectBlocked as redirect:
            if redirects >= MAX_REDIRECTS:
                raise WebFetchError(
                    f"GET from {host} exceeded the {MAX_REDIRECTS} redirect limit"
                ) from None
            current_url = urljoin(current_url, redirect.location)
            redirects += 1
            if cancel is not None:
                cancel.raise_if_cancelled()
        except urllib.error.HTTPError as exc:
            raise WebFetchError(
                f"GET from {host} failed with HTTP {exc.code}"
            ) from None
        except urllib.error.URLError as exc:
            raise WebFetchError(
                f"GET from {host} failed ({type(exc.reason).__name__})"
            ) from None
        except (LookupError, OSError, UnicodeError) as exc:
            raise WebFetchError(
                f"GET from {host} failed ({type(exc).__name__})"
            ) from None
