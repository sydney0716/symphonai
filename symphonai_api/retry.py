"""Shared HTTP retry and error-redaction boundary for real providers."""

from __future__ import annotations

import http.client
import math
import random
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from symphonai_api.call_class import CallClass
from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.providers.base import ProviderError

DEFAULT_MAX_ATTEMPTS = 3
RETRY_BASE_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 8.0
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 529})
OVERLOAD_STATUS_CODES = frozenset({429, 529})
NON_RETRYABLE_SERVER_STATUS_CODES = frozenset({501, 505, 511})
MAX_ERROR_DETAIL_BYTES = 500

_PERMANENT_URL_ERROR_MARKERS = (
    "certificate verify failed",
    "invalid url",
    "no host given",
    "unknown scheme",
    "unknown url type",
    "unsupported url",
)


def redact_secret(text: str, secret: str) -> str:
    """Remove a configured API key from vendor-controlled diagnostic text."""
    if secret:
        return text.replace(secret, "[redacted]")
    return text


def read_with_retry(
    request: urllib.request.Request,
    *,
    timeout: float,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    api_key: str,
    operation: str,
    cancel: CancellationToken | None = None,
    call_class: CallClass = CallClass.FOREGROUND,
) -> bytes:
    """Open ``request`` and return its body, retrying transient failures.

    The already-built Request is reused verbatim on every attempt. ``timeout``
    is one overall deadline covering both requests and backoff sleeps, not a
    fresh timeout granted to each attempt.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")

    deadline = time.monotonic() + timeout

    for attempt in range(1, max_attempts + 1):
        if cancel is not None:
            cancel.raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderError(
                f"{operation} exceeded its {timeout}s overall timeout after "
                f"{_attempt_label(attempt - 1)}"
            )
        try:
            with urllib.request.urlopen(request, timeout=remaining) as response:
                body = response.read()
                if cancel is not None:
                    cancel.raise_if_cancelled()
                return body
        except urllib.error.HTTPError as exc:
            detail = _redacted_error_detail(exc.read(), api_key)
            if exc.code in OVERLOAD_STATUS_CODES and call_class is CallClass.BACKGROUND:
                attempt_text = (
                    "the first attempt" if attempt == 1 else f"attempt {attempt}"
                )
                raise ProviderError(
                    f"{operation} returned HTTP {exc.code} on {attempt_text} "
                    "and did not retry: background calls do not retry provider overload"
                ) from None
            if _is_retryable_status(exc.code) and _wait_before_retry(
                attempt=attempt,
                max_attempts=max_attempts,
                deadline=deadline,
                retry_after=exc.headers.get("Retry-After") if exc.headers else None,
                cancel=cancel,
            ):
                continue
            raise ProviderError(
                f"{operation} returned HTTP {exc.code} after "
                f"{_attempt_label(attempt)}: {detail}"
            ) from None
        except urllib.error.URLError as exc:
            reason = redact_secret(str(exc.reason), api_key)
            if _is_retryable_url_error(exc) and _wait_before_retry(
                attempt=attempt,
                max_attempts=max_attempts,
                deadline=deadline,
                retry_after=None,
                cancel=cancel,
            ):
                continue
            raise ProviderError(
                f"{operation} request failed after {_attempt_label(attempt)}: {reason}"
            ) from None
        except TimeoutError:
            if _wait_before_retry(
                attempt=attempt,
                max_attempts=max_attempts,
                deadline=deadline,
                retry_after=None,
                cancel=cancel,
            ):
                continue
            raise ProviderError(
                f"{operation} request timed out after {_attempt_label(attempt)} "
                f"(overall timeout {timeout}s)"
            ) from None
        except http.client.IncompleteRead as exc:
            reason = redact_secret(str(exc), api_key)
            if _wait_before_retry(
                attempt=attempt,
                max_attempts=max_attempts,
                deadline=deadline,
                retry_after=None,
                cancel=cancel,
            ):
                continue
            raise ProviderError(
                f"{operation} response ended early after {_attempt_label(attempt)}: {reason}"
            ) from None
        except ConnectionError as exc:
            reason = redact_secret(str(exc), api_key)
            if _wait_before_retry(
                attempt=attempt,
                max_attempts=max_attempts,
                deadline=deadline,
                retry_after=None,
                cancel=cancel,
            ):
                continue
            raise ProviderError(
                f"{operation} connection failed after {_attempt_label(attempt)}: {reason}"
            ) from None

    raise AssertionError("retry loop exhausted without returning or raising")


def _redacted_error_detail(raw: bytes, api_key: str) -> str:
    redacted = redact_secret(raw.decode("utf-8", errors="replace"), api_key)
    truncated = redacted.encode("utf-8")[:MAX_ERROR_DETAIL_BYTES]
    return truncated.decode("utf-8", errors="ignore")


def _is_retryable_status(status: int) -> bool:
    return status in RETRYABLE_STATUS_CODES or (
        500 <= status <= 599 and status not in NON_RETRYABLE_SERVER_STATUS_CODES
    )


def _is_retryable_url_error(exc: urllib.error.URLError) -> bool:
    reason = exc.reason
    if isinstance(reason, (ssl.SSLCertVerificationError, ssl.CertificateError, ValueError, TypeError)):
        return False
    lowered = str(reason).lower()
    if any(marker in lowered for marker in _PERMANENT_URL_ERROR_MARKERS):
        return False
    # DNS failures and other OSError reasons are ambiguous: outages and local
    # resolver failures are commonly transient, so retry them within the same
    # overall deadline rather than classifying them as permanent.
    return True


def _wait_before_retry(
    *,
    attempt: int,
    max_attempts: int,
    deadline: float,
    retry_after: str | None,
    cancel: CancellationToken | None = None,
) -> bool:
    if attempt >= max_attempts:
        return False

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False

    hint = _retry_after_delay(retry_after) if retry_after is not None else None
    delay = hint if hint is not None else _exponential_delay(attempt)
    delay = min(delay, MAX_RETRY_DELAY_SECONDS)
    if delay >= remaining:
        return False
    if delay > 0:
        if cancel is None:
            time.sleep(delay)
        elif cancel.wait(delay):
            raise OperationCancelled
    return time.monotonic() < deadline


def _retry_after_delay(raw: str) -> float | None:
    value = raw.strip()
    try:
        numeric = float(value)
    except ValueError:
        numeric = None
    if numeric is not None and math.isfinite(numeric) and numeric >= 0:
        return numeric

    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at is None:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    delay = (retry_at.astimezone(timezone.utc) - _utc_now()).total_seconds()
    return delay if delay > 0 else None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _exponential_delay(attempt: int) -> float:
    exponential = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
    return exponential + random.uniform(0.0, exponential * 0.25)


def _attempt_label(attempt: int) -> str:
    return f"{attempt} attempt" if attempt == 1 else f"{attempt} attempts"
