"""Fixture-free checks for retry."""

from __future__ import annotations

import http.client
import io
import json
import os
import ssl
import threading
import time
import unittest.mock as mock
import urllib.error
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from orchestra_api.cancellation import CancellationToken, OperationCancelled
from orchestra_api.model_discovery import list_models
from orchestra_api.models import Message, ModelRequest, Role
from orchestra_api.providers.anthropic_provider import API_KEY_ENV_VAR as ANTHROPIC_API_KEY_ENV_VAR
from orchestra_api.providers.anthropic_provider import AnthropicProvider
from orchestra_api.providers.base import ProviderError
from orchestra_api.providers.gemini_provider import API_KEY_ENV_VAR as GEMINI_API_KEY_ENV_VAR
from orchestra_api.providers.gemini_provider import GeminiProvider
from orchestra_api.providers.openai_compatible import OpenAICompatibleProvider
from orchestra_api.providers.openai_provider import API_KEY_ENV_VAR, OpenAIProvider
from orchestra_api.retry import read_with_retry
from scripts.checks.harness import check, fail


class _FakeHttpResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

def _http_error(
    status: int,
    body: str,
    *,
    retry_after: str | None = None,
) -> urllib.error.HTTPError:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError(
        url="https://mock.invalid",
        code=status,
        msg="mock error",
        hdrs=headers,
        fp=io.BytesIO(body.encode("utf-8")),
    )


def _openai_success(content: str) -> _FakeHttpResponse:
    payload = {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }
    return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

@check("retry.backoff_wakes_promptly")
def check_retry_backoff_wakes_promptly() -> None:
    retry_request = urllib.request.Request("https://mock.invalid/test")
    backoff_token = CancellationToken()
    timer = threading.Timer(0.02, backoff_token.cancel)
    retryable_error = urllib.error.HTTPError(
        retry_request.full_url,
        503,
        "unavailable",
        {},
        io.BytesIO(b"retry later"),
    )
    started = time.monotonic()
    timer.start()
    try:
        with mock.patch("urllib.request.urlopen", side_effect=retryable_error):
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                try:
                    read_with_retry(
                        retry_request,
                        timeout=2.0,
                        api_key="test-key",
                        operation="cancel test",
                        cancel=backoff_token,
                    )
                except OperationCancelled as exc:
                    if isinstance(exc, ProviderError):
                        fail("OperationCancelled was wrapped as ProviderError")
                else:
                    fail("interruptible retry backoff did not raise OperationCancelled")
                if sleep_mock.called:
                    fail("cancellable retry backoff called time.sleep")
    finally:
        timer.cancel()
        timer.join()
    if time.monotonic() - started >= 0.3:
        fail("retry backoff cancellation did not wake promptly")

@check("retry.cancel_none_uses_time_sleep")
def check_retry_cancel_none_uses_time_sleep() -> None:
    retry_request = urllib.request.Request("https://mock.invalid/test")
    retryable_then_success = urllib.error.HTTPError(
        retry_request.full_url,
        503,
        "unavailable",
        {},
        io.BytesIO(b"retry later"),
    )
    with mock.patch(
        "urllib.request.urlopen",
        side_effect=[retryable_then_success, _FakeHttpResponse(b"ok")],
    ):
        with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
            with mock.patch("orchestra_api.retry.random.uniform", return_value=0.0):
                raw = read_with_retry(
                    retry_request,
                    timeout=2.0,
                    api_key="test-key",
                    operation="none token test",
                    cancel=None,
                )
    if raw != b"ok":
        fail(f"cancel=None retry returned unexpected payload: {raw!r}")
    sleep_mock.assert_called_once_with(0.5)

@check("retry.http_503_succeeds")
def check_retry_http_503_succeeds() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    def _headers(request) -> dict[str, str]:  # noqa: ANN001
        return {name.lower(): value for name, value in request.header_items()}

    def _http_error(
        status: int,
        body: str,
        *,
        retry_after: str | None = None,
    ) -> urllib.error.HTTPError:
        headers = {"Retry-After": retry_after} if retry_after is not None else {}
        return urllib.error.HTTPError(
            url="https://mock.invalid",
            code=status,
            msg="mock error",
            hdrs=headers,
            fp=io.BytesIO(body.encode("utf-8")),
        )

    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])

    def _openai_success(content: str) -> _FakeHttpResponse:
        payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
        return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

    # -- shared HTTP retry: Gemini retries a transient 503 and parses
    # the successful response without rebuilding the vendor request. --
    gemini_retry_key = "gemini-retry-test-key-do-not-use"
    gemini_success = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "retry worked"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {},
    }
    with mock.patch.dict(os.environ, {GEMINI_API_KEY_ENV_VAR: gemini_retry_key}):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[
                _http_error(503, '{"error":"high demand"}'),
                _FakeHttpResponse(json.dumps(gemini_success).encode("utf-8")),
            ],
        ) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                with mock.patch("orchestra_api.retry.random.uniform", return_value=0.0):
                    retry_response = GeminiProvider().create_response(basic_request)
    if retry_response.message.text != "retry worked" or urlopen_mock.call_count != 2:
        fail("expected Gemini HTTP 503 to retry once and then succeed")
    if urlopen_mock.call_args_list[0].args[0] is not urlopen_mock.call_args_list[1].args[0]:
        fail("expected retry transport to resend the same prepared Request object")
    sleep_mock.assert_called_once_with(0.5)

@check("retry.timeout_error_succeeds")
def check_retry_timeout_error_succeeds() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    # -- the motivating transport timeout retries under the same overall
    # deadline and succeeds without any real sleeping. --
    transport_retry_key = "transport-retry-test-key-do-not-use"
    with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: transport_retry_key}):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[TimeoutError("mock timeout"), _openai_success("timeout recovered")],
        ) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                with mock.patch("orchestra_api.retry.random.uniform", return_value=0.0):
                    timeout_response = OpenAIProvider().create_response(basic_request)
    if timeout_response.message.text != "timeout recovered" or urlopen_mock.call_count != 2:
        fail("expected TimeoutError to retry once and then succeed")
    sleep_mock.assert_called_once_with(0.5)

@check("retry.transient_urlerror_succeeds")
def check_retry_transient_urlerror_succeeds() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    transport_retry_key = "transport-retry-test-key-do-not-use"
    # -- ambiguous connection-level URLErrors retry because resolver and
    # peer-reset failures are often transient. --
    with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: transport_retry_key}):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[
                urllib.error.URLError(ConnectionResetError("peer reset")),
                _openai_success("URL error recovered"),
            ],
        ) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                with mock.patch("orchestra_api.retry.random.uniform", return_value=0.0):
                    url_error_response = OpenAIProvider().create_response(basic_request)
    if url_error_response.message.text != "URL error recovered" or urlopen_mock.call_count != 2:
        fail("expected transient URLError to retry once and then succeed")
    sleep_mock.assert_called_once_with(0.5)

@check("retry.certificate_urlerror_fails_immediately")
def check_retry_certificate_urlerror_fails_immediately() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    transport_retry_key = "transport-retry-test-key-do-not-use"
    # -- certificate verification is a permanent URLError cause and must
    # not consume retries or backoff time. --
    certificate_error = ssl.SSLCertVerificationError(1, "certificate verify failed")
    with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: transport_retry_key}):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(certificate_error),
        ) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                try:
                    OpenAIProvider().create_response(basic_request)
                except ProviderError:
                    pass
                else:
                    fail("expected certificate verification URLError to fail immediately")
    if urlopen_mock.call_count != 1:
        fail(f"permanent URLError must make one attempt, got {urlopen_mock.call_count}")
    sleep_mock.assert_not_called()

@check("retry.truncated_body_succeeds")
def check_retry_truncated_body_succeeds() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    transport_retry_key = "transport-retry-test-key-do-not-use"
    # -- an incomplete response body is transient and should be retried. --
    with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: transport_retry_key}):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[
                http.client.IncompleteRead(b"partial", 20),
                _openai_success("incomplete read recovered"),
            ],
        ) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                with mock.patch("orchestra_api.retry.random.uniform", return_value=0.0):
                    incomplete_response = OpenAIProvider().create_response(basic_request)
    if incomplete_response.message.text != "incomplete read recovered" or urlopen_mock.call_count != 2:
        fail("expected IncompleteRead to retry once and then succeed")
    sleep_mock.assert_called_once_with(0.5)

@check("retry.permanent_http_statuses_fail_immediately")
def check_retry_permanent_http_statuses_fail_immediately() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    transport_retry_key = "transport-retry-test-key-do-not-use"
    # -- protocol/configuration 5xx statuses are permanent despite being
    # in the server-error range. --
    for permanent_status in (501, 505, 511):
        with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: transport_retry_key}):
            with mock.patch(
                "urllib.request.urlopen",
                side_effect=_http_error(permanent_status, '{"error":"permanent"}'),
            ) as urlopen_mock:
                with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                    try:
                        OpenAIProvider().create_response(basic_request)
                    except ProviderError:
                        pass
                    else:
                        fail(f"expected HTTP {permanent_status} to fail immediately")
        if urlopen_mock.call_count != 1:
            fail(f"HTTP {permanent_status} must make one attempt, got {urlopen_mock.call_count}")
        sleep_mock.assert_not_called()

@check("retry.overall_deadline")
def check_retry_overall_deadline() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    transport_retry_key = "transport-retry-test-key-do-not-use"
    # -- request time and sleep time share one deadline. Simulate the
    # first request consuming 29.75s of a 30s budget: an 8s wait cannot
    # fit, so no second attempt or sleep is allowed. --
    with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: transport_retry_key}):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=_http_error(503, '{"error":"late failure"}', retry_after="8"),
        ) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                deadline_clock = iter([100.0, 100.0])
                with mock.patch(
                    "orchestra_api.retry.time.monotonic",
                    side_effect=lambda: next(deadline_clock, 129.75),
                ):
                    try:
                        OpenAIProvider().create_response(basic_request)
                    except ProviderError as exc:
                        deadline_message = str(exc)
                    else:
                        fail("expected exhausted overall deadline to stop before retry")
    if urlopen_mock.call_count != 1 or "1 attempt" not in deadline_message:
        fail(
            "overall deadline must stop after the request consumes its budget, "
            f"got {urlopen_mock.call_count} attempts and {deadline_message!r}"
        )
    sleep_mock.assert_not_called()

@check("retry.http_400_fails_immediately")
def check_retry_http_400_fails_immediately() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    # -- permanent 4xx failures must fail immediately, even when the
    # provider allows more attempts. --
    anthropic_retry_key = "anthropic-retry-test-key-do-not-use"

    def _always_400(request, timeout=None):  # noqa: ANN001
        raise _http_error(400, '{"error":"bad request"}')

    with mock.patch.dict(os.environ, {ANTHROPIC_API_KEY_ENV_VAR: anthropic_retry_key}):
        with mock.patch("urllib.request.urlopen", side_effect=_always_400) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                try:
                    AnthropicProvider().create_response(basic_request)
                except ProviderError:
                    pass
                else:
                    fail("expected Anthropic HTTP 400 to raise ProviderError")
    if urlopen_mock.call_count != 1:
        fail(f"HTTP 400 must make exactly one attempt, got {urlopen_mock.call_count}")
    sleep_mock.assert_not_called()

@check("retry.exhausted_transient_failures")
def check_retry_exhausted_transient_failures() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    # -- exhausting the default three attempts must report that count. --
    openai_retry_key = "openai-retry-test-key-do-not-use"

    def _always_503(request, timeout=None):  # noqa: ANN001
        raise _http_error(503, '{"error":"still overloaded"}')

    with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: openai_retry_key}):
        with mock.patch("urllib.request.urlopen", side_effect=_always_503) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                with mock.patch("orchestra_api.retry.random.uniform", return_value=0.0):
                    try:
                        OpenAIProvider().create_response(basic_request)
                    except ProviderError as exc:
                        exhausted_message = str(exc)
                    else:
                        fail("expected repeated OpenAI HTTP 503 responses to exhaust retries")
    if urlopen_mock.call_count != 3 or sleep_mock.call_count != 2:
        fail(
            "expected exhausted retries to make three attempts and two patched sleeps, "
            f"got {urlopen_mock.call_count} attempts and {sleep_mock.call_count} sleeps"
        )
    if "3 attempts" not in exhausted_message:
        fail(f"exhausted retry error must name the attempt count: {exhausted_message!r}")

@check("retry.numeric_retry_after")
def check_retry_numeric_retry_after() -> None:
    # -- model discovery shares the retry transport and honours a
    # numeric Retry-After value instead of exponential backoff. --
    compatible_key_env = "ORCHESTRA_RETRY_AFTER_TEST_KEY"
    compatible_retry_key = "compatible-retry-test-key-do-not-use"
    compatible_provider = OpenAICompatibleProvider(
        api_key_env_var=compatible_key_env,
        base_url="https://mock.invalid/v1",
        max_attempts=2,
    )
    with mock.patch.dict(os.environ, {compatible_key_env: compatible_retry_key}):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[
                _http_error(429, '{"error":"slow down"}', retry_after="2"),
                _FakeHttpResponse(b'{"data":[{"id":"retry-after-model"}]}'),
            ],
        ) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                retry_after_models = list_models(compatible_provider)
    if retry_after_models != ["retry-after-model"] or urlopen_mock.call_count != 2:
        fail("expected model discovery to retry HTTP 429 and return the model listing")
    sleep_mock.assert_called_once_with(2.0)

@check("retry.http_date_retry_after")
def check_retry_http_date_retry_after() -> None:
    compatible_key_env = "ORCHESTRA_RETRY_AFTER_TEST_KEY"
    compatible_retry_key = "compatible-retry-test-key-do-not-use"
    compatible_provider = OpenAICompatibleProvider(
        api_key_env_var=compatible_key_env,
        base_url="https://mock.invalid/v1",
        max_attempts=2,
    )
    # -- RFC 9110 also permits Retry-After as an HTTP-date. --
    retry_now = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
    retry_at = format_datetime(retry_now + timedelta(seconds=4), usegmt=True)
    with mock.patch.dict(os.environ, {compatible_key_env: compatible_retry_key}):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[
                _http_error(429, '{"error":"wait until date"}', retry_after=retry_at),
                _FakeHttpResponse(b'{"data":[{"id":"http-date-model"}]}'),
            ],
        ) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                with mock.patch("orchestra_api.retry._utc_now", return_value=retry_now):
                    http_date_models = list_models(compatible_provider)
    if http_date_models != ["http-date-model"] or urlopen_mock.call_count != 2:
        fail("expected HTTP-date Retry-After to delay and retry model discovery")
    sleep_mock.assert_called_once_with(4.0)

@check("retry.malformed_retry_after")
def check_retry_malformed_retry_after() -> None:
    compatible_key_env = "ORCHESTRA_RETRY_AFTER_TEST_KEY"
    compatible_retry_key = "compatible-retry-test-key-do-not-use"
    compatible_provider = OpenAICompatibleProvider(
        api_key_env_var=compatible_key_env,
        base_url="https://mock.invalid/v1",
        max_attempts=2,
    )
    # -- malformed Retry-After is not a zero-second hint; it falls back
    # to the normal exponential delay. --
    with mock.patch.dict(os.environ, {compatible_key_env: compatible_retry_key}):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[
                _http_error(429, '{"error":"bad hint"}', retry_after="not-a-date"),
                _FakeHttpResponse(b'{"data":[{"id":"fallback-model"}]}'),
            ],
        ) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                with mock.patch("orchestra_api.retry.random.uniform", return_value=0.0):
                    fallback_models = list_models(compatible_provider)
    if fallback_models != ["fallback-model"] or urlopen_mock.call_count != 2:
        fail("expected malformed Retry-After to fall back and retry")
    sleep_mock.assert_called_once_with(0.5)

@check("retry.retry_after_capped")
def check_retry_retry_after_capped() -> None:
    compatible_key_env = "ORCHESTRA_RETRY_AFTER_TEST_KEY"
    compatible_retry_key = "compatible-retry-test-key-do-not-use"
    compatible_provider = OpenAICompatibleProvider(
        api_key_env_var=compatible_key_env,
        base_url="https://mock.invalid/v1",
        max_attempts=2,
    )
    # -- an excessive Retry-After is bounded by the per-sleep cap. --
    with mock.patch.dict(os.environ, {compatible_key_env: compatible_retry_key}):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[
                _http_error(429, '{"error":"long hint"}', retry_after="120"),
                _FakeHttpResponse(b'{"data":[{"id":"capped-model"}]}'),
            ],
        ) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                capped_models = list_models(compatible_provider)
    if capped_models != ["capped-model"] or urlopen_mock.call_count != 2:
        fail("expected capped Retry-After to retry model discovery")
    sleep_mock.assert_called_once_with(8.0)

@check("retry.keys_redacted")
def check_retry_keys_redacted() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    # -- vendor-controlled HTTP error bodies pass through the same key
    # redaction boundary for OpenAI-compatible providers too. --
    redaction_key_env = "ORCHESTRA_RETRY_REDACTION_TEST_KEY"
    redaction_key = "secret-key-that-must-never-escape"
    redaction_provider = OpenAICompatibleProvider(
        api_key_env_var=redaction_key_env,
        base_url="https://mock.invalid/v1",
    )
    with mock.patch.dict(os.environ, {redaction_key_env: redaction_key}):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=_http_error(401, f'{{"error":"bad key {redaction_key}"}}'),
        ) as urlopen_mock:
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                try:
                    redaction_provider.create_response(basic_request)
                except ProviderError as exc:
                    redacted_message = str(exc)
                else:
                    fail("expected OpenAI-compatible HTTP 401 to raise ProviderError")
    if redaction_key in redacted_message or "[redacted]" not in redacted_message:
        fail(f"provider error body did not redact the API key: {redacted_message!r}")
    if urlopen_mock.call_count != 1:
        fail(f"HTTP 401 must not retry, got {urlopen_mock.call_count} attempts")
    sleep_mock.assert_not_called()

@check("retry.key_prefix_boundary_redacted")
def check_retry_key_prefix_boundary_redacted() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    # -- redact before the 500-byte diagnostic cap. With truncation-first,
    # the first ten characters of this key would survive at the boundary. --
    boundary_key_env = "ORCHESTRA_RETRY_BOUNDARY_KEY"
    boundary_key = "BOUNDARY_SECRET_PREFIX_0123456789"
    boundary_prefix = boundary_key[:10]
    boundary_provider = OpenAICompatibleProvider(
        api_key_env_var=boundary_key_env,
        base_url="https://mock.invalid/v1",
    )
    boundary_body = ("x" * 490) + boundary_key
    with mock.patch.dict(os.environ, {boundary_key_env: boundary_key}):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=_http_error(401, boundary_body),
        ):
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                try:
                    boundary_provider.create_response(basic_request)
                except ProviderError as exc:
                    boundary_message = str(exc)
                else:
                    fail("expected boundary redaction request to raise ProviderError")
    if boundary_prefix in boundary_message:
        fail(
            "API-key prefix survived redact-before-truncate boundary: "
            f"{boundary_prefix!r} in {boundary_message!r}"
        )
    if "[redacted]" not in boundary_message:
        fail(f"expected redaction marker in boundary error: {boundary_message!r}")
    sleep_mock.assert_not_called()
