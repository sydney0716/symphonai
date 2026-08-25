#!/usr/bin/env python3
"""Smoke test for the orchestra_api runtime, using FakeModelProvider only.

Verifies:
  - a full ApiAgent run through runner.run_task(): a tool-call turn
    (read_file) followed by a final-answer turn
  - the allow path: read_file/list_files/write_file all succeed inside a
    temp dir that is both repo_root and the explicit allowed write scope
  - the deny path: write outside the allowed write scope, a
    forbidden-pattern path (.env), a `..` path-traversal attempt, run_shell
    denied by default, and an always-deny command (rm) denied even when
    explicitly allowlisted
  - regression check: runner.run_task() with a real OpenAIProvider actually
    includes schemas for all four standard tools in its outgoing request
    (via mocked urllib.request.urlopen) -- guards against ApiAgent/runner
    silently never telling a real model any tool exists
  - request-level model overrides reach real-provider wire requests, and
    malformed/non-object HTTP 200 JSON raises ProviderError
  - regression check: model discovery lists OpenAI, Anthropic, and Gemini
    models via mocked urllib.request.urlopen, with the correct listing URL,
    auth header, Anthropic version header, and Gemini generateContent filter
  - regression check: a real GeminiProvider round-trips a functionCall
    thoughtSignature across a two-turn tool call loop (via mocked
    urllib.request.urlopen), because Gemini rejects the second stateless
    request when that signature is omitted
  - shared HTTP retry behaviour with mocked failures and patched sleeps:
    transient HTTP/timeout/URL/incomplete-read success, permanent failures,
    overall deadline, Retry-After variants, and boundary-safe API-key redaction
  - context compaction stays pure and deterministic: under budget is
    untouched, over budget preserves required context, and impossible
    budgets fail clearly before any provider call

No real network call is ever made in this script. FakeModelProvider is
used for every check except the real-provider regression checks above,
which exercise real request-building code against a mocked HTTP layer.
"""

from __future__ import annotations

import http.client
import io
import json
import os
import ssl
import sys
import tempfile
import unittest.mock as mock
import urllib.error
from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.compaction import ContextCompactionError, compact_messages_for_budget  # noqa: E402
from orchestra_api.model_discovery import list_models  # noqa: E402
from orchestra_api.models import (  # noqa: E402
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    TextBlock,
    ToolCall,
    ToolResult,
)
from orchestra_api.permissions import PermissionPolicy  # noqa: E402
from orchestra_api.providers.anthropic_provider import (  # noqa: E402
    ANTHROPIC_VERSION,
    API_KEY_ENV_VAR as ANTHROPIC_API_KEY_ENV_VAR,
)
from orchestra_api.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from orchestra_api.providers.anthropic_provider import _build_request_body as _build_anthropic_body  # noqa: E402
from orchestra_api.providers.base import ProviderError  # noqa: E402
from orchestra_api.providers.fake import FakeModelProvider  # noqa: E402
from orchestra_api.providers.gemini_provider import (  # noqa: E402
    API_KEY_ENV_VAR as GEMINI_API_KEY_ENV_VAR,
)
from orchestra_api.providers.gemini_provider import GeminiProvider  # noqa: E402
from orchestra_api.providers.gemini_provider import _build_request_body as _build_gemini_body  # noqa: E402
from orchestra_api.providers.gemini_provider import _parse_response as _parse_gemini_response  # noqa: E402
from orchestra_api.providers.openai_compatible import OpenAICompatibleProvider  # noqa: E402
from orchestra_api.providers.openai_provider import API_KEY_ENV_VAR, OpenAIProvider  # noqa: E402
from orchestra_api.providers.openai_provider import (  # noqa: E402
    _build_request_body as _build_openai_body,
)
from orchestra_api.providers.openai_provider import _parse_response as _parse_openai_response  # noqa: E402
from orchestra_api.runner import run_task, standard_tool_registry  # noqa: E402


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def main() -> None:
    normalized = Message(role=Role.USER, content="hi")
    if normalized.content != (TextBlock(text="hi"),) or normalized.text != "hi":
        fail(f"string content was not normalized: {normalized!r}")
    empty = Message(role=Role.USER, content="")
    omitted = Message(role=Role.USER)
    if empty.content != () or empty.text != "" or omitted.content != () or omitted.text != "":
        fail(f"empty content was not normalized: empty={empty!r}, omitted={omitted!r}")
    blocks = Message(role=Role.USER, content=[TextBlock("a"), TextBlock("b")])
    if blocks.text != "ab":
        fail(f"text blocks did not concatenate: {blocks!r}")
    try:
        Message(role=Role.USER, content=123)
    except TypeError:
        pass
    else:
        fail("unsupported message content should raise TypeError")
    ok("Message content normalizes to immutable text blocks")

    canonical_id = "canonical_internal_id"
    vendor_id = "call_abc"
    wire_messages = [
        Message(role=Role.SYSTEM, content="system", turn_id="turn_internal"),
        Message(role=Role.USER, content="question"),
        Message(
            role=Role.ASSISTANT,
            content="calling",
            tool_calls=[
                ToolCall(
                    id=canonical_id,
                    vendor_id=vendor_id,
                    name="lookup",
                    arguments={"key": "value"},
                    provider_metadata={"thoughtSignature": "opaque-signature"},
                )
            ],
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id=canonical_id, ok=True, content="result"),
        ),
    ]
    wire_request = ModelRequest(messages=wire_messages)
    wire_bodies = [
        _build_openai_body(wire_request, "test-model"),
        _build_anthropic_body(wire_request, "test-model", 100),
        _build_gemini_body(wire_request),
    ]
    for body in wire_bodies:
        serialized = json.dumps(body)
        if vendor_id not in serialized or canonical_id in serialized:
            fail(f"provider body did not preserve the vendor id boundary: {body!r}")
        if "turn_internal" in serialized or "schema_version" in serialized:
            fail(f"internal record fields leaked onto the wire: {body!r}")

    fallback_messages = [
        Message(
            role=Role.ASSISTANT,
            tool_calls=[
                ToolCall(
                    id=canonical_id,
                    name="lookup",
                    provider_metadata={"thoughtSignature": "opaque-signature"},
                )
            ],
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id=canonical_id, ok=True, content="result"),
        ),
    ]
    fallback_request = ModelRequest(messages=fallback_messages)
    fallback_bodies = [
        _build_openai_body(fallback_request, "test-model"),
        _build_anthropic_body(fallback_request, "test-model", 100),
        _build_gemini_body(fallback_request),
    ]
    if not all(canonical_id in json.dumps(body) for body in fallback_bodies):
        fail(f"canonical id fallback did not reach every provider wire body: {fallback_bodies!r}")
    ok("provider request bodies keep internal fields private and echo the correct tool-call id")

    missing_id_response = _parse_openai_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}]
                    }
                }
            ]
        }
    )
    missing_id_call = missing_id_response.message.tool_calls[0]
    if not missing_id_call.id or missing_id_call.vendor_id is not None:
        fail(f"OpenAI missing-id fallback was not canonical-only: {missing_id_call!r}")
    ok("OpenAI tool calls missing a vendor id receive a non-empty canonical id")

    # Synthesized ids must not be position-based: the same tool at the same
    # index in a later turn would otherwise reuse the id, leaving two distinct
    # calls in one conversation sharing a tool_call_id.
    two_missing_ids = {
        _parse_openai_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "lookup", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            }
        ).message.tool_calls[0].id
        for _ in range(2)
    }
    gemini_missing_ids = {
        _parse_gemini_response(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"functionCall": {"name": "lookup", "args": {}}}]
                        }
                    }
                ]
            }
        ).message.tool_calls[0].id
        for _ in range(2)
    }
    if len(two_missing_ids) != 2 or len(gemini_missing_ids) != 2:
        fail(
            "synthesized tool-call ids collided across turns: "
            f"openai={two_missing_ids!r}, gemini={gemini_missing_ids!r}"
        )
    ok("synthesized tool-call ids are unique per call, not position-based")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "existing.txt").write_text("hello from disk")
        (root / ".env").write_text("SECRET=do-not-read-me")

        # repo_root and allowed write scope are the same temp dir here.
        policy = PermissionPolicy(repo_root=root, allowed_write_scope=[root])

        # -- full agent run: tool-call turn (read_file) then final answer --
        tool_turn = ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "existing.txt"})],
            )
        )
        final_turn = ModelResponse(message=Message(role=Role.ASSISTANT, content="task complete"))
        provider = FakeModelProvider(responses=[tool_turn, final_turn])

        result = run_task(provider, policy, "read existing.txt then finish")
        if result.stopped_reason != "final_response":
            fail(f"expected stopped_reason='final_response', got {result.stopped_reason!r}")
        if result.final_response.message.text != "task complete":
            fail("final response content mismatch")
        tool_messages = [m for m in result.messages if m.role == Role.TOOL]
        if not tool_messages or not tool_messages[0].tool_result.ok:
            fail("expected the read_file tool call to succeed")
        if tool_messages[0].tool_result.content != "hello from disk":
            fail("read_file returned unexpected content")
        appended = [message for message in result.messages if message.role in (Role.ASSISTANT, Role.TOOL)]
        if any(message.turn_id is None for message in appended):
            fail(f"agent-appended messages were not stamped with turn ids: {appended!r}")
        if appended[0].turn_id != appended[1].turn_id:
            fail(f"assistant tool call and result must share a turn id: {appended!r}")
        if result.run.agent_id != result.agent.agent_id:
            fail(f"agent run owner link is inconsistent: {result!r}")
        if result.final_response.message.turn_id != result.messages[-1].turn_id:
            fail(
                "final_response carries a different turn identity than the same "
                f"message in the transcript: {result.final_response.message!r}"
            )
        if result.final_response.message.turn_id is None:
            fail("final_response.message was returned without a turn id")
        ok("full ApiAgent run (tool-call turn -> final answer) via FakeModelProvider")

        # -- allow path: read/list/write inside repo_root + allowed_write_scope --
        tools = standard_tool_registry()

        r = tools["read_file"].execute(ToolCall(id="a1", name="read_file", arguments={"path": "existing.txt"}), policy)
        if not r.ok:
            fail(f"read_file should be allowed inside repo_root: {r.error}")
        ok("allow: read_file inside repo_root")

        r = tools["list_files"].execute(ToolCall(id="a2", name="list_files", arguments={"path": "."}), policy)
        if not r.ok or "existing.txt" not in r.content:
            fail(f"list_files should be allowed and show existing.txt: {r}")
        ok("allow: list_files inside repo_root")

        r = tools["write_file"].execute(
            ToolCall(id="a3", name="write_file", arguments={"path": "new.txt", "content": "written by smoke test"}),
            policy,
        )
        if not r.ok or not (root / "new.txt").exists():
            fail(f"write_file should be allowed inside allowed_write_scope: {r}")
        ok("allow: write_file inside allowed_write_scope")

        # -- deny path: write outside the allowed write scope --
        no_write_policy = PermissionPolicy(repo_root=root)  # allowed_write_scope defaults to empty
        r = tools["write_file"].execute(
            ToolCall(id="d1", name="write_file", arguments={"path": "should_not_exist.txt", "content": "x"}),
            no_write_policy,
        )
        if r.ok or (root / "should_not_exist.txt").exists():
            fail("write_file should be denied when allowed_write_scope is empty")
        ok("deny: write_file outside the explicit allowed write scope")

        # -- deny path: forbidden pattern (.env) --
        r = tools["read_file"].execute(ToolCall(id="d2", name="read_file", arguments={"path": ".env"}), policy)
        if r.ok:
            fail("read_file should deny a forbidden-pattern path (.env)")
        ok("deny: read_file on a forbidden-pattern path (.env)")

        # -- deny path: .. path traversal --
        r = tools["read_file"].execute(
            ToolCall(id="d3", name="read_file", arguments={"path": "../outside.txt"}), policy
        )
        if r.ok:
            fail("read_file should deny a .. path-traversal attempt")
        ok("deny: read_file on a .. path-traversal attempt")

        # -- deny path: run_shell disabled by default --
        r = tools["run_shell"].execute(
            ToolCall(id="d4", name="run_shell", arguments={"argv": ["echo", "hi"]}), policy
        )
        if r.ok:
            fail("run_shell should be denied by default")
        ok("deny: run_shell disabled by default")

        # -- deny path: always-deny command wins even when explicitly allowlisted --
        risky_policy = PermissionPolicy(
            repo_root=root, shell_enabled=True, shell_allowlist=[("rm",)]
        )
        r = tools["run_shell"].execute(
            ToolCall(id="d5", name="run_shell", arguments={"argv": ["rm", "-rf", "existing.txt"]}), risky_policy
        )
        if r.ok or not (root / "existing.txt").exists():
            fail("run_shell must deny 'rm' even when explicitly allowlisted")
        ok("deny: run_shell always-deny command ('rm') overrides an explicit allowlist")

        # -- regression: a real provider's run_task() request must include
        # schemas for all four standard tools --
        captured: dict = {}

        class _FakeHttpResponse:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def read(self) -> bytes:
                return self._body

            def __enter__(self) -> "_FakeHttpResponse":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

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
        ok("retry: HTTP 503 retries and succeeds with sleep patched")

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
        ok("retry: transport TimeoutError retries and succeeds with sleep patched")

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
        ok("retry: transient URLError retries and succeeds with sleep patched")

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
        ok("retry: permanent certificate URLError fails immediately")

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
        ok("retry: truncated response body retries and succeeds with sleep patched")

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
        ok("retry: HTTP 501/505/511 fail immediately")

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
        ok("retry: request and backoff time share one overall deadline")

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
        ok("retry: HTTP 400 fails immediately after exactly one attempt")

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
        ok("retry: exhausted transient failures report the default three attempts")

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
        ok("retry: model discovery honours numeric Retry-After with sleep patched")

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
        ok("retry: HTTP-date Retry-After is parsed and respected")

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
        ok("retry: malformed Retry-After falls back to exponential delay")

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
        ok("retry: Retry-After is capped at eight seconds")

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
        ok("retry: API keys in vendor error bodies are redacted")

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
        ok("retry: no API-key prefix survives the 500-byte truncation boundary")

        # -- regression: model discovery uses each provider's listing
        # endpoint, sends keys only as headers, and parses the documented ids.
        openai_model_key = "sk-openai-model-list-key-do-not-use"
        os.environ[API_KEY_ENV_VAR] = openai_model_key

        def _fake_openai_models_urlopen(request, timeout=None):  # noqa: ANN001
            if request.get_method() != "GET":
                fail(f"expected OpenAI model listing to use GET, got {request.get_method()!r}")
            if request.full_url != "https://api.openai.com/v1/models":
                fail(f"expected OpenAI model listing URL, got {request.full_url!r}")
            if openai_model_key in request.full_url:
                fail("OpenAI model listing key must never appear in the request URL")
            headers = _headers(request)
            if headers.get("authorization") != f"Bearer {openai_model_key}":
                fail(f"expected OpenAI Authorization bearer header, got {request.header_items()!r}")
            payload = {"data": [{"id": "gpt-list-a"}, {"id": "text-embedding-list-b"}]}
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

        with mock.patch("urllib.request.urlopen", side_effect=_fake_openai_models_urlopen):
            openai_models = list_models(OpenAIProvider())
            openai_models_all = list_models(OpenAIProvider(), include_all=True)
        del os.environ[API_KEY_ENV_VAR]
        # Default drops the embedding model; include_all keeps the raw listing.
        if openai_models != ["gpt-list-a"]:
            fail(f"expected the embedding model filtered out by default, got {openai_models!r}")
        if openai_models_all != ["gpt-list-a", "text-embedding-list-b"]:
            fail(f"expected include_all=True to return the raw listing, got {openai_models_all!r}")
        ok("model_discovery lists OpenAI wire models unfiltered, key only in Authorization header")

        anthropic_model_key = "anthropic-model-list-key-do-not-use"
        os.environ[ANTHROPIC_API_KEY_ENV_VAR] = anthropic_model_key

        def _fake_anthropic_models_urlopen(request, timeout=None):  # noqa: ANN001
            if request.get_method() != "GET":
                fail(f"expected Anthropic model listing to use GET, got {request.get_method()!r}")
            if request.full_url != "https://api.anthropic.com/v1/models":
                fail(f"expected Anthropic model listing URL, got {request.full_url!r}")
            if anthropic_model_key in request.full_url:
                fail("Anthropic model listing key must never appear in the request URL")
            headers = _headers(request)
            if headers.get("x-api-key") != anthropic_model_key:
                fail(f"expected Anthropic x-api-key header, got {request.header_items()!r}")
            if headers.get("anthropic-version") != ANTHROPIC_VERSION:
                fail(f"expected Anthropic version header, got {request.header_items()!r}")
            payload = {"data": [{"id": "claude-list-a"}, {"id": "claude-list-b"}]}
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

        with mock.patch("urllib.request.urlopen", side_effect=_fake_anthropic_models_urlopen):
            anthropic_models = list_models(AnthropicProvider())
        del os.environ[ANTHROPIC_API_KEY_ENV_VAR]
        if anthropic_models != ["claude-list-a", "claude-list-b"]:
            fail(f"expected unfiltered Anthropic model ids, got {anthropic_models!r}")
        ok("model_discovery lists Anthropic models with x-api-key and anthropic-version headers")

        gemini_model_key = "gemini-model-list-key-do-not-use"
        os.environ[GEMINI_API_KEY_ENV_VAR] = gemini_model_key

        def _fake_gemini_models_urlopen(request, timeout=None):  # noqa: ANN001
            if request.get_method() != "GET":
                fail(f"expected Gemini model listing to use GET, got {request.get_method()!r}")
            expected_url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"
            if request.full_url != expected_url:
                fail(f"expected Gemini model listing URL {expected_url!r}, got {request.full_url!r}")
            if gemini_model_key in request.full_url:
                fail("Gemini model listing key must never appear in the request URL")
            headers = _headers(request)
            if headers.get("x-goog-api-key") != gemini_model_key:
                fail(f"expected Gemini x-goog-api-key header, got {request.header_items()!r}")
            payload = {
                "models": [
                    {
                        "name": "models/gemini-generate-a",
                        "supportedGenerationMethods": ["generateContent", "countTokens"],
                    },
                    {
                        "name": "models/gemini-embed-b",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                    {
                        "name": "gemini-generate-c",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                ]
            }
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

        with mock.patch("urllib.request.urlopen", side_effect=_fake_gemini_models_urlopen):
            gemini_models = list_models(GeminiProvider())
        del os.environ[GEMINI_API_KEY_ENV_VAR]
        if gemini_models != ["gemini-generate-a", "gemini-generate-c"]:
            fail(f"expected Gemini generateContent ids with models/ stripped, got {gemini_models!r}")
        ok("model_discovery filters Gemini to generateContent models and strips models/ prefix")

        # -- the coding-model filter: non-text modalities out, coding models in --
        from orchestra_api.model_discovery import is_probably_text_model

        must_keep = [
            "gpt-5-codex",            # "codex" must NOT be treated as non-text
            "gpt-5.1-codex-mini",
            "gpt-4o-search-preview",  # search variants are ordinary chat models
            "gemini-omni-flash-preview",
            "claude-opus-5",
            "gemini-3.5-flash-lite",
            "brand-new-model-9",      # unknown families must survive the filter
        ]
        must_drop = [
            "tts-1-hd",
            "whisper-1",
            "text-embedding-3-large",
            "omni-moderation-latest",
            "gpt-realtime",
            "gpt-4o-transcribe",
            "dall-e-3",
            "gemini-2.5-flash-preview-tts",
            "gemini-3-pro-image",
            "lyria-3-pro-preview",
            "nano-banana-pro-preview",
            "gemini-robotics-er-2-preview",
            "babbage-002",
            "davinci-002",
        ]
        for model_id in must_keep:
            if not is_probably_text_model(model_id):
                fail(f"{model_id!r} is a text/coding model but the filter dropped it")
        for model_id in must_drop:
            if is_probably_text_model(model_id):
                fail(f"{model_id!r} is not a text model but the filter kept it")
        ok("model_discovery text-model filter keeps codex/search/unknown, drops tts/image/audio/embedding")

        # -- OpenAI shutdown_date: past retires the model, future keeps it --
        past = (date.today() - timedelta(days=1)).isoformat()
        future = (date.today() + timedelta(days=365)).isoformat()

        def _fake_openai_models_shutdown_urlopen(request, timeout=None):  # noqa: ANN001
            payload = {
                "data": [
                    {"id": "gpt-live-model", "shutdown_date": future},
                    {"id": "gpt-retired-model", "shutdown_date": past},
                    {"id": "gpt-no-date-model"},
                    {"id": "gpt-bad-date-model", "shutdown_date": "not-a-date"},
                ]
            }
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

        os.environ[API_KEY_ENV_VAR] = "sk-openai-fake-test-key-do-not-use"
        with mock.patch("urllib.request.urlopen", side_effect=_fake_openai_models_shutdown_urlopen):
            live_models = list_models(OpenAIProvider())
            all_models = list_models(OpenAIProvider(), include_all=True)
        del os.environ[API_KEY_ENV_VAR]

        if "gpt-retired-model" in live_models:
            fail(f"expected a past shutdown_date to retire the model, got {live_models!r}")
        for expected in ("gpt-live-model", "gpt-no-date-model", "gpt-bad-date-model"):
            if expected not in live_models:
                fail(f"expected {expected!r} to survive shutdown_date filtering, got {live_models!r}")
        if "gpt-retired-model" not in all_models:
            fail(f"include_all=True must bypass shutdown filtering, got {all_models!r}")
        ok("model_discovery drops models whose shutdown_date has passed, include_all bypasses it")

        def _fake_urlopen(request, timeout=None):  # noqa: ANN001
            captured["body"] = json.loads(request.data.decode("utf-8"))
            payload = json.dumps(
                {
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {},
                }
            ).encode("utf-8")
            return _FakeHttpResponse(payload)

        os.environ[API_KEY_ENV_VAR] = "sk-openai-fake-test-key-do-not-use"
        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            run_task(
                OpenAIProvider(model="constructor-default-model"),
                policy,
                "hello",
                model="request-override-model",
            )
        del os.environ[API_KEY_ENV_VAR]

        sent_tool_names = {t["function"]["name"] for t in captured.get("body", {}).get("tools", [])}
        expected_tool_names = {"read_file", "write_file", "list_files", "run_shell"}
        if sent_tool_names != expected_tool_names:
            fail(f"expected run_task() request to include {expected_tool_names}, got {sent_tool_names!r}")
        if captured.get("body", {}).get("model") != "request-override-model":
            fail(f"run_task(model=...) did not reach the wire: {captured.get('body')!r}")
        ok("real provider's run_task() request includes all four standard tool schemas")
        ok("run_task(model=...) overrides the provider's default model on the wire")

        # Anthropic and OpenAI-compatible providers use the same request-level
        # override contract (Gemini is checked on its URL below).
        anthropic_override_body: dict = {}

        def _fake_anthropic_override_urlopen(request, timeout=None):  # noqa: ANN001
            anthropic_override_body.update(json.loads(request.data.decode("utf-8")))
            payload = {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
                "stop_reason": "end_turn",
            }
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

        with mock.patch.dict(os.environ, {ANTHROPIC_API_KEY_ENV_VAR: "anthropic-override-key"}):
            with mock.patch("urllib.request.urlopen", side_effect=_fake_anthropic_override_urlopen):
                AnthropicProvider(model="anthropic-constructor-default").create_response(
                    ModelRequest(
                        messages=basic_request.messages,
                        model="anthropic-wire-override",
                    )
                )
        if anthropic_override_body.get("model") != "anthropic-wire-override":
            fail(f"Anthropic request model override did not reach body: {anthropic_override_body!r}")

        compatible_override_env = "ORCHESTRA_MODEL_OVERRIDE_TEST_KEY"
        compatible_override_body: dict = {}

        def _fake_compatible_override_urlopen(request, timeout=None):  # noqa: ANN001
            compatible_override_body.update(json.loads(request.data.decode("utf-8")))
            return _openai_success("ok")

        with mock.patch.dict(os.environ, {compatible_override_env: "compatible-override-key"}):
            with mock.patch("urllib.request.urlopen", side_effect=_fake_compatible_override_urlopen):
                OpenAICompatibleProvider(
                    api_key_env_var=compatible_override_env,
                    base_url="https://mock.invalid/v1",
                    model="compatible-constructor-default",
                ).create_response(
                    ModelRequest(
                        messages=basic_request.messages,
                        model="compatible-wire-override",
                    )
                )
        if compatible_override_body.get("model") != "compatible-wire-override":
            fail(
                "OpenAI-compatible request model override did not reach body: "
                f"{compatible_override_body!r}"
            )
        ok("Anthropic and OpenAI-compatible request model overrides reach the wire")

        # -- regression: a real GeminiProvider request must carry the four
        # tools as sanitized tools[].functionDeclarations, not raw schemas --
        os.environ[GEMINI_API_KEY_ENV_VAR] = "AIza-fake-test-key-do-not-use"
        gemini_captured: dict = {}

        def _fake_gemini_urlopen(request, timeout=None):  # noqa: ANN001
            gemini_captured["url"] = request.full_url
            gemini_captured["body"] = json.loads(request.data.decode("utf-8"))
            payload = json.dumps(
                {
                    "candidates": [
                        {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
                }
            ).encode("utf-8")
            return _FakeHttpResponse(payload)

        with mock.patch("urllib.request.urlopen", side_effect=_fake_gemini_urlopen):
            run_task(GeminiProvider(model="gemini-constructor-default"), policy, "hello", model="gemini-wire-override")
        del os.environ[GEMINI_API_KEY_ENV_VAR]

        if "AIza-fake-test-key-do-not-use" in gemini_captured.get("url", ""):
            fail("Gemini API key must never appear in the request URL")
        if "/models/gemini-wire-override:generateContent" not in gemini_captured.get("url", ""):
            fail(f"Gemini request model override did not reach URL: {gemini_captured.get('url')!r}")
        declarations = (gemini_captured.get("body", {}).get("tools") or [{}])[0].get(
            "functionDeclarations", []
        )
        gemini_tool_names = {d.get("name") for d in declarations}
        if gemini_tool_names != expected_tool_names:
            fail(f"expected Gemini request to declare {expected_tool_names}, got {gemini_tool_names!r}")
        argv_schema = next(
            (d["parameters"] for d in declarations if d["name"] == "run_shell"), {}
        )
        if argv_schema.get("type") != "object" or "argv" not in argv_schema.get("properties", {}):
            fail(f"expected run_shell's Gemini parameters to survive sanitization, got {argv_schema!r}")
        ok("real Gemini request carries sanitized tools[].functionDeclarations, key not in URL")

        # -- every real provider normalizes malformed or non-object HTTP 200
        # JSON into ProviderError rather than leaking decoder/parser errors. --
        malformed_compatible_env = "ORCHESTRA_MALFORMED_JSON_TEST_KEY"
        malformed_providers = [
            (OpenAIProvider(max_attempts=1), API_KEY_ENV_VAR, "openai-malformed-key"),
            (
                AnthropicProvider(max_attempts=1),
                ANTHROPIC_API_KEY_ENV_VAR,
                "anthropic-malformed-key",
            ),
            (GeminiProvider(max_attempts=1), GEMINI_API_KEY_ENV_VAR, "gemini-malformed-key"),
            (
                OpenAICompatibleProvider(
                    api_key_env_var=malformed_compatible_env,
                    base_url="https://mock.invalid/v1",
                    max_attempts=1,
                ),
                malformed_compatible_env,
                "compatible-malformed-key",
            ),
        ]
        for malformed_provider, malformed_env, malformed_key in malformed_providers:
            for malformed_body, expected_error_text in (
                (b"not valid json", "invalid JSON"),
                (b"[]", "non-object JSON"),
            ):
                with mock.patch.dict(os.environ, {malformed_env: malformed_key}):
                    with mock.patch(
                        "urllib.request.urlopen",
                        return_value=_FakeHttpResponse(malformed_body),
                    ):
                        try:
                            malformed_provider.create_response(basic_request)
                        except ProviderError as exc:
                            if expected_error_text not in str(exc):
                                fail(
                                    f"expected {expected_error_text!r} from "
                                    f"{malformed_provider.name}, got {exc!r}"
                                )
                        else:
                            fail(
                                f"expected {malformed_provider.name} malformed HTTP 200 "
                                "response to raise ProviderError"
                            )
        ok("all real providers normalize malformed/non-object HTTP 200 JSON to ProviderError")

        # -- regression: Gemini thinking tool calls carry a thoughtSignature
        # sibling on the functionCall part, and stateless follow-up requests
        # must echo it byte-identically on the model-role tool-call turn.
        os.environ[GEMINI_API_KEY_ENV_VAR] = "AIza-fake-test-key-do-not-use"
        thought_signature = (
            "EqsCCqgCARFNMg8IpGYi0elDNnCgmlGxXzZYE3vHXw+E9+uwvV9azoV1Tyk"
            "GZZz4WVUAmOcSJuP27nhJ"
        )
        gemini_two_turn_requests: list[dict] = []
        gemini_two_turn_call_count = [0]

        def _fake_gemini_two_turn_urlopen(request, timeout=None):  # noqa: ANN001
            gemini_two_turn_call_count[0] += 1
            gemini_two_turn_requests.append(json.loads(request.data.decode("utf-8")))
            if gemini_two_turn_call_count[0] == 1:
                payload = {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "read_file",
                                            "args": {"path": "existing.txt"},
                                            "id": "call_142486",
                                        },
                                        "thoughtSignature": thought_signature,
                                    }
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
                }
            elif gemini_two_turn_call_count[0] == 2:
                payload = {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": "read complete"}],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
                }
            else:
                raise AssertionError("expected exactly two Gemini HTTP calls")
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

        with mock.patch("urllib.request.urlopen", side_effect=_fake_gemini_two_turn_urlopen):
            gemini_two_turn_result = run_task(GeminiProvider(), policy, "read existing.txt")
        del os.environ[GEMINI_API_KEY_ENV_VAR]

        if gemini_two_turn_result.stopped_reason != "final_response":
            fail(
                "expected Gemini two-turn run to stop on final_response, "
                f"got {gemini_two_turn_result.stopped_reason!r}"
            )
        if gemini_two_turn_call_count[0] != 2:
            fail(f"expected exactly two Gemini HTTP calls, got {gemini_two_turn_call_count[0]}")
        second_gemini_body = gemini_two_turn_requests[1]
        echoed_function_call_parts = [
            part
            for content in second_gemini_body.get("contents", [])
            if content.get("role") == "model"
            for part in content.get("parts", [])
            if part.get("functionCall", {}).get("name") == "read_file"
        ]
        if len(echoed_function_call_parts) != 1:
            fail(
                "expected the second Gemini request to echo one model-role "
                f"read_file functionCall part, got {echoed_function_call_parts!r}"
            )
        echoed_part = echoed_function_call_parts[0]
        if echoed_part.get("thoughtSignature") != thought_signature:
            fail("expected exact Gemini thoughtSignature on second request")
        if echoed_part.get("functionCall", {}).get("id") != "call_142486":
            fail(f"expected real Gemini functionCall id to round-trip, got {echoed_part!r}")
        ok("real Gemini two-turn tool loop echoes thoughtSignature on the second request")

        # -- context compaction: under budget leaves messages untouched --
        compact_under_messages = [
            Message(role=Role.SYSTEM, content="stay concise"),
            Message(role=Role.USER, content="first goal"),
            Message(role=Role.ASSISTANT, content="short answer"),
        ]
        compact_under = compact_messages_for_budget(compact_under_messages, budget=1_000)
        if compact_under.changed or compact_under.messages != compact_under_messages:
            fail(f"under-budget compaction should leave messages untouched, got {compact_under}")
        ok("context compaction leaves under-budget conversations untouched")

        # -- context compaction: over budget drops old middle while staying coherent --
        compact_over_messages = [
            Message(role=Role.SYSTEM, content="system prompt must stay"),
            Message(role=Role.USER, content="earliest user goal must stay"),
            Message(role=Role.ASSISTANT, content="old assistant detail " * 120),
            Message(role=Role.USER, content="old follow-up " * 120),
            Message(role=Role.ASSISTANT, content="old tool analysis " * 120),
            Message(role=Role.USER, content="latest user request must stay"),
        ]
        compact_over = compact_messages_for_budget(
            compact_over_messages,
            budget=140,
            recent_turns=1,
        )
        compacted_contents = [message.text for message in compact_over.messages]
        if not all(
            isinstance(message, Message) and isinstance(message.content, tuple)
            for message in compact_over.messages
        ):
            fail(f"compaction returned non-canonical message shapes: {compact_over.messages!r}")
        if not compact_over.changed or compact_over.dropped_messages < 1:
            fail(f"expected over-budget conversation to compact, got {compact_over}")
        if compact_over.after_tokens > compact_over.budget:
            fail(f"compacted conversation still exceeds budget: {compact_over}")
        if "system prompt must stay" not in compacted_contents:
            fail("compaction did not preserve the system prompt")
        if "earliest user goal must stay" not in compacted_contents:
            fail("compaction did not preserve the earliest user goal")
        if "latest user request must stay" not in compacted_contents:
            fail("compaction did not preserve the latest user turn")
        if not any("Earlier conversation compacted" in content for content in compacted_contents):
            fail(f"compaction did not insert a useful summary: {compacted_contents!r}")
        ok("context compaction preserves system, earliest goal, and recent turns under budget")

        # -- context compaction: impossible budget raises a clear local error --
        impossible_messages = [
            Message(role=Role.SYSTEM, content="system prompt"),
            Message(role=Role.USER, content="x" * 4_000),
        ]
        try:
            compact_messages_for_budget(impossible_messages, budget=100, recent_turns=1)
        except ContextCompactionError as exc:
            message = str(exc)
            if "Increase the budget" not in message or "recent_turns" not in message:
                fail(f"compaction error was not actionable: {message!r}")
        else:
            fail("expected impossible compaction to raise ContextCompactionError")
        ok("context compaction raises a clear error when preserved context cannot fit")

        print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
