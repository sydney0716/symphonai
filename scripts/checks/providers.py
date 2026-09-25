"""Fixture-free checks for providers."""

from __future__ import annotations

import json
import os
import unittest.mock as mock
from symphonai_api.agent_spec import Effort
from symphonai_api.models import Message, ModelRequest, Role
from symphonai_api.providers.anthropic_provider import API_KEY_ENV_VAR as ANTHROPIC_API_KEY_ENV_VAR
from symphonai_api.providers.anthropic_provider import (
    AnthropicProvider,
    _build_request_body as _build_anthropic_body,
)
from symphonai_api.providers.base import ProviderError
from symphonai_api.providers.gemini_provider import API_KEY_ENV_VAR as GEMINI_API_KEY_ENV_VAR
from symphonai_api.providers.gemini_provider import (
    GeminiProvider,
    _build_request_body as _build_gemini_body,
)
from symphonai_api.providers.openai_compatible import OpenAICompatibleProvider
from symphonai_api.providers.openai_provider import (
    API_KEY_ENV_VAR,
    OpenAIProvider,
    _build_request_body as _build_openai_body,
)
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


@check("providers.effort_request_bodies")
def check_effort_request_bodies() -> None:
    message = Message(role=Role.USER, content="hello")
    high = ModelRequest(messages=[message], effort=Effort.HIGH)
    bodies = {
        "openai": _build_openai_body(high, "gpt-5.4-mini"),
        "anthropic": _build_anthropic_body(high, "claude-sonnet-5", 1024),
        "gemini-3": _build_gemini_body(high, "gemini-3.5-flash"),
        "gemini-2.5": _build_gemini_body(high, "gemini-2.5-flash"),
    }
    expected = {
        "openai": "high",
        "anthropic": {"effort": "high"},
        "gemini-3": {"thinkingLevel": "high"},
        "gemini-2.5": {"thinkingBudget": 8192},
    }
    actual = {
        "openai": bodies["openai"].get("reasoning_effort"),
        "anthropic": bodies["anthropic"].get("output_config"),
        "gemini-3": bodies["gemini-3"].get("generationConfig", {}).get(
            "thinkingConfig"
        ),
        "gemini-2.5": bodies["gemini-2.5"].get("generationConfig", {}).get(
            "thinkingConfig"
        ),
    }
    if actual != expected:
        fail(f"provider effort request fields were wrong: {actual!r}")


@check("providers.default_effort_body_identity")
def check_default_effort_body_identity() -> None:
    request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    actual = {
        "openai": _build_openai_body(request, "gpt-5.4-mini"),
        "anthropic": _build_anthropic_body(request, "claude-sonnet-5", 1024),
        "gemini": _build_gemini_body(request, "gemini-3.5-flash"),
    }
    expected = {
        "openai": {
            "model": "gpt-5.4-mini",
            "messages": [{"role": "user", "content": "hello"}],
        },
        "anthropic": {
            "model": "claude-sonnet-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
        },
        "gemini": {
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        },
    }
    if actual != expected:
        fail(f"default effort changed provider request bodies: {actual!r}")
    unknown = {
        "openai": _build_openai_body(
            ModelRequest(messages=request.messages, effort=Effort.HIGH),
            "unclassified-model",
        ),
        "anthropic": _build_anthropic_body(
            ModelRequest(messages=request.messages, effort=Effort.HIGH),
            "unclassified-model",
            1024,
        ),
        "gemini": _build_gemini_body(
            ModelRequest(messages=request.messages, effort=Effort.HIGH),
            "unclassified-model",
        ),
    }
    if unknown != {
        "openai": {
            "model": "unclassified-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
        "anthropic": {
            "model": "unclassified-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
        },
        "gemini": {
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        },
    }:
        fail(f"unknown models received guessed effort controls: {unknown!r}")


@check("providers.effort_reaches_transports")
def check_effort_reaches_transports() -> None:
    request = ModelRequest(
        messages=[Message(role=Role.USER, content="hello")],
        effort=Effort.HIGH,
    )
    captured: dict[str, dict] = {}

    def openai_urlopen(http_request, timeout=None):  # noqa: ANN001
        captured["openai"] = json.loads(http_request.data.decode("utf-8"))
        return _openai_success("ok")

    with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: "openai-effort-key"}), mock.patch(
        "urllib.request.urlopen", side_effect=openai_urlopen
    ):
        OpenAIProvider(model="gpt-5.4-mini").create_response(request)

    def anthropic_urlopen(http_request, timeout=None):  # noqa: ANN001
        captured["anthropic"] = json.loads(http_request.data.decode("utf-8"))
        payload = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {},
            "stop_reason": "end_turn",
        }
        return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

    with mock.patch.dict(
        os.environ, {ANTHROPIC_API_KEY_ENV_VAR: "anthropic-effort-key"}
    ), mock.patch("urllib.request.urlopen", side_effect=anthropic_urlopen):
        AnthropicProvider(model="claude-sonnet-5").create_response(request)

    def gemini_urlopen(http_request, timeout=None):  # noqa: ANN001
        captured["gemini"] = json.loads(http_request.data.decode("utf-8"))
        payload = {
            "candidates": [
                {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {},
        }
        return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

    with mock.patch.dict(
        os.environ, {GEMINI_API_KEY_ENV_VAR: "gemini-effort-key"}
    ), mock.patch("urllib.request.urlopen", side_effect=gemini_urlopen):
        GeminiProvider(model="gemini-3.5-flash").create_response(request)

    compatible_env = "SYMPHONAI_EFFORT_COMPATIBLE_KEY"

    def compatible_urlopen(http_request, timeout=None):  # noqa: ANN001
        captured["compatible"] = json.loads(http_request.data.decode("utf-8"))
        return _openai_success("ok")

    with mock.patch.dict(os.environ, {compatible_env: "compatible-effort-key"}), mock.patch(
        "urllib.request.urlopen", side_effect=compatible_urlopen
    ):
        OpenAICompatibleProvider(
            api_key_env_var=compatible_env,
            base_url="https://mock.invalid/v1",
            model="gpt-5.4-mini",
            provider_label="grok",
        ).create_response(request)

    actual = {
        "openai": captured["openai"].get("reasoning_effort"),
        "anthropic": captured["anthropic"].get("output_config"),
        "gemini": captured["gemini"].get("generationConfig", {}).get(
            "thinkingConfig"
        ),
        "compatible": captured["compatible"].get("reasoning_effort"),
    }
    expected = {
        "openai": "high",
        "anthropic": {"effort": "high"},
        "gemini": {"thinkingLevel": "high"},
        "compatible": "high",
    }
    if actual != expected:
        fail(f"effort did not reach fake HTTP transports: {actual!r}")

@check("providers.model_overrides")
def check_providers_model_overrides() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
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

    compatible_override_env = "SYMPHONAI_MODEL_OVERRIDE_TEST_KEY"
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

@check("providers.malformed_json")
def check_providers_malformed_json() -> None:
    basic_request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    # -- every real provider normalizes malformed or non-object HTTP 200
    # JSON into ProviderError rather than leaking decoder/parser errors. --
    malformed_compatible_env = "SYMPHONAI_MALFORMED_JSON_TEST_KEY"
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
