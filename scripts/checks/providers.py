"""Fixture-free checks for providers."""

from __future__ import annotations

import json
import os
import unittest.mock as mock
from symphonai_api.models import Message, ModelRequest, Role
from symphonai_api.providers.anthropic_provider import API_KEY_ENV_VAR as ANTHROPIC_API_KEY_ENV_VAR
from symphonai_api.providers.anthropic_provider import AnthropicProvider
from symphonai_api.providers.base import ProviderError
from symphonai_api.providers.gemini_provider import API_KEY_ENV_VAR as GEMINI_API_KEY_ENV_VAR
from symphonai_api.providers.gemini_provider import GeminiProvider
from symphonai_api.providers.openai_compatible import OpenAICompatibleProvider
from symphonai_api.providers.openai_provider import API_KEY_ENV_VAR, OpenAIProvider
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
