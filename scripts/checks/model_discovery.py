"""Fixture-free checks for model discovery."""

from __future__ import annotations

import json
import os
import unittest.mock as mock
from datetime import date, timedelta
from orchestra_api.model_discovery import list_models
from orchestra_api.providers.anthropic_provider import ANTHROPIC_VERSION, API_KEY_ENV_VAR as ANTHROPIC_API_KEY_ENV_VAR
from orchestra_api.providers.anthropic_provider import AnthropicProvider
from orchestra_api.providers.gemini_provider import API_KEY_ENV_VAR as GEMINI_API_KEY_ENV_VAR
from orchestra_api.providers.gemini_provider import GeminiProvider
from orchestra_api.providers.openai_provider import API_KEY_ENV_VAR, OpenAIProvider
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

def _headers(request) -> dict[str, str]:  # noqa: ANN001
    return {name.lower(): value for name, value in request.header_items()}

@check("discovery.openai_models")
def check_discovery_openai_models() -> None:
    previous_api_key = os.environ.get(API_KEY_ENV_VAR)
    try:
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
    finally:
        if previous_api_key is None:
            os.environ.pop(API_KEY_ENV_VAR, None)
        else:
            os.environ[API_KEY_ENV_VAR] = previous_api_key

@check("discovery.anthropic_models")
def check_discovery_anthropic_models() -> None:
    previous_api_key = os.environ.get(ANTHROPIC_API_KEY_ENV_VAR)
    try:
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
    finally:
        if previous_api_key is None:
            os.environ.pop(ANTHROPIC_API_KEY_ENV_VAR, None)
        else:
            os.environ[ANTHROPIC_API_KEY_ENV_VAR] = previous_api_key

@check("discovery.gemini_models")
def check_discovery_gemini_models() -> None:
    previous_api_key = os.environ.get(GEMINI_API_KEY_ENV_VAR)
    try:
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
    finally:
        if previous_api_key is None:
            os.environ.pop(GEMINI_API_KEY_ENV_VAR, None)
        else:
            os.environ[GEMINI_API_KEY_ENV_VAR] = previous_api_key

@check("discovery.text_model_filter")
def check_discovery_text_model_filter() -> None:
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

@check("discovery.shutdown_date_filter")
def check_discovery_shutdown_date_filter() -> None:
    previous_api_key = os.environ.get(API_KEY_ENV_VAR)
    try:
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
    finally:
        if previous_api_key is None:
            os.environ.pop(API_KEY_ENV_VAR, None)
        else:
            os.environ[API_KEY_ENV_VAR] = previous_api_key
