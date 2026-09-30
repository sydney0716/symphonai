"""Fixture-free checks for providers."""

from __future__ import annotations

import json
import os
import unittest.mock as mock
import symphonai_api.model_table as model_table_module
from symphonai_api.model_table import (
    ModelCapability,
    ModelEffort,
    _models_from_json,
    model_capabilities,
    resolve_effort,
)
from symphonai_api.models import Message, ModelRequest, Role, ToolResult
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
    _build_responses_request_body,
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


def _responses_success(content: str) -> _FakeHttpResponse:
    payload = {
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        }],
        "usage": {},
    }
    return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))


@check("providers.effort_request_bodies")
def check_effort_request_bodies() -> None:
    message = Message(role=Role.USER, content="hello")
    bodies = {
        "openai": _build_openai_body(
            ModelRequest(messages=[message], effort="high"),
            "invented-openai-model",
        ),
        "anthropic": _build_anthropic_body(
            ModelRequest(messages=[message], effort="xhigh"),
            "claude-sonnet-5",
            1024,
        ),
        "gemini": _build_gemini_body(
            ModelRequest(messages=[message], effort="minimal"),
            "gemini-3.5-flash",
        ),
    }
    expected = {
        "openai": "high",
        "anthropic": {"effort": "xhigh"},
        "gemini": {"thinkingLevel": "minimal"},
    }
    actual = {
        "openai": bodies["openai"].get("reasoning_effort"),
        "anthropic": bodies["anthropic"].get("output_config"),
        "gemini": bodies["gemini"].get("generationConfig", {}).get(
            "thinkingConfig"
        ),
    }
    if actual != expected:
        fail(f"provider effort request fields were wrong: {actual!r}")


@check("providers.model_table_schema")
def check_model_table_schema() -> None:
    capabilities = model_capabilities()
    indexed = {(item.wire_format, item.model): item for item in capabilities}
    required = {
        (2, "claude-haiku-4-5"),
        (2, "claude-sonnet-5"),
        (3, "gemini-3-pro"),
        (3, "gemini-3.5-flash"),
    }
    if not required.issubset(indexed):
        fail(f"model effort table omitted required entries: {indexed!r}")
    gemini_efforts = {
        model: tuple(option.id for option in indexed[(3, model)].efforts)
        for model in ("gemini-3-pro", "gemini-3.5-flash")
    }
    if gemini_efforts != {
        "gemini-3-pro": ("low", "high"),
        "gemini-3.5-flash": ("minimal", "low", "medium", "high"),
    }:
        fail(f"Gemini model efforts were flattened across models: {gemini_efforts!r}")
    if _models_from_json({"schema_version": 2, "notes": "bad", "models": []}) != ():
        fail("a mismatched model table schema did not fail closed")
    integer_capabilities = _models_from_json({
        "schema_version": 1,
        "notes": "integer wire value fixture",
        "models": [{
            "provider": "fixture",
            "wire_format": 3,
            "id": "integer-value-fixture",
            "efforts": [{"id": "8192", "value": 8192}],
        }],
    })
    if (
        len(integer_capabilities) != 1
        or integer_capabilities[0].efforts[0].value != 8192
        or type(integer_capabilities[0].efforts[0].value) is not int
    ):
        fail(f"model table loader rejected an integer wire value: {integer_capabilities!r}")
    with mock.patch.object(model_table_module, "model_capabilities", return_value=()):
        if resolve_effort(3, "invented-unavailable-table-model", "future") != "future":
            fail("an unavailable model table did not degrade to pass-through")


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


@check("providers.unlisted_effort_passthrough")
def check_unlisted_effort_passthrough() -> None:
    message = Message(role=Role.USER, content="hello")
    unknown_model = "invented-model-that-is-not-in-the-25p-table"
    request = ModelRequest(messages=[message], effort="vendor-special")
    bodies = {
        "openai": _build_openai_body(
            request,
            unknown_model,
        ),
        "anthropic": _build_anthropic_body(
            request,
            unknown_model,
            1024,
        ),
        "gemini": _build_gemini_body(
            request,
            unknown_model,
        ),
    }
    actual = {
        "openai": bodies["openai"].get("reasoning_effort"),
        "anthropic": bodies["anthropic"].get("output_config", {}).get("effort"),
        "gemini": bodies["gemini"].get("generationConfig", {})
        .get("thinkingConfig", {})
        .get("thinkingLevel"),
    }
    if actual != {name: "vendor-special" for name in actual}:
        fail(f"unlisted model effort was not passed through: {actual!r}")


@check("providers.listed_effort_rejection")
def check_listed_effort_rejection() -> None:
    try:
        _build_anthropic_body(
            ModelRequest(
                messages=[Message(role=Role.USER, content="hello")],
                effort="high",
            ),
            "claude-haiku-4-5",
            1024,
        )
    except ValueError as exc:
        message = str(exc)
        if "claude-haiku-4-5" not in message or "accepted efforts: none" not in message:
            fail(f"listed effort rejection was not actionable: {message!r}")
    else:
        fail("a listed model accepted an undeclared effort")


@check("providers.dated_model_effort_families")
def check_dated_model_effort_families() -> None:
    request = ModelRequest(
        messages=[Message(role=Role.USER, content="hello")],
        effort="high",
    )
    try:
        _build_anthropic_body(
            request,
            "claude-haiku-4-5-20251001",
            1024,
        )
    except ValueError as exc:
        if "claude-haiku-4-5-20251001" not in str(exc) or "accepted efforts: none" not in str(exc):
            fail(f"dated Haiku model did not use its no-effort row: {exc}")
    else:
        fail("dated Haiku model accepted an effort")

    for effort in ("low", "medium", "high"):
        resolved = resolve_effort(3, "gemini-3.1-pro-preview", effort)
        if resolved != effort:
            fail(f"Gemini preview effort {effort!r} resolved as {resolved!r}")

    if resolve_effort(3, "gemini-3-flash-preview", "minimal") != "minimal":
        fail("Gemini Flash preview did not use the Flash family row")
    for model in (
        "gemini-3-flash-lite",
        "gemini-3-flash-lite-preview",
        "claude-opus-4-5x",
        "claude-opus-4-5-20251",
        "unlisted-family-model",
    ):
        if resolve_effort(3 if model.startswith("gemini") else 2, model, "vendor-effort") != "vendor-effort":
            fail(f"unmatched model {model!r} borrowed a family effort row")


@check("providers.exact_model_capability_wins")
def check_exact_model_capability_wins() -> None:
    family = ModelCapability(
        "anthropic",
        2,
        "claude-haiku-4-5",
        (ModelEffort("family", "family-wire"),),
    )
    exact = ModelCapability(
        "anthropic",
        2,
        "claude-haiku-4-5-20251001",
        (ModelEffort("exact", "exact-wire"),),
    )
    with mock.patch.object(model_table_module, "model_capabilities", return_value=(family, exact)):
        if resolve_effort(2, exact.model, "exact") != "exact-wire":
            fail("a suffixed exact row did not take precedence over the family")
        try:
            resolve_effort(2, exact.model, "family")
        except ValueError as exc:
            if "accepted efforts: exact" not in str(exc):
                fail(f"family efforts overrode the exact row: {exc}")
        else:
            fail("the exact row accepted only the family's effort")


@check("providers.effort_reaches_transports")
def check_effort_reaches_transports() -> None:
    request = ModelRequest(
        messages=[Message(role=Role.USER, content="hello")],
        effort="high",
    )
    captured: dict[str, dict] = {}

    def openai_urlopen(http_request, timeout=None):  # noqa: ANN001
        captured["openai"] = json.loads(http_request.data.decode("utf-8"))
        return _responses_success("ok")

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
        "openai": captured["openai"].get("reasoning"),
        "anthropic": captured["anthropic"].get("output_config"),
        "gemini": captured["gemini"].get("generationConfig", {}).get(
            "thinkingConfig"
        ),
        "compatible": captured["compatible"].get("reasoning_effort"),
    }
    expected = {
        "openai": {"effort": "high"},
        "anthropic": {"effort": "high"},
        "gemini": {"thinkingLevel": "high"},
        "compatible": "high",
    }
    if actual != expected:
        fail(f"effort did not reach fake HTTP transports: {actual!r}")


@check("providers.openai_responses_request_and_compatible_chat")
def check_openai_responses_request_and_compatible_chat() -> None:
    request = ModelRequest(
        messages=[Message(role=Role.USER, content="hello")],
        tools=[{
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        effort="high",
    )
    captured: dict[str, object] = {}

    def openai_urlopen(http_request, timeout=None):  # noqa: ANN001
        captured["openai_url"] = http_request.full_url
        captured["openai_body"] = json.loads(http_request.data.decode("utf-8"))
        return _responses_success("ok")

    with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: "responses-request-key"}), mock.patch(
        "urllib.request.urlopen", side_effect=openai_urlopen,
    ):
        OpenAIProvider(model="gpt-5.6-sol", base_url="https://api.invalid/v1").create_response(request)

    compatible_env = "SYMPHONAI_RESPONSES_COMPATIBLE_KEY"

    def compatible_urlopen(http_request, timeout=None):  # noqa: ANN001
        captured["compatible_url"] = http_request.full_url
        captured["compatible_body"] = json.loads(http_request.data.decode("utf-8"))
        return _openai_success("ok")

    with mock.patch.dict(os.environ, {compatible_env: "compatible-chat-key"}), mock.patch(
        "urllib.request.urlopen", side_effect=compatible_urlopen,
    ):
        OpenAICompatibleProvider(
            api_key_env_var=compatible_env,
            base_url="https://compatible.invalid/v1",
            model="gpt-5.6-sol",
        ).create_response(request)

    responses_body = captured["openai_body"]
    compatible_body = captured["compatible_body"]
    if captured["openai_url"] != "https://api.invalid/v1/responses":
        fail(f"OpenAI did not post to Responses: {captured['openai_url']!r}")
    if responses_body != {
        "model": "gpt-5.6-sol",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        "tools": [{
            "type": "function",
            "name": "weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        }],
        "reasoning": {"effort": "high"},
    }:
        fail(f"OpenAI Responses body did not translate the request: {responses_body!r}")
    if (
        captured["compatible_url"] != "https://compatible.invalid/v1/chat/completions"
        or compatible_body != {
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": request.tools,
            "reasoning_effort": "high",
        }
    ):
        fail(f"OpenAI-compatible Chat wire request changed: {captured!r}")


@check("providers.openai_responses_effort")
def check_openai_responses_effort() -> None:
    request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    no_effort = _build_responses_request_body(request, "gpt-5.6-sol")
    with_effort = _build_responses_request_body(
        ModelRequest(messages=request.messages, effort="high"), "gpt-5.6-sol",
    )
    if "reasoning" in no_effort or with_effort.get("reasoning") != {"effort": "high"}:
        fail(f"Responses effort field was absent or misplaced: {no_effort!r}, {with_effort!r}")


@check("providers.openai_effort_capabilities")
def check_openai_effort_capabilities() -> None:
    expected = {
        "gpt-5.6-terra": ("none", "low", "medium", "high", "xhigh"),
        "gpt-5.6-sol": ("none", "low", "medium", "high", "xhigh"),
        "gpt-5.6-luna": ("none", "low", "medium", "high", "xhigh"),
        "gpt-6-sol": ("none", "low", "medium", "high", "xhigh"),
        "gpt-6-luna": ("none", "low", "medium", "high", "xhigh"),
        "gpt-6-astra": ("low", "medium", "high", "xhigh"),
    }
    indexed = {
        capability.model: tuple(option.id for option in capability.efforts)
        for capability in model_capabilities()
        if capability.wire_format == 1 and capability.provider == "openai"
    }
    actual = {model: indexed.get(model, ()) for model in expected}
    if actual != expected:
        fail(f"OpenAI model effort rows were wrong: {actual!r}")
    for model, efforts in expected.items():
        for effort in efforts:
            if resolve_effort(1, model, effort) != effort:
                fail(f"{model} did not preserve listed effort {effort!r}")
    try:
        resolve_effort(1, "gpt-6-astra", "none")
    except ValueError:
        pass
    else:
        fail("gpt-6-astra accepted none effort")


@check("providers.openai_reasoning_round_trip")
def check_openai_reasoning_round_trip() -> None:
    reasoning = {
        "type": "reasoning",
        "id": "rs_opaque_123",
        "encrypted_content": "opaque-encrypted-value",
        "summary": [{"type": "summary_text", "text": "private thought"}],
    }
    function_call = {
        "type": "function_call",
        "call_id": "fc_123",
        "name": "weather",
        "arguments": '{"city":"Seoul"}',
    }
    replies = [
        _FakeHttpResponse(json.dumps({
            "status": "completed",
            "output": [reasoning, function_call],
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }).encode("utf-8")),
        _responses_success("done"),
    ]
    captured: list[dict] = []

    def openai_urlopen(http_request, timeout=None):  # noqa: ANN001
        captured.append(json.loads(http_request.data.decode("utf-8")))
        return replies.pop(0)

    user = Message(role=Role.USER, content="weather?")
    with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: "reasoning-round-trip-key"}), mock.patch(
        "urllib.request.urlopen", side_effect=openai_urlopen,
    ):
        first = OpenAIProvider().create_response(ModelRequest(messages=[user]))
        call = first.message.tool_calls[0]
        second_messages = [
            user,
            first.message,
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(tool_call_id="fc_123", ok=True, content="sunny"),
            ),
        ]
        OpenAIProvider().create_response(ModelRequest(messages=second_messages))
    second_input = captured[1]["input"]
    if call.provider_metadata.get("responses_reasoning_items") != [reasoning]:
        fail(f"Responses reasoning item was not carried by the tool call: {call!r}")
    if (
        second_input[1] != reasoning
        or second_input[2].get("type") != "function_call"
        or second_input[2].get("call_id") != "fc_123"
        or json.loads(second_input[2].get("arguments", "{}")) != {"city": "Seoul"}
        or second_input[3] != {
            "type": "function_call_output", "call_id": "fc_123", "output": "sunny"
        }
    ):
        fail(f"second Responses request did not replay reasoning and tool items verbatim: {second_input!r}")

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
