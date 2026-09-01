"""Workspace-backed checks for providers live."""

from __future__ import annotations

import json
import os
import unittest.mock as mock
from symphonai_api.providers.gemini_provider import API_KEY_ENV_VAR as GEMINI_API_KEY_ENV_VAR
from symphonai_api.providers.gemini_provider import GeminiProvider
from symphonai_api.providers.openai_provider import API_KEY_ENV_VAR, OpenAIProvider
from symphonai_api.runner import run_task
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


class _FakeHttpResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

@check("providers_live.openai_tools_and_model_override")
def check_providers_live_openai_tools_and_model_override() -> None:
    previous_api_key = os.environ.get(API_KEY_ENV_VAR)
    try:
        with workspace() as ws:
            root = ws.root
            outside_tmp = str(ws.outside)
            policy = ws.policy
            tools = ws.tools
            captured: dict = {}
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
            expected_tool_names = {
                "read_file",
                "write_file",
                "edit_file",
                "multi_edit_file",
                "list_files",
                "glob",
                "grep",
                "run_shell",
                "web_fetch",
            }
            if sent_tool_names != expected_tool_names:
                fail(f"expected run_task() request to include {expected_tool_names}, got {sent_tool_names!r}")
            if captured.get("body", {}).get("model") != "request-override-model":
                fail(f"run_task(model=...) did not reach the wire: {captured.get('body')!r}")
    finally:
        if previous_api_key is None:
            os.environ.pop(API_KEY_ENV_VAR, None)
        else:
            os.environ[API_KEY_ENV_VAR] = previous_api_key

@check("providers_live.gemini_tools_and_model_override")
def check_providers_live_gemini_tools_and_model_override() -> None:
    previous_api_key = os.environ.get(GEMINI_API_KEY_ENV_VAR)
    try:
        with workspace() as ws:
            root = ws.root
            outside_tmp = str(ws.outside)
            policy = ws.policy
            tools = ws.tools
            expected_tool_names = {
                "read_file",
                "write_file",
                "edit_file",
                "multi_edit_file",
                "list_files",
                "glob",
                "grep",
                "run_shell",
                "web_fetch",
            }
            # -- regression: a real GeminiProvider request must carry the eight
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
    finally:
        if previous_api_key is None:
            os.environ.pop(GEMINI_API_KEY_ENV_VAR, None)
        else:
            os.environ[GEMINI_API_KEY_ENV_VAR] = previous_api_key

@check("providers_live.gemini_thought_signature")
def check_providers_live_gemini_thought_signature() -> None:
    previous_api_key = os.environ.get(GEMINI_API_KEY_ENV_VAR)
    try:
        with workspace() as ws:
            root = ws.root
            outside_tmp = str(ws.outside)
            policy = ws.policy
            tools = ws.tools
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
    finally:
        if previous_api_key is None:
            os.environ.pop(GEMINI_API_KEY_ENV_VAR, None)
        else:
            os.environ[GEMINI_API_KEY_ENV_VAR] = previous_api_key
