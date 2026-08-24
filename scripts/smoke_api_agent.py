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
  - regression check: model discovery lists OpenAI, Anthropic, and Gemini
    models via mocked urllib.request.urlopen, with the correct listing URL,
    auth header, Anthropic version header, and Gemini generateContent filter
  - regression check: a real GeminiProvider round-trips a functionCall
    thoughtSignature across a two-turn tool call loop (via mocked
    urllib.request.urlopen), because Gemini rejects the second stateless
    request when that signature is omitted
  - context compaction stays pure and deterministic: under budget is
    untouched, over budget preserves required context, and impossible
    budgets fail clearly before any provider call

No real network call is ever made in this script. FakeModelProvider is
used for every check except the real-provider regression checks above,
which exercise real request-building code against a mocked HTTP layer.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.compaction import ContextCompactionError, compact_messages_for_budget  # noqa: E402
from orchestra_api.model_discovery import list_models  # noqa: E402
from orchestra_api.models import Message, ModelResponse, Role, ToolCall  # noqa: E402
from orchestra_api.permissions import PermissionPolicy  # noqa: E402
from orchestra_api.providers.anthropic_provider import (  # noqa: E402
    ANTHROPIC_VERSION,
    API_KEY_ENV_VAR as ANTHROPIC_API_KEY_ENV_VAR,
)
from orchestra_api.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from orchestra_api.providers.fake import FakeModelProvider  # noqa: E402
from orchestra_api.providers.gemini_provider import (  # noqa: E402
    API_KEY_ENV_VAR as GEMINI_API_KEY_ENV_VAR,
)
from orchestra_api.providers.gemini_provider import GeminiProvider  # noqa: E402
from orchestra_api.providers.openai_provider import API_KEY_ENV_VAR, OpenAIProvider  # noqa: E402
from orchestra_api.runner import run_task, standard_tool_registry  # noqa: E402


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def main() -> None:
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
        if result.final_response.message.content != "task complete":
            fail("final response content mismatch")
        tool_messages = [m for m in result.messages if m.role == Role.TOOL]
        if not tool_messages or not tool_messages[0].tool_result.ok:
            fail("expected the read_file tool call to succeed")
        if tool_messages[0].tool_result.content != "hello from disk":
            fail("read_file returned unexpected content")
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
        from datetime import date, timedelta

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
            run_task(OpenAIProvider(), policy, "hello")
        del os.environ[API_KEY_ENV_VAR]

        sent_tool_names = {t["function"]["name"] for t in captured.get("body", {}).get("tools", [])}
        expected_tool_names = {"read_file", "write_file", "list_files", "run_shell"}
        if sent_tool_names != expected_tool_names:
            fail(f"expected run_task() request to include {expected_tool_names}, got {sent_tool_names!r}")
        ok("real provider's run_task() request includes all four standard tool schemas")

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
            run_task(GeminiProvider(), policy, "hello")
        del os.environ[GEMINI_API_KEY_ENV_VAR]

        if "AIza-fake-test-key-do-not-use" in gemini_captured.get("url", ""):
            fail("Gemini API key must never appear in the request URL")
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
        compacted_contents = [message.content for message in compact_over.messages]
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
