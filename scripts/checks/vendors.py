"""Core vendor contracts through public providers and scripted HTTP exchanges."""

from __future__ import annotations

import io
import json
import os
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from uuid import uuid4

from scripts.checks.harness import check
from scripts.checks.workspace import workspace
from symphonai_api.agent_loop import ApiAgent
from symphonai_api.call_class import CallClass
from symphonai_api.cost import ModelPrice, PriceTable, UsageTotals, total_cost
from symphonai_api.models import Message, ModelRequest, Role, ToolResult, Usage
from symphonai_api.providers.anthropic_provider import AnthropicProvider
from symphonai_api.providers.base import ContextLengthExceededError, ProviderError
from symphonai_api.providers.gemini_provider import GeminiProvider
from symphonai_api.providers.openai_compatible import OpenAICompatibleProvider
from symphonai_api.providers.openai_provider import OpenAIProvider
from symphonai_api.streaming import StreamAssembler

KEY = "core-vendor-inert-key"
ENV = {name: KEY for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "CORE_CHAT_KEY")}
USER = Message(Role.USER, "core vendor task")


class Response(io.BytesIO):
    status = 200
    headers = {}

    def __init__(self, value):
        super().__init__(value if isinstance(value, bytes) else json.dumps(value).encode())


def sse(*events):
    return "".join((f"event: {name}\n" if name else "") + "data: " + json.dumps(data) + "\n\n"
                   for name, data in events).encode()


def error(status, message):
    return HTTPError("https://core.invalid", status, "scripted rejection", {},
                     io.BytesIO(json.dumps({"error": {"message": message}}).encode()))


@contextmanager
def exchange(*responses):
    bodies, requests, timeouts = [], [], []
    pending = iter(responses)

    def send(request, **options):
        requests.append(request)
        bodies.append(json.loads(request.data))
        timeouts.append(options["timeout"])
        response = next(pending)
        if isinstance(response, Exception):
            raise response
        return Response(response)

    with mock.patch.dict(os.environ, ENV), mock.patch("urllib.request.urlopen", side_effect=send), mock.patch("time.sleep"):
        yield bodies, requests, timeouts


def provider(kind, **options):
    # Unique endpoints isolate public capability learning without touching caches.
    url = f"https://core-{uuid4().hex}.invalid/v1"
    if kind == "anthropic":
        return AnthropicProvider(base_url=url, **options)
    if kind == "openai":
        return OpenAIProvider(base_url=url, **options)
    if kind == "gemini":
        return GeminiProvider(base_url=url, **options)
    return OpenAICompatibleProvider("CORE_CHAT_KEY", url, model="core-chat", **options)


def anthropic(content=None, **options):
    return {"model": "claude-opus-5-5", "content": content or [{"type": "text", "text": "answer"}],
            "usage": {"input_tokens": 10, "output_tokens": 3}, "stop_reason": "end_turn", **options}


def responses(output=None, **options):
    return {"output": output or [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 3}, **options}


def chat(message=None, finish="stop"):
    return {"choices": [{"message": message or {"content": "answer"}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3}}


def gemini(parts=None, finish="STOP", **options):
    return {"candidates": [{"content": {"role": "model", "parts": parts or [{"text": "answer"}]},
                             "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3}, **options}


def streamed(kind, value):
    if kind == "openai":
        return sse(("response.completed", {"response": value}))
    if kind == "gemini":
        return sse((None, value))
    if kind == "chat":
        choice = value["choices"][0]
        delta = dict(choice["message"])
        if "tool_calls" in delta:
            delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(delta["tool_calls"])]
        return sse((None, {"choices": [{"delta": delta, "finish_reason": choice["finish_reason"]}],
                           "usage": value["usage"]})) + b"data: [DONE]\n\n"
    events = [("message_start", {"message": {"model": value["model"], "usage": value["usage"]}})]
    for index, block in enumerate(value["content"]):
        initial = dict(block)
        field = {"text": "text", "thinking": "thinking", "tool_use": "input"}.get(block["type"])
        if field:
            initial[field] = {} if field == "input" else ""
        events.append(("content_block_start", {"index": index, "content_block": initial}))
        if field:
            delta = ({"type": "input_json_delta", "partial_json": json.dumps(block[field])} if field == "input"
                     else {"type": field + "_delta", field: block[field]})
            events.append(("content_block_delta", {"index": index, "delta": delta}))
    events.extend([("message_delta", {"delta": {"stop_reason": value["stop_reason"],
                                               "stop_details": value.get("stop_details")},
                                      "usage": {"output_tokens": value["usage"]["output_tokens"]}}),
                   ("message_stop", {})])
    return sse(*events)


def invoke(p, request=None, *, stream=False):
    request = request or ModelRequest([USER])
    if not stream:
        return p.create_response(request)
    assembler = StreamAssembler()
    for chunk in p.create_response_stream(request):
        assembler.add(chunk)
    return assembler.finish()


def next_messages(first):
    return [USER, first.message, *[Message(Role.TOOL, tool_result=ToolResult(call.id, True, "tool result"))
                                  for call in first.message.tool_calls]]


def run(p, value, *, stream=False, request_model=None):
    with workspace() as ws, exchange(streamed(p.name if p.name != "openai-compatible" else "chat", value) if stream else value):
        return ApiAgent(p, {}, ws.policy, stream=stream).run([USER], model=request_model)


@check("vendors.anthropic_thinking_round_trips")
def anthropic_thinking_round_trips():
    blocks = [{"type": "thinking", "thinking": "private reasoning", "signature": "opaque-signature"},
              {"type": "redacted_thinking", "data": "opaque-redacted"},
              {"type": "tool_use", "id": "tool-1", "name": "read_file", "input": {"path": "note"}}]
    p = provider("anthropic")
    for stream in (False, True):
        with exchange(streamed("anthropic", anthropic(blocks)) if stream else anthropic(blocks), anthropic()) as (bodies, _, _):
            first = invoke(p, stream=stream)
            invoke(p, ModelRequest(next_messages(first)))
            assert bodies[1]["messages"][1]["content"] == blocks
            assert bodies[1]["messages"][2]["content"][0]["tool_use_id"] == "tool-1"


@check("vendors.anthropic_system_cache_is_one_hour")
def anthropic_system_cache_is_one_hour():
    with exchange(anthropic()) as (bodies, _, _):
        invoke(provider("anthropic"), ModelRequest([Message(Role.SYSTEM, "stable"), Message(Role.SYSTEM, "dynamic"), USER]))
        assert bodies[0]["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert "cache_control" not in bodies[0]["system"][1]


@check("vendors.anthropic_flagged_model_requests_server_fallback")
def anthropic_flagged_model_requests_server_fallback():
    with exchange(anthropic(), anthropic()) as (bodies, requests, _):
        invoke(provider("anthropic", model="claude-opus-5-5"))
        invoke(provider("anthropic", model="claude-haiku-4-5"))
        assert bodies[0]["fallbacks"] == "default"
        assert requests[0].get_header("Anthropic-beta")
        assert "fallbacks" not in bodies[1]


@check("vendors.anthropic_refusal_ends_the_run")
def anthropic_refusal_ends_the_run():
    value = anthropic(stop_reason="refusal", stop_details={"category": "fixture-category"})
    for stream in (False, True):
        result = run(provider("anthropic"), value, stream=stream)
        assert result.stopped_reason == "refused"
        assert result.final_response.refusal_category == "fixture-category"
        assert not result.final_response.message.text


@check("vendors.anthropic_answering_model_is_billed")
def anthropic_answering_model_is_billed():
    prices = PriceTable({"claude-opus-5-5": ModelPrice(Decimal(10), Decimal(20)),
                         "claude-sonnet-5-5": ModelPrice(Decimal(1), Decimal(2))}, "USD")
    for model, billed in (("claude-sonnet-5-5", "claude-sonnet-5-5"), ("claude-opus-5-5-20261001", "claude-opus-5-5")):
        result = run(provider("anthropic"), anthropic(model=model))
        assert set(result.usage_by_model) == {billed}
        assert total_cost(result.usage_by_model, prices) == (Decimal("0.000016") if billed.endswith("sonnet-5-5") else Decimal("0.000160"))


def resend_rule(rule):
    resent = []
    for stream in (False, True):
        p = provider("anthropic")
        message = {"tokens": "max_tokens must be <= maximum 1024", "thinking": "thinking unsupported", "fallback": "fallbacks unsupported"}[rule]
        success = streamed("anthropic", anthropic()) if stream else anthropic()
        with exchange(error(400, message), success, success) as (bodies, requests, _):
            invoke(p, ModelRequest([USER], effort="high"), stream=stream)
            invoke(p, ModelRequest([USER], effort="high"), stream=stream)
            assert len(bodies) == 3
            for body in bodies[1:]:
                if rule == "tokens":
                    assert body["max_tokens"] == 1024
                elif rule == "thinking":
                    assert "thinking" not in body
                else:
                    assert "fallbacks" not in body
            assert bodies[0]["max_tokens"] > 1024
            assert "thinking" in bodies[0] and "fallbacks" in bodies[0]
            assert bodies[1] == bodies[2]  # remembered on a fresh call, not just a resend
            assert requests[1].headers == requests[2].headers
            resent.append(({k: v for k, v in bodies[1].items() if k != "stream"}, requests[1].headers))
    assert resent[0] == resent[1]


@check("vendors.anthropic_remembers_output_token_rejection")
def anthropic_remembers_output_token_rejection():
    resend_rule("tokens")


@check("vendors.anthropic_remembers_thinking_rejection")
def anthropic_remembers_thinking_rejection():
    resend_rule("thinking")


@check("vendors.anthropic_remembers_fallback_rejection")
def anthropic_remembers_fallback_rejection():
    resend_rule("fallback")


@check("vendors.responses_reasoning_and_call_outputs_round_trip")
def responses_reasoning_and_call_outputs_round_trip():
    reasoning = {"type": "reasoning", "id": "reason-1", "encrypted_content": "opaque", "summary": []}
    calls = [{"type": "function_call", "call_id": f"call-{i}", "name": "read_file", "arguments": '{"path":"note"}'} for i in (1, 2)]
    p = provider("openai")
    with exchange(responses([reasoning, *calls]), responses()) as (bodies, _, _):
        first = invoke(p)
        invoke(p, ModelRequest(next_messages(first)))
        items = bodies[1]["input"]
        assert reasoning in items
        sent = [x["call_id"] for x in items if x.get("type") == "function_call"]
        outputs = [x["call_id"] for x in items if x.get("type") == "function_call_output"]
        assert sent == outputs == ["call-1", "call-2"]


@check("vendors.responses_tools_are_not_strict")
def responses_tools_are_not_strict():
    schema = {"type": "object", "properties": {"path": {"type": "string"}, "optional": {"type": "integer"}}, "required": ["path"]}
    with exchange(responses()) as (bodies, _, _):
        invoke(provider("openai"), ModelRequest([USER], tools=[{"type": "function", "function": {"name": "read", "parameters": schema}}]))
        assert bodies[0]["tools"][0]["strict"] is False
        assert bodies[0]["tools"][0]["parameters"]["required"] == ["path"]


@check("vendors.responses_refusal_and_filter_end_the_run")
def responses_refusal_and_filter_end_the_run():
    values = [responses([{"type": "message", "content": [{"type": "refusal", "refusal": "fixture refusal"}]}]),
              responses(status="incomplete", incomplete_details={"reason": "content_filter"})]
    for value in values:
        for stream in (False, True):
            assert run(provider("openai"), value, stream=stream).stopped_reason == "refused"


@check("vendors.chat_tool_ids_round_trip")
def chat_tool_ids_round_trip():
    p = provider("chat")
    value = chat({"content": None, "tool_calls": [{"id": "opaque-chat-id", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"note"}'}}]}, "tool_calls")
    with exchange(value, chat()) as (bodies, _, _):
        first = invoke(p)
        invoke(p, ModelRequest(next_messages(first)))
        assert bodies[1]["messages"][1]["tool_calls"][0]["id"] == "opaque-chat-id"
        assert bodies[1]["messages"][2]["tool_call_id"] == "opaque-chat-id"


@check("vendors.chat_refusal_and_filter_end_the_run")
def chat_refusal_and_filter_end_the_run():
    for value in (chat({"refusal": "fixture refusal"}), chat(finish="content_filter")):
        for stream in (False, True):
            assert run(provider("chat"), value, stream=stream).stopped_reason == "refused"


@check("vendors.gemini_signature_and_named_result_round_trip")
def gemini_signature_and_named_result_round_trip():
    part = {"functionCall": {"name": "read_file", "args": {"path": "note"}, "id": "gem-call"}, "thoughtSignature": "opaque-thought"}
    for stream in (False, True):
        p = provider("gemini")
        with exchange(streamed("gemini", gemini([part])) if stream else gemini([part]), gemini()) as (bodies, _, _):
            first = invoke(p, stream=stream)
            invoke(p, ModelRequest(next_messages(first)))
            assert bodies[1]["contents"][1]["parts"] == [part]
            assert bodies[1]["contents"][2]["parts"][0]["functionResponse"]["name"] == "read_file"


@check("vendors.gemini_safety_ends_the_run")
def gemini_safety_ends_the_run():
    for stream in (False, True):
        result = run(provider("gemini"), gemini(finish="SAFETY"), stream=stream)
        assert result.stopped_reason == "refused" and result.final_response.refusal_category == "safety"


@check("vendors.gemini_malformed_call_reports_its_reason")
def gemini_malformed_call_reports_its_reason():
    for stream in (False, True):
        with exchange(streamed("gemini", gemini(finish="MALFORMED_FUNCTION_CALL")) if stream else gemini(finish="MALFORMED_FUNCTION_CALL")):
            try:
                invoke(provider("gemini"), stream=stream)
            except ProviderError as exc:
                assert "MALFORMED_FUNCTION_CALL" in str(exc)
            else:
                raise AssertionError("malformed call accepted")


@check("vendors.streams_match_plain_responses")
def streams_match_plain_responses():
    fixtures = {
        "anthropic": anthropic([{"type": "text", "text": "answer"}, {"type": "tool_use", "id": "call-1", "name": "read", "input": {"path": "note"}}]),
        "openai": responses([{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}, {"type": "function_call", "call_id": "call-1", "name": "read", "arguments": '{"path":"note"}'}]),
        "chat": chat({"content": "answer", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read", "arguments": '{"path":"note"}'}}]}, "tool_calls"),
        "gemini": gemini([{"text": "answer"}, {"functionCall": {"id": "call-1", "name": "read", "args": {"path": "note"}}, "thoughtSignature": "sig"}]),
    }
    for kind, value in fixtures.items():
        p = provider(kind)
        with exchange(value, streamed(kind, value)):
            plain, stream = invoke(p), invoke(p, stream=True)
        assert plain.message == stream.message, (kind, plain, stream)
        assert plain.usage == stream.usage and plain.stop_reason == stream.stop_reason, kind


@check("vendors.transient_status_retries_within_deadline")
def transient_status_retries_within_deadline():
    for kind, value in (("anthropic", anthropic()), ("openai", responses()), ("chat", chat()), ("gemini", gemini())):
        for stream in (False, True):
            with exchange(error(503, "temporary"), streamed(kind, value) if stream else value) as (bodies, _, timeouts):
                invoke(provider(kind, timeout_seconds=10), stream=stream)
                assert len(bodies) == 2 and bodies[0] == bodies[1]
                assert 0 < timeouts[1] <= timeouts[0] <= 10


@check("vendors.background_retries_rate_limits_not_overload")
def background_retries_rate_limits_not_overload():
    for kind, value in (("anthropic", anthropic()), ("openai", responses()), ("chat", chat()), ("gemini", gemini())):
        for stream in (False, True):
            request = ModelRequest([USER], call_class=CallClass.BACKGROUND)
            limited = error(429, "request reached organization max RPM")
            limited.headers["Retry-After"] = "0"
            with exchange(limited, streamed(kind, value) if stream else value) as (bodies, _, _):
                result = invoke(provider(kind), request, stream=stream)
                assert result.message.text == "answer" and len(bodies) == 2
            for status, message in ((529, "overloaded"), (503, "model overloaded"),
                                    (503, "model unavailable"), (429, "request reached organization max RPM")):
                failure = error(status, message)
                if status == 429:
                    failure.headers["Retry-After"] = "120"
                with exchange(failure) as (bodies, _, _), mock.patch("time.sleep") as sleep:
                    try:
                        invoke(provider(kind), request, stream=stream)
                    except ProviderError as exc:
                        assert len(bodies) == 1
                        sleep.assert_not_called()
                        if status == 429:
                            assert message in str(exc)
                        else:
                            assert "did not retry" in str(exc)
                    else:
                        raise AssertionError("background overload or long rate-limit wait accepted")


@check("vendors.context_length_errors_keep_token_counts")
def context_length_errors_keep_token_counts():
    for kind, message, counts in (("anthropic", "prompt is too long: 150 tokens > 100", (150, 100)),
                                   ("gemini", "input token count (150) exceeds the maximum number of tokens allowed (100)", (150, 100)),
                                   ("openai", "context_length_exceeded", (None, None))):
        for stream in (False, True):
            with exchange(error(400, message)):
                try:
                    invoke(provider(kind), stream=stream)
                except ContextLengthExceededError as exc:
                    assert (exc.actual_tokens, exc.limit_tokens) == counts
                else:
                    raise AssertionError("context error lost its type")


@check("vendors.errors_do_not_expose_api_keys")
def errors_do_not_expose_api_keys():
    for kind in ("anthropic", "openai", "chat", "gemini"):
        for stream in (False, True):
            with exchange(error(400, f"invalid key {KEY}")) as (_, requests, _):
                try:
                    invoke(provider(kind), stream=stream)
                except ProviderError as exc:
                    assert KEY not in str(exc)
                    assert KEY not in requests[0].full_url and KEY.encode() not in requests[0].data
                    assert any(KEY in value for value in requests[0].headers.values())
                else:
                    raise AssertionError("vendor error accepted")


@check("vendors.cache_usage_has_separate_prices")
def cache_usage_has_separate_prices():
    value = anthropic(usage={"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 40,
                             "cache_creation_input_tokens": 30, "cache_creation": {"ephemeral_1h_input_tokens": 20}})
    for stream in (False, True):
        with exchange(streamed("anthropic", value) if stream else value):
            usage = invoke(provider("anthropic"), stream=stream).usage
        assert usage == Usage(170, 10, cache_read_tokens=40, cache_write_tokens=30, cache_write_1h_tokens=20)
        price = PriceTable({"core": ModelPrice(Decimal(2), Decimal(3), Decimal(1), Decimal(4), cache_write_1h_per_million=Decimal(5))}, "USD")
        assert price.cost("core", UsageTotals.from_usage(usage)) == Decimal("0.000410")


@check("vendors.long_context_tier_starts_above_threshold")
def long_context_tier_starts_above_threshold():
    table = PriceTable({"gpt-6-luna": ModelPrice(Decimal(1), Decimal(2), long_context=ModelPrice(Decimal(3), Decimal(4)))}, "USD")
    for count in (272000, 272001):
        result = run(provider("openai", model="gpt-6-luna"), responses(usage={"input_tokens": count, "output_tokens": 10}))
        totals = result.usage_by_model["gpt-6-luna"]
        assert totals.long_context_input_tokens == (count if count > 272000 else 0)
        assert table.cost("gpt-6-luna", totals) == Decimal(count * (3 if count > 272000 else 1) + 10 * (4 if count > 272000 else 2)) / 1000000


@check("vendors.unknown_price_has_no_cost")
def unknown_price_has_no_cost():
    table = PriceTable({"known": ModelPrice(Decimal(1), Decimal(2))}, "USD")
    assert table.cost("unknown", UsageTotals(input_tokens=100)) is None
    assert total_cost({"known": UsageTotals(input_tokens=100), "unknown": UsageTotals(input_tokens=100)}, table) is None


@check("vendors.recorded_exchanges_replay")
def recorded_exchanges_replay():
    paths = sorted((Path(__file__).parent / "cassettes").glob("*.json"))
    assert paths
    for path in paths:
        recording = json.loads(path.read_text())
        kind, model = recording["provider"], recording["model"]
        kind = "openai" if kind == "openai-responses" else kind
        p = provider(kind, model=model)
        first_body = recording["exchanges"][0]["request"]["body"]
        tools = first_body.get("tools", [])
        if kind == "openai":
            tools = [{"type": "function", "function": {k: v for k, v in t.items() if k != "type"}} for t in tools]
        elif kind == "gemini":
            tools = [t for group in tools for t in group["functionDeclarations"]]
        messages = [Message(Role.USER, recording["task"])] if any(k in first_body for k in ("messages", "input", "contents")) else []
        for index, recorded in enumerate(recording["exchanges"]):
            expected = recorded["request"]["body"]
            with exchange("".join(recorded["response"]["body_lines"]).encode()) as (bodies, requests, _):
                result = invoke(p, ModelRequest(messages, tools=tools, model=model), stream=True)
            actual = bodies[0]
            # Synthetic cassettes intentionally record only some request fields.
            for key, value in expected.items():
                built = actual[key]
                if key == "input":
                    def normalized(items):
                        return [{**item, "arguments": json.loads(item["arguments"])} if item.get("type") == "function_call" else item for item in items]
                    built, value = normalized(built), normalized(value)
                assert built == value, (path.name, index, key)
            assert requests[0].method == recorded["request"]["method"]
            messages.append(result.message)
            if index + 1 < len(recording["exchanges"]):
                following = recording["exchanges"][index + 1]["request"]["body"]
                if kind == "anthropic":
                    outputs = [part["content"] for m in following["messages"] for part in (m["content"] if isinstance(m["content"], list) else []) if part.get("type") == "tool_result"]
                elif kind == "openai":
                    outputs = [item["output"] for item in following["input"] if item.get("type") == "function_call_output"]
                else:
                    outputs = [part["functionResponse"]["response"]["output"] for m in following["contents"] for part in m["parts"] if "functionResponse" in part]
                calls = result.message.tool_calls
                assert calls and len(outputs) >= len(calls)
                messages.extend(Message(Role.TOOL, tool_result=ToolResult(call.id, True, text))
                                for call, text in zip(calls, outputs[-len(calls):]))
        assert result.message.text == recording["final_answer"], path.name
