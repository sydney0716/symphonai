"""Checks for provider-neutral response streaming and assembly."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from unittest import mock

from symphonai_api.agent_loop import ApiAgent
from symphonai_api.cancellation import CancellationToken
from symphonai_api.events import AssistantTextDelta, CollectingSink
from symphonai_api.identity import AgentRef, RunRef, TurnRef
from symphonai_api.models import (
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    ToolCall,
    Usage,
)
from symphonai_api.providers.anthropic_provider import AnthropicProvider
from symphonai_api.providers.base import ModelProvider, ProviderError
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.providers.gemini_provider import GeminiProvider, _build_request_body
from symphonai_api.providers import openai_compatible
from symphonai_api.providers.openai_compatible import OpenAICompatibleProvider
from symphonai_api.providers.openai_provider import OpenAIProvider
from symphonai_api.runner import run_task
from symphonai_api.serialization import message_to_json
from symphonai_api.streaming import (
    StreamAssembler,
    StreamCompleted,
    TextDelta,
    ToolCallDelta,
    open_stream_with_retry,
    sse_events,
)
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


class _DefaultProvider(ModelProvider):
    @property
    def name(self) -> str:
        return "default-stream"

    @property
    def wire_format(self) -> int:
        return 4

    def create_response(self, request, *, cancel=None) -> ModelResponse:
        return ModelResponse(Message(role=Role.ASSISTANT, content="complete"))


def _empty_completion(
    *, usage: Usage | None = None, stop_reason: str = "end_turn"
) -> StreamCompleted:
    return StreamCompleted(
        ModelResponse(
            Message(role=Role.ASSISTANT),
            usage=usage or Usage(),
            stop_reason=stop_reason,
        )
    )


def _fixed_identity():
    return (
        mock.patch(
            "symphonai_api.agent_loop.new_run_ref",
            return_value=RunRef("run-fixed", "agent-fixed"),
        ),
        mock.patch(
            "symphonai_api.agent_loop.new_turn_ref",
            return_value=TurnRef("turn-fixed", "run-fixed", 1),
        ),
    )


@check("streaming.default_yields_one_completion")
def check_default_yields_one_completion() -> None:
    request = ModelRequest(messages=[Message(role=Role.USER, content="hello")])
    chunks = list(_DefaultProvider().create_response_stream(request))
    if len(chunks) != 1 or not isinstance(chunks[0], StreamCompleted):
        fail(f"default stream did not yield one completion: {chunks!r}")
    if chunks[0].response.message.text != "complete":
        fail(f"default stream changed the response: {chunks[0]!r}")
    if hasattr(chunks[0], "schema_version"):
        fail("transport chunk gained a schema_version")
    overridden = (
        OpenAIProvider,
        AnthropicProvider,
        OpenAICompatibleProvider,
        GeminiProvider,
    )
    if any(
        cls.create_response_stream is ModelProvider.create_response_stream
        for cls in overridden
    ):
        fail("a real wire-format provider did not override streaming")
    fake_chunks = list(
        FakeModelProvider(
            [ModelResponse(Message(role=Role.ASSISTANT, content="fake complete"))]
        ).create_response_stream(request)
    )
    if len(fake_chunks) != 1 or fake_chunks[0].response.message.text != "fake complete":
        fail(f"fake provider's default stream changed: {fake_chunks!r}")


@check("streaming.text_accumulates")
def check_text_accumulates() -> None:
    assembler = StreamAssembler()
    for chunk in (
        TextDelta("one"),
        TextDelta(" "),
        TextDelta("two"),
        _empty_completion(),
    ):
        assembler.add(chunk)
    if assembler.finish().message.text != "one two":
        fail("text deltas were not concatenated in arrival order")


@check("streaming.tool_fragments_by_index")
def check_tool_fragments_by_index() -> None:
    first_metadata = {"signature": {"nested": [1, {"two": 2}]}}
    assembler = StreamAssembler()
    chunks = (
        ToolCallDelta(
            1,
            id="call-one",
            name="second_tool",
            arguments_fragment='{"b":',
            vendor_id="vendor-one",
            provider_metadata=first_metadata,
        ),
        ToolCallDelta(0, id="call-zero", name="first_tool", arguments_fragment="{}"),
        ToolCallDelta(
            1,
            id=None,
            name=None,
            arguments_fragment="2}",
            vendor_id=None,
            provider_metadata={"thought": ["verbatim"]},
        ),
        _empty_completion(),
    )
    for chunk in chunks:
        assembler.add(chunk)
    calls = assembler.finish().message.tool_calls
    if [call.id for call in calls] != ["call-zero", "call-one"]:
        fail(f"tool calls were not accumulated by index: {calls!r}")
    second = calls[1]
    if (
        second.name != "second_tool"
        or second.vendor_id != "vendor-one"
        or second.arguments != {"b": 2}
        or second.provider_metadata
        != {**first_metadata, "thought": ["verbatim"]}
    ):
        fail(f"later fragments erased or replaced tool fields: {second!r}")


@check("streaming.arguments_parsed_once")
def check_arguments_parsed_once() -> None:
    real_loads = json.loads
    assembler = StreamAssembler()
    # The patch has to cover `add` as well as `finish`: parsing a fragment as
    # it arrives is the failure this check exists to catch, and a patch that
    # starts at `finish` cannot see it.
    with mock.patch("symphonai_api.streaming.json.loads", wraps=real_loads) as loads:
        assembler.add(
            ToolCallDelta(
                0,
                id="valid",
                name="valid_tool",
                arguments_fragment='{"x"',
            )
        )
        assembler.add(ToolCallDelta(0, arguments_fragment=": 1}"))
        assembler.add(_empty_completion())
        if loads.call_count != 0:
            fail(f"arguments were parsed while fragments were still arriving: {loads.call_count}")
        call = assembler.finish().message.tool_calls[0]
    if loads.call_count != 1 or call.arguments != {"x": 1}:
        fail(
            "arguments were not parsed exactly once at finish: "
            f"{loads.call_count}, {call!r}"
        )

    empty = StreamAssembler()
    empty.add(ToolCallDelta(0, id="empty", name="empty_tool"))
    empty.add(_empty_completion())
    with mock.patch("symphonai_api.streaming.json.loads", wraps=real_loads) as loads:
        empty_call = empty.finish().message.tool_calls[0]
    if loads.call_count != 0 or empty_call.arguments != {}:
        fail("an empty argument stream did not become an unparsed empty object")

    invalid = StreamAssembler()
    invalid.add(
        ToolCallDelta(7, id="invalid", name="broken_tool", arguments_fragment='{"x":')
    )
    invalid.add(_empty_completion())
    try:
        invalid.finish()
    except ProviderError as exc:
        if "broken_tool" not in str(exc) or "7" not in str(exc):
            fail(f"invalid arguments error omitted tool and index: {exc}")
    else:
        fail("invalid streamed arguments were accepted")


@check("streaming.synthesized_ids_unique")
def check_synthesized_ids_unique() -> None:
    assembler = StreamAssembler()
    assembler.add(ToolCallDelta(0, name="first"))
    assembler.add(ToolCallDelta(1, name="second"))
    assembler.add(_empty_completion())
    ids = [call.id for call in assembler.finish().message.tool_calls]
    # Uniqueness alone would accept a position-derived id like "call_0", which
    # repeats across turns; the id has to have new_id's random shape.
    shape = re.compile(r"^call_[0-9a-f]{32}$")
    if len(set(ids)) != 2 or not all(shape.fullmatch(result_id) for result_id in ids):
        fail(f"missing tool ids were not synthesized uniquely: {ids!r}")


@check("streaming.no_completion_raises")
def check_no_completion_raises() -> None:
    assembler = StreamAssembler()
    assembler.add(TextDelta("partial"))
    try:
        assembler.finish()
    except ProviderError as exc:
        if str(exc) != "stream ended without a completion event":
            fail(f"missing-completion error changed: {exc}")
    else:
        fail("assembler returned a partial response without completion")


@check("streaming.terminal_response_wins")
def check_terminal_response_wins() -> None:
    terminal_call = ToolCall("terminal-id", "terminal_tool", {"complete": True})
    terminal = ModelResponse(
        Message(
            role=Role.ASSISTANT,
            content="terminal text",
            tool_calls=[terminal_call],
        ),
        usage=Usage(input_tokens=3, output_tokens=5),
        stop_reason="tool_use",
    )
    winning = StreamAssembler()
    winning.add(TextDelta("ignored partial"))
    winning.add(ToolCallDelta(0, name="ignored", arguments_fragment="invalid"))
    winning.add(StreamCompleted(terminal))
    if winning.finish() != terminal:
        fail("complete terminal message did not win over deltas")

    usage = Usage(input_tokens=7, output_tokens=11)
    assembled = StreamAssembler()
    assembled.add(TextDelta("assembled"))
    assembled.add(
        ToolCallDelta(
            0,
            id="assembled-id",
            name="tool",
            arguments_fragment="{}",
        )
    )
    assembled.add(_empty_completion(usage=usage, stop_reason="assembled-stop"))
    response = assembled.finish()
    if (
        response.message.text != "assembled"
        or response.message.tool_calls[0].id != "assembled-id"
        or response.usage != usage
        or response.stop_reason != "assembled-stop"
    ):
        fail(
            "empty terminal did not take assembled content and terminal metadata: "
            f"{response!r}"
        )


@check("streaming.loop_matches_non_streaming")
def check_loop_matches_non_streaming() -> None:
    usage = Usage(input_tokens=2, output_tokens=4)
    complete = ModelResponse(
        Message(role=Role.ASSISTANT, content="hello world"),
        usage=usage,
        stop_reason="done",
    )
    streamed_terminal = ModelResponse(
        Message(role=Role.ASSISTANT), usage=usage, stop_reason="done"
    )
    with workspace() as ws:
        run_patch, turn_patch = _fixed_identity()
        with run_patch, turn_patch:
            plain = ApiAgent(
                FakeModelProvider([complete]),
                {},
                ws.policy,
                agent_ref=AgentRef("agent-fixed", "agent"),
            ).run([Message(role=Role.USER, content="prompt")])
            streamed = ApiAgent(
                FakeModelProvider(
                    streams=[
                        [
                            TextDelta("hello"),
                            TextDelta(" world"),
                            StreamCompleted(streamed_terminal),
                        ]
                    ]
                ),
                {},
                ws.policy,
                agent_ref=AgentRef("agent-fixed", "agent"),
                stream=True,
            ).run([Message(role=Role.USER, content="prompt")])
        if plain.messages != streamed.messages:
            fail(
                "streamed and plain conversations differ: "
                f"{plain.messages!r}, {streamed.messages!r}"
            )
        if sum(message.role == Role.ASSISTANT for message in streamed.messages) != 1:
            fail(f"streaming appended partial assistant messages: {streamed.messages!r}")

        default_plain = ApiAgent(
            FakeModelProvider(
                [complete],
                streams=[[TextDelta("wrong"), _empty_completion()]],
            ),
            {},
            ws.policy,
        ).run([Message(role=Role.USER, content="default")])
        if default_plain.final_response.message.text != "hello world":
            fail("stream=False default changed the non-streaming path")

        forwarded = run_task(
            FakeModelProvider(
                streams=[[TextDelta("forwarded"), _empty_completion()]]
            ),
            ws.policy,
            "runner",
            stream=True,
        )
        if forwarded.final_response.message.text != "forwarded":
            fail("run_task did not forward stream=True")


@check("streaming.deltas_emitted")
def check_deltas_emitted() -> None:
    secret_fragment = "half-written-secret-path"
    first = [
        TextDelta("thinking"),
        ToolCallDelta(
            0,
            id="stream-tool",
            name="unknown_tool",
            arguments_fragment=json.dumps({"path": secret_fragment}),
        ),
        _empty_completion(stop_reason="tool_use"),
    ]
    second = [TextDelta("done"), _empty_completion()]
    with workspace() as ws:
        sink = CollectingSink()
        ApiAgent(
            FakeModelProvider(streams=[first, second]),
            {},
            ws.policy,
            stream=True,
            events=sink,
        ).run([Message(role=Role.USER, content="stream")])
    deltas = sink.of_type(AssistantTextDelta)
    if [event.text for event in deltas] != ["thinking", "done"]:
        fail(f"text delta events did not match chunks: {deltas!r}")
    encoded_events = json.dumps([event.__dict__ for event in sink.events], default=str)
    if secret_fragment in encoded_events:
        fail("an event exposed partial tool arguments")


@check("streaming.dropped_events_change_nothing")
def check_dropped_events_change_nothing() -> None:
    def run_with_sink(sink):
        run_patch, turn_patch = _fixed_identity()
        with workspace() as ws, run_patch, turn_patch:
            return ApiAgent(
                FakeModelProvider(
                    streams=[[TextDelta("same"), _empty_completion()]]
                ),
                {},
                ws.policy,
                agent_ref=AgentRef("agent-fixed", "agent"),
                stream=True,
                events=sink,
            ).run([Message(role=Role.USER, content="prompt")])

    expected = run_with_sink(None)

    def dropping_sink(event) -> None:
        raise RuntimeError("drop every event")

    dropped = run_with_sink(dropping_sink)
    if (
        dropped.messages != expected.messages
        or dropped.final_response != expected.final_response
    ):
        fail("exceptions from the event sink changed the streamed result")


class _LinesResponse:
    def __init__(self, lines) -> None:
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def __iter__(self):
        for chunk in self._lines:
            yield from chunk.splitlines(keepends=True)


class _BytesResponse(_LinesResponse):
    def __init__(self, body: bytes) -> None:
        super().__init__([])
        self._body = body

    def read(self) -> bytes:
        return self._body


def _assemble(chunks) -> ModelResponse:
    assembler = StreamAssembler()
    for chunk in chunks:
        assembler.add(chunk)
    return assembler.finish()


def _request() -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content="stream please")])


def _anthropic_transcript() -> list[bytes]:
    return [
        b'event: message_start\n',
        b'data: {"message":{"usage":{"input_tokens":3}}}\n\n',
        b'event: content_block_start\n',
        b'data: {"index":0,"content_block":{"type":"text"}}\n\n',
        b'event: content_block_delta\n',
        b'data: {"index":0,"delta":{"type":"text_delta","text":"hello "}}\n\n',
        b'event: content_block_start\n',
        b'data: {"index":1,"content_block":{"type":"tool_use","id":"tool-1","name":"weather"}}\n\n',
        b'event: content_block_delta\n',
        b'data: {"index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":\\"Seoul\\"}"}}\n\n',
        b'event: content_block_delta\n',
        b'data: {"index":0,"delta":{"type":"text_delta","text":"world"}}\n\n',
        b'event: message_delta\n',
        b'data: {"delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}\n\n',
        b'event: message_stop\n',
        b'data: {}\n\n',
    ]


def _openai_transcript() -> list[bytes]:
    return [
        b'data: {"choices":[{"delta":{"content":"hello "},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tool-1","function":{"name":"weather","arguments":"{\\"city\\":"}}]},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"world","tool_calls":[{"index":0,"function":{"arguments":"\\"Seoul\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":5}}\n\n',
        b'data: [DONE]\n\n',
    ]


@check("streaming.retry_before_first_line_only")
def check_retry_before_first_line_only() -> None:
    request = urllib.request.Request("https://stream.invalid")
    with mock.patch(
        "symphonai_api.streaming.urllib.request.urlopen",
        side_effect=[urllib.error.URLError("temporary"), _LinesResponse([b"line\n"])],
    ) as opener, mock.patch(
        "symphonai_api.streaming._wait_before_retry", return_value=True
    ):
        lines = list(
            open_stream_with_retry(
                request,
                timeout=5,
                max_attempts=3,
                api_key="key",
                operation="test",
            )
        )
    if lines != [b"line\n"] or opener.call_count != 2:
        fail(f"pre-line failure did not retry once: {lines!r}, {opener.call_count}")

    api_key = "stream-secret-key"

    class _FailsAfterLine(_LinesResponse):
        def __iter__(self):
            yield b"first\n"
            raise urllib.error.URLError(f"failure contains {api_key}")

    with mock.patch(
        "symphonai_api.streaming.urllib.request.urlopen",
        return_value=_FailsAfterLine([]),
    ) as opener:
        iterator = open_stream_with_retry(
            request,
            timeout=5,
            max_attempts=3,
            api_key=api_key,
            operation="test",
        )
        if next(iterator) != b"first\n":
            fail("stream did not yield its first line")
        try:
            next(iterator)
        except ProviderError as exc:
            if api_key in str(exc) or "[redacted]" not in str(exc):
                fail(f"streaming error exposed its API key: {exc}")
        else:
            fail("post-line stream failure did not raise")
    if opener.call_count != 1:
        fail(f"post-line failure replayed the stream {opener.call_count} times")


@check("streaming.cancel_mid_stream")
def check_cancel_mid_stream() -> None:
    token = CancellationToken()

    class _CancellingStream(FakeModelProvider):
        def create_response_stream(self, request, *, cancel=None):
            yield TextDelta("partial")
            assert cancel is not None
            cancel.cancel()
            yield _empty_completion()

    with workspace() as ws:
        result = ApiAgent(
            _CancellingStream(),
            {},
            ws.policy,
            stream=True,
        ).run(
            [Message(role=Role.USER, content="cancel")],
            cancel=token,
        )
    if result.stopped_reason != "cancelled":
        fail(f"mid-stream cancellation escaped the agent boundary: {result!r}")
    if any(message.role == Role.ASSISTANT for message in result.messages):
        fail(f"partial streamed response entered the conversation: {result.messages!r}")


@check("streaming.sse_parsing")
def check_sse_parsing() -> None:
    lines = [
        b": keepalive\n",
        b"event: content\r\n",
        b"data: first\n",
        b"data: second\n",
        b"\n",
        b"data: [DONE]\n",
        b"\n",
    ]
    events = list(sse_events(lines))
    if events != [("content", "first\nsecond"), ("message", "[DONE]")]:
        fail(f"SSE lines were parsed incorrectly: {events!r}")


@check("streaming.anthropic_matches_non_streaming")
def check_anthropic_matches_non_streaming() -> None:
    body = {
        "content": [
            {"type": "text", "text": "hello world"},
            {"type": "tool_use", "id": "tool-1", "name": "weather", "input": {"city": "Seoul"}},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 5},
        "stop_reason": "tool_use",
    }
    with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthropic-stream-key"}):
        with mock.patch(
            "urllib.request.urlopen", return_value=_BytesResponse(json.dumps(body).encode())
        ):
            expected = AnthropicProvider().create_response(_request())
        with mock.patch(
            "urllib.request.urlopen", return_value=_LinesResponse(_anthropic_transcript())
        ):
            actual = _assemble(AnthropicProvider().create_response_stream(_request()))
    if actual != expected:
        fail(f"Anthropic streaming response differed from non-streaming: {actual!r}")


@check("streaming.openai_matches_non_streaming")
def check_openai_matches_non_streaming() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "content": "hello world",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "function": {
                                "name": "weather",
                                "arguments": json.dumps({"city": "Seoul"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5},
    }
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "openai-stream-key"}):
        with mock.patch(
            "urllib.request.urlopen", return_value=_BytesResponse(json.dumps(body).encode())
        ):
            expected = OpenAIProvider().create_response(_request())
        with mock.patch(
            "urllib.request.urlopen", return_value=_LinesResponse(_openai_transcript())
        ):
            actual = _assemble(OpenAIProvider().create_response_stream(_request()))
    if actual != expected:
        fail(f"OpenAI streaming response differed from non-streaming: {actual!r}")


@check("streaming.compatible_shares_openai_mapping")
def check_compatible_shares_openai_mapping() -> None:
    environment = "SYMPHONAI_COMPATIBLE_STREAM_KEY"
    provider = OpenAICompatibleProvider(
        api_key_env_var=environment, base_url="https://compatible.invalid/v1"
    )
    with mock.patch.dict(os.environ, {environment: "compatible-stream-key"}):
        with mock.patch(
            "urllib.request.urlopen", return_value=_LinesResponse(_openai_transcript())
        ), mock.patch.object(
            openai_compatible,
            "_openai_stream_chunks",
            wraps=openai_compatible._openai_stream_chunks,
        ) as mapping:
            response = _assemble(provider.create_response_stream(_request()))
    if not mapping.called or response.message.tool_calls[0].vendor_id != "tool-1":
        fail("OpenAI-compatible streaming did not call the shared OpenAI mapper")


@check("streaming.anthropic_partial_json")
def check_anthropic_partial_json() -> None:
    transcript = [
        b'event: message_start\n',
        b'data: {"message":{"usage":{"input_tokens":1}}}\n\n',
        b'event: content_block_start\n',
        b'data: {"index":2,"content_block":{"type":"tool_use","id":"tool-partial","name":"search"}}\n\n',
        b'event: content_block_delta\n',
        b'data: {"index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"query\\":"}}\n\n',
        b'event: content_block_delta\n',
        b'data: {"index":2,"delta":{"type":"input_json_delta","partial_json":"\\"cats\\"}"}}\n\n',
        b'event: message_delta\n',
        b'data: {"delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":1}}\n\n',
        b'event: message_stop\n',
        b'data: {}\n\n',
    ]
    with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthropic-partial-key"}):
        with mock.patch("urllib.request.urlopen", return_value=_LinesResponse(transcript)):
            response = _assemble(AnthropicProvider().create_response_stream(_request()))
    if response.message.tool_calls[0].arguments != {"query": "cats"}:
        fail(f"Anthropic partial JSON was not assembled: {response!r}")


@check("streaming.openai_parallel_tool_calls")
def check_openai_parallel_tool_calls() -> None:
    transcript = [
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"second","function":{"name":"second_tool","arguments":"{\\"b\\":"}},{"index":0,"id":"first","function":{"name":"first_tool","arguments":"{\\"a\\":"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}},{"index":1,"function":{"arguments":"2}"}}]},"finish_reason":"tool_calls"}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "openai-parallel-key"}):
        with mock.patch("urllib.request.urlopen", return_value=_LinesResponse(transcript)):
            calls = _assemble(OpenAIProvider().create_response_stream(_request())).message.tool_calls
    if [(call.id, call.name, call.arguments) for call in calls] != [
        ("first", "first_tool", {"a": 1}),
        ("second", "second_tool", {"b": 2}),
    ]:
        fail(f"OpenAI tool calls were not assembled by their indexes: {calls!r}")


@check("streaming.anthropic_error_event")
def check_anthropic_error_event() -> None:
    key = "anthropic-secret-to-redact"
    transcript = [
        b'event: error\n',
        b'data: {"error":{"message":"vendor rejected anthropic-secret-to-redact"}}\n\n',
    ]
    with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": key}):
        with mock.patch("urllib.request.urlopen", return_value=_LinesResponse(transcript)):
            try:
                list(AnthropicProvider().create_response_stream(_request()))
            except ProviderError as exc:
                if "vendor rejected" not in str(exc) or key in str(exc):
                    fail(f"Anthropic error did not retain/redact the vendor message: {exc}")
            else:
                fail("Anthropic SSE error event did not raise ProviderError")


@check("streaming.openai_empty_choices_ignored")
def check_openai_empty_choices_ignored() -> None:
    transcript = [
        b'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":9}}\n\n',
        b'data: [DONE]\n\n',
    ]
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "openai-empty-key"}):
        with mock.patch("urllib.request.urlopen", return_value=_LinesResponse(transcript)):
            response = _assemble(OpenAIProvider().create_response_stream(_request()))
    if response.message.text != "done" or response.usage != Usage(7, 9):
        fail(f"OpenAI usage-only payload was not accepted: {response!r}")


@check("streaming.compatible_without_stream_usage")
def check_compatible_without_stream_usage() -> None:
    environment = "SYMPHONAI_COMPATIBLE_NO_USAGE_KEY"
    captured: dict[str, object] = {}

    def open_stream(request, timeout=None):
        captured.update(json.loads(request.data.decode("utf-8")))
        return _LinesResponse([
            b'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n',
        ])

    provider = OpenAICompatibleProvider(
        api_key_env_var=environment,
        base_url="https://compatible.invalid/v1",
        stream_usage=False,
    )
    with mock.patch.dict(os.environ, {environment: "compatible-no-usage-key"}):
        with mock.patch("urllib.request.urlopen", side_effect=open_stream):
            response = _assemble(provider.create_response_stream(_request()))
    if response.usage != Usage() or "stream_options" in captured:
        fail(f"compatible stream_usage=False changed the request or usage: {captured!r}, {response!r}")


@check("streaming.truncated_stream_raises")
def check_truncated_stream_raises() -> None:
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "openai-truncated-key"}):
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_LinesResponse([b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n']),
        ):
            chunks = OpenAIProvider().create_response_stream(_request())
            try:
                _assemble(chunks)
            except ProviderError as exc:
                if str(exc) != "stream ended without a completion event":
                    fail(f"truncated stream raised the wrong error: {exc}")
            else:
                fail("truncated stream returned a partial response")


@check("streaming.stream_flag_present")
def check_stream_flag_present() -> None:
    providers = [
        (AnthropicProvider(), "ANTHROPIC_API_KEY", "anthropic-flag-key", _anthropic_transcript()),
        (OpenAIProvider(), "OPENAI_API_KEY", "openai-flag-key", _openai_transcript()),
        (OpenAICompatibleProvider("SYMPHONAI_FLAG_COMPATIBLE", "https://compatible.invalid/v1"), "SYMPHONAI_FLAG_COMPATIBLE", "compatible-flag-key", _openai_transcript()),
    ]
    for provider, environment, key, transcript in providers:
        bodies: list[dict] = []

        def non_stream(request, timeout=None):
            bodies.append(json.loads(request.data.decode("utf-8")))
            if isinstance(provider, AnthropicProvider):
                return _BytesResponse(b'{"content":[],"usage":{},"stop_reason":"end_turn"}')
            return _BytesResponse(b'{"choices":[{"message":{},"finish_reason":"stop"}],"usage":{}}')

        with mock.patch.dict(os.environ, {environment: key}):
            with mock.patch("urllib.request.urlopen", side_effect=non_stream):
                provider.create_response(_request())
            def stream(request, timeout=None):
                bodies.append(json.loads(request.data.decode("utf-8")))
                return _LinesResponse(transcript)

            with mock.patch("urllib.request.urlopen", side_effect=stream):
                _assemble(provider.create_response_stream(_request()))
        if "stream" in bodies[0]:
            fail(f"non-streaming {provider.name} request unexpectedly set stream: {bodies[0]!r}")
        if bodies[1].get("stream") is not True:
            fail(f"streaming {provider.name} request omitted stream=true: {bodies[1]!r}")


# Captured from commit 9d5b88d -- the last tree before streaming existed -- by
# running each provider's create_response against the bodies below. Frozen
# rather than recomputed from `git archive HEAD`: that would compare this
# change with itself once committed, and fails outright in a tree with no .git,
# which is exactly what publish.sh checks before pushing.
_NON_STREAMING_BEFORE_05 = """
{
  "anthropic": {"stop_reason": "tool_use", "usage": [3, 5]},
  "openai": {"stop_reason": "tool_calls", "usage": [3, 5]},
  "compatible": {"stop_reason": "tool_calls", "usage": [3, 5]},
  "gemini": {"stop_reason": "STOP", "usage": [3, 5]}
}
"""
_GEMINI_MESSAGE_BEFORE_05 = """
{
  "content": [{"kind": "text", "schema_version": 1, "text": "hello world"}],
  "role": "assistant",
  "schema_version": 1,
  "tool_calls": [{"arguments": {"city": "Seoul"}, "id": "tool-1", "name": "weather",
                  "provider_metadata": {"thoughtSignature": "frozen-signature"},
                  "schema_version": 1, "vendor_id": "tool-1"}],
  "tool_result": null,
  "turn_id": null
}
"""
_NON_STREAMING_MESSAGE_BEFORE_05 = """
{
  "content": [{"kind": "text", "schema_version": 1, "text": "hello world"}],
  "role": "assistant",
  "schema_version": 1,
  "tool_calls": [{"arguments": {"city": "Seoul"}, "id": "tool-1", "name": "weather",
                  "provider_metadata": {}, "schema_version": 1, "vendor_id": "tool-1"}],
  "tool_result": null,
  "turn_id": null
}
"""


@check("streaming.non_streaming_path_unchanged")
def check_non_streaming_path_unchanged() -> None:
    anthropic_body = {
        "content": [
            {"type": "text", "text": "hello world"},
            {"type": "tool_use", "id": "tool-1", "name": "weather", "input": {"city": "Seoul"}},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 5},
        "stop_reason": "tool_use",
    }
    openai_body = {
        "choices": [
            {
                "message": {
                    "content": "hello world",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "function": {
                                "name": "weather",
                                "arguments": json.dumps({"city": "Seoul"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5},
    }
    gemini_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "hello world"},
                        {
                            "functionCall": {
                                "id": "tool-1",
                                "name": "weather",
                                "args": {"city": "Seoul"},
                            },
                            "thoughtSignature": "frozen-signature",
                        },
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5},
    }
    expected = json.loads(_NON_STREAMING_BEFORE_05)
    expected_message = json.loads(_NON_STREAMING_MESSAGE_BEFORE_05)
    gemini_message = json.loads(_GEMINI_MESSAGE_BEFORE_05)
    cases = (
        ("anthropic", AnthropicProvider(), "ANTHROPIC_API_KEY", anthropic_body, expected_message),
        ("openai", OpenAIProvider(), "OPENAI_API_KEY", openai_body, expected_message),
        (
            "compatible",
            OpenAICompatibleProvider("SYMPHONAI_FROZEN", "https://compatible.invalid/v1"),
            "SYMPHONAI_FROZEN",
            openai_body,
            expected_message,
        ),
        ("gemini", GeminiProvider(), "GEMINI_API_KEY", gemini_body, gemini_message),
    )
    for label, provider, environment, body, expected_message in cases:
        with mock.patch.dict(os.environ, {environment: "frozen-key"}):
            with mock.patch(
                "urllib.request.urlopen",
                return_value=_BytesResponse(json.dumps(body).encode()),
            ):
                response = provider.create_response(_request())
        produced = {
            "stop_reason": response.stop_reason,
            "usage": [response.usage.input_tokens, response.usage.output_tokens],
        }
        if produced != expected[label] or message_to_json(response.message) != expected_message:
            fail(
                f"{label} non-streaming parse changed since 9d5b88d: "
                f"{produced!r}, {message_to_json(response.message)!r}"
            )


@check("streaming.openai_error_event")
def check_openai_error_event() -> None:
    key = "openai-secret-to-redact"
    transcript = [
        b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
        b'data: {"error":{"message":"rate limit exceeded for openai-secret-to-redact"}}\n\n',
    ]
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": key}):
        with mock.patch("urllib.request.urlopen", return_value=_LinesResponse(transcript)):
            try:
                _assemble(OpenAIProvider().create_response_stream(_request()))
            except ProviderError as exc:
                if "rate limit exceeded" not in str(exc) or key in str(exc):
                    fail(f"OpenAI error did not retain/redact the vendor message: {exc}")
            else:
                fail("OpenAI SSE error payload did not raise ProviderError")


def _gemini_transcript(*, signature: bool = True, two_calls: bool = False) -> list[bytes]:
    if two_calls:
        return [
            b'data: {"candidates":[{"content":{"parts":[{"text":"hello "},{"functionCall":{"name":"first","args":{"a":1}}},{"functionCall":{"name":"second","args":{"b":2}}},{"text":"world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":5}}\n\n'
        ]
    if signature:
        return [
            b'data: {"candidates":[{"content":{"parts":[{"text":"hello "},{"functionCall":{"id":"tool-1","name":"weather","args":{"city":"Seoul"}},"thoughtSignature":"streamed-thought"},{"text":"world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":5}}\n\n'
        ]
    return [
        b'data: {"candidates":[{"content":{"parts":[{"text":"hello "},{"functionCall":{"id":"tool-1","name":"weather","args":{"city":"Seoul"}}},{"text":"world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":5}}\n\n'
    ]


@check("streaming.gemini_matches_non_streaming")
def check_gemini_matches_non_streaming() -> None:
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "hello "},
                        {"functionCall": {"id": "tool-1", "name": "weather", "args": {"city": "Seoul"}}},
                        {"text": "world"},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5},
    }
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-stream-key"}):
        with mock.patch(
            "symphonai_api.providers.gemini_provider._synthesize_tool_call_id",
            return_value="call-fixed",
        ), mock.patch("symphonai_api.streaming.new_id", return_value="call-fixed"):
            with mock.patch(
                "urllib.request.urlopen", return_value=_BytesResponse(json.dumps(body).encode())
            ):
                expected = GeminiProvider().create_response(_request())
            with mock.patch(
                "urllib.request.urlopen", return_value=_LinesResponse(_gemini_transcript(signature=False))
            ):
                actual = _assemble(GeminiProvider().create_response_stream(_request()))
    if actual != expected:
        fail(f"Gemini streaming response differed from non-streaming: {actual!r}")


@check("streaming.gemini_thought_signature_round_trip")
def check_gemini_thought_signature_round_trip() -> None:
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-thought-key"}):
        with mock.patch(
            "urllib.request.urlopen", return_value=_LinesResponse(_gemini_transcript())
        ):
            response = _assemble(GeminiProvider().create_response_stream(_request()))
    tool_call = response.message.tool_calls[0]
    if tool_call.provider_metadata.get("thoughtSignature") != "streamed-thought":
        fail(f"streamed thoughtSignature was not retained: {tool_call!r}")
    next_body = _build_request_body(
        ModelRequest(messages=[Message(role=Role.USER, content="continue"), response.message])
    )
    outgoing_part = next(
        part
        for part in next_body["contents"][1]["parts"]
        if "functionCall" in part
    )
    if outgoing_part.get("thoughtSignature") != "streamed-thought":
        fail(f"next Gemini request dropped thoughtSignature: {outgoing_part!r}")


@check("streaming.gemini_no_signature_no_key")
def check_gemini_no_signature_no_key() -> None:
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-no-signature-key"}):
        with mock.patch(
            "urllib.request.urlopen", return_value=_LinesResponse(_gemini_transcript(signature=False))
        ):
            response = _assemble(GeminiProvider().create_response_stream(_request()))
    metadata = response.message.tool_calls[0].provider_metadata
    if "thoughtSignature" in metadata:
        fail(f"signature-free Gemini part invented a thoughtSignature: {metadata!r}")


@check("streaming.gemini_two_calls")
def check_gemini_two_calls() -> None:
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-two-calls-key"}):
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_LinesResponse(_gemini_transcript(signature=False, two_calls=True)),
        ):
            chunks = list(GeminiProvider().create_response_stream(_request()))
    real_loads = json.loads
    with mock.patch("symphonai_api.streaming.json.loads", wraps=real_loads) as loads:
        calls = _assemble(chunks).message.tool_calls
    if loads.call_count != 2 or [call.name for call in calls] != ["first", "second"]:
        fail(f"Gemini calls were not parsed once each in arrival order: {calls!r}")
    if len({call.id for call in calls}) != 2 or [call.arguments for call in calls] != [{"a": 1}, {"b": 2}]:
        fail(f"Gemini calls were not distinct assembled calls: {calls!r}")


@check("streaming.gemini_block_reason")
def check_gemini_block_reason() -> None:
    key = "gemini-block-key"
    transcript = [b'data: {"promptFeedback":{"blockReason":"SAFETY gemini-block-key"}}\n\n']
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": key}):
        with mock.patch("urllib.request.urlopen", return_value=_LinesResponse(transcript)):
            try:
                _assemble(GeminiProvider().create_response_stream(_request()))
            except ProviderError as exc:
                if "SAFETY" not in str(exc) or key in str(exc):
                    fail(f"Gemini block reason was omitted or leaked the key: {exc}")
            else:
                fail("Gemini prompt block did not raise ProviderError")


@check("streaming.gemini_truncated_stream")
def check_gemini_truncated_stream() -> None:
    class _BrokenGeminiResponse(_LinesResponse):
        def __iter__(self):
            yield b'data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}\n'
            yield b"\n"
            raise ConnectionError("stream closed early")

    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-truncated-key"}):
        with mock.patch("urllib.request.urlopen", return_value=_BrokenGeminiResponse([])):
            try:
                _assemble(GeminiProvider().create_response_stream(_request()))
            except ProviderError:
                pass
            else:
                fail("truncated Gemini stream returned a partial response")


@check("streaming.gemini_url_has_no_key")
def check_gemini_url_has_no_key() -> None:
    key = "gemini-url-secret"
    captured: list[str] = []

    def open_stream(request, timeout=None):
        captured.append(request.full_url)
        return _LinesResponse(_gemini_transcript(signature=False))

    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": key}):
        with mock.patch("urllib.request.urlopen", side_effect=open_stream):
            _assemble(GeminiProvider().create_response_stream(_request()))
    if captured != [
        f"{GeminiProvider().base_url}/models/{GeminiProvider().model}:streamGenerateContent?alt=sse"
    ] or key in captured[0]:
        fail(f"Gemini streaming URL was incorrect or exposed the API key: {captured!r}")
