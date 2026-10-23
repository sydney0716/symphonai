"""Record/replay checks against the real provider implementations."""

from __future__ import annotations

import copy
import getpass
import json
import os
import tempfile
import unittest.mock as mock
from pathlib import Path
from typing import Any
from urllib.request import Request

from scripts.checks.harness import check, fail, ok
from symphonai_api.agent_loop import ApiAgent
from symphonai_api.models import Message, Role
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.anthropic_provider import API_KEY_ENV_VAR as ANTHROPIC_KEY, AnthropicProvider
from symphonai_api.providers.gemini_provider import API_KEY_ENV_VAR as GEMINI_KEY, GeminiProvider
from symphonai_api.providers.openai_provider import API_KEY_ENV_VAR as OPENAI_KEY, OpenAIProvider
from symphonai_api.runner import standard_tool_registry
from symphonai_api.tool_schema import tool_registry_schemas

CASSETTE_DIR = Path(__file__).parent / "cassettes"
SYNTHETIC_CASSETTE = "synthetic-openai-responses.json"
FILES = {
    "notes.txt": "The launch code is amber-otter-73041.\n",
    "field-observations.txt": "This file is discoverable only by listing the project.\n",
}
RECORDER_TASK = (
    "Read notes.txt first. On your next turn, run the harmless command `ls`. "
    "Then answer with the note's exact sentence and every file name from the command output. "
    "Make exactly one tool call per turn."
)
PROVIDER_KEYS = {"openai": OPENAI_KEY, "anthropic": ANTHROPIC_KEY, "gemini": GEMINI_KEY}


class _ReplayResponse:
    def __init__(self, status: int, body: str = "", lines: list[str] | None = None) -> None:
        self.status = status
        self._body = body.encode("utf-8")
        self._lines = [line.encode("utf-8") for line in lines or []]

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self) -> _ReplayResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _provider(name: str, model: str):
    if name == "openai":
        return OpenAIProvider(model=model)
    if name == "anthropic":
        return AnthropicProvider(model=model)
    if name == "gemini":
        return GeminiProvider(model=model)
    raise ValueError(f"unsupported cassette provider: {name}")


def _tools(names: list[str]):
    tools = standard_tool_registry(names=names)
    return tools


def _replace_root(value: Any, root: str) -> Any:
    if isinstance(value, str):
        return value.replace(root, "<PROJECT_ROOT>")
    if isinstance(value, list):
        return [_replace_root(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _replace_root(item, root) for key, item in value.items()}
    return value


def _request_record(request: Request, root: str) -> dict[str, Any]:
    raw = request.data or b"{}"
    body = json.loads(raw.decode("utf-8"))
    return {
        "method": request.get_method(),
        "url": _replace_root(request.full_url, root),
        "body": _replace_root(body, root),
    }


def privacy_violations(cassette: dict[str, Any], api_key: str, home: str, username: str) -> list[str]:
    serialized = json.dumps(cassette, ensure_ascii=False)
    violations = [
        label
        for label, value in (("API key", api_key), ("home path", home), ("user name", username))
        if value and value in serialized
    ]
    temp_segments = {
        segment for segment in Path(tempfile.gettempdir()).parts if len(segment) >= 8
    }
    violations.extend(
        f"temporary directory segment ({segment})"
        for segment in sorted(temp_segments)
        if segment in serialized
    )
    if "/var/folders" in serialized:
        violations.append("temporary directory path (/var/folders)")
    return violations


def require_private(cassette: dict[str, Any], api_key: str, home: str, username: str) -> None:
    violations = privacy_violations(cassette, api_key, home, username)
    if violations:
        raise ValueError("cassette contains private values: " + ", ".join(violations))


def _string_leaves(value: Any, location: str = ""):
    if isinstance(value, str):
        yield location or "$", value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _string_leaves(item, f"{location}.{key}" if location else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _string_leaves(item, f"{location}[{index}]")


def needed_back_values(cassette: dict[str, Any]) -> list[tuple[int, int, str, str]]:
    exchanges = cassette["exchanges"]
    request_texts = [
        json.dumps(exchange["request"].get("body", {}), ensure_ascii=False)
        for exchange in exchanges
    ]
    needed = []
    for earlier, exchange in enumerate(exchanges):
        response = exchange["response"]
        response_values = []
        if "body_lines" in response:
            for line in response["body_lines"]:
                if line.startswith("data: "):
                    try:
                        response_values.append(json.loads(line[6:].strip()))
                    except json.JSONDecodeError:
                        continue
        else:
            try:
                response_values.append(json.loads(response.get("body", "")))
            except json.JSONDecodeError:
                pass
        for later in range(earlier + 1, len(exchanges)):
            for response_value in response_values:
                for location, value in _string_leaves(response_value, f"exchanges[{earlier}].response"):
                    appeared_in_request = any(value in request_texts[index] for index in range(earlier + 1))
                    if len(value) >= 16 and not appeared_in_request and value in request_texts[later]:
                        item = (earlier, later, location, value)
                        if item not in needed:
                            needed.append(item)
    return needed


def replay_cassette(
    cassette: dict[str, Any], request_transform=None,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    provider_name = cassette["provider"]
    api_key_name = PROVIDER_KEYS[provider_name]
    outgoing: list[dict[str, Any]] = []
    index = 0

    def replay_urlopen(request: Request, timeout=None):  # noqa: ANN001
        nonlocal index
        if index >= len(cassette["exchanges"]):
            raise AssertionError("replay made more provider calls than the cassette")
        record = _request_record(request, str(_current_root[0]))
        if request_transform is not None:
            record = request_transform(copy.deepcopy(record))
        outgoing.append(record)
        response = cassette["exchanges"][index]["response"]
        index += 1
        return _ReplayResponse(response["status"], response.get("body", ""), response.get("body_lines"))

    dummy_key = "vendor-replay-dummy-key-0123456789"
    _current_root: list[str] = []
    with tempfile.TemporaryDirectory(prefix="symphonai-vendor-replay-") as temporary:
        _current_root.append(str(Path(temporary).resolve()))
        for relative, content in cassette.get("files", {}).items():
            destination = Path(temporary) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        tools = _tools(cassette["tools"])
        policy = PermissionPolicy(
            repo_root=Path(temporary), allowed_write_scope=[Path(temporary)],
            shell_enabled=True, shell_allowlist=[("ls",)],
        )
        provider = _provider(provider_name, cassette["model"])
        agent = ApiAgent(
            provider, tools, policy, max_turns=6,
            tool_schemas=tool_registry_schemas(tools, provider.wire_format), stream=True,
        )
        with mock.patch.dict(os.environ, {api_key_name: dummy_key}), mock.patch(
            "urllib.request.urlopen", side_effect=replay_urlopen,
        ):
            result = agent.run([Message(Role.USER, cassette["task"])])
    failures = []
    for earlier, later, location, value in needed_back_values(cassette):
        request_text = json.dumps(outgoing[later].get("body", {}), ensure_ascii=False) if later < len(outgoing) else ""
        if value not in request_text:
            failures.append(f"{location} (needed in exchange[{later}] request)")
    if index != len(cassette["exchanges"]):
        failures.append(f"replayed {index} of {len(cassette['exchanges'])} exchanges")
    if result.final_response.message.text != cassette["final_answer"]:
        failures.append("final answer differed from the cassette")
    if len(outgoing) != len(cassette["exchanges"]):
        failures.append(f"replayed {len(outgoing)} model calls, expected {len(cassette['exchanges'])}")
    return outgoing, result.final_response.message.text, failures


def _cassette_capture(provider_name: str, model: str, key: str, output: Path) -> None:
    key_env = PROVIDER_KEYS[provider_name]
    provider = _provider(provider_name, model)
    tools = _tools(["read_file", "run_shell"])
    exchanges: list[dict[str, Any]] = []
    project_root: list[str] = []
    original_urlopen = __import__("urllib.request", fromlist=["urlopen"]).urlopen

    class CaptureResponse:
        def __init__(self, response, exchange):  # noqa: ANN001
            self.response = response
            self.exchange = exchange
            self.status = getattr(response, "status", response.getcode())

        def getcode(self):
            return self.status

        def read(self):
            raw = self.response.read()
            self.exchange["response"] = {"status": self.status, "body": raw.decode("utf-8")}
            return raw

        def __iter__(self):
            lines = []
            try:
                for line in self.response:
                    lines.append(line.decode("utf-8"))
                    yield line
            finally:
                self.exchange["response"] = {"status": self.status, "body_lines": lines}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self.response.__exit__(*exc)

    def capture_urlopen(request, timeout=None):  # noqa: ANN001
        record = _request_record(request, project_root[0])
        exchange: dict[str, Any] = {"request": record}
        exchanges.append(exchange)
        return CaptureResponse(original_urlopen(request, timeout=timeout), exchange)

    with tempfile.TemporaryDirectory(prefix="symphonai-vendor-project-") as temporary:
        root = Path(temporary).resolve()
        project_root.append(str(root))
        for relative, content in FILES.items():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        policy = PermissionPolicy(
            repo_root=root, allowed_write_scope=[root], shell_enabled=True,
            shell_allowlist=[("ls",)],
        )
        agent = ApiAgent(
            provider, tools, policy, max_turns=6,
            tool_schemas=tool_registry_schemas(tools, provider.wire_format), stream=True,
        )
        with mock.patch("urllib.request.urlopen", side_effect=capture_urlopen):
            result = agent.run([Message(Role.USER, RECORDER_TASK)])
        for exchange in exchanges:
            exchange["request"] = _replace_root(exchange["request"], str(root))
            exchange["response"] = _replace_root(exchange.get("response", {}), str(root))

    tool_calls = [
        call for message in result.messages if message.role == Role.ASSISTANT
        for call in message.tool_calls
    ]
    if (
        len(exchanges) < 3 or len(tool_calls) < 2 or result.turns_used < 3
        or not {"read_file", "run_shell"}.issubset({call.name for call in tool_calls})
    ):
        raise RuntimeError("recording task did not produce two tool calls across multiple turns")
    cassette = {
        "version": 1,
        "provider": provider_name,
        "model": model,
        "task": RECORDER_TASK,
        "files": FILES,
        "tools": ["read_file", "run_shell"],
        "final_answer": _replace_root(result.final_response.message.text, str(root)),
        "exchanges": exchanges,
    }
    require_private(cassette, key, str(Path.home()), getpass.getuser())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cassette, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@check("vendor_replay.harness_catches_a_dropped_value")
def check_harness_catches_a_dropped_value() -> None:
    cassette = json.loads((CASSETTE_DIR / SYNTHETIC_CASSETTE).read_text(encoding="utf-8"))
    outgoing, answer, failures = replay_cassette(cassette)
    if failures:
        fail(f"normal replay failed: {failures!r}")
    opaque = needed_back_values(cassette)[0][3]

    def strip_value(record: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(record, ensure_ascii=False).replace(opaque, "")
        return json.loads(encoded)

    _dropped_requests, _dropped_answer, dropped_failures = replay_cassette(cassette, strip_value)
    location = needed_back_values(cassette)[0][2]
    if not any(location in failure for failure in dropped_failures):
        fail(f"dropped value was not reported at {location}: {dropped_failures!r}")
    if answer != cassette["final_answer"] or len(outgoing) != 2:
        fail("normal replay did not preserve the final answer and two model calls")
    ok("real OpenAI Responses provider round-trips the opaque call id")
    ok(f"stripping the value fails with its response location: {location}")

    fake_key, fake_home, fake_user = "fake-api-key-123", "/fake/home/path", "fake-user-name"
    for field, value in (("key", fake_key), ("home", fake_home), ("user", fake_user)):
        try:
            require_private({"field": value}, fake_key if field == "key" else "", fake_home if field == "home" else "", fake_user if field == "user" else "")
        except ValueError:
            continue
        fail(f"privacy refusal accepted a cassette containing the fake {field}")
    ok("recorder refuses API key, home path, and username values")


@check("vendor_replay.cassettes")
def check_cassettes() -> None:
    paths = sorted(path for path in CASSETTE_DIR.glob("*.json") if path.name != SYNTHETIC_CASSETTE)
    if not paths:
        ok("nothing to replay: no recorded vendor cassettes are present")
        return
    for path in paths:
        cassette = json.loads(path.read_text(encoding="utf-8"))
        _outgoing, _answer, failures = replay_cassette(cassette)
        if failures:
            fail(f"{path.name}: {failures!r}")
        ok(f"replayed {path.name}")


def _openai_response(output: list[dict[str, Any]]) -> dict[str, Any]:
    event = {"type": "response.completed", "response": {"status": "completed", "output": output}}
    return {
        "status": 200,
        "body_lines": [f"event: response.completed\n", f"data: {json.dumps(event)}\n", "\n"],
    }


@check("vendor_replay.privacy_rejects_temp_path_fragments")
def check_privacy_rejects_temp_path_fragments() -> None:
    segments = [segment for segment in Path(tempfile.gettempdir()).parts if len(segment) >= 8]
    for fragment in [*segments, "/var/folders"]:
        violations = privacy_violations({"response": fragment}, "", "", "")
        if not violations:
            fail(f"privacy check accepted temporary path fragment {fragment!r}")
    if privacy_violations({"response": "safe cassette content"}, "", "", ""):
        fail("privacy check rejected content with no temporary path fragment")
    ok("privacy check rejects split temp-directory segments and /var/folders")


@check("vendor_replay.recording_task_replays")
def check_recording_task_replays() -> None:
    filenames = sorted(FILES)
    answer = f"{FILES['notes.txt'].strip()} Files: {', '.join(filenames)}"
    cassette = {
        "version": 1,
        "provider": "openai",
        "model": "gpt-6.1-sol",
        "task": RECORDER_TASK,
        "files": FILES,
        "tools": ["read_file", "run_shell"],
        "final_answer": answer,
        "exchanges": [
            {"request": {"method": "POST", "url": "https://api.openai.com/v1/responses", "body": {}},
             "response": _openai_response([{
                 "type": "function_call", "name": "read_file", "call_id": "call_recording_read_0123456789",
                 "arguments": json.dumps({"path": "notes.txt"}),
             }])},
            {"request": {"method": "POST", "url": "https://api.openai.com/v1/responses", "body": {}},
             "response": _openai_response([{
                 "type": "function_call", "name": "run_shell", "call_id": "call_recording_ls_0123456789",
                 "arguments": json.dumps({"command": "ls"}),
             }])},
            {"request": {"method": "POST", "url": "https://api.openai.com/v1/responses", "body": {}},
             "response": _openai_response([{
                 "type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": answer}],
             }])},
        ],
    }
    outgoing, replayed_answer, failures = replay_cassette(cassette)
    if failures:
        fail(f"recording task replay failed: {failures!r}")
    if replayed_answer != answer or len(outgoing) != 3:
        fail("recording task replay did not preserve the expected answer and three exchanges")
    ok("recording task replays its read, ls, and final-answer turns")


@check("vendor_replay.needed_back_values_excludes_request_echoes")
def check_needed_back_values_excludes_request_echoes() -> None:
    openai = json.loads((CASSETTE_DIR / "openai.json").read_text(encoding="utf-8"))
    openai_needed = needed_back_values(openai)
    if any(location.endswith("description") for _earlier, _later, location, _value in openai_needed):
        fail("OpenAI tool description was incorrectly listed as needed back")
    if not any(location.endswith("call_id") for _earlier, _later, location, _value in openai_needed):
        fail("OpenAI call_id values were not listed as needed back")

    gemini = json.loads((CASSETTE_DIR / "gemini.json").read_text(encoding="utf-8"))
    gemini_needed = needed_back_values(gemini)
    if not any("thoughtSignature" in location for _earlier, _later, location, _value in gemini_needed):
        fail("Gemini thoughtSignature values were not listed as needed back")

    echoed = "a request string that is repeated by the vendor"
    opaque_id = "opaque-round-trip-id-0123456789"
    synthetic = {
        "exchanges": [
            {
                "request": {"body": {"text": echoed}},
                "response": {"body": json.dumps({"echo": echoed, "id": opaque_id})},
            },
            {
                "request": {"body": {"echo": echoed, "id": opaque_id}},
                "response": {"body": "{}"},
            },
        ],
    }
    synthetic_needed = needed_back_values(synthetic)
    if [(location.rsplit(".", 1)[-1], value) for _earlier, _later, location, value in synthetic_needed] != [("id", opaque_id)]:
        fail(f"synthetic echo cassette listed unexpected needed-back values: {synthetic_needed!r}")
    ok("needed-back values exclude request echoes and retain vendor-originated IDs")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(PROVIDER_KEYS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    key = os.environ.get(PROVIDER_KEYS[arguments.provider], "").strip()
    if not key:
        parser.error(f"{PROVIDER_KEYS[arguments.provider]} is not set")
    _cassette_capture(arguments.provider, arguments.model, key, arguments.out)


if __name__ == "__main__":
    main()
