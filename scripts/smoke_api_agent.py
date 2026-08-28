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
    includes schemas for all eight standard tools in its outgoing request
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
import signal
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import unittest.mock as mock
import urllib.error
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api import (  # noqa: E402
    FAIL_CLOSED,
    InterruptBehavior,
    ResultHint,
    SCHEMA_VERSION,
    ToolEffect,
    ToolMetadata,
    safe_metadata,
)
from orchestra_api.agent_loop import ApiAgent  # noqa: E402
from orchestra_api.cancellation import CancellationToken, OperationCancelled  # noqa: E402
import orchestra_api.content as content  # noqa: E402
from orchestra_api.compaction import (  # noqa: E402
    ContextCompactionError,
    _message_excerpts,
    compact_messages_for_budget,
    estimate_message_tokens,
)
from orchestra_api.content import (  # noqa: E402
    content_block_from_bytes,
    content_block_from_path,
    detect_media_type,
)
from orchestra_api.events import (  # noqa: E402
    CollectingSink,
    RunFailed,
    RunFinished,
    RunStarted,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
    TurnStarted,
)
from orchestra_api.identity import TurnRef  # noqa: E402
from orchestra_api.model_discovery import list_models  # noqa: E402
from orchestra_api.models import (  # noqa: E402
    DocumentBlock,
    ImageBlock,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    TextBlock,
    ToolCall,
    ToolResult,
    has_attachments,
)
from orchestra_api.permissions import PermissionPolicy  # noqa: E402
from orchestra_api.providers.anthropic_provider import (  # noqa: E402
    ANTHROPIC_VERSION,
    API_KEY_ENV_VAR as ANTHROPIC_API_KEY_ENV_VAR,
)
from orchestra_api.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from orchestra_api.providers.anthropic_provider import _build_request_body as _build_anthropic_body  # noqa: E402
from orchestra_api.providers.anthropic_provider import _parse_response as _parse_anthropic_response  # noqa: E402
from orchestra_api.providers.base import ModelProvider, ProviderError  # noqa: E402
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
from orchestra_api.retry import read_with_retry  # noqa: E402
from orchestra_api.runner import run_task, standard_tool_registry  # noqa: E402
from orchestra_api.tool_schema import to_provider_tool_schema  # noqa: E402
from orchestra_api.tools.base import LocalTool  # noqa: E402
import orchestra_api.tools.filesystem as filesystem_tools  # noqa: E402
from orchestra_api.tools.shell import (  # noqa: E402
    MAX_OUTPUT_CHARS,
    RunShellTool,
    _terminate_process_group,
)


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def _assert_openai_tool_calls_answered(messages: list[Message], context: str) -> None:
    wire_messages = _build_openai_body(ModelRequest(messages=messages), "test-model")["messages"]
    for index, message in enumerate(wire_messages):
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            continue
        expected_ids = [tool_call["id"] for tool_call in tool_calls]
        actual_ids = [
            candidate.get("tool_call_id")
            for candidate in wire_messages[index + 1 : index + 1 + len(expected_ids)]
            if candidate.get("role") == "tool"
        ]
        if actual_ids != expected_ids:
            fail(
                f"{context} left unanswered OpenAI tool calls: "
                f"expected={expected_ids!r}, actual={actual_ids!r}, body={wire_messages!r}"
            )


class _CancellingTool(LocalTool):
    @property
    def name(self) -> str:
        return "cancel_work"

    @property
    def description(self) -> str:
        return "Cancel the active test turn."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.DESTRUCTIVE,
            concurrency_safe=False,
            paths=None,
        )

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        assert cancel is not None
        cancel.cancel()
        raise OperationCancelled


class _ToolContractStub(LocalTool):
    @property
    def name(self) -> str:
        return "contract_stub"

    @property
    def description(self) -> str:
        return "Exercise the LocalTool contract."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}


class _MissingMetadataTool(_ToolContractStub):
    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        return ToolResult(tool_call_id=tool_call.id, ok=True)


class _MissingExecuteTool(_ToolContractStub):
    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(),
        )


class _RaisingMetadataTool(_ToolContractStub):
    def __init__(self, error: Exception) -> None:
        self._error = error

    def metadata(self, arguments: dict) -> ToolMetadata:
        raise self._error

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        return ToolResult(tool_call_id=tool_call.id, ok=True)


class _ValidationStageTool(_ToolContractStub):
    def __init__(self) -> None:
        self.executed = False

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(),
        )

    def validate(self, arguments: dict) -> str | None:
        return "scripted validation failure"

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        self.executed = True
        return ToolResult(tool_call_id=tool_call.id, ok=True)


class _RaisingProvider(ModelProvider):
    @property
    def name(self) -> str:
        return "raising"

    @property
    def wire_format(self) -> int:
        return 4

    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        raise ProviderError("event test failure")


def main() -> None:
    image_block = ImageBlock(data="aW1hZ2U=", media_type="image/png")
    document_block = DocumentBlock(data="cGRm", filename="notes.pdf")
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
    attached = Message(role=Role.USER, content=[TextBlock("look"), image_block])
    if attached.content != (TextBlock("look"), image_block) or attached.text != "look":
        fail(f"mixed content did not normalize in order: {attached!r}")
    bare_attachment = Message(role=Role.USER, content=image_block)
    if bare_attachment.content != (image_block,) or bare_attachment.text != "":
        fail(f"bare attachment did not normalize to one block: {bare_attachment!r}")
    if (
        has_attachments(normalized)
        or has_attachments(empty)
        or not has_attachments(attached)
        or not has_attachments(Message(role=Role.USER, content=document_block))
    ):
        fail("has_attachments misclassified text, empty, image, or document content")
    try:
        Message(role=Role.USER, content=123)
    except TypeError:
        pass
    else:
        fail("unsupported message content should raise TypeError")
    ok("Message content normalizes to immutable text and attachment blocks")

    png_raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    jpeg_raw = b"\xff\xd8\xff" + b"\x00" * 32
    gif_raw = b"GIF89a" + b"\x00" * 32
    webp_raw = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 32
    pdf_raw = b"%PDF-1.7\n" + b"\x00" * 32
    detected_types = {
        detect_media_type(png_raw),
        detect_media_type(jpeg_raw),
        detect_media_type(gif_raw),
        detect_media_type(webp_raw),
        detect_media_type(pdf_raw),
    }
    expected_detected_types = {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "application/pdf",
    }
    if detected_types != expected_detected_types or detect_media_type(b"not media") is not None:
        fail(f"attachment magic-byte detection changed: {detected_types!r}")

    with mock.patch.object(content, "MAX_ATTACHMENT_BYTES", 8):
        try:
            content_block_from_bytes(png_raw)
        except ValueError as exc:
            expected_size_error = "attachment is 40 bytes, over the 8 byte limit"
            if str(exc) != expected_size_error:
                fail(f"oversize attachment error changed: {exc!r}")
        else:
            fail("oversize attachment was accepted")
    try:
        content_block_from_bytes(b"not media")
    except ValueError as exc:
        expected_type_error = (
            "unsupported attachment type; expected PNG, JPEG, GIF, WEBP, or PDF"
        )
        if str(exc) != expected_type_error:
            fail(f"unsupported attachment error changed: {exc!r}")
    else:
        fail("unsupported attachment bytes were accepted")

    with tempfile.TemporaryDirectory() as attachment_tmp:
        attachment_root = Path(attachment_tmp)
        pdf_path = attachment_root / "report.pdf"
        png_path = attachment_root / "renamed.bin"
        pdf_path.write_bytes(pdf_raw)
        png_path.write_bytes(png_raw)
        path_pdf = content_block_from_path(pdf_path)
        path_png = content_block_from_path(png_path)
    if (
        not isinstance(path_pdf, DocumentBlock)
        or path_pdf.filename != "report.pdf"
        or path_pdf.media_type != "application/pdf"
        or not isinstance(path_png, ImageBlock)
        or path_png.media_type != "image/png"
    ):
        fail(f"path attachment construction changed: pdf={path_pdf!r}, png={path_png!r}")
    ok("attachment construction detects, bounds, and base64-encodes supported media")

    anthropic_image_content = _build_anthropic_body(
        ModelRequest(messages=[Message(Role.USER, [TextBlock("look"), image_block])]),
        "test-model",
        100,
    )["messages"][0]["content"]
    expected_anthropic_image = [
        {"type": "text", "text": "look"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "aW1hZ2U=",
            },
        },
    ]
    if anthropic_image_content != expected_anthropic_image:
        fail(f"Anthropic image content changed: {anthropic_image_content!r}")
    anthropic_document_content = _build_anthropic_body(
        ModelRequest(messages=[Message(Role.USER, document_block)]),
        "test-model",
        100,
    )["messages"][0]["content"]
    expected_anthropic_document = [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "cGRm",
            },
            "title": "notes.pdf",
        }
    ]
    if anthropic_document_content != expected_anthropic_document:
        fail(f"Anthropic document content changed: {anthropic_document_content!r}")

    openai_image_content = _build_openai_body(
        ModelRequest(messages=[Message(Role.USER, [TextBlock("look"), image_block])]),
        "test-model",
    )["messages"][0]["content"]
    expected_openai_image = [
        {"type": "text", "text": "look"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
        },
    ]
    if openai_image_content != expected_openai_image:
        fail(f"OpenAI image content changed: {openai_image_content!r}")
    openai_document_content = _build_openai_body(
        ModelRequest(messages=[Message(Role.USER, DocumentBlock(data="cGRm"))]),
        "test-model",
    )["messages"][0]["content"]
    expected_openai_document = [
        {
            "type": "file",
            "file": {
                "filename": "document.pdf",
                "file_data": "data:application/pdf;base64,cGRm",
            },
        }
    ]
    if openai_document_content != expected_openai_document:
        fail(f"OpenAI document content changed: {openai_document_content!r}")

    gemini_pdf_parts = _build_gemini_body(
        ModelRequest(messages=[Message(Role.USER, [TextBlock("read"), document_block])])
    )["contents"][0]["parts"]
    expected_gemini_pdf = [
        {"text": "read"},
        {"inlineData": {"mimeType": "application/pdf", "data": "cGRm"}},
    ]
    if gemini_pdf_parts != expected_gemini_pdf:
        fail(f"Gemini PDF content changed: {gemini_pdf_parts!r}")

    merged_message = Message(
        Role.USER,
        [TextBlock("adjacent "), TextBlock("text"), image_block],
    )
    merged_anthropic = _build_anthropic_body(
        ModelRequest(messages=[merged_message]), "test-model", 100
    )["messages"][0]["content"]
    merged_openai = _build_openai_body(
        ModelRequest(messages=[merged_message]), "test-model"
    )["messages"][0]["content"]
    merged_gemini = _build_gemini_body(
        ModelRequest(messages=[merged_message])
    )["contents"][0]["parts"]
    if (
        merged_anthropic[0] != {"type": "text", "text": "adjacent text"}
        or merged_openai[0] != {"type": "text", "text": "adjacent text"}
        or merged_gemini[0] != {"text": "adjacent text"}
        or len(merged_anthropic) != 2
        or len(merged_openai) != 2
        or len(merged_gemini) != 2
    ):
        fail("adjacent text blocks were not merged before attachment encoding")

    fixed_messages = [
        Message(Role.SYSTEM, "system"),
        Message(Role.USER, "user"),
        Message(
            Role.ASSISTANT,
            "assistant",
            tool_calls=[
                ToolCall(
                    id="canonical",
                    vendor_id="vendor",
                    name="lookup",
                    arguments={"key": "value"},
                    provider_metadata={"thoughtSignature": "signature"},
                )
            ],
        ),
        Message(
            Role.TOOL,
            tool_result=ToolResult(
                tool_call_id="canonical", ok=True, content="result"
            ),
        ),
    ]
    fixed_request = ModelRequest(messages=fixed_messages)
    actual_legacy_bodies = (
        json.dumps(_build_openai_body(fixed_request, "test-model"), sort_keys=True),
        json.dumps(
            _build_anthropic_body(fixed_request, "test-model", 100), sort_keys=True
        ),
        json.dumps(_build_gemini_body(fixed_request), sort_keys=True),
    )
    expected_legacy_bodies = (
        '{"messages": [{"content": "system", "role": "system"}, {"content": "user", "role": "user"}, {"content": "assistant", "role": "assistant", "tool_calls": [{"function": {"arguments": "{\\"key\\": \\"value\\"}", "name": "lookup"}, "id": "vendor", "type": "function"}]}, {"content": "result", "role": "tool", "tool_call_id": "vendor"}], "model": "test-model"}',
        '{"max_tokens": 100, "messages": [{"content": "user", "role": "user"}, {"content": [{"text": "assistant", "type": "text"}, {"id": "vendor", "input": {"key": "value"}, "name": "lookup", "type": "tool_use"}], "role": "assistant"}, {"content": [{"content": "result", "is_error": false, "tool_use_id": "vendor", "type": "tool_result"}], "role": "user"}], "model": "test-model", "system": "system"}',
        '{"contents": [{"parts": [{"text": "user"}], "role": "user"}, {"parts": [{"text": "assistant"}, {"functionCall": {"args": {"key": "value"}, "id": "vendor", "name": "lookup"}, "thoughtSignature": "signature"}], "role": "model"}, {"parts": [{"functionResponse": {"name": "lookup", "response": {"output": "result"}}}], "role": "user"}], "system_instruction": {"parts": [{"text": "system"}]}}',
    )
    if actual_legacy_bodies != expected_legacy_bodies:
        fail(
            "attachment-free provider bodies changed: "
            f"actual={actual_legacy_bodies!r}"
        )

    text_only_tokens = estimate_message_tokens(Message(Role.USER, "look"))
    large_image = ImageBlock(data="A" * 133_336, media_type="image/png")
    attached_tokens = estimate_message_tokens(
        Message(Role.USER, [TextBlock("look"), large_image])
    )
    if text_only_tokens != 7 or attached_tokens <= text_only_tokens + 100:
        fail(
            "attachment token budgeting changed: "
            f"text={text_only_tokens}, attached={attached_tokens}"
        )
    excerpts = _message_excerpts(
        [Message(Role.USER, DocumentBlock(data="cGRm"))], limit=1
    )
    if excerpts != ["user: [1 attachment: application/pdf]"]:
        fail(f"attachment-only compaction excerpt changed: {excerpts!r}")
    ok("providers encode attachments without changing text-only wire bodies")

    for incomplete_tool, missing_member in (
        (_MissingMetadataTool, "metadata"),
        (_MissingExecuteTool, "_execute"),
    ):
        try:
            incomplete_tool()
        except TypeError:
            pass
        else:
            fail(f"LocalTool subclass missing {missing_member} was instantiable")
    ok("LocalTool requires both metadata and _execute declarations")

    metadata_tools = standard_tool_registry()
    execute_overrides = [
        name
        for name, tool in metadata_tools.items()
        if type(tool).execute is not LocalTool.execute
    ]
    if execute_overrides:
        fail(f"tools override the base validation pipeline: {execute_overrides!r}")

    metadata_arguments = {
        "read_file": {"path": "sample.txt"},
        "write_file": {"path": "sample.txt", "content": "replacement"},
        "edit_file": {
            "path": "sample.txt",
            "old_string": "before",
            "new_string": "after",
        },
        "multi_edit_file": {
            "path": "sample.txt",
            "edits": [{"old_string": "before", "new_string": "after"}],
        },
        "list_files": {"path": "sample-dir"},
        "glob": {"pattern": "**/*.py", "path": "sample-dir"},
        "grep": {"pattern": "needle", "path": "sample-dir"},
        "run_shell": {"argv": ["ls"]},
    }
    expected_metadata = {
        "read_file": ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=("sample.txt",),
            result_hint=ResultHint.TEXT,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "write_file": ToolMetadata(
            effect=ToolEffect.DESTRUCTIVE,
            concurrency_safe=False,
            paths=("sample.txt",),
            result_hint=ResultHint.TEXT,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "edit_file": ToolMetadata(
            effect=ToolEffect.MUTATING,
            concurrency_safe=False,
            paths=("sample.txt",),
            result_hint=ResultHint.DIFF,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "multi_edit_file": ToolMetadata(
            effect=ToolEffect.MUTATING,
            concurrency_safe=False,
            paths=("sample.txt",),
            result_hint=ResultHint.DIFF,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "list_files": ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=("sample-dir",),
            result_hint=ResultHint.FILE_LIST,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "glob": ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=("sample-dir",),
            result_hint=ResultHint.FILE_LIST,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "grep": ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=("sample-dir",),
            result_hint=ResultHint.FILE_LIST,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "run_shell": ToolMetadata(
            effect=ToolEffect.DESTRUCTIVE,
            concurrency_safe=False,
            paths=None,
            result_hint=ResultHint.TEXT,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
    }
    actual_metadata = {
        name: tool.metadata(metadata_arguments[name])
        for name, tool in metadata_tools.items()
    }
    if actual_metadata != expected_metadata:
        fail(
            "standard tool metadata did not match the literal contract: "
            f"actual={actual_metadata!r}, expected={expected_metadata!r}"
        )
    if any(item.schema_version != SCHEMA_VERSION for item in actual_metadata.values()):
        fail(f"tool metadata schema version drifted: {actual_metadata!r}")

    raw_path = "../outside.txt"
    if metadata_tools["read_file"].metadata({"path": raw_path}).paths != (raw_path,):
        fail("read_file metadata resolved or discarded its raw path")
    if metadata_tools["write_file"].metadata({"path": raw_path}).paths != (raw_path,):
        fail("write_file metadata resolved or discarded its raw path")
    if metadata_tools["list_files"].metadata({}).paths != (".",):
        fail("list_files metadata did not expose its default path")
    for name in ("glob", "grep"):
        if metadata_tools[name].metadata({}).paths != (".",):
            fail(f"{name} metadata did not expose its default path")
        if metadata_tools[name].metadata({"path": 3}).paths is not None:
            fail(f"{name} metadata accepted a non-string path")
    if metadata_tools["grep"].metadata({"output_mode": "content"}).result_hint != ResultHint.TEXT:
        fail("grep content metadata did not declare a text result")
    for name in ("read_file", "write_file", "edit_file", "multi_edit_file"):
        if metadata_tools[name].metadata({}).paths is not None:
            fail(f"{name} metadata treated a missing path as an empty path set")
        if metadata_tools[name].metadata({"path": 3}).paths is not None:
            fail(f"{name} metadata accepted a non-string path")
    if metadata_tools["list_files"].metadata({"path": 3}).paths is not None:
        fail("list_files metadata accepted a non-string path")
    if metadata_tools["run_shell"].metadata({"argv": ["ls"]}).paths is not None:
        fail("run_shell classified a path before the 02e argv classifier")

    if safe_metadata(_RaisingMetadataTool(ValueError("bad metadata")), {}) is not FAIL_CLOSED:
        fail("safe_metadata did not fail closed after a metadata exception")
    try:
        safe_metadata(_RaisingMetadataTool(OperationCancelled()), {})
    except OperationCancelled:
        pass
    else:
        fail("safe_metadata swallowed OperationCancelled")

    for name, item in actual_metadata.items():
        if item.concurrency_safe and item.effect != ToolEffect.READ_ONLY:
            fail(f"concurrency-safe registry call was not read-only: {name}={item!r}")
    if FAIL_CLOSED.concurrency_safe and FAIL_CLOSED.effect != ToolEffect.READ_ONLY:
        fail(f"FAIL_CLOSED violated the concurrency invariant: {FAIL_CLOSED!r}")
    if FAIL_CLOSED.schema_version != SCHEMA_VERSION:
        fail(f"FAIL_CLOSED schema version drifted: {FAIL_CLOSED!r}")

    frozen_metadata = expected_metadata["read_file"]
    try:
        frozen_metadata.effect = ToolEffect.DESTRUCTIVE
    except FrozenInstanceError:
        pass
    else:
        fail("ToolMetadata fields were mutable")
    ok("tool metadata is exact, per-call, frozen, and fail-closed")

    metadata_field_names = (
        "effect",
        "concurrency_safe",
        "paths",
        "result_hint",
        "interrupt_behavior",
        "schema_version",
    )
    read_parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read, relative to the allowed root.",
            },
            "offset": {
                "type": "integer",
                "description": "1-based first line to read.",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to read; 0 reads to end of file. Defaults to 2000.",
                "default": 2000,
            },
        },
        "required": ["path"],
    }
    gemini_read_parameters = json.loads(json.dumps(read_parameters))
    gemini_read_parameters["properties"]["offset"].pop("default")
    gemini_read_parameters["properties"]["limit"].pop("default")
    read_description = "Read a numbered line range from a text file inside the allowed scope."
    literal_read_schemas = {
        1: {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": read_description,
                "parameters": read_parameters,
            },
        },
        2: {
            "name": "read_file",
            "description": read_description,
            "input_schema": read_parameters,
        },
        3: {
            "name": "read_file",
            "description": read_description,
            "parameters": gemini_read_parameters,
        },
        4: {
            "name": "read_file",
            "description": read_description,
            "properties": read_parameters,
        },
    }
    for wire_format in (1, 2, 3, 4):
        for name, tool in metadata_tools.items():
            schema = to_provider_tool_schema(tool, wire_format)
            serialized = json.dumps(schema)
            leaked = [field for field in metadata_field_names if field in serialized]
            if leaked:
                fail(
                    f"metadata leaked into {name} wire format {wire_format}: "
                    f"fields={leaked!r}, schema={schema!r}"
                )
        read_schema = to_provider_tool_schema(
            metadata_tools["read_file"], wire_format
        )
        if read_schema != literal_read_schemas[wire_format]:
            fail(
                f"read_file wire format {wire_format} changed: "
                f"actual={read_schema!r}, expected={literal_read_schemas[wire_format]!r}"
            )
    ok("metadata stays out of every provider tool schema")

    try:
        ToolCall(id="", name="x")
    except ValueError as exc:
        if str(exc) != "ToolCall.id must be a non-empty string":
            fail(f"empty ToolCall id raised the wrong error: {exc!r}")
    else:
        fail("ToolCall accepted an empty canonical id")

    malformed_anthropic = _parse_anthropic_response(
        {
            "content": [
                {"type": "tool_use", "id": "", "name": "lookup", "input": {}}
            ]
        }
    ).message.tool_calls[0]
    if not malformed_anthropic.id or malformed_anthropic.vendor_id is not None:
        fail(f"Anthropic missing-id fallback was not canonical-only: {malformed_anthropic!r}")

    real_anthropic_id = "toolu_real_vendor_id"
    normal_anthropic = _parse_anthropic_response(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": real_anthropic_id,
                    "name": "lookup",
                    "input": {"key": "value"},
                }
            ]
        }
    ).message.tool_calls[0]
    normal_anthropic_body = _build_anthropic_body(
        ModelRequest(
            messages=[
                Message(role=Role.ASSISTANT, tool_calls=[normal_anthropic]),
                Message(
                    role=Role.TOOL,
                    tool_result=ToolResult(
                        tool_call_id=normal_anthropic.id,
                        ok=True,
                        content="result",
                    ),
                ),
            ]
        ),
        "test-model",
        100,
    )
    if json.dumps(normal_anthropic_body).count(real_anthropic_id) != 2:
        fail(f"Anthropic real tool-use id did not round-trip unchanged: {normal_anthropic_body!r}")
    ok("ToolCall ids are non-empty and Anthropic ids preserve the vendor boundary")

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
    anthropic_missing_ids = {
        _parse_anthropic_response(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "",
                        "name": "lookup",
                        "input": {},
                    }
                ]
            }
        ).message.tool_calls[0].id
        for _ in range(2)
    }
    if (
        len(two_missing_ids) != 2
        or len(gemini_missing_ids) != 2
        or len(anthropic_missing_ids) != 2
    ):
        fail(
            "synthesized tool-call ids collided across turns: "
            f"openai={two_missing_ids!r}, gemini={gemini_missing_ids!r}, "
            f"anthropic={anthropic_missing_ids!r}"
        )
    ok("synthesized tool-call ids are unique per call, not position-based")

    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
        root = Path(tmp)
        (root / "existing.txt").write_text("hello from disk")
        (root / ".env").write_text("SECRET=do-not-read-me")

        # repo_root and allowed write scope are the same temp dir here.
        policy = PermissionPolicy(repo_root=root, allowed_write_scope=[root])

        final_events = CollectingSink()
        final_event_result = ApiAgent(
            FakeModelProvider(
                responses=[ModelResponse(Message(Role.ASSISTANT, "event answer"))]
            ),
            {},
            policy,
            events=final_events,
        ).run([Message(Role.USER, "event test")])
        if len(final_events.of_type(RunStarted)) != 1:
            fail(f"final run did not emit one RunStarted: {final_events.events!r}")
        final_terminals = final_events.of_type(RunFinished) + final_events.of_type(RunFailed)
        if len(final_terminals) != 1 or final_terminals[0].stopped_reason != "final_response":
            fail(f"final run terminal events are invalid: {final_events.events!r}")
        if any(
            event.agent_id != final_event_result.agent.agent_id
            or event.run_id != final_event_result.run.run_id
            for event in final_events.events
        ):
            fail(f"event identity does not match the agent result: {final_events.events!r}")
        turn_ids = {
            message.turn_id
            for message in final_event_result.messages
            if message.turn_id is not None
        }
        if any(
            event.turn_id is not None and event.turn_id not in turn_ids
            for event in final_events.events
        ):
            fail(f"event turn identity does not match the transcript: {final_events.events!r}")
        if len(final_events.of_type(TurnStarted)) != 1 or len(final_events.of_type(TurnFinished)) != 1:
            fail(f"completed final turn was not bracketed: {final_events.events!r}")
        ok("final ApiAgent events have one terminal and canonical run/turn identity")

        tool_events = CollectingSink()
        tool_event_result = ApiAgent(
            FakeModelProvider(
                responses=[
                    ModelResponse(
                        Message(
                            Role.ASSISTANT,
                            tool_calls=[
                                ToolCall(
                                    id="event-read",
                                    name="read_file",
                                    arguments={"path": "existing.txt"},
                                )
                            ],
                        )
                    ),
                    ModelResponse(Message(Role.ASSISTANT, "done")),
                ]
            ),
            standard_tool_registry(),
            policy,
            events=tool_events,
        ).run([Message(Role.USER, "read")])
        tool_started = tool_events.of_type(ToolCallStarted)
        tool_finished = tool_events.of_type(ToolCallFinished)
        if (
            len(tool_started) != 1
            or len(tool_finished) != 1
            or tool_started[0].tool_call_id != "event-read"
            or tool_finished[0].tool_call_id != "event-read"
            or not tool_finished[0].ok
        ):
            fail(f"successful tool events did not bracket execution: {tool_events.events!r}")
        if tool_event_result.stopped_reason != "final_response":
            fail(f"event-producing tool run changed its result: {tool_event_result!r}")

        unknown_events = CollectingSink()
        unknown_result = ApiAgent(
            FakeModelProvider(
                responses=[
                    ModelResponse(
                        Message(
                            Role.ASSISTANT,
                            tool_calls=[ToolCall(id="event-unknown", name="missing")],
                        )
                    )
                ]
            ),
            {},
            policy,
            max_turns=1,
            events=unknown_events,
        ).run([Message(Role.USER, "unknown")])
        unknown_finished = unknown_events.of_type(ToolCallFinished)
        if len(unknown_finished) != 1 or unknown_finished[0].ok:
            fail(f"unknown-tool completion event was not ok=False: {unknown_events.events!r}")
        unknown_terminals = unknown_events.of_type(RunFinished) + unknown_events.of_type(RunFailed)
        if (
            unknown_result.stopped_reason != "max_turns"
            or len(unknown_terminals) != 1
            or unknown_terminals[0].stopped_reason != "max_turns"
        ):
            fail(f"max-turn event terminal is invalid: {unknown_events.events!r}")
        ok("tool events bracket success and unknown-tool execution")

        failed_events = CollectingSink()
        try:
            ApiAgent(_RaisingProvider(), {}, policy, events=failed_events).run(
                [Message(Role.USER, "fail")]
            )
        except ProviderError:
            pass
        else:
            fail("raising provider did not propagate ProviderError")
        failed_terminals = failed_events.of_type(RunFinished) + failed_events.of_type(RunFailed)
        if len(failed_events.of_type(RunStarted)) != 1 or len(failed_terminals) != 1:
            fail(f"failed run terminal cardinality is invalid: {failed_events.events!r}")
        if not isinstance(failed_terminals[0], RunFailed):
            fail(f"raising provider did not emit RunFailed: {failed_events.events!r}")
        ok("raising provider emits exactly one RunFailed")

        def _broken_sink(event) -> None:  # noqa: ANN001
            raise RuntimeError("broken observer")

        broken_sink_result = ApiAgent(
            FakeModelProvider(
                responses=[ModelResponse(Message(Role.ASSISTANT, "still works"))]
            ),
            {},
            policy,
            events=_broken_sink,
        ).run([Message(Role.USER, "ignore observer")])
        if broken_sink_result.stopped_reason != "final_response":
            fail(f"raising event sink broke the run: {broken_sink_result!r}")

        def _cancelling_sink(event) -> None:  # noqa: ANN001
            raise OperationCancelled

        sink_cancel_result = ApiAgent(
            FakeModelProvider(), {}, policy, events=_cancelling_sink
        ).run([Message(Role.USER, "cancel from sink")])
        if sink_cancel_result.stopped_reason != "cancelled":
            fail(f"OperationCancelled from sink was swallowed: {sink_cancel_result!r}")
        ok("broken sinks are isolated while sink cancellation stops the run")

        inert_response = ModelResponse(Message(Role.ASSISTANT, "same result"))

        def _fixed_turn(run_id: str, index: int) -> TurnRef:
            return TurnRef(turn_id=f"turn_fixed_{index}", run_id=run_id, index=index)

        with mock.patch("orchestra_api.agent_loop.new_turn_ref", side_effect=_fixed_turn):
            without_events = ApiAgent(
                FakeModelProvider([inert_response]), {}, policy
            ).run([Message(Role.USER, "same input")])
            with_events = ApiAgent(
                FakeModelProvider([inert_response]), {}, policy, events=CollectingSink()
            ).run([Message(Role.USER, "same input")])
        if (
            without_events.messages != with_events.messages
            or without_events.stopped_reason != with_events.stopped_reason
        ):
            fail("collecting events changed the agent result")
        ok("dropping the event stream does not change agent results")

        pre_cancel = CancellationToken()
        pre_cancel_events = CollectingSink()
        pre_cancel.cancel()
        pre_cancel_provider = FakeModelProvider()
        pre_cancel_result = ApiAgent(pre_cancel_provider, {}, policy).run(
            [Message(role=Role.USER, content="stop")],
            cancel=pre_cancel,
            events=pre_cancel_events,
        )
        if pre_cancel_result.stopped_reason != "cancelled":
            fail(f"pre-cancelled run did not return cancelled: {pre_cancel_result!r}")
        if pre_cancel_provider.call_count != 0:
            fail(f"pre-cancelled run called its provider {pre_cancel_provider.call_count} times")
        pre_cancel_terminals = pre_cancel_events.of_type(RunFinished) + pre_cancel_events.of_type(RunFailed)
        if (
            len(pre_cancel_events.of_type(RunStarted)) != 1
            or len(pre_cancel_terminals) != 1
            or pre_cancel_terminals[0].stopped_reason != "cancelled"
        ):
            fail(f"cancelled run terminal cardinality is invalid: {pre_cancel_events.events!r}")
        ok("pre-cancelled ApiAgent run stops before calling the provider")

        during_tool_token = CancellationToken()
        cancelling_call = ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(id="cancel-1", name="cancel_work"),
                    ToolCall(id="cancel-2", name="cancel_work"),
                ],
            )
        )
        during_tool_result = ApiAgent(
            FakeModelProvider(responses=[cancelling_call]),
            {"cancel_work": _CancellingTool()},
            policy,
        ).run([Message(role=Role.USER, content="cancel in tool")], cancel=during_tool_token)
        if during_tool_result.stopped_reason != "cancelled":
            fail(f"tool cancellation escaped the agent boundary: {during_tool_result!r}")
        if not any(message.role == Role.ASSISTANT for message in during_tool_result.messages):
            fail(f"tool cancellation discarded the assistant message: {during_tool_result!r}")
        returned_calls = [
            tool_call
            for message in during_tool_result.messages
            for tool_call in message.tool_calls
        ]
        returned_results = {
            message.tool_result.tool_call_id: message.tool_result
            for message in during_tool_result.messages
            if message.tool_result is not None
        }
        if {tool_call.id for tool_call in returned_calls} != set(returned_results):
            fail(f"tool cancellation left an unanswered call: {during_tool_result.messages!r}")
        repaired = returned_results["cancel-1"]
        if repaired.ok or not repaired.cancelled:
            fail(f"cancelled tool result has the wrong flags: {repaired!r}")
        if returned_results["cancel-2"].ok or not returned_results["cancel-2"].cancelled:
            fail(f"unstarted tool call has the wrong cancellation result: {returned_results['cancel-2']!r}")
        next_turn_messages = [
            *during_tool_result.messages,
            Message(role=Role.USER, content="continue after cancellation"),
        ]
        _assert_openai_tool_calls_answered(next_turn_messages, "cancelled agent transcript")
        ok("tool cancellation repairs every tool call for the next provider request")

        compact_cancel = CancellationToken()
        compact_cancel.cancel()
        try:
            compact_messages_for_budget(
                [Message(role=Role.USER, content="cancel compaction")],
                cancel=compact_cancel,
            )
        except OperationCancelled:
            pass
        else:
            fail("pre-cancelled compaction did not raise OperationCancelled")
        ok("compaction checks cancellation at entry")

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
        if tool_messages[0].tool_result.content != "1\thello from disk":
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

        validation_tool = _ValidationStageTool()
        validation_result = validation_tool.execute(
            ToolCall(id="validation-stage", name=validation_tool.name),
            policy,
        )
        if (
            validation_result.ok
            or validation_result.error != "scripted validation failure"
            or validation_tool.executed
        ):
            fail(
                "validation did not short-circuit before permitted work: "
                f"result={validation_result!r}, executed={validation_tool.executed!r}"
            )

        invalid_shell = tools["run_shell"].execute(
            ToolCall(
                id="invalid-shell-arguments",
                name="run_shell",
                arguments={"argv": ["ls", 3]},
            ),
            policy,
        )
        expected_shell_error = (
            "missing or invalid required argument: argv "
            "(must be a non-empty list of strings)"
        )
        if invalid_shell.ok or invalid_shell.error != expected_shell_error:
            fail(f"run_shell validation message changed: {invalid_shell!r}")

        missing_read_path = tools["read_file"].execute(
            ToolCall(id="missing-read-path", name="read_file"),
            policy,
        )
        if missing_read_path.ok or missing_read_path.error != "missing required argument: path":
            fail(f"read_file validation message changed: {missing_read_path!r}")

        missing_write_content = tools["write_file"].execute(
            ToolCall(
                id="missing-write-content",
                name="write_file",
                arguments={"path": "missing-content.txt"},
            ),
            policy,
        )
        if (
            missing_write_content.ok
            or missing_write_content.error != "missing required argument: content"
        ):
            fail(f"write_file validation message changed: {missing_write_content!r}")

        empty_write = tools["write_file"].execute(
            ToolCall(
                id="empty-write-content",
                name="write_file",
                arguments={"path": "empty-content.txt", "content": ""},
            ),
            policy,
        )
        if not empty_write.ok or (root / "empty-content.txt").read_text() != "":
            fail(f"write_file rejected valid empty content: {empty_write!r}")

        default_list = tools["list_files"].execute(
            ToolCall(id="default-list-path", name="list_files"),
            policy,
        )
        if not default_list.ok or "existing.txt" not in default_list.content:
            fail(f"list_files no longer defaults to the repo root: {default_list!r}")

        invalid_cancel = CancellationToken()
        invalid_cancel.cancel()
        try:
            tools["run_shell"].execute(
                ToolCall(
                    id="cancel-before-validation",
                    name="run_shell",
                    arguments={"argv": ["ls", 3]},
                ),
                policy,
                cancel=invalid_cancel,
            )
        except OperationCancelled:
            pass
        else:
            fail("LocalTool validated invalid arguments before checking cancellation")
        ok("base validation preserves errors, defaults, and cancellation ordering")

        # -- glob/grep fixture: ordering, permission gates, bounds, and modes --
        search_root = root / "search-fixture"
        nested = search_root / "nested"
        ordered = search_root / "ordered"
        cancel_root = search_root / "cancel"
        nested.mkdir(parents=True)
        ordered.mkdir()
        cancel_root.mkdir()
        (search_root / "top.py").write_text("needle\nother\nneedle\nneedle\n")
        (search_root / "top.txt").write_text("ordinary text\n")
        (nested / "a.py").write_text("first\nneedle\n")
        (nested / "b.txt").write_text("needle\n")
        os.utime(search_root / "top.py", (600, 600))
        os.utime(nested / "a.py", (500, 500))
        os.utime(nested / "b.txt", (400, 400))
        secret_value = "SEARCH_FIXTURE_SECRET_62491"
        (search_root / ".env").write_text(secret_value)
        (search_root / "secret.pem").write_text(secret_value)
        (search_root / "node_modules").mkdir()
        (search_root / "node_modules" / "hidden.py").write_text(secret_value)
        (search_root / "skip-binary.txt").write_bytes(b"needle\xff\xfe")
        (search_root / "skip-large.txt").write_bytes(
            b"needle\n" + b"x" * filesystem_tools.MAX_READ_BYTES
        )
        outside_file = Path(outside_tmp) / "outside.txt"
        outside_file.write_text(f"needle {secret_value}")
        (search_root / "escape.txt").symlink_to(outside_file)
        outside_directory = Path(outside_tmp) / "outside-directory"
        outside_directory.mkdir()
        (outside_directory / "followed.py").write_text("needle")
        (search_root / "linked-directory").symlink_to(outside_directory, target_is_directory=True)

        ordered_mtimes = {
            "newest.ord": 500,
            "alpha.ord": 400,
            "beta.ord": 400,
            "older.ord": 300,
            "statfail.ord": 200,
        }
        for filename, mtime in ordered_mtimes.items():
            path = ordered / filename
            path.write_text(filename)
            os.utime(path, (mtime, mtime))

        top_level_glob = tools["glob"].execute(
            ToolCall(
                id="glob-top-level",
                name="glob",
                arguments={"path": "search-fixture", "pattern": "*.py", "head_limit": 0},
            ),
            policy,
        )
        recursive_glob = tools["glob"].execute(
            ToolCall(
                id="glob-recursive",
                name="glob",
                arguments={"path": "search-fixture", "pattern": "**/*.py", "head_limit": 0},
            ),
            policy,
        )
        if top_level_glob.content != "search-fixture/top.py":
            fail(f"*.py crossed directories: {top_level_glob!r}")
        if set(recursive_glob.content.splitlines()) != {
            "search-fixture/top.py",
            "search-fixture/nested/a.py",
        }:
            fail(f"**/*.py did not cover both depths: {recursive_glob!r}")

        secret_glob = tools["glob"].execute(
            ToolCall(
                id="glob-secrets",
                name="glob",
                arguments={"path": "search-fixture", "pattern": "**/*", "head_limit": 0},
            ),
            policy,
        )
        forbidden_names = (".env", "secret.pem", "node_modules", "escape.txt", "followed.py")
        if not secret_glob.ok or any(name in secret_glob.content for name in forbidden_names):
            fail(f"glob exposed a forbidden or escaped file: {secret_glob!r}")
        secret_grep = tools["grep"].execute(
            ToolCall(
                id="grep-secrets",
                name="grep",
                arguments={"path": "search-fixture", "pattern": secret_value, "head_limit": 0},
            ),
            policy,
        )
        if secret_grep.content != "no matches" or secret_value in secret_grep.content:
            fail(f"grep exposed forbidden contents: {secret_grep!r}")

        for tool_name, arguments in (
            (
                "glob",
                {"path": "search-fixture/ordered", "pattern": "*.ord", "head_limit": 0},
            ),
            (
                "grep",
                {"path": "search-fixture", "glob": "**/*.py", "pattern": "needle"},
            ),
        ):
            with mock.patch.object(policy, "check_read", wraps=policy.check_read) as read_gate:
                gated_result = tools[tool_name].execute(
                    ToolCall(
                        id=f"{tool_name}-file-gates",
                        name=tool_name,
                        arguments=arguments,
                    ),
                    policy,
                )
            checked_paths = {
                Path(call.args[0]).resolve()
                for call in read_gate.call_args_list
                if call.args
            }
            result_paths = [
                line.split(":", 1)[0]
                for line in gated_result.content.splitlines()
                if line and not line.startswith("[")
            ]
            if any((root / result_path).resolve() not in checked_paths for result_path in result_paths):
                fail(f"{tool_name} included a file without calling check_read: {read_gate.call_args_list!r}")
        ok("glob and grep enforce per-file gates, prune denied trees, and exclude symlink escapes")

        ordered_call = {
            "path": "search-fixture/ordered",
            "pattern": "*.ord",
            "head_limit": 0,
        }
        ordered_result = tools["glob"].execute(
            ToolCall(id="glob-order", name="glob", arguments=ordered_call), policy
        )
        expected_order = [
            "search-fixture/ordered/newest.ord",
            "search-fixture/ordered/alpha.ord",
            "search-fixture/ordered/beta.ord",
            "search-fixture/ordered/older.ord",
            "search-fixture/ordered/statfail.ord",
        ]
        if ordered_result.content.splitlines() != expected_order:
            fail(f"glob mtime/path ordering changed: {ordered_result!r}")

        original_stat = Path.stat

        def _failing_stat(path: Path, *args, **kwargs):  # noqa: ANN002, ANN003
            if path.name == "statfail.ord":
                raise OSError("scripted stat race")
            return original_stat(path, *args, **kwargs)

        with mock.patch.object(Path, "stat", _failing_stat):
            stat_race = tools["glob"].execute(
                ToolCall(id="glob-stat-race", name="glob", arguments=ordered_call), policy
            )
        if not stat_race.ok or stat_race.content.splitlines() != expected_order:
            fail(f"glob did not tolerate a failed stat as mtime zero: {stat_race!r}")

        no_cap = tools["glob"].execute(
            ToolCall(
                id="glob-no-cap",
                name="glob",
                arguments={**ordered_call, "head_limit": 10},
            ),
            policy,
        )
        capped = tools["glob"].execute(
            ToolCall(
                id="glob-cap",
                name="glob",
                arguments={**ordered_call, "head_limit": 2},
            ),
            policy,
        )
        unlimited = tools["glob"].execute(
            ToolCall(id="glob-unlimited", name="glob", arguments=ordered_call), policy
        )
        notice = "[2 of 5 results; pass offset=2 for the next page]"
        if "results; pass offset=" in no_cap.content or capped.content.splitlines()[-1] != notice:
            fail(f"glob cap notice did not report only a real truncation: {no_cap!r}, {capped!r}")
        if unlimited.content.splitlines() != expected_order or "results; pass offset=" in unlimited.content:
            fail(f"glob head_limit=0 did not return all results: {unlimited!r}")
        page = tools["glob"].execute(
            ToolCall(
                id="glob-page",
                name="glob",
                arguments={**ordered_call, "head_limit": 2, "offset": 2},
            ),
            policy,
        )
        if page.content.splitlines()[:2] != expected_order[2:4]:
            fail(f"glob pagination changed result ordering: {page!r}")
        past_glob = tools["glob"].execute(
            ToolCall(
                id="glob-past-end",
                name="glob",
                arguments={**ordered_call, "offset": 50},
            ),
            policy,
        )
        if past_glob.content != "[no results at offset 50; 5 results total]":
            fail(f"glob past-end pagination returned an ambiguous result: {past_glob!r}")
        ok("glob ordering, stat races, caps, unlimited results, and pagination are deterministic")

        grep_files = tools["grep"].execute(
            ToolCall(
                id="grep-files",
                name="grep",
                arguments={
                    "path": "search-fixture",
                    "glob": "**/*.py",
                    "pattern": "needle",
                    "head_limit": 1,
                },
            ),
            policy,
        )
        if grep_files.content.splitlines() != [
            "search-fixture/top.py",
            "[1 of 2 results; pass offset=1 for the next page]",
        ]:
            fail(f"grep files mode did not cap matching files: {grep_files!r}")
        grep_content = tools["grep"].execute(
            ToolCall(
                id="grep-content",
                name="grep",
                arguments={
                    "path": "search-fixture",
                    "glob": "top.py",
                    "pattern": "NEEDLE",
                    "case_insensitive": True,
                    "output_mode": "content",
                    "head_limit": 2,
                },
            ),
            policy,
        )
        if grep_content.content.splitlines() != [
            "search-fixture/top.py:1\tneedle",
            "search-fixture/top.py:3\tneedle",
            "[2 of 3 results; pass offset=2 for the next page]",
        ]:
            fail(f"grep content mode did not cap matching lines: {grep_content!r}")
        skipped = tools["grep"].execute(
            ToolCall(
                id="grep-skips",
                name="grep",
                arguments={
                    "path": "search-fixture",
                    "glob": "skip-*.txt",
                    "pattern": "needle",
                },
            ),
            policy,
        )
        if not skipped.ok or skipped.content != "no matches":
            fail(f"grep did not silently skip binary and oversized files: {skipped!r}")
        past_grep = tools["grep"].execute(
            ToolCall(
                id="grep-past-end",
                name="grep",
                arguments={
                    "path": "search-fixture",
                    "glob": "**/*.py",
                    "pattern": "needle",
                    "offset": 50,
                },
            ),
            policy,
        )
        if past_grep.content != "[no results at offset 50; 2 results total]":
            fail(f"grep past-end pagination returned an ambiguous result: {past_grep!r}")
        empty_search = search_root / "empty"
        empty_search.mkdir()
        empty_glob = tools["glob"].execute(
            ToolCall(
                id="glob-empty-tree",
                name="glob",
                arguments={"path": "search-fixture/empty", "pattern": "**/*", "offset": 50},
            ),
            policy,
        )
        empty_grep = tools["grep"].execute(
            ToolCall(
                id="grep-empty-tree",
                name="grep",
                arguments={"path": "search-fixture/empty", "pattern": "needle", "offset": 50},
            ),
            policy,
        )
        if empty_glob.content != "no files matched" or empty_grep.content != "no matches":
            fail(f"empty search messages changed: glob={empty_glob!r}, grep={empty_grep!r}")
        ok("grep modes count files or lines and skip unreadable text inputs")

        with mock.patch.object(
            tools["grep"], "_execute", side_effect=AssertionError("validation was bypassed")
        ):
            invalid_regex = tools["grep"].execute(
                ToolCall(id="grep-invalid", name="grep", arguments={"pattern": "["}), policy
            )
        if invalid_regex.ok or not invalid_regex.error.startswith("invalid regular expression: "):
            fail(f"grep invalid-regex validation changed: {invalid_regex!r}")

        for index in range(70):
            (cancel_root / f"{index:03}.cancel").write_text("needle")
        for tool_name, arguments in (
            ("glob", {"path": "search-fixture/cancel", "pattern": "*.cancel"}),
            ("grep", {"path": "search-fixture/cancel", "pattern": "needle"}),
        ):
            cancel_token = CancellationToken()
            real_check_read = policy.check_read
            seen_cancel_files = [0]

            def _cancel_during_gate(path):  # noqa: ANN001
                decision = real_check_read(path)
                if Path(path).suffix == ".cancel":
                    seen_cancel_files[0] += 1
                    if seen_cancel_files[0] == 1:
                        cancel_token.cancel()
                return decision

            try:
                with mock.patch.object(policy, "check_read", side_effect=_cancel_during_gate):
                    tools[tool_name].execute(
                        ToolCall(
                            id=f"{tool_name}-cancel",
                            name=tool_name,
                            arguments=arguments,
                        ),
                        policy,
                        cancel=cancel_token,
                    )
            except OperationCancelled:
                pass
            else:
                fail(f"{tool_name} did not observe cancellation inside its file walk")
        ok("glob and grep observe cancellation during a large file walk")

        read_fixture = root / "read-fixture.txt"
        read_fixture.write_text("a\nb\nc")
        full_read = tools["read_file"].execute(
            ToolCall(id="read-full", name="read_file", arguments={"path": "read-fixture.txt"}),
            policy,
        )
        partial_read = tools["read_file"].execute(
            ToolCall(
                id="read-partial",
                name="read_file",
                arguments={"path": "read-fixture.txt", "offset": 2, "limit": 1},
            ),
            policy,
        )
        end_read = tools["read_file"].execute(
            ToolCall(
                id="read-end",
                name="read_file",
                arguments={"path": "read-fixture.txt", "offset": 3},
            ),
            policy,
        )
        if full_read.content != "1\ta\n2\tb\n3\tc":
            fail(f"full read gained a marker or wrong line numbers: {full_read!r}")
        if partial_read.content != "2\tb\n[lines 2-2; more follow, pass offset=3]":
            fail(f"partial read marker changed: {partial_read!r}")
        if end_read.content != "3\tc\n[lines 3-3; end of file]":
            fail(f"end read marker changed: {end_read!r}")
        past_read = tools["read_file"].execute(
            ToolCall(
                id="read-past-end",
                name="read_file",
                arguments={"path": "read-fixture.txt", "offset": 999999},
            ),
            policy,
        )
        if past_read.content != "[no lines at offset 999999; file has 3 lines]":
            fail(f"past-end read returned an ambiguous result: {past_read!r}")
        (root / "trailing-newline.txt").write_text("a\nb\nc\n")
        trailing_read = tools["read_file"].execute(
            ToolCall(
                id="read-trailing",
                name="read_file",
                arguments={"path": "trailing-newline.txt"},
            ),
            policy,
        )
        if trailing_read.content != "1\ta\n2\tb\n3\tc":
            fail(f"trailing newline produced an empty numbered line: {trailing_read!r}")

        oversized_path = root / "oversized-read.txt"
        oversized_path.write_bytes(b"x\n" * (filesystem_tools.MAX_READ_BYTES // 2 + 1))
        refused = tools["read_file"].execute(
            ToolCall(id="read-byte-cap", name="read_file", arguments={"path": oversized_path.name}),
            policy,
        )
        expected_byte_error = (
            f"file exceeds {filesystem_tools.MAX_READ_BYTES} byte read limit; "
            "read part of it with offset and limit"
        )
        if refused.ok or refused.error != expected_byte_error:
            fail(f"unranged oversized read was not refused: {refused!r}")
        ranged = tools["read_file"].execute(
            ToolCall(
                id="read-byte-range",
                name="read_file",
                arguments={"path": oversized_path.name, "offset": 2, "limit": 1},
            ),
            policy,
        )
        if not ranged.ok or not ranged.content.startswith("2\tx\n[lines 2-2; more follow"):
            fail(f"explicit range of oversized file did not succeed: {ranged!r}")
        (root / "lookahead-byte-cap.txt").write_text("12345\n67890\nabcde\n")
        with mock.patch.object(filesystem_tools, "MAX_READ_BYTES", 12):
            lookahead_allowed = tools["read_file"].execute(
                ToolCall(
                    id="read-lookahead-byte-cap",
                    name="read_file",
                    arguments={"path": "lookahead-byte-cap.txt", "limit": 2},
                ),
                policy,
            )
        if lookahead_allowed.content != (
            "1\t12345\n2\t67890\n[lines 1-2; more follow, pass offset=3]"
        ):
            fail(f"read_file charged its lookahead line to the byte cap: {lookahead_allowed!r}")
        (root / "stream-byte-cap.txt").write_text("12345\n67890\n")
        for arguments in (
            {"path": "stream-byte-cap.txt", "offset": 1, "limit": 0},
            {"path": "stream-byte-cap.txt", "limit": 2000},
        ):
            with mock.patch.object(filesystem_tools, "MAX_READ_BYTES", 10):
                stream_refused = tools["read_file"].execute(
                    ToolCall(
                        id="read-stream-byte-cap",
                        name="read_file",
                        arguments=arguments,
                    ),
                    policy,
                )
            if stream_refused.ok or stream_refused.error != (
                "selected range exceeds 10 byte read limit; narrow it with offset and limit"
            ):
                fail(f"streamed range exceeded its byte accumulator: {stream_refused!r}")
        (root / "single-line-byte-cap.txt").write_text("abcdefghij\n")
        with mock.patch.object(filesystem_tools, "MAX_READ_BYTES", 4):
            single_line_byte_refused = tools["read_file"].execute(
                ToolCall(
                    id="read-single-line-byte-cap",
                    name="read_file",
                    arguments={"path": "single-line-byte-cap.txt", "limit": 1},
                ),
                policy,
            )
        if single_line_byte_refused.ok or single_line_byte_refused.error != (
            "line 1 is 11 characters, over the 4 byte read limit; "
            "search the file with grep instead"
        ):
            fail(f"single-line byte refusal did not name grep: {single_line_byte_refused!r}")
        (root / "later-line-byte-cap.txt").write_text("ab\nabcdefghij\n")
        with mock.patch.object(filesystem_tools, "MAX_READ_BYTES", 4):
            later_line_byte_refused = tools["read_file"].execute(
                ToolCall(
                    id="read-later-line-byte-cap",
                    name="read_file",
                    arguments={"path": "later-line-byte-cap.txt", "offset": 1, "limit": 2},
                ),
                policy,
            )
            narrowed_before_long_line = tools["read_file"].execute(
                ToolCall(
                    id="read-before-later-line-byte-cap",
                    name="read_file",
                    arguments={"path": "later-line-byte-cap.txt", "offset": 1, "limit": 1},
                ),
                policy,
            )
            direct_later_line_refused = tools["read_file"].execute(
                ToolCall(
                    id="read-direct-later-line-byte-cap",
                    name="read_file",
                    arguments={"path": "later-line-byte-cap.txt", "offset": 2, "limit": 1},
                ),
                policy,
            )
        if later_line_byte_refused.ok or later_line_byte_refused.error != (
            "selected range exceeds 4 byte read limit; narrow it with offset and limit"
        ):
            fail(f"later-line byte refusal did not recommend narrowing: {later_line_byte_refused!r}")
        if narrowed_before_long_line.content != (
            "1\tab\n[lines 1-1; more follow, pass offset=2]"
        ):
            fail(f"later-line byte refusal named an ineffective remedy: {narrowed_before_long_line!r}")
        if direct_later_line_refused.ok or direct_later_line_refused.error != (
            "line 2 is 11 characters, over the 4 byte read limit; "
            "search the file with grep instead"
        ):
            fail(f"direct oversized-line read did not name its absolute line: {direct_later_line_refused!r}")
        (root / "token-cap.txt").write_text("abcdefghij" * 8)
        with mock.patch.object(filesystem_tools, "MAX_READ_TOKENS", 2):
            token_refused = tools["read_file"].execute(
                ToolCall(
                    id="read-token-cap",
                    name="read_file",
                    arguments={"path": "token-cap.txt", "offset": 1, "limit": 1},
                ),
                policy,
            )
        if token_refused.ok or token_refused.error != (
            "line 1 is about 21 tokens, over the 2 token limit; "
            "search the file with grep instead"
        ):
            fail(f"single-line token refusal did not name grep: {token_refused!r}")
        (root / "many-lines-token-cap.txt").write_text("a\na\na")
        with mock.patch.object(filesystem_tools, "MAX_READ_TOKENS", 2):
            many_lines_refused = tools["read_file"].execute(
                ToolCall(
                    id="read-many-lines-token-cap",
                    name="read_file",
                    arguments={"path": "many-lines-token-cap.txt", "offset": 1, "limit": 3},
                ),
                policy,
            )
        if many_lines_refused.ok or many_lines_refused.error != (
            "selected range is about 3 tokens, over the 2 token limit; "
            "narrow it with offset and limit"
        ):
            fail(f"multi-line token refusal lost its narrowing remedy: {many_lines_refused!r}")
        ok("read_file numbers ranges, marks partial views, and refuses byte/token overages")

        # -- targeted edits, read-before-write staleness, and structured diffs --
        not_read_path = root / "edit-not-read.txt"
        not_read_path.write_text("before\n")
        not_read = tools["edit_file"].execute(
            ToolCall(
                id="edit-not-read",
                name="edit_file",
                arguments={
                    "path": not_read_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        not_read_error = "file has not been read yet; read it with read_file before editing it"
        if not_read.ok or not_read.error != not_read_error:
            fail(f"edit_file no-read refusal changed: {not_read!r}")

        stale_path = root / "edit-stale.txt"
        stale_path.write_text("before\n")
        tools["read_file"].execute(
            ToolCall(id="read-stale", name="read_file", arguments={"path": stale_path.name}),
            policy,
        )
        stale_stat = stale_path.stat()
        stale_path.write_text("outside change\n")
        os.utime(
            stale_path,
            ns=(stale_stat.st_atime_ns, stale_stat.st_mtime_ns + 1_000_000_000),
        )
        stale = tools["edit_file"].execute(
            ToolCall(
                id="edit-stale",
                name="edit_file",
                arguments={
                    "path": stale_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        stale_error = "file has changed since it was read; read it again before editing it"
        if stale.ok or stale.error != stale_error:
            fail(f"edit_file stale refusal changed: {stale!r}")

        same_content_path = root / "edit-same-content.txt"
        same_content_path.write_text("before\n")
        tools["read_file"].execute(
            ToolCall(
                id="read-same-content",
                name="read_file",
                arguments={"path": same_content_path.name},
            ),
            policy,
        )
        same_stat = same_content_path.stat()
        same_content_path.write_text("before\n")
        os.utime(
            same_content_path,
            ns=(same_stat.st_atime_ns, same_stat.st_mtime_ns + 1_000_000_000),
        )
        same_content = tools["edit_file"].execute(
            ToolCall(
                id="edit-same-content",
                name="edit_file",
                arguments={
                    "path": same_content_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        if not same_content.ok or same_content_path.read_text() != "after\n":
            fail(f"identical-content mtime bump was treated as stale: {same_content!r}")

        ranged_path = root / "edit-ranged-stale.txt"
        ranged_path.write_text("before\nafter\n")
        tools["read_file"].execute(
            ToolCall(
                id="read-ranged-stale",
                name="read_file",
                arguments={"path": ranged_path.name, "limit": 1},
            ),
            policy,
        )
        ranged_stat = ranged_path.stat()
        ranged_path.write_text("before\nafter\n")
        os.utime(
            ranged_path,
            ns=(ranged_stat.st_atime_ns, ranged_stat.st_mtime_ns + 1_000_000_000),
        )
        ranged_stale = tools["edit_file"].execute(
            ToolCall(
                id="edit-ranged-stale",
                name="edit_file",
                arguments={
                    "path": ranged_path.name,
                    "old_string": "before",
                    "new_string": "changed",
                },
            ),
            policy,
        )
        if ranged_stale.ok or ranged_stale.error != stale_error:
            fail(f"ranged read incorrectly proved unchanged content: {ranged_stale!r}")

        long_unranged_path = root / "edit-long-unranged-stale.txt"
        long_unranged_path.write_text("one\ntwo\nthree\n")
        with mock.patch.object(filesystem_tools, "MAX_READ_LINES", 2):
            long_unranged_read = tools["read_file"].execute(
                ToolCall(
                    id="read-long-unranged-stale",
                    name="read_file",
                    arguments={"path": long_unranged_path.name},
                ),
                policy,
            )
        if not long_unranged_read.ok or "more follow" not in long_unranged_read.content:
            fail(f"long unranged fixture did not truncate: {long_unranged_read!r}")
        long_unranged_stat = long_unranged_path.stat()
        long_unranged_path.write_text("one\ntwo\nthree\n")
        os.utime(
            long_unranged_path,
            ns=(
                long_unranged_stat.st_atime_ns,
                long_unranged_stat.st_mtime_ns + 1_000_000_000,
            ),
        )
        long_unranged_edit = tools["edit_file"].execute(
            ToolCall(
                id="edit-long-unranged-stale",
                name="edit_file",
                arguments={
                    "path": long_unranged_path.name,
                    "old_string": "one",
                    "new_string": "changed",
                },
            ),
            policy,
        )
        if long_unranged_edit.ok or long_unranged_edit.error != stale_error:
            fail(
                "truncated unranged read incorrectly proved unchanged content: "
                f"{long_unranged_edit!r}"
            )

        single_open_path = root / "edit-single-open.txt"
        single_open_path.write_text("before\n")
        real_path_open = Path.open
        real_path_read_text = Path.read_text
        open_count = [0]
        read_text_count = [0]

        def _count_path_open(path, *args, **kwargs):  # noqa: ANN001
            open_count[0] += 1
            return real_path_open(path, *args, **kwargs)

        def _count_path_read_text(path, *args, **kwargs):  # noqa: ANN001
            read_text_count[0] += 1
            return real_path_read_text(path, *args, **kwargs)

        with mock.patch.object(
            Path, "open", autospec=True, side_effect=_count_path_open
        ), mock.patch.object(
            Path, "read_text", autospec=True, side_effect=_count_path_read_text
        ):
            single_open_read = tools["read_file"].execute(
                ToolCall(
                    id="read-single-open",
                    name="read_file",
                    arguments={"path": single_open_path.name},
                ),
                policy,
            )
        if (
            not single_open_read.ok
            or open_count != [1]
            or read_text_count != [0]
        ):
            fail(
                "short unranged read did not open exactly once: "
                f"result={single_open_read!r}, opens={open_count!r}, "
                f"read_text={read_text_count!r}"
            )
        single_open_stat = single_open_path.stat()
        single_open_path.write_text("before\n")
        os.utime(
            single_open_path,
            ns=(
                single_open_stat.st_atime_ns,
                single_open_stat.st_mtime_ns + 1_000_000_000,
            ),
        )
        single_open_edit = tools["edit_file"].execute(
            ToolCall(
                id="edit-single-open",
                name="edit_file",
                arguments={
                    "path": single_open_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        if not single_open_edit.ok or single_open_path.read_text() != "after\n":
            fail(f"short full read no longer forgives an identical rewrite: {single_open_edit!r}")

        match_path = root / "edit-matches.txt"
        match_path.write_text("one one\n")
        tools["read_file"].execute(
            ToolCall(id="read-matches", name="read_file", arguments={"path": match_path.name}),
            policy,
        )
        missing_match = tools["edit_file"].execute(
            ToolCall(
                id="edit-zero-match",
                name="edit_file",
                arguments={
                    "path": match_path.name,
                    "old_string": "absent",
                    "new_string": "new",
                },
            ),
            policy,
        )
        if missing_match.ok or missing_match.error != "old_string not found in edit-matches.txt":
            fail(f"zero-match error changed: {missing_match!r}")
        multi_match = tools["edit_file"].execute(
            ToolCall(
                id="edit-multi-match",
                name="edit_file",
                arguments={
                    "path": match_path.name,
                    "old_string": "one",
                    "new_string": "two",
                },
            ),
            policy,
        )
        expected_multi_match = (
            "old_string matches 2 times in edit-matches.txt; set replace_all=true to "
            "change every match, or add surrounding context to identify one"
        )
        if multi_match.ok or multi_match.error != expected_multi_match:
            fail(f"multi-match error changed: {multi_match!r}")
        replace_all = tools["edit_file"].execute(
            ToolCall(
                id="edit-replace-all",
                name="edit_file",
                arguments={
                    "path": match_path.name,
                    "old_string": "one",
                    "new_string": "two",
                    "replace_all": True,
                },
            ),
            policy,
        )
        if not replace_all.ok or match_path.read_text() != "two two\n":
            fail(f"replace_all did not replace every occurrence: {replace_all!r}")

        one_match_path = root / "edit-one-match.txt"
        one_match_path.write_text("before keep\n")
        tools["read_file"].execute(
            ToolCall(id="read-one-match", name="read_file", arguments={"path": one_match_path.name}),
            policy,
        )
        one_match = tools["edit_file"].execute(
            ToolCall(
                id="edit-one-match",
                name="edit_file",
                arguments={
                    "path": one_match_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        if not one_match.ok or one_match_path.read_text() != "after keep\n":
            fail(f"default single replacement changed: {one_match!r}")

        batch_path = root / "multi-edit.txt"
        batch_path.write_text("alpha beta gamma\n")
        tools["read_file"].execute(
            ToolCall(id="read-multi-edit", name="read_file", arguments={"path": batch_path.name}),
            policy,
        )
        ordered_batch = tools["multi_edit_file"].execute(
            ToolCall(
                id="multi-edit-ordered",
                name="multi_edit_file",
                arguments={
                    "path": batch_path.name,
                    "edits": [
                        {"old_string": "alpha", "new_string": "delta"},
                        {"old_string": "delta beta", "new_string": "joined"},
                    ],
                },
            ),
            policy,
        )
        if not ordered_batch.ok or batch_path.read_text() != "joined gamma\n":
            fail(f"multi_edit_file did not apply edits in order: {ordered_batch!r}")
        if (
            ordered_batch.payload is None
            or ordered_batch.payload["kind"] != "file_diff"
            or ordered_batch.payload["path"] != "multi-edit.txt"
            or json.loads(json.dumps(ordered_batch.payload)) != ordered_batch.payload
            or not ordered_batch.content.startswith("--- multi-edit.txt\n+++ multi-edit.txt\n")
        ):
            fail(f"multi_edit_file did not return a structured unified diff: {ordered_batch!r}")
        before_failed_batch = batch_path.read_text()
        failed_batch = tools["multi_edit_file"].execute(
            ToolCall(
                id="multi-edit-atomic",
                name="multi_edit_file",
                arguments={
                    "path": batch_path.name,
                    "edits": [
                        {"old_string": "joined", "new_string": "changed"},
                        {"old_string": "absent", "new_string": "never"},
                    ],
                },
            ),
            policy,
        )
        if failed_batch.ok or failed_batch.error != "edit 2: old_string not found in multi-edit.txt":
            fail(f"multi_edit_file failure message changed: {failed_batch!r}")
        if batch_path.read_text() != before_failed_batch:
            fail("multi_edit_file wrote a partial batch")

        consecutive_second = tools["edit_file"].execute(
            ToolCall(
                id="edit-consecutive",
                name="edit_file",
                arguments={
                    "path": batch_path.name,
                    "old_string": "joined",
                    "new_string": "final",
                },
            ),
            policy,
        )
        if not consecutive_second.ok or batch_path.read_text() != "final gamma\n":
            fail(f"writer did not refresh the ledger after its own write: {consecutive_second!r}")

        diff_path = root / "structured-diff.txt"
        diff_path.write_text("alpha\nkeep\nomega\n")
        tools["read_file"].execute(
            ToolCall(id="read-diff", name="read_file", arguments={"path": diff_path.name}),
            policy,
        )
        diff_result = tools["edit_file"].execute(
            ToolCall(
                id="edit-diff",
                name="edit_file",
                arguments={
                    "path": diff_path.name,
                    "old_string": "keep",
                    "new_string": "changed",
                },
            ),
            policy,
        )
        expected_diff = (
            "--- structured-diff.txt\n+++ structured-diff.txt\n@@ -1,3 +1,3 @@\n"
            " alpha\n-keep\n+changed\n omega"
        )
        if diff_result.content != expected_diff:
            fail(f"unified diff rendering changed: {diff_result!r}")
        expected_payload = {
            "kind": "file_diff",
            "path": "structured-diff.txt",
            "lines_added": 1,
            "lines_removed": 1,
            "truncated": False,
            "hunks": [
                {
                    "old_start": 1,
                    "old_lines": 3,
                    "new_start": 1,
                    "new_lines": 3,
                    "lines": [
                        {"op": "context", "text": "alpha"},
                        {"op": "remove", "text": "keep"},
                        {"op": "add", "text": "changed"},
                        {"op": "context", "text": "omega"},
                    ],
                }
            ],
        }
        if diff_result.payload != expected_payload:
            fail(f"structured diff payload changed: {diff_result.payload!r}")
        if json.loads(json.dumps(diff_result.payload)) != diff_result.payload:
            fail(f"structured diff payload was not JSON-stable: {diff_result.payload!r}")
        diff_ops = {
            line["op"]
            for hunk in diff_result.payload["hunks"]
            for line in hunk["lines"]
        }
        if diff_ops != {"context", "add", "remove"}:
            fail(f"structured diff line operations changed: {diff_ops!r}")

        truncated_path = root / "truncated-diff.txt"
        truncated_path.write_text("a long original line\n")
        tools["read_file"].execute(
            ToolCall(
                id="read-truncated-diff",
                name="read_file",
                arguments={"path": truncated_path.name},
            ),
            policy,
        )
        with mock.patch.object(filesystem_tools, "MAX_READ_BYTES", 10):
            truncated_diff = tools["edit_file"].execute(
                ToolCall(
                    id="edit-truncated-diff",
                    name="edit_file",
                    arguments={
                        "path": truncated_path.name,
                        "old_string": "a long original line",
                        "new_string": "a substantially different line",
                    },
                ),
                policy,
            )
        expected_omitted = (
            "[diff omitted: 1 lines added, 1 removed, over the 10 byte diff limit]"
        )
        if (
            not truncated_diff.ok
            or truncated_diff.content != expected_omitted
            or truncated_diff.payload["hunks"] != []
            or truncated_diff.payload["truncated"] is not True
        ):
            fail(f"oversized diff was not safely omitted: {truncated_diff!r}")

        write_created = tools["write_file"].execute(
            ToolCall(
                id="write-create-unread",
                name="write_file",
                arguments={"path": "write-created.txt", "content": "created"},
            ),
            policy,
        )
        if not write_created.ok or (root / "write-created.txt").read_text() != "created":
            fail(f"write_file did not create an unread path: {write_created!r}")
        write_unread_path = root / "write-unread-existing.txt"
        write_unread_path.write_text("original")
        write_unread = tools["write_file"].execute(
            ToolCall(
                id="write-unread-existing",
                name="write_file",
                arguments={"path": write_unread_path.name, "content": "replacement"},
            ),
            policy,
        )
        if write_unread.ok or write_unread.error != not_read_error:
            fail(f"write_file overwrote an unread existing file: {write_unread!r}")
        if write_unread_path.read_text() != "original":
            fail("write_file changed an unread existing file despite refusing it")

        first_registry = standard_tool_registry()
        second_registry = standard_tool_registry()
        isolated_path = root / "isolated-ledger.txt"
        isolated_path.write_text("before\n")
        first_registry["read_file"].execute(
            ToolCall(id="isolated-read", name="read_file", arguments={"path": isolated_path.name}),
            policy,
        )
        isolated_edit = second_registry["edit_file"].execute(
            ToolCall(
                id="isolated-edit",
                name="edit_file",
                arguments={
                    "path": isolated_path.name,
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            policy,
        )
        if isolated_edit.ok or isolated_edit.error != not_read_error:
            fail(f"read ledger leaked between registries: {isolated_edit!r}")
        if ToolResult(tool_call_id="payload-default", ok=True).payload is not None:
            fail("existing ToolResult callers gained a non-empty payload")
        ok("edit tools enforce fresh reads, apply atomically, and return structured diffs")

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
        # schemas for all eight standard tools --
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

        delayed_token = CancellationToken()

        class _DelayedCancellingResponse(_FakeHttpResponse):
            def read(self) -> bytes:
                time.sleep(0.01)
                delayed_token.cancel()
                return super().read()

        delayed_payload = json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "late answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
        ).encode("utf-8")
        with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: "delayed-cancel-test-key"}):
            with mock.patch(
                "urllib.request.urlopen",
                return_value=_DelayedCancellingResponse(delayed_payload),
            ):
                try:
                    OpenAIProvider().create_response(
                        ModelRequest(messages=[Message(role=Role.USER, content="wait")]),
                        cancel=delayed_token,
                    )
                except ProviderError as exc:
                    fail(f"transport cancellation was wrapped as ProviderError: {exc!r}")
                except OperationCancelled:
                    pass
                else:
                    fail("successful HTTP body bypassed cancellation after response.read()")
        ok("successful HTTP reads recheck cancellation without wrapping it")

        late_agent_token = CancellationToken()

        class _LateCancellingProvider(ModelProvider):
            @property
            def name(self) -> str:
                return "late-cancelling"

            @property
            def wire_format(self) -> int:
                return 4

            def create_response(
                self,
                request: ModelRequest,
                *,
                cancel: CancellationToken | None = None,
            ) -> ModelResponse:
                assert cancel is not None
                time.sleep(0.01)
                cancel.cancel()
                return ModelResponse(Message(Role.ASSISTANT, "late answer"))

        late_agent_result = ApiAgent(
            _LateCancellingProvider(), {}, policy
        ).run(
            [Message(role=Role.USER, content="wait")],
            cancel=late_agent_token,
        )
        if late_agent_result.stopped_reason != "cancelled":
            fail(f"late successful response bypassed agent cancellation: {late_agent_result!r}")
        if not any(
            message.role == Role.ASSISTANT and message.text == "late answer"
            for message in late_agent_result.messages
        ):
            fail(f"late assistant response was not retained: {late_agent_result.messages!r}")
        ok("ApiAgent retains a late assistant response before reporting cancellation")

        retry_request = urllib.request.Request("https://mock.invalid/test")
        backoff_token = CancellationToken()
        timer = threading.Timer(0.02, backoff_token.cancel)
        retryable_error = urllib.error.HTTPError(
            retry_request.full_url,
            503,
            "unavailable",
            {},
            io.BytesIO(b"retry later"),
        )
        started = time.monotonic()
        timer.start()
        try:
            with mock.patch("urllib.request.urlopen", side_effect=retryable_error):
                with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                    try:
                        read_with_retry(
                            retry_request,
                            timeout=2.0,
                            api_key="test-key",
                            operation="cancel test",
                            cancel=backoff_token,
                        )
                    except OperationCancelled as exc:
                        if isinstance(exc, ProviderError):
                            fail("OperationCancelled was wrapped as ProviderError")
                    else:
                        fail("interruptible retry backoff did not raise OperationCancelled")
                    if sleep_mock.called:
                        fail("cancellable retry backoff called time.sleep")
        finally:
            timer.cancel()
            timer.join()
        if time.monotonic() - started >= 0.3:
            fail("retry backoff cancellation did not wake promptly")
        ok("retry backoff wakes promptly without wrapping cancellation as ProviderError")

        retryable_then_success = urllib.error.HTTPError(
            retry_request.full_url,
            503,
            "unavailable",
            {},
            io.BytesIO(b"retry later"),
        )
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[retryable_then_success, _FakeHttpResponse(b"ok")],
        ):
            with mock.patch("orchestra_api.retry.time.sleep") as sleep_mock:
                with mock.patch("orchestra_api.retry.random.uniform", return_value=0.0):
                    raw = read_with_retry(
                        retry_request,
                        timeout=2.0,
                        api_key="test-key",
                        operation="none token test",
                        cancel=None,
                    )
        if raw != b"ok":
            fail(f"cancel=None retry returned unexpected payload: {raw!r}")
        sleep_mock.assert_called_once_with(0.5)
        ok("cancel=None keeps the existing time.sleep retry path")

        shell_token = CancellationToken()
        shell_policy = PermissionPolicy(
            repo_root=root,
            shell_enabled=True,
            shell_allowlist=[(sys.executable,)],
        )
        same_group_proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            child_pgid = os.getpgid(same_group_proc.pid)
            current_pgid = os.getpgid(0)
            if child_pgid != current_pgid:
                fail(
                    "same-group termination test did not exercise the guard: "
                    f"child={child_pgid}, current={current_pgid}"
                )
            with mock.patch("orchestra_api.tools.shell.os.killpg") as killpg_mock:
                _terminate_process_group(same_group_proc)
            if killpg_mock.called:
                fail("same-group termination attempted to signal orchestra's process group")
            try:
                same_group_proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                fail("same-group termination did not kill the child process")
        finally:
            if same_group_proc.poll() is None:
                same_group_proc.kill()
                same_group_proc.wait(timeout=1.0)
        ok("process-group cleanup falls back to child-only kill for orchestra's group")

        real_popen = subprocess.Popen
        children: list[subprocess.Popen] = []

        def _capturing_popen(*args, **kwargs):  # noqa: ANN002, ANN003
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child

        shell_timer = threading.Timer(0.05, shell_token.cancel)
        shell_started = time.monotonic()
        shell_timer.start()
        try:
            with mock.patch(
                "orchestra_api.tools.shell.subprocess.Popen",
                side_effect=_capturing_popen,
            ):
                try:
                    RunShellTool().execute(
                        ToolCall(
                            id="cancel-shell",
                            name="run_shell",
                            arguments={
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    "import time; time.sleep(30)",
                                ]
                            },
                        ),
                        shell_policy,
                        cancel=shell_token,
                    )
                except OperationCancelled:
                    pass
                else:
                    fail("cancelled run_shell returned a ToolResult")
        finally:
            shell_timer.cancel()
            shell_timer.join()
        if time.monotonic() - shell_started >= 1.0:
            fail("cancelled run_shell did not return promptly")
        if len(children) != 1 or children[0].poll() is None:
            fail(f"cancelled run_shell left its child running: {children!r}")
        ok("cancelled run_shell kills and reaps its child process")

        descendant_pid_path = root / "descendant.pid"
        descendant_token = CancellationToken()
        descendant_script = (
            "import subprocess, sys, time; "
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(30)'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            "open(sys.argv[1], 'w').write(str(child.pid)); "
            "time.sleep(30)"
        )
        descendant_timer = threading.Timer(0.2, descendant_token.cancel)
        descendant_timer.start()
        try:
            try:
                shell_tool_for_tree = RunShellTool()
                shell_tool_for_tree.execute(
                    ToolCall(
                        id="cancel-shell-tree",
                        name="run_shell",
                        arguments={
                            "argv": [
                                sys.executable,
                                "-c",
                                descendant_script,
                                str(descendant_pid_path),
                            ]
                        },
                    ),
                    shell_policy,
                    cancel=descendant_token,
                )
            except OperationCancelled:
                pass
            else:
                fail("cancelled process-tree command returned a ToolResult")
        finally:
            descendant_timer.cancel()
            descendant_timer.join()
        if not descendant_pid_path.exists():
            fail("process-tree command did not record its descendant pid before cancellation")
        descendant_pid = int(descendant_pid_path.read_text())
        descendant_deadline = time.monotonic() + 2.0
        descendant_alive = True
        while time.monotonic() < descendant_deadline:
            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                descendant_alive = False
                break
            time.sleep(0.02)
        if descendant_alive:
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            fail(f"cancelled run_shell left descendant pid {descendant_pid} alive")
        ok("cancelled run_shell terminates the complete child process group")

        inherited_pipe_token = CancellationToken()
        inherited_pipe_script = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)']); "
            "time.sleep(30)"
        )
        inherited_pipe_timer = threading.Timer(0.2, inherited_pipe_token.cancel)
        inherited_pipe_started = time.monotonic()
        inherited_pipe_timer.start()
        try:
            try:
                RunShellTool().execute(
                    ToolCall(
                        id="cancel-shell-inherited-pipe",
                        name="run_shell",
                        arguments={
                            "argv": [sys.executable, "-c", inherited_pipe_script]
                        },
                    ),
                    shell_policy,
                    cancel=inherited_pipe_token,
                )
            except OperationCancelled:
                pass
            else:
                fail("cancelled inherited-pipe command returned a ToolResult")
        finally:
            inherited_pipe_timer.cancel()
            inherited_pipe_timer.join()
        inherited_pipe_elapsed = time.monotonic() - inherited_pipe_started
        if inherited_pipe_elapsed >= 1.5:
            fail(
                "cancelled run_shell blocked on an inherited pipe for "
                f"{inherited_pipe_elapsed:.2f}s"
            )
        ok("cancelled run_shell cleanup stays bounded with inherited pipes")

        # -- run_shell behaviour unchanged by the subprocess.run -> Popen rewrite.
        # These are the paths the rewrite could silently break; none of them were
        # covered before, so a regression would have shipped green.
        shell_tool = RunShellTool()

        def _shell(argv: list[str], policy: PermissionPolicy = shell_policy) -> ToolResult:
            return shell_tool.execute(
                ToolCall(id="shell-behaviour", name="run_shell", arguments={"argv": argv}),
                policy,
            )

        success = _shell([sys.executable, "-c", "print('to stdout')"])
        if not success.ok or success.content != "to stdout\n" or success.error is not None:
            fail(f"run_shell success path changed: {success!r}")

        merged = _shell(
            [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"]
        )
        if "out" not in merged.content or "err" not in merged.content:
            fail(f"run_shell no longer merges stdout and stderr: {merged!r}")

        failing = _shell([sys.executable, "-c", "print('partial'); raise SystemExit(7)"])
        if failing.ok or failing.error != "exit code 7" or "partial" not in failing.content:
            fail(f"run_shell nonzero-exit path changed: {failing!r}")

        oversized = _shell(
            [sys.executable, "-c", f"print('x' * {MAX_OUTPUT_CHARS + 500})"]
        )
        if not oversized.content.endswith("\n...(truncated)"):
            fail("run_shell no longer appends the truncation suffix")
        if len(oversized.content) != MAX_OUTPUT_CHARS + len("\n...(truncated)"):
            fail(f"run_shell truncated to the wrong length: {len(oversized.content)}")

        timing_out = shell_tool.execute(
            ToolCall(
                id="shell-timeout",
                name="run_shell",
                arguments={"argv": [sys.executable, "-c", "import time; time.sleep(30)"]},
            ),
            PermissionPolicy(
                repo_root=root,
                shell_enabled=True,
                shell_allowlist=[(sys.executable,)],
                shell_timeout_seconds=0.3,
            ),
        )
        if timing_out.ok or not (timing_out.error or "").startswith("error running command:"):
            fail(f"run_shell timeout path changed: {timing_out!r}")
        ok("run_shell success, stderr, exit-code, truncation, and timeout paths unchanged")

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
        expected_tool_names = {
            "read_file",
            "write_file",
            "edit_file",
            "multi_edit_file",
            "list_files",
            "glob",
            "grep",
            "run_shell",
        }
        if sent_tool_names != expected_tool_names:
            fail(f"expected run_task() request to include {expected_tool_names}, got {sent_tool_names!r}")
        if captured.get("body", {}).get("model") != "request-override-model":
            fail(f"run_task(model=...) did not reach the wire: {captured.get('body')!r}")
        ok("real provider's run_task() request includes all eight standard tool schemas")
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
