"""Fixture-free checks for models and content."""

from __future__ import annotations

import json
import tempfile
import unittest.mock as mock
from pathlib import Path
import symphonai_api.content as content
from symphonai_api.compaction import _message_excerpts, estimate_message_tokens
from symphonai_api.content import content_block_from_bytes, content_block_from_path, detect_media_type
from symphonai_api.models import DocumentBlock, ImageBlock, Message, ModelRequest, Role, TextBlock, ToolCall, ToolResult, has_attachments, reject_system_attachments
from symphonai_api.providers.anthropic_provider import _build_request_body as _build_anthropic_body
from symphonai_api.providers.gemini_provider import _build_request_body as _build_gemini_body
from symphonai_api.providers.openai_provider import _build_request_body as _build_openai_body
from scripts.checks.harness import check, fail


@check("content.message_normalization")
def check_content_message_normalization() -> None:
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

@check("content.attachment_construction")
def check_content_attachment_construction() -> None:
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

@check("content.system_attachments_refused")
def check_content_system_attachments_refused() -> None:
    image_block = ImageBlock(data="aW1hZ2U=", media_type="image/png")
    system_attachment_error = (
        "system-role messages cannot carry attachments; put the "
        "image or document on a user message"
    )
    text_system_messages = [Message(Role.SYSTEM, "instructions")]
    if reject_system_attachments(text_system_messages) is not None:
        fail("text-only system message was rejected")
    system_attachment_messages = [Message(Role.SYSTEM, image_block)]
    try:
        reject_system_attachments(system_attachment_messages)
    except ValueError as exc:
        if str(exc) != system_attachment_error:
            fail(f"system attachment refusal changed: {exc!r}")
    else:
        fail("reject_system_attachments accepted a system image")
    system_attachment_request = ModelRequest(messages=system_attachment_messages)
    attachment_builders = {
        "OpenAI": lambda: _build_openai_body(
            system_attachment_request, "test-model"
        ),
        "Anthropic": lambda: _build_anthropic_body(
            system_attachment_request, "test-model", 100
        ),
        "Gemini": lambda: _build_gemini_body(system_attachment_request),
    }
    for provider_name, build in attachment_builders.items():
        try:
            build()
        except ValueError as exc:
            if str(exc) != system_attachment_error:
                fail(f"{provider_name} system attachment refusal changed: {exc!r}")
        else:
            fail(f"{provider_name} accepted a system-role attachment")

@check("content.provider_encoding")
def check_content_provider_encoding() -> None:
    image_block = ImageBlock(data="aW1hZ2U=", media_type="image/png")
    document_block = DocumentBlock(data="cGRm", filename="notes.pdf")
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
