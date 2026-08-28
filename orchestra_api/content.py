"""Safe construction of base64 image and PDF content blocks."""

from __future__ import annotations

import base64
from pathlib import Path

from orchestra_api.models import DocumentBlock, ImageBlock

MAX_ATTACHMENT_BYTES = 5_000_000
SUPPORTED_IMAGE_MEDIA_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")
DOCUMENT_MEDIA_TYPE = "application/pdf"


def detect_media_type(raw: bytes) -> str | None:
    """The media type of `raw` from its magic bytes, or None if unrecognized."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"%PDF-"):
        return DOCUMENT_MEDIA_TYPE
    return None


def content_block_from_bytes(
    raw: bytes, *, filename: str | None = None
) -> ImageBlock | DocumentBlock:
    """Build the block for `raw`, or raise ValueError explaining why not."""
    if len(raw) > MAX_ATTACHMENT_BYTES:
        raise ValueError(
            f"attachment is {len(raw)} bytes, over the {MAX_ATTACHMENT_BYTES} byte limit"
        )
    media_type = detect_media_type(raw)
    if media_type is None:
        raise ValueError(
            "unsupported attachment type; expected PNG, JPEG, GIF, WEBP, or PDF"
        )
    data = base64.b64encode(raw).decode("ascii")
    if media_type == DOCUMENT_MEDIA_TYPE:
        return DocumentBlock(data=data, filename=filename)
    return ImageBlock(data=data, media_type=media_type)


def content_block_from_path(path: str | Path) -> ImageBlock | DocumentBlock:
    """Read `path` and build its block. `filename` comes from the path's name."""
    resolved = Path(path)
    return content_block_from_bytes(resolved.read_bytes(), filename=resolved.name)
