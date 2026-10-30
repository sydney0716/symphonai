"""Mask secret environment values and key-shaped text in outputs."""

from __future__ import annotations

import os
import re
from dataclasses import replace

from symphonai_api.models import ToolResult


_SECRET_NAME_PARTS = ("KEY", "SECRET", "TOKEN", "PASSWORD")
_MIN_SECRET_LENGTH = 8


_REDACT_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED_SECRET]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_SECRET]"),
    (
        re.compile(r"(?i:\bBearer)[ \t]+[A-Za-z0-9._~+/-]{16,}=*"),
        "Bearer [REDACTED_SECRET]",
    ),
    (
        re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b(\s*[:=]\s*)([\"']?)[^\s\"']{8,}"),
        r"\1\2[REDACTED_SECRET]",
    ),
)

def mask_secrets(text: str) -> str:
    """Mask environment secret values, then recognizable secret patterns."""
    secrets = [
        (name, value)
        for name, value in os.environ.items()
        if any(part in name.upper() for part in _SECRET_NAME_PARTS)
        and len(value) >= _MIN_SECRET_LENGTH
    ]

    replacements: dict[str, str] = {}
    for name, value in secrets:
        replacements.setdefault(value, f"[redacted {name}]")
    if replacements:
        pattern = re.compile("|".join(re.escape(value) for value in sorted(replacements, key=len, reverse=True)))
        text = pattern.sub(lambda match: replacements[match.group()], text)
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def mask_tool_result(result: ToolResult) -> ToolResult:
    content = mask_secrets(result.content)
    error = None if result.error is None else mask_secrets(result.error)
    if content == result.content and error == result.error:
        return result
    return replace(result, content=content, error=error)
