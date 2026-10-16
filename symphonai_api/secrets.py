"""Mask secret environment values before tool results enter a transcript."""

from __future__ import annotations

import os
import re
from dataclasses import replace

from symphonai_api.models import ToolResult


_SECRET_NAME_PARTS = ("KEY", "SECRET", "TOKEN", "PASSWORD")
_MIN_SECRET_LENGTH = 8


def mask_tool_result(result: ToolResult) -> ToolResult:
    secrets = [
        (name, value)
        for name, value in os.environ.items()
        if any(part in name.upper() for part in _SECRET_NAME_PARTS)
        and len(value) >= _MIN_SECRET_LENGTH
    ]
    if not secrets:
        return result

    replacements: dict[str, str] = {}
    for name, value in secrets:
        replacements.setdefault(value, f"[redacted {name}]")
    pattern = re.compile("|".join(re.escape(value) for value in sorted(replacements, key=len, reverse=True)))

    def mask(text: str) -> str:
        return pattern.sub(lambda match: replacements[match.group()], text)

    content = mask(result.content)
    error = None if result.error is None else mask(result.error)
    if content == result.content and error == result.error:
        return result
    return replace(result, content=content, error=error)
