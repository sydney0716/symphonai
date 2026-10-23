"""Validation and review helpers for loop card specs."""

from __future__ import annotations

import hashlib
import re


def spec_validation(text: str) -> list[str]:
    validation = []
    in_section = False
    in_fence = False
    section_level = 0
    for line in text.splitlines():
        if not in_fence:
            heading = re.match(r"^(#{1,6})[ \t]+(.+?)\s*$", line)
            if heading:
                level = len(heading.group(1))
                title = heading.group(2).rstrip("# ").strip()
                if in_section:
                    if level <= section_level:
                        break
                elif title.casefold() == "validation":
                    in_section = True
                    section_level = level
                continue
        if not in_section:
            continue
        if line.startswith("```"):
            if in_fence:
                break
            in_fence = line[3:].strip() in ("", "bash", "sh")
            continue
        if in_fence and line.strip():
            validation.append(line)
    return validation


def review_verdict(answer: str) -> tuple[str, list[str]]:
    last = next((line.strip() for line in reversed(answer.splitlines()) if line.strip()), "")
    if last == "Verdict: pass":
        return "passed", []
    prefix = "Verdict: follow-ups:"
    if last.startswith(prefix):
        paths = [part.strip() for part in last[len(prefix):].split(",") if part.strip()]
        if paths:
            return "follow-ups", paths
    return "no-verdict", []


def patch_digest(patch: str) -> str:
    return hashlib.sha256(patch.encode("utf-8", errors="surrogateescape")).hexdigest()
