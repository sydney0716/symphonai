"""Validation and review helpers for loop card specs."""

from __future__ import annotations


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
