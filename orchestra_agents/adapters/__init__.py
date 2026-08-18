"""Concrete SubagentProvider implementations.

Only `FakeProvider` exists today (in-memory, no external calls). Real
CLI-backed adapters for claude/codex/gemini are intentionally not
implemented yet -- see `docs/orchestra-agent-adapters.md`.
"""

from __future__ import annotations

from orchestra_agents.adapters.fake import FakeProvider

__all__ = ["FakeProvider"]
