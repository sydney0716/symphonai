"""Optional Textual UI package for the SymphonAI API leader."""

from __future__ import annotations

from typing import Any

__all__ = ["SymphonAITuiApp"]


def __getattr__(name: str) -> Any:
    if name == "SymphonAITuiApp":
        from symphonai_tui.app import SymphonAITuiApp

        return SymphonAITuiApp
    raise AttributeError(name)
