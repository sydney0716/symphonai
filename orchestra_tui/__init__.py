"""Optional Textual UI package for the Orchestra API leader."""

from __future__ import annotations

from typing import Any

__all__ = ["OrchestraTuiApp"]


def __getattr__(name: str) -> Any:
    if name == "OrchestraTuiApp":
        from orchestra_tui.app import OrchestraTuiApp

        return OrchestraTuiApp
    raise AttributeError(name)
