"""Classification of provider calls by their source and retry pressure."""

from __future__ import annotations

from enum import Enum


class CallClass(str, Enum):
    FOREGROUND = "foreground"
    """The user's own turn. Visible latency, one at a time."""

    BACKGROUND = "background"
    """Automatic or delegated work that must not amplify provider overload."""
