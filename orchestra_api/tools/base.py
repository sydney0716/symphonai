"""The LocalTool contract every locally-executed agent tool implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from orchestra_api.models import ToolCall, ToolResult

if TYPE_CHECKING:
    from orchestra_api.cancellation import CancellationToken
    from orchestra_api.permissions import PermissionPolicy


class LocalTool(ABC):
    """Abstract contract for a tool the ApiAgent can execute locally.

    Every implementation must consult the given `PermissionPolicy` before
    doing anything with side effects (filesystem, subprocess, ...) and
    return a `ToolResult` with `ok=False` on denial rather than raising.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable tool name as the model refers to it, e.g. "read_file"."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short, human/model-readable description of what this tool does."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON-Schema-shaped {"type": "object", "properties": {...}, "required": [...]}
        describing this tool's arguments, in a vendor-neutral shape. Used by
        `orchestra_api.tool_schema` to build a provider-specific tool
        definition so a real model is actually shown this tool exists.
        """

    @abstractmethod
    def execute(
        self,
        tool_call: ToolCall,
        policy: "PermissionPolicy",
        cancel: "CancellationToken | None" = None,
    ) -> ToolResult:
        """Run this tool for `tool_call`, gated by `policy`."""
