"""The LocalTool contract every locally-executed agent tool implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from orchestra_api.models import ToolCall, ToolResult
from orchestra_api.tools.metadata import ToolMetadata

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

    def execute(
        self,
        tool_call: ToolCall,
        policy: "PermissionPolicy",
        cancel: "CancellationToken | None" = None,
    ) -> ToolResult:
        """Run the fixed stages, then this tool's work.

        cancel -> validate -> the tool's own `_execute`, which is where
        `PermissionPolicy` is consulted. Permissions are never reached by a
        call that failed validation.
        """
        if cancel is not None:
            cancel.raise_if_cancelled()
        error = self.validate(tool_call.arguments)
        if error is not None:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=error)
        return self._execute(tool_call, policy, cancel=cancel)

    def validate(self, arguments: dict) -> str | None:
        """Reject malformed arguments before permissions or work.

        Returns an error message, or None when the arguments are usable.
        Must be pure: no filesystem, no subprocess, no network. Checking
        *shape* only -- whether a path is allowed is `PermissionPolicy`'s
        answer, not this one.
        """
        return None

    @abstractmethod
    def metadata(self, arguments: dict) -> ToolMetadata:
        """What the call described by `arguments` does to the world.

        A tool whose answer does not vary ignores the argument. Must be pure
        and cheap: it is called before the call runs, possibly on arguments
        that will fail validation, and possibly many times.
        """

    @abstractmethod
    def _execute(
        self,
        tool_call: ToolCall,
        policy: "PermissionPolicy",
        cancel: "CancellationToken | None" = None,
    ) -> ToolResult:
        """This tool's actual work, gated by `policy`."""
