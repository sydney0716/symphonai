"""Per-call descriptions of local-tool effects and result shapes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from symphonai_api.cancellation import OperationCancelled
from symphonai_api.identity import SCHEMA_VERSION

if TYPE_CHECKING:
    from symphonai_api.tools.base import LocalTool


class ToolEffect(str, Enum):
    """What a call does to state outside this process.

    Metadata is descriptive, not enforcement. A call declared READ_ONLY that
    writes is a declaration bug, not a security boundary.
    """

    READ_ONLY = "read_only"
    """Observes only. Running it twice changes nothing."""

    MUTATING = "mutating"
    """Changes state that could be reconstructed or undone."""

    DESTRUCTIVE = "destructive"
    """Can lose data irrecoverably, or can do anything at all."""


class ResultHint(str, Enum):
    """How a consumer should render this call's result.

    A hint names a *shape*, never a widget: `symphonai_api` must stay
    ignorant of any UI. Later phases extend this; nothing outside
    metadata.py may assume the set is closed.
    """

    TEXT = "text"
    FILE_LIST = "file_list"
    DIFF = "diff"


class InterruptBehavior(str, Enum):
    """What a mid-run user interrupt does to this call."""

    CANCEL = "cancel"
    """Stop it and discard its work. The current behaviour of every tool."""

    BLOCK = "block"
    """Let it finish while the interrupting message waits."""


@dataclass(frozen=True)
class ToolMetadata:
    """What one specific call does. Declared per call, not per tool."""

    effect: ToolEffect
    concurrency_safe: bool
    """safe to execute in parallel with other concurrency-safe calls *in the
    same turn*. A call that mutates shared state is not, even when it is fast.
    Phase 03 reads this to parallelise; a wrong `True` here is a data race later,
    so when in doubt, `False`.
    """
    paths: tuple[str, ...] | None
    """every path this call can touch, exactly as the arguments spell them:
    unresolved and unvalidated, because resolution is `PermissionPolicy`'s
    job and this must stay pure. `None` means the blast radius is **not
    derivable from the arguments** — `run_shell`'s argv can reach anywhere. An
    empty tuple means the call touches no paths at all. The two are different
    answers and callers must treat `None` as the dangerous one. A tool whose path
    argument is missing or of the wrong type returns `None`, not `()`.
    """
    result_hint: ResultHint = ResultHint.TEXT
    interrupt_behavior: InterruptBehavior = InterruptBehavior.CANCEL
    schema_version: int = SCHEMA_VERSION


FAIL_CLOSED = ToolMetadata(
    effect=ToolEffect.DESTRUCTIVE,
    concurrency_safe=False,
    paths=None,
)
"""What a caller must assume when a call's metadata cannot be determined."""


def safe_metadata(tool: "LocalTool", arguments: dict) -> ToolMetadata:
    """`tool.metadata(arguments)`, or `FAIL_CLOSED` if it raises.

    `metadata` receives raw model-supplied arguments and may parse them, so it
    can raise on input no one anticipated. Every caller that decides something
    consequential -- phase 03's scheduler above all -- must go through this,
    never through `tool.metadata` directly.

    `OperationCancelled` is re-raised, never converted. Swallowing it would
    make a cancelled turn look like an ordinary unsafe call, which is the same
    rule `symphonai_api.events.py:emit` follows.
    """
    try:
        return tool.metadata(arguments)
    except OperationCancelled:
        raise
    except Exception:
        return FAIL_CLOSED
