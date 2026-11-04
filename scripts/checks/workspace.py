"""Per-check temporary workspaces for tool and agent checks."""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from symphonai_api.permissions import PermissionPolicy  # noqa: E402
from symphonai_api.runner import standard_tool_registry  # noqa: E402
from symphonai_api.tools.base import LocalTool  # noqa: E402


@dataclass(frozen=True)
class Workspace:
    root: Path
    outside: Path
    policy: PermissionPolicy
    tools: dict[str, LocalTool]


def handoff_spec(goal: str, command: str = "true") -> str:
    return (
        f"## Goal\n{goal}\n\n## Scope\nCheck fixture\n\n"
        f"## Validation\n```sh\n{command}\n```"
    )


@contextmanager
def workspace() -> Iterator[Workspace]:
    """A seeded temp root, an unrelated outside dir, a policy, and a registry."""
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
        root = Path(tmp).resolve()
        outside = Path(outside_tmp).resolve()
        (root / "existing.txt").write_text("hello from disk")
        (root / ".env").write_text("SECRET=do-not-read-me")
        policy = PermissionPolicy(repo_root=root, allowed_write_scope=[root])
        yield Workspace(
            root=root,
            outside=outside,
            policy=policy,
            tools=standard_tool_registry(),
        )
