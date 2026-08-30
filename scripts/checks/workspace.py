"""Per-check temporary workspaces for tool and agent checks."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from orchestra_api.models import ToolCall  # noqa: E402
from orchestra_api.permissions import PermissionPolicy  # noqa: E402
from orchestra_api.runner import standard_tool_registry  # noqa: E402
from orchestra_api.tools.base import LocalTool  # noqa: E402
import orchestra_api.tools.filesystem as filesystem_tools  # noqa: E402


@dataclass(frozen=True)
class Workspace:
    root: Path
    outside: Path
    policy: PermissionPolicy
    tools: dict[str, LocalTool]


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


@contextmanager
def search_tree() -> Iterator[Workspace]:
    """`workspace()` plus the glob/grep tree the search checks share."""
    with workspace() as ws:
        search_root = ws.root / "search-fixture"
        nested = search_root / "nested"
        ordered = search_root / "ordered"
        cancel_root = search_root / "cancel"
        nested.mkdir(parents=True)
        ordered.mkdir()
        cancel_root.mkdir()
        (search_root / "top.py").write_text("needle\nother\nneedle\nneedle\n")
        (search_root / "top.txt").write_text("ordinary text\n")
        (nested / "a.py").write_text("first\nneedle\n")
        (nested / "b.txt").write_text("needle\n")
        os.utime(search_root / "top.py", (600, 600))
        os.utime(nested / "a.py", (500, 500))
        os.utime(nested / "b.txt", (400, 400))
        secret_value = "SEARCH_FIXTURE_SECRET_62491"
        (search_root / ".env").write_text(secret_value)
        (search_root / "secret.pem").write_text(secret_value)
        (search_root / "node_modules").mkdir()
        (search_root / "node_modules" / "hidden.py").write_text(secret_value)
        (search_root / "skip-binary.txt").write_bytes(b"needle\xff\xfe")
        (search_root / "skip-large.txt").write_bytes(
            b"needle\n" + b"x" * filesystem_tools.MAX_READ_BYTES
        )
        outside_file = ws.outside / "outside.txt"
        outside_file.write_text(f"needle {secret_value}")
        (search_root / "escape.txt").symlink_to(outside_file)
        outside_directory = ws.outside / "outside-directory"
        outside_directory.mkdir()
        (outside_directory / "followed.py").write_text("needle")
        (search_root / "linked-directory").symlink_to(
            outside_directory,
            target_is_directory=True,
        )

        ordered_mtimes = {
            "newest.ord": 500,
            "alpha.ord": 400,
            "beta.ord": 400,
            "older.ord": 300,
            "statfail.ord": 200,
        }
        for filename, mtime in ordered_mtimes.items():
            path = ordered / filename
            path.write_text(filename)
            os.utime(path, (mtime, mtime))
        yield ws


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _selfcheck() -> None:
    with workspace() as first, workspace() as second:
        first_root = first.root
        first_outside = first.outside
        second_root = second.root
        second_outside = second.outside
        _require(first.root != second.root, "workspace roots were reused")
        _require(first.tools is not second.tools, "workspace registries were reused")
        _require(
            first.tools["read_file"] is not second.tools["read_file"],
            "workspace tool instances were reused",
        )
        _require(
            (first.root / "existing.txt").read_text() == "hello from disk",
            "existing.txt seed changed",
        )
        _require(
            (first.root / ".env").read_text() == "SECRET=do-not-read-me",
            ".env seed changed",
        )
        _require(
            not first.outside.is_relative_to(first.root),
            "outside directory is under the workspace root",
        )
        _require(first.policy.repo_root == first.root, "policy root changed")
        _require(
            first.policy.allowed_write_scope == [first.root],
            "policy write scope changed",
        )
        _require(len(first.tools) == 8, "workspace registry is not complete")

        read = first.tools["read_file"].execute(
            ToolCall(
                id="workspace-ledger-read",
                name="read_file",
                arguments={"path": "existing.txt"},
            ),
            first.policy,
        )
        _require(read.ok, f"first workspace read failed: {read!r}")
        isolated_edit = second.tools["edit_file"].execute(
            ToolCall(
                id="workspace-ledger-edit",
                name="edit_file",
                arguments={
                    "path": "existing.txt",
                    "old_string": "hello",
                    "new_string": "goodbye",
                },
            ),
            second.policy,
        )
        _require(
            isolated_edit.error
            == "file has not been read yet; read it with read_file before editing it",
            f"workspace registries shared a read ledger: {isolated_edit!r}",
        )

    _require(not first_root.exists(), "first workspace root was not removed")
    _require(not first_outside.exists(), "first outside directory was not removed")
    _require(not second_root.exists(), "second workspace root was not removed")
    _require(not second_outside.exists(), "second outside directory was not removed")

    with search_tree() as ws:
        top_py = ws.root / "search-fixture" / "top.py"
        _require(top_py.stat().st_mtime == 600, "top.py mtime is not pinned to 600")
    print("workspace selfcheck passed")


if __name__ == "__main__":
    _selfcheck()
