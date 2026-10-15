"""Workspace-backed checks for search."""

from __future__ import annotations

import unittest.mock as mock
import shutil
import subprocess
import tempfile
from pathlib import Path
from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.models import ToolCall
from symphonai_api.permissions import DEFAULT_FORBIDDEN_PATTERNS, PermissionPolicy
from symphonai_api.tools.search import GlobTool, GrepTool
from scripts.checks.harness import check, fail
from scripts.checks.workspace import search_tree


@check("search.permission_gates")
def check_search_permission_gates() -> None:
    with search_tree() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        search_root = root / "search-fixture"
        nested = search_root / "nested"
        ordered = search_root / "ordered"
        cancel_root = search_root / "cancel"
        secret_value = "SEARCH_FIXTURE_SECRET_62491"
        top_level_glob = tools["glob"].execute(
            ToolCall(
                id="glob-top-level",
                name="glob",
                arguments={"path": "search-fixture", "pattern": "*.py", "head_limit": 0},
            ),
            policy,
        )
        recursive_glob = tools["glob"].execute(
            ToolCall(
                id="glob-recursive",
                name="glob",
                arguments={"path": "search-fixture", "pattern": "**/*.py", "head_limit": 0},
            ),
            policy,
        )
        if top_level_glob.content != "search-fixture/top.py":
            fail(f"*.py crossed directories: {top_level_glob!r}")
        if set(recursive_glob.content.splitlines()) != {
            "search-fixture/top.py",
            "search-fixture/nested/a.py",
        }:
            fail(f"**/*.py did not cover both depths: {recursive_glob!r}")

        secret_glob = tools["glob"].execute(
            ToolCall(
                id="glob-secrets",
                name="glob",
                arguments={"path": "search-fixture", "pattern": "**/*", "head_limit": 0},
            ),
            policy,
        )
        forbidden_names = (".env", "secret.pem", "node_modules", "escape.txt", "followed.py")
        if not secret_glob.ok or any(name in secret_glob.content for name in forbidden_names):
            fail(f"glob exposed a forbidden or escaped file: {secret_glob!r}")
        secret_grep = tools["grep"].execute(
            ToolCall(
                id="grep-secrets",
                name="grep",
                arguments={"path": "search-fixture", "pattern": secret_value, "head_limit": 0},
            ),
            policy,
        )
        if secret_grep.content != "no matches" or secret_value in secret_grep.content:
            fail(f"grep exposed forbidden contents: {secret_grep!r}")

        for tool_name, arguments in (
            (
                "glob",
                {"path": "search-fixture/ordered", "pattern": "*.ord", "head_limit": 0},
            ),
            (
                "grep",
                {"path": "search-fixture", "glob": "**/*.py", "pattern": "needle"},
            ),
        ):
            with mock.patch.object(policy, "check_read", wraps=policy.check_read) as read_gate:
                gated_result = tools[tool_name].execute(
                    ToolCall(
                        id=f"{tool_name}-file-gates",
                        name=tool_name,
                        arguments=arguments,
                    ),
                    policy,
                )
            checked_paths = {
                Path(call.args[0]).resolve()
                for call in read_gate.call_args_list
                if call.args
            }
            result_paths = [
                line.split(":", 1)[0]
                for line in gated_result.content.splitlines()
                if line and not line.startswith("[")
            ]
            if any((root / result_path).resolve() not in checked_paths for result_path in result_paths):
                fail(f"{tool_name} included a file without calling check_read: {read_gate.call_args_list!r}")

@check("search.ordering_and_pagination")
def check_search_ordering_and_pagination() -> None:
    with search_tree() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        search_root = root / "search-fixture"
        nested = search_root / "nested"
        ordered = search_root / "ordered"
        cancel_root = search_root / "cancel"
        secret_value = "SEARCH_FIXTURE_SECRET_62491"
        ordered_call = {
            "path": "search-fixture/ordered",
            "pattern": "*.ord",
            "head_limit": 0,
        }
        ordered_result = tools["glob"].execute(
            ToolCall(id="glob-order", name="glob", arguments=ordered_call), policy
        )
        expected_order = [
            "search-fixture/ordered/newest.ord",
            "search-fixture/ordered/alpha.ord",
            "search-fixture/ordered/beta.ord",
            "search-fixture/ordered/older.ord",
            "search-fixture/ordered/statfail.ord",
        ]
        if ordered_result.content.splitlines() != expected_order:
            fail(f"glob mtime/path ordering changed: {ordered_result!r}")

        original_stat = Path.stat

        def _failing_stat(path: Path, *args, **kwargs):  # noqa: ANN002, ANN003
            if path.name == "statfail.ord":
                raise OSError("scripted stat race")
            return original_stat(path, *args, **kwargs)

        with mock.patch.object(Path, "stat", _failing_stat):
            stat_race = tools["glob"].execute(
                ToolCall(id="glob-stat-race", name="glob", arguments=ordered_call), policy
            )
        if not stat_race.ok or stat_race.content.splitlines() != expected_order:
            fail(f"glob did not tolerate a failed stat as mtime zero: {stat_race!r}")

        no_cap = tools["glob"].execute(
            ToolCall(
                id="glob-no-cap",
                name="glob",
                arguments={**ordered_call, "head_limit": 10},
            ),
            policy,
        )
        capped = tools["glob"].execute(
            ToolCall(
                id="glob-cap",
                name="glob",
                arguments={**ordered_call, "head_limit": 2},
            ),
            policy,
        )
        unlimited = tools["glob"].execute(
            ToolCall(id="glob-unlimited", name="glob", arguments=ordered_call), policy
        )
        notice = "[2 of 5 results; pass offset=2 for the next page]"
        if "results; pass offset=" in no_cap.content or capped.content.splitlines()[-1] != notice:
            fail(f"glob cap notice did not report only a real truncation: {no_cap!r}, {capped!r}")
        if unlimited.content.splitlines() != expected_order or "results; pass offset=" in unlimited.content:
            fail(f"glob head_limit=0 did not return all results: {unlimited!r}")
        page = tools["glob"].execute(
            ToolCall(
                id="glob-page",
                name="glob",
                arguments={**ordered_call, "head_limit": 2, "offset": 2},
            ),
            policy,
        )
        if page.content.splitlines()[:2] != expected_order[2:4]:
            fail(f"glob pagination changed result ordering: {page!r}")
        past_glob = tools["glob"].execute(
            ToolCall(
                id="glob-past-end",
                name="glob",
                arguments={**ordered_call, "offset": 50},
            ),
            policy,
        )
        if past_glob.content != "[no results at offset 50; 5 results total]":
            fail(f"glob past-end pagination returned an ambiguous result: {past_glob!r}")

@check("search.grep_modes")
def check_search_grep_modes() -> None:
    with search_tree() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        search_root = root / "search-fixture"
        nested = search_root / "nested"
        ordered = search_root / "ordered"
        cancel_root = search_root / "cancel"
        secret_value = "SEARCH_FIXTURE_SECRET_62491"
        grep_files = tools["grep"].execute(
            ToolCall(
                id="grep-files",
                name="grep",
                arguments={
                    "path": "search-fixture",
                    "glob": "**/*.py",
                    "pattern": "needle",
                    "head_limit": 1,
                },
            ),
            policy,
        )
        if grep_files.content.splitlines() != [
            "search-fixture/top.py",
            "[1 of 2 results; pass offset=1 for the next page]",
        ]:
            fail(f"grep files mode did not cap matching files: {grep_files!r}")
        grep_content = tools["grep"].execute(
            ToolCall(
                id="grep-content",
                name="grep",
                arguments={
                    "path": "search-fixture",
                    "glob": "top.py",
                    "pattern": "NEEDLE",
                    "case_insensitive": True,
                    "output_mode": "content",
                    "head_limit": 2,
                },
            ),
            policy,
        )
        if grep_content.content.splitlines() != [
            "search-fixture/top.py:1\tneedle",
            "search-fixture/top.py:3\tneedle",
            "[2 of 3 results; pass offset=2 for the next page]",
        ]:
            fail(f"grep content mode did not cap matching lines: {grep_content!r}")
        skipped = tools["grep"].execute(
            ToolCall(
                id="grep-skips",
                name="grep",
                arguments={
                    "path": "search-fixture",
                    "glob": "skip-*.txt",
                    "pattern": "needle",
                },
            ),
            policy,
        )
        if not skipped.ok or skipped.content != "no matches":
            fail(f"grep did not silently skip binary and oversized files: {skipped!r}")
        past_grep = tools["grep"].execute(
            ToolCall(
                id="grep-past-end",
                name="grep",
                arguments={
                    "path": "search-fixture",
                    "glob": "**/*.py",
                    "pattern": "needle",
                    "offset": 50,
                },
            ),
            policy,
        )
        if past_grep.content != "[no results at offset 50; 2 results total]":
            fail(f"grep past-end pagination returned an ambiguous result: {past_grep!r}")
        empty_search = search_root / "empty"
        empty_search.mkdir()
        empty_glob = tools["glob"].execute(
            ToolCall(
                id="glob-empty-tree",
                name="glob",
                arguments={"path": "search-fixture/empty", "pattern": "**/*", "offset": 50},
            ),
            policy,
        )
        empty_grep = tools["grep"].execute(
            ToolCall(
                id="grep-empty-tree",
                name="grep",
                arguments={"path": "search-fixture/empty", "pattern": "needle", "offset": 50},
            ),
            policy,
        )
        if empty_glob.content != "no files matched" or empty_grep.content != "no matches":
            fail(f"empty search messages changed: glob={empty_glob!r}, grep={empty_grep!r}")

@check("search.cancellation")
def check_search_cancellation() -> None:
    with search_tree() as ws:
        root = ws.root
        outside_tmp = str(ws.outside)
        policy = ws.policy
        tools = ws.tools
        search_root = root / "search-fixture"
        nested = search_root / "nested"
        ordered = search_root / "ordered"
        cancel_root = search_root / "cancel"
        secret_value = "SEARCH_FIXTURE_SECRET_62491"
        with mock.patch.object(
            tools["grep"], "_execute", side_effect=AssertionError("validation was bypassed")
        ):
            invalid_regex = tools["grep"].execute(
                ToolCall(id="grep-invalid", name="grep", arguments={"pattern": "["}), policy
            )
        if invalid_regex.ok or not invalid_regex.error.startswith("invalid regular expression: "):
            fail(f"grep invalid-regex validation changed: {invalid_regex!r}")

        for index in range(70):
            (cancel_root / f"{index:03}.cancel").write_text("needle")
        for tool_name, arguments in (
            ("glob", {"path": "search-fixture/cancel", "pattern": "*.cancel"}),
            ("grep", {"path": "search-fixture/cancel", "pattern": "needle"}),
        ):
            cancel_token = CancellationToken()
            real_check_read = policy.check_read
            seen_cancel_files = [0]

            def _cancel_during_gate(path):  # noqa: ANN001
                decision = real_check_read(path)
                if Path(path).suffix == ".cancel":
                    seen_cancel_files[0] += 1
                    if seen_cancel_files[0] == 1:
                        cancel_token.cancel()
                return decision

            try:
                with mock.patch.object(policy, "check_read", side_effect=_cancel_during_gate):
                    tools[tool_name].execute(
                        ToolCall(
                            id=f"{tool_name}-cancel",
                            name=tool_name,
                            arguments=arguments,
                        ),
                        policy,
                        cancel=cancel_token,
                    )
            except OperationCancelled:
                pass
            else:
                fail(f"{tool_name} did not observe cancellation inside its file walk")


@check("search.gitignore_respected")
def check_gitignore_respected() -> None:
    if shutil.which("git") is None:
        print("SKIP: git is unavailable")
        return
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        (root / ".gitignore").write_text("build/\ndist/\n*.log\n", encoding="utf-8")
        for relative in (
            "src/a.py", "build/b.py", "build/kept.py", "dist/c.py", "notes.log",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("needle\n", encoding="utf-8")
        subprocess.run(["git", "add", ".gitignore", "src/a.py"], cwd=root, check=True)
        subprocess.run(["git", "add", "-f", "build/kept.py"], cwd=root, check=True)

        policy = PermissionPolicy(
            repo_root=root,
            forbidden_patterns=tuple(
                pattern for pattern in DEFAULT_FORBIDDEN_PATTERNS
                if pattern not in ("build/", "dist/")
            ),
        )
        glob = GlobTool()
        grep = GrepTool()
        root_glob = glob.execute(
            ToolCall(id="gitignore-glob-root", name="glob", arguments={"pattern": "**/*.py", "head_limit": 0}),
            policy,
        )
        if set(root_glob.content.splitlines()) != {"src/a.py", "build/kept.py"}:
            fail(f"root glob did not skip ignored untracked paths and keep the tracked file: {root_glob!r}")
        root_grep = grep.execute(
            ToolCall(
                id="gitignore-grep-root", name="grep",
                arguments={"pattern": "needle", "output_mode": "content", "head_limit": 0},
            ),
            policy,
        )
        if any(path in root_grep.content for path in ("build/b.py", "dist/c.py", "notes.log")):
            fail(f"root grep returned an ignored path: {root_grep!r}")
        build_glob = glob.execute(
            ToolCall(
                id="gitignore-glob-named", name="glob",
                arguments={"path": "build", "pattern": "*.py", "head_limit": 0},
            ),
            policy,
        )
        if set(build_glob.content.splitlines()) != {"build/b.py", "build/kept.py"}:
            fail(f"a named ignored directory was filtered: {build_glob!r}")
