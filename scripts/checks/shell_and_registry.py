"""Checks for shell classification, policy bounds, and registry subsets."""

from __future__ import annotations

from pathlib import Path

from orchestra_api import ToolEffect, ToolMetadata
from orchestra_api.models import ToolCall
from orchestra_api.permissions import (
    MAX_SHELL_OUTPUT_CHARS,
    MIN_SHELL_OUTPUT_CHARS,
    PermissionPolicy,
)
from orchestra_api.runner import standard_tool_registry
from orchestra_api.tools.shell_classify import classify
from scripts.checks.harness import check, fail, ok


REPO_ROOT = Path(__file__).resolve().parents[2]


@check("shell.classify_table")
def check_shell_classify_table() -> None:
    git_status_entry = classify(["git", "status"])
    ls_entry = classify(["ls", "-la"])
    if git_status_entry is None or git_status_entry.concurrency_safe:
        fail(f"git status classification changed: {git_status_entry!r}")
    if ls_entry is None or not ls_entry.concurrency_safe:
        fail(f"ls classification changed: {ls_entry!r}")
    for unsafe_argv in (["make"], ["ls", "--color=always"], []):
        if classify(unsafe_argv) is not None:
            fail(f"unsafe argv was classified read-only: {unsafe_argv!r}")
    if classify(["git", "log", "--max-count=3", "main"]) is None:
        fail("git log rejected a safe equals-form flag or operand")
    if classify(["cat", "--", "-weird-name"]) is None:
        fail("cat rejected a post-double-dash operand")


@check("shell.metadata_contract")
def check_shell_metadata_contract() -> None:
    read_only_git_metadata = ToolMetadata(
        effect=ToolEffect.READ_ONLY,
        concurrency_safe=False,
        paths=None,
    )
    read_only_parallel_metadata = ToolMetadata(
        effect=ToolEffect.READ_ONLY,
        concurrency_safe=True,
        paths=None,
    )
    destructive_shell_metadata = ToolMetadata(
        effect=ToolEffect.DESTRUCTIVE,
        concurrency_safe=False,
        paths=None,
    )
    shell_metadata = standard_tool_registry()["run_shell"]
    if shell_metadata.metadata({"argv": ["git", "diff"]}) != read_only_git_metadata:
        fail("git diff metadata changed from the literal read-only serial contract")
    if shell_metadata.metadata({"argv": ["ls"]}) != read_only_parallel_metadata:
        fail("ls metadata changed from the literal read-only parallel contract")


@check("shell.metadata_fails_closed")
def check_shell_metadata_fails_closed() -> None:
    destructive_shell_metadata = ToolMetadata(
        effect=ToolEffect.DESTRUCTIVE,
        concurrency_safe=False,
        paths=None,
    )
    shell_metadata = standard_tool_registry()["run_shell"]
    for unsafe_arguments in (
        {"argv": ["rm", "-rf", "/"]},
        {"argv": ["make"]},
        {},
        {"argv": "ls"},
        {"argv": [3]},
        {"argv": 3},
    ):
        actual = shell_metadata.metadata(unsafe_arguments)
        if actual != destructive_shell_metadata:
            fail(
                "unsafe or malformed shell metadata did not fail closed: "
                f"arguments={unsafe_arguments!r}, metadata={actual!r}"
            )


@check("shell.permission_not_granted")
def check_shell_permission_not_granted() -> None:
    metadata_tools = standard_tool_registry()
    disabled_ls = metadata_tools["run_shell"].execute(
        ToolCall(
            id="classified-but-disabled",
            name="run_shell",
            arguments={"argv": ["ls"]},
        ),
        PermissionPolicy(repo_root=REPO_ROOT),
    )
    if disabled_ls.ok or disabled_ls.error != "run_shell is disabled by this policy":
        fail(f"read-only classification granted shell permission: {disabled_ls!r}")


@check("policy.output_limit_clamp")
def check_policy_output_limit_clamp() -> None:
    clamped_low = PermissionPolicy(repo_root=REPO_ROOT, shell_output_limit_chars=10)
    clamped_high = PermissionPolicy(
        repo_root=REPO_ROOT, shell_output_limit_chars=10_000_000
    )
    kept_limit = PermissionPolicy(repo_root=REPO_ROOT, shell_output_limit_chars=12_345)
    if (
        clamped_low.shell_output_limit_chars != MIN_SHELL_OUTPUT_CHARS
        or clamped_high.shell_output_limit_chars != MAX_SHELL_OUTPUT_CHARS
        or kept_limit.shell_output_limit_chars != 12_345
    ):
        fail(
            "shell output policy limits did not clamp or preserve correctly: "
            f"low={clamped_low.shell_output_limit_chars}, "
            f"high={clamped_high.shell_output_limit_chars}, "
            f"kept={kept_limit.shell_output_limit_chars}"
        )


@check("registry.subsets")
def check_registry_subsets() -> None:
    metadata_tools = standard_tool_registry()
    expected_full_order = [
        "read_file",
        "write_file",
        "edit_file",
        "multi_edit_file",
        "list_files",
        "glob",
        "grep",
        "run_shell",
        "web_fetch",
    ]
    if list(metadata_tools) != expected_full_order:
        fail(f"full standard registry order changed: {list(metadata_tools)!r}")
    narrowed_tools = standard_tool_registry(["grep", "read_file"])
    if list(narrowed_tools) != ["read_file", "grep"]:
        fail(f"narrowed registry lost canonical ordering: {list(narrowed_tools)!r}")
    try:
        standard_tool_registry(["read_fil"])
    except ValueError as exc:
        if str(exc) != "unknown tool name: 'read_fil'":
            fail(f"unknown registry name error changed: {exc!r}")
    else:
        fail("unknown registry tool name was accepted")
    try:
        standard_tool_registry([])
    except ValueError as exc:
        if str(exc) != "names must not be empty; omit it for the full registry":
            fail(f"empty registry name error changed: {exc!r}")
    else:
        fail("empty narrowed registry was accepted")
    ok("shell classification, policy bounds, and registry subsets are exact")
