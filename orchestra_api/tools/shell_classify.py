"""Pure allow-list classification for provably read-only shell commands."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ReadOnlyCommand:
    """One entry in the read-only table: which flags keep it read-only."""

    safe_flags: frozenset[str]
    concurrency_safe: bool = True


READ_ONLY_COMMANDS: dict[tuple[str, ...], ReadOnlyCommand] = {
    ("git", "status"): ReadOnlyCommand(
        frozenset({"--short", "-s", "--porcelain", "--branch", "-b"}),
        concurrency_safe=False,
    ),
    ("git", "log"): ReadOnlyCommand(
        frozenset(
            {"--oneline", "--stat", "--graph", "-n", "--max-count", "--format", "--pretty"}
        ),
        concurrency_safe=False,
    ),
    ("git", "diff"): ReadOnlyCommand(
        frozenset({"--stat", "--cached", "--staged", "--name-only", "--check"}),
        concurrency_safe=False,
    ),
    ("git", "show"): ReadOnlyCommand(
        frozenset({"--stat", "--name-only", "--format", "--pretty"}),
        concurrency_safe=False,
    ),
    ("ls",): ReadOnlyCommand(
        frozenset({"-l", "-a", "-la", "-al", "-h", "-lh", "-R", "-1"})
    ),
    ("cat",): ReadOnlyCommand(frozenset({"-n"})),
    ("head",): ReadOnlyCommand(frozenset({"-n", "-c"})),
    ("tail",): ReadOnlyCommand(frozenset({"-n", "-c"})),
    ("wc",): ReadOnlyCommand(frozenset({"-l", "-c", "-w"})),
    ("pwd",): ReadOnlyCommand(frozenset()),
    ("whoami",): ReadOnlyCommand(frozenset()),
    ("date",): ReadOnlyCommand(frozenset()),
    ("which",): ReadOnlyCommand(frozenset()),
    ("file",): ReadOnlyCommand(frozenset()),
    ("echo",): ReadOnlyCommand(frozenset()),
}


def classify(argv: Sequence[str]) -> ReadOnlyCommand | None:
    """The table entry for `argv` when every token is safe, else None.

    None means "not provably read-only", which is the answer for anything
    this table does not fully understand.
    """
    if not argv or not all(isinstance(token, str) for token in argv):
        return None
    argv_tuple = tuple(argv)
    matched_prefix = next(
        (
            prefix
            for prefix in sorted(READ_ONLY_COMMANDS, key=len, reverse=True)
            if argv_tuple[: len(prefix)] == prefix
        ),
        None,
    )
    if matched_prefix is None:
        return None
    entry = READ_ONLY_COMMANDS[matched_prefix]
    for token in argv[len(matched_prefix) :]:
        if token == "--":
            return entry
        if not token.startswith("-"):
            continue
        flag = token.split("=", 1)[0]
        if flag not in entry.safe_flags:
            return None
    return entry
