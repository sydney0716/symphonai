"""Bounded file search for composer references."""

from __future__ import annotations

import os
from pathlib import Path

from symphonai_api.permissions import PermissionPolicy


MAX_SEARCH_FILES = 20_000


def _is_subsequence(query: str, value: str) -> bool:
    position = iter(value)
    return all(any(character == candidate for candidate in position) for character in query)


def repository_files(policy: PermissionPolicy, query: str, limit: int) -> tuple[list[str], bool]:
    """Return readable repository paths ranked for an @file query."""
    root = policy.repo_root
    paths: list[str] = []
    visited = 0
    truncated = False
    for directory, names, files in os.walk(root):
        names[:] = sorted(name for name in names if name != ".git")
        for name in sorted(files):
            path = Path(directory) / name
            if not path.is_file():
                continue
            visited += 1
            if visited > MAX_SEARCH_FILES:
                truncated = True
                break
            if policy.check_read(path).allowed:
                paths.append(path.relative_to(root).as_posix())
        if truncated:
            break

    if not query:
        return paths[:limit], truncated

    needle = query.casefold()

    def rank(path: str) -> tuple[int, int, str]:
        value = path.casefold()
        basename = value.rsplit("/", 1)[-1]
        if basename.startswith(needle):
            priority = 0
        elif needle in basename:
            priority = 1
        elif needle in value:
            priority = 2
        else:
            priority = 3
        return priority, len(path), value

    matches = [
        path for path in paths
        if needle in path.casefold() or _is_subsequence(needle, path.casefold())
    ]
    matches.sort(key=rank)
    return matches[:limit], truncated
