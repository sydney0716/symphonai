"""Bounded file search for composer references."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from symphonai_api.permissions import PermissionPolicy


MAX_SEARCH_FILES = 20_000
MAX_TREE_ENTRIES = 2_000
_CACHE_SECONDS = 10.0
_CACHE_LOCK = threading.Lock()
_CANDIDATE_CACHE: dict[Path, tuple[float, tuple[str, ...], bool]] = {}


def _is_subsequence(query: str, value: str) -> bool:
    position = iter(value)
    return all(any(character == candidate for candidate in position) for character in query)


def _candidate_paths(policy: PermissionPolicy, root: Path) -> tuple[tuple[str, ...], bool]:
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CANDIDATE_CACHE.get(root)
        if cached is not None and now - cached[0] < _CACHE_SECONDS:
            return cached[1], cached[2]
    from symphonai_api.tools.search import _walk_candidates
    paths_list = []
    truncated = False
    for candidate in _walk_candidates(root, policy, breadth_first=True, include_directories=True):
        if len(paths_list) == MAX_SEARCH_FILES:
            truncated = True
            break
        paths_list.append(candidate.repo_relative + ("/" if candidate.is_directory else ""))
    paths = tuple(paths_list)
    with _CACHE_LOCK:
        _CANDIDATE_CACHE[root] = (time.monotonic(), paths, truncated)
    return paths, truncated


def repository_files(policy: PermissionPolicy, query: str, limit: int) -> tuple[list[str], bool]:
    """Return readable files and folders ranked together for an @ query."""
    root = policy.repo_root.resolve()
    paths, truncated = _candidate_paths(policy, root)
    if not query:
        return list(paths[:limit]), truncated

    needle = query.casefold()

    def rank(path: str) -> tuple[int, int, str]:
        value = path.casefold()
        basename = value.rstrip("/").rsplit("/", 1)[-1]
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


def repository_tree(policy: PermissionPolicy, directory: Path) -> tuple[list[dict[str, str]], bool]:
    """Return the readable direct children of a repository directory."""
    entries: list[dict[str, str]] = []
    truncated = False
    try:
        children = sorted(
            directory.iterdir(),
            key=lambda path: (not (path.is_dir() and not path.is_symlink()), path.name),
        )
    except OSError:
        return entries, truncated

    for path in children:
        if not policy.check_read(path).allowed:
            continue
        if path.is_dir():
            if path.is_symlink():
                continue
            kind = "dir"
        elif path.is_file():
            kind = "file"
        else:
            continue
        if len(entries) == MAX_TREE_ENTRIES:
            truncated = True
            break
        entries.append({"name": path.name, "kind": kind})
    return entries, truncated
