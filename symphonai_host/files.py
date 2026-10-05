"""Bounded file search for composer references."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from symphonai_api.permissions import PermissionPolicy


MAX_SEARCH_FILES = 20_000
_CACHE_SECONDS = 10.0
_CACHE_LOCK = threading.Lock()
_CANDIDATE_CACHE: dict[Path, tuple[float, tuple[str, ...], bool]] = {}


def _is_subsequence(query: str, value: str) -> bool:
    position = iter(value)
    return all(any(character == candidate for candidate in position) for character in query)


def _walk_candidates(policy: PermissionPolicy, root: Path) -> tuple[tuple[str, ...], bool]:
    paths: list[str] = []
    visited = 0
    truncated = False
    level = [root]
    while level:
        next_level: list[Path] = []
        for directory in sorted(level, key=lambda path: path.relative_to(root).as_posix()):
            try:
                entries = sorted(directory.iterdir(), key=lambda path: path.name)
            except OSError:
                continue
            for path in entries:
                if path.is_dir() and not path.is_symlink():
                    if policy.check_read(path).allowed:
                        next_level.append(path)
                elif path.is_file() and policy.check_read(path).allowed:
                    visited += 1
                    if visited > MAX_SEARCH_FILES:
                        truncated = True
                        break
                    paths.append(path.relative_to(root).as_posix())
            if truncated:
                break
        if truncated:
            break
        level = next_level
    return tuple(paths), truncated


def _candidate_paths(policy: PermissionPolicy, root: Path) -> tuple[tuple[str, ...], bool]:
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CANDIDATE_CACHE.get(root)
        if cached is not None and now - cached[0] < _CACHE_SECONDS:
            return cached[1], cached[2]
    paths, truncated = _walk_candidates(policy, root)
    with _CACHE_LOCK:
        _CANDIDATE_CACHE[root] = (time.monotonic(), paths, truncated)
    return paths, truncated


def repository_files(policy: PermissionPolicy, query: str, limit: int) -> tuple[list[str], bool]:
    """Return readable repository paths ranked for an @file query."""
    root = policy.repo_root.resolve()
    paths, truncated = _candidate_paths(policy, root)
    if not query:
        return list(paths[:limit]), truncated

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
