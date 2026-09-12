"""Bounded, permission-gated repository metadata survey."""

from __future__ import annotations

import os
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

from symphonai_api.permissions import PermissionPolicy


DOCUMENT_READ_LIMIT = 4 * 1024
_ENTRY_POINT_NAMES = frozenset(
    {
        "cargo.toml",
        "deno.json",
        "deno.jsonc",
        "go.mod",
        "makefile",
        "manifest.json",
        "package.json",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
    }
)
_TEST_DIRECTORY_NAMES = frozenset({"__tests__", "test", "tests"})


@dataclass(frozen=True)
class Survey:
    root: Path
    languages: tuple[tuple[str, int], ...]
    by_directory: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    entry_points: tuple[Path, ...]
    docs: tuple[Path, ...]
    tests: tuple[Path, ...]
    tree_summary: str
    truncated_directories: tuple[str, ...]
    stopped: bool
    file_count: int


def _allowed(policy: PermissionPolicy, path: Path) -> bool:
    try:
        return policy.check_read(path).allowed
    except Exception:  # Permission checks fail closed at this boundary.
        return False


def _is_document(relative: Path) -> bool:
    return (
        relative.name.casefold().startswith("readme")
        or bool(relative.parts) and relative.parts[0].casefold() == "docs"
    )


def _is_entry_point(path: Path) -> bool:
    name = path.name.casefold()
    return (
        name in _ENTRY_POINT_NAMES
        or name == "__main__.py"
        or path.stem.casefold() == "main"
    )


def _sample_document(policy: PermissionPolicy, path: Path) -> None:
    if not _allowed(policy, path):
        return
    try:
        with path.open("rb") as document:
            document.read(DOCUMENT_READ_LIMIT)
    except OSError:
        return


def _entries(policy: PermissionPolicy, directory: Path):  # noqa: ANN201
    if not _allowed(policy, directory):
        return ()
    try:
        with os.scandir(directory) as scanner:
            entries = sorted(
                scanner,
                key=lambda entry: (entry.name.casefold(), entry.name),
            )
    except OSError:
        return ()
    return tuple(
        entry for entry in entries if _allowed(policy, Path(entry.path))
    )


def _render_tree(
    root: Path,
    children: dict[Path, list[tuple[str, bool]]],
    *,
    stopped: bool,
    file_count: int,
) -> str:
    lines = ["."]

    def ordered(items: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        return sorted(items, key=lambda item: (not item[1], item[0].casefold(), item[0]))

    top_level = ordered(children.get(root, []))
    if len(top_level) > 20:
        lines.append(f"  … {len(top_level)} entries")
    else:
        for name, is_directory in top_level:
            lines.append(f"  {name}{'/' if is_directory else ''}")
            if not is_directory:
                continue
            nested = ordered(children.get(root / name, []))
            if len(nested) > 20:
                lines.append(f"    … {len(nested)} entries")
            else:
                lines.extend(
                    f"    {child_name}{'/' if child_is_directory else ''}"
                    for child_name, child_is_directory in nested
                )
    if stopped:
        lines.append(f"… stopped after {file_count} files")
    return "\n".join(lines)


def survey_repository(
    *,
    policy: PermissionPolicy,
    max_files: int = 5000,
) -> Survey:
    """Describe a repository without writing it or bypassing its read policy."""
    if not isinstance(max_files, int) or isinstance(max_files, bool) or max_files < 1:
        raise ValueError("max_files must be a positive integer")

    root = policy.repo_root
    languages: Counter[str] = Counter()
    entry_points: set[Path] = set()
    docs: set[Path] = set()
    tests: set[Path] = set()
    children: dict[Path, list[tuple[str, bool]]] = {}
    directory_languages: dict[str, Counter[str]] = {}
    directory_file_counts: dict[str, int] = {}
    file_count = 0

    def classify(entry):  # noqa: ANN001, ANN202
        try:
            return (
                entry.is_dir(follow_symlinks=False),
                entry.is_file(follow_symlinks=False),
            )
        except OSError:
            return False, False

    def count_file(path: Path, bucket: str | None) -> None:
        nonlocal file_count
        file_count += 1
        if bucket is not None:
            directory_file_counts[bucket] += 1
        extension = path.suffix.casefold()
        if extension:
            languages[extension] += 1
            if bucket is not None:
                directory_languages[bucket][extension] += 1
        relative = path.relative_to(root)
        if _is_entry_point(path):
            entry_points.add(path)
        if _is_document(relative):
            docs.add(path)
            _sample_document(policy, path)

    root_directories: list[Path] = []
    root_files: list[Path] = []
    for entry in _entries(policy, root):
        path = Path(entry.path)
        is_directory, is_file = classify(entry)
        if not is_directory and not is_file:
            continue
        children.setdefault(root, []).append((path.name, is_directory))
        if is_directory:
            root_directories.append(path)
            if path.name.casefold() in _TEST_DIRECTORY_NAMES:
                tests.add(path)
        else:
            root_files.append(path)

    for path in root_files:
        if file_count >= max_files:
            break
        count_file(path, None)

    top_level_count = len(root_directories)
    directory_budget = (
        max(200, max_files // top_level_count)
        if top_level_count > 0
        else max_files
    )
    for directory in root_directories:
        directory_languages[directory.name] = Counter()
        directory_file_counts[directory.name] = 0

    def directory_files(start: Path):  # noqa: ANN202
        directories = deque([start])
        while directories:
            directory = directories.popleft()
            for entry in _entries(policy, directory):
                path = Path(entry.path)
                is_directory, is_file = classify(entry)
                if not is_directory and not is_file:
                    continue
                relative = path.relative_to(root)
                if len(relative.parts) <= 2:
                    children.setdefault(path.parent, []).append(
                        (path.name, is_directory)
                    )
                if is_directory:
                    if path.name.casefold() in _TEST_DIRECTORY_NAMES:
                        tests.add(path)
                    directories.append(path)
                else:
                    yield path

    walkers = {
        directory.name: iter(directory_files(directory))
        for directory in root_directories
    }
    exhausted: set[str] = set()
    pending: dict[str, Path] = {}

    def next_file(bucket: str) -> Path | None:
        if bucket in pending:
            return pending.pop(bucket)
        try:
            return next(walkers[bucket])
        except StopIteration:
            exhausted.add(bucket)
            return None

    bucket_names = sorted(walkers, key=lambda name: (name.casefold(), name))
    for bucket in bucket_names:
        while (
            file_count < max_files
            and directory_file_counts[bucket] < directory_budget
        ):
            path = next_file(bucket)
            if path is None:
                break
            count_file(path, bucket)

    refill = [name for name in bucket_names if name not in exhausted]
    while file_count < max_files and refill:
        next_refill: list[str] = []
        for bucket in refill:
            path = next_file(bucket)
            if path is None:
                continue
            count_file(path, bucket)
            next_refill.append(bucket)
            if file_count >= max_files:
                break
        refill = next_refill

    truncated: list[str] = []
    for bucket in bucket_names:
        if bucket in exhausted:
            continue
        path = next_file(bucket)
        if path is not None:
            pending[bucket] = path
            truncated.append(bucket)

    stopped = file_count == max_files

    language_counts = tuple(
        sorted(languages.items(), key=lambda item: (-item[1], item[0]))
    )
    by_directory = tuple(
        (
            name,
            tuple(
                sorted(
                    directory_languages[name].items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        )
        for name in sorted(
            directory_languages,
            key=lambda name: (-directory_file_counts[name], name.casefold(), name),
        )
    )
    ordered_paths = lambda paths: tuple(  # noqa: E731
        sorted(paths, key=lambda path: path.relative_to(root).as_posix())
    )
    return Survey(
        root=root,
        languages=language_counts,
        by_directory=by_directory,
        entry_points=ordered_paths(entry_points),
        docs=ordered_paths(docs),
        tests=ordered_paths(tests),
        tree_summary=_render_tree(
            root,
            children,
            stopped=stopped,
            file_count=file_count,
        ),
        truncated_directories=tuple(truncated),
        stopped=stopped,
        file_count=file_count,
    )
