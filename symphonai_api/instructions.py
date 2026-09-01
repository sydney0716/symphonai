"""Discover, process, and render repository instruction hierarchies."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.read_ledger import ReadLedger


# These are reference values inherited from the documented implementation;
# SymphonAI has not measured or tuned them independently.
MAX_INSTRUCTION_FILE_CHARS = 40_000
MAX_INCLUDE_DEPTH = 5
INSTRUCTION_FILENAMES: tuple[str, ...] = ("CLAUDE.md", "AGENTS.md")


class InstructionScope(str, Enum):
    USER = "user"
    PROJECT = "project"
    DIRECTORY = "directory"
    AGENT = "agent"


@dataclass(frozen=True)
class LoadedInstruction:
    path: Path | None
    scope: InstructionScope
    depth: int
    parent: Path | None
    text: str
    characters: int


@dataclass(frozen=True)
class InstructionSet:
    entries: tuple[LoadedInstruction, ...]
    warnings: tuple[str, ...]
    _repo_root: Path | None = field(default=None, repr=False, compare=False)

    def display_path(self, path: Path) -> str:
        if self._repo_root is not None:
            try:
                return str(path.relative_to(self._repo_root))
            except ValueError:
                pass
        return str(path)

    def render_entry(self, entry: LoadedInstruction) -> str:
        """Render one entry's block exactly as `render()` includes it.

        Returns "" for an entry that contributes nothing, so callers that
        attribute cost per entry skip precisely what `render()` skips.
        """

        if not entry.text.strip():
            return ""
        provenance = entry.scope.value
        if entry.path is not None:
            path = self.display_path(entry.path)
            if entry.parent is None:
                provenance = f"{provenance} {path}"
            else:
                parent = self.display_path(entry.parent)
                provenance = f"{provenance} {parent} -> {path}"
        return f"# instructions: {provenance}\n{entry.text}"

    def render(self) -> str:
        rendered = (self.render_entry(entry) for entry in self.entries)
        return "\n\n".join(block for block in rendered if block)


def _strip_frontmatter(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return text
    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") == "---":
            return "".join(lines[index + 1 :])
    return text


def _process_text(raw: str) -> tuple[str, tuple[str, ...]]:
    without_frontmatter = _strip_frontmatter(raw)
    without_comments = re.sub(r"<!--.*?(?:-->|$)", "", without_frontmatter, flags=re.DOTALL)
    content_lines: list[str] = []
    includes: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    for line in without_comments.splitlines(keepends=True):
        stripped = line.strip()
        if fence_character is not None:
            content_lines.append(line)
            if (
                stripped
                and set(stripped) == {fence_character}
                and len(stripped) >= fence_length
            ):
                fence_character = None
                fence_length = 0
            continue

        if stripped.startswith(("```", "~~~")):
            fence_character = stripped[0]
            fence_length = len(stripped) - len(stripped.lstrip(fence_character))
            content_lines.append(line)
            continue

        target = line.rstrip("\r\n")[1:] if line.startswith("@") else ""
        if target and not any(character.isspace() for character in target):
            includes.append(target)
        else:
            content_lines.append(line)
    return "".join(content_lines).strip("\r\n"), tuple(includes)


def load_instructions(
    policy: PermissionPolicy,
    *,
    working_dir: Path | None = None,
    user_home: Path | None = None,
    agent_instructions: str | None = None,
    ledger: ReadLedger | None = None,
) -> InstructionSet:
    """Load known instruction scopes with reference depth and size bounds."""
    repo_root = policy.repo_root
    entries: list[LoadedInstruction] = []
    warnings: list[str] = []
    visited: set[Path] = set()

    def load_file(
        path: Path,
        *,
        scope: InstructionScope,
        depth: int,
        parent: Path | None,
        included: bool,
        policy_gated: bool,
        include_root: Path | None,
    ) -> None:
        resolved = path.resolve()
        if included and include_root is not None:
            if resolved != include_root and include_root not in resolved.parents:
                warnings.append(
                    f"instruction file {resolved} was denied: "
                    f"path is outside user_home: {resolved!r}"
                )
                return
            # Reuse the policy's pattern list so it cannot drift, but match
            # relative to user_home because the policy's base is repo_root.
            relative_parts = resolved.relative_to(include_root).parts
            forbidden = next(
                (
                    pattern
                    for pattern in policy.forbidden_patterns
                    if any(
                        fnmatch.fnmatch(part, pattern.rstrip("/"))
                        for part in relative_parts
                    )
                ),
                None,
            )
            if forbidden is not None:
                warnings.append(
                    f"instruction file {resolved} was denied: path matches forbidden "
                    f"pattern {forbidden!r}: {resolved!r}"
                )
                return
        elif policy_gated:
            decision = policy.check_read(resolved)
            if not decision.allowed:
                warnings.append(f"instruction file {resolved} was denied: {decision.reason}")
                return
        if resolved in visited:
            if included:
                warnings.append(f"instruction include cycle or duplicate skipped: {resolved}")
            return
        if included and depth > MAX_INCLUDE_DEPTH:
            warnings.append(
                f"instruction include depth {depth} exceeds {MAX_INCLUDE_DEPTH}: {resolved}"
            )
            return
        try:
            raw = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            warnings.append(f"instruction file could not be read: {resolved}: {exc}")
            return

        visited.add(resolved)
        if len(raw) > MAX_INSTRUCTION_FILE_CHARS:
            warnings.append(
                f"instruction file {resolved} has {len(raw)} characters, over the "
                f"{MAX_INSTRUCTION_FILE_CHARS} character reference limit; loaded in full"
            )
        text, includes = _process_text(raw)
        entries.append(
            LoadedInstruction(
                path=resolved,
                scope=scope,
                depth=depth,
                parent=parent,
                text=text,
                characters=len(raw),
            )
        )
        if ledger is not None:
            try:
                ledger.record(resolved, full=False, content=None, partial_view=True)
            except OSError as exc:
                warnings.append(f"instruction ledger record failed: {resolved}: {exc}")

        for include in includes:
            load_file(
                resolved.parent / include,
                scope=scope,
                depth=depth + 1,
                parent=resolved,
                included=True,
                policy_gated=True,
                include_root=include_root,
            )

    selected_user_home = (
        Path(user_home) if user_home is not None else Path.home() / ".symphonai"
    ).resolve()
    user_file = selected_user_home / "CLAUDE.md"
    if user_file.is_file():
        load_file(
            user_file,
            scope=InstructionScope.USER,
            depth=0,
            parent=None,
            included=False,
            policy_gated=False,
            include_root=selected_user_home,
        )

    for name in INSTRUCTION_FILENAMES:
        project_file = repo_root / name
        if project_file.is_file():
            load_file(
                project_file,
                scope=InstructionScope.PROJECT,
                depth=0,
                parent=None,
                included=False,
                policy_gated=True,
                include_root=None,
            )

    if working_dir is not None:
        resolved_working_dir = Path(working_dir).resolve()
        try:
            relative_working_dir = resolved_working_dir.relative_to(repo_root)
        except ValueError:
            warnings.append(
                f"working directory is outside repo_root; directory instructions skipped: "
                f"{resolved_working_dir}"
            )
        else:
            current = repo_root
            for part in relative_working_dir.parts:
                current /= part
                for name in INSTRUCTION_FILENAMES:
                    directory_file = current / name
                    if directory_file.is_file():
                        load_file(
                            directory_file,
                            scope=InstructionScope.DIRECTORY,
                            depth=0,
                            parent=None,
                            included=False,
                            policy_gated=True,
                            include_root=None,
                        )

    if agent_instructions is not None:
        entries.append(
            LoadedInstruction(
                path=None,
                scope=InstructionScope.AGENT,
                depth=0,
                parent=None,
                text=agent_instructions,
                characters=len(agent_instructions),
            )
        )

    return InstructionSet(tuple(entries), tuple(warnings), repo_root)
