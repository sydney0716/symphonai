"""Load progressively disclosed Markdown procedures with TOML frontmatter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import tomllib

from symphonai_api.compaction import estimate_text_tokens


class SkillError(ValueError):
    """A malformed skill file, naming the file and the key."""


_FRONTMATTER_KEYS = {"name", "description", "when_to_use"}


def _raise(path: Path, key: str, detail: str) -> None:
    raise SkillError(f"{path}: {key}: {detail}")


def _read_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _raise(path, "file", f"could not be read: {exc}")
    if "\x00" in text:
        _raise(path, "file", "contains a NUL byte")
    return text


def _without_line_ending(line: str) -> str:
    return line.removesuffix("\n").removesuffix("\r")


def _parts(path: Path) -> tuple[Mapping[str, object], str]:
    text = _read_text(path)
    lines = text.splitlines(keepends=True)
    if not lines or _without_line_ending(lines[0]) != "+++":
        _raise(path, "frontmatter", "must start with a +++ delimiter")
    closing = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if _without_line_ending(line) == "+++"
        ),
        None,
    )
    if closing is None:
        _raise(path, "frontmatter", "is missing its closing +++ delimiter")
    raw_frontmatter = "".join(lines[1:closing])
    try:
        frontmatter = tomllib.loads(raw_frontmatter)
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        _raise(path, "frontmatter", f"could not be parsed: {exc}")
    if not isinstance(frontmatter, Mapping):
        _raise(path, "frontmatter", "must be a table")
    return frontmatter, "".join(lines[closing + 1 :])


def _required_text(
    path: Path,
    frontmatter: Mapping[str, object],
    key: str,
) -> str:
    if key not in frontmatter:
        _raise(path, key, "is required")
    value = frontmatter[key]
    if not isinstance(value, str):
        _raise(path, key, "must be a string")
    if not value.strip():
        _raise(path, key, "must not be blank")
    return value


def _discoverability_text(name: str, description: str, when_to_use: str) -> str:
    return (
        f"name: {name}\n"
        f"description: {description}\n"
        f"when_to_use: {when_to_use}"
    )


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    when_to_use: str
    path: Path
    always_on_tokens: int

    def discoverability_text(self) -> str:
        """The exact text ``always_on_tokens`` counted."""
        return _discoverability_text(self.name, self.description, self.when_to_use)

    def body(self) -> str:
        """Read and return the Markdown body. Raises SkillError if unreadable."""
        _, body = _parts(self.path)
        return body


def load_skill(path: Path) -> Skill:
    """Load one Markdown skill without retaining its body."""
    source = Path(path)
    frontmatter, _ = _parts(source)
    if "hooks" in frontmatter:
        _raise(source, "hooks", "skills are documents and may not declare hooks")
    unknown = frontmatter.keys() - _FRONTMATTER_KEYS
    if unknown:
        _raise(source, str(sorted(unknown)[0]), "unknown key")
    name = _required_text(source, frontmatter, "name")
    description = _required_text(source, frontmatter, "description")
    when_to_use = _required_text(source, frontmatter, "when_to_use")
    if name != source.stem:
        _raise(
            source,
            "name",
            f"declared name {name!r} must match filename stem {source.stem!r}",
        )
    always_on_tokens = estimate_text_tokens(
        _discoverability_text(name, description, when_to_use)
    )
    return Skill(name, description, when_to_use, source, always_on_tokens)


def load_skill_directory(path: Path) -> dict[str, Skill]:
    """Load every direct ``*.md`` child, keyed by its validated name."""
    directory = Path(path)
    if not directory.exists():
        return {}
    return {
        skill.name: skill
        for file in sorted(directory.glob("*.md"))
        if file.is_file()
        for skill in (load_skill(file),)
    }


def roster_text(skills: Iterable[Skill], *, separator: str = "\n\n") -> str:
    """Render a roster for a system prompt, in the given order."""
    return separator.join(skill.discoverability_text() for skill in skills)


def roster_cost(skills: Iterable[Skill]) -> int:
    """Return the discoverability cost of a skill roster."""
    return sum(skill.always_on_tokens for skill in skills)
