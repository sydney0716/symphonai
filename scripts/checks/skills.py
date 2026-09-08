"""Registered checks for progressively disclosed Markdown skills."""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import symphonai_api.skills as skills_module
from symphonai_api.compaction import estimate_text_tokens
from symphonai_api.skills import (
    Skill,
    SkillError,
    load_skill,
    load_skill_directory,
    roster_cost,
    roster_text,
)
from scripts.checks.harness import check, fail


def _document(
    name: str,
    description: str,
    when_to_use: str,
    body: str = "\n# Procedure\n",
) -> str:
    return (
        "+++\n"
        f"name = {json.dumps(name)}\n"
        f"description = {json.dumps(description)}\n"
        f"when_to_use = {json.dumps(when_to_use)}\n"
        "+++\n"
        f"{body}"
    )


def _write(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def _expect_error(path: Path, key: str, detail: str | None = None) -> str:
    try:
        load_skill(path)
    except SkillError as exc:
        message = str(exc)
        if str(path) not in message or key not in message:
            fail(f"skill error omitted {path!s} or {key!r}: {message!r}")
        if detail is not None and detail not in message:
            fail(f"skill error omitted {detail!r}: {message!r}")
        return message
    except Exception as exc:
        fail(f"skill exposed {type(exc).__name__} for {key}: {exc!r}")
    fail(f"skill accepted invalid {key}")


@check("skills.load_and_measure")
def load_and_measure() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        path = _write(
            directory,
            "release-checklist.md",
            _document(
                "release-checklist",
                "Steps to cut and verify a release build.",
                "The user asks to cut, tag, or verify a release.",
            ),
        )
        skill = load_skill(path)
        discoverability = (
            "name: release-checklist\n"
            "description: Steps to cut and verify a release build.\n"
            "when_to_use: The user asks to cut, tag, or verify a release."
        )
        expected_tokens = estimate_text_tokens(discoverability)
        if (
            skill.name != "release-checklist"
            or skill.description != "Steps to cut and verify a release build."
            or skill.when_to_use
            != "The user asks to cut, tag, or verify a release."
            or skill.path != path
            or skill.always_on_tokens != expected_tokens
        ):
            fail(f"well-formed skill loaded incorrectly: {skill!r}")

        path.write_text(
            _document(
                skill.name,
                skill.description,
                skill.when_to_use,
                "\n# Procedure\n" + ("x" * 10_000),
            ),
            encoding="utf-8",
        )
        reloaded = load_skill(path)
        if reloaded.always_on_tokens != skill.always_on_tokens:
            fail("10,000 body characters changed the always-on token cost")

        if skills_module.estimate_text_tokens is not estimate_text_tokens:
            fail("skills.py does not import the shared token estimator")
        for text in ("", "a", "abcd", "abcde", "two words", "한글 procedure"):
            if skills_module.estimate_text_tokens(text) != estimate_text_tokens(text):
                fail(f"skill token measurement diverged for {text!r}")

        try:
            skill.name = "changed"  # type: ignore[misc]
        except Exception:
            pass
        else:
            fail("Skill is not frozen")


@check("skills.measured_text_is_reachable")
def measured_text_is_reachable() -> None:
    fixtures = (
        ("a", "b", "c"),
        ("release", "Release checklist.", "Use before publishing."),
        ("한글", "배포 절차입니다.", "릴리스 요청에 사용합니다."),
        ("numbers-123", "Checks 1, 2, and 3.", "Use for numbered checks."),
        (
            "punctuation",
            "Stops: commas, colons; and dots.",
            "Use when punctuation matters!",
        ),
        (
            "longer-name",
            "A deliberately longer description.",
            "Use for a longer trigger phrase.",
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        loaded = []
        for name, description, when_to_use in fixtures:
            path = _write(
                directory,
                f"{name}.md",
                _document(name, description, when_to_use, ""),
            )
            skill = load_skill(path)
            expected = skills_module._discoverability_text(  # noqa: SLF001
                name,
                description,
                when_to_use,
            )
            if skill.discoverability_text() != expected:
                fail(f"discoverability text differed for {name!r}")
            if (
                estimate_text_tokens(skill.discoverability_text())
                != skill.always_on_tokens
            ):
                fail(f"reachable text and measured cost differed for {name!r}")
            loaded.append(skill)

        original = loaded[-1]
        original_text = original.discoverability_text()
        original.path.write_text(
            _document(
                original.name,
                original.description,
                original.when_to_use,
                "\n# Procedure\n" + ("x" * 10_000),
            ),
            encoding="utf-8",
        )
        reloaded = load_skill(original.path)
        if (
            reloaded.discoverability_text() != original_text
            or reloaded.always_on_tokens != original.always_on_tokens
        ):
            fail("10,000 body characters changed discoverability text or cost")


@check("skills.roster_cost")
def roster_cost_is_sum() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        fixtures = (
            ("alpha", "Alpha procedure.", "Use for alpha work."),
            ("bravo", "Bravo procedure with more detail.", "Use for bravo work."),
            ("charlie", "Charlie procedure.", "Use for charlie work."),
        )
        for name, description, when_to_use in fixtures:
            _write(
                directory,
                f"{name}.md",
                _document(name, description, when_to_use),
            )
        roster = load_skill_directory(directory)
        expected = sum(skill.always_on_tokens for skill in roster.values())
        if len(roster) != 3 or roster_cost(roster.values()) != expected:
            fail(f"three-skill roster cost differed: {roster!r}")


@check("skills.roster_renders_and_sums")
def roster_renders_and_sums() -> None:
    fixtures = (
        ("alpha", "Alpha procedure.", "Use for alpha work."),
        ("bravo", "Bravo procedure with more detail.", "Use for bravo work."),
        ("charlie", "Charlie procedure.", "Use for charlie work."),
    )
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        roster = {}
        for name, description, when_to_use in fixtures:
            roster[name] = load_skill(
                _write(
                    directory,
                    f"{name}.md",
                    _document(name, description, when_to_use),
                )
            )

        forward = (roster["charlie"], roster["alpha"], roster["bravo"])
        reverse = tuple(reversed(forward))
        expected_forward = "\n\n".join(
            skill.discoverability_text() for skill in forward
        )
        expected_reverse = "\n\n".join(
            skill.discoverability_text() for skill in reverse
        )
        if roster_text(forward) != expected_forward:
            fail("default roster rendering did not preserve forward order")
        if roster_text(reverse) != expected_reverse:
            fail("default roster rendering did not preserve reverse order")

        custom_separator = "\n--- skill ---\n"
        if roster_text(forward, separator=custom_separator) != custom_separator.join(
            skill.discoverability_text() for skill in forward
        ):
            fail("custom roster separator was not used exactly")

        zero_cost = Skill("zero", "Zero.", "Use zero.", directory / "zero.md", 0)
        with_zero = (forward[0], zero_cost, forward[1])
        if roster_text(with_zero) != "\n\n".join(
            skill.discoverability_text() for skill in with_zero
        ):
            fail("roster rendering dropped a zero-cost skill")

        if roster_text(()) != "" or roster_cost(()) != 0:
            fail("empty roster text or cost was not empty or zero")
        summed_cost = roster_cost(forward)
        rendered_cost = estimate_text_tokens(roster_text(forward))
        if summed_cost != 62 or rendered_cost != 63:
            fail(
                "known roster estimates differed: "
                f"per-skill={summed_cost}, rendered={rendered_cost}"
            )
        if summed_cost == rendered_cost:
            fail("per-skill and rendered roster estimates unexpectedly matched")


@check("skills.body_is_live")
def body_is_live() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        path = _write(
            directory,
            "live.md",
            _document("live", "A live body.", "Use when testing edits.", "\n# First\n"),
        )
        skill = load_skill(path)
        if skill.body() != "\n# First\n":
            fail(f"frontmatter delimiters leaked into the body: {skill.body()!r}")
        path.write_text(
            _document(
                "live",
                "A live body.",
                "Use when testing edits.",
                "\n# Second\nEdited between calls.\n",
            ),
            encoding="utf-8",
        )
        if skill.body() != "\n# Second\nEdited between calls.\n":
            fail("body() cached its first disk read")
        path.unlink()
        try:
            skill.body()
        except SkillError as exc:
            if str(path) not in str(exc):
                fail(f"unreadable body error omitted its path: {exc!r}")
        except Exception as exc:
            fail(f"unreadable body exposed {type(exc).__name__}: {exc!r}")
        else:
            fail("body() accepted a missing skill file")


@check("skills.frontmatter_rejections")
def frontmatter_rejections() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        mismatch = _write(
            directory,
            "filename.md",
            _document("declared", "Description.", "Use now."),
        )
        message = _expect_error(mismatch, "name")
        if "declared" not in message or "filename" not in message:
            fail(f"name mismatch did not name both values: {message!r}")

        cases = (
            (
                "missing-description.md",
                '+++\nname = "missing-description"\nwhen_to_use = "Use now."\n+++\n',
                "description",
                None,
            ),
            (
                "blank-description.md",
                _document("blank-description", "  \t", "Use now."),
                "description",
                None,
            ),
            (
                "missing-when.md",
                '+++\nname = "missing-when"\ndescription = "Description."\n+++\n',
                "when_to_use",
                None,
            ),
            (
                "blank-when.md",
                _document("blank-when", "Description.", "\n\t"),
                "when_to_use",
                None,
            ),
            (
                "unknown.md",
                _document("unknown", "Description.", "Use now.").replace(
                    "+++\n\n# Procedure",
                    'extra = true\n+++\n\n# Procedure',
                ),
                "extra",
                None,
            ),
            (
                "hooks.md",
                _document("hooks", "Description.", "Use now.").replace(
                    "+++\n\n# Procedure",
                    'hooks = ["unsafe"]\n+++\n\n# Procedure',
                ),
                "hooks",
                "skills are documents",
            ),
        )
        for filename, content, key, detail in cases:
            _expect_error(_write(directory, filename, content), key, detail)


@check("skills.malformed_files")
def malformed_files() -> None:
    cases = (
        ("no-frontmatter.md", b"# Procedure\n"),
        (
            "missing-close.md",
            b'+++\nname = "missing-close"\ndescription = "d"\nwhen_to_use = "w"\n',
        ),
        ("empty-frontmatter.md", b"+++\n+++\n"),
        ("non-table.md", b'+++\n["not", "a", "table"]\n+++\n'),
        ("nul.md", b'+++\nname = "nul\x00"\n+++\n'),
        ("invalid-utf8.md", b"+++\nname = \xff\n+++\n"),
        (
            "blank-name.md",
            b'+++\nname = " "\ndescription = "d"\nwhen_to_use = "w"\n+++\n',
        ),
        (
            "blank-description.md",
            b'+++\nname = "blank-description"\ndescription = ""\nwhen_to_use = "w"\n+++\n',
        ),
        (
            "blank-when.md",
            b'+++\nname = "blank-when"\ndescription = "d"\nwhen_to_use = ""\n+++\n',
        ),
        (
            "duplicate.md",
            b'+++\nname = "duplicate"\nname = "again"\ndescription = "d"\nwhen_to_use = "w"\n+++\n',
        ),
        ("unterminated.md", b'+++\nname = "unterminated\n+++\n'),
        (
            "description-type.md",
            b'+++\nname = "description-type"\ndescription = 1\nwhen_to_use = "w"\n+++\n',
        ),
        (
            "when-type.md",
            b'+++\nname = "when-type"\ndescription = "d"\nwhen_to_use = []\n+++\n',
        ),
        ("wrong-delimiter.md", b"---\nname = 'wrong-delimiter'\n---\n"),
        (
            "truncated-delimiter.md",
            b'+++\nname = "truncated-delimiter"\n+++ extra\n',
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for filename, content in cases:
            path = directory / filename
            path.write_bytes(content)
            try:
                load_skill(path)
            except SkillError as exc:
                if str(path) not in str(exc):
                    fail(f"malformed skill error omitted its path: {exc!r}")
            except Exception as exc:
                fail(
                    f"{filename} exposed {type(exc).__name__} instead of "
                    f"SkillError: {exc!r}"
                )
            else:
                fail(f"malformed skill was accepted: {filename}")


@check("skills.directory_roster")
def directory_roster() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        missing = root / "missing"
        if load_skill_directory(missing) != {}:
            fail("missing skill directory was not empty")

        directory = root / "skills"
        directory.mkdir()
        _write(directory, "alpha.md", _document("alpha", "Alpha.", "Use alpha."))
        _write(directory, "beta.md", _document("beta", "Beta.", "Use beta."))
        _write(directory, "ignored.txt", "not a skill")
        _write(directory, "ignored.MD", _document("ignored", "Ignored.", "Never."))
        nested = directory / "nested"
        nested.mkdir()
        _write(nested, "nested.md", _document("nested", "Nested.", "Never."))
        roster = load_skill_directory(directory)
        if list(roster) != ["alpha", "beta"]:
            fail(f"directory filtering or ordering differed: {list(roster)!r}")

        malformed = _write(directory, "broken.md", "# no frontmatter\n")
        try:
            load_skill_directory(directory)
        except SkillError as exc:
            if str(malformed) not in str(exc):
                fail(f"malformed roster error omitted its file: {exc!r}")
        except Exception as exc:
            fail(f"roster exposed {type(exc).__name__}: {exc!r}")
        else:
            fail("malformed skill was silently skipped")


@check("skills.no_runtime_imports")
def no_runtime_imports() -> None:
    path = Path(__file__).resolve().parents[2] / "symphonai_api/skills.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {
        "agent_loop",
        "leader",
        "runner",
        "agent_run",
        "agent_spec",
        "agent_file",
        "child_context",
        "hooks",
        "provider_catalog",
        "providers",
    }
    imported = {
        node.module.split(".")[1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("symphonai_api.")
    }
    if imported & forbidden:
        fail(f"skills.py imports forbidden modules: {sorted(imported & forbidden)!r}")
    shared_estimator = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "symphonai_api.compaction"
        and any(alias.name == "estimate_text_tokens" for alias in node.names)
        for node in ast.walk(tree)
    )
    if not shared_estimator:
        fail("skills.py does not import compaction.estimate_text_tokens")
