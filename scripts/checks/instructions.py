"""Checks for instruction discovery, processing, provenance, and wiring."""

from __future__ import annotations

import os
import unittest.mock as mock

import orchestra_api.instructions as instructions
from orchestra_api.instructions import InstructionScope, load_instructions
from orchestra_api.models import Message, ModelResponse, Role, ToolCall
from orchestra_api.providers.fake import FakeModelProvider
from orchestra_api.runner import run_task, standard_tool_registry
from orchestra_api.tools.read_ledger import ReadLedger
from scripts.checks.harness import check, fail
from scripts.checks.workspace import workspace


PARTIAL_VIEW_ERROR = (
    "only a processed view of this file has been read; "
    "read it with read_file before editing it"
)


@check("instructions.scope_order")
def check_scope_order() -> None:
    with workspace() as ws:
        user_home = ws.root / "user-config"
        user_home.mkdir()
        (user_home / "CLAUDE.md").write_text("user rule")
        (ws.root / "CLAUDE.md").write_text("project claude")
        (ws.root / "AGENTS.md").write_text("project agents")
        nested = ws.root / "src"
        deep = nested / "package"
        deep.mkdir(parents=True)
        (nested / "CLAUDE.md").write_text("directory claude")
        (deep / "AGENTS.md").write_text("directory agents")

        loaded = load_instructions(
            ws.policy,
            working_dir=deep,
            user_home=user_home,
            agent_instructions="agent rule",
        )
        expected = [
            (InstructionScope.USER, user_home / "CLAUDE.md"),
            (InstructionScope.PROJECT, ws.root / "CLAUDE.md"),
            (InstructionScope.PROJECT, ws.root / "AGENTS.md"),
            (InstructionScope.DIRECTORY, nested / "CLAUDE.md"),
            (InstructionScope.DIRECTORY, deep / "AGENTS.md"),
            (InstructionScope.AGENT, None),
        ]
        actual = [(entry.scope, entry.path) for entry in loaded.entries]
        if actual != expected:
            fail(f"instruction scope order changed: {actual!r}")
        if any(entry.depth != 0 or entry.parent is not None for entry in loaded.entries):
            fail(f"discovered instruction provenance was wrong: {loaded.entries!r}")
        if loaded.entries[-1].characters != len("agent rule"):
            fail(f"agent instruction metadata was wrong: {loaded.entries[-1]!r}")

        root_only = load_instructions(
            ws.policy,
            working_dir=ws.root,
            user_home=ws.root / "missing-user-home",
        )
        if [entry.scope for entry in root_only.entries] != [
            InstructionScope.PROJECT,
            InstructionScope.PROJECT,
        ] or root_only.warnings:
            fail(f"root or missing-user discovery changed: {root_only!r}")


@check("instructions.outside_working_dir")
def check_outside_working_dir() -> None:
    with workspace() as ws:
        loaded = load_instructions(
            ws.policy,
            working_dir=ws.outside,
            user_home=ws.root / "missing-user-home",
        )
        if loaded.entries or len(loaded.warnings) != 1:
            fail(f"outside working directory did not warn and skip: {loaded!r}")
        warning = loaded.warnings[0]
        if str(ws.outside) not in warning or "outside repo_root" not in warning:
            fail(f"outside working-directory warning lacked provenance: {warning!r}")


@check("instructions.include_provenance")
def check_include_provenance() -> None:
    with workspace() as ws:
        docs = ws.root / "docs"
        docs.mkdir()
        project = ws.root / "CLAUDE.md"
        agents = ws.root / "AGENTS.md"
        included = docs / "style.md"
        project.write_text("parent before\n@docs/style.md\nparent after\n")
        agents.write_text("@docs/style.md\nagents rule\n")
        included.write_text("style rule\n")

        loaded = load_instructions(ws.policy, user_home=ws.root / "missing-user-home")
        if [entry.path for entry in loaded.entries] != [project, included, agents]:
            fail(f"included file order or deduplication changed: {loaded.entries!r}")
        child = loaded.entries[1]
        if child.parent != project or child.depth != 1 or child.scope != InstructionScope.PROJECT:
            fail(f"include provenance was wrong: {child!r}")
        if "@docs/style.md" in loaded.entries[0].text:
            fail(f"include directive remained in parent text: {loaded.entries[0]!r}")
        if len(loaded.warnings) != 1 or str(included) not in loaded.warnings[0]:
            fail(f"duplicate include did not warn once: {loaded.warnings!r}")
        rendered = loaded.render()
        if (
            "# instructions: project CLAUDE.md\nparent before\nparent after" not in rendered
            or "# instructions: project CLAUDE.md -> docs/style.md\nstyle rule" not in rendered
        ):
            fail(f"rendered provenance was wrong: {rendered!r}")


@check("instructions.depth_cap")
def check_depth_cap() -> None:
    with workspace() as ws:
        paths = [ws.root / name for name in ("CLAUDE.md", "one.md", "two.md", "three.md")]
        paths[0].write_text("@one.md\nroot")
        paths[1].write_text("@two.md\none")
        paths[2].write_text("@three.md\ntwo")
        paths[3].write_text("three")
        with mock.patch.object(instructions, "MAX_INCLUDE_DEPTH", 2):
            loaded = load_instructions(ws.policy, user_home=ws.root / "missing-user-home")
        if [entry.path for entry in loaded.entries] != paths[:3]:
            fail(f"include depth cap loaded the wrong files: {loaded.entries!r}")
        if [entry.depth for entry in loaded.entries] != [0, 1, 2]:
            fail(f"include depths were wrong: {loaded.entries!r}")
        if (
            len(loaded.warnings) != 1
            or str(paths[3]) not in loaded.warnings[0]
            or "3" not in loaded.warnings[0]
        ):
            fail(f"depth-cap warning was wrong: {loaded.warnings!r}")


@check("instructions.cycle_detection")
def check_cycle_detection() -> None:
    with workspace() as ws:
        project = ws.root / "CLAUDE.md"
        child = ws.root / "child.md"
        project.write_text("@child.md\nroot")
        child.write_text("@CLAUDE.md\nchild")
        loaded = load_instructions(ws.policy, user_home=ws.root / "missing-user-home")
        if [entry.path for entry in loaded.entries] != [project, child]:
            fail(f"cycle loaded a file more than once: {loaded.entries!r}")
        if len(loaded.warnings) != 1 or str(project) not in loaded.warnings[0]:
            fail(f"cycle warning was wrong: {loaded.warnings!r}")


@check("instructions.forbidden_include")
def check_forbidden_include() -> None:
    with workspace() as ws:
        (ws.root / "CLAUDE.md").write_text("safe\n@.env\nstill safe\n")
        loaded = load_instructions(ws.policy, user_home=ws.root / "missing-user-home")
        secret = "SECRET=do-not-read-me"
        if secret in loaded.render() or any(secret in entry.text for entry in loaded.entries):
            fail("forbidden .env value leaked into loaded instructions")
        if (
            len(loaded.warnings) != 1
            or ".env" not in loaded.warnings[0]
            or "forbidden" not in loaded.warnings[0]
        ):
            fail(f"forbidden include warning was wrong: {loaded.warnings!r}")


@check("instructions.processing_and_size_warning")
def check_processing_and_size_warning() -> None:
    with workspace() as ws:
        raw = "---\ntitle: hidden\n---\nVisible\n<!-- hidden\ncomment -->\nTail\n"
        path = ws.root / "CLAUDE.md"
        path.write_text(raw)
        with mock.patch.object(instructions, "MAX_INSTRUCTION_FILE_CHARS", 10):
            loaded = load_instructions(ws.policy, user_home=ws.root / "missing-user-home")
        entry = loaded.entries[0]
        if entry.text != "Visible\n\nTail" or entry.characters != len(raw):
            fail(f"instruction processing or raw count changed: {entry!r}")
        if len(entry.text) <= 10:
            fail(f"oversized instruction was truncated: {entry!r}")
        if (
            len(loaded.warnings) != 1
            or str(path) not in loaded.warnings[0]
            or str(len(raw)) not in loaded.warnings[0]
        ):
            fail(f"oversize warning was wrong: {loaded.warnings!r}")


@check("instructions.ledger_partial_view")
def check_ledger_partial_view() -> None:
    with workspace() as ws:
        path = ws.root / "CLAUDE.md"
        path.write_text("project rule")
        ledger = ws.tools["read_file"]._ledger
        loaded = load_instructions(
            ws.policy,
            user_home=ws.root / "missing-user-home",
            agent_instructions="agent rule",
            ledger=ledger,
        )
        if ledger.check(path.resolve()) != PARTIAL_VIEW_ERROR:
            fail("instruction file did not receive the literal partial-view refusal")
        if len(ledger._records) != 1 or loaded.entries[-1].path is not None:
            fail(f"agent instructions created a ledger record: {ledger._records!r}")

        with mock.patch.object(
            ReadLedger,
            "record",
            autospec=True,
            side_effect=AssertionError("ledger work ran with ledger=None"),
        ):
            without_ledger = load_instructions(
                ws.policy,
                user_home=ws.root / "missing-user-home",
                ledger=None,
            )
        if not without_ledger.entries:
            fail("ledger=None suppressed instruction loading")


@check("instructions.run_task_wiring")
def check_run_task_wiring() -> None:
    with workspace() as ws:
        path = ws.root / "CLAUDE.md"
        path.write_text("project rule")
        baseline = run_task(
            FakeModelProvider(),
            ws.policy,
            "user prompt",
            system_prompt="system prompt",
        )
        baseline_messages = [(message.role, message.text) for message in baseline.messages]
        expected_baseline = [
            (Role.SYSTEM, "system prompt"),
            (Role.USER, "user prompt"),
            (Role.ASSISTANT, "(no scripted response configured)"),
        ]
        if baseline_messages != expected_baseline:
            fail(f"default run_task messages changed from the HEAD baseline: {baseline_messages!r}")

        first_registry = standard_tool_registry()
        second_registry = standard_tool_registry()
        if first_registry["read_file"]._ledger is second_registry["read_file"]._ledger:
            fail("default standard registries stopped constructing isolated ledgers")

        provider = FakeModelProvider(
            [
                ModelResponse(
                    Message(
                        Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="edit-instruction",
                                name="edit_file",
                                arguments={
                                    "path": path.name,
                                    "old_string": "project rule",
                                    "new_string": "changed",
                                },
                            )
                        ],
                    )
                ),
                ModelResponse(Message(Role.ASSISTANT, "done")),
            ]
        )
        wired = run_task(provider, ws.policy, "edit it", include_instructions=True)
        if (
            wired.messages[0].role != Role.SYSTEM
            or "project CLAUDE.md" not in wired.messages[0].text
        ):
            fail(f"automatic instructions were not prepended: {wired.messages!r}")
        tool_results = [
            message.tool_result for message in wired.messages if message.role == Role.TOOL
        ]
        if len(tool_results) != 1 or tool_results[0].error != PARTIAL_VIEW_ERROR:
            fail(f"run_task did not share the instruction ledger with its tools: {tool_results!r}")


@check("instructions.directive_boundaries")
def check_directive_boundaries() -> None:
    with workspace() as ws:
        raw = (
            "```python\n"
            "  @dataclass(frozen=True)\n"
            "class ReadRecord:\n"
            "```\n"
            "    @decorator\n"
            "@sydney0716 please review this\n"
            "~~~python\n"
            "@tilde_decorator\n"
            "~~~\n"
        )
        path = ws.root / "CLAUDE.md"
        path.write_text(raw)
        loaded = load_instructions(ws.policy, user_home=ws.root / "missing-user-home")
        entry = loaded.entries[0]
        required_lines = (
            "  @dataclass(frozen=True)",
            "    @decorator",
            "@sydney0716 please review this",
            "@tilde_decorator",
        )
        if any(line not in entry.text for line in required_lines):
            fail(f"content-like @ lines were removed or reindented: {entry.text!r}")
        if loaded.warnings:
            fail(f"content-like @ lines produced include warnings: {loaded.warnings!r}")
        run_task_docstring = run_task.__doc__ or ""
        if (
            "single task to completion" not in run_task_docstring
            or "omit directory scope" not in run_task_docstring
        ):
            fail(f"run_task docstring lost its purpose or caveat: {run_task_docstring!r}")


@check("instructions.fence_run_length")
def check_fence_run_length() -> None:
    with workspace() as ws:
        raw = "````markdown\n```\n@inside-four-ticks\n```\n````\n"
        (ws.root / "CLAUDE.md").write_text(raw)
        loaded = load_instructions(ws.policy, user_home=ws.root / "missing-user-home")
        if "@inside-four-ticks" not in loaded.entries[0].text:
            fail(f"shorter nested fence exposed an @ line: {loaded.entries[0].text!r}")
        if loaded.warnings:
            fail(f"four-backtick content produced include warnings: {loaded.warnings!r}")


@check("instructions.user_scope_include")
def check_user_scope_include() -> None:
    with workspace() as ws:
        user_home = ws.outside / "user-home"
        user_home.mkdir()
        user_file = user_home / "CLAUDE.md"
        included = user_home / "style.md"
        user_file.write_text("user rule\n@style.md\n")
        included.write_text("user style\n")
        ledger = ws.tools["read_file"]._ledger
        loaded = load_instructions(ws.policy, user_home=user_home, ledger=ledger)
        if [entry.path for entry in loaded.entries] != [user_file, included]:
            fail(f"user-scope neighbour include did not load: {loaded!r}")
        child = loaded.entries[1]
        if (
            child.scope != InstructionScope.USER
            or child.depth != 1
            or child.parent != user_file
        ):
            fail(f"user-scope include provenance was wrong: {child!r}")
        if ledger.check(included.resolve()) != PARTIAL_VIEW_ERROR:
            fail("included user instruction was not recorded as a partial view")
        if loaded.warnings:
            fail(f"allowed user-scope include warned: {loaded.warnings!r}")


@check("instructions.user_scope_guards")
def check_user_scope_guards() -> None:
    with workspace() as ws:
        user_home = ws.outside / "user-home"
        user_home.mkdir()
        escaped = ws.outside / "elsewhere.md"
        forbidden = user_home / ".env"
        escaped.write_text("escaped content")
        forbidden.write_text("USER_SECRET=do-not-read")
        (user_home / "CLAUDE.md").write_text("@../elsewhere.md\n@.env\nuser rule\n")
        loaded = load_instructions(ws.policy, user_home=user_home)
        if len(loaded.entries) != 1:
            fail(f"denied user-scope includes loaded: {loaded.entries!r}")
        if "escaped content" in loaded.render() or "USER_SECRET" in loaded.render():
            fail("denied user-scope include content leaked into rendered instructions")
        if len(loaded.warnings) != 2:
            fail(f"user-scope guards produced the wrong warnings: {loaded.warnings!r}")
        if str(escaped) not in loaded.warnings[0] or "outside" not in loaded.warnings[0]:
            fail(f"user-home escape warning was wrong: {loaded.warnings[0]!r}")
        if str(forbidden) not in loaded.warnings[1] or ".env" not in loaded.warnings[1]:
            fail(f"user denylist warning did not name the matched pattern: {loaded.warnings[1]!r}")


@check("instructions.user_scope_relative_denylist")
def check_user_scope_relative_denylist() -> None:
    with workspace() as ws:
        user_home = ws.outside / "build" / ".orchestra"
        user_home.mkdir(parents=True)
        user_file = user_home / "CLAUDE.md"
        included = user_home / "style.md"
        forbidden = user_home / ".env"
        escaped = ws.outside / "elsewhere.md"
        included.write_text("allowed user style")
        forbidden.write_text("RELATIVE_SECRET=do-not-read")
        escaped.write_text("escaped content")
        user_file.write_text("@style.md\n@.env\n@../../elsewhere.md\nuser rule\n")

        loaded = load_instructions(ws.policy, user_home=user_home)
        if [entry.path for entry in loaded.entries] != [user_file, included]:
            fail(f"denylisted ancestor blocked a safe user include: {loaded!r}")
        if len(loaded.warnings) != 2:
            fail(f"relative user guards produced the wrong warnings: {loaded.warnings!r}")
        if str(forbidden) not in loaded.warnings[0] or ".env" not in loaded.warnings[0]:
            fail(f"relative denylist warning was wrong: {loaded.warnings[0]!r}")
        if str(escaped) not in loaded.warnings[1] or "outside" not in loaded.warnings[1]:
            fail(f"containment warning was wrong: {loaded.warnings[1]!r}")
        rendered = loaded.render()
        if "RELATIVE_SECRET" in rendered or "escaped content" in rendered or any(
            "RELATIVE_SECRET" in entry.text or "escaped content" in entry.text
            for entry in loaded.entries
        ):
            fail("relative user-scope guards leaked denied content")


@check("instructions.tab_indented_content")
def check_tab_indented_content() -> None:
    with workspace():
        text, includes = instructions._process_text("\t@foo\nreal\n")
        if not text.startswith("\t@foo\n"):
            fail(f"tab-indented @ line was removed or reindented: {text!r}")
        if includes:
            fail(f"tab-indented @ line became an include: {includes!r}")


@check("instructions.symlinked_user_home")
def check_symlinked_user_home() -> None:
    with workspace() as ws:
        real_home = ws.outside / "real-user-home"
        linked_home = ws.outside / "linked-user-home"
        real_home.mkdir()
        user_file = real_home / "CLAUDE.md"
        included = real_home / "style.md"
        user_file.write_text("@style.md\nuser rule\n")
        included.write_text("symlinked user style\n")
        os.symlink(real_home, linked_home, target_is_directory=True)

        loaded = load_instructions(ws.policy, user_home=linked_home)
        if [entry.path for entry in loaded.entries] != [user_file, included]:
            fail(f"symlinked user_home did not load its neighbour: {loaded!r}")
        child = loaded.entries[1]
        if (
            child.scope != InstructionScope.USER
            or child.depth != 1
            or child.parent != user_file
        ):
            fail(f"symlinked user include provenance was wrong: {child!r}")
        if loaded.warnings:
            fail(f"symlinked user_home produced warnings: {loaded.warnings!r}")


@check("instructions.whitespace_only_entry")
def check_whitespace_only_entry() -> None:
    with workspace() as ws:
        user_home = ws.root / "user-home"
        user_home.mkdir()
        (user_home / "CLAUDE.md").write_text("user rule")
        (ws.root / "CLAUDE.md").write_text("   \n ")
        (ws.root / "AGENTS.md").write_text("project rule")
        loaded = load_instructions(ws.policy, user_home=user_home)

    whitespace_entry = loaded.entries[1]
    project_entry = loaded.entries[2]
    user_block = loaded.render_entry(loaded.entries[0])
    project_block = loaded.render_entry(project_entry)
    rendered = loaded.render()
    if loaded.render_entry(whitespace_entry) != "":
        fail(f"whitespace-only entry rendered a block: {whitespace_entry!r}")
    if "# instructions: project CLAUDE.md" in rendered:
        fail(f"whitespace-only instruction header survived: {rendered!r}")
    if not project_block or project_block != "# instructions: project AGENTS.md\nproject rule":
        fail(f"non-empty project block changed: {project_block!r}")
    if rendered != f"{user_block}\n\n{project_block}":
        fail(f"surrounding entries were not joined by one blank line: {rendered!r}")
