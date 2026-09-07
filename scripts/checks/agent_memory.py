"""Checks for bounded per-agent memory."""

from __future__ import annotations

import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

from symphonai_api.agent_file import AgentFileError, memory_settings
from symphonai_api.agent_memory import (
    MAX_ENTRIES,
    MAX_ENTRY_CHARS,
    MAX_TOTAL_CHARS,
    AgentMemory,
    MemorySettings,
)
from scripts.checks.agent_spec import FORBIDDEN_IMPORTS, _forbidden_imports
from scripts.checks.harness import check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def _settings_error(path: Path, key: str) -> str:
    try:
        memory_settings(path)
    except AgentFileError as exc:
        message = str(exc)
        if str(path) not in message or key not in message:
            fail(f"memory settings error omitted file or key: {message!r}")
        return message
    except Exception as exc:
        fail(f"memory settings exposed {type(exc).__name__}: {exc!r}")
    fail(f"memory settings accepted invalid {key}")


@check("agent_memory.read_write_and_order")
def read_write_and_order() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = AgentMemory(Path(temporary) / "memory")
        if store.read("reviewer") != ():
            fail("unknown agent had memory")
        first = store.write("reviewer", "first", run_id="run-1")
        if (
            first.text != "first"
            or first.agent_name != "reviewer"
            or first.run_id != "run-1"
            or first.written_at == 0
            or store.read("reviewer") != (first,)
        ):
            fail(f"written memory lost attribution: {first!r}")
        for index in range(2, 6):
            store.write("reviewer", f"entry-{index}", run_id=f"run-{index}")
        entries = store.read("reviewer")
        if [entry.text for entry in entries] != [
            "first",
            "entry-2",
            "entry-3",
            "entry-4",
            "entry-5",
        ]:
            fail(f"memory was not returned oldest first: {entries!r}")


@check("agent_memory.bounds_drop_oldest")
def bounds_drop_oldest() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        count_store = AgentMemory(directory / "count")
        for index in range(MAX_ENTRIES + 5):
            count_store.write("reviewer", f"entry-{index}", run_id=f"run-{index}")
        count_entries = count_store.read("reviewer")
        if (
            len(count_entries) != MAX_ENTRIES
            or count_entries[0].text != "entry-5"
            or count_entries[-1].text != f"entry-{MAX_ENTRIES + 4}"
        ):
            fail(f"entry bound did not retain the newest entries: {count_entries!r}")

        total_store = AgentMemory(directory / "total")
        entry_count = MAX_TOTAL_CHARS // MAX_ENTRY_CHARS + 1
        for index in range(entry_count):
            text = f"{index:02d}" + "x" * (MAX_ENTRY_CHARS - 2)
            total_store.write("reviewer", text, run_id=f"run-{index}")
        total_entries = total_store.read("reviewer")
        if (
            entry_count >= MAX_ENTRIES
            or len(total_entries) != MAX_TOTAL_CHARS // MAX_ENTRY_CHARS
            or total_entries[0].text[:2] != "01"
            or total_entries[-1].text[:2] != f"{entry_count - 1:02d}"
            or sum(len(entry.text) for entry in total_entries) > MAX_TOTAL_CHARS
        ):
            fail(f"total bound did not drop the oldest entry: {total_entries!r}")


@check("agent_memory.refuses_bad_entries")
def refuses_bad_entries() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = AgentMemory(Path(temporary) / "memory")
        store.write("reviewer", "existing", run_id="run-1")
        before = store.read("reviewer")
        oversized = "x" * (MAX_ENTRY_CHARS + 1)
        try:
            store.write("reviewer", oversized, run_id="run-2")
        except ValueError as exc:
            message = str(exc)
            if str(MAX_ENTRY_CHARS) not in message or str(len(oversized)) not in message:
                fail(f"over-long error omitted limit or actual length: {message!r}")
        else:
            fail("over-long memory entry was accepted")
        if store.read("reviewer") != before:
            fail("refused over-long entry changed memory")
        for blank in ("", " ", "\t\n"):
            try:
                store.write("reviewer", blank, run_id="run-blank")
            except ValueError:
                pass
            else:
                fail(f"blank memory entry was accepted: {blank!r}")
            if store.read("reviewer") != before:
                fail("refused blank entry changed memory")


@check("agent_memory.forget_and_isolation")
def forget_and_isolation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "memory"
        store = AgentMemory(root)
        values = {
            "reviewer": "lower",
            "Reviewer": "upper",
        }
        for index, (agent_name, text) in enumerate(values.items()):
            store.write(agent_name, text, run_id=f"run-{index}")
        store.write("../escaped", "traversal", run_id="run-traversal")
        outside = base / "escaped.json"
        if outside.exists():
            fail(f"agent name wrote a memory file outside root: {outside!s}")
        store.write("nested/name", "separator", run_id="run-separator")
        values["../escaped"] = "traversal"
        values["nested/name"] = "separator"
        for agent_name, text in values.items():
            entries = store.read(agent_name)
            if len(entries) != 1 or entries[0].text != text:
                fail(f"agent memories shared storage: {agent_name!r}, {entries!r}")
        for file in base.rglob("*.json"):
            try:
                file.relative_to(root)
            except ValueError:
                fail(f"memory file escaped root: {file!s}")
        store.forget("reviewer")
        store.forget("reviewer")
        if store.read("reviewer") != ():
            fail("forget did not remove the selected agent")
        if store.read("Reviewer")[0].text != "upper":
            fail("forget disturbed a case-distinct agent")


@check("agent_memory.corrupt_file_is_inert")
def corrupt_file_is_inert() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "memory"
        store = AgentMemory(root)
        store.write("reviewer", "valid", run_id="run-1")
        files = list(root.glob("*.json"))
        if len(files) != 1:
            fail(f"write did not create one memory file: {files!r}")
        path = files[0]
        corrupt = b'{"not": "a list"}'
        path.write_bytes(corrupt)
        if store.read("reviewer") != ():
            fail("corrupt memory did not read as empty")
        if not path.exists() or path.read_bytes() != corrupt:
            fail("reading corrupt memory changed or deleted the file")


@check("agent_memory.settings_from_the_file")
def settings_from_the_file() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        plain = _write(directory, "plain.toml", 'prompt = "Review."\n')
        plain_settings = memory_settings(plain)
        if plain_settings.enabled is not False or plain_settings.max_entries != MAX_ENTRIES:
            fail("missing memory table changed the settings default")
        enabled = _write(
            directory,
            "enabled.toml",
            'prompt = "Review."\n[memory]\nenabled = true\nmax_entries = 16\n',
        )
        if memory_settings(enabled) != MemorySettings(enabled=True, max_entries=16):
            fail("memory settings were not loaded")
        default_count = _write(
            directory,
            "default-count.toml",
            'prompt = "Review."\n[memory]\nenabled = true\n',
        )
        if memory_settings(default_count).max_entries != MAX_ENTRIES:
            fail("memory max_entries default changed")
        for value in (0, MAX_ENTRIES + 1):
            path = _write(
                directory,
                f"bad-count-{value}.toml",
                f'prompt = "Review."\n[memory]\nmax_entries = {value}\n',
            )
            message = _settings_error(path, "max_entries")
            if str(MAX_ENTRIES) not in message:
                fail(f"max_entries error omitted its ceiling: {message!r}")
        unknown = _write(
            directory,
            "unknown.toml",
            'prompt = "Review."\n[memory]\nunknown = true\n',
        )
        _settings_error(unknown, "unknown")
        settings = memory_settings(enabled)
        try:
            settings.enabled = False  # type: ignore[misc]
        except FrozenInstanceError:
            pass
        else:
            fail("MemorySettings was mutable")


@check("agent_memory.no_runtime_imports")
def no_runtime_imports() -> None:
    source = (REPO_ROOT / "symphonai_api/agent_memory.py").read_text(encoding="utf-8")
    original_forbidden = set(FORBIDDEN_IMPORTS)
    FORBIDDEN_IMPORTS.update({"agent_run", "agent_spec", "child_context"})
    try:
        found = _forbidden_imports(source)
        if found:
            fail(f"agent_memory imports runtime wiring: {found!r}")
        probes = (
            "from symphonai_api.agent_loop import ApiAgent\n",
            "from symphonai_api.leader import Leader\n",
            "import symphonai_api.runner\n",
            "from symphonai_api import agent_run\n",
            "from . import agent_spec\n",
            "from symphonai_api.child_context import seed_messages\n",
            "from symphonai_api.provider_catalog import providers\n",
            "from symphonai_api.providers.fake import FakeModelProvider\n",
        )
        for probe in probes:
            if not _forbidden_imports(probe):
                fail(f"import inspection missed {probe.strip()!r}")
    finally:
        FORBIDDEN_IMPORTS.clear()
        FORBIDDEN_IMPORTS.update(original_forbidden)
