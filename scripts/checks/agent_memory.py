"""Checks for bounded project and global memory."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path
import stat
from types import SimpleNamespace
from unittest import mock

import symphonai_api.session as session_module
from symphonai_api.agent_file import AgentFileError, memory_settings
from symphonai_api.agent_memory import (
    MAX_ENTRIES,
    MAX_ENTRY_CHARS,
    MAX_TOTAL_CHARS,
    AgentMemory,
    MemorySettings,
    MemoryUnavailable,
)
from symphonai_api.leader import _memory_entries
from symphonai_api.models import ToolCall
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.memory import MemoryTool
from scripts.checks.harness import check, fail


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


def _expect_unavailable(
    operation: Callable[[], object],
    *,
    path: Path,
    unreadable: bool = False,
) -> None:
    try:
        operation()
    except MemoryUnavailable as exc:
        message = str(exc)
        if not unreadable and str(path) not in message:
            fail(f"MemoryUnavailable omitted path: {message!r}")
        if not isinstance(exc, RuntimeError):
            fail("MemoryUnavailable was not a RuntimeError")
        if unreadable:
            if not exc.unreadable:
                fail("unreadable memory did not mark its failure")
        elif not isinstance(exc.__cause__, OSError):
            fail("MemoryUnavailable did not wrap an OSError as a RuntimeError")
    except OSError as exc:
        fail(f"memory operation leaked bare {type(exc).__name__}: {exc}")
    else:
        fail("unavailable memory operation returned normally")


@check("agent_memory.read_write_and_order")
def read_write_and_order() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = AgentMemory(Path(temporary) / "memory")
        if store.read() != ():
            fail("new project set was not empty")
        first = store.write("first", agent_name="reviewer", run_id="run-1")
        if (
            not first.id
            or first.text != "first"
            or first.agent_name != "reviewer"
            or first.run_id != "run-1"
            or first.written_at == 0
            or store.read() != (first,)
        ):
            fail(f"written memory lost attribution: {first!r}")
        for index in range(2, 6):
            store.write(f"entry-{index}", agent_name="reviewer", run_id=f"run-{index}")
        entries = store.read()
        if [entry.text for entry in entries] != [
            "first",
            "entry-2",
            "entry-3",
            "entry-4",
            "entry-5",
        ]:
            fail(f"memory was not returned oldest first: {entries!r}")


@check("agent_memory.bounds_refuse_when_full")
def bounds_refuse_when_full() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        count_store = AgentMemory(directory / "count")
        for index in range(MAX_ENTRIES + 1):
            try:
                count_store.write(f"entry-{index}", agent_name="reviewer", run_id=f"run-{index}")
            except ValueError as exc:
                if index != MAX_ENTRIES or "full" not in str(exc).lower() or "settings" not in str(exc).lower():
                    fail(f"unexpected full memory refusal at entry {index}: {exc}")
            else:
                if index >= MAX_ENTRIES:
                    fail("memory accepted an entry after reaching its count limit")
        count_entries = count_store.read()
        if len(count_entries) != MAX_ENTRIES or count_entries[0].text != "entry-0":
            fail(f"full memory changed existing entries: {count_entries!r}")

        total_store = AgentMemory(directory / "total")
        entry_count = MAX_TOTAL_CHARS // MAX_ENTRY_CHARS + 1
        for index in range(entry_count):
            text = f"{index:02d}" + "x" * (MAX_ENTRY_CHARS - 2)
            try:
                total_store.write(text, agent_name="reviewer", run_id=f"run-{index}")
            except ValueError as exc:
                if index != entry_count - 1 or "full" not in str(exc).lower():
                    fail(f"unexpected total memory refusal at entry {index}: {exc}")
        total_entries = total_store.read()
        if len(total_entries) != entry_count - 1 or total_entries[0].text[:2] != "00":
            fail(f"full memory evicted or changed existing entries: {total_entries!r}")


@check("agent_memory.refuses_bad_entries")
def refuses_bad_entries() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = AgentMemory(Path(temporary) / "memory")
        store.write("existing", agent_name="reviewer", run_id="run-1")
        before = store.read()
        oversized = "x" * (MAX_ENTRY_CHARS + 1)
        try:
            store.write(oversized, agent_name="reviewer", run_id="run-2")
        except ValueError as exc:
            message = str(exc)
            if str(MAX_ENTRY_CHARS) not in message or str(len(oversized)) not in message:
                fail(f"over-long error omitted limit or actual length: {message!r}")
        else:
            fail("over-long memory entry was accepted")
        if store.read() != before:
            fail("refused over-long entry changed memory")
        for blank in ("", " ", "\t\n"):
            try:
                store.write(blank, agent_name="reviewer", run_id="run-blank")
            except ValueError:
                pass
            else:
                fail(f"blank memory entry was accepted: {blank!r}")
            if store.read() != before:
                fail("refused blank entry changed memory")


@check("agent_memory.project_and_global_scopes")
def project_and_global_scopes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary).resolve()
        root = base / "memory"
        first = AgentMemory(root, base / "repo-one")
        second = AgentMemory(root, base / "repo-two")
        first_project = first.write("project one", agent_name="leader", run_id="one")
        first_global = first.write("shared preference", agent_name="reviewer", run_id="two", scope="global")
        second_project = second.write("project two", agent_name="leader", run_id="three")
        if first.read() != (first_project,) or second.read() != (second_project,):
            fail("repositories did not receive separate project memory sets")
        if first.read("global") != (first_global,) or second.read("global") != (first_global,):
            fail("global memory was not shared across repositories")
        defaulted = first.write("defaults to project", agent_name="leader", run_id="four")
        if defaulted.scope != "project" or first.read()[-1] != defaulted:
            fail("memory write without scope did not use the project set")


@check("agent_memory.update_and_delete_by_id")
def update_and_delete_by_id() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = AgentMemory(Path(temporary) / "memory", Path(temporary) / "repo")
        original = store.write(
            "old text", agent_name="leader", run_id="run-1", scope="global"
        )
        updated = store.update(original.id, "new text")
        if (
            updated is None
            or updated.id != original.id
            or updated.text != "new text"
            or updated.agent_name != original.agent_name
            or store.read("global") != (updated,)
        ):
            fail(f"memory update did not preserve the entry identity: {updated!r}")
        if store.update("unknown", "new text") is not None:
            fail("update accepted an unknown memory id")
        if not store.delete(original.id) or store.read("global"):
            fail("delete did not remove the selected memory")
        if store.delete(original.id):
            fail("delete accepted an unknown memory id")


@check("agent_memory.corrupt_file_is_inert")
def corrupt_file_is_inert() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "memory"
        store = AgentMemory(root)
        store.write("valid", agent_name="reviewer", run_id="run-1")
        files = list(root.rglob("*.json"))
        if len(files) != 1:
            fail(f"write did not create one memory file: {files!r}")
        legacy = root / ("a" * 64 + ".json")
        legacy_text = '[{"text":"old agent note"}]'
        legacy.write_text(legacy_text, encoding="utf-8")
        path = files[0]
        corrupt = b'{"not": "a list"}'
        path.write_bytes(corrupt)
        if store.read() != ():
            fail("corrupt memory did not read as empty")
        if not path.exists() or path.read_bytes() != corrupt:
            fail("reading corrupt memory changed or deleted the file")
        if legacy.read_text(encoding="utf-8") != legacy_text:
            fail("legacy per-agent memory was modified")

        unavailable_root = Path(temporary) / "unavailable"
        unavailable_root.mkdir()
        unavailable = AgentMemory(unavailable_root)
        write_path = unavailable._path("project")
        write_path.parent.mkdir(parents=True)
        original_mode = stat.S_IMODE(write_path.parent.stat().st_mode)
        write_path.parent.chmod(0o500)
        try:
            _expect_unavailable(
                lambda: unavailable.write("note", agent_name="reviewer", run_id="run-write"),
                path=write_path,
            )
        finally:
            write_path.parent.chmod(original_mode)


@check("agent_memory.unreadable_files_are_preserved")
def unreadable_files_are_preserved() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "memory"
        store = AgentMemory(root, Path(temporary) / "repo")
        for index in range(5):
            store.write(f"entry {index}", agent_name="leader", run_id=f"run-{index}")
        global_entry = store.write(
            "global preference",
            agent_name="leader",
            run_id="global-run",
            scope="global",
        )
        project_path = store._path("project")
        valid_bytes = project_path.read_bytes()
        payload = json.loads(valid_bytes)
        payload[2]["schema_version"] = 999
        project_path.write_text(json.dumps(payload), encoding="utf-8")
        before = project_path.read_bytes()

        tool = MemoryTool(store, "leader", lambda: "run-new")
        result = tool.execute(
            ToolCall("remember-corrupt", "remember", {"text": "new note"}),
            PermissionPolicy(Path(temporary)),
        )
        if result.ok or "could not be read" not in result.error or "nothing was saved" not in result.error:
            fail(f"remember did not explain the unreadable file: {result!r}")
        if project_path.read_bytes() != before:
            fail("remember changed a set containing an unreadable entry")

        project_path.write_bytes(valid_bytes)
        valid_status = store.read_status("project")
        if valid_status != (store.read("project"), False) or len(valid_status[0]) != 5:
            fail(f"normal memory timestamps no longer read: {valid_status!r}")
        valid_result = tool.execute(
            ToolCall("remember-valid", "remember", {"text": "normal timestamp"}),
            PermissionPolicy(Path(temporary)),
        )
        if not valid_result.ok:
            fail(f"remember rejected a normal timestamp: {valid_result!r}")

        for index, timestamp in enumerate((1.7e12, float("inf"))):
            timestamp_payload = json.loads(valid_bytes)
            timestamp_payload[0]["written_at"] = timestamp
            project_path.write_text(json.dumps(timestamp_payload), encoding="utf-8")
            timestamp_before = project_path.read_bytes()
            if store.read_status("project") != ((), True):
                fail(f"unrepresentable timestamp {timestamp!r} was readable")
            result = tool.execute(
                ToolCall(f"remember-timestamp-{index}", "remember", {"text": "new note"}),
                PermissionPolicy(Path(temporary)),
            )
            if result.ok or "could not be read" not in result.error or "nothing was saved" not in result.error:
                fail(f"remember did not reject timestamp {timestamp!r}: {result!r}")
            if project_path.read_bytes() != timestamp_before:
                fail(f"remember changed a set with timestamp {timestamp!r}")

        seeded = _memory_entries(store, SimpleNamespace(memory=MemorySettings(enabled=True)))
        if [entry.id for entry in seeded] != [global_entry.id]:
            fail(f"unreadable project memory hid the global seed: {seeded!r}")

        invalid_root = Path(temporary) / "invalid-json"
        invalid_store = AgentMemory(invalid_root, Path(temporary) / "repo")
        invalid_path = invalid_store._path("project")
        invalid_path.parent.mkdir(parents=True)
        invalid_path.write_bytes(b"not json")
        invalid_before = invalid_path.read_bytes()
        _expect_unavailable(
            lambda: invalid_store.update("entry-id", "changed"),
            path=invalid_path,
            unreadable=True,
        )
        _expect_unavailable(
            lambda: invalid_store.delete("entry-id"),
            path=invalid_path,
            unreadable=True,
        )
        if invalid_path.read_bytes() != invalid_before:
            fail("update or delete changed a non-JSON memory file")

        missing_store = AgentMemory(Path(temporary) / "new-memory", Path(temporary) / "repo")
        first = missing_store.write("first note", agent_name="leader", run_id="first-run")
        if missing_store.read() != (first,):
            fail("a missing memory file did not accept its first entry")


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


@check("agent_memory.default_root")
def default_root() -> None:
    harness_sessions = Path(os.environ["SYMPHONAI_SESSIONS_DIR"]).resolve()
    harness_memory = Path(os.environ["SYMPHONAI_MEMORY_DIR"]).resolve()
    if (
        harness_sessions.name != "sessions"
        or harness_memory.name != "memory"
        or harness_memory.parent.parent != harness_sessions.parent
    ):
        fail(
            "check harness did not isolate its memory root beside the sessions root"
        )
    with tempfile.TemporaryDirectory() as temporary:
        home = Path(temporary) / "home"
        override = Path(temporary) / "elsewhere"
        with mock.patch.object(session_module, "symphonai_home", return_value=home):
            with mock.patch.dict("os.environ", {}, clear=True):
                if session_module.default_memory_root() != home / "memory":
                    fail("default memory root was not beside the other user data")
            with mock.patch.dict(
                "os.environ",
                {"SYMPHONAI_MEMORY_DIR": str(override)},
                clear=True,
            ):
                if session_module.default_memory_root() != override:
                    fail("SYMPHONAI_MEMORY_DIR did not override the memory root")


@check("agent_memory.tool_writes_and_reopens")
def tool_writes_and_reopens() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "memory"
        store = AgentMemory(root)
        tool = MemoryTool(store, "reviewer", lambda: "run-current")
        result = tool.execute(
            ToolCall("remember-1", "remember", {"text": "Prefer short reports."}),
            PermissionPolicy(Path(temporary)),
        )
        if not result.ok:
            fail(f"memory tool rejected a valid lesson: {result!r}")
        entries = AgentMemory(root).read()
        if (
            len(entries) != 1
            or entries[0].text != "Prefer short reports."
            or entries[0].run_id != "run-current"
        ):
            fail(f"memory did not survive a reconstructed store: {entries!r}")
        description = tool.description.lower()
        for phrase in ("durable lesson", "person taught", "task state", "finding", "summary"):
            if phrase not in description:
                fail(f"memory tool description omitted {phrase!r}: {description!r}")


@check("agent_memory.remember_replaces_and_returns_id")
def remember_replaces_and_returns_id() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "memory"
        store = AgentMemory(root, Path(temporary))
        tool = MemoryTool(store, "reviewer", lambda: "run-current")
        policy = PermissionPolicy(Path(temporary))
        saved = tool.execute(
            ToolCall("remember-new", "remember", {"text": "Use spaces."}), policy
        )
        entry_id = saved.content.removeprefix("Remembered as ").removesuffix(".")
        if not saved.ok or saved.content != f"Remembered as {entry_id}." or not entry_id:
            fail(f"new memory did not return its id: {saved!r}")
        original = store.read()[0]
        updated = tool.execute(
            ToolCall("remember-replace", "remember", {
                "text": "Use tabs.", "replaces": entry_id, "scope": "ignored",
            }),
            policy,
        )
        current = store.read()
        if (
            not updated.ok
            or updated.content != f"Updated {entry_id}."
            or len(current) != 1
            or current[0].id != original.id
            or current[0].scope != original.scope
            or current[0].text != "Use tabs."
        ):
            fail(f"memory replacement changed identity, scope, or count: {updated!r}, {current!r}")
        missing = tool.execute(
            ToolCall("remember-missing", "remember", {
                "text": "No write", "replaces": "missing-id",
            }),
            policy,
        )
        if missing.ok or missing.error != "no memory has that id; nothing was saved" or store.read() != current:
            fail(f"unknown replacement id changed memory: {missing!r}")


@check("agent_memory.forget_requires_exact_text")
def forget_requires_exact_text() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "memory"
        store = AgentMemory(root, Path(temporary))
        tool = MemoryTool(store, "reviewer", lambda: "run-current")
        policy = PermissionPolicy(Path(temporary))
        saved = tool.execute(
            ToolCall("remember-new", "remember", {"text": "Prefer concise answers."}), policy
        )
        entry_id = saved.content.removeprefix("Remembered as ").removesuffix(".")
        forget = tool.forget_tool
        stale = forget.execute(
            ToolCall("forget-stale", "forget", {"id": entry_id, "text": "Prefer concise answer."}),
            policy,
        )
        if stale.ok or "memory changed since it was read" not in (stale.error or "") or len(store.read()) != 1:
            fail(f"forget accepted text that differed by one character: {stale!r}")
        removed = forget.execute(
            ToolCall("forget-current", "forget", {"id": entry_id, "text": "Prefer concise answers."}),
            policy,
        )
        if not removed.ok or removed.content != f"Forgot {entry_id}." or store.read():
            fail(f"forget did not remove the exact matching memory: {removed!r}")


@check("agent_memory.forget_unreadable_and_guarded_delete")
def forget_unreadable_and_guarded_delete() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "memory"
        repo = Path(temporary) / "repo"
        store = AgentMemory(root, repo)
        entry = store.write("current text", agent_name="leader", run_id="run")
        if store.delete(entry.id, expected_text="old"):
            fail("guarded delete removed an entry whose text had changed")
        if store.read() != (entry,):
            fail("guarded delete changed the entry after a text mismatch")

        path = store._path("project")
        corrupt = b'{"unreadable": true}'
        path.write_bytes(corrupt)
        before = path.read_bytes()
        tool = MemoryTool(store, "leader", lambda: "run").forget_tool
        result = tool.execute(
            ToolCall("forget-unreadable", "forget", {
                "id": entry.id,
                "text": entry.text,
            }),
            PermissionPolicy(repo),
        )
        if not result.ok and result.error != "memory file could not be read; nothing was removed":
            fail(f"forget returned the wrong unreadable-set error: {result!r}")
        if result.ok or path.read_bytes() != before:
            fail(f"forget changed an unreadable memory file: {result!r}")
