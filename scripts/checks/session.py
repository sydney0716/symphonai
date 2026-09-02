"""Registered checks for append-only run transcripts and session layout."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import stat
import tempfile
import unittest.mock as mock
from collections import Counter
from datetime import datetime
from pathlib import Path

from symphonai_api.agent_loop import AgentRunResult, ApiAgent
from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.identity import SCHEMA_VERSION, new_agent_ref, new_id
from symphonai_api.leader import Leader, LeaderConfig
from symphonai_api.models import (
    DocumentBlock,
    ImageBlock,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    TextBlock,
    ToolCall,
    ToolResult,
)
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import ModelProvider
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.runner import resume_task, run_task, standard_tool_registry
from symphonai_api.serialization import message_to_json
from symphonai_api.session import (
    RunState,
    SessionError,
    SessionStore,
    TranscriptError,
    TranscriptWriter,
    TurnState,
    classify_run,
    default_sessions_root,
    fork_run,
    load_run,
    load_run_for_resume,
    read_records,
    resume_run,
    tool_result_search_path,
)
from symphonai_api.repair import unanswered_tool_call_ids
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from scripts.checks.harness import check, fail


def _session_paths(root: Path) -> tuple[Path, Path, SessionStore]:
    repo = root / "repo"
    sessions = root / "private-sessions"
    repo.mkdir()
    store = SessionStore(sessions, "session-check")
    return repo, sessions, store


def _final_response(text: str = "done") -> ModelResponse:
    return ModelResponse(message=Message(role=Role.ASSISTANT, content=text))


class _CancellingTool(LocalTool):
    @property
    def name(self) -> str:
        return "cancel_work"

    @property
    def description(self) -> str:
        return "Cancel this check's tool turn."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.DESTRUCTIVE,
            concurrency_safe=False,
            paths=None,
        )

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        assert cancel is not None
        cancel.cancel()
        raise OperationCancelled


class _FailingProvider(ModelProvider):
    @property
    def name(self) -> str:
        return "failing-check-provider"

    @property
    def wire_format(self) -> int:
        return 4

    def create_response(
        self,
        request: ModelRequest,
        *,
        cancel: CancellationToken | None = None,
    ) -> ModelResponse:
        raise RuntimeError("scripted provider failure")


class _RecordingProvider(FakeModelProvider):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__(responses)
        self.requests: list[ModelRequest] = []

    def create_response(
        self,
        request: ModelRequest,
        *,
        cancel: CancellationToken | None = None,
    ) -> ModelResponse:
        self.requests.append(request)
        return super().create_response(request, cancel=cancel)


@check("session.no_transcript_no_files")
def check_no_transcript_no_files() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        home = root / "home"
        home.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(home)}, clear=False):
            expected_default = default_sessions_root()
            ApiAgent(
                FakeModelProvider([_final_response()]),
                {},
                PermissionPolicy(repo_root=repo),
            ).run([Message(role=Role.USER, content="hello")])
        if expected_default.exists():
            fail("running without a transcript created the default sessions root")


@check("session.record_sequence")
def check_record_sequence() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo, _, store = _session_paths(Path(temporary))
        run_task(
            FakeModelProvider([_final_response()]),
            PermissionPolicy(repo_root=repo),
            "hello",
            system_prompt="system guidance",
            session=store,
        )
        records, dropped = read_records(store.directory / "run.jsonl")
        expected = [
            "run_started",
            "message",
            "message",
            "turn_started",
            "request",
            "message",
            "turn_finished",
            "run_finished",
        ]
        if [record["type"] for record in records] != expected or dropped:
            fail(f"unexpected transcript sequence: {records!r}, dropped={dropped}")
        envelope = [
            "schema_version",
            "record_id",
            "ts",
            "type",
            "run_id",
            "agent_id",
            "turn_id",
            "data",
        ]
        for record in records:
            if list(record) != envelope or record["schema_version"] != 1:
                fail(f"invalid transcript envelope: {record!r}")
            parsed = datetime.fromisoformat(record["ts"].replace("Z", "+00:00"))
            if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
                fail(f"transcript timestamp is not UTC: {record['ts']!r}")
        store.close()


@check("session.tool_records")
def check_tool_records() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo, _, store = _session_paths(Path(temporary))
        (repo / "note.txt").write_text("hello")
        provider = FakeModelProvider(
            [
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="transcript-tool-call",
                                name="read_file",
                                arguments={"path": "note.txt"},
                            )
                        ],
                    )
                ),
                _final_response(),
            ]
        )
        run_task(provider, PermissionPolicy(repo_root=repo), "read it", session=store)
        records, _ = read_records(store.directory / "run.jsonl")
        tool_started = [record for record in records if record["type"] == "tool_started"]
        tool_results = [record for record in records if record["type"] == "tool_result"]
        tool_messages = [
            record
            for record in records
            if record["type"] == "message" and record["data"]["role"] == "tool"
        ]
        ids = {
            tool_started[0]["data"]["tool_call_id"],
            tool_results[0]["data"]["tool_call_id"],
            tool_messages[0]["data"]["tool_result"]["tool_call_id"],
        } if tool_started and tool_results and tool_messages else set()
        if ids != {"transcript-tool-call"}:
            fail(f"tool transcript records do not match: {records!r}")
        positions = [record["type"] for record in records]
        if not (
            positions.index("tool_started")
            < next(
                index
                for index, record in enumerate(records)
                if record["type"] == "message" and record["data"]["role"] == "tool"
            )
            < positions.index("tool_result")
        ):
            fail(f"tool records are out of order: {positions!r}")
        store.close()


@check("session.cancellation_record")
def check_cancellation_record() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo, _, store = _session_paths(Path(temporary))
        agent_ref = new_agent_ref("cancel-check")
        provider = FakeModelProvider(
            [
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(id="repair-1", name="cancel_work"),
                            ToolCall(id="repair-2", name="cancel_work"),
                        ],
                    )
                )
            ]
        )
        token = CancellationToken()
        result = ApiAgent(
            provider,
            {"cancel_work": _CancellingTool()},
            PermissionPolicy(repo_root=repo),
            agent_ref=agent_ref,
            transcript=store.writer_for(agent_ref.agent_id, is_root=True),
        ).run([Message(role=Role.USER, content="cancel")], cancel=token)
        records, _ = read_records(store.directory / "run.jsonl")
        cancellations = [record for record in records if record["type"] == "cancellation"]
        finishes = [record for record in records if record["type"] == "run_finished"]
        if (
            result.stopped_reason != "cancelled"
            or len(cancellations) != 1
            or cancellations[0]["data"]["repaired_tool_call_ids"]
            != ["repair-1", "repair-2"]
            or finishes[-1]["data"]["stopped_reason"] != "cancelled"
        ):
            fail(f"cancelled transcript is incomplete: {records!r}")
        store.close()


@check("session.failure_record")
def check_failure_record() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo, _, store = _session_paths(Path(temporary))
        agent_ref = new_agent_ref("failure-check")
        try:
            ApiAgent(
                _FailingProvider(),
                {},
                PermissionPolicy(repo_root=repo),
                agent_ref=agent_ref,
                transcript=store.writer_for(agent_ref.agent_id, is_root=True),
            ).run([Message(role=Role.USER, content="fail")])
        except RuntimeError as exc:
            if str(exc) != "scripted provider failure":
                fail(f"provider exception changed: {exc!r}")
        else:
            fail("provider failure did not propagate")
        records, _ = read_records(store.directory / "run.jsonl")
        failures = [record for record in records if record["type"] == "run_failed"]
        if len(failures) != 1 or failures[0]["data"]["error"] != "scripted provider failure":
            fail(f"provider failure was not persisted: {records!r}")
        store.close()


def _write_three_records(path: Path) -> None:
    writer = TranscriptWriter(path)
    for index in range(3):
        writer.append(
            "turn_started",
            run_id="run-reader",
            agent_id="agent-reader",
            turn_id=f"turn-{index}",
            data={"index": index},
        )
    writer.close()


@check("session.truncated_tail_recovers")
def check_truncated_tail_recovers() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "run.jsonl"
        _write_three_records(path)
        raw = path.read_bytes()
        path.write_bytes(raw[:-12])
        records, dropped = read_records(path)
        if len(records) != 2 or dropped <= 0:
            fail(f"truncated tail was not recovered: records={records!r}, dropped={dropped}")


@check("session.corrupt_middle_raises")
def check_corrupt_middle_raises() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "run.jsonl"
        _write_three_records(path)
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(lines[0] + b"not-json\n" + b"".join(lines[1:]))
        try:
            read_records(path)
        except TranscriptError:
            pass
        else:
            fail("malformed middle record was treated as a crash tail")


@check("session.subagent_transcript_separate")
def check_subagent_transcript_separate() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo, _, store = _session_paths(Path(temporary))
        leader_provider = FakeModelProvider(
            [
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="dispatch-one",
                                name="dispatch_subagent",
                                arguments={"subagent_name": "worker", "task": "answer"},
                            )
                        ],
                    )
                ),
                _final_response("leader done"),
            ]
        )
        leader = Leader(
            LeaderConfig(
                leader_provider=leader_provider,
                subagent_provider=FakeModelProvider([_final_response("worker done")]),
                repo_root=str(repo),
            ),
            session=store,
        )
        result = leader.run("delegate")
        subagent_id = result.subagents["worker"].agent_ref.agent_id
        subagent_path = store.directory / f"agent-{subagent_id}.jsonl"
        root_records, _ = read_records(store.directory / "run.jsonl")
        subagent_records, _ = read_records(subagent_path)
        if not subagent_records:
            fail("subagent transcript was not written")
        if any(record["agent_id"] == subagent_id for record in root_records):
            fail("subagent records leaked into run.jsonl")
        if any(record["agent_id"] != subagent_id for record in subagent_records):
            fail("subagent transcript contains another agent's records")
        store.close()


@check("session.meta_atomic_replace")
def check_meta_atomic_replace() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        _, _, store = _session_paths(Path(temporary))
        writer = store.writer_for("root-agent", is_root=True)
        writer.append(
            "run_started",
            run_id="run-meta",
            agent_id="root-agent",
            turn_id=None,
            data={"agent_name": "root", "parent_run_id": None, "model": "fake"},
        )
        transcript_before = (store.directory / "run.jsonl").read_bytes()
        meta = store.read_meta()
        meta["title"] = "updated atomically"
        with mock.patch("symphonai_api.session.os.replace", wraps=os.replace) as replaced:
            store.write_meta(meta)
        if replaced.call_count != 1:
            fail("metadata rewrite did not use one atomic os.replace")
        if store.read_meta()["title"] != "updated atomically":
            fail("metadata rewrite did not persist the complete object")
        if (store.directory / "run.jsonl").read_bytes() != transcript_before:
            fail("metadata rewrite touched the append-only transcript")
        store.close()


@check("session.request_record_has_no_secret")
def check_request_record_has_no_secret() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo, _, store = _session_paths(Path(temporary))
        secret = "TRANSCRIPT_SECRET_MUST_NOT_APPEAR_68192"
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=False):
            run_task(
                FakeModelProvider([_final_response()]),
                PermissionPolicy(repo_root=repo),
                "hello",
                session=store,
            )
        records, _ = read_records(store.directory / "run.jsonl")
        request = next(record for record in records if record["type"] == "request")
        encoded = json.dumps(request)
        if secret in encoded or set(request["data"]) != {"model", "message_count", "tool_names"}:
            fail(f"request transcript exposes extra transport data: {request!r}")
        store.close()


@check("session.directory_outside_repo_root")
def check_directory_outside_repo_root() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo, _, store = _session_paths(Path(temporary))
        agent_ref = new_agent_ref("mode-check")
        writer = store.writer_for(agent_ref.agent_id, is_root=True)
        writer.append(
            "run_started",
            run_id="run-mode",
            agent_id=agent_ref.agent_id,
            turn_id=None,
            data={"agent_name": "mode", "parent_run_id": None, "model": "fake"},
        )
        policy = PermissionPolicy(repo_root=repo)
        decision = policy.check_read(store.directory / "meta.json")
        directory_mode = stat.S_IMODE(store.directory.stat().st_mode)
        tool_results_mode = stat.S_IMODE((store.directory / "tool-results").stat().st_mode)
        file_modes = {
            stat.S_IMODE((store.directory / name).stat().st_mode)
            for name in ("meta.json", "run.jsonl")
        }
        if store.directory.is_relative_to(repo) or decision.allowed:
            fail("session directory is reachable through the workspace policy")
        if directory_mode != 0o700 or tool_results_mode != 0o700 or file_modes != {0o600}:
            fail(
                "private session modes changed: "
                f"directory={oct(directory_mode)}, tool_results={oct(tool_results_mode)}, "
                f"files={file_modes!r}"
            )
        store.close()


@check("session.sessions_root_env_override")
def check_sessions_root_env_override() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        override = Path(temporary) / "override-root"
        with mock.patch.dict(
            os.environ, {"SYMPHONAI_SESSIONS_DIR": str(override)}, clear=False
        ):
            resolved = default_sessions_root()
        if resolved != override or override.exists():
            fail(
                "sessions-root override was ignored or created as a read side effect: "
                f"resolved={resolved!r}"
            )


def _seed_persisted_messages(
    sessions_root: Path,
    name: str,
    messages: list[Message],
) -> tuple[SessionStore, str]:
    store = SessionStore(sessions_root, name)
    agent_id = f"agent-{name}"
    writer = store.writer_for(agent_id, is_root=True)
    writer.append(
        "run_started",
        run_id=f"run-{name}",
        agent_id=agent_id,
        turn_id=None,
        data={"agent_name": name, "parent_run_id": None, "model": "fake"},
    )
    for message in messages:
        writer.append(
            "message",
            run_id=f"run-{name}",
            agent_id=agent_id,
            turn_id=message.turn_id,
            data=message_to_json(message),
        )
    writer.append(
        "run_finished",
        run_id=f"run-{name}",
        agent_id=agent_id,
        turn_id=None,
        data={"stopped_reason": "final_response", "turns_used": 1},
    )
    store.close()
    return store, agent_id


@check("session.load_round_trip")
def check_load_round_trip() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        (repo / "note.txt").write_text("persist me")
        store = SessionStore(root / "sessions", "load-round-trip")
        agent_ref = new_agent_ref("load-check")
        tool_call = ToolCall(
            id="load-tool-call",
            name="read_file",
            arguments={"path": "note.txt"},
            provider_metadata={"thoughtSignature": {"nested": [1, {"two": 2}]}},
            vendor_id="vendor-load-tool-call",
        )
        provider = FakeModelProvider(
            [
                ModelResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        content=(
                            TextBlock("inspect"),
                            ImageBlock(data="aW1hZ2U=", media_type="image/png"),
                            DocumentBlock(data="cGRm", filename="note.pdf"),
                        ),
                        tool_calls=[tool_call],
                    )
                ),
                _final_response("loaded"),
            ]
        )
        tools = standard_tool_registry(["read_file"])
        seed = [
            Message(role=Role.SYSTEM, content="load system"),
            Message(role=Role.USER, content="load note.txt"),
        ]
        result = ApiAgent(
            provider,
            tools,
            PermissionPolicy(repo_root=repo),
            agent_ref=agent_ref,
            transcript=store.writer_for(agent_ref.agent_id, is_root=True),
        ).run(seed)
        loaded = load_run(store)
        records, _ = read_records(store.directory / "run.jsonl")
        expected_record_ids = [
            record["record_id"]
            for record in records
            if record["type"] == "message"
        ]
        if loaded.messages != result.messages:
            fail(
                "persisted messages did not round-trip into the run conversation: "
                f"loaded={loaded.messages!r}, result={result.messages!r}"
            )
        if loaded.record_ids != expected_record_ids or len(loaded.record_ids) != len(loaded.messages):
            fail(f"message record ids are not aligned: {loaded!r}")
        store.close()


@check("session.load_ignores_unknown_record_type")
def check_load_ignores_unknown_record_type() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store, _ = _seed_persisted_messages(
            Path(temporary) / "sessions",
            "unknown-type",
            [Message(role=Role.ASSISTANT, content="known", turn_id="turn-known")],
        )
        path = store.directory / "run.jsonl"
        future_record = {
            "schema_version": SCHEMA_VERSION,
            "record_id": new_id("rec"),
            "ts": "2000-01-01T00:00:00.000Z",
            "type": "future_thing",
            "run_id": "run-unknown-type",
            "agent_id": "agent-unknown-type",
            "turn_id": None,
            "data": {"future": True},
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(future_record) + "\n")
        loaded = load_run(store)
        if [message.text for message in loaded.messages] != ["known"]:
            fail(f"unknown record type changed the loaded conversation: {loaded!r}")


def _resume_source_messages() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="one system", turn_id="turn-system"),
        Message(role=Role.USER, content="old question", turn_id="turn-user"),
        Message(role=Role.ASSISTANT, content="old answer", turn_id="turn-answer"),
    ]


@check("session.resume_continues_conversation")
def check_resume_continues_conversation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        source, _ = _seed_persisted_messages(
            root / "sessions", "resume-source", _resume_source_messages()
        )
        destination = SessionStore(root / "sessions", "resume-destination")
        provider = _RecordingProvider([_final_response("new answer")])
        resume_task(
            provider,
            PermissionPolicy(repo_root=repo),
            "new question",
            store=source,
            new_store=destination,
        )
        expected = [
            *_resume_source_messages(),
            Message(role=Role.USER, content="new question"),
        ]
        if provider.requests[0].messages != expected:
            fail(f"resume request did not continue the loaded conversation: {provider.requests[0]!r}")
        if sum(message.role == Role.SYSTEM for message in provider.requests[0].messages) != 1:
            fail("resume duplicated the persisted system message")
        destination.close()


@check("session.resume_is_a_new_run")
def check_resume_is_a_new_run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        source, _ = _seed_persisted_messages(
            root / "sessions", "new-run-source", _resume_source_messages()
        )
        original_path = source.directory / "run.jsonl"
        original_bytes = original_path.read_bytes()
        loaded = load_run(source)
        destination = SessionStore(root / "sessions", "new-run-destination")
        result = resume_task(
            FakeModelProvider([_final_response("continued")]),
            PermissionPolicy(repo_root=repo),
            "continue",
            store=source,
            new_store=destination,
        )
        destination_records, _ = read_records(destination.directory / "run.jsonl")
        started = destination_records[0]
        if (
            result.run.run_id == loaded.run_id
            or result.run.parent_run_id != loaded.run_id
            or started["data"]["parent_run_id"] != loaded.run_id
            or source.directory == destination.directory
            or not (destination.directory / "run.jsonl").is_file()
        ):
            fail(
                "resume did not create a distinct descendant run: "
                f"result={result!r}, started={started!r}"
            )
        if original_path.read_bytes() != original_bytes:
            fail("resume appended to the original transcript")
        destination.close()


@check("session.fork_prefix_only")
def check_fork_prefix_only() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        messages = _resume_source_messages()
        with mock.patch(
            "symphonai_api.session._timestamp",
            return_value="2000-01-01T00:00:00.000Z",
        ):
            source, _ = _seed_persisted_messages(
                root / "sessions", "fork-prefix-source", messages
            )
        source_records, _ = read_records(source.directory / "run.jsonl")
        message_records = [
            record for record in source_records if record["type"] == "message"
        ]
        through = message_records[1]["record_id"]
        through_index = next(
            index
            for index, record in enumerate(source_records)
            if record["record_id"] == through
        )
        source_prefix = [
            record
            for record in source_records[: through_index + 1]
            if record["type"] != "run_started"
        ]
        destination = SessionStore(root / "sessions", "fork-prefix-destination")
        forked = fork_run(
            source,
            through_record_id=through,
            new_store=destination,
        )
        fork_records, _ = read_records(destination.directory / "run.jsonl")
        copied = fork_records[1:]
        if forked.messages != messages[:2] or len(copied) != len(source_prefix):
            fail(f"fork did not contain exactly the selected prefix: {fork_records!r}")
        for original, replacement in zip(source_prefix, copied):
            if (
                replacement["type"] != original["type"]
                or replacement["turn_id"] != original["turn_id"]
                or replacement["data"] != original["data"]
                or replacement["record_id"] == original["record_id"]
                or replacement["ts"] == original["ts"]
            ):
                fail(
                    "forked record identity or payload is wrong: "
                    f"original={original!r}, replacement={replacement!r}"
                )
        if fork_records[0]["data"]["parent_run_id"] != load_run(source).run_id:
            fail("fork run_started does not descend from the source run")
        destination.close()


@check("session.fork_rejects_non_message")
def check_fork_rejects_non_message() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, agent_id = _seed_persisted_messages(
            root / "sessions",
            "fork-non-message-source",
            [Message(role=Role.ASSISTANT, content="done", turn_id="turn-done")],
        )
        writer = TranscriptWriter(source.directory / "run.jsonl")
        target_id = writer.append(
            "tool_started",
            run_id="run-fork-non-message-source",
            agent_id=agent_id,
            turn_id="turn-tool-start",
            data={"tool_call_id": "tool-only", "tool_name": "read_file"},
        )
        writer.close()
        destination = SessionStore(root / "sessions", "fork-non-message-destination")
        try:
            fork_run(source, through_record_id=target_id, new_store=destination)
        except SessionError as exc:
            if target_id not in str(exc):
                fail(f"non-message fork error did not name the record: {exc}")
        else:
            fail("fork accepted a tool_started record")
        if (destination.directory / "run.jsonl").exists():
            fail("rejected non-message fork wrote a transcript")


@check("session.fork_rejects_unknown_record")
def check_fork_rejects_unknown_record() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, _ = _seed_persisted_messages(
            root / "sessions",
            "fork-unknown-source",
            [Message(role=Role.ASSISTANT, content="done", turn_id="turn-done")],
        )
        destination = SessionStore(root / "sessions", "fork-unknown-destination")
        missing = "rec_missing_from_transcript"
        try:
            fork_run(source, through_record_id=missing, new_store=destination)
        except SessionError as exc:
            if missing not in str(exc):
                fail(f"unknown-record fork error did not name the id: {exc}")
        else:
            fail("fork accepted an unknown record id")


@check("session.fork_rejects_unanswered_tool_calls")
def check_fork_rejects_unanswered_tool_calls() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, _ = _seed_persisted_messages(
            root / "sessions",
            "fork-unanswered-source",
            [
                Message(
                    role=Role.ASSISTANT,
                    tool_calls=[
                        ToolCall(id="unanswered-1", name="read_file"),
                        ToolCall(id="unanswered-2", name="read_file"),
                    ],
                    turn_id="turn-unanswered",
                )
            ],
        )
        through = load_run(source).record_ids[0]
        destination = SessionStore(root / "sessions", "fork-unanswered-destination")
        try:
            fork_run(source, through_record_id=through, new_store=destination)
        except SessionError as exc:
            if "unanswered-1" not in str(exc) or "unanswered-2" not in str(exc):
                fail(f"inconsistent-fork error omitted unanswered ids: {exc}")
        else:
            fail("fork accepted unanswered tool calls")
        if (destination.directory / "run.jsonl").exists():
            fail("inconsistent fork wrote records before validation")


@check("session.fork_leaves_original_untouched")
def check_fork_leaves_original_untouched() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, _ = _seed_persisted_messages(
            root / "sessions", "fork-untouched-source", _resume_source_messages()
        )
        source_directory = source.directory
        source_run_id = source.run_id

        def snapshot() -> dict[str, tuple[bytes, int]]:
            return {
                str(path.relative_to(source_directory)): (
                    path.read_bytes(),
                    path.stat().st_mtime_ns,
                )
                for path in source_directory.rglob("*")
                if path.is_file()
            }

        before = snapshot()
        source = SessionStore.open(root / "sessions", source_run_id)
        through = load_run(source).record_ids[-1]
        destination = SessionStore(root / "sessions", "fork-untouched-destination")
        fork_run(source, through_record_id=through, new_store=destination)
        fork_path = destination.directory / "run.jsonl"
        if not fork_path.is_file() or not read_records(fork_path)[0]:
            fail("untouched-source check never produced the fork transcript")
        if snapshot() != before:
            fail("fork modified source bytes or mtimes")
        destination.close()


@check("session.resume_after_truncated_tail")
def check_resume_after_truncated_tail() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        source, _ = _seed_persisted_messages(
            root / "sessions", "truncated-resume-source", _resume_source_messages()
        )
        with (source.directory / "run.jsonl").open("ab") as handle:
            handle.write(b'{"schema_version":1,"record_id":"truncated')
        loaded = load_run(source)
        if loaded.dropped_bytes <= 0:
            fail("truncated transcript did not report dropped bytes")
        resumed_messages, resumed_id = resume_run(source)
        destination = SessionStore(root / "sessions", "truncated-resume-destination")
        result = resume_task(
            FakeModelProvider([_final_response("continued")]),
            PermissionPolicy(repo_root=repo),
            "continue",
            store=source,
            new_store=destination,
        )
        if resumed_messages != loaded.messages or resumed_id != loaded.run_id:
            fail("resume_run changed a crash-tail-recovered load")
        if result.run.parent_run_id != loaded.run_id:
            fail("resume after a truncated tail lost lineage")
        destination.close()


@check("session.schema_version_from_the_future")
def check_schema_version_from_the_future() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store, _ = _seed_persisted_messages(
            Path(temporary) / "sessions",
            "future-schema",
            [Message(role=Role.ASSISTANT, content="future", turn_id="turn-future")],
        )
        path = store.directory / "run.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        future_version = SCHEMA_VERSION + 1
        first["schema_version"] = future_version
        lines[0] = json.dumps(first, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            load_run(store)
        except SessionError as exc:
            error = str(exc)
            if str(future_version) not in error or str(SCHEMA_VERSION) not in error:
                fail(f"future-schema error did not name both versions: {error}")
        else:
            fail("future transcript schema was accepted")


def _real_run(
    root: Path, name: str
) -> tuple[Path, SessionStore, AgentRunResult]:
    repo = root / f"repo-{name}"
    repo.mkdir()
    store = SessionStore(root / "sessions", name)
    result = run_task(
        FakeModelProvider([_final_response("original answer")]),
        PermissionPolicy(repo_root=repo),
        "original question",
        system_prompt="original system",
        session=store,
    )
    return repo, store, result


def _snapshot_directory(directory: Path) -> dict[str, tuple]:
    snapshot: dict[str, tuple] = {
        ".": ("directory", directory.stat().st_mtime_ns)
    }
    for path in sorted(directory.rglob("*")):
        relative = str(path.relative_to(directory))
        if path.is_file():
            snapshot[relative] = ("file", path.read_bytes(), path.stat().st_mtime_ns)
        elif path.is_dir():
            snapshot[relative] = ("directory", path.stat().st_mtime_ns)
    return snapshot


@check("session.seed_messages_persisted")
def check_seed_messages_persisted() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        _, store, result = _real_run(Path(temporary), "seed-persisted")
        records, _ = read_records(store.directory / "run.jsonl")
        expected_types = [
            "run_started",
            "message",
            "message",
            "turn_started",
            "request",
            "message",
            "turn_finished",
            "run_finished",
        ]
        message_records = [record for record in records if record["type"] == "message"]
        if [record["type"] for record in records] != expected_types:
            fail(f"seed messages were not persisted before the first turn: {records!r}")
        if [record["data"]["role"] for record in message_records[:2]] != [
            "system",
            "user",
        ]:
            fail(f"seed role ordering changed: {message_records!r}")
        if any(record["turn_id"] is not None for record in message_records[:2]):
            fail(f"fresh seed messages gained envelope turn ids: {message_records!r}")
        if load_run(store).messages != result.messages:
            fail("real run did not load back into its complete conversation")
        store.close()


@check("session.resume_round_trips_a_real_run")
def check_resume_round_trips_a_real_run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, source, original = _real_run(root, "real-resume-source")
        loaded = load_run(source)
        if loaded.messages != original.messages:
            fail(f"real source run lost its seed conversation: {loaded!r}")
        destination = SessionStore(root / "sessions", "real-resume-destination")
        provider = _RecordingProvider([_final_response("resumed answer")])
        resume_task(
            provider,
            PermissionPolicy(repo_root=repo),
            "new prompt",
            store=source,
            new_store=destination,
        )
        expected = [*original.messages, Message(role=Role.USER, content="new prompt")]
        request_messages = provider.requests[0].messages
        if request_messages != expected:
            fail(f"real resume request lost or reordered messages: {request_messages!r}")
        if sum(message.role == Role.SYSTEM for message in request_messages) != 1:
            fail("real resume request did not contain exactly one system message")
        destination.close()


@check("session.chat_persists_each_turn_once")
def check_chat_persists_each_turn_once() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        store = SessionStore(root / "sessions", "chat-once")
        leader = Leader(
            LeaderConfig(
                leader_provider=FakeModelProvider(
                    [_final_response("first answer"), _final_response("second answer")]
                ),
                subagent_provider=FakeModelProvider([_final_response("unused")]),
                repo_root=str(repo),
            ),
            session=store,
        )
        leader.chat("first question")
        second = leader.chat("second question")
        records, _ = read_records(store.directory / "run.jsonl")
        persisted = Counter(
            json.dumps(record["data"], sort_keys=True, separators=(",", ":"))
            for record in records
            if record["type"] == "message"
        )
        held = Counter(
            json.dumps(message_to_json(message), sort_keys=True, separators=(",", ":"))
            for message in second.leader_messages
        )
        if persisted != held or any(count != 1 for count in persisted.values()):
            fail(
                "chat transcript duplicated or omitted accumulated history: "
                f"persisted={persisted!r}, held={held!r}"
            )
        store.close()


@check("session.resumed_run_transcript_is_self_contained")
def check_resumed_run_transcript_is_self_contained() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, source, _ = _real_run(root, "self-contained-source")
        first_destination = SessionStore(
            root / "sessions", "self-contained-first-resume"
        )
        first_result = resume_task(
            FakeModelProvider([_final_response("first resumed answer")]),
            PermissionPolicy(repo_root=repo),
            "first resumed prompt",
            store=source,
            new_store=first_destination,
        )
        first_loaded = load_run(first_destination)
        if first_loaded.messages != first_result.messages:
            fail("resumed run transcript is not self-contained")
        second_destination = SessionStore(
            root / "sessions", "self-contained-second-resume"
        )
        provider = _RecordingProvider([_final_response("second resumed answer")])
        resume_task(
            provider,
            PermissionPolicy(repo_root=repo),
            "second resumed prompt",
            store=first_destination,
            new_store=second_destination,
        )
        expected = [
            *first_result.messages,
            Message(role=Role.USER, content="second resumed prompt"),
        ]
        if provider.requests[0].messages != expected:
            fail("a second resume lost history from the first resumed run")
        first_destination.close()
        second_destination.close()


@check("session.construct_does_not_clobber_meta")
def check_construct_does_not_clobber_meta() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _, store, result = _real_run(root, "construct-existing")
        store.close()
        path = store.directory / "meta.json"
        before_bytes = path.read_bytes()
        before = json.loads(before_bytes)
        reopened = SessionStore(root / "sessions", store.run_id)
        after_bytes = path.read_bytes()
        after = reopened.read_meta()
        if before_bytes != after_bytes:
            fail("constructing an existing SessionStore rewrote meta.json")
        if (
            after["created_at"] != before["created_at"]
            or after["agent_id"] != result.agent.agent_id
            or after["stopped_reason"] != "final_response"
        ):
            fail(f"existing session metadata was reset: before={before!r}, after={after!r}")
        reopened.close()


@check("session.open_preserves_meta")
def check_open_preserves_meta() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _, store, result = _real_run(root, "open-existing")
        store.close()
        directory = store.directory
        run_id = store.run_id
        before = _snapshot_directory(directory)
        opened = SessionStore.open(root / "sessions", run_id)
        loaded = load_run(opened)
        after = _snapshot_directory(directory)
        if after != before:
            fail("opening and loading an existing run changed bytes or mtimes")
        if (
            loaded.meta.get("agent_id") != result.agent.agent_id
            or loaded.meta.get("stopped_reason") != "final_response"
        ):
            fail(f"opened run returned reset metadata: {loaded.meta!r}")
        opened.close()


@check("session.open_missing_run_raises")
def check_open_missing_run_raises() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        sessions_root = Path(temporary) / "sessions"
        missing_run_id = "missing-run-id"
        try:
            SessionStore.open(sessions_root, missing_run_id)
        except SessionError as exc:
            if missing_run_id not in str(exc):
                fail(f"missing-run error did not name the run id: {exc}")
        else:
            fail("opening a missing run silently created it")
        if sessions_root.exists():
            fail("opening a missing run created the sessions root or an entry")


def _digest_for_check(message: Message) -> str:
    canonical = json.dumps(
        message_to_json(message), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.blake2b(canonical, digest_size=16).hexdigest()


def _compacted_chat(
    root: Path, name: str
) -> tuple[Leader, SessionStore, list[tuple[list[str], list[str], list[Message]]]]:
    repo = root / f"repo-{name}"
    repo.mkdir()
    store = SessionStore(root / "sessions", name)
    leader = Leader(
        LeaderConfig(
            leader_provider=FakeModelProvider(
                [
                    _final_response("a" * 320),
                    _final_response("b" * 320),
                    _final_response("c" * 320),
                    _final_response("d" * 40),
                ]
            ),
            subagent_provider=FakeModelProvider([_final_response("unused")]),
            repo_root=str(repo),
            chat_token_budget=260,
            chat_recent_turns=1,
        ),
        session=store,
    )
    observations: list[tuple[list[str], list[str], list[Message]]] = []
    original_run = leader._agent.run

    def observed_run(messages: list[Message], **kwargs: object) -> AgentRunResult:
        observations.append(
            (
                list(leader._agent._persisted_digests),
                [_digest_for_check(message) for message in messages],
                list(messages),
            )
        )
        return original_run(messages, **kwargs)

    leader._agent.run = observed_run
    questions = [
        "q0 " + "u" * 320,
        "q1 " + "v" * 320,
        "q2 " + "w" * 320,
        "q3 final question",
    ]
    for question in questions:
        leader.chat(question)
    return leader, store, observations


@check("session.compacted_chat_persists_every_message")
def check_compacted_chat_persists_every_message() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        leader, store, observations = _compacted_chat(
            Path(temporary), "compacted-every-message"
        )
        records, _ = read_records(store.directory / "run.jsonl")
        if not any(record["type"] == "compaction" for record in records):
            fail("four-turn scenario did not actually compact")
        equal_length_divergence = [
            messages
            for persisted, current, messages in observations
            if len(persisted) == len(current) and persisted != current
        ]
        if not equal_length_divergence or not any(
            messages
            and messages[-1].role == Role.USER
            and messages[-1].text == "q3 final question"
            for messages in equal_length_divergence
        ):
            fail(f"check did not reproduce the equal-length q3 rewrite: {observations!r}")
        raw_messages = [
            message_from_record["data"]
            for message_from_record in records
            if message_from_record["type"] == "message"
        ]
        q3_json = message_to_json(Message(role=Role.USER, content="q3 final question"))
        if q3_json not in raw_messages:
            fail("the post-compaction q3 user message was never persisted")
        loaded = load_run(store)
        if loaded.messages != leader._chat_messages:
            fail(
                "compacted transcript does not rebuild the leader conversation: "
                f"loaded={loaded.messages!r}, held={leader._chat_messages!r}"
            )
        store.close()


@check("session.conversation_rewritten_record_shape")
def check_conversation_rewritten_record_shape() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _, store, observations = _compacted_chat(root, "rewrite-shape")
        records, _ = read_records(store.directory / "run.jsonl")
        rewrites = [
            record for record in records if record["type"] == "conversation_rewritten"
        ]
        expected: list[dict[str, int]] = []
        for persisted, current, _ in observations:
            kept = 0
            for old_digest, new_digest in zip(persisted, current):
                if old_digest != new_digest:
                    break
                kept += 1
            if kept < len(persisted):
                expected.append(
                    {"kept_prefix": kept, "replaced": len(persisted) - kept}
                )
        if [record["data"] for record in rewrites] != expected:
            fail(f"rewrite record shape or cardinality changed: {rewrites!r}")

        plain_repo = root / "plain-repo"
        plain_repo.mkdir()
        plain_store = SessionStore(root / "sessions", "plain-no-rewrite")
        plain_leader = Leader(
            LeaderConfig(
                leader_provider=FakeModelProvider(
                    [_final_response("one"), _final_response("two")]
                ),
                subagent_provider=FakeModelProvider([_final_response("unused")]),
                repo_root=str(plain_repo),
            ),
            session=plain_store,
        )
        plain_leader.chat("first")
        plain_leader.chat("second")
        plain_records, _ = read_records(plain_store.directory / "run.jsonl")
        if any(record["type"] == "conversation_rewritten" for record in plain_records):
            fail("plain append-only chat wrote a conversation_rewritten record")
        store.close()
        plain_store.close()


@check("session.load_run_honours_a_rewrite")
def check_load_run_honours_a_rewrite() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        leader, store, _ = _compacted_chat(root, "load-rewrite")
        loaded = load_run(store)
        if loaded.messages != leader._chat_messages:
            fail("load_run concatenated superseded and compacted conversations")
        if len(loaded.record_ids) != len(loaded.messages):
            fail("rewrite handling broke message/record-id alignment")

        _, plain_store, plain_result = _real_run(root, "load-without-rewrite")
        if load_run(plain_store).messages != plain_result.messages:
            fail("load_run changed behavior for a transcript without rewrites")

        reorder_repo = root / "reorder-repo"
        reorder_repo.mkdir()
        reorder_store = SessionStore(root / "sessions", "load-reordered-prefix")
        reorder_agent_ref = new_agent_ref("reorder-check")
        reorder_agent = ApiAgent(
            FakeModelProvider(
                [_final_response("first order"), _final_response("second order")]
            ),
            {},
            PermissionPolicy(repo_root=reorder_repo),
            agent_ref=reorder_agent_ref,
            transcript=reorder_store.writer_for(
                reorder_agent_ref.agent_id, is_root=True
            ),
        )
        first = Message(role=Role.USER, content="first seed")
        second = Message(role=Role.USER, content="second seed")
        reorder_agent.run([first, second])
        reordered_result = reorder_agent.run([second, first])
        if load_run(reorder_store).messages != reordered_result.messages:
            fail("load_run treated reordered digests as an unchanged prefix")
        store.close()
        plain_store.close()
        reorder_store.close()


@check("session.rewrite_prefix_beyond_messages_raises")
def check_rewrite_prefix_beyond_messages_raises() -> None:
    # A negative prefix would slice silently from the end, and `bool` is an
    # `int`, so both must be rejected as loudly as one that is too large.
    for index, kept_prefix in enumerate((99, -1, True)):
        with tempfile.TemporaryDirectory() as temporary:
            name = f"impossible-rewrite-{index}"
            source, agent_id = _seed_persisted_messages(
                Path(temporary) / "sessions",
                name,
                [Message(role=Role.ASSISTANT, content="only one", turn_id="turn-one")],
            )
            writer = TranscriptWriter(source.directory / "run.jsonl")
            writer.append(
                "conversation_rewritten",
                run_id=f"run-{name}",
                agent_id=agent_id,
                turn_id=None,
                data={"kept_prefix": kept_prefix, "replaced": 1},
            )
            writer.close()
            try:
                loaded = load_run(source)
            except SessionError as exc:
                error = str(exc)
                if source.run_id not in error or repr(kept_prefix) not in error:
                    fail(
                        "impossible-rewrite error omitted required values: "
                        f"kept_prefix={kept_prefix!r}, error={error}"
                    )
            else:
                fail(
                    "load_run accepted an impossible rewrite prefix "
                    f"{kept_prefix!r}: loaded={loaded.messages!r}"
                )


@check("session.resume_a_compacted_chat")
def check_resume_a_compacted_chat() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        leader, source, _ = _compacted_chat(root, "resume-compacted")
        destination = SessionStore(root / "sessions", "resume-compacted-destination")
        provider = _RecordingProvider([_final_response("resumed compacted answer")])
        resume_task(
            provider,
            PermissionPolicy(repo_root=Path(leader._config.repo_root)),
            "resume prompt",
            store=source,
            new_store=destination,
        )
        expected = [
            *leader._chat_messages,
            Message(role=Role.USER, content="resume prompt"),
        ]
        request_messages = provider.requests[0].messages
        if request_messages != expected:
            fail(f"compacted resume included superseded messages: {request_messages!r}")
        held_systems = sum(
            message.role == Role.SYSTEM for message in leader._chat_messages
        )
        resumed_systems = sum(message.role == Role.SYSTEM for message in request_messages)
        if resumed_systems != held_systems:
            fail("compacted resume duplicated a system message")
        source.close()
        destination.close()


@check("session.one_shot_after_chat_repersists")
def check_one_shot_after_chat_repersistence() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        store = SessionStore(root / "sessions", "one-shot-after-chat")
        leader = Leader(
            LeaderConfig(
                leader_provider=FakeModelProvider(
                    [_final_response("chat answer"), _final_response("run answer")]
                ),
                subagent_provider=FakeModelProvider([_final_response("unused")]),
                repo_root=str(repo),
            ),
            session=store,
        )
        leader.chat("chat question")
        result = leader.run("fresh run question", system_prompt="fresh run system")
        records, _ = read_records(store.directory / "run.jsonl")
        rewrites = [
            record for record in records if record["type"] == "conversation_rewritten"
        ]
        if not rewrites or rewrites[-1]["data"] != {
            "kept_prefix": 0,
            "replaced": 2,
        }:
            fail(f"fresh one-shot run did not supersede prior chat: {rewrites!r}")
        if load_run(store).messages != result.leader_messages:
            fail("fresh one-shot conversation was skipped after chat")
        store.close()


def _diagnosis(store: SessionStore):
    loaded = load_run(store)
    records, _ = read_records(store.directory / "run.jsonl")
    return loaded, classify_run(loaded, records)


def _partial_crash_store(root: Path, name: str) -> SessionStore:
    store = SessionStore(root / "sessions", name)
    writer = store.writer_for(f"agent-{name}", is_root=True)
    run_id = f"run-{name}"
    turn_id = f"turn-{name}"
    writer.append(
        "run_started",
        run_id=run_id,
        agent_id=f"agent-{name}",
        turn_id=None,
        data={"agent_name": name, "parent_run_id": None, "model": "fake"},
    )
    writer.append(
        "turn_started",
        run_id=run_id,
        agent_id=f"agent-{name}",
        turn_id=turn_id,
        data={"index": 1},
    )
    assistant = Message(
        role=Role.ASSISTANT,
        tool_calls=[
            ToolCall(id="crash-answered", name="read_file"),
            ToolCall(id="crash-unanswered", name="read_file"),
        ],
        turn_id=turn_id,
    )
    answered = Message(
        role=Role.TOOL,
        tool_result=ToolResult(
            tool_call_id="crash-answered", ok=True, content="answer"
        ),
        turn_id=turn_id,
    )
    for message in (assistant, answered):
        writer.append(
            "message",
            run_id=run_id,
            agent_id=f"agent-{name}",
            turn_id=turn_id,
            data=message_to_json(message),
        )
    writer.close()
    return store


def _three_turn_chat(root: Path, name: str) -> tuple[Leader, SessionStore, list[str]]:
    repo = root / f"repo-{name}"
    repo.mkdir()
    store = SessionStore(root / "sessions", name)
    leader = Leader(
        LeaderConfig(
            leader_provider=FakeModelProvider(
                [_final_response("one"), _final_response("two"), _final_response("three")]
            ),
            subagent_provider=FakeModelProvider([_final_response("unused")]),
            repo_root=str(repo),
        ),
        session=store,
    )
    for question in ("question one", "question two", "question three"):
        leader.chat(question)
    records, _ = read_records(store.directory / "run.jsonl")
    run_ids = [record["run_id"] for record in records if record["type"] == "run_started"]
    return leader, store, run_ids


@check("session.diagnose_completed")
def check_diagnose_completed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        completed = SessionStore(root / "sessions", "completed")
        run_task(
            FakeModelProvider([_final_response("complete")]),
            PermissionPolicy(repo_root=repo),
            "finish",
            session=completed,
        )
        _, final_diagnosis = _diagnosis(completed)
        if (
            final_diagnosis.state != RunState.COMPLETED
            or final_diagnosis.stopped_reason != "final_response"
        ):
            fail(f"final response was diagnosed incorrectly: {final_diagnosis!r}")

        capped = SessionStore(root / "sessions", "max-turns")
        tool_response = ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(id="unknown", name="not_registered")],
            )
        )
        run_task(
            FakeModelProvider([tool_response]),
            PermissionPolicy(repo_root=repo),
            "hit cap",
            max_turns=1,
            session=capped,
        )
        _, capped_diagnosis = _diagnosis(capped)
        if (
            capped_diagnosis.state != RunState.COMPLETED
            or capped_diagnosis.stopped_reason != "max_turns"
        ):
            fail(f"max-turn run was diagnosed incorrectly: {capped_diagnosis!r}")
        completed.close()
        capped.close()


@check("session.diagnose_cancelled")
def check_diagnose_cancelled() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        store = SessionStore(root / "sessions", "cancelled")
        token = CancellationToken()
        response = ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(id="cancelled-call", name="cancel_work")],
            )
        )
        agent_ref = new_agent_ref("cancelled-diagnosis")
        ApiAgent(
            FakeModelProvider([response]),
            {"cancel_work": _CancellingTool()},
            PermissionPolicy(repo_root=repo),
            agent_ref=agent_ref,
            transcript=store.writer_for(agent_ref.agent_id, is_root=True),
        ).run([Message(role=Role.USER, content="cancel")], cancel=token)
        _, diagnosis = _diagnosis(store)
        if diagnosis.state != RunState.CANCELLED or diagnosis.stopped_reason != "cancelled":
            fail(f"cancelled run was diagnosed incorrectly: {diagnosis!r}")
        store.close()


@check("session.diagnose_failed")
def check_diagnose_failed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        store = SessionStore(root / "sessions", "failed")
        try:
            run_task(
                _FailingProvider(),
                PermissionPolicy(repo_root=repo),
                "fail",
                session=store,
            )
        except RuntimeError:
            pass
        else:
            fail("scripted provider failure did not escape")
        _, diagnosis = _diagnosis(store)
        if diagnosis.state != RunState.FAILED or diagnosis.stopped_reason != "failed":
            fail(f"failed run was diagnosed incorrectly: {diagnosis!r}")
        store.close()


@check("session.diagnose_crashed")
def check_diagnose_crashed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        store = SessionStore(root / "sessions", "crashed")
        writer = store.writer_for("agent-crashed", is_root=True)
        writer.append(
            "run_started",
            run_id="run-crashed",
            agent_id="agent-crashed",
            turn_id=None,
            data={"agent_name": "crashed", "parent_run_id": None, "model": "fake"},
        )
        writer.close()
        with (store.directory / "run.jsonl").open("ab") as handle:
            handle.write(b'{"truncated":')
        _, diagnosis = _diagnosis(store)
        if diagnosis.state != RunState.CRASHED or diagnosis.dropped_bytes == 0:
            fail(f"truncated crash was diagnosed incorrectly: {diagnosis!r}")


@check("session.turn_states")
def check_turn_states() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        store = SessionStore(root / "sessions", "turn-states")
        writer = store.writer_for("agent-turns", is_root=True)
        writer.append(
            "run_started",
            run_id="run-turns",
            agent_id="agent-turns",
            turn_id=None,
            data={"agent_name": "turns", "parent_run_id": None, "model": "fake"},
        )
        completed_assistant = Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="done-call", name="read_file")],
            turn_id="turn-completed",
        )
        completed_result = Message(
            role=Role.TOOL,
            tool_result=ToolResult("done-call", True, "done"),
            turn_id="turn-completed",
        )
        partial_assistant = Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="partial-call", name="read_file")],
            turn_id="turn-partial",
        )
        fixtures = (
            ("turn-completed", (completed_assistant, completed_result)),
            ("turn-partial", (partial_assistant,)),
            ("turn-empty", ()),
        )
        for index, (turn_id, messages) in enumerate(fixtures, start=1):
            writer.append(
                "turn_started",
                run_id="run-turns",
                agent_id="agent-turns",
                turn_id=turn_id,
                data={"index": index},
            )
            for message in messages:
                writer.append(
                    "message",
                    run_id="run-turns",
                    agent_id="agent-turns",
                    turn_id=turn_id,
                    data=message_to_json(message),
                )
        writer.close()
        _, diagnosis = _diagnosis(store)
        expected = (
            ("turn-completed", TurnState.COMPLETED),
            ("turn-partial", TurnState.PARTIAL),
            ("turn-empty", TurnState.EMPTY),
        )
        if diagnosis.turns != expected:
            fail(f"turn states were classified incorrectly: {diagnosis.turns!r}")


@check("session.diagnose_does_not_mutate")
def check_diagnose_does_not_mutate() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = _partial_crash_store(Path(temporary), "observe-only")
        loaded = load_run(store)
        before = list(loaded.messages)
        records, _ = read_records(store.directory / "run.jsonl")
        classify_run(loaded, records)
        if loaded.messages != before:
            fail("diagnosis repaired or otherwise mutated loaded messages")


@check("session.repair_on_resume")
def check_repair_on_resume() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        store = _partial_crash_store(root, "repair")
        loaded, diagnosis, repaired = load_run_for_resume(store)
        if repaired != ["crash-unanswered"]:
            fail(f"repair returned the wrong ids: {repaired!r}")
        repair = loaded.messages[-1].tool_result
        if (
            diagnosis.state != RunState.CRASHED
            or repair is None
            or repair.ok
            or repair.cancelled
            or "session ended" not in (repair.error or "")
        ):
            fail(f"crash repair has the wrong state or result: {diagnosis!r}, {repair!r}")
        if unanswered_tool_call_ids(loaded.messages) != []:
            fail("repaired conversation still has unanswered tool calls")
        repo = root / "repo-repair"
        repo.mkdir()
        destination = SessionStore(root / "sessions", "repair-destination")
        resume_task(
            FakeModelProvider([_final_response("resumed")]),
            PermissionPolicy(repo_root=repo),
            "continue",
            store=store,
            new_store=destination,
        )
        records, _ = read_records(destination.directory / "run.jsonl")
        persisted_repairs = [
            record
            for record in records
            if record["type"] == "message"
            and (record["data"].get("tool_result") or {}).get("tool_call_id")
            == "crash-unanswered"
        ]
        if len(persisted_repairs) != 1:
            fail(f"descendant transcript did not persist the repair: {records!r}")
        destination.close()


@check("session.repair_leaves_transcript_untouched")
def check_repair_leaves_transcript_untouched() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = _partial_crash_store(Path(temporary), "repair-bytes")
        path = store.directory / "run.jsonl"
        before = path.read_bytes()
        load_run_for_resume(store)
        if path.read_bytes() != before:
            fail("repair rewrote the original transcript")


@check("session.diagnose_last_run_of_many")
def check_diagnose_last_run_of_many() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _, completed, run_ids = _three_turn_chat(root, "many-completed")
        loaded, diagnosis = _diagnosis(completed)
        if (
            diagnosis.state != RunState.COMPLETED
            or diagnosis.run_count != 3
            or diagnosis.run_id != run_ids[-1]
            or loaded.run_id != run_ids[-1]
            or len(diagnosis.turns) != 1
        ):
            fail(f"last completed run was not isolated: {diagnosis!r}")

        _, crashed, crashed_ids = _three_turn_chat(root, "many-crashed")
        path = crashed.directory / "run.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        last_finish = max(
            index
            for index, line in enumerate(lines)
            if json.loads(line)["type"] == "run_finished"
        )
        path.write_text("".join(lines[:last_finish] + lines[last_finish + 1 :]), encoding="utf-8")
        crashed_loaded, crashed_diagnosis = _diagnosis(crashed)
        if (
            crashed_diagnosis.state != RunState.CRASHED
            or crashed_diagnosis.run_id != crashed_ids[-1]
            or crashed_loaded.run_id != crashed_ids[-1]
            or crashed_diagnosis.run_count != 3
        ):
            fail(f"earlier terminal record masked the final crash: {crashed_diagnosis!r}")
        completed.close()
        crashed.close()


@check("session.loaded_run_id_is_the_last_run")
def check_loaded_run_id_is_the_last_run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        leader, source, run_ids = _three_turn_chat(root, "last-run-id")
        loaded = load_run(source)
        destination = SessionStore(root / "sessions", "last-run-destination")
        result = resume_task(
            FakeModelProvider([_final_response("resumed")]),
            PermissionPolicy(repo_root=Path(leader._config.repo_root)),
            "resume",
            store=source,
            new_store=destination,
        )
        if loaded.run_id != run_ids[-1] or loaded.run_count != 3:
            fail(f"LoadedRun did not identify the final run: {loaded!r}")
        if result.run.parent_run_id != run_ids[-1]:
            fail(f"resume descended from {result.run.parent_run_id!r}, not the last run")
        source.close()
        destination.close()


@check("session.diagnose_a_compacted_transcript")
def check_diagnose_a_compacted_transcript() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        leader, store, _ = _compacted_chat(Path(temporary), "diagnose-compacted")
        records, _ = read_records(store.directory / "run.jsonl")
        loaded, diagnosis, repaired = load_run_for_resume(store)
        expected_compactions = sum(record["type"] == "compaction" for record in records)
        if loaded.messages != leader._chat_messages:
            fail("recovery did not load the conversation held after compaction")
        if (
            diagnosis.state != RunState.COMPLETED
            or diagnosis.compactions != expected_compactions
            or expected_compactions == 0
            or repaired
        ):
            fail(f"compacted transcript diagnosis was wrong: {diagnosis!r}, {repaired!r}")
        store.close()


@check("session.search_path_walks_ancestry")
def check_search_path_walks_ancestry() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        sessions = Path(temporary) / "sessions"
        first = SessionStore(sessions, "first")
        second = SessionStore(sessions, "second")
        third = SessionStore(sessions, "third")
        second.set_parent_session(first.run_id)
        third.set_parent_session(second.run_id)
        expected = (
            third.directory / "tool-results",
            second.directory / "tool-results",
            first.directory / "tool-results",
        )
        if tool_result_search_path(third) != expected:
            fail(f"three-deep search path was wrong: {tool_result_search_path(third)!r}")
        first.set_parent_session(third.run_id)
        if tool_result_search_path(third) != expected:
            fail("sidecar cycle changed or failed to terminate the search path")

        deleted_first = SessionStore(sessions, "deleted-first")
        deleted_second = SessionStore(sessions, "deleted-second")
        deleted_third = SessionStore(sessions, "deleted-third")
        deleted_second.set_parent_session(deleted_first.run_id)
        deleted_third.set_parent_session(deleted_second.run_id)
        shutil.rmtree(deleted_first.directory)
        deleted_path = tool_result_search_path(deleted_third)
        if deleted_path != (
            deleted_third.directory / "tool-results",
            deleted_second.directory / "tool-results",
        ):
            fail(f"deleted ancestor did not terminate safely: {deleted_path!r}")


@check("session.search_path_creates_nothing")
def check_search_path_creates_nothing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sessions = root / "sessions"
        first = SessionStore(sessions, "path-first")
        middle = SessionStore(sessions, "path-middle")
        last = SessionStore(sessions, "path-last")
        middle.set_parent_session(first.run_id)
        last.set_parent_session(middle.run_id)
        missing = middle.directory / "tool-results"
        missing.rmdir()
        middle_before = _snapshot_directory(middle.directory)
        paths = tool_result_search_path(last)
        if missing.exists() or _snapshot_directory(middle.directory) != middle_before:
            fail("search path created or modified the middle result directory")
        if paths != (
            last.directory / "tool-results",
            first.directory / "tool-results",
        ):
            fail(f"missing result directory was not skipped: {paths!r}")

        repo, existing_source, _ = _real_run(root, "readonly-existing-source")
        existing_before = _snapshot_directory(existing_source.directory)
        existing_destination = SessionStore(sessions, "readonly-existing-destination")
        resume_task(
            FakeModelProvider([_final_response("done")]),
            PermissionPolicy(repo_root=repo),
            "resume",
            store=existing_source,
            new_store=existing_destination,
            offload_tool_results=True,
        )
        if _snapshot_directory(existing_source.directory) != existing_before:
            fail("resume changed the source run with an existing result directory")

        missing_repo, missing_source, _ = _real_run(root, "readonly-missing-source")
        missing_source_results = missing_source.directory / "tool-results"
        missing_source_results.rmdir()
        missing_before = _snapshot_directory(missing_source.directory)
        missing_destination = SessionStore(sessions, "readonly-missing-destination")
        resume_task(
            FakeModelProvider([_final_response("done")]),
            PermissionPolicy(repo_root=missing_repo),
            "resume",
            store=missing_source,
            new_store=missing_destination,
            offload_tool_results=True,
        )
        if (
            missing_source_results.exists()
            or _snapshot_directory(missing_source.directory) != missing_before
        ):
            fail("resume created or modified a missing source result directory")
        existing_source.close()
        existing_destination.close()
        missing_source.close()
        missing_destination.close()
