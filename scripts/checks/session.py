"""Registered checks for append-only run transcripts and session layout."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest.mock as mock
from datetime import datetime
from pathlib import Path

from orchestra_api.agent_loop import ApiAgent
from orchestra_api.cancellation import CancellationToken, OperationCancelled
from orchestra_api.identity import new_agent_ref
from orchestra_api.leader import Leader, LeaderConfig
from orchestra_api.models import Message, ModelRequest, ModelResponse, Role, ToolCall, ToolResult
from orchestra_api.permissions import PermissionPolicy
from orchestra_api.providers.base import ModelProvider
from orchestra_api.providers.fake import FakeModelProvider
from orchestra_api.runner import run_task
from orchestra_api.session import (
    SessionStore,
    TranscriptError,
    TranscriptWriter,
    default_sessions_root,
    read_records,
)
from orchestra_api.tools.base import LocalTool
from orchestra_api.tools.metadata import ToolEffect, ToolMetadata
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
            session=store,
        )
        records, dropped = read_records(store.directory / "run.jsonl")
        expected = [
            "run_started",
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
        with mock.patch("orchestra_api.session.os.replace", wraps=os.replace) as replaced:
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
            os.environ, {"ORCHESTRA_SESSIONS_DIR": str(override)}, clear=False
        ):
            resolved = default_sessions_root()
        if resolved != override or override.exists():
            fail(
                "sessions-root override was ignored or created as a read side effect: "
                f"resolved={resolved!r}"
            )
