"""Background process jobs owned by one conversation."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import codecs
import locale
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections.abc import Callable

from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import DEFAULT_SHELL_OUTPUT_CHARS, PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata

MAX_RUNNING_JOBS = 8
MAX_OUTPUT_CHARS = 100_000


def _timestamp(value: float | None) -> str | None:
    return None if value is None else datetime.fromtimestamp(value, timezone.utc).isoformat()


@dataclass
class BackgroundJob:
    id: str
    process: subprocess.Popen[bytes] | None
    started_at: float
    owner: str
    label: str
    status: str = "running"
    exit_code: int | None = None
    reason: str | None = None
    ended_at: float | None = None
    output: str = ""
    total_chars: int = 0
    stopped: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)
    reader: threading.Thread | None = None


class JobManager:
    """Starts, reads, and stops process groups for a single conversation."""

    def __init__(
        self,
        output_limit: int = DEFAULT_SHELL_OUTPUT_CHARS,
        *,
        owner: str = "agent",
        on_change: Callable[[BackgroundJob], None] | None = None,
    ) -> None:
        self._jobs: dict[str, BackgroundJob] = {}
        self._lock = threading.RLock()
        self._closed = False
        self.output_limit = output_limit
        self.owner = owner
        self._on_change = on_change

    def start(
        self, argv: list[str], cwd: str, env: dict[str, str], *, label: str = "",
    ) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("job manager is closed")
            running = sum(job.status == "running" for job in self._jobs.values())
            if running >= MAX_RUNNING_JOBS:
                raise RuntimeError(f"at most {MAX_RUNNING_JOBS} background jobs may run at once")
            job_id = f"job_{os.urandom(8).hex()}"
            started_at = time.time()
            try:
                process = subprocess.Popen(
                    argv,
                    shell=False,
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                job = BackgroundJob(
                    id=job_id, process=None, started_at=started_at,
                    owner=self.owner, label=label,
                    status="failed", reason=str(exc), ended_at=time.time(),
                )
                self._jobs[job_id] = job
                self._notify(job)
                return job_id
            job = BackgroundJob(
                id=job_id, process=process, started_at=started_at,
                owner=self.owner, label=label,
            )
            self._jobs[job_id] = job
            self._notify(job)
            job.reader = threading.Thread(target=self._read_job, args=(job,), daemon=True)
            job.reader.start()
            return job_id

    def _notify(self, job: BackgroundJob) -> None:
        if self._on_change is not None:
            try:
                self._on_change(job)
            except Exception:
                pass

    def _read_job(self, job: BackgroundJob) -> None:
        process = job.process
        assert process is not None and process.stdout is not None
        decoder = codecs.getincrementaldecoder(locale.getpreferredencoding(False))(errors="replace")
        try:
            while chunk := os.read(process.stdout.fileno(), 4096):
                decoded = decoder.decode(chunk)
                if decoded:
                    with job.lock:
                        job.total_chars += len(decoded)
                        job.output = (job.output + decoded)[-MAX_OUTPUT_CHARS:]
            decoded = decoder.decode(b"", final=True)
            if decoded:
                with job.lock:
                    job.total_chars += len(decoded)
                    job.output = (job.output + decoded)[-MAX_OUTPUT_CHARS:]
        except (OSError, UnicodeError) as exc:
            with job.lock:
                job.status = "failed"
                job.reason = str(exc)
        code = process.wait()
        with job.lock:
            job.exit_code = code
            if job.status != "failed":
                job.status = "stopped" if job.stopped else "exited"
            job.ended_at = time.time()
        self._notify(job)

    def get(self, job_id: str) -> BackgroundJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def read(self, job_id: str, since: int = 0, limit: int = MAX_OUTPUT_CHARS) -> dict | None:
        job = self.get(job_id)
        if job is None:
            return None
        with job.lock:
            dropped_chars = max(0, job.total_chars - len(job.output))
            start = max(0, since - dropped_chars)
            output = job.output[start : start + limit]
            next_offset = max(since, dropped_chars) + len(output)
            return {
                "id": job.id,
                "status": job.status,
                "exit_code": job.exit_code,
                "reason": job.reason,
                "output": output,
                "since": next_offset,
                "output_dropped": since < dropped_chars,
            }

    def stop(self, job_id: str) -> dict | None:
        job = self.get(job_id)
        if job is None:
            return None
        process = job.process
        with job.lock:
            stopping = job.status == "running"
            if stopping:
                job.stopped = True
        if process is not None and stopping:
            try:
                # start_new_session records the group id even after its leader exits.
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError, PermissionError):
                if process.poll() is None:
                    try:
                        process.kill()
                    except (OSError, ProcessLookupError):
                        pass
        reader = job.reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2.0)
        if process is not None and stopping:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError, PermissionError):
                pass
            if process.stdout is not None:
                process.stdout.close()
            with job.lock:
                job.status = "stopped"
                job.exit_code = process.poll()
                job.ended_at = time.time()
            self._notify(job)
        return self.read(job_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            job_ids = [job.id for job in self._jobs.values() if job.status == "running"]
        for job_id in job_ids:
            self.stop(job_id)

    def list(self) -> list[BackgroundJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.started_at, reverse=True)

    def has_running(self) -> bool:
        with self._lock:
            return any(job.status == "running" for job in self._jobs.values())


class JobOutputTool(LocalTool):
    def __init__(self, manager: JobManager, output_limit: int) -> None:
        self._manager = manager
        self._output_limit = output_limit

    @property
    def name(self) -> str:
        return "job_output"

    @property
    def description(self) -> str:
        return "Read output from one of this conversation's background jobs."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "job_id": {"minLength": 1, "type": "string"},
            "since": {"type": "integer", "minimum": 0},
        }, "required": ["job_id"], "additionalProperties": False}

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(effect=ToolEffect.READ_ONLY, concurrency_safe=True, paths=None)

    def _execute(self, tool_call: ToolCall, policy: PermissionPolicy, cancel=None) -> ToolResult:
        result = self._manager.read(
            tool_call.arguments["job_id"], tool_call.arguments.get("since", 0), self._output_limit,
        )
        if result is None:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error="unknown job id")
        dropped = "Earlier output was dropped.\n" if result["output_dropped"] else ""
        status = result["status"]
        if result["exit_code"] is not None:
            status += f" (exit code {result['exit_code']})"
        if result["reason"]:
            status += f" ({result['reason']})"
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=True,
            content=f"{status}\n{dropped}{result['output']}\nNext since: {result['since']}",
        )


class JobStopTool(LocalTool):
    def __init__(self, manager: JobManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "job_stop"

    @property
    def description(self) -> str:
        return "Stop one of this conversation's background jobs."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "job_id": {"minLength": 1, "type": "string"},
        }, "required": ["job_id"], "additionalProperties": False}

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(effect=ToolEffect.READ_ONLY, concurrency_safe=False, paths=None)

    def _execute(self, tool_call: ToolCall, policy: PermissionPolicy, cancel=None) -> ToolResult:
        result = self._manager.stop(tool_call.arguments["job_id"])
        if result is None:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error="unknown job id")
        return ToolResult(tool_call_id=tool_call.id, ok=True, content=f"{result['status']}")
