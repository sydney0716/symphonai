"""Configured subprocess hooks for observing events and guarding tool calls."""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from symphonai_api.config import ConfigError, ResolvedConfig, Scope
from symphonai_api.events import Event


DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_TIMEOUT_SECONDS = 30.0
CLEANUP_TIMEOUT_SECONDS = 1.0

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class HookSpec:
    events: tuple[str, ...]
    command: tuple[str, ...]
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    blocking: bool = False
    source: Path | None = None


@dataclass(frozen=True)
class HookOutcome:
    ok: bool
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str


def _valid_event_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            {"Event", "PreToolUse"}
            | {event_type.__name__ for event_type in Event.__subclasses__()}
        )
    )


def _config_error(
    source: Path | None,
    index: int,
    field: str,
    detail: str,
) -> ConfigError:
    location = str(source) if source is not None else "<session>"
    return ConfigError(f"{location}: hooks[{index}].{field}: {detail}")


def hooks_from_config(
    config: ResolvedConfig,
    *,
    repo_root: Path,
) -> tuple[HookSpec, ...]:
    """Parse the resolved hook array and retain its winning provenance."""
    del repo_root  # Commands run relative to the runner's cwd, not while parsing.
    raw_hooks = config.get("hooks", [])
    origin = config.provenance.get("hooks")
    source = origin.source if origin is not None else None
    if origin is not None and origin.scope in (Scope.PROJECT, Scope.PRIVATE):
        raise ConfigError(
            f"{source}: hooks: hooks are not read from configuration inside "
            "a repository; define them in ~/.symphonai/config.toml"
        )
    if not isinstance(raw_hooks, list):
        raise _config_error(source, 0, "hooks", "must be an array of tables")

    valid_names = _valid_event_names()
    valid_set = set(valid_names)
    parsed: list[HookSpec] = []
    for index, raw_hook in enumerate(raw_hooks):
        if not isinstance(raw_hook, Mapping):
            raise _config_error(source, index, "hook", "must be a table")
        unknown = raw_hook.keys() - {
            "on",
            "command",
            "timeout_seconds",
            "blocking",
        }
        if unknown:
            field = str(sorted(unknown)[0])
            raise _config_error(source, index, field, "unknown field")

        raw_events = raw_hook.get("on")
        if (
            not isinstance(raw_events, list)
            or not raw_events
            or not all(isinstance(item, str) for item in raw_events)
        ):
            raise _config_error(source, index, "on", "must be a non-empty array of strings")
        events = tuple(raw_events)
        unknown_event = next((name for name in events if name not in valid_set), None)
        if unknown_event is not None:
            raise _config_error(
                source,
                index,
                "on",
                f"unknown event {unknown_event!r}; valid values: {', '.join(valid_names)}",
            )

        raw_command = raw_hook.get("command")
        if (
            not isinstance(raw_command, list)
            or not raw_command
            or not all(isinstance(item, str) and item for item in raw_command)
        ):
            raise _config_error(
                source,
                index,
                "command",
                "must be a non-empty argv array of strings",
            )
        command = tuple(raw_command)

        raw_timeout = raw_hook.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        if type(raw_timeout) not in (int, float):
            raise _config_error(source, index, "timeout_seconds", "must be a number")
        timeout_seconds = float(raw_timeout)
        if (
            not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds > MAX_TIMEOUT_SECONDS
        ):
            raise _config_error(
                source,
                index,
                "timeout_seconds",
                f"must be positive and no greater than {MAX_TIMEOUT_SECONDS}",
            )

        raw_blocking = raw_hook.get("blocking", False)
        if type(raw_blocking) is not bool:
            raise _config_error(source, index, "blocking", "must be a boolean")
        if "PreToolUse" in events and events != ("PreToolUse",):
            raise _config_error(
                source,
                index,
                "on",
                "PreToolUse cannot be mixed with event-channel names",
            )
        if events == ("PreToolUse",) and not raw_blocking:
            raise _config_error(
                source,
                index,
                "blocking",
                "PreToolUse is the veto point; a non-blocking hook would never run",
            )
        if raw_blocking and events != ("PreToolUse",):
            raise _config_error(
                source,
                index,
                "blocking",
                "the event channel is one-way; blocking is valid only for on = ['PreToolUse']",
            )
        parsed.append(
            HookSpec(
                events=events,
                command=command,
                timeout_seconds=timeout_seconds,
                blocking=raw_blocking,
                source=source,
            )
        )
    return tuple(parsed)


def _signal_group(proc: subprocess.Popen[bytes], signal_number: int) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal_number)
        elif signal_number == signal.SIGTERM:
            proc.terminate()
        else:
            proc.kill()
    except (OSError, ProcessLookupError, PermissionError):
        pass


def _stop_process_group(proc: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    _signal_group(proc, signal.SIGTERM)
    try:
        return proc.communicate(timeout=CLEANUP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_group(proc, signal.SIGKILL)
        try:
            return proc.communicate(timeout=CLEANUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            try:
                return proc.communicate(timeout=CLEANUP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                if proc.stdout is not None:
                    proc.stdout.close()
                if proc.stderr is not None:
                    proc.stderr.close()
                try:
                    proc.wait(timeout=CLEANUP_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
                return exc.output or b"", exc.stderr or b""


class HookRunner:
    def __init__(self, hooks: Sequence[HookSpec], *, cwd: Path) -> None:
        self._hooks = tuple(hooks)
        self._cwd = Path(cwd)

    def _run(
        self,
        hook: HookSpec,
        payload: Mapping[str, object],
    ) -> tuple[HookOutcome, str | None]:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            proc = subprocess.Popen(
                hook.command,
                shell=False,
                cwd=self._cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            outcome = HookOutcome(False, None, False, "", str(exc))
            mode = "missing executable" if isinstance(exc, FileNotFoundError) else "could not start"
            return outcome, mode

        try:
            raw_stdout, raw_stderr = proc.communicate(
                input=encoded,
                timeout=hook.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            raw_stdout, raw_stderr = _stop_process_group(proc)
            stdout = raw_stdout.decode("utf-8", errors="replace")
            stderr = raw_stderr.decode("utf-8", errors="replace")
            return HookOutcome(False, None, True, stdout, stderr), "timeout"

        try:
            stdout = raw_stdout.decode("utf-8", errors="strict")
            stderr = raw_stderr.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return HookOutcome(False, proc.returncode, False, "", ""), "unparseable output"
        outcome = HookOutcome(
            ok=proc.returncode == 0,
            exit_code=proc.returncode,
            timed_out=False,
            stdout=stdout,
            stderr=stderr,
        )
        return outcome, None

    def __call__(self, event: Event) -> None:
        event_name = type(event).__name__
        payload = {"type": event_name, **asdict(event)}
        for hook in self._hooks:
            if hook.blocking or event_name not in hook.events:
                continue
            outcome, problem = self._run(hook, payload)
            if problem is not None or not outcome.ok:
                _LOG.warning(
                    "observational hook failed for %s: %s",
                    event_name,
                    problem or f"exit code {outcome.exit_code}",
                )

    def pre_tool(self, tool_name: str, tool_call_id: str) -> str | None:
        payload = {
            "type": "PreToolUse",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
        }
        for hook in self._hooks:
            if not hook.blocking or hook.events != ("PreToolUse",):
                continue
            outcome, problem = self._run(hook, payload)
            if problem == "timeout":
                return "blocking hook timed out"
            if problem == "missing executable":
                return "blocking hook missing executable"
            if problem == "unparseable output":
                return "blocking hook produced unparseable output"
            if problem is not None:
                return f"blocking hook could not start: {outcome.stderr}"
            if outcome.exit_code != 0:
                return f"blocking hook exited non-zero ({outcome.exit_code})"
            denial = next(
                (
                    line[5:].strip()
                    for line in outcome.stdout.splitlines()
                    if line.startswith("deny:")
                ),
                None,
            )
            if denial is not None:
                return denial or "blocking hook denied the tool call"
        return None
