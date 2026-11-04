"""Configured event observers and blocking tool/Stop decision hooks."""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from symphonai_api.web import RedirectBlocker, valid_service_url
from symphonai_api.config import ConfigError, ResolvedConfig, Scope
from symphonai_api.events import Event
from symphonai_api.cancellation import CancellationToken
from symphonai_api.secrets import mask_secrets
from symphonai_api.trust import TrustList


DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_TIMEOUT_SECONDS = 30.0

_LOG = logging.getLogger(__name__)
_DECISION_POINTS = (("PreToolUse",), ("Stop",))


@dataclass(frozen=True)
class HookSpec:
    events: tuple[str, ...]
    command: tuple[str, ...] = ()
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    blocking: bool = False
    source: Path | None = None
    url: str | None = None
    prompt: str | None = None
    agents: tuple[str, ...] | None = None

    @property
    def label(self) -> str:
        return str(self.source) if self.source is not None else (
            f"command hook {self.command[0]}" if self.command else "URL hook" if self.url else "prompt hook"
        )


@dataclass(frozen=True)
class HookOutcome:
    ok: bool
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str


@dataclass(frozen=True)
class HookInput:
    arguments: dict
    context: str = ""
    updated_by: str | None = None


def _mask_payload(value):
    if isinstance(value, str):
        return mask_secrets(value)
    if isinstance(value, Mapping):
        return {mask_secrets(key) if isinstance(key, str) else key: _mask_payload(item)
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_mask_payload(item) for item in value]
    return value



def _valid_event_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            {"Event"} | {name for (name,) in _DECISION_POINTS}
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
    trust: TrustList | None = None,
) -> tuple[HookSpec, ...]:
    """Parse the resolved hook array and retain its winning provenance."""
    raw_hooks = config.get("hooks", [])
    origin = config.provenance.get("hooks")
    source = origin.source if origin is not None else None
    if (
        origin is not None
        and origin.scope in (Scope.PROJECT, Scope.PRIVATE)
        and not (trust is not None and trust.allows(repo_root, "hooks"))
    ):
        raise ConfigError(
            f"{source}: hooks: hooks were refused and are not read from configuration inside "
            "a repository; define them in ~/.symphonai/config.toml or grant "
            f"{str(Path(repo_root).resolve())!r} under [[trust.repositories]]"
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
            "url",
            "prompt",
            "timeout_seconds",
            "blocking",
            "agents",
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

        kinds = [name for name in ("command", "url", "prompt") if name in raw_hook]
        if len(kinds) != 1:
            raise _config_error(source, index, "hook", "must contain exactly one of command, url or prompt")
        kind = kinds[0]
        command = ()
        if kind == "command":
            raw_command = raw_hook["command"]
            if (not isinstance(raw_command, list) or not raw_command
                or not all(isinstance(item, str) and item for item in raw_command)):
                raise _config_error(source, index, "command", "must be a non-empty argv array of strings")
            command = tuple(raw_command)
        elif kind == "url" and not valid_service_url(raw_hook["url"]):
            raise _config_error(source, index, "url", "must be https, or http on localhost only")
        elif kind == "prompt":
            if not isinstance(raw_hook["prompt"], str) or not raw_hook["prompt"].strip():
                raise _config_error(source, index, "prompt", "must be non-blank text")
            if events not in _DECISION_POINTS:
                raise _config_error(source, index, "prompt", "allowed only on ['PreToolUse'] or ['Stop']")

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
        for (decision_point,) in _DECISION_POINTS:
            if decision_point in events and events != (decision_point,):
                raise _config_error(source, index, "on", f"{decision_point} cannot be mixed with event-channel names")
        if events in _DECISION_POINTS and not raw_blocking:
            raise _config_error(source, index, "blocking", f"{events[0]} must be blocking")
        if raw_blocking and events not in _DECISION_POINTS:
            raise _config_error(source, index, "blocking", "the event channel is one-way; blocking is valid only for PreToolUse or Stop")
        agents = raw_hook.get("agents")
        if agents is not None:
            if (events != ("Stop",) or not isinstance(agents, list) or not agents
                or not all(isinstance(name, str) and name.strip() for name in agents)):
                raise _config_error(source, index, "agents", "must be a non-empty list of agent names on a Stop hook")

        parsed.append(
            HookSpec(
                events=events,
                command=command,
                timeout_seconds=timeout_seconds,
                blocking=raw_blocking,
                source=source,
                url=raw_hook.get("url"), prompt=raw_hook.get("prompt"),
                agents=None if agents is None else tuple(agents),
            )
        )
    return tuple(parsed)






class HookRunner:
    def __init__(self, hooks: Sequence[HookSpec], *, cwd: Path) -> None:
        self._hooks = tuple(hooks)
        self._cwd = Path(cwd)

    def _run(
        self,
        hook: HookSpec,
        payload: Mapping[str, object],
        *, ask_model=None, cancel=None,
    ) -> tuple[HookOutcome, str | None]:
        if hook.url is not None or hook.prompt is not None:
            masked = json.dumps(_mask_payload(payload), separators=(",", ":"))
            completed = queue.Queue(maxsize=1)
            token = cancel.child(deadline_seconds=hook.timeout_seconds) if cancel is not None else CancellationToken(deadline_seconds=hook.timeout_seconds)

            def invoke():
                try:
                    if hook.url is not None:
                        if not valid_service_url(hook.url):
                            raise ValueError("invalid hook URL")

                        request = urllib.request.Request(hook.url, data=masked.encode("utf-8"),
                                                         headers={"Content-Type": "application/json"}, method="POST")
                        with urllib.request.build_opener(RedirectBlocker()).open(request, timeout=hook.timeout_seconds) as response:
                            if not 200 <= response.status < 300:
                                raise ValueError("non-2xx hook response")
                            text = response.read().decode("utf-8")
                    else:
                        if ask_model is None:
                            raise ValueError("no model callback")
                        reply_format = (
                            'Reply with one JSON object: {"decision": "allow" | "deny", "reason": "<one sentence>"}. '
                            'Add "updated_input": {…full new arguments…} only to change the call, and '
                            '"context": "<text>" only to add a note for the agent. Leave out any field you do not use.'
                            if payload["type"] == "PreToolUse" else
                            'Reply with one JSON object: {"decision": "continue" | "stop", "reason": "<text>"}.'
                        )
                        response = ask_model(
                            "Act as a decision hook. " + reply_format + "\n" + mask_secrets(hook.prompt) + "\nPayload:\n" + masked,
                            token,
                        )
                        if response.stop_reason == "refusal" or response.refusal_category:
                            raise ValueError("hook model refused")
                        text = response.message.text.strip()
                        if text.startswith("```json\n") and text.endswith("\n```"):
                            text = text[len("```json\n"):-len("\n```")]
                        if not isinstance(json.loads(text), dict):
                            raise ValueError("hook model must reply with a JSON object")
                    completed.put((text, None))
                except Exception as exc:
                    completed.put(("", "unparseable output" if isinstance(exc, (json.JSONDecodeError, UnicodeError)) else "hook request failed"))

            threading.Thread(target=invoke, name="symphonai-hook", daemon=True).start()
            try:
                text, problem = completed.get(timeout=hook.timeout_seconds)
                return HookOutcome(problem is None, 0 if problem is None else None, False, text, ""), problem
            except queue.Empty:
                token.cancel()
                return HookOutcome(False, None, True, "", ""), "timeout"
            finally:
                token.close()
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        from symphonai_api.process import run_process
        result = run_process(
            hook.command, self._cwd, input=encoded, timeout=hook.timeout_seconds,
            merge_stderr=False, output_limit=None,
        )
        if result.error is not None:
            outcome = HookOutcome(False, None, False, "", str(result.error))
            mode = "missing executable" if isinstance(result.error, FileNotFoundError) else "could not start"
            return outcome, mode
        if result.timed_out:
            return HookOutcome(False, None, True,
                               result.stdout.decode("utf-8", errors="replace"),
                               result.stderr.decode("utf-8", errors="replace")), "timeout"
        try:
            stdout = result.stdout.decode("utf-8", errors="strict")
            stderr = result.stderr.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return HookOutcome(False, result.exit_code, False, "", ""), "unparseable output"
        return HookOutcome(result.exit_code == 0, result.exit_code, False, stdout, stderr), None

    def __call__(self, event: Event) -> None:
        event_name = type(event).__name__
        payload = {"type": event_name, **asdict(event)}
        for hook in self._hooks:
            if hook.blocking or (event_name not in hook.events and "Event" not in hook.events):
                continue
            outcome, problem = self._run(hook, payload)
            if problem is not None or not outcome.ok:
                _LOG.warning(
                    "observational hook failed for %s: %s",
                    event_name,
                    problem or f"exit code {outcome.exit_code}",
                )

    def pre_tool(self, tool_name: str, tool_call_id: str, arguments: dict | None = None,
                 agent_name: str = "", *, ask_model=None, cancel=None) -> str | HookInput | None:
        current = dict(arguments or {})
        notes = []
        updated_by = None
        for hook in self._hooks:
            if not hook.blocking or hook.events != ("PreToolUse",):
                continue
            payload = {"type": "PreToolUse", "tool_name": tool_name, "tool_call_id": tool_call_id,
                       "arguments": current, "agent_name": agent_name}
            outcome, problem = self._run(hook, payload, ask_model=ask_model, cancel=cancel)
            if problem is not None or not outcome.ok:
                return f"{hook.label}: blocking hook failed ({problem or 'non-zero exit'})"
            denial = next((line[5:].strip() for line in outcome.stdout.splitlines() if line.startswith("deny:")), None)
            if denial is not None:
                return denial or f"{hook.label}: blocking hook denied the tool call"
            if not outcome.stdout.strip():
                continue
            try:
                text = outcome.stdout.strip()
                if text.startswith("```json\n") and text.endswith("\n```"):
                    text = text[len("```json\n"):-len("\n```")]
                try:
                    reply = json.loads(text)
                except ValueError:
                    if hook.command:
                        notes.append(outcome.stdout.strip())
                        continue
                    raise
                if not isinstance(reply, dict) and hook.command:
                    notes.append(outcome.stdout.strip())
                    continue
                if not isinstance(reply, dict) or reply.get("decision") not in ("allow", "deny"):
                    raise ValueError()
                if reply["decision"] == "deny":
                    return reply.get("reason") if isinstance(reply.get("reason"), str) and reply["reason"].strip() else f"{hook.label}: tool call denied"
                updated = reply.get("updated_input")
                if updated is not None:
                    if not isinstance(updated, dict):
                        raise ValueError()
                    if updated != current:
                        current = updated
                        updated_by = hook.label
                context = reply.get("context")
                if context is not None:
                    if isinstance(context, str):
                        if context:
                            notes.append(context)
                    else:
                        _LOG.warning("blocking hook %s ignored non-text context", hook.label)
            except (ValueError, TypeError):
                return f"{hook.label}: blocking hook produced unparseable output"
        return HookInput(current, "\n".join(notes), updated_by) if updated_by or notes else None

    def stop(self, *, agent_name: str, session_id: str | None, final_text: str,
             turns_used: int, stop_continues: int, ask_model=None, cancel=None) -> str | None:
        payload = {"type": "Stop", "agent_name": agent_name, "session_id": session_id,
                   "final_text": final_text, "turns_used": turns_used, "stop_continues": stop_continues}
        reasons = []
        for hook in self._hooks:
            if not hook.blocking or hook.events != ("Stop",) or (hook.agents is not None and agent_name not in hook.agents):
                continue
            try:
                outcome, problem = self._run(hook, payload, ask_model=ask_model, cancel=cancel)
            except Exception:
                continue
            if problem is not None or not outcome.ok:
                continue
            try:
                text = outcome.stdout.strip()
                if text.startswith("```json\n") and text.endswith("\n```"):
                    text = text[len("```json\n"):-len("\n```")]
                reply = json.loads(text)
                if (not isinstance(reply, dict) or reply.get("decision") != "continue"
                    or not isinstance(reply.get("reason"), str) or not reply["reason"].strip()):
                    continue
                reasons.append(reply["reason"])
            except (ValueError, TypeError):
                continue
        return "\n".join(reasons) if reasons else None
