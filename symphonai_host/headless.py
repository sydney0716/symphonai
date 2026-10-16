"""Run one SymphonAI turn without starting the browser host."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from symphonai_api.config import ConfigError
from symphonai_api.events import ModelFallback, RunFailed, RunFinished
from symphonai_api.extensions import load_extensions
from symphonai_api.lsp import LspManager
from symphonai_api.mcp import McpError
from symphonai_api.mcp_pool import McpPool
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.roles import LEADER_BASE_PROMPT, SUBAGENT_BASE_PROMPT
from symphonai_api.session import SessionError, SessionStore, default_sessions_root, load_run
from symphonai_host.broker import EventBroker
from symphonai_host.credentials import CredentialError, apply_to_environment, load
from symphonai_host.projects import build_project
from symphonai_host.protocol import encode_event
from symphonai_host.run import ModeSelectionError, ProviderSelectionError
from symphonai_host.server import _provider, _provider_for_start


class _UsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="symphonai", description="Run SymphonAI agents")
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser("run", help="run one task without the app")
    run.add_argument("task", nargs="?")
    run.add_argument("--repo-root", type=Path, default=Path.cwd())
    run.add_argument("--provider")
    run.add_argument("--model")
    run.add_argument("--effort")
    run.add_argument("--mode", choices=("ask", "plan", "allow"), default="ask")
    run.add_argument("--max-turns", type=int)
    run.add_argument("--resume")
    output = run.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true")
    output.add_argument("--stream-json", action="store_true")
    return parser


def _select_provider(arguments):  # noqa: ANN001
    if arguments.provider is not None:
        provider = _provider(arguments.provider, arguments.model)
        choice = {"name": arguments.provider}
    else:
        provider, choice = _provider_for_start(_provider)
        if provider is not None and arguments.model is not None:
            base_url = None if choice is None else choice.get("base_url")
            provider = _provider(provider.name, arguments.model, base_url)
    if provider is None:
        raise ProviderSelectionError("no configured provider; add an API key in Settings")
    choice = dict(choice or {"name": provider.name})
    choice["name"] = provider.name
    model = arguments.model if arguments.model is not None else choice.get("model")
    if arguments.model is not None:
        choice["model"] = arguments.model
    elif getattr(provider, "model", None) is not None:
        model = provider.model
    effort = arguments.effort if arguments.effort is not None else choice.get("effort")
    if effort is not None:
        choice["effort"] = effort
    return provider, model, effort, choice


def _wait_for_terminal(run, subscription, stream_json: bool) -> tuple[RunFinished | RunFailed, bool, str | None]:  # noqa: ANN001
    interrupted = False
    fallback_models = {}
    while True:
        try:
            event = subscription.get(timeout=0.1)
        except KeyboardInterrupt:
            interrupted = True
            run.stop()
            continue
        if event is None:
            continue
        if stream_json:
            encoded = encode_event(event)
            if encoded["type"] != "SessionStarted":
                print(json.dumps(encoded, ensure_ascii=False), flush=True)
        if isinstance(event, ModelFallback):
            fallback_models[event.agent_id] = event.to_model
        if isinstance(event, RunFailed) and event.agent_name == "leader":
            return event, interrupted, fallback_models.get(event.agent_id)
        if isinstance(event, RunFinished) and event.agent_name == "leader":
            return event, interrupted, fallback_models.get(event.agent_id)


def _saved_result(runtime, session_id: str, before_message_count: int) -> str:  # noqa: ANN001
    session = SessionStore.open(runtime.run.sessions_root, session_id)
    try:
        loaded = load_run(session)
    finally:
        session.close()
    messages = loaded.messages
    if len(messages) >= before_message_count:
        messages = messages[before_message_count:]
    return next(
        (message.text for message in reversed(messages) if message.role.value == "assistant"),
        "",
    )


def _files_changed(run, previous_keys: set[str]) -> list[str]:  # noqa: ANN001
    changes = run.changes()
    return sorted({
        path
        for turn in changes.get("turns", [])
        if turn.get("key") not in previous_keys
        for path in turn.get("paths", [])
    })


def _git_status(root: Path) -> dict[str, str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "-z"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    fields = result.stdout.split(b"\0")
    status: dict[str, str] = {}
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if not entry:
            continue
        state = os.fsdecode(entry[:2])
        path = os.fsdecode(entry[3:])
        status[path] = state
        if "R" in state or "C" in state:
            if index < len(fields):
                status[os.fsdecode(fields[index])] = state
                index += 1
    return status


def _execute(arguments, task: str) -> int:  # noqa: ANN001
    apply_to_environment(os.environ, load())
    provider, model, effort, choice = _select_provider(arguments)
    root = arguments.repo_root.expanduser().resolve()
    if not root.is_dir():
        raise _UsageError(f"project directory does not exist: {root}")
    if arguments.max_turns is not None and arguments.max_turns < 1:
        raise _UsageError("--max-turns must be greater than 0")

    status_before = _git_status(root)

    broker = EventBroker()
    runtime = None
    subscription = None
    original_threading_excepthook = threading.excepthook
    threading.excepthook = lambda _args: None
    try:
        runtime = build_project(
            root,
            broker,
            provider,
            publish_approval=lambda _approval: False,
            base_prompt=LEADER_BASE_PROMPT,
            subagent_base_prompt=SUBAGENT_BASE_PROMPT,
            max_turns=arguments.max_turns,
            sessions_root=default_sessions_root(),
            provider_factory=lambda name, selected_model, base_url: _provider(
                name, selected_model, base_url
            ),
            extensions_loader=load_extensions,
            pool_factory=McpPool,
            lsp_factory=LspManager,
            policy_factory=PermissionPolicy,
        )
        previous_keys: set[str] = set()
        before_message_count = 0
        if arguments.resume:
            runtime.run.open_session(arguments.resume)
            previous = runtime.run.changes()
            previous_keys = {turn["key"] for turn in previous.get("turns", [])}
            session = SessionStore.open(runtime.run.sessions_root, arguments.resume)
            try:
                metadata = session.read_meta()
                metadata["headless"] = True
                session.write_meta(metadata)
            finally:
                session.close()
            loaded_session = SessionStore.open(runtime.run.sessions_root, arguments.resume)
            try:
                before_message_count = len(load_run(loaded_session).messages)
            finally:
                loaded_session.close()
        runtime.run.select_provider(provider, model, effort, choice)
        runtime.run.select_mode(arguments.mode)

        subscription = broker.subscribe()
        started = time.perf_counter()
        host_run_id = runtime.run.start(task, _session_meta={"headless": True})
        session_id = arguments.resume or host_run_id
        if not (arguments.json or arguments.stream_json):
            print(f"Running session {session_id}", file=sys.stderr, flush=True)
        terminal, interrupted, fallback_model = _wait_for_terminal(
            runtime.run, subscription, arguments.stream_json,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        stats = runtime.run.conversation_stats() or {}
        usage = stats.get("usage") or {
            "input_tokens": 0, "output_tokens": 0, "calls": 0,
            "total_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
        }
        usage = dict(usage)
        cost = usage.pop("cost", None)
        result_text = _saved_result(runtime, session_id, before_message_count)
        status = "failed" if isinstance(terminal, RunFailed) else (
            "finished" if terminal.stopped_reason == "final_response" and not interrupted else "stopped"
        )
        changed_paths = set(_files_changed(runtime.run, previous_keys))
        if status_before is not None:
            status_after = _git_status(root) or {}
            changed_paths.update(
                path
                for path in status_before.keys() | status_after.keys()
                if status_before.get(path) != status_after.get(path)
            )
        error = terminal.error if isinstance(terminal, RunFailed) else None
        selection = runtime.run.selection() or {}
        result = {
            "session_id": session_id,
            "status": status,
            "stopped_reason": (
                "cancelled" if interrupted else
                terminal.stopped_reason if isinstance(terminal, RunFinished) else None
            ),
            "result": result_text,
            "error": error,
            "provider": selection.get("provider"),
            "model": fallback_model or selection.get("model"),
            "usage": usage,
            "cost": cost,
            "duration_ms": duration_ms,
            "files_changed": sorted(changed_paths),
        }
        if arguments.stream_json:
            print(json.dumps({"type": "result", **result}, ensure_ascii=False), flush=True)
        elif arguments.json:
            print(json.dumps(result, ensure_ascii=False), flush=True)
        elif result_text:
            print(result_text, flush=True)
        if isinstance(terminal, RunFailed):
            print(f"symphonai: {terminal.error}", file=sys.stderr, flush=True)
        if interrupted:
            return 130
        if isinstance(terminal, RunFailed):
            return 1
        return 0 if status == "finished" else 3
    finally:
        if subscription is not None:
            subscription.close()
        if runtime is not None:
            runtime.close()
        threading.excepthook = original_threading_excepthook
        broker.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except _UsageError as exc:
        print(f"symphonai: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        return int(exc.code or 0)
    if arguments.command is None:
        parser.print_help()
        return 0
    task = arguments.task
    try:
        if task is None or task == "-":
            if task is None and sys.stdin.isatty():
                raise _UsageError("a task is required when stdin is a terminal")
            task = sys.stdin.read()
        if not task.strip():
            raise _UsageError("task must not be empty")
        return _execute(arguments, task)
    except (ConfigError, CredentialError, McpError, ModeSelectionError, ProviderSelectionError,
            SessionError, OSError, ValueError) as exc:
        print(f"symphonai: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
