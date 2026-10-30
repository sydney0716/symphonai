"""Run one SymphonAI turn without starting the browser host."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from symphonai_api.agent_spec import IOContract, validate_output
from symphonai_api.config import ConfigError
from symphonai_api.events import ModelFallback, RunFailed, RunFinished
from symphonai_api.extensions import load_extensions
from symphonai_api.lsp import LspManager
from symphonai_api.mcp import McpError
from symphonai_api.mcp_pool import McpPool
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.paths import symphonai_home
from symphonai_api.role_prompts import load_role_additions
from symphonai_api.providers.base import ProviderError
from symphonai_api.roles import (
    EXPLORER_PROMPT,
    IMPLEMENTER_PROMPT, LEADER_BASE_PROMPT, PLANNER_PROMPT, REVIEWER_PROMPT,
    ROLE_GUIDANCE, SUBAGENT_BASE_PROMPT, WORKER_PROMPT,
)
from symphonai_api.session import SessionError, SessionStore, default_sessions_root, load_run
from symphonai_host.broker import EventBroker
from symphonai_host.credentials import CredentialError, apply_to_environment, load
from symphonai_host.__main__ import main as _host_main
from symphonai_host.loops import LoopStore
from symphonai_host.projects import build_project
from symphonai_host.protocol import encode_event
from symphonai_host.run import LoopChanged, ModeSelectionError, ProviderSelectionError, RunActiveError
from symphonai_host.server import PROVIDERS, _app_root, _provider, _provider_for_start
from symphonai_api.tools.shell import _sandbox_available


class _UsageError(ValueError):
    pass


REVIEW_PROMPT = """Review the uncommitted changes in this repository. Do not change any file.

1. Read them with `git status` and `git diff HEAD`, and read every untracked file that is not ignored.
2. Read the code each change depends on, and run the narrowest tests that exercise it.
3. Report findings, most severe first. For each: the file and line, what is wrong, and the input or situation that shows it. Leave out style preferences.
4. If nothing is worth fixing, say so in one line."""


class _HeadlessKeyError(ValueError):
    pass


class _StatusCapturingProvider:
    def __init__(self, provider) -> None:  # noqa: ANN001
        self._provider = provider
        self.key_error_status: int | None = None

    def __getattr__(self, name: str):  # noqa: ANN204
        return getattr(self._provider, name)

    def create_response(self, request, *, cancel=None):  # noqa: ANN001
        try:
            return self._provider.create_response(request, cancel=cancel)
        except ProviderError as exc:
            self._capture_status(exc)
            raise

    def stream_response(self, request, *, cancel=None):  # noqa: ANN001
        try:
            yield from self._provider.stream_response(request, cancel=cancel)
        except ProviderError as exc:
            self._capture_status(exc)
            raise

    def create_response_stream(self, request, *, cancel=None):  # noqa: ANN001
        try:
            yield from self._provider.create_response_stream(request, cancel=cancel)
        except ProviderError as exc:
            self._capture_status(exc)
            raise

    def _capture_status(self, error: ProviderError) -> None:
        status = getattr(error, "status", None)
        if self.key_error_status is None and status in (401, 403):
            self.key_error_status = status


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="symphonai", description="Run SymphonAI agents")
    commands = parser.add_subparsers(dest="command")
    app = commands.add_parser("app", help="open the SymphonAI app")
    app.add_argument("--repo-root", type=Path)
    app.add_argument("--no-open", action="store_true")
    doctor = commands.add_parser("doctor", help="check whether this machine is ready")
    doctor.add_argument("--repo-root", type=Path, default=Path.cwd())
    doctor.add_argument("--json", action="store_true")
    prompt = commands.add_parser("prompt", help="show an assembled role prompt")
    prompt.add_argument("name")
    prompt.add_argument("--stage", action="store_true")
    prompt.add_argument("--repo-root", type=Path, default=Path.cwd())
    run = commands.add_parser("run", help="run one task without the app")
    run.add_argument("task", nargs="?")
    run.add_argument("--repo-root", type=Path, default=Path.cwd())
    run.add_argument("--provider")
    run.add_argument("--model")
    run.add_argument("--effort")
    run.add_argument("--mode", choices=("ask", "plan", "allow"), default="ask")
    run.add_argument("--max-turns", type=int)
    run.add_argument("--resume")
    run.add_argument("--output-schema", type=Path)
    run.add_argument("--allowed-tools")
    run.add_argument("--disallowed-tools")
    output = run.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true")
    output.add_argument("--stream-json", action="store_true")

    loop = commands.add_parser("loop", help="run a spec loop without the app")
    loop.add_argument("goal", nargs="?")
    loop.add_argument("--resume")
    loop.add_argument("--repo-root", type=Path, default=Path.cwd())
    loop.add_argument("--provider")
    loop.add_argument("--model")
    loop.add_argument("--effort")
    loop.add_argument("--json", action="store_true")

    review = commands.add_parser("review", help="review changes without editing files")
    review.add_argument("--repo-root", type=Path, default=Path.cwd())
    review.add_argument("--provider")
    review.add_argument("--model")
    review.add_argument("--effort")
    review.add_argument("--max-turns", type=int)
    review.add_argument("--resume")
    review.add_argument("--output-schema", type=Path)
    review.add_argument("--allowed-tools")
    review.add_argument("--disallowed-tools")
    review.add_argument("--base")
    review.add_argument("--focus")
    review_output = review.add_mutually_exclusive_group()
    review_output.add_argument("--json", action="store_true")
    review_output.add_argument("--stream-json", action="store_true")
    review.set_defaults(mode="plan")
    return parser


def _run_app(arguments) -> int:  # noqa: ANN001
    def on_started(address: str) -> None:
        print(f"SymphonAI is running at {address}", file=sys.stderr, flush=True)
        print("Press Ctrl+C to stop.", file=sys.stderr, flush=True)
        if not arguments.no_open:
            webbrowser.open(address)

    host_arguments = [] if arguments.repo_root is None else ["--repo-root", str(arguments.repo_root)]
    try:
        _host_main(host_arguments, on_started=on_started)
    except KeyboardInterrupt:
        return 0
    return 0


def _run_doctor(arguments) -> int:  # noqa: ANN001
    checks = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    add(
        "Python",
        "ok" if sys.version_info >= (3, 11) else "fail",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    git = shutil.which("git")
    add("git", "ok" if git else "fail", "on PATH" if git else "not found on PATH")

    home = symphonai_home()
    credentials_path = home / "credentials.json"
    provider_keys = {variable for _, variable, _ in PROVIDERS}
    present_keys = {name for name in provider_keys if os.environ.get(name)}
    try:
        home_info = home.lstat()
    except FileNotFoundError:
        add("Home folder", "warn", f"{home} does not exist yet")
    except OSError:
        add("Home folder", "fail", f"{home} cannot be inspected")
    else:
        if not stat.S_ISDIR(home_info.st_mode):
            add("Home folder", "fail", f"{home} is not a directory")
        else:
            mode = stat.S_IMODE(home_info.st_mode)
            home_status = "warn" if mode & ~0o700 else "ok"
            home_details = [f"directory mode {mode:04o}"]
            if mode & ~0o700:
                home_details.append("directory mode is fixed on the next key save")
            try:
                credentials_info = credentials_path.lstat()
                if stat.S_ISREG(credentials_info.st_mode):
                    credentials_mode = stat.S_IMODE(credentials_info.st_mode)
                    if credentials_mode & ~0o600:
                        home_status = "warn"
                        home_details.append(
                            f"credentials file mode {credentials_mode:04o}; fixed on the next key save"
                        )
                    else:
                        home_details.append(f"credentials file mode {credentials_mode:04o}")
                    try:
                        stored = json.loads(credentials_path.read_text(encoding="utf-8"))
                    except (OSError, UnicodeError, ValueError):
                        stored = {}
                    keys = stored.get("keys", {}) if isinstance(stored, dict) else {}
                    if isinstance(keys, dict):
                        present_keys.update(
                            key for key, value in keys.items()
                            if key in provider_keys and isinstance(value, str) and value
                        )
                else:
                    home_status = "warn"
                    home_details.append("credentials path is not a regular file")
            except FileNotFoundError:
                pass
            except OSError:
                home_status = "warn"
                home_details.append("credentials file cannot be inspected")
            add("Home folder", home_status, "; ".join(home_details))
    add(
        "API keys",
        "ok" if present_keys else "fail",
        "available: " + ", ".join(sorted(present_keys)) if present_keys else "none found",
    )
    sandbox = _sandbox_available()
    add("Sandbox", "ok" if sandbox else "warn", "available" if sandbox else "unavailable (allow mode runs unsandboxed)")
    app_root = _app_root()
    add("App", "ok" if app_root.is_dir() else "fail", "installed" if app_root.is_dir() else f"missing: {app_root}")
    root = arguments.repo_root.expanduser()
    if not root.is_dir():
        add("Project", "warn", f"path does not exist: {root}")
    elif not git:
        add("Project", "warn", "cannot check repository because git is missing")
    else:
        result = subprocess.run(
            [git, "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        add("Project", "ok" if result.returncode == 0 else "warn", "git repository" if result.returncode == 0 else "not a git repository")

    if arguments.json:
        print(json.dumps({"checks": checks}, ensure_ascii=False))
    else:
        for check in checks:
            print(f"{check['status']} {check['name']}: {check['detail']}")
    return 1 if any(check["status"] == "fail" for check in checks) else 0


def _show_prompt(arguments) -> int:  # noqa: ANN001
    from symphonai_api.extensions import load_extensions

    root = arguments.repo_root.resolve()
    try:
        extensions = load_extensions(repo_root=root)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"symphonai: {exc}", file=sys.stderr)
        return 2
    builtins = {
        "worker": WORKER_PROMPT,
        "explorer": EXPLORER_PROMPT,
        "planner": PLANNER_PROMPT,
        "implementer": IMPLEMENTER_PROMPT,
        "reviewer": REVIEWER_PROMPT,
    }
    known = {"leader", *ROLE_GUIDANCE, *extensions.agents}
    name = arguments.name
    if name not in known:
        print(f"symphonai: unknown role {name!r}; known roles: {', '.join(sorted(known))}", file=sys.stderr)
        return 2
    if arguments.stage and name not in {"planner", "implementer", "reviewer"}:
        print("symphonai: --stage is only valid for planner, implementer, or reviewer", file=sys.stderr)
        return 2
    spec = extensions.agents.get(name)
    definition_prompt = spec.prompt if spec is not None else builtins.get(name, "")
    base = LEADER_BASE_PROMPT if name == "leader" or arguments.stage else SUBAGENT_BASE_PROMPT
    policy = PermissionPolicy(repo_root=root, allowed_write_scope=[root])
    addition, sources, warnings = load_role_additions(
        name, repo_root=root, policy=policy
    )
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    layers = [base]
    if definition_prompt:
        layers.append(definition_prompt)
    if addition:
        layers.append(addition)
    print("\n\n".join(layers))
    print(f"built-in: {'LEADER_BASE_PROMPT' if base == LEADER_BASE_PROMPT else 'SUBAGENT_BASE_PROMPT'}", file=sys.stderr)
    if definition_prompt:
        agent_path = _agent_definition_path(name, root, spec)
        print(f"{agent_path if agent_path is not None else 'built-in: ' + name}", file=sys.stderr)
    for source in sources:
        print(source, file=sys.stderr)
    return 0


def _agent_definition_path(name: str, root: Path, spec) -> str | None:  # noqa: ANN001
    if spec is None:
        return None
    candidates = (
        root / ".symphonai" / "agents" / f"{name}.toml",
        symphonai_home() / "agents" / f"{name}.toml",
    )
    return str(next((path for path in candidates if path.is_file()), candidates[0]))


def _review_task(arguments) -> str:  # noqa: ANN001
    prompt = REVIEW_PROMPT
    if arguments.base is not None:
        prompt = prompt.replace(
            "Review the uncommitted changes in this repository.",
            f"Review the changes between {arguments.base} and the working tree (git diff {arguments.base}).",
            1,
        )
    if arguments.focus is not None:
        prompt += f"\n\nFocus on: {arguments.focus}"
    return prompt


def _load_output_schema(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        IOContract(output_schema=schema)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise _UsageError(f"invalid output schema {path}: {exc}") from None
    return schema


def _tool_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = tuple(dict.fromkeys(name.strip() for name in value.split(",")))
    if not names or any(not name for name in names):
        raise _UsageError("tool lists must contain comma-separated names")
    return names


def _select_provider(arguments):  # noqa: ANN001
    if arguments.provider is not None:
        try:
            provider = _provider(arguments.provider, arguments.model)
        except ProviderSelectionError as exc:
            if str(exc) == f"{arguments.provider} has no API key":
                env_var = next((variable for name, variable, _ in PROVIDERS if name == arguments.provider), None)
                if env_var is not None:
                    raise _HeadlessKeyError(
                        f"no API key found; set {env_var} in the environment"
                    ) from None
            raise
        choice = {"name": arguments.provider}
    else:
        provider, choice = _provider_for_start(_provider)
        if provider is not None and arguments.model is not None:
            base_url = None if choice is None else choice.get("base_url")
            provider = _provider(provider.name, arguments.model, base_url)
    if provider is None:
        variables = ", ".join(dict.fromkeys(variable for _, variable, _ in PROVIDERS))
        raise _HeadlessKeyError(
            f"no API key found; set one of {variables} in the environment"
        )
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
    return _StatusCapturingProvider(provider), model, effort, choice


def _rejected_key_message(provider_name: str, status: int) -> str:
    env_var = next(
        (variable for name, variable, _ in PROVIDERS if name == provider_name),
        "the configured environment variable",
    )
    return f"{provider_name} rejected the API key in {env_var} (HTTP {status})"


def _wait_for_terminal(
    run, subscription, stream_json: bool, provider=None, provider_name: str | None = None,  # noqa: ANN001
) -> tuple[RunFinished | RunFailed, bool, str | None]:
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
            if isinstance(event, RunFailed) and provider is not None and provider.key_error_status in (401, 403):
                encoded["error"] = _rejected_key_message(provider_name or provider.name, provider.key_error_status)
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
    output_schema = _load_output_schema(arguments.output_schema)
    allowed_tools = _tool_list(arguments.allowed_tools)
    disallowed_tools = _tool_list(arguments.disallowed_tools) or ()
    if (arguments.command == "review"
            and (allowed_tools is None or "dispatch_subagent" not in allowed_tools)):
        disallowed_tools = (*disallowed_tools, "dispatch_subagent")
    if output_schema is not None:
        task += "\n\nReply with only a JSON value matching this schema:\n" + json.dumps(
            output_schema, ensure_ascii=False,
        )
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
            None,
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
        valid_tools = runtime.run.available_tool_names()
        requested_tools = set(allowed_tools or ()) | set(disallowed_tools)
        unknown_tools = sorted(requested_tools - valid_tools)
        if unknown_tools:
            raise _UsageError(
                f"unknown tool name(s): {', '.join(unknown_tools)}; valid tools: "
                + ", ".join(sorted(valid_tools))
            )
        if allowed_tools is not None or disallowed_tools:
            selected_tools = (set(allowed_tools) if allowed_tools is not None else valid_tools)
            selected_tools.difference_update(disallowed_tools)
            runtime.run.restrict_tool_names(selected_tools)
        provider, model, effort, choice = _select_provider(arguments)
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
            runtime.run, subscription, arguments.stream_json, provider, choice.get("name", provider.name),
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
        if isinstance(terminal, RunFinished) and terminal.stopped_reason == "refused":
            session = SessionStore.open(runtime.run.sessions_root, session_id)
            try:
                from symphonai_api.session import read_records
                records, _ = read_records(session.directory / "run.jsonl")
                category = next((record.get("data", {}).get("refusal_category")
                                 for record in reversed(records)
                                 if record.get("type") == "run_finished"), None)
            finally:
                session.close()
            error = f"refused: {category}" if category else "refused"
            result_text = ""
        output_value = None
        if output_schema is not None and status == "finished":
            output_value, output_error = validate_output(
                IOContract(output_schema=output_schema), result_text,
            )
            if output_error is not None:
                status = "failed"
                error = f"final answer did not match the output schema: {output_error}"
        key_error_status = provider.key_error_status
        if error is not None and key_error_status in (401, 403):
            selected_name = choice.get("name", provider.name)
            error = _rejected_key_message(selected_name, key_error_status)
            status = "failed"
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
        if output_schema is not None and error is None:
            result["output"] = output_value
        if arguments.stream_json:
            print(json.dumps({"type": "result", **result}, ensure_ascii=False), flush=True)
        elif arguments.json:
            print(json.dumps(result, ensure_ascii=False), flush=True)
        elif result_text:
            print(result_text, flush=True)
        if isinstance(terminal, RunFailed):
            print(f"symphonai: {error}", file=sys.stderr, flush=True)
        if interrupted:
            return 130
        if isinstance(terminal, RunFailed):
            return 2 if key_error_status in (401, 403) else 1
        if status == "failed":
            return 1
        return 0 if status == "finished" else 3
    finally:
        if subscription is not None:
            subscription.close()
        if runtime is not None:
            runtime.close()
        threading.excepthook = original_threading_excepthook
        broker.close()


def _print_loop_result(loop: dict, json_output: bool, reason: str | None = None) -> None:
    result = {
        "loop_id": loop["id"],
        "state": loop.get("state", "unknown"),
        "reason": loop.get("reason", "") if reason is None else reason,
        "branch": loop.get("branch"),
        "cards": [
            {
                "id": card.get("id"),
                "title": card.get("title", ""),
                "state": card.get("state", "unknown"),
                "commit": card.get("commit"),
            }
            for card in loop.get("cards", [])
        ],
        "summary": loop.get("summary", {}),
        "retries": loop.get("retries", []),
        "usage": loop.get("usage", {"cards": {}, "total": {
            "input_tokens": 0,
            "output_tokens": 0,
            "calls": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }}),
    }
    if json_output:
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    print(f"Branch: {result['branch'] or ''}")
    for retry in result["retries"]:
        stage = retry.get("stage", "stage")
        card = retry.get("card")
        label = f"{stage} card {card}" if card else stage
        print(f"Retrying {label}: {retry.get('reason', '')}")
    for card in result["cards"]:
        print(f"{card['id']} · {card['state']} · {card['commit'] or ''}")
    for validation in result["summary"].get("validation", []):
        print(f"{validation.get('command', '')} · {validation.get('exit')}")
    usage = result["usage"].get("total", {})
    print(
        f"Tokens: {usage.get('input_tokens', 0):,} in · "
        f"{usage.get('output_tokens', 0):,} out · {usage.get('calls', 0):,} calls"
    )
    if result["reason"]:
        print(f"Reason: {result['reason']}")


def _loop_retry_target(reason: str) -> tuple[str | None, str | None]:
    """Classify only loop failures that can be retried automatically."""
    if reason.startswith(("planner failed:", "planner stopped:", "plan.json is missing", "plan.json is invalid:")):
        return "planner", None
    card_id_pattern = r"([a-h](?:F2?)?)"
    match = re.match(rf"card {card_id_pattern} could not (start|finish): (.*)", reason)
    if match and "validation still failing after" not in match.group(3):
        return "implementer", match.group(1)
    match = re.match(rf"card {card_id_pattern} review failed:", reason)
    no_verdict = re.match(rf"card {card_id_pattern} review gave no verdict", reason)
    if match or no_verdict:
        return "reviewer", (match or no_verdict).group(1)
    if reason == "review gave no verdict":
        return "reviewer", None
    return None, None


def _run_loop(arguments, goal: str | None) -> int:  # noqa: ANN001
    apply_to_environment(os.environ, load())
    root = arguments.repo_root.expanduser().resolve()
    if not root.is_dir():
        raise _UsageError(f"project directory does not exist: {root}")

    broker = EventBroker()
    runtime = None
    subscription = None
    original_threading_excepthook = threading.excepthook
    threading.excepthook = lambda _args: None
    loop = None
    approval_error = None
    retries = []
    retried_stages = set()
    try:
        runtime = build_project(
            root,
            broker,
            None,
            publish_approval=lambda _approval: False,
            base_prompt=LEADER_BASE_PROMPT,
            subagent_base_prompt=SUBAGENT_BASE_PROMPT,
            sessions_root=default_sessions_root(),
            provider_factory=lambda name, selected_model, base_url: _provider(
                name, selected_model, base_url
            ),
            extensions_loader=load_extensions,
            pool_factory=McpPool,
            lsp_factory=LspManager,
            policy_factory=PermissionPolicy,
        )
        provider, model, effort, choice = _select_provider(arguments)
        runtime.run.select_provider(provider, model, effort, choice)
        subscription = broker.subscribe()
        try:
            if arguments.resume is not None:
                loop = runtime.run.resume_loop(
                    arguments.resume, runner_pid=os.getpid(), runner_lock=True,
                )
            else:
                loop = runtime.run.start_loop(
                    goal, runner_pid=os.getpid(), runner_lock=True,
                )
        except KeyError:
            print(f"loop {arguments.resume} is unknown", file=sys.stderr, flush=True)
            return 1
        except RunActiveError as exc:
            print(f"loop {arguments.resume} is {exc}; cannot resume", file=sys.stderr, flush=True)
            return 1
        store = LoopStore(root)
        approved = False
        last_reported_state = None
        terminal_states = ("awaiting_commit", "needs_attention", "cancelled", "done")
        while True:
            try:
                event = subscription.get(timeout=0.1)
            except KeyboardInterrupt:
                runtime.run.cancel_loop(loop["id"])
                print(f"loop {loop['id']} cancelled", file=sys.stderr, flush=True)
                return 130
            if event is not None and (
                not isinstance(event, LoopChanged) or event.loop_id != loop["id"]
            ):
                continue
            current = store.load(loop["id"])
            if isinstance(event, LoopChanged):
                if event.change == "card":
                    card = next(
                        (item for item in current.get("cards", []) if item.get("id") == event.card_id),
                        None,
                    )
                    if card is not None:
                        print(f"card {event.card_id} {card.get('state', 'unknown')}", file=sys.stderr, flush=True)
                elif event.state != last_reported_state:
                    print(f"loop {loop['id']} {event.state}", file=sys.stderr, flush=True)
                    last_reported_state = event.state
            else:
                if current.get("state") != last_reported_state:
                    print(f"loop {loop['id']} {current.get('state', 'unknown')}", file=sys.stderr, flush=True)
                    last_reported_state = current.get("state")
            if current.get("state") == "awaiting_plan" and not approved:
                approved = True
                try:
                    runtime.run.approve_loop_plan(loop["id"])
                except (RunActiveError, ValueError) as exc:
                    approval_error = str(exc)
                    loop = current
                    break
            elif current.get("state") == "needs_attention":
                reason = str(current.get("reason", ""))
                stage, card_id = _loop_retry_target(reason)
                key = ("planner", None) if stage == "planner" else ("card", card_id)
                if stage is not None and key not in retried_stages:
                    retried_stages.add(key)
                    entry = {"stage": stage, "card": card_id, "reason": reason}
                    retries.append(entry)
                    label = f"{stage} card {card_id}" if card_id else stage
                    print(f"retrying {label}: {reason}", file=sys.stderr, flush=True)
                    if stage == "planner":
                        (store.path(loop["id"], "plan.json")).unlink(missing_ok=True)
                        specs_folder = store.path(loop["id"], "specs")
                        for path in specs_folder.iterdir():
                            if path.is_dir():
                                shutil.rmtree(path)
                            else:
                                path.unlink()
                        store.set_state(current, "planning", "")
                        try:
                            runtime.run._start_loop_planner(current, "state")
                        except Exception as exc:
                            store.set_state(current, "needs_attention", f"planner failed: {exc}")
                    else:
                        runtime.run.resume_loop(loop["id"])
                    continue
                loop = current
                break
            elif (
                current.get("state") in terminal_states
                and (event is None or event.state == current.get("state"))
            ):
                loop = current
                break
        loop = store.load(loop["id"])
        loop["retries"] = retries
        loop["usage"] = runtime.run.loop_usage(loop)
        _print_loop_result(loop, arguments.json, approval_error)
        return 1 if approval_error or loop.get("state") in ("needs_attention", "cancelled") else 0
    except KeyboardInterrupt:
        if runtime is not None and loop is not None:
            try:
                runtime.run.cancel_loop(loop["id"])
            except (KeyError, OSError, RunActiveError, ValueError):
                pass
            print(f"loop {loop['id']} cancelled", file=sys.stderr, flush=True)
        return 130
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
    if arguments.command == "doctor":
        return _run_doctor(arguments)
    if arguments.command == "prompt":
        return _show_prompt(arguments)
    if arguments.command == "app":
        return _run_app(arguments)
    if arguments.command == "loop":
        try:
            if arguments.resume is not None:
                if arguments.goal is not None:
                    raise _UsageError("--resume cannot be combined with a goal")
                return _run_loop(arguments, None)
            goal = arguments.goal
            if goal is None or goal == "-":
                if goal is None and sys.stdin.isatty():
                    raise _UsageError("a goal is required when stdin is a terminal")
                goal = sys.stdin.read()
            if not goal.strip():
                raise _UsageError("goal must not be empty")
            return _run_loop(arguments, goal)
        except _HeadlessKeyError as exc:
            print(f"symphonai: {exc}", file=sys.stderr, flush=True)
            return 2
        except (ConfigError, CredentialError, McpError, ModeSelectionError,
                ProviderSelectionError, SessionError, OSError, ValueError) as exc:
            print(f"symphonai: {exc}", file=sys.stderr, flush=True)
            return 2
    review = arguments.command == "review"
    task = _review_task(arguments) if review else arguments.task
    try:
        if not review and (task is None or task == "-"):
            if task is None and sys.stdin.isatty():
                raise _UsageError("a task is required when stdin is a terminal")
            task = sys.stdin.read()
        if not review and not task.strip():
            raise _UsageError("task must not be empty")
        return _execute(arguments, task)
    except _HeadlessKeyError as exc:
        message = str(exc)
        if arguments.json:
            print(json.dumps({"status": "failed", "error": message}, ensure_ascii=False), flush=True)
        elif arguments.stream_json:
            print(json.dumps({"type": "result", "status": "failed", "error": message}, ensure_ascii=False), flush=True)
        print(f"symphonai: {message}", file=sys.stderr)
        return 2
    except (ConfigError, CredentialError, McpError, ModeSelectionError, ProviderSelectionError,
            SessionError, OSError, ValueError) as exc:
        print(f"symphonai: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
