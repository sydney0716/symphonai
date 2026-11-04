"""The run_shell local tool: executes an allowlisted argv command.

Every invocation is gated by `PermissionPolicy.check_shell()` before
`subprocess.Popen` is ever called, always with `shell=False` and a timeout.
A denied command returns `ToolResult(ok=False, ...)` without ever touching
`subprocess`.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.environment import project_virtualenv
from symphonai_api.jobs import JobManager
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata, call_target
from symphonai_api.tools.shell_classify import classify
from symphonai_api.tools.shell_parse import commands_for_argv, parse_command_line

CANCEL_POLL_SECONDS = 0.05
CLEANUP_TIMEOUT_SECONDS = 1.0
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
BWRAP_EXEC = "bwrap"


def shell_executable() -> str:
    return "/bin/bash" if os.path.isfile("/bin/bash") else "/bin/sh"


def shell_environment(
    repo_root: str | Path | None = None, *, for_agent: bool = False,
) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not any(part in key.upper() for part in ("KEY", "SECRET", "TOKEN"))
    }
    virtualenv = None if repo_root is None else project_virtualenv(repo_root)
    if virtualenv is not None:
        environment["PATH"] = str(virtualenv / "bin") + os.pathsep + environment.get("PATH", os.defpath)
        environment["VIRTUAL_ENV"] = str(virtualenv)
        environment.pop("PYTHONHOME", None)
    if for_agent:
        environment.update({
            "PAGER": "cat",
            "GIT_PAGER": "cat",
            "GIT_EDITOR": "true",
            "EDITOR": "true",
            "VISUAL": "true",
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "never",
        })
    return environment


def _seatbelt_profile(policy: PermissionPolicy) -> str:
    from symphonai_api.tools.search import DEFAULT_SKIP_DIRECTORIES

    def quote_path(path: str) -> str:
        return '"' + path.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r") + '"'

    writable = {
        str(os.path.realpath(os.environ.get("TMPDIR", tempfile.gettempdir()))),
        str(Path("/private/tmp").resolve()),
        str(Path("/private/var/folders").resolve()),
    }
    writable.update(
        str(path.resolve())
        for path in (
            [policy.repo_root] if policy.sandbox_write_scope is None
            else policy.sandbox_write_scope
        )
    )
    rules = ["(version 1)", "(debug deny)", "(allow default)", "(deny file-write*)"]
    rules.extend(
        f"(allow file-write* (subpath {quote_path(path)}))"
        for path in sorted(writable)
    )
    rules.extend(("(allow file-write* (literal \"/dev/null\"))", "(allow file-write* (literal \"/dev/tty\"))"))
    if not policy.sandbox_network:
        rules.extend((
            "(deny network*)", "(allow network* (local unix))",
            '(allow network-bind (local ip "localhost:*"))',
            '(allow network-inbound (local ip "localhost:*"))',
            '(allow network-outbound (remote ip "localhost:*"))',
        ))
    def component_regex(pattern: str) -> str:
        import fnmatch

        pattern = pattern.rstrip("/").casefold()
        result = ""
        index = 0
        while index < len(pattern):
            char = pattern[index]
            if char == "*":
                result += "[^/]*"
            elif char == "?":
                result += "[^/]"
            elif char == "[":
                end = index + 1
                if end < len(pattern) and pattern[end] == "!":
                    end += 1
                if end < len(pattern) and pattern[end] == "]":
                    end += 1
                end = pattern.find("]", end)
                if end == -1:
                    result += r"\["
                else:
                    fragment = fnmatch.translate(pattern[index:end + 1])[4:-3]
                    if fragment.startswith("["):
                        negated = fragment.startswith("[^")
                        body = fragment[2 if negated else 1:-1]
                        result += "[" + ("^/" if negated else "") + body + body.upper() + "]"
                    else:
                        result += fragment
                    index = end
            elif char.isascii() and char.isalpha():
                result += f"[{char}{char.upper()}]"
            else:
                result += re.escape(char)
            index += 1
        return result

    def regex_filter(expression: str) -> str:
        return f"(regex {quote_path(expression)})"

    roots = {policy.repo_root.resolve(), *(
        path.resolve() for path in (policy.sandbox_write_scope or ())
    )}
    for root in sorted(roots):
        for name in (".git", ".symphonai"):
            path = root / name
            rules.append(f"(deny file-write* (subpath {quote_path(str(path))}))")
            if path.is_symlink():
                rules.append(f"(deny file-write* (subpath {quote_path(str(path.resolve()))}))")
        git = root / ".git"
        if git.is_file():
            pointer = git.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir: ") or not pointer[8:].strip():
                raise ValueError(f"invalid gitdir pointer: {git}")
            target = (root / pointer[8:].strip()).resolve()
            rules.append(f"(deny file-write* (subpath {quote_path(str(target))}))")
        prefix = "^" + re.escape(str(root).rstrip("/")) + "/([^/]+/)*"
        skipped = "|".join(re.escape(name) for name in sorted(DEFAULT_SKIP_DIRECTORIES))
        excluded = regex_filter(prefix + "(" + skipped + r"|[^/]*\.egg-info)(/|$)")
        for pattern in policy.forbidden_patterns:
            if pattern.rstrip("/").casefold() == ".git":
                continue  # Git metadata stays readable, but never writable.
            matched = regex_filter(prefix + component_regex(pattern) + "(/|$)")
            denied = f"(require-all {matched} (require-not {excluded}))"
            rules.append(f"(deny file-write* {denied})")
            if pattern not in policy.sandbox_readable:
                rules.append(f"(deny file-read* {denied})")
    rules.extend(f"(deny file-read* file-write* (subpath {quote_path(str(folder))}))"
                 for folder in policy.protected_paths)
    return "\n".join(rules)


def _mac_sandbox_denial(
    output: str, pid: int | None = None, *,
    policy: PermissionPolicy | None = None, cwd: Path | None = None,
) -> str | None:
    """Return a Seatbelt denial that explains a permission failure, if available."""
    if sys.platform != "darwin":
        return None
    lowered_output = output.casefold()
    if not any(
        marker in lowered_output
        for marker in ("operation not permitted", "eperm", "permission denied", "read-only file system")
    ):
        return None

    def read_denial(resource: str) -> str | None:
        if policy is None:
            return None
        from symphonai_api.permissions import _matching_forbidden_component
        from symphonai_api.tools.search import _default_skipped_directory

        path = Path(resource)
        if not path.is_absolute():
            path = (cwd or policy.repo_root) / path
        try:
            path = path.resolve()
        except (OSError, ValueError, RuntimeError):
            return None
        if not path.is_relative_to(policy.repo_root):
            return None
        parts = path.relative_to(policy.repo_root).parts
        if any(part.casefold() == ".git" for part in parts):
            return None
        directories = parts if path.is_dir() else parts[:-1]
        if any(_default_skipped_directory(part) for part in directories):
            return None
        patterns = tuple(
            pattern for pattern in policy.forbidden_patterns
            if pattern.rstrip("/").casefold() != ".git" and pattern not in policy.sandbox_readable
        )
        if any(_matching_forbidden_component(part, patterns) for part in parts):
            return f"read of {path} denied; the person can allow it with [sandbox] readable in user config"
        return None

    for line in output.splitlines():
        if not any(marker in line.casefold() for marker in ("operation not permitted", "permission denied", "eperm")):
            continue
        for candidate in line.split(":"):
            denial = read_denial(candidate.strip().strip("\"'"))
            if denial:
                return denial
    try:
        report = subprocess.run(
            [
                "/usr/bin/log", "show", "--last", "2m", "--style", "compact",
                "--predicate",
                'sender == "Sandbox" AND eventMessage CONTAINS[c] "deny(1)"',
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in report.stdout.splitlines():
        match = re.search(r"\(\d+\) deny\(1\) (file-read\S*|file-write\S*|network\S*)\s*(.*)$", line)
        if not match:
            continue
        operation, resource = match.groups()
        if operation.startswith("file-read"):
            denial = read_denial(resource.strip())
            if denial and (resource.strip() in output or (pid is not None and f"({pid})" in line)):
                return denial
            continue
        if operation.startswith("network"):
            resource = resource.strip()
            port = re.search(r":(\d+)(?:\s|$)", resource)
            record_pid = re.search(r"\((\d+)\)\s+deny\(1\)", line)
            same_process = pid is not None and record_pid is not None and int(record_pid[1]) == pid
            mentions_port = port is not None and re.search(
                rf"(?:\bport\s+|[:,]\s*){re.escape(port[1])}(?!\d)",
                output,
                re.IGNORECASE,
            )
            if same_process or mentions_port:
                return "network access"
            continue
        resource = resource.strip()
        if resource and not resource.startswith("/dev/") and resource in output:
            return f"write to {resource} outside the project"
    return None


def _terminate_process_group(proc: subprocess.Popen) -> None:
    try:
        if hasattr(os, "killpg"):
            pgid = os.getpgid(proc.pid)
            if pgid != os.getpgid(0):
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        else:
            proc.kill()
    except (OSError, ProcessLookupError, PermissionError):
        pass


@lru_cache(maxsize=1)
def _linux_sandbox_available() -> bool:
    if shutil.which(BWRAP_EXEC) is None:
        return False
    try:
        probe = subprocess.run(
            [
                BWRAP_EXEC, "--ro-bind", "/", "/", "--dev", "/dev",
                "--proc", "/proc", "/bin/true",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _sandbox_available() -> bool:
    if sys.platform == "linux":
        return _linux_sandbox_available()
    if sys.platform != "darwin" or not os.path.isfile(SANDBOX_EXEC):
        return False
    try:
        probe = subprocess.run(
            [SANDBOX_EXEC, "-p", "(version 1) (allow default)", "/usr/bin/true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _bwrap_command(policy: PermissionPolicy, argv: list[str], cwd: Path | None = None) -> list[str]:
    from symphonai_api.permissions import _matching_forbidden_component
    from symphonai_api.tools.search import _default_skipped_directory

    root = str(policy.repo_root.resolve())
    working_directory = root if cwd is None else str(cwd.resolve())
    temporary = os.path.realpath(os.environ.get("TMPDIR", tempfile.gettempdir()))
    command = [
        BWRAP_EXEC,
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
    ]
    writable = [root] if policy.sandbox_write_scope is None else [
        str(path.resolve()) for path in policy.sandbox_write_scope
    ]
    for path in writable:
        if path != root and not Path(path).exists():
            continue
        command.extend(("--bind", path, path))
    command.extend(("--bind", temporary, temporary, "--bind", "/tmp", "/tmp"))
    patterns = tuple(
        pattern for pattern in policy.forbidden_patterns
        if pattern.rstrip("/").casefold() != ".git"
    )
    unreadable = tuple(pattern for pattern in patterns if pattern not in policy.sandbox_readable)

    def walk_error(error: OSError) -> None:
        raise error

    roots = {Path(root), *(Path(path) for path in writable)}
    for directory_root in sorted(roots):
        for name in (".git", ".symphonai"):
            path = directory_root / name
            if path.exists():
                resolved = str(path.resolve())
                command.extend(("--ro-bind", resolved, resolved))
        git = directory_root / ".git"
        if git.is_file():
            pointer = git.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir: ") or not pointer[8:].strip():
                raise ValueError(f"invalid gitdir pointer: {git}")
            target = (directory_root / pointer[8:].strip()).resolve()
            if target.exists():
                command.extend(("--ro-bind", str(target), str(target)))
        for directory, directories, files in os.walk(directory_root, onerror=walk_error):
            for name in list(directories):
                path = Path(directory) / name
                if _default_skipped_directory(name):
                    directories.remove(name)
                elif _matching_forbidden_component(name, patterns):
                    resolved = str(path.resolve())
                    if _matching_forbidden_component(name, unreadable):
                        command.extend(("--tmpfs", resolved, "--remount-ro", resolved))
                        directories.remove(name)
                    else:
                        command.extend(("--ro-bind", resolved, resolved))
                elif path.is_symlink():
                    directories.remove(name)
            for name in files:
                path = Path(directory) / name
                if _matching_forbidden_component(name, patterns) and path.exists():
                    resolved = str(path.resolve())
                    source = "/dev/null" if _matching_forbidden_component(name, unreadable) else resolved
                    command.extend(("--ro-bind", source, resolved))
    from symphonai_api.paths import ensure_private_directory
    for folder in policy.protected_paths:
        # Mount before a sign-in can create the first token during this shell.
        ensure_private_directory(folder)
        command.extend(("--tmpfs", str(folder), "--remount-ro", str(folder)))
    if not policy.sandbox_network:
        # Each call gets its own loopback; only server and client in one command can connect.
        command.append("--unshare-net")
    command.extend(("--die-with-parent", "--chdir", working_directory, "--", *argv))
    return command


def sandbox_available() -> bool:
    """Whether this platform's sandbox can execute commands."""
    return _sandbox_available()


def sandbox_command(policy: PermissionPolicy, argv: list[str], cwd: Path | None = None) -> list[str]:
    """Wrap argv for the policy's platform sandbox, failing closed if required."""
    available = sandbox_available()
    if policy.shell_sandbox and not available:
        raise ValueError("sandbox requested but unavailable on this platform")
    if available and (policy.shell_sandbox or policy.mode == "allow"):
        if sys.platform == "linux":
            return _bwrap_command(policy, argv, cwd)
        return [SANDBOX_EXEC, "-p", _seatbelt_profile(policy), *argv]
    return argv


class RunShellTool(LocalTool):
    """Run an allowlisted command line or argv list and return its output."""

    def __init__(self, jobs: JobManager | None = None) -> None:
        self._jobs = jobs
        self._cwd: Path | None = None

    @property
    def name(self) -> str:
        return "run_shell"

    @property
    def description(self) -> str:
        return (
            "Run an allowlisted shell command line or argv list and return its output. "
            "Timeout defaults to 120 seconds and cannot exceed the policy limit."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "argv": {
                    "minItems": 1,
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Run one program without a shell, as a list of strings, "
                        "e.g. ['git', 'status']."
                    ),
                },
                "command": {
                    "minLength": 1,
                    "type": "string",
                    "description": (
                        "A shell command line where pipes, &&, ;, || and redirects work."
                    ),
                },
                "timeout_seconds": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Optional timeout in seconds; defaults to 120 and is capped by policy.",
                },
                "background": {
                    "type": "boolean",
                    "description": "Start the command as a background job.",
                },
            },
        }

    def metadata(self, arguments: dict) -> ToolMetadata:
        argv = arguments.get("argv")
        commands = (
            parse_command_line(arguments["command"])
            if "command" in arguments and isinstance(arguments["command"], str)
            else commands_for_argv(argv)
            if isinstance(argv, (list, tuple))
            else None
        )
        entries = None if commands is None else [
            classify(command.argv) if not command.writes else None
            for command in commands
        ]
        if not entries or any(entry is None for entry in entries):
            return ToolMetadata(
                effect=ToolEffect.DESTRUCTIVE,
                concurrency_safe=False,
                paths=None,
            )
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=all(entry.concurrency_safe for entry in entries if entry is not None),
            paths=None,
        )

    def validate(self, arguments: dict) -> str | None:
        if "argv" in arguments and "command" in arguments:
            return "pass either argv or command, not both"
        if "argv" not in arguments and "command" not in arguments:
            return "missing required argument: command or argv"
        return None

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        root = policy.repo_root.resolve()
        cwd = self._cwd or root
        reset_message = ""
        if not cwd.is_dir() or not cwd.is_relative_to(root):
            cwd = root
            reset_message = "Shell working directory reset to the project root."
        self._cwd = cwd
        command_line = tool_call.arguments.get("command")
        if isinstance(command_line, str):
            argv = [shell_executable(), "-c", command_line]
        else:
            argv = tool_call.arguments.get("argv")
        timeout_seconds = min(
            tool_call.arguments.get("timeout_seconds") or 120,
            policy.shell_timeout_seconds,
        )
        decision = policy.check_shell(argv, cwd=cwd)
        if not decision.allowed:
            error = decision.reason + (f"\n{reset_message}" if reset_message else "")
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=error)
        command = argv
        cwd_marker = f"\036SYMPHONAI_CWD_{os.urandom(12).hex()}_"
        if isinstance(command_line, str) and tool_call.arguments.get("background") is not True:
            command = [
                shell_executable(), "-c",
                f"{command_line}\n__symphonai_status=$?; printf '{cwd_marker}%s\\037' \"$PWD\"; exit \"$__symphonai_status\"",
            ]
        unwrapped = command
        try:
            command = sandbox_command(policy, command, cwd)
        except ValueError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        sandboxed = command is not unwrapped
        env = shell_environment(policy.repo_root, for_agent=True)
        if tool_call.arguments.get("background") is True:
            if self._jobs is None:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    ok=False,
                    error="background jobs are not available here",
                )
            try:
                job_id = self._jobs.start(
                    command, str(cwd), env,
                    label=call_target("run_shell", tool_call.arguments),
                )
            except RuntimeError as exc:
                return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=True,
                content=f"Started {job_id}. Read its output with job_output and stop it with job_stop.",
            )
        try:
            proc = subprocess.Popen(
                command,
                shell=False,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout_seconds
            while True:
                if cancel is not None and cancel.cancelled:
                    _terminate_process_group(proc)
                    try:
                        proc.communicate(timeout=CLEANUP_TIMEOUT_SECONDS)
                    except subprocess.TimeoutExpired:
                        pass
                    raise OperationCancelled
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate()
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _terminate_process_group(proc)
                    try:
                        proc.communicate(timeout=CLEANUP_TIMEOUT_SECONDS)
                    except subprocess.TimeoutExpired:
                        pass
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                delay = min(CANCEL_POLL_SECONDS, remaining)
                try:
                    stdout, stderr = proc.communicate(timeout=delay)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=f"error running command: {exc}")
        stdout = stdout or ""
        if isinstance(command_line, str):
            marker_at = stdout.rfind(cwd_marker)
            if marker_at >= 0 and stdout.endswith("\x1f"):
                captured = stdout[marker_at + len(cwd_marker):-1]
                stdout = stdout[:marker_at]
                ended = Path(captured).resolve()
                if ended.is_dir() and ended.is_relative_to(root):
                    self._cwd = ended
                else:
                    self._cwd = root
                    reset_message = "Shell working directory reset to the project root."
        if not self._cwd.is_dir() or not self._cwd.is_relative_to(root):
            self._cwd = root
            reset_message = "Shell working directory reset to the project root."
        output = stdout + (stderr or "")
        denial = _mac_sandbox_denial(output, proc.pid, policy=policy, cwd=cwd) if sandboxed and proc.returncode != 0 else None
        if self._cwd != root:
            output = f"(in {self._cwd.relative_to(root)})\n" + output
        limit = policy.shell_output_limit_chars
        if len(output) > limit:
            output = (
                output[:limit]
                + f"\n[output truncated: {len(output)} chars, over the {limit} char limit]"
            )
        if reset_message:
            output = (output + "\n" if output else "") + reset_message
        error = None
        if proc.returncode != 0:
            error = (
                f"blocked by the sandbox: {denial}" + (f"\n{output}" if output else "")
                if denial else f"exit code {proc.returncode}"
            )
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=proc.returncode == 0,
            content=output,
            error=error,
        )
