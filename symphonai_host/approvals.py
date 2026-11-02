"""Synchronous approval callbacks backed by pending HTTP decisions."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from symphonai_api.identity import new_id
from symphonai_api.permissions import (
    DenialReason,
    PermissionDecision,
    ToolApprovalRequest,
    shell_grant_prefix,
)
from symphonai_api.paths import ensure_private_directory, write_atomic


DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class PendingApproval:
    approval_id: str
    operation: str
    target: str
    details: str
    tool_call_id: str = ""
    remember: str = ""
    session_id: str | None = None
    kind: str = "approval"
    options: tuple[str, ...] = ()
    diff: str = ""


@dataclass
class _Pending:
    approval: PendingApproval
    event: threading.Event
    decision: PermissionDecision | None = None
    grant_prefixes: tuple[tuple[str, ...], ...] = ()
    grant_hosts: tuple[str, ...] = ()
    answer: str = ""


class InvalidApprovalAnswer(ValueError):
    """An interactive answer does not match its pending request."""


class PermissionGrantError(OSError):
    """A project permission grant could not be read or stored privately."""


def load_project_permission_grants(
    path: Path,
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Read the validated, versioned project grant file."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return (), ()
    except OSError as exc:
        raise PermissionGrantError(f"could not read project permission grants at {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "shell", "fetch"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or not isinstance(payload["shell"], list)
            or not isinstance(payload["fetch"], list)
            or any(
                not isinstance(prefix, list)
                or not prefix
                or any(not isinstance(arg, str) or not arg.strip() for arg in prefix)
                for prefix in payload["shell"]
            )
            or any(not isinstance(host, str) or not host.strip() for host in payload["fetch"])
        ):
            raise ValueError("invalid schema")
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PermissionGrantError(f"invalid project permission grants at {path}: {exc}") from exc
    return (
        tuple(tuple(prefix) for prefix in payload["shell"]),
        tuple(dict.fromkeys(host.casefold().rstrip(".") for host in payload["fetch"])),
    )


def save_project_permission_grant(path: Path, kind: str, grant: object) -> None:
    """Atomically add grants to a private, validated project grant file."""
    if kind not in {"shell", "fetch"}:
        raise PermissionGrantError("unsupported project permission grant")
    path = Path(path)
    shell, fetch = load_project_permission_grants(path)
    if kind == "shell":
        shell = tuple(dict.fromkeys((*shell, *(tuple(prefix) for prefix in grant))))
    else:
        fetch = tuple(dict.fromkeys((*fetch, *(str(host).casefold().rstrip(".") for host in grant))))
    write_project_permission_grants(path, shell, fetch)


def remove_project_permission_grant(path: Path, kind: str, value: object) -> bool:
    """Remove one saved project grant without accepting malformed grant files."""
    if kind == "shell" and isinstance(value, (list, tuple)) and all(isinstance(arg, str) for arg in value):
        shell, fetch = load_project_permission_grants(path)
        updated = tuple(prefix for prefix in shell if prefix != tuple(value))
        if updated == shell:
            return False
        write_project_permission_grants(path, updated, fetch)
        return True
    if kind == "fetch" and isinstance(value, str):
        shell, fetch = load_project_permission_grants(path)
        normalized = value.casefold().rstrip(".")
        updated = tuple(host for host in fetch if host != normalized)
        if updated == fetch:
            return False
        write_project_permission_grants(path, shell, updated)
        return True
    raise PermissionGrantError("permission removal requires a shell prefix or fetch host")


def write_project_permission_grants(
    path: Path,
    shell: tuple[tuple[str, ...], ...],
    fetch: tuple[str, ...],
) -> None:
    """Atomically write the versioned project permission file."""
    path = Path(path)
    payload = {
        "schema_version": 1,
        "shell": [list(prefix) for prefix in shell],
        "fetch": list(fetch),
    }
    try:
        ensure_private_directory(path.parent)
        write_atomic(
            path, json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            mode=0o600, fsync=True,
        )
    except OSError as exc:
        raise PermissionGrantError(f"could not write project permission grants at {path}: {exc}") from exc


class ApprovalBroker:
    """Turns a synchronous ApprovalCallback into an HTTP round trip."""

    def __init__(
        self,
        publish: Callable[[PendingApproval], bool],
        *,
        timeout: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        session_id: str | None = None,
        shell_grants: tuple[tuple[str, ...], ...] = (),
        fetch_grants: tuple[str, ...] = (),
        review: Callable[[ToolApprovalRequest], PermissionDecision | None] | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("approval timeout must be greater than 0")
        self._publish = publish
        self._review = review
        self._timeout = timeout
        self._session_id = session_id
        self._pending: dict[str, _Pending] = {}
        self._granted_shell_prefixes: set[tuple[str, ...]] = set()
        self._project_shell_prefixes = set(shell_grants)
        self._granted_fetch_hosts: set[str] = set()
        self._project_fetch_hosts = {host.casefold().rstrip(".") for host in fetch_grants}
        self._lock = threading.Lock()

    def callback(self, request: ToolApprovalRequest) -> PermissionDecision:
        commands = request.commands
        if commands is None:
            commands = (request.command,) if request.command else ()
        command_writes = request.command_writes
        grant_prefixes = (
            tuple(shell_grant_prefix(command) for command in commands)
            if request.operation == "run_shell"
            else ()
        )
        try:
            fetch_host = (urlsplit(request.target).hostname or "").casefold().rstrip(".")
        except (TypeError, ValueError):
            fetch_host = ""
        with self._lock:
            if request.operation == "run_shell" and commands and not any(command_writes) and all(
                any(
                    command[:len(prefix)] == prefix
                    for prefix in self._granted_shell_prefixes | self._project_shell_prefixes
                )
                for command in commands
            ):
                return PermissionDecision.allow()
            if request.operation == "web_fetch" and fetch_host and fetch_host in (
                self._granted_fetch_hosts | self._project_fetch_hosts
            ):
                return PermissionDecision.allow()
        if self._review is not None:
            try:
                decision = self._review(request)
            except Exception:
                decision = None
            if decision is not None:
                return decision
        distinct_prefixes = tuple(dict.fromkeys(grant_prefixes))
        remember = (
            ", ".join(" ".join(prefix) for prefix in distinct_prefixes)
            if commands and grant_prefixes and all(grant_prefixes)
            else ""
        )
        grant_hosts = (fetch_host,) if request.operation == "web_fetch" and fetch_host else ()
        if grant_hosts:
            remember = fetch_host
        approval = PendingApproval(
            approval_id=new_id("appr"),
            operation=request.operation,
            target=request.target,
            details=request.details,
            diff=request.diff,
            tool_call_id=request.tool_call_id,
            remember=remember,
            session_id=self._session_id,
        )
        decision, _ = self.request_interaction(
            approval, grant_prefixes=grant_prefixes, grant_hosts=grant_hosts,
        )
        return decision

    def request_interaction(
        self,
        approval: PendingApproval,
        *,
        grant_prefixes: tuple[tuple[str, ...], ...] = (),
        grant_hosts: tuple[str, ...] = (),
    ) -> tuple[PermissionDecision, str]:
        pending = _Pending(
            approval, threading.Event(), grant_prefixes=grant_prefixes,
            grant_hosts=grant_hosts,
        )
        with self._lock:
            self._pending[approval.approval_id] = pending
        try:
            try:
                published = self._publish(approval)
            except Exception:
                published = False
            if not published:
                return PermissionDecision.deny(
                    "no client is connected to answer approvals",
                    denial=DenialReason.NO_APPROVAL_CALLBACK,
                ), ""
            if not pending.event.wait(self._timeout):
                with self._lock:
                    if pending.decision is None:
                        pending.decision = PermissionDecision.deny(
                            f"no approval decision arrived within {self._timeout:g}s",
                            denial=DenialReason.APPROVAL_FAILED,
                        )
                return pending.decision, pending.answer
            return pending.decision or PermissionDecision.deny(
                "approval was resolved without a decision", denial=DenialReason.APPROVAL_FAILED
            ), pending.answer
        except Exception:
            return PermissionDecision.deny(
                "approval callback failed", denial=DenialReason.APPROVAL_FAILED
            ), ""
        finally:
            with self._lock:
                if self._pending.get(approval.approval_id) is pending:
                    del self._pending[approval.approval_id]

    def resolve(
        self,
        approval_id: str,
        *,
        allowed: bool,
        reason: str,
        remember: bool = False,
        answer: str = "",
    ) -> bool:
        with self._lock:
            pending = self._pending.get(approval_id)
            if pending is None or pending.decision is not None:
                return False
            if allowed and pending.approval.kind in {"question", "plan"}:
                if not isinstance(answer, str) or not answer.strip():
                    raise InvalidApprovalAnswer("answer must be a non-blank string")
                if pending.approval.kind == "plan" and answer not in pending.approval.options:
                    raise InvalidApprovalAnswer("answer must be one of the offered plan modes")
            pending.decision = (
                PermissionDecision.allow()
                if allowed
                else PermissionDecision.deny(
                    reason or "approval denied by user", denial=DenialReason.DENIED_BY_USER
                )
            )
            pending.answer = answer if allowed else ""
            if allowed and remember and pending.approval.remember:
                if pending.approval.operation == "run_shell":
                    self._granted_shell_prefixes.update(pending.grant_prefixes)
                elif pending.approval.operation == "web_fetch":
                    self._granted_fetch_hosts.update(pending.grant_hosts)
            pending.event.set()
            return True

    def ask_user(
        self, question: str, options: tuple[str, ...], tool_call_id: str = "",
    ) -> tuple[PermissionDecision, str]:
        approval = PendingApproval(
            approval_id=new_id("question"),
            operation="ask_user",
            target=question,
            details="",
            tool_call_id=tool_call_id,
            session_id=self._session_id,
            kind="question",
            options=options,
        )
        return self.request_interaction(approval)

    def clear_grants(self) -> None:
        with self._lock:
            self._granted_shell_prefixes.clear()
            self._granted_fetch_hosts.clear()


    def pending_grant(
        self, approval_id: str,
    ) -> tuple[PendingApproval, tuple[tuple[str, ...], ...], tuple[str, ...]] | None:
        with self._lock:
            pending = self._pending.get(approval_id)
            if pending is None or pending.decision is not None:
                return None
            return pending.approval, pending.grant_prefixes, pending.grant_hosts

    def add_project_grant(
        self,
        operation: str,
        *,
        shell_prefixes: tuple[tuple[str, ...], ...] = (),
        fetch_hosts: tuple[str, ...] = (),
    ) -> None:
        with self._lock:
            if operation == "run_shell":
                self._project_shell_prefixes.update(shell_prefixes)
            elif operation == "web_fetch":
                self._project_fetch_hosts.update(
                    host.casefold().rstrip(".") for host in fetch_hosts
                )

    def set_project_grants(
        self,
        shell_prefixes: tuple[tuple[str, ...], ...],
        fetch_hosts: tuple[str, ...],
    ) -> None:
        with self._lock:
            self._project_shell_prefixes = set(shell_prefixes)
            self._project_fetch_hosts = {
                host.casefold().rstrip(".") for host in fetch_hosts
            }

    def pending(self) -> tuple[PendingApproval, ...]:
        """Every approval still waiting for a decision, oldest first."""
        with self._lock:
            return tuple(
                pending.approval
                for pending in self._pending.values()
                if pending.decision is None
            )

    def cancel_all(self, *, reason: str) -> int:
        with self._lock:
            pending = tuple(self._pending.values())
            for item in pending:
                if item.decision is None:
                    item.decision = PermissionDecision.deny(
                        f"run stopped while waiting for approval: {reason}",
                        denial=DenialReason.APPROVAL_CANCELLED,
                    )
                    item.event.set()
            return len(pending)
