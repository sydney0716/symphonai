"""Synchronous approval callbacks backed by pending HTTP decisions."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from symphonai_api.identity import new_id
from symphonai_api.permissions import (
    DenialReason,
    PermissionDecision,
    ToolApprovalRequest,
    shell_grant_prefix,
)


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


@dataclass
class _Pending:
    approval: PendingApproval
    event: threading.Event
    decision: PermissionDecision | None = None
    grant_prefixes: tuple[tuple[str, ...], ...] = ()
    answer: str = ""


class InvalidApprovalAnswer(ValueError):
    """An interactive answer does not match its pending request."""


class ApprovalBroker:
    """Turns a synchronous ApprovalCallback into an HTTP round trip."""

    def __init__(
        self,
        publish: Callable[[PendingApproval], bool],
        *,
        timeout: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        session_id: str | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("approval timeout must be greater than 0")
        self._publish = publish
        self._timeout = timeout
        self._session_id = session_id
        self._pending: dict[str, _Pending] = {}
        self._granted_shell_prefixes: set[tuple[str, ...]] = set()
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
        with self._lock:
            if request.operation == "run_shell" and commands and not any(command_writes) and all(
                any(command[:len(prefix)] == prefix for prefix in self._granted_shell_prefixes)
                for command in commands
            ):
                return PermissionDecision.allow()
        distinct_prefixes = tuple(dict.fromkeys(grant_prefixes))
        remember = (
            ", ".join(" ".join(prefix) for prefix in distinct_prefixes)
            if commands and grant_prefixes and all(grant_prefixes)
            else ""
        )
        approval = PendingApproval(
            approval_id=new_id("appr"),
            operation=request.operation,
            target=request.target,
            details=request.details,
            tool_call_id=request.tool_call_id,
            remember=remember,
            session_id=self._session_id,
        )
        decision, _ = self.request_interaction(approval, grant_prefixes=grant_prefixes)
        return decision

    def request_interaction(
        self,
        approval: PendingApproval,
        *,
        grant_prefixes: tuple[tuple[str, ...], ...] = (),
    ) -> tuple[PermissionDecision, str]:
        pending = _Pending(approval, threading.Event(), grant_prefixes=grant_prefixes)
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
                self._granted_shell_prefixes.update(pending.grant_prefixes)
            pending.event.set()
            return True

    def ask_user(self, question: str, options: tuple[str, ...]) -> tuple[PermissionDecision, str]:
        approval = PendingApproval(
            approval_id=new_id("question"),
            operation="ask_user",
            target=question,
            details="",
            session_id=self._session_id,
            kind="question",
            options=options,
        )
        return self.request_interaction(approval)

    def clear_grants(self) -> None:
        with self._lock:
            self._granted_shell_prefixes.clear()

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
