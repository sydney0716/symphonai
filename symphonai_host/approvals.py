"""Synchronous approval callbacks backed by pending HTTP decisions."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from symphonai_api.identity import new_id
from symphonai_api.permissions import DenialReason, PermissionDecision, ToolApprovalRequest


DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class PendingApproval:
    approval_id: str
    operation: str
    target: str
    details: str


@dataclass
class _Pending:
    approval: PendingApproval
    event: threading.Event
    decision: PermissionDecision | None = None


class ApprovalBroker:
    """Turns a synchronous ApprovalCallback into an HTTP round trip."""

    def __init__(
        self,
        publish: Callable[[PendingApproval], bool],
        *,
        timeout: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    ) -> None:
        if timeout <= 0:
            raise ValueError("approval timeout must be greater than 0")
        self._publish = publish
        self._timeout = timeout
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    def callback(self, request: ToolApprovalRequest) -> PermissionDecision:
        approval = PendingApproval(new_id("appr"), request.operation, request.target, request.details)
        pending = _Pending(approval, threading.Event())
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
                )
            if not pending.event.wait(self._timeout):
                with self._lock:
                    if pending.decision is None:
                        pending.decision = PermissionDecision.deny(
                            f"no approval decision arrived within {self._timeout:g}s",
                            denial=DenialReason.APPROVAL_FAILED,
                        )
                return pending.decision
            return pending.decision or PermissionDecision.deny(
                "approval was resolved without a decision", denial=DenialReason.APPROVAL_FAILED
            )
        except Exception:
            return PermissionDecision.deny(
                "approval callback failed", denial=DenialReason.APPROVAL_FAILED
            )
        finally:
            with self._lock:
                if self._pending.get(approval.approval_id) is pending:
                    del self._pending[approval.approval_id]

    def resolve(self, approval_id: str, *, allowed: bool, reason: str) -> bool:
        with self._lock:
            pending = self._pending.get(approval_id)
            if pending is None or pending.decision is not None:
                return False
            pending.decision = (
                PermissionDecision.allow()
                if allowed
                else PermissionDecision.deny(
                    reason or "approval denied by user", denial=DenialReason.DENIED_BY_USER
                )
            )
            pending.event.set()
            return True

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
                        denial=DenialReason.APPROVAL_FAILED,
                    )
                    item.event.set()
            return len(pending)
