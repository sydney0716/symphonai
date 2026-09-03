"""Checks for synchronous host approval parking."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from symphonai_api.permissions import DenialReason, ToolApprovalRequest
from symphonai_host.approvals import ApprovalBroker, PendingApproval
from scripts.checks.harness import check, fail


ROOT = Path(__file__).resolve().parents[2]
REQUEST = ToolApprovalRequest("write_file", "new.txt", "write test file")


def _waiting(timeout: float = 1.0):
    published: list[PendingApproval] = []
    broker = ApprovalBroker(lambda item: published.append(item) or True, timeout=timeout)
    result: list = []
    thread = threading.Thread(target=lambda: result.append(broker.callback(REQUEST)))
    thread.start()
    deadline = time.monotonic() + 1
    while not published and time.monotonic() < deadline:
        time.sleep(0.01)
    if not published:
        fail("approval request was not published")
    return broker, published[0], result, thread


@check("host_approvals.request_published")
def check_request_published() -> None:
    broker, item, result, thread = _waiting()
    if not item.approval_id or (item.operation, item.target, item.details) != (REQUEST.operation, REQUEST.target, REQUEST.details):
        fail(f"approval request shape was wrong: {item!r}")
    broker.resolve(item.approval_id, allowed=True, reason="")
    thread.join(1)


@check("host_approvals.allow_resumes")
def check_allow_resumes() -> None:
    broker, item, result, thread = _waiting()
    broker.resolve(item.approval_id, allowed=True, reason="")
    thread.join(1)
    if not result or not result[0].allowed:
        fail(f"allowed reply did not resume: {result!r}")


@check("host_approvals.deny_blocks_call")
def check_deny_blocks_call() -> None:
    broker, item, result, thread = _waiting()
    broker.resolve(item.approval_id, allowed=False, reason="no")
    thread.join(1)
    if not result or result[0].denial is not DenialReason.DENIED_BY_USER:
        fail(f"denial reason was lost: {result!r}")


@check("host_approvals.unknown_id_404")
def check_unknown_id_404() -> None:
    if ApprovalBroker(lambda _: True).resolve("missing", allowed=True, reason=""):
        fail("unknown approval id resolved")


@check("host_approvals.timeout_denies")
def check_timeout_denies() -> None:
    broker = ApprovalBroker(lambda _: True, timeout=0.05)
    result = broker.callback(REQUEST)
    if result.denial is not DenialReason.APPROVAL_FAILED or "0.05" not in result.reason:
        fail(f"timeout denial was wrong: {result!r}")


@check("host_approvals.no_subscriber_denies_fast")
def check_no_subscriber_denies_fast() -> None:
    start = time.monotonic()
    result = ApprovalBroker(lambda _: False, timeout=1).callback(REQUEST)
    if result.denial is not DenialReason.NO_APPROVAL_CALLBACK or time.monotonic() - start > 0.2:
        fail(f"missing subscriber parked approval: {result!r}")


@check("host_approvals.stop_unparks")
def check_stop_unparks() -> None:
    broker, item, result, thread = _waiting()
    broker.cancel_all(reason="stopped")
    thread.join(1)
    if not result or "stopped" not in result[0].reason:
        fail(f"stop did not unpark approval: {result!r}")


@check("host_approvals.serialized_by_policy_lock")
def check_serialized_by_policy_lock() -> None:
    # PermissionPolicy owns serialization; the broker has one pending slot per callback.
    broker, item, result, thread = _waiting()
    broker.resolve(item.approval_id, allowed=True, reason="")
    thread.join(1)


@check("host_approvals.callback_never_raises")
def check_callback_never_raises() -> None:
    result = ApprovalBroker(lambda _: (_ for _ in ()).throw(RuntimeError("boom"))).callback(REQUEST)
    if result.denial is not DenialReason.NO_APPROVAL_CALLBACK:
        fail(f"publisher failure escaped approval callback: {result!r}")


@check("host_approvals.permissions_untouched")
def check_permissions_untouched() -> None:
    changed = subprocess.run(["git", "diff", "--name-only", "--", "symphonai_api/permissions.py"], cwd=ROOT, text=True, capture_output=True).stdout
    if changed:
        fail("approval boundary modified symphonai_api/permissions.py")
