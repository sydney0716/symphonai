"""Checks for synchronous host approval parking."""

from __future__ import annotations

import threading
import time
import http.client
from pathlib import Path

from symphonai_api.permissions import DenialReason, ToolApprovalRequest
from symphonai_host.approvals import ApprovalBroker, PendingApproval
from symphonai_api.models import Message, ModelResponse, Role
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_host.server import HostServer
from symphonai_host.broker import EventBroker
from symphonai_api.events import RunStarted
from symphonai_host.protocol import decode_frame
from scripts.checks.host_server import _event_stream, _headers, _next_sse, _request
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


@check("host_approvals.broker_cancel_all_unparks")
def check_broker_cancel_all_unparks() -> None:
    broker, item, result, thread = _waiting()
    broker.cancel_all(reason="stopped")
    thread.join(1)
    if not result or "stopped" not in result[0].reason:
        fail(f"stop did not unpark approval: {result!r}")


@check("host_approvals.callback_never_raises")
def check_callback_never_raises() -> None:
    result = ApprovalBroker(lambda _: (_ for _ in ()).throw(RuntimeError("boom"))).callback(REQUEST)
    if result.denial is not DenialReason.NO_APPROVAL_CALLBACK:
        fail(f"publisher failure escaped approval callback: {result!r}")


@check("host_approvals.permissions_untouched")
def check_permissions_untouched() -> None:
    source = ROOT / "symphonai_api" / "permissions.py"
    if not source.is_file() or "class PermissionPolicy" not in source.read_text(encoding="utf-8"):
        fail("permissions module was not present for the approval boundary")


@check("host_approvals.pending_listing")
def check_pending_listing() -> None:
    broker, item, result, thread = _waiting()
    if broker.pending() != (item,):
        fail(f"pending approvals omitted request: {broker.pending()!r}")
    broker.resolve(item.approval_id, allowed=True, reason="")
    thread.join(1)
    if broker.pending():
        fail("resolved approval remained pending")


@check("host_approvals.pending_endpoint")
def check_pending_endpoint() -> None:
    host = HostServer(FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]), PermissionPolicy(repo_root=ROOT))
    host.start()
    try:
        for headers, expected in (({}, 401), ({"Authorization": f"Bearer {host.token}"}, 200)):
            connection = http.client.HTTPConnection("127.0.0.1", host.port, timeout=2)
            connection.request("GET", "/approvals", headers=headers)
            response = connection.getresponse()
            body = response.read()
            connection.close()
            if response.status != expected or (expected == 200 and body != b'{"pending": []}'):
                fail(f"approval listing response was wrong: {response.status}, {body!r}")
    finally:
        host.close()


def _host(*, broker: EventBroker | None = None, approval_timeout: float = 1) -> HostServer:
    host = HostServer(
        FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
        PermissionPolicy(repo_root=ROOT),
        broker=broker,
        approval_timeout=approval_timeout,
        keepalive_seconds=0.05,
    )
    host.start()
    return host


@check("host_approvals.round_trip_over_http")
def check_round_trip_over_http() -> None:
    host = _host()
    result = []
    try:
        connection, response = _event_stream(host)
        try:
            thread = threading.Thread(target=lambda: result.append(host.run.approvals.callback(REQUEST)))
            thread.start()
            frame = _next_sse(connection, response, timeout=5)
            if not isinstance(frame, tuple) or frame[0] != "approval_requested":
                fail(f"approval frame was not published over SSE: {frame!r}")
            approval_id = frame[1].get("approval_id")
            reply_connection, reply = _request(
                host, "POST", "/approval", body={"approval_id": approval_id, "allowed": True}, headers=_headers(host)
            )
            try:
                if reply.status != 200:
                    fail(f"approval reply was not accepted: {reply.status}")
            finally:
                reply_connection.close()
            thread.join(5)
            if thread.is_alive() or not result or not result[0].allowed:
                fail(f"approval reply did not resume callback: {result!r}")
        finally:
            connection.close()
    finally:
        host.close()


@check("host_approvals.survives_a_dropped_frame")
def check_survives_a_dropped_frame() -> None:
    broker = EventBroker(max_queued_events=1)
    host = _host(broker=broker)
    result = []
    subscription = broker.subscribe()
    try:
        broker.publish(RunStarted(agent_id="agent", run_id="before", agent_name="agent"))
        thread = threading.Thread(target=lambda: result.append(host.run.approvals.callback(REQUEST)))
        thread.start()
        deadline = time.monotonic() + 5
        pending = []
        while not pending and time.monotonic() < deadline:
            pending = host.pending_approvals()
            time.sleep(0.01)
        if not pending or subscription.take_dropped() == 0:
            fail(f"dropped approval was not recoverable: {pending!r}")
        reply_connection, reply = _request(
            host, "POST", "/approval", body={"approval_id": pending[0]["approval_id"], "allowed": True}, headers=_headers(host)
        )
        try:
            if reply.status != 200:
                fail(f"reconciled approval was not accepted: {reply.status}")
        finally:
            reply_connection.close()
        thread.join(5)
        if thread.is_alive() or not result or not result[0].allowed:
            fail(f"dropped approval did not resume: {result!r}")
    finally:
        subscription.close()
        host.close()


@check("host_approvals.no_subscriber_over_http")
def check_no_subscriber_over_http() -> None:
    host = _host(approval_timeout=5)
    try:
        callback = host.run._policy.approval_callback
        if callback is None:
            fail("host did not install an approval callback")
        started = time.monotonic()
        decision = callback(REQUEST)
        elapsed = time.monotonic() - started
        if decision.allowed or decision.denial is not DenialReason.NO_APPROVAL_CALLBACK or elapsed >= 1:
            fail(f"no-subscriber approval did not deny promptly: {decision!r}, elapsed={elapsed:.3f}s")
    finally:
        host.close()


@check("host_approvals.stop_unparks_over_http")
def check_stop_unparks_over_http() -> None:
    host = _host(approval_timeout=5)
    result = []
    try:
        connection, response = _event_stream(host)
        try:
            callback = host.run._policy.approval_callback
            if callback is None:
                fail("host did not install an approval callback")
            thread = threading.Thread(target=lambda: result.append(callback(REQUEST)))
            thread.start()
            deadline = time.monotonic() + 5
            while not host.run.approvals.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            pending = host.run.approvals.pending()
            if not pending:
                fail(f"approval did not park before stop; result={result!r}")
            host.run.stop()
            thread.join(5)
            if thread.is_alive() or not result:
                fail(f"stop did not unpark approval; pending={pending!r}, result={result!r}")
            decision = result[0]
            if decision.allowed or decision.denial is not DenialReason.APPROVAL_FAILED or "stopped" not in decision.reason:
                fail(f"stop returned the wrong approval decision: {decision!r}")
        finally:
            connection.close()
    finally:
        host.close()
