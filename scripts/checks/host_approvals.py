"""Checks for synchronous host approval parking."""

from __future__ import annotations

import http.client
import hashlib
import json
import subprocess
import tempfile
import threading
import time
from dataclasses import fields
from pathlib import Path
import unittest.mock as mock

from symphonai_api.permissions import ApprovalOutcome, DenialReason, ToolApprovalRequest
from symphonai_host.approvals import ApprovalBroker, PendingApproval
from symphonai_api.models import Message, ModelResponse, Role
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.extensions import load_extensions
from symphonai_host.server import HostServer
from symphonai_host.run import HostRun
from symphonai_host.broker import EventBroker
from symphonai_api.events import RunStarted
from symphonai_host.protocol import ApprovalRequested, decode_frame
from scripts.checks.host_server import _await_sse, _headers, _request, _subscribed_stream, _wait_until
from scripts.checks.harness import check, fail


ROOT = Path(__file__).resolve().parents[2]
REQUEST = ToolApprovalRequest("write_file", "new.txt", "write test file")


def _waiting(timeout: float = 1.0, invoke=None):  # noqa: ANN001
    published: list[PendingApproval] = []
    broker = ApprovalBroker(lambda item: published.append(item) or True, timeout=timeout)
    result: list = []
    operation = (
        (lambda: broker.callback(REQUEST))
        if invoke is None
        else (lambda: invoke(broker))
    )
    thread = threading.Thread(target=lambda: result.append(operation()))
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
    if not item.approval_id or (
        item.operation,
        item.target,
        item.details,
        item.tool_call_id,
    ) != (
        REQUEST.operation,
        REQUEST.target,
        REQUEST.details,
        "",
    ):
        fail(f"approval request shape was wrong: {item!r}")
    broker.resolve(item.approval_id, allowed=True, reason="")
    thread.join(1)


def _prompted_write(broker: ApprovalBroker, tool_call_id: str | None):
    policy = PermissionPolicy(
        repo_root=ROOT,
        mode="ask",
        approval_callback=broker.callback,
    )
    if tool_call_id is None:
        return policy.check_write("new.txt")
    with policy.event_context(
        None,
        agent_id="agent",
        run_id="run",
        turn_id="turn",
        tool_name="write_file",
        tool_call_id=tool_call_id,
    ):
        return policy.check_write("new.txt")


@check("host_approvals.tool_call_identity")
def check_tool_call_identity() -> None:
    for supplied, expected in (("call-from-runtime", "call-from-runtime"), (None, "")):
        broker, item, result, thread = _waiting(
            invoke=lambda active, value=supplied: _prompted_write(active, value)
        )
        broker.resolve(item.approval_id, allowed=True, reason="")
        thread.join(1)
        if item.tool_call_id != expected:
            fail(
                f"approval tool-call identity was wrong for {supplied!r}: {item!r}"
            )
        if thread.is_alive() or not result or not result[0].allowed:
            fail(f"approval did not finish for {supplied!r}: {result!r}")


@check("host_approvals.allow_resumes")
def check_allow_resumes() -> None:
    broker, item, result, thread = _waiting()
    broker.resolve(item.approval_id, allowed=True, reason="")
    thread.join(1)
    if (
        not result
        or not result[0].allowed
        or result[0].outcome is not ApprovalOutcome.ALLOWED
    ):
        fail(f"allowed reply did not resume: {result!r}")


@check("host_approvals.deny_blocks_call")
def check_deny_blocks_call() -> None:
    broker, item, result, thread = _waiting()
    broker.resolve(item.approval_id, allowed=False, reason="no")
    thread.join(1)
    if (
        not result
        or result[0].denial is not DenialReason.DENIED_BY_USER
        or result[0].outcome is not ApprovalOutcome.REJECTED
    ):
        fail(f"denial reason was lost: {result!r}")


@check("host_approvals.unknown_id_404")
def check_unknown_id_404() -> None:
    if ApprovalBroker(lambda _: True).resolve("missing", allowed=True, reason=""):
        fail("unknown approval id resolved")


@check("host_approvals.timeout_denies")
def check_timeout_denies() -> None:
    broker = ApprovalBroker(lambda _: True, timeout=0.05)
    result = broker.callback(REQUEST)
    if (
        result.denial is not DenialReason.APPROVAL_FAILED
        or result.outcome is not ApprovalOutcome.UNAVAILABLE
        or "0.05" not in result.reason
    ):
        fail(f"timeout denial was wrong: {result!r}")


@check("host_approvals.no_subscriber_denies_fast")
def check_no_subscriber_denies_fast() -> None:
    start = time.monotonic()
    result = ApprovalBroker(lambda _: False, timeout=1).callback(REQUEST)
    if (
        result.denial is not DenialReason.NO_APPROVAL_CALLBACK
        or result.outcome is not ApprovalOutcome.UNAVAILABLE
        or time.monotonic() - start > 0.2
    ):
        fail(f"missing subscriber parked approval: {result!r}")


@check("host_approvals.broker_cancel_all_unparks")
def check_broker_cancel_all_unparks() -> None:
    broker, item, result, thread = _waiting()
    broker.cancel_all(reason="stopped")
    thread.join(1)
    if (
        not result
        or result[0].denial is not DenialReason.APPROVAL_CANCELLED
        or result[0].outcome is not ApprovalOutcome.CANCELLED
        or "stopped" not in result[0].reason
    ):
        fail(f"stop did not unpark approval: {result!r}")


@check("host_approvals.callback_never_raises")
def check_callback_never_raises() -> None:
    result = ApprovalBroker(lambda _: (_ for _ in ()).throw(RuntimeError("boom"))).callback(REQUEST)
    if (
        result.denial is not DenialReason.NO_APPROVAL_CALLBACK
        or result.outcome is not ApprovalOutcome.UNAVAILABLE
    ):
        fail(f"publisher failure escaped approval callback: {result!r}")


def _start_shell_check(policy: PermissionPolicy, argv: list[str]):
    result = []
    thread = threading.Thread(target=lambda: result.append(policy.check_shell(argv)))
    thread.start()
    return result, thread


def _await_pending(broker: ApprovalBroker) -> PendingApproval:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        pending = broker.pending()
        if pending:
            return pending[0]
        time.sleep(0.01)
    fail("shell approval was not published")


@check("host_approvals.remembered_shell_prefix")
def check_remembered_shell_prefix() -> None:
    published: list[PendingApproval] = []
    broker = ApprovalBroker(lambda item: published.append(item) or True, timeout=1)
    policy = PermissionPolicy(repo_root=ROOT, mode="ask", approval_callback=broker.callback)
    result, thread = _start_shell_check(policy, ["pytest", "-x"])
    approval = _await_pending(broker)
    if approval.remember != "pytest":
        fail(f"pytest approval offered the wrong remembered prefix: {approval.remember!r}")
    if not broker.resolve(approval.approval_id, allowed=True, reason="", remember=True):
        fail("remembered pytest approval did not resolve")
    thread.join(1)
    if thread.is_alive() or not result or not result[0].allowed:
        fail(f"remembered pytest approval did not resume: {result!r}")

    for argv in (["pytest"], ["pytest", "-q"], ["pytest", "tests/test_a.py"]):
        decision = policy.check_shell(argv)
        if not decision.allowed:
            fail(f"remembered prefix did not allow {argv!r}: {decision!r}")
    if len(published) != 1:
        fail(f"remembered pytest commands published extra approvals: {published!r}")

    result, thread = _start_shell_check(policy, ["ruff", "check", "."])
    ruff = _await_pending(broker)
    if ruff.target != "ruff check .":
        fail(f"ungranted command was not prompted: {ruff!r}")
    broker.resolve(ruff.approval_id, allowed=True, reason="")
    thread.join(1)
    if thread.is_alive() or not result or not result[0].allowed:
        fail(f"unremembered approval did not resume: {result!r}")


@check("host_approvals.command_line_grants")
def check_command_line_grants() -> None:
    published: list[PendingApproval] = []
    broker = ApprovalBroker(lambda item: published.append(item) or True, timeout=1)
    policy = PermissionPolicy(repo_root=ROOT, mode="ask", approval_callback=broker.callback)
    line = ["bash", "-c", "git status && pytest -x"]
    result, thread = _start_shell_check(policy, line)
    approval = _await_pending(broker)
    if approval.remember != "git status, pytest":
        fail(f"multi-command approval offered the wrong grant label: {approval!r}")
    if not broker.resolve(approval.approval_id, allowed=True, reason="", remember=True):
        fail("multi-command remembered approval did not resolve")
    thread.join(1)
    if thread.is_alive() or not result or not result[0].allowed:
        fail(f"multi-command approval did not resume: {result!r}")

    if not policy.check_shell(line).allowed or len(published) != 1:
        fail(f"remembered commands prompted again: {published!r}")
    result, thread = _start_shell_check(policy, ["bash", "-c", "git status >> notes.txt"])
    writing = _await_pending(broker)
    if writing.target != "git status >> notes.txt":
        fail(f"remembered grant auto-allowed a writing command: {writing!r}")
    broker.resolve(writing.approval_id, allowed=True, reason="")
    thread.join(1)
    if thread.is_alive() or not result or not result[0].allowed:
        fail(f"writing command approval did not resume: {result!r}")

    result, thread = _start_shell_check(policy, ["bash", "-c", "git status && make"])
    other = _await_pending(broker)
    if other.target != "git status && make" or other.remember != "git status, make":
        fail(f"partially granted command line had the wrong approval: {other!r}")
    broker.resolve(other.approval_id, allowed=True, reason="")
    thread.join(1)
    if thread.is_alive() or not result or not result[0].allowed:
        fail(f"new command approval did not resume: {result!r}")


@check("host_approvals.denied_shell_grant_not_remembered")
def check_denied_shell_grant_not_remembered() -> None:
    broker = ApprovalBroker(lambda _: True, timeout=1)
    policy = PermissionPolicy(repo_root=ROOT, mode="ask", approval_callback=broker.callback)
    result, thread = _start_shell_check(policy, ["pytest", "-x"])
    first = _await_pending(broker)
    if not broker.resolve(first.approval_id, allowed=False, reason="no", remember=True):
        fail("denied approval did not resolve")
    thread.join(1)
    if thread.is_alive() or not result or result[0].allowed:
        fail(f"denied approval did not refuse its shell call: {result!r}")

    result, thread = _start_shell_check(policy, ["pytest", "-x"])
    second = _await_pending(broker)
    if second.approval_id == first.approval_id or second.remember != "pytest":
        fail(f"denied remembered approval did not prompt again: {second!r}")
    broker.resolve(second.approval_id, allowed=False, reason="no")
    thread.join(1)


@check("host_approvals.grants_clear")
def check_grants_clear() -> None:
    broker = ApprovalBroker(lambda _: True, timeout=1)
    request = ToolApprovalRequest(
        "run_shell", "pytest -x", command=("pytest", "-x")
    )
    result = []
    thread = threading.Thread(target=lambda: result.append(broker.callback(request)))
    thread.start()
    approval = _await_pending(broker)
    broker.resolve(approval.approval_id, allowed=True, reason="", remember=True)
    thread.join(1)
    if thread.is_alive() or not result or not result[0].allowed:
        fail("remembered shell callback did not finish")
    if not broker.callback(request).allowed:
        fail("remembered grant did not bypass publication")

    broker.clear_grants()
    result.clear()
    thread = threading.Thread(target=lambda: result.append(broker.callback(request)))
    thread.start()
    cleared = _await_pending(broker)
    broker.resolve(cleared.approval_id, allowed=False, reason="no")
    thread.join(1)
    if thread.is_alive() or not result or result[0].allowed:
        fail("cleared shell grant did not prompt again")


@check("host_approvals.always_deny_precedes_remembered_grant")
def check_always_deny_precedes_remembered_grant() -> None:
    broker = ApprovalBroker(lambda _: True, timeout=1)
    request = ToolApprovalRequest("run_shell", "rm", command=("rm",))
    result = []
    thread = threading.Thread(target=lambda: result.append(broker.callback(request)))
    thread.start()
    approval = _await_pending(broker)
    broker.resolve(approval.approval_id, allowed=True, reason="", remember=True)
    thread.join(1)
    if thread.is_alive() or not result or not result[0].allowed:
        fail("test setup did not grant rm")

    policy = PermissionPolicy(repo_root=ROOT, mode="ask", approval_callback=broker.callback)
    decision = policy.check_shell(["rm", "-rf", "/"])
    if decision.allowed or broker.pending():
        fail(f"always-denied rm consulted or bypassed a remembered grant: {decision!r}")




@check("host_approvals.approval_records_match")
def check_approval_records_match() -> None:
    pending_fields = tuple(field.name for field in fields(PendingApproval))
    requested_fields = tuple(field.name for field in fields(ApprovalRequested))
    if pending_fields != requested_fields:
        fail(
            "approval record fields differ: "
            f"PendingApproval={pending_fields!r}, "
            f"ApprovalRequested={requested_fields!r}"
        )

    protocol = (ROOT / "symphonai_host" / "PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    approvals = protocol.partition("## Approvals\n")[2].partition("\n## ")[0]
    missing = [name for name in requested_fields if f"`{name}`" not in approvals]
    if missing:
        fail(f"approval protocol paragraph omits fields: {missing!r}")


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


@check("host_approvals.session_scoped_grants_and_tags")
def check_session_scoped_grants_and_tags() -> None:
    published_a: list[PendingApproval] = []
    published_b: list[PendingApproval] = []
    broker_a = ApprovalBroker(lambda item: published_a.append(item) or True, timeout=1, session_id="session-a")
    broker_b = ApprovalBroker(lambda item: published_b.append(item) or True, timeout=1, session_id="session-b")
    request = ToolApprovalRequest("run_shell", "echo", "run a command", command=("echo", "hello"))
    result: list = []
    thread = threading.Thread(target=lambda: result.append(broker_a.callback(request)))
    thread.start()
    deadline = time.monotonic() + 1
    while not published_a and time.monotonic() < deadline:
        time.sleep(0.005)
    if not published_a or published_a[0].session_id != "session-a":
        fail(f"first approval was not tagged with its session: {published_a!r}")
    if not broker_a.resolve(published_a[0].approval_id, allowed=True, reason="", remember=True):
        fail("first session's remembered grant did not resolve")
    thread.join(timeout=1)
    if thread.is_alive() or not result or not result[0].allowed:
        fail("first session's approval did not resume")
    if not broker_a.callback(request).allowed:
        fail("first session did not honor its own remembered grant")
    other: list = []
    second = threading.Thread(target=lambda: other.append(broker_b.callback(request)))
    second.start()
    deadline = time.monotonic() + 1
    while not published_b and time.monotonic() < deadline:
        time.sleep(0.005)
    if not published_b or published_b[0].session_id != "session-b":
        fail("second session inherited the first session's shell grant")
    broker_b.resolve(published_b[0].approval_id, allowed=False, reason="no")
    second.join(timeout=1)


@check("host_approvals.transport_tool_call_id")
def check_transport_tool_call_id() -> None:
    host = _host()
    request = ToolApprovalRequest(
        operation="write_file",
        target="transport.txt",
        details="transport identity",
        tool_call_id="call-over-wire",
        diff="--- a.txt\n+++ a.txt\n@@ -1 +1 @@\n-old\n+new",
    )
    result = []
    stream_connection = None
    thread = None
    approval_id = None
    frame_payload = {}
    listing = {}
    try:
        stream_connection, response = _subscribed_stream(host)
        thread = threading.Thread(
            target=lambda: result.append(host.run.approvals.callback(request))
        )
        thread.start()
        frame = _await_sse(
            stream_connection,
            response,
            lambda candidate: isinstance(candidate, tuple)
            and candidate[0] == "approval_requested",
            what="approval request with tool-call identity",
        )
        frame_payload = frame[1]
        approval_id = frame_payload.get("approval_id")

        listing_connection = http.client.HTTPConnection(
            "127.0.0.1", host.port, timeout=2
        )
        try:
            listing_connection.request(
                "GET", "/approvals", headers={"Authorization": f"Bearer {host.token}"}
            )
            listing_response = listing_connection.getresponse()
            body = listing_response.read()
            if listing_response.status != 200:
                fail(
                    "approval listing failed while an approval was pending: "
                    f"{listing_response.status}, {body!r}"
                )
            listing = json.loads(body)
        finally:
            listing_connection.close()

        host.run.approvals.resolve(approval_id, allowed=True, reason="")
        thread.join(1)
    finally:
        if isinstance(approval_id, str):
            host.run.approvals.resolve(approval_id, allowed=True, reason="")
        if thread is not None:
            thread.join(1)
        if stream_connection is not None:
            stream_connection.close()
        host.close()

    pending = listing.get("pending", [])
    frame_id = frame_payload.get("tool_call_id")
    pending_id = pending[0].get("tool_call_id") if len(pending) == 1 else None
    frame_diff = frame_payload.get("diff")
    pending_diff = pending[0].get("diff") if len(pending) == 1 else None
    if (
        frame_id != request.tool_call_id
        or pending_id != request.tool_call_id
        or frame_diff != request.diff
        or pending_diff != request.diff
    ):
        fail(
            "approval transport lost tool-call identity: "
            f"frame={frame_payload!r}, listing={listing!r}"
        )
    if thread is None or thread.is_alive() or not result or not result[0].allowed:
        fail(f"transport approval did not finish: {result!r}")


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


def _git_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "check@example.test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Spec Check"], cwd=root, check=True)
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "seed.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "seed"], cwd=root, check=True)


def _project_remember(host: HostServer, operation: str, expected: str) -> None:
    result = []
    check = (
        (lambda: host.run.policy.check_shell(["pnpm", "test", "--run"]))
        if operation == "run_shell"
        else (lambda: host.run.policy.check_fetch("https://Remember-This-Host.Example./guide"))
    )
    connection, response = _subscribed_stream(host)
    thread = threading.Thread(target=lambda: result.append(check()))
    thread.start()
    try:
        frame = _await_sse(
            connection, response,
            lambda candidate: isinstance(candidate, tuple) and candidate[0] == "approval_requested",
            what=f"{operation} approval",
        )
        if frame[1].get("remember") != expected:
            fail(f"{operation} approval offered the wrong remembered value: {frame[1]!r}")
        connection_out, reply = _request(
            host, "POST", "/approval",
            body={"approval_id": frame[1]["approval_id"], "allowed": True, "remember": "project"},
            headers=_headers(host),
        )
        body = reply.read()
        connection_out.close()
        if reply.status != 200:
            fail(f"project remember was refused: {reply.status}, {body!r}")
        thread.join(1)
        if thread.is_alive() or not result or not result[0].allowed:
            fail(f"project remembered approval did not resume: {result!r}")
    finally:
        if thread.is_alive():
            pending = host.run.pending_approvals()
            if pending:
                host.resolve_approval(pending[0].approval_id, allowed=False, reason="test cleanup")
            thread.join(1)
        connection.close()


@check("host_approvals.always_allow_persists")
def always_allow_persists() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, home = base / "project", base / "home"
        _git_repo(root)
        extensions = load_extensions(repo_root=root, home=home)
        host = HostServer(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            PermissionPolicy(repo_root=root, shell_enabled=True),
            extensions=extensions, home=home, keepalive_seconds=0.05,
        )
        host.start()
        try:
            _project_remember(host, "run_shell", "pnpm test")
            _project_remember(host, "web_fetch", "remember-this-host.example")
            project_key = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()
            grants_path = home / ".symphonai" / "permissions" / f"{project_key}.json"
            saved = json.loads(grants_path.read_text(encoding="utf-8"))
            if saved != {
                "schema_version": 1,
                "shell": [["pnpm", "test"]],
                "fetch": ["remember-this-host.example"],
            }:
                fail(f"project remember wrote the wrong grant file: {saved!r}")
            if grants_path.stat().st_mode & 0o777 != 0o600:
                fail(f"project permission grant file was not private: {grants_path.stat().st_mode:o}")
            host.run.end_conversation()
            run_id = host.run.start("A new conversation")
            _wait_until(
                lambda: run_id not in host.run._active_by_session,
                "new conversation did not finish",
            )
            if not host.run.policy.check_shell(["pnpm", "test", "--run"]).allowed:
                fail("project shell grant did not reach a new conversation")
            if not host.run.policy.check_fetch("https://REMEMBER-THIS-HOST.EXAMPLE./new").allowed:
                fail("project fetch grant did not reach a new conversation")
        finally:
            host.close()

        fresh_extensions = load_extensions(repo_root=root, home=home)
        fresh = HostRun(
            FakeModelProvider(), PermissionPolicy(repo_root=root, shell_enabled=True),
            EventBroker(), extensions=fresh_extensions,
            project_grants_root=home / ".symphonai" / "permissions",
        )
        try:
            if not fresh.policy.check_shell(["pnpm", "test", "--run"]).allowed:
                fail("project shell grant did not reach a new HostRun")
            if not fresh.policy.check_fetch("https://remember-this-host.example/next").allowed:
                fail("project fetch grant did not reach a new HostRun")
        finally:
            fresh.close()

        management = HostServer(
            FakeModelProvider(), PermissionPolicy(repo_root=root, shell_enabled=True),
            extensions=load_extensions(repo_root=root, home=home), home=home,
            keepalive_seconds=0.05,
        )
        management.start()
        try:
            connection, response = _request(management, "GET", "/permissions", headers=_headers(management))
            listed = json.loads(response.read())
            connection.close()
            if response.status != 200 or listed != {
                "shell": [["pnpm", "test"]], "fetch": ["remember-this-host.example"],
            }:
                fail(f"GET /permissions returned the wrong project grants: {response.status}, {listed!r}")
            connection, response = _request(
                management, "POST", "/permissions/remove",
                body={"kind": "fetch", "value": "remember-this-host.example"},
                headers=_headers(management),
            )
            removed = json.loads(response.read())
            connection.close()
            if response.status != 200 or removed != {"removed": True}:
                fail(f"POST /permissions/remove failed: {response.status}, {removed!r}")
        finally:
            management.close()

        other_root = base / "other-project"
        _git_repo(other_root)
        other = HostServer(
            FakeModelProvider(), PermissionPolicy(repo_root=other_root, shell_enabled=True),
            extensions=load_extensions(repo_root=other_root, home=home), home=home,
            keepalive_seconds=0.05,
        )
        other.start()
        try:
            _project_remember(other, "run_shell", "pnpm test")
        finally:
            other.close()


@check("host_approvals.always_allow_stays_private")
def always_allow_stays_private() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, home = base / "project", base / "home"
        _git_repo(root)
        host = HostServer(
            FakeModelProvider(), PermissionPolicy(repo_root=root, shell_enabled=True),
            extensions=load_extensions(repo_root=root, home=home), home=home,
            keepalive_seconds=0.05,
        )
        host.start()
        try:
            _project_remember(host, "run_shell", "pnpm test")
        finally:
            host.close()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, check=True,
        ).stdout.decode()
        if ".symphonai/config.local.toml" in status:
            fail(f"private permission config appeared in git status: {status!r}")
        if status:
            fail(f"project grant created repository changes: {status!r}")
        if (root / ".symphonai" / "config.local.toml").exists():
            fail("project grant was written inside the repository")
        if not list((home / ".symphonai" / "permissions").glob("*.json")):
            fail("project grant was not saved under the SymphonAI home")


@check("host_approvals.project_grant_validation_and_ceiling")
def project_grant_validation_and_ceiling() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, home = base / "project", base / "home"
        _git_repo(root)
        grants_dir = home / ".symphonai" / "permissions"
        grants_dir.mkdir(parents=True)
        grant_key = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()
        grants_path = grants_dir / f"{grant_key}.json"
        invalid_contents = b'{"schema_version":2,"shell":[["pnpm","test"]],"fetch":[]}\n'
        grants_path.write_bytes(invalid_contents)
        pending = []
        published = threading.Event()
        host = HostRun(
            FakeModelProvider(), PermissionPolicy(repo_root=root, shell_enabled=True),
            EventBroker(), extensions=load_extensions(repo_root=root, home=home),
            project_grants_root=grants_dir, approval_timeout=0.2,
            publish_approval=lambda approval: (pending.append(approval), published.set(), True)[-1],
        )
        result = []
        worker = threading.Thread(target=lambda: result.append(host.policy.check_shell(["pnpm", "test"])))
        try:
            with mock.patch("symphonai_host.run._LOG.error") as reported:
                worker.start()
                if not published.wait(2) or not pending:
                    fail("invalid project grants bypassed the normal approval")
                host.resolve_approval(pending[0].approval_id, allowed=True, remember="project")
                worker.join(2)
                if worker.is_alive() or not result or not result[0].allowed:
                    fail(f"one-time approval did not work with invalid project grants: {result!r}")
                if not reported.called:
                    fail("invalid project grants were not reported")
            if grants_path.read_bytes() != invalid_contents:
                fail("invalid project grant file was overwritten")
        finally:
            host.close()

        ceiling_home = base / "ceiling-home"
        ceiling_config = ceiling_home / ".symphonai" / "config.toml"
        ceiling_config.parent.mkdir(parents=True)
        ceiling_config.write_text(
            '[agents.ceiling]\nshell_allowlist = [["pnpm", "test"]]\n', encoding="utf-8",
        )
        ceiling_dir = ceiling_home / ".symphonai" / "permissions"
        ceiling_dir.mkdir(parents=True)
        ceiling_path = ceiling_dir / f"{grant_key}.json"
        ceiling_path.write_text(
            '{"schema_version":1,"shell":[["python3"]],"fetch":[]}\n', encoding="utf-8",
        )
        requests = []
        ceiling_host = HostRun(
            FakeModelProvider(), PermissionPolicy(repo_root=root, shell_enabled=True),
            EventBroker(), extensions=load_extensions(repo_root=root, home=ceiling_home),
            project_grants_root=ceiling_dir, approval_timeout=0.05,
            publish_approval=lambda approval: (requests.append(approval), True)[-1],
        )
        try:
            decision = ceiling_host.policy.check_shell(["python3"])
            if decision.allowed or not requests:
                fail("project grant outside the capability ceiling bypassed approval")
        finally:
            ceiling_host.close()


@check("host_approvals.round_trip_over_http")
def check_round_trip_over_http() -> None:
    host = _host()
    result = []
    try:
        connection, response = _subscribed_stream(host)
        try:
            thread = threading.Thread(target=lambda: result.append(host.run.approvals.callback(REQUEST)))
            thread.start()
            frame = _await_sse(
                connection,
                response,
                lambda candidate: isinstance(candidate, tuple)
                and candidate[0] == "approval_requested",
                what="approval request",
            )
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
        if (
            decision.allowed
            or decision.denial is not DenialReason.NO_APPROVAL_CALLBACK
            or decision.outcome is not ApprovalOutcome.UNAVAILABLE
            or elapsed >= 1
        ):
            fail(f"no-subscriber approval did not deny promptly: {decision!r}, elapsed={elapsed:.3f}s")
    finally:
        host.close()


@check("host_approvals.stop_unparks_over_http")
def check_stop_unparks_over_http() -> None:
    host = _host(approval_timeout=5)
    result = []
    try:
        connection, response = _subscribed_stream(host)
        try:
            callback = host.run._policy.approval_callback
            if callback is None:
                fail("host did not install an approval callback")
            thread = threading.Thread(target=lambda: result.append(callback(REQUEST)))
            thread.start()
            _await_sse(
                connection,
                response,
                lambda frame: isinstance(frame, tuple)
                and frame[0] == "approval_requested",
                what="parked approval request",
            )
            pending = host.run.approvals.pending()
            if not pending:
                fail(f"approval did not park before stop; result={result!r}")
            host.run.stop()
            thread.join(5)
            if thread.is_alive() or not result:
                fail(f"stop did not unpark approval; pending={pending!r}, result={result!r}")
            decision = result[0]
            if (
                decision.allowed
                or decision.denial is not DenialReason.APPROVAL_CANCELLED
                or decision.outcome is not ApprovalOutcome.CANCELLED
                or "stopped" not in decision.reason
            ):
                fail(f"stop returned the wrong approval decision: {decision!r}")
        finally:
            connection.close()
    finally:
        host.close()


@check("host_approvals.pending_session_identity_and_activity")
def check_pending_session_identity_and_activity() -> None:
    host = _host(approval_timeout=2)
    thread = None
    connection = None
    try:
        run_id = host.run.start("create conversation")
        deadline = time.monotonic() + 2
        while host.run.active and time.monotonic() < deadline:
            time.sleep(0.005)
        if host.run.active:
            fail("setup conversation did not finish")
        connection, stream = _subscribed_stream(host)
        result = []
        thread = threading.Thread(target=lambda: result.append(host.run.approvals.callback(REQUEST)))
        thread.start()
        frame = _await_sse(
            connection, stream,
            lambda item: isinstance(item, tuple) and item[0] == "approval_requested",
            what="session-tagged approval",
        )
        payload = frame[1]
        if payload.get("session_id") != run_id:
            fail(f"approval event did not identify its session: {payload!r}")
        connection_list, response = _request(host, "GET", "/approvals", headers=_headers(host))
        try:
            listing = json.loads(response.read())
            if response.status != 200 or listing.get("pending") != [payload]:
                fail(f"approval listing and event differed: {listing!r}, {payload!r}")
        finally:
            connection_list.close()
        connection_list, response = _request(host, "GET", "/sessions", headers=_headers(host))
        try:
            sessions = json.loads(response.read())
            item = next(entry for entry in sessions if entry["run_id"] == run_id)
            if item.get("activity") != "waiting":
                fail(f"pending approval did not mark its session waiting: {item!r}")
        finally:
            connection_list.close()
        connection_list, response = _request(
            host, "POST", "/approval",
            body={"approval_id": payload["approval_id"], "allowed": True},
            headers=_headers(host),
        )
        try:
            reply = json.loads(response.read())
            if response.status != 200 or reply != {"resolved": True}:
                fail(f"session approval did not resolve: {response.status}, {reply!r}")
        finally:
            connection_list.close()
        thread.join(1)
        if thread.is_alive() or not result or not result[0].allowed:
            fail("session-scoped approval callback did not resume")
    finally:
        if thread is not None:
            thread.join(1)
        if connection is not None:
            connection.close()
        host.close()


@check("host_approvals.runtime_event_session_tag")
def check_runtime_event_session_tag() -> None:
    host = _host()
    connection = None
    try:
        connection, stream = _subscribed_stream(host)
        connection_prompt, response = _request(
            host, "POST", "/prompt", body={"prompt": "tag events"}, headers=_headers(host)
        )
        try:
            run_id = json.loads(response.read())["run_id"]
        finally:
            connection_prompt.close()
        terminal = _await_sse(
            connection, stream,
            lambda item: isinstance(item, tuple) and item[0] == "event"
            and item[1].get("type") in ("RunFinished", "RunFailed"),
            what="tagged terminal event",
        )
        if terminal[1].get("session_id") != run_id:
            fail(f"runtime event frame omitted its conversation id: {terminal[1]!r}")
    finally:
        if connection is not None:
            connection.close()
        host.close()
