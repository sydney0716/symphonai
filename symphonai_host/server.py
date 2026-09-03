"""Loopback-only HTTP/SSE boundary for a single SymphonAI host run."""

from __future__ import annotations

import json
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import ModelProvider
from symphonai_host.broker import EventBroker, Subscription
from symphonai_host.protocol import (
    ApprovalRequested,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_request,
    encode_event,
    encode_frame,
)
from symphonai_host.run import HostRun, RunActiveError


class HostServer:
    """Own a loopback HTTP server, event broker, and one active runtime run."""

    def __init__(
        self,
        provider: ModelProvider,
        policy: PermissionPolicy,
        *,
        token: str | None = None,
        broker: EventBroker | None = None,
        keepalive_seconds: float = 15.0,
        system_prompt: str | None = None,
        max_turns: int = 20,
        model: str | None = None,
        approval_timeout: float = 300.0,
    ) -> None:
        if keepalive_seconds <= 0:
            raise ValueError("keepalive_seconds must be greater than 0")
        self.token = token or secrets.token_urlsafe(32)
        self.broker = broker or EventBroker()
        self.run = HostRun(
            provider,
            policy,
            self.broker,
            system_prompt=system_prompt,
            max_turns=max_turns,
            model=model,
            publish_approval=self._publish_approval,
            approval_timeout=approval_timeout,
        )
        self.keepalive_seconds = keepalive_seconds
        self._handshake_printed = False
        self._thread: threading.Thread | None = None
        self._serving = False
        self._close_lock = threading.Lock()
        self._closed = False
        handler = self._handler_type()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._httpd.daemon_threads = True

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def address(self) -> tuple[str, int]:
        return "127.0.0.1", self.port

    def handshake(self) -> dict[str, Any]:
        return {"port": self.port, "token": self.token}

    def _publish_approval(self, approval: Any) -> bool:
        if self.broker.subscriber_count == 0:
            return False
        self.broker.publish(
            ApprovalRequested(
                approval.approval_id, approval.operation, approval.target, approval.details
            )
        )
        return True

    def pending_approvals(self) -> list[dict[str, str]]:
        return [approval.__dict__ for approval in self.run.approvals.pending()]

    def print_handshake(self) -> None:
        if not self._handshake_printed:
            print(json.dumps(self.handshake()), flush=True)
            self._handshake_printed = True

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self.serve_forever,
                name="symphonai-host-http",
                daemon=True,
            )
            self._thread.start()

    def serve_forever(self) -> None:
        self._serving = True
        try:
            self._httpd.serve_forever()
        finally:
            self._serving = False

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self.run.stop()
        self.broker.close()
        if self._serving:
            self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        host = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def handle(self) -> None:
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError):
                    return

            def log_message(self, format: str, *args: object) -> None:
                return

            def _authorized(self) -> bool:
                supplied = self.headers.get("Authorization", "")
                expected = f"Bearer {host.token}"
                if not secrets.compare_digest(supplied, expected):
                    self.send_response(HTTPStatus.UNAUTHORIZED)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return False
                return True

            def _json(self, status: HTTPStatus, body: dict) -> None:
                encoded = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _read_object(self) -> dict:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    data = json.loads(self.rfile.read(length).decode("utf-8"))
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProtocolError(f"invalid JSON payload: {exc}") from None
                if not isinstance(data, dict):
                    raise ProtocolError("request payload must be an object")
                return data

            def _not_found(self) -> None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._json(
                        HTTPStatus.OK,
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "state": "active" if host.run.active else "idle",
                            "run_id": host.run.active_run_id,
                            "runtime_run_id": host.run.runtime_run_id,
                        },
                    )
                    return
                if self.path == "/approvals":
                    if not self._authorized():
                        return
                    self._json(HTTPStatus.OK, {"pending": host.pending_approvals()})
                    return
                if self.path != "/events":
                    self._not_found()
                    return
                if not self._authorized():
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                subscription = host.broker.subscribe()
                try:
                    self._stream_events(subscription)
                except OSError:
                    return
                finally:
                    subscription.close()

            def _stream_events(self, subscription: Subscription) -> None:
                while not subscription.closed:
                    dropped = subscription.take_dropped()
                    if dropped:
                        self._sse(encode_frame("error", {"dropped": dropped}))
                    event = subscription.get(timeout=host.keepalive_seconds)
                    if event is None:
                        if subscription.closed:
                            return
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    if isinstance(event, ApprovalRequested):
                        self._sse(encode_frame("approval_requested", event.__dict__))
                    else:
                        self._sse(encode_frame("event", encode_event(event)))

            def _sse(self, frame: str) -> None:
                self.wfile.write(f"data: {frame}\n\n".encode("utf-8"))
                self.wfile.flush()

            def do_POST(self) -> None:
                if self.path not in ("/prompt", "/stop", "/approval"):
                    self._not_found()
                    return
                if not self._authorized():
                    return
                kind = self.path.removeprefix("/")
                try:
                    request = decode_request(kind, self._read_object())
                except ProtocolError as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                if kind == "prompt":
                    try:
                        run_id = host.run.start(request.prompt)
                    except RunActiveError as exc:
                        self._json(HTTPStatus.CONFLICT, {"error": str(exc), "run_id": exc.run_id})
                        return
                    self._json(HTTPStatus.OK, {"accepted": True, "run_id": run_id})
                    return
                if kind == "approval":
                    if not host.run.approvals.resolve(
                        request.approval_id, allowed=request.allowed, reason=request.reason
                    ):
                        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown approval id"})
                        return
                    self._json(HTTPStatus.OK, {"resolved": True})
                    return
                host.run.stop()
                self._json(HTTPStatus.OK, {"accepted": True})

        return Handler
