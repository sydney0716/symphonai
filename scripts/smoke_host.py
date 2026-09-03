#!/usr/bin/env python3
"""Headless loopback smoke for the plain-text host client."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from symphonai_api.events import RunFinished
from symphonai_api.models import Message, ModelResponse, Role, ToolCall
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_host.client import HostAddress, HostClient, HostClientError
from symphonai_host.protocol import decode_event
from symphonai_host.server import HostServer


class _SmokeProvider(FakeModelProvider):
    """Replay an approval turn, then wait until the stop request cancels it."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__(responses)
        self.block = False
        self.blocking_started = threading.Event()

    def create_response(self, request, *, cancel=None):
        if self.block:
            self.blocking_started.set()
            if cancel is None or not cancel.wait(5):
                raise RuntimeError("stop did not cancel the second prompt")
            cancel.raise_if_cancelled()
        return super().create_response(request, cancel=cancel)


def main(binary: str | None = None) -> None:
    if binary:
        _smoke_binary(Path(binary))
        return
    _smoke_interpreter()


def _smoke_binary(binary: Path) -> None:
    """Drive the frozen host through a local OpenAI-compatible provider."""
    binary = binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"binary launch failed: {binary} is missing")
    with tempfile.TemporaryDirectory() as directory:
        with _ScriptedProvider() as provider:
            _require_loopback_base_url(provider.base_url)
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.endswith("_API_KEY")
            }
            # This is an inert local value, not a developer credential. The
            # real provider requires a non-empty header before it will make a
            # request, and the configured base URL is the loopback fixture.
            environment["OPENAI_API_KEY"] = "sidecar-smoke-key"
            environment["SYMPHONAI_SESSIONS_DIR"] = str(Path(directory) / "sessions")
            try:
                child = subprocess.Popen(
                    [str(binary), "--repo-root", directory, "--base-url", provider.base_url],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    cwd=directory,
                    env=environment,
                )
            except OSError as exc:
                raise RuntimeError(f"binary launch failed: {exc}") from None
            client: HostClient | None = None
            try:
                address = _read_binary_handshake(child)
                client = HostClient(address)
                finished = threading.Event()
                approved = threading.Event()
                second_approval = threading.Event()
                failures: list[str] = []
                approval_count = 0

                def consume() -> None:
                    nonlocal approval_count
                    try:
                        for kind, payload in client.events():
                            if kind == "approval_requested":
                                approval_count += 1
                                if approval_count == 1:
                                    if not client.send_approval(
                                        payload["approval_id"], allowed=True
                                    ).get("resolved"):
                                        failures.append("approval reply was rejected")
                                        return
                                    approved.set()
                                else:
                                    second_approval.set()
                            elif kind == "event" and isinstance(
                                decode_event(payload), RunFinished
                            ):
                                finished.set()
                    except Exception as exc:
                        failures.append(f"event stream failed: {exc}")

                threading.Thread(target=consume, daemon=True).start()
                if client.health().get("state") != "idle":
                    raise RuntimeError("connect through handshake failed: host was not idle")
                print("OK:   client connected through handshake address")
                if not client.send_prompt("smoke").get("accepted"):
                    raise RuntimeError("prompt failed: host rejected the prompt")
                if not approved.wait(5):
                    raise RuntimeError(f"approval failed: {failures}")
                print("OK:   client answered the approval")
                if not finished.wait(5):
                    raise RuntimeError(f"run completion failed: {failures}")
                if failures:
                    raise RuntimeError(f"run completion failed: {failures[0]}")
                if (Path(directory) / "approved.txt").read_text(encoding="utf-8") != "ok":
                    raise RuntimeError("run completion failed: approved tool did not execute")
                print("OK:   approved run finished")
                sessions = client.list_sessions()
                if len(sessions) != 1 or client.open_session(sessions[0]["run_id"])["replayed"] < 2:
                    raise RuntimeError("reopen failed: finished session did not replay")
                print("OK:   client reopened the finished session")
                if not client.send_prompt("continue then stop").get("accepted"):
                    raise RuntimeError("continue failed: host rejected the prompt")
                if not second_approval.wait(5):
                    raise RuntimeError(f"stop failed: no active approval arrived: {failures}")
                if not client.stop().get("accepted"):
                    raise RuntimeError("stop failed: host rejected the stop request")
                deadline = time.monotonic() + 5
                while client.health().get("state") == "active" and time.monotonic() < deadline:
                    time.sleep(0.05)
                if client.health().get("state") != "idle":
                    raise RuntimeError("stop failed: active run did not finish")
                print("OK:   client stopped an active host run")
                if provider.requests != 3:
                    raise RuntimeError(
                        f"provider routing failed: expected 3 loopback requests, got {provider.requests}"
                    )
            finally:
                if client is not None:
                    client.close()
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def _require_loopback_base_url(base_url: str) -> None:
    """Refuse a smoke run before a synthetic key can reach the internet."""
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None:
        raise RuntimeError(f"local-only guard failed: {base_url!r} is not a 127.0.0.1 URL")


def _read_binary_handshake(child: subprocess.Popen[str]) -> HostAddress:
    if child.stdout is None:
        raise RuntimeError("handshake failed: binary has no stdout pipe")
    line: list[str] = []
    ready = threading.Event()

    def read() -> None:
        line.append(child.stdout.readline())
        ready.set()

    threading.Thread(target=read, daemon=True).start()
    if not ready.wait(5):
        raise RuntimeError("handshake failed: binary did not emit a line within 5 seconds")
    try:
        return HostAddress.from_handshake(line[0])
    except HostClientError as exc:
        raise RuntimeError(f"handshake failed: {exc}") from None


class _ScriptedProvider:
    """Three local OpenAI-compatible replies for the binary smoke scenario."""

    def __init__(self) -> None:
        self.requests = 0
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_type())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __enter__(self) -> "_ScriptedProvider":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/v1/chat/completions":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                if self.headers.get("Authorization") != "Bearer sidecar-smoke-key":
                    self.send_error(HTTPStatus.UNAUTHORIZED)
                    return
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                provider.requests += 1
                payload = json.dumps(provider._response()).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return

        return Handler

    def _response(self) -> dict:
        if self.requests == 1:
            return self._tool_response("smoke-write", "approved.txt", "ok")
        if self.requests == 2:
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }],
                "usage": {},
            }
        if self.requests == 3:
            return self._tool_response("smoke-stop", "stopped.txt", "not written")
        return {"choices": [], "usage": {}}

    @staticmethod
    def _tool_response(call_id: str, path: str, content: str) -> dict:
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps({"path": path, "content": content}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        }


def _smoke_interpreter() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provider = _SmokeProvider([
            ModelResponse(Message(
                Role.ASSISTANT,
                tool_calls=[ToolCall("smoke-write", "write_file", {"path": "approved.txt", "content": "ok"})],
            )),
            ModelResponse(Message(Role.ASSISTANT, "done")),
        ])
        host = HostServer(
            provider,
            PermissionPolicy(repo_root=root, mode="prompt"),
            sessions_root=root / "sessions",
        )
        host.start()
        client = HostClient(HostAddress(host.port, host.token))
        finished = threading.Event()
        approved = threading.Event()
        failed: list[str] = []

        def consume() -> None:
            try:
                for kind, payload in client.events():
                    if kind == "approval_requested":
                        if not client.send_approval(payload["approval_id"], allowed=True).get("resolved"):
                            failed.append("approval reply was rejected")
                            return
                        approved.set()
                    elif kind == "event" and isinstance(decode_event(payload), RunFinished):
                        finished.set()
                        return
            except Exception as exc:
                failed.append(f"event stream failed: {exc}")

        stream = threading.Thread(target=consume, daemon=True)
        stream.start()
        try:
            if client.health().get("state") != "idle":
                raise RuntimeError("health did not report idle host")
            print("OK:   client connected through handshake address")
            if not client.send_prompt("smoke").get("accepted"):
                raise RuntimeError("prompt was not accepted")
            if not approved.wait(5):
                raise RuntimeError(f"approval did not arrive: {failed}")
            print("OK:   client answered the approval")
            if not finished.wait(5):
                raise RuntimeError(f"run did not finish: {failed}")
            if failed:
                raise RuntimeError(failed[0])
            deadline = time.monotonic() + 5
            while client.health().get("state") == "active" and time.monotonic() < deadline:
                time.sleep(0.05)
            if (root / "approved.txt").read_text(encoding="utf-8") != "ok":
                raise RuntimeError("approved tool did not execute")
            print("OK:   approved run finished")
            sessions = client.list_sessions()
            if len(sessions) != 1 or client.open_session(sessions[0]["run_id"])["replayed"] < 2:
                raise RuntimeError("finished session did not reopen with its history")
            print("OK:   client reopened the finished session")
            provider.block = True
            if not client.send_prompt("continue then stop").get("accepted"):
                raise RuntimeError("continuation prompt was not accepted")
            if not provider.blocking_started.wait(5):
                raise RuntimeError("second prompt did not become active")
            if not client.stop().get("accepted"):
                raise RuntimeError("stop request was not accepted")
            deadline = time.monotonic() + 5
            while client.health().get("state") == "active" and time.monotonic() < deadline:
                time.sleep(0.05)
            if client.health().get("state") != "idle":
                raise RuntimeError("stop did not finish the active run")
            print("OK:   client stopped an active host run")
        finally:
            client.close()
            host.close()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(); parser.add_argument("--binary")
        main(parser.parse_args().binary)
    except Exception as exc:
        print(f"FAIL: host client smoke: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
