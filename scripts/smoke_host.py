#!/usr/bin/env python3
"""Headless loopback smoke for the plain-text host client."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from symphonai_api.events import RunFinished
from symphonai_api.models import Message, ModelResponse, Role, ToolCall
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_host.client import HostAddress, HostClient
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


def main() -> None:
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
        main()
    except Exception as exc:
        print(f"FAIL: host client smoke: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
