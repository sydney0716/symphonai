"""Plain-text interactive client for a SymphonAI host."""

from __future__ import annotations

import queue
import select
import sys
import threading

from symphonai_host.client import HostClient


def run(client: HostClient) -> None:
    approvals: queue.Queue[dict] = queue.Queue()
    seen: set[str] = set()
    seen_lock = threading.Lock()

    def enqueue_approval(approval: dict) -> None:
        approval_id = approval.get("approval_id")
        if not isinstance(approval_id, str):
            print("approval_requested: invalid approval id", file=sys.stderr)
            return
        with seen_lock:
            if approval_id in seen:
                return
            seen.add(approval_id)
        approvals.put(approval)

    def describe(kind: str, payload: dict) -> str:
        parts = [payload.get("type", kind)]
        for field in ("run_id", "agent_id", "turn_id", "tool_call_id", "approval_id"):
            value = payload.get(field)
            if isinstance(value, str):
                parts.append(f"{field}={value[:8]}")
        for field in ("tool_name", "text", "error", "stopped_reason"):
            value = payload.get(field)
            if value:
                parts.append(f"{field}={value}")
        return " ".join(str(part) for part in parts)

    def reader() -> None:
        try:
            for kind, payload in client.events():
                if kind == "approval_requested":
                    print(f"approve {payload.get('operation')} {payload.get('target')}? [y/N]")
                    enqueue_approval(payload)
                elif kind == "error" and payload.get("dropped"):
                    print(describe(kind, payload))
                    for approval in client._request("GET", "/approvals").get("pending", []):
                        enqueue_approval(approval)
                else:
                    print(describe(kind, payload))
        except Exception as exc:
            print(f"events ended: {exc}", file=sys.stderr)

    threading.Thread(target=reader, daemon=True).start()
    outstanding: dict | None = None
    while True:
        if outstanding is None:
            try:
                outstanding = approvals.get_nowait()
            except queue.Empty:
                pass
        # POSIX-only: this proof client targets macOS/Linux; phase 18 owns UI portability.
        readable, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not readable:
            continue
        line = sys.stdin.readline()
        if line == "":
            client.stop()
            return
        line = line.rstrip("\n")
        if outstanding is not None:
            client.send_approval(outstanding["approval_id"], allowed=line.lower() in {"y", "yes"})
            outstanding = None
        elif not line:
            continue
        elif line == "/quit":
            client.stop()
            return
        elif line == "/stop":
            client.stop()
        else:
            client.send_prompt(line)
