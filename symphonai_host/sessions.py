"""Read-only session discovery for the loopback host."""

from __future__ import annotations

from pathlib import Path

from symphonai_api.session import (
    SessionStore,
    TranscriptError,
    classify_run,
    load_run,
    read_records,
)


def list_sessions(root: Path) -> list[dict]:
    """List every session directory, retaining damaged entries for recovery."""
    root = Path(root)
    if not root.is_dir():
        return []
    sessions: list[dict] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        item = {
            "run_id": directory.name,
            "title": None,
            "created_at": None,
            "updated_at": None,
            "stopped_reason": None,
            "parent_run_id": None,
            "state": "unreadable",
            "message_count": 0,
        }
        try:
            store = SessionStore.open(root, directory.name)
            meta = store.read_meta()
            item.update({
                key: meta.get(key)
                for key in ("run_id", "title", "created_at", "updated_at", "stopped_reason", "parent_run_id")
            })
            loaded = load_run(store)
            records, _ = read_records(store.directory / "run.jsonl")
            item["state"] = classify_run(loaded, records).state.value
            item["message_count"] = len(loaded.messages)
        except (OSError, TranscriptError, ValueError, KeyError, TypeError):
            pass
        sessions.append(item)
    return sorted(sessions, key=lambda item: item["updated_at"] or "", reverse=True)
