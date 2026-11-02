"""Session discovery and retention for the loopback host."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from symphonai_api.session import (
    SessionStore,
    TranscriptError,
    classify_run,
    load_run,
    read_records,
)
from symphonai_api.serialization import message_from_json
from symphonai_api.models import Role


DEFAULT_CLEANUP_PERIOD_DAYS = 30
DEFAULT_HISTORY_LIMIT = 100


def prompt_history(root: Path, repo_root: Path, *, limit: int = DEFAULT_HISTORY_LIMIT) -> list[str]:
    """Return recent person prompts for this repository from session transcripts."""
    root = Path(root)
    if not root.is_dir():
        return []
    sessions = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            if not isinstance(meta, dict) or meta.get("repo_root") != str(repo_root):
                continue
            updated_at = meta.get("updated_at")
            sessions.append((updated_at if isinstance(updated_at, str) else "", directory))
        except (OSError, ValueError, TypeError, UnicodeError):
            continue
    sessions.sort(key=lambda item: item[0], reverse=True)

    prompts = []
    previous = None
    for _, directory in sessions:
        try:
            records, _ = read_records(directory / "run.jsonl")
            session_prompts = []
            for record in reversed(records):
                if record.get("type") != "message":
                    continue
                message = message_from_json(record["data"])
                if message.role != Role.USER:
                    continue
                text = message.text
                if (
                    not text
                    or text.startswith("Goal check failed (round ")
                    or (
                        text.startswith("Round ")
                        and " ended without the goal reported complete." in text
                    )
                ):
                    continue
                session_prompts.append(text)
        except (OSError, TranscriptError, ValueError, KeyError, TypeError, UnicodeError):
            continue
        for text in session_prompts:
            if text == previous:
                continue
            prompts.append(text)
            previous = text
            if len(prompts) >= limit:
                return prompts
    return prompts


def prune_sessions(root: Path, *, period_days: int, now: datetime) -> int:
    """Remove dated session directories strictly older than the cutoff."""
    if period_days <= 0:
        return 0
    root = Path(root)
    if not root.is_dir():
        return 0
    cutoff = now - timedelta(days=period_days)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    try:
        directories = list(root.iterdir())
    except OSError:
        return 0
    removed = 0
    for directory in directories:
        try:
            if not directory.is_dir():
                continue
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            updated_at = meta.get("updated_at") if isinstance(meta, dict) else None
            if not isinstance(updated_at, str):
                continue
            updated = datetime.fromisoformat(updated_at)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            if updated < cutoff:
                shutil.rmtree(directory)
                removed += 1
        except (OSError, ValueError, TypeError, UnicodeError):
            continue
    return removed


def list_sessions(
    root: Path, *, limit: int | None = None, activity: dict[str, str] | None = None,
    query: str | None = None, archived: bool = False,
) -> list[dict]:
    """List every session directory, retaining damaged entries for recovery."""
    root = Path(root)
    if not root.is_dir():
        return []
    sessions: list[tuple[dict, SessionStore | None]] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        if directory.name == "loops":
            continue
        item = {
            "run_id": directory.name,
            "title": None,
            "created_at": None,
            "updated_at": None,
            "stopped_reason": None,
            "parent_run_id": None,
            "parent_session_id": None,
            "repo_root": "",
            "archived": False,
            "state": "unreadable",
            "message_count": 0,
            "activity": (activity or {}).get(directory.name, "idle"),
            "unread": False,
        }
        store = None
        try:
            store = SessionStore.open(root, directory.name)
            meta = store.read_meta()
            item.update({
                key: meta.get(key)
                for key in ("run_id", "title", "created_at", "updated_at", "stopped_reason", "parent_run_id", "parent_session_id")
            })
            item["archived"] = meta.get("archived") is True
            item["unread"] = not meta.get("viewed_at") and "spec_run" not in meta
            repo_root = meta.get("repo_root")
            item["repo_root"] = repo_root if isinstance(repo_root, str) else ""
        except (OSError, TranscriptError, ValueError, KeyError, TypeError):
            store = None
        sessions.append((item, store))
    sessions = [entry for entry in sessions if entry[0]["archived"] is archived]
    sessions.sort(key=lambda entry: entry[0]["updated_at"] or "", reverse=True)
    if query:
        sessions = [(item, store) for item, store in sessions
                    if matching_snippets(item["title"], store, query)]
    selected = sessions if limit is None else sessions[:limit]
    for item, store in selected:
        if store is None:
            continue
        try:
            loaded = load_run(store)
            records, _ = read_records(store.directory / "run.jsonl")
            item["state"] = classify_run(loaded, records).state.value
            item["message_count"] = len(loaded.messages)
            meta = store.read_meta()
            viewed_at = meta.get("viewed_at")
            ended_at = next((record.get("ts") for record in reversed(records)
                             if record.get("type") in ("run_finished", "run_failed")), None)
            if item["stopped_reason"] and isinstance(viewed_at, str) and isinstance(ended_at, str):
                item["unread"] = datetime.fromisoformat(ended_at) > datetime.fromisoformat(viewed_at)
        except (OSError, TranscriptError, ValueError, KeyError, TypeError):
            pass
    for item, _ in selected:
        item.pop("archived", None)
    return [item for item, _ in selected]


def matching_snippets(title: str | None, store: SessionStore | None, query: str) -> list[str]:
    """Share sidebar and agent matching over titles and conversation text."""
    needle = query.casefold()
    snippets = []

    def add(text):
        index = text.casefold().find(needle)
        if index >= 0:
            start = max(0, index - 80)
            snippets.append(text[start:start + 200])

    if isinstance(title, str):
        add(title)
    if store is not None and len(snippets) < 3:
        try:
            records, _ = read_records(store.directory / "run.jsonl")
            for record in records:
                if record.get("type") == "message":
                    message = message_from_json(record["data"])
                    if message.role in (Role.USER, Role.ASSISTANT):
                        add(message.text)
                        if len(snippets) >= 3:
                            break
        except (OSError, TranscriptError, ValueError, KeyError, TypeError, UnicodeError):
            pass
    return snippets
