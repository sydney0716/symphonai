"""Durable storage for spec loops owned by a project."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
from symphonai_api.paths import write_atomic


_ID_PATTERN = re.compile(r"loop_[0-9a-f]{12}\Z")
_LOOP_TRANSITIONS = {
    "planning": {"awaiting_plan", "needs_attention", "cancelled"},
    "awaiting_plan": {"planning", "running", "cancelled"},
    "running": {"validating", "needs_attention", "cancelled"},
    "validating": {"awaiting_commit", "needs_attention", "cancelled"},
    "awaiting_commit": {"done", "cancelled"},
    "needs_attention": {"planning", "running", "validating", "cancelled"},
    "done": set(),
    "cancelled": set(),
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(goal: str, loop_id: str) -> str:
    words = re.findall(r"[a-z0-9]+", goal.lower())
    return "-".join(words)[:40].rstrip("-") or f"loop-{loop_id.removeprefix('loop_')[:6]}"


class LoopStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.loops_root = self.project_root / ".symphonai" / "loops"

    def _folder(self, loop_id: str) -> Path:
        if not isinstance(loop_id, str) or _ID_PATTERN.fullmatch(loop_id) is None:
            raise KeyError(loop_id)
        return self.loops_root / loop_id

    def create(self, goal: str) -> dict:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must not be blank")
        self.loops_root.mkdir(parents=True, exist_ok=True)
        while True:
            loop_id = f"loop_{secrets.token_hex(6)}"
            folder = self._folder(loop_id)
            try:
                folder.mkdir()
                break
            except FileExistsError:
                continue
        now = _timestamp()
        (folder / ".gitignore").write_text("*\n", encoding="utf-8")
        (folder / "specs").mkdir()
        (folder / "reports").mkdir()
        loop = {
            "schema_version": 1,
            "id": loop_id,
            "goal": goal,
            "slug": _slug(goal, loop_id),
            "state": "planning",
            "reason": "",
            "created_at": now,
            "updated_at": now,
            "branch": None,
            "planner_session_id": None,
            "lanes": [],
            "cards": [],
        }
        self.save(loop)
        return loop

    def load(self, loop_id: str) -> dict:
        folder = self._folder(loop_id)
        if not folder.exists():
            raise KeyError(loop_id)
        path = folder / "loop.json"
        try:
            loop = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid loop record: {loop_id}") from exc
        if not isinstance(loop, dict) or loop.get("id") != loop_id:
            raise ValueError(f"invalid loop record: {loop_id}")
        return loop

    def list(self) -> list[dict]:
        if not self.loops_root.is_dir():
            return []
        try:
            folders = self.loops_root.iterdir()
        except OSError:
            return []
        loops = []
        for folder in folders:
            if _ID_PATTERN.fullmatch(folder.name) is None or not folder.is_dir():
                continue
            try:
                loops.append(self.load(folder.name))
            except (KeyError, ValueError):
                continue
        return sorted(loops, key=lambda loop: loop.get("created_at", ""), reverse=True)

    def save(self, loop: dict) -> None:
        if not isinstance(loop, dict):
            raise ValueError("loop must be an object")
        folder = self._folder(loop.get("id"))
        folder.mkdir(parents=True, exist_ok=True)
        loop["updated_at"] = _timestamp()
        write_atomic(
            folder / "loop.json", json.dumps(loop, ensure_ascii=False, indent=2) + "\n",
            mode=0o600, fsync=True,
        )

    def set_state(self, loop: dict, state: str, reason: str = "") -> dict:
        current = loop.get("state")
        allowed = _LOOP_TRANSITIONS.get(current, set()) if isinstance(current, str) else set()
        if not isinstance(state, str) or state not in allowed:
            raise ValueError(f"invalid loop state transition: {current!r} -> {state!r}")
        if state != "needs_attention":
            loop.pop("failure_stage", None)
        loop["state"] = state
        loop["reason"] = reason
        self.save(loop)
        return loop

    def path(self, loop_id: str, relative: str | Path) -> Path:
        folder = self._folder(loop_id).resolve()
        candidate = (folder / relative).resolve()
        if not candidate.is_relative_to(folder):
            raise ValueError("path escapes loop folder")
        return candidate
