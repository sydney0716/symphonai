"""Durable storage for spec loops owned by a project."""

from __future__ import annotations

from symphonai_api.paths import utc_iso

import json
from pathlib import Path
import re
import secrets
from typing import TYPE_CHECKING
from symphonai_api.paths import write_atomic


if TYPE_CHECKING:
    from symphonai_host.spec_loop import Loop


_ID_PATTERN = re.compile(r"loop_[0-9a-f]{12}\Z")


def _slug(goal: str, loop_id: str) -> str:
    words = re.findall(r"[a-z0-9]+", goal.lower())
    return "-".join(words)[:40].rstrip("-") or f"loop-{loop_id.removeprefix('loop_')[:6]}"


class LoopStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.loops_root = self.project_root / ".symphonai" / "loops"

    @staticmethod
    def validate_id(loop_id: str) -> None:
        if not isinstance(loop_id, str) or _ID_PATTERN.fullmatch(loop_id) is None:
            raise KeyError(loop_id)

    def _folder(self, loop_id: str) -> Path:
        self.validate_id(loop_id)
        return self.loops_root / loop_id

    def create(self, goal: str) -> Loop:
        from symphonai_host.spec_loop import Loop
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
        now = utc_iso()
        (folder / ".gitignore").write_text("*\n", encoding="utf-8")
        (folder / "specs").mkdir()
        (folder / "reports").mkdir()
        loop = Loop.from_payload({
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
        })
        self.save(loop)
        return loop

    def _read(self, loop_id: str) -> dict:
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

    def load_record(self, loop_id: str):
        from symphonai_host.spec_loop import Loop
        return Loop.from_payload(self._read(loop_id))

    def load(self, loop_id: str) -> dict:
        return self.load_record(loop_id).to_payload()

    def records(self):
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
                loops.append(self.load_record(folder.name))
            except (KeyError, ValueError):
                continue
        return sorted(loops, key=lambda loop: loop.created_at, reverse=True)

    def list(self) -> list[dict]:
        return [loop.to_payload() for loop in self.records()]

    def save(self, loop: Loop) -> None:
        from symphonai_host.spec_loop import Loop
        if not isinstance(loop, Loop):
            raise ValueError("loop must be a Loop")
        loop.updated_at = utc_iso()
        payload = loop.to_payload()
        folder = self._folder(payload.get("id"))
        folder.mkdir(parents=True, exist_ok=True)
        write_atomic(
            folder / "loop.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            mode=0o600, fsync=True,
        )

    def set_state(self, loop: Loop, state: str, reason: str = "") -> Loop:
        from symphonai_host.spec_loop import LoopState, STATE_RULES
        current = loop.state
        allowed = STATE_RULES["loop"].get(current, (set(), set()))[0]
        if not isinstance(state, str) or state not in allowed:
            raise ValueError(f"invalid loop state transition: {str(current) if isinstance(current, LoopState) else current!r} -> {state!r}")
        if state != "needs_attention":
            loop.failure_stage = None
            loop._present.discard("failure_stage")
        loop.state = LoopState(state)
        loop.reason = reason
        self.save(loop)
        return loop

    def path(self, loop_id: str, relative: str | Path) -> Path:
        folder = self._folder(loop_id).resolve()
        candidate = (folder / relative).resolve()
        if not candidate.is_relative_to(folder):
            raise ValueError("path escapes loop folder")
        return candidate
