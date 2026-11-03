"""Durable per-project roadmap storage."""

from __future__ import annotations

import json
from pathlib import Path
import re
import secrets
from symphonai_api.paths import write_atomic


_ID_PATTERN = re.compile(r"goal_[0-9a-f]{12}\Z")


class InvalidRoadmap(ValueError):
    pass


class RoadmapStore:
    def __init__(self, project_root: Path) -> None:
        self.path = Path(project_root).expanduser().resolve() / ".symphonai" / "roadmap.json"

    def load(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "goals": []}
        try:
            roadmap = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidRoadmap(str(exc)) from None
        if (
            not isinstance(roadmap, dict)
            or set(roadmap) != {"schema_version", "goals"}
            or type(roadmap.get("schema_version")) is not int
            or roadmap.get("schema_version") != 1
            or not isinstance(roadmap.get("goals"), list)
        ):
            raise InvalidRoadmap("expected schema_version 1 and a goals list")
        ids = set()
        for goal in roadmap["goals"]:
            if (
                not isinstance(goal, dict)
                or set(goal) != {"id", "goal", "loop_id", "done"}
                or not isinstance(goal.get("id"), str)
                or _ID_PATTERN.fullmatch(goal["id"]) is None
                or goal["id"] in ids
                or not isinstance(goal.get("goal"), str)
                or not goal["goal"].strip()
                or (goal.get("loop_id") is not None and not isinstance(goal["loop_id"], str))
                or not isinstance(goal.get("done"), bool)
            ):
                raise InvalidRoadmap("goal entries must have id, goal, loop_id, and done")
            if goal["loop_id"] is not None and not re.fullmatch(r"loop_[0-9a-f]{12}", goal["loop_id"]):
                raise InvalidRoadmap("goal loop_id must be null or a loop id")
            ids.add(goal["id"])
        return roadmap

    def save(self, roadmap: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(
            self.path, json.dumps(roadmap, ensure_ascii=False, indent=2) + "\n",
            mode=0o600, fsync=True,
        )

    @staticmethod
    def add(roadmap: dict, goal: str) -> dict:
        while True:
            goal_id = f"goal_{secrets.token_hex(6)}"
            if all(item["id"] != goal_id for item in roadmap["goals"]):
                break
        item = {"id": goal_id, "goal": goal, "loop_id": None, "done": False}
        roadmap["goals"].append(item)
        return item

    @staticmethod
    def find(roadmap: dict, goal_id: str) -> dict | None:
        return next((item for item in roadmap["goals"] if item["id"] == goal_id), None)

    def mark_loop_done(self, loop_id: str) -> None:
        roadmap = self.load()
        goal = next((item for item in roadmap["goals"] if item["loop_id"] == loop_id), None)
        if goal is not None and not goal["done"]:
            goal["done"] = True
            self.save(roadmap)

    def record_loop(self, loop_id: str, text: str, goal_id: str | None = None) -> None:
        roadmap = self.load()
        if any(goal["loop_id"] == loop_id for goal in roadmap["goals"]):
            return
        goal = self.find(roadmap, goal_id) if goal_id is not None else self.add(roadmap, text)
        if goal is None:
            raise KeyError(goal_id)
        goal["loop_id"] = loop_id
        goal["done"] = False
        self.save(roadmap)

    def reply(self, roadmap: dict) -> dict:
        from symphonai_host.loops import LoopStore

        loops = LoopStore(self.path.parents[1])
        result = []
        for item in roadmap["goals"]:
            state = "not_started"
            if item["loop_id"] is not None:
                try:
                    state = loops.load(item["loop_id"]).get("state", "unknown")
                except KeyError:
                    state = "done" if item["done"] else "unknown"
            result.append({**item, "state": state})
        return {"goals": result}
