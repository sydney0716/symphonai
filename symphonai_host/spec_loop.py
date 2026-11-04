"""Typed, durable spec-loop state and project loop orchestration."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from types import UnionType
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, get_args, get_origin, get_type_hints

from symphonai_api.cost import UsageTotals
from symphonai_api.events import RunFailed, RunFinished
from symphonai_api.models import Role
from symphonai_api.roles import failure_reviewer_prompt, loop_planner_prompt, spec_reviewer_prompt
from symphonai_api.session import SessionError, SessionStore, TranscriptError, load_run
from symphonai_api.tools.filesystem import MAX_READ_BYTES
from symphonai_api.tools.shell import shell_environment, shell_executable
from symphonai_api.worktree import WorktreeDiff, WorktreeError, _linked_paths, create_worktree, remove_worktree, stage_worktree, worktree_diff
from symphonai_host.loops import LoopStore
from symphonai_host.goal import Goal
from symphonai_host.roadmap import InvalidRoadmap, RoadmapStore

if TYPE_CHECKING:
    from symphonai_host.run import HostRun
    from symphonai_api.leader import Leader


class LoopState(str, Enum):
    PLANNING = "planning"
    AWAITING_PLAN = "awaiting_plan"
    RUNNING = "running"
    VALIDATING = "validating"
    AWAITING_COMMIT = "awaiting_commit"
    NEEDS_ATTENTION = "needs_attention"
    DONE = "done"
    CANCELLED = "cancelled"

    def __str__(self) -> str:
        return self.value


class CardState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    REVIEW = "review"
    DONE = "done"
    FAILED = "failed"
    DROPPED = "dropped"

    def __str__(self) -> str:
        return self.value


# One state table supplies transitions and operation preconditions. Asynchronous
# callbacks still ignore obsolete states; user operations raise LoopStateError.
STATE_RULES = {
    "loop": {
        LoopState.PLANNING: ({"awaiting_plan", "needs_attention", "cancelled"}, {"cancel"}),
        LoopState.AWAITING_PLAN: ({"planning", "running", "cancelled"}, {"approve", "drop", "revise", "cancel"}),
        LoopState.RUNNING: ({"validating", "needs_attention", "cancelled"}, {"advance", "cancel"}),
        LoopState.VALIDATING: ({"awaiting_commit", "needs_attention", "cancelled"}, {"validate", "cancel"}),
        LoopState.AWAITING_COMMIT: ({"done", "cancelled"}, {"commit", "cancel"}),
        LoopState.NEEDS_ATTENTION: ({"planning", "running", "validating", "cancelled"}, {"resume", "cancel"}),
        LoopState.DONE: (set(), set()),
        LoopState.CANCELLED: (set(), set()),
    },
    "card": {
        CardState.PLANNED: ({"running", "failed", "dropped"}, set()),
        CardState.RUNNING: ({"review", "failed", "planned"}, set()),
        CardState.REVIEW: ({"done", "failed", "planned"}, set()),
        CardState.FAILED: ({"planned", "failed"}, set()),
        CardState.DONE: ({"dropped"}, set()),
        CardState.DROPPED: (set(), set()),
    },
}


def _matches_type(value, expected) -> bool:
    origin = get_origin(expected)
    if origin is UnionType:
        return any(_matches_type(value, member) for member in get_args(expected))
    if origin is list:
        return isinstance(value, list) and all(_matches_type(item, get_args(expected)[0]) for item in value)
    if expected is dict:
        return isinstance(value, dict)
    if expected in (str, int, bool, type(None)):
        return type(value) is expected
    return isinstance(value, expected)


@dataclass
class Record:
    # Preserve absent optional fields and extensions from older/newer loop files.
    _present: set[str] = field(default_factory=set, init=False, repr=False)
    _extra: dict = field(default_factory=dict, init=False, repr=False)

    def __setattr__(self, name, value):
        if name == "state":
            value = (LoopState if isinstance(self, Loop) else CardState)(value)
        if name == "state" and isinstance(self, Card) and "state" in self.__dict__:
            current = self.state
            if value != current and value not in STATE_RULES["card"][current][0]:
                raise ValueError(f"invalid card state transition: {current!r} -> {value!r}")
        object.__setattr__(self, name, value)
        if not name.startswith("_") and "_present" in self.__dict__:
            self._present.add(name)

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError(f"{cls.__name__.lower()} must be an object")
        names = {item.name for item in fields(cls) if not item.name.startswith("_")}
        values = {key: value for key, value in payload.items() if key in names}
        if cls is Loop:
            cards = values.get("cards", [])
            if not isinstance(cards, list):
                raise ValueError("loop cards must be a list")
            values["cards"] = [Card.from_payload(card) for card in cards]
        hints = get_type_hints(cls)
        for key, value in values.items():
            if key == "state":
                value = (LoopState if cls is Loop else CardState)(value)
                values[key] = value
            if not _matches_type(value, hints[key]):
                raise ValueError(f"invalid {cls.__name__.lower()} field: {key}")
        result = cls(**values)
        result._present = set(payload) & names
        result._extra = {key: value for key, value in payload.items() if key not in names}
        return result

    def to_payload(self):
        result = dict(self._extra)
        for item in fields(self):
            if item.name.startswith("_"):
                continue
            value = getattr(self, item.name)
            # A previously absent list can acquire entries during a resumed run.
            if item.name not in self._present and not (isinstance(value, list) and value):
                continue
            if isinstance(value, Enum):
                value = value.value
            elif isinstance(value, list):
                value = [entry.to_payload() if isinstance(entry, Record) else entry for entry in value]
            result[item.name] = value
        return result

    def get(self, key, default=None):
        if key in self._present:
            return getattr(self, key) if hasattr(self, key) else self._extra[key]
        value = getattr(self, key, None)
        return value if isinstance(value, list) and value else default

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return self._extra[key]

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self._extra[key] = value

    def __contains__(self, key):
        return key in self._present or key in self._extra

    def setdefault(self, key, default):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, default=None):
        result = self.get(key, default)
        self._present.discard(key)
        if hasattr(self, key):
            object.__setattr__(self, key, None)
        self._extra.pop(key, None)
        return result

    def update(self, **values):
        for key, value in values.items():
            self[key] = value


@dataclass
class Card(Record):
    id: str = ""
    title: str = ""
    spec: str = ""
    report: str | None = None
    lane: int = 0
    after: list[str] = field(default_factory=list)
    state: CardState = CardState.PLANNED
    rounds: int = 0
    follow_up_of: str | None = None
    session_id: str | None = None
    previous_session_ids: list[str] = field(default_factory=list)
    commit: str | None = None


@dataclass
class Loop(Record):
    schema_version: int = 1
    id: str = ""
    goal: str = ""
    slug: str = ""
    state: LoopState = LoopState.PLANNING
    reason: str = ""
    created_at: str = ""
    updated_at: str = ""
    branch: str | None = None
    planner_session_id: str | None = None
    previous_planner_session_ids: list[str] = field(default_factory=list)
    lanes: list[dict] = field(default_factory=list)
    cards: list[Card] = field(default_factory=list)
    runner_pid: int | None = None
    runner_lock: bool = False
    base_commit: str | None = None
    summary: dict = field(default_factory=dict)
    failure_stage: str | None = None

    def allows(self, operation: str) -> bool:
        return operation in STATE_RULES["loop"][self.state][1]

    def require(self, operation: str, expected: str) -> None:
        if not self.allows(operation):
            from symphonai_host.run import LoopStateError
            raise LoopStateError(self.id, self.state.value, expected)


def _loop_git(cwd: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=cwd, capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(str(exc)) from exc
    if check and result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or f"git {' '.join(arguments)} failed")
    return result


class LoopController:
    def __init__(self, host: HostRun, store: LoopStore):
        self.host = host
        self.store = store
        self._loop_validation_cancel: dict[str, threading.Event] = {}
        self._loop_validation_processes: dict[str, subprocess.Popen] = {}
        self._loop_runner_lock_fds: dict[str, int] = {}

    def start_spec_run(self, spec: dict) -> tuple[str, str]:
        """Start an isolated implementation conversation for a parsed spec."""
        from symphonai_host.run import MAX_ACTIVE_CONVERSATIONS, RunActiveError

        with self.host._lock:
            if len(self.host._active_by_session) >= MAX_ACTIVE_CONVERSATIONS:
                raise RunActiveError(description=f"{MAX_ACTIVE_CONVERSATIONS} conversations are already running")
            policy = replace(self.host._base_policy, repo_root=self.host._repo_root, mode="allow")
            objective = f"{spec['text']}\n\nWrite your report at {spec['report']}."
            if spec["validation"]:
                check = ("/bin/sh", "-c", "set -e\n" + "\n".join(spec["validation"]))
            else:
                check = ()
            goal = Goal(objective, check, max_rounds=5)
            run_id = self.host.start(
                objective, _new_goal=goal, session_policy=policy,
                _session_meta={"spec_run": {
                    "spec": spec["path"], "report": spec["report"],
                    "kind": "implement", "worktree": "worktree", "role": "implementer",
                    **({"loop_id": spec["loop_id"], "card_id": spec["card_id"]}
                       if spec.get("loop_id") and spec.get("card_id") else {}),
                    **({"start_ref": spec["start_ref"]} if spec.get("start_ref") else {}),
                }},
                _title=f"Run {Path(spec['path']).name}",
            )
            return run_id, run_id


    def start_spec_review(self, session_id: str, *, failure: str | None = None) -> tuple[str, str]:
        from symphonai_host.run import RunActiveError

        from symphonai_host.spec_run import patch_digest

        with self.host._lock:
            source = SessionStore.open(self.host._sessions_root, session_id)
            try:
                source_meta = source.read_meta()
            finally:
                source.close()
            info = source_meta.get("spec_run")
            if (
                not isinstance(info, dict)
                or info.get("kind") != "implement"
                or not isinstance(info.get("loop_id"), str)
            ):
                raise KeyError(session_id)
            if info.get("state") not in ("finished", "blocked", "stopped"):
                raise RunActiveError("spec run has not finished")
            if session_id in self.host._active_by_session or session_id in self.host._goal_checks_by_session:
                raise RunActiveError("spec run is still active")
            prior_review = source_meta.get("review")
            if isinstance(prior_review, dict) and prior_review.get("session_id") in self.host._active_by_session:
                raise RunActiveError("review is already running")
            worktree = self.host._sessions_root / session_id / "worktree"
            diff = worktree_diff(worktree)
            baseline_files = {}
            for name in diff.files:
                file_path = worktree / name
                baseline_files[name] = (
                    hashlib.sha256(file_path.read_bytes()).hexdigest()
                    if file_path.is_file() else None
                )
            spec_path = self.store.path(
                info["loop_id"],
                Path(info["spec"]).relative_to(f".symphonai/loops/{info['loop_id']}").as_posix(),
            )
            spec = {"text": spec_path.read_text(encoding="utf-8")}
            report_path = self.host._repo_root / info["report"]
            if not report_path.is_file():
                report_path = worktree / info["report"]
            report = report_path.read_text(encoding="utf-8") if report_path.is_file() else "No report was written."
            patch = diff.patch[:MAX_READ_BYTES]
            prompt = spec_reviewer_prompt(spec["text"], report, patch, Path(info["spec"]).parent)
            if failure is not None:
                prompt = failure_reviewer_prompt(
                    spec["text"], report, (source_meta.get("goal") or {}).get("last_check"),
                    failure, patch, Path(info["spec"]).parent,
                )
            policy = replace(self.host._base_policy, repo_root=worktree)
            review_scope = worktree / Path(info["spec"]).parent
            policy.allowed_write_scope = [review_scope]
            policy.mode = "allow"
            meta = {
                "spec": info["spec"], "report": info["report"], "kind": "review",
                "of": session_id, "worktree_path": str(worktree),
                "baseline": patch_digest(diff.patch), "baseline_files": baseline_files,
                "role": "reviewer",
                **({"failure": failure} if failure is not None else {}),
                **({"loop_id": info["loop_id"], "card_id": info.get("card_id")}
                   if info.get("loop_id") else {}),
            }
            run_id = self.host.start(
                prompt, session_policy=policy, _session_meta={"spec_run": meta},
                _title=f"Review {Path(info['spec']).name}",
            )
            review_id = run_id
            source = SessionStore.open(self.host._sessions_root, session_id)
            try:
                def change(source_meta):
                    source_meta["review"] = {"session_id": review_id, "verdict": "running", "follow_ups": [], "not_copied": []}
                source.update_meta(change)
            finally:
                source.close()
            return review_id, run_id


    def _acquire_loop_runner_lock(self, loop_id: str) -> int | None:
        path = self.store.path(loop_id, "runner.lock")
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return None
        return descriptor


    def start_loop(
        self, goal: str, *, runner_pid: int | None = None, runner_lock: bool = False,
        roadmap_goal_id: str | None = None,
    ) -> Loop:

        from symphonai_host.run import ProviderSelectionError, RunActiveError

        if self.host._extensions is not None:
            for role in ("planner", "implementer", "reviewer", "worker", "explorer"):
                provider_name = self.host._extensions.config.get(f"roles.{role}.provider")
                if provider_name is None or (
                    self.host._provider is not None and self.host._provider.name == provider_name
                ):
                    continue
                if self.host._provider_factory is None:
                    raise ProviderSelectionError(
                        f"role {role} provider {provider_name!r} is unavailable"
                    )
                model = self.host._extensions.config.get(f"roles.{role}.model")
                try:
                    provider = self.host._provider_factory(provider_name, model, None)
                except ProviderSelectionError:
                    provider = None
                if provider is None:
                    raise ProviderSelectionError(
                        f"role {role} provider {provider_name!r} is unavailable"
                    )
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must not be blank")
        from symphonai_host.run import MAX_ACTIVE_CONVERSATIONS
        with self.host._lock:
            if len(self.host._active_by_session) >= MAX_ACTIVE_CONVERSATIONS:
                raise RunActiveError(description=f"{MAX_ACTIVE_CONVERSATIONS} conversations are already running")
            roadmap_store = None
            if self.host._roadmap_tools_enabled or roadmap_goal_id is not None:
                roadmap_store = RoadmapStore(self.host._repo_root)
                roadmap = roadmap_store.load()
                if roadmap_goal_id is not None and roadmap_store.find(roadmap, roadmap_goal_id) is None:
                    raise KeyError(roadmap_goal_id)
            store = self.store
            loop = Loop.from_payload(store.create(goal))
            try:
                if runner_pid is not None:
                    loop.runner_pid = runner_pid
                if runner_lock:
                    descriptor = self._acquire_loop_runner_lock(loop.id)
                    if descriptor is None:
                        raise RunActiveError(loop.id)
                    self._loop_runner_lock_fds[loop.id] = descriptor
                    loop.runner_lock = True
                if runner_pid is not None or runner_lock:
                    store.save(loop)
                if roadmap_store is not None:
                    roadmap_store.record_loop(loop.id, goal, roadmap_goal_id)
            except Exception:
                descriptor = self._loop_runner_lock_fds.pop(loop.id, None)
                if descriptor is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
                shutil.rmtree(store.path(loop.id, "."))
                raise
            try:
                return self._start_loop_planner(loop, "created")
            except Exception as exc:
                loop.failure_stage = "planning"
                store.set_state(loop, "needs_attention", f"planner failed: {exc}")
                self._loop_changed(loop, "state")
                return loop



    def start_roadmap_goal(self, goal_id: str) -> tuple[dict, Loop]:
        from symphonai_host.run import RunActiveError

        with self.host._lock:
            roadmap_store = RoadmapStore(self.host._repo_root)
            roadmap = roadmap_store.load()
            goal = roadmap_store.find(roadmap, goal_id)
            if goal is None:
                raise KeyError(goal_id)
            loops = self.store
            if goal["loop_id"] is not None:
                try:
                    existing = loops.load(goal["loop_id"])
                except (KeyError, ValueError):
                    raise RunActiveError("roadmap goal already has a loop") from None
                if existing.get("state") != "cancelled":
                    raise RunActiveError("roadmap goal already has a loop")
            loop = self.start_loop(goal["goal"], roadmap_goal_id=goal_id)
            return roadmap_store.reply(roadmap_store.load()), loop


    def _start_loop_planner(
        self, loop: Loop, change: str, *, note: str | None = None,
        current_plan: dict | None = None,
    ) -> Loop:

        from symphonai_host.run import MAX_ACTIVE_CONVERSATIONS, RunActiveError

        from symphonai_api.tools.shell import sandbox_available

        store = self.store
        folder = store.path(loop.id, ".")
        specs_folder = folder / "specs"
        plan_path = folder / "plan.json"
        has_sandbox = sandbox_available()
        specs_folder.mkdir(parents=True, exist_ok=True)
        prompt = loop_planner_prompt(loop.goal, folder, has_sandbox, note=note, current_plan=current_plan)
        with self.host._lock:
            if len(self.host._active_by_session) >= MAX_ACTIVE_CONVERSATIONS:
                raise RunActiveError(description=f"{MAX_ACTIVE_CONVERSATIONS} conversations are already running")
            policy = replace(self.host._base_policy, repo_root=self.host._repo_root)
            policy.allowed_write_scope = [specs_folder, plan_path]
            policy.sandbox_write_scope = [specs_folder, plan_path]
            policy.shell_enabled = has_sandbox
            policy.shell_allowlist = [()] if has_sandbox else []
            policy.mode = "allow"
            tool_restriction = None
            if not has_sandbox:
                tool_restriction = frozenset(
                    set(self.host.available_tool_names()) - {"run_shell"}
                )
            try:
                run_id = self.host.start(
                    prompt, session_policy=policy, tool_restriction=tool_restriction,
                    _session_meta={"spec_run": {
                        "kind": "loop_plan", "loop_id": loop.id, "role": "planner",
                    }},
                    _title=f"Plan: {loop.goal[:60]}",
                )
                previous_sessions = loop.setdefault("previous_planner_session_ids", [])
                previous_session_id = loop.planner_session_id
                if isinstance(previous_session_id, str) and previous_session_id not in previous_sessions:
                    previous_sessions.append(previous_session_id)
                loop.planner_session_id = run_id
                store.save(loop)
                self._loop_changed(loop, change)
                return loop
            except Exception:
                if change == "created":

                    self._loop_changed(loop, change)
                raise


    def _loop_changed(self, loop: Loop, change: str, card_id: str = "") -> None:
        from symphonai_host.run import LoopChanged

        self.host._broker.publish(LoopChanged(
            agent_id="", run_id="", loop_id=loop.id, state=loop.state,
            change=change, card_id=card_id, reason=loop.reason,
        ))


    def approve_loop_plan(self, loop_id: str) -> Loop:
        with self.host._lock:
            store = self.store
            loop = store.load_record(loop_id)
            loop.require('approve', 'awaiting_plan')
            if not any(card.state != "dropped" for card in loop.cards):
                raise ValueError("nothing left to run")
            branch_prefix = f"symphonai/{loop.slug}"
            integration_admin = self.host._sessions_root / "loops" / loop_id / "integration"
            branch_created = False
            integration_created = False
            branch = branch_prefix
            try:
                _loop_git(self.host._repo_root, "rev-parse", "--show-toplevel")
                suffix = 1
                while True:
                    branch = branch_prefix if suffix == 1 else f"{branch_prefix}-{suffix}"
                    result = _loop_git(
                        self.host._repo_root, "show-ref", "--verify", "--quiet",
                        f"refs/heads/{branch}", check=False,
                    )
                    if result.returncode != 0:
                        break
                    suffix += 1
                _loop_git(self.host._repo_root, "branch", branch)
                branch_created = True
                loop.base_commit = os.fsdecode(
                    _loop_git(self.host._repo_root, "rev-parse", "HEAD").stdout.strip(),
                )
                integration_root = create_worktree(
                    self.host._repo_root, integration_admin, start=branch,
                )
                integration_created = True
                _loop_git(integration_root, "switch", branch)
                loop.branch = branch
                store.set_state(loop, "running")
            except Exception as exc:
                if integration_created or integration_admin.exists():
                    try:
                        remove_worktree(self.host._repo_root, integration_admin)
                    except (OSError, WorktreeError):
                        shutil.rmtree(integration_admin, ignore_errors=True)
                        try:
                            _loop_git(self.host._repo_root, "worktree", "prune")
                        except RuntimeError:
                            pass
                for directory in (integration_admin.parent, integration_admin.parent.parent):
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
                if branch_created:
                    try:
                        _loop_git(self.host._repo_root, "branch", "-D", branch)
                    except RuntimeError:
                        pass
                try:
                    persisted = store.load_record(loop_id)
                    if persisted.state != "awaiting_plan" or persisted.branch is not None:
                        persisted.state = "awaiting_plan"
                        persisted.branch = None
                        store.save(persisted)
                except (KeyError, OSError, ValueError):
                    pass
                raise ValueError(str(exc)) from exc
            self._loop_changed(loop, "state")
            self._advance_loop(loop_id)
            loop = store.load_record(loop_id)
            return loop


    def drop_loop_cards(self, loop_id: str, card_ids: list[str]) -> Loop:
        with self.host._lock:
            store = self.store
            loop = store.load_record(loop_id)
            loop.require('drop', 'awaiting_plan')
            cards = loop.cards
            known = {card.id for card in cards}
            unknown = sorted(set(card_ids) - known)
            if unknown:
                raise ValueError(f"unknown card id: {unknown[0]}")
            dropped = set(card_ids)
            for card in cards:
                if card.id not in dropped and dropped.intersection(card.after):
                    raise ValueError(f"kept card {card.id} depends on a dropped card")
            changed = []
            for card in cards:
                if card.id in dropped and card.state != "dropped":
                    card.state = "dropped"
                    changed.append(card.id)
            if changed:
                store.save(loop)
                for card_id in changed:
                    self._loop_changed(loop, "card", card_id)
            return loop


    def revise_loop_plan(self, loop_id: str, note: str) -> Loop:
        from symphonai_host.run import LoopStateError, MAX_ACTIVE_CONVERSATIONS, RunActiveError

        with self.host._lock:
            store = self.store
            loop = store.load_record(loop_id)
            loop.require('revise', 'awaiting_plan')
            if len(self.host._active_by_session) >= MAX_ACTIVE_CONVERSATIONS:
                raise RunActiveError(description=f"{MAX_ACTIVE_CONVERSATIONS} conversations are already running")
            current_plan = {"lanes": loop.lanes, "cards": [card.to_payload() for card in loop.cards]}
            plan_path = store.path(loop_id, "plan.json")
            previous_plan = plan_path.read_bytes() if plan_path.is_file() else None
            plan_path.unlink(missing_ok=True)
            store.set_state(loop, "planning")
            try:
                return self._start_loop_planner(
                    loop, "state", note=note, current_plan=current_plan,
                )
            except Exception:
                loop = store.load_record(loop_id)
                store.set_state(loop, "awaiting_plan")
                if previous_plan is not None:
                    plan_path.write_bytes(previous_plan)
                self._loop_changed(loop, "state")
                raise


    def cancel_loop(self, loop_id: str) -> Loop:
        worktree_ids = set()
        session_ids = set()
        with self.host._lock:
            store = self.store
            loop = store.load_record(loop_id)
            loop.require('cancel', 'a non-terminal state')
            if isinstance(loop.planner_session_id, str):
                session_ids.add(loop.planner_session_id)
            for card in loop.cards:
                session_id = card.session_id
                if isinstance(session_id, str):
                    session_ids.add(session_id)
                    worktree_ids.add(session_id)
                    try:
                        source = SessionStore.open(self.host._sessions_root, session_id)
                        try:
                            review_id = source.read_meta().get("review", {}).get("session_id")
                            if isinstance(review_id, str):
                                session_ids.add(review_id)
                        finally:
                            source.close()
                    except (OSError, SessionError, AttributeError):
                        pass
            for session_id in session_ids:
                active = self.host._active_by_session.get(session_id)
                if active is not None:
                    active.cancel.cancel()
                approvals = self.host._approvals_by_session.get(session_id)
                if approvals is not None:
                    approvals.cancel_all(reason="loop cancelled")
                goal_check = self.host._goal_checks_by_session.get(session_id)
                if goal_check is not None:
                    goal_check.cancel.set()
            validation_cancel = self._loop_validation_cancel.get(loop_id)
            if validation_cancel is not None:
                validation_cancel.set()
            process = self._loop_validation_processes.get(loop_id)
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    process.kill()
            if loop.state in ("running", "validating"):
                store.set_state(loop, "needs_attention", "cancelling")
                self._loop_changed(loop, "state")
            active_runs = [self.host._active_by_session[sid] for sid in session_ids if sid in self.host._active_by_session]
            goal_thread = self.host._goal_check_thread
        for active in active_runs:
            if active.thread is not threading.current_thread():
                active.thread.join(timeout=5)
        if goal_thread is not None and goal_thread is not threading.current_thread():
            goal_thread.join(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:

            with self.host._lock:
                checks_running = any(sid in self.host._goal_checks_by_session for sid in session_ids)
                validation_running = loop_id in self._loop_validation_cancel
            if not checks_running and not validation_running:
                break
            time.sleep(0.01)
        try:
            for session_id in worktree_ids:
                worktree = self.host._sessions_root / session_id / "worktree"
                if worktree.exists():
                    remove_worktree(self.host._repo_root, worktree)
            integration_admin = self.host._sessions_root / "loops" / loop_id / "integration"
            if integration_admin.exists():
                remove_worktree(self.host._repo_root, integration_admin)
            branch = loop.branch
            if isinstance(branch, str):
                branches = _loop_git(self.host._repo_root, "branch", "--list", branch).stdout
                if branches.strip():
                    _loop_git(self.host._repo_root, "branch", "-D", branch)
        except (OSError, RuntimeError) as exc:
            with self.host._lock:
                loop = store.load_record(loop_id)
                loop.reason = f"cancel failed: {exc}"
                store.save(loop)
                self._loop_changed(loop, "state")
            raise WorktreeError(str(exc)) from exc
        with self.host._lock:
            loop = store.load_record(loop_id)
            store.set_state(loop, "cancelled")
            self._loop_changed(loop, "state")
            return loop



    def integrate(self, loop: Loop, card: Card, worktree: Path) -> str:
        from symphonai_host.run import WorktreeApplyConflict

        loop_id = loop.id
        self.store.validate_id(loop_id)
        integration_root = self._loop_integration_root(loop_id)
        diff = self._loop_card_diff(loop_id, Path(worktree))
        if not diff.patch:
            return ""
        linked = _linked_paths(integration_root.resolve())
        status = _loop_git(
            integration_root, "status", "--porcelain", "--untracked-files=all", "--", ".",
            *(f":(exclude){path}" for path in linked),
        ).stdout
        if status:
            raise RuntimeError("integration worktree has uncommitted changes")
        patch = diff.patch.encode("utf-8", errors="surrogateescape")
        checked = subprocess.run(
            ["git", "apply", "--check", "--binary"], cwd=integration_root,
            input=patch, capture_output=True, check=False,
        )
        if checked.returncode:
            message = checked.stderr.decode("utf-8", errors="replace").strip()
            raise WorktreeApplyConflict(message or "git apply --check failed")
        try:
            applied = subprocess.run(
                ["git", "apply", "--binary"], cwd=integration_root,
                input=patch, capture_output=True, check=False,
            )
            if applied.returncode:
                message = applied.stderr.decode("utf-8", errors="replace").strip()
                raise WorktreeApplyConflict(message or "git apply failed")
            stage_worktree(integration_root)
            _loop_git(integration_root, "commit", "-m", f"{card.id}: {card.title}")
            return os.fsdecode(_loop_git(integration_root, "rev-parse", "HEAD").stdout.strip())
        except (OSError, RuntimeError) as exc:
            reason = str(exc)
            try:
                _loop_git(integration_root, "reset", "--hard", "HEAD")
                _loop_git(integration_root, "clean", "-fd", "--", ".",
                          *(f":(exclude){path}" for path in linked))
            except (OSError, RuntimeError) as cleanup_error:
                reason = f"{reason}; integration cleanup failed: {cleanup_error}"
            card["failure"] = reason
            card["reason"] = reason
            loop.failure_stage = "integration"
            self._loop_attention(loop, f"card {card.id} integration failed: {reason}", card)
            raise RuntimeError(reason) from exc



    @staticmethod
    def _loop_card_diff(loop_id: str, worktree: Path) -> WorktreeDiff:
        stage_worktree(worktree)
        loop_files = f":(exclude).symphonai/loops/{loop_id}/**"
        files = _loop_git(
            worktree, "diff", "--cached", "--name-only", "--relative", "-z", "HEAD",
            "--", ".", loop_files,
        ).stdout
        patch = _loop_git(
            worktree, "diff", "--cached", "--binary", "HEAD", "--", ".", loop_files,
        ).stdout
        return WorktreeDiff(
            patch=os.fsdecode(patch),
            files=tuple(os.fsdecode(item) for item in files.split(b"\0") if item),
        )


    def _loop_integration_root(self, loop_id: str) -> Path:
        self.store.validate_id(loop_id)
        top = Path(os.fsdecode(_loop_git(
            self.host._repo_root, "rev-parse", "--show-toplevel",
        ).stdout.strip())).resolve()
        relative_root = self.host._repo_root.resolve().relative_to(top)
        return self.host._sessions_root / "loops" / loop_id / "integration" / relative_root


    def _loop_card_changed(self, loop: Loop, card: Card) -> None:
        self.store.save(loop)
        self._loop_changed(loop, "card", str(card.id))


    def _loop_attention(self, loop: Loop, reason: str, card: Card | None = None) -> None:
        store = self.store
        if card is not None:
            card.state = "failed"
            self._loop_card_changed(loop, card)
        if loop.state == "running":
            store.set_state(loop, "needs_attention", reason)
        else:
            loop.reason = reason
            store.save(loop)
        self._loop_changed(loop, "state")


    def _advance_loop(self, loop_id: str) -> None:
        """Start ready lane cards while conversation slots are available."""
        from symphonai_host.run import MAX_ACTIVE_CONVERSATIONS, RunActiveError

        with self.host._lock:
            store = self.store
            loop = store.load_record(loop_id)
            if loop.state == "validating":
                self._start_loop_validation(loop_id)
                return
            if not loop.allows("advance"):
                return
            integration = self._loop_integration_root(loop_id)
            active_count = len(self.host._active_by_session)
            lanes = loop.lanes
            lane_order = [lane.get("id", lane.get("lane", index + 1)) for index, lane in enumerate(lanes)]
            cards = loop.cards
            for card in cards:
                if active_count >= MAX_ACTIVE_CONVERSATIONS:
                    break
                if card.state != "review" or not card.session_id:
                    continue
                try:
                    source = SessionStore.open(self.host._sessions_root, card.session_id)
                    try:
                        review = source.read_meta().get("review")
                    finally:
                        source.close()
                    if isinstance(review, dict) and review.get("session_id"):
                        continue
                    self.start_spec_review(card.session_id, failure=card.to_payload().get("failure"))
                    active_count += 1
                except RunActiveError:
                    break
                except Exception as exc:
                    self._loop_attention(loop, f"card {card.id} review could not start: {exc}", card)
                    return
            def dependency_done(dep_id, current):
                dependency = next((item for item in cards if item.id == dep_id), None)
                if dependency is None or dependency.state not in ("done", "failed"):
                    return False
                replacements = [item for item in cards if item.follow_up_of and dep_id in item.after]
                if dependency.state == "failed" and not replacements:
                    return False
                return all((current is not None and item.id == current.id)
                           or dependency_done(item.id, current) for item in replacements)

            for lane_id in lane_order:
                if active_count >= MAX_ACTIVE_CONVERSATIONS:
                    break
                lane_cards = [card for card in cards if card.lane == lane_id]
                if any(card.state in ("running", "review") for card in lane_cards):
                    continue
                card = next((item for item in lane_cards if item.state == "planned"
                             and all(dependency_done(dep_id, item) for dep_id in item.after)), None)
                if card is None:
                    continue
                spec_value = card.spec
                try:
                    if not isinstance(spec_value, str):
                        raise ValueError("card has no spec path")
                    spec_path = store.path(loop_id, spec_value)
                    source = spec_path.read_text(encoding="utf-8")
                    from symphonai_api.spec_text import spec_validation
                    validation = spec_validation(source)
                    if not validation:
                        raise ValueError(f"card {card.id} has no Validation commands")
                    tip = os.fsdecode(_loop_git(integration, "rev-parse", "HEAD").stdout.strip())
                    card_spec_path = spec_path.relative_to(self.host._repo_root).as_posix()
                    report_path = store.path(
                        loop_id, card.get("report", f"reports/{card.id}-report.md"),
                    )
                    relative_report = report_path.relative_to(self.host._repo_root).as_posix()
                    _, session_id = self.start_spec_run({
                        "text": source, "validation": validation,
                        "path": card_spec_path, "report": relative_report,
                        "loop_id": loop_id, "card_id": card.id, "start_ref": tip,
                    })
                except RunActiveError:
                    break
                except Exception as exc:
                    self._loop_attention(loop, f"card {card.id} could not start: {exc}", card)
                    return
                card.state = "running"
                card.session_id = session_id
                self._loop_card_changed(loop, card)
                active_count += 1
            non_dropped = [card for card in cards if card.state != "dropped"]
            if non_dropped and all(dependency_done(card.id, None) for card in non_dropped):
                store.set_state(loop, "validating")
                self._loop_changed(loop, "state")
                self._start_loop_validation(loop_id)


    def _finish_loop_card_run(
        self, session_id: str, state: str, *, error: str = "", goal_reason: str = "",
    ) -> None:
        info = {}
        try:
            source = SessionStore.open(self.host._sessions_root, session_id)
            try:
                info = source.read_meta().get("spec_run", {})
            finally:
                source.close()
            loop_id, card_id = info.get("loop_id"), info.get("card_id")
            if not loop_id or not card_id:
                return
            with self.host._lock:
                store = self.store
                loop = store.load_record(loop_id)
                if loop.state == "cancelled":
                    return
                card = next(item for item in loop.cards if item.get("id") == card_id)
                if state == "finished":
                    card.pop("failure", None)
                    card.state = "review"
                    self._loop_card_changed(loop, card)
                    self._advance_loop(loop_id)
                    return
                cause = error or goal_reason or state
                card["failure"] = f"card {card_id} could not finish: {cause}"
                card.state = "review"
                self._loop_card_changed(loop, card)
                self._advance_loop(loop_id)
        except (OSError, KeyError, ValueError, SessionError) as exc:
            if isinstance(info, dict) and info.get("loop_id"):
                with self.host._lock:
                    loop = self.store.load_record(info["loop_id"])
                    self._loop_attention(loop, f"card {info.get('card_id')} could not finish: {exc}")


    def _finish_loop_card_review(self, review_info: dict) -> None:
        loop_id, card_id = review_info.get("loop_id"), review_info.get("card_id")
        if not loop_id or not card_id:
            return
        with self.host._lock:
            store = self.store
            loop = store.load_record(loop_id)
            if loop.state == "cancelled":
                return
            if review_info.get("failure") is not None and loop.state == "needs_attention":
                return
            card = next(item for item in loop.cards if item.get("id") == card_id)
            source = SessionStore.open(self.host._sessions_root, review_info["of"])
            try:
                result = source.read_meta().get("review", {})
            finally:
                source.close()
            verdict = result.get("verdict")
            if verdict not in ("passed", "follow-ups"):
                error = result.get("error")
                if error:
                    reason = f"card {card_id} review failed: {error}"
                elif verdict == "tree-changed":
                    reason = f"card {card_id} review changed files other than its follow-up specs"
                elif verdict == "no-verdict":
                    reason = f"card {card_id} review gave no verdict"
                else:
                    reason = f"card {card_id} review did not pass"
                self._loop_attention(loop, reason, card)
                return
            failure = review_info.get("failure")
            if failure is not None and verdict != "follow-ups":
                self._loop_attention(loop, f"card {card_id} failure review passed a failed card", card)
                return
            card_worktree = self.host._sessions_root / review_info["of"] / "worktree"
            commit = ""
            if failure is None:
                try:
                    commit = self.integrate(loop, card, card_worktree)
                except (OSError, RuntimeError) as exc:
                    card["failure"] = str(exc)
                    self._loop_card_changed(loop, card)
                    try:
                        self.start_spec_review(review_info["of"], failure=str(exc))
                    except Exception as review_error:
                        self._loop_attention(loop, f"card {card_id} failure review could not start: {review_error}", card)
                    self._advance_loop(loop_id)
                    return
            if card_worktree.exists():
                remove_worktree(self.host._repo_root, card_worktree)
            if commit:
                card.commit = commit
            card.state = "failed" if failure is not None else "done"
            self._loop_card_changed(loop, card)
            if verdict == "follow-ups":
                original_id = card.follow_up_of or card_id
                rounds = int(card.rounds) + 1
                if rounds > 2:
                    reason = (
                        f"card {original_id} still fails after 2 follow-up rounds"
                        if failure is not None else
                        f"card {original_id} still has findings after 2 follow-up rounds"
                    )
                    self._loop_attention(loop, reason)
                    return
                follow_ups = result.get("follow_ups", [])
                added = False
                for follow_path in follow_ups:
                    relative = Path(follow_path)
                    prefix = f".symphonai/loops/{loop_id}/"
                    if not follow_path.startswith(prefix) or ".." in relative.parts:
                        continue
                    source_path = self.host._repo_root / relative
                    if not source_path.is_file():
                        continue
                    suffix = "F" if rounds == 1 else "F2"
                    follow_id = f"{original_id}{suffix}"
                    relative_spec = relative.relative_to(prefix).as_posix()
                    if any(item.get("id") == follow_id for item in loop.cards):
                        continue
                    follow = Card.from_payload({
                        "id": follow_id, "title": f"Follow up: {card.title}",
                        "spec": relative_spec, "lane": card.lane,
                        "after": [card_id], "state": "planned",
                        "follow_up_of": original_id, "rounds": rounds,
                    })
                    loop.cards.append(follow)
                    self._loop_card_changed(loop, follow)
                    added = True
                    break
                if failure is not None and not added:
                    self._loop_attention(loop, f"card {card_id} failure review supplied no usable follow-up")
                    return
            self._advance_loop(loop_id)


    def _start_loop_validation(self, loop_id: str) -> None:
        with self.host._lock:
            if loop_id in self._loop_validation_cancel or self.host._closing:
                return
            loop = self.store.load_record(loop_id)
            if not loop.allows("validate"):
                return
            cancelled = threading.Event()
            self._loop_validation_cancel[loop_id] = cancelled
            thread = threading.Thread(
                target=self._validate_loop, args=(loop_id, cancelled),
                name=f"symphonai-loop-validation-{loop_id}", daemon=True,
            )
            try:
                thread.start()
            except Exception:
                self._loop_validation_cancel.pop(loop_id, None)
                raise


    def _run_loop_validation_command(
        self, loop_id: str, command: str, cwd: Path, cancelled: threading.Event,
    ) -> dict:
        from symphonai_api.process import run_process
        from symphonai_api.tools.shell import sandbox_command
        policy = replace(self.host._base_policy, repo_root=cwd)
        policy.mode = "allow"
        try:
            argv = sandbox_command(policy, [shell_executable(), "-c", command], cwd)
        except ValueError as exc:
            return {"command": command, "exit": None, "output": str(exc)}
        def started(process):
            with self.host._lock:
                self._loop_validation_processes[loop_id] = process
        try:
            result = run_process(
                argv, cwd, env=shell_environment(cwd, for_agent=True), timeout=600,
                cancel=cancelled.is_set, output_limit=4000, on_start=started,
            )
            output = str(result.error) if result.error is not None else result.stdout.decode("utf-8", errors="replace")
            return {"command": command, "exit": result.exit_code, "output": output[-4000:]}
        finally:
            with self.host._lock:
                self._loop_validation_processes.pop(loop_id, None)


    def _validate_loop(self, loop_id: str, cancelled: threading.Event) -> None:
        from symphonai_api.spec_text import spec_validation

        store = self.store
        try:
            loop = store.load_record(loop_id)
            if not loop.allows("validate"):
                return
            commands = []
            seen = set()
            for card in loop.cards:
                if card.state != "done":
                    continue
                spec = store.path(loop_id, card.spec).read_text(encoding="utf-8")
                for line in spec_validation(spec):
                    if line not in seen:
                        commands.append(line)
                        seen.add(line)
            integration = self._loop_integration_root(loop_id)
            validation = [
                self._run_loop_validation_command(loop_id, command, integration, cancelled)
                for command in commands
            ]
            if cancelled.is_set():
                return
            done_cards = [card for card in loop.cards if card.state == "done"]
            commits = [
                {"card": card.id, "commit": card.commit, "title": card.title}
                for card in done_cards
                if card.commit
            ]
            base_commit = loop.base_commit
            head = os.fsdecode(_loop_git(integration, "rev-parse", "HEAD").stdout.strip())
            files = _loop_git(integration, "diff", "--name-only", f"{base_commit}..{head}").stdout
            summary = {
                "commits": commits,
                "files": sorted(item for item in os.fsdecode(files).splitlines() if item),
                "validation": validation,
            }
            with self.host._lock:
                loop = store.load_record(loop_id)
                if not loop.allows("validate") or cancelled.is_set():
                    return
                loop.summary = summary
                failed = next((item for item in validation if item["exit"] != 0), None)
                if failed is None:
                    store.set_state(loop, "awaiting_commit")
                else:
                    loop.failure_stage = "validation"
                    store.set_state(loop, "needs_attention", f"validation failed: {failed['command']}")
                self._loop_changed(loop, "state")
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            with self.host._lock:
                try:
                    loop = store.load_record(loop_id)
                    if loop.state == "validating":
                        loop.failure_stage = "validation"
                        store.set_state(loop, "needs_attention", f"validation failed: {exc}")
                        self._loop_changed(loop, "state")
                except (KeyError, ValueError, OSError):
                    pass
        finally:
            with self.host._lock:
                self._loop_validation_cancel.pop(loop_id, None)


    def commit_loop(self, loop_id: str, action: str) -> Loop:
        with self.host._lock:
            store = self.store
            loop = store.load_record(loop_id)
            loop.require('commit', 'awaiting_commit')
            if action not in ("approve", "decline"):
                raise ValueError("loop commit action must be approve or decline")
            integration_admin = self.host._sessions_root / "loops" / loop_id / "integration"
            branch = loop.branch
            try:
                if integration_admin.exists():
                    remove_worktree(self.host._repo_root, integration_admin)
                for card in loop.cards:
                    session_id = card.session_id
                    if isinstance(session_id, str):
                        card_worktree = self.host._sessions_root / session_id / "worktree"
                        if card_worktree.exists():
                            remove_worktree(self.host._repo_root, card_worktree)
                if action == "decline" and isinstance(branch, str):
                    _loop_git(self.host._repo_root, "branch", "-D", branch)
            except (OSError, WorktreeError, RuntimeError) as exc:
                raise ValueError(str(exc)) from exc
            if action == "approve":
                store.set_state(loop, "done")
                try:
                    RoadmapStore(self.host._repo_root).mark_loop_done(loop_id)
                except InvalidRoadmap:
                    pass
            else:
                store.set_state(loop, "cancelled", "declined")
            self._loop_changed(loop, "state")
            return loop


    def resume_loop(
        self, loop_id: str, *, runner_pid: int | None = None, runner_lock: bool = False,
    ) -> Loop:
        from symphonai_host.run import LoopStateError, RunActiveError

        with self.host._lock:
            store = self.store
            loop = store.load_record(loop_id)
            loop.require('resume', 'needs_attention')
            if runner_pid is not None:
                loop.runner_pid = runner_pid
            if runner_lock:
                descriptor = self._acquire_loop_runner_lock(loop_id)
                if descriptor is None:
                    raise RunActiveError(loop_id)
                self._loop_runner_lock_fds[loop_id] = descriptor
                loop.runner_lock = True
            if runner_pid is not None or runner_lock:
                store.save(loop)
            planning_failed = loop.failure_stage == "planning" or str(loop.reason).startswith(
                ("planner failed:", "planner stopped:", "plan.json", "plan ", "spec ")
            ) or not loop.cards
            if not planning_failed:
                try:
                    plan = json.loads(store.path(loop_id, "plan.json").read_text(encoding="utf-8"))
                    planning_failed = bool(self._loop_plan_problem(store, loop_id, plan))
                except (OSError, UnicodeError, ValueError):
                    planning_failed = True
            if planning_failed:
                store.set_state(loop, "planning", "")
                try:
                    return self._start_loop_planner(loop, "state")
                except Exception as exc:
                    loop.failure_stage = "planning"
                    store.set_state(loop, "needs_attention", f"planner failed: {exc}")
                    self._loop_changed(loop, "state")
                    raise
            for card in loop.cards:
                if card.state != "failed" or any(
                    item.follow_up_of and card.id in item.after for item in loop.cards
                ):
                    continue
                session_id = card.session_id
                if isinstance(session_id, str):
                    try:
                        source = SessionStore.open(self.host._sessions_root, session_id)
                        try:
                            review = source.read_meta().get("review", {})
                        finally:
                            source.close()
                    except (OSError, SessionError, TranscriptError):
                        review = {}
                    review_id = review.get("session_id") if isinstance(review, dict) else None
                    active = self.host._active_by_session.get(review_id)
                    if active is not None:
                        active.cancel.cancel()
                    approvals = self.host._approvals_by_session.get(review_id)
                    if approvals is not None:
                        approvals.cancel_all(reason="loop resumed")
                    worktree = self.host._sessions_root / session_id / "worktree"
                    previous_sessions = card.setdefault("previous_session_ids", [])
                    if session_id not in previous_sessions:
                        previous_sessions.append(session_id)
                card.pop("reason", None)
                card.update(state="planned", session_id=None)
                self._loop_card_changed(loop, card)
                if isinstance(session_id, str) and worktree.exists():
                    remove_worktree(self.host._repo_root, worktree)
            validation_failed = loop.failure_stage == "validation"
            if "failure_stage" not in loop:
                validation_failed = str(loop.reason).startswith("validation failed:")
            if validation_failed:
                store.set_state(loop, "validating", "")
                self._loop_changed(loop, "state")
                self._start_loop_validation(loop_id)
            else:
                store.set_state(loop, "running", "")
                self._loop_changed(loop, "state")
                self._advance_loop(loop_id)
            return store.load_record(loop_id)


    def loop_card_details(self, loop_id: str, card_id: str) -> dict:
        store = self.store
        loop = store.load_record(loop_id)
        card = next((item for item in loop.cards if item.get("id") == card_id), None)
        if card is None:
            raise KeyError(card_id)

        def read_text(relative: str) -> str:
            try:
                path = store.path(loop_id, relative)
                with path.open("rb") as source:
                    return source.read(MAX_READ_BYTES).decode("utf-8", errors="replace")
            except (OSError, ValueError, TypeError):
                return ""

        spec = read_text(card.spec)
        report = read_text(card.get("report", ""))
        diff_bytes = b""
        commit = card.commit
        if not isinstance(commit, str):
            commit = next((
                item.get("commit") for item in loop.summary.get("commits", [])
                if item.get("card") == card_id
            ), None)
        branch = loop.branch
        valid_branch = isinstance(branch, str) and re.fullmatch(r"symphonai/[a-z0-9-]+", branch)
        if (isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40,64}", commit)
                and valid_branch):
            ancestry = _loop_git(
                self.host._repo_root, "merge-base", "--is-ancestor", commit, branch, check=False,
            )
            if ancestry.returncode == 0:
                diff_bytes = _loop_git(
                    self.host._repo_root, "show", "--format=", "--binary", commit,
                ).stdout
        elif isinstance(card.session_id, str):
            worktree = self.host._sessions_root / card.session_id / "worktree"
            if worktree.is_dir():
                try:
                    diff_bytes = worktree_diff(worktree).patch.encode("utf-8", errors="surrogateescape")
                except (OSError, RuntimeError):
                    diff_bytes = b""
        truncated = len(diff_bytes) > MAX_READ_BYTES
        diff = diff_bytes[:MAX_READ_BYTES].decode("utf-8", errors="replace")

        review_meta = {}
        session_id = card.session_id if isinstance(card.session_id, str) else None
        if session_id is not None:
            try:
                source = SessionStore.open(self.host._sessions_root, session_id)
                try:
                    review_meta = source.read_meta().get("review", {})
                finally:
                    source.close()
            except (OSError, SessionError, AttributeError):
                review_meta = {}
        verdict = review_meta.get("verdict", "") if isinstance(review_meta, dict) else ""
        if verdict not in ("passed", "follow-ups", "running"):
            verdict = ""
        review_text = ""
        review_id = review_meta.get("session_id") if isinstance(review_meta, dict) else None
        if isinstance(review_id, str) and verdict != "running":
            try:
                review_store = SessionStore.open(self.host._sessions_root, review_id)
                try:
                    review_run = load_run(review_store)
                    review_text = next((
                        message.text for message in reversed(review_run.messages)
                        if message.role == Role.ASSISTANT and message.text.strip()
                    ), "")
                finally:
                    review_store.close()
            except (OSError, SessionError):
                review_text = ""
        follow_up_ids = [
            item["id"] for item in loop.cards
            if card_id in item.get("after", []) and item.get("follow_up_of")
        ]
        return {
            "card": card.to_payload(),
            "spec": spec,
            "report": report,
            "diff": diff,
            "diff_truncated": truncated,
            "review": {"verdict": verdict, "text": review_text, "follow_ups": follow_up_ids},
            "session_id": session_id,
        }


    def recover_loops(self) -> None:
        store = self.store
        for loop in store.records():
            with self.host._lock:
                if loop.state == "planning":
                    if loop.planner_session_id not in self.host._active_by_session:
                        loop.failure_stage = "planning"
                        store.set_state(loop, "needs_attention", "planner stopped: host restarted")
                        self._loop_changed(loop, "state")
                    continue
                if loop.state == "validating":
                    self._start_loop_validation(loop.id)
                    continue
                if not loop.allows("advance"):
                    continue
                if loop.runner_lock is True:
                    descriptor = self._acquire_loop_runner_lock(loop.id)
                    if descriptor is None:
                        continue
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
                else:
                    runner_pid = loop.runner_pid
                    if type(runner_pid) is int and runner_pid > 0:
                        try:
                            os.kill(runner_pid, 0)
                        except ProcessLookupError:
                            pass
                        except OSError:
                            continue
                        else:
                            continue
                changed = False
                changed_cards = []
                for card in loop.cards:
                    if card.state not in ("running", "review"):
                        continue
                    session_id = card.session_id
                    if isinstance(session_id, str) and session_id in self.host._active_by_session:
                        continue
                    if card.state == "review" and isinstance(session_id, str):
                        try:
                            source = SessionStore.open(self.host._sessions_root, session_id)
                            try:
                                review_id = source.read_meta().get("review", {}).get("session_id")
                            finally:
                                source.close()
                        except (OSError, SessionError, AttributeError):
                            review_id = None
                        if isinstance(review_id, str) and review_id in self.host._active_by_session:
                            continue
                    if isinstance(session_id, str):
                        worktree = self.host._sessions_root / session_id / "worktree"
                        if worktree.exists():
                            remove_worktree(self.host._repo_root, worktree)
                    if isinstance(session_id, str):
                        previous_sessions = card.setdefault("previous_session_ids", [])
                        if session_id not in previous_sessions:
                            previous_sessions.append(session_id)
                    card.update(state="planned", session_id=None)
                    changed_cards.append(card.id)
                    changed = True
                if changed:
                    store.save(loop)
                    for card_id in changed_cards:
                        self._loop_changed(loop, "card", card_id)
                self._advance_loop(loop.id)


    def _finish_loop_plan(
        self, session_id: str, terminal_event: RunFinished | RunFailed | None,
        *, error: Exception | None = None,
    ) -> None:
        session = SessionStore.open(self.host._sessions_root, session_id)
        try:
            meta = session.read_meta()
            info = meta.get("spec_run")
        finally:
            session.close()
        if not isinstance(info, dict) or info.get("kind") != "loop_plan":
            return
        loop_id = info.get("loop_id")
        if not isinstance(loop_id, str):
            return
        store = self.store
        try:
            loop = store.load_record(loop_id)
        except (KeyError, ValueError):
            return
        if loop.state != "planning":
            return
        reason = ""
        if error is not None:
            reason = f"planner failed: {error}"
        elif isinstance(terminal_event, RunFailed):
            reason = f"planner failed: {terminal_event.error}"
        elif not isinstance(terminal_event, RunFinished) or terminal_event.stopped_reason != "final_response":
            stopped_reason = terminal_event.stopped_reason if isinstance(terminal_event, RunFinished) else "unknown"
            reason = f"planner stopped: {stopped_reason}"
        else:
            plan_path = store.path(loop_id, "plan.json")
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                reason = "plan.json is missing"
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                reason = f"plan.json is invalid: {exc}"
            else:
                reason = self._loop_plan_problem(store, loop_id, plan)
                if not reason:
                    templates = plan["cards"]
                    expanded_ids = {
                        card["id"]: ([f"{card['id']}{index}" for index in range(1, len(card["items"]) + 1)]
                                     if "items" in card else [card["id"]])
                        for card in templates
                    }
                    cards = []
                    next_lane = 4
                    for template in templates:
                        after = [value for parent in template["after"] for value in expanded_ids[parent]]
                        if "items" not in template:
                            cards.append({**template, "after": after})
                            continue
                        path = store.path(loop_id, template["spec"])
                        text = path.read_text(encoding="utf-8")
                        for index, item in enumerate(template["items"], 1):
                            expanded_path = path.with_name(f"{path.stem}-{index}{path.suffix}")
                            try:
                                expanded_path.write_text(text.replace("{item}", item), encoding="utf-8")
                            except OSError as exc:
                                store.set_state(loop, "needs_attention", f"item spec could not be written: {exc}")
                                self._loop_changed(loop, "state")
                                return
                            cards.append({
                                **template, "id": f"{template['id']}{index}",
                                "title": f"{template['title']}: {item}",
                                "spec": expanded_path.relative_to(store.path(loop_id, ".")).as_posix(),
                                "lane": next_lane, "after": after,
                            })
                            next_lane += 1
                    loop.cards = [
                        Card.from_payload({
                            "id": card["id"],
                            "title": card["title"],
                            "spec": card["spec"],
                            "report": "reports/" + card["id"] + "-" + (re.sub(
                                r"[^a-z0-9]+", "-", card["title"].lower(),
                            ).strip("-")[:40].rstrip("-") or "spec") + "-report.md",
                            "lane": card["lane"],
                            "after": card["after"],
                            "state": "planned",
                            "rounds": 0,
                            "follow_up_of": None,
                            "session_id": None,
                        })
                        for card in cards
                    ]
                    lane_ids = sorted({card["lane"] for card in cards})
                    loop.lanes = [
                        {"id": lane_id, "cards": [card["id"] for card in cards if card["lane"] == lane_id]}
                        for lane_id in lane_ids
                    ]
                    store.set_state(loop, "awaiting_plan")
                    self._loop_changed(loop, "planned")
                    return
        loop.reason = reason
        store.set_state(loop, "needs_attention", reason)
        self._loop_changed(loop, "state")


    @staticmethod
    def _loop_plan_problem(store: LoopStore, loop_id: str, plan: object) -> str:
        if not isinstance(plan, dict) or not isinstance(plan.get("cards"), list):
            return "plan.json must contain a cards list"
        cards = plan["cards"]
        if not 1 <= len(cards) <= 8:
            return f"plan must contain 1 to 8 cards; found {len(cards)}"
        ids = []
        for card in cards:
            if not isinstance(card, dict):
                return "plan card must be an object"
            card_id = card.get("id")
            if not isinstance(card_id, str) or card_id not in set("abcdefgh"):
                return f"invalid card id: {card_id!r}"
            if card_id in ids:
                return f"duplicate card id: {card_id}"
            ids.append(card_id)
            if "items" in card:
                items = card["items"]
                if (not isinstance(items, list) or not 1 <= len(items) <= 50
                    or any(not isinstance(item, str) or not item.strip() or len(item) > 200 for item in items)
                    or len(set(items)) != len(items)):
                    return f"card {card_id} items must contain 1 to 50 distinct non-blank strings of at most 200 characters"
            if not isinstance(card.get("title"), str) or not card["title"].strip():
                return f"card {card_id} has no title"
            if not isinstance(card.get("spec"), str):
                return f"card {card_id} has an invalid spec path"
            if type(card.get("lane")) is not int or not 1 <= card["lane"] <= 3:
                return f"card {card_id} has an invalid lane"
            if not isinstance(card.get("after"), list) or not all(
                isinstance(item, str) for item in card["after"]
            ):
                return f"card {card_id} has an invalid after list"
        known_ids = set(ids)
        graph = {card["id"]: card["after"] for card in cards}
        for card in cards:
            card_id = card["id"]
            if any(parent not in known_ids for parent in card["after"]):
                return f"card {card_id} has an unknown after id"
            try:
                path = store.path(loop_id, card["spec"])
            except (KeyError, ValueError, OSError):
                return f"card {card_id} spec path escapes the loop folder"
            specs_root = store.path(loop_id, "specs")
            if not path.is_relative_to(specs_root) or not path.is_file():
                return f"card {card_id} spec file is missing or outside loop specs"
            from symphonai_api.spec_text import spec_validation
            if not spec_validation(path.read_text(encoding="utf-8")):
                return f"card {card_id} has no Validation commands"
        template_paths = {store.path(loop_id, card["spec"]) for card in cards}
        generated_paths = set()
        for card in cards:
            path = store.path(loop_id, card["spec"])
            for index in range(1, len(card.get("items", [])) + 1):
                expanded_path = path.with_name(f"{path.stem}-{index}{path.suffix}")
                if expanded_path in template_paths or expanded_path in generated_paths or expanded_path.is_symlink():
                    return f"card {card['id']} expanded spec path already exists"
                generated_paths.add(expanded_path)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(card_id: str) -> bool:
            if card_id in visiting:
                return False
            if card_id in visited:
                return True
            visiting.add(card_id)
            if not all(visit(parent) for parent in graph[card_id]):
                return False
            visiting.remove(card_id)
            visited.add(card_id)
            return True

        if not all(visit(card_id) for card_id in ids):
            return "card dependencies contain a cycle"
        return ""


    def loop_usage(self, loop: Loop) -> dict:
        usage: dict = {"cards": {}}
        total = UsageTotals()

        def add_stage(
            target: dict, stage: str,
            session_ids: str | list[str] | tuple[str, ...] | None,
        ) -> None:
            nonlocal total
            if isinstance(session_ids, str):
                session_ids = [session_ids]
            if not isinstance(session_ids, (list, tuple)):
                return
            stage_usage = UsageTotals()
            for session_id in dict.fromkeys(item for item in session_ids if isinstance(item, str)):
                stage_usage = stage_usage.merged(self.host._session_usage(session_id))
            if stage_usage.calls == 0:
                return
            target[stage] = {
                "input_tokens": stage_usage.input_tokens,
                "output_tokens": stage_usage.output_tokens,
                "calls": stage_usage.calls,
                "cache_read_tokens": stage_usage.cache_read_tokens,
                "cache_write_tokens": stage_usage.cache_write_tokens,
                "cache_write_1h_tokens": stage_usage.cache_write_1h_tokens,
            }
            total = total.merged(stage_usage)

        add_stage(usage, "planning", [
            *loop.previous_planner_session_ids, loop.planner_session_id,
        ])
        for card in loop.cards:
            card_stages: dict = {}
            session_ids = [*card.previous_session_ids, card.session_id]
            session_ids = list(dict.fromkeys(item for item in session_ids if isinstance(item, str)))
            add_stage(card_stages, "implement", session_ids)
            review_ids = []
            for session_id in session_ids:
                try:
                    source = SessionStore.open(self.host._sessions_root, session_id)
                    try:
                        review = source.read_meta().get("review")
                    finally:
                        source.close()
                except (OSError, SessionError, TranscriptError):
                    review = None
                review_id = review.get("session_id") if isinstance(review, dict) else None
                if isinstance(review_id, str):
                    review_ids.append(review_id)
            add_stage(card_stages, "review", review_ids)
            if card_stages:
                usage["cards"][card.id] = card_stages
        usage["total"] = {
            "input_tokens": total.input_tokens,
            "output_tokens": total.output_tokens,
            "calls": total.calls,
            "cache_read_tokens": total.cache_read_tokens,
            "cache_write_tokens": total.cache_write_tokens,
            "cache_write_1h_tokens": total.cache_write_1h_tokens,
        }
        return usage




    def _finish_spec_run(self, session_id: str, phase: str, *, error: str = "") -> None:
        goal_reason = ""
        try:
            store = SessionStore.open(self.host._sessions_root, session_id)
            try:
                meta = store.read_meta()
                info = meta.get("spec_run")
                if not isinstance(info, dict) or info.get("kind") != "implement":
                    return
                goal = meta.get("goal")
                if isinstance(goal, dict) and isinstance(goal.get("reason"), str):
                    goal_reason = goal["reason"]
                    check = goal.get("check")
                    rounds = goal.get("rounds")
                    last_check = goal.get("last_check")
                    if (
                        goal_reason == "rounds exhausted"
                        and isinstance(check, list) and len(check) >= 3
                        and isinstance(check[2], str)
                        and type(rounds) is int
                        and isinstance(last_check, dict) and "exit" in last_check
                    ):
                        commands = "; ".join(
                            line for line in check[2].splitlines() if line != "set -e"
                        )
                        goal_reason = (
                            f"validation still failing after {rounds} rounds "
                            f"(exit {last_check['exit']}): {commands}"
                        )
                root = self.host._sessions_root / session_id / "worktree"
                report = info.get("report")
                copied = False
                if isinstance(report, str):
                    source = root / report
                    ignored = subprocess.run(
                        ["git", "check-ignore", "-q", "--", report], cwd=self.host._repo_root,
                        capture_output=True, check=False,
                    ).returncode == 0
                    if source.is_file() and ignored and ".env" not in Path(report).parts:
                        target = self.host._repo_root / report
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                        copied = True
                def change(meta):
                    info = meta.get("spec_run", {})
                    info.update({
                        "state": {"complete": "finished", "blocked": "blocked", "paused": "stopped"}.get(phase, "stopped"),
                        "report_copied": copied,
                    })
                    meta["spec_run"] = info
                store.update_meta(change)
            finally:
                store.close()
        except (OSError, SessionError, ValueError):
            return
        self._finish_loop_card_run(
            session_id,
            {"complete": "finished", "blocked": "blocked", "paused": "stopped"}.get(phase, "stopped"),
            error=error, goal_reason=goal_reason,
        )


    def _finish_spec_review(self, session_id: str, leader: Leader, terminal_event: RunFinished | RunFailed | None) -> None:
        from symphonai_host.spec_run import patch_digest, review_verdict

        with self.host._lock:
            review_store = SessionStore.open(self.host._sessions_root, session_id)
            try:
                review_meta = review_store.read_meta()
                review_info = review_meta.get("spec_run")
            finally:
                review_store.close()
            if not isinstance(review_info, dict) or review_info.get("kind") != "review":
                return
            source_id = review_info.get("of")
            if not isinstance(source_id, str):
                return
            source = SessionStore.open(self.host._sessions_root, source_id)
            try:
                source_meta = source.read_meta()
                original = source_meta.get("spec_run")
                if not isinstance(original, dict):
                    return
                loop_id, card_id = review_info.get("loop_id"), review_info.get("card_id")
                if loop_id and card_id:
                    loop = self.store.load_record(loop_id)
                    card = next((item for item in loop.cards if item.id == card_id), None)
                    if (loop.state == "cancelled" or card is None or card.session_id != source_id
                            or card.state not in ("review", "failed")):
                        return
                prior = source_meta.get("review")
                if isinstance(prior, dict) and prior.get("session_id") != session_id:
                    return
                worktree = Path(review_info.get("worktree_path", ""))
                try:
                    current = worktree_diff(worktree)
                except (OSError, WorktreeError) as exc:
                    reason = (f"review worktree is missing: {worktree}" if not worktree.is_dir()
                              else f"review worktree could not be read: {exc}")
                    source.update_meta(lambda meta: meta.update({"review": {
                        "session_id": session_id, "verdict": "failed", "error": reason,
                        "follow_ups": [], "not_copied": [],
                    }}))
                    if loop_id and card_id:
                        card["reason"] = reason
                        self._loop_card_changed(loop, card)
                    self._finish_loop_card_review(review_info)
                    return
                answer = next((message.text for message in reversed(leader._chat_messages) if message.role == Role.ASSISTANT and message.text.strip()), "")
                verdict, follow_ups = review_verdict(answer)
                if not isinstance(terminal_event, RunFinished) or terminal_event.stopped_reason != "final_response":
                    verdict, follow_ups = "no-verdict", []
                baseline_files = review_info.get("baseline_files", {})
                unchanged = isinstance(baseline_files, dict) and all(
                    (
                        (worktree / name).is_file()
                        and hashlib.sha256((worktree / name).read_bytes()).hexdigest() == digest
                        if digest is not None else not (worktree / name).exists()
                    )
                    for name, digest in baseline_files.items()
                )
                allowed_additions = set(follow_ups) if verdict == "follow-ups" else set()
                additions = set(current.files) - set(baseline_files if isinstance(baseline_files, dict) else {})
                if review_info.get("loop_id"):
                    loop_follow_ups = {
                        value for value in follow_ups
                        if (worktree / value).is_file()
                    }
                    unchanged = unchanged and loop_follow_ups == set(follow_ups)
                unchanged = unchanged and additions <= allowed_additions
                if not unchanged:
                    verdict = "tree-changed"
                    follow_ups = []
                copied: list[str] = []
                not_copied: list[str] = []
                spec_directory = Path(str(original.get("spec", ""))).parent
                if verdict == "follow-ups":
                    for value in follow_ups:
                        relative = Path(value)
                        if relative.is_absolute() or ".." in relative.parts or not relative.as_posix().startswith(spec_directory.as_posix() + "/"):
                            not_copied.append(value)
                            continue
                        source_file = worktree / relative
                        target_file = self.host._repo_root / relative
                        if not source_file.is_file() or target_file.exists():
                            not_copied.append(value)
                            continue
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_file, target_file)
                        copied.append(value)
                review = {
                    "session_id": session_id, "verdict": verdict,
                    "follow_ups": copied, "not_copied": not_copied,
                }
                if isinstance(terminal_event, RunFinished):
                    review["stopped_reason"] = terminal_event.stopped_reason
                elif isinstance(terminal_event, RunFailed):
                    review["error"] = terminal_event.error
                source.update_meta(lambda meta: meta.update({"review": review}))
            finally:
                source.close()
            if isinstance(review_info, dict):
                self._finish_loop_card_review(review_info)

