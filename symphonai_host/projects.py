"""Project scoped runtime resources owned by a host."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import hashlib
import threading
from pathlib import Path
from collections.abc import Callable, Mapping

from symphonai_api.extensions import load_extensions
from symphonai_api.lsp import LspManager
from symphonai_api.mcp_pool import McpPool
from symphonai_api.paths import ensure_private_directory, write_atomic
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.runner import standard_tool_registry
from symphonai_host.run import HostRun


@dataclass
class ProjectRuntime:
    root: Path
    run: HostRun
    pool: McpPool | None = None
    lsp: LspManager | None = None
    _closed: bool = field(default=False, init=False)
    _reload_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _retired_pools: list = field(default_factory=list, init=False)
    _skill_digests: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.run.extensions is not None:
            self._skill_digests = {name: hashlib.sha256(skill.body().encode()).hexdigest()
                                   for name, skill in self.run.extensions.skills.items()}

    def reload_extensions(self, *, home: Path | None = None) -> dict:
        """Prepare outside the host lock, then publish a complete replacement."""
        with self._reload_lock:
            previous = self.run.extensions
            changes = {kind: {"added": [], "removed": [], "changed": [], "failed": []}
                       for kind in ("skills", "agents", "plugins", "mcp_servers")}
            try:
                extensions = load_extensions(repo_root=self.root, home=home)
                if extensions.config.warnings:
                    raise ValueError("; ".join(extensions.config.warnings))
                skill_digests = {name: hashlib.sha256(skill.body().encode()).hexdigest()
                                 for name, skill in extensions.skills.items()}
            except (ValueError, OSError) as exc:
                for kind in ("skills", "agents", "plugins"):
                    if kind in str(exc):
                        changes[kind]["failed"].append({"error": str(exc)})
                return {"changes": changes, "error": str(exc)}
            for kind, delta in changes.items():
                before = {} if previous is None else getattr(previous, kind)
                after = getattr(extensions, kind)
                if kind == "mcp_servers":
                    before = {spec.name: spec for spec in before}
                    after = {spec.name: spec for spec in after}
                delta["added"] = sorted(after.keys() - before.keys())
                delta["removed"] = sorted(before.keys() - after.keys())
                if kind == "skills":
                    delta["changed"] = sorted(name for name in before.keys() & after.keys()
                                              if (before[name].discoverability_text(), before[name].path,
                                                  self._skill_digests.get(name)) !=
                                                 (after[name].discoverability_text(), after[name].path,
                                                  skill_digests[name]))
                else:
                    delta["changed"] = sorted(name for name in before.keys() & after.keys()
                                              if before[name] != after[name])
            reuse = self.pool is not None and previous is not None and previous.mcp_servers == extensions.mcp_servers
            pool = self.pool if reuse else McpPool(
                extensions.mcp_servers, cwd=self.root, reserved_names=set(standard_tool_registry()))
            try:
                tools = pool.start()
                tools.update(pool.resource_tools)
                changes["mcp_servers"]["failed"] = [
                    {"name": name, "error": state["error"]}
                    for name, state in pool.states.items() if state["state"] != "started"
                ]
                retained = self.run.replace_extensions(extensions, pool, tools)
            except BaseException:
                if not reuse:
                    pool.close()
                raise
            old, self.pool = self.pool, pool
            self._skill_digests = skill_digests
            if old is not None and old is not pool:
                if retained:
                    self._retired_pools.append(old)
                else:
                    old.close()
            return {"changes": changes, "error": None}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.run.close()
        if self.lsp is not None:
            self.lsp.close()
        if self.pool is not None:
            self.pool.close()
        for pool in self._retired_pools:
            pool.close()


def build_project(
    root: Path,
    broker,
    provider,
    *,
    publish_approval,
    base_prompt: str = "",
    subagent_base_prompt: str = "",
    max_turns: int | None = None,
    sessions_root: Path | None = None,
    memory_root: Path | None = None,
    provider_factory: Callable | None = None,
    extensions_loader=load_extensions,
    pool_factory=McpPool,
    lsp_factory=LspManager,
    policy_factory=PermissionPolicy,
) -> ProjectRuntime:
    root = root.resolve()
    extensions = extensions_loader(repo_root=root)
    pool = pool_factory(
        extensions.mcp_servers, cwd=root,
        reserved_names=set(standard_tool_registry()),
    )
    lsp = None
    try:
        mcp_tools = pool.start()
        mcp_tools.update(pool.resource_tools)
        lsp = lsp_factory(extensions.lsp_servers, root=root)
        policy = policy_factory(
            repo_root=root,
            allowed_write_scope=[root],
            shell_enabled=True,
            shell_allowlist=[()],
            shell_sandbox=extensions.config.get("sandbox.shell", False),
            sandbox_network=extensions.config.get("sandbox.network", False),
            fetch_enabled=True,
        )
        ceiling = getattr(extensions, "ceiling", None)
        reload_policy = policy
        if ceiling is not None:
            policy = ceiling.bound(policy)
        run = HostRun(
            provider, policy, broker,
            base_prompt=base_prompt,
            subagent_base_prompt=subagent_base_prompt,
            max_turns=max_turns,
            provider_factory=provider_factory,
            publish_approval=publish_approval,
            sessions_root=sessions_root,
            memory_root=memory_root,
            extensions=extensions,
            mcp_tools=mcp_tools,
            mcp_pool=pool,
            mcp_instructions=pool.instructions,
            lsp=lsp,
            reload_policy=reload_policy,
        )
        return ProjectRuntime(root, run, pool, lsp)
    except BaseException:
        if lsp is not None:
            lsp.close()
        pool.close()
        raise


def record_recent_project(home: Path, root: Path) -> None:
    path = home / "projects.json"
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        recent = values if isinstance(values, list) else []
    except (OSError, ValueError):
        recent = []
    root_string = str(root.resolve())
    recent = [root_string, *(item for item in recent if item != root_string)][:20]
    try:
        ensure_private_directory(home)
        write_atomic(path, json.dumps(recent), mode=0o600, fsync=True)
    except OSError:
        return


def recent_projects(home: Path) -> list[str]:
    try:
        values = json.loads((home / "projects.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return []
    return [value for value in values if isinstance(value, str)] if isinstance(values, list) else []


def choose_start_root(
    explicit: Path | None, cwd: Path, home: Path, project_home: Path | None = None,
) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    current = cwd.resolve()
    home = home.resolve()
    if current != Path("/") and current != home:
        return current
    recent = recent_projects(project_home if project_home is not None else home / ".symphonai")
    return next((Path(value).resolve() for value in recent
                 if Path(value).is_dir()), None)
