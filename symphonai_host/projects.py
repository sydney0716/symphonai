"""Project scoped runtime resources owned by a host."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from collections.abc import Callable, Mapping

from symphonai_api.extensions import load_extensions
from symphonai_api.lsp import LspManager
from symphonai_api.mcp_pool import McpPool
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

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.run.close()
        if self.lsp is not None:
            self.lsp.close()
        if self.pool is not None:
            self.pool.close()


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
            lsp=lsp,
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
        home.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(recent, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError:
        return
    finally:
        if "temporary" in locals():
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


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
