"""Checks for subagent worktree isolation and diffs."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest import mock

from symphonai_api.agent_spec import AgentSpec, ModelSelector
from symphonai_api.config import ConfigError, load_config
from symphonai_api.checkpoints import CheckpointStore
from symphonai_api.models import Message, ModelRequest, ModelResponse, Role, ToolCall
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.session import SessionStore
from symphonai_api.leader import DispatchSubagentTool, dispatch_subagent_tool_schema

from scripts.checks.harness import check, fail


class _RecordingProvider(FakeModelProvider):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__(responses)
        self.requests: list[ModelRequest] = []

    def create_response(self, request: ModelRequest, *, cancel=None) -> ModelResponse:
        self.requests.append(request)
        return super().create_response(request, cancel=cancel)


def _git(root: Path, *arguments: str, input: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        input=input,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        fail(f"git {' '.join(arguments)} failed: {result.stderr.decode(errors='replace')}")
    return result.stdout


def _repository(root: Path, *, subdirectory: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "worktree-check@example.test")
    _git(root, "config", "user.name", "Worktree Check")
    repo_root = root / subdirectory if subdirectory else root
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "a.py").write_text("value = 1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "initial")
    return repo_root


def _response(text: str = "", *calls: ToolCall) -> ModelResponse:
    return ModelResponse(Message(Role.ASSISTANT, text, tool_calls=list(calls)))


def _call(name: str, task: str, isolation: str | None = None) -> ToolCall:
    arguments = {"subagent_name": name, "task": task}
    if isolation is not None:
        arguments["isolation"] = isolation
    return ToolCall("dispatch", "dispatch_subagent", arguments)


def _dispatch_tool(
    repo_root: Path,
    provider: _RecordingProvider,
    session: SessionStore | None,
    *,
    checkpoints: CheckpointStore | None = None,
    policy: PermissionPolicy | None = None,
) -> tuple[DispatchSubagentTool, PermissionPolicy]:
    policy = policy or PermissionPolicy(
        repo_root,
        allowed_write_scope=[repo_root],
        shell_enabled=True,
        shell_allowlist=[("pwd",)],
    )
    spec = AgentSpec(
        name="worker",
        prompt="",
        model=ModelSelector("fake", "test-model"),
        policy_ceiling=policy,
    )
    tool = DispatchSubagentTool(
        subagent_provider=provider,
        leader_policy=policy,
        session=session,
        subagent_specs={"worker": spec},
        subagent_max_turns=8,
        checkpoints=checkpoints,
    )
    return tool, policy


def _session(root: Path, repo_root: Path) -> SessionStore:
    return SessionStore(root / "sessions", "session", repo_root=repo_root)


@check("worktree.dependencies_are_linked_by_default")
def check_dependencies_are_linked_by_default() -> None:
    from symphonai_api.worktree import create_worktree, remove_worktree

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repo = _repository(base / "repo")
        home = base / "home"
        dependencies = {
            "node_modules": "node_modules/p",
            ".venv": ".venv/p",
            "venv": "venv/p",
        }
        for directory, probe in dependencies.items():
            target = repo / probe
            target.parent.mkdir(parents=True)
            target.write_text(directory, encoding="utf-8")
        with mock.patch.dict("os.environ", {"SYMPHONAI_HOME": str(home)}):
            default_path = base / "sessions" / "default"
            default = create_worktree(repo, default_path)
            try:
                for directory, probe in dependencies.items():
                    linked = default / probe
                    if not linked.is_file() or linked.read_text(encoding="utf-8") != directory:
                        fail(f"default dependency {directory} was not linked into the worktree")
                    if not (default / directory).is_symlink():
                        fail(f"default dependency {directory} is not a symlink")
            finally:
                remove_worktree(repo, default_path)

            project_config = repo / ".symphonai" / "config.toml"
            project_config.parent.mkdir(parents=True)
            project_config.write_text("[worktree]\nsymlink = []\n", encoding="utf-8")
            empty_path = base / "sessions" / "empty"
            empty = create_worktree(repo, empty_path)
            try:
                if any((empty / name).is_symlink() for name in dependencies):
                    fail("empty worktree.symlink configuration created dependency links")
            finally:
                remove_worktree(repo, empty_path)

    schema = dispatch_subagent_tool_schema(1)["function"]
    description = schema["description"]
    isolation = schema["parameters"]["properties"]["isolation"]["description"]
    expected_description = (
        "Set isolation to worktree only for a task that should start from the last commit and stay apart from the person's tree. "
        "Such a subagent does not see uncommitted changes, including yours, and its diff comes back in the result for you to apply. "
        "Explorers, and follow-ups on uncommitted work, share the tree."
    )
    if expected_description not in description or isolation != (
        "Run the subagent in its own git worktree, from the last commit."
    ):
        fail(f"dispatch description did not explain worktree isolation: {description!r}, {isolation!r}")


@check("worktree.linked_paths_stay_out_of_the_diff")
def check_linked_paths_stay_out_of_the_diff() -> None:
    from symphonai_api.worktree import create_worktree, remove_worktree, worktree_diff

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repo = _repository(base / "repo")
        (repo / "a.txt").write_text("original\n", encoding="utf-8")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "--quiet", "-m", "add source file")
        (repo / "tracked-link").symlink_to("a.txt")
        _git(repo, "add", "tracked-link")
        _git(repo, "commit", "--quiet", "-m", "add tracked symlink")
        (repo / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
        _git(repo, "add", ".gitignore")
        _git(repo, "commit", "--quiet", "-m", "ignore dependencies")
        (repo / "node_modules").mkdir()
        (repo / "node_modules" / "p").write_text("package", encoding="utf-8")
        with mock.patch.dict("os.environ", {"SYMPHONAI_HOME": str(base / "home")}):
            worktree_path = base / "sessions" / "worktree"
            worktree = create_worktree(repo, worktree_path)
            try:
                (worktree / "a.txt").write_text("edited\n", encoding="utf-8")
                (worktree / "new-link").symlink_to("b.txt")
                (worktree / "tracked-link").unlink()
                (worktree / "tracked-link").symlink_to("b.txt")
                diff = worktree_diff(worktree)
                if diff.files != ("a.txt", "new-link", "tracked-link"):
                    fail(f"worker symlink changes were omitted or dependency link leaked: {diff!r}")
                if any(
                    marker not in diff.patch
                    for marker in (
                        "diff --git a/new-link b/new-link",
                        "diff --git a/tracked-link b/tracked-link",
                        "-a.txt",
                        "+b.txt",
                    )
                ) or "node_modules" in diff.patch:
                    fail(f"worktree patch lost symlink changes or included dependency link: {diff.patch!r}")
            finally:
                remove_worktree(repo, worktree_path)

        real_repo = _repository(base / "real-repo")
        (real_repo / "node_modules").mkdir()
        (real_repo / "node_modules" / "p").write_text("package", encoding="utf-8")
        with mock.patch.dict("os.environ", {"SYMPHONAI_HOME": str(base / "home")}):
            real_path = base / "sessions" / "real-directory"
            real_worktree = create_worktree(real_repo, real_path)
            try:
                (real_worktree / "node_modules").unlink()
                (real_worktree / "node_modules").mkdir()
                (real_worktree / "node_modules" / "p").write_text("worker package", encoding="utf-8")
                diff = worktree_diff(real_worktree)
                if diff.files != ("node_modules/p",) or "worker package" not in diff.patch:
                    fail(f"worker-owned dependency directory was omitted from its diff: {diff!r}")
            finally:
                remove_worktree(real_repo, real_path)

        top = base / "monorepo"
        package = _repository(top, subdirectory="pkg")
        (top / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
        _git(top, "add", ".gitignore")
        _git(top, "commit", "--quiet", "-m", "ignore dependencies")
        (package / "a.txt").write_text("original\n", encoding="utf-8")
        _git(top, "add", "pkg/a.txt")
        _git(top, "commit", "--quiet", "-m", "add package source")
        (package / "node_modules").mkdir()
        (package / "node_modules" / "p").write_text("package", encoding="utf-8")
        with mock.patch.dict("os.environ", {"SYMPHONAI_HOME": str(base / "home")}):
            package_worktree_path = base / "sessions" / "monorepo-worktree"
            package_root = create_worktree(package, package_worktree_path)
            try:
                (package_root / "a.txt").write_text("edited\n", encoding="utf-8")
                package_diff = worktree_diff(package_root)
                top_diff = worktree_diff(package_worktree_path)
                if package_diff.files != ("a.txt",) or "node_modules" in package_diff.patch:
                    fail(f"project-root diff included a linked dependency: {package_diff!r}")
                if top_diff.files != ("pkg/a.txt",) or "node_modules" in top_diff.patch:
                    fail(f"worktree-top diff included a linked dependency: {top_diff!r}")
            finally:
                remove_worktree(package, package_worktree_path)


@check("worktree.diff_isolated_and_applyable")
def check_diff_isolated_and_applyable() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repo = _repository(base / "repo")
        main_file = repo / "a.py"
        session = _session(base, repo)
        checkpoints = CheckpointStore(base / "checkpoints", repo)
        provider = _RecordingProvider([
            _response("", ToolCall("read", "read_file", {"path": "a.py"})),
            _response("", ToolCall("edit", "edit_file", {
                "path": "a.py", "old_string": "value = 1", "new_string": "value = 2",
            })),
            _response("", ToolCall("write", "write_file", {
                "path": "new.py", "content": "created = True\n",
            })),
            _response("finished"),
        ])
        tool, policy = _dispatch_tool(repo, provider, session, checkpoints=checkpoints)
        try:
            result = tool.execute(_call("worker", "edit the files", "worktree"), policy)
            if not result.ok:
                fail(f"worktree dispatch failed: {result.error}")
            if main_file.read_text(encoding="utf-8") != "value = 1\n" or (repo / "new.py").exists():
                fail("worktree edits changed the main tree")
            if result.payload != {
                "kind": "worktree_diff",
                "subagent": "worker",
                "worktree": str(tool.pool["worker"].worktree_path),
                "files": ["a.py", "new.py"],
            }:
                fail(f"worktree diff payload was incomplete: {result.payload!r}")
            if "Worktree " not in result.content or "diff --git a/a.py b/a.py" not in result.content:
                fail(f"worktree result omitted its patch: {result.content!r}")
            patch = result.content[result.content.index("diff --git"):]
            _git(repo, "apply", "--check", input=patch.encode("utf-8", errors="surrogateescape"))
            record = tool.pool["worker"]
            if record.agent._tools["write_file"]._checkpoints is not None or checkpoints.entries():
                fail("worktree child used the leader's checkpoint store")
        finally:
            session.close()


@check("worktree.tools_use_the_isolated_root")
def check_tools_use_the_isolated_root() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repo = _repository(base / "repo")
        (repo.parent / "outside").write_text("outside", encoding="utf-8")
        session = _session(base, repo)
        provider = _RecordingProvider([
            _response("", ToolCall("pwd", "run_shell", {"argv": ["pwd"]})),
            _response("", ToolCall("outside", "read_file", {"path": "../outside"})),
            _response("", ToolCall("inside", "read_file", {"path": "a.py"})),
            _response("done"),
        ])
        tool, policy = _dispatch_tool(repo, provider, session)
        try:
            result = tool.execute(_call("worker", "inspect", "worktree"), policy)
            worktree = tool.pool["worker"].worktree_path
            tool_results = [
                message.tool_result
                for request in provider.requests
                for message in request.messages
                if message.tool_result is not None
            ]
            if not result.ok or worktree is None:
                fail(f"worktree inspection dispatch failed: {result.error!r}")
            if not any(worktree.as_posix() in item.content for item in tool_results):
                fail(f"run_shell did not execute in the worktree: {tool_results!r}")
            if not any(not item.ok and item.error for item in tool_results):
                fail("read_file accepted a path outside the worktree root")
            if not any(item.ok and "value = 1" in item.content for item in tool_results):
                fail("read_file could not read a file inside the worktree")
        finally:
            session.close()


@check("worktree.reuses_named_workspace")
def check_reuses_named_workspace() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repo = _repository(base / "repo")
        session = _session(base, repo)
        provider = _RecordingProvider([
            _response("", ToolCall("first", "write_file", {
                "path": "first.py", "content": "first = 1\n",
            })),
            _response("first done"),
            _response("", ToolCall("read", "read_file", {"path": "first.py"})),
            _response("", ToolCall("second", "write_file", {
                "path": "second.py", "content": "second = 2\n",
            })),
            _response("second done"),
        ])
        tool, policy = _dispatch_tool(repo, provider, session)
        try:
            first = tool.execute(_call("worker", "start", "worktree"), policy)
            first_path = tool.pool["worker"].worktree_path
            second = tool.execute(_call("worker", "continue"), policy)
            second_path = tool.pool["worker"].worktree_path
            read_results = [
                message.tool_result
                for request in provider.requests[3:]
                for message in request.messages
                if message.tool_result is not None and message.tool_result.tool_call_id == "read"
            ]
            if not first.ok or not second.ok or first_path != second_path:
                fail("a named subagent did not reuse its worktree")
            if not read_results or not any("first = 1" in item.content for item in read_results):
                fail(f"the second dispatch did not see first-dispatch edits: {read_results!r}")
            if second.payload["files"] != ["first.py", "second.py"]:
                fail(f"the second result omitted accumulated changes: {second.payload!r}")
        finally:
            session.close()


@check("worktree.starts_from_head")
def check_starts_from_head() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repo = _repository(base / "repo")
        (repo / "a.py").write_text("uncommitted = True\n", encoding="utf-8")
        session = _session(base, repo)
        provider = _RecordingProvider([
            _response("", ToolCall("read", "read_file", {"path": "a.py"})),
            _response("done"),
        ])
        tool, policy = _dispatch_tool(repo, provider, session)
        try:
            result = tool.execute(_call("worker", "read", "worktree"), policy)
            read_result = next(
                message.tool_result
                for request in provider.requests
                for message in request.messages
                if message.tool_result is not None and message.tool_result.tool_call_id == "read"
            )
            if not result.ok or "value = 1" not in read_result.content or "uncommitted" in read_result.content:
                fail(f"new worktree inherited main-tree uncommitted edits: {read_result!r}")
        finally:
            session.close()


@check("worktree.subdirectory_root")
def check_subdirectory_root() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        top = base / "repo"
        repo = _repository(top, subdirectory="nested")
        session = _session(base, repo)
        provider = _RecordingProvider([
            _response("", ToolCall("pwd", "run_shell", {"argv": ["pwd"]})),
            _response("", ToolCall("write", "write_file", {
                "path": "new.py", "content": "new = True\n",
            })),
            _response("done"),
        ])
        tool, policy = _dispatch_tool(repo, provider, session)
        try:
            result = tool.execute(_call("worker", "work here", "worktree"), policy)
            worktree_root = tool.pool["worker"].worktree_path
            if not result.ok or worktree_root != (session.directory / "worktrees" / "worker" / "nested").resolve():
                fail(f"worktree root did not mirror the repository subdirectory: {worktree_root!r}")
            if (
                repo.joinpath("new.py").exists()
                or (repo / "a.py").read_text(encoding="utf-8") != "value = 1\n"
            ):
                fail("subdirectory worktree changed the main tree")
            if str(worktree_root) not in result.content:
                fail(f"shell or worktree report did not name the child root: {result.content!r}")
            patch = result.content[result.content.index("diff --git"):]
            _git(top, "apply", "--check", input=patch.encode("utf-8", errors="surrogateescape"))
            if "diff --git a/nested/new.py b/nested/new.py" not in patch:
                fail(f"subdirectory worktree patch used the wrong repository paths: {patch!r}")
        finally:
            session.close()


@check("worktree.refusals_leave_no_worktree")
def check_refusals_leave_no_worktree() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repo = _repository(base / "repo")
        provider = _RecordingProvider([_response("shared")])
        policy = PermissionPolicy(repo, allowed_write_scope=[repo])
        no_session_tool, _ = _dispatch_tool(repo, provider, None, policy=policy)
        no_session = no_session_tool.execute(_call("worker", "task", "worktree"), policy)
        if no_session.ok or "requires a session" not in (no_session.error or ""):
            fail(f"worktree dispatch without a session was not refused: {no_session!r}")

        outside = base / "outside-repo"
        outside.mkdir()
        outside_session = _session(base / "outside-session", outside)
        outside_provider = _RecordingProvider([_response("unused")])
        outside_tool, outside_policy = _dispatch_tool(
            outside, outside_provider, outside_session,
        )
        try:
            non_git = outside_tool.execute(_call("worker", "task", "worktree"), outside_policy)
            if non_git.ok or "not a git repository" not in (non_git.error or "").casefold():
                fail(f"non-git worktree dispatch omitted git's error: {non_git!r}")
            if (outside_session.directory / "worktrees" / "worker").exists():
                fail("failed worktree creation left a worktree behind")
        finally:
            outside_session.close()

        session = _session(base, repo)
        shared_tool, shared_policy = _dispatch_tool(repo, provider, session)
        try:
            first = shared_tool.execute(_call("worker", "shared task"), shared_policy)
            refused = shared_tool.execute(_call("worker", "switch", "worktree"), shared_policy)
            if not first.ok or refused.ok or "already works in the shared tree" not in (refused.error or ""):
                fail(f"shared-tree subagent changed isolation mode: {refused!r}")
            if (session.directory / "worktrees" / "worker").exists():
                fail("refusing shared-to-worktree switch left a worktree behind")
        finally:
            session.close()


@check("worktree.policy_rerooting")
def check_policy_rerooting() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "repo"
        new_root = base / "worktree"
        callback = lambda *args: True
        policy = PermissionPolicy(
            root,
            allowed_write_scope=[root / "src", base / "elsewhere"],
            forbidden_patterns=(".env",),
            shell_enabled=True,
            shell_allowlist=[("pwd",)],
            fetch_enabled=True,
            fetch_allowlist=["example.com"],
            shell_timeout_seconds=12,
            shell_output_limit_chars=12345,
            mode="ask",
            approval_callback=callback,
        )
        rerooted = policy.rerooted(new_root)
        if rerooted.repo_root != new_root.resolve() or rerooted.allowed_write_scope != [(new_root / "src").resolve()]:
            fail(f"rerooted policy mapped an invalid write scope: {rerooted!r}")
        if (
            rerooted.forbidden_patterns != policy.forbidden_patterns
            or rerooted.shell_enabled != policy.shell_enabled
            or rerooted.shell_allowlist != policy.shell_allowlist
            or rerooted.fetch_enabled != policy.fetch_enabled
            or rerooted.fetch_allowlist != policy.fetch_allowlist
            or rerooted.shell_timeout_seconds != policy.shell_timeout_seconds
            or rerooted.shell_output_limit_chars != policy.shell_output_limit_chars
            or rerooted.mode != policy.mode
            or rerooted.approval_callback is not callback
        ):
            fail("rerooted policy changed a non-path permission field")


@check("worktree.shared_dispatch_stays_shared")
def check_shared_dispatch_stays_shared() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repo = _repository(base / "repo")
        session = _session(base, repo)
        provider = _RecordingProvider([
            _response("", ToolCall("write", "write_file", {
                "path": "shared.py", "content": "shared = True\n",
            })),
            _response("done"),
        ])
        tool, policy = _dispatch_tool(repo, provider, session)
        try:
            result = tool.execute(_call("worker", "work in place"), policy)
            if not result.ok or result.payload is not None or not (repo / "shared.py").is_file():
                fail(f"dispatch without isolation changed its established behavior: {result!r}")
            if tool.pool["worker"].worktree_path is not None:
                fail("ordinary dispatch unexpectedly created a worktree")
        finally:
            session.close()


@check("worktree.configured_symlink_is_added")
def check_configured_symlink_is_added() -> None:
    from symphonai_api.worktree import create_worktree, remove_worktree

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = _repository(root / "repo")
        (repo / ".symphonai").mkdir()
        (repo / ".symphonai" / "config.toml").write_text('[worktree]\nsymlink = [".venv"]\n')
        (repo / ".venv").mkdir()
        (repo / ".venv" / "probe").write_text("ok")
        admin = root / "sessions" / "worktree"
        child = create_worktree(repo, admin)
        try:
            link = child / ".venv"
            if not link.is_symlink() or (link / "probe").read_text() != "ok":
                fail("configured main-tree directory was not linked into its worktree")
        finally:
            remove_worktree(repo, admin)


@check("worktree.configured_symlink_config_validation")
def check_configured_symlink_config_validation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        (repo / ".symphonai").mkdir(parents=True)
        config = repo / ".symphonai" / "config.toml"
        config.write_text('[worktree]\nsymlink = [".venv", "tools/cache"]\n')
        loaded = load_config(repo_root=repo, home=root / "home")
        if loaded.get("worktree.symlink") != [".venv", "tools/cache"]:
            fail("configured worktree symlink paths were not loaded")


@check("worktree.configured_symlink_escape_refused")
def check_configured_symlink_escape_refused() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        (repo / ".symphonai").mkdir(parents=True)
        config = repo / ".symphonai" / "config.toml"
        for entry in ("../outside", "/absolute"):
            config.write_text(f'[worktree]\nsymlink = ["{entry}"]\n')
            try:
                load_config(repo_root=repo, home=root / "home")
            except ConfigError as exc:
                if "worktree.symlink" not in str(exc):
                    fail(f"invalid symlink path error omitted its key: {exc}")
            else:
                fail(f"unsafe worktree symlink path was accepted: {entry}")
