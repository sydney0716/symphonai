"""Checks for layered configuration and load-time capability ceilings."""

from __future__ import annotations

import ast
import itertools
import tempfile
from pathlib import Path
from unittest import mock

from symphonai_api.agent_file import AgentFileError, load_agent_directory, load_agent_file
from symphonai_api.agent_spec import ModelSelector
import symphonai_api.config as config_module
from symphonai_api.config import (
    CapabilityCeiling,
    ConfigError,
    Provenance,
    Scope,
    load_config,
)
from symphonai_api.permissions import PermissionPolicy
from scripts.checks.agent_spec import _forbidden_imports
from scripts.checks.harness import check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_CONFIG_IMPORTS = {
    "agent_loop",
    "leader",
    "runner",
    "agent_run",
    "agent_spec",
    "agent_file",
    "child_context",
    "provider_catalog",
    "providers",
}


def _roots(temporary: str) -> tuple[Path, Path]:
    root = Path(temporary)
    return root / "repo", root / "home"


def _scope_path(scope: Scope, repo_root: Path, home: Path) -> Path:
    if scope is Scope.USER:
        return home / ".symphonai" / "config.toml"
    if scope is Scope.PROJECT:
        return repo_root / ".symphonai" / "config.toml"
    if scope is Scope.PRIVATE:
        return repo_root / ".symphonai" / "config.local.toml"
    raise ValueError("session scope has no path")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _settings(value: str) -> str:
    return f'[agents]\ndirectory = "{value}"\n'


def _load_subset(scopes: tuple[Scope, ...], temporary: str):
    repo_root, home = _roots(temporary)
    session = None
    for scope in scopes:
        if scope is Scope.SESSION:
            session = {"agents": {"directory": scope.value}}
        else:
            _write(_scope_path(scope, repo_root, home), _settings(scope.value))
    return load_config(repo_root=repo_root, home=home, session=session)


@check("config.scope_precedence")
def scope_precedence() -> None:
    ordered = tuple(Scope)
    for size in range(1, len(ordered) + 1):
        for selected in itertools.combinations(ordered, size):
            with tempfile.TemporaryDirectory() as temporary:
                resolved = _load_subset(selected, temporary)
            expected = max(selected, key=ordered.index)
            if resolved.get("agents.directory") != expected.value:
                fail(f"wrong winner for {selected!r}: {resolved.values!r}")
            if resolved.scope_of("agents.directory") is not expected:
                fail(f"wrong winning scope for {selected!r}: {resolved.provenance!r}")


@check("config.merges_per_leaf")
def merges_per_leaf() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo_root, home = _roots(temporary)
        _write(
            _scope_path(Scope.USER, repo_root, home),
            (
                '[agents]\ndirectory = "user-agents"\n'
                '[agents.ceiling]\nshell_enabled = true\n'
                'fetch_allowlist = ["user.example", "lower-only.example"]\n'
            ),
        )
        _write(
            _scope_path(Scope.PROJECT, repo_root, home),
            (
                "[agents.ceiling]\nfetch_enabled = true\n"
                'fetch_allowlist = ["project.example"]\n'
            ),
        )
        resolved = load_config(repo_root=repo_root, home=home)
        expected = {
            "agents.directory": "user-agents",
            "agents.ceiling.shell_enabled": True,
            "agents.ceiling.fetch_enabled": True,
            "agents.ceiling.fetch_allowlist": ["project.example"],
        }
        if dict(resolved.values) != expected:
            fail(f"per-leaf merge or list replacement changed: {resolved.values!r}")


@check("config.missing_and_malformed")
def missing_and_malformed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo_root, home = _roots(temporary)
        empty = load_config(repo_root=repo_root, home=home)
        if empty.values or empty.provenance:
            fail(f"missing scopes contributed values: {empty!r}")
        for scope in (Scope.USER, Scope.PROJECT, Scope.PRIVATE):
            path = _scope_path(scope, repo_root, home)
            _write(path, "[agents\n")
            try:
                load_config(
                    repo_root=repo_root,
                    home=home,
                    session={"agents": {"directory": "higher"}},
                )
            except ConfigError as exc:
                if str(path) not in str(exc):
                    fail(f"malformed {scope.value} error omitted its path: {exc!r}")
            else:
                fail(f"malformed {scope.value} scope was skipped")
            path.unlink()
        for malformed_session in ({"agents": []}, {"agents": None}):
            try:
                load_config(repo_root=repo_root, home=home, session=malformed_session)
            except ConfigError as exc:
                if "<session>" not in str(exc) or "agents" not in str(exc):
                    fail(f"malformed session error lacked source or key: {exc!r}")
            else:
                fail("malformed session mapping was accepted")


@check("config.provenance")
def provenance() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo_root, home = _roots(temporary)
        user_path = _scope_path(Scope.USER, repo_root, home)
        project_path = _scope_path(Scope.PROJECT, repo_root, home)
        private_path = _scope_path(Scope.PRIVATE, repo_root, home)
        _write(user_path, '[agents]\ndirectory = "user"\n')
        _write(project_path, "[agents.ceiling]\nshell_enabled = true\n")
        _write(private_path, "[agents.ceiling]\nfetch_enabled = true\n")
        with mock.patch.object(
            config_module.Path,
            "home",
            side_effect=AssertionError("Path.home was consulted"),
        ):
            resolved = load_config(
                repo_root=repo_root,
                home=home,
                session={"agents": {"ceiling": {"modes": ["plan"]}}},
            )
        expected = {
            "agents.directory": Provenance("agents.directory", Scope.USER, user_path),
            "agents.ceiling.shell_enabled": Provenance(
                "agents.ceiling.shell_enabled", Scope.PROJECT, project_path
            ),
            "agents.ceiling.fetch_enabled": Provenance(
                "agents.ceiling.fetch_enabled", Scope.PRIVATE, private_path
            ),
            "agents.ceiling.modes": Provenance(
                "agents.ceiling.modes", Scope.SESSION, None
            ),
        }
        if dict(resolved.provenance) != expected:
            fail(f"provenance differed: {resolved.provenance!r}")
        if any(
            (origin.scope is Scope.SESSION) != (origin.source is None)
            for origin in resolved.provenance.values()
        ):
            fail("source was None outside SESSION or non-None inside SESSION")


def _expect_refusal(
    field: str,
    ceiling: CapabilityCeiling,
    policy: PermissionPolicy,
    source: Path,
) -> None:
    try:
        ceiling.refuse(policy, source=source)
    except ConfigError as exc:
        if str(source) not in str(exc) or field not in str(exc):
            fail(f"ceiling refusal omitted {source!s} or {field!r}: {exc!r}")
    else:
        fail(f"ceiling accepted an over-reaching {field}")


@check("config.ceiling_refuses")
def ceiling_refuses() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo_root = Path(temporary) / "repo"
        source = Path(temporary) / "agent.toml"
        cases = (
            (
                "allowed_write_scope",
                CapabilityCeiling(
                    allowed_write_scope=((repo_root / "src").resolve(),)
                ),
                PermissionPolicy(repo_root, allowed_write_scope=[repo_root / "srcextra"]),
            ),
            (
                "shell_enabled",
                CapabilityCeiling(shell_enabled=False),
                PermissionPolicy(repo_root, shell_enabled=True),
            ),
            (
                "shell_allowlist",
                CapabilityCeiling(shell_allowlist=(("python3",),)),
                PermissionPolicy(repo_root, shell_allowlist=[("make",)]),
            ),
            (
                "fetch_enabled",
                CapabilityCeiling(fetch_enabled=False),
                PermissionPolicy(repo_root, fetch_enabled=True),
            ),
            (
                "fetch_allowlist",
                CapabilityCeiling(fetch_allowlist=("docs.example",)),
                PermissionPolicy(repo_root, fetch_allowlist=["other.example"]),
            ),
            (
                "modes",
                CapabilityCeiling(modes=("plan",)),
                PermissionPolicy(repo_root, mode="auto"),
            ),
        )
        for field, ceiling, policy in cases:
            _expect_refusal(field, ceiling, policy, source)
        equal = CapabilityCeiling(
            allowed_write_scope=((repo_root / "src").resolve(),),
            shell_enabled=True,
            shell_allowlist=(("python3",),),
            fetch_enabled=True,
            fetch_allowlist=("docs.example",),
            modes=("accept_edits",),
        )
        equal.refuse(
            PermissionPolicy(
                repo_root,
                allowed_write_scope=[repo_root / "src"],
                shell_enabled=True,
                shell_allowlist=[("python3",)],
                fetch_enabled=True,
                fetch_allowlist=["DOCS.EXAMPLE."],
                mode="accept_edits",
            ),
            source=source,
        )


def _ceiling_toml() -> str:
    return (
        "[agents.ceiling]\n"
        'allowed_write_scope = ["src"]\n'
        "shell_enabled = true\n"
        'shell_allowlist = [["python3", "scripts/check.py"]]\n'
        "fetch_enabled = true\n"
        'fetch_allowlist = ["Docs.Example."]\n'
        'modes = ["auto", "plan"]\n'
    )


@check("config.empty_ceiling_refuses_nothing")
def empty_ceiling_refuses_nothing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo_root, home = _roots(temporary)
        policies = (
            PermissionPolicy(repo_root, allowed_write_scope=[repo_root.parent / "outside"]),
            PermissionPolicy(repo_root, shell_enabled=True, shell_allowlist=[("python3",)]),
            PermissionPolicy(repo_root, fetch_enabled=True, fetch_allowlist=["x.example"]),
            PermissionPolicy(repo_root, mode="auto"),
            PermissionPolicy(repo_root, mode="prompt"),
            PermissionPolicy(repo_root, mode="plan"),
            PermissionPolicy(repo_root, mode="accept_edits"),
            PermissionPolicy(
                repo_root,
                allowed_write_scope=[repo_root.parent / "outside"],
                shell_enabled=True,
                shell_allowlist=[("make",)],
                fetch_enabled=True,
                fetch_allowlist=["wide.example"],
                mode="accept_edits",
            ),
        )
        unconstrained = CapabilityCeiling()
        for index, policy in enumerate(policies):
            try:
                unconstrained.refuse(policy, source=Path(temporary) / f"{index}.toml")
            except ConfigError as exc:
                fail(f"all-None ceiling refused policy {index}: {exc!r}")
        empty = load_config(repo_root=repo_root, home=home)
        if CapabilityCeiling.from_config(empty, repo_root=repo_root) != unconstrained:
            fail("absent configuration did not create an all-None ceiling")

        expected = CapabilityCeiling(
            allowed_write_scope=((repo_root / "src").resolve(),),
            shell_enabled=True,
            shell_allowlist=(("python3", "scripts/check.py"),),
            fetch_enabled=True,
            fetch_allowlist=("docs.example",),
            modes=("auto", "plan"),
        )
        for scope in Scope:
            with tempfile.TemporaryDirectory() as scoped_temporary:
                scoped_root, scoped_home = _roots(scoped_temporary)
                if scope is Scope.SESSION:
                    session = {
                        "agents": {
                            "ceiling": {
                                "allowed_write_scope": ["src"],
                                "shell_enabled": True,
                                "shell_allowlist": [["python3", "scripts/check.py"]],
                                "fetch_enabled": True,
                                "fetch_allowlist": ["Docs.Example."],
                                "modes": ["auto", "plan"],
                            }
                        }
                    }
                else:
                    _write(_scope_path(scope, scoped_root, scoped_home), _ceiling_toml())
                    session = None
                actual = CapabilityCeiling.from_config(
                    load_config(repo_root=scoped_root, home=scoped_home, session=session),
                    repo_root=scoped_root,
                )
                scoped_expected = CapabilityCeiling(
                    allowed_write_scope=((scoped_root / "src").resolve(),),
                    shell_enabled=expected.shell_enabled,
                    shell_allowlist=expected.shell_allowlist,
                    fetch_enabled=expected.fetch_enabled,
                    fetch_allowlist=expected.fetch_allowlist,
                    modes=expected.modes,
                )
                if actual != scoped_expected:
                    fail(f"{scope.value} ceiling was not honoured: {actual!r}")


@check("config.ceiling_applies_to_agent_files")
def ceiling_applies_to_agent_files() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        repo_root = directory / "repo"
        path = directory / "worker.toml"
        _write(
            path,
            (
                'prompt = "Review."\n'
                '[policy]\nallowed_write_scope = ["../outside"]\n'
                "shell_enabled = true\n"
            ),
        )
        default_model = ModelSelector("fake")
        implicit = load_agent_file(path, repo_root=repo_root, default_model=default_model)
        explicit = load_agent_file(
            path,
            repo_root=repo_root,
            default_model=default_model,
            ceiling=None,
        )
        if implicit != explicit:
            fail("ceiling=None changed agent-file loading")
        ceiling = CapabilityCeiling(
            allowed_write_scope=((repo_root / "src").resolve(),),
            shell_enabled=False,
        )
        try:
            load_agent_file(
                path,
                repo_root=repo_root,
                default_model=default_model,
                ceiling=ceiling,
            )
        except ConfigError as exc:
            fail(f"agent file exposed ConfigError: {exc!r}")
        except AgentFileError as exc:
            if str(path) not in str(exc) or "allowed_write_scope" not in str(exc):
                fail(f"agent-file refusal omitted path or field: {exc!r}")
        else:
            fail("agent file accepted an out-of-root write scope under a ceiling")
        try:
            load_agent_directory(
                directory,
                repo_root=repo_root,
                default_model=default_model,
                ceiling=ceiling,
            )
        except AgentFileError:
            pass
        else:
            fail("agent directory did not forward its ceiling")


def _config_forbidden_imports(source: str) -> list[str]:
    found = _forbidden_imports(source)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = ([node.module] if node.module is not None else []) + [
                alias.name for alias in node.names
            ]
        else:
            continue
        for module in modules:
            if (
                any(part in _FORBIDDEN_CONFIG_IMPORTS for part in module.split("."))
                and module not in found
            ):
                found.append(module)
    return found


@check("config.no_runtime_imports")
def no_runtime_imports() -> None:
    source = (REPO_ROOT / "symphonai_api/config.py").read_text(encoding="utf-8")
    found = _config_forbidden_imports(source)
    if found:
        fail(f"config imports runtime wiring: {found!r}")
    probes = tuple(
        f"from symphonai_api.{module} import Probe\n"
        for module in sorted(_FORBIDDEN_CONFIG_IMPORTS - {"providers"})
    ) + ("from symphonai_api.providers.fake import FakeModelProvider\n",)
    for probe in probes:
        if not _config_forbidden_imports(probe):
            fail(f"import inspection missed {probe.strip()!r}")
    if _config_forbidden_imports(
        "from symphonai_api.permissions import PermissionPolicy\n"
    ):
        fail("import inspection rejected permissions")


@check("config.layers_are_per_scope")
def layers_are_per_scope() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo_root, home = _roots(temporary)
        user_path = _scope_path(Scope.USER, repo_root, home)
        project_path = _scope_path(Scope.PROJECT, repo_root, home)
        private_path = _scope_path(Scope.PRIVATE, repo_root, home)
        _write(
            user_path,
            '[agents]\ndirectory = "user"\n[hooks.nested]\nlevel = 1\n',
        )
        _write(project_path, '[skills]\ndirectory = "project-skills"\n')
        _write(private_path, "")
        resolved = load_config(
            repo_root=repo_root,
            home=home,
            session={"plugins": {"nested": {"enabled": True}}},
        )
        expected = (
            (
                Scope.USER,
                user_path,
                {
                    "agents.directory": "user",
                    "hooks.nested.level": 1,
                },
            ),
            (
                Scope.PROJECT,
                project_path,
                {"skills.directory": "project-skills"},
            ),
            (
                Scope.SESSION,
                None,
                {"plugins.nested.enabled": True},
            ),
        )
        actual = tuple(
            (layer.scope, layer.source, dict(layer.values))
            for layer in resolved.layers
        )
        if actual != expected:
            fail(f"per-scope layers differed: {actual!r}")
        empty = load_config(
            repo_root=repo_root / "empty-repo",
            home=home / "empty-home",
        )
        if empty.layers:
            fail(f"missing scopes produced layers: {empty.layers!r}")


@check("config.ceiling_meet_is_the_tighter_side")
def ceiling_meet_is_the_tighter_side() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo_root = Path(temporary).resolve()
        src = repo_root / "src"
        package = src / "package"
        docs = repo_root / "docs"
        tests = repo_root / "tests"
        samples = (
            CapabilityCeiling(allowed_write_scope=(src,)),
            CapabilityCeiling(shell_enabled=False),
            CapabilityCeiling(shell_allowlist=(("git", "status"),)),
            CapabilityCeiling(fetch_enabled=True),
            CapabilityCeiling(fetch_allowlist=("one.example", "two.example")),
            CapabilityCeiling(modes=("plan",)),
            CapabilityCeiling(
                allowed_write_scope=(docs,),
                shell_enabled=True,
                modes=("auto", "plan"),
            ),
        )
        unconstrained = CapabilityCeiling()
        for sample in samples:
            if unconstrained.meet(sample) != sample or sample.meet(unconstrained) != sample:
                fail(f"all-None ceiling was not meet identity: {sample!r}")

        cases = (
            (
                CapabilityCeiling(allowed_write_scope=(src, docs)),
                CapabilityCeiling(allowed_write_scope=(package, tests)),
                CapabilityCeiling(allowed_write_scope=(package,)),
            ),
            (
                CapabilityCeiling(shell_enabled=True),
                CapabilityCeiling(shell_enabled=False),
                CapabilityCeiling(shell_enabled=False),
            ),
            (
                CapabilityCeiling(
                    shell_allowlist=(("git",), ("python3",)),
                ),
                CapabilityCeiling(
                    shell_allowlist=(("git", "status"), ("make",)),
                ),
                CapabilityCeiling(shell_allowlist=(("git", "status"),)),
            ),
            (
                CapabilityCeiling(fetch_enabled=False),
                CapabilityCeiling(fetch_enabled=True),
                CapabilityCeiling(fetch_enabled=False),
            ),
            (
                CapabilityCeiling(
                    fetch_allowlist=(
                        "one.example",
                        "two.example",
                        "one.example",
                        "three.example",
                    ),
                ),
                CapabilityCeiling(
                    fetch_allowlist=("three.example", "one.example"),
                ),
                CapabilityCeiling(
                    fetch_allowlist=("one.example", "three.example"),
                ),
            ),
            (
                CapabilityCeiling(modes=("auto", "plan")),
                CapabilityCeiling(modes=("plan", "prompt")),
                CapabilityCeiling(modes=("plan",)),
            ),
        )
        for left, right, expected in cases:
            actual = left.meet(right)
            if actual != expected:
                fail(f"ceiling meet was not tighter: {left!r}, {right!r}, {actual!r}")


def _ceiling_values(*, tight: bool) -> dict[str, object]:
    if tight:
        return {
            "allowed_write_scope": ["src"],
            "shell_enabled": False,
            "fetch_enabled": False,
            "modes": ["plan"],
        }
    return {
        "allowed_write_scope": ["."],
        "shell_enabled": True,
        "fetch_enabled": True,
        "modes": ["auto", "prompt", "plan", "accept_edits"],
    }


def _ceiling_document(*, tight: bool) -> str:
    values = _ceiling_values(tight=tight)
    write_scope = values["allowed_write_scope"]
    modes = values["modes"]
    return (
        "[agents.ceiling]\n"
        f"allowed_write_scope = {write_scope!r}\n".replace("'", '"')
        + f"shell_enabled = {str(values['shell_enabled']).lower()}\n"
        + f"fetch_enabled = {str(values['fetch_enabled']).lower()}\n"
        + f"modes = {modes!r}\n".replace("'", '"')
    )


def _load_ceiling_pair(
    temporary: str,
    lower: Scope,
    higher: Scope,
    *,
    lower_is_tight: bool,
):
    repo_root, home = _roots(temporary)
    session = None
    for scope, tight in ((lower, lower_is_tight), (higher, not lower_is_tight)):
        if scope is Scope.SESSION:
            session = {"agents": {"ceiling": _ceiling_values(tight=tight)}}
        else:
            _write(
                _scope_path(scope, repo_root, home),
                _ceiling_document(tight=tight),
            )
    resolved = load_config(repo_root=repo_root, home=home, session=session)
    return repo_root, CapabilityCeiling.from_config(resolved, repo_root=repo_root)


@check("config.ceiling_cannot_be_raised_from_below")
def ceiling_cannot_be_raised_from_below() -> None:
    scopes = tuple(Scope)
    for lower_index, lower in enumerate(scopes):
        for higher in scopes[lower_index + 1 :]:
            for lower_is_tight in (True, False):
                with tempfile.TemporaryDirectory() as temporary:
                    repo_root, actual = _load_ceiling_pair(
                        temporary,
                        lower,
                        higher,
                        lower_is_tight=lower_is_tight,
                    )
                expected = CapabilityCeiling(
                    allowed_write_scope=((repo_root / "src").resolve(),),
                    shell_enabled=False,
                    fetch_enabled=False,
                    modes=("plan",),
                )
                if actual != expected:
                    direction = "lower tight" if lower_is_tight else "higher tight"
                    fail(
                        f"scope ceiling was raised for {lower.value}->{higher.value} "
                        f"({direction}): {actual!r}"
                    )

    with tempfile.TemporaryDirectory() as temporary:
        repo_root, home = _roots(temporary)
        empty = load_config(repo_root=repo_root, home=home)
        if CapabilityCeiling.from_config(empty, repo_root=repo_root) != CapabilityCeiling():
            fail("no ceiling layers did not return an unconstrained ceiling")
        for scope in Scope:
            with tempfile.TemporaryDirectory() as scoped_temporary:
                scoped_root, scoped_home = _roots(scoped_temporary)
                if scope is Scope.SESSION:
                    session = {"agents": {"ceiling": _ceiling_values(tight=True)}}
                else:
                    _write(
                        _scope_path(scope, scoped_root, scoped_home),
                        _ceiling_document(tight=True),
                    )
                    session = None
                actual = CapabilityCeiling.from_config(
                    load_config(
                        repo_root=scoped_root,
                        home=scoped_home,
                        session=session,
                    ),
                    repo_root=scoped_root,
                )
                expected = CapabilityCeiling(
                    allowed_write_scope=((scoped_root / "src").resolve(),),
                    shell_enabled=False,
                    fetch_enabled=False,
                    modes=("plan",),
                )
                if actual != expected:
                    fail(f"single {scope.value} ceiling differed: {actual!r}")


@check("config.shell_allowlist_uses_prefixes")
def shell_allowlist_uses_prefixes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo_root = Path(temporary)
        source = repo_root / "agent.toml"
        permitted = CapabilityCeiling(shell_allowlist=(("git",),))
        permitted.refuse(
            PermissionPolicy(repo_root, shell_allowlist=[("git", "status")]),
            source=source,
        )
        refused = (
            (
                CapabilityCeiling(shell_allowlist=(("git", "status"),)),
                PermissionPolicy(repo_root, shell_allowlist=[("git",)]),
            ),
            (
                CapabilityCeiling(shell_allowlist=(("git",),)),
                PermissionPolicy(repo_root, shell_allowlist=[("gitk",)]),
            ),
        )
        for ceiling, policy in refused:
            _expect_refusal("shell_allowlist", ceiling, policy, source)


@check("config.refusal_agrees_with_narrowing")
def refusal_agrees_with_narrowing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo_root = Path(temporary).resolve()
        write_pairs = (
            (repo_root / "src", repo_root / "src"),
            (repo_root, repo_root / "src"),
            (repo_root / "src", repo_root),
            (repo_root / "src", repo_root / "docs"),
            (repo_root / "src", repo_root / "src" / "package"),
            (repo_root / "src", repo_root / "srcextra"),
        )
        shell_pairs = (
            (("git", "status"), ("git", "status")),
            (("git",), ("git", "status")),
            (("git", "status"), ("git",)),
            (("git",), ("gitk",)),
            (("git", "status"), ("git", "status", "--short")),
            (("python3",), ("make",)),
        )
        cases: list[tuple[str, CapabilityCeiling, PermissionPolicy, PermissionPolicy]] = []
        for allowed, requested in write_pairs:
            ceiling = CapabilityCeiling(allowed_write_scope=(allowed,))
            policy = PermissionPolicy(repo_root, allowed_write_scope=[requested])
            ceiling_policy = PermissionPolicy(repo_root, allowed_write_scope=[allowed])
            cases.append(("allowed_write_scope", ceiling, policy, ceiling_policy))
        for allowed, requested in shell_pairs:
            ceiling = CapabilityCeiling(shell_allowlist=(allowed,))
            policy = PermissionPolicy(
                repo_root,
                shell_enabled=True,
                shell_allowlist=[requested],
            )
            ceiling_policy = PermissionPolicy(
                repo_root,
                shell_enabled=True,
                shell_allowlist=[allowed],
            )
            cases.append(("shell_allowlist", ceiling, policy, ceiling_policy))

        source = repo_root / "agent.toml"
        for field, ceiling, policy, ceiling_policy in cases:
            narrowed = policy.narrowed(ceiling_policy)
            before = getattr(policy, field)
            after = getattr(narrowed, field)
            expected_refusal = after != before
            try:
                ceiling.refuse(policy, source=source)
            except ConfigError:
                refused = True
            else:
                refused = False
            if refused != expected_refusal:
                fail(
                    f"refusal disagreed with narrowing for {field}: "
                    f"{before!r} -> {after!r}, refused={refused}"
                )


@check("config.sections_are_open")
def sections_are_open() -> None:
    sections = ("hooks", "skills", "mcp", "plugins")
    for scope in Scope:
        for section in sections:
            with tempfile.TemporaryDirectory() as temporary:
                repo_root, home = _roots(temporary)
                if scope is Scope.SESSION:
                    session = {section: {"nested": {"anything": [1, "two"]}}}
                    source = None
                else:
                    source = _scope_path(scope, repo_root, home)
                    _write(
                        source,
                        f'[{section}.nested]\nanything = [1, "two"]\n',
                    )
                    session = None
                resolved = load_config(repo_root=repo_root, home=home, session=session)
                key = f"{section}.nested.anything"
                if resolved.get(key) != [1, "two"]:
                    fail(f"{scope.value} {section} section was not flattened")
                if resolved.provenance.get(key) != Provenance(key, scope, source):
                    fail(f"{scope.value} {section} provenance differed")

    with tempfile.TemporaryDirectory() as temporary:
        repo_root, home = _roots(temporary)
        unknown_path = _scope_path(Scope.PROJECT, repo_root, home)
        _write(unknown_path, "[hook]\nenabled = true\n")
        try:
            load_config(repo_root=repo_root, home=home)
        except ConfigError as exc:
            if str(unknown_path) not in str(exc) or "hook" not in str(exc):
                fail(f"unknown top-level file key lacked path or key: {exc!r}")
        else:
            fail("unknown top-level file section was accepted")
        unknown_path.unlink()
        invalid_sessions = (
            ({"hook": {"enabled": True}}, "hook"),
            ({"agents": {"unknown": True}}, "agents.unknown"),
            (
                {"agents": {"ceiling": {"unknown": True}}},
                "agents.ceiling.unknown",
            ),
        )
        for session, key in invalid_sessions:
            try:
                load_config(repo_root=repo_root, home=home, session=session)
            except ConfigError as exc:
                if "<session>" not in str(exc) or key not in str(exc):
                    fail(f"unknown session key lacked source or key: {exc!r}")
            else:
                fail(f"unknown session key was accepted: {key}")
