"""Checks for trust-gated cross-scope extension discovery."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from unittest import mock

import symphonai_api.discovery as discovery_module
from symphonai_api.agent_file import AgentFileError
from symphonai_api.config import CapabilityCeiling, Scope
from symphonai_api.discovery import DiscoveryError, Offered, discover
from symphonai_api.plugins import PluginError
from symphonai_api.skills import SkillError
from symphonai_api.trust import RepositoryTrust, TrustList
from scripts.checks.agent_spec import _forbidden_imports
from scripts.checks.harness import check, fail


KINDS = ("agents", "skills", "plugins")


def _write_member(
    base: Path,
    kind: str,
    name: str,
    *,
    malformed: bool = False,
    shell_enabled: bool = False,
) -> None:
    directory = base / kind
    directory.mkdir(parents=True, exist_ok=True)
    if kind == "agents":
        body = "" if malformed else (
            f'prompt = "{name}"\n'
            '[model]\nprovider = "fake"\n'
            + ('[policy]\nshell_enabled = true\n' if shell_enabled else "")
        )
        (directory / f"{name}.toml").write_text(body, encoding="utf-8")
    elif kind == "skills":
        body = "bad" if malformed else (
            "+++\n"
            f'name = "{name}"\n'
            f'description = "{name} description"\n'
            f'when_to_use = "use {name}"\n'
            "+++\nbody\n"
        )
        (directory / f"{name}.md").write_text(body, encoding="utf-8")
    else:
        plugin = directory / name
        plugin.mkdir()
        body = "" if malformed else (
            f'name = "{name}"\nversion = "1"\ndescription = "{name} plugin"\n'
        )
        (plugin / "plugin.toml").write_text(body, encoding="utf-8")


def _trust(root: Path, *capabilities: str) -> TrustList:
    return TrustList(
        (
            RepositoryTrust(
                root.resolve(),
                frozenset(capabilities),
                None,
            ),
        )
    )


def _mapping(discovered, kind: str):  # noqa: ANN001, ANN202
    return getattr(discovered, kind)


@check("discovery.user_missing_and_safe_default")
def user_missing_and_safe_default() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "repo"
        home = base / "home"
        root.mkdir()
        user_root = home / ".symphonai"
        project_root = root / ".symphonai"
        for kind in KINDS:
            _write_member(user_root, kind, f"user_{kind}")
            _write_member(project_root, kind, f"repo_{kind}")
        loaded = discover(repo_root=root, home=home, trust=None)
        for kind in KINDS:
            if tuple(_mapping(loaded, kind)) != (f"user_{kind}",):
                fail(f"trust=None did not load only user {kind}: {loaded!r}")
        expected_withheld = tuple(
            Offered(
                Scope.PROJECT,
                project_root / kind,
                (f"repo_{kind}",),
            )
            for kind in KINDS
        )
        if loaded.withheld != expected_withheld:
            fail(f"trust=None did not report every repository offer: {loaded!r}")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        home = Path(temporary) / "home"
        root.mkdir()
        missing = discover(repo_root=root, home=home)
        if any((missing.agents, missing.skills, missing.plugins, missing.withheld)):
            fail(f"missing scope directories contributed members: {missing!r}")


@check("discovery.trust_matrix")
def trust_matrix() -> None:
    different = {"agents": "skills", "skills": "plugins", "plugins": "agents"}
    for kind in KINDS:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "repo"
            home = base / "home"
            root.mkdir()
            _write_member(root / ".symphonai", kind, "repo_member")
            cases = (
                ("granted", _trust(root, kind), True),
                ("different capability", _trust(root, different[kind]), False),
                ("different root", _trust(base / "other", kind), False),
                ("no list", None, False),
            )
            for label, trust, accepted in cases:
                loaded = discover(repo_root=root, home=home, trust=trust)
                members = _mapping(loaded, kind)
                if ("repo_member" in members) is not accepted:
                    fail(f"{kind} {label} trust result was wrong: {loaded!r}")
                offered = [item for item in loaded.withheld if item.directory.name == kind]
                if accepted and offered:
                    fail(f"trusted {kind} was still reported withheld: {offered!r}")
                if not accepted and (
                    offered
                    != [
                        Offered(
                            Scope.PROJECT,
                            root / ".symphonai" / kind,
                            ("repo_member",),
                        )
                    ]
                ):
                    fail(f"untrusted {kind} offer was not exact: {offered!r}")


@check("discovery.withheld_without_loading")
def withheld_without_loading() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        home = Path(temporary) / "home"
        root.mkdir()
        project = root / ".symphonai"
        for kind in KINDS:
            _write_member(project, kind, f"listed_{kind}", malformed=True)

        def parsed_untrusted(directory, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            if Path(directory).is_relative_to(project):
                raise AssertionError("untrusted member content was parsed")
            return {}

        with (
            mock.patch.object(
                discovery_module,
                "load_agent_directory",
                side_effect=parsed_untrusted,
            ),
            mock.patch.object(
                discovery_module,
                "load_skill_directory",
                side_effect=parsed_untrusted,
            ),
            mock.patch.object(
                discovery_module,
                "load_plugin_directory",
                side_effect=parsed_untrusted,
            ),
        ):
            loaded = discover(repo_root=root, home=home, trust=None)
        if any((loaded.agents, loaded.skills, loaded.plugins)):
            fail(f"untrusted content entered a discovered mapping: {loaded!r}")
        expected = tuple(
            Offered(
                Scope.PROJECT,
                project / kind,
                (f"listed_{kind}",),
            )
            for kind in KINDS
        )
        if loaded.withheld != expected:
            fail(f"withheld filename listing was incomplete: {loaded.withheld!r}")


@check("discovery.cross_scope_collisions")
def cross_scope_collisions() -> None:
    for kind in KINDS:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "repo"
            home = base / "home"
            root.mkdir()
            user_directory = home / ".symphonai" / kind
            project_directory = root / ".symphonai" / kind
            _write_member(home / ".symphonai", kind, "deploy")
            _write_member(root / ".symphonai", kind, "deploy")
            try:
                discover(repo_root=root, home=home, trust=_trust(root, kind))
            except DiscoveryError as exc:
                message = str(exc)
                required = (
                    "deploy",
                    "user",
                    "project",
                    str(user_directory),
                    str(project_directory),
                )
                if not all(fragment in message for fragment in required):
                    fail(f"{kind} collision omitted context: {message!r}")
            else:
                fail(f"trusted repository {kind} shadowed the user member")

            untrusted = discover(repo_root=root, home=home, trust=None)
            if tuple(_mapping(untrusted, kind)) != ("deploy",):
                fail(f"untrusted {kind} caused a collision or won: {untrusted!r}")


@check("discovery.member_failures_are_atomic")
def member_failures_are_atomic() -> None:
    expected_errors = {
        "agents": AgentFileError,
        "skills": SkillError,
        "plugins": PluginError,
    }
    for kind in KINDS:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            home = Path(temporary) / "home"
            root.mkdir()
            directory = home / ".symphonai" / kind
            _write_member(home / ".symphonai", kind, "broken", malformed=True)
            try:
                discover(repo_root=root, home=home)
            except DiscoveryError as exc:
                cause = exc.__cause__
                expected_type = expected_errors[kind]
                if (
                    not isinstance(cause, expected_type)
                    or expected_type.__name__ not in str(exc)
                    or str(directory) not in str(exc)
                    or "user" not in str(exc)
                    or str(cause) not in str(exc)
                ):
                    fail(f"{kind} loader failure was not preserved: {exc!r}")
            else:
                fail(f"malformed user {kind} member was skipped")


@check("discovery.ceiling_forwarding")
def ceiling_forwarding() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        home = Path(temporary) / "home"
        root.mkdir()
        _write_member(
            root / ".symphonai",
            "agents",
            "worker",
            shell_enabled=True,
        )
        try:
            discover(
                repo_root=root,
                home=home,
                trust=_trust(root, "agents"),
                ceiling=CapabilityCeiling(shell_enabled=False),
            )
        except DiscoveryError as exc:
            if not isinstance(exc.__cause__, AgentFileError) or "shell_enabled" not in str(exc):
                fail(f"ceiling refusal was not preserved: {exc!r}")
        else:
            fail("discovery did not forward the constraining agent ceiling")
        accepted = discover(
            repo_root=root,
            home=home,
            trust=_trust(root, "agents"),
            ceiling=CapabilityCeiling(),
        )
        if tuple(accepted.agents) != ("worker",):
            fail(f"unconstrained ceiling rejected the agent: {accepted!r}")


@check("discovery.determinism_and_imports")
def determinism_and_imports() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        home = Path(temporary) / "home"
        root.mkdir()
        for kind in KINDS:
            for name in ("user_b", "user_a"):
                _write_member(home / ".symphonai", kind, name)
            for name in ("repo_b", "repo_a"):
                _write_member(root / ".symphonai", kind, name)
        trust = _trust(root, *KINDS)
        first = discover(repo_root=root, home=home, trust=trust)
        second = discover(repo_root=root, home=home, trust=trust)
        expected = ("user_a", "user_b", "repo_a", "repo_b")
        for kind in KINDS:
            first_names = tuple(_mapping(first, kind))
            second_names = tuple(_mapping(second, kind))
            if first_names != expected or second_names != expected:
                fail(
                    f"{kind} discovery order was not deterministic: "
                    f"{first_names!r}, {second_names!r}"
                )

    source = Path(discovery_module.__file__ or "").read_text(encoding="utf-8")
    shared_forbidden = _forbidden_imports(source)
    forbidden = {
        "agent_loop",
        "leader",
        "runner",
        "agent_run",
        "child_context",
        "extensions",
        "provider_catalog",
        "providers",
    }
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module] if node.module is not None else []
        else:
            continue
        imported.extend(
            module
            for module in modules
            if any(part in forbidden for part in module.split("."))
        )
    if shared_forbidden or imported:
        fail(
            "discovery imports runtime orchestration: "
            f"{shared_forbidden + imported!r}"
        )
