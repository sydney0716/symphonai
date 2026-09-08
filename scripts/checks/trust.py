"""Checks for owner-controlled repository extension trust."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile

from symphonai_api.config import ConfigError, Scope, load_config
from symphonai_api.trust import (
    CAPABILITIES,
    RepositoryTrust,
    TrustList,
    trust_from_config,
)
from scripts.checks.agent_spec import _forbidden_imports
from scripts.checks.harness import check, fail


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _scope_path(scope: Scope, repo_root: Path, home: Path) -> Path:
    if scope is Scope.USER:
        return home / ".symphonai" / "config.toml"
    if scope is Scope.PROJECT:
        return repo_root / ".symphonai" / "config.toml"
    if scope is Scope.PRIVATE:
        return repo_root / ".symphonai" / "config.local.toml"
    raise ValueError("session scope has no path")


def _trust_table(root: Path, allow: tuple[str, ...] = ("hooks",)) -> str:
    encoded_allow = ", ".join(json.dumps(item) for item in allow)
    return (
        "[[trust.repositories]]\n"
        f"root = {json.dumps(str(root))}\n"
        f"allow = [{encoded_allow}]\n"
    )


def _load_scope(temporary: str, scope: Scope, content: str):
    base = Path(temporary)
    repo_root = base / "repo"
    home = base / "home"
    source = None
    session = None
    if scope is Scope.SESSION:
        root = base / "trusted"
        session = {
            "trust": {
                "repositories": [
                    {"root": str(root), "allow": ["hooks", "mcp"]}
                ]
            }
        }
    else:
        source = _scope_path(scope, repo_root, home)
        _write(source, content)
    return load_config(repo_root=repo_root, home=home, session=session), source


def _expect_config_error(config, fragments: tuple[str, ...]) -> str:  # noqa: ANN001
    try:
        trust_from_config(config)
    except ConfigError as exc:
        message = str(exc)
        if not all(fragment in message for fragment in fragments):
            fail(f"trust error omitted {fragments!r}: {message!r}")
        return message
    fail(f"trust config was accepted despite expected fragments {fragments!r}")


@check("trust.owner_only")
def owner_only() -> None:
    for scope in Scope:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            trusted_root = base / "trusted"
            content = _trust_table(trusted_root, ("hooks", "mcp"))
            config, source = _load_scope(temporary, scope, content)
            if scope in (Scope.USER, Scope.SESSION):
                trust = trust_from_config(config)
                expected_root = (
                    trusted_root
                    if scope is not Scope.SESSION
                    else base / "trusted"
                ).resolve()
                if trust != TrustList(
                    (
                        RepositoryTrust(
                            expected_root,
                            frozenset(("hooks", "mcp")),
                            source,
                        ),
                    )
                ):
                    fail(f"owner trust parsed incorrectly for {scope.value}: {trust!r}")
            else:
                _expect_config_error(
                    config,
                    (str(source), "may not grant itself trust"),
                )

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repo_root = base / "repo"
        home = base / "home"
        trusted_root = base / "trusted"
        _write(
            _scope_path(Scope.USER, repo_root, home),
            _trust_table(trusted_root, ("hooks",)),
        )
        project = _scope_path(Scope.PROJECT, repo_root, home)
        _write(project, _trust_table(trusted_root, ("mcp",)))
        config = load_config(repo_root=repo_root, home=home)
        if config.scope_of("trust.repositories") is not Scope.PROJECT:
            fail("project trust list did not win as one leaf")
        _expect_config_error(config, (str(project), "may not grant itself trust"))


@check("trust.vocabulary_and_shape")
def vocabulary_and_shape() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repo_root = base / "repo"
        home = base / "home"
        source = _scope_path(Scope.USER, repo_root, home)
        trusted_root = base / "trusted"

        _write(source, _trust_table(trusted_root, ("hook",)))
        message = _expect_config_error(
            load_config(repo_root=repo_root, home=home),
            (str(source), "[0].allow", "'hook'"),
        )
        if not all(capability in message for capability in CAPABILITIES):
            fail(f"unknown capability error omitted valid names: {message!r}")

        _write(source, _trust_table(trusted_root, ()))
        empty_grant = trust_from_config(load_config(repo_root=repo_root, home=home))
        if len(empty_grant.entries) != 1 or empty_grant.entries[0].allow:
            fail(f"empty capability list did not parse exactly: {empty_grant!r}")
        if any(empty_grant.allows(trusted_root, name) for name in CAPABILITIES):
            fail("allow=[] granted a capability")

        duplicate = _trust_table(trusted_root) + _trust_table(trusted_root / ".")
        _write(source, duplicate)
        _expect_config_error(
            load_config(repo_root=repo_root, home=home),
            (str(source), "duplicate root", "indices 0 and 1"),
        )

        for label, root_line in (("missing", ""), ("blank", 'root = "  "\n')):
            _write(
                source,
                "[[trust.repositories]]\n"
                f"{root_line}"
                'allow = ["hooks"]\n',
            )
            _expect_config_error(
                load_config(repo_root=repo_root, home=home),
                (str(source), "[0].root", "non-empty string"),
            )

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        absent = trust_from_config(
            load_config(repo_root=base / "repo", home=base / "home")
        )
        if absent != TrustList():
            fail(f"absent trust did not produce TrustList(): {absent!r}")
        probes = (*CAPABILITIES, "unknown")
        if any(absent.allows(base / "any", capability) for capability in probes):
            fail("empty TrustList allowed a root or capability")

        bad = base / "repo" / ".symphonai" / "config.toml"
        _write(bad, "[outside]\nvalue = true\n")
        try:
            load_config(repo_root=base / "repo", home=base / "home")
        except ConfigError as exc:
            if str(bad) not in str(exc) or "outside" not in str(exc):
                fail(f"outside-section error omitted source or key: {exc!r}")
        else:
            fail("widening _SECTIONS accepted an unrelated top-level key")


@check("trust.matching_is_exact")
def matching_is_exact() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        granted = parent / "repo"
        child = granted / "child"
        sibling = parent / "repoextra"
        alias = parent / "alias"
        child.mkdir(parents=True)
        sibling.mkdir()
        alias.symlink_to(granted, target_is_directory=True)
        trust = TrustList(
            (
                RepositoryTrust(
                    granted.resolve(),
                    frozenset(("hooks",)),
                    None,
                ),
            )
        )
        cases = (
            (granted, "hooks", True),
            (alias, "hooks", True),
            (parent, "hooks", False),
            (sibling, "hooks", False),
            (child, "hooks", False),
            (granted, "not-a-capability", False),
        )
        for root, capability, expected in cases:
            actual = trust.allows(root, capability)
            if actual is not expected:
                fail(
                    f"exact trust differed for {root!s}/{capability}: "
                    f"actual={actual}, expected={expected}"
                )


@check("trust.no_runtime_imports")
def no_runtime_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "symphonai_api/trust.py"
    source = path.read_text(encoding="utf-8")
    forbidden = _forbidden_imports(source)
    expanded = {
        "agent_loop",
        "leader",
        "runner",
        "agent_run",
        "agent_spec",
        "agent_file",
        "child_context",
        "hooks",
        "mcp",
        "skills",
        "permissions",
        "provider_catalog",
        "providers",
    }
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module] if node.module is not None else []
        else:
            continue
        forbidden.extend(
            module
            for module in modules
            if set(module.split(".")) & expanded
        )
    if forbidden:
        fail(f"trust.py imports forbidden runtime modules: {sorted(set(forbidden))!r}")

    for consumer in ("hooks.py", "mcp.py"):
        consumer_source = (root / "symphonai_api" / consumer).read_text(
            encoding="utf-8"
        )
        consumer_forbidden = _forbidden_imports(consumer_source)
        if consumer_forbidden:
            fail(
                f"{consumer} imports forbidden runtime modules: "
                f"{sorted(set(consumer_forbidden))!r}"
            )
        consumer_tree = ast.parse(consumer_source)
        trust_imports = [
            node
            for node in ast.walk(consumer_tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "symphonai_api.trust"
        ]
        if len(trust_imports) != 1:
            fail(f"{consumer} did not gain exactly one trust import")
