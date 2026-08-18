"""Deny-by-default permission policy gating all filesystem and shell access.

Everything defaults closed: `read_file`/`list_files` are scoped to
`repo_root` minus a denylist of secret/build/cache/dependency patterns;
`write_file` additionally requires an explicit allowed write scope, empty
by default; `run_shell` is disabled by default and, even when enabled,
only runs commands matching an explicit allowlist -- and a hardcoded
always-deny set is enforced ahead of, and regardless of, that allowlist.

This module never calls `subprocess` itself; it only decides whether a
caller (see `orchestra_api.tools`) is allowed to.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_FORBIDDEN_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    ".git/",
    ".ssh/",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
    ".venv/",
    "__pycache__/",
    "*.egg-info/",
    "node_modules/",
    "dist/",
    "build/",
)

# Argv prefixes that are always denied, regardless of shell_enabled or
# shell_allowlist. Checked before, and independent of, the allowlist.
ALWAYS_DENY_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("rm",),
    ("mv",),
    ("chmod",),
    ("chown",),
    ("sudo",),
    ("curl",),
    ("wget",),
    ("ssh",),
    ("scp",),
    ("git", "push"),
    ("git", "merge"),
    ("git", "commit"),
    ("open",),
    ("osascript",),
)


@dataclass(frozen=True)
class PermissionDecision:
    """The outcome of a permission check: allowed, or denied with a reason."""

    allowed: bool
    reason: str = ""

    @classmethod
    def allow(cls) -> "PermissionDecision":
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str) -> "PermissionDecision":
        return cls(allowed=False, reason=reason)


@dataclass
class PermissionPolicy:
    """Deny-by-default policy for filesystem and shell access.

    `repo_root` is the only directory read/list access is ever scoped to.
    `allowed_write_scope` is a list of directories writes may target,
    empty by default so nothing is writable until explicitly configured.
    `shell_enabled` and `shell_allowlist` gate `run_shell`; a command must
    pass both, and `ALWAYS_DENY_COMMANDS` overrides either.
    """

    repo_root: Path
    allowed_write_scope: list[Path] = field(default_factory=list)
    forbidden_patterns: tuple[str, ...] = DEFAULT_FORBIDDEN_PATTERNS
    shell_enabled: bool = False
    shell_allowlist: list[tuple[str, ...]] = field(default_factory=list)
    shell_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self.allowed_write_scope = [Path(p).resolve() for p in self.allowed_write_scope]

    # -- path checks ------------------------------------------------------

    def _resolve_within_root(self, path: str | Path) -> Path | None:
        """Resolve `path` (relative paths are taken as relative to repo_root)
        and return it only if the resolved path is inside `repo_root`.

        Resolution happens before the containment check, so a `..`
        component or a symlink that points outside `repo_root` is caught
        here rather than trusted.
        """
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        resolved = candidate.resolve()
        if resolved == self.repo_root or self.repo_root in resolved.parents:
            return resolved
        return None

    def _matches_forbidden(self, resolved: Path) -> str | None:
        try:
            rel_parts = resolved.relative_to(self.repo_root).parts
        except ValueError:
            rel_parts = resolved.parts
        for pattern in self.forbidden_patterns:
            glob = pattern.rstrip("/")
            if any(fnmatch.fnmatch(part, glob) for part in rel_parts):
                return pattern
        return None

    def check_read(self, path: str | Path) -> PermissionDecision:
        resolved = self._resolve_within_root(path)
        if resolved is None:
            return PermissionDecision.deny(f"path escapes repo_root: {path!r}")
        forbidden = self._matches_forbidden(resolved)
        if forbidden is not None:
            return PermissionDecision.deny(
                f"path matches forbidden pattern {forbidden!r}: {path!r}"
            )
        return PermissionDecision.allow()

    # list_files follows exactly the same rule as read_file.
    check_list = check_read

    def check_write(self, path: str | Path) -> PermissionDecision:
        read_decision = self.check_read(path)
        if not read_decision.allowed:
            return read_decision
        resolved = self._resolve_within_root(path)
        assert resolved is not None  # check_read already validated this
        for allowed_root in self.allowed_write_scope:
            if resolved == allowed_root or allowed_root in resolved.parents:
                return PermissionDecision.allow()
        return PermissionDecision.deny(
            f"path is outside the explicit allowed write scope: {path!r}"
        )

    # -- shell checks -------------------------------------------------------

    def check_shell(self, argv: list[str]) -> PermissionDecision:
        if not argv:
            return PermissionDecision.deny("empty command")
        argv_tuple = tuple(argv)
        for denied_prefix in ALWAYS_DENY_COMMANDS:
            if argv_tuple[: len(denied_prefix)] == denied_prefix:
                return PermissionDecision.deny(
                    f"command matches always-deny rule: {' '.join(denied_prefix)!r}"
                )
        if not self.shell_enabled:
            return PermissionDecision.deny("run_shell is disabled by this policy")
        for allowed_prefix in self.shell_allowlist:
            if argv_tuple[: len(allowed_prefix)] == allowed_prefix:
                return PermissionDecision.allow()
        return PermissionDecision.deny(
            f"command does not match the shell allowlist: {list(argv)}"
        )
