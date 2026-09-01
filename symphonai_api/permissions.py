"""Deny-by-default permission policy gating all filesystem and shell access.

Everything defaults closed: `read_file`/`list_files` are scoped to
`repo_root` minus a denylist of secret/build/cache/dependency patterns;
`write_file` additionally requires an explicit allowed write scope, empty
by default; `run_shell` is disabled by default and, even when enabled,
only runs commands matching an explicit allowlist -- and a hardcoded
always-deny set is enforced ahead of, and regardless of, that allowlist.

This module never calls `subprocess` itself; it only decides whether a
caller (see `symphonai_api.tools`) is allowed to.
"""

from __future__ import annotations

import fnmatch
import ipaddress
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlsplit

from symphonai_api.web_domains import preapproved_domains

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
DEFAULT_SHELL_OUTPUT_CHARS = 20_000
MIN_SHELL_OUTPUT_CHARS = 1_000
MAX_SHELL_OUTPUT_CHARS = 200_000

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


class DenialReason(str, Enum):
    """Why a check said no, as a value a caller can branch on.

    `PermissionDecision.reason` stays the human- and model-readable sentence.
    This names the kind so consumers never have to parse that prose.
    """

    OUTSIDE_ROOT = "outside_root"
    FORBIDDEN_PATTERN = "forbidden_pattern"
    OUTSIDE_WRITE_SCOPE = "outside_write_scope"
    EMPTY_COMMAND = "empty_command"
    ALWAYS_DENY = "always_deny"
    SHELL_DISABLED = "shell_disabled"
    NOT_ALLOWLISTED = "not_allowlisted"
    NO_APPROVAL_CALLBACK = "no_approval_callback"
    APPROVAL_FAILED = "approval_failed"
    DENIED_BY_USER = "denied_by_user"
    INVALID_APPROVAL = "invalid_approval"
    PLAN_MODE = "plan_mode"
    UNSUPPORTED_SCHEME = "unsupported_scheme"
    BLOCKED_HOST = "blocked_host"
    DOMAIN_NOT_APPROVED = "domain_not_approved"


@dataclass(frozen=True)
class PermissionDecision:
    """The outcome of a permission check: allowed, or denied with a reason."""

    allowed: bool
    reason: str = ""
    denial: DenialReason | None = None

    @classmethod
    def allow(cls) -> "PermissionDecision":
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str, *, denial: DenialReason) -> "PermissionDecision":
        return cls(allowed=False, reason=reason, denial=denial)


PermissionMode = Literal["auto", "prompt", "plan", "accept_edits"]


@dataclass(frozen=True)
class ToolApprovalRequest:
    """A side-effectful tool action that needs an interactive decision."""

    operation: str
    target: str
    details: str = ""


ApprovalCallback = Callable[[ToolApprovalRequest], PermissionDecision | bool]


@dataclass
class PermissionPolicy:
    """Deny-by-default policy for filesystem and shell access.

    `repo_root` is the only directory read/list access is ever scoped to.
    `allowed_write_scope` is a list of directories writes may target,
    empty by default so nothing is writable until explicitly configured.
    `shell_enabled` and `shell_allowlist` gate `run_shell`; a command must
    pass both, and `ALWAYS_DENY_COMMANDS` overrides either.

    `mode="auto"` uses only the static rules. `prompt` asks before writes or
    shell calls, `plan` permits reads only, and `accept_edits` permits scoped
    writes without asking while still prompting for shell calls.
    """

    repo_root: Path
    allowed_write_scope: list[Path] = field(default_factory=list)
    forbidden_patterns: tuple[str, ...] = DEFAULT_FORBIDDEN_PATTERNS
    shell_enabled: bool = False
    shell_allowlist: list[tuple[str, ...]] = field(default_factory=list)
    fetch_enabled: bool = False
    fetch_allowlist: list[str] = field(default_factory=list)
    shell_timeout_seconds: float = 10.0
    shell_output_limit_chars: int = DEFAULT_SHELL_OUTPUT_CHARS
    mode: PermissionMode = "auto"
    approval_callback: ApprovalCallback | None = None

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self.allowed_write_scope = [Path(p).resolve() for p in self.allowed_write_scope]
        self.fetch_allowlist = [
            host.casefold().rstrip(".") for host in self.fetch_allowlist
        ]
        self.shell_output_limit_chars = max(
            MIN_SHELL_OUTPUT_CHARS,
            min(MAX_SHELL_OUTPUT_CHARS, int(self.shell_output_limit_chars)),
        )
        if self.mode not in ("auto", "prompt", "plan", "accept_edits"):
            raise ValueError(
                f"unknown permission mode {self.mode!r}; expected "
                "'auto', 'prompt', 'plan', or 'accept_edits'"
            )
        self._approval_lock = threading.Lock()

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
            return PermissionDecision.deny(
                f"path escapes repo_root: {path!r}",
                denial=DenialReason.OUTSIDE_ROOT,
            )
        forbidden = self._matches_forbidden(resolved)
        if forbidden is not None:
            return PermissionDecision.deny(
                f"path matches forbidden pattern {forbidden!r}: {path!r}",
                denial=DenialReason.FORBIDDEN_PATTERN,
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
        if self.mode == "plan":
            return PermissionDecision.deny(
                "plan mode allows reads only; this call would change the world",
                denial=DenialReason.PLAN_MODE,
            )
        if self.mode == "prompt":
            return self._ask_approval(
                operation="write_file",
                target=str(path),
                details=f"write inside repo root: {resolved}",
            )
        for allowed_root in self.allowed_write_scope:
            if resolved == allowed_root or allowed_root in resolved.parents:
                return PermissionDecision.allow()
        return PermissionDecision.deny(
            f"path is outside the explicit allowed write scope: {path!r}",
            denial=DenialReason.OUTSIDE_WRITE_SCOPE,
        )

    # -- fetch checks -------------------------------------------------------

    def check_fetch(self, url: str) -> PermissionDecision:
        try:
            parsed = urlsplit(url)
            scheme = parsed.scheme.casefold()
        except (TypeError, ValueError):
            scheme = ""
            parsed = None
        if scheme not in ("http", "https"):
            return PermissionDecision.deny(
                "web_fetch supports only http and https URLs",
                denial=DenialReason.UNSUPPORTED_SCHEME,
            )

        try:
            host = (parsed.hostname or "").casefold().rstrip(".")
        except ValueError:
            host = ""
        blocked = not host or host == "localhost" or host.endswith(
            (".localhost", ".local")
        )
        if not blocked:
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                address = None
            if address is not None:
                blocked = any(
                    (
                        address.is_loopback,
                        address.is_private,
                        address.is_link_local,
                        address.is_reserved,
                        address.is_unspecified,
                        address.is_multicast,
                    )
                )
        if blocked:
            return PermissionDecision.deny(
                f"web_fetch blocks host {host or '[missing]'}",
                denial=DenialReason.BLOCKED_HOST,
            )

        if host in preapproved_domains() or host in self.fetch_allowlist:
            return PermissionDecision.allow()
        if self.mode in ("prompt", "accept_edits"):
            return self._ask_approval(
                operation="web_fetch",
                target=url,
                details=f"HTTP GET from {host}",
            )
        if self.fetch_enabled:
            return PermissionDecision.allow()
        return PermissionDecision.deny(
            f"domain is not approved for web_fetch: {host}",
            denial=DenialReason.DOMAIN_NOT_APPROVED,
        )

    # -- shell checks -------------------------------------------------------

    def check_shell(self, argv: list[str]) -> PermissionDecision:
        if not argv:
            return PermissionDecision.deny(
                "empty command", denial=DenialReason.EMPTY_COMMAND
            )
        argv_tuple = tuple(argv)
        for denied_prefix in ALWAYS_DENY_COMMANDS:
            if argv_tuple[: len(denied_prefix)] == denied_prefix:
                return PermissionDecision.deny(
                    f"command matches always-deny rule: {' '.join(denied_prefix)!r}",
                    denial=DenialReason.ALWAYS_DENY,
                )
        if self.mode == "plan":
            return PermissionDecision.deny(
                "plan mode allows reads only; this call would change the world",
                denial=DenialReason.PLAN_MODE,
            )
        if self.mode in ("prompt", "accept_edits"):
            return self._ask_approval(
                operation="run_shell",
                target=" ".join(argv),
                details=f"run in repo root: {self.repo_root}",
            )
        if not self.shell_enabled:
            return PermissionDecision.deny(
                "run_shell is disabled by this policy",
                denial=DenialReason.SHELL_DISABLED,
            )
        for allowed_prefix in self.shell_allowlist:
            if argv_tuple[: len(allowed_prefix)] == allowed_prefix:
                return PermissionDecision.allow()
        return PermissionDecision.deny(
            f"command does not match the shell allowlist: {list(argv)}",
            denial=DenialReason.NOT_ALLOWLISTED,
        )

    def _ask_approval(
        self,
        *,
        operation: str,
        target: str,
        details: str = "",
    ) -> PermissionDecision:
        with self._approval_lock:
            if self.approval_callback is None:
                return PermissionDecision.deny(
                    f"{operation} requires approval, but no approval callback is configured",
                    denial=DenialReason.NO_APPROVAL_CALLBACK,
                )
            try:
                decision = self.approval_callback(
                    ToolApprovalRequest(
                        operation=operation, target=target, details=details
                    )
                )
            except Exception as exc:  # noqa: BLE001
                return PermissionDecision.deny(
                    f"approval callback failed ({type(exc).__name__}): {exc}",
                    denial=DenialReason.APPROVAL_FAILED,
                )
            if isinstance(decision, PermissionDecision):
                return decision
            if decision is True:
                return PermissionDecision.allow()
            if decision is False:
                return PermissionDecision.deny(
                    f"{operation} denied by user",
                    denial=DenialReason.DENIED_BY_USER,
                )
            return PermissionDecision.deny(
                f"approval callback returned an invalid decision for {operation}",
                denial=DenialReason.INVALID_APPROVAL,
            )
