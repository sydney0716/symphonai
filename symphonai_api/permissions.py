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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlsplit

from symphonai_api.events import (
    EventSink,
    PermissionDenied,
    PermissionRequested,
    emit,
)
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
        self._event_sink: EventSink | None = None
        self._event_agent_id = ""
        self._event_run_id = ""
        self._event_local = threading.local()

    def attach_event_sink(
        self,
        sink: EventSink | None,
        *,
        agent_id: str,
        run_id: str,
    ) -> None:
        """Attach observation identity for direct permission checks."""
        self._event_sink = sink
        self._event_agent_id = agent_id
        self._event_run_id = run_id

    @contextmanager
    def event_context(
        self,
        sink: EventSink | None,
        *,
        agent_id: str,
        run_id: str,
        turn_id: str | None,
        tool_name: str,
        tool_call_id: str,
    ) -> Iterator[None]:
        """Supply one thread's tool identity while a permission check runs."""
        previous = getattr(self._event_local, "context", None)
        self._event_local.context = (
            sink,
            agent_id,
            run_id,
            turn_id,
            tool_name,
            tool_call_id,
        )
        try:
            yield
        finally:
            if previous is None:
                del self._event_local.context
            else:
                self._event_local.context = previous

    def _event_identity(
        self,
        default_tool_name: str,
    ) -> tuple[EventSink | None, str, str, str | None, str, str]:
        context = getattr(self._event_local, "context", None)
        if context is not None:
            return context
        return (
            self._event_sink,
            self._event_agent_id,
            self._event_run_id,
            None,
            default_tool_name,
            "",
        )

    def _deny(
        self,
        reason: str,
        *,
        denial: DenialReason,
        tool_name: str,
    ) -> PermissionDecision:
        decision = PermissionDecision.deny(reason, denial=denial)
        sink, agent_id, run_id, turn_id, actual_tool_name, tool_call_id = (
            self._event_identity(tool_name)
        )
        emit(
            sink,
            PermissionDenied(
                agent_id=agent_id,
                run_id=run_id,
                turn_id=turn_id,
                tool_name=actual_tool_name,
                tool_call_id=tool_call_id,
                reason=decision.reason,
            ),
        )
        return decision

    def _permission_requested(self, tool_name: str) -> None:
        sink, agent_id, run_id, turn_id, actual_tool_name, tool_call_id = (
            self._event_identity(tool_name)
        )
        emit(
            sink,
            PermissionRequested(
                agent_id=agent_id,
                run_id=run_id,
                turn_id=turn_id,
                tool_name=actual_tool_name,
                tool_call_id=tool_call_id,
                mode=self.mode,
            ),
        )

    def narrowed(self, ceiling: "PermissionPolicy") -> "PermissionPolicy":
        """Return a new policy allowing only what both policies allow."""
        if not _contains_path(self.repo_root, ceiling.repo_root):
            raise ValueError(
                "cannot narrow repo_root "
                f"{self.repo_root!s} with outside root {ceiling.repo_root!s}"
            )
        for component in ceiling.repo_root.relative_to(self.repo_root).parts:
            forbidden = _matching_forbidden_component(component, self.forbidden_patterns)
            if forbidden is not None:
                raise ValueError(
                    f"cannot narrow to forbidden root {ceiling.repo_root!s}: {forbidden!r}"
                )
        if ceiling.mode != self.mode and ceiling.mode != "plan":
            raise ValueError(
                "cannot narrow permission mode "
                f"{self.mode!r} with {ceiling.mode!r}"
            )
        return PermissionPolicy(
            repo_root=ceiling.repo_root,
            allowed_write_scope=_intersect_write_scopes(
                self.allowed_write_scope, ceiling.allowed_write_scope
            ),
            forbidden_patterns=_ordered_union(
                self.forbidden_patterns, ceiling.forbidden_patterns
            ),
            shell_enabled=self.shell_enabled and ceiling.shell_enabled,
            shell_allowlist=_intersect_shell_allowlists(
                self.shell_allowlist, ceiling.shell_allowlist
            ),
            fetch_enabled=self.fetch_enabled and ceiling.fetch_enabled,
            fetch_allowlist=[
                host for host in self.fetch_allowlist if host in ceiling.fetch_allowlist
            ],
            shell_timeout_seconds=min(
                self.shell_timeout_seconds, ceiling.shell_timeout_seconds
            ),
            shell_output_limit_chars=min(
                self.shell_output_limit_chars, ceiling.shell_output_limit_chars
            ),
            mode=ceiling.mode,
            approval_callback=(
                ceiling.approval_callback
                if ceiling.approval_callback is not None
                else self.approval_callback
            ),
        )

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
            if any(_matching_forbidden_component(part, (pattern,)) for part in rel_parts):
                return pattern
        return None

    def check_read(self, path: str | Path) -> PermissionDecision:
        resolved = self._resolve_within_root(path)
        if resolved is None:
            return self._deny(
                f"path escapes repo_root: {path!r}",
                denial=DenialReason.OUTSIDE_ROOT,
                tool_name="read_file",
            )
        forbidden = self._matches_forbidden(resolved)
        if forbidden is not None:
            return self._deny(
                f"path matches forbidden pattern {forbidden!r}: {path!r}",
                denial=DenialReason.FORBIDDEN_PATTERN,
                tool_name="read_file",
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
            return self._deny(
                "plan mode allows reads only; this call would change the world",
                denial=DenialReason.PLAN_MODE,
                tool_name="write_file",
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
        return self._deny(
            f"path is outside the explicit allowed write scope: {path!r}",
            denial=DenialReason.OUTSIDE_WRITE_SCOPE,
            tool_name="write_file",
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
            return self._deny(
                "web_fetch supports only http and https URLs",
                denial=DenialReason.UNSUPPORTED_SCHEME,
                tool_name="web_fetch",
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
            return self._deny(
                f"web_fetch blocks host {host or '[missing]'}",
                denial=DenialReason.BLOCKED_HOST,
                tool_name="web_fetch",
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
        return self._deny(
            f"domain is not approved for web_fetch: {host}",
            denial=DenialReason.DOMAIN_NOT_APPROVED,
            tool_name="web_fetch",
        )

    # -- shell checks -------------------------------------------------------

    def check_shell(self, argv: list[str]) -> PermissionDecision:
        if not argv:
            return self._deny(
                "empty command",
                denial=DenialReason.EMPTY_COMMAND,
                tool_name="run_shell",
            )
        argv_tuple = tuple(argv)
        for denied_prefix in ALWAYS_DENY_COMMANDS:
            if argv_tuple[: len(denied_prefix)] == denied_prefix:
                return self._deny(
                    f"command matches always-deny rule: {' '.join(denied_prefix)!r}",
                    denial=DenialReason.ALWAYS_DENY,
                    tool_name="run_shell",
                )
        if self.mode == "plan":
            return self._deny(
                "plan mode allows reads only; this call would change the world",
                denial=DenialReason.PLAN_MODE,
                tool_name="run_shell",
            )
        if self.mode in ("prompt", "accept_edits"):
            return self._ask_approval(
                operation="run_shell",
                target=" ".join(argv),
                details=f"run in repo root: {self.repo_root}",
            )
        if not self.shell_enabled:
            return self._deny(
                "run_shell is disabled by this policy",
                denial=DenialReason.SHELL_DISABLED,
                tool_name="run_shell",
            )
        for allowed_prefix in self.shell_allowlist:
            if argv_tuple[: len(allowed_prefix)] == allowed_prefix:
                return PermissionDecision.allow()
        return self._deny(
            f"command does not match the shell allowlist: {list(argv)}",
            denial=DenialReason.NOT_ALLOWLISTED,
            tool_name="run_shell",
        )

    def check_opaque_tool(
        self,
        tool_name: str,
        *,
        target: str,
        details: str = "",
    ) -> PermissionDecision:
        """Decide a call whose blast radius is not derivable from its arguments."""
        if self.mode == "plan":
            return self._deny(
                "a read-only mode cannot permit an opaque effect",
                denial=DenialReason.PLAN_MODE,
                tool_name=tool_name,
            )
        if self.mode in ("prompt", "accept_edits"):
            return self._ask_approval(
                operation=tool_name,
                target=target,
                details=details,
            )
        return PermissionDecision.allow()

    def _ask_approval(
        self,
        *,
        operation: str,
        target: str,
        details: str = "",
    ) -> PermissionDecision:
        with self._approval_lock:
            self._permission_requested(operation)
            if self.approval_callback is None:
                return self._deny(
                    f"{operation} requires approval, but no approval callback is configured",
                    denial=DenialReason.NO_APPROVAL_CALLBACK,
                    tool_name=operation,
                )
            try:
                decision = self.approval_callback(
                    ToolApprovalRequest(
                        operation=operation, target=target, details=details
                    )
                )
            except Exception as exc:  # noqa: BLE001
                return self._deny(
                    f"approval callback failed ({type(exc).__name__}): {exc}",
                    denial=DenialReason.APPROVAL_FAILED,
                    tool_name=operation,
                )
            if isinstance(decision, PermissionDecision):
                if not decision.allowed:
                    sink, agent_id, run_id, turn_id, tool_name, tool_call_id = (
                        self._event_identity(operation)
                    )
                    emit(
                        sink,
                        PermissionDenied(
                            agent_id=agent_id,
                            run_id=run_id,
                            turn_id=turn_id,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            reason=decision.reason or "permission denied",
                        ),
                    )
                return decision
            if decision is True:
                return PermissionDecision.allow()
            if decision is False:
                return self._deny(
                    f"{operation} denied by user",
                    denial=DenialReason.DENIED_BY_USER,
                    tool_name=operation,
                )
            return self._deny(
                f"approval callback returned an invalid decision for {operation}",
                denial=DenialReason.INVALID_APPROVAL,
                tool_name=operation,
            )


def _contains_path(parent: Path, child: Path) -> bool:
    return child == parent or parent in child.parents


def _matching_forbidden_component(
    component: str, patterns: tuple[str, ...]
) -> str | None:
    for pattern in patterns:
        if fnmatch.fnmatch(component, pattern.rstrip("/")):
            return pattern
    return None


def _intersect_write_scopes(
    parent_scopes: list[Path], ceiling_scopes: list[Path]
) -> list[Path]:
    intersections: list[Path] = []
    for parent_scope in parent_scopes:
        for ceiling_scope in ceiling_scopes:
            if _contains_path(parent_scope, ceiling_scope):
                candidate = ceiling_scope
            elif _contains_path(ceiling_scope, parent_scope):
                candidate = parent_scope
            else:
                continue
            if candidate not in intersections:
                intersections.append(candidate)
    return intersections


def _ordered_union(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*first, *second)))


def _intersect_shell_allowlists(
    parent_allowlist: list[tuple[str, ...]], ceiling_allowlist: list[tuple[str, ...]]
) -> list[tuple[str, ...]]:
    intersections: list[tuple[str, ...]] = []
    for parent_prefix in parent_allowlist:
        for ceiling_prefix in ceiling_allowlist:
            if _is_prefix(parent_prefix, ceiling_prefix):
                candidate = ceiling_prefix
            elif _is_prefix(ceiling_prefix, parent_prefix):
                candidate = parent_prefix
            else:
                continue
            if candidate not in intersections:
                intersections.append(candidate)
    return intersections


def _is_prefix(prefix: tuple[str, ...], value: tuple[str, ...]) -> bool:
    return value[: len(prefix)] == prefix
