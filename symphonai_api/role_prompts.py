"""Load user and project prompt additions for agent roles."""

from __future__ import annotations

from pathlib import Path

from symphonai_api.permissions import PermissionPolicy, check_home_read
from symphonai_api.paths import symphonai_home

MAX_ROLE_PROMPT_CHARS = 40_000


def load_role_additions(
    name: str,
    *,
    repo_root: Path,
    policy: PermissionPolicy,
    user_home: Path | None = None,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return rendered additions, their source paths, and any warnings."""
    home = (Path(user_home) if user_home is not None else symphonai_home()).resolve()
    candidates = (
        (home / "roles" / f"{name}.md", "user"),
        (Path(repo_root) / ".symphonai" / "roles" / f"{name}.md", "project"),
    )
    additions: list[str] = []
    sources: list[str] = []
    warnings: list[str] = []
    for path, scope in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if scope == "user":
            reason = check_home_read(resolved, home, policy)
            if reason is not None:
                warnings.append(f"role prompt file {resolved} was denied: {reason}")
                continue
        else:
            decision = policy.check_read(resolved)
            if not decision.allowed:
                warnings.append(f"role prompt file {resolved} was denied: {decision.reason}")
                continue
        try:
            text = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            warnings.append(f"role prompt file could not be read: {resolved}: {exc}")
            continue
        if len(text) > MAX_ROLE_PROMPT_CHARS:
            warnings.append(
                f"role prompt file {resolved} has {len(text)} characters, over the "
                f"{MAX_ROLE_PROMPT_CHARS} character limit; skipped"
            )
            continue
        if not text.strip():
            continue
        additions.append(f"Additional instructions for {name} from {('your SymphonAI settings' if scope == 'user' else 'this project')}:\n{text}")
        sources.append(str(resolved))
    return "\n\n".join(additions), tuple(sources), tuple(warnings)
