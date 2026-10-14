"""Registration and execution support for repository checks."""

from __future__ import annotations

import os
import signal
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn


class CheckFailed(Exception):
    """One check's assertion failed. Caught by the driver, not by a check."""


REGISTRY: dict[str, Callable[[], None]] = {}
NEEDS_REPOSITORY: set[str] = set()
_CURRENT_LABELS: list[str] | None = None
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECK_TIMEOUT_SECONDS = 60


class _CheckTimedOut(Exception):
    """Raised by the alarm handler to end a stuck check."""


def fail(msg: str) -> NoReturn:
    """Abort the current check. Same call signature the smoke scripts use."""
    raise CheckFailed(msg)


def ok(msg: str) -> None:
    """Record a passing assertion inside the current check."""
    if _CURRENT_LABELS is None:
        raise RuntimeError("ok() called outside a running check")
    _CURRENT_LABELS.append(msg)


def check(
    name: str,
    *,
    needs_repository: bool = False,
) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """Register a zero-argument check function under `name`."""

    def register(function: Callable[[], None]) -> Callable[[], None]:
        if name in REGISTRY:
            raise ValueError(f"duplicate check name: {name!r}")
        REGISTRY[name] = function
        if needs_repository:
            NEEDS_REPOSITORY.add(name)
        return function

    return register


def names() -> list[str]:
    """Every registered check name, in registration order."""
    return list(REGISTRY)


def repository_names() -> list[str]:
    """Return repository-dependent checks in registration order."""
    return [name for name in REGISTRY if name in NEEDS_REPOSITORY]


def run(selector: str | None = None) -> int:
    """Run the selected checks. Returns 0 when all passed, 1 otherwise."""
    if selector is None:
        selected = list(REGISTRY.items())
    else:
        normalized_selector = selector.casefold()
        selected = [
            (name, function)
            for name, function in REGISTRY.items()
            if normalized_selector in name.casefold()
        ]
    if not selected:
        print(f"no check matches {selector!r}")
        return 1
    if any(name.startswith("selfcheck.") for name in REGISTRY) and not all(
        name.startswith("selfcheck.") for name, _ in selected
    ):
        print("refusing to mix selfcheck fixtures with real checks")
        return 1

    # A clone has .git and none of the documents these checks read, so .git was
    # the wrong question: it made six checks fail for anyone who cloned.
    # docs/roadmap.json is present in the working repository and gitignored
    # here, which is the real distinction.
    repository_present = (REPOSITORY_ROOT / "docs" / "roadmap.json").is_file()

    global _CURRENT_LABELS
    passed = 0
    failed = 0
    skipped = 0
    previous_sessions_root = os.environ.get("SYMPHONAI_SESSIONS_DIR")
    previous_memory_root = os.environ.get("SYMPHONAI_MEMORY_DIR")
    timeout_seconds = int(
        os.environ.get("SYMPHONAI_CHECK_TIMEOUT", DEFAULT_CHECK_TIMEOUT_SECONDS)
    )
    if timeout_seconds < 1:
        raise ValueError("SYMPHONAI_CHECK_TIMEOUT must be at least 1")
    alarm_available = hasattr(signal, "SIGALRM")

    def timeout_check(signum, frame) -> None:  # noqa: ANN001
        raise _CheckTimedOut

    with tempfile.TemporaryDirectory() as temporary:
        os.environ["SYMPHONAI_SESSIONS_DIR"] = str(Path(temporary) / "sessions")
        os.environ["SYMPHONAI_MEMORY_DIR"] = str(Path(temporary) / "memory")
        try:
            for index, (name, function) in enumerate(selected):
                if not repository_present and name in NEEDS_REPOSITORY:
                    skipped += 1
                    print(f"SKIP  {name}: needs the working repository")
                    continue
                os.environ["SYMPHONAI_MEMORY_DIR"] = str(
                    Path(temporary) / str(index) / "memory"
                )
                labels: list[str] = []
                _CURRENT_LABELS = labels
                failure: str | None = None
                previous_handler = None
                previous_alarm = 0
                try:
                    if alarm_available:
                        previous_handler = signal.signal(signal.SIGALRM, timeout_check)
                        previous_alarm = signal.alarm(timeout_seconds)
                    function()
                except _CheckTimedOut:
                    failure = f"timed out after {timeout_seconds}s"
                except CheckFailed as exc:
                    failure = str(exc)
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                finally:
                    if alarm_available:
                        signal.alarm(0)
                        if previous_handler is not None:
                            signal.signal(signal.SIGALRM, previous_handler)
                        if previous_alarm:
                            signal.alarm(previous_alarm)
                    _CURRENT_LABELS = None

                if failure is None:
                    passed += 1
                    print(f"PASS  {name}")
                else:
                    failed += 1
                    print(f"FAIL  {name}: {failure}")
                for label in labels:
                    print(f"  OK:   {label}")
        finally:
            if previous_sessions_root is None:
                os.environ.pop("SYMPHONAI_SESSIONS_DIR", None)
            else:
                os.environ["SYMPHONAI_SESSIONS_DIR"] = previous_sessions_root
            if previous_memory_root is None:
                os.environ.pop("SYMPHONAI_MEMORY_DIR", None)
            else:
                os.environ["SYMPHONAI_MEMORY_DIR"] = previous_memory_root

    print(
        f"{passed} passed, {failed} failed, {skipped} skipped, "
        f"{len(selected)} selected of {len(REGISTRY)} registered"
    )
    return 1 if failed else 0
