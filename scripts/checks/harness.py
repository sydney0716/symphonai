"""Registration and execution support for repository checks."""

from __future__ import annotations

import os
import signal
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn


class CheckFailed(Exception):
    """One check's assertion failed. Caught by the driver, not by a check."""


class CheckSkipped(Exception):
    """A check cannot run in the current environment."""


def skip(reason: str) -> NoReturn:
    raise CheckSkipped(reason)


REGISTRY: dict[str, Callable[[], None]] = {}
DEFAULT_CHECK_TIMEOUT_SECONDS = 60


class _CheckTimedOut(Exception):
    """Raised by the alarm handler to end a stuck check."""


def fail(msg: str) -> NoReturn:
    """Abort the current check. Same call signature the smoke scripts use."""
    raise CheckFailed(msg)


def check(name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """Register a zero-argument check function under `name`."""

    def register(function: Callable[[], None]) -> Callable[[], None]:
        if name in REGISTRY:
            raise ValueError(f"duplicate check name: {name!r}")
        REGISTRY[name] = function
        return function

    return register


def names() -> list[str]:
    """Every registered check name, in registration order."""
    return list(REGISTRY)


def run(selector: str | None = None) -> int:
    """Run the selected checks. Returns 0 when all passed, 1 otherwise."""
    if sys.flags.optimize:
        print("checks require assertions; optimized Python (-O/PYTHONOPTIMIZE) is not supported", file=sys.stderr)
        return 1
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
    passed = 0
    failed = 0
    skipped = 0
    timeout_seconds = int(
        os.environ.get("SYMPHONAI_CHECK_TIMEOUT", DEFAULT_CHECK_TIMEOUT_SECONDS)
    )
    if timeout_seconds < 1:
        raise ValueError("SYMPHONAI_CHECK_TIMEOUT must be at least 1")
    alarm_available = hasattr(signal, "SIGALRM")

    def timeout_check(signum, frame) -> None:  # noqa: ANN001
        raise _CheckTimedOut

    for name, function in selected:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            isolated = {
                "HOME": str(root / "user"),
                "SYMPHONAI_HOME": str(root / "home"),
                "SYMPHONAI_CREDENTIALS_FILE": str(root / "home" / "credentials.json"),
                "SYMPHONAI_SESSIONS_DIR": str(root / "sessions"),
                "SYMPHONAI_MEMORY_DIR": str(root / "memory"),
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
            }
            previous_environment = {key: os.environ.get(key) for key in isolated}
            os.environ.update(isolated)
            failure: str | None = None
            skip_reason: str | None = None
            previous_handler = None
            previous_alarm = 0
            try:
                if alarm_available:
                    previous_handler = signal.signal(signal.SIGALRM, timeout_check)
                    previous_alarm = signal.alarm(timeout_seconds)
                function()
            except _CheckTimedOut:
                failure = f"timed out after {timeout_seconds}s"
            except CheckSkipped as exc:
                skip_reason = str(exc)
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
                for key, value in previous_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            if skip_reason is not None:
                skipped += 1
                print(f"SKIP  {name}: {skip_reason}")
            elif failure is None:
                passed += 1
                print(f"PASS  {name}")
            else:
                failed += 1
                print(f"FAIL  {name}: {failure}")

    print(
        f"{passed} passed, {failed} failed, {skipped} skipped, "
        f"{len(selected)} selected of {len(REGISTRY)} registered"
    )
    return 1 if failed else 0
