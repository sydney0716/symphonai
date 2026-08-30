"""Registration and execution support for repository checks."""

from __future__ import annotations

import sys


sys.dont_write_bytecode = True

import atexit
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn


def _remove_check_bytecode() -> None:
    # A direct submodule import starts loading the package before this flag runs.
    shutil.rmtree(Path(__file__).parent / "__pycache__", ignore_errors=True)


atexit.register(_remove_check_bytecode)


class CheckFailed(Exception):
    """One check's assertion failed. Caught by the driver, not by a check."""


REGISTRY: dict[str, Callable[[], None]] = {}
_CURRENT_LABELS: list[str] | None = None


def fail(msg: str) -> NoReturn:
    """Abort the current check. Same call signature the smoke scripts use."""
    raise CheckFailed(msg)


def ok(msg: str) -> None:
    """Record a passing assertion inside the current check."""
    if _CURRENT_LABELS is None:
        raise RuntimeError("ok() called outside a running check")
    _CURRENT_LABELS.append(msg)


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

    global _CURRENT_LABELS
    passed = 0
    failed = 0
    for name, function in selected:
        labels: list[str] = []
        _CURRENT_LABELS = labels
        failure: str | None = None
        try:
            function()
        except CheckFailed as exc:
            failure = str(exc)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            _CURRENT_LABELS = None

        if failure is None:
            passed += 1
            print(f"PASS  {name}")
        else:
            failed += 1
            print(f"FAIL  {name}: {failure}")
        for label in labels:
            print(f"  OK:   {label}")

    print(
        f"{passed} passed, {failed} failed, {len(selected)} selected of "
        f"{len(REGISTRY)} registered"
    )
    return 1 if failed else 0
