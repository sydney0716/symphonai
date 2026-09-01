"""Thread-safe consecutive-failure circuit breaking."""

from __future__ import annotations

import threading


DEFAULT_MAX_CONSECUTIVE_FAILURES = 3


class CircuitOpen(RuntimeError):
    """Raised when a repair loop has failed too many times in a row."""


class ConsecutiveFailureBreaker:
    """Counts consecutive failures of one automatic operation and gives up."""

    def __init__(
        self,
        name: str,
        *,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    ) -> None:
        if max_consecutive_failures < 1:
            raise ValueError(
                "max_consecutive_failures must be >= 1, "
                f"got {max_consecutive_failures}"
            )
        self._name = name
        self._max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._consecutive_failures >= self._max_consecutive_failures

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self) -> int:
        with self._lock:
            self._consecutive_failures += 1
            return self._consecutive_failures

    def raise_if_open(self) -> None:
        with self._lock:
            count = self._consecutive_failures
            is_open = count >= self._max_consecutive_failures
        if is_open:
            raise CircuitOpen(
                f"{self._name} gave up after {count} consecutive failures"
            )

    def reset(self) -> None:
        self.record_success()
