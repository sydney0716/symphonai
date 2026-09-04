"""Turn-scoped cancellation primitives for SymphonAI runtime operations."""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import Enum


class OperationCancelled(Exception):
    """Raised inside a cancelled operation to unwind to the turn boundary.

    Callers outside symphonai_api should not see this: ``ApiAgent.run()``
    catches it and reports ``stopped_reason="cancelled"`` instead.
    """


class CancelReason(str, Enum):
    """The first cause that cancelled a token."""

    EXPLICIT = "explicit"
    PARENT = "parent"
    DEADLINE = "deadline"


class CancellationToken:
    """A one-way switch, shared by everything running inside one turn.

    Backed by ``threading.Event`` so ``wait()`` is an interruptible sleep: a
    retry backoff parks on the token instead of on ``time.sleep``, and wakes
    the moment another thread calls ``cancel()``.

    Cancellation is one-way; there is no reset. A cancelled turn is done.
    Mid-request HTTP cancellation remains bounded by the socket timeout because
    ``urlopen`` cannot be interrupted without moving it to another thread.
    """

    def __init__(self, *, deadline_seconds: float | None = None) -> None:
        if deadline_seconds is not None and not deadline_seconds > 0:
            raise ValueError("deadline_seconds must be positive")
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: CancelReason | None = None
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._next_callback_id = 0
        self._parent_unsubscribe: Callable[[], None] | None = None
        self._deadline_timer: threading.Timer | None = None
        if deadline_seconds is not None:
            timer = threading.Timer(deadline_seconds, self._deadline_elapsed)
            timer.daemon = True
            self._deadline_timer = timer
            timer.start()

    def cancel(self) -> None:
        """Request cancellation. Idempotent, and safe from any thread."""
        self._cancel(CancelReason.EXPLICIT)

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> CancelReason | None:
        """Why this token is cancelled, or ``None`` while it is live."""
        with self._lock:
            return self._reason

    def child(self, *, deadline_seconds: float | None = None) -> "CancellationToken":
        """Create a token cancelled by this token, but never vice versa."""
        child = CancellationToken(deadline_seconds=deadline_seconds)
        unsubscribe = self.on_cancel(lambda: child._cancel(CancelReason.PARENT))
        with child._lock:
            child._parent_unsubscribe = unsubscribe
        return child

    def on_cancel(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register ``callback`` and return an idempotent unsubscriber."""
        with self._lock:
            if self._event.is_set():
                callback_id: int | None = None
            else:
                callback_id = self._next_callback_id
                self._next_callback_id += 1
                self._callbacks[callback_id] = callback

        if callback_id is None:
            self._invoke(callback)

            def unsubscribe() -> None:
                return

            return unsubscribe

        def unsubscribe() -> None:
            with self._lock:
                self._callbacks.pop(callback_id, None)

        return unsubscribe

    def close(self) -> None:
        """Detach from the parent and stop the deadline timer."""
        with self._lock:
            parent_unsubscribe = self._parent_unsubscribe
            self._parent_unsubscribe = None
            deadline_timer = self._deadline_timer
            self._deadline_timer = None
        if parent_unsubscribe is not None:
            parent_unsubscribe()
        if deadline_timer is not None:
            deadline_timer.cancel()

    def raise_if_cancelled(self) -> None:
        """Raise ``OperationCancelled`` if already cancelled, else return."""
        if self.cancelled:
            raise OperationCancelled

    def wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``, waking early if cancelled.

        Returns ``True`` if the token was cancelled during the wait.
        """
        return self._event.wait(seconds)

    def _deadline_elapsed(self) -> None:
        self._cancel(CancelReason.DEADLINE)

    def _cancel(self, reason: CancelReason) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()
            callbacks = list(self._callbacks.values())
            self._callbacks.clear()
            deadline_timer = self._deadline_timer
            self._deadline_timer = None
        if deadline_timer is not None:
            deadline_timer.cancel()
        for callback in callbacks:
            self._invoke(callback)

    @staticmethod
    def _invoke(callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            pass
