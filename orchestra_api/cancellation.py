"""Turn-scoped cancellation primitives for Orchestra runtime operations."""

from __future__ import annotations

import threading


class OperationCancelled(Exception):
    """Raised inside a cancelled operation to unwind to the turn boundary.

    Callers outside orchestra_api should not see this: ``ApiAgent.run()``
    catches it and reports ``stopped_reason="cancelled"`` instead.
    """


class CancellationToken:
    """A one-way switch, shared by everything running inside one turn.

    Backed by ``threading.Event`` so ``wait()`` is an interruptible sleep: a
    retry backoff parks on the token instead of on ``time.sleep``, and wakes
    the moment another thread calls ``cancel()``.

    Cancellation is one-way; there is no reset. A cancelled turn is done.
    Mid-request HTTP cancellation remains bounded by the socket timeout because
    ``urlopen`` cannot be interrupted without moving it to another thread.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation. Idempotent, and safe from any thread."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise ``OperationCancelled`` if already cancelled, else return."""
        if self.cancelled:
            raise OperationCancelled

    def wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``, waking early if cancelled.

        Returns ``True`` if the token was cancelled during the wait.
        """
        return self._event.wait(seconds)
