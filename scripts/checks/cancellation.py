"""Checks for composable cancellation tokens."""

from __future__ import annotations

import threading

from symphonai_api.cancellation import CancelReason, CancellationToken, OperationCancelled
from scripts.checks.harness import check, fail


@check("cancellation.plain_token_unchanged")
def check_plain_token_unchanged() -> None:
    token = CancellationToken()
    if token._deadline_timer is not None:  # noqa: SLF001
        fail("a plain cancellation token started a thread")
    if token.cancelled or token.reason is not None:
        fail("a new token was not live")
    if token.wait(0.01):
        fail("a live token woke its wait")
    wake_token = CancellationToken()
    timer = threading.Timer(0.01, wake_token.cancel)
    timer.daemon = True
    timer.start()
    if not wake_token.wait(0.5):
        fail("a cancelled token did not wake its wait")
    try:
        wake_token.raise_if_cancelled()
    except OperationCancelled:
        pass
    else:
        fail("a cancelled token did not raise OperationCancelled")


@check("cancellation.parent_cancels_the_subtree")
def check_parent_cancels_the_subtree() -> None:
    parent = CancellationToken()
    child = parent.child()
    grandchild = child.child()
    great_grandchild = grandchild.child()
    parent.cancel()
    for token in (child, grandchild, great_grandchild):
        if not token.cancelled or token.reason is not CancelReason.PARENT:
            fail(f"parent cancellation did not reach descendant: {token.reason!r}")


@check("cancellation.child_never_cancels_the_parent")
def check_child_never_cancels_the_parent() -> None:
    parent = CancellationToken()
    child = parent.child()
    sibling = parent.child()
    child.cancel()
    if parent.cancelled or sibling.cancelled:
        fail("a child cancellation propagated upward or sideways")
    if child.reason is not CancelReason.EXPLICIT:
        fail(f"child had wrong explicit reason: {child.reason!r}")


@check("cancellation.reasons")
def check_reasons() -> None:
    token = CancellationToken()
    if token.reason is not None:
        fail("a live token had a cancellation reason")
    token.cancel()
    if token.reason is not CancelReason.EXPLICIT:
        fail(f"explicit cancellation had wrong reason: {token.reason!r}")

    parent = CancellationToken()
    child = parent.child()
    child.cancel()
    parent.cancel()
    if child.reason is not CancelReason.EXPLICIT:
        fail("a later parent cancellation overwrote the first reason")

    deadline = CancellationToken(deadline_seconds=0.01)
    fired = threading.Event()
    deadline.on_cancel(fired.set)
    if not fired.wait(0.5) or deadline.reason is not CancelReason.DEADLINE:
        fail(f"deadline did not set its reason: {deadline.reason!r}")
    deadline.cancel()
    if deadline.reason is not CancelReason.DEADLINE:
        fail("explicit cancellation overwrote a deadline reason")


@check("cancellation.child_of_a_cancelled_parent")
def check_child_of_a_cancelled_parent() -> None:
    parent = CancellationToken()
    parent.cancel()
    listener_count = len(parent._callbacks)  # noqa: SLF001
    child = parent.child()
    if not child.cancelled or child.reason is not CancelReason.PARENT:
        fail(f"child of cancelled parent was not born cancelled: {child.reason!r}")
    if len(parent._callbacks) != listener_count:  # noqa: SLF001
        fail("child of a cancelled parent registered a listener")


@check("cancellation.deadline_fires_and_is_cancellable")
def check_deadline_fires_and_is_cancellable() -> None:
    try:
        CancellationToken(deadline_seconds=0)
    except ValueError:
        pass
    else:
        fail("zero deadline was accepted")
    try:
        CancellationToken(deadline_seconds=-1)
    except ValueError:
        pass
    else:
        fail("negative deadline was accepted")

    deadline = CancellationToken(deadline_seconds=0.01)
    fired = threading.Event()
    deadline.on_cancel(fired.set)
    if not fired.wait(0.5) or not deadline.cancelled:
        fail("deadline did not cancel without another thread touching it")

    closed = CancellationToken(deadline_seconds=0.05)
    closed_fired = threading.Event()
    closed.on_cancel(closed_fired.set)
    closed.close()
    if closed_fired.wait(0.1) or closed.cancelled:
        fail("closing a live deadline token did not stop its deadline")


@check("cancellation.listeners_fire_once")
def check_listeners_fire_once() -> None:
    token = CancellationToken()
    fired: list[str] = []

    def broken_listener() -> None:
        fired.append("broken")
        raise OperationCancelled

    token.on_cancel(broken_listener)
    token.on_cancel(lambda: fired.append("later"))
    removed = token.on_cancel(lambda: fired.append("removed"))
    removed()
    removed()
    token.cancel()
    if fired != ["broken", "later"]:
        fail(f"listeners did not fire once in order: {fired!r}")
    removed()

    immediate: list[str] = []
    already_cancelled = CancellationToken()
    already_cancelled.cancel()
    unsubscribe = already_cancelled.on_cancel(lambda: immediate.append("now"))
    if immediate != ["now"]:
        fail("a listener on a cancelled token did not fire before return")
    unsubscribe()
    unsubscribe()


@check("cancellation.close_removes_the_listener")
def check_close_removes_the_listener() -> None:
    parent = CancellationToken()
    initial_listener_count = len(parent._callbacks)  # noqa: SLF001
    children = [parent.child() for _ in range(200)]
    if len(parent._callbacks) != initial_listener_count + len(children):  # noqa: SLF001
        fail("children did not register one parent listener each")
    for child in children:
        child.close()
        child.close()
    if len(parent._callbacks) != initial_listener_count:  # noqa: SLF001
        fail("closing children left parent listener entries behind")
    live_child = parent.child()
    live_child.close()
    if live_child.cancelled:
        fail("close cancelled a live child")
    live_child.cancel()
    if live_child.reason is not CancelReason.EXPLICIT:
        fail("a closed live child was no longer usable")


@check("cancellation.no_deadlock_from_a_callback")
def check_no_deadlock_from_a_callback() -> None:
    token = CancellationToken()
    completed = threading.Event()
    timed_out = threading.Event()

    def callback() -> None:
        token.close()
        token.cancel()

    token.on_cancel(callback)
    watchdog = threading.Timer(0.5, timed_out.set)
    watchdog.daemon = True
    worker = threading.Thread(target=lambda: (token.cancel(), completed.set()), daemon=True)
    watchdog.start()
    worker.start()
    finished = completed.wait(0.75)
    watchdog.cancel()
    if timed_out.is_set() or not finished:
        fail("cancellation deadlocked while a callback re-entered the token")


@check("cancellation.concurrent_cancel_and_child")
def check_concurrent_cancel_and_child() -> None:
    for _ in range(200):
        parent = CancellationToken()
        start = threading.Barrier(2)
        cancelled = threading.Event()

        def cancel_parent() -> None:
            start.wait()
            parent.cancel()
            cancelled.set()

        worker = threading.Thread(target=cancel_parent)
        worker.start()
        start.wait()
        child = parent.child()
        if not cancelled.wait(0.5):
            fail("concurrent parent cancellation did not complete")
        worker.join(timeout=0.5)
        if worker.is_alive() or not child.cancelled:
            fail("a live child escaped concurrent parent cancellation")
