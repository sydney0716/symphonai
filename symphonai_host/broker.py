"""Non-blocking fan-out for the host's ephemeral runtime event stream."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

from symphonai_api.events import Event


@dataclass
class Subscription:
    """One independently bounded event queue owned by an EventBroker."""

    _broker: "EventBroker"
    _queue: queue.Queue[Event | None]
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _dropped: int = 0
    _closed: bool = False

    def get(self, timeout: float | None = None) -> Event | None:
        """Return an event, None on timeout, or None after broker closure."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def take_dropped(self) -> int:
        """Return and reset the number of old events discarded for this peer."""
        with self._lock:
            dropped = self._dropped
            self._dropped = 0
            return dropped

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        self._broker.unsubscribe(self)


class EventBroker:
    """Fan out events without allowing a slow subscriber to stall a run."""

    MAX_QUEUED_EVENTS = 1000

    def __init__(self, *, max_queued_events: int = MAX_QUEUED_EVENTS) -> None:
        if max_queued_events < 1:
            raise ValueError("max_queued_events must be at least 1")
        self._max_queued_events = max_queued_events
        self._subscriptions: list[Subscription] = []
        self._lock = threading.Lock()
        self._closed = False

    def subscribe(self) -> Subscription:
        subscription = Subscription(self, queue.Queue(maxsize=self._max_queued_events))
        with self._lock:
            if self._closed:
                subscription._closed = True
                subscription._queue.put_nowait(None)
            else:
                self._subscriptions.append(subscription)
        return subscription

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscriptions)

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)
            with subscription._lock:
                subscription._closed = True

    def publish(self, event: Event) -> None:
        with self._lock:
            subscriptions = tuple(self._subscriptions)
        for subscription in subscriptions:
            with subscription._lock:
                if subscription._closed:
                    continue
                try:
                    subscription._queue.put_nowait(event)
                except queue.Full:
                    # Runtime events are explicitly lossy observation. Removing
                    # the oldest preserves the latest state for a slow client.
                    subscription._queue.get_nowait()
                    subscription._dropped += 1
                    subscription._queue.put_nowait(event)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            subscriptions = tuple(self._subscriptions)
            self._subscriptions.clear()
        for subscription in subscriptions:
            with subscription._lock:
                subscription._closed = True
                try:
                    subscription._queue.put_nowait(None)
                except queue.Full:
                    subscription._queue.get_nowait()
                    subscription._queue.put_nowait(None)
