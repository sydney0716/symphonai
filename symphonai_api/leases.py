"""In-process advisory leases for exclusive workspace writes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading


class LeaseConflict(RuntimeError):
    """Raised when a prefix is already leased by another holder."""


@dataclass(frozen=True)
class Lease:
    holder: str
    prefix: str
    root: Path


@dataclass
class _HeldLease:
    lease: Lease
    path: Path
    count: int = 1


def _contains_path(parent: Path, child: Path) -> bool:
    return parent == child or parent in child.parents


class WorkspaceLeases:
    """One writer at a time per path prefix, within this process."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()
        self._lock = threading.Lock()
        self._held: list[_HeldLease] = []

    def _resolve_prefix(self, prefix: str | None) -> tuple[str, Path]:
        if prefix is None:
            requested = "."
            candidate = self._root
        else:
            requested = prefix
            candidate = Path(prefix)
            if not candidate.is_absolute():
                candidate = self._root / candidate
        resolved = candidate.resolve()
        if not _contains_path(self._root, resolved):
            raise ValueError(
                f"cannot lease prefix {resolved!s} outside root {self._root!s}"
            )
        return requested, resolved

    def acquire(self, holder: str, prefix: str | None) -> Lease:
        requested, path = self._resolve_prefix(prefix)
        lease = Lease(holder=holder, prefix=requested, root=self._root)
        with self._lock:
            for held in self._held:
                if held.lease.holder == holder:
                    continue
                if _contains_path(held.path, path) or _contains_path(path, held.path):
                    raise LeaseConflict(
                        f"prefix {requested!r} conflicts with holder {held.lease.holder!r}"
                    )
            for held in self._held:
                if held.lease == lease:
                    held.count += 1
                    return lease
            self._held.append(_HeldLease(lease=lease, path=path))
        return lease

    def release(self, lease: Lease) -> None:
        with self._lock:
            for index, held in enumerate(self._held):
                if held.lease != lease:
                    continue
                held.count -= 1
                if held.count == 0:
                    self._held.pop(index)
                return

    def holder_for(self, path: str | Path) -> str | None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve()
        if not _contains_path(self._root, resolved):
            return None
        with self._lock:
            containing = [held for held in self._held if _contains_path(held.path, resolved)]
            if not containing:
                return None
            return max(containing, key=lambda held: len(held.path.parts)).lease.holder

    @contextmanager
    def held(self, holder: str, prefix: str | None):
        lease = self.acquire(holder, prefix)
        try:
            yield lease
        finally:
            self.release(lease)
