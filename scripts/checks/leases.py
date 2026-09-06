"""Checks for advisory in-process workspace leases."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from symphonai_api.leases import Lease, LeaseConflict, WorkspaceLeases
from scripts.checks.agent_spec import FORBIDDEN_IMPORTS, _forbidden_imports
from scripts.checks.harness import check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]


def _root() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory()


def _expect_conflict(leases: WorkspaceLeases, holder: str, prefix: str | None) -> None:
    try:
        leases.acquire(holder, prefix)
    except LeaseConflict as exc:
        if repr(prefix if prefix is not None else ".") not in str(exc):
            fail(f"conflict omitted prefix: {exc!r}")
        if "first" not in str(exc):
            fail(f"conflict omitted existing holder: {exc!r}")
        return
    fail(f"conflicting prefix was acquired: {prefix!r}")


@check("leases.acquire_and_release")
def acquire_and_release() -> None:
    with _root() as temporary:
        root = Path(temporary)
        leases = WorkspaceLeases(root)
        whole_root = leases.acquire("first", None)
        if whole_root != Lease("first", ".", root.resolve()):
            fail(f"whole-root lease changed shape: {whole_root!r}")
        leases.release(whole_root)
        leases.release(whole_root)
        lease = leases.acquire("first", "src")
        leases.release(Lease("second", "src", root.resolve()))
        if leases.holder_for("src/file.py") != "first":
            fail("releasing an unknown lease freed the held lease")
        leases.release(lease)
        if leases.holder_for("src/file.py") is not None:
            fail("released lease remained held")


@check("leases.conflicts_by_containment")
def conflicts_by_containment() -> None:
    with _root() as temporary:
        root = Path(temporary)
        leases = WorkspaceLeases(root)
        leases.acquire("first", "src")
        _expect_conflict(leases, "second", "src")
        _expect_conflict(leases, "second", "src/api")
        if leases.acquire("second", "docs").holder != "second":
            fail("disjoint lease was rejected")

        leases = WorkspaceLeases(root)
        leases.acquire("first", "src/api")
        _expect_conflict(leases, "second", "src")

        leases = WorkspaceLeases(root)
        leases.acquire("first", None)
        _expect_conflict(leases, "second", "src")

        leases = WorkspaceLeases(root)
        leases.acquire("first", "src")
        _expect_conflict(leases, "second", None)


@check("leases.same_holder_reentrant")
def same_holder_reentrant() -> None:
    with _root() as temporary:
        leases = WorkspaceLeases(Path(temporary))
        first = leases.acquire("holder", "src")
        second = leases.acquire("holder", "src")
        nested = leases.acquire("holder", "src/api")
        leases.release(second)
        if leases.holder_for("src/file.py") != "holder":
            fail("one release freed a re-entrant lease")
        leases.release(nested)
        if leases.holder_for("src/file.py") != "holder":
            fail("nested release freed its containing lease")
        leases.release(first)
        if leases.holder_for("src/file.py") is not None:
            fail("final release did not free the lease")


@check("leases.holder_for_path")
def holder_for_path() -> None:
    with _root() as temporary:
        root = Path(temporary)
        leases = WorkspaceLeases(root)
        whole_root = leases.acquire("root", None)
        if leases.holder_for("docs/readme.md") != "root":
            fail("whole-root lease did not contain a descendant")
        leases.release(whole_root)
        outer = leases.acquire("outer", "src")
        inner = Lease("inner", "src/api", root.resolve())
        leases._held.append(type(leases._held[0])(inner, root.resolve() / "src/api"))
        if leases.holder_for("src/api/file.py") != "inner":
            fail("holder_for did not select the innermost lease")
        if leases.holder_for("docs/readme.md") is not None:
            fail("holder_for returned a lease outside its prefix")
        leases.release(outer)


@check("leases.prefix_outside_root")
def prefix_outside_root() -> None:
    with _root() as temporary:
        root = Path(temporary) / "root"
        root.mkdir()
        leases = WorkspaceLeases(root)
        try:
            leases.acquire("holder", "../outside")
        except ValueError as exc:
            if str(root.resolve()) not in str(exc) or "outside" not in str(exc):
                fail(f"outside-prefix error omitted paths: {exc!r}")
        else:
            fail("outside prefix was accepted")
        if leases.acquire("holder", "src/../docs").holder != "holder":
            fail("inside prefix containing '..' was rejected")


@check("leases.held_releases_on_exception")
def held_releases_on_exception() -> None:
    with _root() as temporary:
        leases = WorkspaceLeases(Path(temporary))
        try:
            with leases.held("holder", "src"):
                raise RuntimeError("inside")
        except RuntimeError as exc:
            if str(exc) != "inside":
                fail(f"context manager changed exception: {exc!r}")
        else:
            fail("context manager swallowed exception")
        if leases.holder_for("src/file.py") is not None:
            fail("context manager did not release on exception")


@check("leases.concurrent_acquire_has_one_winner")
def concurrent_acquire_has_one_winner() -> None:
    with _root() as temporary:
        root = Path(temporary)
        for round_number in range(20):
            leases = WorkspaceLeases(root)
            barrier = threading.Barrier(16)
            winners: list[Lease] = []
            conflicts: list[LeaseConflict] = []
            results_lock = threading.Lock()

            def acquire_one(index: int) -> None:
                barrier.wait()
                try:
                    lease = leases.acquire(f"holder-{index}", "src")
                except LeaseConflict as exc:
                    with results_lock:
                        conflicts.append(exc)
                else:
                    with results_lock:
                        winners.append(lease)

            threads = [threading.Thread(target=acquire_one, args=(index,)) for index in range(16)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if len(winners) != 1 or len(conflicts) != 15:
                fail(
                    f"round {round_number} had {len(winners)} winners and "
                    f"{len(conflicts)} conflicts"
                )
            leases.release(winners[0])
            leases.acquire("after", "src")

        leases = WorkspaceLeases(root)
        leases.acquire("holder", "src")
        original_resolve = Path.resolve

        def checked_resolve(path: Path, *args, **kwargs) -> Path:
            if leases._lock.locked():
                fail("holder_for resolved a path while the registry lock was held")
            return original_resolve(path, *args, **kwargs)

        Path.resolve = checked_resolve
        try:
            if leases.holder_for("src/file.py") != "holder":
                fail("holder_for changed while checking its lock scope")
        finally:
            Path.resolve = original_resolve
        started = threading.Event()
        returned = threading.Event()

        def query_holder() -> None:
            started.set()
            leases.holder_for("src/file.py")
            returned.set()

        leases._lock.acquire()
        thread = threading.Thread(target=query_holder)
        try:
            thread.start()
            if not started.wait(1):
                fail("holder_for thread did not start")
            if returned.wait(0.05):
                fail("holder_for bypassed the registry lock")
        finally:
            leases._lock.release()
        if not returned.wait(1):
            fail("holder_for did not return after the registry lock was released")
        thread.join()


@check("leases.no_symphonai_imports")
def no_symphonai_imports() -> None:
    source = (REPO_ROOT / "symphonai_api/leases.py").read_text()
    original_forbidden = set(FORBIDDEN_IMPORTS)
    FORBIDDEN_IMPORTS.add("symphonai_api")
    try:
        if _forbidden_imports(source):
            fail("leases imports symphonai_api")
        if not _forbidden_imports("import symphonai_api\n"):
            fail("import inspection missed the symphonai_api package")
    finally:
        FORBIDDEN_IMPORTS.clear()
        FORBIDDEN_IMPORTS.update(original_forbidden)
