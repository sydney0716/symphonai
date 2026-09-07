"""Checks for advisory in-process workspace leases."""

from __future__ import annotations

import random
import tempfile
import threading
from pathlib import Path

import symphonai_api.leases as leases_module
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


def _holder_selection(leases: WorkspaceLeases, path: Path) -> tuple[str | None, str | None]:
    resolved = path.resolve()
    with leases._lock:
        containing = [
            held
            for held in leases._held
            if held.path == resolved or held.path in resolved.parents
        ]
    if not containing:
        return None, None
    innermost = max(containing, key=lambda held: len(held.path.parts)).lease.holder
    outermost = min(containing, key=lambda held: len(held.path.parts)).lease.holder
    return innermost, outermost


def _overlap_violation(leases: WorkspaceLeases) -> str | None:
    with leases._lock:
        held_entries = list(leases._held)
    for index, left in enumerate(held_entries):
        for right in held_entries[index + 1 :]:
            overlaps = (
                left.path == right.path
                or left.path in right.path.parents
                or right.path in left.path.parents
            )
            if overlaps and left.lease.holder != right.lease.holder:
                return (
                    f"distinct holders overlap: {left.lease!r} and "
                    f"{right.lease!r}"
                )
    return None


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
        outer = leases.acquire("holder", "src")
        inner = leases.acquire("holder", "src/api")
        for path in (root / "src/api/file.py", root / "src/other.py"):
            innermost, outermost = _holder_selection(leases, path)
            actual = leases.holder_for(path)
            if actual != innermost or actual != outermost:
                fail(
                    "reachable nested leases disagreed by selection: "
                    f"actual={actual!r}, inner={innermost!r}, outer={outermost!r}"
                )
        outside = root / "docs/readme.md"
        innermost, outermost = _holder_selection(leases, outside)
        if leases.holder_for(outside) != innermost or innermost != outermost:
            fail("holder_for returned a lease outside reachable prefixes")
        leases.release(inner)
        leases.release(outer)
        whole_root = leases.acquire("root", None)
        rooted = root / "docs/readme.md"
        innermost, outermost = _holder_selection(leases, rooted)
        if (
            leases.holder_for(rooted) != "root"
            or innermost != "root"
            or outermost != "root"
        ):
            fail("whole-root holder disagreed by selection")
        leases.release(whole_root)


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
            barrier_failures: list[str] = []
            results_lock = threading.Lock()

            def acquire_one(index: int) -> None:
                try:
                    barrier.wait(timeout=1)
                except threading.BrokenBarrierError:
                    with results_lock:
                        barrier_failures.append(
                            f"winner barrier broke for holder-{index}"
                        )
                    return
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
            if barrier_failures:
                fail(barrier_failures[0])
            if len(winners) != 1 or len(conflicts) != 15:
                fail(
                    f"round {round_number} had {len(winners)} winners and "
                    f"{len(conflicts)} conflicts"
                )
            leases.release(winners[0])
            leases.acquire("after", "src")

        leases = WorkspaceLeases(root)
        prefixes = (None, "src", "src/api", "src/ui", "docs", "docs/api", "tests")
        sweep_barrier = threading.Barrier(12)
        sweep_failures: list[str] = []
        sweep_failures_lock = threading.Lock()

        def wait_for_sweep(holder_index: int) -> bool:
            try:
                sweep_barrier.wait(timeout=1)
                return True
            except threading.BrokenBarrierError:
                sweep_barrier.abort()
                with sweep_failures_lock:
                    sweep_failures.append(
                        f"sweep barrier broke for holder-{holder_index}"
                    )
                return False

        def sweep(holder_index: int) -> None:
            randomizer = random.Random(holder_index)
            for _ in range(120):
                prefix = randomizer.choice(prefixes)
                try:
                    with leases.held(f"sweep-{holder_index}", prefix):
                        if not wait_for_sweep(holder_index):
                            return
                        violation = _overlap_violation(leases)
                        if violation is not None:
                            with sweep_failures_lock:
                                sweep_failures.append(violation)
                        if not wait_for_sweep(holder_index):
                            return
                except LeaseConflict:
                    if not wait_for_sweep(holder_index):
                        return
                    if not wait_for_sweep(holder_index):
                        return
                except Exception as exc:
                    sweep_barrier.abort()
                    with sweep_failures_lock:
                        sweep_failures.append(
                            f"sweep holder {holder_index} failed: {exc!r}"
                        )
                    return

        sweep_threads = [
            threading.Thread(target=sweep, args=(index,)) for index in range(12)
        ]
        for thread in sweep_threads:
            thread.start()
        for thread in sweep_threads:
            thread.join()
        if sweep_failures:
            fail(sweep_failures[0])
        if any(thread.is_alive() for thread in sweep_threads):
            fail("randomized lease invariant sweep did not finish")
        try:
            with leases.held("exception-probe", "exception"):
                raise RuntimeError("expected sweep exception")
        except RuntimeError:
            pass
        if leases._held:
            fail(f"randomized lease invariant sweep leaked entries: {leases._held!r}")

        probe_leases = WorkspaceLeases(root)
        probe_leases.acquire("existing", None)
        probe_barrier = threading.Barrier(2)
        probe_timeout = 0.2
        if probe_timeout <= 0:
            fail("mutual-exclusion probe timeout must be positive")
        both_inside = threading.Event()
        probe_outcomes: list[str] = []
        probe_outcomes_lock = threading.Lock()
        probe_call_counts = threading.local()
        original_contains_path = leases_module._contains_path

        def synchronized_contains_path(parent: Path, child: Path) -> bool:
            call_count = getattr(probe_call_counts, "value", 0) + 1
            probe_call_counts.value = call_count
            if call_count == 1:
                return original_contains_path(parent, child)
            try:
                probe_barrier.wait(timeout=probe_timeout)
            except threading.BrokenBarrierError:
                raise
            both_inside.set()
            return original_contains_path(parent, child)

        def probe_acquire(holder: str) -> None:
            try:
                probe_leases.acquire(holder, "src")
            except threading.BrokenBarrierError:
                outcome = "timed-out"
            except LeaseConflict:
                outcome = "conflict"
            except Exception as exc:
                outcome = f"error:{exc!r}"
            else:
                outcome = "acquired"
            with probe_outcomes_lock:
                probe_outcomes.append(f"{holder}:{outcome}")

        probe_threads = [
            threading.Thread(target=probe_acquire, args=(holder,))
            for holder in ("probe-one", "probe-two")
        ]
        try:
            leases_module._contains_path = synchronized_contains_path
            for thread in probe_threads:
                thread.start()
            for thread in probe_threads:
                thread.join(2)
        finally:
            leases_module._contains_path = original_contains_path
            probe_barrier.abort()
        if any(thread.is_alive() for thread in probe_threads):
            fail("mutual-exclusion probe threads did not finish")
        if both_inside.is_set():
            fail("probe-one and probe-two entered the critical section together")
        if sorted(probe_outcomes) != [
            "probe-one:timed-out",
            "probe-two:timed-out",
        ]:
            fail(f"mutual-exclusion probe had unexpected outcomes: {probe_outcomes!r}")

        append_scope_leases = WorkspaceLeases(root)
        append_outside_lock: list[Lease] = []

        class LockCheckingList(list):
            def append(self, item) -> None:  # noqa: ANN001
                if not append_scope_leases._lock.locked():
                    append_outside_lock.append(item.lease)
                super().append(item)

        append_scope_leases._held = LockCheckingList()
        append_scope_leases.acquire("append-probe", "src")
        if append_outside_lock:
            fail(f"lease was appended outside the registry lock: {append_outside_lock!r}")

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
