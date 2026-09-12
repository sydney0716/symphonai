"""Checks for the bounded, permission-gated repository survey."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import symphonai_api.survey as survey_module
from symphonai_api.models import Message, ModelResponse, Role
from symphonai_api.permissions import DenialReason, PermissionDecision, PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.survey import DOCUMENT_READ_LIMIT, survey_repository
from symphonai_host.broker import EventBroker
from symphonai_host.run import HostRun
from scripts.checks.harness import check, fail


ROOT = Path(__file__).resolve().parents[2]


def _policy(root: Path) -> PermissionPolicy:
    return PermissionPolicy(repo_root=root)


def _relative(root: Path, paths: tuple[Path, ...]) -> tuple[str, ...]:
    return tuple(path.relative_to(root).as_posix() for path in paths)


class _Entries:
    def __init__(self, entries) -> None:  # noqa: ANN001
        self.entries = entries

    def __enter__(self):  # noqa: ANN204
        return iter(self.entries)

    def __exit__(self, *_args) -> None:  # noqa: ANN002
        return None


@check("survey.languages")
def check_languages() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        (root / "one.py").write_text("", encoding="utf-8")
        (root / "two.PY").write_text("", encoding="utf-8")
        (root / "app.js").write_text("", encoding="utf-8")
        result = survey_repository(policy=_policy(root))
        if result.root != root or result.languages != ((".py", 2), (".js", 1)):
            fail(f"language survey was wrong: {result!r}")


@check("survey.deterministic")
def check_deterministic() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bucket = root / "bucket"
        bucket.mkdir()
        for index in range(125):
            (bucket / f"a-{index:03}.py").write_text("", encoding="utf-8")
            (bucket / f"z-{index:03}.js").write_text("", encoding="utf-8")

        original_scandir = survey_module.os.scandir
        visits: dict[Path, int] = {}

        def alternating_scandir(directory):  # noqa: ANN001, ANN202
            path = Path(directory)
            with original_scandir(path) as scanner:
                entries = list(scanner)
            visits[path] = visits.get(path, 0) + 1
            if visits[path] % 2 == 0:
                entries.reverse()
            return _Entries(entries)

        with mock.patch.object(
            survey_module.os,
            "scandir",
            side_effect=alternating_scandir,
        ):
            first = survey_repository(policy=_policy(root), max_files=200)
            second = survey_repository(policy=_policy(root), max_files=200)

        fields = ("languages", "entry_points", "docs", "tests", "by_directory")
        if any(getattr(first, name) != getattr(second, name) for name in fields):
            fail(f"directory iteration order changed the survey: {first!r}, {second!r}")


@check("survey.breadth_first")
def check_breadth_first() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bucket = root / "bucket"
        shallow = bucket / "a-shallow"
        deep = bucket / "z-deep" / "level-2" / "level-3"
        deep.mkdir(parents=True)
        shallow.mkdir(parents=True)
        (root / "main.py").write_text("", encoding="utf-8")
        for index in range(200):
            (shallow / f"file-{index:03}.py").write_text("", encoding="utf-8")
        (deep / "deep.ts").write_text("", encoding="utf-8")

        original_scandir = survey_module.os.scandir

        def ascending_scandir(directory):  # noqa: ANN001, ANN202
            with original_scandir(directory) as scanner:
                entries = sorted(scanner, key=lambda entry: entry.name)
            return _Entries(entries)

        with mock.patch.object(
            survey_module.os,
            "scandir",
            side_effect=ascending_scandir,
        ):
            result = survey_repository(policy=_policy(root), max_files=201)
        if root / "main.py" not in result.entry_points:
            fail(f"shallow top-level file was omitted: {result!r}")
        if any(extension == ".ts" for extension, _ in result.languages):
            fail(f"deep file was visited before shallow files: {result!r}")


@check("survey.directory_budgets")
def check_directory_budgets() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        heavy = root / "a-heavy"
        python = root / "b-python"
        javascript = root / "c-javascript"
        for directory in (heavy, python, javascript):
            directory.mkdir()
        for index in range(1000):
            (heavy / f"reference-{index:04}.ts").write_text("", encoding="utf-8")
        for index in range(10):
            (python / f"module-{index:02}.py").write_text("", encoding="utf-8")
            (javascript / f"module-{index:02}.js").write_text("", encoding="utf-8")

        result = survey_repository(policy=_policy(root), max_files=600)
        buckets = dict(result.by_directory)
        if tuple(buckets) != ("a-heavy", "b-python", "c-javascript"):
            fail(f"directory buckets were missing or unordered: {result.by_directory!r}")
        if buckets["a-heavy"] != ((".ts", 200),):
            fail(f"dominant directory exceeded its share: {result.by_directory!r}")
        if buckets["b-python"] != ((".py", 10),):
            fail(f"Python directory did not contribute: {result.by_directory!r}")
        if buckets["c-javascript"] != ((".js", 10),):
            fail(f"JavaScript directory did not contribute: {result.by_directory!r}")
        totals = dict(result.languages)
        for counts in buckets.values():
            for extension, count in counts:
                if count > totals.get(extension, 0):
                    fail(f"directory count exceeded repository total: {result!r}")


@check("survey.structured_truncation")
def check_structured_truncation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        (root / "one.py").write_text("", encoding="utf-8")
        (root / "two.py").write_text("", encoding="utf-8")
        complete = survey_repository(policy=_policy(root), max_files=3)
        stopped = survey_repository(policy=_policy(root), max_files=1)
        if complete.stopped or complete.file_count != 2:
            fail(f"complete survey reported wrong bounds: {complete!r}")
        if not stopped.stopped or stopped.file_count != 1:
            fail(f"truncated survey reported wrong bounds: {stopped!r}")


@check("survey.host_policy_accessor")
def check_host_policy_accessor() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        policy = _policy(Path(temporary).resolve())
        run = HostRun(
            FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
            policy,
            EventBroker(),
        )
        if run.policy is not policy:
            fail("HostRun policy accessor returned a different policy")
        try:
            run.policy = _policy(ROOT)  # type: ignore[misc]
        except AttributeError:
            pass
        else:
            fail("HostRun policy accessor was writable")
        source = (ROOT / "symphonai_host" / "server.py").read_text(encoding="utf-8")
        if "run._policy" in source:
            fail("host server still accessed the run's private policy")


@check("survey.categories")
def check_categories() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        (root / "docs").mkdir()
        (root / "tests").mkdir()
        (root / "src").mkdir()
        (root / "README.md").write_text("read me", encoding="utf-8")
        (root / "docs" / "guide.txt").write_text("guide", encoding="utf-8")
        (root / "pyproject.toml").write_text("", encoding="utf-8")
        (root / "src" / "main.py").write_text("", encoding="utf-8")
        result = survey_repository(policy=_policy(root))
        if _relative(root, result.entry_points) != ("pyproject.toml", "src/main.py"):
            fail(f"entry points were wrong: {result.entry_points!r}")
        if _relative(root, result.docs) != ("README.md", "docs/guide.txt"):
            fail(f"documentation was wrong: {result.docs!r}")
        if _relative(root, result.tests) != ("tests",):
            fail(f"test directories were wrong: {result.tests!r}")

    with tempfile.TemporaryDirectory() as temporary:
        empty = survey_repository(policy=_policy(Path(temporary).resolve()))
        if empty.languages or empty.entry_points or empty.docs or empty.tests:
            fail(f"empty repository produced categories: {empty!r}")


@check("survey.policy_boundary")
def check_policy_boundary() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        private = root / "private"
        private.mkdir()
        (private / "hidden.py").write_text("hidden", encoding="utf-8")
        (root / "visible.js").write_text("visible", encoding="utf-8")
        policy = _policy(root)
        original = policy.check_read
        asked: list[Path] = []

        def recording_check(path):  # noqa: ANN001
            resolved = Path(path).resolve()
            asked.append(resolved)
            if resolved == private or private in resolved.parents:
                return PermissionDecision.deny(
                    "fixture refusal", denial=DenialReason.FORBIDDEN_PATTERN
                )
            return original(path)

        policy.check_read = recording_check  # type: ignore[method-assign]
        result = survey_repository(policy=policy)
        expected_asked = {root.resolve(), private.resolve(), (root / "visible.js").resolve()}
        if not expected_asked.issubset(set(asked)):
            fail(f"survey used paths without asking the policy: {asked!r}")
        if "private" in result.tree_summary or result.languages != ((".js", 1),):
            fail(f"refused subtree reached the survey: {result!r}")


class _RecordedReader:
    def __init__(self, path: Path, handle, reads: list[tuple[Path, int]]) -> None:  # noqa: ANN001
        self.path = path
        self.handle = handle
        self.reads = reads

    def __enter__(self):  # noqa: ANN204
        self.handle.__enter__()
        return self

    def __exit__(self, *args):  # noqa: ANN002, ANN204
        return self.handle.__exit__(*args)

    def read(self, size: int = -1):  # noqa: ANN201
        self.reads.append((self.path, size))
        return self.handle.read(size)


@check("survey.content_bounds")
def check_content_bounds() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        (root / "docs").mkdir()
        readme = root / "README.md"
        guide = root / "docs" / "guide.md"
        source = root / "source.py"
        readme.write_bytes(b"r" * (1024 * 1024))
        guide.write_text("guide", encoding="utf-8")
        source.write_text("source contents must stay unread", encoding="utf-8")
        reads: list[tuple[Path, int]] = []
        original_open = Path.open

        def recording_open(path, *args, **kwargs):  # noqa: ANN001, ANN202
            return _RecordedReader(
                path,
                original_open(path, *args, **kwargs),
                reads,
            )

        with mock.patch.object(Path, "open", recording_open):
            survey_repository(policy=_policy(root))

        read_paths = {path for path, _ in reads}
        if read_paths != {readme, guide}:
            fail(f"survey read unexpected file contents: {reads!r}")
        if any(size != DOCUMENT_READ_LIMIT for _, size in reads):
            fail(f"documentation read exceeded its bound: {reads!r}")
        if source in read_paths:
            fail("source file contents were read")


@check("survey.max_files")
def check_max_files() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        for index in range(5):
            (root / f"file-{index}.py").write_text("", encoding="utf-8")
        result = survey_repository(policy=_policy(root), max_files=2)
        if sum(count for _, count in result.languages) != 2:
            fail(f"max_files did not bound surveyed files: {result.languages!r}")
        if "stopped after 2 files" not in result.tree_summary:
            fail(f"bounded survey did not record its stop: {result.tree_summary!r}")


@check("survey.tree_summary")
def check_tree_summary() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        source = root / "src"
        crowded = root / "crowded"
        source.mkdir()
        crowded.mkdir()
        (source / "main.py").write_text("", encoding="utf-8")
        for index in range(30):
            (crowded / f"entry-{index:02}.txt").write_text("", encoding="utf-8")
        tree = survey_repository(policy=_policy(root)).tree_summary
        for expected in ("src/", "main.py", "crowded/", "… 30 entries"):
            if expected not in tree:
                fail(f"tree summary omitted {expected!r}: {tree!r}")
        if "entry-00.txt" in tree:
            fail(f"crowded directory was not elided: {tree!r}")


@check("survey.forbidden_paths")
def check_forbidden_paths() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text("private", encoding="utf-8")
        (root / ".env").write_text("secret", encoding="utf-8")
        (root / "public.py").write_text("", encoding="utf-8")
        result = survey_repository(policy=_policy(root))
        if ".git" in result.tree_summary or ".env" in result.tree_summary:
            fail(f"forbidden paths reached the survey: {result.tree_summary!r}")
        if result.languages != ((".py", 1),):
            fail(f"forbidden paths affected language counts: {result.languages!r}")


def _snapshot(root: Path):  # noqa: ANN201
    paths = tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*")))
    files = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    return paths, files


@check("survey.read_only")
def check_read_only() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        (root / "docs").mkdir()
        (root / "README.md").write_text("read me", encoding="utf-8")
        (root / "docs" / "guide.md").write_text("guide", encoding="utf-8")
        (root / "source.py").write_text("source", encoding="utf-8")
        before = _snapshot(root)
        survey_repository(policy=_policy(root))
        after = _snapshot(root)
        if after != before:
            fail(f"survey changed its repository: before={before!r}, after={after!r}")


@check("survey.import_boundary")
def check_import_boundary() -> None:
    from scripts.checks.agent_spec import _forbidden_imports

    source = (ROOT / "symphonai_api" / "survey.py").read_text(encoding="utf-8")
    forbidden = _forbidden_imports(source)
    if forbidden:
        fail(f"survey imported forbidden runtime modules: {forbidden!r}")
