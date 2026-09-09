"""Schema and spec-binding checks for the roadmap data."""

from __future__ import annotations

import copy
import json
import re
import tempfile
from pathlib import Path

from scripts.checks.harness import check, fail, ok


REPO_ROOT = Path(__file__).resolve().parents[2]
BOUND_PHASES = ("10", "18", "19")
BASE_SPEC = re.compile(
    r"^specs/(?P<phase>[^/]+)/(?P<id>[0-9]+[A-Za-z])-[^/]+\.md$"
)
FOLLOW_UP_SPEC = re.compile(
    r"^specs/(?P<phase>[^/]+)/(?P<id>[0-9]+[A-Za-z])F[0-9]*-[^/]+\.md$"
)

# Specs that are deliberately not roadmap items, and why each is not.
UNBOUND_BY_DESIGN: dict[str, str] = {
    "specs/18/18j-five-flaky-checks-one-race.md":
        "test hygiene: five host checks racing a fifty-millisecond keepalive",
    "specs/18/18k-a-page-you-can-open.md":
        "development browser shell omitted from the phase 18 roadmap",
}


def _is_type(value, expected: str) -> bool:  # noqa: ANN001
    types = {
        "array": list,
        "integer": int,
        "object": dict,
        "string": str,
    }
    return isinstance(value, types[expected]) and not (
        expected == "integer" and isinstance(value, bool)
    )


def _schema_errors(value, schema: dict, path: str = "$") -> list[str]:  # noqa: ANN001
    if "oneOf" in schema:
        matches = [
            option
            for option in schema["oneOf"]
            if not _schema_errors(value, option, path)
        ]
        return [] if len(matches) == 1 else [f"{path}: expected exactly one schema"]

    errors = []
    expected_type = schema.get("type")
    if expected_type is not None and not _is_type(value, expected_type):
        return [f"{path}: expected {expected_type}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not in enum")
    if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
        errors.append(f"{path}: value does not match pattern")
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing {name}")
        for name, child in schema.get("properties", {}).items():
            if name in value:
                errors.extend(_schema_errors(value[name], child, f"{path}.{name}"))
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(_schema_errors(item, schema["items"], f"{path}[{index}]"))
    if isinstance(value, list) and len(value) < schema.get("minItems", 0):
        errors.append(f"{path}: expected at least {schema['minItems']} items")
    return errors


def _load(path: Path):  # noqa: ANN202
    return json.loads(path.read_text(encoding="utf-8"))


@check("roadmap_data.schema")
def schema_validation() -> None:
    schema = _load(REPO_ROOT / "docs" / "roadmap.schema.json")
    roadmap = _load(REPO_ROOT / "docs" / "roadmap.json")
    errors = _schema_errors(roadmap, schema)
    if errors:
        fail(f"real roadmap does not match its schema: {errors!r}")

    cases = [
        ("string", "specs/18/18c-the-roadmap-pane.md", True),
        (
            "array",
            ["specs/10/10b-the-missing-events.md", "specs/10/10c-hooks.md"],
            True,
        ),
        ("number", 3, False),
        ("empty array", [], False),
        ("array containing a number", ["specs/18/18c-the-roadmap-pane.md", 3], False),
    ]
    for label, spec, accepted in cases:
        candidate = copy.deepcopy(roadmap)
        structured = next(
            item
            for phase in candidate["phases"]
            for item in phase["items"]
            if isinstance(item, dict) and "spec" in item
        )
        structured["spec"] = spec
        matches = not _schema_errors(candidate, schema)
        if matches != accepted:
            fail(f"roadmap schema handled the {label} spec form incorrectly")


def _spec_paths(item: dict) -> list[str]:
    spec = item.get("spec")
    if isinstance(spec, str):
        return [spec]
    if isinstance(spec, list):
        return [path for path in spec if isinstance(path, str)]
    return []


def _unbound_set_errors(unbound_by_design: dict[str, str]) -> list[str]:
    expected = {
        "specs/18/18j-five-flaky-checks-one-race.md",
        "specs/18/18k-a-page-you-can-open.md",
    }
    return sorted(set(unbound_by_design) ^ expected)


def _report_path(root: Path, spec_path: str) -> Path:
    spec = Path(spec_path)
    return root / "specs" / "report" / spec.parent.name / f"{spec.stem}-report.md"


def _binding_status(
    roadmap: dict,
    root: Path,
    unbound_by_design: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    exclusions = unbound_by_design or {}
    excluded = set(exclusions)
    named = {
        path
        for phase in roadmap.get("phases", [])
        if phase.get("id") in BOUND_PHASES
        for item in phase.get("items", [])
        if isinstance(item, dict)
        for path in _spec_paths(item)
    }
    expected = {
        path.relative_to(root).as_posix()
        for phase in BOUND_PHASES
        for path in (root / "specs" / phase).glob("*.md")
        if not path.name.endswith("-PLAN.md")
    }
    completed = {path for path in expected if _report_path(root, path).is_file()}
    pending = sorted(expected - completed)
    required = {
        path
        for path in completed
        if FOLLOW_UP_SPEC.fullmatch(path) is None and path not in excluded
    }
    errors = [path for path in named if not (root / path).is_file()]
    errors.extend(required - named)
    errors.extend(path for path in excluded if not (root / path).is_file())
    errors.extend(
        path
        for path, reason in exclusions.items()
        if not isinstance(reason, str) or not reason.strip()
    )
    errors.extend(excluded & named)
    acceptable_parents = named | excluded
    for path in expected:
        follow_up = FOLLOW_UP_SPEC.fullmatch(path)
        if follow_up is None:
            continue
        parents = {
            candidate
            for candidate in expected
            if (base := BASE_SPEC.fullmatch(candidate)) is not None
            and base["phase"] == follow_up["phase"]
            and base["id"] == follow_up["id"]
        }
        if not parents or parents.isdisjoint(acceptable_parents):
            errors.append(path)
    return sorted(set(errors)), pending


def _binding_errors(
    roadmap: dict,
    root: Path,
    unbound_by_design: dict[str, str] | None = None,
) -> list[str]:
    return _binding_status(roadmap, root, unbound_by_design)[0]


@check("roadmap_data.spec_bindings")
def spec_bindings() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = root / "specs" / "18"
        directory.mkdir(parents=True)
        parent = directory / "18a-parent.md"
        follow_up = directory / "18aF-review-finding.md"
        parent.write_text("parent", encoding="utf-8")
        follow_up.write_text("finding", encoding="utf-8")
        fixture = {
            "phases": [
                {
                    "id": "18",
                    "items": [
                        {"title": "parent", "spec": "specs/18/18a-parent.md"}
                    ],
                }
            ]
        }
        if _binding_errors(fixture, root):
            fail("an unbound follow-up with a bound parent was rejected")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = root / "specs" / "18"
        directory.mkdir(parents=True)
        base_path = "specs/18/18a-hygiene.md"
        follow_up_path = "specs/18/18aF-review-finding.md"
        nested_follow_up_path = "specs/18/18aF2-second-review-finding.md"
        for path in (base_path, follow_up_path, nested_follow_up_path):
            (root / path).write_text("spec", encoding="utf-8")
        fixture = {"phases": [{"id": "18", "items": []}]}
        fixture_errors = _binding_errors(
            fixture,
            root,
            {base_path: "test hygiene"},
        )
        if fixture_errors:
            fail(
                "follow-ups did not inherit an exempted base parent's standing: "
                f"{fixture_errors!r}"
            )

    exact_errors = _unbound_set_errors(UNBOUND_BY_DESIGN)
    if exact_errors:
        fail(
            "UNBOUND_BY_DESIGN does not match the approved set: "
            f"{exact_errors!r}"
        )

    extra_exclusion = dict(UNBOUND_BY_DESIGN)
    extra_exclusion["specs/18/18z-extra.md"] = "not approved"
    if not _unbound_set_errors(extra_exclusion):
        fail("UNBOUND_BY_DESIGN exact-set check accepted an extra entry")

    roadmap = _load(REPO_ROOT / "docs" / "roadmap.json")
    errors, pending = _binding_status(roadmap, REPO_ROOT, UNBOUND_BY_DESIGN)
    if errors:
        fail(f"roadmap spec bindings are incomplete: {errors!r}")

    missing = copy.deepcopy(roadmap)
    phase = next(item for item in missing["phases"] if item["id"] == "18")
    phase["items"].append(
        {"title": "missing", "spec": "specs/18/18z-does-not-exist.md"}
    )
    missing_errors = _binding_errors(missing, REPO_ROOT, UNBOUND_BY_DESIGN)
    if "specs/18/18z-does-not-exist.md" not in missing_errors:
        fail("a roadmap binding to a missing spec was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = root / "specs" / "18"
        directory.mkdir(parents=True)
        bound = directory / "18a-bound.md"
        unbound = directory / "18b-unbound.md"
        bound.write_text("bound", encoding="utf-8")
        unbound.write_text("unbound", encoding="utf-8")
        report_directory = root / "specs" / "report" / "18"
        report_directory.mkdir(parents=True)
        (report_directory / "18b-unbound-report.md").write_text(
            "report",
            encoding="utf-8",
        )
        fixture = {
            "phases": [
                {
                    "id": "18",
                    "items": [
                        {"title": "bound", "spec": "specs/18/18a-bound.md"}
                    ],
                }
            ]
        }
        fixture_errors = _binding_errors(fixture, root)
        if "specs/18/18b-unbound.md" not in fixture_errors:
            fail("a spec with no roadmap item was not reported")

        stale_path = "specs/18/18z-missing.md"
        if stale_path not in _binding_errors(
            fixture,
            root,
            {stale_path: "no longer present"},
        ):
            fail("an exclusion for a missing spec was accepted")

        empty_reason = directory / "18c-hygiene.md"
        empty_reason.write_text("hygiene", encoding="utf-8")
        empty_reason_path = "specs/18/18c-hygiene.md"
        if empty_reason_path not in _binding_errors(
            fixture,
            root,
            {empty_reason_path: ""},
        ):
            fail("an exclusion with an empty reason was accepted")

        also_bound_path = "specs/18/18a-bound.md"
        if also_bound_path not in _binding_errors(
            fixture,
            root,
            {also_bound_path: "test hygiene"},
        ):
            fail("a spec that was excluded and bound was accepted")

    for expected_pending, completed_names in [
        (0, {"18a-first", "18b-second"}),
        (1, {"18a-first"}),
        (2, set()),
    ]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "specs" / "18"
            directory.mkdir(parents=True)
            report_directory = root / "specs" / "report" / "18"
            report_directory.mkdir(parents=True)
            paths = ["specs/18/18a-first.md", "specs/18/18b-second.md"]
            for path in paths:
                (root / path).write_text("spec", encoding="utf-8")
                if Path(path).stem in completed_names:
                    (report_directory / f"{Path(path).stem}-report.md").write_text(
                        "report",
                        encoding="utf-8",
                    )
            fixture = {
                "phases": [
                    {
                        "id": "18",
                        "items": [
                            {"title": "completed", "spec": path}
                            for path in paths
                            if Path(path).stem in completed_names
                        ],
                    }
                ]
            }
            fixture_errors, fixture_pending = _binding_status(fixture, root)
            if fixture_errors:
                fail(
                    f"the {expected_pending}-pending fixture failed: "
                    f"{fixture_errors!r}"
                )
            if len(fixture_pending) != expected_pending:
                fail(
                    f"expected {expected_pending} pending specs, got "
                    f"{fixture_pending!r}"
                )

    for report_exists in (False, True):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_path = "specs/18/18z-missing.md"
            if report_exists:
                report = root / "specs" / "report" / "18" / "18z-missing-report.md"
                report.parent.mkdir(parents=True)
                report.write_text("report", encoding="utf-8")
            fixture = {
                "phases": [
                    {
                        "id": "18",
                        "items": [{"title": "missing", "spec": missing_path}],
                    }
                ]
            }
            if missing_path not in _binding_errors(fixture, root):
                fail(
                    "a missing bound spec was accepted with "
                    f"report_exists={report_exists}"
                )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = root / "specs" / "18" / "18z-PLAN.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("plan", encoding="utf-8")
        fixture = {"phases": [{"id": "18", "items": []}]}
        plan_errors, plan_pending = _binding_status(fixture, root)
        if plan_errors or plan_pending:
            fail("a PLAN file was treated as a roadmap spec")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = root / "specs" / "18"
        directory.mkdir(parents=True)
        orphan = directory / "18aF-review-finding.md"
        orphan.write_text("finding", encoding="utf-8")
        fixture = {"phases": [{"id": "18", "items": []}]}
        orphan_path = "specs/18/18aF-review-finding.md"
        if orphan_path not in _binding_errors(fixture, root):
            fail("a follow-up whose parent is missing was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = root / "specs" / "18"
        directory.mkdir(parents=True)
        parent = directory / "18a-parent.md"
        follow_up = directory / "18aF-review-finding.md"
        parent.write_text("parent", encoding="utf-8")
        follow_up.write_text("finding", encoding="utf-8")
        fixture = {"phases": [{"id": "18", "items": []}]}
        follow_up_path = "specs/18/18aF-review-finding.md"
        if follow_up_path not in _binding_errors(fixture, root):
            fail("a follow-up whose parent is unbound was accepted")

    ok(f"{len(pending)} roadmap specs pending: {pending!r}")
