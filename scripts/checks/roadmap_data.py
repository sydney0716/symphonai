"""Schema and spec-binding checks for the roadmap data."""

from __future__ import annotations

import copy
import json
import re
import tempfile
from pathlib import Path

from scripts.checks.harness import check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]
BOUND_PHASES = ("10", "18", "19")
BASE_SPEC = re.compile(
    r"^specs/(?P<phase>[^/]+)/(?P<id>[0-9]+[A-Za-z])-[^/]+\.md$"
)
FOLLOW_UP_SPEC = re.compile(
    r"^specs/(?P<phase>[^/]+)/(?P<id>[0-9]+[A-Za-z])F[0-9]*-[^/]+\.md$"
)


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


def _binding_errors(roadmap: dict, root: Path) -> list[str]:
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
    required = {path for path in expected if FOLLOW_UP_SPEC.fullmatch(path) is None}
    errors = [path for path in named if not (root / path).is_file()]
    errors.extend(required - named)
    for path in expected - required:
        follow_up = FOLLOW_UP_SPEC.fullmatch(path)
        parents = {
            candidate
            for candidate in expected
            if (base := BASE_SPEC.fullmatch(candidate)) is not None
            and base["phase"] == follow_up["phase"]
            and base["id"] == follow_up["id"]
        }
        if not parents or parents.isdisjoint(named):
            errors.append(path)
    return sorted(set(errors))


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

    roadmap = _load(REPO_ROOT / "docs" / "roadmap.json")
    errors = _binding_errors(roadmap, REPO_ROOT)
    if errors:
        fail(f"roadmap spec bindings are incomplete: {errors!r}")

    missing = copy.deepcopy(roadmap)
    phase = next(item for item in missing["phases"] if item["id"] == "18")
    phase["items"].append(
        {"title": "missing", "spec": "specs/18/18z-does-not-exist.md"}
    )
    missing_errors = _binding_errors(missing, REPO_ROOT)
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
