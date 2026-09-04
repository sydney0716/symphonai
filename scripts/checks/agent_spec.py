"""Checks for declarative agent roles."""
from __future__ import annotations
import ast
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from symphonai_api.agent_spec import AgentSpec, ContextInheritance, IOContract, Isolation, ModelSelector, validate_output
from symphonai_api.permissions import PermissionPolicy
from scripts.checks.harness import check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_IMPORTS = {"agent_loop", "leader", "runner", "provider_catalog", "providers"}


def _forbidden_imports(source: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = ([node.module] if node.module is not None else []) + [
                alias.name for alias in node.names
            ]
        else:
            continue
        for module in modules:
            if any(part in FORBIDDEN_IMPORTS for part in module.split(".")):
                found.append(module)
    return found

def _spec(root: Path, **changes) -> AgentSpec:
    values = {
        "name": "worker",
        "prompt": "",
        "model": ModelSelector("fake"),
        "policy_ceiling": PermissionPolicy(repo_root=root),
    }
    values.update(changes)
    return AgentSpec(**values)

@check("agent_spec.defaults_and_frozen")
def defaults_and_frozen() -> None:
    with tempfile.TemporaryDirectory() as d:
        spec = _spec(Path(d))
        if (
            spec.tool_names is not None
            or spec.budget is not None
            or spec.deadline_seconds is not None
            or spec.isolation != Isolation()
            or spec.io != IOContract()
            or spec.call_class.value != "background"
            or spec.max_depth != 0
            or spec.schema_version != 1
        ):
            fail("documented defaults changed")
        try: spec.name = "other"  # type: ignore[misc]
        except FrozenInstanceError: return
        fail("AgentSpec was mutable")

@check("agent_spec.model_selector")
def model_selector() -> None:
    if ModelSelector("fake").model is not None: fail("None model default changed")
    for value in ("", " "):
        try: ModelSelector(value)
        except ValueError: continue
        fail("empty provider accepted")

@check("agent_spec.isolation_rules")
def isolation_rules() -> None:
    for inherit in ContextInheritance:
        for tail in (-2, -1, 0, 1, 2):
            expected = tail >= 1 if inherit is ContextInheritance.TAIL else tail == 0
            try:
                Isolation(inherit, tail)
            except ValueError as exc:
                if expected:
                    fail(f"valid isolation rejected: {inherit!r}, {tail}")
                if inherit.value not in str(exc) or str(tail) not in str(exc):
                    fail("isolation error omitted values")
            else:
                if not expected:
                    fail(f"invalid isolation accepted: {inherit!r}, {tail}")
    for prefix in ("", "/absolute", "a/../b"):
        try: Isolation(workspace_prefix=prefix)
        except ValueError: continue
        fail("invalid workspace prefix accepted")

@check("agent_spec.io_contract_shape")
def io_contract_shape() -> None:
    for schema in ([], {"type": "string"}):
        try: IOContract(output_schema=schema)  # type: ignore[arg-type]
        except ValueError: continue
        fail("invalid schema accepted")
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    contract = IOContract(output_schema=schema); schema["properties"]["x"]["type"] = "integer"
    if contract.output_schema["properties"]["x"]["type"] != "string": fail("schema was not frozen")

@check("agent_spec.field_validation")
def field_validation() -> None:
    with tempfile.TemporaryDirectory() as d:
        for field, changes in (("name", {"name":""}), ("tool_names", {"tool_names":()}), ("tool_names", {"tool_names":("x","x")}), ("deadline_seconds", {"deadline_seconds":0}), ("max_depth", {"max_depth":-1})):
            try:
                _spec(Path(d), **changes)
            except ValueError as exc:
                if field not in str(exc):
                    fail(f"validation error omitted {field}: {exc!r}")
                continue
            fail(f"invalid field accepted: {changes!r}")

@check("agent_spec.validate_output_passthrough")
def passthrough() -> None:
    if validate_output(IOContract(), " raw ") != (" raw ", None): fail("passthrough changed text")

@check("agent_spec.validate_output_parsing")
def parsing() -> None:
    c=IOContract(output_schema={"type":"object"})
    for text in ('{"x": 1}', ' ```json\n{"x": 1}\n``` ', ' \n{"x": 1}\t'):
        if validate_output(c,text)[1] is not None: fail("valid JSON was rejected")
    if validate_output(c,"not json")[0] is not None: fail("malformed JSON did not return error")

@check("agent_spec.validate_output_subset")
def subset() -> None:
    c=IOContract(output_schema={"type":"object","properties":{"n":{"type":"integer","minLength":9},"color":{"enum":["red","green"]},"tags":{"type":"array","items":{"type":"string"}}},"required":["n"]})
    if validate_output(c,'{"n": 1, "tags": ["x"]}')[1] is not None: fail("subset rejected valid output")
    if validate_output(c, '{"n":1,"color":"red"}')[1] is not None:
        fail("nested enum rejected an allowed value")
    for text in ('{}','{"n": true}','{"n":1,"tags":[2]}','{"n":1,"color":"purple"}'):
        if validate_output(c,text)[0] is not None: fail("subset accepted invalid output")

@check("agent_spec.validate_output_never_raises")
def never_raises() -> None:
    schema={"type":"object","properties":{"a":{"type":"array","items":{"type":"object","properties":{"b":{"type":"array","items":{"type":"string"}}}}}}}
    c=IOContract(output_schema=schema)
    for text in ("", "{", "[]", "null", '{"a":[{"b":["x"]}]}', "true", "42", "not json", '{"a":[{}]}', '{"a":"x"}', '{"a":[{"b":[2]}]}', "{}"):
        try: validate_output(c,text)
        except Exception as exc: fail(f"validate_output raised {exc!r}")

@check("agent_spec.with_overrides")
def overrides() -> None:
    with tempfile.TemporaryDirectory() as d:
        s=_spec(Path(d)); changed=s.with_overrides(max_depth=2)
        if changed is s or s.max_depth or changed.max_depth != 2: fail("override changed original")
        try:
            s.with_overrides(max_depth=-1)
        except ValueError:
            pass
        else:
            fail("override skipped validation")
        try: s.with_overrides(missing=True)
        except TypeError: return
        fail("unknown override accepted")

@check("agent_spec.no_runtime_imports")
def no_runtime_imports() -> None:
    source = (REPO_ROOT / "symphonai_api/agent_spec.py").read_text()
    found = _forbidden_imports(source)
    if found:
        fail(f"agent_spec imports runtime wiring: {found!r}")
    probes = [
        ("from symphonai_api.runner import standard_tool_registry\n", True),
        ("from symphonai_api.leader import Leader\n", True),
        ("from symphonai_api.agent_loop import ApiAgent\n", True),
        ("from symphonai_api.providers.openai import OpenAIProvider\n", True),
        ("import symphonai_api.runner\n", True),
        ("from . import runner\n", True),
        ("from symphonai_api import runner\n", True),
        ("from symphonai_api import leader, budgets\n", True),
        ("import symphonai_api.budgets\n", False),
        ("from symphonai_api.budgets import RunBudget\n", False),
    ]
    for line, should_be_caught in probes:
        caught = bool(_forbidden_imports(line))
        if caught != should_be_caught:
            fail(f"import inspection got {line.strip()!r} wrong")
