"""Checks for declarative TOML agent files."""

from __future__ import annotations

import ast
import itertools
import json
import tempfile
from decimal import Decimal
from pathlib import Path

import symphonai_api.agent_file as agent_file_module
from symphonai_api.agent_file import (
    AgentFileError,
    load_agent_directory,
    load_agent_file,
)
from symphonai_api.agent_spec import (
    AgentSpec,
    ContextInheritance,
    Effort,
    IOContract,
    Isolation,
    ModelSelector,
)
from symphonai_api.call_class import CallClass
from symphonai_api.cost import ModelPrice, PriceTable
from symphonai_api.identity import SCHEMA_VERSION
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.runner import standard_tool_registry
from scripts.checks.agent_spec import _forbidden_imports
from scripts.checks.harness import check, fail


REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_AGENT_FILE_IMPORTS = {
    "agent_loop",
    "leader",
    "runner",
    "agent_run",
    "child_context",
    "provider_catalog",
    "providers",
}
STANDARD_TOOL_NAMES = (
    "read_file",
    "write_file",
    "edit_file",
    "multi_edit_file",
    "list_files",
    "glob",
    "grep",
    "run_shell",
    "web_fetch",
)


def _write(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def _price_table() -> PriceTable:
    return PriceTable(
        prices={
            "claude-sonnet-5": ModelPrice(
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            )
        },
        currency="USD",
    )


def _expect_error(
    path: Path,
    key: str,
    *,
    repo_root: Path,
    price_table: PriceTable | None = None,
    default_model: ModelSelector | None = None,
) -> str:
    try:
        load_agent_file(
            path,
            repo_root=repo_root,
            price_table=price_table,
            default_model=default_model,
        )
    except AgentFileError as exc:
        message = str(exc)
        if str(path) not in message or key not in message:
            fail(f"agent file error omitted {path!s} or {key!r}: {message!r}")
        return message
    except Exception as exc:
        fail(f"agent file exposed {type(exc).__name__} for {key}: {exc!r}")
    fail(f"agent file accepted invalid {key}")


@check("agent_file.minimal_and_full")
def minimal_and_full() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        repo_root = directory / "repo"
        default_model = ModelSelector("default", "default-model", Effort.LOW)
        minimal = _write(directory, "minimal.toml", 'prompt = "Review."\n')
        loaded = load_agent_file(
            minimal,
            repo_root=repo_root,
            default_model=default_model,
        )
        if (
            loaded.name != "minimal"
            or loaded.prompt != "Review."
            or loaded.model is not default_model
            or loaded.policy_ceiling.repo_root != repo_root.resolve()
            or loaded.tool_names is not None
            or loaded.budget is not None
            or loaded.deadline_seconds is not None
            or loaded.isolation != Isolation()
            or loaded.io != IOContract()
            or loaded.call_class is not CallClass.BACKGROUND
            or loaded.max_depth != 0
            or loaded.schema_version != 1
        ):
            fail(f"minimal agent did not preserve defaults: {loaded!r}")

        _expect_error(minimal, "model", repo_root=repo_root)
        effort_only = _write(
            directory,
            "effort-only.toml",
            'prompt = "Review."\n[model]\neffort = "high"\n',
        )
        effort_loaded = load_agent_file(
            effort_only,
            repo_root=repo_root,
            default_model=default_model,
        )
        if effort_loaded.model != ModelSelector(
            "default",
            "default-model",
            Effort.HIGH,
        ):
            fail("effort-only model did not merge over the default")
        provider_only = _write(
            directory,
            "provider-only.toml",
            'prompt = "Review."\n[model]\nprovider = "anthropic"\n',
        )
        provider_loaded = load_agent_file(provider_only, repo_root=repo_root)
        if provider_loaded.model != ModelSelector("anthropic"):
            fail("provider-only model did not use component defaults")
        missing_provider = _write(
            directory,
            "missing-provider.toml",
            'prompt = "Review."\n[model]\neffort = "high"\n',
        )
        _expect_error(missing_provider, "provider", repo_root=repo_root)

        full = _write(
            directory,
            "reviewer.toml",
            """prompt = '''
You are a code reviewer.

Read the diff before the report.
'''
deadline_seconds = 120.0
call_class = "background"
max_depth = 0

[model]
provider = "anthropic"
model = "claude-sonnet-5"
effort = "high"

[isolation]
inherit = "tail"
inherit_tail = 2
workspace_prefix = "src"

[budget]
max_turns = 12
wall_seconds = 300
max_total_tokens = 200000
max_cost = "1.50"

[policy]
allowed_write_scope = ["src", "tests"]
shell_enabled = true
shell_allowlist = [["git", "status"], ["git", "diff"]]
fetch_enabled = false
mode = "auto"

[io]
output_schema = { type = "object", properties = { verdict = { type = "string" } } }
""",
        )
        prices = _price_table()
        full_loaded = load_agent_file(
            full,
            repo_root=repo_root,
            price_table=prices,
            default_model=default_model,
        )
        if full_loaded.model != ModelSelector(
            "anthropic",
            "claude-sonnet-5",
            Effort.HIGH,
        ):
            fail(f"full model was not loaded: {full_loaded.model!r}")
        without_default = load_agent_file(
            full,
            repo_root=repo_root,
            price_table=prices,
        )
        if without_default.model != full_loaded.model:
            fail("complete model fields unexpectedly depended on default_model")
        if (
            full_loaded.isolation.inherit is not ContextInheritance.TAIL
            or full_loaded.isolation.inherit_tail != 2
            or full_loaded.isolation.workspace_prefix != "src"
        ):
            fail(f"full isolation was not loaded: {full_loaded.isolation!r}")
        budget = full_loaded.budget
        if (
            budget is None
            or budget.max_turns != 12
            or budget.wall_seconds != 300
            or budget.max_total_tokens != 200000
            or budget.max_cost != Decimal("1.50")
            or budget.price_table is not prices
        ):
            fail(f"full budget was not loaded: {budget!r}")
        policy = full_loaded.policy_ceiling
        if (
            policy.allowed_write_scope
            != [(repo_root / "src").resolve(), (repo_root / "tests").resolve()]
            or policy.shell_allowlist != [("git", "status"), ("git", "diff")]
            or not policy.shell_enabled
            or policy.fetch_enabled
            or policy.mode != "auto"
        ):
            fail(f"full policy was not loaded: {policy!r}")
        output_schema = full_loaded.io.output_schema
        if (
            output_schema is None
            or output_schema["properties"]["verdict"]["type"] != "string"
            or full_loaded.deadline_seconds != 120.0
            or full_loaded.call_class is not CallClass.BACKGROUND
            or full_loaded.max_depth != 0
        ):
            fail(f"full top-level or IO values were not loaded: {full_loaded!r}")


@check("agent_file.unknown_and_forbidden_keys")
def unknown_and_forbidden_keys() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        default_model = ModelSelector("fake")
        accepted_positions: list[str] = []
        positions = (None, *agent_file_module._TABLE_KEYS)
        for position in positions:
            header = "" if position is None else f"\n[{position}]"
            path = _write(
                directory,
                f"unknown-{position or 'top'}.toml",
                f'prompt = "Review."{header}\nunknown_key = true\n',
            )
            try:
                load_agent_file(
                    path,
                    repo_root=directory,
                    default_model=default_model,
                )
            except AgentFileError as exc:
                if "unknown_key" not in str(exc):
                    fail(f"unknown key error omitted its key at {position}: {exc!r}")
            else:
                accepted_positions.append(position or "top")
        if accepted_positions:
            fail(f"unknown keys accepted at positions: {accepted_positions!r}")

        named = _write(
            directory,
            "filename.toml",
            'prompt = "Review."\nname = "override"\n',
        )
        _expect_error(
            named,
            "name",
            repo_root=directory,
            default_model=default_model,
        )
        forbidden = (
            ("approval_callback", "[policy]\napproval_callback = \"callback\"\n"),
            ("price_table", "[budget]\nprice_table = \"prices.json\"\n"),
        )
        for key, table in forbidden:
            path = _write(
                directory,
                f"forbidden-{key}.toml",
                f'prompt = "Review."\n{table}',
            )
            message = _expect_error(
                path,
                key,
                repo_root=directory,
                default_model=default_model,
            )
            if "cannot come from" not in message:
                fail(f"forbidden key error did not explain the boundary: {message!r}")


@check("agent_file.money_is_a_string")
def money_is_a_string() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        default_model = ModelSelector("fake")
        prices = _price_table()
        floating = _write(
            directory,
            "floating.toml",
            'prompt = "Review."\n[budget]\nmax_cost = 1.50\n',
        )
        message = _expect_error(
            floating,
            "max_cost",
            repo_root=directory,
            price_table=prices,
            default_model=default_model,
        )
        if "quote" not in message:
            fail(f"floating max_cost error did not say to quote it: {message!r}")
        quoted = _write(
            directory,
            "quoted.toml",
            'prompt = "Review."\n[budget]\nmax_cost = "1.50"\n',
        )
        quoted_loaded = load_agent_file(
            quoted,
            repo_root=directory,
            price_table=prices,
            default_model=default_model,
        )
        if (
            quoted_loaded.budget is None
            or quoted_loaded.budget.max_cost != Decimal("1.50")
            or quoted_loaded.budget.price_table is not prices
        ):
            fail(f"quoted max_cost did not retain its exact value: {quoted_loaded!r}")
        missing_prices = _expect_error(
            quoted,
            "max_cost",
            repo_root=directory,
            default_model=default_model,
        )
        if "price table" not in missing_prices:
            fail(f"missing price table error was not actionable: {missing_prices!r}")
        no_cost = _write(
            directory,
            "no-cost.toml",
            'prompt = "Review."\n[budget]\nmax_turns = 2\n',
        )
        for price_table in (None, prices):
            loaded = load_agent_file(
                no_cost,
                repo_root=directory,
                price_table=price_table,
                default_model=default_model,
            )
            if loaded.budget is None or loaded.budget.max_turns != 2:
                fail("budget without max_cost did not load")


@check("agent_file.validation_errors_name_the_key")
def validation_errors_name_the_key() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        default_model = ModelSelector("fake")
        cases = (
            ("provider", '[model]\nprovider = ""\n'),
            ("inherit_tail", '[isolation]\ninherit = "tail"\ninherit_tail = 0\n'),
            ("inherit_tail", '[isolation]\ninherit = "all"\ninherit_tail = 1\n'),
            ("max_turns", "[budget]\nmax_turns = 0\n"),
            ("max_depth", "max_depth = -1\n"),
            ("output_schema", '[io]\noutput_schema = { type = "string" }\n'),
            ("workspace_prefix", '[isolation]\nworkspace_prefix = "/tmp"\n'),
            ("workspace_prefix", '[isolation]\nworkspace_prefix = ""\n'),
            ("workspace_prefix", '[isolation]\nworkspace_prefix = "a/../b"\n'),
        )
        for index, (key, body) in enumerate(cases):
            path = _write(
                directory,
                f"invalid-{index}.toml",
                f'prompt = "Review."\n{body}',
            )
            _expect_error(
                path,
                key,
                repo_root=directory,
                default_model=default_model,
            )
        malformed = _write(directory, "malformed.toml", 'prompt = ["unterminated"\n')
        _expect_error(
            malformed,
            "toml",
            repo_root=directory,
            default_model=default_model,
        )
        bad_effort = _write(
            directory,
            "bad-effort.toml",
            'prompt = "Review."\n[model]\neffort = "extreme"\n',
        )
        message = _expect_error(
            bad_effort,
            "effort",
            repo_root=directory,
            default_model=default_model,
        )
        for value in ("default", "low", "medium", "high"):
            if value not in message:
                fail(f"effort error omitted valid value {value!r}: {message!r}")


@check("agent_file.directory_roster")
def directory_roster() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        default_model = ModelSelector("fake")
        missing = load_agent_directory(
            directory / "missing",
            repo_root=directory,
            default_model=default_model,
        )
        if missing != {}:
            fail(f"missing directory did not return an empty roster: {missing!r}")
        _write(directory, "alpha.toml", 'prompt = "Alpha"\n')
        _write(directory, "beta.toml", 'prompt = "Beta"\n')
        _write(directory, "ignored.txt", 'prompt = "Ignored"\n')
        nested = directory / "nested"
        nested.mkdir()
        _write(nested, "nested.toml", 'prompt = "Nested"\n')
        roster = load_agent_directory(
            directory,
            repo_root=directory,
            default_model=default_model,
        )
        if set(roster) != {"alpha", "beta"}:
            fail(f"directory roster selected the wrong files: {roster!r}")
        bad = _write(directory, "broken.toml", 'prompt = ["broken"\n')
        try:
            load_agent_directory(
                directory,
                repo_root=directory,
                default_model=default_model,
            )
        except AgentFileError as exc:
            if str(bad) not in str(exc):
                fail(f"directory error omitted bad file: {exc!r}")
        else:
            fail("directory silently dropped a malformed agent file")


@check("agent_file.tool_allow_and_deny")
def tool_allow_and_deny() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        default_model = ModelSelector("fake")
        if tuple(standard_tool_registry()) != STANDARD_TOOL_NAMES:
            fail("test's canonical tool names do not match the runtime registry")

        allow_only = _write(
            directory,
            "allow-only.toml",
            'prompt = "Review."\ntools = ["run_shell", "read_file", "grep"]\n',
        )
        allowed = load_agent_file(
            allow_only,
            repo_root=directory,
            default_model=default_model,
        )
        if allowed.tool_names != ("read_file", "grep", "run_shell"):
            fail(f"allow list was not canonicalized: {allowed.tool_names!r}")

        deny_only = _write(
            directory,
            "deny-only.toml",
            'prompt = "Review."\ndeny_tools = ["run_shell", "write_file"]\n',
        )
        denied = load_agent_file(
            deny_only,
            repo_root=directory,
            default_model=default_model,
        )
        expected_denied = tuple(
            name
            for name in STANDARD_TOOL_NAMES
            if name not in {"run_shell", "write_file"}
        )
        if denied.tool_names is None or denied.tool_names != expected_denied:
            fail(f"deny-only list did not remain explicit: {denied.tool_names!r}")
        if tuple(standard_tool_registry(denied.tool_names)) != expected_denied:
            fail("deny-only tuple widened when handed to the runtime registry")

        both = _write(
            directory,
            "both.toml",
            (
                'prompt = "Review."\n'
                'tools = ["run_shell", "grep", "read_file"]\n'
                'deny_tools = ["run_shell"]\n'
            ),
        )
        narrowed = load_agent_file(
            both,
            repo_root=directory,
            default_model=default_model,
        )
        if narrowed.tool_names != ("read_file", "grep"):
            fail(f"deny did not win over allow: {narrowed.tool_names!r}")

        unknown_cases = (
            ("tools", "unknown_tool"),
            ("deny_tools", "unknown_tool"),
            ("tools", "web_search"),
            ("deny_tools", "web_search"),
            ("tools", "read_tool_result"),
            ("deny_tools", "read_tool_result"),
        )
        for index, (key, unknown) in enumerate(unknown_cases):
            path = _write(
                directory,
                f"unknown-tool-{index}.toml",
                (
                    'prompt = "Review."\n'
                    f'{key} = ["read_file", "{unknown}"]\n'
                ),
            )
            message = _expect_error(
                path,
                key,
                repo_root=directory,
                default_model=default_model,
            )
            if unknown not in message:
                fail(f"unknown tool error omitted {unknown!r}: {message!r}")

        empty = _write(
            directory,
            "empty-tools.toml",
            'prompt = "Review."\ntools = []\n',
        )
        _expect_error(
            empty,
            "tools",
            repo_root=directory,
            default_model=default_model,
        )
        cancelled = _write(
            directory,
            "cancelled-tools.toml",
            (
                'prompt = "Review."\n'
                'tools = ["read_file"]\n'
                'deny_tools = ["read_file"]\n'
            ),
        )
        cancelled_message = _expect_error(
            cancelled,
            "tools and deny_tools",
            repo_root=directory,
            default_model=default_model,
        )
        if "empty" not in cancelled_message:
            fail(f"empty resolved set error was unclear: {cancelled_message!r}")
        try:
            standard_tool_registry(())
        except ValueError:
            pass
        else:
            fail("runtime registry accepted the empty tuple")

        for size in range(3):
            for selected in itertools.combinations(STANDARD_TOOL_NAMES, size):
                for denied_name in STANDARD_TOOL_NAMES:
                    path = _write(
                        directory,
                        "enumerated.toml",
                        (
                            'prompt = "Review."\n'
                            f"tools = {json.dumps(selected)}\n"
                            f'deny_tools = ["{denied_name}"]\n'
                        ),
                    )
                    expected = tuple(
                        name
                        for name in STANDARD_TOOL_NAMES
                        if name in selected and name != denied_name
                    )
                    if not selected or not expected:
                        _expect_error(
                            path,
                            "tools",
                            repo_root=directory,
                            default_model=default_model,
                        )
                        continue
                    loaded = load_agent_file(
                        path,
                        repo_root=directory,
                        default_model=default_model,
                    )
                    if (
                        loaded.tool_names != expected
                        or not set(loaded.tool_names).issubset(STANDARD_TOOL_NAMES)
                        or denied_name in loaded.tool_names
                    ):
                        fail(
                            "enumerated tool resolution was not a canonical narrowing: "
                            f"{selected!r}, {denied_name!r}, {loaded.tool_names!r}"
                        )
                    registry = standard_tool_registry(loaded.tool_names)
                    if tuple(registry) != loaded.tool_names:
                        fail(f"runtime rejected or reordered tool tuple: {loaded.tool_names!r}")


@check("agent_file.effort_leaves_schema_version_alone")
def effort_leaves_schema_version_alone() -> None:
    if [value.value for value in Effort] != ["default", "low", "medium", "high"]:
        fail(f"unexpected effort values: {list(Effort)!r}")
    model = ModelSelector("fake")
    if model.effort is not Effort.DEFAULT:
        fail(f"model effort default changed: {model.effort!r}")
    with tempfile.TemporaryDirectory() as temporary:
        policy = PermissionPolicy(Path(temporary))
        spec = AgentSpec("worker", "Review.", model, policy)
        versions = (
            SCHEMA_VERSION,
            model.schema_version,
            Isolation().schema_version,
            IOContract().schema_version,
            spec.schema_version,
        )
        if versions != (1, 1, 1, 1, 1):
            fail(f"effort changed schema versions: {versions!r}")


def _agent_file_forbidden_imports(source: str) -> list[str]:
    found = _forbidden_imports(source)
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
            if (
                any(part in FORBIDDEN_AGENT_FILE_IMPORTS for part in module.split("."))
                and module not in found
            ):
                found.append(module)
    return found


@check("agent_file.no_runtime_imports")
def no_runtime_imports() -> None:
    source = (REPO_ROOT / "symphonai_api/agent_file.py").read_text(encoding="utf-8")
    found = _agent_file_forbidden_imports(source)
    if found:
        fail(f"agent_file imports runtime wiring: {found!r}")
    probes = (
        "from symphonai_api.agent_loop import ApiAgent\n",
        "from symphonai_api.leader import Leader\n",
        "import symphonai_api.runner\n",
        "from symphonai_api import agent_run\n",
        "from . import child_context\n",
        "from symphonai_api.provider_catalog import PROVIDERS\n",
        "from symphonai_api.providers.fake import FakeModelProvider\n",
    )
    for probe in probes:
        if not _agent_file_forbidden_imports(probe):
            fail(f"import inspection missed {probe.strip()!r}")
    safe = (
        "from symphonai_api.agent_spec import AgentSpec\n",
        "from symphonai_api.budgets import RunBudget\n",
    )
    for probe in safe:
        if _agent_file_forbidden_imports(probe):
            fail(f"import inspection rejected {probe.strip()!r}")
