"""Offline checks for the Terminal-Bench adapter."""

from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
import importlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.bench import (
    populate_agent_context,
    price_agent_usage,
    usage_summary,
    usage_table_data,
    verify_user_config,
)
from scripts.checks.harness import check, fail
from symphonai_api.cost import UsageTotals, load_price_table, total_cost


@check("bench.pilot_task_selection")
def pilot_task_selection() -> None:
    runner_spec = importlib.util.spec_from_file_location(
        "terminal_bench_runner", Path(__file__).resolve().parents[1] / "bench.py"
    )
    if runner_spec is None or runner_spec.loader is None:
        fail("could not load the Terminal-Bench runner")
    bench_module = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(bench_module)
    tasks = bench_module._tasks()
    if bench_module._select_tasks(tasks, None) != tasks:
        fail("no --task selection did not keep the five pilot tasks")
    selected = bench_module._select_tasks(tasks, ["regex-log"])
    if selected != ["terminal-bench/regex-log"]:
        fail(f"short task name selected the wrong pilot task: {selected!r}")
    if bench_module._select_tasks(tasks, ["terminal-bench/regex-log"]) != [
        "terminal-bench/regex-log",
    ]:
        fail("full pilot task name no longer resolves")
    selected = bench_module._select_tasks(
        tasks, ["regex-log", "bn-fit-modify"]
    )
    expected = [
        "terminal-bench/regex-log",
        "terminal-bench/bn-fit-modify",
    ]
    if selected != expected:
        fail(f"pilot task was not followed by the qualified dataset task: {selected!r}")
    selected = bench_module._select_tasks(
        tasks, ["bn-fit-modify", "chess-best-move", "fix-ocaml-gc"]
    )
    if selected != [
        "terminal-bench/chess-best-move",
        "terminal-bench/bn-fit-modify",
        "terminal-bench/fix-ocaml-gc",
    ]:
        fail(f"mixed task names did not preserve pilot-first resolution: {selected!r}")
    unknown_task = bench_module._select_tasks(tasks, ["no-such-task"])
    if unknown_task != [
        "terminal-bench/no-such-task",
    ]:
        fail("an unknown task was rejected before Harbor could resolve it")
    arguments = SimpleNamespace(
        n_concurrent=1, attempts=3, oracle=True, smoke=False, agent="symphonai",
        model=None, effort=None, prices=None,
    )
    with mock.patch.object(bench_module, "_harbor_executable", return_value="harbor"):
        command = bench_module._harbor_command(arguments, Path("/tmp/out"), unknown_task, None)
    if command[command.index("--n-attempts") + 1] != "3":
        fail(f"Harbor attempts option was not forwarded: {command!r}")
    passed_tasks = [command[index + 1] for index, value in enumerate(command[:-1])
                    if value == "--include-task-name"]
    if passed_tasks != ["terminal-bench/no-such-task"]:
        fail(f"normalized task name was not passed to Harbor: {passed_tasks!r}")
    summary = bench_module._attempt_summary([
        {"task": "a", "resolved": True}, {"task": "a", "resolved": False},
        {"task": "b", "resolved": True},
    ])
    if summary["tasks"]["a"]["resolved_rate"] != 0.5 or summary["resolved_rate"] != 2 / 3:
        fail(f"attempt summary resolved rates are incorrect: {summary!r}")


@check("bench.billed_notice_precedes_trial_run")
def billed_notice_precedes_trial_run() -> None:
    runner_spec = importlib.util.spec_from_file_location(
        "terminal_bench_runner_notice", Path(__file__).resolve().parents[1] / "bench.py"
    )
    if runner_spec is None or runner_spec.loader is None:
        fail("could not load the Terminal-Bench runner")
    bench_module = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(bench_module)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "output"
        output.mkdir()
        observed: list[str] = []

        def run_harbor(_arguments, _output, tasks, _roles_path):
            observed.append(captured.getvalue())
            if tasks != ["terminal-bench/resolved-task"]:
                fail(f"trial run did not receive resolved task names: {tasks!r}")
            return 1

        captured = io.StringIO()
        with (
            mock.patch.object(sys, "argv", [
                "bench.py", "--agent", "symphonai", "--model", "anthropic/test-model",
                "--task", "unresolved-alias", "--out", str(output),
            ]),
            mock.patch.object(bench_module, "_new_output", return_value=output),
            mock.patch.object(
                bench_module, "_resolve_harbor_tasks",
                return_value=(["terminal-bench/resolved-task"], 0),
            ),
            mock.patch.object(bench_module, "_run_harbor", side_effect=run_harbor),
            redirect_stdout(captured),
        ):
            result = bench_module.main()
    notice = "BILLED RUN: tasks=terminal-bench/resolved-task"
    if result != 1 or not observed or notice not in observed[0]:
        fail(f"billed notice did not precede Harbor's trial run: {observed!r}")


@check("bench.harbor_usage_cost_is_normalized")
def harbor_usage_cost_is_normalized() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        price_file = root / "prices.json"
        price_file.write_text(json.dumps({"version": 1, "currency": "USD", "models": {
            "model-a": {"input": 2, "output": 4, "cache_read": 1, "cache_write": 3},
        }}), encoding="utf-8")
        prices = load_price_table(price_file)
        trial = root / "trial"
        agent_logs = trial / "agent"
        agent_logs.mkdir(parents=True)
        (agent_logs / "trajectory.json").write_text(json.dumps({
            "final_metrics": {"extra": {"total_cache_creation_input_tokens": 100}},
        }), encoding="utf-8")
        usage = price_agent_usage(
            {"model-a": {"n_input_tokens": 1000, "n_cache_tokens": 200,
                         "n_output_tokens": 100, "cost_usd": 99}},
            prices, trajectory_path=agent_logs / "trajectory.json",
        )
        if usage["models"]["model-a"] != {
            "input": 1000, "uncached_input": 700, "cache_read": 200,
            "cache_write": 100, "cache_write_1h": 0, "output": 100,
        }:
            fail(f"Harbor token counts were not split exactly once: {usage!r}")
        if usage["total"] != "0.0023" or usage["estimated"]:
            fail(f"Harbor token cost was incorrect or marked estimated: {usage!r}")


@check("bench.symphonai_and_agent_reported_cost")
def symphonai_and_agent_reported_cost() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        price_file = root / "prices.json"
        price_file.write_text(json.dumps({"version": 1, "currency": "USD", "models": {
            "claude-opus-5-5": {"input": 4, "output": 20, "cache_read": 0.20,
                                "cache_write": 5, "cache_write_1h": 8},
            "gpt-6-luna": {"input": 0.10, "output": 0.50, "cache_read": 0.01,
                           "cache_write": 0.125, "long_context": {
                               "input": 0.20, "output": 0.75,
                               "cache_read": 0.02, "cache_write": 0.25,
                           }},
            "model-a": {"input": 2, "output": 4, "cache_read": 1, "cache_write": 3},
        }}), encoding="utf-8")
        prices = load_price_table(price_file)
        opus_record = {"type": "usage", "data": {
            "model": "claude-opus-5-5", "input_tokens": 100000,
            "output_tokens": 2000, "calls": 1, "cache_read_tokens": 95000,
            "cache_write_tokens": 4000,
        }}
        opus_usage = UsageTotals(100000, 2000, 1, 95000, 4000)
        opus_cost = total_cost({"claude-opus-5-5": opus_usage}, prices)
        opus = usage_summary([opus_record], prices)
        if (opus["cost"]["total"] != "0.083"
                or opus["largest_single_request"] != 100000
                or opus["context"]["n_input_tokens"] != 100000
                or opus["context"]["n_cache_tokens"] != 95000):
            fail(f"Opus whole-prompt usage was normalized incorrectly: {opus!r}")
        if opus["cost"]["total"] != str(opus_cost):
            fail(f"Opus bench cost differs from symphonai_api.cost: {opus!r}, {opus_cost!r}")
        opus_report_cost = price_agent_usage(None, prices, symphonai_usage=opus)
        if (opus_report_cost["total"] != "0.083"
                or opus_report_cost["models"]["claude-opus-5-5"]["input"] != 100000):
            fail(f"Opus displayed cost or prompt size was incorrect: {opus_report_cost!r}")

        hour_record = {"type": "usage", "data": {
            "model": "claude-opus-5-5", "input_tokens": 10000,
            "output_tokens": 0, "calls": 1, "cache_write_tokens": 10000,
            "cache_write_1h_tokens": 8000,
        }}
        hour_summary = usage_summary([hour_record], prices)
        hour_usage = UsageTotals(
            10000, 0, 1, 0, 10000, cache_write_1h_tokens=8000,
        )
        hour_cost = total_cost({"claude-opus-5-5": hour_usage}, prices)
        if (hour_summary["models"]["claude-opus-5-5"]["cache_write_1h"] != 8000
                or hour_summary["cost"]["total"] != "0.074"
                or hour_summary["cost"]["total"] != str(hour_cost)):
            fail(f"SymphonAI one-hour write count or cost was incorrect: {hour_summary!r}")
        hour_report_cost = price_agent_usage(None, prices, symphonai_usage=hour_summary)
        if (hour_report_cost["models"]["claude-opus-5-5"]["cache_write_1h"] != 8000
                or hour_report_cost["total"] != "0.074"
                or hour_report_cost["estimated"]):
            fail(f"SymphonAI one-hour write report was incorrect: {hour_report_cost!r}")

        luna_record = {"type": "usage", "data": {
            "model": "gpt-6-luna", "input_tokens": 300000,
            "output_tokens": 1000, "calls": 1, "cache_read_tokens": 200000,
            "cache_write_tokens": 50000, "long_context_input_tokens": 300000,
            "long_context_output_tokens": 1000,
            "long_context_cache_read_tokens": 200000,
            "long_context_cache_write_tokens": 50000,
        }}
        luna_usage = UsageTotals(
            300000, 1000, 1, 200000, 50000, 300000, 1000, 200000, 50000,
        )
        luna_cost = total_cost({"gpt-6-luna": luna_usage}, prices)
        luna = usage_summary([luna_record], prices)
        if luna["cost"]["total"] != "0.02725":
            fail(f"Luna long-context cost was incorrect: {luna!r}")
        if luna["cost"]["total"] != str(luna_cost):
            fail(f"Luna bench cost differs from symphonai_api.cost: {luna!r}, {luna_cost!r}")
        luna_report_cost = price_agent_usage(None, prices, symphonai_usage=luna)
        if luna_report_cost["total"] != "0.02725":
            fail(f"Luna report did not use long-context rates: {luna_report_cost!r}")

        context = SimpleNamespace()
        populate_agent_context(context, opus, prices)
        if (context.n_input_tokens, context.n_cache_tokens, context.cost_usd) != (100000, 95000, 0.083):
            fail(f"Opus Harbor context counts or cost were incorrect: {context!r}")
        unknown = price_agent_usage(
            {"unlisted": {"n_input_tokens": 100, "n_cache_tokens": 0,
                           "n_output_tokens": 2, "cost_usd": 0}}, prices,
        )
        if unknown["total"] is not None or unknown["unknown_models"] != ["unlisted"]:
            fail(f"unpriced model did not produce an empty cost: {unknown!r}")

        runner_spec = importlib.util.spec_from_file_location(
            "terminal_bench_runner_cost", Path(__file__).resolve().parents[1] / "bench.py"
        )
        if runner_spec is None or runner_spec.loader is None:
            fail("could not load the Terminal-Bench runner for reported-cost check")
        runner = importlib.util.module_from_spec(runner_spec)
        runner_spec.loader.exec_module(runner)
        agent_logs = root / "agent"
        agent_logs.mkdir()
        (agent_logs / "trajectory.json").write_text(json.dumps({
            "final_metrics": {"extra": {"total_cache_write_input_tokens": 100}},
        }), encoding="utf-8")
        rows = [{"task": "fixture", "resolved": True, "wall_seconds": 1,
                 "trial_path": root, "result": {"agent_result": {
                     "cost_usd": 0.123,
                     "model_usage": {"model-a": {"n_input_tokens": 1000,
                         "n_cache_tokens": 200, "n_output_tokens": 100, "cost_usd": 99}},
                 }}}]
        rendered = runner._usage_from_trials(rows, prices)[0]
        if rendered["agent_reported_cost"] != 0.123:
            fail(f"agent-reported cost was not preserved separately: {rendered!r}")
        if rendered["model_usage"]["model-a"].get("cost_usd") is not None:
            fail(f"Harbor cost leaked into computed model usage: {rendered!r}")
        if rendered["cost"]["total"] != "0.0023":
            fail(f"agent-reported cost affected computed cost: {rendered!r}")


@check("bench.provider_prefixed_model_prices")
def provider_prefixed_model_prices() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        price_file = Path(temporary) / "prices.json"
        price_file.write_text(json.dumps({"version": 1, "currency": "USD", "models": {
            "gpt-6.1-sol": {"input": 2, "cache_read": 0.10, "output": 10},
        }}), encoding="utf-8")
        prices = load_price_table(price_file)
        priced = price_agent_usage({"openai/gpt-6.1-sol": {
            "n_input_tokens": 60347, "n_cache_tokens": 57600,
            "n_output_tokens": 1589,
        }}, prices)
    if priced["total"] != "0.027144" or priced["unknown_models"]:
        fail(f"provider-prefixed model was not priced by its base name: {priced!r}")


@check("bench.results_include_leader_shell_changes")
def results_include_leader_shell_changes() -> None:
    runner_spec = importlib.util.spec_from_file_location(
        "terminal_bench_runner_leader_changes", Path(__file__).resolve().parents[1] / "bench.py"
    )
    if runner_spec is None or runner_spec.loader is None:
        fail("could not load the runner for leader shell changes fixture")
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    with tempfile.TemporaryDirectory() as temporary:
        trial = Path(temporary)
        transcript = trial / "agent" / "sessions" / "run.jsonl"
        transcript.parent.mkdir(parents=True)
        transcript.write_text(json.dumps({
            "type": "leader_shell_changes",
            "data": {"changed_existing_paths": ["src/main.py", "README.md"]},
        }) + "\n", encoding="utf-8")
        rows = [{
            "task": "fixture", "resolved": True, "wall_seconds": 1,
            "trial_path": trial, "result": {"agent_result": {
                "model_usage": {}, "metadata": {},
            }},
        }]
        rendered = runner._usage_from_trials(rows)[0]
    if rendered.get("leader_shell_changes") != ["src/main.py", "README.md"]:
        fail(f"results row omitted recorded leader changes: {rendered!r}")


@check("bench.claude_cache_write_lifetimes")
def claude_cache_write_lifetimes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        price_file = root / "prices.json"
        price_file.write_text(json.dumps({"version": 1, "currency": "USD", "models": {
            "claude-opus-5-5": {"input": 4, "cache_read": 0.20,
                                "cache_write": 5, "cache_write_1h": 8, "output": 20},
        }}), encoding="utf-8")
        stream = root / "claude-code.txt"
        message = {
            "type": "assistant",
            "message": {
                "id": "message-1", "model": "claude-opus-5-5",
                "usage": {"cache_creation": {
                    "ephemeral_5m_input_tokens": 0,
                    "ephemeral_1h_input_tokens": 6641,
                }},
            },
        }
        stream.write_text("\n".join(json.dumps(event) for event in (
            message, message, {**message, "type": "content_block_delta"},
        )), encoding="utf-8")
        prices = load_price_table(price_file)
        usage = {"claude-opus-5-5": {
            "n_input_tokens": 34365, "n_cache_tokens": 27720,
            "n_output_tokens": 1580,
        }}
        rates = {"claude-opus-5-5": "8"}
        priced = price_agent_usage(
            usage, prices, claude_stream_path=stream, cache_write_1h_rates=rates,
        )
        fallback = price_agent_usage(usage, prices, claude_stream_path=stream)
    if priced["total"] != "0.090288" or priced["estimated"]:
        fail(f"one-hour cache writes did not use their rate exactly: {priced!r}")
    buckets = priced["models"]["claude-opus-5-5"]
    if (buckets.get("cache_write_5m") != 0 or buckets.get("cache_write_1h") != 6641
            or buckets.get("uncached_input") != 4):
        fail(f"one-hour cache writes were not kept in separate buckets: {buckets!r}")
    if fallback["total"] != "0.070365" or not fallback["estimated"]:
        fail(f"missing one-hour rate was not reported as an estimate: {fallback!r}")


def _agent_module_with_harbor_stubs():
    packages = (
        "harbor", "harbor.agents", "harbor.agents.installed", "harbor.environments",
        "harbor.models", "harbor.models.agent",
    )
    modules = {name: types.ModuleType(name) for name in packages}
    for module in modules.values():
        module.__path__ = []

    auth = types.ModuleType("harbor.agents.agent_auth_resolution")
    auth.ModelConnectionSpec = type("ModelConnectionSpec", (), {})

    base = types.ModuleType("harbor.agents.installed.base")
    class BaseInstalledAgent:
        pass
    base.BaseInstalledAgent = BaseInstalledAgent
    base.with_prompt_template = lambda function: function

    environment_module = types.ModuleType("harbor.environments.base")
    environment_module.BaseEnvironment = type("BaseEnvironment", (), {})

    context_module = types.ModuleType("harbor.models.agent.context")
    context_module.AgentContext = type("AgentContext", (), {})
    context_module.ModelUsage = type("ModelUsage", (), {
        "__init__": lambda self, **values: self.__dict__.update(values),
    })

    modules.update({
        auth.__name__: auth,
        base.__name__: base,
        environment_module.__name__: environment_module,
        context_module.__name__: context_module,
    })
    with mock.patch.dict(sys.modules, modules):
        sys.modules.pop("scripts.bench.agent", None)
        return importlib.import_module("scripts.bench.agent")


@check("bench.pre_agent_failure_rows")
def pre_agent_failure_rows() -> None:
    runner_spec = importlib.util.spec_from_file_location(
        "terminal_bench_runner_setup_failure", Path(__file__).resolve().parents[1] / "bench.py"
    )
    if runner_spec is None or runner_spec.loader is None:
        fail("could not load the runner for setup-failure fixture")
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        trial = output / "harbor" / "job" / "regex-log__failed"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(json.dumps({
            "task_name": "terminal-bench/regex-log",
            "agent_result": None,
            "verifier_result": None,
            "exception_info": {
                "exception_type": "NonZeroAgentExitCodeError",
                "exception_message": "python3 is unavailable",
            },
            "started_at": "2026-10-28T00:00:00",
            "finished_at": "2026-10-28T00:00:05",
        }), encoding="utf-8")
        rows = runner._trial_rows(output)
        rendered = runner._usage_from_trials(rows)
        if len(rendered) != 1 or rendered[0]["resolved"] is not False:
            fail(f"pre-agent failure did not become an unresolved row: {rendered!r}")
        expected = {
            "type": "NonZeroAgentExitCodeError",
            "message": "python3 is unavailable",
        }
        if rendered[0].get("exception") != expected:
            fail(f"pre-agent exception was not retained: {rendered!r}")
        if any(key in rendered[0] for key in ("model_usage", "usage", "cost", "agent_reported_cost")):
            fail(f"pre-agent failure row contains token or cost fields: {rendered!r}")
        table = io.StringIO()
        with redirect_stdout(table):
            runner._print_table(rows)
        if "NonZeroAgentExitCodeError: python3 is unavailable" not in table.getvalue():
            fail(f"failure table omitted the exception: {table.getvalue()!r}")

    agent_module = _agent_module_with_harbor_stubs()
    agent = object.__new__(agent_module.SymphonaiAgent)
    calls = []

    async def no_python(_environment, command, *, timeout_sec):
        calls.append(command)
        return SimpleNamespace(return_code=0, stdout="absent", stderr="")

    agent.exec_as_agent = no_python
    if asyncio.run(agent._task_python(None)) != "absent":
        fail("task Python probe failed to record the absent state")
    if len(calls) != 1 or "else printf 'absent'" not in calls[0]:
        fail(f"task Python probe did not handle command absence safely: {calls!r}")


@check("bench.agent_python_path")
def agent_python_path() -> None:
    agent_module = _agent_module_with_harbor_stubs()
    base_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    class FakeEnvironment:
        async def upload_file(self, _local, _remote):
            return None

        async def download_dir(self, _remote, local):
            Path(local).mkdir(parents=True, exist_ok=True)

    async def run_with_python_state(root: Path, task_python: str) -> tuple[str, str, dict]:
        agent = object.__new__(agent_module.SymphonaiAgent)
        agent._task_python_before = task_python
        agent.model_name = "openai/test-model"
        agent.roles = {}
        agent.effort = None
        agent.prices_path = None
        agent.logs_dir = root / "logs"
        calls: list[tuple[str, dict]] = []

        async def exec_as_agent(_environment, command, *, timeout_sec):
            if command == "pwd":
                return SimpleNamespace(return_code=0, stdout="/task\n", stderr="")
            if command == "printf '%s' \"$PATH\"":
                return SimpleNamespace(return_code=0, stdout=base_path, stderr="")
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def exec_with_env(_environment, command, **kwargs):
            calls.append((command, kwargs["env"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        agent.exec_as_agent = exec_as_agent
        agent._exec_with_env = exec_with_env
        context = SimpleNamespace(metadata=None, model_usage={})
        await agent.run("instruction", FakeEnvironment(), context)
        return calls[0][1]["PATH"], calls[0][0], context.metadata

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        absent_path, absent_command, absent_metadata = asyncio.run(
            run_with_python_state(root / "absent", "absent")
        )
        present_path, present_command, present_metadata = asyncio.run(
            run_with_python_state(root / "present", "/usr/bin/python3\nPython 3.13.0")
        )
    expected_added = base_path + ":/tmp/symphonai-agent-python"
    if absent_path != expected_added or "export PATH=" not in absent_command:
        fail(f"missing task Python did not append the standalone directory: {absent_path!r}")
    if present_path != base_path:
        fail(f"existing task PATH was changed: {present_path!r}")
    if absent_metadata["symphonai_agent_python"]["added"] is not True:
        fail(f"run metadata omitted the added Python state: {absent_metadata!r}")
    if present_metadata["symphonai_agent_python"]["added"] is not False:
        fail(f"run metadata incorrectly reported an added Python: {present_metadata!r}")


@check("bench.user_config_roundtrips")
def user_config_roundtrips() -> None:
    roles = {
        "planner": {"provider": "anthropic", "model": "planner-model", "effort": "high"},
        "implementer": {"provider": "openai", "model": "coder-model"},
        "reviewer": {"provider": "anthropic", "model": "review-model"},
        "worker": {"provider": "openai", "model": "worker-model"},
        "explorer": {"provider": "openai", "model": "explorer-model"},
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        verify_user_config(root / "home", repo, roles)


@check("bench.usage_populates_context_and_report")
def usage_populates_context_and_report() -> None:
    records = [
        {"type": "usage", "agent_id": "leader", "data": {
            "model": "model-a", "agent_name": "leader", "input_tokens": 10,
            "output_tokens": 2, "calls": 1, "cache_read_tokens": 3,
            "cache_write_tokens": 1,
        }},
        {"type": "usage", "agent_id": "worker-1", "data": {
            "model": "model-a", "agent_name": "worker", "input_tokens": 20,
            "output_tokens": 4, "calls": 2, "cache_read_tokens": 5,
            "cache_write_tokens": 2,
        }},
        {"type": "usage", "agent_id": "explorer-1", "data": {
            "model": "model-b", "agent_name": "explorer", "input_tokens": 7,
            "output_tokens": 3, "calls": 1, "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }},
    ]
    summary = usage_summary(records)
    if "cost" in summary:
        fail("cost appeared without a price file")
    if summary["models"]["model-a"] != {
        "input": 30, "output": 6, "cache_read": 8, "cache_write": 3,
        "cache_write_1h": 0, "calls": 3,
    }:
        fail(f"per-model usage is incorrect: {summary['models']!r}")
    if summary["agents"]["worker"]["calls"] != 2 or summary["agents"]["explorer"]["input"] != 7:
        fail(f"per-agent usage is incorrect: {summary['agents']!r}")
    if summary["largest_single_request"] != 20:
        fail(f"largest single request is incorrect: {summary['largest_single_request']!r}")

    table_data = usage_table_data(summary)
    if (table_data["models"]["model-a"]["input"] != 30
            or table_data["agents"]["worker"]["calls"] != 2
            or table_data["agents"]["explorer"]["input"] != 7):
        fail(f"model and agent table data is incorrect: {table_data!r}")

    context = SimpleNamespace()
    populate_agent_context(context, summary)
    if (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) != (37, 8, 9):
        fail(f"Harbor context totals are incorrect: {context!r}")
    if context.model_usage["model-a"]["n_input_tokens"] != 30 or context.cost_usd is not None:
        fail(f"Harbor model usage or optional cost is incorrect: {context!r}")

    with tempfile.TemporaryDirectory() as temporary:
        price_file = Path(temporary) / "prices.json"
        price_file.write_text(json.dumps({"version": 1, "currency": "USD", "models": {
            "model-a": {"input": 1, "output": 2},
            "model-b": {"input": 1, "output": 2},
        }}), encoding="utf-8")
        priced = usage_summary(records, load_price_table(price_file))
        if "cost" not in priced or priced["cost"]["total"] in (None, "0"):
            fail(f"price file did not add usage costs: {priced!r}")
        context = SimpleNamespace()
        populate_agent_context(context, priced, load_price_table(price_file))
        if context.cost_usd is None or context.cost_usd <= 0:
            fail(f"priced usage was not copied to AgentContext: {context!r}")


@check("bench.failed_and_cancelled_runs_keep_usage")
def failed_and_cancelled_runs_keep_usage() -> None:
    agent_module = _agent_module_with_harbor_stubs()
    usage_record = {
        "type": "usage", "agent_id": "leader", "data": {
            "model": "model-a", "agent_name": "leader", "input_tokens": 12,
            "output_tokens": 3, "calls": 1, "cache_read_tokens": 2,
            "cache_write_tokens": 1,
        },
    }

    class FakeEnvironment:
        def __init__(self):
            self.downloads = []

        async def upload_file(self, _local, _remote):
            return None

        async def download_dir(self, remote, local):
            self.downloads.append((remote, Path(local)))
            destination = Path(local) / "run"
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "usage.jsonl").write_text(
                json.dumps(usage_record) + "\n", encoding="utf-8",
            )

    def make_agent(logs_dir, exec_behavior):
        agent = object.__new__(agent_module.SymphonaiAgent)
        agent.model_name = "openai/model-a"
        agent.roles = {}
        agent.effort = None
        agent.prices_path = None
        agent.logs_dir = Path(logs_dir)

        async def exec_as_agent(_environment, command, *, timeout_sec):
            if command == "pwd":
                return SimpleNamespace(return_code=0, stdout="/task\n", stderr="")
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def exec_with_env(_environment, _command, **_kwargs):
            return await exec_behavior()

        agent.exec_as_agent = exec_as_agent
        agent._exec_with_env = exec_with_env
        return agent

    def check_context(context):
        if (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) != (12, 2, 3):
            fail(f"failed run usage totals were not copied to the agent context: {context!r}")
        if context.model_usage["model-a"].n_input_tokens != 12:
            fail(f"failed run model usage was not copied to the agent context: {context.model_usage!r}")
        if context.metadata["symphonai_usage"]["agents"]["leader"]["calls"] != 1:
            fail(f"failed run usage metadata was not retained: {context.metadata!r}")

    async def failed_exec():
        return SimpleNamespace(return_code=1, stderr="scripted run failure")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        logs_dir = root / "failed-logs"
        environment = FakeEnvironment()
        context = SimpleNamespace(metadata=None, model_usage={})
        agent = make_agent(logs_dir, failed_exec)
        try:
            asyncio.run(agent.run("task", environment, context))
        except RuntimeError as exc:
            if "scripted run failure" not in str(exc):
                fail(f"failed SymphonAI run error was replaced: {exc!r}")
        else:
            fail("non-zero SymphonAI run did not raise")
        if len(environment.downloads) != 1 or not (logs_dir / "sessions" / "run" / "usage.jsonl").is_file():
            fail("failed SymphonAI run sessions were not downloaded")
        check_context(context)

    class DownloadFails(FakeEnvironment):
        async def download_dir(self, _remote, _local):
            raise OSError("scripted download failure")

    with tempfile.TemporaryDirectory() as temporary:
        logs_dir = Path(temporary) / "collection-failure-logs"
        context = SimpleNamespace(metadata=None, model_usage={})
        agent = make_agent(logs_dir, failed_exec)
        try:
            asyncio.run(agent.run("task", DownloadFails(), context))
        except RuntimeError as exc:
            if "scripted run failure" not in str(exc):
                fail(f"session collection failure replaced the run error: {exc!r}")
        else:
            fail("failed run did not preserve its error when collection also failed")
        error_log = logs_dir / "symphonai-collection-error.txt"
        if not error_log.is_file() or "scripted download failure" not in error_log.read_text(encoding="utf-8"):
            fail("session collection failure was not written to trial logs")

    async def cancelled_exec():
        raise asyncio.CancelledError

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        logs_dir = root / "cancelled-logs"
        environment = FakeEnvironment()
        context = SimpleNamespace(metadata=None, model_usage={})
        agent = make_agent(logs_dir, cancelled_exec)
        try:
            asyncio.run(agent.run("task", environment, context))
        except asyncio.CancelledError:
            pass
        else:
            fail("cancelled SymphonAI run did not propagate cancellation")
        if len(environment.downloads) != 1 or not (logs_dir / "sessions" / "run" / "usage.jsonl").is_file():
            fail("cancelled SymphonAI run sessions were not downloaded")
        check_context(context)
