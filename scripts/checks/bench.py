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
    selected = bench_module._select_tasks(tasks, ["regex-log"])
    if selected != ["terminal-bench/regex-log"]:
        fail(f"short task name selected the wrong pilot task: {selected!r}")
    selected = bench_module._select_tasks(
        tasks, ["regex-log", "chess-best-move"]
    )
    expected = [
        "terminal-bench/chess-best-move",
        "terminal-bench/regex-log",
    ]
    if selected != expected:
        fail(f"task selection did not preserve pilot-file order: {selected!r}")
    try:
        bench_module._select_tasks(tasks, ["nope"])
    except ValueError as exc:
        if any(task not in str(exc) for task in tasks):
            fail(f"unknown task error omitted pilot tasks: {exc!r}")
    else:
        fail("unknown pilot task was accepted")
    arguments = SimpleNamespace(
        n_concurrent=1, attempts=3, oracle=True, smoke=False, agent="symphonai",
        model=None, effort=None, prices=None,
    )
    with mock.patch.object(bench_module, "_harbor_executable", return_value="harbor"):
        command = bench_module._harbor_command(arguments, Path("/tmp/out"), selected, None)
    if command[command.index("--n-attempts") + 1] != "3":
        fail(f"Harbor attempts option was not forwarded: {command!r}")
    summary = bench_module._attempt_summary([
        {"task": "a", "resolved": True}, {"task": "a", "resolved": False},
        {"task": "b", "resolved": True},
    ])
    if summary["tasks"]["a"]["resolved_rate"] != 0.5 or summary["resolved_rate"] != 2 / 3:
        fail(f"attempt summary resolved rates are incorrect: {summary!r}")


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
            "cache_write": 100, "output": 100,
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
                                "cache_write": 5},
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
        "input": 30, "output": 6, "cache_read": 8, "cache_write": 3, "calls": 3,
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
