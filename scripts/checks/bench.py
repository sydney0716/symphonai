"""Offline checks for the Terminal-Bench adapter."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.bench import (
    populate_agent_context,
    usage_summary,
    usage_table_data,
    verify_user_config,
)
from scripts.checks.harness import check, fail
from symphonai_api.cost import load_price_table


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
    if summary["largest_single_request"] != 27:
        fail(f"largest single request is incorrect: {summary['largest_single_request']!r}")

    table_data = usage_table_data(summary)
    if (table_data["models"]["model-a"]["input"] != 30
            or table_data["agents"]["worker"]["calls"] != 2
            or table_data["agents"]["explorer"]["input"] != 7):
        fail(f"model and agent table data is incorrect: {table_data!r}")

    context = SimpleNamespace()
    populate_agent_context(context, summary)
    if (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) != (48, 11, 9):
        fail(f"Harbor context totals are incorrect: {context!r}")
    if context.model_usage["model-a"]["n_input_tokens"] != 41 or context.cost_usd is not None:
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
        if (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) != (15, 3, 3):
            fail(f"failed run usage totals were not copied to the agent context: {context!r}")
        if context.model_usage["model-a"].n_input_tokens != 15:
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
