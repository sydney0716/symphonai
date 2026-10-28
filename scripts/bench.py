#!/usr/bin/env python3
"""Run the Terminal-Bench pilot through Harbor without a SymphonAI-specific harness."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.bench import price_agent_usage, read_usage_files, usage_table_data  # noqa: E402
from symphonai_api.cost import load_price_table  # noqa: E402
from symphonai_api.paths import symphonai_home  # noqa: E402


DATASET = "terminal-bench/terminal-bench-2-1"
AGENT_IMPORT = "scripts.bench.agent:SymphonaiAgent"
TASKS_FILE = REPO_ROOT / "scripts" / "bench" / "terminal_bench_pilot.txt"
SMOKE_ANTHROPIC_KEY = "symphonai-smoke-anthropic-key"
SMOKE_OPENAI_KEY = "symphonai-smoke-openai-key"


def _tasks() -> list[str]:
    tasks = [line.strip() for line in TASKS_FILE.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(tasks) != 5 or len(set(tasks)) != 5:
        raise ValueError("terminal_bench_pilot.txt must contain exactly five unique task names")
    return tasks


def _select_tasks(tasks: list[str], requested: list[str] | None) -> list[str]:
    if not requested:
        return tasks
    pilot_names = {
        name
        for task in tasks
        for name in (task, task.rsplit("/", 1)[-1])
    }
    selected_pilot: set[str] = set()
    selected_other: list[str] = []
    for name in requested:
        if name in pilot_names:
            selected_pilot.add(next(
                task for task in tasks
                if name == task or name == task.rsplit("/", 1)[-1]
            ))
        else:
            full_name = name if name.startswith("terminal-bench/") else f"terminal-bench/{name}"
            if full_name not in selected_other:
                selected_other.append(full_name)
    return [task for task in tasks if task in selected_pilot] + selected_other


def _roles(arguments: argparse.Namespace) -> dict[str, dict[str, str]]:
    roles: dict[str, dict[str, str]] = {}
    for entry in arguments.role:
        role, separator, selection = entry.partition("=")
        provider, colon, model = selection.partition(":")
        if (not separator or not colon or not role or not provider or not model
                or role not in {"planner", "implementer", "reviewer", "worker", "explorer"}):
            raise ValueError(f"invalid role override {entry!r}; expected NAME=PROVIDER:MODEL")
        roles[role] = {"provider": provider, "model": model}
    for value in roles.values():
        if arguments.effort:
            value["effort"] = arguments.effort
    return roles


def _parse_model(value: str) -> tuple[str, str]:
    provider, separator, model = value.partition("/")
    if not separator or not provider or not model:
        raise ValueError("--model must be PROVIDER/MODEL")
    return provider, model


def _new_output(path: str | None) -> Path:
    if path:
        output = Path(path).expanduser().resolve()
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = symphonai_home() / "bench" / stamp
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    print(f"Output: {output}")
    return output


def _harbor_command(
    arguments: argparse.Namespace, output: Path, task_names: list[str],
    roles_path: Path | None,
) -> list[str]:
    command = [
        _harbor_executable(), "run",
        "--dataset", DATASET,
        "--n-concurrent", str(arguments.n_concurrent),
        "--n-attempts", str(arguments.attempts),
        "--jobs-dir", str(output / "harbor"),
        "--yes",
    ]
    for task in task_names:
        command.extend(("--include-task-name", task))
    if arguments.oracle:
        command.extend(("--agent", "oracle"))
    elif arguments.smoke:
        command.extend(("--agent", AGENT_IMPORT, "--model", "openai/smoke",
                        "--ak", f"roles_path={roles_path}", "--ak", "smoke=true",
                        "--install-only"))
    else:
        agent = arguments.agent
        if agent == "symphonai":
            agent = AGENT_IMPORT
        command.extend(("--agent", agent))
        if arguments.model:
            command.extend(("--model", arguments.model))
        if roles_path is not None:
            command.extend(("--ak", f"roles_path={roles_path}"))
        if arguments.effort and arguments.agent == "symphonai":
            command.extend(("--ak", f"effort={arguments.effort}"))
        elif arguments.effort:
            command.extend(("--effort", arguments.effort))
        if arguments.prices and arguments.agent == "symphonai":
            command.extend(("--ak", f"prices_path={Path(arguments.prices).expanduser().resolve()}"))
    return command


def _harbor_executable() -> str:
    sibling = Path(sys.executable).with_name("harbor")
    executable = shutil.which("harbor") or (str(sibling) if sibling.is_file() else None)
    if executable is None:
        raise FileNotFoundError("Harbor is not installed in this Python environment")
    return executable


def _run_harbor(
    arguments: argparse.Namespace, output: Path, task_names: list[str],
    roles_path: Path | None,
) -> int:
    command = _harbor_command(arguments, output, task_names, roles_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    if arguments.oracle:
        environment.pop("ANTHROPIC_API_KEY", None)
        environment.pop("OPENAI_API_KEY", None)
    elif arguments.smoke:
        environment["ANTHROPIC_API_KEY"] = SMOKE_ANTHROPIC_KEY
        environment["OPENAI_API_KEY"] = SMOKE_OPENAI_KEY
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    return completed.returncode


def _resolve_harbor_tasks(
    arguments: argparse.Namespace, output: Path, task_names: list[str],
    roles_path: Path | None,
) -> tuple[list[str] | None, int]:
    command = _harbor_command(arguments, output, task_names, roles_path)
    command.append("--print-config")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    configured = subprocess.run(
        command, cwd=REPO_ROOT, env=environment, text=True,
        capture_output=True, check=False,
    )
    if configured.returncode:
        print(configured.stdout, end="")
        print(configured.stderr, end="", file=sys.stderr)
        return None, configured.returncode

    harbor_executable = Path(_harbor_executable()).resolve()
    harbor_python = harbor_executable.with_name("python")
    if not harbor_python.is_file():
        harbor_python = harbor_executable.with_name("python3")
    if not harbor_python.is_file():
        harbor_python = Path(sys.executable)
    resolver = """import asyncio, json, sys
from harbor.job_plan import JobPlan
from harbor.models.job.config import JobConfig

async def main():
    config = JobConfig.model_validate(json.load(sys.stdin))
    tasks = await JobPlan.resolve_task_configs(config)
    print(json.dumps([task.name for task in tasks]))

try:
    asyncio.run(main())
except Exception as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(1)
"""
    resolved = subprocess.run(
        [str(harbor_python), "-c", resolver], cwd=REPO_ROOT, env=environment,
        input=configured.stdout, text=True, capture_output=True, check=False,
    )
    if resolved.returncode:
        print(resolved.stdout, end="")
        print(resolved.stderr, end="", file=sys.stderr)
        return None, resolved.returncode
    try:
        names = json.loads(resolved.stdout)
    except json.JSONDecodeError:
        print("Harbor returned invalid resolved task names", file=sys.stderr)
        return None, 1
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        print("Harbor returned invalid resolved task names", file=sys.stderr)
        return None, 1
    return names, 0


def _trial_rows(output: Path) -> list[dict[str, Any]]:
    rows = []
    for result_path in output.glob("harbor/*/*/result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        trial_name = result_path.parent.name
        task = str(result.get("task_name", trial_name)).split("/", 1)[-1]
        reward = _reward(result)
        started = result.get("started_at")
        finished = result.get("finished_at")
        rows.append({
            "task": task,
            "resolved": reward == 1.0,
            "reward": reward,
            "wall_seconds": _duration(started, finished),
            "result": result,
            "trial_path": result_path.parent,
            "trial_name": trial_name,
        })
    return sorted(rows, key=lambda row: (row["task"], row["trial_name"]))


def _reward(result: dict[str, Any]) -> float | None:
    for key in ("verifier_result", "verifier", "reward"):
        value = result.get(key)
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, dict):
            reward = value.get("reward")
            if isinstance(reward, (float, int)):
                return float(reward)
            rewards = value.get("rewards")
            if isinstance(rewards, dict) and isinstance(rewards.get("reward"), (float, int)):
                return float(rewards["reward"])
    return None


def _duration(started: Any, finished: Any) -> float | None:
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        return (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
    except ValueError:
        return None


def _smoke_rows(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in output.glob("harbor/*/*/agent/**/symphonai-smoke.json"):
        try:
            details = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        trial_root = next((parent for parent in path.parents if (parent / "result.json").is_file()), None)
        if trial_root is None:
            continue
        task = trial_root.name.split("__", 1)[0]
        if (not isinstance(details.get("result"), dict)
                or not details.get("host_keys_reached_container")):
            continue
        before = details.get("task_python_before")
        after = details.get("task_python_after")
        rows.append({"task": task, "doctor": f"JSON (exit {details.get('return_code')})",
                     "task_python_unchanged": isinstance(before, str) and before == after,
                     "details": details})
    return sorted(rows, key=lambda row: row["task"])


def _search_smoke_keys(output: Path) -> bool:
    needles = (SMOKE_ANTHROPIC_KEY.encode(), SMOKE_OPENAI_KEY.encode())
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if any(needle in data for needle in needles):
            return True
    return False


def _usage_from_trials(
    rows: list[dict[str, Any]], prices: Any = None,
    cache_write_1h_rates: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rendered = []
    for row in rows:
        result = row["result"]
        context = result.get("agent_result")
        if not isinstance(context, dict):
            exception_info = result.get("exception_info")
            exception = None
            if isinstance(exception_info, dict):
                exception = {
                    "type": exception_info.get("exception_type"),
                    "message": exception_info.get("exception_message"),
                }
            rendered.append({
                "task": row["task"], "resolved": False,
                "wall_seconds": row["wall_seconds"], "exception": exception,
            })
            continue
        reported_cost = context.get("cost_usd") if isinstance(context, dict) else None
        model_usage = context.get("model_usage") if isinstance(context, dict) else None
        if isinstance(model_usage, dict):
            model_usage = {
                model: {key: value for key, value in values.items() if key != "cost_usd"}
                if isinstance(values, dict) else values
                for model, values in model_usage.items()
            }
        rendered_row = {
            "task": row["task"],
            "resolved": row["resolved"],
            "wall_seconds": row["wall_seconds"],
            "model_usage": model_usage,
            "metadata": context.get("metadata"),
            "agent_reported_cost": reported_cost,
        }
        if "usage" in row:
            rendered_row["usage"] = row["usage"]
        if prices is not None:
            metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
            symphonai = metadata.get("symphonai_usage") if isinstance(metadata, dict) else None
            rendered_row["cost"] = price_agent_usage(
                context.get("model_usage") if isinstance(context, dict) else None,
                prices,
                symphonai_usage=symphonai,
                trajectory_path=row["trial_path"] / "agent" / "trajectory.json",
                claude_stream_path=row["trial_path"] / "agent" / "claude-code.txt",
                cache_write_1h_rates=cache_write_1h_rates,
            )
            row["cost"] = rendered_row["cost"]
        rendered.append(rendered_row)
    return rendered


def _cache_write_1h_rates(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        return {}
    return {
        model: entry["cache_write_1h"]
        for model, entry in models.items()
        if isinstance(model, str) and isinstance(entry, dict) and "cache_write_1h" in entry
    }


def _print_table(rows: list[dict[str, Any]], *, smoke: bool = False, prices: bool = False) -> None:
    if smoke:
        print("task | doctor | task interpreter unchanged")
        for row in rows:
            print(f"{row['task']} | {row['doctor']} | {row['task_python_unchanged']}")
        return
    header = "task | resolved | wall seconds | tokens per model | tokens per agent | max request | exception"
    print(header + (" | cost" if prices else ""))
    for row in rows:
        if "result" not in row:
            print(f"{row['task']} | {row.get('resolved')} | {row.get('wall_seconds')} | - | - | - | {row.get('exception', '-')}")
            continue
        agent_result = row["result"].get("agent_result", {})
        if not isinstance(agent_result, dict):
            info = row["result"].get("exception_info")
            detail = "-"
            if isinstance(info, dict):
                detail = f"{info.get('exception_type', 'Exception')}: {info.get('exception_message', '')}"
                detail = " ".join(detail.split())
            print(f"{row['task']} | False | {row['wall_seconds']} | - | - | - | {detail}")
            continue
        metadata = agent_result.get("metadata", {})
        usage = row.get("usage") or (
            metadata.get("symphonai_usage", {}) if isinstance(metadata, dict) else {}
        )
        table_data = usage_table_data(usage) if usage else {}
        models = table_data.get("models", agent_result.get("model_usage", {}))
        agents = table_data.get("agents", {})
        line = (f"{row['task']} | {row['resolved']} | {row['wall_seconds']} | "
                f"{json.dumps(models, sort_keys=True)} | "
                f"{json.dumps(agents, sort_keys=True)} | "
                f"{table_data.get('largest_single_request', '-')} | -")
        if prices:
            line += f" | {json.dumps(row.get('cost') or {}, sort_keys=True)}"
        print(line)


def _attempt_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    tasks = {
        task: {"resolved": sum(row["resolved"] is True for row in attempts),
               "attempts": len(attempts),
               "resolved_rate": (sum(row["resolved"] is True for row in attempts) / len(attempts))}
        for task, attempts in sorted(by_task.items())
    }
    total_attempts = sum(value["attempts"] for value in tasks.values())
    resolved = sum(value["resolved"] for value in tasks.values())
    return {"tasks": tasks, "resolved": resolved, "attempts": total_attempts,
            "resolved_rate": resolved / total_attempts if total_attempts else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--oracle", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    parser.add_argument("--agent", choices=("symphonai", "claude-code", "codex"), default="symphonai")
    parser.add_argument("--model", help="Harbor provider/model string")
    parser.add_argument("--role", action="append", default=[], metavar="NAME=PROVIDER:MODEL")
    parser.add_argument("--task", action="append", metavar="NAME",
                        help="run a pilot task by full name or final path component (repeatable)")
    parser.add_argument("--effort")
    parser.add_argument("--prices")
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--out")
    arguments = parser.parse_args()

    if not arguments.oracle and not arguments.smoke and not arguments.model:
        parser.error("--model is required for billed agent runs")
    if arguments.agent != "symphonai" and arguments.role:
        parser.error("--role applies only to --agent symphonai")
    if arguments.n_concurrent < 1:
        parser.error("--n-concurrent must be at least 1")
    if arguments.attempts < 1:
        parser.error("--attempts must be at least 1")
    try:
        task_names = _select_tasks(_tasks(), arguments.task)
        roles = _roles(arguments)
        if arguments.model:
            _parse_model(arguments.model)
        if arguments.prices:
            load_price_table(arguments.prices)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    resolution_output = (
        Path(arguments.out).expanduser().resolve()
        if arguments.out else Path(tempfile.gettempdir()) / "symphonai-bench-resolution"
    )
    resolved_tasks, resolve_code = _resolve_harbor_tasks(
        arguments, resolution_output, task_names, None,
    )
    if resolve_code:
        return resolve_code
    task_names = resolved_tasks or []
    output = _new_output(arguments.out)
    roles_path = None
    if arguments.agent == "symphonai" and not arguments.oracle:
        roles_path = output / "roles.json"
        roles_path.write_text(json.dumps(roles, indent=2), encoding="utf-8")
    if not arguments.oracle and not arguments.smoke:
        models = {name: value for name, value in roles.items()}
        models.setdefault("implementer", {"provider": _parse_model(arguments.model)[0],
                                           "model": _parse_model(arguments.model)[1]})
        print(
            f"BILLED RUN: tasks={', '.join(task_names)}; agent={arguments.agent}; "
            f"models per role={json.dumps(models, sort_keys=True)}"
        )
    return_code = _run_harbor(arguments, output, task_names, roles_path)
    if return_code:
        return return_code
    if arguments.smoke:
        rows = _smoke_rows(output)
        key_found = _search_smoke_keys(output)
        (output / "results.json").write_text(json.dumps({"smoke": rows,
            "dummy_key_search_found": key_found}, indent=2), encoding="utf-8")
        _print_table(rows, smoke=True)
        unchanged = all(row["task_python_unchanged"] for row in rows)
        if len(rows) != len(task_names) or key_found or not unchanged:
            print(f"Smoke incomplete: {len(rows)}/{len(task_names)} tasks; dummy key found={key_found}", file=sys.stderr)
            return 1
        print("Dummy key search: clean")
        return 0

    rows = _trial_rows(output)
    if arguments.agent == "symphonai" and not arguments.oracle:
        for row in rows:
            matches = list(row["trial_path"].glob("agent/**/sessions"))
            records = [record for match in matches for record in read_usage_files(match)]
            if records:
                from scripts.bench import usage_summary
                prices = load_price_table(arguments.prices) if arguments.prices else None
                row["usage"] = usage_summary(records, prices)
    prices = load_price_table(arguments.prices) if arguments.prices else None
    report_rows = _usage_from_trials(rows, prices, _cache_write_1h_rates(arguments.prices))
    report = {
        "dataset": DATASET, "agent": "oracle" if arguments.oracle else arguments.agent,
        "tasks": report_rows,
    }
    if arguments.attempts > 1:
        report["summary"] = _attempt_summary(report_rows)
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_table(rows, prices=bool(arguments.prices))
    if arguments.attempts > 1:
        print(f"Attempts summary: {json.dumps(report['summary'], sort_keys=True)}")
    expected_rows = len(task_names) * arguments.attempts
    if len(rows) != expected_rows:
        print(f"Harbor returned {len(rows)} of {expected_rows} task results", file=sys.stderr)
        return 1
    if arguments.oracle and not all(row["resolved"] for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
