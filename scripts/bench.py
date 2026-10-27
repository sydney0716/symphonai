#!/usr/bin/env python3
"""Run the Terminal-Bench pilot through Harbor without a SymphonAI-specific harness."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.bench import read_usage_files, usage_table_data  # noqa: E402
from symphonai_api.cost import load_price_table  # noqa: E402
from symphonai_api.paths import symphonai_home  # noqa: E402


DATASET = "terminal-bench/terminal-bench@4.0.0"
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
        })
    return sorted(rows, key=lambda row: row["task"])


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
        if not isinstance(details.get("result"), dict) or not details.get("host_keys_reached_container"):
            continue
        rows.append({"task": task, "doctor": f"JSON (exit {details.get('return_code')})",
                     "task_python_unchanged": True,
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


def _usage_from_trials(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered = []
    for row in rows:
        context = row["result"].get("agent_result", {})
        rendered_row = {
            "task": row["task"],
            "resolved": row["resolved"],
            "wall_seconds": row["wall_seconds"],
            "model_usage": context.get("model_usage"),
            "metadata": context.get("metadata"),
        }
        if "usage" in row:
            rendered_row["usage"] = row["usage"]
        rendered.append(rendered_row)
    return rendered


def _print_table(rows: list[dict[str, Any]], *, smoke: bool = False, prices: bool = False) -> None:
    if smoke:
        print("task | doctor | task interpreter unchanged")
        for row in rows:
            print(f"{row['task']} | {row['doctor']} | {row['task_python_unchanged']}")
        return
    header = "task | resolved | wall seconds | tokens per model | tokens per agent | max request"
    print(header + (" | cost" if prices else ""))
    for row in rows:
        if "result" not in row:
            print(f"{row['task']} | {row.get('resolved')} | {row.get('wall_seconds')} | - | - | -")
            continue
        agent_result = row["result"].get("agent_result", {})
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
                f"{table_data.get('largest_single_request', '-')}")
        if prices:
            line += f" | {json.dumps(table_data.get('cost') or {}, sort_keys=True)}"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--oracle", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    parser.add_argument("--agent", choices=("symphonai", "claude-code", "codex"), default="symphonai")
    parser.add_argument("--model", help="Harbor provider/model string")
    parser.add_argument("--role", action="append", default=[], metavar="NAME=PROVIDER:MODEL")
    parser.add_argument("--effort")
    parser.add_argument("--prices")
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--out")
    arguments = parser.parse_args()

    if not arguments.oracle and not arguments.smoke and not arguments.model:
        parser.error("--model is required for billed agent runs")
    if arguments.agent != "symphonai" and arguments.role:
        parser.error("--role applies only to --agent symphonai")
    if arguments.n_concurrent < 1:
        parser.error("--n-concurrent must be at least 1")
    try:
        task_names = _tasks()
        roles = _roles(arguments)
        if arguments.model:
            _parse_model(arguments.model)
        if arguments.prices:
            load_price_table(arguments.prices)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    output = _new_output(arguments.out)
    roles_path = None
    if arguments.agent == "symphonai" and not arguments.oracle:
        roles_path = output / "roles.json"
        roles_path.write_text(json.dumps(roles, indent=2), encoding="utf-8")
    if not arguments.oracle and not arguments.smoke:
        models = {name: value for name, value in roles.items()}
        models.setdefault("implementer", {"provider": _parse_model(arguments.model)[0],
                                           "model": _parse_model(arguments.model)[1]})
        print(f"BILLED RUN: {len(task_names)} tasks; agent={arguments.agent}; models per role={json.dumps(models, sort_keys=True)}")

    return_code = _run_harbor(arguments, output, task_names, roles_path)
    if return_code:
        return return_code
    if arguments.smoke:
        rows = _smoke_rows(output)
        key_found = _search_smoke_keys(output)
        (output / "results.json").write_text(json.dumps({"smoke": rows,
            "dummy_key_search_found": key_found}, indent=2), encoding="utf-8")
        _print_table(rows, smoke=True)
        if len(rows) != len(task_names) or key_found:
            print(f"Smoke incomplete: {len(rows)}/{len(task_names)} tasks; dummy key found={key_found}", file=sys.stderr)
            return 1
        print("Dummy key search: clean")
        return 0

    rows = _trial_rows(output)
    if arguments.agent == "symphonai" and not arguments.oracle:
        for row in rows:
            matches = list((output / "harbor").glob(f"*/{row['task']}__*/agent/**/sessions"))
            records = []
            for match in matches:
                records.extend(read_usage_files(match))
            if records:
                from scripts.bench import usage_summary
                prices = load_price_table(arguments.prices) if arguments.prices else None
                row["usage"] = usage_summary(records, prices)
    report_rows = _usage_from_trials(rows)
    if not arguments.prices:
        for result_row in report_rows:
            model_usage = result_row.get("model_usage") or {}
            for values in model_usage.values():
                if isinstance(values, dict):
                    values.pop("cost_usd", None)
    (output / "results.json").write_text(json.dumps({
        "dataset": DATASET, "agent": "oracle" if arguments.oracle else arguments.agent,
        "tasks": report_rows,
    }, indent=2), encoding="utf-8")
    _print_table(rows, prices=bool(arguments.prices))
    if len(rows) != len(task_names):
        print(f"Harbor returned {len(rows)} of {len(task_names)} task results", file=sys.stderr)
        return 1
    if arguments.oracle and not all(row["resolved"] for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
