"""Harbor-installed SymphonAI agent for Terminal-Bench."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from harbor.agents.agent_auth_resolution import ModelConnectionSpec
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext, ModelUsage

from scripts.bench import (
    populate_agent_context,
    read_usage_files,
    usage_summary,
    write_user_config,
)
from symphonai_api.cost import load_price_table


REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ITEMS = (
    "pyproject.toml",
    "symphonai_api",
    "symphonai_host",
    "symphonai_app",
    "symphonai_tui",
)


def _source_archive() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="symphonai-source-", suffix=".tar.gz", delete=False)
    archive = Path(handle.name)
    handle.close()
    with tarfile.open(archive, "w:gz") as stream:
        for item in _SOURCE_ITEMS:
            source = REPO_ROOT / item
            if source.is_dir():
                for path in source.rglob("*"):
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                        stream.add(path, arcname=path.relative_to(REPO_ROOT))
            elif source.is_file():
                stream.add(source, arcname=item)
    return archive


class SymphonaiAgent(BaseInstalledAgent):
    """Run the checked-out SymphonAI CLI inside each Harbor task environment."""

    MODEL_CONNECTION = ModelConnectionSpec()

    def __init__(
        self,
        *args: Any,
        roles_path: str | None = None,
        effort: str | None = None,
        prices_path: str | None = None,
        smoke: str | bool = False,
        **kwargs: Any,
    ) -> None:
        self.roles_path = Path(roles_path) if roles_path else None
        self.effort = effort or None
        self.prices_path = Path(prices_path) if prices_path else None
        self.smoke = smoke is True or str(smoke).casefold() == "true"
        self.roles: dict[str, dict[str, str]] = {}
        if self.roles_path:
            self.roles = json.loads(self.roles_path.read_text(encoding="utf-8"))
        self._doctor: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "symphonai"

    async def install(self, environment: BaseEnvironment) -> None:
        before = await self._task_python(environment)
        await self.ensure_system_dependencies(environment, ("curl",))
        archive = _source_archive()
        try:
            await environment.upload_file(archive, "/tmp/symphonai-source.tar.gz")
        finally:
            archive.unlink(missing_ok=True)
        result = await self.exec_as_root(
            environment,
            command=(
                "set -eu; mkdir -p /tmp/symphonai-source; "
                "tar -xzf /tmp/symphonai-source.tar.gz -C /tmp/symphonai-source; "
                "rm -f /tmp/symphonai-source.tar.gz; "
                "python_bin=; install_mode=; "
                "for candidate in python3.14 python3.13 python3.12 python3.11 python3; do "
                "if command -v \"$candidate\" >/dev/null 2>&1 && "
                "\"$candidate\" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' "
                " >/dev/null 2>&1 && \"$candidate\" -m venv /tmp/symphonai-venv "
                " >/dev/null 2>&1; then python_bin=$(command -v \"$candidate\"); install_mode=pip; break; "
                "elif command -v \"$candidate\" >/dev/null 2>&1 && "
                "\"$candidate\" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' "
                " >/dev/null 2>&1 && \"$candidate\" -m venv --without-pip /tmp/symphonai-venv "
                " >/dev/null 2>&1; then python_bin=$(command -v \"$candidate\"); install_mode=copy; break; "
                "else rm -rf /tmp/symphonai-venv; fi; done; "
                "if [ -z \"$python_bin\" ]; then "
                "export UV_INSTALL_DIR=/tmp/symphonai-uv; "
                "export UV_PYTHON_INSTALL_DIR=/tmp/symphonai-python; "
                "export UV_PYTHON_BIN_DIR=/tmp/symphonai-python/bin; "
                "export PATH=\"$UV_INSTALL_DIR:$PATH\"; "
                "curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh; "
                "sh /tmp/uv-install.sh >/tmp/uv-install.log 2>&1; rm -f /tmp/uv-install.sh; "
                "/tmp/symphonai-uv/uv python install 3.11 >/tmp/uv-python-install.log 2>&1; "
                "/tmp/symphonai-uv/uv venv --python 3.11 /tmp/symphonai-venv; "
                "install_mode=uv; fi; "
                "if [ \"$install_mode\" = copy ]; then "
                "site_packages=$(/tmp/symphonai-venv/bin/python -c "
                "'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])'); "
                "mkdir -p \"$site_packages\"; "
                "cp -R /tmp/symphonai-source/symphonai_api /tmp/symphonai-source/symphonai_host "
                "/tmp/symphonai-source/symphonai_tui /tmp/symphonai-source/symphonai_app \"$site_packages/\"; "
                "cat > /tmp/symphonai-venv/bin/symphonai <<'PY'\n"
                "#!/tmp/symphonai-venv/bin/python\n"
                "from symphonai_host.headless import main\n"
                "raise SystemExit(main())\n"
                "PY\nchmod +x /tmp/symphonai-venv/bin/symphonai; "
                "elif [ \"$install_mode\" = uv ]; then "
                "/tmp/symphonai-uv/uv pip install --python /tmp/symphonai-venv/bin/python "
                "--no-progress /tmp/symphonai-source; "
                "chmod -R a+rX /tmp/symphonai-uv /tmp/symphonai-python /tmp/symphonai-venv; "
                "else /tmp/symphonai-venv/bin/python -m pip install "
                "--disable-pip-version-check --no-input /tmp/symphonai-source; fi"
            ),
            timeout_sec=900,
        )
        if result.return_code:
            raise RuntimeError(f"SymphonAI install failed: {result.stderr[-3000:]}")

        after = await self._task_python(environment)
        if before != after:
            raise RuntimeError(f"task python changed during install: {before!r} -> {after!r}")
        if self.smoke:
            keys = {
                key: value for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
                if (value := os.environ.get(key))
            }
            key_check = await self._exec_with_env(
                environment,
                "test -n \"$ANTHROPIC_API_KEY\" && test -n \"$OPENAI_API_KEY\"",
                env=keys,
                timeout_sec=15,
            )
            if key_check.return_code:
                raise RuntimeError("smoke credentials did not reach the task container")
            doctor = await self._exec_with_env(
                environment,
                command="/tmp/symphonai-venv/bin/symphonai doctor --json",
                env={},
                timeout_sec=120,
            )
            try:
                doctor_data = json.loads(doctor.stdout)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(f"symphonai doctor did not return JSON: {exc}") from None
            self._doctor = {
                "result": doctor_data, "return_code": doctor.return_code,
                "task_python": before, "task_python_before": before,
                "task_python_after": after, "host_keys_reached_container": True,
            }
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / "symphonai-smoke.json").write_text(
                json.dumps(self._doctor, indent=2), encoding="utf-8",
            )

    async def _task_python(self, environment: BaseEnvironment) -> str:
        result = await self.exec_as_agent(
            environment,
            command=(
                "if command -v python3 >/dev/null 2>&1; then "
                "command -v python3; python3 --version; "
                "else printf 'absent'; fi"
            ),
            timeout_sec=15,
        )
        if result.return_code:
            raise RuntimeError(f"could not check task python3: {result.stderr[-1000:]}")
        return result.stdout.strip()

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext,
    ) -> None:
        task_root_result = await self.exec_as_agent(environment, command="pwd", timeout_sec=15)
        if task_root_result.return_code:
            raise RuntimeError(f"could not determine task root: {task_root_result.stderr[-1000:]}")
        task_root = task_root_result.stdout.strip().splitlines()[-1]
        provider, model = self._provider_model()
        sessions = "/tmp/symphonai-sessions"
        remote_config = "/tmp/symphonai-user-config.toml"
        with tempfile.TemporaryDirectory(prefix="symphonai-bench-config-") as temporary:
            config_path = write_user_config(Path(temporary), self.roles)
            await environment.upload_file(config_path, remote_config)
        command_parts = [
            "/tmp/symphonai-venv/bin/symphonai", "run", instruction,
            "--repo-root", task_root, "--provider", provider, "--model", model,
            "--mode", "allow", "--json",
        ]
        if self.effort:
            command_parts.extend(("--effort", self.effort))
        command = " ".join(shlex.quote(part) for part in command_parts)
        api_env = {
            key: value for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
            if (value := os.environ.get(key))
        }
        env = {
            "SYMPHONAI_HOME": "/tmp/symphonai-home/.symphonai",
            "SYMPHONAI_SESSIONS_DIR": sessions,
            "SYMPHONAI_MEMORY_DIR": "/tmp/symphonai-memory",
            "PYTHONDONTWRITEBYTECODE": "1",
            **api_env,
        }
        await self.exec_as_agent(
            environment,
            command=(
                "mkdir -p /tmp/symphonai-home/.symphonai; "
                "cp /tmp/symphonai-user-config.toml /tmp/symphonai-home/.symphonai/config.toml"
            ),
            timeout_sec=15,
        )
        run_error: BaseException | None = None
        try:
            result = await self._exec_with_env(
                environment, command, cwd=task_root, env=env, timeout_sec=28_800,
            )
            if result.return_code:
                run_error = RuntimeError(f"symphonai run failed: {result.stderr[-3000:]}")
        except asyncio.CancelledError as exc:
            run_error = exc
        except Exception as exc:
            run_error = exc

        sessions_dir = self.logs_dir / "sessions"
        collection_error: Exception | None = None
        try:
            sessions_dir.mkdir(parents=True, exist_ok=True)
            await environment.download_dir(sessions, sessions_dir)
            price_table = load_price_table(self.prices_path) if self.prices_path else None
            summary = usage_summary(read_usage_files(sessions_dir), price_table)
            populate_agent_context(context, summary, price_table)
            context.model_usage = {
                name: ModelUsage(**values) for name, values in context.model_usage.items()
            }
            context.metadata = dict(context.metadata or {})
            context.metadata["symphonai_usage"] = summary
        except Exception as exc:
            collection_error = exc
            message = f"SymphonAI session and usage collection failed: {type(exc).__name__}: {exc}"
            try:
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                (self.logs_dir / "symphonai-collection-error.txt").write_text(
                    message + "\n", encoding="utf-8",
                )
            except OSError:
                print(message, file=sys.stderr)

        if isinstance(run_error, asyncio.CancelledError):
            raise run_error
        if run_error is not None:
            raise run_error
        if collection_error is not None:
            raise RuntimeError(
                f"symphonai session and usage collection failed: {collection_error}"
            ) from collection_error

    def _provider_model(self) -> tuple[str, str]:
        if not self.model_name or "/" not in self.model_name:
            raise RuntimeError("Harbor model must use provider/model form")
        provider, model = self.model_name.split("/", 1)
        return provider, model

    async def _exec_with_env(
        self,
        environment: BaseEnvironment,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str],
        timeout_sec: int,
    ) -> Any:
        """Execute with secrets in Docker's API body, never in docker CLI args."""
        return await asyncio.to_thread(
            self._docker_exec_with_env, environment, command, cwd, env, timeout_sec,
        )

    @staticmethod
    def _docker_exec_with_env(
        environment: BaseEnvironment,
        command: str,
        cwd: str | None,
        env: dict[str, str],
        timeout_sec: int,
    ) -> Any:
        import docker

        project = re.sub(r"[^a-z0-9_-]", "-", environment.session_id.lower())
        if not re.match(r"^[a-z0-9]", project):
            project = "0" + project
        client = docker.from_env(timeout=timeout_sec + 60)
        try:
            containers = client.containers.list(filters={
                "label": [
                    f"com.docker.compose.project={project}",
                    "com.docker.compose.service=main",
                ],
            })
            if len(containers) != 1:
                raise RuntimeError(f"expected one task container for {project}, found {len(containers)}")
            options: dict[str, Any] = {
                "environment": env,
                "workdir": cwd,
                "demux": True,
            }
            if environment.default_user is not None:
                options["user"] = environment.default_user
            result = containers[0].exec_run(["/bin/sh", "-lc", command], **options)
            stdout, stderr = result.output if isinstance(result.output, tuple) else (result.output, b"")
            return_code = result.exit_code
            return type("ExecResult", (), {
                "stdout": (stdout or b"").decode(errors="replace"),
                "stderr": (stderr or b"").decode(errors="replace"),
                "return_code": return_code,
            })()
        finally:
            client.close()
