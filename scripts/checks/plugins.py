"""Checks for plugin bundles of existing extension formats."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import tempfile

import symphonai_api.plugins as plugins_module
from symphonai_api.agent_file import load_agent_directory
from symphonai_api.agent_spec import ModelSelector
from symphonai_api.config import CapabilityCeiling
from symphonai_api.events import RunFinished
from symphonai_api.hooks import HookRunner, HookSpec, hooks_from_config
from symphonai_api.mcp import McpClient, McpTool, mcp_servers_from_config
from symphonai_api.plugins import (
    PluginError,
    append_plugin_hooks,
    load_plugin,
    load_plugin_directory,
)
from symphonai_api.skills import load_skill_directory
from scripts.checks.agent_spec import _forbidden_imports
from scripts.checks.harness import check, fail


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _manifest(
    name: str,
    *,
    version: str = "1.0.0",
    description: str = "A test plugin.",
) -> str:
    return (
        f"name = {json.dumps(name)}\n"
        f"version = {json.dumps(version)}\n"
        f"description = {json.dumps(description)}\n"
    )


def _plugin(root: Path, name: str) -> Path:
    path = root / name
    _write(path / "plugin.toml", _manifest(name))
    return path


def _agent(path: Path, name: str = "worker", body: str = 'prompt = "Work."\n') -> Path:
    return _write(path / "agents" / f"{name}.toml", body)


def _skill(path: Path, name: str = "guide") -> Path:
    return _write(
        path / "skills" / f"{name}.md",
        (
            "+++\n"
            f"name = {json.dumps(name)}\n"
            'description = "A guide."\n'
            'when_to_use = "When guidance is needed."\n'
            "+++\n"
            "\n# Guide\n"
        ),
    )


def _config(
    path: Path,
    *,
    hook_command: tuple[str, ...] | None = None,
    server_name: str = "docs",
) -> Path:
    command = hook_command or ("observe",)
    return _write(
        path / "config.toml",
        (
            "[[hooks]]\n"
            'on = ["RunFinished"]\n'
            f"command = {json.dumps(command)}\n"
            "[[mcp.servers]]\n"
            f"name = {json.dumps(server_name)}\n"
            'command = ["fake-server", "--stdio"]\n'
        ),
    )


def _expect_error(
    path: Path,
    repo_root: Path,
    fragments: tuple[str, ...],
    **loader_kwargs,
) -> str:
    try:
        load_plugin(path, repo_root=repo_root, **loader_kwargs)
    except PluginError as exc:
        message = str(exc)
        if not all(fragment in message for fragment in fragments):
            fail(f"plugin error omitted {fragments!r}: {message!r}")
        return message
    except Exception as exc:
        fail(f"plugin exposed {type(exc).__name__}: {exc!r}")
    fail(f"plugin {path.name!r} was accepted")


@check("plugins.all_members_use_existing_loaders")
def all_members_use_existing_loaders() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo_root = root / "repo"
        path = _plugin(root, "release_tools")
        _agent(path, "releaser")
        _skill(path, "checklist")
        _config(path)
        default_model = ModelSelector("fake")

        plugin = load_plugin(
            path,
            repo_root=repo_root,
            default_model=default_model,
        )
        direct_agents = load_agent_directory(
            path / "agents",
            repo_root=repo_root,
            default_model=default_model,
        )
        direct_skills = load_skill_directory(path / "skills")
        direct_config = plugins_module._member_config(path)
        direct_hooks = hooks_from_config(direct_config, repo_root=repo_root)
        direct_servers = mcp_servers_from_config(
            direct_config,
            repo_root=repo_root,
        )
        expected_servers = tuple(
            replace(server, name=f"release_tools__{server.name}")
            for server in direct_servers
        )
        if plugin.agents != {"release_tools/releaser": direct_agents["releaser"]}:
            fail(f"plugin agent differed from agent_file loader: {plugin.agents!r}")
        if plugin.skills != {"release_tools/checklist": direct_skills["checklist"]}:
            fail(f"plugin skill differed from skills loader: {plugin.skills!r}")
        if plugin.hooks != direct_hooks:
            fail(f"plugin hooks differed from hooks loader: {plugin.hooks!r}")
        if plugin.mcp_servers != expected_servers:
            fail(f"plugin MCP servers differed from MCP loader: {plugin.mcp_servers!r}")
        if (
            plugin.name,
            plugin.version,
            plugin.description,
            plugin.path,
        ) != ("release_tools", "1.0.0", "A test plugin.", path):
            fail(f"plugin manifest or path was not retained: {plugin!r}")


@check("plugins.optional_members")
def optional_members() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo_root = root / "repo"
        skills_only = _plugin(root, "skills_only")
        _skill(skills_only)
        agents_only = _plugin(root, "agents_only")
        _agent(agents_only)
        empty = _plugin(root, "empty")

        loaded_skill = load_plugin(skills_only, repo_root=repo_root)
        if (
            set(loaded_skill.skills) != {"skills_only/guide"}
            or loaded_skill.agents
            or loaded_skill.hooks
            or loaded_skill.mcp_servers
        ):
            fail(f"skills-only plugin loaded unexpected members: {loaded_skill!r}")
        loaded_agent = load_plugin(
            agents_only,
            repo_root=repo_root,
            default_model=ModelSelector("fake"),
        )
        if (
            set(loaded_agent.agents) != {"agents_only/worker"}
            or loaded_agent.skills
            or loaded_agent.hooks
            or loaded_agent.mcp_servers
        ):
            fail(f"agents-only plugin loaded unexpected members: {loaded_agent!r}")
        loaded_empty = load_plugin(empty, repo_root=repo_root)
        if any(
            (
                loaded_empty.agents,
                loaded_empty.skills,
                loaded_empty.hooks,
                loaded_empty.mcp_servers,
            )
        ):
            fail(f"manifest-only plugin was not empty: {loaded_empty!r}")


@check("plugins.manifest_validation")
def manifest_validation() -> None:
    fields = ("name", "version", "description")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo_root = root / "repo"
        for field in fields:
            path = root / f"missing_{field}"
            values = {
                "name": path.name,
                "version": "1.0.0",
                "description": "Description.",
            }
            del values[field]
            _write(
                path / "plugin.toml",
                "".join(f"{key} = {json.dumps(value)}\n" for key, value in values.items()),
            )
            _expect_error(path, repo_root, (path.name, field, "required"))

            blank = root / f"blank_{field}"
            values = {
                "name": blank.name,
                "version": "1.0.0",
                "description": "Description.",
            }
            values[field] = "  "
            _write(
                blank / "plugin.toml",
                "".join(f"{key} = {json.dumps(value)}\n" for key, value in values.items()),
            )
            _expect_error(blank, repo_root, (blank.name, field, "blank"))

        mismatch = root / "directory_name"
        _write(mismatch / "plugin.toml", _manifest("declared_name"))
        _expect_error(
            mismatch,
            repo_root,
            ("declared_name", "directory_name", "name"),
        )

        unknown = _plugin(root, "unknown_key")
        _write(unknown / "plugin.toml", _manifest(unknown.name) + "mystery = true\n")
        _expect_error(unknown, repo_root, (unknown.name, "mystery", "unknown key"))

        for invalid_name in (
            "my-plugin",
            "my plugin",
            "my.plugin",
            "café",
            "9plugin",
        ):
            invalid = root / invalid_name
            _write(invalid / "plugin.toml", _manifest(invalid_name))
            _expect_error(
                invalid,
                repo_root,
                ("name", repr(invalid_name), "^[A-Za-z_][A-Za-z0-9_]*$"),
            )

        for valid_name in ("_plugin", "myplugin", "my_plugin2"):
            valid = _plugin(root, valid_name)
            loaded = load_plugin(valid, repo_root=repo_root)
            if loaded.name != valid_name:
                fail(f"identifier-shaped plugin name did not load: {loaded!r}")

        docs = (
            Path(__file__).resolve().parents[2] / "docs/symphonai-api-runtime.md"
        ).read_text(encoding="utf-8")
        new_shape = "^[A-Za-z_][A-Za-z0-9_]*$"
        old_shape = "^[A-Za-z0-9_]+$"
        if new_shape not in docs or old_shape in docs:
            fail("runtime documentation did not state only the corrected plugin shape")


@check("plugins.member_failures_are_atomic")
def member_failures_are_atomic() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo_root = root / "repo"
        cases: list[tuple[Path, tuple[str, ...]]] = []

        agent = _plugin(root, "bad_agent")
        agent_source = _agent(agent, body='tools = ["read_file"]\n')
        cases.append((agent, (agent.name, "agents", str(agent_source), "prompt")))

        skill = _plugin(root, "bad_skill")
        skill_source = _write(skill / "skills" / "guide.md", "no frontmatter\n")
        cases.append((skill, (skill.name, "skills", str(skill_source), "frontmatter")))

        hook = _plugin(root, "bad_hook")
        hook_source = _write(
            hook / "config.toml",
            '[[hooks]]\non = ["NotAnEvent"]\ncommand = ["observe"]\n',
        )
        cases.append((hook, (hook.name, "hooks", str(hook_source), "NotAnEvent")))

        mcp = _plugin(root, "bad_mcp")
        mcp_source = _write(
            mcp / "config.toml",
            '[[mcp.servers]]\nname = "docs"\ncommand = []\n',
        )
        cases.append((mcp, (mcp.name, "mcp_servers", str(mcp_source), "command")))

        for path, fragments in cases:
            _expect_error(
                path,
                repo_root,
                fragments,
                default_model=ModelSelector("fake"),
            )


@check("plugins.namespaces_do_not_shadow")
def namespaces_do_not_shadow() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo_root = root / "repo"
        local_root = root / "local"
        _agent(local_root, body='prompt = "Local work."\n')
        _skill(local_root)
        _config(local_root)
        default_model = ModelSelector("fake")
        local_agents = load_agent_directory(
            local_root / "agents",
            repo_root=repo_root,
            default_model=default_model,
        )
        local_skills = load_skill_directory(local_root / "skills")
        local_config = plugins_module._member_config(local_root)
        local_servers = {
            server.name: server
            for server in mcp_servers_from_config(local_config, repo_root=repo_root)
        }

        loaded = []
        for name in ("alpha", "beta"):
            path = _plugin(root / "plugins", name)
            _agent(path)
            _skill(path)
            _config(path)
            loaded.append(
                load_plugin(
                    path,
                    repo_root=repo_root,
                    default_model=default_model,
                )
            )

        agent_registry = dict(local_agents)
        skill_registry = dict(local_skills)
        server_registry = dict(local_servers)
        for plugin in loaded:
            agent_registry.update(plugin.agents)
            skill_registry.update(plugin.skills)
            server_registry.update(
                {server.name: server for server in plugin.mcp_servers}
            )

        tool_names = {
            McpTool(
                McpClient(plugin.mcp_servers[0], cwd=root),
                server_tool_name="search",
                description="Search.",
                parameters={},
            ).name
            for plugin in loaded
        }
        if tool_names != {"mcp__alpha__docs__search", "mcp__beta__docs__search"}:
            fail(f"plugin MCP tool names collided: {tool_names!r}")
        if agent_registry["worker"] != local_agents["worker"]:
            fail("plugin replaced the local agent")
        if skill_registry["guide"] != local_skills["guide"]:
            fail("plugin replaced the local skill")
        if server_registry["docs"] != local_servers["docs"]:
            fail("plugin replaced the local MCP server")
        if set(agent_registry) != {"worker", "alpha/worker", "beta/worker"}:
            fail(f"agent namespaces collided: {agent_registry!r}")
        if set(skill_registry) != {"guide", "alpha/guide", "beta/guide"}:
            fail(f"skill namespaces collided: {skill_registry!r}")
        if set(server_registry) != {"docs", "alpha__docs", "beta__docs"}:
            fail(f"MCP namespaces collided: {server_registry!r}")
        if any(
            plugin.agents[f"{plugin.name}/worker"].name != "worker"
            or plugin.skills[f"{plugin.name}/guide"].name != "guide"
            for plugin in loaded
        ):
            fail("agent or skill value names were namespaced instead of mapping keys")


@check("plugins.composed_mcp_names_are_usable")
def composed_mcp_names_are_usable() -> None:
    combinations = (
        ("alpha", "docs", "search"),
        ("A1", "server_2", "tool-name"),
        ("_private", "docs", "search_1"),
        ("release2", "code", "grep"),
        ("p" * 20, "s" * 15, "t" * 10),
        ("p", "s", "t"),
    )
    vendor_pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo_root = root / "repo"
        for plugin_name, server_name, tool_name in combinations:
            path = _plugin(root, plugin_name)
            _config(path, server_name=server_name)
            plugin = load_plugin(path, repo_root=repo_root)
            if len(plugin.mcp_servers) != 1:
                fail(f"plugin server did not load for {plugin_name!r}")
            server = plugin.mcp_servers[0]
            expected_server = f"{plugin_name}__{server_name}"
            tool = McpTool(
                McpClient(server, cwd=root),
                server_tool_name=tool_name,
                description="Test tool.",
                parameters={},
            )
            expected_tool = f"mcp__{plugin_name}__{server_name}__{tool_name}"
            if not server.name.isidentifier():
                fail(f"plugin server name was unusable: {server.name!r}")
            if server.name != expected_server:
                fail(
                    f"plugin server namespace was {server.name!r}, "
                    f"expected {expected_server!r}"
                )
            if vendor_pattern.fullmatch(tool.name) is None:
                fail(f"plugin tool name was unusable: {tool.name!r}")
            if tool.name != expected_tool:
                fail(
                    f"plugin tool namespace was {tool.name!r}, "
                    f"expected {expected_tool!r}"
                )

        for invalid_name in ("my-plugin", "9plugin"):
            invalid = _plugin(root, invalid_name)
            _config(invalid)
            try:
                invalid_plugin = load_plugin(invalid, repo_root=repo_root)
            except PluginError:
                pass
            else:
                server_name = invalid_plugin.mcp_servers[0].name
                if server_name.isidentifier():
                    fail(
                        f"invalid plugin {invalid_name!r} unexpectedly made "
                        f"identifier {server_name!r}"
                    )
                fail(f"composed-name sweep produced non-identifier {server_name!r}")


@check("plugins.hooks_append_in_plugin_order")
def hooks_append_in_plugin_order() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo_root = root / "repo"
        repo_root.mkdir()
        plugins_root = root / "plugins"
        output = root / "order.txt"
        code = (
            "from pathlib import Path; import sys; "
            "p=Path(sys.argv[1]); "
            "p.write_text(p.read_text() + sys.argv[2] + '\\n' "
            "if p.exists() else sys.argv[2] + '\\n', encoding='utf-8')"
        )

        for name, label in (("bravo", "bravo"), ("alpha", "alpha")):
            path = _plugin(plugins_root, name)
            _config(path, hook_command=(sys.executable, "-c", code, str(output), label))
        plugins = load_plugin_directory(plugins_root, repo_root=repo_root)
        local = HookSpec(
            events=("RunFinished",),
            command=(sys.executable, "-c", code, str(output), "local"),
        )
        hooks = append_plugin_hooks((local,), plugins)
        HookRunner(hooks, cwd=repo_root)(RunFinished("agent", "run"))
        actual = output.read_text(encoding="utf-8").splitlines()
        if actual != ["local", "alpha", "bravo"]:
            fail(f"hook append order was {actual!r}")


@check("plugins.agent_ceiling_is_forwarded")
def agent_ceiling_is_forwarded() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo_root = root / "repo"
        path = _plugin(root, "bounded")
        agent = _agent(
            path,
            body=(
                'prompt = "Work."\n'
                "[policy]\n"
                'allowed_write_scope = ["../outside"]\n'
                "shell_enabled = true\n"
            ),
        )
        default_model = ModelSelector("fake")
        load_plugin(path, repo_root=repo_root, default_model=default_model)
        ceiling = CapabilityCeiling(
            allowed_write_scope=((repo_root / "src").resolve(),),
            shell_enabled=False,
        )
        _expect_error(
            path,
            repo_root,
            (path.name, "agents", str(agent), "allowed_write_scope"),
            default_model=default_model,
            ceiling=ceiling,
        )


@check("plugins.directory_and_import_boundary")
def directory_and_import_boundary() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo_root = root / "repo"
        missing = root / "missing"
        if load_plugin_directory(missing, repo_root=repo_root) != {}:
            fail("missing plugin directory was not empty")

        plugins_root = root / "plugins"
        direct = _plugin(plugins_root, "direct")
        _write(plugins_root / "README.md", "not a plugin")
        nested = _plugin(direct / "nested", "nested")
        loaded = load_plugin_directory(plugins_root, repo_root=repo_root)
        if list(loaded) != ["direct"] or loaded["direct"].path != direct:
            fail(f"plugin discovery was recursive or included files: {loaded!r}")
        if nested.name in loaded:
            fail("nested plugin was discovered")

        broken = plugins_root / "broken"
        broken.mkdir()
        try:
            load_plugin_directory(plugins_root, repo_root=repo_root)
        except PluginError as exc:
            if "broken" not in str(exc) or "plugin.toml" not in str(exc):
                fail(f"directory error omitted offending plugin: {exc!r}")
        else:
            fail("plugin directory skipped a malformed direct plugin")

    path = Path(__file__).resolve().parents[2] / "symphonai_api/plugins.py"
    source = path.read_text(encoding="utf-8")
    forbidden = _forbidden_imports(source)
    expanded = {
        "agent_loop",
        "leader",
        "runner",
        "agent_run",
        "child_context",
        "provider_catalog",
        "providers",
    }
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module] if node.module is not None else []
        else:
            continue
        forbidden.extend(
            module
            for module in modules
            if set(module.split(".")) & expanded
        )
    if forbidden:
        fail(f"plugins.py imports forbidden runtime modules: {sorted(set(forbidden))!r}")

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    tomllib_callers = {
        name
        for name, function in functions.items()
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tomllib"
            for node in ast.walk(function)
        )
    }
    if tomllib_callers != {"_load_manifest"}:
        fail(f"member parsers call tomllib: {sorted(tomllib_callers)!r}")
    load_plugin_calls = {
        node.func.id
        for node in ast.walk(functions["load_plugin"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    required_loaders = {
        "load_agent_directory",
        "load_skill_directory",
        "hooks_from_config",
        "mcp_servers_from_config",
    }
    if not required_loaders <= load_plugin_calls:
        fail(
            "load_plugin did not delegate every member parser: "
            f"{sorted(required_loaders - load_plugin_calls)!r}"
        )
