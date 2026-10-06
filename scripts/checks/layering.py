"""Shared import-boundary checks for runtime modules."""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.checks.harness import check, fail


ROOT = Path(__file__).resolve().parents[2]
_BASE_FORBIDDEN = frozenset({"agent_loop", "leader", "runner", "provider_catalog", "providers"})
IMPORT_RULES = (
    {
        "checks": ("agent_file.no_runtime_imports",),
        "paths": ("symphonai_api/agent_file.py",),
        "forbidden": _BASE_FORBIDDEN | {"agent_run", "child_context"},
        "probe": "from symphonai_api.agent_loop import Probe\n",
    },
    {
        "checks": ("agent_memory.no_runtime_imports",),
        "paths": ("symphonai_api/agent_memory.py",),
        "forbidden": _BASE_FORBIDDEN | {"agent_run", "agent_spec", "child_context"},
        "probe": "from symphonai_api.agent_spec import Probe\n",
    },
    {
        "checks": ("agent_run.no_runtime_imports",),
        "paths": ("symphonai_api/agent_run.py",),
        "forbidden": _BASE_FORBIDDEN,
        "probe": "from symphonai_api.runner import Probe\n",
    },
    {
        "checks": ("agent_spec.no_runtime_imports",),
        "paths": ("symphonai_api/agent_spec.py",),
        "forbidden": _BASE_FORBIDDEN,
        "probe": "from symphonai_api.leader import Probe\n",
    },
    {
        "checks": ("child_context.no_runtime_imports",),
        "paths": ("symphonai_api/child_context.py",),
        "forbidden": _BASE_FORBIDDEN | {"agent_run"},
        "probe": "from symphonai_api.agent_run import Probe\n",
    },
    {
        "checks": ("config.no_runtime_imports",),
        "paths": ("symphonai_api/config.py",),
        "forbidden": _BASE_FORBIDDEN | {"agent_run", "agent_spec", "agent_file", "child_context"},
        "probe": "from symphonai_api.agent_file import Probe\n",
    },
    {
        "checks": ("skills.no_runtime_imports",),
        "paths": ("symphonai_api/skills.py",),
        "forbidden": _BASE_FORBIDDEN | {"agent_run", "agent_spec", "agent_file", "child_context", "hooks"},
        "mode": "from_submodule",
        "required_from": ("symphonai_api.compaction", "estimate_text_tokens"),
        "probe": "from symphonai_api.agent_loop import Probe\n",
    },
    {
        "checks": ("trust.no_runtime_imports",),
        "paths": ("symphonai_api/trust.py",),
        "forbidden": _BASE_FORBIDDEN | {"agent_run", "agent_spec", "agent_file", "child_context", "hooks", "mcp", "skills", "permissions"},
        "probe": "from symphonai_api.permissions import Probe\n",
    },
    {
        "checks": ("leases.no_symphonai_imports",),
        "paths": ("symphonai_api/leases.py",),
        "forbidden": frozenset({"symphonai_api"}),
        "probe": "import symphonai_api\n",
    },
    {
        "checks": ("survey.import_boundary",),
        "paths": ("symphonai_api/survey.py",),
        "forbidden": _BASE_FORBIDDEN,
        "probe": "from symphonai_api.providers.fake import Probe\n",
    },
    {
        "checks": ("mcp.import_boundary",),
        "paths": ("symphonai_api/mcp.py",),
        "forbidden": _BASE_FORBIDDEN | {"agent_run", "agent_spec", "agent_file", "child_context", "hooks"},
        "probe": "from symphonai_api.hooks import Probe\n",
    },
    {
        "checks": ("host_client.stdlib_only",),
        "paths": ("symphonai_host/client.py", "symphonai_host/cli.py"),
        "allowed_roots": frozenset({"__future__", "argparse", "dataclasses", "http", "json", "queue", "select", "subprocess", "sys", "threading", "typing"}),
        "mode": "modules",
        "probe": "import third_party_probe\n",
    },
    {
        "checks": ("host_protocol.import_direction", "host_server.api_untouched"),
        "paths": ("symphonai_api/**/*.py",),
        "forbidden_text": "symphonai_host",
        "probe": "# symphonai_host\n",
    },
    {
        "checks": ("discovery.determinism_and_imports",),
        "paths": ("symphonai_api/discovery.py",),
        "forbidden": _BASE_FORBIDDEN | {"agent_run", "child_context", "extensions"},
        "probe": "from symphonai_api.extensions import Probe\n",
    },
    {
        "checks": ("plugins.directory_and_import_boundary",),
        "paths": ("symphonai_api/plugins.py",),
        "forbidden": _BASE_FORBIDDEN | {"agent_run", "child_context"},
        "probe": "from symphonai_api.agent_run import Probe\n",
    },
    {
        "checks": ("extensions.leader_default_and_imports",),
        "paths": ("symphonai_api/extensions.py",),
        "forbidden": _BASE_FORBIDDEN | {"agent_run", "agent_file", "child_context"},
        "probe": "from symphonai_api.agent_file import Probe\n",
    },
    {
        "checks": ("hooks.none_is_unchanged_and_imports",),
        "paths": ("symphonai_api/hooks.py",),
        "forbidden": _BASE_FORBIDDEN | {"agent_run", "agent_spec", "agent_file", "permissions", "child_context"},
        "probe": "from symphonai_api.permissions import Probe\n",
    },
    {
        "checks": ("mcp_pool.default_and_imports",),
        "paths": ("symphonai_api/mcp_pool.py",),
        "forbidden": _BASE_FORBIDDEN | {"agent_run", "agent_spec", "agent_file", "child_context", "extensions"},
        "probe": "from symphonai_api.extensions import Probe\n",
    },
)


def _module_names(source: str, mode: str = "components") -> list[str]:
    modules = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if mode != "from_submodule":
                modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                if mode == "from_submodule":
                    if node.module.startswith("symphonai_api."):
                        modules.append(node.module.split(".")[1])
                else:
                    modules.append(node.module)
            if mode == "components":
                modules.extend(alias.name for alias in node.names)
    return modules


def _violations(source: str, rule: dict) -> list[str]:
    if "forbidden_text" in rule:
        return [rule["forbidden_text"]] if rule["forbidden_text"] in source else []
    names = _module_names(source, rule.get("mode", "components"))
    if "allowed_roots" in rule:
        return [
            name for name in names
            if name.split(".")[0] not in rule["allowed_roots"]
            and not name.split(".")[0].startswith("symphonai_")
        ]
    return [
        name for name in names
        if any(part in rule["forbidden"] for part in name.split("."))
    ]


@check("layering.imports")
def imports() -> None:
    for rule in IMPORT_RULES:
        if not _violations(rule["probe"], rule):
            fail(f"import scanner missed probe for {rule['checks']!r}")
        paths = [path for pattern in rule["paths"] for path in ROOT.glob(pattern)]
        if not paths:
            fail(f"import rule matched no files: {rule['paths']!r}")
        for path in sorted(paths):
            source = path.read_text(encoding="utf-8")
            found = _violations(source, rule)
            if found:
                fail(f"{path.relative_to(ROOT)} violates import rule: {sorted(set(found))!r}")
            required = rule.get("required_from")
            if required is not None:
                module, name = required
                tree = ast.parse(source)
                if not any(
                    isinstance(node, ast.ImportFrom)
                    and node.module == module
                    and any(alias.name == name for alias in node.names)
                    for node in ast.walk(tree)
                ):
                    fail(f"{path.relative_to(ROOT)} does not import {name} from {module}")
