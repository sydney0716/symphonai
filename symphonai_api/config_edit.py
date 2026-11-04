"""Edit one scalar TOML assignment without rewriting the surrounding file."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import tempfile
import tomllib

from symphonai_api.config import load_config
from symphonai_api.paths import ensure_private_directory, symphonai_home, write_atomic


def _key_path(source: str, value: str = "0") -> tuple[str, ...]:
    node = tomllib.loads(f"{source} = {value}")
    parts = []
    while isinstance(node, dict) and len(node) == 1:
        key, node = next(iter(node.items()))
        parts.append(key)
    return tuple(parts)


def _replacement(text: str, key: str, value: object) -> str:
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+", key):
        raise ValueError("key must be a dotted config key")
    if type(value) not in (bool, int, float, str) or (type(value) is float and not math.isfinite(value)):
        raise ValueError("value must be a boolean, finite number, or string; tables and arrays stay in the file")
    parts = tuple(key.split("."))
    current = tomllib.loads(text)
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            current = None
            break
        current = current[part]
    if isinstance(current, (dict, list)):
        raise ValueError("tables and arrays stay in the file")
    encoded = json.dumps(value, ensure_ascii=True) if type(value) is not float else repr(value)
    lines = text.splitlines(keepends=True)
    table = ()
    insertion = None
    newline = "\r\n" if "\r\n" in text else "\n"
    multiline = None
    for index, line in enumerate(lines):
        if multiline is not None:
            if re.search(r"(?<!\\)" + re.escape(multiline), line):
                multiline = None
            continue
        stripped = line.strip()
        if stripped.startswith("["):
            if table == parts[:-1] and insertion is None:
                insertion = index
            if stripped.startswith("[["):
                table = None
            else:
                header = stripped.split("]", 1)[0][1:]
                table = _key_path(header)
            continue
        assignment = re.match(r"^(\s*([^=]+?)\s*=\s*)(.*?)(\r?\n)?$", line)
        if assignment is None or table is None or stripped.startswith("#"):
            continue
        raw = assignment[3]
        if raw.startswith(('"""', "'''")) and not re.search(r"(?<!\\)" + re.escape(raw[:3]), raw[3:]):
            multiline = raw[:3]
        try:
            path = table + _key_path(assignment[2])
        except tomllib.TOMLDecodeError:
            continue
        if path != parts:
            continue
        scalar = re.fullmatch(r'''("(?:[^"\\]|\\.)*"|'[^']*'|[^#'"\r\n]*?)(\s*(?:\#.*)?)''', raw)
        if scalar is None or raw.startswith(('"""', "'''")):
            raise ValueError("this value spans multiple lines; open the config file to edit it")
        lines[index] = assignment[1] + encoded + scalar[2] + (assignment[4] or "")
        return "".join(lines)
    if table == parts[:-1]:
        insertion = len(lines)
    addition = f"{parts[-1]} = {encoded}{newline}"
    if insertion is not None:
        if insertion and not lines[insertion - 1].endswith("\n"):
            lines[insertion - 1] += newline
        lines.insert(insertion, addition)
        return "".join(lines)
    return text + (newline if text and not text.endswith("\n") else "") + f"[{'.'.join(parts[:-1])}]{newline}" + addition


def set_config_value(key: str, value: object, scope: str, *, repo_root: Path, home: Path | None = None) -> Path:
    if scope not in ("user", "project"):
        raise ValueError("scope must be user or project")
    if scope == "project" and (key.split(".", 1)[0] in {"roles", "permissions", "always_deny"}
                               or key == "sandbox.readable"):
        raise ValueError(f"{key} is accepted only in user config")
    user = symphonai_home(home) / "config.toml"
    project = Path(repo_root) / ".symphonai" / "config.toml"
    target = user if scope == "user" else project
    try:
        with target.open(encoding="utf-8", newline="") as source:
            original = source.read()
    except FileNotFoundError:
        original = ""
    candidate = _replacement(original, key, value)
    tomllib.loads(candidate)
    # Run the real layered loader on staged config files before touching the target.
    with tempfile.TemporaryDirectory(prefix="symphonai-config-") as temporary:
        staged = Path(temporary)
        mappings = (
            (user, staged / "home" / ".symphonai" / "config.toml"),
            (project, staged / "repo" / ".symphonai" / "config.toml"),
            (project.with_name("config.local.toml"), staged / "repo" / ".symphonai" / "config.local.toml"),
        )
        def validate(text):
            for source, destination in mappings:
                if source != target and not source.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text if source == target else source.read_text(encoding="utf-8"), encoding="utf-8")
            user_file = mappings[0][1]
            user_values = tomllib.loads(user_file.read_text(encoding="utf-8")) if user_file.exists() else {}
            trust = user_values.get("trust", {})
            grants = trust.get("repositories", []) if isinstance(trust, dict) else []
            # A grant for the real repository applies to its validation copy.
            grants = [
                {**grant, "root": str(staged / "repo")}
                if isinstance(grant, dict) and isinstance(grant.get("root"), str)
                and Path(grant["root"]).expanduser().resolve() == Path(repo_root).resolve()
                else grant for grant in grants
            ] if isinstance(grants, list) else grants
            config = load_config(repo_root=staged / "repo", home=staged / "home",
                                 session={"trust": {"repositories": grants}})
            return set(config.warnings)

        try:
            before = validate(original)
            introduced = validate(candidate) - before
            if introduced:
                raise ValueError("; ".join(sorted(introduced)))
        except ValueError as exc:
            message = str(exc)
            for source, destination in mappings:
                message = message.replace(str(destination), str(source))
            raise ValueError(message) from exc
    destination = target.resolve()
    ensure_private_directory(destination.parent)
    write_atomic(destination, candidate)
    return target
