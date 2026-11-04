"""Helpers for reading small sections from task specifications."""

from __future__ import annotations

import re

from symphonai_api.tools.shell_parse import parse_command_line


def spec_validation(text: str) -> list[str]:
    """Read complete checkable shell commands, raising ValueError on bad syntax."""
    validation = []
    in_section = False
    in_fence = False
    section_level = 0
    for line in text.splitlines():
        if not in_fence:
            heading = re.match(r"^(#{1,6})[ \t]+(.+?)\s*$", line)
            if heading:
                level = len(heading.group(1))
                title = heading.group(2).rstrip("# ").strip()
                if in_section:
                    if level <= section_level:
                        break
                elif title.casefold() == "validation":
                    in_section = True
                    section_level = level
                continue
        if not in_section:
            continue
        if line.startswith("```"):
            if in_fence:
                break
            in_fence = line[3:].strip() in ("", "bash", "sh")
            continue
        if in_fence:
            validation.append(line)
    commands: list[str] = []
    pending: list[str] = []
    for line in validation:
        if not pending and not line.strip():
            continue
        pending.append(line)
        command = "\n".join(pending)
        # A trailing operator needs the following line even though the policy
        # parser can already identify the command before that operator.
        if re.search(r"(?:&&|\|\||\|)\s*$", command):
            continue
        if parse_command_line(command) is not None:
            commands.append(command)
            pending = []
    if pending:
        command = "\n".join(pending)
        raise ValueError(
            f"Validation command {command!r} cannot be parsed: "
            "incomplete or unsupported shell syntax that the policy cannot check"
        )
    return commands
