"""Load a trust-filtered skill body on demand."""

from __future__ import annotations

from collections.abc import Mapping

from symphonai_api.cancellation import CancellationToken
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.skills import Skill, SkillError, roster_text
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata


class UseSkillTool(LocalTool):
    def __init__(self, skills: Mapping[str, Skill]) -> None:
        self._skills = skills

    @property
    def name(self) -> str:
        return "use_skill"

    @property
    def description(self) -> str:
        return (
            "Load the full instructions of a skill. Call it when the task matches a skill's when_to_use."
            "\n\nAvailable skills:\n\n"
            + roster_text(self._skills[name] for name in sorted(self._skills))
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

    def validate(self, arguments: dict) -> str | None:
        if not isinstance(arguments.get("name"), str):
            return "missing or invalid required argument: name (must be a string)"
        return None

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=None,
        )

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        name = tool_call.arguments["name"]
        skill = self._skills.get(name)
        if skill is None:
            available = ", ".join(sorted(self._skills))
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"unknown skill {name!r}; available: {available}",
            )
        try:
            body = skill.body()
        except SkillError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        return ToolResult(tool_call_id=tool_call.id, ok=True, content=body)
