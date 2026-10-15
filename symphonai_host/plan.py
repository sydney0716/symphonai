"""Leader-managed conversation plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from symphonai_api.events import Event
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata


PLAN_STEP_LIMIT = 20
PLAN_STEP_CHARS = 200
PLAN_EXPLANATION_CHARS = 500


@dataclass(frozen=True)
class PlanChanged(Event):
    steps: list[dict] = field(default_factory=list)
    explanation: str = ""
    session_id: str = ""


class UpdatePlanTool(LocalTool):
    def __init__(self, update: Callable[[list[dict], str], None]) -> None:
        self._update = update

    @property
    def name(self) -> str:
        return "update_plan"

    @property
    def description(self) -> str:
        return (
            "Use this for work of three or more steps. Keep exactly one step in progress while working, "
            "and mark steps completed as soon as they are done."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": PLAN_STEP_LIMIT,
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string", "maxLength": PLAN_STEP_CHARS},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        },
                        "required": ["step", "status"],
                    },
                },
                "explanation": {"type": "string", "maxLength": PLAN_EXPLANATION_CHARS},
            },
            "required": ["plan"],
        }

    def validate(self, arguments: dict) -> str | None:
        if set(arguments) - {"plan", "explanation"} or "plan" not in arguments:
            return "arguments must contain plan and optional explanation"
        plan = arguments["plan"]
        if not isinstance(plan, list) or not 1 <= len(plan) <= PLAN_STEP_LIMIT:
            return f"plan must contain 1 to {PLAN_STEP_LIMIT} steps"
        in_progress = 0
        for step in plan:
            if not isinstance(step, dict) or set(step) != {"step", "status"}:
                return "each plan step must contain only step and status"
            text, status = step["step"], step["status"]
            if not isinstance(text, str) or not text.strip() or len(text) > PLAN_STEP_CHARS:
                return f"step must be non-blank and at most {PLAN_STEP_CHARS} characters"
            if not isinstance(status, str) or status not in {"pending", "in_progress", "completed"}:
                return "status must be pending, in_progress, or completed"
            in_progress += status == "in_progress"
        if in_progress > 1:
            return "plan may have at most one in_progress step"
        explanation = arguments.get("explanation", "")
        if not isinstance(explanation, str) or len(explanation) > PLAN_EXPLANATION_CHARS:
            return f"explanation must be a string of at most {PLAN_EXPLANATION_CHARS} characters"
        return None

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(effect=ToolEffect.READ_ONLY, concurrency_safe=False, paths=None)

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel=None,
    ) -> ToolResult:
        plan = [dict(step) for step in tool_call.arguments["plan"]]
        explanation = tool_call.arguments.get("explanation", "")
        self._update(plan, explanation)
        return ToolResult(tool_call_id=tool_call.id, ok=True, content="Plan updated.")


def plan_tools(update: Callable[[list[dict], str], None]) -> dict[str, LocalTool]:
    return {"update_plan": UpdatePlanTool(update)}
