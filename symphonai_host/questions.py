"""Leader questions answered through the approval channel."""

from __future__ import annotations

from typing import Callable

from symphonai_api.identity import new_id
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import ApprovalOutcome, DenialReason, PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_host.approvals import PendingApproval


QUESTION_LIMIT = 1_000
OPTION_LIMIT = 100
PLAN_LIMIT = 20_000


class AskUserTool(LocalTool):
    def __init__(self, ask: Callable[[str, tuple[str, ...]], tuple[object, str]]) -> None:
        self._ask = ask

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return (
            "Ask only what the person alone can answer. Offer options when the likely answers are few; "
            "the person can always answer in their own words."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string", "minLength": 1, "maxLength": QUESTION_LIMIT},
                "options": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {"type": "string", "minLength": 1, "maxLength": OPTION_LIMIT},
                },
            },
            "required": ["question"],
        }

    def validate(self, arguments: dict) -> str | None:
        if set(arguments) - {"question", "options"} or "question" not in arguments:
            return "arguments must contain question and optional options"
        question = arguments["question"]
        if not isinstance(question, str) or not question.strip() or len(question) > QUESTION_LIMIT:
            return f"question must be non-blank and at most {QUESTION_LIMIT} characters"
        options = arguments.get("options")
        if options is not None:
            if not isinstance(options, list) or not 2 <= len(options) <= 4:
                return "options must contain 2 to 4 choices"
            if any(
                not isinstance(option, str)
                or not option.strip()
                or len(option) > OPTION_LIMIT
                for option in options
            ):
                return f"each option must be non-blank and at most {OPTION_LIMIT} characters"
            if len(set(options)) != len(options):
                return "options must be distinct"
        return None

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(effect=ToolEffect.READ_ONLY, concurrency_safe=False, paths=None)

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel=None,
    ) -> ToolResult:
        decision, answer = self._ask(
            tool_call.arguments["question"], tuple(tool_call.arguments.get("options", ())),
        )
        if decision.allowed:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=True,
                content=f"The person answered: {answer}",
            )
        if decision.denial is DenialReason.DENIED_BY_USER:
            return ToolResult(tool_call_id=tool_call.id, ok=True, content="The person skipped the question.")
        reason = (
            "the question was stopped"
            if decision.outcome is ApprovalOutcome.CANCELLED
            else "the question timed out"
            if decision.denial is DenialReason.APPROVAL_FAILED
            else "no client is connected to answer the question"
            if decision.denial is DenialReason.NO_APPROVAL_CALLBACK
            else decision.reason or "the question could not be answered"
        )
        return ToolResult(tool_call_id=tool_call.id, ok=False, error=reason)


def question_tools(
    ask: Callable[[str, tuple[str, ...]], tuple[object, str]],
) -> dict[str, LocalTool]:
    return {"ask_user": AskUserTool(ask)}


class ProposePlanTool(LocalTool):
    def __init__(
        self,
        ask: Callable[[PendingApproval], tuple[object, str]],
        permitted_modes: Callable[[], tuple[str, ...]],
        session_id: str | None,
    ) -> None:
        self._ask = ask
        self._permitted_modes = permitted_modes
        self._session_id = session_id

    @property
    def name(self) -> str:
        return "propose_plan"

    @property
    def description(self) -> str:
        return "In plan mode, present the complete ready plan with this tool. Do not change files before it is approved."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"plan": {"type": "string", "minLength": 1, "maxLength": PLAN_LIMIT}},
            "required": ["plan"],
        }

    def validate(self, arguments: dict) -> str | None:
        plan = arguments.get("plan")
        if set(arguments) != {"plan"}:
            return "arguments must contain only plan"
        if not isinstance(plan, str) or not plan.strip() or len(plan) > PLAN_LIMIT:
            return f"plan must be non-blank and at most {PLAN_LIMIT} characters"
        return None

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(effect=ToolEffect.READ_ONLY, concurrency_safe=False, paths=None)

    def _execute(self, tool_call: ToolCall, policy: PermissionPolicy, cancel=None) -> ToolResult:
        if policy.mode != "plan":
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"propose_plan is for plan mode; this conversation is in {policy.mode} mode",
            )
        options = tuple(mode for mode in ("ask", "allow") if mode in self._permitted_modes())
        approval = PendingApproval(
            approval_id=new_id("plan"),
            operation="propose_plan",
            target=tool_call.arguments["plan"],
            details="",
            session_id=self._session_id,
            kind="plan",
            options=options,
        )
        decision, answer = self._ask(approval)
        if decision.allowed:
            policy.mode = answer
            label = "ask" if answer == "ask" else "allow"
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=True,
                content=f"The person approved the plan. The conversation is now in {label} mode; carry out the plan.",
            )
        if decision.denial is DenialReason.DENIED_BY_USER:
            reason = decision.reason
            suffix = f": {reason}" if reason and reason != "approval denied by user" else "."
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=True,
                content=f"The person wants to keep planning{suffix}",
            )
        reason = (
            "the plan approval was stopped"
            if decision.outcome is ApprovalOutcome.CANCELLED
            else "the plan approval timed out"
            if decision.denial is DenialReason.APPROVAL_FAILED
            else "no client is connected to approve the plan"
            if decision.denial is DenialReason.NO_APPROVAL_CALLBACK
            else decision.reason or "the plan could not be approved"
        )
        return ToolResult(tool_call_id=tool_call.id, ok=False, error=reason)


def plan_proposal_tools(
    ask: Callable[[PendingApproval], tuple[object, str]],
    permitted_modes: Callable[[], tuple[str, ...]],
    *,
    session_id: str | None = None,
) -> dict[str, LocalTool]:
    return {"propose_plan": ProposePlanTool(ask, permitted_modes, session_id)}
