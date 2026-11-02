"""The ApiAgent loop: call the model, execute any tool calls, repeat.

On each turn: call `provider.create_response()`. If the response carries
one or more `ToolCall`s, execute each via the tool registry (gated by
`PermissionPolicy`), append the resulting `ToolResult`(s) as tool-role
`Message`s, and loop. Otherwise the response is final. The loop always
terminates -- either on a final response or once `max_turns` is reached --
and never raises just because `max_turns` was hit.
"""

from __future__ import annotations

import hashlib
import json
import time
import threading
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path

from symphonai_api.budgets import BudgetState, RunBudget
from symphonai_api.call_class import CallClass
from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.compaction import (
    ContextCompactionError,
    DEFAULT_RECENT_TURNS,
    MODEL_SUMMARY_PROMPT,
    budget_for_model,
    compact_messages_for_budget,
    estimate_messages_tokens,
    render_dropped_messages,
)
from symphonai_api.cost import UsageTotals
from symphonai_api.events import (
    AssistantReasoningDelta,
    AssistantTextDelta,
    EventSink,
    ModelFallback,
    PromptSubmitted,
    RunFailed,
    RunFinished,
    RunStarted,
    CompactionApplied,
    SessionNotesUpdated,
    ToolCallFailed,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
    TurnStarted,
    emit,
)
from symphonai_api.identity import (
    AgentRef,
    RunRef,
    TurnRef,
    new_agent_ref,
    new_run_ref,
    new_turn_ref,
)
from symphonai_api.model_table import long_context_threshold
from symphonai_api.hooks import HookRunner
from symphonai_api.agent_run import AgentRun
from symphonai_api.models import (
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    ToolCall,
    ToolResult,
)
from symphonai_api.paths import write_atomic
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import (
    ContextLengthExceededError,
    ModelProvider,
    ProviderUnavailableError,
    tool_result_text,
)
from symphonai_api.repair import repair_unanswered_tool_calls
from symphonai_api.scheduler import MAX_TOOL_CONCURRENCY, partition_tool_calls
from symphonai_api.secrets import mask_tool_result
from symphonai_api.serialization import message_to_json
from symphonai_api.session import TranscriptWriter
from symphonai_api.streaming import ReasoningDelta, StreamAssembler, TextDelta
from symphonai_api.tool_results import ToolResultStore, offload_tool_result
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import FILE_EDIT_TOOLS, call_target
from symphonai_api.tools.read_ledger import ReadLedger

_CUT_TOOL_CALL_ERROR = (
    "not run: the reply reached its output limit before this call was complete. "
    "Split the work into smaller calls."
)


def _message_digest(message: Message) -> str:
    canonical = json.dumps(
        message_to_json(message), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.blake2b(canonical, digest_size=16).hexdigest()


def _tool_result_event_fields(
    result: ToolResult, tool_name: str = "",
) -> dict[str, str | int | bool]:
    output = ""
    output_truncated = False
    hidden_output_tools = {
        "dispatch_subagent", "update_plan", "ask_user", "propose_plan",
        *FILE_EDIT_TOOLS,
    }
    if (
        tool_name not in hidden_output_tools
        and not (isinstance(result.payload, dict) and result.payload.get("kind") == "file_diff")
    ):
        text = tool_result_text(result)
        output = text[:4000]
        output_truncated = len(text) > 4000
    defaults: dict[str, str | int | bool] = {
        "result_kind": "",
        "result_path": "",
        "lines_added": 0,
        "lines_removed": 0,
        "diff": "",
        "truncated": False,
        "output": output,
        "output_truncated": output_truncated,
    }
    payload = result.payload
    if not result.ok or not isinstance(payload, dict):
        return defaults
    if payload.get("kind") != "file_diff":
        return defaults
    path = payload.get("path")
    lines_added = payload.get("lines_added")
    lines_removed = payload.get("lines_removed")
    truncated = payload.get("truncated")
    return {
        "result_kind": "file_diff",
        "result_path": path if isinstance(path, str) else "",
        "lines_added": lines_added if type(lines_added) is int else 0,
        "lines_removed": lines_removed if type(lines_removed) is int else 0,
        "diff": (
            payload["diff"]
            if isinstance(payload.get("diff"), str)
            else result.content if isinstance(result.content, str) else ""
        ),
        "truncated": truncated if type(truncated) is bool else False,
    }


@dataclass
class AgentRunResult:
    """The outcome of an `ApiAgent.run()` call."""

    final_response: ModelResponse
    messages: list[Message]
    turns_used: int
    # "final_response", "max_turns", "cancelled", "budget_turns",
    # "budget_wall_time", "budget_tokens", or "budget_cost"
    stopped_reason: str
    run: RunRef
    agent: AgentRef
    usage_by_model: dict[str, UsageTotals] = field(default_factory=dict)


class ApiAgent:
    """Runs the call-model / execute-tools loop against a `ModelProvider`."""

    def __init__(
        self,
        provider: ModelProvider,
        tools: dict[str, LocalTool],
        policy: PermissionPolicy,
        max_turns: int | None = None,
        tool_schemas: list[dict] | None = None,
        *,
        result_store: ToolResultStore | None = None,
        agent_ref: AgentRef | None = None,
        events: EventSink | None = None,
        budget: RunBudget | None = None,
        call_class: CallClass = CallClass.FOREGROUND,
        transcript: TranscriptWriter | None = None,
        stream: bool = False,
        read_ledger: ReadLedger | None = None,
        model_summary: bool = False,
        session_notes: Path | None = None,
        context_budget: Callable[[str | None], int] | None = None,
        context_overflow: Callable[[int, int | None], None] | None = None,
        context_recent_turns: int = DEFAULT_RECENT_TURNS,
        deferrable_tools: tuple[str, ...] = (),
        tool_history: Path | None = None,
    ) -> None:
        effective_max_turns = budget.max_turns if budget is not None else max_turns
        if effective_max_turns is not None and effective_max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {effective_max_turns}")
        self._previous_answering_model: str | None = None
        self._provider = provider
        self._tools = tools
        self._policy = policy
        self._max_turns = effective_max_turns
        self._budget = budget
        self._call_class = call_class
        self._result_store = result_store
        self._agent_ref = agent_ref or new_agent_ref("agent")
        self._events = events
        self._transcript = transcript
        self._stream = stream
        self._read_ledger = read_ledger
        self._model_summary = model_summary
        self._notes_path = session_notes
        self._notes_lock = threading.Lock()
        self._notes_thread: threading.Thread | None = None
        self._notes_cancel: CancellationToken | None = None
        self._notes_closed = False
        self._notes_text = ""
        self._notes_coverage: tuple[str, ...] = ()
        self._notes_tokens = 0
        self._notes_calls = 0
        self._notes_last_calls = 0
        self._notes_usage: list[tuple[str, UsageTotals]] = []
        if session_notes is not None:
            try:
                with session_notes.open(encoding="utf-8", newline="") as source:
                    text = source.read(48001)
                if text.strip() and len(text) <= 48000:
                    self._notes_text = text
                    state = json.loads(session_notes.with_suffix(".coverage.json").read_text(encoding="utf-8"))
                    if (state.get("notes_digest") == hashlib.sha256(text.encode("utf-8")).hexdigest()
                            and isinstance(state.get("messages"), list)
                            and all(isinstance(item, str) for item in state["messages"])
                            and type(state.get("tokens")) is int):
                        self._notes_coverage = tuple(state["messages"])
                        self._notes_tokens = state["tokens"]
            except (OSError, UnicodeError, ValueError, AttributeError):
                pass
        self._context_budget = context_budget
        self._context_overflow = context_overflow
        self._context_recent_turns = context_recent_turns
        self._persisted_digests: list[str] = []
        # Schemas actually sent to the model so it knows these tools exist.
        # `tools` above is only the *execution* registry, keyed by name --
        # without this, a real provider is never told any tool exists and
        # can never call one, even if `tools` would happily execute it.
        self._tool_schemas = tool_schemas or []
        self._all_tool_schemas = self._tool_schemas
        self._last_request_schemas = None
        self._tool_search = None
        marked_names = set(deferrable_tools) | {name for name in tools if name.startswith("mcp__")}
        if marked_names:
            from symphonai_api.tools.tool_search import DeferredTool, SearchTools

            marked = {name: tools[name] for name in marked_names if name in tools}
            if marked:
                if "search_tools" in tools:
                    raise ValueError("search_tools is reserved for MCP tool discovery")
                history = tool_history or getattr(transcript, "path", None)
                self._tool_search = SearchTools(marked, history)
                self._tools = {name: (DeferredTool(tool, self._tool_search) if name in marked else tool)
                               for name, tool in tools.items()}

    def _write_notes_coverage(self, text, coverage, tokens) -> None:
        write_atomic(self._notes_path.with_suffix(".coverage.json"), json.dumps({
            "notes_digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "messages": coverage, "tokens": tokens,
        }))

    def close_session_notes(self) -> None:
        with self._notes_lock:
            self._notes_closed = True
            if self._notes_cancel is not None:
                self._notes_cancel.cancel()
            thread = self._notes_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def _session_notes_boundary(self, conversation, model, effort, run_id, calls, usage_by_model, record_usage):
        if self._notes_path is None:
            return
        with self._notes_lock:
            for key, usage in self._notes_usage:
                usage_by_model[key] = usage_by_model.get(key, UsageTotals()).merged(usage)
            self._notes_usage.clear()
            self._notes_calls += calls
            tokens = estimate_messages_tokens(conversation)
            if (self._notes_closed or (self._notes_thread is not None and self._notes_thread.is_alive())
                    or tokens < 10000
                    or (self._notes_text and (tokens - self._notes_tokens < 5000
                                             or self._notes_calls - self._notes_last_calls < 3))):
                return
            snapshot = tuple(conversation)
            current_notes = self._notes_text
            call_count = self._notes_calls
            token = CancellationToken()
            self._notes_cancel = token
            provider = self._provider
            tools = list(self._tool_schemas)
            path = self._notes_path

            def update():
                from symphonai_api.models import ImageBlock, TextBlock
                try:
                    # Match normal requests' latest-image policy without rewriting the snapshot.
                    request_messages = list(snapshot)
                    remaining = 5
                    for index in range(len(request_messages) - 1, -1, -1):
                        content = list(request_messages[index].content)
                        changed = False
                        for block_index in range(len(content) - 1, -1, -1):
                            if isinstance(content[block_index], ImageBlock):
                                if remaining:
                                    remaining -= 1
                                else:
                                    content[block_index] = TextBlock(
                                        "[an earlier image was removed from context; ask the person to attach it again if you need it]")
                                    changed = True
                        if changed:
                            request_messages[index] = replace(request_messages[index], content=content)
                    instructions = (
                        "Update this session's running notes from the conversation above. "
                        "Return the complete replacement notes in Markdown only; do not call tools. "
                        "Keep them under about 12,000 tokens, condensing older entries to stay there. "
                        "Use these sections: Current state (in progress, next steps); Task (the person's "
                        "request and decisions); Files and functions; Commands; Errors and corrections; "
                        "Learnings; Key results (exact outputs the person asked for); Work log (terse)."
                        "\n\nCurrent notes:\n" + current_notes
                    )
                    response = provider.create_response(ModelRequest(
                        messages=[*request_messages, Message(Role.USER, instructions)],
                        model=model, effort=effort, tools=tools, call_class=CallClass.BACKGROUND,
                    ), cancel=token)
                    key = model or provider.model or "unknown"
                    threshold = long_context_threshold(provider.wire_format, key)
                    usage = UsageTotals.from_usage(response.usage,
                        long_context=threshold is not None and response.usage.input_tokens > threshold)
                    record_usage(key, usage, turn_id=None)
                    with self._notes_lock:
                        self._notes_usage.append((key, usage))
                    text = response.message.text
                    if response.has_tool_calls or response.stop_reason == "refusal" or not text.strip() or len(text) > 48000:
                        return
                    with self._notes_lock:
                        if self._notes_closed or token.cancelled:
                            return
                        coverage = tuple(_message_digest(message) for message in snapshot)
                        # A hash mismatch after an interrupted pair of atomic writes fails closed.
                        self._write_notes_coverage(text, coverage, tokens)
                        write_atomic(path, text)
                        self._notes_text = text
                        self._notes_coverage = coverage
                        self._notes_tokens = tokens
                        self._notes_last_calls = call_count
                    emit(self._events, SessionNotesUpdated(
                        agent_id=self._agent_ref.agent_id, run_id=run_id,
                        session_id=path.parent.name,
                    ))
                except Exception:
                    # A failed background update must leave the turn and old notes intact.
                    pass
                finally:
                    token.close()

            self._notes_thread = threading.Thread(target=update, name="session-notes", daemon=True)
            self._notes_thread.start()

    def _compaction_recent_files(self, conversation: list[Message]) -> list[tuple[str, str]]:
        recent_files = []
        if self._read_ledger is not None:
            read_paths = {str((self._policy.repo_root / call.arguments["path"]).resolve())
                          for message in conversation for call in message.tool_calls
                          if call.name == "read_file" and isinstance(call.arguments.get("path"), str)}
            for message in conversation:
                if message.text.startswith("Restored after compaction:\n"):
                    try:
                        state = json.loads(message.text.split("\n", 1)[1])
                    except ValueError:
                        continue
                    if isinstance(state, dict):
                        read_paths.update(str((self._policy.repo_root / item["path"]).resolve())
                                          for item in state.get("recent_files", []))
            with self._read_ledger._lock:
                paths = [path for path in reversed(self._read_ledger._records)
                         if str(path) in read_paths][:5]
            for path in paths:
                if not self._policy.check_read(path).allowed:
                    continue
                try:
                    with path.open(encoding="utf-8") as source:
                        content = source.read(20001)
                    label = str(path.relative_to(self._policy.repo_root.resolve()))
                except (OSError, UnicodeError, ValueError):
                    continue
                recent_files.append((label, content))

        return recent_files

    def _compaction_notes_summary(self, conversation: list[Message], dropped: list[Message],
                                  *, cancel: CancellationToken | None = None) -> str | None:
        if self._notes_path is not None:
            with self._notes_lock:
                thread = self._notes_thread
            deadline = time.monotonic() + 15
            while thread is not None and thread.is_alive() and time.monotonic() < deadline:
                if cancel is not None:
                    cancel.raise_if_cancelled()
                thread.join(min(0.05, max(0, deadline - time.monotonic())))
            with self._notes_lock:
                covered = set()
                for digest, message in zip(self._notes_coverage, conversation):
                    if digest != _message_digest(message):
                        break
                    covered.add(id(message))
                if self._notes_text and all(id(message) in covered for message in dropped):
                    return self._notes_text
        return None

    def _compaction_notes_rebased(self, messages: list[Message], before_tokens: int,
                                  after_tokens: int, notes_summary: str | None) -> None:
        if self._notes_path is not None:
            with self._notes_lock:
                if notes_summary is not None and notes_summary == self._notes_text and any(message.text == (
                    "Summary of the earlier conversation, written when it was compacted:\n\n" + notes_summary
                ) for message in messages):
                    self._notes_coverage = tuple(_message_digest(message) for message in messages)
                self._notes_tokens = max(0, self._notes_tokens - (before_tokens - after_tokens))
                try:
                    self._write_notes_coverage(self._notes_text, self._notes_coverage, self._notes_tokens)
                except OSError:
                    pass

    @property
    def agent_ref(self) -> AgentRef:
        return self._agent_ref

    def _execute_tool_call(
        self,
        tool_call: ToolCall,
        cancel: CancellationToken | None,
        event_sink: EventSink | None,
        run_id: str,
        turn_id: str,
        hooks: HookRunner | None,
        *,
        catch_exceptions: bool = True,
    ) -> ToolResult:
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"unknown tool: {tool_call.name!r}",
            )
        try:
            with self._policy.event_context(
                event_sink,
                agent_id=self._agent_ref.agent_id,
                run_id=run_id,
                turn_id=turn_id,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
            ):
                hook_note = ""
                if hooks is not None:
                    from symphonai_api.hooks import HookInput
                    from symphonai_api.schema_validation import validate_schema

                    decision = hooks.pre_tool(
                        tool_call.name, tool_call.id, tool_call.arguments, self._agent_ref.name,
                        ask_model=lambda prompt, token, provider=self._provider, hook_model=self._previous_answering_model or self._provider.model: provider.create_response(ModelRequest(
                            messages=[Message(Role.USER, prompt)], model=hook_model, effort="low",
                        ), cancel=token), cancel=cancel,
                    )
                    if isinstance(decision, str):
                        return ToolResult(tool_call.id, ok=False, error=decision)
                    if isinstance(decision, HookInput):
                        if decision.updated_by:
                            error = validate_schema(decision.arguments, tool.parameters, "arguments")
                            if error is None:
                                error = tool.validate(decision.arguments)
                            if error is not None:
                                return ToolResult(tool_call.id, ok=False,
                                                  error=f"{decision.updated_by}: invalid updated_input: {error}")
                            tool_call = replace(tool_call, arguments=decision.arguments)
                        hook_note = decision.context
                result = tool.execute(tool_call, self._policy, cancel=cancel)
                return replace(result, content=result.content + f"\nHook note: {hook_note}") if hook_note else result
        except OperationCancelled:
            raise
        except Exception as exc:
            if not catch_exceptions:
                raise
            return ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def run(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        effort: str | None = None,
        parent_run_id: str | None = None,
        cancel: CancellationToken | None = None,
        run: AgentRun | None = None,
        events: EventSink | None = None,
        hooks: HookRunner | None = None,
    ) -> AgentRunResult:
        run_ref = new_run_ref(self._agent_ref.agent_id, parent_run_id)
        event_sink = events if events is not None else self._events
        conversation = list(messages)
        response: ModelResponse | None = None
        usage_by_model: dict[str, UsageTotals] = {}
        budget_state = (
            BudgetState(time.monotonic(), usage_by_model)
            if self._budget is not None or run is not None
            else None
        )
        requested_model = model or self._provider.model or "unknown"
        fallback_models = tuple(
            self._provider.fallback_models
        )
        turn = 0
        provider_calls = 0
        stop_continues = 0
        current_turn_ref: TurnRef | None = None
        reported_changes: dict[Path, int | None] = {}
        last_failure: tuple[str, str | None] | None = None
        repeated_failures = 0

        def append_record(
            record_type: str, *, turn_id: str | None, data: dict
        ) -> None:
            if self._transcript is not None:
                self._transcript.append(
                    record_type,
                    run_id=run_ref.run_id,
                    agent_id=self._agent_ref.agent_id,
                    turn_id=turn_id,
                    data=data,
                )

        def record_message(message: Message, *, turn_id: str | None) -> None:
            append_record("message", turn_id=turn_id, data=message_to_json(message))

        def persist_message(message: Message, *, turn_id: str | None) -> None:
            record_message(message, turn_id=turn_id)
            self._persisted_digests.append(_message_digest(message))

        def append_message(message: Message) -> None:
            conversation.append(message)
            persist_message(message, turn_id=message.turn_id)

        def persist_messages(items: list[Message], *, rewritten: bool = False) -> None:
            digests = [_message_digest(message) for message in items]
            kept = 0
            for old, new in zip(self._persisted_digests, digests):
                if old != new:
                    break
                kept += 1
            if rewritten or kept < len(self._persisted_digests):
                append_record(
                    "conversation_rewritten", turn_id=None,
                    data={"kept_prefix": kept, "replaced": len(self._persisted_digests) - kept},
                )
            for message in items[kept:]:
                record_message(message, turn_id=message.turn_id)
            self._persisted_digests = digests

        def record_usage(model: str, usage: UsageTotals, *, turn_id: str | None) -> None:
            append_record(
                "usage", turn_id=turn_id,
                data={"model": model, "agent_name": self._agent_ref.name, **usage.to_record()},
            )

        def finish_run(final_response: ModelResponse, stopped_reason: str, turns_used: int) -> AgentRunResult:
            result = AgentRunResult(
                final_response=final_response, messages=conversation, turns_used=turns_used,
                stopped_reason=stopped_reason, run=run_ref, agent=self._agent_ref,
                usage_by_model=usage_by_model,
            )
            try:
                emit(event_sink, RunFinished(
                    agent_id=self._agent_ref.agent_id, run_id=run_ref.run_id,
                    agent_name=self._agent_ref.name, stopped_reason=stopped_reason,
                ))
            except OperationCancelled:
                if stopped_reason != "cancelled":
                    raise
            data = {"stopped_reason": stopped_reason, "turns_used": turns_used}
            if stopped_reason == "refused":
                data["refusal_category"] = final_response.refusal_category
            append_record("run_finished", turn_id=None, data=data)
            return result

        def compact_conversation(
            budget: int, *, recent_turns: int | None = None, turn_id: str | None
        ) -> bool:
            recent_turns = (
                self._context_recent_turns if recent_turns is None else recent_turns
            )
            before = estimate_messages_tokens(conversation)
            recent_files = self._compaction_recent_files(conversation)

            notes_summary = None

            def summarize(dropped: list[Message]) -> str:
                nonlocal notes_summary
                notes_summary = self._compaction_notes_summary(conversation, dropped, cancel=cancel)
                if notes_summary is not None:
                    return notes_summary
                summary_request = ModelRequest(
                    messages=[
                        Message(role=Role.SYSTEM, content=MODEL_SUMMARY_PROMPT),
                        Message(role=Role.USER, content=render_dropped_messages(dropped)),
                    ],
                    model=model,
                    tools=[],
                    max_tokens=20_000,
                    call_class=self._call_class,
                )
                summary_response = (
                    self._provider.create_response(summary_request, cancel=cancel)
                )
                summary_model = model or self._provider.model or "unknown"
                threshold = long_context_threshold(self._provider.wire_format, summary_model)
                summary_usage = UsageTotals.from_usage(
                    summary_response.usage,
                    long_context=threshold is not None and summary_response.usage.input_tokens > threshold,
                )
                usage_by_model[summary_model] = usage_by_model.get(
                    summary_model, UsageTotals()
                ).merged(summary_usage)
                record_usage(summary_model, summary_usage, turn_id=turn_id)
                return summary_response.message.text

            errors: list[ContextCompactionError] = []
            result = None
            assistant_turn_window = sum(
                message.role == Role.USER for message in conversation
            ) < recent_turns
            for keep_turns in range(recent_turns, 0, -1):
                try:
                    result = compact_messages_for_budget(
                        conversation,
                        budget=budget,
                        recent_turns=keep_turns,
                        assistant_turn_window=assistant_turn_window,
                        cancel=cancel,
                        recent_files=recent_files,
                        summarize=summarize if self._model_summary or self._notes_path is not None else None,
                    )
                    break
                except ContextCompactionError as exc:
                    errors.append(exc)
            if result is None:
                raise errors[-1]
            if not result.changed:
                return False
            rewritten = []
            for message in result.messages:
                calls = [
                    replace(call, provider_metadata={
                        key: value for key, value in call.provider_metadata.items()
                        if key != "anthropic_content"
                    })
                    for call in message.tool_calls
                ]
                rewritten.append(
                    replace(message, tool_calls=calls)
                    if calls != message.tool_calls else message
                )
            persist_messages(rewritten, rewritten=True)
            conversation[:] = rewritten
            self._compaction_notes_rebased(rewritten, before, result.after_tokens, notes_summary)
            append_record(
                "compaction",
                turn_id=turn_id,
                data={
                    "before_tokens": before,
                    "after_tokens": result.after_tokens,
                    "dropped_messages": result.dropped_messages,
                },
            )
            emit(
                event_sink,
                CompactionApplied(
                    agent_id=self._agent_ref.agent_id,
                    run_id=run_ref.run_id,
                    before_tokens=before,
                    after_tokens=result.after_tokens,
                    dropped_messages=result.dropped_messages,
                ),
            )
            return True

        def append_redirects(*, close: bool = False) -> None:
            if run is None:
                return
            redirects = run.close_redirects() if close else run.take_redirects()
            for text in redirects:
                redirect = Message(role=Role.USER, content=text)
                append_message(redirect)

        def append_outside_change_notice() -> None:
            if self._read_ledger is None:
                return
            changed = [
                (path, version)
                for path, version in self._read_ledger.changed_files()
                if path not in reported_changes or reported_changes[path] != version
            ]
            if not changed:
                return
            changed.sort(key=lambda item: item[0].relative_to(self._policy.repo_root).as_posix())
            paths = [
                path.relative_to(self._policy.repo_root).as_posix()
                + (" (deleted)" if version is None else "")
                for path, version in changed
            ]
            shown = paths[:10]
            if len(paths) > 10:
                shown.append(f"and {len(paths) - 10} more")
            notice = (
                "These files changed on disk since you last read them, not through "
                "your edits: " + ", ".join(shown)
                + ". Read them again before relying on or editing them."
            )
            message = Message(role=Role.USER, content=notice)
            append_message(message)
            reported_changes.update(changed)

        def append_repaired_tool_results(
            tool_call_ids: list[str], turn_id: str | None
        ) -> None:
            for repaired_message in conversation[-len(tool_call_ids):] if tool_call_ids else ():
                persist_message(repaired_message, turn_id=turn_id)

        def finish_for_budget(stopped_reason: str) -> AgentRunResult:
            append_redirects(close=True)
            final_response = response or ModelResponse(
                message=Message(role=Role.ASSISTANT), stop_reason=stopped_reason
            )
            return finish_run(final_response, stopped_reason, provider_calls)

        def exceeded_budget(budget: RunBudget | None = None) -> str | None:
            effective_budget = self._budget if budget is None else budget
            if budget_state is None or effective_budget is None:
                return None
            return budget_state.exceeded(effective_budget, requested_model)

        def compact_for_request(turn_ref: TurnRef) -> None:
            context_budget = (
                self._context_budget(model)
                if self._context_budget is not None
                else budget_for_model(self._provider.wire_format, model)
            )
            if estimate_messages_tokens(conversation) > context_budget:
                compact_conversation(context_budget, turn_id=turn_ref.turn_id)

        def request_response(turn_ref: TurnRef) -> str:
            nonlocal response, model, effort
            from symphonai_api.models import ImageBlock, TextBlock
            for request_attempt in range(2):
                request_tools = self._tool_schemas
                if self._tool_search is not None:
                    if self._tool_schemas is not self._last_request_schemas:
                        self._all_tool_schemas = self._tool_schemas
                    self._tool_search.restore_messages(conversation)
                    context_budget = (self._context_budget(model) if self._context_budget is not None
                                      else budget_for_model(self._provider.wire_format, model))
                    request_tools = self._tool_search.request_schemas(
                        self._all_tool_schemas, self._provider.wire_format, context_budget,
                    )
                    self._tool_schemas = self._last_request_schemas = request_tools
                    if self._tool_search.active:
                        self._tools["search_tools"] = self._tool_search
                    else:
                        self._tools.pop("search_tools", None)
                compact_for_request(turn_ref)
                request_messages = list(conversation)
                if self._tool_search is not None and self._tool_search.active:
                    request_messages.insert(0, Message(Role.SYSTEM,
                        "More MCP tools are available; use search_tools to find and load them."))
                remaining_images = 5
                for index in range(len(request_messages) - 1, -1, -1):
                    message = request_messages[index]
                    content = list(message.content)
                    changed = False
                    for block_index in range(len(content) - 1, -1, -1):
                        if not isinstance(content[block_index], ImageBlock):
                            continue
                        if remaining_images:
                            remaining_images -= 1
                        else:
                            content[block_index] = TextBlock(
                                "[an earlier image was removed from context; ask the person to attach it again if you need it]"
                            )
                            changed = True
                    if changed:
                        request_messages[index] = replace(message, content=content)
                request = ModelRequest(
                    messages=request_messages,
                    model=model,
                    tools=request_tools,
                    effort=effort,
                    call_class=self._call_class,
                )
                append_record(
                    "request",
                    turn_id=turn_ref.turn_id,
                    data={
                        "model": model,
                        "message_count": len(request.messages),
                        "tool_names": sorted(self._tools),
                    },
                )
                try:
                    primary_model = request.model or self._provider.model
                    fallback_chain = list(fallback_models)
                    if primary_model in fallback_chain:
                        fallback_chain = fallback_chain[fallback_chain.index(primary_model) + 1:]
                    attempted_model = primary_model
                    candidate_models = [request.model, *(
                        candidate for candidate in fallback_chain
                        if candidate != primary_model
                    )]
                    unavailable: ProviderUnavailableError | None = None
                    used_model = primary_model or "unknown"
                    for candidate_model in candidate_models:
                        candidate_request = request
                        candidate_effective_model = candidate_model or primary_model
                        if candidate_model is not None and candidate_model != request.model:
                            fallback_messages = [
                                replace(
                                    message,
                                    tool_calls=[
                                        replace(call, provider_metadata={}, vendor_id=None)
                                        for call in message.tool_calls
                                    ],
                                ) if message.tool_calls else message
                                for message in request.messages
                            ]
                            from symphonai_api.model_table import model_capabilities

                            capability = next((
                                item for item in model_capabilities()
                                if item.wire_format == self._provider.wire_format
                                and item.model == candidate_model
                            ), None)
                            effort = request.effort
                            if capability is None or not any(
                                option.id == effort for option in capability.efforts
                            ):
                                effort = None
                            candidate_request = replace(
                                request,
                                model=candidate_model,
                                messages=fallback_messages,
                                effort=effort,
                            )
                        try:
                            if self._stream:
                                assembler = StreamAssembler()
                                chunks = (
                                    self._provider.create_response_stream(candidate_request, cancel=cancel)
                                )
                                for chunk in chunks:
                                    if cancel is not None:
                                        cancel.raise_if_cancelled()
                                    if isinstance(chunk, TextDelta):
                                        emit(
                                            event_sink,
                                            AssistantTextDelta(
                                                agent_id=self._agent_ref.agent_id,
                                                run_id=run_ref.run_id,
                                                turn_id=turn_ref.turn_id,
                                                text=chunk.text,
                                            ),
                                        )
                                    elif isinstance(chunk, ReasoningDelta):
                                        emit(
                                            event_sink,
                                            AssistantReasoningDelta(
                                                agent_id=self._agent_ref.agent_id,
                                                run_id=run_ref.run_id,
                                                turn_id=turn_ref.turn_id,
                                                text=chunk.text,
                                            ),
                                        )
                                    assembler.add(chunk)
                                response = assembler.finish()
                            else:
                                response = self._provider.create_response(
                                    candidate_request, cancel=cancel
                                )
                        except ProviderUnavailableError as exc:
                            unavailable = exc
                            attempted_model = candidate_effective_model
                            continue
                        used_model = candidate_effective_model or "unknown"
                        if candidate_effective_model != primary_model:
                            emit(
                                event_sink,
                                ModelFallback(
                                    agent_id=self._agent_ref.agent_id,
                                    run_id=run_ref.run_id,
                                    turn_id=turn_ref.turn_id,
                                    from_model=attempted_model or "unknown",
                                    to_model=used_model,
                                    reason=(
                                        "network" if unavailable.status is None
                                        else str(unavailable.status)
                                    ),
                                ),
                            )
                            model = candidate_effective_model
                        break
                    else:
                        if unavailable is not None:
                            raise unavailable
                except ContextLengthExceededError as exc:
                    request_tokens = estimate_messages_tokens(request.messages)
                    exc.request_tokens = request_tokens
                    if self._context_overflow is not None:
                        self._context_overflow(request_tokens, exc.actual_tokens)
                    if request_attempt:
                        raise
                    budget = (
                        self._context_budget(model)
                        if self._context_budget is not None
                        else budget_for_model(self._provider.wire_format, model)
                    )
                    forced_budget = min(
                        budget,
                        max(1, request_tokens - 1)
                        if exc.actual_tokens is not None
                        else max(1, request_tokens * 3 // 4),
                    )
                    try:
                        changed = compact_conversation(
                            forced_budget, turn_id=turn_ref.turn_id
                        )
                    except ContextCompactionError:
                        raise exc from None
                    if not changed:
                        raise
                    continue
                break
            else:
                raise RuntimeError("request retry loop did not produce a response")
            return used_model

        def record_response_usage(turn_ref: TurnRef, used_model: str) -> None:
            nonlocal provider_calls
            dated_suffix = (response.model or "").removeprefix(used_model + "-")
            same_model = response.model == used_model or (
                response.model is not None and response.model.startswith(used_model + "-")
                and len(dated_suffix) == 8 and dated_suffix.isascii() and dated_suffix.isdigit()
            )
            if response.model and not same_model:
                if response.model != used_model and response.model != self._previous_answering_model:
                    emit(event_sink, ModelFallback(
                        agent_id=self._agent_ref.agent_id, run_id=run_ref.run_id,
                        turn_id=turn_ref.turn_id, from_model=used_model,
                        to_model=response.model, reason="refusal",
                    ))
                used_model = response.model
            self._previous_answering_model = used_model
            provider_calls += 1
            threshold = long_context_threshold(self._provider.wire_format, used_model)
            usage = UsageTotals.from_usage(
                response.usage,
                long_context=threshold is not None and response.usage.input_tokens > threshold,
            )
            usage_by_model[used_model] = usage_by_model.get(
                used_model, UsageTotals()
            ).merged(usage)
            record_usage(used_model, usage, turn_id=turn_ref.turn_id)

        def handle_response(turn_ref: TurnRef) -> bool:
            nonlocal response
            refused = response.stop_reason == "refusal"
            if refused:
                response = replace(response, message=replace(
                    response.message, content="", tool_calls=[], reasoning="",
                ))
            # Stamp once and reuse, so the message in `conversation` and
            # `final_response` carry the same turn identity.
            response = replace(
                response, message=replace(response.message, turn_id=turn_ref.turn_id)
            )
            if not refused:
                append_message(response.message)
            if cancel is not None:
                cancel.raise_if_cancelled()
            return refused

        def run_tools(turn_ref: TurnRef) -> None:
            nonlocal last_failure, repeated_failures
            cut_call = next(
                (
                    call
                    for call in response.message.tool_calls
                    if call.id == response.cut_tool_call_id
                ),
                None,
            )
            executable_calls = [
                call
                for call in response.message.tool_calls
                if call.id != response.cut_tool_call_id
            ]
            batches = partition_tool_calls(executable_calls, self._tools)
            if cut_call is not None:
                batches.append([cut_call])
            for batch in batches:
                for tool_call in batch:
                    append_record(
                        "tool_started",
                        turn_id=turn_ref.turn_id,
                        data={
                            "tool_call_id": tool_call.id,
                            "tool_name": tool_call.name,
                        },
                    )
                    emit(
                        event_sink,
                        ToolCallStarted(
                            agent_id=self._agent_ref.agent_id,
                            run_id=run_ref.run_id,
                            turn_id=turn_ref.turn_id,
                            tool_name=tool_call.name,
                            tool_call_id=tool_call.id,
                            target=call_target(
                                tool_call.name,
                                tool_call.arguments,
                            ),
                        ),
                    )

                cancelled = False
                results: dict[int, ToolResult] = {}
                if batch[0].id == response.cut_tool_call_id:
                    results[0] = ToolResult(
                        tool_call_id=batch[0].id,
                        ok=False,
                        error=_CUT_TOOL_CALL_ERROR,
                    )
                elif len(batch) == 1:
                    tool_call = batch[0]
                    results[0] = self._execute_tool_call(
                        tool_call,
                        cancel,
                        event_sink,
                        run_ref.run_id,
                        turn_ref.turn_id,
                        hooks,
                        catch_exceptions=False,
                    )
                else:
                    executor = ThreadPoolExecutor(
                        max_workers=min(len(batch), MAX_TOOL_CONCURRENCY)
                    )
                    futures = {
                        executor.submit(
                            self._execute_tool_call,
                            tool_call,
                            cancel,
                            event_sink,
                            run_ref.run_id,
                            turn_ref.turn_id,
                            hooks,
                        ): index
                        for index, tool_call in enumerate(batch)
                    }
                    try:
                        for future in as_completed(futures):
                            index = futures[future]
                            try:
                                results[index] = future.result()
                            except OperationCancelled:
                                cancelled = True
                                for pending in futures:
                                    pending.cancel()
                            except CancelledError:
                                pass
                    finally:
                        executor.shutdown(wait=True, cancel_futures=cancelled)

                for index, tool_call in enumerate(batch):
                    tool_result = results.get(index)
                    if tool_result is None:
                        continue
                    if tool_result.ok:
                        last_failure = None
                        repeated_failures = 0
                    else:
                        failure = (tool_call.name, tool_result.error)
                        repeated_failures = (
                            repeated_failures + 1 if failure == last_failure else 1
                        )
                        last_failure = failure
                        if repeated_failures >= 3:
                            tool_result = replace(
                                tool_result,
                                error=(
                                    f"{tool_result.error or ''} This has now failed "
                                    f"{repeated_failures} times in a row with the same error. "
                                    "Do not send it again; change approach, or tell the "
                                    "person what is blocking you."
                                ),
                            )
                    tool_result = mask_tool_result(tool_result)
                    if self._result_store is not None:
                        tool_result = offload_tool_result(
                            tool_result,
                            tool_name=tool_call.name,
                            store=self._result_store,
                        )
                    tool_message = Message(
                        role=Role.TOOL,
                        tool_result=tool_result,
                        turn_id=turn_ref.turn_id,
                    )
                    append_message(tool_message)
                    emit(
                        event_sink,
                        ToolCallFinished(
                            agent_id=self._agent_ref.agent_id,
                            run_id=run_ref.run_id,
                            turn_id=turn_ref.turn_id,
                            tool_name=tool_call.name,
                            tool_call_id=tool_call.id,
                            ok=tool_result.ok,
                            **_tool_result_event_fields(tool_result, tool_call.name),
                        ),
                    )
                    if not tool_result.ok:
                        emit(
                            event_sink,
                            ToolCallFailed(
                                agent_id=self._agent_ref.agent_id,
                                run_id=run_ref.run_id,
                                turn_id=turn_ref.turn_id,
                                tool_name=tool_call.name,
                                tool_call_id=tool_call.id,
                                error=tool_result.error or "tool call failed",
                            ),
                        )
                    append_record(
                        "tool_result",
                        turn_id=turn_ref.turn_id,
                        data={
                            "tool_call_id": tool_call.id,
                            "tool_name": tool_call.name,
                            "ok": tool_result.ok,
                            "cancelled": tool_result.cancelled,
                            "offloaded_id": (
                                None
                                if tool_result.offloaded is None
                                else tool_result.offloaded.id
                            ),
                        },
                    )
                if cancelled:
                    raise OperationCancelled

        def finish_turn(turn_ref: TurnRef) -> None:
            append_record(
                "turn_finished", turn_id=turn_ref.turn_id, data={"index": turn}
            )
            emit(
                event_sink,
                TurnFinished(
                    agent_id=self._agent_ref.agent_id,
                    run_id=run_ref.run_id,
                    turn_id=turn_ref.turn_id,
                    index=turn,
                ),
            )

        def start_turn() -> TurnRef:
            nonlocal current_turn_ref
            turn_ref = new_turn_ref(run_ref.run_id, turn)
            current_turn_ref = turn_ref
            append_record(
                "turn_started", turn_id=turn_ref.turn_id, data={"index": turn}
            )
            emit(
                event_sink,
                TurnStarted(
                    agent_id=self._agent_ref.agent_id,
                    run_id=run_ref.run_id,
                    turn_id=turn_ref.turn_id,
                    index=turn,
                ),
            )
            return turn_ref

        def take_redirects_and_check_budget() -> str | None:
            if cancel is not None:
                cancel.raise_if_cancelled()
            boundary_budget = self._budget
            if run is not None:
                append_redirects()
                boundary_budget = run.budget
                if boundary_budget.max_turns is not None and boundary_budget.max_turns <= provider_calls:
                    return "budget_turns"
            budget_reason = exceeded_budget(boundary_budget)
            if budget_reason is not None:
                return budget_reason
            return None

        def start_run() -> None:
            append_record(
                "run_started",
                turn_id=None,
                data={
                    "agent_name": self._agent_ref.name,
                    "parent_run_id": parent_run_id,
                    "model": requested_model,
                },
            )
            persist_messages(messages)
            emit(
                event_sink,
                RunStarted(
                    agent_id=self._agent_ref.agent_id,
                    run_id=run_ref.run_id,
                    agent_name=self._agent_ref.name,
                ),
            )
            submitted = messages[-1].text if messages else ""
            emit(
                event_sink,
                PromptSubmitted(
                    agent_id=self._agent_ref.agent_id,
                    run_id=run_ref.run_id,
                    text=submitted,
                    message_count=max(len(messages) - 1, 0),
                ),
            )

        try:
            start_run()
            turn = 0
            while self._max_turns is None or turn < self._max_turns:
                turn += 1
                turn_ref = start_turn()
                budget_reason = take_redirects_and_check_budget()
                if budget_reason is not None:
                    return finish_for_budget(budget_reason)
                if provider_calls > 0:
                    append_outside_change_notice()
                used_model = request_response(turn_ref)
                record_response_usage(turn_ref, used_model)
                refused = handle_response(turn_ref)
                if not response.has_tool_calls:
                    self._session_notes_boundary(conversation, model, effort, run_ref.run_id, 0,
                                                 usage_by_model, record_usage)
                    finish_turn(turn_ref)
                    if not refused and run is not None:
                        run_limit = run.budget.max_turns
                        redirect_limit = (
                            run_limit if self._max_turns is None
                            else self._max_turns if run_limit is None
                            else min(self._max_turns, run_limit)
                        )
                        if redirect_limit is None or turn < redirect_limit:
                            if run.wait_for_background is not None and exceeded_budget() is None:
                                run.wait_for_background(lambda: exceeded_budget() is not None)
                                budget_reason = exceeded_budget()
                                if budget_reason is not None:
                                    return finish_for_budget(budget_reason)
                            if run.has_redirects():
                                continue
                    if (not refused and hooks is not None and exceeded_budget() is None
                        and (cancel is None or not cancel.cancelled)
                        and (run is None or redirect_limit is None or turn < redirect_limit)):
                        writer = getattr(self._transcript, "_writer", self._transcript)
                        transcript_path = getattr(writer, "path", None)
                        reason = hooks.stop(
                            agent_name=self._agent_ref.name,
                            session_id=None if transcript_path is None else transcript_path.parent.name,
                            final_text=response.message.text, turns_used=provider_calls,
                            stop_continues=stop_continues,
                            ask_model=lambda prompt, token, provider=self._provider, hook_model=self._previous_answering_model or model: provider.create_response(ModelRequest(
                                messages=[Message(Role.USER, prompt)], model=hook_model, effort="low",
                            ), cancel=token), cancel=cancel,
                        )
                        if reason is not None:
                            if stop_continues < 3 and (self._max_turns is None or turn < self._max_turns):
                                stop_continues += 1
                                append_message(Message(Role.USER, "[Stop hook] " + reason))
                                continue
                            if stop_continues >= 3:
                                note = "\n\n(Stop hook asked to continue; limit of 3 reached)"
                                response = replace(response, message=replace(
                                    response.message, content=[*response.message.content, note],
                                ))
                                conversation[-1] = response.message
                                persist_messages(conversation, rewritten=True)
                                emit(event_sink, AssistantTextDelta(
                                    agent_id=self._agent_ref.agent_id, run_id=run_ref.run_id,
                                    turn_id=turn_ref.turn_id, text=note,
                                ))
                    append_redirects(close=True)
                    return finish_run(response, "refused" if refused else "final_response", turn)
                run_tools(turn_ref)
                self._session_notes_boundary(conversation, model, effort, run_ref.run_id,
                                             len(response.message.tool_calls), usage_by_model, record_usage)
                # Checked once every batch of this turn has been answered, not
                # between batches: a stop with a tool call still outstanding
                # would return a conversation no provider will accept, and
                # unlike the cancellation path below there is nothing here to
                # repair it.
                if cancel is not None:
                    cancel.raise_if_cancelled()
                budget_reason = exceeded_budget()
                if budget_reason is not None:
                    return finish_for_budget(budget_reason)
                finish_turn(turn_ref)
            assert response is not None and self._max_turns is not None
            append_redirects(close=True)
            return finish_run(response, "max_turns", turn)
        except OperationCancelled:
            repair_turn_id = (
                None if current_turn_ref is None else current_turn_ref.turn_id
            )
            repaired_tool_call_ids = repair_unanswered_tool_calls(
                conversation,
                error="cancelled before this tool completed",
                cancelled=True,
                turn_id=repair_turn_id,
            )
            append_repaired_tool_results(repaired_tool_call_ids, repair_turn_id)
            append_redirects(close=True)
            append_record(
                "cancellation",
                turn_id=(
                    None if current_turn_ref is None else current_turn_ref.turn_id
                ),
                data={"repaired_tool_call_ids": repaired_tool_call_ids},
            )
            cancelled_response = response or ModelResponse(
                message=Message(role=Role.ASSISTANT), stop_reason="cancelled"
            )
            return finish_run(cancelled_response, "cancelled", max(turn, 1))
        except Exception as exc:
            repair_turn_id = (
                None if current_turn_ref is None else current_turn_ref.turn_id
            )
            repaired_tool_call_ids = repair_unanswered_tool_calls(
                conversation,
                error="the run failed before this tool completed",
                cancelled=False,
                turn_id=repair_turn_id,
            )
            append_repaired_tool_results(repaired_tool_call_ids, repair_turn_id)
            append_redirects(close=True)
            append_record(
                "run_failed", turn_id=None, data={"error": str(exc)}
            )
            emit(
                event_sink,
                RunFailed(
                    agent_id=self._agent_ref.agent_id,
                    run_id=run_ref.run_id,
                    agent_name=self._agent_ref.name,
                    error=str(exc),
                ),
            )
            raise
