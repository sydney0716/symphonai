"""Checks for live run pause, resume, and paused cancellation."""

from __future__ import annotations

import ast
import tempfile
import threading
from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import symphonai_api.agent_loop as agent_loop_module
import symphonai_api.agent_run as agent_run_module
from symphonai_api.agent_loop import ApiAgent
from symphonai_api.agent_run import AgentRun, PauseGate, RunPhase, new_agent_run
from symphonai_api.agent_spec import AgentSpec, ModelSelector
from symphonai_api.budgets import RunBudget
from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.cost import ModelPrice, PriceTable
from symphonai_api.events import CollectingSink
from symphonai_api.identity import AgentRef, RunRef, TurnRef
from symphonai_api.models import Message, ModelResponse, Role, ToolCall
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from scripts.checks.harness import check, fail


_DEFAULT_BASELINE_COMMIT = "ff3163a"
_CONTROL_BASELINE_COMMIT = "ff3163a"
_DEFAULT_PRE_08D = (
    "final_response",
    1,
    "end_turn",
    (
        ("user", "frozen prompt", None),
        ("assistant", "frozen answer", "turn-fixed"),
    ),
    ("assistant", "frozen answer", "turn-fixed"),
    ("run-fixed", "agent-fixed", None, 1),
    ("agent-fixed", "agent", None, 1),
    (("unknown", 0, 0, 1),),
    (
        ("RunStarted", None, None, "agent", None),
        ("PromptSubmitted", None, None, None, None),
        ("TurnStarted", "turn-fixed", 1, None, None),
        ("TurnFinished", "turn-fixed", 1, None, None),
        ("RunFinished", None, None, "agent", "final_response"),
    ),
    1,
)


def _new_run(root: Path, *, budget: RunBudget | None = None):
    return new_agent_run(
        AgentSpec(
            "worker",
            "",
            ModelSelector("fake"),
            PermissionPolicy(repo_root=root),
            budget=budget,
        )
    )


def _finish_result() -> SimpleNamespace:
    return SimpleNamespace(
        turns_used=1,
        usage_by_model={},
        stopped_reason="final_response",
    )


def _action(run, method: str):
    return {
        "start": lambda: run.start(CancellationToken()),
        "pause": run.pause,
        "resume": run.resume,
        "finish": lambda: run.finish(_finish_result()),
        "fail": lambda: run.fail("broken"),
        "cancel": run.cancel,
    }[method]


class _TrackingToken(CancellationToken):
    def __init__(self) -> None:
        super().__init__()
        self.listener_registered = threading.Event()
        self.unsubscribe_calls = 0

    def on_cancel(self, callback):
        unsubscribe = super().on_cancel(callback)
        self.listener_registered.set()

        def tracked_unsubscribe() -> None:
            self.unsubscribe_calls += 1
            unsubscribe()

        return tracked_unsubscribe


def _listener_count(token: CancellationToken) -> int:
    with token._lock:
        return len(token._callbacks)


def _start_waiter(gate: PauseGate, token: CancellationToken | None = None):
    done = threading.Event()
    errors: list[BaseException] = []

    def wait() -> None:
        try:
            gate.wait_while_paused(token)
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    return done, errors, thread


def _start_agent(
    agent: ApiAgent,
    gate: PauseGate | None,
    token: CancellationToken,
    *,
    controlled_run: AgentRun | None = None,
):
    done = threading.Event()
    results = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(
                agent.run(
                    [Message(Role.USER, "task")],
                    cancel=token,
                    pause=gate,
                    run=controlled_run,
                )
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return done, results, errors, thread


@check("run_control.phase_transitions")
def phase_transitions() -> None:
    allowed = {
        "start": (RunPhase.PENDING,),
        "pause": (RunPhase.RUNNING,),
        "resume": (RunPhase.PAUSED,),
        "finish": (RunPhase.RUNNING,),
        "fail": (RunPhase.RUNNING, RunPhase.PAUSED),
        "cancel": (RunPhase.PENDING, RunPhase.RUNNING, RunPhase.PAUSED),
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        if RunPhase.PAUSED.value != "paused":
            fail("RunPhase.PAUSED has the wrong value")
        for method, legal_phases in allowed.items():
            for phase in RunPhase:
                if phase in legal_phases:
                    continue
                run = _new_run(root)
                run.phase = phase
                try:
                    _action(run, method)()
                except ValueError as exc:
                    message = str(exc)
                else:
                    fail(f"{method} was allowed from {phase.value}")
                for expected in (method, phase.value, *(item.value for item in legal_phases)):
                    if expected not in message:
                        fail(f"{method} error omitted {expected!r}: {message!r}")

        paused = _new_run(root)
        paused.start(CancellationToken())
        paused.pause()
        if paused.phase is not RunPhase.PAUSED:
            fail("pause did not move RUNNING to PAUSED")
        paused.resume()
        if paused.phase is not RunPhase.RUNNING:
            fail("resume did not move PAUSED to RUNNING")

        cancelled = _new_run(root)
        token = CancellationToken()
        cancelled.start(token)
        cancelled.pause()
        cancelled.cancel()
        if cancelled.phase is not RunPhase.CANCELLED or not token.cancelled:
            fail("a paused run did not remain terminable")

        failed = _new_run(root)
        failed.start(CancellationToken())
        failed.pause()
        failed.fail("broken")
        if failed.phase is not RunPhase.FAILED or failed.error != "broken":
            fail("a paused run could not fail")

        heartbeat = _new_run(root)
        with mock.patch.object(
            agent_run_module.time,
            "monotonic",
            side_effect=[10.0, 20.0, 23.0],
        ):
            heartbeat.start(CancellationToken())
            heartbeat.pause()
            heartbeat.heartbeat()
            if heartbeat.phase is not RunPhase.PAUSED or heartbeat.quiet_seconds != 3.0:
                fail("paused heartbeat did not update quiet_seconds")


@check("run_control.gate_blocks_and_releases")
def gate_blocks_and_releases() -> None:
    gate = PauseGate()
    immediate, immediate_errors, immediate_thread = _start_waiter(gate)
    if not immediate.wait(1.0):
        fail("an unpaused gate did not return immediately")
    immediate_thread.join(timeout=1.0)
    if immediate_errors:
        fail(f"an unpaused gate raised: {immediate_errors!r}")

    gate.pause()
    token = _TrackingToken()
    done, errors, thread = _start_waiter(gate, token)
    if not token.listener_registered.wait(1.0):
        fail("paused waiter did not register its cancellation listener")
    if done.wait(0):
        fail("paused gate returned before resume")
    gate.resume()
    if not done.wait(1.0):
        fail("paused gate did not return after resume")
    thread.join(timeout=1.0)
    if errors:
        fail(f"resumed gate raised: {errors!r}")


@check("run_control.paused_run_stays_terminable")
def paused_run_stays_terminable() -> None:
    gate = PauseGate()
    gate.pause()
    token = _TrackingToken()
    done, errors, thread = _start_waiter(gate, token)
    if not token.listener_registered.wait(1.0):
        fail("paused waiter did not register before cancellation")
    token.cancel()
    if not done.wait(1.0):
        fail("cancellation did not wake a paused waiter")
    thread.join(timeout=1.0)
    if len(errors) != 1 or not isinstance(errors[0], OperationCancelled):
        fail(f"paused waiter did not raise OperationCancelled: {errors!r}")

    class CountingGate(PauseGate):
        def __init__(self) -> None:
            super().__init__()
            self.wait_calls = 0

        def wait_while_paused(self, cancel=None) -> None:
            self.wait_calls += 1
            super().wait_while_paused(cancel)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        pre_cancelled_gate = CountingGate()
        pre_cancelled_gate.pause()
        pre_cancelled = CancellationToken()
        pre_cancelled.cancel()
        pre_provider = FakeModelProvider()
        pre_result = ApiAgent(
            pre_provider,
            {},
            PermissionPolicy(repo_root=root),
        ).run(
            [Message(Role.USER, "task")],
            cancel=pre_cancelled,
            pause=pre_cancelled_gate,
        )
        if (
            pre_result.stopped_reason != "cancelled"
            or pre_provider.call_count != 0
            or pre_cancelled_gate.wait_calls != 0
        ):
            fail("the park point ran before the initial cancellation check")

        live_gate = PauseGate()
        live_gate.pause()
        live_token = _TrackingToken()
        live_provider = FakeModelProvider()
        live_agent = ApiAgent(
            live_provider,
            {},
            PermissionPolicy(repo_root=root),
        )
        live_done, results, live_errors, live_thread = _start_agent(
            live_agent,
            live_gate,
            live_token,
        )
        if not live_token.listener_registered.wait(1.0):
            fail("agent run did not park before cancellation")
        live_token.cancel()
        if not live_done.wait(1.0):
            fail("agent run did not terminate while paused")
        live_thread.join(timeout=1.0)
        if (
            live_errors
            or len(results) != 1
            or results[0].stopped_reason != "cancelled"
            or live_provider.call_count != 0
        ):
            fail("cancelled paused run called the provider or ended incorrectly")


@check("run_control.gate_leaks_no_listeners")
def gate_leaks_no_listeners() -> None:
    resume_gate = PauseGate()
    resume_token = _TrackingToken()
    starting = _listener_count(resume_token)
    for _ in range(20):
        resume_gate.pause()
        resume_token.listener_registered.clear()
        done, errors, thread = _start_waiter(resume_gate, resume_token)
        if not resume_token.listener_registered.wait(1.0):
            fail("resume-cycle listener was not registered")
        resume_gate.resume()
        if not done.wait(1.0):
            fail("resume-cycle waiter did not finish")
        thread.join(timeout=1.0)
        if errors or _listener_count(resume_token) != starting:
            fail("resume cycle leaked a cancellation listener")
    if resume_token.unsubscribe_calls != 20:
        fail("resume path did not unsubscribe every listener")

    for _ in range(20):
        cancel_gate = PauseGate()
        cancel_gate.pause()
        cancel_token = _TrackingToken()
        starting = _listener_count(cancel_token)
        done, errors, thread = _start_waiter(cancel_gate, cancel_token)
        if not cancel_token.listener_registered.wait(1.0):
            fail("cancel-cycle listener was not registered")
        cancel_token.cancel()
        if not done.wait(1.0):
            fail("cancel-cycle waiter did not finish")
        thread.join(timeout=1.0)
        if (
            len(errors) != 1
            or not isinstance(errors[0], OperationCancelled)
            or _listener_count(cancel_token) != starting
        ):
            fail("cancel cycle leaked a listener or lost cancellation")
        if cancel_token.unsubscribe_calls != 1:
            fail("cancellation path did not invoke its unsubscriber")


def _default_output(result, sink: CollectingSink, provider: FakeModelProvider):
    messages = tuple(
        (message.role.value, message.text, message.turn_id)
        for message in result.messages
    )
    final = result.final_response.message
    usage = tuple(
        (name, totals.input_tokens, totals.output_tokens, totals.calls)
        for name, totals in sorted(result.usage_by_model.items())
    )
    events = tuple(
        (
            type(event).__name__,
            event.turn_id,
            getattr(event, "index", None),
            getattr(event, "agent_name", None),
            getattr(event, "stopped_reason", None),
        )
        for event in sink.events
    )
    return (
        result.stopped_reason,
        result.turns_used,
        result.final_response.stop_reason,
        messages,
        (final.role.value, final.text, final.turn_id),
        (
            result.run.run_id,
            result.run.agent_id,
            result.run.parent_run_id,
            result.run.schema_version,
        ),
        (
            result.agent.agent_id,
            result.agent.name,
            result.agent.parent_agent_id,
            result.agent.schema_version,
        ),
        usage,
        events,
        provider.call_count,
    )


@check("run_control.default_is_unchanged")
def default_is_unchanged() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        provider = FakeModelProvider(
            [ModelResponse(Message(Role.ASSISTANT, "frozen answer"))]
        )
        sink = CollectingSink()
        agent = ApiAgent(
            provider,
            {},
            PermissionPolicy(repo_root=Path(temporary)),
            agent_ref=AgentRef("agent-fixed", "agent"),
            events=sink,
        )
        with (
            mock.patch.object(
                agent_loop_module,
                "new_run_ref",
                return_value=RunRef("run-fixed", "agent-fixed"),
            ),
            mock.patch.object(
                agent_loop_module,
                "new_turn_ref",
                return_value=TurnRef("turn-fixed", "run-fixed", 1),
            ),
            mock.patch.object(
                agent_loop_module,
                "PauseGate",
                side_effect=AssertionError("pause=None constructed a gate"),
            ),
        ):
            result = agent.run(
                [Message(Role.USER, "frozen prompt")],
                pause=None,
            )
        actual = _default_output(result, sink, provider)
        if actual != _DEFAULT_PRE_08D:
            fail(
                f"default run differs from {_DEFAULT_BASELINE_COMMIT}: "
                f"expected={_DEFAULT_PRE_08D!r}, actual={actual!r}"
            )


@check("run_control.loop_parks_at_a_turn_boundary")
def loop_parks_at_a_turn_boundary() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        provider = FakeModelProvider(
            [ModelResponse(Message(Role.ASSISTANT, "done"))]
        )
        gate = PauseGate()
        gate.pause()
        token = _TrackingToken()
        builds: list[object] = []
        original_request = agent_loop_module.ModelRequest

        def recording_request(*args, **kwargs):
            builds.append((args, kwargs))
            return original_request(*args, **kwargs)

        agent = ApiAgent(
            provider,
            {},
            PermissionPolicy(repo_root=Path(temporary)),
        )
        with mock.patch.object(
            agent_loop_module,
            "ModelRequest",
            side_effect=recording_request,
        ):
            done, results, errors, thread = _start_agent(agent, gate, token)
            if not token.listener_registered.wait(1.0):
                fail("agent did not reach the turn-boundary park point")
            if provider.call_count != 0 or builds or done.wait(0):
                fail("paused boundary built a request or called the provider")
            gate.resume()
            if not done.wait(1.0):
                fail("resumed agent did not finish")
            thread.join(timeout=1.0)
        if (
            errors
            or len(results) != 1
            or results[0].stopped_reason != "final_response"
            or provider.call_count != 1
            or len(builds) != 1
        ):
            fail("resumed boundary did not perform exactly one provider call")


def _price_table(rate: str) -> PriceTable:
    return PriceTable(
        prices={"priced": ModelPrice(Decimal(rate), Decimal(rate))},
        currency="USD",
    )


@check("run_control.budget_only_lowers")
def budget_only_lowers() -> None:
    first_prices = _price_table("1")
    second_prices = _price_table("2")
    field_values = {
        "max_turns": (10, 5, 20),
        "wall_seconds": (10.0, 5.0, 20.0),
        "max_total_tokens": (10, 5, 20),
        "max_cost": (Decimal("10"), Decimal("5"), Decimal("20")),
    }
    base_values = {
        name: values[0]
        for name, values in field_values.items()
    }
    base = RunBudget(**base_values, price_table=first_prices)
    cases = (
        ("none_to_number", None, "smaller", True),
        ("number_to_smaller", "number", "smaller", True),
        ("number_to_equal", "number", "number", True),
        ("number_to_larger", "number", "larger", False),
        ("number_to_none", "number", None, False),
    )
    for field_name, (number, smaller, larger) in field_values.items():
        choices = {"number": number, "smaller": smaller, "larger": larger}
        for label, old_key, new_key, succeeds in cases:
            old_value = choices.get(old_key, old_key)
            new_value = choices.get(new_key, new_key)
            arguments = dict(base_values)
            arguments[field_name] = (
                number
                if field_name == "max_turns" and old_value is None
                else old_value
            )
            original = RunBudget(**arguments, price_table=first_prices)
            if field_name == "max_turns" and old_value is None:
                object.__setattr__(original, field_name, None)
            try:
                lowered = original.lowered(**{field_name: new_value})
            except ValueError as exc:
                if succeeds:
                    fail(f"{field_name} {label} unexpectedly failed: {exc}")
                message = str(exc)
                for expected in (field_name, repr(old_value), repr(new_value)):
                    if expected not in message:
                        fail(
                            f"{field_name} widening error omitted "
                            f"{expected!r}: {message!r}"
                        )
            else:
                if not succeeds:
                    fail(f"{field_name} {label} widening was accepted")
                if lowered is original or getattr(lowered, field_name) != new_value:
                    fail(f"{field_name} {label} did not return the requested copy")
                if getattr(original, field_name) != old_value:
                    fail(f"{field_name} {label} changed the original")

    equal = base.lowered(**base_values)
    try:
        equal.max_turns = 1  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        fail("lowered budget was mutable")
    if base.lowered(price_table=second_prices).price_table is not second_prices:
        fail("price_table replacement was refused")
    no_cost = RunBudget(max_cost=None, price_table=first_prices)
    if no_cost.lowered(price_table=None).price_table is not None:
        fail("price_table=None was refused without max_cost")
    try:
        base.lowered(price_table=None)
    except ValueError as exc:
        if "price_table" not in str(exc) or "max_cost" not in str(exc):
            fail(f"price/max_cost error omitted fields: {exc!r}")
    else:
        fail("price_table=None was accepted with max_cost")

    checks_root = Path(__file__).resolve().parent
    for check_path in checks_root.glob("*.py"):
        source = check_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sha256"
            ):
                call_source = ast.get_source_segment(source, node) or ""
                if "symphonai_api" in call_source:
                    fail(f"runtime file hash pin remains in {check_path.name}")
    agent_file_source = (checks_root / "agent_file.py").read_text(encoding="utf-8")
    if (
        "BUDGETS_SHA256" in agent_file_source
        or "budgets.py changed" in agent_file_source
    ):
        fail("agent_file budget hash pin was not removed")


@check("run_control.cap_is_phase_guarded")
def cap_is_phase_guarded() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for phase in (RunPhase.PENDING, RunPhase.RUNNING, RunPhase.PAUSED):
            run = _new_run(root)
            run.phase = phase
            run.cap_budget(max_turns=5)
            if run.budget.max_turns != 5:
                fail(f"cap_budget did not work from {phase.value}")
        for phase in (RunPhase.FINISHED, RunPhase.FAILED, RunPhase.CANCELLED):
            run = _new_run(root)
            run.phase = phase
            before = run.budget
            try:
                run.cap_budget(max_turns=5)
            except ValueError as exc:
                message = str(exc)
                for expected in (
                    "cap_budget",
                    phase.value,
                    RunPhase.PENDING.value,
                    RunPhase.RUNNING.value,
                    RunPhase.PAUSED.value,
                ):
                    if expected not in message:
                        fail(f"cap_budget phase error omitted {expected!r}: {message!r}")
            else:
                fail(f"cap_budget was allowed from {phase.value}")
            if run.budget is not before:
                fail("phase-rejected cap changed the stored budget")

        run = _new_run(root)
        before = run.budget
        try:
            run.cap_budget(max_turns=before.max_turns + 1)
        except ValueError:
            pass
        else:
            fail("raising cap_budget was accepted")
        if run.budget is not before:
            fail("value-rejected cap changed the stored budget")


@check("run_control.redirect_queue")
def redirect_queue() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run = _new_run(Path(temporary))
        for blank in ("", " ", "\t\n"):
            try:
                run.redirect(blank)
            except ValueError:
                pass
            else:
                fail(f"blank redirect was accepted: {blank!r}")
        run.redirect("first")
        run.redirect("second")
        if run.take_redirects() != ("first", "second"):
            fail("redirects were not returned oldest first")
        if run.take_redirects() != ():
            fail("redirect queue was not cleared exactly once")

        for phase in (RunPhase.PENDING, RunPhase.RUNNING, RunPhase.PAUSED):
            phase_run = _new_run(Path(temporary))
            phase_run.phase = phase
            phase_run.redirect(phase.value)
            if phase_run.take_redirects() != (phase.value,):
                fail(f"redirect did not work from {phase.value}")
        for phase in (RunPhase.FINISHED, RunPhase.FAILED, RunPhase.CANCELLED):
            phase_run = _new_run(Path(temporary))
            phase_run.phase = phase
            try:
                phase_run.redirect("late")
            except ValueError as exc:
                message = str(exc)
                if "redirect" not in message or phase.value not in message:
                    fail(f"redirect phase error was incomplete: {message!r}")
            else:
                fail(f"redirect was allowed from {phase.value}")

        run.phase = RunPhase.RUNNING
        barrier = threading.Barrier(2)
        done = threading.Event()
        errors: list[BaseException] = []

        def control() -> None:
            try:
                barrier.wait(timeout=1.0)
                run.redirect("from another thread")
                run.cap_budget(max_turns=5)
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=control, daemon=True)
        thread.start()
        barrier.wait(timeout=1.0)
        if not done.wait(1.0):
            fail("second-thread controls did not complete")
        thread.join(timeout=1.0)
        if (
            errors
            or run.take_redirects() != ("from another thread",)
            or run.budget.max_turns != 5
        ):
            fail("second-thread redirect or cap was not observed")


class _BoundaryProvider(FakeModelProvider):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__(responses)
        self.first_call_started = threading.Event()
        self.release_first_call = threading.Event()
        self.requests = []

    def create_response(self, request, *, cancel=None):
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_call_started.set()
            if not self.release_first_call.wait(1.0):
                raise RuntimeError("first provider call was not released")
        return super().create_response(request, cancel=cancel)


def _tool_response(call_id: str) -> ModelResponse:
    return ModelResponse(
        Message(
            Role.ASSISTANT,
            tool_calls=[ToolCall(id=call_id, name="missing")],
        )
    )


@check("run_control.redirect_reaches_the_next_turn")
def redirect_reaches_the_next_turn() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = _BoundaryProvider(
            [
                _tool_response("first"),
                _tool_response("second"),
                ModelResponse(Message(Role.ASSISTANT, "done")),
            ]
        )
        controlled = _new_run(root, budget=RunBudget(max_turns=3))
        token = CancellationToken()
        controlled.start(token)
        agent = ApiAgent(
            provider,
            {},
            PermissionPolicy(repo_root=root),
            budget=controlled.budget,
        )
        done, results, errors, thread = _start_agent(
            agent,
            None,
            token,
            controlled_run=controlled,
        )
        if not provider.first_call_started.wait(1.0):
            fail("first provider call did not start")
        controlled.redirect("change direction")
        provider.release_first_call.set()
        if not done.wait(1.0):
            fail("redirected run did not finish")
        thread.join(timeout=1.0)
        if errors or len(results) != 1 or provider.call_count != 3:
            fail(f"redirected run ended incorrectly: {errors!r}, {results!r}")
        for index, request in enumerate(provider.requests):
            redirects = [
                message
                for message in request.messages
                if message.role is Role.USER and message.text == "change direction"
            ]
            expected = 0 if index == 0 else 1
            if len(redirects) != expected:
                fail(f"request {index + 1} contained {len(redirects)} redirects")
        delivered = [
            message
            for message in results[0].messages
            if message.role is Role.USER and message.text == "change direction"
        ]
        if len(delivered) != 1 or controlled.take_redirects() != ():
            fail("redirect was omitted, replayed, or left queued")


@check("run_control.capped_turns_stop_the_run")
def capped_turns_stop_the_run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        capped_provider = _BoundaryProvider(
            [_tool_response("cap"), ModelResponse(Message(Role.ASSISTANT, "late"))]
        )
        controlled = _new_run(root, budget=RunBudget(max_turns=3))
        token = CancellationToken()
        controlled.start(token)
        capped_agent = ApiAgent(
            capped_provider,
            {},
            PermissionPolicy(repo_root=root),
            budget=controlled.budget,
        )
        done, results, errors, thread = _start_agent(
            capped_agent,
            None,
            token,
            controlled_run=controlled,
        )
        if not capped_provider.first_call_started.wait(1.0):
            fail("capped provider call did not start")
        controlled.cap_budget(max_turns=1)
        capped_provider.release_first_call.set()
        if not done.wait(1.0):
            fail("capped run did not stop")
        thread.join(timeout=1.0)
        if (
            errors
            or len(results) != 1
            or results[0].stopped_reason != "budget_turns"
            or results[0].turns_used != 1
            or capped_provider.call_count != 1
        ):
            fail("lowered max_turns did not stop with budget_turns")

        exhausted_provider = FakeModelProvider([_tool_response("exhaust")])
        exhausted = ApiAgent(
            exhausted_provider,
            {},
            PermissionPolicy(repo_root=root),
            max_turns=1,
        ).run([Message(Role.USER, "task")])
        if exhausted.stopped_reason != "max_turns" or exhausted_provider.call_count != 1:
            fail("ordinary max_turns exhaustion was not distinguishable")

        source = (
            Path(__file__).resolve().parents[2] / "symphonai_api/agent_loop.py"
        ).read_text()
        if '"budget_turns"' not in source.split("class ApiAgent", 1)[0]:
            fail("AgentRunResult stopped-reason comment omitted budget_turns")


def _frozen_control_probe(*, attached: bool):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = FakeModelProvider(
            [ModelResponse(Message(Role.ASSISTANT, "frozen answer"))]
        )
        sink = CollectingSink()
        agent = ApiAgent(
            provider,
            {},
            PermissionPolicy(repo_root=root),
            agent_ref=AgentRef("agent-fixed", "agent"),
            events=sink,
        )
        controlled = _new_run(root) if attached else None
        if controlled is not None:
            controlled.phase = RunPhase.RUNNING
        with (
            mock.patch.object(
                agent_loop_module,
                "new_run_ref",
                return_value=RunRef("run-fixed", "agent-fixed"),
            ),
            mock.patch.object(
                agent_loop_module,
                "new_turn_ref",
                return_value=TurnRef("turn-fixed", "run-fixed", 1),
            ),
        ):
            result = agent.run(
                [Message(Role.USER, "frozen prompt")],
                pause=None,
                run=controlled,
            )
        return _default_output(result, sink, provider)


@check("run_control.control_is_unchanged_by_default")
def control_is_unchanged_by_default() -> None:
    for attached in (False, True):
        actual = _frozen_control_probe(attached=attached)
        if actual != _DEFAULT_PRE_08D:
            fail(
                f"default control differs from {_CONTROL_BASELINE_COMMIT}: "
                f"attached={attached}, expected={_DEFAULT_PRE_08D!r}, actual={actual!r}"
            )


@check("run_control.control_lock_scope")
def control_lock_scope() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        controlled = _new_run(root)
        controlled.phase = RunPhase.RUNNING
        cap_observations: list[bool] = []
        original_lowered = RunBudget.lowered

        def observing_lowered(budget, **changes):
            cap_observations.append(controlled._control_lock.locked())
            return original_lowered(budget, **changes)

        with mock.patch.object(RunBudget, "lowered", new=observing_lowered):
            controlled.cap_budget(max_turns=5)
        controlled.redirect("lock probe")
        observations: list[bool] = []
        original_message = agent_loop_module.Message

        def observing_message(*args, **kwargs):
            message = original_message(*args, **kwargs)
            if message.role is Role.USER and message.text == "lock probe":
                observations.append(controlled._control_lock.locked())
            return message

        with mock.patch.object(
            agent_loop_module,
            "Message",
            side_effect=observing_message,
        ):
            result = ApiAgent(
                FakeModelProvider(
                    [ModelResponse(Message(Role.ASSISTANT, "done"))]
                ),
                {},
                PermissionPolicy(repo_root=root),
            ).run(
                [Message(Role.USER, "task")],
                run=controlled,
            )
        if (
            result.stopped_reason != "final_response"
            or cap_observations != [False]
            or observations != [False]
        ):
            fail(
                "control callback or redirect append ran under the lock: "
                f"cap={cap_observations!r}, redirect={observations!r}"
            )
