"""Checks for live run pause, resume, and paused cancellation."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import symphonai_api.agent_loop as agent_loop_module
import symphonai_api.agent_run as agent_run_module
from symphonai_api.agent_loop import ApiAgent
from symphonai_api.agent_run import PauseGate, RunPhase, new_agent_run
from symphonai_api.agent_spec import AgentSpec, ModelSelector
from symphonai_api.cancellation import CancellationToken, OperationCancelled
from symphonai_api.events import CollectingSink
from symphonai_api.identity import AgentRef, RunRef, TurnRef
from symphonai_api.models import Message, ModelResponse, Role
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from scripts.checks.harness import check, fail


_DEFAULT_BASELINE_COMMIT = "8f0891240951b771201a60ba22f72928b067e0ea"
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
        ("TurnStarted", "turn-fixed", 1, None, None),
        ("TurnFinished", "turn-fixed", 1, None, None),
        ("RunFinished", None, None, "agent", "final_response"),
    ),
    1,
)


def _new_run(root: Path):
    return new_agent_run(
        AgentSpec(
            "worker",
            "",
            ModelSelector("fake"),
            PermissionPolicy(repo_root=root),
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


def _start_agent(agent: ApiAgent, gate: PauseGate, token: CancellationToken):
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
