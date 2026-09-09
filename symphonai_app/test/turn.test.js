import test from "node:test";
import assert from "node:assert/strict";

import {
  DISPATCHING,
  IDLE,
  RUNNING,
  createTurnState,
} from "../src/turn.js";

function running(text = "first", runId = "run-1") {
  const turn = createTurnState();
  turn.submit(text);
  turn.accepted(runId);
  return turn;
}

function queueOne(turn, text = "queued") {
  turn.submit(text);
  return turn.queue.at(-1);
}

test("idle submit starts one dispatch and holds its text", () => {
  const turn = createTurnState();
  const actions = turn.submit("first");
  assert.equal(turn.state, DISPATCHING);
  assert.deepEqual(turn.queue, []);
  assert.equal(turn.inFlight.text, "first");
  assert.deepEqual(actions, [
    { kind: "prompt", id: turn.inFlight.id, text: "first" },
  ]);
});

test("submit queues in dispatching and running without changing state", () => {
  const dispatching = createTurnState();
  dispatching.submit("first");
  assert.deepEqual(dispatching.submit("second"), []);
  assert.equal(dispatching.state, DISPATCHING);
  assert.equal(dispatching.queue[0].text, "second");

  const active = running();
  assert.deepEqual(active.submit("second"), []);
  assert.equal(active.state, RUNNING);
  assert.equal(active.queue[0].text, "second");
});

test("accepted records the active run and enters running", () => {
  const turn = createTurnState();
  turn.submit("first");
  assert.deepEqual(turn.accepted("run-1"), []);
  assert.equal(turn.state, RUNNING);
  assert.equal(turn.activeRunId, "run-1");
});

test("only matching terminal events end a running turn", () => {
  for (const type of ["RunFinished", "RunFailed"]) {
    const turn = running("first", `${type}-run`);
    assert.deepEqual(turn.event({ type, run_id: "some-other-run" }), []);
    assert.equal(turn.state, RUNNING);
    assert.deepEqual(turn.event({ type, run_id: `${type}-run` }), []);
    assert.equal(turn.state, IDLE);
    assert.equal(turn.activeRunId, null);
  }
});

test("rejection restores the in-flight text to the queue front", () => {
  const turn = createTurnState();
  turn.submit("first");
  turn.submit("second");
  const actions = turn.rejected("offline");
  assert.equal(turn.state, IDLE);
  assert.deepEqual(turn.queue.map(({ text }) => text), ["first", "second"]);
  assert.equal(turn.inFlight, null);
  assert.deepEqual(actions, [{ kind: "rejected", error: "offline" }]);
});

test("only queued messages can be edited or withdrawn", () => {
  const combinations = [
    ["queued", "edit", true],
    ["queued", "withdraw", true],
    ["in-flight", "edit", false],
    ["in-flight", "withdraw", false],
  ];
  assert.equal(combinations.length, 4);
  for (const [target, intent, changes] of combinations) {
    const turn = createTurnState();
    turn.submit("in flight");
    const queued = queueOne(turn);
    const id = target === "queued" ? queued.id : turn.inFlight.id;
    const before = turn.queue;
    const actions = intent === "edit"
      ? turn.edit(id, "edited")
      : turn.withdraw(id);
    assert.equal(actions.length > 0, changes, `${target} ${intent}`);
    if (!changes) {
      assert.deepEqual(turn.queue, before, `${target} ${intent}`);
      assert.equal(turn.inFlight.text, "in flight", `${target} ${intent}`);
    }
  }
});

test("terminal events drain three queued messages one at a time", () => {
  const turn = running();
  for (const text of ["second", "third", "fourth"]) {
    turn.submit(text);
  }

  for (const [runId, text, remaining] of [
    ["run-1", "second", 2],
    ["run-2", "third", 1],
    ["run-3", "fourth", 0],
  ]) {
    const actions = turn.event({ type: "RunFinished", run_id: runId });
    assert.deepEqual(actions, [
      { kind: "prompt", id: turn.inFlight.id, text },
    ]);
    assert.equal(turn.state, DISPATCHING);
    assert.equal(turn.queue.length, remaining);
    turn.accepted(`run-${Number(runId.at(-1)) + 1}`);
  }
});

test("stop in the dispatch gap cancels acceptance and waits for termination", () => {
  const turn = createTurnState();
  turn.submit("first");
  assert.deepEqual(turn.stop(), []);
  assert.equal(turn.inFlight.cancelled, true);
  assert.deepEqual(turn.accepted("run-1"), [{ kind: "stop" }]);
  assert.equal(turn.state, DISPATCHING);
  assert.equal(turn.activeRunId, "run-1");
  turn.event({ type: "RunFinished", run_id: "run-1" });
  assert.equal(turn.state, IDLE);
});

test("stop is inert when idle and waits for a terminal event when running", () => {
  const idle = createTurnState();
  assert.deepEqual(idle.stop(), []);
  assert.equal(idle.state, IDLE);

  const active = running();
  assert.deepEqual(active.stop(), [{ kind: "stop" }]);
  assert.equal(active.state, RUNNING);
  active.event({ type: "RunFinished", run_id: "run-1" });
  assert.equal(active.state, IDLE);
});

test("a terminal event needs no preceding turn event", () => {
  const turn = createTurnState();
  turn.submit("first");
  turn.accepted("run-1");
  turn.event({ type: "RunFinished", run_id: "run-1" });
  assert.equal(turn.state, IDLE);
});

test("unknown and out-of-order events are inert", () => {
  const turn = running();
  const before = {
    state: turn.state,
    queue: turn.queue,
    activeRunId: turn.activeRunId,
    inFlight: turn.inFlight,
  };
  for (const value of [
    { type: "FutureEvent", known: false, fields: { run_id: "run-1" } },
    { type: "TurnFinished", run_id: "run-1" },
    { type: "RunFinished", run_id: "other" },
  ]) {
    assert.deepEqual(turn.event(value), []);
  }
  assert.deepEqual(
    {
      state: turn.state,
      queue: turn.queue,
      activeRunId: turn.activeRunId,
      inFlight: turn.inFlight,
    },
    before,
  );
  turn.event({ type: "RunFinished", run_id: "run-1" });
  assert.deepEqual(turn.event({ type: "RunFailed", run_id: "run-1" }), []);
  assert.equal(turn.state, IDLE);
});

function machineFor(state) {
  const turn = createTurnState();
  turn.submit("first");
  turn.submit("queued");
  if (state === IDLE) {
    turn.rejected("fixture");
  } else if (state === RUNNING) {
    turn.accepted("run-1");
  }
  assert.equal(turn.state, state);
  return turn;
}

test("the complete seven-intent by three-state table is exercised", () => {
  const states = [IDLE, DISPATCHING, RUNNING];
  const intents = [
    ["submit", (turn) => turn.submit("new")],
    ["edit", (turn) => turn.edit(turn.queue[0].id, "edited")],
    ["withdraw", (turn) => turn.withdraw(turn.queue[0].id)],
    ["stop", (turn) => turn.stop()],
    ["accepted", (turn) => turn.accepted("accepted-run")],
    ["rejected", (turn) => turn.rejected("rejected")],
    ["event", (turn) => turn.event({ type: "FutureEvent", known: false })],
  ];
  const cells = [];
  for (const state of states) {
    for (const [intent, invoke] of intents) {
      const turn = machineFor(state);
      assert.doesNotThrow(() => invoke(turn), `${intent} from ${state}`);
      cells.push(`${intent}:${state}`);
    }
  }
  assert.equal(cells.length, 21);
  assert.equal(new Set(cells).size, 21);
  assert.deepEqual(states, [IDLE, DISPATCHING, RUNNING]);
});
