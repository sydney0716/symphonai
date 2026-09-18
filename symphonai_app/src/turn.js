export const IDLE = "idle";
export const DISPATCHING = "dispatching";
export const RUNNING = "running";

const TERMINAL_EVENTS = new Set(["RunFinished", "RunFailed"]);

export function createTurnState() {
  let state = IDLE;
  let activeRunId = null;
  let inFlight = null;
  let nextMessageId = 1;
  const queue = [];

  function message(text) {
    const value = { id: `message-${nextMessageId}`, text };
    nextMessageId += 1;
    return value;
  }

  function promptAction(value) {
    return { kind: "prompt", id: value.id, text: value.text };
  }

  function dispatch(value) {
    state = DISPATCHING;
    activeRunId = null;
    inFlight = { ...value, cancelled: false };
    return [promptAction(value)];
  }

  function drainOne() {
    if (state !== IDLE || queue.length === 0) {
      return [];
    }
    return dispatch(queue.shift());
  }

  function terminalEvent(value) {
    if (!value || value.known === false || !TERMINAL_EVENTS.has(value.type)) {
      return false;
    }
    const runId = value.run_id ?? value.fields?.run_id;
    return activeRunId !== null && runId === activeRunId;
  }

  return {
    get state() {
      return state;
    },

    get queue() {
      return queue.map((value) => ({ ...value }));
    },

    get activeRunId() {
      return activeRunId;
    },

    get inFlight() {
      return inFlight === null ? null : { ...inFlight };
    },

    submit(text) {
      queue.push(message(text));
      return drainOne();
    },

    edit(id, text) {
      const value = queue.find((candidate) => candidate.id === id);
      if (value === undefined) {
        return [];
      }
      value.text = text;
      return [{ kind: "edited", id, text }];
    },

    withdraw(id) {
      const index = queue.findIndex((candidate) => candidate.id === id);
      if (index === -1) {
        return [];
      }
      const [value] = queue.splice(index, 1);
      return [{ kind: "withdrawn", id: value.id }];
    },

    stop() {
      if (state === RUNNING) {
        return [{ kind: "stop" }];
      }
      if (state === DISPATCHING && inFlight !== null) {
        inFlight.cancelled = true;
      }
      return [];
    },

    accepted() {
      if (state !== DISPATCHING || inFlight === null) {
        return [];
      }
      if (inFlight.cancelled) {
        return [{ kind: "stop" }];
      }
      state = RUNNING;
      return [];
    },

    adopted(runId) {
      if (state !== DISPATCHING || inFlight === null) {
        return [];
      }
      queue.unshift({ id: inFlight.id, text: inFlight.text });
      inFlight = null;
      activeRunId = typeof runId === "string" && runId.length > 0 ? runId : null;
      state = RUNNING;
      return [];
    },

    rejected(error) {
      if (state !== DISPATCHING || inFlight === null) {
        return [];
      }
      const rejected = { id: inFlight.id, text: inFlight.text };
      queue.unshift(rejected);
      inFlight = null;
      activeRunId = null;
      state = IDLE;
      return [{ kind: "rejected", error }];
    },

    event(value) {
      if (
        activeRunId === null &&
        (state === DISPATCHING || state === RUNNING) &&
        value?.known !== false &&
        value?.type === "RunStarted" &&
        typeof value.run_id === "string"
      ) {
        activeRunId = value.run_id;
        return [];
      }
      if (!terminalEvent(value)) {
        return [];
      }
      state = IDLE;
      activeRunId = null;
      inFlight = null;
      return drainOne();
    },
  };
}
