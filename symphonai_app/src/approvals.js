const STALE_REASON = "This approval expired or was answered elsewhere.";

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function payloadOf(frame) {
  return isObject(frame?.payload) ? frame.payload : frame;
}

function questionOf(value) {
  if (
    !isObject(value) ||
    typeof value.approval_id !== "string" ||
    typeof value.operation !== "string" ||
    typeof value.target !== "string" ||
    typeof value.details !== "string"
  ) {
    return null;
  }
  return {
    operation: value.operation,
    target: value.target,
    details: value.details,
    state: "open",
    ...(typeof value.tool_call_id === "string"
      ? { tool_call_id: value.tool_call_id }
      : {}),
  };
}

function isNotFound(value) {
  return value?.status === 404 || /\b404\b/.test(String(value?.message ?? ""));
}

export function createApprovals({ client }) {
  if (
    !client ||
    typeof client.approve !== "function" ||
    typeof client.approvals !== "function"
  ) {
    throw new TypeError("approvals require an approval client");
  }

  const questions = new Map();

  function open(value) {
    if (questions.has(value?.approval_id)) {
      return;
    }
    const question = questionOf(value);
    if (question !== null) {
      questions.set(value.approval_id, question);
    }
  }

  async function resync() {
    const reply = await client.approvals();
    const pending = Array.isArray(reply?.pending) ? reply.pending : [];
    const outstanding = new Set(
      pending
        .filter((value) => typeof value?.approval_id === "string")
        .map((value) => value.approval_id),
    );
    for (const [id, question] of questions) {
      if (
        !outstanding.has(id) &&
        question.state !== "resolved" &&
        question.state !== "stale"
      ) {
        questions.set(id, {
          ...question,
          state: "stale",
          reason: STALE_REASON,
        });
      }
    }
    for (const value of pending) {
      open(value);
    }
  }

  async function onFrame(frame) {
    if (frame?.kind === "approval_requested") {
      open(payloadOf(frame));
      return;
    }
    const payload = payloadOf(frame);
    if (
      frame?.kind === "error" &&
      Number.isInteger(payload?.dropped) &&
      payload.dropped > 0
    ) {
      await resync();
    }
  }

  async function answer(id, allowed, reason = "") {
    const question = questions.get(id);
    if (question === undefined || question.state !== "open") {
      return;
    }
    questions.set(id, { ...question, state: "answering" });
    let reply;
    try {
      reply = await client.approve(id, allowed, reason);
    } catch (error) {
      if (!isNotFound(error)) {
        throw error;
      }
      questions.set(id, {
        ...questions.get(id),
        state: "stale",
        reason: STALE_REASON,
      });
      return;
    }
    if (reply?.status === 404) {
      questions.set(id, {
        ...questions.get(id),
        state: "stale",
        reason: STALE_REASON,
      });
      return;
    }
    if (reply?.resolved !== true) {
      throw new Error("approval reply was not resolved");
    }
    questions.set(id, { ...questions.get(id), state: "resolved" });
  }

  return {
    get state() {
      return new Map(
        [...questions].map(([id, question]) => [id, { ...question }]),
      );
    },
    onFrame,
    answer,
    resync,
  };
}
