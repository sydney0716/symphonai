function eventFrame(type, fields) {
  return { type, fields };
}

function unpack(frame) {
  const payload = frame?.payload;
  if (frame?.kind !== "event" || payload === null || typeof payload !== "object") {
    return null;
  }
  const { type, ...fields } = payload;
  return typeof type === "string" ? eventFrame(type, fields) : null;
}

function callTarget(fields) {
  const arguments_ = fields.arguments;
  return fields.target
    ?? fields.path
    ?? arguments_?.path
    ?? arguments_?.query
    ?? arguments_?.pattern
    ?? null;
}

function diffResult(fields) {
  const result = fields.result ?? fields.tool_result;
  const payload = result?.payload ?? fields.payload ?? fields;
  const path = payload?.path;
  const diff = result?.content ?? payload?.diff ?? fields.diff;
  const added = payload?.lines_added ?? payload?.added;
  const removed = payload?.lines_removed ?? payload?.removed;
  const hasDiffShape = typeof path === "string"
    && typeof diff === "string"
    && Number.isInteger(added)
    && Number.isInteger(removed);
  return hasDiffShape ? { path, added, removed, diff } : null;
}

export function createTranscript() {
  const model = [];
  const calls = new Map();
  const questions = new Map();
  const endedRuns = new Set();
  let turnSerial = 0;
  let activeTurn = null;
  const activityTurns = new WeakMap();

  function turnKey(fields) {
    return fields.turn_id ?? activeTurn ?? `implicit:${fields.run_id ?? ""}:${turnSerial}`;
  }

  function appendActivity(fields) {
    const key = turnKey(fields);
    const previous = model.at(-1);
    const entry = previous?.type === "activity" && activityTurns.get(previous) === key
      ? previous
      : { type: "activity", calls: [] };
    if (entry !== previous) {
      model.push(entry);
      activityTurns.set(entry, key);
    }
    const call = {
      toolCallId: fields.tool_call_id,
      name: fields.tool_name,
      target: callTarget(fields),
      status: "running",
    };
    entry.calls.push(call);
    calls.set(fields.tool_call_id, {
      call,
      entry,
      runId: fields.run_id ?? null,
    });
  }

  function liveCall(fields) {
    const record = calls.get(fields.tool_call_id);
    if (record === undefined || endedRuns.has(record.runId)) {
      return null;
    }
    return record;
  }

  function resolveQuestion(fields, allowed, reason = "") {
    const question = questions.get(fields.tool_call_id);
    if (question === undefined || question.resolved) {
      return;
    }
    question.resolved = true;
    question.allowed = allowed;
    question.reason = reason;
  }

  function replaceWithEdit(record, edit) {
    const activityIndex = model.indexOf(record.entry);
    const callIndex = record.entry.calls.indexOf(record.call);
    if (activityIndex < 0 || callIndex < 0) {
      return;
    }
    const key = activityTurns.get(record.entry);
    const before = record.entry.calls.slice(0, callIndex);
    const after = record.entry.calls.slice(callIndex + 1);
    const replacements = [];
    if (before.length > 0) {
      record.entry.calls = before;
      replacements.push(record.entry);
    }
    replacements.push({
      type: "edit",
      toolCallId: record.call.toolCallId,
      name: record.call.name,
      ...edit,
    });
    if (after.length > 0) {
      const trailing = { type: "activity", calls: after };
      activityTurns.set(trailing, key);
      for (const call of after) {
        calls.get(call.toolCallId).entry = trailing;
      }
      replacements.push(trailing);
    }
    model.splice(activityIndex, 1, ...replacements);
    calls.delete(record.call.toolCallId);
  }

  function applyEvent(event) {
    const { type, fields } = event;
    if (type === "PromptSubmitted") {
      model.push({
        type: "prompt",
        text: fields.text,
        messageCount: fields.message_count,
      });
      return;
    }
    if (type === "AssistantTextDelta") {
      const previous = model.at(-1);
      if (previous?.type === "text") {
        previous.text += fields.text;
      } else {
        model.push({ type: "text", text: fields.text });
      }
      return;
    }
    if (type === "ToolCallStarted") {
      if (!endedRuns.has(fields.run_id) && !calls.has(fields.tool_call_id)) {
        appendActivity(fields);
      }
      resolveQuestion(fields, true);
      return;
    }
    if (type === "ToolCallFinished") {
      const record = liveCall(fields);
      if (record === null) {
        return;
      }
      resolveQuestion(fields, true);
      const edit = diffResult(fields);
      if (fields.ok && edit !== null) {
        replaceWithEdit(record, edit);
        return;
      }
      record.call.status = fields.ok ? "succeeded" : "failed";
      return;
    }
    if (type === "ToolCallFailed") {
      const record = liveCall(fields);
      if (record === null) {
        return;
      }
      record.call.status = "failed";
      record.call.error = fields.error;
      return;
    }
    if (type === "PermissionRequested") {
      const question = {
        type: "question",
        toolCallId: fields.tool_call_id,
        toolName: fields.tool_name,
        mode: fields.mode,
        resolved: false,
      };
      model.push(question);
      questions.set(fields.tool_call_id, question);
      return;
    }
    if (type === "PermissionDenied") {
      resolveQuestion(fields, false, fields.reason);
      return;
    }
    if (type === "CompactionApplied") {
      model.push({
        type: "compaction",
        beforeTokens: fields.before_tokens,
        afterTokens: fields.after_tokens,
        droppedMessages: fields.dropped_messages,
      });
      return;
    }
    if (type === "RunStarted") {
      endedRuns.delete(fields.run_id);
      activeTurn = null;
      turnSerial += 1;
      return;
    }
    if (type === "RunFinished" || type === "RunFailed") {
      endedRuns.add(fields.run_id);
      activeTurn = null;
      return;
    }
    if (type === "TurnStarted") {
      turnSerial += 1;
      activeTurn = fields.turn_id ?? `turn:${fields.run_id ?? ""}:${turnSerial}`;
      return;
    }
    if (type === "TurnFinished") {
      activeTurn = null;
      return;
    }
    if (
      type === "SessionStarted"
      || type === "SessionEnded"
      || type === "SubagentSpawned"
      || type === "SubagentStopped"
    ) {
      return;
    }
    model.push({ type: "unknown", event: { type, ...fields } });
  }

  function apply(frame) {
    const dropped = frame?.dropped ?? frame?.payload?.dropped;
    if (frame?.kind === "error" && Number.isInteger(dropped) && dropped > 0) {
      model.push({ type: "gap", dropped });
      return;
    }
    const event = unpack(frame);
    if (event !== null) {
      applyEvent(event);
    }
  }

  return {
    apply,
    get model() {
      return model;
    },
  };
}
