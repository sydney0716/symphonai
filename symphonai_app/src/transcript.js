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
  return fields.target || null;
}

function diffResult(fields) {
  const hasDiff = fields.result_kind === "file_diff"
    && typeof fields.result_path === "string"
    && fields.result_path.length > 0
    && typeof fields.diff === "string"
    && fields.diff.length > 0;
  return hasDiff
    ? {
      path: fields.result_path,
      added: fields.lines_added,
      removed: fields.lines_removed,
      diff: fields.diff,
      truncated: fields.truncated,
    }
    : null;
}

export function createTranscript({ now = () => Date.now() } = {}) {
  const model = [];
  const calls = new Map();
  const questions = new Map();
  const endedRuns = new Set();
  let turnSerial = 0;
  let sessionId = null;
  let activeTurn = null;
  let rootAgentId = null;
  let currentPrompt = null;
  const activityTurns = new Map();
  const promptEdits = new Map();
  const promptUndone = new Map();
  const promptStarts = new Map();
  const dispatchCalls = [];

  function appendTurnTime(prompt, endedAt, outcome) {
    const startValue = promptStarts.get(prompt);
    const started = typeof startValue === "number" ? startValue : Date.parse(startValue);
    const ended = typeof endedAt === "number" ? endedAt : Date.parse(endedAt);
    if (!Number.isFinite(started) || !Number.isFinite(ended)) return;
    const seconds = Math.max(0, Math.floor((ended - started) / 1000));
    const duration = seconds < 60
      ? `${seconds}s`
      : seconds < 3600
        ? `${Math.floor(seconds / 60)}m ${seconds % 60}s`
        : `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    const label = outcome === "failed" ? "Failed after" : outcome === "cancelled" ? "Stopped after" : "Worked for";
    const time = new Date(ended).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
    const entry = { type: "turn-time", text: `${label} ${duration} · ${time}` };
    model.push(entry);
    return entry;
  }

  function turnKey(fields) {
    const turn = fields.turn_id
      ?? activeTurn
      ?? `implicit:${fields.run_id ?? ""}:${turnSerial}`;
    return JSON.stringify([fields.agent_id ?? null, turn]);
  }

  function appendActivity(fields) {
    const key = turnKey(fields);
    const previous = model.at(-1);
    const entry = previous?.type === "activity" && activityTurns.get(previous) === key
      ? previous
      : { type: "activity", agentId: fields.agent_id, calls: [] };
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
    if (call.name === "dispatch_subagent" && call.target) dispatchCalls.push(call);
    calls.set(fields.tool_call_id, {
      call,
      entry,
      runId: fields.run_id ?? null,
    });
  }

  function appendPlan(fields) {
    model.push({
      type: "plan",
      agentId: fields.agent_id,
      steps: Array.isArray(fields.steps) ? fields.steps.map((step) => ({ ...step })) : [],
      explanation: typeof fields.explanation === "string" ? fields.explanation : "",
    });
  }

  function appendUserQuestion(question, answer = "", agentId) {
    model.push({ type: "user-question", agentId, question, answer });
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

  function appendEditCard(prompt, edits, { canUndo = false, undone = false } = {}) {
    const byPath = new Map();
    for (const edit of edits) {
      const file = byPath.get(edit.path) ?? { path: edit.path, added: 0, removed: 0 };
      file.added += edit.added;
      file.removed += edit.removed;
      byPath.set(edit.path, file);
    }
    const files = [...byPath.values()];
    if (!files.length) return null;
    for (const previous of model) {
      if (previous.type === "edit-card") previous.canUndo = false;
    }
    const card = {
      type: "edit-card",
      checkpointKey: prompt.checkpointKey ?? null,
      files,
      added: files.reduce((total, file) => total + file.added, 0),
      removed: files.reduce((total, file) => total + file.removed, 0),
      canUndo: canUndo && !undone,
      undone,
    };
    model.push(card);
    return card;
  }

  function replaceWithEdit(record, edit, agentId) {
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
      agentId,
      toolCallId: record.call.toolCallId,
      name: record.call.name,
      ...edit,
    });
    const editEntry = replacements.at(-1);
    if (currentPrompt !== null) {
      promptEdits.get(currentPrompt)?.push(editEntry);
    }
    if (after.length > 0) {
      const trailing = {
        type: "activity",
        agentId: record.entry.agentId,
        calls: after,
      };
      activityTurns.set(trailing, key);
      for (const call of after) {
        calls.get(call.toolCallId).entry = trailing;
      }
      replacements.push(trailing);
    }
    model.splice(activityIndex, 1, ...replacements);
    calls.delete(record.call.toolCallId);
  }

  function replayedUserCommand(text) {
    const prefix = "The person ran a command in the repository:\n$ ";
    if (typeof text !== "string" || !text.startsWith(prefix)) return null;
    const lines = text.slice(prefix.length).split("\n");
    const command = lines.shift();
    const close = lines.length - 1;
    if (!command || close < 0 || !/^\((?:exit code \d+|timed out after 120 s|stopped)\)$/.test(lines[close])) return null;
    const outcome = lines[close].slice(1, -1);
    const output = lines.slice(0, close).join("\n").replace(/\n$/, "");
    const entry = { type: "command", command: `! ${command}`, output: [] };
    if (output) entry.output.push(output);
    if (outcome === "stopped") entry.output.push("stopped");
    else if (outcome === "timed out after 120 s") entry.output.push("timed out");
    else if (outcome !== "exit code 0") entry.output.push(outcome.replace("exit code ", "exit "));
    return entry;
  }

  function applyEvent(event) {
    const { type, fields } = event;
    if (type === "HistoryMessage") {
      if (!fields.text && !fields.attachments?.length && !fields.tool_calls?.length && !fields.turn_ended_at) {
        return;
      }
      if (fields.role === "user") {
        const commandEntry = replayedUserCommand(fields.text);
        const entry = commandEntry ?? {
          type: "prompt",
          agentId: fields.agent_id,
          text: fields.text,
          messageCount: fields.message_count,
        };
        if (commandEntry === null) {
          if (typeof fields.checkpoint_key === "string") entry.checkpointKey = fields.checkpoint_key;
          promptEdits.set(entry, (fields.edits ?? []).filter((edit) =>
            typeof edit?.path === "string"
            && Number.isInteger(edit.added)
            && Number.isInteger(edit.removed)
          ));
          promptUndone.set(entry, fields.undone === true);
          if (typeof fields.turn_started_at === "string") promptStarts.set(entry, fields.turn_started_at);
          if (fields.attachments?.length) {
            entry.attachments = fields.attachments.map(({ kind, filename }) => ({ kind, filename }));
          }
        }
        model.push(entry);
        if (commandEntry === null && (rootAgentId === null || fields.agent_id === rootAgentId)) {
          currentPrompt = entry;
        }
        if (commandEntry === null && fields.turn_ended_at) {
          appendEditCard(entry, promptEdits.get(entry) ?? [], {
            undone: promptUndone.get(entry) === true,
          });
          appendTurnTime(entry, fields.turn_ended_at, fields.turn_outcome);
          if (currentPrompt === entry) currentPrompt = null;
        }
      } else if (fields.role === "assistant") {
        if (fields.text) model.push({
          type: "text",
          agentId: fields.agent_id,
          runId: fields.run_id,
          text: fields.text,
        });
        for (const toolCall of fields.tool_calls ?? []) {
          if (toolCall.name === "update_plan" && Array.isArray(toolCall.plan)) {
            appendPlan({
              agent_id: fields.agent_id,
              steps: toolCall.plan,
              explanation: toolCall.explanation,
            });
            continue;
          }
          if (toolCall.name === "ask_user" && typeof toolCall.question === "string") {
            const answer = typeof toolCall.answer === "string"
              ? toolCall.answer
              : toolCall.skipped === true ? "Skipped" : "";
            appendUserQuestion(toolCall.question, answer, fields.agent_id);
            continue;
          }
          if (toolCall.name === "propose_plan" && typeof toolCall.plan === "string") {
            const outcome = toolCall.decision === "ask"
              ? "Approved — ask before changes"
              : toolCall.decision === "allow"
                ? "Approved — allow changes"
                : toolCall.decision === "keep_planning"
                  ? `Kept planning${typeof toolCall.feedback === "string" && toolCall.feedback ? `: ${toolCall.feedback}` : ""}`
                  : "";
            model.push({ type: "user-plan", plan: toolCall.plan, outcome });
            continue;
          }
          if (toolCall.name !== "dispatch_subagent") continue;
          appendActivity({
            agent_id: fields.agent_id,
            run_id: fields.run_id,
            turn_id: fields.turn_id,
            tool_call_id: toolCall.id,
            tool_name: toolCall.name,
            target: toolCall.subagent_name,
          });
          const record = calls.get(toolCall.id);
          if (record) {
            record.call.status = "settled";
            if (typeof toolCall.task === "string") record.call.task = toolCall.task;
            calls.delete(toolCall.id);
          }
        }
        if (
          currentPrompt !== null
          && (!fields.tool_calls?.length || fields.turn_ended_at)
        ) {
          appendEditCard(currentPrompt, promptEdits.get(currentPrompt) ?? [], {
            undone: promptUndone.get(currentPrompt) === true,
          });
          if (fields.turn_ended_at) {
            appendTurnTime(currentPrompt, fields.turn_ended_at, fields.turn_outcome);
          }
          currentPrompt = null;
        }
      }
      return;
    }
    if (type === "PromptSubmitted") {
      if (rootAgentId !== null && fields.agent_id !== rootAgentId) {
        const dispatch = [...dispatchCalls].reverse().find((call) => (
          call.agentId === fields.agent_id
        ));
        if (dispatch) dispatch.task = fields.text;
        return;
      }
      const entry = {
        type: "prompt",
        agentId: fields.agent_id,
        text: fields.text,
        messageCount: fields.message_count,
      };
      if (typeof fields.checkpoint_key === "string") entry.checkpointKey = fields.checkpoint_key;
      promptEdits.set(entry, []);
      promptStarts.set(entry, now());
      model.push(entry);
      if (rootAgentId === null || fields.agent_id === rootAgentId) {
        rootAgentId ??= fields.agent_id;
        currentPrompt = entry;
      }
      return;
    }
    if (type === "AssistantTextDelta") {
      const previous = model.at(-1);
      if (
        previous?.type === "text"
        && previous.agentId === fields.agent_id
        && previous.runId === fields.run_id
      ) {
        previous.text += fields.text;
      } else {
        model.push({
          type: "text",
          agentId: fields.agent_id,
          runId: fields.run_id,
          text: fields.text,
        });
      }
      return;
    }
    if (type === "ToolCallStarted") {
      if (fields.tool_name === "update_plan" || fields.tool_name === "ask_user" || fields.tool_name === "propose_plan") return;
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
        replaceWithEdit(record, edit, fields.agent_id);
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
        agentId: fields.agent_id,
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
        agentId: fields.agent_id,
        beforeTokens: fields.before_tokens,
        afterTokens: fields.after_tokens,
        droppedMessages: fields.dropped_messages,
      });
      return;
    }
    if (type === "GoalChanged") {
      const lastCheck = fields.last_check;
      const result = lastCheck?.exit === null || lastCheck?.exit === undefined
        ? "timed out"
        : `exit ${lastCheck.exit}`;
      let text = "";
      if (fields.change === "set") {
        text = `Goal set: up to ${fields.max_rounds} rounds.`;
      } else if (fields.change === "check" && fields.phase === "complete") {
        text = `Goal check passed. Goal complete after ${fields.rounds} rounds.`;
      } else if (fields.change === "check" && fields.phase === "active") {
        text = `Goal check failed (${result}). Starting round ${fields.rounds + 1} of ${fields.max_rounds}.`;
      } else if (fields.change === "check" && fields.phase === "blocked") {
        text = `Goal check failed (${result}). Goal blocked: ${fields.reason}.`;
      } else if (fields.change === "check" && fields.phase === "paused") {
        text = `Goal check failed (${result}). Goal paused: ${fields.reason}.`;
      } else if (fields.change === "pause") {
        text = `Goal paused: ${fields.reason}.`;
      } else if (fields.change === "resume") {
        text = "Goal resumed.";
      } else if (fields.change === "clear") {
        text = "Goal cleared.";
      } else if (fields.change === "update") {
        text = `Agent marked the goal ${fields.phase}: ${fields.reason}.`;
      } else if (fields.change === "round") {
        text = fields.phase === "blocked"
          ? `Goal blocked: ${fields.reason}.`
          : `No check configured. Starting round ${fields.rounds} of ${fields.max_rounds}.`;
      }
      model.push({
        type: "goal",
        agentId: fields.agent_id,
        text,
        output: fields.change === "check" && typeof lastCheck?.output === "string"
          ? lastCheck.output
          : "",
      });
      return;
    }
    if (type === "PlanChanged") {
      appendPlan(fields);
      return;
    }
    if (type === "RunStarted") {
      rootAgentId ??= fields.agent_id;
      endedRuns.delete(fields.run_id);
      activeTurn = null;
      turnSerial += 1;
      return;
    }
    if (type === "RunFinished" || type === "RunFailed") {
      const edits = currentPrompt === null ? [] : promptEdits.get(currentPrompt) ?? [];
      if (fields.agent_id === rootAgentId && edits.length) {
        appendEditCard(currentPrompt, edits, {
          canUndo: Boolean(currentPrompt.checkpointKey),
        });
      }
      if (fields.agent_id === rootAgentId && currentPrompt !== null) {
        appendTurnTime(
          currentPrompt,
          now(),
          type === "RunFailed" ? "failed" : fields.stopped_reason === "cancelled" ? "cancelled" : "finished",
        );
      }
      for (const [id, record] of calls) {
        if (record.runId === fields.run_id) {
          if (record.call.status === "running") record.call.status = "settled";
          calls.delete(id);
        }
      }
      endedRuns.add(fields.run_id);
      activeTurn = null;
      if (fields.agent_id === rootAgentId) currentPrompt = null;
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
      || type === "SubagentStopped"
    ) {
      return;
    }
    if (type === "SubagentSpawned") {
      const dispatch = [...dispatchCalls].reverse().find((call) => (
        call.target === fields.subagent_name && !call.agentId
      ));
      if (dispatch) dispatch.agentId = fields.subagent_agent_id;
      return;
    }
    model.push({
      type: "unknown",
      agentId: fields.agent_id,
      event: { type, ...fields },
    });
  }

  function apply(frame) {
    const frameSessionId = frame?.payload?.session_id;
    if (typeof frameSessionId === "string" && frameSessionId !== sessionId) {
      return;
    }
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

  function reset() {
    model.length = 0;
    calls.clear();
    questions.clear();
    endedRuns.clear();
    activityTurns.clear();
    promptEdits.clear();
    promptUndone.clear();
    promptStarts.clear();
    dispatchCalls.length = 0;
    turnSerial = 0;
    activeTurn = null;
    rootAgentId = null;
    currentPrompt = null;
  }

  function snapshot() {
    return {
      model: [...model],
      calls: new Map(calls),
      questions: new Map(questions),
      endedRuns: new Set(endedRuns),
      activityTurns: new Map(activityTurns),
      promptEdits: new Map([...promptEdits].map(([entry, edits]) => [entry, [...edits]])),
      promptUndone: new Map(promptUndone),
      promptStarts: new Map(promptStarts),
      dispatchCalls: [...dispatchCalls],
      turnSerial,
      sessionId,
      activeTurn,
      rootAgentId,
      currentPrompt,
    };
  }

  function restore(state) {
    reset();
    model.push(...state.model);
    for (const [key, value] of state.calls) calls.set(key, value);
    for (const [key, value] of state.questions) questions.set(key, value);
    for (const value of state.endedRuns) endedRuns.add(value);
    for (const [key, value] of state.activityTurns) activityTurns.set(key, value);
    for (const [key, value] of state.promptEdits) promptEdits.set(key, [...value]);
    for (const [key, value] of state.promptUndone) promptUndone.set(key, value);
    for (const [key, value] of state.promptStarts) promptStarts.set(key, value);
    dispatchCalls.push(...state.dispatchCalls);
    turnSerial = state.turnSerial;
    sessionId = state.sessionId;
    activeTurn = state.activeTurn;
    rootAgentId = state.rootAgentId;
    currentPrompt = state.currentPrompt;
  }

  return {
    apply,
    reset,
    snapshot,
    restore,
    setSessionId(value) {
      sessionId = typeof value === "string" ? value : null;
    },
    get model() {
      return model;
    },
  };
}
