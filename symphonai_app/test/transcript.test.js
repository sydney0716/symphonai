import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { KNOWN_EVENT_TYPES } from "../src/protocol.js";
import { createTranscript } from "../src/transcript.js";

const BASE = Object.freeze({
  agent_id: "agent-1",
  run_id: "run-1",
  turn_id: "turn-1",
  schema_version: 1,
});

function event(type, fields = {}) {
  return { kind: "event", payload: { type, ...BASE, ...fields } };
}

function call(type, id, fields = {}) {
  return event(type, {
    tool_name: fields.tool_name ?? "read_file",
    tool_call_id: id,
    ...fields,
  });
}

function applyAll(frames) {
  const transcript = createTranscript();
  for (const frame of frames) {
    transcript.apply(frame);
  }
  return transcript;
}

const RECORDED_EVENTS = Object.freeze({
  RunStarted: event("RunStarted", { agent_name: "leader" }),
  RunFinished: event("RunFinished", { agent_name: "leader", stopped_reason: "done" }),
  RunFailed: event("RunFailed", { agent_name: "leader", error: "broken" }),
  TurnStarted: event("TurnStarted", { index: 1 }),
  TurnFinished: event("TurnFinished", { index: 1 }),
  AssistantTextDelta: event("AssistantTextDelta", { text: "hello" }),
  ToolCallStarted: call("ToolCallStarted", "recorded-call", { target: "recorded.txt" }),
  ToolCallFinished: call("ToolCallFinished", "recorded-call", {
    ok: true,
    result_kind: "",
    result_path: "",
    lines_added: 0,
    lines_removed: 0,
    diff: "",
    truncated: false,
  }),
  PromptSubmitted: event("PromptSubmitted", { text: "hello", message_count: 1 }),
  ToolCallFailed: call("ToolCallFailed", "recorded-call", { error: "broken" }),
  PermissionRequested: call("PermissionRequested", "recorded-call", { mode: "prompt" }),
  PermissionDenied: call("PermissionDenied", "recorded-call", { reason: "no" }),
  SessionStarted: event("SessionStarted", { session_run_id: "session-1" }),
  SessionEnded: event("SessionEnded", { session_run_id: "session-1" }),
  SubagentSpawned: event("SubagentSpawned", {
    subagent_name: "worker",
    subagent_agent_id: "agent-2",
  }),
  SubagentStopped: event("SubagentStopped", {
    subagent_name: "worker",
    subagent_agent_id: "agent-2",
  }),
  CompactionApplied: event("CompactionApplied", {
    before_tokens: 100,
    after_tokens: 40,
    dropped_messages: 3,
  }),
});

test("assistant deltas join only while consecutive", () => {
  const transcript = applyAll([
    event("AssistantTextDelta", { text: "hello" }),
    event("AssistantTextDelta", { text: " world" }),
    call("ToolCallStarted", "read-1", { target: "src/auth.py" }),
    call("ToolCallFinished", "read-1", { ok: true }),
    event("AssistantTextDelta", { text: "after" }),
    event("AssistantTextDelta", { text: " tool" }),
  ]);

  assert.deepEqual(transcript.model.map(({ type }) => type), [
    "text",
    "activity",
    "text",
  ]);
  assert.equal(transcript.model[0].text, "hello world");
  assert.equal(transcript.model[0].agentId, "agent-1");
  assert.equal(transcript.model[0].runId, "run-1");
  assert.equal(transcript.model[2].text, "after tool");
});

test("assistant text separates agents even when their run ids collide", () => {
  const transcript = applyAll([
    event("AssistantTextDelta", {
      agent_id: "leader",
      run_id: "shared-run",
      text: "I will delegate this. ",
    }),
    event("SubagentSpawned", {
      agent_id: "leader",
      run_id: "shared-run",
      subagent_name: "reader",
      subagent_agent_id: "reader-agent",
    }),
    event("AssistantTextDelta", {
      agent_id: "reader-agent",
      run_id: "shared-run",
      text: "Reading the file...",
    }),
  ]);

  assert.deepEqual(transcript.model, [
    {
      type: "text",
      agentId: "leader",
      runId: "shared-run",
      text: "I will delegate this. ",
    },
    {
      type: "text",
      agentId: "reader-agent",
      runId: "shared-run",
      text: "Reading the file...",
    },
  ]);
});

test("assistant text separates runs from the same agent", () => {
  const transcript = applyAll([
    event("AssistantTextDelta", { run_id: "run-1", text: "first run" }),
    event("RunStarted", { run_id: "run-2", agent_name: "leader" }),
    event("AssistantTextDelta", { run_id: "run-2", text: "second run" }),
  ]);

  assert.equal(transcript.model.length, 2);
  assert.deepEqual(
    transcript.model.map(({ agentId, runId, text }) => ({ agentId, runId, text })),
    [
      { agentId: "agent-1", runId: "run-1", text: "first run" },
      { agentId: "agent-1", runId: "run-2", text: "second run" },
    ],
  );
});

test("a prompt retains the submitted text", () => {
  const transcript = applyAll([
    event("PromptSubmitted", { text: "Explain this", message_count: 7 }),
  ]);
  assert.deepEqual(transcript.model, [
    {
      type: "prompt",
      agentId: "agent-1",
      text: "Explain this",
      messageCount: 7,
    },
  ]);
});

test("consecutive calls group by turn and retain ordered parts", () => {
  const transcript = applyAll([
    event("TurnStarted", { turn_id: "turn-1", index: 1 }),
    call("ToolCallStarted", "read-1", { target: "src/auth.py" }),
    call("ToolCallFinished", "read-1", { ok: true }),
    call("ToolCallStarted", "grep-1", {
      tool_name: "grep",
      target: "def login",
    }),
    call("ToolCallFinished", "grep-1", { tool_name: "grep", ok: true }),
    event("TurnFinished", { turn_id: "turn-1", index: 1 }),
    event("TurnStarted", { turn_id: "turn-2", index: 2 }),
    call("ToolCallStarted", "write-1", {
      turn_id: "turn-2",
      tool_name: "write_file",
      target: "notes.txt",
    }),
  ]);
  const activities = transcript.model.filter(({ type }) => type === "activity");

  assert.equal(activities.length, 2);
  assert.deepEqual(
    activities[0].calls.map(({ name, target }) => ({ name, target })),
    [
      { name: "read_file", target: "src/auth.py" },
      { name: "grep", target: "def login" },
    ],
  );
  assert.deepEqual(
    activities[1].calls.map(({ name, target }) => ({ name, target })),
    [{ name: "write_file", target: "notes.txt" }],
  );
});

test("activity grouping separates agents with the same turn id", () => {
  const transcript = applyAll([
    call("ToolCallStarted", "leader-read", {
      agent_id: "leader",
      run_id: "leader-run",
      turn_id: "colliding-turn",
      target: "leader.txt",
    }),
    call("ToolCallStarted", "worker-read", {
      agent_id: "worker",
      run_id: "worker-run",
      turn_id: "colliding-turn",
      target: "worker.txt",
    }),
  ]);

  assert.equal(transcript.model.length, 2);
  assert.deepEqual(
    transcript.model.map(({ agentId, calls }) => ({
      agentId,
      ids: calls.map(({ toolCallId }) => toolCallId),
    })),
    [
      { agentId: "leader", ids: ["leader-read"] },
      { agentId: "worker", ids: ["worker-read"] },
    ],
  );
});

test("activity entries contain parts and no rendered sentence", () => {
  const transcript = applyAll([
    call("ToolCallStarted", "read-1", { target: "src/auth.py" }),
    call("ToolCallFinished", "read-1", { ok: true }),
  ]);
  const activity = transcript.model[0];

  assert.deepEqual(Object.keys(activity), ["type", "agentId", "calls"]);
  assert.equal(activity.agentId, "agent-1");
  assert.equal(Object.hasOwn(activity, "text"), false);
  assert.equal(Object.hasOwn(activity, "sentence"), false);
  assert.equal(Object.hasOwn(activity, "summary"), false);
  assert.deepEqual(activity.calls[0], {
    toolCallId: "read-1",
    name: "read_file",
    target: "src/auth.py",
    status: "succeeded",
  });
});

test("a failure annotates one existing activity exactly once", () => {
  const transcript = applyAll([
    call("ToolCallStarted", "failed-1", { target: "missing.txt" }),
    call("ToolCallFinished", "failed-1", { ok: false }),
  ]);
  const finishedEntry = transcript.model[0];
  assert.equal(finishedEntry.calls[0].status, "failed");
  assert.equal(Object.hasOwn(finishedEntry.calls[0], "error"), false);

  transcript.apply(call("ToolCallFailed", "failed-1", { error: "not found" }));
  assert.equal(transcript.model.length, 1);
  assert.equal(transcript.model[0], finishedEntry);
  assert.equal(transcript.model[0].calls.length, 1);
  assert.equal(transcript.model[0].calls[0].status, "failed");
  assert.equal(transcript.model[0].calls[0].error, "not found");
});

test("a successful call carries neither failure nor error text", () => {
  const transcript = applyAll([
    call("ToolCallStarted", "ok-1"),
    call("ToolCallFinished", "ok-1", { ok: true }),
  ]);
  const finished = transcript.model[0].calls[0];
  assert.equal(finished.status, "succeeded");
  assert.equal(Object.hasOwn(finished, "error"), false);
  assert.equal(JSON.stringify(finished).includes("failed"), false);
});

test("an edit carries its complete diff immediately", () => {
  const diff = "--- src/auth.py\n+++ src/auth.py\n-old\n+new";
  const transcript = applyAll([
    call("ToolCallStarted", "edit-1", {
      tool_name: "edit_file",
      target: "src/auth.py",
    }),
    call("ToolCallFinished", "edit-1", {
      tool_name: "edit_file",
      ok: true,
      result_kind: "file_diff",
      result_path: "src/auth.py",
      lines_added: 1,
      lines_removed: 1,
      diff,
      truncated: false,
    }),
  ]);

  assert.deepEqual(transcript.model, [{
    type: "edit",
    agentId: "agent-1",
    toolCallId: "edit-1",
    name: "edit_file",
    path: "src/auth.py",
    added: 1,
    removed: 1,
    diff,
    truncated: false,
  }]);
});

test("an edit keeps its arrival position beside concurrent activity", () => {
  const transcript = applyAll([
    call("ToolCallStarted", "edit-first", { tool_name: "edit_file" }),
    call("ToolCallStarted", "read-second"),
    call("ToolCallFinished", "edit-first", {
      tool_name: "edit_file",
      ok: true,
      result_kind: "file_diff",
      result_path: "ordered.txt",
      lines_added: 1,
      lines_removed: 1,
      diff: "-old\n+new",
      truncated: false,
    }),
  ]);

  assert.deepEqual(transcript.model.map(({ type }) => type), ["edit", "activity"]);
  assert.equal(transcript.model[1].calls[0].toolCallId, "read-second");
  transcript.apply(call("ToolCallFinished", "read-second", { ok: true }));
  assert.equal(transcript.model[1].calls[0].status, "succeeded");
});

test("an edit split preserves attribution and trailing activity grouping", () => {
  const transcript = applyAll([
    call("ToolCallStarted", "before", {
      agent_id: "leader",
      turn_id: "shared-turn",
      target: "before.txt",
    }),
    call("ToolCallStarted", "middle-edit", {
      agent_id: "leader",
      turn_id: "shared-turn",
      tool_name: "edit_file",
      target: "edited.txt",
    }),
    call("ToolCallStarted", "after", {
      agent_id: "leader",
      turn_id: "shared-turn",
      tool_name: "grep",
      target: "needle",
    }),
    call("ToolCallFinished", "middle-edit", {
      agent_id: "finishing-agent",
      turn_id: "shared-turn",
      tool_name: "edit_file",
      ok: true,
      result_kind: "file_diff",
      result_path: "edited.txt",
      lines_added: 1,
      lines_removed: 1,
      diff: "-old\n+new",
      truncated: false,
    }),
  ]);

  assert.deepEqual(
    transcript.model.map(({ type, agentId }) => ({ type, agentId })),
    [
      { type: "activity", agentId: "leader" },
      { type: "edit", agentId: "finishing-agent" },
      { type: "activity", agentId: "leader" },
    ],
  );
  assert.deepEqual(
    transcript.model[2].calls.map(({ toolCallId }) => toolCallId),
    ["after"],
  );

  transcript.apply(call("ToolCallStarted", "later", {
    agent_id: "leader",
    turn_id: "shared-turn",
    tool_name: "write_file",
    target: "later.txt",
  }));
  assert.equal(transcript.model.length, 3);
  assert.deepEqual(
    transcript.model[2].calls.map(({ toolCallId }) => toolCallId),
    ["after", "later"],
  );

  transcript.apply(call("ToolCallStarted", "other-agent", {
    agent_id: "worker",
    turn_id: "shared-turn",
    target: "worker.txt",
  }));
  assert.equal(transcript.model.length, 4);
  assert.equal(transcript.model[3].agentId, "worker");
  assert.deepEqual(
    transcript.model[3].calls.map(({ toolCallId }) => toolCallId),
    ["other-agent"],
  );
});

test("captured runtime frames produce reachable targets and edits", async () => {
  const fixture = JSON.parse(await readFile(
    new URL("./fixtures/transcript-wire.json", import.meta.url),
    "utf8",
  ));
  const transcript = applyAll(fixture.frames);

  assert.deepEqual(transcript.model, [
    {
      type: "edit",
      agentId: "agent-wire",
      toolCallId: "wire-edit",
      name: "edit_file",
      path: "src/auth.py",
      added: 1,
      removed: 1,
      diff: "--- src/auth.py\n+++ src/auth.py\n-old\n+new",
      truncated: false,
    },
    {
      type: "activity",
      agentId: "agent-wire",
      calls: [{
        toolCallId: "wire-read",
        name: "read_file",
        target: "README.md",
        status: "succeeded",
      }],
    },
  ]);
});

test("an empty wire diff remains ordinary activity", () => {
  const transcript = applyAll([
    call("ToolCallStarted", "empty-diff", {
      tool_name: "edit_file",
      target: "empty.txt",
    }),
    call("ToolCallFinished", "empty-diff", {
      tool_name: "edit_file",
      ok: true,
      result_kind: "file_diff",
      result_path: "empty.txt",
      lines_added: 0,
      lines_removed: 0,
      diff: "",
      truncated: false,
    }),
  ]);

  assert.equal(transcript.model.length, 1);
  assert.equal(transcript.model[0].type, "activity");
  assert.equal(transcript.model[0].calls[0].status, "succeeded");
});

test("permission questions resolve only on matching answers", () => {
  const denied = applyAll([
    call("PermissionRequested", "deny-1", { mode: "prompt" }),
    call("PermissionDenied", "other", { reason: "wrong question" }),
  ]);
  assert.equal(denied.model[0].resolved, false);
  denied.apply(call("PermissionDenied", "deny-1", { reason: "user denied" }));
  assert.deepEqual(denied.model[0], {
    type: "question",
    agentId: "agent-1",
    toolCallId: "deny-1",
    toolName: "read_file",
    mode: "prompt",
    resolved: true,
    allowed: false,
    reason: "user denied",
  });

  const allowed = applyAll([
    call("ToolCallStarted", "allow-1"),
    call("PermissionRequested", "allow-1", { mode: "prompt" }),
    call("ToolCallFinished", "other", { ok: true }),
  ]);
  const question = allowed.model.find(({ type }) => type === "question");
  assert.equal(question.resolved, false);
  allowed.apply(call("ToolCallFinished", "allow-1", { ok: true }));
  assert.equal(question.resolved, true);
  assert.equal(question.allowed, true);
});

test("a failed tool completion does not answer its permission question", () => {
  const transcript = applyAll([
    call("ToolCallStarted", "denied-1"),
    call("PermissionRequested", "denied-1", { mode: "prompt" }),
    call("ToolCallFinished", "denied-1", { ok: false }),
  ]);
  const question = transcript.model.find(({ type }) => type === "question");

  assert.equal(question.resolved, false);
  assert.equal(Object.hasOwn(question, "allowed"), false);
});

test("each dropped frame inserts a positional gap", () => {
  const transcript = applyAll([
    event("PromptSubmitted", { text: "before", message_count: 1 }),
    { kind: "error", payload: { dropped: 4 } },
    event("AssistantTextDelta", { text: "after" }),
    { kind: "error", dropped: 2, payload: {} },
  ]);

  assert.deepEqual(transcript.model.map(({ type }) => type), [
    "prompt",
    "gap",
    "text",
    "gap",
  ]);
  assert.deepEqual(
    transcript.model.filter(({ type }) => type === "gap").map(({ dropped }) => dropped),
    [4, 2],
  );
});

test("all documented event recordings are covered and unknown events survive", () => {
  assert.equal(KNOWN_EVENT_TYPES.length, 17);
  assert.deepEqual(Object.keys(RECORDED_EVENTS), KNOWN_EVENT_TYPES);
  for (const frame of Object.values(RECORDED_EVENTS)) {
    const transcript = createTranscript();
    assert.doesNotThrow(() => transcript.apply(frame));
  }

  const payload = {
    type: "RuntimeLearnedToSing",
    ...BASE,
    future: { nested: [1, 2, 3] },
  };
  const transcript = applyAll([{ kind: "event", payload }]);
  assert.deepEqual(transcript.model, [{
    type: "unknown",
    agentId: "agent-1",
    event: payload,
  }]);
});

test("orphan and post-run tool events are inert", () => {
  const transcript = applyAll([
    call("ToolCallFinished", "never-started", { ok: false }),
    call("ToolCallFailed", "never-started", { error: "orphan" }),
    call("PermissionDenied", "never-asked", { reason: "orphan" }),
  ]);
  assert.deepEqual(transcript.model, []);

  transcript.apply(event("RunStarted", { run_id: "run-ended" }));
  transcript.apply(call("ToolCallStarted", "late-1", { run_id: "run-ended" }));
  transcript.apply(event("RunFinished", { run_id: "run-ended" }));
  const before = JSON.stringify(transcript.model);
  assert.doesNotThrow(() => transcript.apply(
    call("ToolCallFinished", "late-1", { run_id: "run-ended", ok: true }),
  ));
  assert.equal(JSON.stringify(transcript.model), before);
});

test("compaction details are retained as data", () => {
  const transcript = applyAll([RECORDED_EVENTS.CompactionApplied]);
  assert.deepEqual(transcript.model, [{
    type: "compaction",
    agentId: "agent-1",
    beforeTokens: 100,
    afterTokens: 40,
    droppedMessages: 3,
  }]);
});

test("every event-derived entry is attributed and gaps are not", () => {
  const entries = applyAll([
    event("PromptSubmitted", {
      agent_id: "prompt-agent",
      text: "prompt",
      message_count: 1,
    }),
    event("AssistantTextDelta", { agent_id: "text-agent", text: "text" }),
    call("ToolCallStarted", "activity-call", {
      agent_id: "activity-agent",
      target: "file.txt",
    }),
    call("PermissionRequested", "question-call", {
      agent_id: "question-agent",
      mode: "prompt",
    }),
    event("CompactionApplied", {
      agent_id: "compaction-agent",
      before_tokens: 10,
      after_tokens: 5,
      dropped_messages: 1,
    }),
    event("FutureEntry", { agent_id: "unknown-agent" }),
    { kind: "error", dropped: 2 },
  ]).model;
  const edit = applyAll([
    call("ToolCallStarted", "edit-call", {
      agent_id: "edit-starter",
      tool_name: "edit_file",
      target: "edit.txt",
    }),
    call("ToolCallFinished", "edit-call", {
      agent_id: "edit-agent",
      tool_name: "edit_file",
      ok: true,
      result_kind: "file_diff",
      result_path: "edit.txt",
      lines_added: 1,
      lines_removed: 1,
      diff: "-old\n+new",
      truncated: false,
    }),
  ]).model[0];

  assert.deepEqual(
    entries.filter(({ type }) => type !== "gap").map(({ type, agentId }) => ({
      type,
      agentId,
    })),
    [
      { type: "prompt", agentId: "prompt-agent" },
      { type: "text", agentId: "text-agent" },
      { type: "activity", agentId: "activity-agent" },
      { type: "question", agentId: "question-agent" },
      { type: "compaction", agentId: "compaction-agent" },
      { type: "unknown", agentId: "unknown-agent" },
    ],
  );
  assert.equal(edit.agentId, "edit-agent");
  const gap = entries.find(({ type }) => type === "gap");
  assert.equal(Object.hasOwn(gap, "agentId"), false);
});
