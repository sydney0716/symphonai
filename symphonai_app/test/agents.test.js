import test from "node:test";
import assert from "node:assert/strict";

import { createAgentBoard } from "../src/agents.js";

function event(type, fields = {}) {
  return { kind: "event", payload: { type, ...fields } };
}

test("agent board follows root and subagent activity in spawn order", () => {
  const board = createAgentBoard();
  assert.deepEqual(board.rows, []);
  board.apply(event("RunStarted", { agent_id: "root", agent_name: "leader" }));
  assert.deepEqual(board.rows, [
    { agentId: "root", parentAgentId: null, name: "leader", state: "running", tool: "", target: "", last: null },
  ]);
  board.apply(event("ToolCallStarted", { agent_id: "root", tool_name: "read" }));
  assert.deepEqual(board.rows, [
    { agentId: "root", parentAgentId: null, name: "leader", state: "running", tool: "read", target: "", last: null },
  ]);
  board.apply(event("SubagentSpawned", { agent_id: "root", subagent_agent_id: "worker", subagent_name: "reader" }));
  assert.deepEqual(board.rows, [
    { agentId: "root", parentAgentId: null, name: "leader", state: "running", tool: "read", target: "", last: null },
    { agentId: "worker", parentAgentId: "root", name: "reader", state: "running", tool: "", target: "", last: null },
  ]);
  board.apply(event("ToolCallStarted", { agent_id: "worker", tool_name: "search" }));
  assert.deepEqual(board.rows[1], { agentId: "worker", parentAgentId: "root", name: "reader", state: "running", tool: "search", target: "", last: null });
  board.apply(event("PermissionRequested", { agent_id: "worker", tool_name: "write" }));
  assert.deepEqual(board.rows[1], { agentId: "worker", parentAgentId: "root", name: "reader", state: "waiting", tool: "write", target: "", last: null });
  board.apply(event("PermissionDenied", { agent_id: "worker" }));
  assert.deepEqual(board.rows[1], { agentId: "worker", parentAgentId: "root", name: "reader", state: "running", tool: "write", target: "", last: null });
  board.apply(event("SubagentStopped", { subagent_agent_id: "worker" }));
  assert.deepEqual(board.rows[1], { agentId: "worker", parentAgentId: "root", name: "reader", state: "done", tool: "", target: "", last: null });
  board.apply(event("RunFinished", { agent_id: "root" }));
  assert.deepEqual(board.rows, [
    { agentId: "root", parentAgentId: null, name: "leader", state: "done", tool: "", target: "", last: null },
    { agentId: "worker", parentAgentId: "root", name: "reader", state: "done", tool: "", target: "", last: null },
  ]);
});

test("a completed tool retains its target as the last activity across turns", () => {
  const board = createAgentBoard();
  board.apply(event("RunStarted", { agent_id: "root", agent_name: "leader" }));
  board.apply(event("ToolCallStarted", { agent_id: "root", tool_name: "edit_file", target: "app.js" }));
  assert.equal(board.rows[0].tool, "edit_file");
  assert.equal(board.rows[0].target, "app.js");
  board.apply(event("ToolCallFinished", { agent_id: "root", tool_name: "edit_file" }));
  assert.deepEqual(board.rows[0].last, { tool: "edit_file", target: "app.js" });
  assert.equal(board.rows[0].tool, "");
  assert.equal(board.rows[0].target, "");
  board.apply(event("RunFinished", { agent_id: "root" }));
  assert.equal(board.rows[0].state, "done");
  board.apply(event("RunStarted", { agent_id: "root", agent_name: "leader" }));
  assert.deepEqual(board.rows[0].last, { tool: "edit_file", target: "app.js" });
  assert.equal(board.rows[0].state, "running");
});

test("failed tools retain their description and clear current activity", () => {
  const board = createAgentBoard();
  board.apply(event("RunStarted", { agent_id: "root" }));
  board.apply(event("ToolCallStarted", { agent_id: "root", tool_name: "run_shell", target: "pytest" }));
  board.apply(event("ToolCallFailed", { agent_id: "root", tool_name: "run_shell" }));
  assert.deepEqual(board.rows[0].last, { tool: "run_shell", target: "pytest" });
  assert.equal(board.rows[0].tool, "");
  assert.equal(board.rows[0].target, "");
});

test("overlapping tools finish with their own target and keep pending activity visible", () => {
  const board = createAgentBoard();
  board.apply(event("RunStarted", { agent_id: "root" }));
  board.apply(event("ToolCallStarted", { agent_id: "root", tool_call_id: "read", tool_name: "read_file", target: "a.py" }));
  board.apply(event("ToolCallStarted", { agent_id: "root", tool_call_id: "edit", tool_name: "edit_file", target: "b.py" }));
  board.apply(event("ToolCallFinished", { agent_id: "root", tool_call_id: "read", tool_name: "read_file" }));
  assert.deepEqual(board.rows[0].last, { tool: "read_file", target: "a.py" });
  assert.equal(board.rows[0].tool, "edit_file");
  assert.equal(board.rows[0].target, "b.py");
  board.apply(event("ToolCallFailed", { agent_id: "root", tool_call_id: "edit", tool_name: "edit_file" }));
  assert.deepEqual(board.rows[0].last, { tool: "edit_file", target: "b.py" });
  assert.equal(board.rows[0].tool, "");
  assert.equal(board.rows[0].target, "");
});

test("waiting for approval keeps the current call target", () => {
  const board = createAgentBoard();
  board.apply(event("RunStarted", { agent_id: "root" }));
  board.apply(event("ToolCallStarted", { agent_id: "root", tool_name: "write_file", target: "a.py" }));
  board.apply(event("PermissionRequested", { agent_id: "root", tool_name: "write_file" }));
  assert.equal(board.rows[0].state, "waiting");
  assert.equal(board.rows[0].tool, "write_file");
  assert.equal(board.rows[0].target, "a.py");
});

test("a new root replaces the finished run and its subagents", () => {
  const board = createAgentBoard();
  const rows = board.rows;
  board.apply(event("RunStarted", { agent_id: "root-a", agent_name: "first" }));
  board.apply(event("SubagentSpawned", { agent_id: "root-a", subagent_agent_id: "child-a", subagent_name: "reader" }));
  board.apply(event("RunStarted", { agent_id: "child-a", agent_name: "reader" }));
  assert.deepEqual(board.rows.map((row) => row.agentId), ["root-a", "child-a"]);
  board.apply(event("SubagentStopped", { subagent_agent_id: "child-a" }));
  board.apply(event("RunFinished", { agent_id: "root-a" }));
  assert.deepEqual(board.rows, [
    { agentId: "root-a", parentAgentId: null, name: "first", state: "done", tool: "", target: "", last: null },
    { agentId: "child-a", parentAgentId: "root-a", name: "reader", state: "done", tool: "", target: "", last: null },
  ]);

  board.apply(event("RunStarted", { agent_id: "root-b", agent_name: "second" }));
  assert.equal(board.rows, rows);
  assert.deepEqual(board.rows, [
    { agentId: "root-b", parentAgentId: null, name: "second", state: "running", tool: "", target: "", last: null },
  ]);
});

test("agent board ignores unrelated and unknown events", () => {
  const board = createAgentBoard();
  board.apply(event("RunStarted", { agent_id: "root" }));
  const before = structuredClone(board.rows);
  board.apply({ kind: "reply", payload: { type: "RunFinished", agent_id: "root" } });
  board.apply(event("AssistantTextDelta", { agent_id: "root", text: "hello" }));
  board.apply(event("ToolCallStarted", { agent_id: "missing", tool_name: "read" }));
  assert.deepEqual(board.rows, before);
});

test("a repeated root start updates its row", () => {
  const board = createAgentBoard();
  board.apply(event("RunStarted", { agent_id: "root" }));
  board.apply(event("RunStarted", { agent_id: "root", agent_name: "leader" }));
  assert.deepEqual(board.rows, [
    { agentId: "root", parentAgentId: null, name: "leader", state: "running", tool: "", target: "", last: null },
  ]);
});

test("failed tool and run events clear the active tool", () => {
  const board = createAgentBoard();
  board.apply(event("RunStarted", { agent_id: "root" }));
  board.apply(event("ToolCallStarted", { agent_id: "root", tool_name: "read" }));
  board.apply(event("ToolCallFailed", { agent_id: "root" }));
  assert.equal(board.rows[0].tool, "");
  board.apply(event("ToolCallStarted", { agent_id: "root", tool_name: "write" }));
  board.apply(event("RunFailed", { agent_id: "root" }));
  assert.deepEqual(board.rows[0], { agentId: "root", parentAgentId: null, name: "agent", state: "done", tool: "", target: "", last: { tool: "read", target: "" } });
});

test("nested spawn parents and independent activity survive a root reset", () => {
  const board = createAgentBoard();
  const rows = board.rows;
  board.apply(event("RunStarted", { agent_id: "root", agent_name: "leader" }));
  board.apply(event("SubagentSpawned", { agent_id: "root", subagent_agent_id: "a", subagent_name: "A" }));
  board.apply(event("SubagentSpawned", { agent_id: "root", subagent_agent_id: "c", subagent_name: "C" }));
  board.apply(event("SubagentSpawned", { agent_id: "a", subagent_agent_id: "b", subagent_name: "B" }));
  assert.deepEqual(rows.map((row) => [row.agentId, row.parentAgentId]), [
    ["root", null], ["a", "root"], ["c", "root"], ["b", "a"],
  ]);
  board.apply(event("ToolCallStarted", { agent_id: "b", tool_name: "search" }));
  board.apply(event("SubagentStopped", { subagent_agent_id: "a" }));
  assert.deepEqual(rows.map((row) => [row.state, row.tool]), [
    ["running", ""], ["done", ""], ["running", ""], ["running", "search"],
  ]);
  board.apply(event("RunStarted", { agent_id: "next", agent_name: "next" }));
  assert.equal(board.rows, rows);
  assert.deepEqual(rows.map((row) => [row.agentId, row.parentAgentId]), [["next", null]]);
});

test("unknown spawner leaves its child visible as a root candidate", () => {
  const board = createAgentBoard();
  board.apply(event("RunStarted", { agent_id: "root" }));
  board.apply(event("SubagentSpawned", { agent_id: "missing", subagent_agent_id: "child", subagent_name: "reader" }));
  assert.deepEqual(board.rows.map((row) => [row.agentId, row.parentAgentId]), [
    ["root", null], ["child", "missing"],
  ]);
});
