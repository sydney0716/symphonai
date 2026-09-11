import test from "node:test";
import assert from "node:assert/strict";

import { createApprovals } from "../src/approvals.js";

function question(id, overrides = {}) {
  return {
    approval_id: id,
    operation: `operation-${id}`,
    target: `target-${id}`,
    details: `details-${id}`,
    ...overrides,
  };
}

function frame(id, overrides = {}) {
  return {
    kind: "approval_requested",
    payload: question(id, overrides),
  };
}

function client({ pending = [], approve } = {}) {
  const calls = { approve: [], approvals: 0 };
  return {
    calls,
    async approve(id, allowed, reason) {
      calls.approve.push({ approval_id: id, allowed, reason });
      return approve ? approve(id, allowed, reason) : { resolved: true };
    },
    async approvals() {
      calls.approvals += 1;
      return { pending };
    },
  };
}

test("approval frames keep only a meaningful tool call id", async () => {
  const boundary = client();
  const approvals = createApprovals({ client: boundary });
  await approvals.onFrame(frame("A", { tool_call_id: "tool-call-A" }));
  await approvals.onFrame(frame("B", { tool_call_id: "" }));
  assert.deepEqual(approvals.state, new Map([
    ["A", {
      operation: "operation-A",
      target: "target-A",
      details: "details-A",
      state: "open",
      tool_call_id: "tool-call-A",
    }],
    ["B", {
      operation: "operation-B",
      target: "target-B",
      details: "details-B",
      state: "open",
    }],
  ]));
});

test("a repeated frame neither duplicates nor resets an answering question", async () => {
  let resolveApproval;
  const pending = new Promise((resolve) => {
    resolveApproval = resolve;
  });
  const boundary = client({ approve: () => pending });
  const approvals = createApprovals({ client: boundary });
  await approvals.onFrame(frame("A"));
  const answering = approvals.answer("A", true, "because");
  assert.equal(approvals.state.get("A").state, "answering");
  await approvals.onFrame(frame("A", { target: "replacement" }));
  assert.equal(approvals.state.size, 1);
  assert.equal(approvals.state.get("A").state, "answering");
  assert.equal(approvals.state.get("A").target, "target-A");
  resolveApproval({ resolved: true });
  await answering;
});

test("answer moves open through answering to resolved and sends exact data", async () => {
  let resolveApproval;
  const pending = new Promise((resolve) => {
    resolveApproval = resolve;
  });
  const boundary = client({ approve: () => pending });
  const approvals = createApprovals({ client: boundary });
  await approvals.onFrame(frame("A"));
  const answering = approvals.answer("A", false, "not this file");
  assert.equal(approvals.state.get("A").state, "answering");
  assert.deepEqual(boundary.calls.approve, [{
    approval_id: "A",
    allowed: false,
    reason: "not this file",
  }]);
  resolveApproval({ resolved: true });
  await answering;
  assert.equal(approvals.state.get("A").state, "resolved");
});

test("resync leaves an in-flight answer live until it resolves", async () => {
  let resolveApproval;
  const pending = new Promise((resolve) => {
    resolveApproval = resolve;
  });
  const boundary = client({ approve: () => pending });
  const approvals = createApprovals({ client: boundary });
  await approvals.onFrame(frame("A"));

  const answer = approvals.answer("A", true, "because");
  assert.equal(approvals.state.get("A").state, "answering");
  await approvals.resync();
  const duringFlight = approvals.state.get("A");
  assert.equal(duringFlight.state, "answering");
  assert.equal(Object.hasOwn(duringFlight, "reason"), false);

  resolveApproval({ resolved: true });
  await answer;
  const resolved = approvals.state.get("A");
  assert.equal(resolved.state, "resolved");
  assert.equal(Object.hasOwn(resolved, "reason"), false);
});

test("a 404 makes an expired question stale with a displayable reason", async () => {
  const missing = new Error("host request failed with status 404");
  const boundary = client({ approve: async () => { throw missing; } });
  const approvals = createApprovals({ client: boundary });
  await approvals.onFrame(frame("A"));
  await approvals.answer("A", true, "");
  const stale = approvals.state.get("A");
  assert.equal(stale.state, "stale");
  assert.match(stale.reason, /expired|answered elsewhere/i);
  assert.ok(!Object.hasOwn(stale, "error"));
});

test("an approval id is answered at most once", async () => {
  const boundary = client();
  const approvals = createApprovals({ client: boundary });
  await approvals.onFrame(frame("A"));
  await approvals.answer("A", true, "first");
  await approvals.answer("A", false, "second");
  assert.equal(boundary.calls.approve.length, 1);
  assert.equal(approvals.state.get("A").state, "resolved");
});

test("resync reconciles missing, retained, and unseen ids together", async () => {
  const boundary = client({ pending: [question("B"), question("C")] });
  const approvals = createApprovals({ client: boundary });
  await approvals.onFrame(frame("A"));
  await approvals.onFrame(frame("B"));
  const beforeB = approvals.state.get("B");
  await approvals.resync();
  assert.equal(approvals.state.get("A").state, "stale");
  assert.deepEqual(approvals.state.get("B"), beforeB);
  assert.equal(approvals.state.get("C").state, "open");
  assert.deepEqual([...approvals.state.keys()], ["A", "B", "C"]);
});

test("every dropped notice triggers exactly one resync", async () => {
  const boundary = client();
  const approvals = createApprovals({ client: boundary });
  await approvals.onFrame({ kind: "error", dropped: 2 });
  await approvals.onFrame({ kind: "error", payload: { dropped: 1 } });
  assert.equal(boundary.calls.approvals, 2);
});

test("a question discovered only by resync can be answered", async () => {
  const boundary = client({ pending: [question("C")] });
  const approvals = createApprovals({ client: boundary });
  await approvals.resync();
  await approvals.answer("C", true, "found it");
  assert.deepEqual(boundary.calls.approve, [{
    approval_id: "C",
    allowed: true,
    reason: "found it",
  }]);
  assert.equal(approvals.state.get("C").state, "resolved");
});

test("an omitted reason is sent only as the protocol empty default", async () => {
  const boundary = client();
  const approvals = createApprovals({ client: boundary });
  await approvals.onFrame(frame("A"));
  await approvals.answer("A", true);
  assert.deepEqual(boundary.calls.approve, [{
    approval_id: "A",
    allowed: true,
    reason: "",
  }]);
});
