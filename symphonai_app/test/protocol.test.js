import test from "node:test";
import assert from "node:assert/strict";

import {
  KNOWN_EVENT_TYPES,
  PROTOCOL_VERSION,
  ProtocolError,
  decodeEvent,
  decodeFrame,
  encodeRequest,
} from "../src/protocol.js";

test("decodeFrame accepts every frame kind", () => {
  for (const kind of ["event", "reply", "error", "approval_requested"]) {
    assert.deepEqual(
      decodeFrame(JSON.stringify({ protocol_version: 1, kind, payload: { kind } })),
      { kind, payload: { kind } },
    );
  }
});

test("decodeFrame enumerates malformed frames", () => {
  const invalid = [
    [
      JSON.stringify({ protocol_version: 2, kind: "event", payload: {} }),
      /2.*1/,
    ],
    [JSON.stringify({ protocol_version: 1, kind: "future", payload: {} }), /kind/],
    [JSON.stringify({ protocol_version: 1, kind: "event" }), /payload/],
    [JSON.stringify({ protocol_version: 1, kind: "event", payload: [] }), /payload/],
    ["{", /JSON/],
  ];
  assert.equal(invalid.length, 5);
  for (const [text, pattern] of invalid) {
    assert.throws(() => decodeFrame(text), (error) => {
      assert.ok(error instanceof ProtocolError);
      assert.match(error.message, pattern);
      return true;
    });
  }
});

test("decodeEvent recognizes all seventeen documented event types", () => {
  const documented = [
    "RunStarted",
    "RunFinished",
    "RunFailed",
    "TurnStarted",
    "TurnFinished",
    "AssistantTextDelta",
    "ToolCallStarted",
    "ToolCallFinished",
    "PromptSubmitted",
    "ToolCallFailed",
    "PermissionRequested",
    "PermissionDenied",
    "SessionStarted",
    "SessionEnded",
    "SubagentSpawned",
    "SubagentStopped",
    "CompactionApplied",
  ];
  assert.equal(documented.length, 17);
  assert.deepEqual(KNOWN_EVENT_TYPES, documented);
  for (const type of documented) {
    assert.deepEqual(decodeEvent({ type, marker: type }), {
      type,
      known: true,
      fields: { marker: type },
    });
  }
});

test("decodeEvent preserves unknown event types", () => {
  const payload = {
    type: "RuntimeLearnedToSing",
    future_field: { nested: [1, 2, 3] },
    never_seen_before: true,
  };
  assert.deepEqual(decodeEvent(payload), {
    type: payload.type,
    known: false,
    fields: {
      future_field: payload.future_field,
      never_seen_before: true,
    },
  });
});

test("decodeEvent preserves unknown fields on known events", () => {
  const future = { nested: ["kept"] };
  assert.deepEqual(decodeEvent({ type: "RunStarted", future }), {
    type: "RunStarted",
    known: true,
    fields: { future },
  });
});

test("encodeRequest maps all request kinds without I/O", () => {
  const cases = [
    ["prompt", "/prompt"],
    ["stop", "/stop"],
    ["approval", "/approval"],
    ["session/open", "/session/open"],
  ];
  for (const [kind, path] of cases) {
    const payload = { marker: kind };
    assert.deepEqual(encodeRequest(kind, payload), { path, body: payload });
  }
  assert.throws(() => encodeRequest("future", {}), ProtocolError);
});

test("the exported protocol version is one", () => {
  assert.equal(PROTOCOL_VERSION, 1);
});
