import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  createClient,
  parseHandshake,
  readEventStream,
} from "../src/client.js";
import { ProtocolError } from "../src/protocol.js";

const TOKEN = "secret-token-that-must-not-leak";
const ENCODER = new TextEncoder();

function response(status, value, { malformed = false, body = null } = {}) {
  return {
    status,
    body,
    async json() {
      if (malformed) {
        throw new SyntaxError("malformed");
      }
      return value;
    },
  };
}

function streamBody(chunks) {
  let index = 0;
  return {
    getReader() {
      return {
        async read() {
          if (index === chunks.length) {
            return { done: true, value: undefined };
          }
          const value = ENCODER.encode(chunks[index]);
          index += 1;
          return { done: false, value };
        },
      };
    },
  };
}

function streamResponse(chunks, status = 200) {
  return response(status, null, { body: streamBody(chunks) });
}

function frame(kind = "event", payload = { type: "RunStarted" }, version = 1) {
  return JSON.stringify({ protocol_version: version, kind, payload });
}

function stringify(value) {
  if (value instanceof Error) {
    return `${value.name}: ${value.message}`;
  }
  return JSON.stringify(value);
}

test("client source does not reference EventSource", async () => {
  const source = await readFile(new URL("../src/client.js", import.meta.url), "utf8");
  assert.ok(!source.includes("EventSource"));
});

test("parseHandshake accepts only a port and nonempty token", () => {
  assert.deepEqual(parseHandshake('{"port":4312,"token":"abc"}'), {
    port: 4312,
    token: "abc",
  });
  for (const line of [
    "not json",
    "[]",
    "{}",
    '{"port":4312}',
    '{"token":"abc"}',
    '{"port":"4312","token":"abc"}',
    '{"port":4312,"token":""}',
  ]) {
    assert.throws(() => parseHandshake(line), ProtocolError);
  }
});

test("all eight calls authorize by header and never by URL", async () => {
  const records = [];
  const fetch = async (url, options) => {
    records.push({ url, options });
    return new URL(url).pathname === "/events"
      ? streamResponse([])
      : response(200, { ok: true });
  };
  const client = createClient({ port: 4312, token: TOKEN, fetch });
  const subscription = client.events(() => {});
  await subscription.done;
  await client.prompt("hello");
  await client.stop("done");
  await client.approve("approval-1", true, "yes");
  await client.openSession("run-1");
  await client.approvals();
  await client.sessions();
  await client.health();

  assert.equal(records.length, 8);
  assert.deepEqual(
    records.map(({ url }) => new URL(url).pathname),
    [
      "/events",
      "/prompt",
      "/stop",
      "/approval",
      "/session/open",
      "/approvals",
      "/sessions",
      "/health",
    ],
  );
  for (const { url, options } of records) {
    assert.equal(options.headers.Authorization, `Bearer ${TOKEN}`);
    assert.ok(!url.includes(TOKEN));
    assert.equal(new URL(url).search, "");
  }
});

test("keepalive comments never become frames", async () => {
  const delivered = [];
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async () =>
      streamResponse([
        ": keepalive\n\n: keepalive\n\n",
        `: keepalive\n\ndata: ${frame()}\n\n`,
      ]),
  });
  await client.events((value) => delivered.push(value)).done;
  assert.equal(delivered.length, 1);
  assert.deepEqual(delivered[0], {
    kind: "event",
    payload: { type: "RunStarted" },
  });
});

test("multiple data lines join and strip exactly one leading space", async () => {
  const received = [];
  await readEventStream(
    streamBody(["data: first\ndata:   second\n\n"]),
    (data) => received.push(data),
  );
  assert.deepEqual(received, ["first\n  second"]);
});

test("chunk boundaries are ignored at all three significant positions", async () => {
  const wire = `data: ${frame()}\n\n`;
  const cases = [
    ["inside line", 2],
    ["inside data value", wire.indexOf("RunStarted") + 3],
    ["between blank-line newlines", wire.length - 1],
  ];
  assert.equal(cases.length, 3);
  for (const [label, offset] of cases) {
    const delivered = [];
    const client = createClient({
      port: 4312,
      token: TOKEN,
      fetch: async () => streamResponse([wire.slice(0, offset), wire.slice(offset)]),
    });
    await client.events((value) => delivered.push(value)).done;
    assert.deepEqual(
      delivered,
      [{ kind: "event", payload: { type: "RunStarted" } }],
      label,
    );
  }
});

test("new reader preserves dropped, unknown, and future-version behavior", async () => {
  const delivered = [];
  const unknown = { type: "FutureEvent", future: { nested: true } };
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async () =>
      streamResponse([
        `data: ${frame("error", { dropped: 3 })}\n\n`,
        `data: ${frame("event", unknown)}\n\n`,
      ]),
  });
  const subscription = client.events((value) => delivered.push(value));
  await subscription.done;
  assert.deepEqual(delivered, [
    { kind: "error", dropped: 3 },
    { kind: "event", payload: unknown },
  ]);
  assert.equal(client.droppedFrames, 3);
  assert.equal(subscription.hasDroppedFrames, true);

  const futureClient = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async () => streamResponse([`data: ${frame("event", {}, 2)}\n\n`]),
  });
  await assert.rejects(futureClient.events(() => {}).done, (error) => {
    assert.ok(error instanceof ProtocolError);
    assert.match(error.message, /2.*1/);
    return true;
  });
});

test("non-200 event streams expose status but never credentials", async () => {
  const records = [];
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async (url, options) => {
      records.push({ url, options });
      return response(401, null);
    },
  });
  let thrown;
  try {
    await client.events(() => {}).done;
  } catch (error) {
    thrown = error;
  }
  assert.ok(thrown instanceof Error);
  assert.match(thrown.message, /401/);
  assert.ok(!stringify(thrown).includes(TOKEN));
  assert.ok(!stringify(thrown).includes("Authorization"));
  assert.equal(records[0].options.headers.Authorization, `Bearer ${TOKEN}`);
  assert.equal(new URL(records[0].url).search, "");
});

test("prompt conflicts preserve only the active run id", async () => {
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async () =>
      response(409, { error: `busy ${TOKEN}`, run_id: "active-run" }),
  });
  const result = await client.prompt("second");
  assert.deepEqual(result, {
    accepted: false,
    conflict: true,
    run_id: "active-run",
  });
  assert.ok(!stringify(result).includes(TOKEN));
  assert.ok(!stringify(result).includes("Authorization"));
});

test("failing requests never expose the token or authorization", async () => {
  const failures = [
    ["401", async () => response(401, null)],
    ["409", async () => response(409, { error: TOKEN })],
    ["malformed", async () => response(200, null, { malformed: true })],
    ["network", async () => { throw new Error(`network failed for ${TOKEN}`); }],
  ];
  assert.equal(failures.length, 4);
  for (const [label, fetch] of failures) {
    const client = createClient({ port: 4312, token: TOKEN, fetch });
    let thrown;
    try {
      await client.sessions();
    } catch (error) {
      thrown = error;
    }
    assert.ok(thrown instanceof Error, `${label} did not throw`);
    assert.ok(!stringify(thrown).includes(TOKEN), `${label} leaked the token`);
    assert.ok(
      !stringify(thrown).includes(`Bearer ${TOKEN}`),
      `${label} leaked authorization`,
    );
  }
});

test("ProtocolError messages created by a token-bearing client are sanitized", () => {
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async () => response(200, {}),
  });
  assert.throws(() => client.events(null), (error) => {
    assert.ok(error instanceof ProtocolError);
    assert.ok(!stringify(error).includes(TOKEN));
    assert.ok(!stringify(error).includes("Authorization"));
    return true;
  });
});
