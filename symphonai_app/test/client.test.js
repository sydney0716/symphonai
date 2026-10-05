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

test("provider choice uses an authenticated JSON request", async () => {
  const records = [];
  const client = createClient({ port: 4312, token: TOKEN, fetch: async (url, options) => {
    records.push({ url, options });
    return response(200, { selected: true });
  } });
  assert.deepEqual(await client.selectProvider({ name: "openai", model: "custom", base_url: "http://127.0.0.1:9000/v1", effort: "high" }), { selected: true });
  assert.equal(records[0].url, "http://127.0.0.1:4312/provider");
  assert.equal(records[0].options.headers.Authorization, `Bearer ${TOKEN}`);
  assert.deepEqual(JSON.parse(records[0].options.body), {
    name: "openai", model: "custom", base_url: "http://127.0.0.1:9000/v1", effort: "high",
  });
});

test("model listing uses authenticated query parameters", async () => {
  const records = [];
  const client = createClient({ port: 4312, token: TOKEN, fetch: async (url, options) => {
    records.push({ url, options });
    return response(200, {
      provider: "openai", state: "available",
      models: [{ id: "gpt-test", efforts: ["low", "high"] }], detail: "",
    });
  } });
  assert.deepEqual(
    await client.models("openai", "http://127.0.0.1:9000/v1"),
    {
      provider: "openai", state: "available",
      models: [{ id: "gpt-test", efforts: ["low", "high"] }], detail: "",
    },
  );
  assert.equal(
    records[0].url,
    "http://127.0.0.1:4312/models?provider=openai&base_url=http%3A%2F%2F127.0.0.1%3A9000%2Fv1",
  );
  assert.equal(records[0].options.method, "GET");
  assert.equal(records[0].options.headers.Authorization, `Bearer ${TOKEN}`);
});

test("permission mode uses an authenticated JSON request", async () => {
  const records = [];
  const client = createClient({ port: 4312, token: TOKEN, fetch: async (url, options) => {
    records.push({ url, options });
    return response(200, { mode: "plan" });
  } });
  assert.deepEqual(await client.selectMode("plan"), { mode: "plan" });
  assert.equal(records[0].url, "http://127.0.0.1:4312/mode");
  assert.equal(records[0].options.headers.Authorization, `Bearer ${TOKEN}`);
  assert.deepEqual(JSON.parse(records[0].options.body), { mode: "plan" });
});

test("permission mode refusal preserves the host message", async () => {
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async () => response(403, { error: "plan mode is forbidden by the workspace ceiling" }),
  });
  await assert.rejects(
    client.selectMode("plan"),
    (error) => error.status === 403 && error.message === "plan mode is forbidden by the workspace ceiling",
  );
});

test("agent controls post the agent id, action, and optional redirect text", async () => {
  const records = [];
  const client = createClient({ port: 4312, token: TOKEN, fetch: async (url, options) => {
    records.push({ url, options });
    return response(200, { agent_id: "agent-2", state: "paused" });
  } });
  assert.deepEqual(await client.controlAgent("agent-2", "pause"), {
    agent_id: "agent-2", state: "paused",
  });
  assert.deepEqual(await client.controlAgent("agent-2", "redirect", "Focus the tests"), {
    agent_id: "agent-2", state: "paused",
  });
  assert.equal(records[0].url, "http://127.0.0.1:4312/agent/control");
  assert.equal(records[0].options.headers.Authorization, `Bearer ${TOKEN}`);
  assert.deepEqual(JSON.parse(records[0].options.body), {
    agent_id: "agent-2", action: "pause",
  });
  assert.deepEqual(JSON.parse(records[1].options.body), {
    agent_id: "agent-2", action: "redirect", text: "Focus the tests",
  });
});

test("changes requests are authenticated and retain conflict paths", async () => {
  const records = [];
  const client = createClient({ port: 4312, token: TOKEN, fetch: async (url, options) => {
    records.push({ url, options });
    return records.length === 1
      ? response(200, { turns: [], files: [] })
      : response(409, { error: "changed outside", paths: ["a.py"] });
  } });
  assert.deepEqual(await client.changes(), { turns: [], files: [] });
  await assert.rejects(
    client.revertChanges({ path: "a.py" }),
    (error) => error.status === 409 && error.paths[0] === "a.py",
  );
  assert.equal(records[0].url, "http://127.0.0.1:4312/changes");
  assert.equal(records[0].options.method, "GET");
  assert.equal(records[0].options.headers.Authorization, `Bearer ${TOKEN}`);
  assert.equal(records[1].url, "http://127.0.0.1:4312/changes/revert");
  assert.equal(records[1].options.method, "POST");
  assert.deepEqual(JSON.parse(records[1].options.body), { path: "a.py" });
});

test("worktree actions post the name and preserve conflict messages", async () => {
  const records = [];
  const client = createClient({ port: 4312, token: TOKEN, fetch: async (url, options) => {
    records.push({ url, options });
    return records.length === 1
      ? response(409, { error: "patch does not apply" })
      : response(200, { applied: ["a.py"] });
  } });
  await assert.rejects(client.applyWorktree("w1"), (error) => error.status === 409 && error.message === "patch does not apply");
  assert.deepEqual(await client.applyWorktree("w1"), { applied: ["a.py"] });
  assert.equal(records[0].url, "http://127.0.0.1:4312/worktree/apply");
  assert.deepEqual(JSON.parse(records[0].options.body), { name: "w1" });
  assert.equal(records[0].options.headers.Authorization, `Bearer ${TOKEN}`);
  assert.equal(records[1].options.method, "POST");
});

test("manual compaction uses the authenticated JSON request", async () => {
  const records = [];
  const client = createClient({ port: 4312, token: TOKEN, fetch: async (url, options) => {
    records.push({ url, options });
    return response(200, {
      changed: true,
      before_tokens: 1200,
      after_tokens: 450,
      dropped_messages: 5,
    });
  } });
  assert.deepEqual(await client.compact("keep the API names"), {
    changed: true,
    before_tokens: 1200,
    after_tokens: 450,
    dropped_messages: 5,
  });
  assert.equal(records[0].url, "http://127.0.0.1:4312/compact");
  assert.equal(records[0].options.method, "POST");
  assert.equal(records[0].options.headers.Authorization, `Bearer ${TOKEN}`);
  assert.deepEqual(JSON.parse(records[0].options.body), { instructions: "keep the API names" });
});

test("agent definitions use authenticated read and write routes", async () => {
  const records = [];
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async (url, options) => {
      records.push({ url, options });
      return response(200, { name: "reviewer", scope: "project", text: 'prompt = "ok"\n' });
    },
  });
  assert.deepEqual(await client.agent("reviewer", "project"), {
    name: "reviewer", scope: "project", text: 'prompt = "ok"\n',
  });
  assert.deepEqual(await client.saveAgent("reviewer", "project", 'prompt = "ok"\n'), {
    name: "reviewer", scope: "project", text: 'prompt = "ok"\n',
  });
  assert.equal(records[0].url, "http://127.0.0.1:4312/agent?name=reviewer&scope=project");
  assert.equal(records[0].options.headers.Authorization, `Bearer ${TOKEN}`);
  assert.equal(records[1].url, "http://127.0.0.1:4312/agent");
  assert.deepEqual(JSON.parse(records[1].options.body), {
    name: "reviewer", scope: "project", text: 'prompt = "ok"\n',
  });
});

test("agent save preserves the host validation message", async () => {
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async () => response(400, { error: "/tmp/reviewer.toml: model: bad provider" }),
  });
  await assert.rejects(
    client.saveAgent("reviewer", "project", "broken"),
    (error) => error.message === "/tmp/reviewer.toml: model: bad provider" && error.status === 400,
  );
});

test("fork sends opaque session and record ids in the authenticated body", async () => {
  const records = [];
  const client = createClient({ port: 4312, token: TOKEN, fetch: async (url, options) => {
    records.push({ url, options });
    return response(200, { run_id: "fork-run", replayed: 1 });
  } });
  assert.deepEqual(await client.forkSession("source-run", "rec-opaque"), {
    run_id: "fork-run", replayed: 1,
  });
  assert.equal(records[0].url, "http://127.0.0.1:4312/session/fork");
  assert.equal(records[0].options.headers.Authorization, `Bearer ${TOKEN}`);
  assert.deepEqual(JSON.parse(records[0].options.body), {
    run_id: "source-run", record_id: "rec-opaque",
  });
});

test("fork sends force only when explicitly requested", async () => {
  const records = [];
  const client = createClient({ port: 4312, token: TOKEN, fetch: async (url, options) => {
    records.push(JSON.parse(options.body));
    return response(200, { run_id: "fork-run", replayed: 1 });
  } });
  await client.forkSession("source-run", "rec-opaque", true);
  assert.deepEqual(records, [{ run_id: "source-run", record_id: "rec-opaque", force: true }]);
});

test("goal methods use authenticated goal routes and conversation stats", async () => {
  const records = [];
  const replies = [
    { accepted: true, run_id: "goal-run", goal: { objective: "fix parser" } },
    { goal: { phase: "paused" } },
    { conversation: { goal: { phase: "paused" } } },
  ];
  const client = createClient({ port: 4312, token: TOKEN, fetch: async (url, options) => {
    records.push({ url, options });
    return response(200, replies[records.length - 1]);
  } });
  assert.deepEqual(await client.setGoal("fix parser", ["python3", "-m", "pytest"]), replies[0]);
  assert.deepEqual(await client.goalState("pause"), replies[1]);
  assert.deepEqual(await client.conversationStats(), replies[2]);
  assert.deepEqual(records.map(({ url, options }) => [url, options.method, options.headers.Authorization]), [
    ["http://127.0.0.1:4312/goal", "POST", `Bearer ${TOKEN}`],
    ["http://127.0.0.1:4312/goal/state", "POST", `Bearer ${TOKEN}`],
    ["http://127.0.0.1:4312/conversation", "GET", `Bearer ${TOKEN}`],
  ]);
  assert.deepEqual(JSON.parse(records[0].options.body), {
    objective: "fix parser", check: ["python3", "-m", "pytest"],
  });
  assert.deepEqual(JSON.parse(records[1].options.body), { action: "pause" });
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
    const path = new URL(url).pathname;
    return path === "/events"
      ? streamResponse([])
      : response(200, path === "/sessions" ? [] : { ok: true });
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

test("approval sends remember only when requested", async () => {
  const bodies = [];
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async (_url, options) => {
      bodies.push(JSON.parse(options.body));
      return response(200, { resolved: true });
    },
  });
  await client.approve("approval-1", true, "yes");
  await client.approve("approval-2", true, "", true);
  assert.deepEqual(bodies, [
    { approval_id: "approval-1", allowed: true, reason: "yes" },
    { approval_id: "approval-2", allowed: true, reason: "", remember: true },
  ]);
});

test("prompt adds attachments only when provided", async () => {
  const bodies = [];
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async (_url, options) => {
      bodies.push(JSON.parse(options.body));
      return response(200, { accepted: true });
    },
  });
  await client.prompt("text");
  await client.prompt("", [{ data: "AQ==", filename: "a.png" }]);
  assert.deepEqual(bodies, [
    { prompt: "text" },
    { prompt: "", attachments: [{ data: "AQ==", filename: "a.png" }] },
  ]);
});

test("files encodes the query and limit as search parameters", async () => {
  const records = [];
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async (url, options) => {
      records.push({ url, options });
      return response(200, { files: ["src/a.py"], truncated: false });
    },
  });
  assert.deepEqual(await client.files("a b", 7), { files: ["src/a.py"], truncated: false });
  assert.equal(new URL(records[0].url).pathname, "/files");
  assert.deepEqual(Object.fromEntries(new URL(records[0].url).searchParams), {
    query: "a b", limit: "7",
  });
});

test("history requests the selected limit", async () => {
  const records = [];
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async (url, options) => {
      records.push({ url, options });
      return response(200, { prompts: ["latest"] });
    },
  });
  assert.deepEqual(await client.history(12), { prompts: ["latest"] });
  assert.equal(new URL(records[0].url).pathname, "/history");
  assert.equal(new URL(records[0].url).searchParams.get("limit"), "12");
});

test("history requests the selected limit", async () => {
  const records = [];
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async (url, options) => {
      records.push({ url, options });
      return response(200, { prompts: ["latest"] });
    },
  });
  assert.deepEqual(await client.history(12), { prompts: ["latest"] });
  assert.equal(new URL(records[0].url).pathname, "/history");
  assert.equal(new URL(records[0].url).searchParams.get("limit"), "12");
});

test("sessions accepts only an array while other routes still require objects", async () => {
  const sessions = [{ run_id: "run-one" }];
  const paths = [];
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async (url) => {
      paths.push(new URL(url).pathname + new URL(url).search);
      return response(200, sessions);
    },
  });
  assert.deepEqual(await client.sessions(200), sessions);
  assert.deepEqual(paths, ["/sessions?limit=200"]);
  await assert.rejects(client.health(), (error) => {
    assert.ok(error instanceof ProtocolError);
    assert.match(error.message, /must be an object/);
    return true;
  });

  const invalidClient = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async () => response(200, { sessions }),
  });
  await assert.rejects(invalidClient.sessions(), (error) => {
    assert.ok(error instanceof ProtocolError);
    assert.match(error.message, /must be an array/);
    return true;
  });
});

test("file encodes every significant query character and returns the reply", async () => {
  const cases = [
    ["space", "specs/18/a space.md"],
    ["plus", "specs/18/a+b.md"],
    ["ampersand", "specs/18/a&b.md"],
    ["hash", "specs/18/a#b.md"],
    ["non-ASCII", "specs/18/한글.md"],
  ];
  assert.equal(cases.length, 5);

  for (const [label, path] of cases) {
    let record;
    const client = createClient({
      port: 4312,
      token: TOKEN,
      fetch: async (url, options) => {
        record = { url, options };
        return response(200, { path, text: `${label} text` });
      },
    });

    assert.deepEqual(await client.file(path), {
      path,
      text: `${label} text`,
    });
    const url = new URL(record.url);
    assert.equal(url.pathname, "/file", label);
    assert.equal(url.searchParams.get("path"), path, label);
    assert.equal(
      record.url,
      `http://127.0.0.1:4312/file?${new URLSearchParams({ path })}`,
      label,
    );
    assert.equal(record.options.method, "GET", label);
    assert.equal(record.options.headers.Authorization, `Bearer ${TOKEN}`, label);
    assert.ok(!record.url.includes(TOKEN), label);
  }
});

test("file failures expose status without exposing credentials", async () => {
  const statuses = [401, 403, 404, 413];
  const failures = new Map();
  assert.equal(statuses.length, 4);

  for (const status of statuses) {
    let recordedUrl;
    const client = createClient({
      port: 4312,
      token: TOKEN,
      fetch: async (url) => {
        recordedUrl = url;
        return response(status, { error: TOKEN, Authorization: `Bearer ${TOKEN}` });
      },
    });
    let thrown;
    try {
      await client.file("specs/18/missing report.md");
    } catch (error) {
      thrown = error;
    }
    assert.ok(thrown instanceof ProtocolError, `${status} was not ProtocolError`);
    assert.equal(thrown.status, status);
    assert.match(thrown.message, new RegExp(String(status)));
    assert.ok(!stringify(thrown).includes(TOKEN));
    assert.ok(!stringify(thrown).includes("Authorization"));
    assert.ok(!recordedUrl.includes(TOKEN));
    failures.set(status, thrown);
  }

  assert.equal(failures.get(404).status, 404);
  assert.notEqual(failures.get(404).status, failures.get(403).status);
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

test("spec workflow methods send the route payloads", async () => {
  const records = [];
  const client = createClient({
    port: 4312,
    token: TOKEN,
    fetch: async (url, options) => {
      records.push({ url, options });
      return response(200, url.endsWith("/spec/runs") ? [] : { session_id: "s", run_id: "r" });
    },
  });
  await client.planSpec("39", 2);
  await client.runSpec("specs/39/39b-run-a-spec.md");
  await client.reviewSpec("s");
  await client.commitSpec("s", "39b: run a spec");
  await client.specRuns();
  assert.deepEqual(records.map(({ url, options }) => [new URL(url).pathname, options.method, options.body && JSON.parse(options.body)]), [
    ["/spec/plan", "POST", { phase: "39", item: 2 }],
    ["/spec/run", "POST", { path: "specs/39/39b-run-a-spec.md" }],
    ["/spec/review", "POST", { session_id: "s" }],
    ["/spec/commit", "POST", { session_id: "s", message: "39b: run a spec" }],
    ["/spec/runs", "GET", undefined],
  ]);
});
