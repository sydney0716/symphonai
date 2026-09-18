import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { start } from "../src/app.js";
import { createClient } from "../src/client.js";
import { DISPATCHING, IDLE, RUNNING } from "../src/turn.js";

const BASE = "specs/18/18a-a-client-for-the-boundary.md";
const FOLLOW_UP = "specs/18/18aF-a-double-more-capable-than-the-real-thing.md";
const REPORT = "specs/report/18/18a-a-client-for-the-boundary-report.md";
const OPEN_BASE = "specs/18/18b-three-states-and-a-queue.md";
const OPEN_REPORT = "specs/report/18/18b-three-states-and-a-queue-report.md";

class FakeElement {
  constructor(tagName, id = "") {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.children = [];
    this.className = "";
    this.open = false;
    this.textContent = "";
    this.type = "";
    this.value = "";
    this.listeners = new Map();
  }

  append(...children) {
    this.children.push(...children);
  }

  replaceChildren(...children) {
    this.children = [...children];
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  async dispatch(type) {
    const event = { preventDefault() {} };
    for (const listener of this.listeners.get(type) ?? []) {
      await listener(event);
    }
  }
}

export class FakeDocument {
  constructor() {
    const tags = {
      "app-shell": "div",
      sidebar: "nav",
      "home-link": "h1",
      "sidebar-toggle": "button",
      "rail-toggle": "button",
      "page-links": "div",
      page: "div",
      "status-rail": "aside",
      "chat-pane": "main",
      "prompt-form": "form",
      prompt: "textarea",
    };
    this.elements = new Map(
      [
        "app-shell",
        "sidebar",
        "home-link",
        "sidebar-toggle",
        "rail-toggle",
        "page-links",
        "page",
        "status-rail",
        "agents",
        "chat-pane",
        "roadmap",
        "spec",
        "run-notice",
        "chat",
        "approvals",
        "prompt-form",
        "prompt",
        "prompt-error",
      ].map((id) => [id, new FakeElement(tags[id] ?? "div", id)]),
    );
    const get = (id) => this.elements.get(id);
    get("app-shell").className = "app-shell";
    get("chat-pane").className = "chat-pane";
    get("sidebar").append(get("home-link"), get("page-links"));
    get("status-rail").append(get("agents"), get("roadmap"), get("spec"));
    get("prompt-form").append(get("prompt"), get("prompt-error"));
    get("chat-pane").append(get("run-notice"), get("chat"), get("approvals"), get("prompt-form"));
    get("page").append(get("chat-pane"));
    get("app-shell").append(get("sidebar"), get("sidebar-toggle"), get("page"), get("status-rail"), get("rail-toggle"));
    this.body = new FakeElement("body", "body");
    this.body.append(get("app-shell"));
  }

  createElement(tagName) {
    return new FakeElement(tagName);
  }

  getElementById(id) {
    return this.elements.get(id) ?? null;
  }
}

function fakeGlobal({ fragment = "", stored = null, storageThrows = false } = {}) {
  const listeners = new Map();
  const writes = [];
  const global = {
    location: { hash: fragment },
    localStorage: {
      getItem(key) {
        if (storageThrows) {
          throw new Error("storage read failed");
        }
        assert.equal(key, "symphonai.route");
        return stored;
      },
      setItem(key, value) {
        if (storageThrows) {
          throw new Error("storage write failed");
        }
        writes.push([key, value]);
      },
    },
    addEventListener(type, listener) {
      listeners.set(type, listener);
    },
  };
  return {
    global,
    writes,
    dispatch(type) {
      return listeners.get(type)?.();
    },
  };
}

function walk(root) {
  return [root, ...root.children.flatMap(walk)];
}

function visibleText(root) {
  return walk(root).map((value) => value.textContent).join("\n");
}

function find(root, predicate) {
  return walk(root).find(predicate);
}

function fixtureRoadmap({ allDone = false } = {}) {
  return JSON.stringify({
    goal: "Ship a visible app",
    phases: [
      {
        id: "17",
        name: "Host process",
        status: "done",
        items: [{ title: "Boundary", spec: BASE, done: true }],
      },
      {
        id: "18",
        name: "Desktop app",
        status: allDone ? "done" : "in_progress",
        items: [{ title: "Open boundary", spec: OPEN_BASE }],
      },
      {
        id: "20",
        name: "Readable app",
        status: allDone ? "done" : "next",
        items: [{ title: "Later phase", spec: FOLLOW_UP }],
      },
    ],
  });
}

export function fakeClient(
  roadmapText = fixtureRoadmap(),
  {
    project = { repo_root: "/work/current", name: "current" },
    sessions = [],
    settings = { settings: {} },
    health = { protocol_version: 1, state: "idle", run_id: null, runtime_run_id: null },
  } = {},
) {
  const calls = { approve: [], credentials: [], file: [], openSession: [], prompt: [], sessions: [], settings: 0 };
  let eventCallback;
  let resolvePrompt;
  const promptReply = new Promise((resolve) => {
    resolvePrompt = resolve;
  });
  const files = new Map([
    ["docs/roadmap.json", { path: "docs/roadmap.json", text: roadmapText }],
    [BASE, { path: BASE, text: "base spec text" }],
    [REPORT, { path: REPORT, text: "base report text" }],
    [OPEN_BASE, { path: OPEN_BASE, text: "open spec text" }],
    [OPEN_REPORT, { path: OPEN_REPORT, text: "open report text" }],
  ]);
  return {
    calls,
    emit(frame) {
      return eventCallback(frame);
    },
    resolvePrompt,
    async file(path) {
      calls.file.push(path);
      const reply = files.get(path);
      if (reply === undefined) {
        const error = new Error("missing");
        error.status = 404;
        throw error;
      }
      return reply;
    },
    events(callback) {
      eventCallback = callback;
      return { close() {}, done: Promise.resolve() };
    },
    prompt(text) {
      calls.prompt.push(text);
      return promptReply;
    },
    async stop() {
      return { accepted: true };
    },
    async approve(id, allowed, reason) {
      calls.approve.push({ id, allowed, reason });
      return { resolved: true };
    },
    async approvals() {
      return { pending: [] };
    },
    async project() {
      return project;
    },
    async settings() {
      calls.settings += 1;
      return settings;
    },
    async health() {
      return health;
    },
    async storeCredential(name, value) {
      calls.credentials.push({ name, value });
      return { stored: true, name };
    },
    async sessions(limit) {
      calls.sessions.push(limit);
      return sessions;
    },
    async openSession(runId) {
      calls.openSession.push(runId);
      return { run_id: runId };
    },
  };
}

function eventFrame(type, fields = {}) {
  return {
    kind: "event",
    payload: {
      type,
      agent_id: "agent-1",
      run_id: "run-1",
      turn_id: "turn-1",
      schema_version: 1,
      ...fields,
    },
  };
}

function toolStarted(id, name, target = "") {
  return eventFrame("ToolCallStarted", {
    tool_name: name,
    tool_call_id: id,
    target,
  });
}

function toolFinished(id, name, fields = {}) {
  return eventFrame("ToolCallFinished", {
    tool_name: name,
    tool_call_id: id,
    ok: true,
    result_kind: "",
    result_path: "",
    lines_added: 0,
    lines_removed: 0,
    diff: "",
    truncated: false,
    ...fields,
  });
}

test("start renders the chat page and prepares the roadmap", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const app = await start({ global: {}, document, client });
  const roadmap = document.getElementById("roadmap");
  const page = document.getElementById("page");

  assert.deepEqual(page.children, [document.getElementById("chat-pane")]);
  assert.equal(app.route().page, "chat");
  assert.ok(!visibleText(roadmap).includes("Ship a visible app"));
  assert.equal(find(roadmap, (value) => value.className === "roadmap-goal"), undefined);
  assert.match(visibleText(roadmap), /18 · Desktop app/);
  assert.match(visibleText(roadmap), /0\/1 · in_progress/);
  assert.equal(roadmap.children.length, 3);
  assert.ok(roadmap.children.every((phase) => phase.tagName === "DETAILS"));
  assert.deepEqual(roadmap.children.map((phase) => phase.open), [false, true, false]);
  assert.deepEqual(
    roadmap.children.map((phase) => phase.children[0].textContent),
    [
      "17 · Host process — 1/1 · done",
      "18 · Desktop app — 0/1 · in_progress",
      "20 · Readable app — 0/1 · next",
    ],
  );
  for (const phase of roadmap.children) {
    const [summary, ...items] = phase.children;
    assert.equal(summary.tagName, "SUMMARY");
    assert.ok(!summary.textContent.includes("Boundary"));
    assert.ok(items.every((item) => item.tagName === "BUTTON"));
    assert.equal(summary.children.length, 0);
  }
  assert.ok(document.getElementById("chat"));
  assert.equal(document.getElementById("turn-state"), null);
  assert.equal(document.getElementById("prompt-error").textContent, "");
  assert.equal(app.client, client);
  assert.ok(app.transcript);
  assert.deepEqual(app.transcript.model, []);
});

test("sidebar groups sessions and keeps only the current project openable", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    project: { repo_root: "/work/current", name: "Current project" },
    sessions: [
      { run_id: "current-old", title: "Current chat", repo_root: "/work/current", updated_at: "2026-01-01" },
      { run_id: "other-new", title: "Other chat", repo_root: "/work/other", updated_at: "2026-03-01" },
      { run_id: "legacy", title: null, repo_root: "", updated_at: "2026-02-01" },
    ],
  });

  await start({ global: {}, document, client });
  const projects = find(document.getElementById("sidebar"), (value) =>
    value.className === "projects"
  );
  const groups = projects.children;

  assert.deepEqual(groups.map((group) => group.children[0].textContent), [
    "Current project",
    "other",
    "Unknown project",
  ]);
  assert.deepEqual(groups.map((group) => group.open), [true, false, false]);
  assert.match(visibleText(groups[1]), /Not openable from this host\./);
  assert.match(visibleText(groups[2]), /legacy/);
  assert.equal(find(groups[1], (value) => value.tagName === "BUTTON"), undefined);

  await find(groups[0], (value) => value.tagName === "BUTTON").dispatch("click");
  assert.deepEqual(client.calls.openSession, ["current-old"]);
});

test("sidebar shows the current project when it has no sessions", async () => {
  const document = new FakeDocument();
  await start({
    global: {},
    document,
    client: fakeClient(fixtureRoadmap(), {
      project: { repo_root: "/work/empty", name: "empty" },
      sessions: [],
    }),
  });

  const projects = find(document.getElementById("sidebar"), (value) =>
    value.className === "projects"
  );
  assert.equal(projects.children.length, 1);
  assert.equal(projects.children[0].open, true);
  assert.equal(projects.children[0].children[0].textContent, "empty");
  assert.match(visibleText(projects.children[0]), /No chats yet\./);
});

test("sidebar requests 200 sessions without a limit notice", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    sessions: Array.from({ length: 200 }, (_, index) => ({
      run_id: `run-${index}`,
      repo_root: "/work/current",
      updated_at: `2026-01-01T00:00:${String(index).padStart(2, "0")}Z`,
    })),
  });
  await start({ global: {}, document, client });

  assert.deepEqual(client.calls.sessions, [200]);
  assert.doesNotMatch(visibleText(document.getElementById("sidebar")), /Showing the .* most recent sessions/);
});

test("page navigation preserves the rendered transcript and stores the route", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal();
  const client = fakeClient();
  const app = await start({ global: browser.global, document, client });
  const page = document.getElementById("page");
  const links = document.getElementById("page-links");
  const chatPane = document.getElementById("chat-pane");

  await client.emit(eventFrame("AssistantTextDelta", { text: "still here" }));
  await find(links, (value) => value.textContent === "Settings").dispatch("click");
  assert.equal(page.children[0].className, "settings-pane");
  assert.equal(app.route().page, "settings");
  assert.equal(browser.global.location.hash, "#/settings");
  assert.deepEqual(browser.writes.at(-1), ["symphonai.route", "#/settings"]);

  await document.getElementById("home-link").dispatch("click");
  assert.deepEqual(page.children, [chatPane]);
  assert.equal(document.getElementById("chat").children[0].textContent, "still here");
});

test("old stored roadmap routes fall back to chat, but a URL fragment wins", async () => {
  const storedDocument = new FakeDocument();
  const stored = fakeGlobal({ stored: "#/roadmap" });
  await start({ global: stored.global, document: storedDocument, client: fakeClient() });
  assert.deepEqual(
    storedDocument.getElementById("page").children,
    [storedDocument.getElementById("chat-pane")],
  );

  const fragmentDocument = new FakeDocument();
  const fragment = fakeGlobal({ fragment: "#/settings/mcp", stored: "#/roadmap" });
  const app = await start({
    global: fragment.global,
    document: fragmentDocument,
    client: fakeClient(),
  });
  assert.equal(app.route().page, "settings");
  assert.equal(app.route().section, "mcp");
  assert.equal(fragmentDocument.getElementById("page").children[0].className, "settings-pane");
});

test("sidebar keeps projects, a single settings link, and a home link", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal({ fragment: "#/settings/general" });
  await start({ global: browser.global, document, client: fakeClient() });
  const sidebar = document.getElementById("sidebar");
  const links = document.getElementById("page-links");

  assert.deepEqual(links.children.map((link) => link.textContent), ["Settings"]);
  assert.doesNotMatch(visibleText(sidebar), /Roadmap|Chat/);
  assert.equal(sidebar.children[0], document.getElementById("home-link"));
  await document.getElementById("home-link").dispatch("click");
  assert.equal(browser.global.location.hash, "#/chat");
  assert.deepEqual(document.getElementById("page").children, [document.getElementById("chat-pane")]);
});

test("status rail keeps the roadmap beside settings and renders live agents", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal({ fragment: "#/settings/general" });
  const client = fakeClient();
  await start({ global: browser.global, document, client });
  const rail = document.getElementById("status-rail");
  const agents = document.getElementById("agents");

  assert.deepEqual(rail.children, [agents, document.getElementById("roadmap"), document.getElementById("spec")]);
  assert.equal(document.getElementById("page").children[0].className, "settings-pane");
  assert.equal(document.getElementById("page").children.length, 1);
  assert.equal(document.getElementById("roadmap").children.length, 3);
  assert.match(visibleText(rail), /18 · Desktop app/);
  assert.equal(visibleText(agents), "\nNothing is running.");

  await client.emit(eventFrame("RunStarted", { agent_name: "leader" }));
  await client.emit(eventFrame("ToolCallStarted", { tool_name: "read", tool_call_id: "one" }));
  await client.emit(eventFrame("SubagentSpawned", {
    subagent_agent_id: "agent-2", subagent_name: "worker",
  }));
  assert.deepEqual(agents.children.map((row) => row.className), ["agent-row", "agent-row"]);
  assert.deepEqual(agents.children.map((row) => row.textContent), [
    "leader · running · read", "worker · running",
  ]);
});

test("settings routes render general origins, model presence, and unknown fallback", async () => {
  const browser = fakeGlobal({ fragment: "#/settings/general" });
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    settings: { settings: {
      config: [
        { key: "z.list", value: ["one", "two"], scope: "project" },
        { key: "a.enabled", value: false, scope: "user" },
        { key: "private.marker", value: "config-only-marker", scope: "private" },
      ],
      providers: [
        { name: "openai", env_var: "OPENAI_API_KEY", key_present: true },
        { name: "gemini", env_var: "GEMINI_API_KEY", key_present: false },
      ],
    } },
  });
  const app = await start({ global: browser.global, document, client });
  const pane = document.getElementById("page").children[0];
  const content = find(pane, (value) => value.className === "settings-content");
  const rowCells = () => walk(content)
    .filter((value) => value.className === "settings-row")
    .map((row) => row.children.map((cell) => cell.textContent));

  assert.equal(client.calls.settings, 1);
  assert.equal(app.route().section, "general");
  assert.deepEqual(rowCells(), [
    ["a.enabled", "off", "user"],
    ["private.marker", "config-only-marker", "private"],
    ["z.list", "one, two", "project"],
  ]);

  const modelsLink = find(pane, (value) => value.tagName === "A" && value.textContent === "Models");
  await modelsLink.dispatch("click");
  assert.equal(browser.global.location.hash, "#/settings/models");
  assert.deepEqual(rowCells(), [
    ["gemini", "GEMINI_API_KEY", "absent"],
    ["openai", "OPENAI_API_KEY", "present"],
  ]);
  const modelText = visibleText(content);
  assert.ok(!modelText.includes("config-only-marker"));
  assert.doesNotMatch(modelText, /\bkey\s*[:=]\s*\S+/i);

  browser.global.location.hash = "#/settings/unknown";
  browser.dispatch("hashchange");
  assert.equal(app.route().section, "unknown");
  assert.equal(content.children[0].textContent, "General");
  assert.equal(rowCells().length, 3);
});

test("model settings save and remove a key without displaying its value", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal({ fragment: "#/settings/models" });
  const client = fakeClient(fixtureRoadmap(), {
    settings: { settings: { providers: [
      { name: "openai", env_var: "OPENAI_API_KEY", key_present: false },
    ] } },
  });
  await start({ global: browser.global, document, client });
  const content = find(document.getElementById("page"), (value) => value.className === "settings-content");
  const controls = find(content, (value) => value.className === "credential-controls");
  const input = find(controls, (value) => value.tagName === "INPUT");
  const status = find(content, (value) => value.className === "settings-row").children[2];
  const secret = "recognisable-app-key-fixture";

  assert.equal(input.type, "password");
  input.value = secret;
  await find(controls, (value) => value.tagName === "BUTTON" && value.textContent === "Save").dispatch("click");
  assert.deepEqual(client.calls.credentials, [{ name: "OPENAI_API_KEY", value: secret }]);
  assert.equal(input.value, "");
  assert.equal(status.textContent, "present");
  assert.ok(!visibleText(content).includes(secret));

  await find(controls, (value) => value.tagName === "BUTTON" && value.textContent === "Remove").dispatch("click");
  assert.deepEqual(client.calls.credentials.at(-1), { name: "OPENAI_API_KEY", value: "" });
  assert.equal(status.textContent, "absent");
});

test("credential client sends the value only in an authenticated POST body", async () => {
  let request;
  const client = createClient({
    port: 4312,
    token: "fixture-token",
    fetch: async (url, options) => {
      request = { url, options };
      return { status: 200, json: async () => ({ stored: true, name: "OPENAI_API_KEY" }) };
    },
  });
  const reply = await client.storeCredential("OPENAI_API_KEY", "recognisable-client-fixture");

  assert.deepEqual(reply, { stored: true, name: "OPENAI_API_KEY" });
  assert.equal(request.url, "http://127.0.0.1:4312/credentials");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.headers.Authorization, "Bearer fixture-token");
  assert.deepEqual(JSON.parse(request.options.body), {
    name: "OPENAI_API_KEY", value: "recognisable-client-fixture",
  });
  assert.ok(!request.url.includes("recognisable-client-fixture"));
});

test("extension settings routes show startup state, complete commands, and withheld reasons", async () => {
  const browser = fakeGlobal({ fragment: "#/settings/mcp" });
  const copied = [];
  browser.global.navigator = { clipboard: { writeText: async (path) => copied.push(path) } };
  const document = new FakeDocument();
  const command = `python -m ${"example.".repeat(600)}server`;
  const client = fakeClient(fixtureRoadmap(), {
    settings: { settings: {
      mcp_servers: [
        { name: "live-at-startup", command, started: true },
        { name: "offline-at-startup", command: "python -m offline", started: false },
      ],
      skills: [
        { name: "zeta", path: "/home/example/.symphonai/skills/zeta.md" },
        { name: "alpha", path: ".symphonai/skills/alpha.md" },
      ],
      plugins: [
        { name: "west", path: "/home/example/.symphonai/plugins/west" },
        { name: "east", path: ".symphonai/plugins/east" },
      ],
      agents: [
        { name: "reviewer", path: "" },
        { name: "builder", path: ".symphonai/agents/builder.toml" },
      ],
      withheld: [
        { scope: "project", directory: ".symphonai/skills", names: ["blocked"], reason: "repository not trusted" },
        { scope: "user", directory: "/users/example/plugins", names: [] },
      ],
    } },
  });
  await start({ global: browser.global, document, client });
  const pane = document.getElementById("page").children[0];
  const content = find(pane, (value) => value.className === "settings-content");
  const rowCells = () => walk(content)
    .filter((value) => value.className === "settings-row")
    .map((row) => row.children.map((cell) => cell.textContent));
  const open = async (name) => {
    const link = find(pane, (value) => value.tagName === "A" && value.textContent === name);
    await link.dispatch("click");
    assert.equal(browser.global.location.hash, `#/settings/${name.toLowerCase()}`);
  };

  assert.deepEqual(rowCells(), [
    ["live-at-startup", command, "started"],
    ["offline-at-startup", "python -m offline", "not started"],
  ]);
  assert.match(visibleText(content), /Started/);
  assert.doesNotMatch(visibleText(content), /running/i);

  await open("Skills");
  assert.deepEqual(rowCells(), [
    ["alpha", ".symphonai/skills/alpha.md"],
    ["zeta", "/home/example/.symphonai/skills/zeta.md"],
  ]);
  await find(content, (value) => value.tagName === "BUTTON" && value.textContent === "Copy").dispatch("click");
  assert.deepEqual(copied, [".symphonai/skills/alpha.md"]);

  await open("Plugins");
  assert.deepEqual(rowCells(), [
    ["east", ".symphonai/plugins/east"],
    ["west", "/home/example/.symphonai/plugins/west"],
  ]);
  delete browser.global.navigator;
  await find(content, (value) => value.tagName === "BUTTON" && value.textContent === "Copy").dispatch("click");
  assert.deepEqual(copied, [".symphonai/skills/alpha.md"]);

  await open("Agents");
  assert.deepEqual(rowCells(), [
    ["builder", ".symphonai/agents/builder.toml"],
    ["reviewer", ""],
  ]);

  await open("Inventory");
  assert.deepEqual(rowCells(), [
    ["project", ".symphonai/skills", "blocked", "repository not trusted"],
    ["user", "/users/example/plugins", "", "No reason recorded."],
  ]);
});

test("definition settings offer no file-write call", async () => {
  const source = await readFile(new URL("../src/client.js", import.meta.url), "utf8");
  const appSource = await readFile(new URL("../src/app.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /request\("POST",\s*"\/file"/);
  assert.doesNotMatch(appSource, /storeDefinition|writeDefinition|editDefinition/);
});

test("an empty extension inventory says nothing was withheld", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal({ fragment: "#/settings/inventory" });
  await start({ global: browser.global, document, client: fakeClient() });
  const content = find(document.getElementById("page"), (value) => value.className === "settings-content");

  assert.match(visibleText(content), /Nothing was withheld\./);
  assert.equal(walk(content).filter((value) => value.className === "settings-row").length, 0);
});

test("throwing local storage cannot stop startup or navigation", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal({ storageThrows: true });
  const app = await start({ global: browser.global, document, client: fakeClient() });

  assert.deepEqual(
    document.getElementById("page").children,
    [document.getElementById("chat-pane")],
  );
  await find(
    document.getElementById("page-links"),
    (value) => value.textContent === "Settings",
  ).dispatch("click");
  assert.equal(app.route().page, "settings");
});

test("frames accumulate while settings is open", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal();
  const client = fakeClient();
  await start({ global: browser.global, document, client });
  const links = document.getElementById("page-links");

  await find(links, (value) => value.textContent === "Settings").dispatch("click");
  await client.emit(eventFrame("AssistantTextDelta", { text: "while " }));
  await client.emit(eventFrame("AssistantTextDelta", { text: "away" }));
  await document.getElementById("home-link").dispatch("click");

  assert.equal(document.getElementById("chat").children[0].textContent, "while away");
});

test("folding the sidebar leaves the current page in place", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient() });
  const shell = document.getElementById("app-shell");
  const toggle = document.getElementById("sidebar-toggle");
  const page = document.getElementById("page");
  const currentPane = page.children[0];

  await toggle.dispatch("click");
  assert.equal(shell.className, "app-shell sidebar-folded");
  assert.equal(toggle.textContent, "Show sidebar");
  assert.equal(page.children[0], currentPane);

  await toggle.dispatch("click");
  assert.equal(shell.className, "app-shell");
  assert.equal(toggle.textContent, "Hide sidebar");
  assert.equal(page.children[0], currentPane);
});

test("status and sidebar folds are independent", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient() });
  const shell = document.getElementById("app-shell");
  const railToggle = document.getElementById("rail-toggle");
  const sidebarToggle = document.getElementById("sidebar-toggle");

  await railToggle.dispatch("click");
  assert.equal(shell.className, "app-shell rail-folded");
  assert.equal(railToggle.textContent, "Show status");
  await sidebarToggle.dispatch("click");
  assert.equal(shell.className, "app-shell rail-folded sidebar-folded");
  await railToggle.dispatch("click");
  assert.equal(shell.className, "app-shell sidebar-folded");
  assert.equal(railToggle.textContent, "Hide status");
});

test("items in folded and open phases open their specs and reports", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const app = await start({ global: {}, document, client });
  const calls = [];
  const open = app.specView.open;
  app.specView.open = (...arguments_) => {
    calls.push(arguments_);
    return open(...arguments_);
  };
  for (const [title, specText, reportText] of [
    ["Boundary", "base spec text", "base report text"],
    ["Open boundary", "open spec text", "open report text"],
  ]) {
    const button = find(
      document.getElementById("roadmap"),
      (value) => value.tagName === "BUTTON" && value.textContent === title,
    );
    await button.dispatch("click");
    const text = visibleText(document.getElementById("spec"));
    assert.match(text, new RegExp(specText));
    assert.match(text, new RegExp(reportText));
  }

  assert.deepEqual(calls.map(([item]) => item.title), ["Boundary", "Open boundary"]);
  assert.deepEqual(calls[0][1], { specPaths: [BASE, OPEN_BASE, FOLLOW_UP] });
});

test("an all-done roadmap leaves every phase folded", async () => {
  const document = new FakeDocument();
  await start({
    global: {},
    document,
    client: fakeClient(fixtureRoadmap({ allDone: true })),
  });

  assert.ok(document.getElementById("roadmap").children.every((phase) => !phase.open));
});

test("an additional fixture phase renders without a copied count", async () => {
  const roadmap = JSON.parse(fixtureRoadmap());
  roadmap.phases.push({
    id: "22",
    name: "Fixture phase",
    status: "planned",
    items: [],
  });
  const document = new FakeDocument();

  await start({
    global: {},
    document,
    client: fakeClient(JSON.stringify(roadmap)),
  });

  assert.equal(document.getElementById("roadmap").children.length, roadmap.phases.length);
});

test("submit dispatches once and assistant deltas render in order", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const app = await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  input.value = "hello";

  const submitting = document.getElementById("prompt-form").dispatch("submit");
  assert.equal(app.turn.state, DISPATCHING);
  assert.deepEqual(client.calls.prompt, ["hello"]);
  client.resolvePrompt({ accepted: true, run_id: "run-host" });
  await submitting;
  assert.equal(app.turn.state, RUNNING);
  assert.equal(app.turn.activeRunId, null);

  await client.emit(eventFrame("RunStarted", { run_id: "run-runtime" }));
  assert.equal(app.turn.activeRunId, "run-runtime");
  await client.emit(eventFrame("AssistantTextDelta", { text: "hello " }));
  await client.emit(eventFrame("AssistantTextDelta", { text: "world" }));
  const chat = document.getElementById("chat");
  assert.equal(chat.children.length, 1);
  assert.equal(chat.children[0].className, "assistant");
  assert.equal(chat.children[0].textContent, "hello world");
  await client.emit(eventFrame("RunFinished", { run_id: "run-runtime" }));
  assert.equal(app.turn.state, "idle");
});

test("a prompt conflict adopts the active runtime run and keeps the message", async () => {
  const document = new FakeDocument();
  const client = fakeClient(undefined, {
    health: { state: "active", run_id: "run-host", runtime_run_id: "run-active" },
  });
  client.prompt = async () => ({ accepted: false, conflict: true, run_id: "run-host" });
  const app = await start({ global: {}, document, client });
  document.getElementById("prompt").value = "first";

  await document.getElementById("prompt-form").dispatch("submit");

  assert.equal(app.turn.state, RUNNING);
  assert.equal(app.turn.activeRunId, "run-active");
  assert.deepEqual(app.turn.queue.map(({ text }) => text), ["first"]);
  assert.equal(document.getElementById("prompt-error").textContent, "A run is already in progress.");
});

test("an idle host after a prompt conflict leaves the message ready to resend", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.prompt = async (text) => {
    client.calls.prompt.push(text);
    return client.calls.prompt.length === 1
      ? { accepted: false, conflict: true, run_id: "run-host" }
      : { accepted: true, run_id: "run-next" };
  };
  const app = await start({ global: {}, document, client });
  document.getElementById("prompt").value = "first";

  await document.getElementById("prompt-form").dispatch("submit");

  assert.equal(app.turn.state, IDLE);
  assert.equal(app.turn.activeRunId, null);
  assert.deepEqual(app.turn.queue.map(({ text }) => text), ["first"]);
  assert.equal(document.getElementById("prompt-error").textContent, "The message was not sent. Send it again.");

  document.getElementById("prompt").value = "second";
  await document.getElementById("prompt-form").dispatch("submit");
  assert.deepEqual(client.calls.prompt, ["first", "first"]);
  assert.deepEqual(app.turn.queue.map(({ text }) => text), ["second"]);

  await client.emit(eventFrame("RunStarted", { run_id: "run-retry" }));
  await client.emit(eventFrame("RunFinished", { run_id: "run-retry" }));
  assert.deepEqual(client.calls.prompt, ["first", "first", "second"]);
});

test("a failed health request after a conflict leaves the message ready to resend", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.health = async () => { throw new Error("offline"); };
  client.prompt = async () => ({ accepted: false, conflict: true, run_id: "run-host" });
  const app = await start({ global: {}, document, client });
  document.getElementById("prompt").value = "first";

  await document.getElementById("prompt-form").dispatch("submit");

  assert.equal(app.turn.state, IDLE);
  assert.equal(app.turn.activeRunId, null);
  assert.deepEqual(app.turn.queue.map(({ text }) => text), ["first"]);
  assert.equal(document.getElementById("prompt-error").textContent, "The message was not sent. Send it again.");
});

test("an active run without a published id drains after RunStarted and RunFinished", async () => {
  const document = new FakeDocument();
  const client = fakeClient(undefined, {
    health: { state: "active", run_id: "run-host", runtime_run_id: null },
  });
  client.prompt = async (text) => {
    client.calls.prompt.push(text);
    return client.calls.prompt.length === 1
      ? { accepted: false, conflict: true, run_id: "run-host" }
      : { accepted: true, run_id: "run-next" };
  };
  const app = await start({ global: {}, document, client });
  document.getElementById("prompt").value = "first";

  await document.getElementById("prompt-form").dispatch("submit");
  assert.equal(app.turn.state, RUNNING);
  assert.equal(app.turn.activeRunId, null);
  assert.deepEqual(app.turn.queue.map(({ text }) => text), ["first"]);

  await client.emit(eventFrame("RunStarted", { run_id: "run-active" }));
  assert.equal(app.turn.activeRunId, "run-active");
  await client.emit(eventFrame("RunFinished", { run_id: "run-active" }));
  assert.deepEqual(client.calls.prompt, ["first", "first"]);
  assert.equal(app.turn.inFlight.text, "first");
  assert.deepEqual(app.turn.queue, []);
});

test("startup shows an active run notice and tolerates failed health", async () => {
  const notice = "A run started before this page was opened is still in progress.";
  const activeDocument = new FakeDocument();
  const activeClient = fakeClient(undefined, {
    health: { state: "active", run_id: "run-host", runtime_run_id: "run-active" },
  });
  await start({
    global: {},
    document: activeDocument,
    client: activeClient,
  });
  assert.equal(activeDocument.getElementById("run-notice").textContent, notice);
  await activeClient.emit(eventFrame("AssistantTextDelta", { text: "still working" }));
  assert.equal(activeDocument.getElementById("chat").children[0].textContent, "still working");
  assert.ok(visibleText(activeDocument.getElementById("chat-pane")).includes(notice));
  await activeClient.emit(eventFrame("RunFinished", { run_id: "run-other" }));
  assert.equal(activeDocument.getElementById("run-notice").textContent, notice);
  await activeClient.emit(eventFrame("RunFinished", { run_id: "run-active" }));
  assert.equal(activeDocument.getElementById("run-notice").textContent, "");

  const idleDocument = new FakeDocument();
  await start({ global: {}, document: idleDocument, client: fakeClient() });
  assert.equal(idleDocument.getElementById("run-notice").textContent, "");

  const failedDocument = new FakeDocument();
  const client = fakeClient();
  client.health = async () => { throw new Error("offline"); };
  await start({ global: {}, document: failedDocument, client });
  assert.deepEqual(failedDocument.getElementById("page").children, [
    failedDocument.getElementById("chat-pane"),
  ]);
  assert.equal(failedDocument.getElementById("run-notice").textContent, "");
});

test("a startup notice without a runtime id clears on the first terminal event", async () => {
  const document = new FakeDocument();
  const client = fakeClient(undefined, {
    health: { state: "active", run_id: "run-host", runtime_run_id: null },
  });
  await start({ global: {}, document, client });
  assert.equal(
    document.getElementById("run-notice").textContent,
    "A run started before this page was opened is still in progress.",
  );

  await client.emit(eventFrame("RunFinished", { run_id: "run-any" }));
  assert.equal(document.getElementById("run-notice").textContent, "");
});

test("two tool calls in one turn render as one activity", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });

  await client.emit(eventFrame("TurnStarted", { index: 1 }));
  await client.emit(toolStarted("call-1", "read_file", "one.txt"));
  await client.emit(toolFinished("call-1", "read_file"));
  await client.emit(toolStarted("call-2", "grep", "needle"));
  await client.emit(toolFinished("call-2", "grep"));

  const chat = document.getElementById("chat");
  assert.equal(chat.children.length, 1);
  assert.equal(chat.children[0].className, "activity");
  assert.match(chat.children[0].textContent, /Read `one\.txt` and searched for `needle`\./);
});

test("approvals stay separate and dropped frames render one model gap", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const app = await start({ global: {}, document, client });

  await client.emit({
    kind: "approval_requested",
    payload: {
      approval_id: "approval-1",
      operation: "write_file",
      target: "result.txt",
      details: "write the result",
    },
  });
  const approvals = document.getElementById("approvals");
  assert.match(visibleText(approvals), /write_file: result.txt/);
  const allow = find(
    approvals,
    (value) => value.tagName === "BUTTON" && value.textContent === "Allow",
  );
  const deny = find(
    approvals,
    (value) => value.tagName === "BUTTON" && value.textContent === "Deny",
  );
  assert.ok(allow);
  assert.ok(deny);
  await allow.dispatch("click");
  await allow.dispatch("click");
  assert.deepEqual(client.calls.approve, [
    { id: "approval-1", allowed: true, reason: "" },
  ]);

  await client.emit({ kind: "error", dropped: 2 });
  const chat = document.getElementById("chat");
  assert.equal(chat.children.length, 1);
  assert.equal(chat.children[0].className, "gap");
  assert.equal(chat.children[0].textContent, "2 events were dropped.");
  assert.deepEqual(app.transcript.model, [{ type: "gap", dropped: 2 }]);
});

test("assistant runs split around tool activity", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });

  await client.emit(eventFrame("AssistantTextDelta", { text: "before " }));
  await client.emit(eventFrame("AssistantTextDelta", { text: "tool" }));
  await client.emit(toolStarted("call-1", "read_file", "file.txt"));
  await client.emit(toolFinished("call-1", "read_file"));
  await client.emit(eventFrame("AssistantTextDelta", { text: "after " }));
  await client.emit(eventFrame("AssistantTextDelta", { text: "tool" }));

  const chat = document.getElementById("chat");
  assert.deepEqual(
    chat.children.map((child) => child.className),
    ["assistant", "activity", "assistant"],
  );
  assert.equal(chat.children[0].textContent, "before tool");
  assert.equal(chat.children[2].textContent, "after tool");
});

test("prompt and edit entries render from wire fields", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });

  await client.emit(eventFrame("PromptSubmitted", {
    text: "change the file",
    message_count: 1,
  }));
  await client.emit(toolStarted("edit-1", "edit_file", "notes.txt"));
  await client.emit(toolFinished("edit-1", "edit_file", {
    result_kind: "file_diff",
    result_path: "notes.txt",
    lines_added: 2,
    lines_removed: 1,
    diff: "@@ -1 +1,2 @@\n-old\n+new\n+line",
  }));

  const chat = document.getElementById("chat");
  assert.equal(chat.children[0].className, "prompt");
  assert.equal(chat.children[0].textContent, "change the file");
  const edit = chat.children[1];
  assert.equal(edit.tagName, "DETAILS");
  assert.equal(edit.className, "edit");
  assert.equal(edit.children[0].tagName, "SUMMARY");
  assert.match(edit.children[0].textContent, /notes\.txt \(\+2 −1\)/);
  assert.equal(edit.children[1].tagName, "PRE");
  assert.equal(edit.children[1].textContent, "@@ -1 +1,2 @@\n-old\n+new\n+line");
});

test("person and agent messages render as separate chat classes", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });

  await client.emit(eventFrame("PromptSubmitted", { text: "person", message_count: 1 }));
  await client.emit(eventFrame("AssistantTextDelta", { text: "agent" }));

  const chat = document.getElementById("chat");
  const prompts = chat.children.filter((child) => child.className === "prompt");
  const assistants = chat.children.filter((child) => child.className === "assistant");
  assert.equal(prompts.length, 1);
  assert.equal(assistants.length, 1);
  assert.notEqual(prompts[0], assistants[0]);
});

test("unknown events render and do not stop later frames", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });

  await client.emit(eventFrame("FutureEvent", { future_value: 7 }));
  await client.emit(eventFrame("AssistantTextDelta", { text: "still live" }));

  const chat = document.getElementById("chat");
  assert.equal(chat.children.length, 2);
  assert.equal(chat.children[0].className, "unknown-event");
  assert.match(chat.children[0].textContent, /FutureEvent/);
  assert.equal(chat.children[1].textContent, "still live");
});

test("each render replaces the chat instead of duplicating prior entries", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const app = await start({ global: {}, document, client });

  await client.emit(eventFrame("PromptSubmitted", { text: "one", message_count: 1 }));
  await client.emit(eventFrame("PromptSubmitted", { text: "two", message_count: 2 }));

  const chat = document.getElementById("chat");
  assert.equal(app.transcript.model.length, 2);
  assert.equal(chat.children.length, 2);
  assert.deepEqual(chat.children.map((child) => child.textContent), ["one", "two"]);
});

test("a failed prompt uses its error region and the next submit clears it", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  let shouldFail = true;
  client.prompt = async () => {
    if (shouldFail) {
      shouldFail = false;
      throw new Error("offline");
    }
    return { accepted: true, run_id: "run-2" };
  };
  await start({ global: {}, document, client });
  document.getElementById("prompt").value = "retry me";

  await document.getElementById("prompt-form").dispatch("submit");

  const promptError = document.getElementById("prompt-error");
  const chat = document.getElementById("chat");
  assert.equal(promptError.textContent, "Prompt failed.");
  assert.equal(chat.textContent, "");
  assert.equal(chat.children.length, 0);

  document.getElementById("prompt").value = "try again";
  const retrying = document.getElementById("prompt-form").dispatch("submit");
  assert.equal(promptError.textContent, "");
  assert.equal(chat.textContent, "");
  assert.equal(chat.children.length, 0);
  await retrying;
  assert.equal(promptError.textContent, "");
});

test("render stays DOM-only and start does not read window", async () => {
  const appTestSource = await readFile(new URL("./app.test.js", import.meta.url), "utf8");
  const renderSource = await readFile(
    new URL("../src/render.js", import.meta.url),
    "utf8",
  );
  const appSource = await readFile(new URL("../src/app.js", import.meta.url), "utf8");
  const cssSource = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const indexSource = await readFile(new URL("../index.html", import.meta.url), "utf8");

  function chatPaneChildIds(markup) {
    const content = markup.match(/<main id="chat-pane"[^>]*>([\s\S]*?)<\/main>/)?.[1];
    assert.ok(content);
    const ids = [];
    let depth = 0;
    for (const [, closing, attributes] of content.matchAll(/<(\/?)[a-z][\w-]*([^>]*)>/gi)) {
      if (closing) {
        depth -= 1;
      } else {
        if (depth === 0) {
          ids.push(attributes.match(/\bid="([^"]+)"/)?.[1] ?? null);
        }
        depth += 1;
      }
    }
    assert.equal(depth, 0);
    return ids;
  }
  const expectedChatChildren = ["run-notice", "chat", "approvals", "prompt-form"];
  assert.deepEqual(chatPaneChildIds(indexSource), expectedChatChildren);
  const withExtraChild = indexSource.replace(
    /(<main id="chat-pane"[^>]*>)/,
    '$1<div id="extra"></div>',
  );
  assert.throws(
    () => assert.deepEqual(chatPaneChildIds(withExtraChild), expectedChatChildren),
    { name: "AssertionError" },
  );

  const chatPaneRule = cssSource.match(/(?:^|\n)\.chat-pane\s*\{([^}]*)\}/)?.[1];
  assert.ok(chatPaneRule);
  assert.deepEqual(
    chatPaneRule.match(/grid-template-rows:\s*([^;]+);/)?.[1].trim().split(/\s+/),
    ["auto", "1fr", "auto", "auto"],
  );
  const sharedPaneRule = cssSource.match(
    /#status-rail,\s*\.chat-pane,\s*\.settings-pane\s*\{([^}]*)\}/,
  )?.[1];
  assert.match(sharedPaneRule, /(?:^|;)\s*overflow:\s*auto\s*;/);

  assert.ok(!/^\s*import\s/m.test(renderSource));
  assert.ok(!/\b(?:if|switch)\s*\(/.test(renderSource));
  assert.match(appSource, /export async function start\(\{ global, document, client \}\)/);
  assert.ok(!appSource.includes("chatLine"));
  assert.ok(!appSource.includes("roadmap.goal"));
  assert.ok(!appSource.includes('getElementById("turn-state")'));
  assert.ok(!start.toString().includes("window"));
  for (const className of [
    "prompt",
    "activity",
    "edit",
    "question",
    "compaction",
    "gap",
    "unknown-event",
  ]) {
    assert.match(cssSource, new RegExp(`\\.${className}\\b`));
  }
  function backgroundFor(className) {
    let background;
    for (const [, selectors, declarations] of cssSource.matchAll(/([^{}]+)\{([^{}]+)\}/g)) {
      if (selectors.split(",").map((selector) => selector.trim()).includes(`.${className}`)) {
        const match = declarations.match(/(?:^|;)\s*background:\s*(#[0-9a-f]{6})\s*;/i);
        if (match) {
          background = match[1].toLowerCase();
        }
      }
    }
    return background;
  }
  const neutral = backgroundFor("activity");
  const promptBackground = backgroundFor("prompt");
  const assistantBackground = backgroundFor("assistant");
  assert.ok(neutral);
  assert.ok(promptBackground);
  assert.ok(assistantBackground);
  for (const className of ["edit", "question", "compaction", "gap", "unknown-event", "approval"]) {
    assert.equal(backgroundFor(className), neutral);
  }
  assert.notEqual(promptBackground, neutral);
  assert.notEqual(assistantBackground, neutral);
  assert.notEqual(promptBackground, assistantBackground);
  assert.match(cssSource, /\.edit summary\s*{[^}]*cursor:\s*pointer/s);
  assert.match(cssSource, /\.edit pre\s*{[^}]*overflow-x:\s*auto/s);
  assert.ok(!/\.(?:tool|dropped)\b/.test(cssSource));
  assert.equal((indexSource.match(/class="chat-pane"/g) ?? []).length, 1);
  assert.equal((indexSource.match(/id="status-rail"/g) ?? []).length, 1);
  assert.equal((indexSource.match(/<!-- symphonai-handshake -->/g) ?? []).length, 1);
  assert.ok(!indexSource.includes("<h2>Chat</h2>"));
  assert.ok(!indexSource.includes('id="turn-state"'));
  assert.ok(!/<label\b/.test(indexSource));
  assert.match(indexSource, /<textarea[^>]*aria-label="Message"/);
  assert.match(
    indexSource,
    /<p id="prompt-error" class="prompt-error" aria-live="polite"><\/p>/,
  );
  assert.ok(!indexSource.includes("window.__symphonai"));
  assert.match(indexSource, /<nav id="sidebar"[^>]*>/);
  assert.ok(indexSource.indexOf('id="sidebar"') < indexSource.indexOf('id="page"'));
  assert.match(cssSource, /\.app-shell\.sidebar-folded\s*{[^}]*grid-template-columns:\s*0/s);
  assert.ok(!/assert\.equal\(phases\.length,\s*\d+\)/.test(appTestSource));
});
