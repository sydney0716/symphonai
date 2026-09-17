import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { start } from "../src/app.js";
import { DISPATCHING, RUNNING } from "../src/turn.js";

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

class FakeDocument {
  constructor() {
    const tags = {
      "app-shell": "div",
      sidebar: "nav",
      "sidebar-toggle": "button",
      "page-links": "div",
      page: "div",
      "roadmap-pane": "aside",
      "chat-pane": "main",
      "prompt-form": "form",
      prompt: "textarea",
    };
    this.elements = new Map(
      [
        "app-shell",
        "sidebar",
        "sidebar-toggle",
        "page-links",
        "page",
        "roadmap-pane",
        "chat-pane",
        "roadmap",
        "spec",
        "chat",
        "approvals",
        "prompt-form",
        "prompt",
        "prompt-error",
      ].map((id) => [id, new FakeElement(tags[id] ?? "div", id)]),
    );
    const get = (id) => this.elements.get(id);
    get("app-shell").className = "app-shell";
    get("roadmap-pane").className = "roadmap-pane";
    get("chat-pane").className = "chat-pane";
    get("sidebar").append(get("page-links"));
    get("roadmap-pane").append(get("roadmap"), get("spec"));
    get("prompt-form").append(get("prompt"), get("prompt-error"));
    get("chat-pane").append(get("chat"), get("approvals"), get("prompt-form"));
    get("page").append(get("roadmap-pane"), get("chat-pane"));
    get("app-shell").append(get("sidebar"), get("sidebar-toggle"), get("page"));
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

function fakeClient(
  roadmapText = fixtureRoadmap(),
  {
    project = { repo_root: "/work/current", name: "current" },
    sessions = [],
    settings = { settings: {} },
  } = {},
) {
  const calls = { approve: [], file: [], openSession: [], prompt: [], sessions: [], settings: 0 };
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

test("sidebar requests 200 sessions and discloses a full page", async () => {
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
  assert.match(visibleText(document.getElementById("sidebar")), /Showing the 200 most recent sessions\./);

  const shorterDocument = new FakeDocument();
  const shorterClient = fakeClient(fixtureRoadmap(), {
    sessions: [{ run_id: "one", repo_root: "/work/current", updated_at: "2026-01-01" }],
  });
  await start({ global: {}, document: shorterDocument, client: shorterClient });
  assert.doesNotMatch(visibleText(shorterDocument.getElementById("sidebar")), /Showing the 200 most recent/);
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
  await find(links, (value) => value.textContent === "Roadmap").dispatch("click");
  assert.deepEqual(page.children, [document.getElementById("roadmap-pane")]);
  assert.equal(app.route().page, "roadmap");
  assert.equal(browser.global.location.hash, "#/roadmap");
  assert.deepEqual(browser.writes.at(-1), ["symphonai.route", "#/roadmap"]);

  await find(links, (value) => value.textContent === "Chat").dispatch("click");
  assert.deepEqual(page.children, [chatPane]);
  assert.equal(document.getElementById("chat").children[0].textContent, "still here");
});

test("stored routes reopen, but a URL fragment wins", async () => {
  const storedDocument = new FakeDocument();
  const stored = fakeGlobal({ stored: "#/roadmap" });
  await start({ global: stored.global, document: storedDocument, client: fakeClient() });
  assert.deepEqual(
    storedDocument.getElementById("page").children,
    [storedDocument.getElementById("roadmap-pane")],
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
    (value) => value.textContent === "Roadmap",
  ).dispatch("click");
  assert.equal(app.route().page, "roadmap");
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
  await find(links, (value) => value.textContent === "Chat").dispatch("click");

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

test("the real roadmap opens phase 18 and no other phase", async () => {
  const roadmapText = await readFile(
    new URL("../../docs/roadmap.json", import.meta.url),
    "utf8",
  );
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient(roadmapText) });
  const phases = document.getElementById("roadmap").children;
  const open = phases.filter((phase) => phase.open);
  const phase20 = phases.find((phase) => phase.children[0].textContent.startsWith("20 ·"));

  assert.equal(phases.length, JSON.parse(roadmapText).phases.length);
  assert.equal(open.length, 1);
  assert.match(open[0].children[0].textContent, /^18 · App —/);
  assert.equal(phase20.open, false);
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
  client.resolvePrompt({ accepted: true, run_id: "run-1" });
  await submitting;
  assert.equal(app.turn.state, RUNNING);

  await client.emit(eventFrame("AssistantTextDelta", { text: "hello " }));
  await client.emit(eventFrame("AssistantTextDelta", { text: "world" }));
  const chat = document.getElementById("chat");
  assert.equal(chat.children.length, 1);
  assert.equal(chat.children[0].className, "assistant");
  assert.equal(chat.children[0].textContent, "hello world");
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
  assert.match(cssSource, /\.edit summary\s*{[^}]*cursor:\s*pointer/s);
  assert.match(cssSource, /\.edit pre\s*{[^}]*overflow-x:\s*auto/s);
  assert.ok(!/\.(?:tool|dropped)\b/.test(cssSource));
  assert.equal((indexSource.match(/class="(?:roadmap|chat)-pane"/g) ?? []).length, 2);
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
