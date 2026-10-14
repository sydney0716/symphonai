import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";

import { INIT_PROMPT, start } from "../src/app.js";
import { COMMANDS, matchCommands } from "../src/commands.js";
import { createClient } from "../src/client.js";
import { parseMarkdown } from "../src/markdown.js";
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
    this._textContent = "";
    this.type = "";
    this.value = "";
    this.listeners = new Map();
    this.attributes = new Map();
  }

  get href() {
    return this._href;
  }

  set href(value) {
    const attribute = String(value);
    this.attributes.set("href", attribute);
    this._href = new URL(attribute, "http://127.0.0.1/app/").href;
  }

  append(...children) {
    for (const child of children) child.parentNode = this;
    this.children.push(...children);
  }

  get textContent() {
    return this.className === "assistant"
      ? this._textContent + this.children.map((child) => child.textContent).join("")
      : this._textContent;
  }

  set textContent(value) {
    this._textContent = String(value);
    this.children = [];
  }

  replaceChildren(...children) {
    for (const child of this.children) child.parentNode = null;
    for (const child of children) child.parentNode = this;
    this.children = [...children];
  }

  before(...siblings) {
    const index = this.parentNode.children.indexOf(this);
    for (const sibling of siblings) sibling.parentNode = this.parentNode;
    this.parentNode.children.splice(index, 0, ...siblings);
  }

  focus() {
    let document = this;
    while (document && !(document instanceof FakeDocument)) {
      document = document.parentNode;
    }
    if (!document) return;
    if (document.activeElement) document.activeElement.focused = false;
    this.focused = true;
    document.activeElement = this;
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  async dispatch(type, fields = {}) {
    const event = { preventDefault() {}, ...fields };
    for (const listener of this.listeners.get(type) ?? []) {
      await listener(event);
    }
  }

  requestSubmit() {
    return this.dispatch("submit");
  }

  click() {
    this.clicked = true;
    return this.dispatch("click");
  }
}

export class FakeDocument {
  constructor() {
    this.documentElement = new FakeElement("html");
    const tags = {
      "app-shell": "div",
      sidebar: "nav",
      "home-link": "h1",
      "new-chat": "button",
      "sidebar-toggle": "button",
      "rail-toggle": "button",
      "page-links": "div",
      page: "div",
      "status-rail": "aside",
      "rail-tabs": "div",
      "rail-tools": "div",
      "chat-pane": "main",
      "prompt-form": "form",
      "jump-latest": "button",
      prompt: "textarea",
      "prompt-attachments": "div",
      files: "div",
    };
    this.elements = new Map(
      [
        "app-shell",
        "sidebar",
        "home-link",
        "new-chat",
        "sidebar-toggle",
        "rail-toggle",
        "page-links",
        "page",
        "status-rail",
        "rail-tabs",
        "agents",
        "chat-pane",
        "roadmap",
        "changes",
        "spec",
        "files",
        "rail-tools",
        "run-notice",
        "chat",
        "approvals",
        "prompt-form",
        "jump-latest",
        "prompt",
        "prompt-attachments",
        "prompt-error",
      ].map((id) => [id, new FakeElement(tags[id] ?? "div", id)]),
    );
    const get = (id) => this.elements.get(id);
    get("app-shell").className = "app-shell";
    get("chat-pane").className = "chat-pane";
    get("new-chat").className = "new-chat";
    get("jump-latest").type = "button";
    get("jump-latest").hidden = true;
    get("sidebar").append(get("home-link"), get("new-chat"), get("page-links"));
    get("status-rail").append(
      get("rail-tabs"), get("agents"), get("roadmap"), get("changes"),
      get("spec"), get("files"), get("rail-tools"),
    );
    get("prompt-form").append(get("prompt"), get("prompt-attachments"), get("prompt-error"));
    get("chat-pane").append(get("run-notice"), get("chat"), get("approvals"), get("jump-latest"), get("prompt-form"));
    get("page").append(get("chat-pane"));
    get("app-shell").append(get("sidebar"), get("sidebar-toggle"), get("page"), get("status-rail"), get("rail-toggle"));
    this.body = new FakeElement("body", "body");
    this.body.append(get("app-shell"));
    this.body.parentNode = this;
  }

  createElement(tagName) {
    return new FakeElement(tagName);
  }

  getElementById(id) {
    return this.elements.get(id) ?? null;
  }
}

function fakeGlobal({ fragment = "", stored = null, themeStored = null, railStored = null, statusStored = null, storageThrows = false } = {}) {
  const listeners = new Map();
  const writes = [];
  const global = {
    location: { hash: fragment },
    localStorage: {
      getItem(key) {
        if (storageThrows) {
          throw new Error("storage read failed");
        }
        if (key === "symphonai.route") return stored;
        if (key === "symphonai.theme") return themeStored;
        if (key === "symphonai.rail-tools") return railStored;
        if (key === "symphonai.status-line") return statusStored;
        assert.fail(`unexpected localStorage key: ${key}`);
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
    dispatch(type, event = {}) {
      return listeners.get(type)?.(event);
    },
  };
}

function walk(root) {
  return [root, ...root.children.flatMap(walk)];
}

function visibleText(root) {
  return walk(root).map((value) => value.textContent).join("\n");
}

function textInDocumentOrder(root) {
  return [root._textContent, ...root.children.map(textInDocumentOrder)].join("");
}

function find(root, predicate) {
  return walk(root).find(predicate);
}

function agentRowText(row) {
  const name = row.children.find((child) => child.className === "agent-name");
  const detail = row.children.find((child) => child.className === "agent-detail");
  return [name?.textContent, detail?.textContent].filter(Boolean).join(" · ");
}

function latestCommandEntry(document) {
  return [...document.getElementById("chat").children]
    .reverse()
    .find((entry) => entry.className === "command");
}

function latestCommandOutput(document) {
  return latestCommandEntry(document)?.children
    .filter((entry) => entry.className === "command-output")
    .map((entry) => entry.textContent) ?? [];
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
    memory = { project: { entries: [] }, global: { entries: [] } },
    memoryFailure = false,
    health = { protocol_version: 1, state: "idle", run_id: null, runtime_run_id: null },
    conversation = null,
    modelListing = { provider: "", state: "unknown", models: [], detail: "Model listing unavailable." },
    fileOverrides = {},
    roadmapFailure = null,
    specFilesReply = { paths: [] },
    specFilesFailure = false,
  } = {},
) {
  const calls = { agent: [], approve: [], compact: [], conversationStats: 0, credentials: [], file: [], files: [], history: [], forkSession: [], goal: [], goalState: [], memory: 0, updateMemory: [], deleteMemory: [], models: [], newSession: 0, openSession: [], prompt: [], promptAttachments: [], saveAgent: [], selectMode: [], selectProvider: [], sessions: [], settings: 0, specFiles: 0, stop: [], changes: 0, revertChanges: [], applyWorktree: [], discardWorktree: [] };
  let memoryData = memory;
  let currentRoadmapFailure = roadmapFailure;
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
  for (const [path, text] of Object.entries(fileOverrides)) files.set(path, { path, text });
  return {
    calls,
    emit(frame) {
      return eventCallback(frame);
    },
    setRoadmapFailure(failure) {
      currentRoadmapFailure = failure;
    },
    resolvePrompt,
    async file(path) {
      calls.file.push(path);
      if (path === "docs/roadmap.json" && currentRoadmapFailure !== null) {
        const error = new Error(currentRoadmapFailure.message ?? "roadmap unavailable");
        if (currentRoadmapFailure.status !== undefined) error.status = currentRoadmapFailure.status;
        throw error;
      }
      const reply = files.get(path);
      if (reply === undefined) {
        const error = new Error("missing");
        error.status = 404;
        throw error;
      }
      return reply;
    },
    async changes() {
      calls.changes += 1;
      return { files: [], turns: [], worktrees: [] };
    },
    async revertChanges(payload) {
      calls.revertChanges.push(payload);
      return {};
    },
    async applyWorktree(name) {
      calls.applyWorktree.push(name);
      return {};
    },
    async discardWorktree(name) {
      calls.discardWorktree.push(name);
      return {};
    },
    async files(query, limit = 20) {
      calls.files.push([query, limit]);
      return { files: [] };
    },
    async specFiles() {
      calls.specFiles += 1;
      if (specFilesFailure) throw new Error("spec listing unavailable");
      return specFilesReply;
    },
    async history(limit = 100) {
      calls.history.push(limit);
      return { prompts: [] };
    },
    events(callback) {
      eventCallback = callback;
      return { close() {}, done: Promise.resolve() };
    },
    prompt(text, attachments = []) {
      calls.prompt.push(text);
      calls.promptAttachments.push(attachments);
      return promptReply;
    },
    async models(provider, baseUrl) {
      calls.models.push([provider, baseUrl]);
      return { ...modelListing, provider };
    },
    async selectProvider(choice) {
      calls.selectProvider.push(choice);
      return { selected: true };
    },
    async selectMode(mode) {
      calls.selectMode.push(mode);
      return { mode };
    },
    async compact(instructions) {
      calls.compact.push(instructions);
      return { changed: false, before_tokens: 0, after_tokens: 0, dropped_messages: 0 };
    },
    async setGoal(objective, check) {
      calls.goal.push({ objective, check });
      return {
        accepted: true,
        run_id: "goal-run",
        goal: { objective, check, phase: "active", rounds: 1, max_rounds: 10, last_check: null, reason: "" },
      };
    },
    async goalState(action) {
      calls.goalState.push(action);
      return { goal: action === "clear" ? null : { objective: "goal", check: ["make", "test"], phase: action === "pause" ? "paused" : "active", rounds: 1, max_rounds: 10, last_check: null, reason: "" } };
    },
    async stop(runId) {
      calls.stop.push(runId);
      return { accepted: true };
    },
    async approve(id, allowed, reason, remember = false) {
      calls.approve.push({ id, allowed, reason, ...(remember ? { remember: true } : {}) });
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
    async memory() {
      calls.memory += 1;
      if (memoryFailure) throw new Error("memory unavailable");
      return memoryData;
    },
    async updateMemory(id, text) {
      calls.updateMemory.push({ id, text });
      for (const group of Object.values(memoryData)) {
        group.entries = group.entries.map((entry) => entry.id === id ? { ...entry, text } : entry);
      }
      return {
        entry: Object.values(memoryData).flatMap((group) => group.entries)
          .find((entry) => entry.id === id),
      };
    },
    async deleteMemory(id) {
      calls.deleteMemory.push({ id });
      for (const group of Object.values(memoryData)) {
        group.entries = group.entries.filter((entry) => entry.id !== id);
      }
      return { deleted: id };
    },
    async agent(name, scope) {
      calls.agent.push([name, scope]);
      return { name, scope, text: 'prompt = "loaded"\n[model]\nprovider = "openai"\n' };
    },
    async saveAgent(name, scope, text) {
      calls.saveAgent.push({ name, scope, text });
      return { written: true, message: "definition saved; it will take effect on the next run" };
    },
    async health() {
      return health;
    },
    async conversationStats() {
      calls.conversationStats += 1;
      return { conversation };
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
    async forkSession(runId, recordId, force = false) {
      calls.forkSession.push([runId, recordId, ...(force ? [true] : [])]);
      return { run_id: "fork-run" };
    },
    async newSession() {
      calls.newSession += 1;
      return { ended: true };
    },
  };
}

async function submitCommand(document, command) {
  document.getElementById("prompt").value = command;
  await document.getElementById("prompt-form").dispatch("submit");
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

test("command matching prioritizes names and also searches descriptions", () => {
  assert.deepEqual(matchCommands("/"), COMMANDS);
  assert.deepEqual(matchCommands("/mo"), [COMMANDS[0], COMMANDS[1]]);
  assert.deepEqual(matchCommands("/MOD"), [COMMANDS[0], COMMANDS[1]]);
  assert.deepEqual(matchCommands("/model"), [COMMANDS[1]]);
  assert.ok(matchCommands("/con").some(({ name }) => name === "context"));
  assert.deepEqual(matchCommands("/choose"), COMMANDS.slice(0, 3));
  assert.deepEqual(matchCommands("/current"), [COMMANDS[2]]);
  assert.deepEqual(matchCommands("/model x"), []);
  assert.deepEqual(matchCommands("hello"), []);
});

test("help lists commands and their aliases in table order", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient() });
  await submitCommand(document, "/help");

  const helpEntry = latestCommandEntry(document);
  assert.equal(helpEntry.children[0].className, "command-echo");
  assert.equal(helpEntry.children[0].textContent, "/help");
  assert.deepEqual(helpEntry.children.slice(1).map((child) => child.className), ["command-output"]);
  const helpOutput = latestCommandOutput(document);
  assert.equal(helpOutput.length, 1);
  assert.deepEqual(helpOutput[0].trim().split("\n"), [
    "/mode — Choose the permission mode",
    "/model [<provider> [<id>]] — Choose the provider and model",
    "/effort [<value>] — Choose the effort for the current model",
    "/help — Show the commands",
    "/new — Start a new chat (also /clear)",
    "/plan — Switch plan mode on or off",
    "/resume [<search>] — Reopen a past conversation in this project (also /continue)",
    "/cost — Show what this conversation has used",
    "/context — Show what fills the context window",
    "/compact [<instructions>] — Summarize the conversation so far to free context",
    "/goal [<objective> [-- <check command>] | pause | resume | clear] — Keep working toward a goal",
    "/init — Have the agent write .symphonai/INSTRUCTIONS.md",
  ]);
  await submitCommand(document, "/help extra");
  assert.match(visibleText(document.getElementById("chat")), /Usage: \/help/);
});

test("pasted images send with text, clear the chips and render in the transcript", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.prompt = async (text, attachments = []) => {
    client.calls.prompt.push(text);
    client.calls.promptAttachments.push(attachments);
    return { accepted: true, run_id: "image-run" };
  };
  const app = await start({ global: {}, document, client });
  const file = {
    name: "shot.png",
    type: "image/png",
    size: 9,
    async arrayBuffer() { return Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10, 0]).buffer; },
  };
  const input = document.getElementById("prompt");
  await input.dispatch("paste", {
    clipboardData: { items: [{ type: "image/png", getAsFile: () => file }] },
  });
  assert.equal(document.getElementById("prompt-attachments").children.length, 1);
  input.value = "what is this";
  await document.getElementById("prompt-form").dispatch("submit");
  assert.deepEqual(client.calls.prompt, ["what is this"]);
  assert.deepEqual(client.calls.promptAttachments, [[{
    data: "iVBORw0KGgoA",
    filename: "shot.png",
  }]]);
  assert.equal(document.getElementById("prompt-attachments").children.length, 0);

  await client.emit(eventFrame("PromptSubmitted", { text: "what is this", message_count: 1 }));
  const prompt = app.transcript.model.at(-1);
  assert.deepEqual(prompt.attachments, [{ kind: "image", filename: "shot.png" }]);
  const transcriptPrompt = find(document.getElementById("chat"), (node) => node.className === "prompt");
  assert.equal(transcriptPrompt.children[0].textContent, "Image · shot.png");
  await client.emit({
    kind: "event",
    payload: {
      type: "HistoryMessage",
      role: "user",
      text: "",
      tool_calls: [],
      turn_id: "turn-history",
      attachments: [{ kind: "document", media_type: "application/pdf", filename: "spec.pdf" }],
    },
  });
  const replayedPrompt = app.transcript.model.at(-1);
  assert.equal(replayedPrompt.text, "");
  assert.deepEqual(replayedPrompt.attachments, [{ kind: "document", filename: "spec.pdf" }]);
  const replayedAttachment = find(document.getElementById("chat"), (node) => node.textContent === "PDF · spec.pdf");
  assert.ok(replayedAttachment);
});

test("dropped PDFs allow an empty prompt and removals or invalid files do not send", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.prompt = async (text, attachments = []) => {
    client.calls.prompt.push(text);
    client.calls.promptAttachments.push(attachments);
    return { accepted: true, run_id: "pdf-run" };
  };
  await start({ global: {}, document, client });
  const form = document.getElementById("prompt-form");
  const pdf = {
    name: "spec.pdf", type: "application/pdf", size: 15,
    async arrayBuffer() { return Uint8Array.from([...new TextEncoder().encode("%PDF-1.7 sample")]).buffer; },
  };
  await form.dispatch("drop", { dataTransfer: { files: [pdf] } });
  await form.dispatch("submit");
  assert.deepEqual(client.calls.prompt, [""]);
  assert.deepEqual(client.calls.promptAttachments[0], [{
    data: "JVBERi0xLjcgc2FtcGxl",
    filename: "spec.pdf",
  }]);

  const remove = document.getElementById("prompt-attachments");
  await form.dispatch("drop", { dataTransfer: { files: [pdf] } });
  await remove.children[0].children[0].click();
  await form.dispatch("submit");
  assert.equal(client.calls.prompt.length, 1);

  await document.getElementById("prompt").dispatch("paste", {
    clipboardData: { items: [{ type: "application/pdf", getAsFile: () => pdf }] },
  });
  assert.equal(document.getElementById("prompt-attachments").children.length, 1);
  document.getElementById("prompt").value = "/help";
  await form.dispatch("submit");
  assert.equal(document.getElementById("prompt-attachments").children.length, 1);
  document.getElementById("prompt").value = "";
  await document.getElementById("prompt-attachments").children[0].children[0].click();

  for (const file of [
    { name: "large.png", type: "image/png", size: 6_000_000 },
    { name: "notes.txt", type: "text/plain", size: 12 },
  ]) {
    await form.dispatch("drop", { dataTransfer: { files: [file] } });
    assert.notEqual(document.getElementById("prompt-error").textContent, "");
    assert.equal(document.getElementById("prompt-attachments").children.length, 0);
  }
  await form.dispatch("submit");
  await form.dispatch("drop", {
    dataTransfer: { files: Array.from({ length: 11 }, (_, index) => ({ name: `${index}.png`, type: "image/png", size: 9 })) },
  });
  assert.equal(document.getElementById("prompt-attachments").children.length, 10);
  assert.match(document.getElementById("prompt-error").textContent, /up to 10/);
  for (let index = 0; index < 10; index += 1) {
    await document.getElementById("prompt-attachments").children[0].children[0].click();
  }
  assert.equal(client.calls.prompt.length, 1);
});

test("composer Escape closes a slash menu before it can stop a running turn", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const app = await start({ global: {}, document, client });
  document.getElementById("prompt").value = "hello";
  const submitting = document.getElementById("prompt-form").dispatch("submit");
  client.resolvePrompt({ accepted: true, run_id: "escape-menu-run" });
  await submitting;
  const input = document.getElementById("prompt");
  input.value = "/";
  await input.dispatch("input");
  await input.dispatch("keydown", { key: "Escape" });
  assert.equal(app.turn.state, RUNNING);
  assert.deepEqual(client.calls.stop, []);
  assert.equal(find(document.getElementById("chat-pane"), (node) => node.className === "command-menu").children.length, 0);
});

test("@ mentions search paths, insert on Enter and leave slash commands alone", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.files = async (query) => {
    client.calls.files.push([query, 20]);
    return { files: ["src/parser.py", "tests/test_parser.py"] };
  };
  await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  input.value = "look at @pars";
  await input.dispatch("input");
  assert.deepEqual(client.calls.files, [["pars", 20]]);
  const menu = find(document.getElementById("chat-pane"), (node) => node.className === "file-menu");
  assert.equal(menu.children[0].textContent, "src/parser.py");
  await input.dispatch("keydown", { key: "Enter" });
  assert.equal(input.value, "look at @src/parser.py ");

  input.value = "me@example.com";
  await input.dispatch("input");
  assert.equal(client.calls.files.length, 1);
  assert.equal(menu.children.length, 0);

  input.value = "/he";
  await input.dispatch("input");
  assert.ok(find(document.getElementById("chat-pane"), (node) => node.className === "command-menu").children.length > 0);
});

test("prompt arrows recall project history and drafts", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.history = async (limit) => {
    client.calls.history.push(limit);
    return { prompts: ["d", "c"] };
  };
  await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  await input.dispatch("keydown", { key: "ArrowUp" });
  assert.equal(input.value, "d");
  await input.dispatch("keydown", { key: "ArrowUp" });
  assert.equal(input.value, "c");
  await input.dispatch("keydown", { key: "ArrowDown" });
  assert.equal(input.value, "d");
  await input.dispatch("keydown", { key: "ArrowDown" });
  assert.equal(input.value, "");
  assert.deepEqual(client.calls.history, [100]);

  await input.dispatch("keydown", { key: "ArrowUp" });
  input.value = "edited draft";
  await input.dispatch("keydown", { key: "ArrowUp" });
  assert.equal(input.value, "edited draft");

  input.value = "/he";
  await input.dispatch("input");
  await input.dispatch("keydown", { key: "ArrowUp" });
  assert.deepEqual(client.calls.history, [100]);
  assert.equal(input.value, "/he");
});

test("a prompt sent from this page becomes the next recalled prompt", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.prompt = async (text) => {
    client.calls.prompt.push(text);
    return { accepted: true, run_id: "history-run" };
  };
  client.history = async (limit) => {
    client.calls.history.push(limit);
    return { prompts: ["older"] };
  };
  await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  input.value = "from this page";
  await document.getElementById("prompt-form").dispatch("submit");
  await input.dispatch("keydown", { key: "ArrowUp" });
  assert.equal(input.value, "from this page");
  assert.deepEqual(client.calls.prompt, ["from this page"]);
});

test("goal sets a check argv and queues a prompt during its first round", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  let active = false;
  client.health = async () => ({ protocol_version: 1, state: active ? "active" : "idle", run_id: null, runtime_run_id: active ? "runtime-goal" : null });
  client.setGoal = async (objective, check) => {
    client.calls.goal.push({ objective, check });
    active = true;
    return { accepted: true, run_id: "goal-run", goal: { objective, check, phase: "active", rounds: 1, max_rounds: 10, last_check: null, reason: "" } };
  };
  client.prompt = async (text) => {
    client.calls.prompt.push(text);
    return { accepted: false, conflict: true, run_id: "goal-run" };
  };
  const app = await start({ global: {}, document, client });
  await submitCommand(document, "/goal fix the parser -- python3 -m pytest tests/parser");
  assert.deepEqual(client.calls.goal, [{
    objective: "fix the parser",
    check: ["python3", "-m", "pytest", "tests/parser"],
  }]);
  assert.match(visibleText(document.getElementById("chat")), /Goal set\. Round 1 of 10 started\./);
  assert.doesNotMatch(find(document.body, (node) => node.className === "conversation-usage").textContent, /Goal active 1\/10/);
  await submitCommand(document, "please finish the parser");
  assert.equal(app.turn.state, RUNNING);
  assert.deepEqual(app.turn.queue.map(({ text }) => text), ["please finish the parser"]);
});

test("goal syntax errors answer usage without posting", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  for (const command of ["/goal fix it --", "/goal -- make test", "/goal pause now"]) {
    await submitCommand(document, command);
    assert.match(visibleText(document.getElementById("chat")), /Usage: \/goal \[<objective> \[-- <check command>\] \| pause \| resume \| clear\]/);
  }
  assert.deepEqual(client.calls.goal, []);
  assert.deepEqual(client.calls.goalState, []);
});

test("goal accepts a check-less objective and rejects an empty explicit check", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await submitCommand(document, "/goal fix the parser");
  assert.deepEqual(client.calls.goal, [{ objective: "fix the parser", check: [] }]);
  assert.match(visibleText(document.getElementById("chat")), /Goal set\. Round 1 of 10 started\./);
  await submitCommand(document, "/goal fix it --");
  assert.match(visibleText(document.getElementById("chat")), /Usage: \/goal/);
  assert.equal(client.calls.goal.length, 1);
});

test("goal set maps 409 to the active-run message", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.setGoal = async () => { throw Object.assign(new Error("conflict"), { status: 409 }); };
  await start({ global: {}, document, client });
  await submitCommand(document, "/goal fix it -- make test");
  assert.match(visibleText(document.getElementById("chat")), /A run is active\. Wait for it to finish, then set the goal\./);
});

test("goal state commands post their action and display host errors", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await submitCommand(document, "/goal pause");
  await submitCommand(document, "/goal resume");
  await submitCommand(document, "/goal clear");
  assert.deepEqual(client.calls.goalState, ["pause", "resume", "clear"]);
  assert.match(visibleText(document.getElementById("chat")), /Goal cleared\./);
  client.goalState = async (action) => {
    client.calls.goalState.push(action);
    throw new Error("no goal");
  };
  await submitCommand(document, "/goal resume");
  assert.match(visibleText(document.getElementById("chat")), /no goal/);
});

test("bare goal shows saved status and the last check result", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    conversation: {
      goal: {
        objective: "fix the parser",
        check: ["python3", "-m", "pytest"],
        phase: "paused",
        rounds: 2,
        max_rounds: 10,
        reason: "interrupted",
        last_check: { exit: 1, ok: false, output: "1 failed" },
      },
    },
  });
  await start({ global: {}, document, client });
  await submitCommand(document, "/goal");
  assert.match(visibleText(document.getElementById("chat")), /fix the parser/);
  assert.match(visibleText(document.getElementById("chat")), /Check: python3 -m pytest/);
  assert.match(visibleText(document.getElementById("chat")), /paused · round 2 of 10 · interrupted/);
  assert.match(visibleText(document.getElementById("chat")), /Last check: exit 1/);
  assert.match(visibleText(document.getElementById("chat")), /1 failed/);

  const emptyDocument = new FakeDocument();
  const emptyClient = fakeClient();
  emptyClient.conversationStats = async () => ({ conversation: null });
  await start({ global: {}, document: emptyDocument, client: emptyClient });
  await submitCommand(emptyDocument, "/goal");
  assert.match(visibleText(emptyDocument.getElementById("chat")), /No goal\. Set one with \/goal <objective> \[-- <check command>\]\./);
});

test("goal events render between rounds and update the status line", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    conversation: {
      provider: "fake",
      mode: "ask",
      goal: { objective: "fix parser", check: ["make", "test"], phase: "active", rounds: 1, max_rounds: 10, reason: "", last_check: null },
    },
  });
  const app = await start({ global: {}, document, client });
  const goalEvent = (change, phase, rounds, last_check = null, reason = "") => eventFrame("GoalChanged", {
    change, phase, rounds, max_rounds: 10, reason, last_check,
  });
  await app.onFrame(goalEvent("set", "active", 1));
  assert.doesNotMatch(find(document.body, (node) => node.className === "conversation-usage").textContent, /Goal active 1\/10/);
  await app.onFrame(eventFrame("RunFinished", { stopped_reason: "final_response" }));
  await app.onFrame(goalEvent("check", "active", 1, { exit: 1, ok: false, output: "1 failed" }));
  await app.onFrame(eventFrame("PromptSubmitted", { text: "round two feedback", message_count: 2 }));
  await app.onFrame(eventFrame("RunFinished", { stopped_reason: "final_response" }));
  await app.onFrame(goalEvent("check", "complete", 2, { exit: 0, ok: true, output: "" }));
  assert.doesNotMatch(find(document.body, (node) => node.className === "conversation-usage").textContent, /Goal complete 2\/10/);
  const entries = app.transcript.model.filter(({ type }) => type === "goal");
  assert.deepEqual(entries.map(({ text }) => text), [
    "Goal set: up to 10 rounds.",
    "Goal check failed (exit 1). Starting round 2 of 10.",
    "Goal check passed. Goal complete after 2 rounds.",
  ]);
  assert.equal(entries[1].output, "1 failed");
  const goalEntry = find(document.getElementById("chat"), (value) => value.className === "goal-check-output");
  assert.ok(goalEntry);
  assert.match(visibleText(goalEntry), /Check output/);
  assert.match(visibleText(goalEntry), /1 failed/);
  await app.onFrame(goalEvent("clear", "", 0));
  assert.doesNotMatch(find(document.body, (node) => node.className === "conversation-usage").textContent, /Goal /);
});

test("goal update and check-less round events render in the transcript", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    conversation: {
      mode: "ask",
      goal: { objective: "finish", check: [], phase: "active", rounds: 1, max_rounds: 2, reason: "", last_check: null },
    },
  });
  const app = await start({ global: {}, document, client });
  await app.onFrame(eventFrame("GoalChanged", {
    change: "update", phase: "blocked", rounds: 1, max_rounds: 2,
    reason: "missing access", last_check: null,
  }));
  assert.equal(app.transcript.model.at(-1).text, "Agent marked the goal blocked: missing access.");
  await app.onFrame(eventFrame("GoalChanged", {
    change: "round", phase: "active", rounds: 2, max_rounds: 2,
    reason: "", last_check: null,
  }));
  assert.equal(app.transcript.model.at(-1).text, "No check configured. Starting round 2 of 2.");
});

test("new and clear commands share the New chat action", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await client.emit(eventFrame("AssistantTextDelta", { text: "Old thread" }));
  assert.match(visibleText(document.getElementById("chat")), /Old thread/);

  await submitCommand(document, "/new");
  assert.equal(client.calls.newSession, 1);
  assert.equal(document.getElementById("chat").children.length, 0);

  await client.emit(eventFrame("AssistantTextDelta", { text: "Another old thread" }));
  await submitCommand(document, "/clear");
  assert.equal(client.calls.newSession, 2);
  assert.equal(document.getElementById("chat").children.length, 0);

  client.newSession = async () => { throw Object.assign(new Error("conflict"), { status: 409 }); };
  await submitCommand(document, "/new");
  assert.equal(document.getElementById("prompt-error").textContent, "Could not start a new chat.");
  await submitCommand(document, "/new extra");
  assert.match(visibleText(document.getElementById("chat")), /Usage: \/new/);
});

test("plan toggles back to the mode that was active before plan", async () => {
  const askDocument = new FakeDocument();
  let askConversation = { provider: "openai", model: "gpt-one", mode: "ask" };
  const askClient = fakeClient(fixtureRoadmap(), {
    settings: { settings: { mode: "ask", ceiling: { modes: ["ask", "allow", "plan"] } } },
    conversation: askConversation,
  });
  askClient.selectMode = async (mode) => {
    askClient.calls.selectMode.push(mode);
    askConversation = { ...askConversation, mode };
    return { mode };
  };
  askClient.conversationStats = async () => ({ conversation: askConversation });
  await start({ global: {}, document: askDocument, client: askClient });
  await submitCommand(askDocument, "/plan");
  await submitCommand(askDocument, "/plan");
  assert.deepEqual(askClient.calls.selectMode, ["plan", "ask"]);

  const allowDocument = new FakeDocument();
  let allowConversation = { provider: "openai", model: "gpt-one", mode: "allow" };
  const allowClient = fakeClient(fixtureRoadmap(), {
    settings: { settings: { mode: "allow", ceiling: { modes: ["ask", "allow", "plan"] } } },
    conversation: allowConversation,
  });
  allowClient.selectMode = async (mode) => {
    allowClient.calls.selectMode.push(mode);
    allowConversation = { ...allowConversation, mode };
    return { mode };
  };
  allowClient.conversationStats = async () => ({ conversation: allowConversation });
  await start({ global: {}, document: allowDocument, client: allowClient });
  await submitCommand(allowDocument, "/plan");
  await submitCommand(allowDocument, "/plan");
  assert.deepEqual(allowClient.calls.selectMode, ["plan", "allow"]);
});

test("plan mode rejects disallowed settings and accepts no arguments", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    settings: { settings: { mode: "ask", ceiling: { modes: ["ask", "allow"] } } },
    conversation: { provider: "openai", model: "gpt-one", mode: "ask" },
  });
  await start({ global: {}, document, client });
  await submitCommand(document, "/plan");
  assert.match(visibleText(document.getElementById("chat")), /Plan mode is not permitted here\./);
  assert.deepEqual(client.calls.selectMode, []);
  await submitCommand(document, "/plan extra");
  assert.match(visibleText(document.getElementById("chat")), /Usage: \/plan/);
});

test("the clear alias is suggested and Enter starts a new chat", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await client.emit(eventFrame("AssistantTextDelta", { text: "Old thread" }));
  const input = document.getElementById("prompt");
  input.focus();
  input.value = "/cl";
  await input.dispatch("input");
  const menu = find(document.getElementById("chat-pane"), (node) => node.className === "command-menu");
  assert.equal(menu.children.length, 1);
  assert.match(menu.children[0].textContent, /^\/new/);
  await input.dispatch("keydown", { key: "Enter" });
  assert.equal(client.calls.newSession, 1);
  assert.equal(document.getElementById("chat").children.length, 0);
});

test("resume lists this project's newest conversations and shares opening with the sidebar", async () => {
  const document = new FakeDocument();
  const sessions = [
    { run_id: "newest", title: "Newest", repo_root: "/work/current", updated_at: "2026-03-03" },
    { run_id: "middle", title: "Middle", repo_root: "/work/current", updated_at: "2026-03-02" },
    { run_id: "oldest", title: null, repo_root: "/work/current", updated_at: "2026-03-01" },
    { run_id: "elsewhere", title: "Elsewhere", repo_root: "/work/other", updated_at: "2026-03-04" },
  ];
  const client = fakeClient(fixtureRoadmap(), { sessions });
  const transcriptAtOpen = [];
  client.openSession = async (runId) => {
    client.calls.openSession.push(runId);
    transcriptAtOpen.push(visibleText(document.getElementById("chat")));
    return { run_id: runId };
  };
  await start({ global: {}, document, client });
  await client.emit(eventFrame("AssistantTextDelta", { text: "Before sidebar open" }));
  const newestLink = find(document.getElementById("sidebar"), (row) => (
    row.className === "session-link" && row.textContent === "Newest"
  ));
  await newestLink.dispatch("click");
  assert.equal(document.getElementById("chat").children.length, 0);

  await client.emit(eventFrame("AssistantTextDelta", { text: "Before resume open" }));
  await submitCommand(document, "/resume");
  const picker = find(document.getElementById("chat-pane"), (row) => row.className === "picker");
  assert.equal(find(picker, (row) => row.className === "picker-title").textContent, "Resume a conversation");
  const rows = find(picker, (row) => row.className === "picker-list").children;
  assert.deepEqual(rows.map((row) => row.children[0].textContent), ["Newest", "Middle", "oldest"]);
  assert.match(rows[0].className, /current/);
  assert.match(rows[1].className, /focused/);
  assert.deepEqual(client.calls.sessions, [200, undefined]);
  await picker.dispatch("keydown", { key: "Enter" });
  assert.deepEqual(client.calls.openSession, ["newest", "middle"]);
  assert.deepEqual(transcriptAtOpen, ["", ""]);
  assert.equal(document.getElementById("chat").children.length, 0);
});

test("resume search opens one match directly, reports no matches, and cancels its picker", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    sessions: [
      { run_id: "run-refactor", title: "Refactor branch", repo_root: "/work/current" },
      { run_id: "run-other", title: "Other work", repo_root: "/work/current" },
    ],
  });
  await start({ global: {}, document, client });
  await submitCommand(document, "/resume REFACTOR");
  assert.deepEqual(client.calls.openSession, ["run-refactor"]);
  assert.equal(find(document.getElementById("chat-pane"), (row) => row.className === "picker"), undefined);

  await submitCommand(document, "/resume missing");
  assert.match(visibleText(document.getElementById("chat")), /No conversation in this project matches "missing"\./);
  await submitCommand(document, "/resume");
  const picker = find(document.getElementById("chat-pane"), (row) => row.className === "picker");
  await picker.dispatch("keydown", { key: "Escape" });
  assert.equal(find(document.getElementById("chat-pane"), (row) => row.className === "picker"), undefined);
  assert.match(visibleText(document.getElementById("chat")), /Kept the current conversation\./);
});

test("resume reports empty and failed session listings", async () => {
  const emptyDocument = new FakeDocument();
  const emptyClient = fakeClient();
  await start({ global: {}, document: emptyDocument, client: emptyClient });
  await submitCommand(emptyDocument, "/resume");
  assert.match(visibleText(emptyDocument.getElementById("chat")), /No past conversations in this project\./);

  const failedDocument = new FakeDocument();
  const failedClient = fakeClient();
  failedClient.sessions = async (limit) => {
    failedClient.calls.sessions.push(limit);
    if (limit === undefined) throw new Error("Session listing unavailable.");
    return [];
  };
  await start({ global: {}, document: failedDocument, client: failedClient });
  await submitCommand(failedDocument, "/resume");
  assert.match(visibleText(failedDocument.getElementById("chat")), /Session listing unavailable\./);
});

test("cost prints agent and total usage with optional cache counts and cost", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  let conversation = null;
  client.conversationStats = async () => {
    client.calls.conversationStats += 1;
    return { conversation };
  };
  await start({ global: {}, document, client });
  conversation = {
    agents: [{
      name: "leader",
      input_tokens: 1070,
      output_tokens: 35,
      total_tokens: 1105,
      cache_read_tokens: 900,
      cache_write_tokens: 50,
      cost: { amount: "0.00141", currency: "USD" },
    }],
    usage: {
      input_tokens: 1073,
      output_tokens: 37,
      total_tokens: 1110,
      cache_read_tokens: 900,
      cache_write_tokens: 50,
      cost: { amount: "0.00141", currency: "USD" },
    },
  };
  await submitCommand(document, "/cost");
  assert.deepEqual(latestCommandOutput(document), [[
    "leader: 1105 tokens (input 1070, output 35, cache read 900, cache write 50) · 0.00141 USD",
    "Total: 1110 tokens (input 1073, output 37, cache read 900, cache write 50) · 0.00141 USD",
  ].join("\n")]);

  conversation = {
    agents: [{ name: "leader", input_tokens: 4, output_tokens: 1, total_tokens: 5 }],
    usage: { input_tokens: 4, output_tokens: 1, total_tokens: 5 },
  };
  await submitCommand(document, "/cost");
  assert.deepEqual(latestCommandOutput(document), [[
    "leader: 5 tokens (input 4, output 1)",
    "Total: 5 tokens (input 4, output 1)",
  ].join("\n")]);

  conversation = null;
  await submitCommand(document, "/cost");
  assert.deepEqual(latestCommandOutput(document), ["No usage recorded in this conversation yet."]);
  await submitCommand(document, "/cost extra");
  assert.deepEqual(latestCommandOutput(document), ["Usage: /cost"]);
});

test("resume and continue appear in command suggestions", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient() });
  const input = document.getElementById("prompt");
  input.value = "/res";
  await input.dispatch("input");
  let menu = find(document.getElementById("chat-pane"), (row) => row.className === "command-menu");
  assert.match(menu.children[0].textContent, /^\/resume/);
  input.value = "/cont";
  await input.dispatch("input");
  menu = find(document.getElementById("chat-pane"), (row) => row.className === "command-menu");
  assert.match(menu.children[0].textContent, /^\/resume/);
});

test("compact forwards instructions, answers the result, and refreshes conversation", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.compact = async (instructions) => {
    client.calls.compact.push(instructions);
    return { changed: true, before_tokens: 1200, after_tokens: 450, dropped_messages: 5 };
  };
  await start({ global: {}, document, client });
  await submitCommand(document, "/compact keep the API names");
  assert.deepEqual(client.calls.compact, ["keep the API names"]);
  assert.deepEqual(latestCommandOutput(document), [
    "Compacting…",
    "Compacted: 1200 → 450 tokens, 5 messages summarized.",
  ]);
  assert.equal(latestCommandEntry(document).children[0].textContent, "/compact keep the API names");
  assert.equal(client.calls.conversationStats, 2);

  client.compact = async (instructions) => {
    client.calls.compact.push(instructions);
    return { changed: false, before_tokens: 450, after_tokens: 450, dropped_messages: 0 };
  };
  await submitCommand(document, "/compact");
  assert.equal(client.calls.compact.at(-1), undefined);
  assert.deepEqual(latestCommandOutput(document), ["Compacting…", "Nothing to compact yet."]);
});

test("init sends the fixed prompt in ask mode and clears the composer", async () => {
  const expected = `Please analyze this repository and write .symphonai/INSTRUCTIONS.md, which SymphonAI loads into every conversation in this project.

Include:
1. The commands used most often: how to build, lint and run the tests, including how to run a single test.
2. The high-level architecture: the big picture that takes reading several files to understand.

Rules:
- If .symphonai/INSTRUCTIONS.md already exists, read it and improve it instead of starting over.
- Keep it short: only what an agent would get wrong without it. Do not list every file or directory, and do not add generic advice such as writing tests or handling errors.
- If README.md, CLAUDE.md, AGENTS.md, .cursorrules, .cursor/rules/ or .github/copilot-instructions.md exist, carry over the parts that matter.
- Do not invent sections or facts the repository does not support.`;
  assert.equal(INIT_PROMPT, expected);

  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    conversation: { provider: "openai", model: "gpt-one", mode: "ask" },
  });
  client.prompt = async (text) => {
    client.calls.prompt.push(text);
    return { accepted: true, run_id: "run-init" };
  };
  await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  input.value = "/init";
  await document.getElementById("prompt-form").dispatch("submit");
  assert.deepEqual(client.calls.prompt, [expected]);
  assert.equal(input.value, "");
});

test("init queues behind a running turn and sends when that turn finishes", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.prompt = async (text) => {
    client.calls.prompt.push(text);
    return { accepted: true, run_id: `run-${client.calls.prompt.length}` };
  };
  const app = await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  input.value = "first request";
  await document.getElementById("prompt-form").dispatch("submit");
  await client.emit(eventFrame("RunStarted", { run_id: "run-active" }));

  input.value = "/init";
  await document.getElementById("prompt-form").dispatch("submit");
  assert.deepEqual(client.calls.prompt, ["first request"]);
  assert.deepEqual(app.turn.queue.map(({ text }) => text), [INIT_PROMPT]);
  assert.equal(input.value, "");

  await client.emit(eventFrame("RunFinished", { run_id: "run-active" }));
  assert.deepEqual(client.calls.prompt, ["first request", INIT_PROMPT]);
  assert.deepEqual(app.turn.queue, []);
});

test("init is refused in plan mode and rejects arguments", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    conversation: { provider: "openai", model: "gpt-one", mode: "plan" },
  });
  await start({ global: {}, document, client });
  await submitCommand(document, "/init");
  assert.deepEqual(latestCommandOutput(document), ["Plan mode cannot write files. Switch with /plan, then run /init."]);
  assert.deepEqual(client.calls.prompt, []);
  await submitCommand(document, "/init now");
  assert.deepEqual(latestCommandOutput(document), ["Usage: /init"]);
  assert.deepEqual(client.calls.prompt, []);
});

test("slash suggestions render above the composer without taking focus", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient() });
  const input = document.getElementById("prompt");
  input.focus();
  input.value = "/";
  await input.dispatch("input");

  const menu = find(document.getElementById("chat-pane"), (node) => node.className === "command-menu");
  assert.deepEqual(menu.children.map((row) => row.textContent), [
    "/mode    Choose the permission mode",
    "/model  [<provider> [<id>]]  Choose the provider and model",
    "/effort  [<value>]  Choose the effort for the current model",
    "/help    Show the commands",
    "/new    Start a new chat",
    "/plan    Switch plan mode on or off",
    "/resume  [<search>]  Reopen a past conversation in this project",
    "/cost    Show what this conversation has used",
    "/context    Show what fills the context window",
    "/compact  [<instructions>]  Summarize the conversation so far to free context",
    "/goal  [<objective> [-- <check command>] | pause | resume | clear]  Keep working toward a goal",
    "/init    Have the agent write .symphonai/INSTRUCTIONS.md",
  ]);
  assert.equal(document.activeElement, input);
  assert.equal(menu.parentNode, document.getElementById("chat-pane"));

  input.value = "/con";
  await input.dispatch("input");
  assert.deepEqual(menu.children.map((row) => row.textContent), [
    "/resume  [<search>]  Reopen a past conversation in this project",
    "/context    Show what fills the context window",
  ]);

  input.value = "/mode ";
  await input.dispatch("input");
  assert.equal(menu.children.length, 0);
  assert.equal(document.activeElement, input);
});

test("slash menu navigation runs the highlighted command and Tab accepts without running", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    conversation: { provider: "openai", model: "gpt-one", mode: "ask" },
  });
  await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  input.focus();
  input.value = "/mo";
  await input.dispatch("input");
  await input.dispatch("keydown", { key: "ArrowDown" });
  assert.equal(document.activeElement, input);
  await input.dispatch("keydown", { key: "Enter" });

  const picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  assert.equal(find(picker, (node) => node.className === "picker-title").textContent, "Models · openai");
  assert.equal(input.value, "");
  assert.equal(latestCommandEntry(document).children[0].textContent, "/model");

  const tabDocument = new FakeDocument();
  const tabClient = fakeClient(fixtureRoadmap(), {
    conversation: { provider: "openai", model: "gpt-one", mode: "ask" },
  });
  await start({ global: {}, document: tabDocument, client: tabClient });
  const tabInput = tabDocument.getElementById("prompt");
  tabInput.focus();
  tabInput.value = "/ef";
  await tabInput.dispatch("input");
  await tabInput.dispatch("keydown", { key: "Tab" });

  assert.equal(tabInput.value, "/effort ");
  assert.equal(find(tabDocument.getElementById("chat-pane"), (node) => node.className === "command-menu").children.length, 0);
  assert.equal(tabDocument.activeElement, tabInput);
  assert.deepEqual(tabClient.calls.models, []);
  assert.deepEqual(tabClient.calls.selectProvider, []);
  assert.equal(visibleText(tabDocument.getElementById("chat")), "");
});

test("Escape dismisses slash suggestions without changing text, and row clicks run commands", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient() });
  const input = document.getElementById("prompt");
  input.focus();
  input.value = "/mo";
  await input.dispatch("input");
  await input.dispatch("keydown", { key: "Escape" });
  const menu = find(document.getElementById("chat-pane"), (node) => node.className === "command-menu");
  assert.equal(input.value, "/mo");
  assert.equal(menu.children.length, 0);
  assert.equal(document.activeElement, input);

  input.value = "/";
  await input.dispatch("input");
  const modeRow = menu.children[0];
  await modeRow.dispatch("mousedown");
  assert.equal(document.activeElement, input);
  await modeRow.dispatch("click");
  assert.equal(find(document.getElementById("chat-pane"), (node) => node.className === "picker").className, "picker");
  assert.equal(input.value, "");
});

test("Escape from a command picker keeps its answer under the command echo", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient() });
  await submitCommand(document, "/mode");
  const picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  await picker.dispatch("keydown", { key: "Escape" });

  const chat = document.getElementById("chat");
  assert.equal(chat.children.length, 1);
  assert.equal(chat.children[0].className, "command");
  assert.equal(latestCommandEntry(document).tagName, "DIV");
  assert.equal(latestCommandEntry(document).children[0].textContent, "/mode");
  assert.deepEqual(latestCommandOutput(document), ["Kept mode as ask."]);
  assert.equal(chat.children.some((entry) => entry.className === "assistant"), false);
});

test("Enter submits the composer through the form", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.prompt = async (text) => {
    client.calls.prompt.push(text);
    return { accepted: true, run_id: "run-keyboard" };
  };
  await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  input.focus();
  input.value = "hello";
  await input.dispatch("keydown", { key: "Enter" });

  assert.deepEqual(client.calls.prompt, ["hello"]);
  assert.equal(input.value, "");
});

test("Shift, Alt, and Ctrl+Enter insert a newline at the cursor", async () => {
  for (const modifiers of [{ shiftKey: true }, { altKey: true }, { ctrlKey: true }]) {
    const document = new FakeDocument();
    const client = fakeClient();
    await start({ global: {}, document, client });
    const input = document.getElementById("prompt");
    input.value = "ab";
    input.selectionStart = 1;
    input.selectionEnd = 1;
    let prevented = false;
    await input.dispatch("keydown", {
      key: "Enter",
      ...modifiers,
      preventDefault() { prevented = true; },
    });

    assert.equal(input.value, "a\nb");
    assert.equal(input.selectionStart, 2);
    assert.equal(input.selectionEnd, 2);
    assert.equal(prevented, true);
    assert.deepEqual(client.calls.prompt, []);
  }
});

test("composition Enter events leave the composer alone", async () => {
  for (const fields of [{ isComposing: true }, { keyCode: 229 }]) {
    const document = new FakeDocument();
    const client = fakeClient();
    await start({ global: {}, document, client });
    const input = document.getElementById("prompt");
    input.value = "hello";
    let prevented = false;
    await input.dispatch("keydown", {
      key: "Enter",
      ...fields,
      preventDefault() { prevented = true; },
    });

    assert.equal(input.value, "hello");
    assert.equal(prevented, false);
    assert.deepEqual(client.calls.prompt, []);
  }
});

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

test("turn edit cards collapse files, keep only latest undo, and retry conflicts", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.changes = async () => ({
    files: [],
    turns: [
      { key: "chk-first", paths: ["a.py", "b.py", "c.py", "d.py", "e.py"] },
      { key: "chk-second", paths: ["z.py"] },
    ],
    worktrees: [],
  });
  const attempted = [];
  client.revertChanges = async (payload) => {
    attempted.push(payload);
    if (!payload.force) {
      const error = new Error("changed outside");
      error.status = 409;
      error.paths = ["x.py"];
      throw error;
    }
    return {};
  };
  const app = await start({ global: {}, document, client });
  const addEdit = async (id, path, added, removed, run_id) => {
    await client.emit({ ...toolStarted(id, "edit_file", path), payload: {
      ...toolStarted(id, "edit_file", path).payload, run_id,
    } });
    await client.emit(toolFinished(id, "edit_file", {
      run_id,
      result_kind: "file_diff", result_path: path, lines_added: added,
      lines_removed: removed, diff: "-old\n+new",
    }));
  };
  await client.emit(eventFrame("RunStarted", { agent_id: "agent-1", run_id: "run-1" }));
  await client.emit(eventFrame("PromptSubmitted", {
    agent_id: "agent-1", run_id: "run-1", text: "first", checkpoint_key: "chk-first",
  }));
  for (const [index, path] of ["a.py", "b.py", "c.py", "d.py", "e.py"].entries()) {
    await addEdit(`first-${index}`, path, index + 1, 0, "run-1");
  }
  await client.emit(eventFrame("RunFinished", { agent_id: "agent-1", run_id: "run-1" }));
  await client.emit(eventFrame("RunStarted", { agent_id: "agent-1", run_id: "run-2" }));
  await client.emit(eventFrame("PromptSubmitted", {
    agent_id: "agent-1", run_id: "run-2", text: "second", checkpoint_key: "chk-second",
  }));
  await addEdit("second-edit", "z.py", 3, 1, "run-2");
  await client.emit(eventFrame("RunFinished", { agent_id: "agent-1", run_id: "run-2" }));

  const cards = () => walk(document.getElementById("chat")).filter((node) => node.className === "edit-card");
  assert.equal(cards().length, 2);
  assert.match(visibleText(cards()[1]), /Edited 1 file[\s\S]*\+3 −1/);
  assert.equal(find(cards()[0], (node) => node.tagName === "BUTTON" && node.textContent === "Undo"), undefined);
  const more = find(cards()[0], (node) => node.tagName === "BUTTON" && node.textContent === "Show 2 more files");
  assert.ok(more);
  await more.dispatch("click");
  assert.equal(cards()[0].children.find((node) => node.className === "edit-card-files").children.length, 5);

  const view = find(cards()[1], (node) => node.tagName === "BUTTON" && node.textContent === "View changes");
  await view.dispatch("click");
  assert.equal(document.getElementById("changes").hidden, false);

  const undo = find(cards()[1], (node) => node.tagName === "BUTTON" && node.textContent === "Undo");
  await undo.dispatch("click");
  assert.deepEqual(attempted, [{ key: "chk-second" }]);
  assert.match(visibleText(cards()[1]), /x\.py[\s\S]*Revert anyway/);
  await find(cards()[1], (node) => node.tagName === "BUTTON" && node.textContent === "Revert anyway").dispatch("click");
  assert.deepEqual(attempted, [{ key: "chk-second" }, { key: "chk-second", force: true }]);
  assert.match(visibleText(cards()[1]), /Undone/);
  assert.equal(find(cards()[1], (node) => node.tagName === "BUTTON" && node.textContent === "Undo"), undefined);
});

test("live edit cards only list paths covered by their checkpoint", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.changes = async () => ({
    files: [], turns: [{ key: "chk-live", paths: ["a.py"] }], worktrees: [],
  });
  await start({ global: {}, document, client });

  await client.emit(eventFrame("RunStarted", { agent_id: "leader", run_id: "root-run" }));
  await client.emit(eventFrame("PromptSubmitted", {
    agent_id: "leader", run_id: "root-run", text: "edit files", checkpoint_key: "chk-live",
  }));
  const addEdit = async (id, agent_id, run_id, path) => {
    await client.emit(eventFrame("ToolCallStarted", {
      agent_id, run_id, tool_name: "edit_file", tool_call_id: id, target: path,
    }));
    await client.emit(eventFrame("ToolCallFinished", {
      agent_id, run_id, tool_name: "edit_file", tool_call_id: id, ok: true,
      result_kind: "file_diff", result_path: path,
      lines_added: 1, lines_removed: 1, diff: "-old\n+new", truncated: false,
    }));
  };
  await addEdit("leader-edit", "leader", "root-run", "a.py");
  await addEdit("worktree-edit", "worker", "worktree-run", "b.py");
  await client.emit(eventFrame("RunFinished", { agent_id: "worker", run_id: "worktree-run" }));
  await client.emit(eventFrame("RunFinished", { agent_id: "leader", run_id: "root-run" }));

  const card = find(document.getElementById("chat"), (node) => node.className === "edit-card");
  assert.ok(card);
  const rows = card.children.find((node) => node.className === "edit-card-files").children;
  assert.equal(rows.length, 1);
  assert.match(visibleText(rows[0]), /a\.py/);
  assert.doesNotMatch(visibleText(card), /b\.py/);
  assert.ok(find(card, (node) => node.tagName === "BUTTON" && node.textContent === "Undo"));
});

test("live edit cards retain shared subagent files covered by their checkpoint", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.changes = async () => ({
    files: [], turns: [{ key: "chk-live", paths: ["a.py", "c.py"] }], worktrees: [],
  });
  await start({ global: {}, document, client });
  await client.emit(eventFrame("RunStarted", { agent_id: "leader", run_id: "root-run" }));
  await client.emit(eventFrame("PromptSubmitted", {
    agent_id: "leader", run_id: "root-run", text: "edit files", checkpoint_key: "chk-live",
  }));
  for (const [id, agent_id, run_id, path] of [
    ["leader-edit", "leader", "root-run", "a.py"],
    ["worker-edit", "worker", "worker-run", "c.py"],
  ]) {
    await client.emit(eventFrame("ToolCallStarted", {
      agent_id, run_id, tool_name: "edit_file", tool_call_id: id, target: path,
    }));
    await client.emit(eventFrame("ToolCallFinished", {
      agent_id, run_id, tool_name: "edit_file", tool_call_id: id, ok: true,
      result_kind: "file_diff", result_path: path,
      lines_added: 1, lines_removed: 1, diff: "-old\n+new", truncated: false,
    }));
  }
  await client.emit(eventFrame("RunFinished", { agent_id: "leader", run_id: "root-run" }));
  const card = find(document.getElementById("chat"), (node) => node.className === "edit-card");
  const rows = card.children.find((node) => node.className === "edit-card-files").children;
  assert.equal(rows.length, 2);
  assert.match(visibleText(rows[0]), /a\.py/);
  assert.match(visibleText(rows[1]), /c\.py/);
  assert.ok(find(card, (node) => node.tagName === "BUTTON" && node.textContent === "Undo"));
});

test("live edit cards have no Undo when changes has no matching checkpoint", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.changes = async () => ({
    files: [], turns: [{ key: "another-checkpoint", paths: ["a.py"] }], worktrees: [],
  });
  await start({ global: {}, document, client });
  await client.emit(eventFrame("RunStarted", { agent_id: "leader", run_id: "root-run" }));
  await client.emit(eventFrame("PromptSubmitted", {
    agent_id: "leader", run_id: "root-run", text: "edit a file", checkpoint_key: "chk-live",
  }));
  await client.emit(eventFrame("ToolCallStarted", {
    agent_id: "leader", run_id: "root-run", tool_name: "edit_file", tool_call_id: "edit", target: "a.py",
  }));
  await client.emit(eventFrame("ToolCallFinished", {
    agent_id: "leader", run_id: "root-run", tool_name: "edit_file", tool_call_id: "edit", ok: true,
    result_kind: "file_diff", result_path: "a.py", lines_added: 1, lines_removed: 1,
    diff: "-old\n+new", truncated: false,
  }));
  await client.emit(eventFrame("RunFinished", { agent_id: "leader", run_id: "root-run" }));
  const card = find(document.getElementById("chat"), (node) => node.className === "edit-card");
  assert.ok(card);
  assert.equal(find(card, (node) => node.tagName === "BUTTON" && node.textContent === "Undo"), undefined);
});

test("start renders the chat page and prepares the whole roadmap graph in the rail", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const app = await start({ global: {}, document, client });
  const roadmap = document.getElementById("roadmap");
  const page = document.getElementById("page");

  assert.deepEqual(page.children, [document.getElementById("chat-pane")]);
  assert.equal(app.route().page, "chat");
  assert.ok(find(roadmap, (node) => node.className === "roadmap-overview"));
  assert.equal(find(roadmap, (node) => node.className === "roadmap-graph-error"), undefined);
  assert.deepEqual(
    walk(roadmap).filter((node) => node.className.startsWith("roadmap-phase-box"))
      .map((node) => node.attributes.get("data-phase-id")),
    ["17", "18", "20"],
  );
  assert.equal(find(roadmap, (node) => node.className === "roadmap-open"), undefined);
  assert.ok(document.getElementById("chat"));
  assert.equal(document.getElementById("turn-state"), null);
  assert.equal(document.getElementById("prompt-error").textContent, "");
  assert.equal(app.client, client);
  assert.ok(app.transcript);
  assert.deepEqual(app.transcript.model, []);
  assert.equal(find(document.body, (value) => value.className === "conversation-usage").textContent, "");
});

test("missing roadmap does not stop the chat app from starting", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    roadmapFailure: { status: 404, message: "host file request failed with status 404" },
  });
  await start({ global: {}, document, client });

  assert.ok(document.getElementById("prompt-form"));
  assert.equal(
    find(document.getElementById("roadmap"), (node) => node.className === "roadmap-message").textContent,
    "No roadmap in this repository.",
  );
});

test("invalid roadmap does not stop startup and reports the parse error", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient("not json") });

  assert.ok(document.getElementById("prompt-form"));
  assert.equal(
    find(document.getElementById("roadmap"), (node) => node.className === "roadmap-message").textContent,
    "The roadmap couldn't be read: roadmap is not valid JSON",
  );
});

test("a missing roadmap during refresh updates the tool without replacing the page", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.specRuns = async () => [{ session_id: "spec-session", kind: "implement", state: "running", spec: "specs/18/example.md" }];
  const app = await start({ global: {}, document, client });
  const chatPane = document.getElementById("page").children[0];
  const composer = document.getElementById("prompt-form");

  client.setRoadmapFailure({ status: 404, message: "host file request failed with status 404" });
  await client.emit(eventFrame("RunFinished", { session_id: "spec-session" }));

  assert.equal(document.getElementById("page").children[0], chatPane);
  assert.equal(document.getElementById("prompt-form"), composer);
  assert.equal(
    find(document.getElementById("roadmap"), (node) => node.className === "roadmap-message").textContent,
    "No roadmap in this repository.",
  );
});

test("opening phase 18 loads each primary spec and follow-up once", async () => {
  const roadmap = {
    goal: "Shared specs",
    phases: [{
      id: "18", name: "Client", status: "done",
      items: [
        { title: "First", spec: BASE },
        { title: "Same spec", spec: BASE },
        { title: "Second", spec: OPEN_BASE },
        { title: "Same second spec", spec: OPEN_BASE },
      ],
    }],
  };
  const client = fakeClient(JSON.stringify(roadmap), {
    specFilesReply: { paths: [BASE, OPEN_BASE, FOLLOW_UP] },
    fileOverrides: { [FOLLOW_UP]: "# 18aF — Follow-up\nbody" },
  });
  const document = new FakeDocument();
  await start({ global: {}, document, client });
  assert.equal(client.calls.file.includes(BASE), false);
  const rail = document.getElementById("roadmap");
  await find(rail, (node) => node.className === "roadmap-phase-header").dispatch("click");
  await new Promise((resolve) => setTimeout(resolve, 0));
  for (const path of [BASE, OPEN_BASE, FOLLOW_UP]) {
    assert.equal(client.calls.file.filter((value) => value === path).length, 1, path);
  }
  assert.equal(walk(rail).filter((node) => node.className.startsWith("roadmap-task ")).length, 2);
});

test("phase waves load full spec titles and open tasks and follow-ups", async () => {
  const roadmap = JSON.parse(fixtureRoadmap());
  roadmap.phases[0].status = "in_progress";
  delete roadmap.phases[0].items[0].done;
  roadmap.phases[0].items[0].after = [];
  roadmap.phases[0].items.push({ title: "Second spec", spec: OPEN_BASE, after: ["18a"] });
  const client = fakeClient(JSON.stringify(roadmap), {
    fileOverrides: {
      [BASE]: "# 18a — A beautifully long title\nbody",
      [FOLLOW_UP]: "# 18aF — Fix the goal\nbody",
    },
  });
  const originalFile = client.file.bind(client);
  let releaseFiles;
  const filesReady = new Promise((resolve) => { releaseFiles = resolve; });
  client.file = async (path) => {
    if ([BASE, FOLLOW_UP].includes(path)) await filesReady;
    return originalFile(path);
  };
  const document = new FakeDocument();
  await start({ global: {}, document, client });
  const roadmapRail = document.getElementById("roadmap");
  const phase = find(roadmapRail, (node) => node.attributes.get("data-phase-id") === "17");
  const initialWaves = find(phase, (node) => node.className === "roadmap-waves");
  assert.equal(find(initialWaves, (node) => node.className.startsWith("roadmap-task ")).textContent, "18a  Boundary");
  releaseFiles();
  await new Promise((resolve) => setTimeout(resolve, 0));

  const refreshedPhase = find(roadmapRail, (node) => node.attributes.get("data-phase-id") === "17");
  const waves = find(refreshedPhase, (node) => node.className === "roadmap-waves");
  assert.ok(waves);
  assert.equal(walk(waves).filter((node) => node.className.startsWith("roadmap-task ")).length, 2);
  assert.equal(find(waves, (node) => node.className.startsWith("roadmap-task ")).textContent, "18a  A beautifully long title");
  assert.equal(find(waves, (node) => node.className.startsWith("roadmap-task ")).attributes.get("title"), "A beautifully long title");
  assert.equal(walk(waves).some((node) => node.className.startsWith("roadmap-edge")), false);
  assert.equal(find(waves, (node) => node.className === "roadmap-follow-up").textContent, "18aF  Fix the goal");

  await find(waves, (node) => node.className.startsWith("roadmap-task ")).dispatch("click");
  assert.ok(client.calls.file.includes(BASE));
  await find(waves, (node) => node.className === "roadmap-follow-up").dispatch("click");
  assert.ok(client.calls.file.includes(FOLLOW_UP));
  assert.ok(!find(roadmapRail, (node) => node.tagName === "SVG"));
});

test("phase 43 renders task waves with full titles and clickable follow-ups", async () => {
  const followUpPath = "specs/43/43iF-forks-and-merges-inside-a-phase.md";
  const phase = {
    id: "43",
    name: "Waves",
    status: "in_progress",
    progress: { done: 0, total: 10 },
    after: [],
    items: [
      { title: "Root", spec: "specs/43/43a-root.md", after: [] },
      { title: "Controls: plain and primary buttons", spec: "specs/43/43b-buttons.md", after: ["43a"] },
      { title: "The sidebar shows where you are", spec: "specs/43/43c-sidebar.md", after: ["43b"] },
      { title: "Make the primary button clearer", spec: "specs/43/43d-primary.md", after: ["43b"] },
      { title: "The transcript is flat", spec: "specs/43/43e-transcript.md", after: ["43a"] },
      { title: "Each column scrolls on its own", spec: "specs/43/43g-columns.md", after: ["43a"] },
      { title: "Roadmap arrows connect", spec: "specs/43/43i-arrows.md", after: ["43a"] },
      { title: "A task finishes", spec: "specs/43/43h-finish.md", after: ["43e"] },
      { title: "One task per line", spec: "specs/43/43j-wave.md", after: ["43i"] },
      { title: "Roadmap waves", spec: "specs/43/43f-waves.md", after: ["43c", "43d", "43e"] },
    ],
  };
  const roadmap = JSON.stringify({ goal: "Phase 43 graph", phases: [phase] });
  const client = fakeClient(roadmap, {
    specFilesReply: { paths: [followUpPath] },
    fileOverrides: { [followUpPath]: "# 43iF — Forks and merges inside a phase\nbody" },
  });
  const document = new FakeDocument();
  await start({ global: {}, document, client });
  await new Promise((resolve) => setTimeout(resolve, 0));

  const phaseNode = find(document.getElementById("roadmap"), (node) => node.attributes.get("data-phase-id") === "43");
  const waves = find(phaseNode, (node) => node.className === "roadmap-waves");
  const waveNodes = waves.children.filter((node) => node.className === "roadmap-wave");
  assert.equal(waveNodes.length, 4);
  const secondWaveTasks = walk(waveNodes[1]).filter((node) => node.className.startsWith("roadmap-task "));
  assert.deepEqual(secondWaveTasks.map((task) => task.textContent), [
    "43b  Controls: plain and primary buttons", "43e  The transcript is flat",
    "43g  Each column scrolls on its own", "43i  Roadmap arrows connect",
  ]);
  assert.deepEqual(secondWaveTasks.map((task) => task.attributes.get("title")), secondWaveTasks.map((task) => task.textContent.slice(5)));
  const followUp = find(waveNodes[1], (node) => node.className === "roadmap-follow-up");
  assert.equal(followUp.textContent, "43iF  Forks and merges inside a phase");
  assert.equal(walk(phaseNode).some((node) => node.className.startsWith("roadmap-edge")), false);
  await followUp.dispatch("click");
  assert.ok(client.calls.file.includes(followUpPath));
});

test("two-column phase tasks keep their titles in the buttons", async () => {
  const roadmap = JSON.stringify({
    goal: "Two columns",
    phases: [{
      id: "43", name: "Two tasks", status: "in_progress", after: [], progress: { done: 0, total: 2 },
      items: [
        { title: "Root", spec: "specs/43/43a-root.md", after: [] },
        { title: "Task B", spec: "specs/43/43b-task.md", after: ["43a"] },
        { title: "Task C", spec: "specs/43/43c-task.md", after: ["43a"] },
      ],
    }],
  });
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient(roadmap) });
  const phaseNode = find(document.getElementById("roadmap"), (node) => node.attributes.get("data-phase-id") === "43");
  const graph = find(phaseNode, (node) => node.className === "roadmap-waves");
  const buttons = walk(graph).filter((node) => node.className.startsWith("roadmap-task "));
  assert.deepEqual(buttons.slice(1).map((button) => button.textContent), ["43b  Task B", "43c  Task C"]);
});

test("roadmap phases use waves and the overview arrow stays a CSS shape", async () => {
  const roadmap = JSON.stringify({ goal: "Waves", phases: [
    { id: "43", name: "Current", status: "in_progress", after: [], items: [
      { title: "Root", spec: "specs/43/43a-root.md", after: [] },
      { title: "Parallel", spec: "specs/43/43b-parallel.md", after: ["43a"] },
    ] },
    { id: "44", name: "Next", status: "next", items: [] },
  ] });
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient(roadmap) });
  const rail = document.getElementById("roadmap");
  const phase = find(rail, (node) => node.attributes.get("data-phase-id") === "43");
  const waves = find(phase, (node) => node.className === "roadmap-waves");
  assert.ok(waves);
  assert.equal(walk(phase).some((node) => node.className.startsWith("roadmap-edge")), false);
  assert.equal(walk(rail).filter((node) => node.className === "roadmap-edge").length, 1);

  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const rule = (selector) => css.match(
    new RegExp(`(?:^|\\n)${selector}\\s*\\{([^}]*)\\}`),
  )?.[1] ?? "";
  const edge = rule("\\.roadmap-edge");
  const edgeStem = rule("\\.roadmap-edge::before");
  const edgeHead = rule("\\.roadmap-edge::after");
  assert.match(edgeStem, /left:\s*50%;/);
  assert.match(edgeStem, /border-left:\s*1px solid var\(--ui-text-secondary\);/);
  assert.match(edgeHead, /border-top:\s*0\.3rem solid var\(--ui-text-secondary\);/);
  assert.match(edge, /position:\s*relative;/);
  assert.doesNotMatch(edgeStem + edgeHead, /content:\s*["'][^"']+["']/);
  assert.doesNotMatch(css, /--edge-bend|roadmap-edge-through|roadmap-graph-compact|\.roadmap-box\b/);
});

test("independent phases stack in a parallel rail group and task clicks open the spec", async () => {
  const roadmap = {
    goal: "Parallel roadmap",
    phases: [
      { id: "37", name: "Independent A", status: "in_progress", after: [], items: [{ title: "Task A", spec: "specs/37/37a-task-a.md", after: [] }] },
      { id: "38", name: "Independent B", status: "in_progress", after: [], items: [{ title: "Task B", spec: "specs/38/38a-task-b.md", after: [] }] },
    ],
  };
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient(JSON.stringify(roadmap)) });

  const rail = document.getElementById("roadmap");
  const group = find(rail, (node) => node.className === "roadmap-phase-group");
  assert.deepEqual(group.children.map((node) => node.attributes.get("data-phase-id")), ["37", "38"]);
  const phase37 = find(rail, (node) => node.attributes.get("data-phase-id") === "37");
  const task = find(phase37, (node) => node.className.startsWith("roadmap-task "));
  await task.dispatch("click");
  assert.match(visibleText(document.getElementById("spec")), /37a-task-a\.md/);
});

test("spec file listing connects cross-phase follow-ups to the graph and viewer", async () => {
  const followUpPath = "specs/37/37eF-fix-the-goal.md";
  const roadmap = {
    goal: "Show follow-ups",
    phases: [
      { id: "37", name: "Sandbox", status: "in_progress", items: [{ title: "Sandbox", spec: "specs/37/37e-test.md", after: [] }] },
      { id: "39", name: "Workflow", status: "next", items: [{ title: "Run", spec: "specs/39/39b-run.md", after: ["37eF"] }] },
    ],
  };
  const client = fakeClient(JSON.stringify(roadmap), {
    specFilesReply: { paths: [followUpPath] },
    fileOverrides: {
      "specs/37/37e-test.md": "# 37e — Shell sandbox\nbody",
      [followUpPath]: "# 37eF — Fix the goal\nbody",
      "specs/39/39b-run.md": "# 39b — Run a spec\nbody",
    },
  });
  const document = new FakeDocument();
  await start({ global: {}, document, client });
  await new Promise((resolve) => setTimeout(resolve, 0));

  const rail = document.getElementById("roadmap");
  const priorPhase = find(rail, (node) => node.attributes.get("data-phase-id") === "37");
  const priorGraph = find(priorPhase, (node) => node.className === "roadmap-waves");
  assert.equal(find(priorGraph, (node) => node.className === "roadmap-follow-up").textContent, "37eF  Fix the goal");
  const currentPhase = find(rail, (node) => node.attributes.get("data-phase-id") === "39");
  assert.ok(find(currentPhase, (node) => node.className === "roadmap-waves"));
  assert.ok(!find(currentPhase, (node) => node.className === "roadmap-graph-error"));

  await find(priorGraph, (node) => node.className.startsWith("roadmap-task ")).dispatch("click");
  assert.match(visibleText(document.getElementById("spec")), /37eF-fix-the-goal/);
  assert.equal(client.calls.specFiles, 1);
});

test("spec file listing failure keeps the bound roadmap paths", async () => {
  const roadmap = {
    goal: "Keep the roadmap available",
    phases: [{ id: "18", name: "Client", status: "in_progress", items: [{ title: "Open boundary", spec: BASE, after: [] }] }],
  };
  const client = fakeClient(JSON.stringify(roadmap), { specFilesFailure: true });
  const document = new FakeDocument();
  await start({ global: {}, document, client });
  await new Promise((resolve) => setTimeout(resolve, 0));

  const phase = find(document.getElementById("roadmap"), (node) => node.attributes.get("data-phase-id") === "18");
  assert.ok(find(phase, (node) => node.className === "roadmap-waves"));
  assert.equal(find(phase, (node) => node.className.startsWith("roadmap-task ")).textContent, "18a  Open boundary");
});

test("conversation context and per-agent usage render and refresh after compaction", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const secret = "recognisable-conversation-secret-24f";
  const absolutePath = "/recognisable/absolute/conversation/path-24f";
  const snapshots = [
    {
      fixture_secret: secret,
      repo_root: absolutePath,
      provider: "openai", model: "gpt-test", mode: "ask",
      context: { used_tokens: 120, budget_tokens: 160, remaining_tokens: 40, by_source: { user: 70, assistant: 50 } },
      usage: { input_tokens: 30, output_tokens: 12, calls: 3, total_tokens: 42 },
      agents: [
        { agent_id: "leader-id", name: "leader", state: "running", input_tokens: 20, output_tokens: 10, calls: 2, total_tokens: 30, cost: { amount: "0.004", currency: "USD" } },
        { agent_id: "child-id", name: "researcher", state: "running", input_tokens: 10, output_tokens: 2, calls: 1, total_tokens: 12 },
      ],
    },
    {
      context: { used_tokens: 65, budget_tokens: 160, remaining_tokens: 95, by_source: { user: 35, assistant: 30 } },
      usage: { input_tokens: 35, output_tokens: 15, calls: 4, total_tokens: 50, cost: { amount: "0.007", currency: "USD" } },
      agents: [
        { agent_id: "leader-id", name: "leader", state: "running", input_tokens: 25, output_tokens: 13, calls: 3, total_tokens: 38, cost: { amount: "0.005", currency: "USD" } },
        { agent_id: "child-id", name: "researcher", state: "running", input_tokens: 10, output_tokens: 2, calls: 1, total_tokens: 12 },
      ],
    },
  ];
  client.conversationStats = async () => {
    client.calls.conversationStats += 1;
    return { conversation: snapshots[Math.min(client.calls.conversationStats - 1, 1)] };
  };
  await start({ global: {}, document, client });

  const usage = find(document.body, (value) => value.className === "conversation-usage");
  assert.equal(usage.textContent, "gpt-test");
  assert.doesNotMatch(usage.textContent, /USD|undefined/);
  assert.match(agentRowText(find(document.getElementById("agents"), (row) => row.className === "agent-row")), /leader · running · thinking · 30 tokens · USD 0\.004/);
  assert.match(agentRowText(find(document.getElementById("agents"), (row) => row.className === "agent-row" && agentRowText(row).startsWith("researcher"))), /researcher · running · thinking · 12 tokens/);
  assert.doesNotMatch(visibleText(document.body), new RegExp(`${secret}|${absolutePath}`));

  await client.emit(eventFrame("RunFinished", { agent_id: "leader-id" }));
  assert.equal(usage.textContent, "");
  assert.equal(client.calls.conversationStats, 2);
});

test("context command shows the measured source breakdown and model window", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.conversationStats = async () => ({ conversation: {
    provider: "anthropic",
    model: "claude-opus-5-5",
    context: {
      used_tokens: 12400,
      budget_tokens: 955000,
      remaining_tokens: 942600,
      window_tokens: 1000000,
      by_source: { system_prompt: 2100, user: 800, assistant: 3000, tool_result: 6500 },
    },
  } });
  await start({ global: {}, document, client });
  await submitCommand(document, "/context");
  assert.deepEqual(latestCommandOutput(document), [[
    "Context · anthropic / claude-opus-5-5",
    "Used 12,400 of 955,000 tokens before compaction (1.3%) · window 1,000,000",
    "    System prompt   2,100",
    "    Your messages   800",
    "    Replies         3,000",
    "    Tool results    6,500",
    "Remaining before compaction: 942,600",
    "Counts are estimates, about 4 characters per token.",
  ].join("\n")]);
  await submitCommand(document, "/context now");
  assert.deepEqual(latestCommandOutput(document), ["Usage: /context"]);
});

test("context command omits an unknown window and handles unmeasured context", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  let conversation = {
    provider: "anthropic",
    model: "claude-custom",
    context: {
      used_tokens: 13,
      budget_tokens: 100,
      remaining_tokens: 87,
      window_tokens: null,
      by_source: { user: 8, assistant: 5 },
    },
  };
  client.conversationStats = async () => ({ conversation });
  await start({ global: {}, document, client });
  await submitCommand(document, "/context");
  assert.deepEqual(latestCommandOutput(document), [[
    "Context · anthropic / claude-custom",
    "Used 13 of 100 tokens before compaction (13.0%)",
    "    Your messages   8",
    "    Replies         5",
    "Remaining before compaction: 87",
    "Counts are estimates, about 4 characters per token.",
  ].join("\n")]);

  conversation = { provider: "anthropic", model: "claude-custom" };
  await submitCommand(document, "/context");
  assert.deepEqual(latestCommandOutput(document), ["No context measured yet. Send a message first."]);
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
  const sessionLinks = walk(projects).filter((value) => value.className.includes("session-link"));
  assert.deepEqual(sessionLinks.map((value) => value.textContent), [
    "Current chat",
    "Other chat",
    "legacy",
  ]);
  assert.equal(document.getElementById("new-chat").className, "new-chat");

  await find(groups[0], (value) => value.tagName === "BUTTON").dispatch("click");
  assert.deepEqual(client.calls.openSession, ["current-old"]);
});

test("new chat clears the transcript and the next prompt adds a session link", async () => {
  const document = new FakeDocument();
  const initial = {
    run_id: "old-run",
    title: "Old chat",
    repo_root: "/work/current",
    updated_at: "2026-01-01",
  };
  const added = {
    run_id: "new-run",
    title: "New chat title",
    repo_root: "/work/current",
    updated_at: "2026-02-01",
  };
  const client = fakeClient(fixtureRoadmap(), { sessions: [initial] });
  client.prompt = async (text) => {
    client.calls.prompt.push(text);
    return { accepted: true, run_id: "host-run" };
  };
  client.sessions = async (limit) => {
    client.calls.sessions.push(limit);
    return client.calls.sessions.length === 1 ? [initial] : [added, initial];
  };
  await start({ global: {}, document, client });
  await client.emit({ kind: "event", payload: {
    type: "HistoryMessage", role: "assistant", text: "Old answer", tool_calls: [], turn_id: "turn-old",
  } });
  assert.equal(document.getElementById("chat").children.length, 1);

  const sessionLinks = () => walk(document.getElementById("sidebar"))
    .filter((value) => value.className === "session-link");
  assert.deepEqual(sessionLinks().map((value) => value.textContent), ["Old chat"]);
  await document.getElementById("new-chat").dispatch("click");
  assert.equal(client.calls.newSession, 1);
  assert.equal(document.getElementById("chat").children.length, 0);
  assert.deepEqual(sessionLinks().map((value) => value.textContent), ["Old chat"]);

  document.getElementById("prompt").value = "Start the next chat";
  await document.getElementById("prompt-form").dispatch("submit");
  assert.deepEqual(client.calls.prompt, ["Start the next chat"]);
  assert.deepEqual(client.calls.sessions, [200, 200]);
  assert.deepEqual(sessionLinks().map((value) => value.textContent), [
    "New chat title",
    "Old chat",
  ]);
});

test("composer box contains only the text, attachment chips, and error line", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient() });
  const form = document.getElementById("prompt-form");
  assert.deepEqual(form.children.map((child) => child.id || child.className), [
    "prompt", "prompt-attachments", "prompt-error",
  ]);
  assert.equal(walk(form).some((node) => node.tagName === "SELECT"), false);
  assert.equal(walk(form).some((node) => node.tagName === "BUTTON" || node.tagName === "INPUT"), false);
  const usage = find(document.getElementById("chat-pane"), (node) => node.className.includes("conversation-usage"));
  assert.equal(usage.parentNode, document.getElementById("chat-pane"));
  assert.equal(document.getElementById("chat-pane").children.indexOf(usage), document.getElementById("chat-pane").children.indexOf(form) + 1);
});

test("status line defaults to model and effort and always shows a non-ask mode", async () => {
  const document = new FakeDocument();
  let current = {
    provider: "openai", model: "gpt-5.6-terra", effort: "low", mode: "ask",
    context: { used_tokens: 4, budget_tokens: 10 }, usage: { total_tokens: 8 },
  };
  const client = fakeClient(fixtureRoadmap(), { conversation: current });
  client.conversationStats = async () => ({ conversation: current });
  client.selectMode = async (mode) => {
    client.calls.selectMode.push(mode);
    current = { ...current, mode };
    return { mode };
  };
  await start({ global: {}, document, client });
  const status = find(document.getElementById("chat-pane"), (node) => node.className.includes("conversation-usage"));
  assert.equal(status.textContent, "gpt-5.6-terra low");
  // The live mode picker path exercises the same status renderer.
  document.getElementById("prompt").value = "/mode";
  await document.getElementById("prompt-form").dispatch("submit");
  const picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  await picker.dispatch("keydown", { key: "ArrowDown" });
  await picker.dispatch("keydown", { key: "Enter" });
  assert.equal(status.textContent, "gpt-5.6-terra low · plan");
});

test("status line settings use a heading and provider and model toggle independently", async () => {
  const document = new FakeDocument();
  const conversation = {
    provider: "openai", model: "gpt-5.6-terra", effort: "low", mode: "ask",
  };
  const client = fakeClient(fixtureRoadmap(), { conversation });
  client.conversationStats = async () => ({ conversation });
  await start({ global: fakeGlobal({ fragment: "#/settings/general" }).global, document, client });

  const page = document.getElementById("page");
  const group = find(page, (node) => node.className === "status-line-settings");
  assert.equal(group.tagName, "SECTION");
  assert.equal(group.children[0].tagName, "H3");
  assert.equal(group.children[0].textContent, "Status line");
  assert.equal(walk(group).some((node) => node.tagName === "LEGEND"), false);

  const status = find(document.getElementById("chat-pane"), (node) => node.className.includes("conversation-usage"));
  const provider = find(group, (node) => node.attributes.get("aria-label") === "Provider");
  const model = find(group, (node) => node.attributes.get("aria-label") === "Model");
  const effort = find(group, (node) => node.attributes.get("aria-label") === "Effort");
  model.checked = false;
  await model.dispatch("change");
  effort.checked = false;
  await effort.dispatch("change");
  provider.checked = true;
  await provider.dispatch("change");
  assert.equal(status.textContent, "openai");
  model.checked = true;
  await model.dispatch("change");
  assert.equal(status.textContent, "openai / gpt-5.6-terra");
  effort.checked = true;
  await effort.dispatch("change");
  assert.equal(status.textContent, "openai / gpt-5.6-terra low");
});

test("status line settings persist checked fields and storage failures use defaults", async () => {
  const browser = fakeGlobal({ fragment: "#/settings/general" });
  const document = new FakeDocument();
  const conversation = {
    provider: "openai", model: "gpt-5.6-terra", effort: "low", mode: "ask",
    context: { used_tokens: 4, budget_tokens: 10 },
  };
  const client = fakeClient(fixtureRoadmap(), { conversation });
  await start({ global: browser.global, document, client });
  const context = find(document.getElementById("page"), (node) => node.attributes.get("aria-label") === "Context");
  assert.equal(context.checked, false);
  context.checked = true;
  await context.dispatch("change");
  const saved = browser.writes.find(([key]) => key === "symphonai.status-line")[1];
  assert.match(find(document.getElementById("chat-pane"), (node) => node.className.includes("conversation-usage")).textContent, /Context 4 \/ 10 tokens/);

  const reloaded = new FakeDocument();
  await start({
    global: fakeGlobal({ fragment: "#/settings/general", statusStored: saved }).global,
    document: reloaded,
    client: fakeClient(fixtureRoadmap(), { conversation }),
  });
  assert.equal(find(reloaded.getElementById("page"), (node) => node.attributes.get("aria-label") === "Context").checked, true);

  const blockedDocument = new FakeDocument();
  await start({
    global: fakeGlobal({ storageThrows: true }).global,
    document: blockedDocument,
    client: fakeClient(fixtureRoadmap(), { conversation }),
  });
  assert.equal(find(blockedDocument.getElementById("chat-pane"), (node) => node.className.includes("conversation-usage")).textContent, "gpt-5.6-terra low");
});

test("every CSS id selector targets an element in index.html", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const index = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const selectors = css
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/url\([^)]*\)/gi, "")
    .replace(/#(?:[\da-f]{3,8})(?![\w-])/gi, "");
  const ids = new Set([...selectors.matchAll(/#([\w-]+)/g)].map((match) => match[1]));
  const missing = [...ids].filter((id) => !new RegExp(`\\bid=[\"']${id}[\"']`).test(index));
  assert.deepEqual(missing, []);
});

test("every FakeDocument id exists in index.html", async () => {
  const index = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const indexIds = new Set([...index.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));
  const fakeIds = new FakeDocument().elements.keys();
  assert.deepEqual([...fakeIds].filter((id) => !indexIds.has(id)), []);
});

test("composer is one bordered box with the attachments, input, and bottom line in order", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const rule = (selector) => css.match(new RegExp(`${selector}\\s*\\{([^}]*)\\}`))?.[1] ?? "";
  const box = rule("#prompt-form");
  const focusBox = rule("#prompt-form:focus-within");
  const attachments = rule("#prompt-attachments");
  const textarea = rule("#prompt-form textarea");
  const textareaFocus = rule("#prompt-form textarea:focus");
  const usage = rule("\\.conversation-usage");
  const error = [...css.matchAll(/\.prompt-error\s*\{([^}]*)\}/g)].at(-1)?.[1] ?? "";

  assert.match(box, /border:\s*1px solid var\(--ui-control-border\);/);
  assert.match(box, /border-radius:\s*var\(--radius-composer\);/);
  assert.match(box, /background:\s*var\(--ui-panel\);/);
  assert.match(box, /padding:\s*var\(--space-2\) var\(--space-3\);/);
  assert.match(focusBox, /border-color:\s*var\(--ui-accent\);/);
  assert.match(focusBox, /outline:\s*1px solid var\(--ui-accent\);/);

  const formGrid = rule("\\.prompt-form");
  assert.match(formGrid, /grid-template-columns:\s*minmax\(0, 1fr\);/);
  assert.match(usage, /text-align:\s*left;/);
  assert.match(rule("#prompt-attachments:empty"), /display:\s*none;/);

  assert.match(textarea, /border:\s*0;/);
  assert.match(textarea, /background:\s*transparent;/);
  assert.match(textarea, /padding:\s*0;/);
  assert.match(textarea, /resize:\s*none;/);
  assert.match(textarea, /min-height:\s*calc\(1\.55em \* 3\);/);
  assert.match(textareaFocus, /outline:\s*none;/);
  assert.match(usage, /white-space:\s*nowrap;/);
  assert.match(usage, /text-overflow:\s*ellipsis;/);
});

test("mode picker marks the current mode and selects with keyboard without prompting", async () => {
  const document = new FakeDocument();
  let current = { provider: "openai", model: "gpt-old", mode: "ask" };
  const client = fakeClient(fixtureRoadmap(), {
    settings: { settings: { mode: "ask", ceiling: { modes: ["ask", "plan"] } } },
    conversation: current,
  });
  client.selectMode = async (mode) => { client.calls.selectMode.push(mode); current = { ...current, mode }; return { mode }; };
  client.conversationStats = async () => ({ conversation: current });
  await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  input.value = "/mode";
  await document.getElementById("prompt-form").dispatch("submit");
  const picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  assert.equal(picker.focused, true);
  let rows = find(picker, (node) => node.className === "picker-list").children;
  assert.deepEqual(rows.map((row) => row.children[0].textContent), ["ask", "plan"]);
  assert.match(rows[0].className, /focused current/);
  assert.equal(input.value, "");
  await picker.dispatch("keydown", { key: "ArrowDown" });
  rows = find(picker, (node) => node.className === "picker-list").children;
  assert.match(rows[1].className, /focused/);
  await picker.dispatch("keydown", { key: "Enter" });
  assert.deepEqual(client.calls.selectMode, ["plan"]);
  assert.deepEqual(client.calls.prompt, []);
  assert.equal(input.focused, true);
  const status = find(document.getElementById("chat-pane"), (node) => node.className.includes("conversation-usage"));
  assert.equal(status.textContent, "gpt-old · plan");
  assert.match(status.className, /plan/);
  assert.match(visibleText(document.getElementById("chat")), /Mode set to plan/);
});

test("model picker selects models and preserves only supported effort", async () => {
  const document = new FakeDocument();
  let current = { provider: "openai", model: "gpt-current", effort: "high", mode: "ask" };
  const client = fakeClient(fixtureRoadmap(), {
    conversation: current,
    modelListing: { provider: "openai", state: "available", models: [
      { id: "gpt-current", efforts: ["low", "high"] },
      { id: "gpt-next", efforts: ["medium", "high"] },
      { id: "gpt-basic", efforts: [] },
      { id: "gpt-unknown", efforts: null },
    ], detail: "" },
  });
  client.selectProvider = async (choice) => {
    client.calls.selectProvider.push(choice);
    current = { ...current, provider: choice.name, model: choice.model };
    if (Object.hasOwn(choice, "effort")) current.effort = choice.effort;
    else delete current.effort;
  };
  client.conversationStats = async () => ({ conversation: current });
  await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  input.value = "/model";
  await document.getElementById("prompt-form").dispatch("submit");
  let picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  assert.equal(picker.focused, true);
  let rows = find(picker, (node) => node.className === "picker-list").children;
  assert.deepEqual(rows.map((row) => row.children[0].textContent), ["gpt-current", "gpt-next", "gpt-basic", "gpt-unknown"]);
  assert.match(rows[0].className, /focused current/);
  assert.equal(rows[0].children[1].textContent, "Current");
  assert.equal(rows[0].children.length, 2);
  assert.deepEqual(client.calls.models, [["openai", undefined]]);
  await picker.dispatch("keydown", { key: "ArrowRight" });
  rows = find(picker, (node) => node.className === "picker-list").children;
  assert.match(rows[0].className, /focused current/);
  await picker.dispatch("keydown", { key: "ArrowDown" });
  await picker.dispatch("keydown", { key: "Enter" });
  assert.deepEqual(client.calls.selectProvider, [{ name: "openai", model: "gpt-next", effort: "high" }]);
  assert.match(visibleText(document.getElementById("chat")), /Model set to openai \/ gpt-next · effort high\./);
  assert.equal(input.value, "");
  input.value = "/model";
  await document.getElementById("prompt-form").dispatch("submit");
  picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  await find(picker, (node) => node.className.split(" ").includes("picker-row") && node.children[0]?.textContent === "gpt-basic").dispatch("click");
  assert.deepEqual(client.calls.selectProvider.at(-1), { name: "openai", model: "gpt-basic" });
  assert.deepEqual(client.calls.prompt, []);
  assert.match(visibleText(document.getElementById("chat")), /Model set to openai \/ gpt-basic · effort reset to the model's default\./);
  input.value = "/model";
  await document.getElementById("prompt-form").dispatch("submit");
  picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  await find(picker, (node) => node.className.split(" ").includes("picker-row") && node.children[0]?.textContent === "gpt-unknown").dispatch("click");
  assert.deepEqual(client.calls.selectProvider.at(-1), { name: "openai", model: "gpt-unknown" });
  input.value = "/model openai gpt-unknown experimental";
  await document.getElementById("prompt-form").dispatch("submit");
  assert.match(visibleText(document.getElementById("chat")), /Usage: \/model \[<provider> \[<id>\]\]/);
  assert.deepEqual(client.calls.selectProvider.at(-1), { name: "openai", model: "gpt-unknown" });
});

test("escape cancels through the keymap and keeps the current model", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), { conversation: { provider: "openai", model: "gpt-one", mode: "ask" } });
  await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  input.value = "/model openai";
  await document.getElementById("prompt-form").dispatch("submit");
  const picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  assert.equal(latestCommandEntry(document).children[0].textContent, "/model openai");
  await picker.dispatch("keydown", { key: "Escape" });
  assert.equal(find(document.getElementById("chat-pane"), (node) => node.className === "picker"), undefined);
  assert.match(visibleText(document.getElementById("chat")), /Kept model as openai \/ gpt-one/);
  assert.deepEqual(client.calls.selectProvider, []);
  assert.deepEqual(client.calls.prompt, []);
  assert.equal(input.value, "");
  assert.equal(input.focused, true);
});

test("bare model command in a conversation can switch to another keyed provider", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    settings: { settings: { providers: [
      { name: "openai", key_present: true },
      { name: "anthropic", key_present: true },
    ] } },
    conversation: { provider: "openai", model: "gpt-old", mode: "ask" },
  });
  client.models = async (provider) => ({
    provider,
    state: "available",
    models: [{ id: `${provider}-model`, efforts: [] }],
    detail: "",
  });
  await start({ global: {}, document, client });
  document.getElementById("prompt").value = "/model";
  await document.getElementById("prompt-form").dispatch("submit");
  let picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  let rows = find(picker, (node) => node.className === "picker-list").children;
  assert.deepEqual(rows.map((row) => row.children[0].textContent), ["openai", "anthropic"]);
  assert.match(rows[0].className, /current/);
  await find(picker, (node) => node.className.split(" ").includes("picker-row") && node.children[0]?.textContent === "anthropic").dispatch("click");
  picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  rows = find(picker, (node) => node.className === "picker-list").children;
  assert.deepEqual(rows.map((row) => row.children[0].textContent), ["anthropic-model"]);
  assert.deepEqual(client.calls.selectProvider, []);
  assert.deepEqual(client.calls.prompt, []);
  await picker.dispatch("keydown", { key: "Escape" });
  picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  rows = find(picker, (node) => node.className === "picker-list").children;
  assert.deepEqual(rows.map((row) => row.children[0].textContent), ["openai", "anthropic"]);
  assert.match(rows[1].className, /focused/);
  assert.doesNotMatch(visibleText(document.getElementById("chat")), /Kept model as/);
  await picker.dispatch("keydown", { key: "Escape" });
  assert.match(visibleText(document.getElementById("chat")), /Kept model as openai \/ gpt-old/);
});

test("a model hidden from discovery can still be selected by ID", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    conversation: { provider: "openai", model: "gpt-current", mode: "ask" },
    modelListing: { provider: "openai", state: "available", models: [{ id: "gpt-visible", efforts: [] }], detail: "" },
  });
  client.models = async (provider) => ({
    provider,
    state: "available",
    models: [{ id: "gpt-visible", efforts: [] }],
    detail: "",
  });
  await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  input.value = "/model openai gpt-hidden-by-preference";
  await document.getElementById("prompt-form").dispatch("submit");
  assert.deepEqual(client.calls.selectProvider, [{ name: "openai", model: "gpt-hidden-by-preference" }]);
  assert.deepEqual(client.calls.prompt, []);
});

test("model picker opens before a conversation and choosing among providers opens their models", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    settings: { settings: { providers: [
      { name: "openai", key_present: true },
      { name: "anthropic", key_present: true },
    ] } },
    conversation: null,
    modelListing: { provider: "openai", state: "available", models: [{ id: "gpt-alpha", efforts: [] }], detail: "" },
  });
  client.models = async (provider) => ({
    provider,
    state: "available",
    models: [{ id: `${provider}-model`, efforts: [] }],
    detail: "",
  });
  await start({ global: {}, document, client });
  document.getElementById("prompt").value = "/model";
  await document.getElementById("prompt-form").dispatch("submit");
  let picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  let rows = find(picker, (node) => node.className === "picker-list").children;
  assert.deepEqual(rows.map((row) => row.children[0].textContent), ["openai", "anthropic"]);
  assert.equal(latestCommandEntry(document).children[0].textContent, "/model");
  assert.deepEqual(latestCommandOutput(document), []);
  await find(picker, (node) => node.className.split(" ").includes("picker-row") && node.children[0]?.textContent === "anthropic").dispatch("click");
  picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  rows = find(picker, (node) => node.className === "picker-list").children;
  assert.deepEqual(rows.map((row) => row.children[0].textContent), ["anthropic-model"]);
  assert.deepEqual(client.calls.prompt, []);
});

test("unknown model listings show the reason and keep the typed fallback", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    conversation: { provider: "openai", model: "custom-old", mode: "ask" },
  });
  client.models = async (provider) => ({ provider, state: "unknown", models: [], detail: "Catalogue unavailable for this endpoint." });
  await start({ global: {}, document, client });
  document.getElementById("prompt").value = "/model";
  await document.getElementById("prompt-form").dispatch("submit");
  let picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  assert.equal(find(picker, (node) => node.className === "picker-message").textContent, "Catalogue unavailable for this endpoint.");
  assert.equal(find(picker, (node) => node.className === "picker-list").children.length, 0);
  document.getElementById("prompt").value = "/model custom-provider custom-id";
  await document.getElementById("prompt-form").dispatch("submit");
  assert.deepEqual(client.calls.selectProvider, [{ name: "custom-provider", model: "custom-id" }]);
  assert.deepEqual(client.calls.prompt, []);
  assert.match(visibleText(document.getElementById("chat")), /Model set to custom-provider \/ custom-id/);
});

test("effort picker starts at the current level and applies the keyboard choice", async () => {
  const document = new FakeDocument();
  let conversation = { provider: "anthropic", model: "claude-sonnet-5", effort: "high", mode: "ask" };
  const client = fakeClient(fixtureRoadmap(), { conversation });
  client.models = async (provider) => ({
    provider,
    state: "available",
    models: [{ id: "claude-sonnet-5", efforts: ["low", "medium", "high", "xhigh", "max"] }],
    detail: "",
  });
  client.selectProvider = async (choice) => {
    client.calls.selectProvider.push(choice);
    conversation = { ...conversation, effort: choice.effort };
  };
  client.conversationStats = async () => ({ conversation });
  await start({ global: {}, document, client });

  await submitCommand(document, "/effort");
  assert.equal(document.getElementById("prompt").value, "");
  const picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  const rows = find(picker, (node) => node.className === "picker-list").children;
  assert.equal(find(picker, (node) => node.className === "picker-title").textContent, "Effort · anthropic / claude-sonnet-5");
  assert.deepEqual(rows.map((row) => row.children[0].textContent), ["low", "medium", "high", "xhigh", "max"]);
  assert.match(rows[2].className, /focused current/);
  await picker.dispatch("keydown", { key: "ArrowDown" });
  await picker.dispatch("keydown", { key: "Enter" });

  assert.deepEqual(client.calls.selectProvider, [{ name: "anthropic", model: "claude-sonnet-5", effort: "xhigh" }]);
  assert.equal(document.getElementById("prompt").value, "");
  assert.equal(find(document.getElementById("chat-pane"), (node) => node.className === "picker"), undefined);
  assert.match(visibleText(document.getElementById("chat")), /Effort set to xhigh\./);
  assert.match(find(document.getElementById("chat-pane"), (node) => node.className.includes("conversation-usage")).textContent, /xhigh/);
});

test("typed effort validates listed values and applies valid values without a picker", async () => {
  const document = new FakeDocument();
  let conversation = { provider: "anthropic", model: "claude-sonnet-5", effort: "high", mode: "ask" };
  const client = fakeClient(fixtureRoadmap(), { conversation });
  client.models = async (provider) => ({
    provider,
    state: "available",
    models: [{ id: "claude-sonnet-5", efforts: ["low", "medium", "high", "xhigh", "max"] }],
    detail: "",
  });
  client.selectProvider = async (choice) => {
    client.calls.selectProvider.push(choice);
    conversation = { ...conversation, effort: choice.effort };
  };
  client.conversationStats = async () => ({ conversation });
  await start({ global: {}, document, client });

  await submitCommand(document, "/effort low");
  assert.equal(document.getElementById("prompt").value, "");
  await submitCommand(document, "/effort turbo");

  assert.deepEqual(client.calls.selectProvider, [{ name: "anthropic", model: "claude-sonnet-5", effort: "low" }]);
  assert.equal(find(document.getElementById("chat-pane"), (node) => node.className === "picker"), undefined);
  assert.match(visibleText(document.getElementById("chat")), /claude-sonnet-5 accepts: low, medium, high, xhigh, max\./);
  assert.match(find(document.getElementById("chat-pane"), (node) => node.className.includes("conversation-usage")).textContent, /low/);
});

test("effort handles models without effort support and unknown effort listings", async () => {
  const document = new FakeDocument();
  const conversation = { provider: "openai", model: "gpt-basic", mode: "ask" };
  const client = fakeClient(fixtureRoadmap(), { conversation });
  client.models = async (provider) => ({
    provider,
    state: "available",
    models: [{ id: "gpt-basic", efforts: [] }],
    detail: "",
  });
  await start({ global: {}, document, client });

  await submitCommand(document, "/effort");
  await submitCommand(document, "/effort high");
  assert.deepEqual(client.calls.selectProvider, []);
  assert.equal((visibleText(document.getElementById("chat")).match(/gpt-basic does not take an effort\./g) ?? []).length, 2);

  const unknownDocument = new FakeDocument();
  let unknownConversation = { provider: "openai", model: "gpt-unknown", mode: "ask" };
  const unknownClient = fakeClient(fixtureRoadmap(), { conversation: unknownConversation });
  unknownClient.models = async (provider) => ({
    provider,
    state: "available",
    models: [{ id: "gpt-unknown", efforts: null }],
    detail: "",
  });
  unknownClient.selectProvider = async (choice) => {
    unknownClient.calls.selectProvider.push(choice);
    unknownConversation = { ...unknownConversation, effort: choice.effort };
  };
  unknownClient.conversationStats = async () => ({ conversation: unknownConversation });
  await start({ global: {}, document: unknownDocument, client: unknownClient });

  await submitCommand(unknownDocument, "/effort");
  assert.match(visibleText(unknownDocument.getElementById("chat")), /Effort levels for gpt-unknown are unknown; type \/effort <value>\./);
  assert.equal(find(unknownDocument.getElementById("chat-pane"), (node) => node.className === "picker"), undefined);
  await submitCommand(unknownDocument, "/effort vendor-x");
  assert.deepEqual(unknownClient.calls.selectProvider, [{ name: "openai", model: "gpt-unknown", effort: "vendor-x" }]);
});

test("effort accepts typed values for models absent from discovery", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    conversation: { provider: "custom", model: "custom-model", mode: "ask" },
  });
  client.models = async (provider) => ({ provider, state: "available", models: [], detail: "" });
  await start({ global: {}, document, client });

  await submitCommand(document, "/effort experimental");

  assert.deepEqual(client.calls.selectProvider, [{ name: "custom", model: "custom-model", effort: "experimental" }]);
});

test("effort reports missing models, usage errors, and unavailable listings", async () => {
  const emptyDocument = new FakeDocument();
  const emptyClient = fakeClient(fixtureRoadmap());
  await start({ global: {}, document: emptyDocument, client: emptyClient });

  await submitCommand(emptyDocument, "/effort");
  assert.match(visibleText(emptyDocument.getElementById("chat")), /Choose a model first with \/model\./);
  assert.deepEqual(emptyClient.calls.models, []);

  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    conversation: { provider: "openai", model: "gpt-one", mode: "ask" },
  });
  client.models = async (provider) => ({ provider, state: "unknown", models: [], detail: "Catalogue unavailable." });
  await start({ global: {}, document, client });
  await submitCommand(document, "/effort high");
  assert.match(visibleText(document.getElementById("chat")), /Catalogue unavailable\./);
  client.models = async () => { throw new Error("Model service offline."); };
  await submitCommand(document, "/effort high");
  assert.match(visibleText(document.getElementById("chat")), /Model service offline\./);
  await submitCommand(document, "/effort low high");
  assert.match(visibleText(document.getElementById("chat")), /Usage: \/effort \[<value>\]/);
});

test("escape keeps the current effort without selecting another", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    conversation: { provider: "anthropic", model: "claude-sonnet-5", effort: "high", mode: "ask" },
  });
  client.models = async (provider) => ({
    provider,
    state: "available",
    models: [{ id: "claude-sonnet-5", efforts: ["low", "high", "max"] }],
    detail: "",
  });
  await start({ global: {}, document, client });

  await submitCommand(document, "/effort");
  assert.equal(document.getElementById("prompt").value, "");
  const picker = find(document.getElementById("chat-pane"), (node) => node.className === "picker");
  await picker.dispatch("keydown", { key: "Escape" });

  assert.equal(document.getElementById("prompt").value, "");
  assert.match(visibleText(document.getElementById("chat")), /Kept effort as high\./);
  assert.deepEqual(client.calls.selectProvider, []);
});

test("commands clear the composer before running", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), { conversation: { provider: "openai", model: "gpt-one", mode: "ask" } });
  await start({ global: {}, document, client });
  const input = document.getElementById("prompt");
  input.value = "/nope keep this";
  await document.getElementById("prompt-form").dispatch("submit");
  assert.equal(input.value, "");
  assert.deepEqual(latestCommandOutput(document), ["Unknown command: /nope. Type / to see commands."]);
  input.value = "/mode";
  await document.getElementById("prompt-form").dispatch("submit");
  assert.equal(input.value, "");
  assert.deepEqual(client.calls.prompt, []);
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
  await find(document.getElementById("sidebar"), (value) => value.textContent === "Settings").dispatch("click");
  assert.equal(page.children[0].className, "settings-pane");
  assert.equal(app.route().page, "settings");
  assert.equal(browser.global.location.hash, "#/settings");
  assert.deepEqual(browser.writes.at(-1), ["symphonai.route", "#/settings"]);

  await document.getElementById("home-link").dispatch("click");
  assert.deepEqual(page.children, [chatPane]);
  assert.equal(document.getElementById("chat").children[0].textContent, "still here");
});

test("the removed roadmap route falls back to chat unless a URL fragment wins", async () => {
  const storedDocument = new FakeDocument();
  const stored = fakeGlobal({ stored: "#/roadmap" });
  const storedApp = await start({ global: stored.global, document: storedDocument, client: fakeClient() });
  assert.equal(storedDocument.getElementById("page").children[0].className, "chat-pane");
  assert.equal(storedApp.route().page, "chat");

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

test("sidebar places Settings after the scrolling conversation list", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal({ fragment: "#/settings/general" });
  const longTitle = "A very long session title wider than the sidebar";
  const client = fakeClient(fixtureRoadmap(), {
    sessions: [{ run_id: "long", title: longTitle, repo_root: "/work/current" }],
  });
  await start({ global: browser.global, document, client });
  const sidebar = document.getElementById("sidebar");
  const links = document.getElementById("page-links");

  const settingsLink = sidebar.children.at(-1);
  assert.deepEqual(links.children.map((link) => link.textContent), ["Chat"]);
  assert.equal(settingsLink.textContent, "Settings");
  assert.equal(settingsLink.className, "icon-settings");
  assert.equal(settingsLink.attributes.get("aria-current"), "page");
  assert.ok(find(document.getElementById("status-rail"), (node) => node.id === "changes"));
  assert.doesNotMatch(visibleText(sidebar), /Roadmap/);
  assert.deepEqual(sidebar.children, [
    document.getElementById("home-link"),
    document.getElementById("new-chat"),
    links,
    find(sidebar, (node) => node.className === "projects"),
    settingsLink,
  ]);
  assert.equal(sidebar.children[0], document.getElementById("home-link"));
  const sessionLink = find(sidebar, (node) => node.className === "session-link");
  assert.equal(sessionLink.attributes.get("title"), longTitle);
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  assert.match(css, /\.icon-settings\s*\{[^}]*--icon-image:\s*url\("data:image\/svg\+xml,[^"]*%3Cpath[^}]*%3Ccircle/s);
  assert.match(css, /\.changes-file,\s*\.changes-worktree,/);
  await links.children[0].dispatch("click");
  assert.equal(browser.global.location.hash, "#/chat");
  assert.deepEqual(document.getElementById("page").children, [document.getElementById("chat-pane")]);
  await document.getElementById("home-link").dispatch("click");
  assert.equal(browser.global.location.hash, "#/chat");
  assert.deepEqual(document.getElementById("page").children, [document.getElementById("chat-pane")]);
  assert.ok(find(document.getElementById("roadmap"), (node) => node.className === "roadmap-overview"));
});

test("sidebar tracks the active page and Settings section", async () => {
  const document = new FakeDocument();
  const { global } = fakeGlobal();
  await start({ global, document, client: fakeClient() });
  const links = document.getElementById("page-links");
  const settingsLink = find(document.getElementById("sidebar"), (link) => link.className === "icon-settings");
  const page = document.getElementById("page");
  const currentPages = () => [...links.children, settingsLink]
    .filter((link) => link.attributes.get("aria-current") === "page");

  assert.deepEqual(currentPages().map((link) => link.textContent), ["Chat"]);
  await settingsLink.dispatch("click");
  assert.deepEqual(currentPages().map((link) => link.textContent), ["Settings"]);
  const sections = find(page, (node) => node.className === "settings-sections");
  assert.deepEqual(sections.children.filter((link) => link.attributes.get("aria-current"))
    .map((link) => link.textContent), ["General"]);

  const mcp = sections.children.find((link) => link.textContent === "Mcp");
  await mcp.dispatch("click");
  assert.deepEqual(currentPages().map((link) => link.textContent), ["Settings"]);
  assert.deepEqual(sections.children.filter((link) => link.attributes.get("aria-current"))
    .map((link) => link.textContent), ["Mcp"]);
  await links.children[0].dispatch("click");
  assert.deepEqual(currentPages().map((link) => link.textContent), ["Chat"]);
  assert.deepEqual(sections.children.filter((link) => link.attributes.get("aria-current")), []);
});

test("settings section highlight compares route sections with browser-style hrefs", async () => {
  const document = new FakeDocument();
  const { global } = fakeGlobal({ fragment: "#/settings/mcp" });
  await start({ global, document, client: fakeClient() });
  const sections = find(document.getElementById("page"), (node) => node.className === "settings-sections");
  const selected = sections.children.filter((link) => link.attributes.get("aria-current") === "page");

  assert.deepEqual(selected.map((link) => link.textContent), ["Mcp"]);
  assert.equal(selected[0].href, "http://127.0.0.1/app/#/settings/mcp");
  assert.equal(selected[0].attributes.get("href"), "#/settings/mcp");
});

test("session buttons keep the centred navigation row styles", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const buttonRule = css.match(/button\.session-link\s*\{([^}]*)\}/)?.[1] ?? "";
  const sessionType = [...css.matchAll(/\.session-link\s*\{([^}]*)\}/g)]
    .find((match) => /font-size:/.test(match[1]))?.[1] ?? "";
  const metadataSelectors = css.match(/(?:^|\n)(\.conversation-usage,[\s\S]*?\.roadmap-task)\s*\{/)?.[1] ?? "";

  assert.match(buttonRule, /width:\s*calc\(100% - 1rem\);/);
  assert.match(buttonRule, /text-align:\s*left;/);
  assert.doesNotMatch(buttonRule, /display:\s*block;|padding:\s*(?:var\(--space-\d+\)|\d+(?:\.\d+)?(?:px|rem)?)\s/);
  assert.match(sessionType, /font-size:\s*0\.8125rem;/);
  assert.ok(metadataSelectors);
  assert.doesNotMatch(metadataSelectors, /\.session-link/);
});

test("sidebar marks the open conversation and rolls back selection on failure", async () => {
  const document = new FakeDocument();
  const sessions = [
    { run_id: "a", title: "Conversation A", repo_root: "/work/current" },
    { run_id: "b", title: "Conversation B", repo_root: "/work/current" },
  ];
  const client = fakeClient(fixtureRoadmap(), { sessions });
  const openSession = client.openSession;
  client.openSession = async (runId) => {
    if (runId === "a") throw new Error("could not open");
    return openSession(runId);
  };
  await start({ global: {}, document, client });
  const sidebar = document.getElementById("sidebar");
  const sessionLink = (title) => find(sidebar, (node) =>
    node.className === "session-link" && node.textContent === title
  );
  const currentSessions = () => walk(sidebar).filter((node) =>
    node.className === "session-link" && node.attributes.get("aria-current") === "true"
  );

  assert.deepEqual(currentSessions(), []);
  await sessionLink("Conversation B").dispatch("click");
  assert.deepEqual(currentSessions().map((node) => node.textContent), ["Conversation B"]);
  await assert.rejects(sessionLink("Conversation A").dispatch("click"), /could not open/);
  assert.deepEqual(currentSessions().map((node) => node.textContent), ["Conversation B"]);
  await document.getElementById("new-chat").dispatch("click");
  assert.deepEqual(currentSessions(), []);
});

test("sidebar navigation rows use selected and hover styles", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const current = css.match(/#page-links a\[aria-current="page"\][\s\S]*?\{([^}]*)\}/)?.[1] ?? "";
  const settingsCurrent = css.match(/#sidebar > \.icon-settings\[aria-current="page"\][\s\S]*?\{([^}]*)\}/)?.[1] ?? "";
  const session = css.match(/\.session-link\[aria-current="true"\][\s\S]*?\{([^}]*)\}/)?.[1] ?? "";
  const rows = css.match(/#page-links a,[\s\S]*?#sidebar > \.icon-settings,[\s\S]*?\.session-link\s*\{([^}]*)\}/)?.[1] ?? "";

  assert.match(current, /background:\s*var\(--ui-selected\);/);
  assert.match(settingsCurrent, /background:\s*var\(--ui-selected\);/);
  assert.match(session, /background:\s*var\(--ui-selected\);/);
  assert.match(current, /font-weight:\s*600;/);
  assert.match(session, /font-weight:\s*600;/);
  assert.match(rows, /min-height:\s*2rem;/);
  assert.match(rows, /align-items:\s*center;/);
  assert.match(rows, /padding:\s*0 var\(--space-2\);/);
  assert.match(rows, /border-radius:\s*var\(--radius-control\);/);
  assert.match(rows, /text-decoration:\s*none;/);
  assert.match(css, /#page-links a:hover,[\s\S]*?background:\s*var\(--ui-hover\);/);
  assert.match(css, /#sidebar > \.icon-settings:hover,[\s\S]*?background:\s*var\(--ui-hover\);/);
  assert.match(css, /#home-link\s*\{[^}]*font-size:\s*0\.875rem;[^}]*font-weight:\s*600;[^}]*cursor:\s*pointer;/);
});

test("session list scrolls below page links inside the fixed sidebar", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const sidebar = css.match(/#sidebar\s*\{([^}]*)\}/)?.[1] ?? "";
  const projects = css.match(/\.projects\s*\{([^}]*)\}/)?.[1] ?? "";

  assert.match(sidebar, /display:\s*flex;/);
  assert.match(sidebar, /flex-direction:\s*column;/);
  assert.match(sidebar, /height:\s*100vh;/);
  assert.match(sidebar, /overflow:\s*hidden;/);
  assert.match(sidebar, /overflow-x:\s*hidden;/);
  assert.match(projects, /overflow-x:\s*hidden;/);
  const sessionLink = css.match(/\.project-status,\s*\.project-empty,\s*\.session-link\s*\{([^}]*)\}/)?.[1] ?? "";
  assert.match(sessionLink, /white-space:\s*nowrap;/);
  assert.match(sessionLink, /overflow:\s*hidden;/);
  assert.match(sessionLink, /text-overflow:\s*ellipsis;/);
  assert.match(projects, /flex:\s*1;/);
  assert.match(projects, /min-height:\s*0;/);
  assert.match(projects, /overflow-y:\s*auto;/);
  assert.doesNotMatch(css, /#page-links\s*\{[^}]*display:\s*none;/);
});

test("agent Save remains the primary action", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  assert.doesNotMatch(html, /id="(?:attach-button|send-button|stop-button|attachment-picker)"/);
  assert.equal((html.match(/class="[^"]*\bprimary\b[^"]*"/g) ?? []).length, 0);

  const settingsLink = find(document.getElementById("sidebar"), (value) => value.className === "icon-settings");
  await settingsLink.dispatch("click");
  await find(document.getElementById("page"), (value) => value.textContent === "Agents").dispatch("click");
  const settings = document.getElementById("page");
  await find(settings, (value) => value.textContent === "New agent").dispatch("click");
  assert.equal(find(settings, (value) => value.className === "agent-save primary").textContent, "Save");
});

test("approvals from another conversation show its title and open that session", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    sessions: [
      { run_id: "session-a", title: "Background work", repo_root: "/work/current", activity: "waiting" },
    ],
  });
  await start({ global: {}, document, client });
  await client.emit({ kind: "approval_requested", payload: {
    approval_id: "approval-a", operation: "write_file", target: "a.py",
    details: "write file", session_id: "session-a",
  } });
  const approvals = document.getElementById("approvals");
  assert.match(visibleText(approvals), /Background work/);
  await find(approvals, (value) => value.textContent === "Open").dispatch("click");
  assert.deepEqual(client.calls.openSession, ["session-a"]);
  const waiting = find(document.getElementById("sidebar"), (value) => (
    value.className === "session-link" && value.textContent.includes("Background work")
  ));
  assert.match(waiting.textContent, /waiting/);
});

test("agent rail tools replace section headings and keep separated rows", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  assert.match(html, /id="rail-tabs" class="rail-tabs" role="tablist"/);
  assert.match(html, /id="agents" class="rail-pane"/);
  assert.match(html, /id="roadmap" class="rail-pane" role="tabpanel" hidden/);
  assert.match(html, /id="changes" class="rail-pane changes-pane" role="tabpanel" hidden/);
  assert.match(html, /id="spec" class="rail-pane spec-view"[^>]*hidden/);
  assert.match(html, /id="files" class="rail-pane files-pane" role="tabpanel" hidden/);
  assert.match(html, /id="rail-tools" class="rail-tools"/);
  assert.doesNotMatch(html, /<h2 class="rail-title/);
  assert.match(css, /\.rail-tab\[aria-selected="true"\]\s*\{[^}]*background:\s*var\(--ui-selected\);/s);
  assert.match(css, /\.rail-pane,\s*\.rail-tools\s*\{[^}]*overflow:\s*auto;/s);
  assert.match(css, /\.agent-row\s*\{[^}]*padding:\s*var\(--space-2\) 0;/s);
  assert.match(css, /#agents\s*>\s*\.agent-row\s*\+\s*\.agent-row\s*\{[^}]*border-top:\s*1px solid var\(--ui-border\);/s);
  assert.doesNotMatch(css, /\.agent-children\s*>\s*\.agent-row\s*\+\s*\.agent-row\s*\{[^}]*border-top:/s);
  assert.match(css, /\.icon-tool-files\s*\{[^}]*--icon-image:\s*url\("data:image\/svg\+xml,/s);
  assert.match(css, /\.files-name\s*\{[^}]*text-overflow:\s*ellipsis/s);
  assert.match(css, /\.files-source\s*\{[^}]*overflow:\s*auto[^}]*grid-template-columns:/s);
  assert.match(css, /\.files-line-numbers\s*\{[^}]*user-select:\s*none/s);
});

test("rail tabs open tools, close to the Tools list, and persist the selected layout", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal();
  const client = fakeClient();
  await start({ global: browser.global, document, client });
  const tabs = document.getElementById("rail-tabs");
  const agents = document.getElementById("agents");
  const roadmap = document.getElementById("roadmap");
  const changes = document.getElementById("changes");
  const tools = document.getElementById("rail-tools");
  const tabButtons = () => find(tabs, (node) => node.className === "rail-tab-item")
    ? tabs.children.filter((node) => node.className === "rail-tab-item").map((row) => row.children[0])
    : [];

  assert.deepEqual(tabButtons().map((tab) => tab.textContent || tab.attributes.get("aria-label")), ["Agents", "Roadmap"]);
  assert.equal(tabButtons()[0].attributes.get("aria-selected"), "true");
  assert.equal(agents.hidden, false);
  assert.equal(roadmap.hidden, true);
  assert.equal(changes.hidden, true);
  assert.equal(document.getElementById("spec").hidden, true);
  assert.equal(document.getElementById("files").hidden, true);
  assert.equal(tools.hidden, true);

  const add = tabs.children.at(-1);
  assert.equal(add.attributes.get("aria-label"), "Open a tool");
  await find(tabs, (node) => node.attributes.get("aria-label") === "Close Agents").dispatch("click");
  await find(tabs, (node) => node.attributes.get("aria-label") === "Close Roadmap").dispatch("click");
  assert.equal(tools.hidden, false);
  assert.deepEqual(tools.children.slice(1).map((row) => row.textContent), ["Agents", "Roadmap", "Changes", "Spec", "Files"]);
  await tools.children.find((row) => row.textContent === "Changes").dispatch("click");
  assert.equal(client.calls.changes, 1);
  assert.deepEqual(tabButtons().map((tab) => tab.textContent || tab.attributes.get("aria-label")), ["Changes"]);
  assert.equal(tabButtons()[0].attributes.get("aria-selected"), "true");
  assert.equal(changes.hidden, false);
  const saved = browser.writes.filter(([key]) => key === "symphonai.rail-tools").at(-1);
  assert.deepEqual(JSON.parse(saved[1]), { open: ["Changes"], selected: "Changes" });

  const restoredDocument = new FakeDocument();
  const restored = fakeGlobal({
    railStored: JSON.stringify({ open: ["Roadmap", "Changes"], selected: "Changes" }),
  });
  const restoredClient = fakeClient();
  await start({ global: restored.global, document: restoredDocument, client: restoredClient });
  const restoredTabs = restoredDocument.getElementById("rail-tabs").children;
  assert.deepEqual(restoredTabs.slice(0, -1).map((row) => row.children[0].attributes.get("aria-label")), ["Roadmap", "Changes"]);
  assert.equal(restoredTabs[1].children[0].attributes.get("aria-selected"), "true");
  assert.equal(restoredDocument.getElementById("changes").hidden, false);
  assert.equal(restoredClient.calls.changes, 1);

  const throwingDocument = new FakeDocument();
  await start({ global: fakeGlobal({ storageThrows: true }).global, document: throwingDocument, client: fakeClient() });
  assert.deepEqual(throwingDocument.getElementById("rail-tabs").children.slice(0, -1).map((row) => row.children[0].attributes.get("aria-label")), ["Agents", "Roadmap"]);
  assert.equal(throwingDocument.getElementById("agents").hidden, false);
});

test("roadmap task clicks open and reuse the Spec tool tab", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  const tasks = () => walk(document.getElementById("roadmap"))
    .filter((node) => node.className.startsWith("roadmap-task "));
  const firstTask = tasks()[0];
  assert.ok(firstTask);
  await firstTask.dispatch("click");
  assert.equal(document.getElementById("spec").hidden, false);
  assert.deepEqual(
    document.getElementById("rail-tabs").children.slice(0, -1).map((row) => row.children[0].attributes.get("aria-label")),
    ["Agents", "Roadmap", "Spec"],
  );
  assert.equal(document.getElementById("rail-tabs").children[2].children[0].attributes.get("aria-selected"), "true");
  await tasks()[1].dispatch("click");
  assert.equal(
    document.getElementById("rail-tabs").children.filter((row) => row.children[0]?.attributes.get("aria-label") === "Spec").length,
    1,
  );
});

test("Changes tool restores revert conflict and worktree actions", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const reply = {
    files: [{ path: "a.py", status: "modified", diff: "@@ -1 +1 @@\n-old\n+new" }],
    turns: [{ key: "turn-1", prompt: "change a.py", paths: ["a.py"] }],
    worktrees: [{ name: "w1", files: ["b.py"], diff: "+worktree" }],
  };
  client.changes = async () => {
    client.calls.changes += 1;
    return reply;
  };
  client.revertChanges = async (payload) => {
    client.calls.revertChanges.push(payload);
    if (client.calls.revertChanges.length === 1) {
      const error = new Error("changed outside");
      error.status = 409;
      error.paths = ["a.py"];
      throw error;
    }
    return {};
  };
  await start({ global: {}, document, client });
  await find(document.getElementById("rail-tabs"), (node) => node.attributes.get("aria-label") === "Open a tool").dispatch("click");
  await document.getElementById("rail-tools").children.find((row) => row.textContent === "Changes").dispatch("click");
  const changes = document.getElementById("changes");
  await find(changes, (node) => node.textContent === "Revert").dispatch("click");
  assert.deepEqual(client.calls.revertChanges, [{ path: "a.py" }]);
  assert.match(visibleText(changes), /These files changed outside the agent:.*a\.py/s);
  await find(changes, (node) => node.textContent === "Revert anyway").dispatch("click");
  assert.deepEqual(client.calls.revertChanges[1], { path: "a.py", force: true });
  await find(changes, (node) => node.textContent === "Revert this prompt and later").dispatch("click");
  assert.deepEqual(client.calls.revertChanges[2], { key: "turn-1" });
  await find(changes, (node) => node.textContent === "Apply worktree").dispatch("click");
  await find(changes, (node) => node.textContent === "Discard worktree").dispatch("click");
  assert.deepEqual(client.calls.applyWorktree, ["w1"]);
  assert.deepEqual(client.calls.discardWorktree, ["w1"]);
  assert.equal(client.calls.changes, 5);
});

test("rail icon tabs show the selected name and move the only close button", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient() });
  const tabs = document.getElementById("rail-tabs");
  const openFromTools = async (name) => {
    await tabs.children.at(-1).dispatch("click");
    await document.getElementById("rail-tools").children
      .find((row) => row.textContent === name).dispatch("click");
  };
  await openFromTools("Changes");
  await openFromTools("Spec");
  const rows = tabs.children.slice(0, -1);
  assert.deepEqual(rows.map((row) => row.children[0].textContent), ["", "", "", "Spec"]);
  assert.deepEqual(rows.map((row) => row.children[0].attributes.get("aria-label")), ["Agents", "Roadmap", "Changes", "Spec"]);
  assert.deepEqual(rows.map((row) => row.children[0].attributes.get("title")), ["Agents", "Roadmap", "Changes", "Spec"]);
  assert.equal(rows[3].attributes.get("aria-selected"), "true");
  assert.deepEqual(rows.map((row) => row.children.length), [1, 1, 1, 2]);
  assert.equal(walk(tabs).filter((node) => node.className === "rail-tab-close").length, 1);
  assert.match(visibleText(document.getElementById("spec")), /Choose a task in Roadmap to open its spec\./);

  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const rule = (selector) => css.match(new RegExp(`${selector}\\s*\\{([^}]*)\\}`))?.[1] ?? "";
  const railTabs = rule("\\.rail-tabs");
  const railTab = rule("\\.rail-tab");
  assert.doesNotMatch(railTabs, /overflow-x:\s*(?:auto|scroll);/);
  assert.match(railTabs, /width:\s*calc\(100% - var\(--space-8\) - var\(--space-3\)\);/);
  assert.match(railTab, /width:\s*1\.5rem;/);
  assert.doesNotMatch(railTab, /text-overflow:\s*ellipsis/);
  for (const icon of ["agents", "roadmap", "changes", "spec", "files"]) {
    assert.match(css, new RegExp(`\\.icon-tool-${icon}\\s*\\{[^}]*--icon-image:\\s*url\\("data:image/svg\\+xml,`, "s"));
  }

  await rows[1].children[0].dispatch("click");
  const selectedRows = tabs.children.slice(0, -1);
  assert.deepEqual(selectedRows.map((row) => row.children[0].textContent), ["", "Roadmap", "", ""]);
  assert.equal(selectedRows[1].children[0].attributes.get("aria-selected"), "true");
  assert.deepEqual(selectedRows.map((row) => row.children.length), [1, 2, 1, 1]);
  assert.equal(walk(tabs).filter((node) => node.className === "rail-tab-close").length, 1);
});

test("Files opens on demand, expands in place, keeps line numbers out of source text, and restores its tree", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal();
  const client = fakeClient();
  client.calls.tree = [];
  client.calls.source = [];
  client.tree = async (path = "") => {
    client.calls.tree.push(path);
    return path === ""
      ? { path, entries: [{ name: "src", kind: "dir" }], truncated: false }
      : { path, entries: [{ name: "a.py", kind: "file" }], truncated: false };
  };
  client.source = async (path) => {
    client.calls.source.push(path);
    return { path, text: "alpha\nbeta\ngamma" };
  };
  await start({ global: browser.global, document, client });

  const tabs = document.getElementById("rail-tabs");
  const tools = document.getElementById("rail-tools");
  const files = document.getElementById("files");
  assert.deepEqual(tabs.children.slice(0, -1).map((row) => row.children[0].attributes.get("aria-label")), ["Agents", "Roadmap"]);
  assert.equal(files.hidden, true);
  await tabs.children.at(-1).dispatch("click");
  await tools.children.find((row) => row.textContent === "Files").dispatch("click");
  assert.deepEqual(client.calls.tree, [""]);
  assert.equal(files.hidden, false);

  const directory = find(files, (node) => node.className.includes("files-directory"));
  assert.equal(directory.children[1].textContent, "src");
  await directory.dispatch("click");
  assert.deepEqual(client.calls.tree, ["", "src"]);
  const file = find(files, (node) => node.className.includes("files-file"));
  assert.equal(file.children[1].textContent, "a.py");
  assert.match(file.attributes.get("style"), /space-4.*1/);
  await file.dispatch("click");
  assert.deepEqual(client.calls.source, ["src/a.py"]);
  const source = find(files, (node) => node.className === "files-source");
  assert.equal(find(source, (node) => node.className === "files-code").textContent, "alpha\nbeta\ngamma");
  assert.equal(find(source, (node) => node.className === "files-line-numbers").textContent, "1\n2\n3");
  await find(files, (node) => node.className === "files-back").dispatch("click");
  const expanded = find(files, (node) => node.className.includes("files-directory"));
  assert.equal(expanded.attributes.get("aria-expanded"), "true");
  assert.ok(find(files, (node) => node.className.includes("files-file")));
  assert.deepEqual(client.calls.tree, ["", "src"]);

  const saved = browser.writes.filter(([key]) => key === "symphonai.rail-tools").at(-1);
  assert.ok(JSON.parse(saved[1]).open.includes("Files"));
  const restoredDocument = new FakeDocument();
  const restoredClient = fakeClient();
  restoredClient.calls.tree = [];
  restoredClient.tree = async (path = "") => {
    restoredClient.calls.tree.push(path);
    return { path, entries: [], truncated: false };
  };
  await start({
    global: fakeGlobal({ railStored: saved[1] }).global,
    document: restoredDocument,
    client: restoredClient,
  });
  assert.deepEqual(restoredDocument.getElementById("rail-tabs").children.slice(0, -1)
    .map((row) => row.children[0].attributes.get("aria-label")), ["Agents", "Roadmap", "Files"]);
  assert.equal(restoredDocument.getElementById("files").hidden, false);
  assert.deepEqual(restoredClient.calls.tree, [""]);
});

test("Files displays readable source errors without throwing", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.tree = async (path = "") => ({
    path, entries: [{ name: "image.png", kind: "file" }], truncated: false,
  });
  client.source = async () => {
    const error = new Error("host request failed with status 415");
    error.status = 415;
    throw error;
  };
  await start({ global: {}, document, client });
  const tabs = document.getElementById("rail-tabs");
  await tabs.children.at(-1).dispatch("click");
  await document.getElementById("rail-tools").children.find((row) => row.textContent === "Files").dispatch("click");
  const files = document.getElementById("files");
  await find(files, (node) => node.className.includes("files-file")).dispatch("click");
  assert.equal(find(files, (node) => node.className === "files-source-error").textContent, "Not a text file");
});

test("selected Changes reloads after opening a session, starting a chat, and finishing a run", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    sessions: [
      { run_id: "session-one", title: "First chat", repo_root: "/work/current" },
      { run_id: "session-two", title: "Second chat", repo_root: "/work/current" },
    ],
  });
  await start({ global: {}, document, client });
  const tabs = document.getElementById("rail-tabs");
  await tabs.children.at(-1).dispatch("click");
  await document.getElementById("rail-tools").children
    .find((row) => row.textContent === "Changes").dispatch("click");
  assert.equal(client.calls.changes, 1);

  await find(document.getElementById("sidebar"), (node) =>
    node.className === "session-link" && node.textContent.startsWith("Second chat")
  ).dispatch("click");
  assert.equal(client.calls.changes, 2);

  await document.getElementById("new-chat").dispatch("click");
  assert.equal(client.calls.changes, 3);

  await client.emit(eventFrame("RunFinished", { run_id: "run-current" }));
  assert.equal(client.calls.changes, 4);
});

test("agent name and detail styles are scoped to rail rows", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const selectors = [...css.matchAll(/([^{}]+)\{[^{}]*\}/g)]
    .flatMap(([, group]) => group.split(",").map((selector) => selector.trim()))
    .filter((selector) => /\.agent-(?:name|detail)\b/.test(selector));
  assert.ok(selectors.length > 0);
  for (const selector of selectors) {
    assert.match(
      selector,
      /(?:^|[\s>+~])(?:\.agent-row|#agents)(?=[\s>+~]|$).*\.agent-(?:name|detail)\b/,
      selector,
    );
  }
});

test("DOM child collections are not called with array-only methods", async () => {
  const names = await readdir(new URL("../src/", import.meta.url));
  const arrayOnlyDomChildMethods = /\b(?:line|parent|element|container|row|root|card|taskBlock)\s*(?:\.|\?\.)\s*(?:children|childNodes)\s*(?:\.|\?\.)\s*(?:at|find|filter|map|some|every|reduce|flatMap|slice|includes|indexOf|findIndex|lastIndexOf|forEach|sort|reverse|concat|join|keys|values|entries|copyWithin|fill|flat|toReversed|toSorted|toSpliced|with)\s*\(/;
  for (const name of names.filter((file) => file.endsWith(".js"))) {
    const source = await readFile(new URL(`../src/${name}`, import.meta.url), "utf8");
    assert.doesNotMatch(source, arrayOnlyDomChildMethods, name);
  }
});

test("agent rail shows current activity and state without controls", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  let conversation = null;
  client.conversationStats = async () => ({ conversation });
  await start({ global: {}, document, client });
  const agents = document.getElementById("agents");
  assert.equal(visibleText(agents), "\nNothing is running.");

  await client.emit(eventFrame("RunStarted", { agent_name: "leader" }));
  await client.emit(eventFrame("ToolCallStarted", { tool_name: "edit_file", target: "app.js" }));
  await client.emit(eventFrame("SubagentSpawned", { subagent_agent_id: "worker", subagent_name: "worker" }));
  await client.emit(eventFrame("ToolCallStarted", { agent_id: "worker", tool_name: "run_shell", target: "pytest" }));
  await client.emit(eventFrame("ToolCallFinished", { agent_id: "worker", tool_name: "run_shell" }));
  conversation = { agents: [{ agent_id: "researcher", parent_agent_id: "agent-1", name: "researcher" }] };
  await client.emit(eventFrame("RunFinished", { agent_id: "worker" }));

  let leader = agents.children[0];
  let children = leader.children.find((row) => row.className === "agent-children");
  assert.equal(leader.children.find((row) => row.className === "agent-name").textContent, "leader");
  assert.equal(leader.children.find((row) => row.className === "agent-detail").textContent, "running · editing app.js");
  assert.equal(agentRowText(leader), "leader · running · editing app.js");
  assert.deepEqual(children.children.map(agentRowText), [
    "worker · finished", "researcher",
  ]);
  assert.equal(children.children[1].children.some((row) => row.className === "agent-detail"), false);
  assert.equal(walk(agents).some((row) => ["agent-controls", "agent-redirect", "agent-control-error"].includes(row.className)), false);

  await client.emit(eventFrame("PermissionRequested", { tool_name: "edit_file" }));
  assert.equal(agentRowText(agents.children[0]), "leader · waiting for approval · editing app.js");
  assert.equal(walk(agents).some((row) => ["agent-controls", "agent-redirect", "agent-control-error"].includes(row.className)), false);
  await client.emit(eventFrame("PermissionDenied", { tool_name: "edit_file" }));
  await client.emit(eventFrame("ToolCallFinished", { tool_name: "edit_file" }));
  assert.equal(agentRowText(agents.children[0]), "leader · running · thinking");
  await client.emit(eventFrame("RunFinished"));

  leader = agents.children[0];
  children = leader.children.find((row) => row.className === "agent-children");
  assert.equal(agentRowText(leader), "leader · finished");
  assert.deepEqual(children.children.map(agentRowText), [
    "worker · finished", "researcher",
  ]);
  assert.equal(walk(agents).some((row) => ["agent-controls", "agent-redirect", "agent-control-error"].includes(row.className)), false);
  assert.doesNotMatch(visibleText(agents), /Nothing is running/);
});

test("conversation agents with missing or unknown states show no state word", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.conversationStats = async () => ({ conversation: { agents: [
    { agent_id: "a", name: "researcher", parent_agent_id: null },
    { agent_id: "b", name: "reviewer", parent_agent_id: null, state: "blue" },
  ] } });
  await start({ global: {}, document, client });

  const agents = document.getElementById("agents");
  assert.deepEqual(agents.children.map(agentRowText), ["researcher", "reviewer"]);
  assert.equal(agents.children.every((row) => !row.children.some((child) => child.className === "agent-detail")), true);
  assert.equal(find(agents, (row) => ["agent-controls", "agent-redirect", "agent-control-error"].includes(row.className)), undefined);
});

test("agent activity uses the tool verbs and optional targets while running", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  const agents = document.getElementById("agents");
  for (const [tool, target, present] of [
    ["read_file", "a.py", "reading a.py"],
    ["edit_file", "a.py", "editing a.py"],
    ["multi_edit_file", "a.py", "editing a.py"],
    ["write_file", "a.py", "editing a.py"],
    ["run_shell", "pytest", "running pytest"],
    ["grep", "src", "searching src"],
    ["glob", "src", "searching src"],
    ["list_files", "src", "searching src"],
    ["web_fetch", "example.com", "fetching example.com"],
    ["web_search", "docs", "searching the web for docs"],
    ["dispatch_subagent", "worker", "dispatching worker"],
    ["custom_tool", "item", "custom_tool item"],
    ["read_file", undefined, "reading"],
  ]) {
    await client.emit(eventFrame("RunStarted", { agent_name: "leader" }));
    await client.emit(eventFrame("ToolCallStarted", { tool_name: tool, ...(target === undefined ? {} : { target }) }));
    assert.equal(agentRowText(agents.children[0]), `leader · running · ${present}`, tool);
    await client.emit(eventFrame("ToolCallFailed", { tool_name: tool }));
    assert.equal(agentRowText(agents.children[0]), "leader · running · thinking", tool);
    await client.emit(eventFrame("RunFinished"));
    assert.equal(agentRowText(agents.children[0]), "leader · finished", tool);
  }
});

test("dispatch activity expands its matched subagent task with click and Enter", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await client.emit(eventFrame("RunStarted", { agent_id: "leader", run_id: "root-run", agent_name: "leader" }));
  await client.emit(eventFrame("PromptSubmitted", { agent_id: "leader", run_id: "root-run", text: "investigate" }));
  await client.emit(eventFrame("ToolCallStarted", {
    agent_id: "leader", run_id: "root-run", tool_call_id: "dispatch", tool_name: "dispatch_subagent", target: "explorer",
  }));
  await client.emit(eventFrame("SubagentSpawned", {
    agent_id: "leader", run_id: "root-run", subagent_name: "explorer", subagent_agent_id: "child",
  }));
  await client.emit(eventFrame("PromptSubmitted", { agent_id: "child", run_id: "child-run", text: "find the bug" }));
  const chat = document.getElementById("chat");
  const dispatch = find(chat, (node) => node.className.includes("activity"));
  assert.equal(find(dispatch, (node) => node.className === "activity-text").children.map((node) => node.textContent).join(""), "Dispatched explorer.");
  assert.equal(find(chat, (node) => node.className === "prompt" && node.textContent === "find the bug"), undefined);
  assert.equal(find(dispatch, (node) => node.className === "dispatch-tasks"), undefined);
  await dispatch.click();
  assert.equal(visibleText(find(find(chat, (node) => node.className.includes("activity")), (node) => node.className === "dispatch-tasks")).trim(), "explorer: find the bug");
  const expanded = find(chat, (node) => node.className.includes("activity"));
  await expanded.dispatch("keydown", { key: "Enter", preventDefault() {} });
  assert.equal(find(find(chat, (node) => node.className.includes("activity")), (node) => node.className === "dispatch-tasks"), undefined);
});

test("reopened history renders its collapsed dispatch task and turn time", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.changes = async () => {
    client.calls.changes += 1;
    return { files: [], turns: [{ key: "checkpoint-replayed", paths: ["a.py"] }], worktrees: [] };
  };
  await start({ global: {}, document, client });
  await client.emit(eventFrame("HistoryMessage", {
    role: "user", text: "investigate", tool_calls: [], turn_id: null,
    checkpoint_key: "checkpoint-replayed",
    edits: [{ path: "a.py", added: 1, removed: 0 }],
    turn_started_at: "2026-10-12T17:21:00.000Z",
  }));
  await client.emit(eventFrame("HistoryMessage", {
    role: "assistant", text: "", turn_id: "turn-replayed",
    tool_calls: [{ id: "dispatch-replayed", name: "dispatch_subagent", subagent_name: "explorer", task: "find the bug" }],
    turn_ended_at: "2026-10-12T17:27:34.000Z",
    turn_outcome: "finished",
  }));
  const chat = document.getElementById("chat");
  const dispatch = find(chat, (node) => node.className.includes("activity"));
  assert.equal(find(dispatch, (node) => node.className === "dispatch-tasks"), undefined);
  const card = find(chat, (node) => node.className === "edit-card");
  assert.ok(card);
  assert.ok(find(card, (node) => node.tagName === "BUTTON" && node.textContent === "Undo"));
  assert.ok(chat.children.indexOf(card) < chat.children.indexOf(find(chat, (node) => node.className === "turn-time")));
  assert.equal(client.calls.changes, 1);
  assert.match(visibleText(chat), /Worked for 6m 34s ·/);
  await dispatch.dispatch("keydown", { key: "Enter", preventDefault() {} });
  assert.match(visibleText(chat), /explorer: find the bug/);
});

test("status rail renders the whole roadmap beside settings and live agents", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal({ fragment: "#/settings/general" });
  const client = fakeClient();
  await start({ global: browser.global, document, client });
  const rail = document.getElementById("status-rail");
  const agents = document.getElementById("agents");

  assert.deepEqual(rail.children, [
    document.getElementById("rail-tabs"), agents, document.getElementById("roadmap"),
    document.getElementById("changes"), document.getElementById("spec"),
    document.getElementById("files"), document.getElementById("rail-tools"),
  ]);
  assert.equal(document.getElementById("page").children[0].className, "settings-pane");
  assert.equal(document.getElementById("page").children.length, 1);
  assert.ok(find(document.getElementById("roadmap"), (node) => node.className === "roadmap-overview"));
  assert.equal(find(document.getElementById("roadmap"), (node) => node.className === "roadmap-open"), undefined);
  assert.match(visibleText(rail), /18 Desktop app/);
  assert.equal(visibleText(agents), "\nNothing is running.");

  await client.emit(eventFrame("RunStarted", { agent_name: "leader" }));
  await client.emit(eventFrame("ToolCallStarted", { tool_name: "read", tool_call_id: "one" }));
  await client.emit(eventFrame("SubagentSpawned", {
    subagent_agent_id: "agent-2", subagent_name: "worker",
  }));
  await client.emit(eventFrame("SubagentSpawned", {
    subagent_agent_id: "agent-3", subagent_name: "sibling",
  }));
  await client.emit(eventFrame("SubagentSpawned", {
    agent_id: "agent-2", subagent_agent_id: "agent-4", subagent_name: "grandchild",
  }));
  await client.emit(eventFrame("ToolCallStarted", { agent_id: "agent-4", tool_name: "search" }));
  await client.emit(eventFrame("SubagentStopped", { subagent_agent_id: "agent-2" }));
  const root = agents.children[0];
  const children = root.children.find((row) => row.className === "agent-children");
  assert.equal(children.className, "agent-children");
  assert.deepEqual(children.children.map(agentRowText), ["worker · finished", "sibling · running · thinking"]);
  const workerChildren = children.children[0].children.find((row) => row.className === "agent-children");
  assert.deepEqual(workerChildren.children.map(agentRowText), ["grandchild · running · search"]);
  assert.deepEqual(agents.children.map(agentRowText), ["leader · running · read"]);

  await client.emit(eventFrame("SubagentSpawned", {
    agent_id: "missing", subagent_agent_id: "orphan", subagent_name: "orphan",
  }));
  assert.deepEqual(agents.children.map(agentRowText), [
    "leader · running · read", "orphan · running · thinking",
  ]);
});

test("reopened history and conversation stats render a nested agent tree without run events", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    sessions: [{ run_id: "prior", title: "Prior", repo_root: "/work/current" }],
  });
  let conversation = null;
  client.conversationStats = async () => ({ conversation });
  client.openSession = async (runId) => {
    client.calls.openSession.push(runId);
    await client.emit({ kind: "event", payload: {
      type: "HistoryMessage", role: "assistant", text: "Prior answer", tool_calls: [], turn_id: "prior-turn",
    } });
    conversation = { agents: [
      { agent_id: "root", name: "leader", parent_agent_id: null, state: "finished" },
      { agent_id: "a", name: "researcher", parent_agent_id: "root", state: "finished" },
      { agent_id: "b", name: "B", parent_agent_id: "a", state: "finished" },
      { agent_id: "c", name: "C", parent_agent_id: "root", state: "finished" },
    ] };
    return { run_id: runId };
  };
  await start({ global: {}, document, client });
  const agents = document.getElementById("agents");
  assert.equal(visibleText(agents), "\nNothing is running.");
  const link = find(document.getElementById("sidebar"), (row) => row.className === "session-link");
  await link.dispatch("click");
  assert.deepEqual(client.calls.openSession, ["prior"]);
  assert.deepEqual(agents.children.map(agentRowText), ["leader · finished"]);
  const children = agents.children[0].children.find((row) => row.className === "agent-children");
  assert.equal(children.className, "agent-children");
  assert.deepEqual(children.children.map(agentRowText), ["researcher · finished", "C · finished"]);
  const researcherChildren = children.children[0].children.find((row) => row.className === "agent-children");
  assert.deepEqual(researcherChildren.children.map(agentRowText), ["B · finished"]);
  assert.equal(find(agents, (row) => ["agent-controls", "agent-redirect", "agent-control-error"].includes(row.className)), undefined);
  assert.equal(find(document.body, (row) => row.className === "conversation-usage").textContent, "");
  assert.match(visibleText(document.getElementById("chat")), /Prior answer/);
  assert.doesNotMatch(visibleText(agents), /tokens|USD/);
});

test("agent rail keeps finished parents and shows running worker state", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.conversationStats = async () => ({ conversation: { agents: [
    { agent_id: "leader", name: "leader", state: "finished" },
    { agent_id: "researcher", name: "researcher", parent_agent_id: "leader", state: "finished" },
    { agent_id: "worker", name: "worker", parent_agent_id: "leader", state: "running" },
  ] } });
  await start({ global: {}, document, client });

  const agents = document.getElementById("agents");
  assert.deepEqual(agents.children.map(agentRowText), ["leader · finished"]);
  const children = agents.children[0].children.find((row) => row.className === "agent-children");
  assert.equal(children.className, "agent-children");
  assert.deepEqual(children.children.map(agentRowText), ["researcher · finished", "worker · running · thinking"]);
  assert.equal(find(agents, (row) => ["agent-controls", "agent-redirect", "agent-control-error"].includes(row.className)), undefined);
  assert.equal(agents.children[0].children.some((row) => row.className === "agent-controls"), false);
  assert.equal(children.children[0].children.some((child) => child.className === "agent-children"), false);
});

test("opening another conversation and starting a new chat clears prior agents", async () => {
  const document = new FakeDocument();
  const sessions = [
    { run_id: "conversation-a", title: "Conversation A", repo_root: "/work/current" },
    { run_id: "conversation-b", title: "Conversation B", repo_root: "/work/current" },
  ];
  const client = fakeClient(fixtureRoadmap(), { sessions });
  const agents = document.getElementById("agents");
  let conversation = null;
  const replayBoard = [];
  client.sessions = async () => sessions;
  client.conversationStats = async () => ({ conversation });
  client.openSession = async (runId) => {
    client.calls.openSession.push(runId);
    await client.emit({ kind: "event", payload: {
      type: "HistoryMessage", role: "assistant", text: `History for ${runId}`,
      tool_calls: [], turn_id: `turn-${runId}`,
    } });
    replayBoard.push(visibleText(agents));
    conversation = { agents: runId === "conversation-a"
      ? [
        { agent_id: "leader-a", name: "leader", parent_agent_id: null },
        { agent_id: "worker-a", name: "worker", parent_agent_id: "leader-a" },
      ]
      : [{ agent_id: "leader-b", name: "leader", parent_agent_id: null }],
    };
    return { run_id: runId };
  };
  await start({ global: {}, document, client });

  const sessionLink = (title) => find(document.getElementById("sidebar"), (row) =>
    row.className === "session-link" && row.textContent === title
  );
  await sessionLink("Conversation A").dispatch("click");
  assert.deepEqual(walk(agents).filter((row) => row.className === "agent-row").map(agentRowText), [
    "leader", "worker",
  ]);

  await sessionLink("Conversation B").dispatch("click");
  assert.deepEqual(client.calls.openSession, ["conversation-a", "conversation-b"]);
  assert.deepEqual(replayBoard, ["\nNothing is running.", "\nNothing is running."]);
  assert.deepEqual(agents.children.map(agentRowText), ["leader"]);
  assert.equal(find(agents, (row) => row.className === "agent-name" && row.textContent.includes("worker")), undefined);

  await document.getElementById("new-chat").dispatch("click");
  assert.equal(visibleText(agents), "\nNothing is running.");
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

test("settings navigation renders hooks, ceiling, and trust before inventory", async () => {
  const browser = fakeGlobal({ fragment: "#/settings/hooks" });
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    settings: { settings: {
      config: [{ key: "visible.setting", value: true, scope: "user" }],
      hooks: [
        { event: "turn", command: "z-hook" },
        { event: "prompt", command: "p-hook" },
        { event: "turn", command: "a-hook" },
      ],
      ceiling: { allowed_write_scope: [], modes: ["plan", "allow"] },
      trust: [
        { root: "/z", allow: [] },
        { root: "/a", allow: ["agents", "skills"] },
      ],
    } },
  });
  await start({ global: browser.global, document, client });
  const pane = document.getElementById("page").children[0];
  const sections = find(pane, (value) => value.className === "settings-sections");
  const content = find(pane, (value) => value.className === "settings-content");
  const rowCells = () => walk(content)
    .filter((value) => value.className === "settings-row")
    .map((row) => row.children.map((cell) => cell.textContent));
  const open = async (name) => {
    await find(sections, (value) => value.textContent === name).dispatch("click");
    assert.equal(browser.global.location.hash, `#/settings/${name.toLowerCase()}`);
  };

  assert.deepEqual(sections.children.map((link) => link.textContent), [
    "General", "Models", "Mcp", "Skills", "Plugins", "Agents",
    "Memory", "Hooks", "Ceiling", "Trust", "Inventory",
  ]);
  assert.equal(content.children[0].textContent, "Hooks");
  assert.deepEqual(rowCells(), [
    ["prompt", "p-hook"], ["turn", "a-hook"], ["turn", "z-hook"],
  ]);

  await open("Ceiling");
  assert.equal(content.children[0].textContent, "Ceiling");
  assert.equal(content.children[1].textContent,
    'The most any agent may be granted. "not set" means the ceiling does not limit this.');
  assert.deepEqual(rowCells(), [
    ["allowed_write_scope", "none"],
    ["fetch_allowlist", "not set"],
    ["fetch_enabled", "not set"],
    ["modes", "plan, allow"],
    ["shell_allowlist", "not set"],
    ["shell_enabled", "not set"],
  ]);

  await open("Trust");
  assert.equal(content.children[0].textContent, "Trust");
  assert.deepEqual(rowCells(), [["/a", "agents, skills"], ["/z", "nothing"]]);

  browser.global.location.hash = "#/settings/unknown";
  browser.dispatch("hashchange");
  assert.equal(content.children[0].textContent, "General");
  assert.deepEqual(rowCells(), [["visible.setting", "on", "user"]]);
});

test("Memory settings loads project and global entries each time it opens", async () => {
  const browser = fakeGlobal({ fragment: "#/settings/agents" });
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    memory: {
      project: { entries: [{ id: "p1", text: "Keep reports concise.", written_at: "2026-10-13T00:00:00.000Z", agent_name: "leader" }] },
      global: { entries: [{ id: "g1", text: "Use metric units.", written_at: "2026-10-12T00:00:00.000Z", agent_name: "reviewer" }] },
    },
  });
  await start({ global: browser.global, document, client });
  const pane = document.getElementById("page").children[0];
  const sections = find(pane, (node) => node.className === "settings-sections");
  const content = find(pane, (node) => node.className === "settings-content");
  const memoryLink = sections.children.find((link) => link.textContent === "Memory");

  assert.deepEqual(sections.children.slice(5, 8).map((link) => link.textContent), ["Agents", "Memory", "Hooks"]);
  await memoryLink.dispatch("click");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(client.calls.memory, 1);
  assert.deepEqual(content.children.slice(0, 2).map((node) => node.textContent), [
    "Memory",
    "What the agent remembered from your conversations. It reads these at the start of each new conversation.",
  ]);
  const groups = content.children.filter((node) => node.className === "memory-group");
  assert.deepEqual(groups.map((group) => group.children[0].textContent), ["Project", "Global"]);
  assert.match(visibleText(groups[0]), /Keep reports concise\./);
  assert.match(visibleText(groups[0]), /leader/);
  assert.match(visibleText(groups[1]), /Use metric units\./);
  await sections.children.find((link) => link.textContent === "Hooks").dispatch("click");
  await memoryLink.dispatch("click");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(client.calls.memory, 2);
});

test("Memory settings explains unreadable groups without offering changes", async () => {
  const browser = fakeGlobal({ fragment: "#/settings/memory" });
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    memory: {
      project: { entries: [], error: "unreadable" },
      global: { entries: [{ id: "g1", text: "Use metric units.", agent_name: "leader" }] },
    },
  });
  await start({ global: browser.global, document, client });
  await new Promise((resolve) => setTimeout(resolve, 0));
  const content = find(document.getElementById("page"), (node) => node.className === "settings-content");
  const groups = content.children.filter((node) => node.className === "memory-group");
  const project = groups[0];
  const global = groups[1];

  assert.match(visibleText(project), /This memory file couldn't be read, so nothing in it was changed\./);
  assert.equal(walk(project).some((node) => ["Edit", "Delete"].includes(node.textContent)), false);
  assert.equal(walk(project).some((node) => node.textContent === "Nothing remembered yet."), false);
  assert.match(visibleText(global), /Use metric units\./);
  assert.equal(walk(global).some((node) => node.textContent === "Edit"), true);
  assert.equal(walk(global).some((node) => node.textContent === "Delete"), true);
});

test("Memory settings edits entries and keeps a rejected edit open", async () => {
  const browser = fakeGlobal({ fragment: "#/settings/memory" });
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    memory: {
      project: { entries: [{ id: "p1", text: "Original note", written_at: "2026-10-13T00:00:00.000Z", agent_name: "leader" }] },
      global: { entries: [] },
    },
  });
  const update = client.updateMemory.bind(client);
  let saves = 0;
  client.updateMemory = async (id, text) => {
    saves += 1;
    if (saves === 2) throw new Error("text is too long");
    return update(id, text);
  };
  await start({ global: browser.global, document, client });
  await new Promise((resolve) => setTimeout(resolve, 0));
  const content = find(document.getElementById("page"), (node) => node.className === "settings-content");
  let row = find(content, (node) => node.className === "memory-entry");
  await find(row, (node) => node.textContent === "Edit").dispatch("click");
  let textarea = find(content, (node) => node.className === "memory-edit-input");
  textarea.value = "Updated note";
  await find(content, (node) => node.textContent === "Save").dispatch("click");
  assert.match(visibleText(content), /Updated note/);
  assert.deepEqual(client.calls.updateMemory, [{ id: "p1", text: "Updated note" }]);

  row = find(content, (node) => node.className === "memory-entry");
  await find(row, (node) => node.textContent === "Edit").dispatch("click");
  textarea = find(content, (node) => node.className === "memory-edit-input");
  textarea.value = "Rejected note";
  await find(content, (node) => node.textContent === "Save").dispatch("click");
  assert.ok(find(content, (node) => node.className === "memory-edit-input"));
  assert.equal(find(content, (node) => node.className === "memory-edit-error").textContent, "text is too long");
});

test("Memory settings confirms deletion inline and cancellation preserves the entry", async () => {
  const browser = fakeGlobal({ fragment: "#/settings/memory" });
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    memory: {
      project: { entries: [{ id: "p1", text: "Delete carefully.", written_at: "2026-10-13T00:00:00.000Z", agent_name: "leader" }] },
      global: { entries: [] },
    },
  });
  await start({ global: browser.global, document, client });
  await new Promise((resolve) => setTimeout(resolve, 0));
  const content = find(document.getElementById("page"), (node) => node.className === "settings-content");
  let row = find(content, (node) => node.className === "memory-entry");
  await find(row, (node) => node.textContent === "Delete").dispatch("click");
  row = find(content, (node) => node.className === "memory-entry");
  assert.match(visibleText(row), /Delete this memory\?/);
  await find(row, (node) => node.textContent === "Cancel").dispatch("click");
  assert.match(visibleText(content), /Delete carefully\./);
  assert.deepEqual(client.calls.deleteMemory, []);

  row = find(content, (node) => node.className === "memory-entry");
  await find(row, (node) => node.textContent === "Delete").dispatch("click");
  row = find(content, (node) => node.className === "memory-entry");
  await find(row, (node) => node.textContent === "Delete").dispatch("click");
  assert.deepEqual(client.calls.deleteMemory, [{ id: "p1" }]);
  assert.match(visibleText(content), /Nothing remembered yet\./);
});

test("Memory settings shows empty and retry states", async () => {
  const browser = fakeGlobal({ fragment: "#/settings/memory" });
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), { memoryFailure: true });
  let attempts = 0;
  client.memory = async () => {
    attempts += 1;
    if (attempts === 1) throw new Error("offline");
    return { project: { entries: [] }, global: { entries: [] } };
  };
  await start({ global: browser.global, document, client });
  await new Promise((resolve) => setTimeout(resolve, 0));
  const content = find(document.getElementById("page"), (node) => node.className === "settings-content");
  assert.match(visibleText(content), /Couldn't load memory\./);
  await find(content, (node) => node.textContent === "Retry").dispatch("click");
  assert.equal(walk(content).filter((node) => node.className === "memory-empty").length, 2);
  assert.equal(walk(content).filter((node) => node.textContent === "Nothing remembered yet.").length, 2);
});

test("empty hooks and trust show messages while an empty ceiling keeps six rows", async () => {
  const browser = fakeGlobal({ fragment: "#/settings/hooks" });
  const document = new FakeDocument();
  await start({
    global: browser.global,
    document,
    client: fakeClient(fixtureRoadmap(), {
      settings: { settings: { hooks: [], trust: [], ceiling: {} } },
    }),
  });
  const pane = document.getElementById("page").children[0];
  const sections = find(pane, (value) => value.className === "settings-sections");
  const content = find(pane, (value) => value.className === "settings-content");
  const tables = () => walk(content).filter((value) => value.tagName === "TABLE");

  assert.match(visibleText(content), /No hooks are configured\./);
  assert.equal(tables().length, 0);
  await find(sections, (value) => value.textContent === "Trust").dispatch("click");
  assert.match(visibleText(content), /No directories are trusted\./);
  assert.equal(tables().length, 0);
  await find(sections, (value) => value.textContent === "Ceiling").dispatch("click");
  assert.equal(tables().length, 1);
  assert.equal(walk(content).filter((value) => value.className === "settings-row").length, 6);
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
  assert.equal(walk(document.getElementById("prompt-form")).some((node) => node.tagName === "SELECT"), false);
  document.getElementById("prompt").value = "wait for a key";
  await document.getElementById("prompt-form").dispatch("submit");
  assert.deepEqual(client.calls.prompt, []);
  assert.equal(document.getElementById("prompt").value, "wait for a key");
  input.value = secret;
  await find(controls, (value) => value.tagName === "BUTTON" && value.textContent === "Save").dispatch("click");
  assert.deepEqual(client.calls.credentials, [{ name: "OPENAI_API_KEY", value: secret }]);
  assert.equal(input.value, "");
  assert.equal(status.textContent, "present");
  assert.ok(!visibleText(content).includes(secret));

  await find(controls, (value) => value.tagName === "BUTTON" && value.textContent === "Remove").dispatch("click");
  assert.deepEqual(client.calls.credentials.at(-1), { name: "OPENAI_API_KEY", value: "" });
  assert.equal(status.textContent, "absent");
  assert.equal(walk(document.getElementById("prompt-form")).some((node) => node.tagName === "SELECT"), false);
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
    ["builder", "project", ".symphonai/agents/builder.toml", "", ""],
    ["reviewer", "user", "", "", ""],
  ]);

  await open("Inventory");
  assert.deepEqual(rowCells(), [
    ["project", ".symphonai/skills", "blocked", "repository not trusted"],
    ["user", "/users/example/plugins", "", "No reason recorded."],
  ]);
});

test("definition settings use the agent definition route", async () => {
  const source = await readFile(new URL("../src/client.js", import.meta.url), "utf8");
  const appSource = await readFile(new URL("../src/app.js", import.meta.url), "utf8");
  assert.match(source, /"\/agent"/);
  assert.match(appSource, /saveAgent/);
});

test("agent settings open, save, preserve refused edits, and create definitions", async () => {
  const browser = fakeGlobal({ fragment: "#/settings/agents" });
  const document = new FakeDocument();
  const settings = { settings: {
    agents: [{ name: "reviewer", path: ".symphonai/agents/reviewer.toml" }],
    withheld: [{ scope: "project", directory: ".symphonai/agents", names: ["blocked"], reason: "repository not trusted" }],
  } };
  const client = fakeClient(fixtureRoadmap(), { settings });
  client.agent = async () => ({
    name: "reviewer",
    scope: "project",
    text: 'prompt = "loaded prompt"\n\ntools = ["read_file"] # keep this comment\n'
      + 'deadline_seconds = 30\n'
      + '[model]\nprovider = "openai"\nmodel = "gpt-test"\neffort = "high"\n'
      + '[budget]\nmax_turns = 4\n[memory]\nenabled = true\n[policy]\nmode = "ask"\n',
  });
  const saved = [];
  client.saveAgent = async (name, scope, text) => {
    saved.push({ name, scope, text });
    return { written: true, message: "definition saved; it will take effect on the next run" };
  };
  await start({ global: browser.global, document, client });
  const content = find(document.getElementById("page"), (value) => value.className === "settings-content");
  const rows = walk(content).filter((value) => value.className === "settings-row");
  assert.equal(rows.length, 2);
  assert.equal(visibleText(rows[0]), "\nblocked\nproject\n.symphonai/agents/blocked.toml\nrepository not trusted\n\nWithheld");
  assert.equal(rows[0].children[4].children[0].disabled, true);

  await find(rows[1], (value) => value.tagName === "BUTTON").dispatch("click");
  const form = find(content, (value) => value.className === "agent-form");
  const controls = Object.fromEntries(
    walk(form).filter((value) => value.className?.startsWith("agent-")).map((value) => [value.className, value]),
  );
  assert.equal(controls["agent-prompt"].value, "loaded prompt");
  assert.equal(controls["agent-provider"].value, "openai");
  controls["agent-prompt"].value = "edited prompt";
  await form.dispatch("submit");
  assert.equal(saved.length, 1);
  assert.equal(saved[0].name, "reviewer");
  assert.equal(saved[0].scope, "project");
  assert.match(saved[0].text, /prompt = "edited prompt"/);
  assert.match(saved[0].text, /deadline_seconds = 30/);
  assert.match(saved[0].text, /tools = \["read_file"\] # keep this comment/);
  assert.match(saved[0].text, /\[budget\]\nmax_turns = 4/);
  assert.match(saved[0].text, /\[memory\]\nenabled = true/);
  assert.match(saved[0].text, /\[policy\]\nmode = "ask"/);
  assert.match(visibleText(form), /definition saved; it will take effect on the next run/);

  client.saveAgent = async () => { throw new Error("/tmp/reviewer.toml: model: bad provider"); };
  controls["agent-model"].value = "kept-model";
  await form.dispatch("submit");
  assert.equal(controls["agent-model"].value, "kept-model");
  assert.match(visibleText(form), /\/tmp\/reviewer\.toml: model: bad provider/);

  await find(content, (value) => value.tagName === "BUTTON" && value.textContent === "New agent").dispatch("click");
  const newForm = find(content, (value) => value.className === "agent-form");
  const newControls = Object.fromEntries(
    walk(newForm).filter((value) => value.className?.startsWith("agent-")).map((value) => [value.className, value]),
  );
  newControls["agent-name"].value = "new-agent";
  newControls["agent-provider"].value = "anthropic";
  newControls["agent-prompt"].value = "new prompt";
  client.saveAgent = async (name, scope, text) => {
    saved.push({ name, scope, text });
    return { written: true, message: "definition saved; it will take effect on the next run" };
  };
  await newForm.dispatch("submit");
  assert.equal(saved.at(-1).name, "new-agent");
  assert.equal(saved.at(-1).scope, "project");
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
    document.getElementById("sidebar"),
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

  await find(document.getElementById("sidebar"), (value) => value.textContent === "Settings").dispatch("click");
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

  assert.equal(toggle.textContent, "");
  assert.equal(toggle.attributes.get("aria-label"), "Hide sidebar");
  assert.equal(toggle.attributes.get("title"), "Hide sidebar");
  await toggle.dispatch("click");
  assert.equal(shell.className, "app-shell sidebar-folded");
  assert.equal(toggle.textContent, "");
  assert.equal(toggle.attributes.get("aria-label"), "Show sidebar");
  assert.equal(toggle.attributes.get("title"), "Show sidebar");
  assert.equal(page.children[0], currentPane);

  await toggle.dispatch("click");
  assert.equal(shell.className, "app-shell");
  assert.equal(toggle.textContent, "");
  assert.equal(toggle.attributes.get("aria-label"), "Hide sidebar");
  assert.equal(toggle.attributes.get("title"), "Hide sidebar");
  assert.equal(page.children[0], currentPane);
});

test("status and sidebar folds are independent", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient() });
  const shell = document.getElementById("app-shell");
  const railToggle = document.getElementById("rail-toggle");
  const sidebarToggle = document.getElementById("sidebar-toggle");

  assert.equal(railToggle.textContent, "");
  assert.equal(railToggle.attributes.get("aria-label"), "Hide status");
  assert.equal(railToggle.attributes.get("title"), "Hide status");
  assert.equal(sidebarToggle.textContent, "");
  assert.equal(sidebarToggle.attributes.get("aria-label"), "Hide sidebar");
  assert.equal(sidebarToggle.attributes.get("title"), "Hide sidebar");
  await railToggle.dispatch("click");
  assert.equal(shell.className, "app-shell rail-folded");
  assert.equal(railToggle.textContent, "");
  assert.equal(railToggle.attributes.get("aria-label"), "Show status");
  assert.equal(railToggle.attributes.get("title"), "Show status");
  await sidebarToggle.dispatch("click");
  assert.equal(shell.className, "app-shell rail-folded sidebar-folded");
  await railToggle.dispatch("click");
  assert.equal(shell.className, "app-shell sidebar-folded");
  assert.equal(railToggle.textContent, "");
  assert.equal(railToggle.attributes.get("aria-label"), "Hide status");
  assert.equal(railToggle.attributes.get("title"), "Hide status");
});

test("action icons use current-color SVG masks and hidden buttons stay hidden", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const appSource = await readFile(new URL("../src/app.js", import.meta.url), "utf8");
  const icons = [
    "plus", "pause", "play", "check", "trash",
    "undo", "left-panel", "right-panel", "settings",
  ];

  for (const icon of icons) {
    assert.match(css, new RegExp(`\\.icon-${icon}\\s*\\{[^}]*--icon-image:\\s*url\\("data:image/svg\\+xml,`));
  }
  const maskRule = css.match(/\.icon-plus::before,[\s\S]*?\.icon-settings::before\s*\{([^}]*)\}/)?.[1] ?? "";
  assert.match(maskRule, /background-color:\s*currentColor;/);
  assert.match(maskRule, /mask:\s*var\(--icon-image\) center \/ 1rem 1rem no-repeat;/);

  for (const [id, icon] of [
    ["new-chat", "plus"],
    ["sidebar-toggle", "left-panel"], ["rail-toggle", "right-panel"],
  ]) {
    assert.match(html, new RegExp(`<button[^>]*id="${id}"[^>]*class="[^"]*icon-${icon}[^"]*"`));
  }
  assert.doesNotMatch(appSource, /controlAgent|agent-controls|agent-redirect|agent-control-error/);
  assert.match(css, /\[hidden\]\s*\{[^}]*display:\s*none\s*!important;/);
  assert.match(css, /\.sidebar-toggle,\s*\.rail-toggle\s*\{[^}]*width:\s*2rem;[^}]*height:\s*2rem;/);
});

test("items in folded and open roadmap phases open their specs and reports", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const app = await start({ global: {}, document, client });
  const calls = [];
  const open = app.specView.open;
  app.specView.open = (...arguments_) => {
    calls.push(arguments_);
    return open(...arguments_);
  };
  const roadmap = document.getElementById("roadmap");
  for (const [title, specText, reportText] of [
    ["Boundary", "base spec text", "base report text"],
    ["Open boundary", "open spec text", "open report text"],
  ]) {
    if (title === "Boundary") {
      const completed = find(roadmap, (node) => node.attributes.get("data-phase-id") === "17");
      await find(completed, (node) => node.className === "roadmap-phase-header").dispatch("click");
    }
    const button = find(
      roadmap,
      (value) => value.tagName === "BUTTON" && value.textContent.includes(title),
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

  const roadmap = document.getElementById("roadmap");
  const chain = find(roadmap, (node) => node.className.startsWith("roadmap-phase-chain-summary"));
  assert.ok(chain, visibleText(roadmap));
  assert.equal(find(roadmap, (node) => node.className === "roadmap-phase-tasks"), undefined);
});

test("done phase chains expand to one-line phases and then to a task graph", async () => {
  const roadmap = {
    goal: "Collapse completed phases",
    phases: [
      { id: "01", name: "First", status: "done", items: [{ title: "First task", spec: "specs/01/01a-first.md" }] },
      { id: "02", name: "Second", status: "done", items: [{ title: "Second task", spec: "specs/02/02a-second.md" }] },
      { id: "03", name: "Third", status: "done", items: [{ title: "Third task", spec: "specs/03/03a-third.md" }] },
      { id: "04", name: "Open phase", status: "next", items: [{ title: "Open task", spec: "specs/04/04a-open.md" }] },
    ],
  };
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient(JSON.stringify(roadmap)) });
  let page = document.getElementById("roadmap");
  const chain = find(page, (node) => node.className.startsWith("roadmap-phase-chain-summary"));
  assert.equal(chain?.textContent, "✓ 01–03 · 3 phases", visibleText(page));
  await chain.dispatch("click");
  page = document.getElementById("roadmap");
  for (const id of ["01", "02", "03"]) {
    assert.ok(find(page, (node) => node.attributes.get("data-phase-id") === id));
  }
  const first = find(page, (node) => node.attributes.get("data-phase-id") === "01");
  assert.equal(first.children.length, 1);
  await find(first, (node) => node.className === "roadmap-phase-header").dispatch("click");
  const expandedFirst = find(document.getElementById("roadmap"), (node) => node.attributes.get("data-phase-id") === "01");
  assert.ok(find(expandedFirst, (node) => node.className === "roadmap-phase-tasks"));
});

test("the removed roadmap route falls back to chat while the rail keeps its graph", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal();
  const app = await start({ global: browser.global, document, client: fakeClient() });
  browser.global.location.hash = "#/roadmap";
  browser.dispatch("hashchange");
  assert.equal(document.getElementById("page").children[0].className, "chat-pane");
  assert.equal(app.route().page, "chat");
  const phase = find(document.getElementById("roadmap"), (node) => node.attributes.get("data-phase-id") === "18");
  assert.ok(find(phase, (node) => node.className === "roadmap-phase-tasks"));
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

  assert.equal(
    walk(document.getElementById("roadmap")).filter((node) => node.className.startsWith("roadmap-phase-box")).length,
    roadmap.phases.length,
  );
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

test("Escape stops a running turn from the composer", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const app = await start({ global: {}, document, client });
  document.getElementById("prompt").value = "hello";
  const submitting = document.getElementById("prompt-form").dispatch("submit");
  assert.equal(app.turn.state, DISPATCHING);
  client.resolvePrompt({ accepted: true, run_id: "run-stop-key" });
  await submitting;
  assert.equal(app.turn.state, RUNNING);
  await document.getElementById("prompt").dispatch("keydown", { key: "Escape" });
  assert.deepEqual(client.calls.stop, [""]);
  await client.emit(eventFrame("RunStarted", { run_id: "run-stop-key" }));
  await client.emit(eventFrame("RunFinished", { run_id: "run-stop-key" }));
  assert.equal(app.turn.state, IDLE);
});

test("Mod+. stops a running turn from Settings and is a no-op while idle", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal({ fragment: "#/settings" });
  const client = fakeClient();
  const app = await start({ global: browser.global, document, client });
  assert.equal(app.route().page, "settings");

  document.getElementById("prompt").value = "hello";
  const submitting = document.getElementById("prompt-form").dispatch("submit");
  client.resolvePrompt({ accepted: true, run_id: "run-stop-key" });
  await submitting;
  let prevented = 0;
  await browser.dispatch("keydown", {
    key: ".",
    ctrlKey: true,
    preventDefault() { prevented += 1; },
  });
  assert.deepEqual(client.calls.stop, [""]);
  assert.equal(prevented, 1);

  await client.emit(eventFrame("RunStarted", { run_id: "run-stop-key" }));
  await client.emit(eventFrame("RunFinished", { run_id: "run-stop-key" }));
  await browser.dispatch("keydown", {
    key: ".",
    ctrlKey: true,
    preventDefault() { prevented += 1; },
  });
  assert.deepEqual(client.calls.stop, [""]);
  assert.equal(prevented, 2);
});

test("Mod+. cancels a dispatching prompt and stops it when accepted", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal();
  const client = fakeClient();
  const app = await start({ global: browser.global, document, client });
  document.getElementById("prompt").value = "hello";
  const submitting = document.getElementById("prompt-form").dispatch("submit");
  assert.equal(app.turn.state, DISPATCHING);
  await browser.dispatch("keydown", {
    key: ".",
    ctrlKey: true,
    preventDefault() {},
  });
  assert.deepEqual(client.calls.stop, []);

  client.resolvePrompt({ accepted: true, run_id: "run-cancel-dispatch" });
  await submitting;
  assert.deepEqual(client.calls.stop, [""]);
  await client.emit(eventFrame("RunStarted", { run_id: "run-cancel-dispatch" }));
  await client.emit(eventFrame("RunFinished", { run_id: "run-cancel-dispatch" }));
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
  await client.emit(toolStarted("call-1", "read_file", "README.md"));
  await client.emit(toolFinished("call-1", "read_file"));
  await client.emit(toolStarted("call-2", "grep", "needle"));
  await client.emit(toolFinished("call-2", "grep"));

  const chat = document.getElementById("chat");
  assert.equal(chat.children.length, 1);
  assert.ok(chat.children[0].className.split(" ").includes("activity"));
  const activity = chat.children[0];
  assert.deepEqual(walk(activity).filter((node) => node.tagName === "CODE").map((node) => node.textContent), [
    "README.md", "needle",
  ]);
  assert.equal(textInDocumentOrder(activity), "Read README.md and searched for needle.");
  assert.doesNotMatch(textInDocumentOrder(activity), /`/);
});

test("late extractor memory activity follows the finished turn as quoted prose", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await client.emit(eventFrame("RunStarted", {
    agent_id: "leader", run_id: "leader-run", agent_name: "leader",
  }));
  await client.emit(eventFrame("PromptSubmitted", {
    agent_id: "leader", run_id: "leader-run", text: "remember Run checks.",
  }));
  await client.emit(eventFrame("RunFinished", {
    agent_id: "leader", run_id: "leader-run", stopped_reason: "final_response",
  }));
  await client.emit(eventFrame("ToolCallStarted", {
    agent_id: "leader", run_id: "memory_run_1", tool_call_id: "remember-1",
    tool_name: "remember", target: "Run checks.",
  }));
  await client.emit(eventFrame("ToolCallFinished", {
    agent_id: "leader", run_id: "memory_run_1", tool_call_id: "remember-1",
    tool_name: "remember", ok: true,
  }));
  const chat = document.getElementById("chat");
  const line = chat.children.at(-1);
  assert.equal(chat.children.at(-2).className, "turn-time");
  assert.equal(textInDocumentOrder(line), 'Remembered "Run checks."');
  assert.ok(line.className.includes("activity-memory"));
  assert.deepEqual(walk(line).filter((node) => node.tagName === "CODE"), []);
});

test("forget activity reads as a successful memory removal", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await client.emit(eventFrame("TurnStarted", { index: 1 }));
  await client.emit(toolStarted("forget-1", "forget", "Prefer concise answers"));
  await client.emit(toolFinished("forget-1", "forget"));
  const line = document.getElementById("chat").children[0];
  assert.equal(textInDocumentOrder(line), 'Forgot "Prefer concise answers"');
  assert.ok(line.className.includes("activity-memory"));
  assert.deepEqual(walk(line).filter((node) => node.tagName === "CODE"), []);
});

test("activity targets render as code while failures and missing targets keep their wording", async () => {
  const failedDocument = new FakeDocument();
  const failedClient = fakeClient();
  await start({ global: {}, document: failedDocument, client: failedClient });
  await failedClient.emit(eventFrame("TurnStarted", { index: 1 }));
  await failedClient.emit(toolStarted("failed-call", "run_shell", "x"));
  await failedClient.emit(eventFrame("ToolCallFailed", {
    tool_call_id: "failed-call",
    error: "boom",
  }));
  const failedLine = failedDocument.getElementById("chat").children[0];
  assert.equal(failedLine.className, "activity activity-shell failed");
  assert.equal(failedLine.attributes.get("title"), "Ran x (failed: boom).");
  assert.equal(textInDocumentOrder(failedLine), "Ran x (failed: boom).");
  assert.deepEqual(walk(failedLine).filter((node) => node.tagName === "CODE").map((node) => node.textContent), ["x"]);

  const untargetedDocument = new FakeDocument();
  const untargetedClient = fakeClient();
  await start({ global: {}, document: untargetedDocument, client: untargetedClient });
  await untargetedClient.emit(eventFrame("TurnStarted", { index: 1 }));
  await untargetedClient.emit(toolStarted("untargeted-call", "run_shell", null));
  await untargetedClient.emit(toolFinished("untargeted-call", "run_shell"));
  const untargetedLine = untargetedDocument.getElementById("chat").children[0];
  assert.equal(textInDocumentOrder(untargetedLine), "Ran.");
  assert.deepEqual(walk(untargetedLine).filter((node) => node.tagName === "CODE"), []);
});

test("activity lines use their first tool kind and keep the full sentence in the title", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await client.emit(eventFrame("TurnStarted", { index: 1 }));
  await client.emit(toolStarted("read", "read_file", "src/app.js"));
  await client.emit(toolFinished("read", "read_file"));
  await client.emit(toolStarted("unknown", "custom_tool", "a long target"));
  await client.emit(toolFinished("unknown", "custom_tool"));

  const [readLine] = document.getElementById("chat").children;
  assert.equal(readLine.className, "activity activity-read");
  assert.equal(readLine.children.length, 1);
  assert.equal(readLine.children[0].className, "activity-text");
  assert.equal(
    readLine.attributes.get("title"),
    "Read src/app.js and used custom_tool a long target.",
  );

  const unknownDocument = new FakeDocument();
  const unknownClient = fakeClient();
  await start({ global: {}, document: unknownDocument, client: unknownClient });
  await unknownClient.emit(eventFrame("TurnStarted", { index: 1 }));
  await unknownClient.emit(toolStarted("unknown", "custom_tool", "a long target"));
  await unknownClient.emit(toolFinished("unknown", "custom_tool"));
  const unknownLine = unknownDocument.getElementById("chat").children[0];
  assert.equal(unknownLine.className, "activity");
  assert.equal(unknownLine.attributes.get("title"), "Used custom_tool a long target.");
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

test("shell approvals show and answer the remember button only when offered", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await client.emit({
    kind: "approval_requested",
    payload: {
      approval_id: "shell-approval",
      operation: "run_shell",
      target: "pytest -x",
      details: "run tests",
      remember: "pytest",
    },
  });

  const buttons = walk(document.getElementById("approvals"))
    .filter((node) => node.tagName === "BUTTON");
  assert.deepEqual(buttons.map((button) => button.textContent), [
    "Allow",
    "Allow, and don't ask again for pytest in this chat",
    "Deny",
  ]);
  await buttons[1].dispatch("click");
  assert.deepEqual(client.calls.approve, [{
    id: "shell-approval",
    allowed: true,
    reason: "",
    remember: true,
  }]);

  await client.emit({
    kind: "approval_requested",
    payload: {
      approval_id: "ordinary-approval",
      operation: "write_file",
      target: "note.txt",
      details: "write note",
    },
  });
  const ordinaryButtons = walk(document.getElementById("approvals"))
    .filter((node) => node.tagName === "BUTTON");
  assert.deepEqual(ordinaryButtons.slice(-2).map((button) => button.textContent), ["Allow", "Deny"]);
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
    ["assistant", "activity activity-read", "assistant"],
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

test("assistant text after a command is rendered as its own entry", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });

  await submitCommand(document, "/help");
  await client.emit(eventFrame("AssistantTextDelta", { text: "Agent reply" }));

  const chat = document.getElementById("chat");
  assert.deepEqual(chat.children.map((entry) => entry.className), ["command", "assistant"]);
  assert.equal(latestCommandOutput(document).length, 1);
  assert.equal(latestCommandEntry(document).children[0].textContent, "/help");
  assert.equal(chat.children[1].textContent, "Agent reply");
});

test("assistant replies render Markdown blocks and inline formatting in order", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  const reply = [
    "# Heading",
    "",
    "- first",
    "- second",
    "",
    "1. alpha",
    "2. beta",
    "",
    "```python",
    "print('hello')",
    "```",
    "",
    "> quoted text",
    "",
    "Paragraph with `code`, **strong**, *em* and [link](https://example.com/path).",
  ].join("\n");
  await client.emit(eventFrame("AssistantTextDelta", { text: reply }));

  const assistant = document.getElementById("chat").children[0];
  assert.equal(assistant.className, "assistant");
  assert.deepEqual(assistant.children.map((node) => node.tagName), ["H1", "UL", "OL", "PRE", "BLOCKQUOTE", "P"]);
  assert.deepEqual(assistant.children[1].children.map((item) => item.tagName), ["LI", "LI"]);
  assert.deepEqual(assistant.children[2].children.map((item) => item.tagName), ["LI", "LI"]);
  const code = assistant.children[3].children[0];
  assert.equal(code.tagName, "CODE");
  assert.equal(code.textContent, "print('hello')");
  assert.equal(code.attributes.get("data-language"), "python");
  assert.equal(assistant.children[4].tagName, "BLOCKQUOTE");
  const paragraph = assistant.children[5];
  assert.deepEqual(walk(paragraph).filter((node) => ["CODE", "STRONG", "EM", "A"].includes(node.tagName)).map((node) => node.tagName), [
    "CODE", "STRONG", "EM", "A",
  ]);
  const link = find(paragraph, (node) => node.tagName === "A");
  assert.equal(link.textContent, "link");
  assert.equal(link.attributes.get("href"), "https://example.com/path");
  assert.equal(link.attributes.get("rel"), "noopener noreferrer");
  assert.equal(link.attributes.get("target"), "_blank");
});

test("assistant Markdown keeps HTML and unsafe links as text", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await client.emit(eventFrame("AssistantTextDelta", {
    text: "<script>alert(1)</script>\n\n[x](javascript:alert(1))",
  }));

  const assistant = document.getElementById("chat").children[0];
  assert.deepEqual(assistant.children.map((node) => node.tagName), ["P", "P"]);
  assert.equal(assistant.children[0].textContent, "<script>alert(1)</script>");
  assert.equal(assistant.children[1].textContent, "x");
  assert.equal(walk(assistant).some((node) => node.tagName === "SCRIPT" || node.tagName === "A"), false);
});

test("assistant Markdown renders plain lines and unclosed fences while streaming", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await client.emit(eventFrame("AssistantTextDelta", { text: "A plain reply." }));
  let assistant = document.getElementById("chat").children[0];
  assert.deepEqual(assistant.children.map((node) => node.tagName), ["P"]);
  assert.equal(assistant.children[0].textContent, "A plain reply.");

  const partial = "```python\nprint('still streaming')";
  await client.emit(eventFrame("AssistantTextDelta", { text: `\n\n${partial}` }));
  assistant = document.getElementById("chat").children[0];
  assert.deepEqual(assistant.children.map((node) => node.tagName), ["P", "PRE"]);
  assert.equal(assistant.children[1].children[0].textContent, "print('still streaming')");
  assert.equal(assistant.children[1].children[0].attributes.get("data-language"), "python");

  const streamed = [
    "# Heading", "", "- first", "- second", "", "1. alpha", "2. beta", "", "```python",
    "print('hello')", "```", "", "> quoted text", "", "Paragraph with `code`, **strong**, *em* and [link](https://example.com/path).",
  ].join("\n");
  const streamDocument = new FakeDocument();
  const streamClient = fakeClient();
  await start({ global: {}, document: streamDocument, client: streamClient });
  for (let index = 0; index < streamed.length; index += 1) {
    await streamClient.emit(eventFrame("AssistantTextDelta", { text: streamed[index] }));
  }
  assert.equal(streamDocument.getElementById("chat").children[0].children.length, 6);
});

test("Markdown parses nested lists, ordered starts, and wrapped list text", () => {
  assert.deepEqual(parseMarkdown("- a\n  - b\n  - c\n- d"), [{
    type: "list",
    ordered: false,
    start: null,
    items: [
      {
        children: [{ type: "text", text: "a" }],
        blocks: [{
          type: "list",
          ordered: false,
          start: null,
          items: [
            { children: [{ type: "text", text: "b" }], blocks: [] },
            { children: [{ type: "text", text: "c" }], blocks: [] },
          ],
        }],
      },
      { children: [{ type: "text", text: "d" }], blocks: [] },
    ],
  }]);

  const consecutive = parseMarkdown("1. one\n\n2. two\n\n3. three")[0];
  assert.equal(consecutive.type, "list");
  assert.equal(consecutive.ordered, true);
  assert.equal(consecutive.start, 1);
  assert.deepEqual(consecutive.items.map((item) => item.children[0].text), ["one", "two", "three"]);
  const later = parseMarkdown("3. three\n4. four")[0];
  assert.equal(later.start, 3);
  assert.deepEqual(later.items.map((item) => item.children[0].text), ["three", "four"]);

  assert.deepEqual(parseMarkdown("- item\n  continues here")[0].items[0].children, [
    { type: "text", text: "item" },
    { type: "break" },
    { type: "text", text: "continues here" },
  ]);
});

test("Markdown list items render fenced code blocks and parse list edge cases", async () => {
  const input = "1. Install:\n   ```bash\n   pip install -e .\n   ```\n2. Run tests";
  const parsed = parseMarkdown(input);
  assert.equal(parsed.length, 1);
  assert.equal(parsed[0].type, "list");
  assert.deepEqual(parsed[0].items[0].children, [{ type: "text", text: "Install:" }]);
  assert.deepEqual(parsed[0].items[0].blocks, [{
    type: "code-block",
    language: "bash",
    text: "pip install -e .",
  }]);
  assert.deepEqual(parsed[0].items.map((item) => item.children[0].text), ["Install:", "Run tests"]);

  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await client.emit(eventFrame("AssistantTextDelta", { text: input }));
  const assistant = document.getElementById("chat").children[0];
  const firstItem = assistant.children[0].children[0];
  const pre = find(firstItem, (node) => node.tagName === "PRE");
  assert.ok(pre);
  assert.equal(pre.children[0].tagName, "CODE");
  assert.equal(firstItem.textContent.includes("`"), false);

  const unclosed = parseMarkdown("- a\n  ```\n  x")[0].items[0];
  assert.deepEqual(unclosed.blocks, [{ type: "code-block", language: null, text: "x" }]);

  assert.equal(parseMarkdown("| a | b |\n|-|-|\n| 1 | 2 |")[0].type, "table");
  assert.deepEqual(parseMarkdown("| a |\n|:-:|\n| 1 |")[0].align, ["center"]);

  const indented = parseMarkdown("  - starts indented\n  - second");
  assert.equal(indented.length, 1);
  assert.equal(indented[0].type, "list");
  assert.deepEqual(indented[0].items.map((item) => item.children[0].text), ["starts indented", "second"]);
  assert.equal(parseMarkdown("    - four spaces")[0].type, "paragraph");
});

test("Markdown list code blocks keep blank lines and later item text", async () => {
  const codeInput = "1. Write the file:\n   ```python\n   def a():\n       return 1\n\n   def b():\n       return 2\n   ```\n2. Run it";
  const parsedCode = parseMarkdown(codeInput);
  assert.equal(parsedCode.length, 1);
  assert.equal(parsedCode[0].items.length, 2);
  assert.equal(parsedCode[0].items[0].blocks[0].text, "def a():\n    return 1\n\ndef b():\n    return 2");
  assert.equal(parsedCode[0].items[1].children[0].text, "Run it");

  const continuationInput = "- Run:\n  ```\n  make\n  ```\n  then check the output\n- next";
  const parsedContinuation = parseMarkdown(continuationInput)[0];
  assert.deepEqual(parsedContinuation.items[0].children, [{ type: "text", text: "Run:" }]);
  assert.deepEqual(parsedContinuation.items[0].blocks, [
    { type: "code-block", language: null, text: "make" },
    { type: "paragraph", children: [{ type: "text", text: "then check the output" }] },
  ]);

  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await client.emit(eventFrame("AssistantTextDelta", { text: continuationInput }));
  const item = document.getElementById("chat").children[0].children[0].children[0];
  assert.equal(item.textContent, "Run:");
  assert.deepEqual(item.children.map((node) => node.tagName), ["PRE", "P"]);
  assert.equal(item.children[0].children[0].textContent, "make");
  assert.equal(item.children[1].textContent, "then check the output");

  const underIndented = parseMarkdown("- a\n  ```\nx\n  ```")[0].items[0];
  assert.deepEqual(underIndented.blocks, [{ type: "code-block", language: null, text: "x" }]);
});

test("Markdown tables, separators, and incomplete streaming blocks parse safely", async () => {
  const tableInput = "| a | b |\n|:--|--:|\n| 1 | 2 |";
  assert.deepEqual(parseMarkdown(tableInput), [{
    type: "table",
    align: ["left", "right"],
    header: [[{ type: "text", text: "a" }], [{ type: "text", text: "b" }]],
    rows: [[[{ type: "text", text: "1" }], [{ type: "text", text: "2" }]]],
  }]);
  assert.deepEqual(parseMarkdown("| a | b |\n| 1 | 2 |")[0].type, "paragraph");
  assert.deepEqual(parseMarkdown("---"), [{ type: "rule" }]);
  assert.equal(parseMarkdown("- x")[0].type, "list");
  for (const partial of ["| a | b |", "| a | b |\n|--", "- a\n  "]) {
    assert.doesNotThrow(() => parseMarkdown(partial), partial);
  }

  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  await client.emit(eventFrame("AssistantTextDelta", { text: "3. three\n4. four\n\n" + tableInput + "\n\n---" }));
  const assistant = document.getElementById("chat").children[0];
  const ordered = assistant.children[0];
  assert.equal(ordered.tagName, "OL");
  assert.equal(ordered.attributes.get("start"), "3");
  const wrapper = assistant.children[1];
  assert.equal(wrapper.className, "table-scroll");
  const table = wrapper.children[0];
  assert.deepEqual(table.children.map((part) => part.tagName), ["THEAD", "TBODY"]);
  const header = table.children[0].children[0].children;
  const cells = table.children[1].children[0].children;
  assert.deepEqual(header.map((cell) => cell.tagName), ["TH", "TH"]);
  assert.deepEqual(cells.map((cell) => cell.tagName), ["TD", "TD"]);
  assert.deepEqual(header.map((cell) => cell.textContent), ["a", "b"]);
  assert.deepEqual(cells.map((cell) => cell.textContent), ["1", "2"]);
  assert.deepEqual(header.map((cell) => cell.attributes.get("style")), ["text-align: left;", "text-align: right;"]);
  assert.deepEqual(cells.map((cell) => cell.attributes.get("style")), ["text-align: left;", "text-align: right;"]);
  assert.equal(assistant.children[2].tagName, "HR");

  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  assert.match(css, /\.assistant \.table-scroll\s*\{[^}]*overflow-x:\s*auto;/);
  assert.match(css, /\.assistant th,\s*\.assistant td\s*\{[^}]*padding:\s*var\(--space-1\) var\(--space-2\);[^}]*border:\s*1px solid var\(--ui-border\);/);
  assert.match(css, /\.assistant th\s*\{[^}]*font-size:\s*0\.8125rem;[^}]*font-weight:\s*600;/);
  assert.match(css, /\.assistant hr\s*\{[^}]*border-top:\s*1px solid var\(--ui-border\);/);
  assert.match(css, /\.assistant li > ul,\s*\.assistant li > ol\s*\{[^}]*margin:\s*0;/);
});

test("underscores inside words stay literal while standalone emphasis still renders", () => {
  for (const text of [
    "Call standard_tool_registry with skills_dir set.",
    `Edit symphonai${String.fromCharCode(95)}api/tools/shell_classify.py now.`,
    "MAX_ATTACHMENT_BYTES",
    "__init__",
    "a__b",
    "a_b_c",
    "漢字_名前",
    "école_test",
  ]) {
    assert.deepEqual(parseMarkdown(text), [{
      type: "paragraph",
      children: [{ type: "text", text }],
    }]);
  }

  for (const [text, before, emphasized, after] of [
    ["this is _important_ now", "this is ", "important", " now"],
    ["_important_ now", "", "important", " now"],
    ["this is _important_", "this is ", "important", ""],
    ["(_important_)", "(", "important", ")"],
    ["_漢字_", "", "漢字", ""],
  ]) {
    assert.deepEqual(parseMarkdown(text)[0].children, [
      ...(before ? [{ type: "text", text: before }] : []),
      { type: "em", children: [{ type: "text", text: emphasized }] },
      ...(after ? [{ type: "text", text: after }] : []),
    ]);
  }
});

test("asterisk emphasis, strong, code, and links keep their parsing", () => {
  assert.deepEqual(parseMarkdown("**strong** `code` *em* [link](https://example.com)")[0].children, [
    { type: "strong", children: [{ type: "text", text: "strong" }] },
    { type: "text", text: " " },
    { type: "code", text: "code" },
    { type: "text", text: " " },
    { type: "em", children: [{ type: "text", text: "em" }] },
    { type: "text", text: " " },
    { type: "link", label: "link", url: "https://example.com" },
  ]);
});

test("session history renders as the original conversation", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });

  await client.emit({ kind: "event", payload: {
    type: "HistoryMessage", role: "user", text: "What changed?", tool_calls: [], turn_id: "turn-1",
  } });
  await client.emit({ kind: "event", payload: {
    type: "HistoryMessage", role: "assistant", text: "The config changed.", tool_calls: [], turn_id: "turn-1",
  } });

  const chat = document.getElementById("chat");
  assert.deepEqual(chat.children.map((child) => [child.className, child.textContent]), [
    ["prompt", "What changed?"],
    ["assistant", "The config changed."],
  ]);
  assert.doesNotMatch(visibleText(chat), /Received HistoryMessage\./);
});

test("reopened edit cards keep their rows and reconcile Undo from changes", async () => {
  async function replay({ turns, undone = false }) {
    const document = new FakeDocument();
    const client = fakeClient();
    client.changes = async () => ({ files: [], turns, worktrees: [] });
    await start({ global: {}, document, client });
    await client.emit({ kind: "event", payload: {
      type: "HistoryMessage", role: "user", text: "Edit two files", tool_calls: [],
      checkpoint_key: "checkpoint-1",
      edits: [
        { path: "a.py", added: 2, removed: 1 },
        { path: "b.py", added: 1, removed: 3 },
      ],
      undone,
    } });
    await client.emit({ kind: "event", payload: {
      type: "HistoryMessage", role: "assistant", text: "Done", tool_calls: [],
    } });
    const card = find(document.getElementById("chat"), (node) => node.className === "edit-card");
    return { card, document };
  }

  const available = await replay({ turns: [{ key: "checkpoint-1", paths: ["a.py", "b.py"] }] });
  assert.ok(available.card);
  assert.equal(available.card.children.find((node) => node.className === "edit-card-files").children.length, 2);
  assert.ok(find(available.card, (node) => node.tagName === "BUTTON" && node.textContent === "Undo"));

  const noPaths = await replay({ turns: [{ key: "checkpoint-1", paths: [] }] });
  assert.ok(noPaths.card);
  assert.equal(find(noPaths.card, (node) => node.tagName === "BUTTON" && node.textContent === "Undo"), undefined);

  const alreadyUndone = await replay({
    turns: [{ key: "checkpoint-1", paths: ["a.py", "b.py"] }],
    undone: true,
  });
  assert.ok(alreadyUndone.card);
  assert.equal(find(alreadyUndone.card, (node) => node.tagName === "BUTTON" && node.textContent === "Undo"), undefined);
  assert.match(visibleText(alreadyUndone.card), /Undone/);
});

test("replayed messages with record ids do not render fork controls", async () => {
  const document = new FakeDocument();
  const source = { run_id: "source", title: "Original", repo_root: "/work/current" };
  const client = fakeClient(fixtureRoadmap(), { sessions: [source] });
  await start({ global: {}, document, client });
  const chat = document.getElementById("chat");
  const sourceButton = find(document.getElementById("sidebar"), (value) =>
    value.className === "session-link" && value.textContent === "Original"
  );
  await sourceButton.dispatch("click");
  await client.emit({ kind: "event", payload: {
    type: "HistoryMessage", role: "user", text: "What changed?", record_id: "rec-user",
    tool_calls: [], turn_id: "turn-1",
  } });
  await client.emit({ kind: "event", payload: {
    type: "HistoryMessage", role: "assistant", text: "The config changed.", record_id: "rec-answer",
    tool_calls: [], turn_id: "turn-1",
  } });
  assert.equal(find(chat, (value) => value.className === "fork-message"), undefined);
  assert.match(visibleText(chat), /What changed\?/);
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

test("transcript prose, activity, approvals, and turns use their assigned surfaces", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const declarations = (selector) => [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
    .filter(([, selectors]) => selectors.split(",").map((value) => value.trim()).includes(selector))
    .map(([, , rules]) => rules)
    .join("\n");
  const prompt = declarations(".prompt");
  const approval = declarations(".approval");

  assert.match(prompt, /background:\s*var\(--ui-inset\);/);
  assert.match(prompt, /border:\s*1px solid var\(--ui-border\);/);
  assert.match(prompt, /border-radius:\s*var\(--radius-control\);/);
  assert.match(prompt, /padding:\s*var\(--space-2\) var\(--space-3\);/);
  assert.match(prompt, /align-self:\s*stretch;/);
  assert.match(approval, /background:\s*var\(--ui-panel\);/);
  assert.match(approval, /border:\s*1px solid var\(--ui-control-border\);/);
  assert.match(approval, /border-radius:\s*var\(--radius-control\);/);
  assert.match(approval, /padding:\s*var\(--space-3\);/);
  for (const className of ["activity", "edit", "question", "compaction", "gap", "unknown-event"]) {
    const rules = declarations(`.${className}`);
    assert.match(rules, /color:\s*var\(--ui-text-secondary\);/);
    assert.match(rules, /font-size:\s*0\.8125rem;/);
  }
  assert.match(declarations(".gap"), /color:\s*var\(--ui-warning\);/);
  assert.match(declarations(".activity code"), /font-family:\s*var\(--font-code\);/);
  assert.match(declarations(".activity code"), /background:\s*none;/);
  const activityRules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
    .filter(([, selectors]) => selectors.split(",").some((selector) => /(?:^|[\s>])\.activity(?:[.#:[\s]|$)/.test(selector.trim())));
  assert.ok(activityRules.length > 0);
  assert.ok(activityRules.every(([, , rules]) =>
    [...rules.matchAll(/(?:^|;)\s*overflow(?:-[xy])?\s*:\s*([^;]+);/g)]
      .every(([, value]) => value.trim() === "visible")
  ));
  assert.match(declarations(".activity-text"), /display:\s*block;/);
  assert.match(declarations(".activity-text"), /white-space:\s*nowrap;/);
  assert.match(declarations(".activity-text"), /overflow:\s*hidden;/);
  assert.match(declarations(".activity-text"), /text-overflow:\s*ellipsis;/);
  assert.match(declarations(".edit pre"), /overflow-x:\s*auto;/);
  assert.match(declarations(".edit pre"), /background:\s*var\(--ui-inset\);/);
  assert.match(declarations("#chat"), /gap:\s*var\(--space-3\);/);
  assert.match(declarations("#chat > .prompt:not(:first-child)"), /margin-top:\s*var\(--space-6\);/);
});

test("agent turns use colored dots joined only between consecutive blocks", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const entryMargins = css.match(/#chat\s*>\s*:is\(\.prompt,\s*\.assistant,\s*\.activity,\s*\.edit,\s*\.question,\s*\.compaction,\s*\.gap,\s*\.unknown-event\)\s*\{([^}]*)\}/)?.[1];
  const dotRule = css.match(/#chat\s*>\s*:is\(\.assistant,\s*\.activity,\s*\.edit,\s*\.question,\s*\.compaction,\s*\.gap,\s*\.unknown-event\)::before\s*\{([^}]*)\}/)?.[1];
  const threadRule = css.match(/#chat\s*>\s*:is\(\.assistant,\s*\.activity,\s*\.edit,\s*\.question,\s*\.compaction,\s*\.gap,\s*\.unknown-event\):has\(\+ :is\(\.assistant,\s*\.activity,\s*\.edit,\s*\.question,\s*\.compaction,\s*\.gap,\s*\.unknown-event\)\)::after\s*\{([^}]*)\}/)?.[1];
  assert.match(entryMargins, /margin-top:\s*0;/);
  assert.match(entryMargins, /margin-bottom:\s*0;/);
  assert.match(css, /#chat\s*>\s*\.prompt:not\(:first-child\)\s*\{[^}]*margin-top:\s*var\(--space-6\);/s);
  assert.match(dotRule, /width:\s*0\.5rem;/);
  assert.match(dotRule, /height:\s*0\.5rem;/);
  assert.match(dotRule, /z-index:\s*1;/);
  assert.match(dotRule, /background:\s*var\(--ui-text-secondary\);/);
  assert.match(threadRule, /width:\s*1px;/);
  assert.match(threadRule, /z-index:\s*0;/);
  assert.match(threadRule, /background:\s*var\(--ui-border\);/);
  assert.match(css, /--agent-dot-top:\s*calc\(0\.775em - 0\.25rem\);/);
  assert.match(css, /--agent-dot-center:\s*calc\(var\(--agent-dot-top\) \+ 0\.25rem\);/);
  assert.match(dotRule, /top:\s*var\(--agent-dot-top\);/);
  assert.match(threadRule, /top:\s*var\(--agent-dot-center\);/);
  assert.match(threadRule, /bottom:\s*calc\(0px - var\(--space-3\) - var\(--agent-dot-center\)\);/);
  assert.match(css, /#chat\s*>\s*\.activity::before,\s*#chat\s*>\s*\.edit::before\s*\{[^}]*background:\s*var\(--ui-success\);/s);
  assert.match(css, /#chat\s*>\s*\.activity\.failed::before\s*\{[^}]*background-color:\s*var\(--ui-danger\);/);
  assert.match(css, /#chat\s*>\s*\.question\.open::before\s*\{[^}]*background:\s*var\(--ui-warning\);/);
  assert.match(css, /#chat\s*>\s*\.question\.resolved::before\s*\{[^}]*background:\s*var\(--ui-text-secondary\);/);
  for (const icon of ["page", "magnifier", "pencil", "terminal", "globe"]) {
    assert.match(css, new RegExp(`\\.icon-${icon}\\s*,`));
  }
  for (const kind of ["read", "search", "edit", "shell", "web"]) {
    assert.match(css, new RegExp(`#chat\\s*>\\s*\\.activity-${kind}::before`));
  }
  assert.match(css, /#chat\s*>\s*\.edit::before/);
  assert.match(css, /mask:\s*var\(--icon-image\) center \/ 1rem 1rem no-repeat;/);
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
    for (const [, closing, tag, attributes] of content.matchAll(/<(\/?)([a-z][\w-]*)([^>]*)>/gi)) {
      if (closing) {
        depth -= 1;
      } else {
        if (depth === 0) {
          ids.push(attributes.match(/\bid="([^"]+)"/)?.[1] ?? null);
        }
        if (["input", "img", "br", "hr", "meta", "link"].includes(tag.toLowerCase())) continue;
        depth += 1;
      }
    }
    assert.equal(depth, 0);
    return ids;
  }
  const expectedChatChildren = ["run-notice", "chat", "approvals", "jump-latest", "prompt-form"];
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

  const commandRule = cssSource.match(/(?:^|\n)\.command\s*\{([^}]*)\}/)?.[1];
  const commandOutputRule = cssSource.match(/(?:^|\n)\.command-output\s*\{([^}]*white-space:\s*pre-wrap[^}]*)\}/s)?.[1];
  assert.ok(commandRule);
  assert.ok(commandOutputRule);
  assert.match(commandRule, /(?:^|;)\s*color:\s*var\(--ui-text-secondary\)\s*;/);
  assert.match(commandRule, /(?:^|;)\s*font-family:\s*var\(--font-code\)\s*;/);
  assert.match(commandRule, /(?:^|;)\s*font-size:\s*0\.8125rem\s*;/);
  assert.match(commandRule, /(?:^|;)\s*width:\s*100%\s*;/);
  assert.match(commandRule, /(?:^|;)\s*text-align:\s*left\s*;/);
  assert.match(commandRule, /(?:^|;)\s*background:\s*none\s*;/);
  assert.match(commandRule, /(?:^|;)\s*border-radius:\s*0\s*;/);
  assert.match(commandOutputRule, /(?:^|;)\s*white-space:\s*pre-wrap\s*;/);
  assert.match(cssSource, /\.command-echo::before\s*\{[^}]*content:\s*"❯ "\s*;/s);
  assert.match(cssSource, /\.command-output::before\s*\{[^}]*content:\s*"⎿ "\s*;/s);

  assert.ok(!/^\s*import\s/m.test(renderSource));
  assert.ok(!/\b(?:if|switch)\s*\(/.test(renderSource));
  assert.match(appSource, /export async function start\(\{ global, document, client \}\)/);
  assert.ok(!appSource.includes("chatLine"));
  assert.ok(!appSource.includes("roadmap.goal"));
  assert.ok(!appSource.includes('getElementById("turn-state")'));
  assert.ok(!/\bwindow\s*(?:\.|\[)/.test(start.toString()));
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
      if (!selectors.includes("::") && selectors.split(",").map((selector) => selector.trim()).includes(`.${className}`)) {
        const match = declarations.match(/(?:^|;)\s*background:\s*(var\(--[\w-]+\))\s*;/i);
        if (match) {
          background = match[1];
        }
      }
    }
    return background;
  }
  const promptBackground = backgroundFor("prompt");
  const approvalBackground = backgroundFor("approval");
  assert.ok(promptBackground);
  assert.ok(approvalBackground);
  const commandColor = commandRule.match(/(?:^|;)\s*color:\s*(var\(--[\w-]+\))\s*;/)?.[1];
  assert.equal(promptBackground, "var(--ui-inset)");
  assert.equal(approvalBackground, "var(--ui-panel)");
  assert.notEqual(commandColor, promptBackground);
  assert.equal(backgroundFor("assistant"), undefined);
  for (const className of ["activity", "edit", "question", "compaction", "gap", "unknown-event"]) {
    assert.equal(backgroundFor(className), undefined);
  }
  for (const className of ["prompt", "assistant"]) {
    const rule = cssSource.match(new RegExp(`\\.${className}\\s*\\{([^}]*)\\}`))?.[1];
    assert.match(rule, /(?:^|;)\s*white-space:\s*pre-wrap\s*;/);
    assert.match(rule, /(?:^|;)\s*overflow-wrap:\s*anywhere\s*;/);
  }
  assert.match(cssSource, /\.edit summary\s*{[^}]*cursor:\s*pointer/s);
  assert.match(cssSource, /\.edit pre\s*{[^}]*overflow-x:\s*auto/s);
  assert.match(cssSource, /\.assistant pre\s*{[^}]*overflow-x:\s*auto/s);
  assert.match(cssSource, /\.assistant pre\s*{[^}]*background:\s*var\(--ui-inset\)/);
  assert.match(cssSource, /\.assistant code\s*{[^}]*font-family:\s*var\(--font-code\)/);
  assert.match(cssSource, /\.assistant ul,\s*\.assistant ol\s*{[^}]*padding-left:\s*1\.35em/s);
  assert.match(cssSource, /h1\s*\{[^}]*font-size:\s*1\.125rem/s);
  assert.match(cssSource, /h2,\s*h3,\s*h4,\s*h5,\s*h6\s*\{[^}]*font-size:\s*0\.875rem/s);
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

test("narrow columns leave room for chat and move the rail below it", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const [wide, narrow] = css.split("@media (max-width: 760px)");
  assert.ok(narrow);

  const wideColumns = Object.fromEntries(
    [...wide.matchAll(/(?:^|\n)(\.app-shell(?:\.[\w-]+)*)\s*\{([^}]*)\}/g)]
      .map(([, selector, rules]) => [selector, rules.match(/grid-template-columns:\s*([^;]+);/)?.[1]]),
  );
  assert.deepEqual(wideColumns, {
    ".app-shell": "14rem minmax(0, 1fr) 18rem",
    ".app-shell.sidebar-folded": "0 minmax(0, 1fr) 18rem",
    ".app-shell.rail-folded": "14rem minmax(0, 1fr) 0",
    ".app-shell.sidebar-folded.rail-folded": "0 minmax(0, 1fr) 0",
  });

  const narrowColumns = [...narrow.matchAll(/grid-template-columns:\s*([^;]+);/g)]
    .map(([, columns]) => columns);
  assert.equal(narrowColumns.length, 4);
  for (const columns of narrowColumns) {
    const fixedPixels = [...columns.matchAll(/(\d+(?:\.\d+)?)rem\b/g)]
      .reduce((total, [, rem]) => total + Number(rem) * 16, 0);
    assert.ok(fixedPixels < 760, columns);
    assert.ok(500 - fixedPixels >= 300, columns);
  }
  assert.match(narrow, /grid-template-columns:\s*min\(11rem,\s*25vw\) minmax\(0,\s*1fr\)/);
  assert.match(narrow, /#status-rail\s*\{[^}]*grid-column:\s*1\s*\/\s*-1;[^}]*grid-row:\s*2;/);
  assert.match(narrow, /\.app-shell\.rail-folded #status-rail\s*\{[^}]*display:\s*none;/);
});

test("desktop columns scroll inside the window and narrow pages scroll as a whole", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const [wide, narrow] = css.split("@media (max-width: 760px)");
  const rule = (source, selector) => source.match(
    new RegExp(`(?:^|\\n)\\s*${selector}\\s*\\{([^}]*)\\}`),
  )?.[1] ?? "";
  const shell = rule(wide, "\\.app-shell");
  const pane = rule(wide, "#status-rail,\\s*\\.chat-pane,\\s*\\.settings-pane");
  const overscroll = rule(wide, "#status-rail,\\s*\\.chat-log,\\s*\\.settings-pane,\\s*\\.projects");

  assert.match(shell, /(?:^|;)\s*height:\s*100vh\s*;/);
  assert.match(shell, /(?:^|;)\s*overflow:\s*hidden\s*;/);
  for (const selector of ["#status-rail", "\\.settings-pane"]) {
    assert.ok(pane, selector);
    assert.match(pane, /(?:^|;)\s*height:\s*100vh\s*;/);
    assert.match(pane, /(?:^|;)\s*overflow-y:\s*auto\s*;/);
  }
  for (const selector of ["#status-rail", "\\.chat-log", "\\.settings-pane", "\\.projects"]) {
    assert.ok(overscroll, selector);
    assert.match(overscroll, /(?:^|;)\s*overscroll-behavior:\s*contain\s*;/);
  }

  assert.match(rule(narrow, "\\.app-shell"), /(?:^|;)\s*height:\s*auto\s*;/);
  assert.match(rule(narrow, "\\.app-shell"), /(?:^|;)\s*overflow:\s*visible\s*;/);
  const narrowRail = rule(narrow, "#status-rail");
  const narrowPanes = rule(narrow, "\\.settings-pane");
  assert.match(narrowRail, /(?:^|;)\s*height:\s*auto\s*;/);
  assert.match(narrowRail, /(?:^|;)\s*overflow:\s*visible\s*;/);
  for (const selector of ["\\.settings-pane"]) {
    assert.ok(narrowPanes, selector);
    assert.match(narrowPanes, /(?:^|;)\s*height:\s*auto\s*;/);
    assert.match(narrowPanes, /(?:^|;)\s*overflow:\s*visible\s*;/);
  }
});

test("an empty run notice removes its row while a populated notice keeps the normal rows", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  assert.match(css, /#run-notice:empty\s*\{[^}]*display:\s*none;/);
  assert.match(css, /\.chat-pane:has\(#run-notice:empty\)\s*\{[^}]*grid-template-rows:\s*1fr auto auto;/);
  assert.doesNotMatch(css, /#run-notice\s*\{[^}]*display:\s*none;/);
});

function specWorkflowRoadmap(hasSpec = true) {
  return JSON.stringify({ goal: "Spec workflow test", phases: [{
    id: "18", name: "Workflow", status: "in_progress",
    items: [hasSpec ? { title: "Boundary", spec: BASE } : "Boundary"],
  }] });
}

function specWorkflowClient({ hasSpec = true, runs = [], onAction = {} } = {}) {
  const client = fakeClient(specWorkflowRoadmap(hasSpec));
  let currentRuns = runs;
  let specRunReads = 0;
  client.specRuns = async () => {
    specRunReads += 1;
    return currentRuns;
  };
  client.specRunReadCount = () => specRunReads;
  client.setSpecRuns = (next) => { currentRuns = next; };
  client.planSpec = async (...args) => onAction.planSpec?.(...args);
  client.runSpec = async (...args) => onAction.runSpec?.(...args);
  client.reviewSpec = async (...args) => onAction.reviewSpec?.(...args);
  client.commitSpec = async (...args) => onAction.commitSpec?.(...args);
  return client;
}

async function openWorkflowItem(client) {
  const document = new FakeDocument();
  const app = await start({ global: {}, document, client });
  const roadmapButton = find(document.getElementById("roadmap"), (node) => node.className.startsWith("roadmap-task "));
  assert.ok(roadmapButton, visibleText(document.getElementById("page")));
  await roadmapButton.dispatch("click");
  assert.ok(find(document.getElementById("spec"), (node) => node.className === "spec-workflow"), visibleText(document.getElementById("page")));
  return { app, document, roadmapButton };
}

test("starting a spec plan clears agents from the previous conversation", async () => {
  const client = specWorkflowClient({ hasSpec: false });
  const { document } = await openWorkflowItem(client);
  await client.emit(eventFrame("RunStarted", { agent_id: "leader-a", agent_name: "leader" }));
  await client.emit(eventFrame("SubagentSpawned", {
    agent_id: "leader-a", subagent_agent_id: "worker-a", subagent_name: "worker",
  }));
  assert.match(
    agentRowText(find(document.getElementById("agents"), (row) => row.className === "agent-row" && agentRowText(row).startsWith("worker"))),
    /worker · running/,
  );

  await documentButton(document, "Plan").dispatch("click");
  assert.equal(visibleText(document.getElementById("agents")), "\nNothing is running.");
});

function workflowActions(document) {
  const panel = find(document.getElementById("spec"), (node) => node.className === "spec-workflow");
  return {
    panel,
    step: find(panel, (node) => node.className === "spec-step")?.textContent,
    actions: walk(panel).filter((node) => node.tagName === "BUTTON").map((node) => node.textContent),
  };
}

test("roadmap spec workflow renders the actions for each run state", async () => {
  const reviewRun = (verdict) => ({
    kind: "implement", spec: BASE, session_id: "implement-1", state: "finished",
    review: { session_id: "review-1", verdict },
  });
  const cases = [
    ["unplanned", false, [], ["Plan"]],
    ["planning", false, [{ kind: "plan", phase: "18", item: 0, session_id: "plan-1", state: "running" }], ["Open"]],
    ["planned", true, [], ["Run"]],
    ["running", true, [{ kind: "implement", spec: BASE, session_id: "implement-1", state: "running" }], ["Open"]],
    ["ran", true, [{ kind: "implement", spec: BASE, session_id: "implement-1", state: "finished" }], ["Review", "Open"]],
    ["in review", true, [{ ...reviewRun("running"), state: "finished" }], ["Open"]],
    ["reviewed", true, [reviewRun("passed")], ["Commit…", "Open"]],
    ["reviewed", true, [reviewRun("tree-changed")], ["Review", "Open"]],
    ["committed", true, [{ ...reviewRun("passed"), committed: { sha: "abc123" } }], []],
  ];
  for (const [step, hasSpec, runs, expectedActions] of cases) {
    const { document } = await openWorkflowItem(specWorkflowClient({ hasSpec, runs }));
    const rendered = workflowActions(document);
    assert.equal(rendered.step, step);
    assert.deepEqual(rendered.actions, expectedActions, `${step} actions`);
  }
});

test("unbound planner outcomes show the stop reason and can be reopened or retried", async () => {
  const stopped = await openWorkflowItem(specWorkflowClient({ hasSpec: false, runs: [
    { kind: "plan", phase: "18", item: 0, session_id: "plan-budget", state: "stopped", stopped_reason: "budget_turns", created: [] },
  ] }));
  const stoppedView = workflowActions(stopped.document);
  assert.equal(stoppedView.step, "planner stopped: turn limit");
  assert.deepEqual(stoppedView.actions, ["Plan", "Open"]);

  const empty = await openWorkflowItem(specWorkflowClient({ hasSpec: false, runs: [
    { kind: "plan", phase: "18", item: 0, session_id: "plan-empty", state: "finished", created: [] },
  ] }));
  assert.equal(workflowActions(empty.document).step, "planner wrote no spec");
  assert.deepEqual(workflowActions(empty.document).actions, ["Plan", "Open"]);
});

test("roadmap spec actions post their payloads, refresh, open sessions, and show errors", async () => {
  const planClient = specWorkflowClient({ hasSpec: false, onAction: {
    planSpec(phase, item) {
      assert.deepEqual([phase, item], ["18", 0]);
      planClient.setSpecRuns([{ kind: "plan", phase: "18", item: 0, session_id: "plan-1", state: "running" }]);
    },
  } });
  const plan = await openWorkflowItem(planClient);
  await documentButton(plan.document, "Plan").dispatch("click");
  assert.equal(workflowActions(plan.document).step, "planning");
  assert.deepEqual(workflowActions(plan.document).actions, ["Open"]);

  const runClient = specWorkflowClient({ onAction: {
    runSpec(path) {
      assert.equal(path, BASE);
      runClient.setSpecRuns([{ kind: "implement", spec: BASE, session_id: "implement-1", state: "running" }]);
    },
  } });
  const run = await openWorkflowItem(runClient);
  await find(run.document.getElementById("spec"), (node) => node.tagName === "BUTTON" && node.textContent === "Run").dispatch("click");
  assert.equal(workflowActions(run.document).step, "running");

  const reviewClient = specWorkflowClient({ runs: [{ kind: "implement", spec: BASE, session_id: "implement-1", state: "finished" }], onAction: {
    reviewSpec(sessionId) {
      assert.equal(sessionId, "implement-1");
      reviewClient.setSpecRuns([{ kind: "implement", spec: BASE, session_id: "implement-1", state: "finished", review: { session_id: "review-1", verdict: "running" } }]);
    },
  } });
  const review = await openWorkflowItem(reviewClient);
  await find(review.document.getElementById("spec"), (node) => node.tagName === "BUTTON" && node.textContent === "Review").dispatch("click");
  assert.equal(workflowActions(review.document).step, "in review");

  const commitClient = specWorkflowClient({ runs: [{ kind: "implement", spec: BASE, session_id: "implement-1", state: "finished", review: { session_id: "review-1", verdict: "passed" } }], onAction: {
    commitSpec(sessionId, message) {
      assert.deepEqual([sessionId, message], ["implement-1", "my commit message"]);
      commitClient.setSpecRuns([{ kind: "implement", spec: BASE, session_id: "implement-1", state: "finished", review: { session_id: "review-1", verdict: "passed" }, committed: { sha: "abc123" } }]);
    },
  } });
  const commit = await openWorkflowItem(commitClient);
  const message = find(commit.document.getElementById("spec"), (node) => node.tagName === "INPUT" && node.attributes.get("aria-label") === "Commit message");
  message.value = "my commit message";
  await find(commit.document.getElementById("spec"), (node) => node.tagName === "BUTTON" && node.textContent === "Commit…").dispatch("click");
  assert.equal(workflowActions(commit.document).step, "committed");

  const openClient = specWorkflowClient({ hasSpec: false, runs: [{ kind: "plan", phase: "18", item: 0, session_id: "plan-1", state: "running" }] });
  const opened = await openWorkflowItem(openClient);
  await find(opened.document.getElementById("spec"), (node) => node.tagName === "BUTTON" && node.textContent === "Open").dispatch("click");
  assert.deepEqual(openClient.calls.openSession, ["plan-1"]);

  const errorClient = specWorkflowClient({ onAction: { async runSpec() { throw new Error("run rejected"); } } });
  const failed = await openWorkflowItem(errorClient);
  await find(failed.document.getElementById("spec"), (node) => node.tagName === "BUTTON" && node.textContent === "Run").dispatch("click");
  assert.equal(workflowActions(failed.document).step, "planned");
  assert.match(visibleText(workflowActions(failed.document).panel), /run rejected/);
});

function documentButton(document, label) {
  return find(document.getElementById("spec"), (node) => node.tagName === "BUTTON" && node.textContent === label);
}

function fakeClock(start = 10_000) {
  let now = start;
  let nextId = 0;
  const timers = new Map();
  return {
    Date: { now: () => now },
    setTimeout(callback, delay) {
      const id = ++nextId;
      timers.set(id, { due: now + delay, callback });
      return id;
    },
    async advance(milliseconds) {
      const target = now + milliseconds;
      while (true) {
        const next = [...timers.entries()].sort((left, right) => left[1].due - right[1].due)[0];
        if (!next || next[1].due > target) break;
        timers.delete(next[0]);
        now = next[1].due;
        await next[1].callback();
      }
      now = target;
    },
  };
}

function withClock(clock) {
  const browser = fakeGlobal().global;
  browser.Date = clock.Date;
  browser.setTimeout = clock.setTimeout;
  return browser;
}

test("spec-state refresh fetches immediately and once at the window end", async () => {
  const clock = fakeClock();
  const client = specWorkflowClient({ runs: [{ kind: "implement", spec: BASE, session_id: "implement-1", state: "running" }] });
  const document = new FakeDocument();
  const app = await start({ global: withClock(clock), document, client });
  await find(document.getElementById("roadmap"), (node) => node.className.startsWith("roadmap-task ")).dispatch("click");
  const before = client.specRunReadCount();
  const frame = eventFrame("GoalChanged", { session_id: "implement-1", change: "pause", phase: "paused", rounds: 1, max_rounds: 5, reason: "stopped" });
  await app.onFrame(frame);
  assert.equal(client.specRunReadCount(), before + 1);
  await clock.advance(100);
  client.setSpecRuns([{ kind: "implement", spec: BASE, session_id: "implement-1", state: "finished" }]);
  await app.onFrame(frame);
  assert.equal(client.specRunReadCount(), before + 1);
  assert.equal(workflowActions(document).step, "running");
  await clock.advance(900);
  assert.equal(client.specRunReadCount(), before + 2);
  assert.equal(workflowActions(document).step, "ran");
});

test("three spec-state events inside one window make one trailing refresh", async () => {
  const clock = fakeClock();
  const client = specWorkflowClient({ runs: [{ kind: "implement", spec: BASE, session_id: "implement-1", state: "running" }] });
  const document = new FakeDocument();
  const app = await start({ global: withClock(clock), document, client });
  const before = client.specRunReadCount();
  const frame = eventFrame("GoalChanged", { session_id: "implement-1", change: "round", phase: "active", rounds: 1, max_rounds: 5, reason: "" });
  await app.onFrame(frame);
  await clock.advance(100);
  await app.onFrame(frame);
  await clock.advance(100);
  await app.onFrame(frame);
  assert.equal(client.specRunReadCount(), before + 1);
  await clock.advance(800);
  assert.equal(client.specRunReadCount(), before + 2);
});

test("sidebar activity refreshes immediately and once at the window end", async () => {
  const clock = fakeClock();
  const document = new FakeDocument();
  const initial = { run_id: "background", title: "Background", repo_root: "/work/current", activity: "working" };
  const client = fakeClient(fixtureRoadmap(), { sessions: [initial] });
  const replies = [
    [initial],
    [{ ...initial, activity: "working" }],
    [{ ...initial, activity: "waiting" }],
  ];
  client.sessions = async (limit) => {
    client.calls.sessions.push(limit);
    return replies[Math.min(client.calls.sessions.length - 1, replies.length - 1)];
  };
  await start({ global: withClock(clock), document, client });
  const before = client.calls.sessions.length;
  const frame = eventFrame("RunFinished", { session_id: "background" });
  await client.emit(frame);
  assert.equal(client.calls.sessions.length, before + 1);
  await clock.advance(100);
  await client.emit(frame);
  assert.equal(client.calls.sessions.length, before + 1);
  assert.match(find(document.getElementById("sidebar"), (node) => node.className === "session-link").textContent, /working$/);
  await clock.advance(900);
  assert.equal(client.calls.sessions.length, before + 2);
  assert.match(find(document.getElementById("sidebar"), (node) => node.className === "session-link").textContent, /waiting$/);
});

test("chat follows new output only at the bottom and jump preserves focus", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  const chat = document.getElementById("chat");
  const jump = document.getElementById("jump-latest");
  chat.clientHeight = 100;
  chat.scrollHeight = 300;
  chat.scrollTop = 176;
  await client.emit(eventFrame("AssistantTextDelta", { text: "at bottom" }));
  assert.equal(chat.scrollTop, chat.scrollHeight);
  assert.equal(jump.hidden, true);

  chat.scrollHeight = 600;
  chat.scrollTop = 100;
  await client.emit(eventFrame("AssistantTextDelta", { text: "while reading" }));
  assert.equal(chat.scrollTop, 100);
  assert.equal(jump.hidden, false);

  document.getElementById("prompt").focus();
  const activeElement = document.activeElement;
  chat.scrollHeight = 720;
  await jump.click();
  assert.equal(chat.scrollTop, 720);
  assert.equal(jump.hidden, true);
  assert.equal(document.activeElement, activeElement);

  jump.hidden = false;
  chat.scrollTop = chat.scrollHeight - chat.clientHeight;
  await chat.dispatch("scroll");
  assert.equal(jump.hidden, true);
});

test("chat restores its follow position after visiting Settings", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });
  const chat = document.getElementById("chat");
  const jump = document.getElementById("jump-latest");
  const settings = find(document.getElementById("sidebar"), (value) => value.className === "icon-settings");
  const home = document.getElementById("home-link");

  chat.clientHeight = 100;
  chat.scrollHeight = 300;
  chat.scrollTop = 200;
  await settings.dispatch("click");
  chat.scrollTop = 0;
  chat.scrollHeight = 0;
  chat.clientHeight = 0;
  await client.emit(eventFrame("AssistantTextDelta", { text: "while away" }));
  chat.scrollHeight = 500;
  chat.clientHeight = 100;
  await home.dispatch("click");
  assert.equal(chat.scrollTop, 500);
  assert.equal(jump.hidden, true);

  chat.scrollHeight = 600;
  chat.scrollTop = 100;
  await chat.dispatch("scroll");
  await settings.dispatch("click");
  chat.scrollTop = 0;
  chat.scrollHeight = 0;
  chat.clientHeight = 0;
  await client.emit(eventFrame("AssistantTextDelta", { text: "still away" }));
  chat.scrollHeight = 600;
  chat.clientHeight = 100;
  await home.dispatch("click");
  assert.equal(chat.scrollTop, 100);
  assert.equal(jump.hidden, false);
});

test("navigating from Chat to Chat restores the reader position", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient() });
  const page = document.getElementById("page");
  const chatPane = document.getElementById("chat-pane");
  const chat = document.getElementById("chat");
  const jump = document.getElementById("jump-latest");
  chat.clientHeight = 100;
  chat.scrollHeight = 600;
  chat.scrollTop = 250;
  await chat.dispatch("scroll");

  const replaceChildren = page.replaceChildren.bind(page);
  page.replaceChildren = (...children) => {
    replaceChildren(...children);
    if (children.includes(chatPane)) chat.scrollTop = 0;
  };
  await document.getElementById("home-link").dispatch("click");

  assert.equal(chat.scrollTop, 250);
  assert.equal(jump.hidden, false);
});

test("conversation turns use a reading column and an internal transcript scroller", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const chatPane = css.match(/(?:^|\n)\.chat-pane\s*\{([^}]*)\}/)?.[1];
  const chatLog = css.match(/(?:^|\n)\.chat-log\s*\{([^}]*)\}/)?.[1];
  const prompt = css.match(/(?:^|\n)\.prompt\s*\{([^}]*)\}/)?.[1];
  const assistant = css.match(/(?:^|\n)\.assistant\s*\{([^}]*)\}/)?.[1];
  assert.match(chatPane, /(?:^|;)\s*height:\s*100vh\s*;/);
  assert.match(chatLog, /(?:^|;)\s*min-height:\s*0\s*;/);
  assert.match(chatLog, /(?:^|;)\s*overflow-y:\s*auto\s*;/);
  for (const rule of [prompt, assistant]) {
    assert.doesNotMatch(rule, /max-width:\s*80%|margin-(?:left|right):\s*auto/);
    assert.match(rule, /white-space:\s*pre-wrap/);
    assert.match(rule, /overflow-wrap:\s*anywhere/);
  }
  assert.doesNotMatch(css, /content:\s*["'](?:You|Assistant)["']/);
  assert.match(css, /#chat,\s*#approvals,\s*#prompt-form,\s*\.conversation-usage\s*\{[^}]*max-width:\s*45rem[^}]*margin-left:\s*auto[^}]*margin-right:\s*auto/);
  assert.equal((css.match(/#chat,\s*#approvals,\s*#prompt-form,\s*\.conversation-usage\s*\{/) ?? []).length, 1);
  assert.match(css, /\.status-line-settings\s*\{[^}]*display:\s*flex[^}]*flex-wrap:\s*wrap[^}]*gap:\s*var\(--space-2\)\s+var\(--space-4\)/);
  assert.match(css, /\.status-line-settings\s+h3\s*\{[^}]*flex-basis:\s*100%/);
  assert.match(html, /<button id="jump-latest"[^>]*hidden>Jump to latest<\/button>\s*<form id="prompt-form"/);
});

test("theme tokens cover every app color and match between light and dark blocks", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const light = css.match(/:root\s*\{([^{}]*)\}/)?.[1];
  const mediaDark = css.match(/:root:not\(\[data-theme="light"\]\)\s*\{([^{}]*)\}/)?.[1];
  const explicitDark = css.match(/:root\[data-theme="dark"\]\s*\{([^{}]*)\}/)?.[1];
  assert.ok(light && mediaDark && explicitDark);
  const tokens = (block) => Object.fromEntries(
    [...block.matchAll(/(--[\w-]+):\s*([^;]+);/g)].map(([, name, value]) => [name, value.trim()]),
  );
  const lightTokens = tokens(light);
  assert.deepEqual(Object.keys(tokens(mediaDark)).sort(), Object.keys(lightTokens).sort());
  assert.deepEqual(tokens(explicitDark), tokens(mediaDark));
  const declared = new Set([...css.matchAll(/(--[\w-]+):/g)].map(([, name]) => name));
  for (const [, name] of css.matchAll(/var\((--[\w-]+)\)/g)) assert.ok(declared.has(name), name);

  let outside = css
    .replace(/:root\s*\{[^{}]*\}/, "")
    .replace(/@media\s*\(prefers-color-scheme:\s*dark\)\s*\{\s*:root:not\(\[data-theme="light"\]\)\s*\{[^{}]*\}\s*\}/, "")
    .replace(/:root\[data-theme="light"\]\s*\{[^{}]*\}/, "")
    .replace(/:root\[data-theme="dark"\]\s*\{[^{}]*\}/, "");
  assert.doesNotMatch(outside, /#[\da-f]{3,8}\b|rgba?\s*\(|hsla?\s*\(/i);
});

test("app type, spacing, and shape use the design scale", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const rule = (selector, property = "font-size") => {
    for (const [, selectors, declarations] of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
      if (
        selectors.split(",").map((value) => value.trim()).includes(selector)
        && new RegExp(`${property}\\s*:`).test(declarations)
      ) return declarations;
    }
    return "";
  };

  const fontSizes = [...css.matchAll(/font-size\s*:\s*([^;}]+);/g)].map(([, value]) => value.trim());
  assert.ok(fontSizes.length > 0);
  for (const value of fontSizes) {
    assert.ok(["0.75rem", "0.8125rem", "0.875rem", "1.125rem", "inherit"].includes(value), value);
  }
  const fontWeights = [...css.matchAll(/font-weight\s*:\s*([^;}]+);/g)].map(([, value]) => value.trim());
  assert.ok(fontWeights.length > 0);
  for (const value of fontWeights) assert.ok(["400", "500", "600"].includes(value), value);

  const radii = [...css.matchAll(/border-radius\s*:\s*([^;}]+);/g)].map(([, value]) => value.trim());
  assert.ok(radii.length > 0);
  for (const value of radii) {
    assert.ok(["0", "50%", "var(--radius-control)", "var(--radius-popover)", "var(--radius-composer)"].includes(value), value);
  }
  for (const selector of [".command-menu", ".file-menu", ".picker"]) {
    assert.match(rule(selector, "border-radius"), /border-radius:\s*var\(--radius-popover\);/);
  }

  const spacing = [...css.matchAll(/(?:^|[;{])\s*(?:padding(?:-[\w-]+)?|margin(?:-[\w-]+)?|gap)\s*:\s*([^;}]+);/g)];
  assert.ok(spacing.length > 0);
  for (const [, value] of spacing) assert.doesNotMatch(value, /(?:\d+(?:\.\d+)?)(?:rem|px)\b/, value);

  const expectedTokens = {
    "--space-1": "0.25rem",
    "--space-2": "0.5rem",
    "--space-3": "0.75rem",
    "--space-4": "1rem",
    "--space-6": "1.5rem",
    "--space-8": "2rem",
    "--radius-control": "0.25rem",
    "--radius-popover": "0.375rem",
    "--radius-composer": "0.375rem",
  };
  const rootBlocks = [...css.matchAll(/:root\s*\{([^{}]*)\}/g)];
  const spacingRoot = rootBlocks.find(([, block]) => block.includes("--space-1:"));
  assert.ok(spacingRoot);
  assert.ok(rootBlocks.indexOf(spacingRoot) > 0);
  assert.ok(css.indexOf(spacingRoot[0]) > css.indexOf(':root[data-theme="dark"]'));
  for (const [name, value] of Object.entries(expectedTokens)) {
    assert.equal([...css.matchAll(new RegExp(`${name}:`, "g"))].length, 1, name);
    assert.match(spacingRoot[1], new RegExp(`${name}:\\s*${value.replace(".", "\\.")}\\s*;`));
  }

  assert.match(rule("body"), /font-size:\s*0\.875rem;/);
  assert.match(rule("body"), /line-height:\s*1\.55;/);
  assert.match(rule("h1"), /font-size:\s*1\.125rem;/);
  assert.match(rule("h1"), /font-weight:\s*600;/);
  assert.match(rule("h2"), /font-size:\s*0\.875rem;/);
  assert.match(rule("h2"), /font-weight:\s*600;/);
  assert.match(rule("button"), /font-size:\s*0\.8125rem;/);
  assert.match(rule("textarea"), /font-size:\s*0\.875rem;/);
  assert.match(rule(".conversation-usage"), /font-size:\s*0\.75rem;/);
  assert.match(rule(".command"), /font-size:\s*0\.8125rem;/);
  assert.match(rule(".command"), /line-height:\s*1\.55;/);
});

test("controls, focus, motion, and prose links use the app states", async () => {
  const css = await readFile(new URL("../app.css", import.meta.url), "utf8");
  const button = css.match(/(?:^|\n)button\s*\{([^}]*)\}/)?.[1] ?? "";
  assert.match(button, /min-height:\s*2rem;/);
  assert.match(button, /padding:\s*0\s+var\(--space-3\);/);
  assert.match(button, /border:\s*1px solid var\(--ui-control-border\);/);
  assert.match(button, /border-radius:\s*var\(--radius-control\);/);
  assert.match(button, /background:\s*transparent;/);
  assert.match(button, /color:\s*var\(--ui-text\);/);
  assert.match(button, /font-weight:\s*500;/);
  assert.match(button, /cursor:\s*pointer;/);
  assert.match(button, /transition:\s*background-color 120ms, border-color 120ms, color 120ms;/);
  assert.match(css, /button:not\(:disabled\):hover\s*\{[^}]*background:\s*var\(--ui-hover\);/s);
  assert.match(css, /button:disabled\s*\{(?=[^}]*color:\s*var\(--ui-text-secondary\);)(?=[^}]*border-color:\s*var\(--ui-border\);)(?=[^}]*cursor:\s*default;)[^}]*\}/s);
  const fields = css.match(/input:not\(\[type="checkbox"\]\):not\(\[type="radio"\]\):not\(\[type="file"\]\),\s*select,\s*textarea\s*\{([^}]*)\}/)?.[1] ?? "";
  assert.match(fields, /min-height:\s*2rem;/);
  assert.match(fields, /padding:\s*var\(--space-1\) var\(--space-2\);/);
  assert.match(fields, /border:\s*1px solid var\(--ui-control-border\);/);
  assert.match(fields, /border-radius:\s*var\(--radius-control\);/);
  assert.match(fields, /background:\s*var\(--ui-panel\);/);
  assert.match(fields, /color:\s*var\(--ui-text\);/);
  assert.match(css, /input::placeholder,\s*textarea::placeholder\s*\{[^}]*color:\s*var\(--ui-text-secondary\);/s);

  assert.match(css, /\.primary\s*\{[^}]*background:\s*var\(--ui-accent\);[^}]*color:\s*var\(--ui-on-accent\);/s);
  assert.match(css, /\.primary:not\(:disabled\):hover\s*\{[^}]*background:\s*var\(--ui-accent-hover\);/s);

  const rowButtons = [
    "session-link", "command-suggestion", "file-suggestion", "picker-row",
    "roadmap-phase-header", "roadmap-phase-chain-summary", "roadmap-task",
    "roadmap-follow-up", "roadmap-item",
  ];
  const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)];
  for (const className of rowButtons) {
    const matchingRules = rules.filter(([, selectors]) =>
      selectors.split(",").map((selector) => selector.trim()).includes(`.${className}`)
    );
    assert.ok(matchingRules.some(([, , declarations]) => /border:\s*(?:0|1px solid transparent);/.test(declarations)), className);
  }
  assert.match(css, /\.command-suggestion\.focused\s*\{[^}]*background:\s*var\(--ui-hover\);[^}]*border-color:\s*var\(--ui-accent\);/s);
  assert.match(css, /\.picker-row\.focused\s*\{[^}]*background:\s*var\(--ui-hover\);[^}]*border-color:\s*var\(--ui-accent\);/s);

  assert.match(css, /:focus-visible\s*\{[^}]*outline:\s*2px solid var\(--ui-accent\);[^}]*outline-offset:\s*2px;/s);
  assert.match(css, /\.picker\s*\{[^}]*outline:\s*none;/s);
  assert.match(css, /@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{[^}]*\*\s*\{[^}]*transition:\s*none;/s);
  assert.match(css, /\.assistant a\s*\{[^}]*color:\s*var\(--ui-accent\);[^}]*text-decoration:\s*underline;/s);
});

test("inline theme bootstrap applies only valid explicit themes and never throws", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)?.[1];
  assert.ok(script);
  assert.ok(html.indexOf("<script>") < html.indexOf('rel="stylesheet"'));
  const apply = (stored, throws = false) => {
    const root = new FakeElement("html");
    const localStorage = { getItem() { if (throws) throw new Error("blocked"); return stored; } };
    assert.doesNotThrow(() => new Function("document", "localStorage", script)({ documentElement: root }, localStorage));
    return root.attributes.get("data-theme");
  };
  assert.equal(apply("dark"), "dark");
  assert.equal(apply("light"), "light");
  for (const stored of ["system", null, "blue"]) assert.equal(apply(stored), undefined);
  assert.equal(apply("dark", true), undefined);
});

test("general settings select applies theme even when storage is unavailable", async () => {
  const browser = fakeGlobal({ fragment: "#/settings/general", themeStored: "dark" });
  const document = new FakeDocument();
  await start({ global: browser.global, document, client: fakeClient() });
  const select = find(document.getElementById("page"), (node) => node.attributes.get("aria-label") === "Theme");
  assert.ok(find(document.getElementById("page"), (node) => node.textContent === "Appearance"));
  assert.equal(select.value, "dark");
  document.documentElement.setAttribute("data-theme", "dark");
  assert.equal(document.documentElement.attributes.get("data-theme"), "dark");
  select.value = "system";
  await select.dispatch("change");
  assert.deepEqual(browser.writes.filter(([key]) => key === "symphonai.theme"), [["symphonai.theme", "system"]]);
  assert.equal(document.documentElement.attributes.has("data-theme"), false);

  const invalidBrowser = fakeGlobal({ fragment: "#/settings/general", themeStored: "blue" });
  const invalidDocument = new FakeDocument();
  await start({ global: invalidBrowser.global, document: invalidDocument, client: fakeClient() });
  assert.equal(
    find(invalidDocument.getElementById("page"), (node) => node.attributes.get("aria-label") === "Theme").value,
    "system",
  );

  const blockedBrowser = fakeGlobal({ fragment: "#/settings/general", storageThrows: true });
  const blockedDocument = new FakeDocument();
  await start({ global: blockedBrowser.global, document: blockedDocument, client: fakeClient() });
  const blockedSelect = find(blockedDocument.getElementById("page"), (node) => node.attributes.get("aria-label") === "Theme");
  assert.equal(blockedSelect.value, "system");
  blockedSelect.value = "light";
  await blockedSelect.dispatch("change");
  assert.equal(blockedDocument.documentElement.attributes.get("data-theme"), "light");
});
