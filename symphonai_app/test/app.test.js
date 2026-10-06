import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

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
      "chat-pane": "main",
      "prompt-form": "form",
      prompt: "textarea",
      "prompt-attachments": "div",
      "attachment-picker": "input",
      "attach-button": "button",
      "send-button": "button",
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
        "agents",
        "chat-pane",
        "roadmap",
        "spec",
        "run-notice",
        "chat",
        "approvals",
        "prompt-form",
        "prompt",
        "prompt-attachments",
        "attachment-picker",
        "attach-button",
        "send-button",
        "prompt-error",
      ].map((id) => [id, new FakeElement(tags[id] ?? "div", id)]),
    );
    const get = (id) => this.elements.get(id);
    get("app-shell").className = "app-shell";
    get("chat-pane").className = "chat-pane";
    get("new-chat").className = "new-chat";
    get("attachment-picker").type = "file";
    get("sidebar").append(get("home-link"), get("new-chat"), get("page-links"));
    get("status-rail").append(get("agents"), get("roadmap"), get("spec"));
    get("prompt-form").append(get("prompt"), get("prompt-attachments"), get("attachment-picker"), get("attach-button"), get("send-button"), get("prompt-error"));
    get("chat-pane").append(get("run-notice"), get("chat"), get("approvals"), get("prompt-form"));
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
    health = { protocol_version: 1, state: "idle", run_id: null, runtime_run_id: null },
    conversation = null,
    modelListing = { provider: "", state: "unknown", models: [], detail: "Model listing unavailable." },
    changes = { turns: [], files: [] },
    revertConflictPaths = [],
    worktreeConflict = false,
    fileOverrides = {},
    specFilesReply = { paths: [] },
    specFilesFailure = false,
  } = {},
) {
  const calls = { agent: [], approve: [], applyWorktree: [], changes: 0, compact: [], controlAgent: [], conversationStats: 0, credentials: [], discardWorktree: [], file: [], files: [], history: [], forkSession: [], goal: [], goalState: [], models: [], newSession: 0, openSession: [], prompt: [], promptAttachments: [], revertChanges: [], saveAgent: [], selectMode: [], selectProvider: [], sessions: [], settings: 0, specFiles: 0 };
  let changesReply = changes;
  let worktreeConflictPending = worktreeConflict;
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
    async stop() {
      return { accepted: true };
    },
    async controlAgent(agentId, action, text) {
      calls.controlAgent.push({ agent_id: agentId, action, ...(text === undefined ? {} : { text }) });
      return {
        agent_id: agentId,
        state: action === "pause" ? "paused" : action === "stop" ? "stopping" : "running",
      };
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
    async changes() {
      calls.changes += 1;
      return changesReply;
    },
    async revertChanges(payload) {
      calls.revertChanges.push(payload);
      if (revertConflictPaths.length > 0 && !payload.force) {
        const error = new Error("files changed outside the agent");
        error.status = 409;
        error.paths = revertConflictPaths;
        throw error;
      }
      changesReply = { turns: [], files: [] };
      return { reverted: [payload.path].filter(Boolean) };
    },
    async applyWorktree(name) {
      calls.applyWorktree.push(name);
      if (worktreeConflictPending) {
        worktreeConflictPending = false;
        const error = new Error("patch does not apply");
        error.status = 409;
        throw error;
      }
      changesReply = { turns: [], files: [], worktrees: [] };
      return { applied: [] };
    },
    async discardWorktree(name) {
      calls.discardWorktree.push(name);
      changesReply = { turns: [], files: [], worktrees: [] };
      return { discarded: name };
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

  await document.getElementById("attach-button").click();
  assert.equal(document.getElementById("attachment-picker").clicked, true);
  document.getElementById("attachment-picker").files = [pdf];
  await document.getElementById("attachment-picker").dispatch("change");
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
  assert.match(find(document.body, (node) => node.className === "conversation-usage").textContent, /Goal active 1\/10/);
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
  assert.match(find(document.body, (node) => node.className === "conversation-usage").textContent, /Goal active 1\/10/);
  await app.onFrame(eventFrame("RunFinished", { stopped_reason: "final_response" }));
  await app.onFrame(goalEvent("check", "active", 1, { exit: 1, ok: false, output: "1 failed" }));
  await app.onFrame(eventFrame("PromptSubmitted", { text: "round two feedback", message_count: 2 }));
  await app.onFrame(eventFrame("RunFinished", { stopped_reason: "final_response" }));
  await app.onFrame(goalEvent("check", "complete", 2, { exit: 0, ok: true, output: "" }));
  assert.match(find(document.body, (node) => node.className === "conversation-usage").textContent, /Goal complete 2\/10/);
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

test("Mod+Enter submits the composer through the form", async () => {
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
  await input.dispatch("keydown", { key: "Enter", ctrlKey: true });

  assert.deepEqual(client.calls.prompt, ["hello"]);
  assert.equal(input.value, "");
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
  assert.equal(find(document.body, (value) => value.className === "conversation-usage").textContent, "");
});

test("graph roadmap loads short spec titles and opens items and follow-ups", async () => {
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
  const initialGraph = find(document.getElementById("roadmap").children[0], (node) => node.className === "roadmap-graph");
  assert.equal(find(initialGraph, (node) => node.className === "roadmap-box-button").textContent, "18a  Boundary");
  releaseFiles();
  await new Promise((resolve) => setTimeout(resolve, 0));

  const phase = document.getElementById("roadmap").children[0];
  const graph = find(phase, (node) => node.className === "roadmap-graph");
  assert.ok(graph);
  assert.equal(walk(graph).filter((node) => node.className === "roadmap-box-button").length, 2);
  assert.ok(find(graph, (node) => node.className === "roadmap-connector-row"));
  assert.ok(find(graph, (node) => node.className.startsWith("roadmap-edge ")));
  assert.equal(find(graph, (node) => node.className === "roadmap-box-button").textContent, "18a  A beautifully lo…");
  assert.equal(find(graph, (node) => node.className === "roadmap-box-button").attributes.get("title"), "Boundary");
  assert.equal(find(graph, (node) => node.className === "roadmap-follow-up").textContent, "18aF  Fix the goal");

  await find(graph, (node) => node.className === "roadmap-box-button").dispatch("click");
  assert.ok(client.calls.file.includes(BASE));
  await find(graph, (node) => node.className === "roadmap-follow-up").dispatch("click");
  assert.ok(client.calls.file.includes(FOLLOW_UP));
  assert.ok(!find(document.getElementById("roadmap"), (node) => node.tagName === "SVG"));
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
  const priorGraph = find(rail.children[0], (node) => node.className === "roadmap-graph");
  assert.equal(find(priorGraph, (node) => node.className === "roadmap-follow-up").textContent, "37eF  Fix the goal");
  const currentPhase = rail.children[1];
  assert.ok(find(currentPhase, (node) => node.className === "roadmap-graph"));
  assert.ok(!find(currentPhase, (node) => node.className === "roadmap-graph-error"));

  await find(priorGraph, (node) => node.className === "roadmap-box-button").dispatch("click");
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

  const phase = document.getElementById("roadmap").children[0];
  assert.ok(find(phase, (node) => node.className === "roadmap-graph"));
  assert.equal(find(phase, (node) => node.className === "roadmap-box-button").textContent, "18a  Open boundary");
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
        { agent_id: "leader-id", name: "leader", input_tokens: 20, output_tokens: 10, calls: 2, total_tokens: 30, cost: { amount: "0.004", currency: "USD" } },
        { agent_id: "child-id", name: "researcher", input_tokens: 10, output_tokens: 2, calls: 1, total_tokens: 12 },
      ],
    },
    {
      context: { used_tokens: 65, budget_tokens: 160, remaining_tokens: 95, by_source: { user: 35, assistant: 30 } },
      usage: { input_tokens: 35, output_tokens: 15, calls: 4, total_tokens: 50, cost: { amount: "0.007", currency: "USD" } },
      agents: [
        { agent_id: "leader-id", name: "leader", input_tokens: 25, output_tokens: 13, calls: 3, total_tokens: 38, cost: { amount: "0.005", currency: "USD" } },
        { agent_id: "child-id", name: "researcher", input_tokens: 10, output_tokens: 2, calls: 1, total_tokens: 12 },
      ],
    },
  ];
  client.conversationStats = async () => {
    client.calls.conversationStats += 1;
    return { conversation: snapshots[Math.min(client.calls.conversationStats - 1, 1)] };
  };
  await start({ global: {}, document, client });

  const usage = find(document.body, (value) => value.className === "conversation-usage");
  assert.equal(usage.textContent, "openai / gpt-test · Mode ask · Context 120 / 160 tokens · 42 tokens");
  assert.doesNotMatch(usage.textContent, /USD|undefined/);
  assert.match(visibleText(document.getElementById("agents")), /leader · done · 30 tokens · USD 0\.004/);
  assert.match(visibleText(document.getElementById("agents")), /researcher · done · 12 tokens/);
  assert.doesNotMatch(visibleText(document.body), new RegExp(`${secret}|${absolutePath}`));

  await client.emit(eventFrame("RunFinished", { agent_id: "leader-id" }));
  assert.equal(usage.textContent, "Context 65 / 160 tokens · 50 tokens · USD 0.007");
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

test("composer keeps attachment controls inside the form", async () => {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient() });
  const form = document.getElementById("prompt-form");
  assert.deepEqual(form.children.map((child) => child.id || child.className), [
    "prompt", "prompt-attachments", "attachment-picker", "attach-button",
    "send-button", "prompt-error", "conversation-usage",
  ]);
  assert.equal(walk(form).some((node) => node.tagName === "SELECT"), false);
  assert.equal(document.getElementById("attachment-picker").type, "file");
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
  const status = find(document.getElementById("prompt-form"), (node) => node.className.includes("conversation-usage"));
  assert.match(status.textContent, /Mode plan/);
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
  assert.match(find(document.getElementById("prompt-form"), (node) => node.className.includes("conversation-usage")).textContent, /Effort xhigh/);
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
  assert.match(find(document.getElementById("prompt-form"), (node) => node.className.includes("conversation-usage")).textContent, /Effort low/);
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

test("sidebar keeps projects, settings and changes links, and a home link", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal({ fragment: "#/settings/general" });
  await start({ global: browser.global, document, client: fakeClient() });
  const sidebar = document.getElementById("sidebar");
  const links = document.getElementById("page-links");

  assert.deepEqual(links.children.map((link) => link.textContent), ["Settings", "Changes"]);
  assert.doesNotMatch(visibleText(sidebar), /Roadmap|Chat/);
  assert.equal(sidebar.children[0], document.getElementById("home-link"));
  await document.getElementById("home-link").dispatch("click");
  assert.equal(browser.global.location.hash, "#/chat");
  assert.deepEqual(document.getElementById("page").children, [document.getElementById("chat-pane")]);
});

test("Changes page renders files and prompts, retries outside edits, and refreshes", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    changes: {
      turns: [{ key: "chk-one", prompt: "Update the source", paths: ["a.py"] }],
      files: [{
        path: "a.py",
        status: "modified",
        changed_outside: true,
        diff: "-before\n+after",
        truncated: false,
      }],
    },
    revertConflictPaths: ["a.py"],
  });
  await start({ global: {}, document, client });
  const page = document.getElementById("page");
  await find(document.getElementById("page-links"), (value) => value.textContent === "Changes")
    .dispatch("click");
  await Promise.resolve();

  assert.equal(client.calls.changes, 1);
  assert.match(visibleText(page), /a\.py · modified/);
  assert.match(visibleText(page), /Changed outside the agent/);
  assert.match(visibleText(page), /Update the source/);
  assert.match(visibleText(page), /-before\n\+after/);
  await find(page, (value) => value.textContent === "Revert").dispatch("click");
  assert.deepEqual(client.calls.revertChanges, [{ path: "a.py" }]);
  assert.match(visibleText(page), /Revert anyway/);
  assert.match(visibleText(page), /a\.py/);

  await find(page, (value) => value.textContent === "Revert anyway").dispatch("click");
  assert.deepEqual(client.calls.revertChanges, [
    { path: "a.py" },
    { path: "a.py", force: true },
  ]);
  assert.equal(client.calls.changes, 2);
  assert.match(visibleText(page), /No changes in this conversation/);

  await document.getElementById("home-link").dispatch("click");
  await find(document.getElementById("page-links"), (value) => value.textContent === "Changes")
    .dispatch("click");
  await Promise.resolve();
  assert.equal(client.calls.changes, 3);
});

test("Changes page applies worktrees, shows conflicts, and reloads after success", async () => {
  const document = new FakeDocument();
  const client = fakeClient(fixtureRoadmap(), {
    changes: {
      turns: [], files: [],
      worktrees: [{ name: "w1", files: ["a.py"], diff: "-before\n+after", truncated: false }],
    },
    worktreeConflict: true,
  });
  await start({ global: {}, document, client });
  const page = document.getElementById("page");
  await find(document.getElementById("page-links"), (value) => value.textContent === "Changes")
    .dispatch("click");
  await Promise.resolve();
  assert.match(visibleText(page), /w1/);
  assert.match(visibleText(page), /-before\n\+after/);
  await find(page, (value) => value.textContent === "Apply").dispatch("click");
  await Promise.resolve();
  assert.deepEqual(client.calls.applyWorktree, ["w1"]);
  assert.match(visibleText(page), /patch does not apply/);
  await find(page, (value) => value.textContent === "Apply").dispatch("click");
  await Promise.resolve();
  assert.deepEqual(client.calls.applyWorktree, ["w1", "w1"]);
  assert.equal(client.calls.changes, 2);
  assert.match(visibleText(page), /No changes in this conversation/);
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
  await client.emit(eventFrame("SubagentSpawned", {
    subagent_agent_id: "agent-3", subagent_name: "sibling",
  }));
  await client.emit(eventFrame("SubagentSpawned", {
    agent_id: "agent-2", subagent_agent_id: "agent-4", subagent_name: "grandchild",
  }));
  await client.emit(eventFrame("ToolCallStarted", { agent_id: "agent-4", tool_name: "search" }));
  await client.emit(eventFrame("SubagentStopped", { subagent_agent_id: "agent-2" }));
  const root = agents.children[0];
  const children = root.children[0];
  const worker = children.children[0];
  assert.deepEqual(agents.children.map((row) => row.textContent), ["leader · running · read"]);
  assert.equal(children.className, "agent-children");
  assert.deepEqual(children.children.map((row) => row.textContent), ["worker · done", "sibling · running"]);
  assert.equal(worker.children[0].className, "agent-children");
  assert.deepEqual(worker.children[0].children.map((row) => row.textContent), ["grandchild · running · search"]);

  await client.emit(eventFrame("SubagentSpawned", {
    agent_id: "missing", subagent_agent_id: "orphan", subagent_name: "orphan",
  }));
  assert.deepEqual(agents.children.map((row) => row.textContent), ["leader · running · read", "orphan · running"]);
});

test("agent rail controls pause, redirect, stop, and disappear when done", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal({ fragment: "#/chat" });
  const client = fakeClient();
  await start({ global: browser.global, document, client });
  await client.emit(eventFrame("RunStarted", { agent_id: "leader-id", agent_name: "leader" }));

  let row = find(document.getElementById("agents"), (value) => value.className === "agent-row");
  let controls = row.children.find((value) => value.className === "agent-controls");
  assert.deepEqual(controls.children.map((button) => button.textContent), ["Pause", "Redirect", "Stop"]);
  await controls.children[0].dispatch("click");
  assert.deepEqual(client.calls.controlAgent[0], { agent_id: "leader-id", action: "pause" });
  row = find(document.getElementById("agents"), (value) => value.className === "agent-row");
  assert.match(row.textContent, /leader · paused/);
  controls = row.children.find((value) => value.className === "agent-controls");
  assert.equal(controls.children[0].textContent, "Resume");
  await controls.children[1].dispatch("click");
  let redirectInput = find(row, (value) => value.tagName === "INPUT");
  assert.ok(redirectInput);
  await redirectInput.dispatch("keydown", { key: "Escape", preventDefault() {} });
  assert.equal(find(row, (value) => value.tagName === "INPUT"), undefined);

  await controls.children[1].dispatch("click");
  redirectInput = find(row, (value) => value.tagName === "INPUT");
  redirectInput.value = "Focus on the migration";
  const send = find(row, (value) => value.textContent === "Send");
  await send.dispatch("click");
  assert.deepEqual(client.calls.controlAgent[1], {
    agent_id: "leader-id", action: "redirect", text: "Focus on the migration",
  });
  row = find(document.getElementById("agents"), (value) => value.className === "agent-row");
  controls = row.children.find((value) => value.className === "agent-controls");
  await controls.children[2].dispatch("click");
  assert.deepEqual(client.calls.controlAgent[2], { agent_id: "leader-id", action: "stop" });

  await client.emit(eventFrame("RunFinished", { agent_id: "leader-id", stopped_reason: "cancelled" }));
  row = find(document.getElementById("agents"), (value) => value.className === "agent-row");
  assert.equal(row.textContent, "leader · done");
  assert.equal(row.children.some((value) => value.className === "agent-controls"), false);
});

test("agent rail keeps state unchanged and shows a control error", async () => {
  const document = new FakeDocument();
  const browser = fakeGlobal({ fragment: "#/chat" });
  const client = fakeClient();
  client.controlAgent = async () => {
    throw new Error("agent is already paused");
  };
  await start({ global: browser.global, document, client });
  await client.emit(eventFrame("RunStarted", { agent_id: "leader-id", agent_name: "leader" }));
  let row = find(document.getElementById("agents"), (value) => value.className === "agent-row");
  const controls = row.children.find((value) => value.className === "agent-controls");
  await controls.children[0].dispatch("click");
  row = find(document.getElementById("agents"), (value) => value.className === "agent-row");
  assert.equal(row._textContent, "leader · running");
  assert.equal(
    row.children.find((value) => value.className === "agent-control-error").textContent,
    "agent is already paused",
  );
  assert.equal(row.children.find((value) => value.className === "agent-controls").children[0].textContent, "Pause");
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
      { agent_id: "root", name: "leader", parent_agent_id: null },
      { agent_id: "a", name: "A", parent_agent_id: "root" },
      { agent_id: "b", name: "B", parent_agent_id: "a" },
      { agent_id: "c", name: "C", parent_agent_id: "root" },
    ] };
    return { run_id: runId };
  };
  await start({ global: {}, document, client });
  const agents = document.getElementById("agents");
  assert.equal(visibleText(agents), "\nNothing is running.");
  const link = find(document.getElementById("sidebar"), (row) => row.className === "session-link");
  await link.dispatch("click");
  assert.deepEqual(client.calls.openSession, ["prior"]);
  assert.deepEqual(agents.children.map((row) => row.textContent), ["leader · done"]);
  const children = agents.children[0].children[0];
  assert.deepEqual(children.children.map((row) => row.textContent), ["A · done", "C · done"]);
  assert.deepEqual(children.children[0].children[0].children.map((row) => row.textContent), ["B · done"]);
  assert.equal(find(document.body, (row) => row.className === "conversation-usage").textContent, "");
  assert.match(visibleText(document.getElementById("chat")), /Prior answer/);
  assert.doesNotMatch(visibleText(agents), /tokens|USD/);
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
    "Hooks", "Ceiling", "Trust", "Inventory",
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
      (value) => value.tagName === "BUTTON" && value.textContent.startsWith(title),
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

test("replayed messages offer a fork at the message and show the parent in the sidebar", async () => {
  const document = new FakeDocument();
  const source = { run_id: "source", title: "Original", repo_root: "/work/current" };
  const fork = { run_id: "fork-run", title: "Original", parent_session_id: "source", repo_root: "/work/current" };
  const client = fakeClient(fixtureRoadmap(), { sessions: [source] });
  client.sessions = async (limit) => {
    client.calls.sessions.push(limit);
    return client.calls.sessions.length === 1 ? [source] : [fork, source];
  };
  client.forkSession = async (runId, recordId) => {
    client.calls.forkSession.push([runId, recordId]);
    await client.emit({ kind: "event", payload: {
      type: "HistoryMessage", role: "user", text: "What changed?", record_id: "rec-fork",
      tool_calls: [], turn_id: "turn-fork",
    } });
    return { run_id: "fork-run" };
  };
  await start({ global: {}, document, client });
  const chat = document.getElementById("chat");
  assert.equal(find(chat, (value) => value.className === "fork-message"), undefined);
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
  const controls = walk(chat).filter((value) => value.className === "fork-message");
  assert.equal(controls.length, 2);
  await controls[0].dispatch("click");
  assert.deepEqual(client.calls.forkSession, [["source", "rec-user"]]);
  assert.match(visibleText(chat), /What changed\?/);
  assert.doesNotMatch(visibleText(chat), /The config changed\./);
  assert.match(visibleText(document.getElementById("sidebar")), /Original · fork of Original/);
  const forkControl = find(chat, (value) => value.className === "fork-message");
  await forkControl.dispatch("click");
  assert.deepEqual(client.calls.forkSession[1], ["fork-run", "rec-fork"]);
});

test("fork conflicts show paths and Branch anyway retries with force", async () => {
  const document = new FakeDocument();
  const source = { run_id: "source", title: "Original", repo_root: "/work/current" };
  const fork = { run_id: "fork-run", title: "Original", parent_session_id: "source", repo_root: "/work/current" };
  const client = fakeClient(fixtureRoadmap(), { sessions: [source] });
  client.sessions = async () => [source, fork];
  client.forkSession = async (runId, recordId, force = false) => {
    client.calls.forkSession.push([runId, recordId, ...(force ? [true] : [])]);
    if (!force) {
      throw Object.assign(new Error("files changed outside the agent"), {
        status: 409,
        paths: ["src/a.py", "src/b.py"],
      });
    }
    await client.emit({ kind: "event", payload: {
      type: "HistoryMessage", role: "user", text: "Original prompt", record_id: "rec-fork",
      tool_calls: [], turn_id: "turn-fork",
    } });
    return { run_id: "fork-run" };
  };
  await start({ global: {}, document, client });
  const sourceButton = find(document.getElementById("sidebar"), (value) =>
    value.className === "session-link" && value.textContent === "Original"
  );
  await sourceButton.dispatch("click");
  await client.emit({ kind: "event", payload: {
    type: "HistoryMessage", role: "user", text: "Original prompt", record_id: "rec-user",
    tool_calls: [], turn_id: "turn-1",
  } });
  const forkButton = find(document.getElementById("chat"), (value) => value.className === "fork-message");
  await forkButton.dispatch("click");
  assert.match(visibleText(document.getElementById("chat")), /src\/a\.py, src\/b\.py/);
  const branchAnyway = find(document.getElementById("chat"), (value) => value.textContent === "Branch anyway");
  assert.ok(branchAnyway);
  await branchAnyway.dispatch("click");
  assert.deepEqual(client.calls.forkSession, [["source", "rec-user"], ["source", "rec-user", true]]);
  assert.match(visibleText(document.getElementById("chat")), /Original prompt/);
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
    /#status-rail,\s*\.chat-pane,\s*\.settings-pane,\s*\.changes-pane\s*\{([^}]*)\}/,
  )?.[1];
  assert.match(sharedPaneRule, /(?:^|;)\s*overflow:\s*auto\s*;/);

  const commandRule = cssSource.match(/(?:^|\n)\.command\s*\{([^}]*)\}/)?.[1];
  const commandOutputRule = cssSource.match(/(?:^|\n)\.command-output\s*\{([^}]*white-space:\s*pre-wrap[^}]*)\}/s)?.[1];
  assert.ok(commandRule);
  assert.ok(commandOutputRule);
  assert.match(commandRule, /(?:^|;)\s*color:\s*#9aa7b8\s*;/);
  assert.match(commandRule, /(?:^|;)\s*font-family:\s*ui-monospace, SFMono-Regular, Menlo, monospace\s*;/);
  assert.match(commandRule, /(?:^|;)\s*font-size:\s*0\.9rem\s*;/);
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
  assert.notEqual(backgroundFor("command"), promptBackground);
  assert.notEqual(backgroundFor("command"), assistantBackground);
  for (const className of ["prompt", "assistant"]) {
    const rule = cssSource.match(new RegExp(`\\.${className}\\s*\\{([^}]*)\\}`))?.[1];
    assert.match(rule, /(?:^|;)\s*white-space:\s*pre-wrap\s*;/);
    assert.match(rule, /(?:^|;)\s*overflow-wrap:\s*anywhere\s*;/);
  }
  assert.match(cssSource, /\.edit summary\s*{[^}]*cursor:\s*pointer/s);
  assert.match(cssSource, /\.edit pre\s*{[^}]*overflow-x:\s*auto/s);
  assert.match(cssSource, /\.assistant pre\s*{[^}]*overflow-x:\s*auto/s);
  assert.match(cssSource, /\.assistant pre\s*{[^}]*background:\s*#[0-9a-f]{6}/i);
  assert.match(cssSource, /\.assistant code\s*{[^}]*font-family:\s*ui-monospace,\s*SFMono-Regular,\s*Menlo,\s*monospace/s);
  assert.match(cssSource, /\.assistant ul,\s*\.assistant ol\s*{[^}]*padding-left:\s*1\.35em/s);
  assert.match(cssSource, /\.assistant h1,[\s\S]*?\.assistant h6\s*{[^}]*font-size:\s*1\.1em/s);
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
  const roadmapButton = find(document.getElementById("roadmap"), (node) => node.className === "roadmap-item");
  await roadmapButton.dispatch("click");
  return { app, document, roadmapButton };
}

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

test("roadmap spec actions post their payloads, refresh, open sessions, and show errors", async () => {
  const planClient = specWorkflowClient({ hasSpec: false, onAction: {
    planSpec(phase, item) {
      assert.deepEqual([phase, item], ["18", 0]);
      planClient.setSpecRuns([{ kind: "plan", phase: "18", item: 0, session_id: "plan-1", state: "running" }]);
    },
  } });
  const plan = await openWorkflowItem(planClient);
  await documentButton(plan.document, "Plan").dispatch("click");
  assert.match(find(plan.document.getElementById("roadmap"), (node) => node.className === "roadmap-item").textContent, /planning$/);
  assert.deepEqual(workflowActions(plan.document).actions, ["Open"]);

  const runClient = specWorkflowClient({ onAction: {
    runSpec(path) {
      assert.equal(path, BASE);
      runClient.setSpecRuns([{ kind: "implement", spec: BASE, session_id: "implement-1", state: "running" }]);
    },
  } });
  const run = await openWorkflowItem(runClient);
  await find(run.document.getElementById("spec"), (node) => node.tagName === "BUTTON" && node.textContent === "Run").dispatch("click");
  assert.match(find(run.document.getElementById("roadmap"), (node) => node.className === "roadmap-item").textContent, /running$/);

  const reviewClient = specWorkflowClient({ runs: [{ kind: "implement", spec: BASE, session_id: "implement-1", state: "finished" }], onAction: {
    reviewSpec(sessionId) {
      assert.equal(sessionId, "implement-1");
      reviewClient.setSpecRuns([{ kind: "implement", spec: BASE, session_id: "implement-1", state: "finished", review: { session_id: "review-1", verdict: "running" } }]);
    },
  } });
  const review = await openWorkflowItem(reviewClient);
  await find(review.document.getElementById("spec"), (node) => node.tagName === "BUTTON" && node.textContent === "Review").dispatch("click");
  assert.match(find(review.document.getElementById("roadmap"), (node) => node.className === "roadmap-item").textContent, /in review$/);

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
  assert.match(find(commit.document.getElementById("roadmap"), (node) => node.className === "roadmap-item").textContent, /committed$/);

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
  await find(document.getElementById("roadmap"), (node) => node.className === "roadmap-item").dispatch("click");
  const before = client.specRunReadCount();
  const frame = eventFrame("GoalChanged", { session_id: "implement-1", change: "pause", phase: "paused", rounds: 1, max_rounds: 5, reason: "stopped" });
  await app.onFrame(frame);
  assert.equal(client.specRunReadCount(), before + 1);
  await clock.advance(100);
  client.setSpecRuns([{ kind: "implement", spec: BASE, session_id: "implement-1", state: "finished" }]);
  await app.onFrame(frame);
  assert.equal(client.specRunReadCount(), before + 1);
  assert.match(find(document.getElementById("roadmap"), (node) => node.className === "roadmap-item").textContent, /running$/);
  await clock.advance(900);
  assert.equal(client.specRunReadCount(), before + 2);
  assert.match(find(document.getElementById("roadmap"), (node) => node.className === "roadmap-item").textContent, /ran$/);
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
