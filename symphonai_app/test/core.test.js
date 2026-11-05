import test from "node:test";
import { execFileSync } from "node:child_process";
import assert from "node:assert/strict";
import { start as startApp } from "../src/app.js";
import { renderMemorySettings } from "../src/settings.js";
import { createTranscript } from "../src/transcript.js";
import { renderTranscript } from "../src/render.js";
import { parseMarkdown } from "../src/markdown.js";

function start(options) {
  options.document.defaultView = options.global;
  return startApp(options);
}

function addListener(target, type, listener) {
  const listeners = target.listeners.get(type) ?? [];
  listeners.push(listener);
  target.listeners.set(type, listeners);
}

async function dispatch(target, type, values = {}) {
  let stopped = false;
  const event = { type, target, defaultPrevented: false,
    ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, ...values };
  event.preventDefault = () => { event.defaultPrevented = true; values.preventDefault?.(); };
  event.stopPropagation = () => { stopped = true; values.stopPropagation?.(); };
  const path = [];
  for (let node = target; node; node = node.parentNode) path.push(node);
  const document = target.ownerDocument;
  if (document) path.push(document);
  if (document?.defaultView) path.push(document.defaultView);
  const pending = [];
  for (const current of path) {
    event.currentTarget = current;
    for (const listener of [...(current.listeners.get(type) ?? [])]) {
      pending.push(listener(event));
    }
    if (stopped) break;
  }
  await Promise.all(pending);
  return event;
}

// A small DOM with real parent/child and textContent semantics. Missing ids
// return null, and operations on a non-child fail rather than hiding a crash.
class Node {
  constructor(document, tagName = "", text = "") {
    Object.defineProperties(this, {
      ownerDocument: { value: document, writable: true },
      parentNode: { value: null, writable: true },
    });
    this.nodeType = tagName ? 1 : 3;
    this.tagName = tagName.toUpperCase();
    this.childNodes = [];
    this.data = text;
    this.className = "";
    this.value = "";
    this.hidden = false;
    this.open = false;
    this.disabled = false;
    this.scrollTop = 0;
    this.clientHeight = 300;
    this.scrollHeight = 300;
    this.listeners = new Map();
    this.attributes = new Map();
  }
  get children() { return this.childNodes.filter((node) => node.nodeType === 1); }
  get textContent() { return this.nodeType === 3 ? this.data : this.childNodes.map((node) => node.textContent).join(""); }
  set textContent(text) {
    if (this.nodeType === 3) this.data = String(text);
    else this.replaceChildren(...(String(text) ? [this.ownerDocument.createTextNode(text)] : []));
  }
  append(...nodes) {
    for (let node of nodes) {
      if (typeof node !== "object") node = this.ownerDocument.createTextNode(node);
      node.remove();
      node.parentNode = this;
      this.childNodes.push(node);
    }
  }
  remove() {
    if (this.parentNode) {
      const index = this.parentNode.childNodes.indexOf(this);
      this.parentNode.childNodes.splice(index, 1);
      this.parentNode = null;
    }
  }
  replaceChildren(...nodes) {
    for (const node of this.childNodes) node.parentNode = null;
    this.childNodes = [];
    this.append(...nodes);
  }
  replaceChild(next, previous) {
    const index = this.childNodes.indexOf(previous);
    if (index < 0) throw new Error("not a child");
    next.remove();
    this.childNodes[index] = next;
    next.parentNode = this;
    previous.parentNode = null;
    return previous;
  }
  before(...nodes) {
    if (!this.parentNode) return;
    const parent = this.parentNode;
    for (const node of nodes) {
      node.remove();
      parent.childNodes.splice(parent.childNodes.indexOf(this), 0, node);
      node.parentNode = parent;
    }
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  removeAttribute(name) { this.attributes.delete(name); }
  set href(value) { this.setAttribute("href", new URL(value, "http://127.0.0.1/app/").href); }
  get href() { return this.getAttribute("href"); }
  contains(node) { return node === this || this.childNodes.some((child) => child.contains(node)); }
  focus() { this.ownerDocument.activeElement = this; }
  setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; }
  addEventListener(type, listener) { addListener(this, type, listener); }
  dispatch(type, values = {}) { return dispatch(this, type, values); }
  click() { return this.dispatch("click"); }
  requestSubmit() { return this.dispatch("submit"); }
}

function find(root, predicate) {
  if (predicate(root)) return root;
  for (const node of root.childNodes ?? []) {
    const found = find(node, predicate);
    if (found) return found;
  }
  return null;
}

class Document {
  constructor() {
    this.listeners = new Map();
    this.documentElement = new Node(this, "html");
    this.body = new Node(this, "body");
    this.documentElement.append(this.body);
    const make = (tag, id, parent) => {
      const node = new Node(this, tag);
      node.id = id;
      parent.append(node);
      return node;
    };
    const shell = make("div", "app-shell", this.body);
    const sidebar = make("nav", "sidebar", shell);
    make("h1", "home-link", sidebar);
    make("button", "new-chat", sidebar);
    make("div", "page-links", sidebar);
    make("button", "sidebar-toggle", shell);
    const page = make("div", "page", shell);
    const chat = make("main", "chat-pane", page);
    for (const id of ["run-notice", "chat", "approvals"]) make("div", id, chat);
    make("button", "jump-latest", chat);
    const form = make("form", "prompt-form", chat);
    make("textarea", "prompt", form);
    make("div", "prompt-attachments", form);
    make("div", "prompt-error", form);
    make("section", "terminal-panel", page).hidden = true;
    const rail = make("aside", "status-rail", shell);
    for (const id of ["rail-tabs", "agents", "roadmap", "changes", "spec", "files", "rail-tools"]) make("div", id, rail);
    make("button", "rail-toggle", shell);
  }
  createElement(tag) { return new Node(this, tag); }
  createTextNode(text) { return new Node(this, "", String(text)); }
  getElementById(id) { return find(this.documentElement, (node) => node.id === id); }
  addEventListener(type, listener) { addListener(this, type, listener); }
}

function browser() {
  const storage = new Map();
  return {
    location: { hash: "#/chat" },
    listeners: new Map(),
    addEventListener(type, listener) { addListener(this, type, listener); },
    dispatch(type, values = {}) { return dispatch(this, type, values); },
    localStorage: { getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
  };
}

function replayPage(replayId, history = [], offset = 0, total = history.length) {
  const end = offset + history.length;
  return { replay_id: replayId, history_frames: total, offset, history,
    next_offset: end < total ? end : null };
}

function historyFrame(fields) {
  return { kind: "event", payload: { type: "HistoryMessage", agent_id: "core-agent",
    agent_name: "leader", run_id: "core-run", ...fields } };
}

function animationFrames(global, afterFrame = () => {}) {
  const pending = [];
  global.requestAnimationFrame = callback => pending.push(callback);
  return async () => {
    const callbacks = pending.splice(0);
    await Promise.all(callbacks.map(callback => {
      const result = callback();
      afterFrame();
      return result;
    }));
  };
}

function scriptedClient() {
  let receive;
  const prompts = [];
  let newSessions = 0;
  const client = {
    project: async () => ({ repo_root: "/core-project", name: "Core project" }),
    sessions: async () => [],
    openSession: async () => (replayPage("empty")),
    sessionViewed: async () => ({ viewed_at: "2026-11-02T00:00:00+00:00" }),
    roadmap: async () => ({ goals: [] }),
    settings: async () => ({ settings: { mode: "allow", providers: [{ name: "openai", key_present: true }] } }),
    health: async () => ({ state: "idle", runtime_run_id: null }),
    conversationStats: async () => ({ conversation: { agents: [{ agent_id: "core-agent", name: "leader", parent_agent_id: null }] } }),
    approvals: async () => ({ pending: [] }),
    approve: async () => ({ resolved: true }),
    stop: async () => ({ accepted: true }),
    changes: async () => ({ turns: [], files: [], worktrees: [] }),
    loops: async () => ({ loops: [] }),
    history: async () => ({ prompts: [] }),
    events(callback) { receive = callback; return { ready: Promise.resolve(), close() {} }; },
    prompt: async (text) => { prompts.push(text); return { accepted: true, run_id: "core-run" }; },
    newSession: async () => { newSessions += 1; return { ended: true }; },
  };
  return { client, prompts, newSessions: () => newSessions, frame: value => receive(value), emit: (type, fields = {}) => receive({
    kind: "event", payload: { type, agent_id: "core-agent", agent_name: "leader", run_id: "core-run", ...fields },
  }) };
}

async function submit(document, text) {
  document.getElementById("prompt").value = text;
  await document.getElementById("prompt-form").dispatch("submit");
}

function terminalBrowser() {
  const global = browser();
  const instances = [];
  global.innerHeight = 800;
  global.Terminal = class {
    constructor(options) { this.options = options; this.output = ""; this.cols = 80; this.rows = 24; instances.push(this); }
    loadAddon(addon) { addon.terminal = this; }
    open(view) { this.view = view; }
    write(text, callback) { this.output += text; callback?.(); }
    onData(callback) { this.input = callback; }
    attachCustomKeyEventHandler(callback) { this.key = callback; }
    focus() { this.view.focus(); }
    dispose() { this.disposed = true; }
  };
  global.FitAddon = { FitAddon: class {
    fit() { this.terminal.cols = 96; this.terminal.rows = 12; }
  } };
  global.getComputedStyle = () => ({ getPropertyValue: (name) => ({
    "--font-code": "fixture-monospace", "--ui-text": "#202124", "--ui-inset": "#EFEFED",
    "--ui-danger": "#B4232D",
  })[name] ?? "" });
  const event = (key, fields = {}) => ({
    key, type: "keydown", ctrlKey: false, metaKey: false, altKey: false, shiftKey: false,
    preventDefault() {}, stopPropagation() {}, ...fields,
  });
  return { global, instances, event, dispatch: (type, value) => global.dispatch(type, value) };
}

test("reload refreshes skill commands and the mention roster", async () => {
  const script = scriptedClient();
  let reloaded = false;
  script.client.settings = async () => ({ settings: {
    mode: "allow", providers: [{ name: "openai", key_present: true }],
    skills: reloaded ? [{ name: "new-skill", description: "Fixture skill" }] : [],
  } });
  script.client.reloadExtensions = async () => {
    reloaded = true;
    return { changes: { skills: { added: ["new-skill"], removed: [], changed: [], failed: [] } } };
  };
  const document = new Document();
  const app = await start({ global: browser(), document, client: script.client });
  await submit(document, "/reload");
  assert.match(document.body.textContent, /1 added/);
  await submit(document, "/new-skill");
  assert.equal(script.prompts[0], "Use the new-skill skill.");
  await script.emit("RunStarted", { parent_run_id: null });
  await script.emit("RunFinished", { stopped_reason: "final_response" });
  await submit(document, "@new-skill help");
  assert.match(script.prompts[1], /^Use the new-skill skill\./);
  app.subscription.close();
});

test("General saves a scalar and keeps an invalid draft beside its error", async () => {
  const script = scriptedClient();
  const writes = [];
  let extract = true;
  script.client.settings = async () => ({ settings: { mode: "allow", providers: [], config: [
    { key: "memory.extract", value: extract, scope: "user" },
    { key: "memory.model", value: "fixture", scope: "user" },
  ] } });
  script.client.setConfig = async (key, value, scope) => {
    writes.push({ key, value, scope });
    if (key === "memory.model") throw new Error("invalid model");
    extract = value;
    return { changes: {} };
  };
  const document = new Document();
  const global = browser(); global.location.hash = "#/settings/general";
  const app = await start({ global, document, client: script.client });
  const field = find(document.body, node => node.getAttribute?.("aria-label") === "memory.extract");
  field.checked = false;
  await find(field.parentNode, node => node.tagName === "BUTTON").click();
  assert.deepEqual(writes[0], { key: "memory.extract", value: false, scope: "user" });
  assert.equal(find(document.body, node => node.getAttribute?.("aria-label") === "memory.extract").checked, false);
  const model = find(document.body, node => node.getAttribute?.("aria-label") === "memory.model");
  model.value = "bad draft";
  await find(model.parentNode, node => node.tagName === "BUTTON").click();
  assert.equal(model.value, "bad draft");
  assert.match(model.parentNode.textContent, /invalid model/);
  app.subscription.close();
});

test("Memory shows consolidation changes and restores the selected scope", async () => {
  const document = new Document();
  const content = document.createElement("section");
  const restored = [];
  const context = {
    document, settingsContent: content, memoryLoading: false, memoryLoadError: null,
    memoryReply: { project: { entries: [], has_previous: true, last_consolidation: {
      time: "2026-11-02", merged: [{ ids: ["a", "b"], text: "Merged fact", previous: [{ text: "Old fact" }] }],
      removed: [{ id: "c", text: "Outdated fact", reason: "Superseded" }],
    } } },
    boundary: { restoreMemory: async scope => restored.push(scope) },
    loadMemory: async () => { context.memoryReply.project.entries = [{ id: "a", text: "Restored fact" }]; },
    renderMemoryContent() {},
  };
  renderMemorySettings(context);
  assert.match(content.textContent, /1 merges, 1 removed/);
  assert.match(content.textContent, /Old fact → Merged fact/);
  assert.match(content.textContent, /Outdated fact — Superseded/);
  await find(content, node => node.tagName === "BUTTON" && node.textContent === "Restore previous").click();
  assert.deepEqual(restored, ["project"]);
});

test("Terminal panel starts closed and opens without xterm globals", async () => {
  const document = new Document();
  const script = scriptedClient();
  const app = await start({ global: browser(), document, client: script.client });
  const panel = document.getElementById("terminal-panel");
  assert.ok(panel.hidden);
  assert.equal(app.route().page, "chat");
  await find(document.body, (node) => node.className === "terminal-toggle").click();
  assert.ok(!panel.hidden);
  assert.match(panel.textContent, /Terminal unavailable/);
  await find(document.body, (node) => node.className === "terminal-toggle").click();
  assert.ok(panel.hidden);
});

test("Terminal tabs route input, preserve output while hidden, and reattach on reload", async () => {
  const fixture = terminalBrowser();
  const { global, instances } = fixture;
  const script = scriptedClient();
  let starts = 0;
  const shells = new Map(), inputs = [], stopped = [], sizes = [];
  Object.assign(script.client, {
    terminals: async () => ({ terminals: [...shells.values()] }),
    terminalStart: async () => {
      const id = `shell-${++starts}`;
      shells.set(id, { id, exited: false, output: "" });
      return { id };
    },
    terminalInput: async (id, data) => { inputs.push([id, data]); },
    terminalResize: async (id, cols, rows) => { sizes.push([id, cols, rows]); },
    terminalStop: async (id) => { stopped.push(id); shells.delete(id); },
  });
  let document = new Document();
  await start({ global, document, client: script.client });
  await fixture.dispatch("keydown", fixture.event("`", { ctrlKey: true, target: document.body }));
  const panel = document.getElementById("terminal-panel");
  assert.ok(!panel.hidden);
  assert.equal(starts, 1);
  assert.deepEqual(sizes.at(-1), ["shell-1", 96, 12]);
  assert.equal(instances[0].options.fontFamily, "fixture-monospace");
  assert.equal(instances[0].options.theme.red, "#B4232D");
  instances[0].input("pwd\r");
  instances[0].input("\x03");
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(inputs, [["shell-1", "pwd\r"], ["shell-1", "\x03"]]);
  const output = "\x1b[31mred\x1b[0m\r\n";
  shells.get("shell-1").output = output;
  await script.emit("TerminalOutput", { terminal_id: "shell-1", data: output, exited: false });
  assert.equal(instances[0].output, output, "raw output is passed unchanged to xterm");
  for (let added = 0; added < 3; added++) {
    await find(panel, (node) => node.getAttribute?.("aria-label") === "Add terminal").click();
  }
  assert.equal(starts, 4);
  assert.ok(find(panel, (node) => node.getAttribute?.("aria-label") === "Add terminal").disabled);
  instances[1].input("second\r");
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(inputs.at(-1), ["shell-2", "second\r"]);
  await find(panel, (node) => node.getAttribute?.("aria-label") === "Close terminal 2").click();
  assert.deepEqual(stopped, ["shell-2"]);
  assert.ok(instances[1].disposed);
  await find(document.body, (node) => node.className === "terminal-toggle").click();
  assert.ok(panel.hidden);
  await script.emit("TerminalOutput", { terminal_id: "shell-1", data: "hidden output", exited: false });
  shells.get("shell-1").output += "hidden output";
  assert.ok(instances[0].output.endsWith("hidden output"));
  assert.deepEqual(stopped, ["shell-2"], "hiding the panel does not stop shells");
  await find(document.body, (node) => node.className === "terminal-toggle").click();
  const handle = find(panel, (node) => node.getAttribute?.("role") === "separator");
  await handle.dispatch("pointerdown", { clientY: 500 });
  await handle.dispatch("pointermove", { clientY: 380 });
  await handle.dispatch("pointerup");
  assert.equal(JSON.parse(global.localStorage.getItem("symphonai.terminal-panel")).height, 400);
  document = new Document();
  await start({ global, document, client: script.client });
  assert.equal(starts, 4, "reload reuses all three running shells");
  assert.ok(!document.getElementById("terminal-panel").hidden);
  assert.match(document.getElementById("terminal-panel").getAttribute("style"), /400px/);
  assert.equal(instances.at(-3).output, shells.get("shell-1").output);
});

test("Terminal replay drops stale replies and keystrokes until parsing finishes, then answers live queries", async () => {
  for (const buffered of [false, true]) {
    const { global, instances } = terminalBrowser();
    const script = scriptedClient();
    const inputs = [], writes = [];
    global.Terminal.prototype.write = function (text, callback) {
      this.output += text;
      writes.push({ text, callback });
    };
    const saved = "saved\x1b[6n";
    const pending = "buffered\x1b[c\x1b]11;?\x07\x1b[?25$p";
    Object.assign(script.client, {
      terminals: async () => {
        if (buffered) await script.emit("TerminalOutput", {
          terminal_id: "existing", data: saved + pending, offset: 0, exited: false,
        });
        return { terminals: [{ id: "existing", output: saved, exited: false }] };
      },
      terminalResize: async () => ({}),
      terminalInput: async (id, data) => { inputs.push([id, data]); },
    });
    const document = new Document();
    await start({ global, document, client: script.client });
    await find(document.body, (node) => node.className === "terminal-toggle").click();
    const terminal = instances[0];
    assert.equal(writes.length, 1, "saved and buffered output share a final replay callback");
    assert.equal(writes[0].text, saved + (buffered ? pending : ""));
    terminal.input("typed during replay\r");
    terminal.input("\x1b[2;1R\x1b[?1;2c\x1b]11;rgb:1515/1616/1818\x07\x1b[?25;1$y");
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(inputs, [], "neither keystrokes nor automatic replies reach the shell before the callback");
    // A live frame queued behind replay must be allowed to answer when parsed.
    await script.emit("TerminalOutput", { terminal_id: "existing", data: "\x1b[6n", exited: false });
    writes[0].callback();
    terminal.input("vim\r");
    terminal.input("\x1b[3;1R");
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(inputs, [["existing", "vim\r"], ["existing", "\x1b[3;1R"]]);
    assert.equal(writes[1].text, "\x1b[6n", "live queries still go through xterm");
  }
});

test("Terminal reattach uses character offsets when repeated Unicode output overlaps the snapshot", async () => {
  const { global, instances } = terminalBrowser();
  const script = scriptedClient();
  const line = "😀 repeated\r\n", length = Array.from(line).length;
  script.client.terminals = async () => {
    await script.emit("TerminalOutput", { terminal_id: "repeated", offset: 0, data: line.repeat(30), exited: false });
    await script.emit("TerminalOutput", { terminal_id: "repeated", offset: length * 30, data: line.repeat(20), exited: false });
    return { terminals: [{ id: "repeated", output: line.repeat(25), end: length * 25, exited: false }] };
  };
  script.client.terminalResize = async () => ({});
  const document = new Document();
  await start({ global, document, client: script.client });
  await find(document.body, (node) => node.className === "terminal-toggle").click();
  assert.equal(instances[0].output, line.repeat(50));
  await script.emit("TerminalOutput", { terminal_id: "repeated", offset: 0, data: line.repeat(50), exited: false });
  assert.equal(instances[0].output, line.repeat(50), "already replayed data is ignored");
});

test("Terminal batches input arriving while a request is in flight", async () => {
  const { global, instances } = terminalBrowser();
  const script = scriptedClient();
  const inputs = [];
  let release;
  script.client.terminals = async () => ({ terminals: [{ id: "batch", output: "", end: 0, exited: false }] });
  script.client.terminalResize = async () => ({});
  script.client.terminalInput = async (id, data) => {
    inputs.push([id, data]);
    if (inputs.length === 1) await new Promise((resolve) => { release = resolve; });
  };
  const document = new Document();
  await start({ global, document, client: script.client });
  await find(document.body, (node) => node.className === "terminal-toggle").click();
  instances[0].input("x");
  for (let chunk = 0; chunk < 10; chunk++) instances[0].input("p".repeat(500));
  assert.deepEqual(inputs, [["batch", "x"]]);
  release();
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(inputs, [["batch", "x"], ["batch", "p".repeat(5000)]]);
});

test("Terminal focus isolates shortcuts and the panel height stays within its bounds", async () => {
  const fixture = terminalBrowser();
  const { global, instances } = fixture;
  const script = scriptedClient();
  let stops = 0;
  Object.assign(script.client, {
    terminals: async () => ({ terminals: [{ id: "existing", output: "saved", exited: false }] }),
    terminalResize: async () => ({}),
    stop: async () => { stops++; return {}; },
  });
  const document = new Document();
  await start({ global, document, client: script.client });
  await find(document.body, (node) => node.className === "terminal-toggle").click();
  assert.equal(instances[0].output, "saved");
  let propagationStopped = false;
  assert.equal(instances[0].key(fixture.event(".", { ctrlKey: true, stopPropagation() { propagationStopped = true; } })), true);
  assert.ok(propagationStopped, "shell keystrokes do not bubble to app shortcuts");
  await fixture.dispatch("keydown", fixture.event(".", { ctrlKey: true, target: instances[0].view }));
  assert.equal(stops, 0);
  const panel = document.getElementById("terminal-panel");
  const handle = find(panel, (node) => node.getAttribute?.("role") === "separator");
  await handle.dispatch("pointerdown", { clientY: 500 });
  await handle.dispatch("pointermove", { clientY: -5000 });
  await handle.dispatch("pointerup");
  assert.match(panel.getAttribute("style"), /640px/);
  await handle.dispatch("pointerdown", { clientY: 500 });
  await handle.dispatch("pointermove", { clientY: 5000 });
  await handle.dispatch("pointerup");
  assert.match(panel.getAttribute("style"), /120px/);
  assert.equal(instances[0].key(fixture.event("`", { ctrlKey: true })), false);
  assert.ok(panel.hidden);
  assert.equal(document.activeElement, document.getElementById("prompt"));
});

test("MCP slash prompts show required arguments and submit rendered text", async () => {
  for (const multiple of [false, true]) {
    const document = new Document();
    const script = scriptedClient();
    const calls = [];
    const arguments_ = multiple
      ? [{ name: "topic", required: true }, { name: "style", required: true }]
      : [{ name: "topic", required: true }];
    script.client.mcpPrompts = async () => ({ prompts: [{ server: "fixture", name: "explain",
      description: "Explain a topic", arguments: arguments_ }] });
    script.client.mcpPrompt = async (server, name, arguments_) => {
      calls.push({ server, name, arguments: arguments_ });
      if (!arguments_.topic || (multiple && !arguments_.style)) {
        throw new Error("Required prompt argument is missing");
      }
      return { text: `Explain ${arguments_.topic}${arguments_.style ? ` in ${arguments_.style}` : ""}` };
    };
    await start({ global: browser(), document, client: script.client });
    const input = document.getElementById("prompt");
    input.value = "/mcp.fixture";
    await input.dispatch("input");
    assert.match(document.body.textContent, /mcp.fixture.explain/);
    assert.match(document.body.textContent, /<topic>/);
    await submit(document, "/mcp.fixture.explain");
    assert.deepEqual(script.prompts, []);
    assert.deepEqual(calls, [{ server: "fixture", name: "explain", arguments: {} }]);
    assert.match(document.getElementById("chat").textContent, /Required prompt argument is missing/);
    calls.length = 0;
    await submit(document, multiple ? '/mcp.fixture.explain topic="async IO" style=detail' : "/mcp.fixture.explain async IO");
    assert.deepEqual(calls, [{ server: "fixture", name: "explain", arguments: multiple
      ? { topic: "async IO", style: "detail" } : { topic: "async IO" } }]);
    assert.deepEqual(script.prompts, [multiple ? "Explain async IO in detail" : "Explain async IO"]);
  }
});

test("saved history waits for its exact replay count, paints once, then reconciles edits together", async () => {
  const document = new Document();
  const script = scriptedClient();
  const global = browser();
  const chat = document.getElementById("chat");
  const rendered = [];
  let observedText = "";
  const frame = animationFrames(global, () => {
    if (chat.textContent !== observedText) {
      observedText = chat.textContent;
      if (observedText) rendered.push(observedText);
    }
  });
  script.client.sessions = async (_limit, _query, archived) => archived ? [] : [
    { run_id: "saved", title: "Saved", repo_root: "/core-project" },
  ];
  const history = Array.from({ length: 40 }, (_, i) => historyFrame({
    session_id: "saved", replay_id: "batch", role: "user", text: `Message ${i}`,
    ...(i < 3 ? { checkpoint_key: `edit-${i}`, edits: [{ path: `file-${i}.txt`, added: 1, removed: 0 }],
      turn_ended_at: "2026-11-03T00:00:00Z", turn_outcome: "final_response" } : {}),
  }));
  let opened = false, finishPage, pageRequested;
  const requested = new Promise(resolve => { pageRequested = resolve; });
  script.client.openSession = async () => {
    opened = true;
    return replayPage("batch", history.slice(0, 20), 0, 40);
  };
  script.client.historyPage = (id, offset) => {
    assert.equal(id, "batch");
    assert.equal(offset, 20);
    pageRequested();
    return new Promise(resolve => { finishPage = resolve; });
  };
  script.client.conversationStats = async () => ({ conversation: opened ? { session_id: "saved", agents: [
    { agent_id: "core-agent", parent_agent_id: null },
  ] } : null });
  let changesCalls = 0, finishChanges;
  script.client.changes = () => {
    changesCalls += 1;
    return new Promise(resolve => { finishChanges = resolve; });
  };
  const app = await start({ global, document, client: script.client });
  await frame();
  const opening = find(document.body, node => node.className === "session-link" && node.textContent === "Saved").click();
  await requested;
  await frame();
  assert.equal(chat.textContent, "", "the first page must not paint a partial conversation");
  finishPage(replayPage("batch", history.slice(20), 20, 40));
  await opening;
  assert.equal(chat.textContent, "", "painting waits for an animation frame");
  await frame();
  assert.equal(rendered.length, 1);
  for (let i = 0; i < 40; i += 1) assert.ok(chat.textContent.includes(`Message ${i}`));
  assert.equal(app.transcript.model.filter(entry => entry.type === "edit-card").length, 3);
  assert.equal(chat.scrollTop, chat.scrollHeight);
  assert.equal(changesCalls, 0);
  const reconciliation = frame();
  assert.equal(changesCalls, 1);
  await script.emit("AssistantTextDelta", { session_id: "saved", text: "LIVE FIRST" });
  await frame();
  assert.match(chat.textContent, /LIVE FIRST/);
  await script.emit("AssistantTextDelta", { session_id: "saved", text: " SECOND" });
  await frame();
  assert.match(chat.textContent, /LIVE FIRST SECOND/);
  finishChanges({ turns: [0, 1, 2].map(i => ({ key: `edit-${i}`, paths: [`file-${i}.txt`] })) });
  await reconciliation;
  await frame();
  assert.equal(changesCalls, 1);
  app.subscription.close();
});

test("switched and failed opens never expose partial history", async () => {
  const document = new Document();
  const script = scriptedClient();
  const global = browser();
  const frame = animationFrames(global);
  script.client.sessions = async (_limit, _query, archived) => archived ? [] : ["A", "B"].map(id => ({
    run_id: id, title: id, repo_root: "/core-project",
  }));
  let hostSession = null;
  script.client.conversationStats = async () => ({ conversation: hostSession ? {
    session_id: hostSession, agents: [{ agent_id: "core-agent", parent_agent_id: null }],
  } : null });
  const app = await start({ global, document, client: script.client });
  await script.emit("PromptSubmitted", { text: "Original conversation" });
  await frame();
  const replies = [], pages = new Map();
  script.client.openSession = id => {
    hostSession = id;
    return new Promise((resolve, reject) => replies.push({ id, resolve, reject }));
  };
  script.client.historyPage = id => new Promise(resolve => pages.set(id, resolve));
  const open = id => find(document.body, node => node.className === "session-link" && node.textContent === id).click();
  const first = open("A");
  replies[0].resolve(replayPage("old-A", [historyFrame({ session_id: "A", role: "user", text: "Partial A" })], 0, 2));
  await Promise.resolve();
  assert.ok(pages.has("old-A"));
  await frame();
  assert.equal(document.getElementById("chat").textContent, "");
  const second = open("B");
  pages.get("old-A")(replayPage("old-A", [historyFrame({ session_id: "A", role: "assistant", text: "Rest A" })], 1, 2));
  await first;
  replies[1].resolve(replayPage("B", [historyFrame({ session_id: "B", role: "user", text: "Partial B" })], 0, 2));
  await Promise.resolve();
  assert.ok(pages.has("B"));
  const third = open("A");
  pages.get("B")(replayPage("B", [historyFrame({ session_id: "B", role: "assistant", text: "Rest B" })], 1, 2));
  await second;
  await frame();
  assert.equal(document.getElementById("chat").textContent, "");
  replies[2].reject(new Error("Open failed"));
  await third;
  await frame();
  assert.equal(app.transcript.snapshot().sessionId, "A");
  assert.equal(document.getElementById("chat").textContent, "");
  assert.match(document.getElementById("run-notice").textContent, /Open failed/);
  assert.ok(find(document.getElementById("run-notice"), node => node.textContent === "Retry" && node.tagName === "BUTTON"));
  app.subscription.close();
});

test("leader row returns from an agent view by click or Enter and preserves scroll", async () => {
  const script = scriptedClient();
  script.client.openSession = async () => (replayPage("empty"));
  script.client.conversationStats = async () => ({ conversation: {
    session_id: "session", agents: [
      { agent_id: "root", name: "leader", parent_agent_id: null, state: "finished" },
      { agent_id: "worker", name: "worker", parent_agent_id: "root", state: "finished" },
    ],
  } });
  script.client.agentHistory = async () => ({ messages: [{ role: "assistant", text: "Worker answer" }] });
  const document = new Document();
  const app = await start({ global: browser(), document, client: script.client });
  const chat = document.getElementById("chat");
  for (const keyboard of [false, true]) {
    chat.scrollTop = 123;
    await chat.dispatch("scroll");
    const worker = () => find(document.body, node => node.getAttribute?.("aria-label") === "Open worker's conversation");
    await worker().click();
    assert.equal(worker().getAttribute("aria-current"), "true");
    assert.match(chat.textContent, /Worker answer/);
    const leader = find(document.body, node => node.getAttribute?.("aria-label") === "Back to the conversation");
    assert.equal(leader.getAttribute("role"), "button");
    if (keyboard) await leader.dispatch("keydown", { key: "Enter" });
    else await leader.click();
    assert.equal(chat.scrollTop, 123);
    assert.equal(worker().getAttribute("aria-current"), null);
    assert.doesNotMatch(chat.textContent, /Worker answer/);
  }
  app.subscription.close();
});

test("Usage defaults on, explains missing prices, migrates once and keeps new choices", async () => {
  const global = browser();
  global.localStorage.setItem("symphonai.status-line", JSON.stringify({ usage: false, model: false }));
  const script = scriptedClient();
  script.client.openSession = async () => (replayPage("empty"));
  const usage = { total_tokens: 182340, cost: { amount: "0.42", currency: "USD", estimated: true } };
  script.client.conversationStats = async () => ({ conversation: { session_id: "session", usage } });
  const document = new Document();
  const app = await start({ global, document, client: script.client });
  assert.match(document.body.textContent, /~\$0\.42 · 182,340 tokens/);
  global.location.hash = "#/settings/general";
  await find(document.body, node => node.className === "icon-settings").click();
  const checkbox = find(document.body, node => node.getAttribute?.("aria-label") === "Usage");
  assert.equal(checkbox.checked, true);
  checkbox.checked = false;
  await checkbox.dispatch("change");
  assert.equal(JSON.parse(global.localStorage.getItem("symphonai.status-line.v2")).usage, false);
  app.subscription.close();
  const nextDocument = new Document();
  const next = await start({ global, document: nextDocument, client: script.client });
  await find(nextDocument.body, node => node.className === "icon-settings").click();
  assert.equal(find(nextDocument.body, node => node.getAttribute?.("aria-label") === "Usage").checked, false);
  next.subscription.close();
  for (const reason of [null, "grok-4.20-0309-non-reasoning"]) {
    const fresh = browser();
    script.client.conversationStats = async () => ({ conversation: {
      session_id: "session", usage: { total_tokens: 182340, ...(reason ? { unpriced_model: reason } : {}) },
    } });
    const doc = new Document();
    const instance = await start({ global: fresh, document: doc, client: script.client });
    assert.ok(doc.body.textContent.includes(reason ? `no price for ${reason}` : "no prices set"));
    instance.subscription.close();
  }
});

test("conversation costs use standard decimals and a below-cent indicator", async () => {
  for (const [amount, estimated, currency, expected] of [
    ["1.234567", false, "USD", "$1.23"],
    ["0.000428", false, "USD", "<$0.01"],
    ["0", false, "USD", "$0.00"],
    ["0.42", true, "USD", "~$0.42"],
    ["12.345678", false, "USD", "$12.35"],
    ["0.000428", true, "USD", "~<$0.01"],
    ["1.234567", true, "credits", "~credits 1.234567"],
  ]) {
    const script = scriptedClient();
    script.client.openSession = async () => (replayPage("empty"));
    script.client.conversationStats = async () => ({ conversation: {
      session_id: "session", usage: { total_tokens: 10, cost: { amount, currency, estimated } },
    } });
    const document = new Document();
    const app = await start({ global: browser(), document, client: script.client });
    assert.ok(document.body.textContent.includes(`${expected} · 10 tokens`), document.body.textContent);
    app.subscription.close();
  }
});

test("main transcript hides child work but retains exchanges, pending approvals and child edit totals", () => {
  const main = createTranscript({ now: () => 0, rootAgentId: "root" });
  const worker = createTranscript({ now: () => 0, rootAgentId: "child" });
  const event = (type, agent_id, fields = {}) => ({ kind: "event", payload: {
    type, agent_id, run_id: agent_id === "root" ? "root-run" : "child-run", ...fields,
  } });
  main.apply(event("PromptSubmitted", "root", { text: "Person's request", checkpoint_key: "checkpoint" }));
  main.apply(event("AssistantTextDelta", "root", { text: "Leader text" }));
  main.apply(event("PlanChanged", "root", { steps: [{ step: "Delegate", status: "in_progress" }] }));
  main.apply(event("AutoReviewed", "root", { tool_call_id: "root-review", reason: "Leader review" }));
  main.apply(event("ToolCallStarted", "root", { tool_call_id: "dispatch", tool_name: "dispatch_subagent", target: "worker" }));
  main.apply(event("SubagentSpawned", "root", { subagent_name: "worker", subagent_agent_id: "child" }));
  const child = (type, fields = {}) => {
    const frame = event(type, "child", fields);
    main.apply(frame); worker.apply(frame);
  };
  child("RunStarted");
  child("PromptSubmitted", { text: "Worker task" });
  child("ToolCallStarted", { tool_call_id: "read", tool_name: "read_file", target: "fixture.txt" });
  child("PermissionRequested", { tool_call_id: "read", tool_name: "read_file", mode: "ask" });
  assert.ok(main.model.some(entry => entry.type === "question" && !entry.resolved));
  main.recordDecision({ type: "user-question", question: "Child question", answer: "Yes" }, "read");
  child("ToolCallFinished", { tool_call_id: "read", ok: true, output: "Private read output" });
  assert.ok(!main.model.some(entry => entry.type === "question"));
  child("ToolCallStarted", { tool_call_id: "write", tool_name: "write_file", target: "worker.txt" });
  const edit = { tool_call_id: "write", ok: true, result_kind: "file_diff", result_path: "worker.txt",
    lines_added: 3, lines_removed: 1, diff: "-old\n+new" };
  child("ToolCallFinished", edit);
  main.apply(event("ToolCallFinished", "child", edit));
  child("AssistantTextDelta", { text: "Private worker answer" });
  child("AssistantReasoningDelta", { text: "Private worker thought" });
  child("PlanChanged", { steps: [{ step: "Private plan", status: "completed" }] });
  child("AutoReviewed", { reason: "Private review" });
  child("RunFinished", { stopped_reason: "final_response" });
  for (const [kind, text] of [["progress", "Progress message"], ["question", "Agent asks"], ["message", "Leader sent"]]) {
    main.apply(event("SubagentNotice", "root", { subagent_name: "worker", kind, text }));
  }
  main.apply(event("ToolCallFinished", "root", { tool_call_id: "dispatch", ok: true, output: "Delivered result", output_truncated: false }));
  main.apply(event("RunFinished", "root", { stopped_reason: "final_response" }));
  assert.ok(!main.model.some(entry => entry.agentId === "child"));
  assert.ok(!main.model.some(entry => entry.type === "user-question"));
  assert.equal(main.model.filter(entry => entry.type === "subagent-notice").length, 3);
  assert.ok(main.model.some(entry => entry.type === "auto-reviewed" && entry.reason === "Leader review"));
  assert.deepEqual(main.model.find(entry => entry.type === "edit-card").files, [{ path: "worker.txt", added: 3, removed: 1 }]);
  assert.ok(worker.model.some(entry => entry.type === "edit" && entry.path === "worker.txt"));
  assert.ok(worker.model.some(entry => entry.type === "text" && entry.text === "Private worker answer"));
  assert.ok(!worker.model.some(entry => entry.type === "thought" || entry.text === "Private worker thought"));
  assert.ok(worker.model.some(entry => entry.type === "activity" && entry.calls.some(call => call.name === "read_file")));
});

test("dispatch results and failure reasons render equally from supplied live events and history", async () => {
  for (const error of [null, "dispatch refused: missing Validation section", "validation failed after 5 rounds"]) {
    const live = createTranscript({ rootAgentId: "root" });
    const replay = createTranscript({ rootAgentId: "root" });
    const frame = (type, fields) => ({ kind: "event", payload: { type, agent_id: "root", run_id: "run", ...fields } });
    live.apply(frame("PromptSubmitted", { text: "Request" }));
    live.apply(frame("ToolCallStarted", { tool_call_id: "dispatch", tool_name: "dispatch_subagent", target: "worker" }));
    live.apply(frame("ToolCallFinished", { tool_call_id: "dispatch", ok: !error, output: error ? "" : "Delivered result" }));
    if (error) live.apply(frame("ToolCallFailed", { tool_call_id: "dispatch", error }));
    replay.apply(frame("HistoryMessage", { role: "user", text: "Request" }));
    replay.apply(frame("HistoryMessage", { role: "assistant", tool_calls: [{
      id: "dispatch", name: "dispatch_subagent", subagent_name: "worker", ok: !error,
      ...(error ? { error } : { output: "Delivered result", output_truncated: false }),
    }] }));
    const document = new Document();
    const roots = [document.createElement("div"), document.createElement("div")];
    for (const [index, transcript] of [live, replay].entries()) {
      const root = roots[index];
      renderTranscript(document, root, transcript.model, parseMarkdown, {
        rerender: () => renderTranscript(document, root, transcript.model, parseMarkdown),
      });
      if (error) assert.ok(root.textContent.includes(error));
      else {
        assert.match(root.textContent, /worker \(finished\)/);
        assert.ok(!root.textContent.includes("Delivered result"));
        await find(root, node => node.getAttribute?.("aria-expanded") === "false").click();
        assert.ok(root.textContent.includes("Delivered result"));
      }
    }
    assert.equal(roots[0].textContent, roots[1].textContent);
  }
});

test("Roadmap is read-only, refreshes on changes and opens its linked loop", async () => {
  const script = scriptedClient();
  let goals = [];
  script.client.roadmap = async () => ({ goals });
  script.client.loops = async () => ({ loops: [{ id: "loop_fixture", goal: "x", state: "awaiting_plan" }] });
  const opened = [];
  script.client.loop = async id => { opened.push(id); return { loop: { id, goal: "x", state: "awaiting_plan", cards: [], lanes: [] } }; };
  const document = new Document();
  const app = await start({ global: browser(), document, client: script.client });
  await app.openTool("Roadmap");
  const root = document.getElementById("roadmap");
  assert.match(root.textContent, /No goals yet. Ask the leader to add one./);
  assert.equal(find(root, node => node.tagName === "INPUT"), null);
  goals = [{ id: "goal_fixture", goal: "x", loop_id: "loop_fixture", state: "awaiting_plan" }];
  await script.emit("LoopChanged", { change: "roadmap", loop_id: "", state: "" });
  assert.match(root.textContent, /Awaiting plan/);
  for (const label of ["Add goal", "Move up", "Move down", "Remove", "Start"]) {
    assert.equal(find(root, node => node.tagName === "BUTTON" && node.textContent === label), null);
  }
  await find(root, node => node.className === "roadmap-goal-link").click();
  await new Promise(resolve => setImmediate(resolve));
  assert.ok(opened.includes("loop_fixture"));
  app.subscription.close();
});

test("prompt renders the reply and keeps tool output collapsed until opened", async () => {
  const document = new Document();
  const script = scriptedClient();
  await start({ global: browser(), document, client: script.client });
  await submit(document, "please run the fixture");
  assert.deepEqual(script.prompts, ["please run the fixture"]);
  await script.emit("RunStarted", { parent_run_id: null });
  await script.emit("ToolCallStarted", { tool_call_id: "shell-core", tool_name: "run_shell", target: "printf" });
  await script.emit("ToolCallFinished", { tool_call_id: "shell-core", tool_name: "run_shell", ok: true, output: "core tool output" });
  await script.emit("AssistantTextDelta", { text: "core assistant reply" });
  await script.emit("RunFinished", { stopped_reason: "final_response", turns_used: 2 });
  const chat = document.getElementById("chat");
  assert.match(chat.textContent, /core assistant reply/);
  const activity = find(chat, (node) => node.getAttribute?.("aria-expanded") === "false");
  assert.ok(activity, "tool output has an accessible collapsed control");
  assert.ok(!chat.textContent.includes("core tool output"));
  await activity.dispatch("click");
  assert.match(chat.textContent, /core tool output/);
});

test("slash new invokes the session action", async () => {
  const document = new Document();
  const script = scriptedClient();
  await start({ global: browser(), document, client: script.client });
  await submit(document, "/new");
  assert.equal(script.newSessions(), 1);
  assert.deepEqual(script.prompts, []);
});

test("the real client preserves host errors for conversation and session creation", async () => {
  const document = new Document();
  const global = browser();
  global.__symphonai = { port: 4312, token: "core-inert-token" };
  const calls = [];
  const values = {
    "/project": { repo_root: "/core-project", name: "Core project" },
    "/settings": { settings: { mode: "allow", providers: [{ name: "openai", key_present: true }] } },
    "/sessions": [], "/roadmap": { goals: [] }, "/approvals": { pending: [] },
    "/health": { state: "idle", runtime_run_id: null }, "/changes": { turns: [], files: [], worktrees: [] },
  };
  global.fetch = async (url) => {
    const path = new URL(url).pathname;
    calls.push(path);
    if (path === "/events") return { status: 200, body: new ReadableStream({ start() {} }) };
    if (["/conversation", "/session/new"].includes(path)) return {
      status: 409, json: async () => ({ error: `core host detail: ${path}` }),
    };
    return { status: 200, json: async () => values[path] ?? {} };
  };
  const app = await start({ global, document });
  try {
    await submit(document, "/cost");
    assert.match(document.getElementById("chat").textContent, /core host detail: \/conversation/);
    await submit(document, "/new");
    assert.match(document.getElementById("prompt-error").textContent, /core host detail: \/session\/new/);
    assert.ok(calls.includes("/session/new"));
  } finally {
    app.subscription.close();
  }
});

test("subagent notices render equally live and replayed, with rail progress and questions", async () => {
  const document = new Document();
  const script = scriptedClient();
  await start({ global: browser(), document, client: script.client });
  await script.emit("RunStarted");
  await script.emit("SubagentSpawned", { subagent_name: "worker", subagent_agent_id: "child" });
  await script.emit("RunStarted", { agent_id: "child", agent_name: "worker", run_id: "child-run" });
  const rail = document.getElementById("agents");
  assert.match(rail.textContent, /worker.*running/);
  const kinds = ["progress", "question", "message", "finished"];
  const texts = ["Reading existing.txt\nMore progress", "Which file?\nQuestion details", "Use existing.txt\nMessage details",
    JSON.stringify({ ok: true, content: "Complete result\nValidation: passed", error: null })];
  for (const [index, kind] of kinds.entries()) {
    await script.emit("SubagentNotice", {
      subagent_name: "worker", subagent_agent_id: "child", kind, text: texts[index],
    });
    if (kind === "progress") assert.match(rail.textContent, /Reading existing.txt/);
    if (kind === "question") assert.match(rail.textContent, /Asks: Which file\?/);
    if (kind === "message") assert.ok(!rail.textContent.includes("Asks: Which file?"));
  }
  const chat = document.getElementById("chat");
  const notices = () => chat.children.filter((node) => node.className.split(" ").includes("subagent-notice"));
  const expected = ["worker: Reading existing.txt", "worker asks: Which file?", "To worker: Use existing.txt", "worker finished"];
  assert.deepEqual(notices().map((node) => node.children[0].textContent), expected);
  assert.equal(chat.children.filter((node) => node.className === "prompt").length, 0);
  assert.ok(!chat.textContent.includes("Complete result"));
  const lastToggle = notices().at(-1);
  assert.equal(lastToggle.getAttribute("aria-expanded"), "false");
  await lastToggle.dispatch("click");
  assert.match(chat.textContent, /Complete result\nValidation: passed/);
  assert.equal(notices().at(-1).getAttribute("aria-expanded"), "true");
  assert.match(rail.textContent, /finished/);

  await script.emit("SubagentSpawned", { subagent_name: "stopped-worker", subagent_agent_id: "closed-child" });
  await script.emit("RunFinished", { agent_id: "closed-child", run_id: "closed-run", stopped_reason: "cancelled" });
  await script.emit("SubagentStopped", { subagent_name: "stopped-worker", subagent_agent_id: "closed-child" });
  assert.match(rail.textContent, /stopped-worker.*closed/);
  await script.emit("SubagentSpawned", { subagent_name: "failed-worker", subagent_agent_id: "failed-child" });
  await script.emit("RunFailed", { agent_id: "failed-child", run_id: "failed-run" });
  await script.emit("SubagentStopped", { subagent_name: "failed-worker", subagent_agent_id: "failed-child" });
  assert.match(rail.textContent, /failed-worker.*failed/);
  assert.equal(find(rail, (node) => node.tagName === "BUTTON"), null);

  const replayDocument = new Document();
  const replay = scriptedClient();
  await start({ global: browser(), document: replayDocument, client: replay.client });
  const headers = ["[subagent worker progress]", "[subagent worker asks]", "[message from the leader]", "[subagent worker finished]"];
  for (const [index, header] of headers.entries()) {
    await replay.emit("HistoryMessage", {
      role: "user", notice: true, text: `${header}\n${index === 2 ? "To worker: " : ""}${texts[index]}`,
      tool_calls: [],
    });
  }
  const replayChat = replayDocument.getElementById("chat");
  const replayNotices = replayChat.children.filter((node) => node.className.split(" ").includes("subagent-notice"));
  assert.deepEqual(replayNotices.map((node) => node.children[0].textContent), expected);
  assert.equal(replayChat.children.filter((node) => node.className === "prompt").length, 0);
  await replayNotices.at(-1).dispatch("keydown", { key: "Enter" });
  assert.match(replayChat.textContent, /Complete result\nValidation: passed/);
  await replay.emit("HistoryMessage", { role: "user", notice: false, text: "A person's real prompt", tool_calls: [] });
  assert.equal(replayChat.children.filter((node) => node.className === "prompt").length, 1);
});


test("sidebar unread and waiting marks persist on reload and opening records viewed", async () => {
  const script = scriptedClient();
  const sessions = [
    { run_id: "past", repo_root: "/core-project", title: "Finished work", unread: true, activity: "idle" },
    { run_id: "waiting", repo_root: "/core-project", title: "Needs approval", unread: true, activity: "waiting" },
  ];
  script.client.sessions = async (_limit, _query, archived) => archived ? [] : sessions;
  let opened = null;
  const viewed = [];
  script.client.openSession = async (id) => { opened = id; return replayPage("empty"); };
  script.client.conversationStats = async () => ({ conversation: opened ? { session_id: opened, agents: [] } : null });
  script.client.sessionViewed = async (id) => {
    viewed.push(id);
    sessions.find((session) => session.run_id === id).unread = false;
    return { session_id: id };
  };
  for (let reload = 0; reload < 2; reload += 1) {
    const document = new Document();
    const app = await start({ global: browser(), document, client: script.client });
    assert.ok(find(document.body, (node) => node.className === "session-dot unread"));
    assert.ok(find(document.body, (node) => node.className === "session-dot waiting"));
    assert.match(find(document.body, (node) => node.className === "projects").textContent, /Needs approval waiting/);
    app.subscription.close();
  }
  const document = new Document();
  const app = await start({ global: browser(), document, client: script.client });
  await find(document.body, (node) => node.className === "session-link" && node.textContent === "Finished work").click();
  assert.deepEqual(viewed, ["past"]);
  assert.equal(find(document.body, (node) => node.className === "session-dot unread"), null);
  assert.ok(find(document.body, (node) => node.className === "session-dot waiting"));
  app.subscription.close();
});


test("stats opens Usage with vendor and estimated costs, balances, errors and three ranges", async () => {
  const document = new Document();
  const script = scriptedClient();
  const ranges = [];
  script.client.usage = async (days) => {
    ranges.push(days);
    return {
      providers: [
        { provider: "anthropic", input_tokens: 125618, output_tokens: 20, cache_read_tokens: 10,
          cost: { amount: "3.99", exact_amount: "3.99473345", currency: "USD", source: "vendor", data_refreshed_at: "2026-10-01T04:02:11Z" },
          models: [{ model: "claude-opus-5-5", input_tokens: 125618, output_tokens: 20, cache_read_tokens: 10 }] },
        { provider: "gemini", cost: { amount: "0.25", currency: "USD", source: "estimate" }, models: [] },
        { provider: "kimi", balance: [{ amount: "9.98959", currency: null }], models: [] },
        { provider: "openai", cost: { amount: "10.66", exact_amount: "10.66323600000000000000000000", currency: "USD", source: "vendor" }, models: [] },
        { provider: "deepseek", balance: [{ amount: "4.90", currency: "USD" }], models: [] },
        { provider: "grok", cost: { amount: "0.00", exact_amount: "0.004", currency: "USD", source: "estimate" }, models: [] },
        { provider: "qwen", error: "Vendor cost request returned HTTP 503.", models: [] },
        { provider: "ollama", models: [] }, { provider: "groq", models: [] },
        { provider: "together", models: [] }, { provider: "openrouter", models: [] },
        { provider: "unknown", models: [], cost: null },
      ],
      daily: [{ date: "2026-11-02", input_tokens: 100, output_tokens: 20 }],
      projects: [{ repo_root: "/core-project", input_tokens: 100, output_tokens: 20, cache_read_tokens: 10, cache_write_tokens: 0 }],
    };
  };
  const app = await start({ global: browser(), document, client: script.client });
  await submit(document, "/stats");
  assert.equal(app.route().page, "usage");
  const pane = find(document.body, (node) => node.className === "usage-page");
  const money = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
  assert.ok(pane.textContent.includes(`${money.format(3.99)} · vendor`));
  assert.ok(pane.textContent.includes(`${money.format(10.66)} · vendor`));
  assert.ok(pane.textContent.includes(`${money.format(0.25)} · estimate`));
  assert.ok(pane.textContent.includes(`<${money.format(0.01)} · estimate`));
  assert.ok(pane.textContent.includes(new Intl.NumberFormat().format(125618)));
  for (const name of ["Anthropic", "OpenAI", "Google Gemini", "DeepSeek", "Moonshot (Kimi)",
    "xAI (Grok)", "OpenRouter", "Alibaba Qwen (DashScope)", "Ollama (local)", "Groq", "Together AI"]) {
    assert.ok(find(pane, (node) => node.tagName === "H2" && node.textContent === name), name);
  }
  assert.match(pane.textContent, /Vendor data refreshed at 2026-10-01T04:02:11Z/);
  assert.match(pane.textContent, /Moonshot \(Kimi\)/);
  assert.ok(pane.textContent.includes(`${new Intl.NumberFormat(undefined, { maximumFractionDigits: 20 }).format(9.98959)} balance`));
  assert.match(pane.textContent, /HTTP 503/);
  assert.ok(find(pane, (node) => node.className === "usage-bar"));
  for (const range of [7, 90]) {
    await find(pane, (node) => node.tagName === "BUTTON" && node.textContent === `${range} days`).click();
  }
  assert.deepEqual(ranges, [30, 7, 90]);
  assert.deepEqual(script.prompts, []);
  app.subscription.close();
});

test("preview marks stay in the composer and send together as one prompt", async () => {
  const document = new Document();
  const script = scriptedClient();
  const global = browser();
  const opened = [];
  global.open = (_url, _target, features) => {
    const window = { location: { href: '' }, opener: {}, close() {} };
    opened.push({ window, features });
    return window;
  };
  let marks = [];
  const starts = [], removed = [];
  script.client.previewStart = async (url) => { starts.push(url); return { preview_url: 'http://127.0.0.1:4321/' }; };
  script.client.previewStop = async () => ({ stopped: true });
  script.client.previewMarks = async () => ({ session_id: null, marks });
  script.client.removePreviewMarks = async (ids, sessionId) => {
    removed.push([ids, sessionId]);
    marks = marks.filter((mark) => !ids.includes(mark.id));
  };
  const app = await start({ global, document, client: script.client });
  await submit(document, '/preview http://127.0.0.1:3000');
  assert.deepEqual(starts, ['http://127.0.0.1:3000']);
  assert.equal(opened[0].window.location.href, 'http://127.0.0.1:4321/');
  assert.equal(opened[0].window.opener, null);
  marks = [
    { id: 'one', text: 'Button', comment: 'Make it blue', selector: '#submit',
      html: '<button id="submit">Button</button>', url: 'http://127.0.0.1:4321/', source_hint: 'src/Button.tsx', source_guess: true },
    { id: 'two', text: 'Heading', comment: 'Larger title', selector: 'h1',
      html: '<h1>Heading</h1>', url: 'http://127.0.0.1:4321/', source_hint: 'src/App.tsx:8', source_guess: false },
  ];
  await script.emit('PreviewMarkAdded', { repo_root: '/core-project', session_id: null, mark: marks[0] });
  const composer = document.getElementById('prompt-attachments');
  assert.match(composer.textContent, /Button · Make it blue/);
  assert.match(composer.textContent, /Heading · Larger title/);
  assert.ok(!app.transcript.model.some((entry) => entry.type === 'prompt'));
  await submit(document, 'please fix these');
  assert.equal(script.prompts.length, 1);
  for (const text of ['please fix these', 'Make it blue', 'Larger title', '#submit', 'h1',
    '<button id="submit">Button</button>', '<h1>Heading</h1>', 'Source (guess): src/Button.tsx', 'Source: src/App.tsx:8']) {
    assert.ok(script.prompts[0].includes(text), text);
  }
  assert.deepEqual(removed, [[['one', 'two'], null]]);
  assert.equal(find(composer, (node) => node.className === 'attachment-chip preview-mark-chip'), null);
  app.subscription.close();
});


test("leader notes show before the answer and replay without becoming prompts", async () => {
  const document = new Document();
  const script = scriptedClient();
  const app = await start({ global: browser(), document, client: script.client });
  await script.emit('RunStarted');
  await script.emit('PersonNotice', { text: 'halfway' });
  assert.match(document.getElementById('chat').textContent, /Leader: halfway/);
  assert.equal(app.transcript.model.filter((entry) => entry.type === 'prompt').length, 0);
  await script.emit('AssistantTextDelta', { text: 'Finished work' });
  const live = document.getElementById('chat').textContent;
  assert.ok(live.indexOf('Leader: halfway') < live.indexOf('Finished work'));
  app.transcript.reset();
  app.transcript.setRootAgentId("core-agent");
  await script.emit('HistoryMessage', { role: 'assistant', text: '', tool_calls: [
    { id: 'note', name: 'notify_person', text: 'halfway' },
  ] });
  await script.emit('HistoryMessage', { role: 'assistant', text: 'Finished work', tool_calls: [] });
  assert.match(document.getElementById('chat').textContent, /Leader: halfway/);
  assert.equal(app.transcript.model.filter((entry) => entry.type === 'person-notice').length, 1);
  assert.equal(app.transcript.model.filter((entry) => entry.type === 'prompt').length, 0);
  app.subscription.close();
});


test("real runtime dispatch events carry masked bounded output while other hidden tools stay hidden", async () => {
  const payload = JSON.parse(execFileSync("python3", ["-c", String.raw`
import json, tempfile
from dataclasses import asdict
from pathlib import Path
from symphonai_api.agent_loop import ApiAgent
from symphonai_api.models import Message, ModelResponse, Role, ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata

class ResultTool(LocalTool):
    name = "fixture"
    description = "Scripted result"
    parameters = {"type": "object", "properties": {}}
    def __init__(self, name, text):
        self.name, self.text = name, text
    def metadata(self, arguments):
        return ToolMetadata(ToolEffect.READ_ONLY, concurrency_safe=True, paths=None)
    def _execute(self, call, policy, cancel=None):
        return ToolResult(call.id, True, content=self.text)

secret = "sk-" + "Z" * 32
text = "Worker answer: " + secret + " " + "x" * 4100
names = ["dispatch_subagent", "update_plan", "ask_user", "propose_plan", "write_file", "edit_file", "multi_edit_file", "apply_patch"]
events = []
with tempfile.TemporaryDirectory() as root:
    provider = FakeModelProvider([
        ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall(name, name, {}) for name in names])),
        ModelResponse(Message(Role.ASSISTANT, "Leader answer")),
    ])
    ApiAgent(provider, {name: ResultTool(name, text) for name in names},
             PermissionPolicy(Path(root)), events=events.append).run([Message(Role.USER, "Request")])
print(json.dumps({"secret": secret, "events": [{"type": type(event).__name__, **asdict(event)} for event in events]}))
`], { cwd: new URL("../../", import.meta.url), encoding: "utf8" }));
  const finished = payload.events.filter((event) => event.type === "ToolCallFinished");
  assert.equal(finished.length, 8);
  const dispatch = finished.find((event) => event.tool_name === "dispatch_subagent");
  assert.equal(dispatch.output.length, 4000);
  assert.equal(dispatch.output_truncated, true);
  assert.ok(!dispatch.output.includes(payload.secret));
  assert.ok(dispatch.output.includes("[REDACTED_SECRET]"));
  assert.ok(finished.filter((event) => event.tool_name !== "dispatch_subagent").every((event) => event.output === "" && !event.output_truncated));
  const main = createTranscript({ rootAgentId: dispatch.agent_id });
  const replay = createTranscript({ rootAgentId: dispatch.agent_id });
  const frame = (type, fields) => ({ kind: "event", payload: { type, agent_id: dispatch.agent_id, run_id: dispatch.run_id, ...fields } });
  main.apply(frame("RunStarted", { agent_name: "leader" }));
  main.apply(frame("ToolCallStarted", { tool_call_id: dispatch.tool_call_id, tool_name: "dispatch_subagent", target: "worker" }));
  main.apply(frame("ToolCallFinished", dispatch));
  replay.apply(frame("HistoryMessage", { role: "assistant", tool_calls: [{
    id: dispatch.tool_call_id, name: "dispatch_subagent", subagent_name: "worker", ok: true,
    output: dispatch.output, output_truncated: dispatch.output_truncated,
  }] }));
  const document = new Document();
  const roots = [];
  for (const transcript of [main, replay]) {
    const call = transcript.model.find((entry) => entry.type === "activity").calls[0];
    assert.equal(call.output, dispatch.output);
    assert.equal(call.outputTruncated, true);
    assert.ok(!JSON.stringify(transcript.model).includes(payload.secret));
    const root = document.createElement("div");
    renderTranscript(document, root, transcript.model, parseMarkdown, {
      rerender: () => renderTranscript(document, root, transcript.model, parseMarkdown),
    });
    assert.ok(!root.textContent.includes("Worker answer:"));
    await find(root, (node) => node.getAttribute?.("aria-expanded") === "false").click();
    assert.ok(root.textContent.includes("Worker answer: [REDACTED_SECRET]"));
    assert.ok(!root.textContent.includes(payload.secret));
    roots.push(root);
  }
  assert.equal(roots[0].textContent, roots[1].textContent);
});


test("transcript waits for explicit leader and session facts rather than the first child event", () => {
  const transcript = createTranscript();
  const frame = (type, agent_id, session_id, fields = {}) => ({ kind: "event", payload: {
    type, agent_id, session_id, run_id: `${agent_id}-run`, ...fields,
  } });
  transcript.apply(frame("AssistantTextDelta", "child", "wanted", { text: "PRIVATE CHILD" }));
  transcript.apply(frame("PromptSubmitted", "root", "other", { text: "OTHER CONVERSATION" }));
  transcript.apply(frame("PromptSubmitted", "root", "wanted", { text: "Request" }));
  transcript.apply(frame("AssistantTextDelta", "root", "wanted", { text: "Leader" }));
  assert.deepEqual(transcript.model, []);
  const held = transcript.snapshot();
  transcript.reset();
  transcript.restore(held);
  transcript.setSessionId("wanted");
  assert.deepEqual(transcript.model, []);
  transcript.setRootAgentId("root");
  assert.deepEqual(transcript.model.filter((entry) => entry.type === "prompt" || entry.type === "text").map((entry) => entry.text), ["Request", "Leader"]);
  assert.ok(!JSON.stringify(transcript.model).includes("PRIVATE CHILD"));
  assert.ok(!JSON.stringify(transcript.model).includes("OTHER CONVERSATION"));
});

test("new chat holds early frames, pins the recorded leader and replays the same exchanges", async () => {
  const document = new Document();
  const script = scriptedClient();
  const global = browser();
  const frame = animationFrames(global);
  let serverConversation = null;
  let reply;
  let requestStarted;
  const started = new Promise((resolve) => { requestStarted = resolve; });
  const pendingReply = new Promise((resolve) => { reply = resolve; });
  const agents = [
    { agent_id: "child", name: "worker", parent_agent_id: "root", state: "running" },
    { agent_id: "root", name: "leader", parent_agent_id: null, state: "running" },
  ];
  const emit = async (type, agent, fields = {}) => {
    await script.emit(type, {
      agent_id: agent, agent_name: agent === "root" ? "leader" : "worker",
      run_id: `${agent}-run`, session_id: "new-session", ...fields,
    });
    await frame();
  };
  script.client.conversationStats = async () => ({ conversation: serverConversation });
  script.client.sessions = async (_limit, _query, archived) => archived ? [] : [{
    run_id: "new-session", title: "Test conversation", repo_root: "/core-project", activity: "idle",
  }];
  script.client.prompt = async () => {
    serverConversation = { session_id: "new-session", agents };
    await emit("PromptSubmitted", "root", { text: "Please work" });
    await emit("RunStarted", "root");
    await emit("AssistantTextDelta", "root", { text: "LEADER-TEXT delegating" });
    await emit("ToolCallStarted", "root", { tool_call_id: "dispatch", tool_name: "dispatch_subagent", target: "worker" });
    await emit("SubagentSpawned", "root", { subagent_name: "worker", subagent_agent_id: "child" });
    await emit("RunStarted", "child");
    await emit("ToolCallStarted", "child", { tool_call_id: "read", tool_name: "read_file", target: "in.txt" });
    await emit("ToolCallFinished", "child", { tool_call_id: "read", tool_name: "read_file", ok: true, output: "PRIVATE READ" });
    await emit("ToolCallStarted", "child", { tool_call_id: "write", tool_name: "write_file", target: "out.txt" });
    await emit("PermissionRequested", "child", { tool_call_id: "write", tool_name: "write_file", target: "out.txt" });
    requestStarted();
    return pendingReply;
  };
  script.client.agentHistory = async () => ({ messages: [
    { role: "assistant", text: "", tool_calls: [{ id: "read", name: "read_file", target: "in.txt", ok: true, output: "PRIVATE READ" }] },
  ] });
  const app = await start({ global, document, client: script.client });
  await frame();
  const sending = submit(document, "Please work");
  await started;
  assert.ok(!document.getElementById("chat").textContent.includes("PRIVATE READ"));
  reply({ accepted: true, run_id: "new-session" });
  await sending;
  await frame();
  const chat = document.getElementById("chat");
  assert.match(chat.textContent, /Please work/);
  assert.match(chat.textContent, /LEADER-TEXT delegating/);
  assert.ok(app.transcript.model.some((entry) => entry.type === "question" && !entry.resolved));
  await find(document.body, (node) => node.getAttribute?.("aria-label") === "Open worker's conversation").click();
  await frame();
  assert.match(chat.textContent, /in.txt/);
  await emit("ToolCallStarted", "child", { tool_call_id: "write", tool_name: "write_file", target: "out.txt" });
  await emit("ToolCallFinished", "child", { tool_call_id: "write", tool_name: "write_file", ok: true,
    result_kind: "file_diff", result_path: "out.txt", lines_added: 1, lines_removed: 0, diff: "+output" });
  await emit("AssistantTextDelta", "child", { text: "PRIVATE WORKER ANSWER" });
  assert.match(chat.textContent, /out.txt/);
  assert.match(chat.textContent, /PRIVATE WORKER ANSWER/);
  await emit("RunFinished", "child", { stopped_reason: "final_response" });
  await emit("ToolCallFinished", "root", { tool_call_id: "dispatch", tool_name: "dispatch_subagent", ok: true,
    output: "Worker finished [REDACTED_SECRET]", output_truncated: false });
  await emit("AssistantTextDelta", "root", { text: "LEADER-FINAL all done" });
  await find(document.body, (node) => node.getAttribute?.("aria-label") === "Back to the conversation").click();
  await frame();
  assert.match(chat.textContent, /Dispatched worker \(finished\)/);
  assert.match(chat.textContent, /LEADER-FINAL all done/);
  assert.ok(!chat.textContent.includes("PRIVATE"));
  assert.ok(!chat.textContent.includes("Worker finished"));
  assert.ok(chat.textContent.indexOf("Please work") < chat.textContent.indexOf("LEADER-TEXT"));
  assert.ok(chat.textContent.indexOf("LEADER-TEXT") < chat.textContent.indexOf("Dispatched worker"));
  assert.ok(chat.textContent.indexOf("Dispatched worker") < chat.textContent.indexOf("LEADER-FINAL"));
  script.client.openSession = async () => replayPage("saved", [
    historyFrame({ session_id: "new-session", replay_id: "saved", agent_id: "root", role: "user", text: "Please work" }),
    historyFrame({ session_id: "new-session", replay_id: "saved", agent_id: "root", role: "assistant", text: "LEADER-TEXT delegating", tool_calls: [{
      id: "dispatch", name: "dispatch_subagent", subagent_name: "worker", ok: true,
      output: "Worker finished [REDACTED_SECRET]", output_truncated: false,
    }] }),
    historyFrame({ session_id: "new-session", replay_id: "saved", agent_id: "root", role: "assistant", text: "LEADER-FINAL all done", tool_calls: [] }),
  ]);
  await find(document.body, (node) => node.className === "session-link" && node.textContent === "Test conversation").click();
  await frame();
  assert.match(chat.textContent, /Please work/);
  assert.match(chat.textContent, /LEADER-TEXT delegating/);
  assert.match(chat.textContent, /Dispatched worker \(finished\)/);
  assert.match(chat.textContent, /LEADER-FINAL all done/);
  assert.ok(!chat.textContent.includes("PRIVATE"));
  app.subscription.close();
});


test("agent history replays writes with path and state without exposing edit content", () => {
  const saved = JSON.parse(execFileSync("python3", ["-c", String.raw`
import json, tempfile
from pathlib import Path
from symphonai_api.agent_loop import ApiAgent
from symphonai_api.identity import new_id, new_agent_ref
from symphonai_api.models import Message, ModelResponse, Role, ToolCall
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.runner import standard_tool_registry
from symphonai_api.session import SessionStore
from symphonai_host.broker import EventBroker
from symphonai_host.run import HostRun

with tempfile.TemporaryDirectory() as temporary:
    base = Path(temporary)
    root = base / "project"
    root.mkdir()
    (root / "a.txt").write_text("alpha")
    policy = PermissionPolicy(root, allowed_write_scope=[root])
    sessions = base / "sessions"
    session_id = new_id("run")
    agent = new_agent_ref("worker")
    store = SessionStore(sessions, session_id, repo_root=root)
    provider = FakeModelProvider([
        ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("read", "read_file", {"path": "a.txt"})])),
        ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("write", "write_file", {"path": "out.txt", "content": "UNEXPOSED_EDIT_CONTENT"})])),
        ModelResponse(Message(Role.ASSISTANT, "", tool_calls=[ToolCall("failed", "write_file", {"path": str(base / "outside.txt"), "content": "UNEXPOSED_EDIT_CONTENT"})])),
        ModelResponse(Message(Role.ASSISTANT, "WORKER ANSWER")),
    ])
    ApiAgent(provider, standard_tool_registry(), policy, agent_ref=agent,
             transcript=store.writer_for(agent.agent_id)).run([Message(Role.USER, "Worker task")])
    store.close()
    host = HostRun(FakeModelProvider(), policy, EventBroker(), sessions_root=sessions,
                   memory_root=base / "memory", memory_extraction_enabled=False)
    try:
        history = host.agent_history(session_id, agent.agent_id)
        print(json.dumps({"agent_id": agent.agent_id, "session_id": session_id, "messages": history["messages"]}))
    finally:
        host.close()
`], { cwd: new URL("../../", import.meta.url), encoding: "utf8" }));
  const calls = saved.messages.flatMap((message) => message.tool_calls ?? []);
  const written = calls.find((call) => call.id === "write");
  const failed = calls.find((call) => call.id === "failed");
  assert.equal(written.target, "out.txt");
  assert.equal(written.ok, true);
  assert.equal(failed.ok, false);
  for (const call of [written, failed]) {
    assert.ok(!Object.hasOwn(call, "output"));
    assert.ok(!Object.hasOwn(call, "diff"));
    assert.ok(!JSON.stringify(call).includes("UNEXPOSED_EDIT_CONTENT"));
  }
  const worker = createTranscript({ rootAgentId: saved.agent_id });
  const main = createTranscript({ rootAgentId: "leader" });
  worker.setSessionId(saved.session_id);
  main.setSessionId(saved.session_id);
  for (const message of saved.messages) {
    const frame = { kind: "event", payload: { ...message, type: "HistoryMessage", agent_id: saved.agent_id, session_id: saved.session_id } };
    worker.apply(frame);
    main.apply(frame);
  }
  const document = new Document();
  const view = document.createElement("div");
  renderTranscript(document, view, worker.model, parseMarkdown);
  assert.match(view.textContent, /Read a.txt/);
  assert.match(view.textContent, /Wrote out.txt/);
  assert.match(view.textContent, /WORKER ANSWER/);
  assert.ok(view.textContent.indexOf("Read a.txt") < view.textContent.indexOf("Wrote out.txt"));
  assert.ok(view.textContent.indexOf("Wrote out.txt") < view.textContent.indexOf("WORKER ANSWER"));
  assert.ok(worker.model.some((entry) => entry.type === "activity" && entry.calls.some((call) => call.toolCallId === "failed" && call.status === "failed")));
  assert.ok(!JSON.stringify(main.model).includes("out.txt"));
  assert.ok(!JSON.stringify(worker.model).includes("UNEXPOSED_EDIT_CONTENT"));
});


test("checkpoint-backed worker writes form one matching live and replayed edit card", () => {
  const fixture = JSON.parse(execFileSync("python3", ["-c", String.raw`
import json, subprocess, tempfile
from pathlib import Path
from symphonai_api.models import Message, ModelResponse, Role, ToolCall
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_host.broker import EventBroker
from symphonai_host.run import HostRun
from symphonai_host.protocol import encode_event

class Script(FakeModelProvider):
    def __init__(self):
        super().__init__()
        self.parent_calls = self.child_calls = 0
    def create_response(self, request, *, cancel=None):
        names = {tool.get('name', tool.get('function', {}).get('name')) for tool in request.tools}
        if 'dispatch_subagent' in names:
            self.parent_calls += 1
            calls = [] if self.parent_calls > 1 else [
                ToolCall('dispatch-' + name, 'dispatch_subagent', {'subagent_name': name,
                    'task': 'Write ' + path + '\n\n## Validation\n' + chr(96) * 3 + 'sh\ntrue\n' + chr(96) * 3})
                for name, path in [('worker', 'a.txt'), ('implementer', 'b.txt')]]
            return ModelResponse(Message(Role.ASSISTANT, 'All done' if not calls else 'Delegating', tool_calls=calls))
        index = self.child_calls
        self.child_calls += 1
        path = 'a.txt' if index < 2 else 'b.txt'
        calls = [] if index % 2 else [ToolCall('write-' + path, 'write_file', {'path': path, 'content': 'one line\n'})]
        return ModelResponse(Message(Role.ASSISTANT, 'Worker finished' if not calls else '', tool_calls=calls))

def encode(event):
    return event.payload() if hasattr(event, 'payload') else encode_event(event)

with tempfile.TemporaryDirectory() as temporary:
    base = Path(temporary)
    root = base / 'project'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    broker = EventBroker()
    run = HostRun(Script(), PermissionPolicy(root, allowed_write_scope=[root], shell_enabled=True, shell_allowlist=[()]),
                  broker, sessions_root=base/'sessions', memory_root=base/'memory', memory_extraction_enabled=False)
    subscription = broker.subscribe()
    try:
        run.select_mode('allow')
        session = run.start('Please write both files')
        live = []
        while True:
            event = subscription.get(timeout=10)
            if event is None:
                raise AssertionError('scripted run did not finish')
            live.append(encode(event))
            if type(event).__name__ in ('RunFinished', 'RunFailed') and getattr(event, 'agent_name', '') == 'leader':
                assert type(event).__name__ == 'RunFinished'
                leader = event.agent_id
                break
        page = run.open_session(session)
        replay = [frame['payload'] for frame in page['history']]
        while page['next_offset'] is not None:
            page = run.history_page(page['replay_id'], page['next_offset'])
            replay.extend(frame['payload'] for frame in page['history'])
        changes = run.changes()
        undone = run.revert_changes(key=changes['turns'][-1]['key'])
        assert not (root / 'a.txt').exists() and not (root / 'b.txt').exists()
        print(json.dumps({'session': session, 'leader': leader, 'live': live, 'replay': replay,
                          'changes': changes, 'undone': undone}))
    finally:
        subscription.close()
        run.close()
`], { cwd: new URL("../../", import.meta.url), encoding: "utf8" }));
  const cards = [];
  for (const events of [fixture.live, fixture.replay]) {
    const transcript = createTranscript({ rootAgentId: fixture.leader });
    transcript.setSessionId(fixture.session);
    for (const payload of events) transcript.apply({ kind: "event", payload });
    const edits = transcript.model.filter((entry) => entry.type === "edit-card");
    assert.equal(edits.length, 1, JSON.stringify({ phase: events === fixture.live ? "live" : "replay",
      changes: fixture.changes, summaries: events.filter((event) => event.type === "HistoryMessage" || event.type === "PromptSubmitted") }));
    assert.deepEqual(edits[0].files, [{ path: "a.txt", added: 1, removed: 0 }, { path: "b.txt", added: 1, removed: 0 }]);
    assert.ok(fixture.changes.turns.some((turn) => turn.key === edits[0].checkpointKey && turn.paths.includes("a.txt") && turn.paths.includes("b.txt")));
    assert.ok(!transcript.model.some((entry) => entry.type === "activity" && entry.calls.some((call) => call.name === "write_file")));
    const document = new Document();
    const chat = document.getElementById("chat");
    const pending = [];
    renderTranscript(document, chat, transcript.model, parseMarkdown, {}, {
      requestAnimationFrame: callback => pending.push(callback),
    });
    assert.equal(chat.textContent, "", "edit-card DOM is frame-batched");
    pending.splice(0).forEach(callback => callback());
    assert.equal(chat.children.filter(node => node.className === "edit-card").length, 1);
    const card = find(chat, node => node.className === "edit-card");
    assert.match(card.textContent, /a.txt/);
    assert.match(card.textContent, /b.txt/);
    cards.push(edits[0]);
  }
  assert.equal(cards[0].checkpointKey, cards[1].checkpointKey);
  assert.equal(cards[0].canUndo, true);
  assert.deepEqual(fixture.undone.reverted.sort(), ["a.txt", "b.txt"]);
});


test("a steered replay keeps the original turn's edit card at the end", () => {
  const files = [{ path: "out.txt", added: 2, removed: 0 }];
  for (const steering of [false, true]) {
    const transcript = createTranscript({ rootAgentId: "root" });
    transcript.setSessionId("saved");
    const apply = fields => transcript.apply(historyFrame({
      session_id: "saved", agent_id: "root", ...fields,
    }));
    apply({ role: "user", text: "Original request", checkpoint_key: "prompt-key", edits: files });
    apply({ role: "assistant", text: "Writing", tool_calls: [{
      id: "write", name: "write_file", target: "out.txt", ok: true,
    }] });
    if (steering) apply({ role: "user", text: "Keep both lines" });
    assert.equal(transcript.model.filter(entry => entry.type === "edit-card").length, 0);
    apply({ role: "assistant", text: "Final answer", tool_calls: [] });
    const cards = transcript.model.filter(entry => entry.type === "edit-card");
    assert.equal(cards.length, 1);
    assert.equal(cards[0].checkpointKey, "prompt-key");
    assert.deepEqual(cards[0].files, files);
    assert.ok(transcript.model.indexOf(cards[0]) > transcript.model.findIndex(entry => entry.text === "Final answer"));
    const prompts = transcript.model.filter(entry => entry.type === "prompt");
    assert.deepEqual(prompts.map(entry => entry.text), steering
      ? ["Original request", "Keep both lines"] : ["Original request"]);
    const document = new Document();
    const chat = document.getElementById("chat");
    const pending = [];
    renderTranscript(document, chat, transcript.model, parseMarkdown, {}, {
      requestAnimationFrame: callback => pending.push(callback),
    });
    pending.splice(0).forEach(callback => callback());
    assert.equal(chat.children.filter(node => node.className === "edit-card").length, 1);
    assert.match(find(chat, node => node.className === "edit-card").textContent, /out.txt/);
  }
});


test("permission requests send the Allow and Reject decisions to the host", async () => {
  const document = new Document();
  const script = scriptedClient();
  const answers = [];
  script.client.approve = async (...args) => { answers.push(args); return { resolved: true }; };
  const app = await start({ global: browser(), document, client: script.client });
  try {
    await submit(document, "edit the fixture");
    await script.emit("RunStarted", { parent_run_id: null });
    for (const [id, allowed, button] of [["allow-edit", true, "Allow"], ["reject-edit", false, "Deny"]]) {
      await script.emit("ToolCallStarted", { tool_call_id: id, tool_name: "write_file", target: `${id}.txt` });
      await script.emit("PermissionRequested", { tool_call_id: id, tool_name: "write_file", mode: "ask" });
      await script.frame({ kind: "approval_requested", payload: {
        approval_id: id, tool_call_id: id, operation: "write_file", target: `${id}.txt`, details: "Replace fixture text",
      } });
      const row = find(document.body, node => node.className?.split(" ").includes("approval") && node.textContent.includes(`${id}.txt`));
      assert.ok(row, "the pending permission request is visible");
      await find(row, node => node.tagName === "BUTTON" && node.textContent === button).click();
      assert.deepEqual(answers.at(-1), [id, allowed, "", false, ""]);
    }
    assert.equal(answers.length, 2);
  } finally { app.subscription.close(); }
});

test("composer Stop, Escape and mod-period each stop a running turn once", async () => {
  const stops = [];
  for (const action of ["button", "escape", "shortcut"]) {
    const document = new Document();
    const script = scriptedClient();
    script.client.stop = async () => { stops.push(action); return { accepted: true }; };
    const app = await start({ global: browser(), document, client: script.client });
    try {
      await submit(document, "run the fixture");
      await script.emit("RunStarted", { parent_run_id: null });
      const input = document.getElementById("prompt");
      if (action === "button") {
        const stop = find(document.getElementById("prompt-form"), node => node.tagName === "BUTTON" && node.textContent === "Stop");
        assert.ok(stop && !stop.hidden, "Stop is reachable in the composer");
        await stop.click();
      } else {
        await input.dispatch("keydown", action === "escape" ? { key: "Escape" } : { key: ".", ctrlKey: true });
      }
      assert.equal(stops.length, ["button", "escape", "shortcut"].indexOf(action) + 1);
    } finally { app.subscription.close(); }
  }
});

test("Escape closes Find before it can stop the running turn", async () => {
  const document = new Document();
  const script = scriptedClient();
  let stops = 0;
  script.client.stop = async () => { stops++; return { accepted: true }; };
  const app = await start({ global: browser(), document, client: script.client });
  try {
    await submit(document, "keep running");
    await script.emit("RunStarted", { parent_run_id: null });
    const input = document.getElementById("prompt");
    await input.dispatch("keydown", { key: "f", ctrlKey: true });
    const bar = find(document.body, node => node.className === "transcript-find");
    assert.equal(bar.hidden, false);
    await input.dispatch("keydown", { key: "Escape" });
    assert.equal(bar.hidden, true);
    assert.equal(document.activeElement, input);
    assert.equal(stops, 0);
    await input.dispatch("keydown", { key: "Escape" });
    assert.equal(bar.hidden, true);
    assert.equal(stops, 1, "after Find closes, the next Escape reaches Stop");
  } finally { app.subscription.close(); }
});
