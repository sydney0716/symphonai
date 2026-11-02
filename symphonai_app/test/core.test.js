import test from "node:test";
import assert from "node:assert/strict";
import { start } from "../src/app.js";
import { createClient } from "../src/client.js";

// A small DOM with real parent/child and textContent semantics. Missing ids
// return null, and operations on a non-child fail rather than hiding a crash.
class Node {
  constructor(document, tagName = "", text = "") {
    this.ownerDocument = document;
    this.nodeType = tagName ? 1 : 3;
    this.tagName = tagName.toUpperCase();
    this.parentNode = null;
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
  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }
  async dispatch(type, values = {}) {
    const event = { target: this, currentTarget: this, preventDefault() {}, stopPropagation() {}, ...values };
    for (const listener of this.listeners.get(type) ?? []) await listener(event);
  }
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
  addEventListener(type, listener) { this.listeners.set(type, listener); }
}

function browser() {
  const storage = new Map();
  return {
    location: { hash: "#/chat" },
    addEventListener() {},
    localStorage: { getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
  };
}

function scriptedClient() {
  let receive;
  const prompts = [];
  let newSessions = 0;
  const client = {
    project: async () => ({ repo_root: "/core-project", name: "Core project" }),
    sessions: async () => [],
    sessionViewed: async () => ({ viewed_at: "2026-11-02T00:00:00+00:00" }),
    roadmap: async () => ({ goals: [] }),
    settings: async () => ({ settings: { mode: "allow", providers: [{ name: "openai", key_present: true }] } }),
    health: async () => ({ state: "idle", runtime_run_id: null }),
    conversationStats: async () => ({ conversation: null }),
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
  return { client, prompts, newSessions: () => newSessions, emit: (type, fields = {}) => receive({
    kind: "event", payload: { type, agent_id: "core-agent", agent_name: "leader", run_id: "core-run", ...fields },
  }) };
}

async function submit(document, text) {
  document.getElementById("prompt").value = text;
  await document.getElementById("prompt-form").dispatch("submit");
}

function terminalBrowser() {
  const global = browser();
  const instances = [], listeners = new Map();
  global.innerHeight = 800;
  global.addEventListener = (type, callback) => {
    const callbacks = listeners.get(type) ?? [];
    callbacks.push(callback);
    listeners.set(type, callbacks);
  };
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
  return { global, instances, event, dispatch: async (type, value) => {
    for (const callback of listeners.get(type) ?? []) await callback(value);
  } };
}

test("Terminal panel starts closed and opens without xterm globals", async () => {
  const document = new Document();
  const script = scriptedClient();
  const app = await start({ global: browser(), document, client: script.client });
  const panel = document.getElementById("terminal-panel");
  assert.ok(panel.hidden);
  assert.equal(app.route().page, "chat");
  assert.ok(!find(document.getElementById("rail-tools"), (node) => node.textContent === "Terminal"));
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

test("A tools-only MCP catalog adds no slash commands", async () => {
  const document = new Document();
  const script = scriptedClient();
  script.client.mcpPrompts = async () => ({ prompts: [] });
  await start({ global: browser(), document, client: script.client });
  const input = document.getElementById("prompt");
  input.value = "/mcp.";
  await input.dispatch("input");
  assert.ok(!find(document.body, (node) => node.className?.startsWith("command-suggestion")));
});

test("page starts against the scripted client", async () => {
  const document = new Document();
  const script = scriptedClient();
  const app = await start({ global: browser(), document, client: script.client });
  assert.equal(app.client, script.client);
  assert.equal(app.route().page, "chat");
  assert.ok(document.getElementById("prompt"));
  assert.equal(document.getElementById("chat").textContent, "");
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
  script.client.openSession = async (id) => { opened = id; };
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


test("real client sends session viewed directly over HTTP", async () => {
  const calls = [];
  const client = createClient({
    port: 4312, token: "core-inert-token",
    fetch: async (url, options) => {
      calls.push({ path: new URL(url).pathname, method: options.method, body: JSON.parse(options.body) });
      return { status: 200, json: async () => ({ session_id: "shown", viewed_at: "2026-11-02T00:00:00Z" }) };
    },
  });
  await client.sessionViewed("shown");
  assert.deepEqual(calls, [{ path: "/session/viewed", method: "POST", body: { session_id: "shown" } }]);
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

test("real client requests the selected usage period without the protocol encoder", async () => {
  const calls = [];
  const client = createClient({ port: 4312, token: "core-inert-token", fetch: async (url, options) => {
    calls.push([new URL(url).pathname + new URL(url).search, options.method]);
    return { status: 200, json: async () => ({ days: 90, providers: [] }) };
  } });
  assert.equal((await client.usage(90)).days, 90);
  assert.deepEqual(calls, [["/usage?days=90", "GET"]]);
});
