import test from "node:test";
import assert from "node:assert/strict";
import { start } from "../src/app.js";

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
