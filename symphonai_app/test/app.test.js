import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { start } from "../src/app.js";
import { DISPATCHING, RUNNING } from "../src/turn.js";

const BASE = "specs/18/18a-a-client-for-the-boundary.md";
const FOLLOW_UP = "specs/18/18aF-a-double-more-capable-than-the-real-thing.md";
const REPORT = "specs/report/18/18a-a-client-for-the-boundary-report.md";

class FakeElement {
  constructor(tagName, id = "") {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.children = [];
    this.className = "";
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
    this.elements = new Map(
      ["roadmap", "spec", "chat", "approvals", "prompt-form", "prompt", "turn-state"]
        .map((id) => [id, new FakeElement(id === "prompt-form" ? "form" : "div", id)]),
    );
    this.body = new FakeElement("body", "body");
  }

  createElement(tagName) {
    return new FakeElement(tagName);
  }

  getElementById(id) {
    return this.elements.get(id) ?? null;
  }
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

function fixtureRoadmap() {
  return JSON.stringify({
    goal: "Ship a visible app",
    phases: [
      {
        id: "18",
        name: "Desktop app",
        status: "next",
        items: [
          { title: "Boundary", spec: BASE, done: true },
          { title: "Boundary finding", spec: FOLLOW_UP },
        ],
      },
    ],
  });
}

function fakeClient() {
  const calls = { approve: [], file: [], prompt: [] };
  let eventCallback;
  let resolvePrompt;
  const promptReply = new Promise((resolve) => {
    resolvePrompt = resolve;
  });
  const files = new Map([
    ["docs/roadmap.json", { path: "docs/roadmap.json", text: fixtureRoadmap() }],
    [BASE, { path: BASE, text: "base spec text" }],
    [REPORT, { path: REPORT, text: "base report text" }],
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
  };
}

test("start renders both panes from injected dependencies", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const app = await start({ global: {}, document, client });
  const roadmap = document.getElementById("roadmap");

  assert.match(visibleText(roadmap), /Ship a visible app/);
  assert.match(visibleText(roadmap), /18 · Desktop app/);
  assert.match(visibleText(roadmap), /1\/2 · next/);
  assert.ok(document.getElementById("chat"));
  assert.equal(document.getElementById("turn-state").textContent, "idle");
  assert.equal(app.client, client);
});

test("selecting an item opens its spec, report, and follow-up names", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  const app = await start({ global: {}, document, client });
  const calls = [];
  const open = app.specView.open;
  app.specView.open = (...arguments_) => {
    calls.push(arguments_);
    return open(...arguments_);
  };
  const button = find(
    document.getElementById("roadmap"),
    (value) => value.tagName === "BUTTON" && value.textContent === "Boundary",
  );

  await button.dispatch("click");

  assert.equal(calls.length, 1);
  assert.equal(calls[0][0].title, "Boundary");
  assert.deepEqual(calls[0][1], { specPaths: [BASE, FOLLOW_UP] });
  const text = visibleText(document.getElementById("spec"));
  assert.match(text, /base spec text/);
  assert.match(text, /base report text/);
  assert.match(text, /18aF-a-double-more-capable-than-the-real-thing/);
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

  await client.emit({
    kind: "event",
    payload: { type: "AssistantTextDelta", run_id: "run-1", text: "hello " },
  });
  await client.emit({
    kind: "event",
    payload: { type: "AssistantTextDelta", run_id: "run-1", text: "world" },
  });
  assert.match(visibleText(document.getElementById("chat")), /hello world/);
});

test("tool activity, approvals, and dropped notices stay visible", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  await start({ global: {}, document, client });

  await client.emit({
    kind: "event",
    payload: { type: "ToolCallStarted", tool_name: "read_file" },
  });
  await client.emit({
    kind: "event",
    payload: { type: "ToolCallFinished", tool_name: "read_file", ok: true },
  });
  assert.match(visibleText(document.getElementById("chat")), /Tool started: read_file/);
  assert.match(visibleText(document.getElementById("chat")), /Tool finished: read_file/);

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

  await client.emit({ kind: "error", dropped: 3 });
  assert.match(visibleText(document.getElementById("chat")), /3 events were dropped/);
});

test("render stays DOM-only and start does not read window", async () => {
  const renderSource = await readFile(
    new URL("../src/render.js", import.meta.url),
    "utf8",
  );
  const appSource = await readFile(new URL("../src/app.js", import.meta.url), "utf8");
  const indexSource = await readFile(new URL("../index.html", import.meta.url), "utf8");

  assert.ok(!/^\s*import\s/m.test(renderSource));
  assert.ok(!/\b(?:if|switch)\s*\(/.test(renderSource));
  assert.match(appSource, /export async function start\(\{ global, document, client \}\)/);
  assert.ok(!start.toString().includes("window"));
  assert.equal((indexSource.match(/class="(?:roadmap|chat)-pane"/g) ?? []).length, 2);
  assert.equal((indexSource.match(/<!-- symphonai-handshake -->/g) ?? []).length, 1);
  assert.ok(!indexSource.includes("window.__symphonai"));
});
