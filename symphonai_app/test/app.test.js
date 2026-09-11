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
  assert.ok(app.transcript);
  assert.deepEqual(app.transcript.model, []);
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

test("a failed prompt reports through the turn-state label", async () => {
  const document = new FakeDocument();
  const client = fakeClient();
  client.prompt = async () => {
    throw new Error("offline");
  };
  await start({ global: {}, document, client });
  document.getElementById("prompt").value = "retry me";

  await document.getElementById("prompt-form").dispatch("submit");

  assert.equal(document.getElementById("turn-state").textContent, "Prompt failed.");
  assert.equal(document.getElementById("chat").children.length, 0);
});

test("render stays DOM-only and start does not read window", async () => {
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
  assert.ok(!indexSource.includes("window.__symphonai"));
});
