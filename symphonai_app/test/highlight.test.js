import test from "node:test";
import assert from "node:assert/strict";

import { highlight } from "../src/highlight.js";
import { renderMarkdownBlocks, renderTranscript } from "../src/render.js";

const samples = [
  ["python", "return 7, \"say \\\"hi\\\"\" # note 7"],
  ["javascript", "const count = 7; const text = \"say \\\"hi\\\"\"; // note 7"],
  ["typescript", "const count = 7; const text = \"say \\\"hi\\\"\"; // note 7"],
  ["shell", "if echo 7; then echo \"say \\\"hi\\\"\" # note 7"],
  ["json", "{\"ok\": true, \"text\": \"say \\\"hi\\\"\", \"n\": 7 // note"],
  ["diff", "+const count = 7; const text = \"say \\\"hi\\\"\"; // note 7"],
  ["rust", "let count = 7; let text = \"say \\\"hi\\\"\"; // note 7"],
  ["go", "var count = 7; var text = \"say \\\"hi\\\"\" // note 7"],
  ["toml", "enabled = true\ncount = 7\ntext = \"say \\\"hi\\\"\" # note 7"],
];

test("supported languages highlight keywords, escaped strings, comments and numbers without changing text", () => {
  for (const [language, source] of samples) {
    const tokens = highlight(language, source);
    assert.equal(tokens.map((token) => token.text).join(""), source, language);
    for (const kind of ["keyword", "string", "comment", "number"]) {
      assert.ok(tokens.some((token) => token.kind === kind), language + " has " + kind);
    }
    assert.ok(tokens.some((token) => token.kind === "string" && token.text.includes("\\\"")), language + " keeps escaped quotes in strings");
  }
});

test("diff additions and removals are distinguished while preserving text", () => {
  const source = "+let added = 4;\n-let removed = 2;\n";
  const tokens = highlight("diff", source);
  assert.equal(tokens.map((token) => token.text).join(""), source);
  assert.ok(tokens.some((token) => token.kind === "keyword" && token.text.startsWith("+")));
  assert.ok(tokens.some((token) => token.kind === "comment" && token.text.startsWith("-")));
});

test("comment-like text inside strings stays a string and unknown languages stay plain", () => {
  assert.ok(highlight("py", "\"# not a comment\"").every((token) => token.kind === "string"));
  assert.ok(highlight("js", "'//x'").every((token) => token.kind === "string"));
  assert.deepEqual(highlight("made-up", "let x = 2"), [{ kind: "plain", text: "let x = 2" }]);
  assert.deepEqual(highlight(null, "plain"), [{ kind: "plain", text: "plain" }]);
  for (const [alias, language] of [["sh", "shell"], ["bash", "shell"], ["zsh", "shell"], ["rs", "rust"]]) {
    assert.deepEqual(highlight(alias, "if 7"), highlight(language, "if 7"));
  }
});

class FakeNode {
  constructor(tagName, text = "") {
    this.tagName = tagName.toUpperCase();
    this.textContent = text;
    this.children = [];
    this.attributes = new Map();
    this.className = "";
    this.listeners = new Map();
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  append(...children) {
    this.children.push(...children);
  }

  replaceChildren(...children) {
    this.children = [...children];
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  async dispatch(type) {
    await this.listeners.get(type)?.();
  }
}

const fakeDocument = {
  createElement: (tag) => new FakeNode(tag),
  createTextNode: (text) => new FakeNode("#text", text),
};

test("code blocks stay safe, copy their exact source and render large blocks plain", async () => {
  const source = "<script>alert(\"x\")</script>";
  const [pre] = renderMarkdownBlocks(fakeDocument, source, () => [{
    type: "code-block", language: "javascript", text: source,
  }], { highlight });
  const code = pre.children[0];
  assert.equal(code.children.map((node) => node.textContent).join(""), source);
  assert.equal(code.children.some((node) => node.tagName === "SCRIPT"), false);
  const spans = code.children.filter((node) => node.tagName === "SPAN");
  assert.equal(spans.length, highlight("javascript", source).filter((token) => token.kind !== "plain").length);
  assert.ok(spans.every((node) => node.className.startsWith("syntax-")));

  const large = "x".repeat(200_001);
  assert.deepEqual(highlight("python", large), [{ kind: "plain", text: large }]);

  const copied = [];
  const root = new FakeNode("div");
  renderTranscript(fakeDocument, root, [{ type: "text", text: source }], () => [{
    type: "code-block", language: "javascript", text: source,
  }], { copyText: async (value) => copied.push(value), highlight });
  await root.children[0].children[0].children[1].dispatch("click");
  assert.deepEqual(copied, [source]);
});
