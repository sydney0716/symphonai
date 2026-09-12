import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  KeymapError,
  RESERVED,
  lookup,
  mergeKeymap,
  parseKeymap,
  primaryModifier,
} from "../src/keys.js";


async function loadKeymap(path) {
  return parseKeymap(await readFile(path, "utf8"));
}


test("the configured default file returns every binding", async () => {
  const path = process.env.SYMPHONAI_KEYS_PATH;
  assert.ok(path, "the default keymap path was not passed to the test");
  const text = await readFile(path, "utf8");
  const bindings = await loadKeymap(path);

  assert.equal(bindings.size, Object.keys(JSON.parse(text)).length);
  assert.equal(bindings.get("mod+enter"), "submit");
  for (const chord of RESERVED) {
    assert.ok(bindings.has(chord), `default keymap omitted reserved chord ${chord}`);
  }
});


test("merging changes one leaf, preserves the rest, and adds new actions", async () => {
  const defaults = await loadKeymap(process.env.SYMPHONAI_KEYS_PATH);
  const rebound = mergeKeymap(defaults, parseKeymap('{"mod+enter":"send"}'));

  assert.equal(rebound.bindings.size, defaults.size);
  assert.equal(rebound.bindings.get("mod+enter"), "send");
  assert.equal(rebound.bindings.get("mod+k"), "clear-chat");
  assert.deepEqual(rebound.rejected, []);

  const added = mergeKeymap(
    defaults,
    parseKeymap('{"alt+n":"open-notes"}'),
  );
  assert.equal(added.bindings.size, defaults.size + 1);
  assert.equal(added.bindings.get("alt+n"), "open-notes");
});


test("null unbinds a chord and lookup no longer finds it", async () => {
  const defaults = await loadKeymap(process.env.SYMPHONAI_KEYS_PATH);
  const { bindings, rejected } = mergeKeymap(
    defaults,
    parseKeymap('{"mod+k":null}'),
  );

  assert.equal(bindings.has("mod+k"), false);
  assert.deepEqual(rejected, []);
  assert.equal(
    lookup(bindings, { key: "k", metaKey: true }, { platform: "darwin" }),
    null,
  );
});


test("every reserved chord is rejected, reported, and preserved", async () => {
  const defaults = await loadKeymap(process.env.SYMPHONAI_KEYS_PATH);
  const user = new Map(
    RESERVED.map((chord, index) => [chord, `take-reserved-${index}`]),
  );
  const { bindings, rejected } = mergeKeymap(defaults, user);

  assert.deepEqual(rejected.map(({ chord }) => chord), [...RESERVED]);
  for (const chord of RESERVED) {
    const refusal = rejected.find((item) => item.chord === chord);
    assert.equal(bindings.get(chord), defaults.get(chord));
    assert.notEqual(bindings.get(chord), refusal.action);
    assert.equal(typeof refusal.reason, "string");
    assert.ok(refusal.reason.length > 0);
  }
});


test("duplicate chords name the chord and both actions in either file", () => {
  const cases = [
    '{"mod+shift+x":"first-default","shift+mod+x":"second-default"}',
    '{"ctrl+x":"first-user","ctrl+x":"second-user"}',
  ];
  for (const [index, text] of cases.entries()) {
    assert.throws(
      () => parseKeymap(text),
      (error) => {
        assert.ok(error instanceof KeymapError);
        assert.match(error.message, /x/);
        assert.match(error.message, new RegExp(`first-${index === 0 ? "default" : "user"}`));
        assert.match(error.message, new RegExp(`second-${index === 0 ? "default" : "user"}`));
        return true;
      },
    );
  }
});


test("malformed keymap shapes fail as KeymapError", () => {
  const cases = [
    ["not an object", "[]"],
    ["non-string value", '{"mod+k":42}'],
    ["empty chord", '{"":"action"}'],
    ["unknown modifier", '{"hyper+k":"action"}'],
    ["invalid JSON", "{"],
  ];
  for (const [label, text] of cases) {
    assert.throws(
      () => parseKeymap(text),
      (error) => error instanceof KeymapError,
      label,
    );
  }
});


test("lookup canonicalizes modifier order", () => {
  const first = parseKeymap('{"mod+shift+k":"palette"}');
  const second = new Map([["shift+mod+k", "palette"]]);
  const event = { key: "K", metaKey: true, shiftKey: true };

  assert.equal(lookup(first, event, { platform: "darwin" }), "palette");
  assert.equal(lookup(second, event, { platform: "darwin" }), "palette");
});


test("the caller-provided platform resolves the primary modifier", async () => {
  const bindings = parseKeymap('{"mod+p":"open"}');
  const metaEvent = { key: "p", metaKey: true };
  const controlEvent = { key: "p", ctrlKey: true };

  assert.equal(primaryModifier("darwin"), "meta");
  assert.equal(primaryModifier("linux"), "ctrl");
  assert.equal(lookup(bindings, metaEvent, { platform: "darwin" }), "open");
  assert.equal(lookup(bindings, controlEvent, { platform: "linux" }), "open");
  assert.equal(lookup(bindings, controlEvent, { platform: "darwin" }), null);
  assert.equal(lookup(bindings, metaEvent, { platform: "linux" }), null);

  const source = await readFile(new URL("../src/keys.js", import.meta.url), "utf8");
  assert.equal(source.includes("navigator"), false);
  assert.equal(source.includes("globalThis"), false);
});


test("reserved chords are frozen", () => {
  assert.equal(Object.isFrozen(RESERVED), true);
  assert.throws(() => RESERVED.push("alt+escape"), TypeError);
});
