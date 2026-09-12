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

const DEFAULT_KEYMAP_PATH = process.env.SYMPHONAI_KEYS_PATH
  ?? new URL("../keys.default.json", import.meta.url);


async function loadKeymap(path) {
  return parseKeymap(await readFile(path, "utf8"));
}


test("the configured default file returns every binding", async () => {
  const path = DEFAULT_KEYMAP_PATH;
  const text = await readFile(path, "utf8");
  const bindings = await loadKeymap(path);

  assert.equal(bindings.size, Object.keys(JSON.parse(text)).length);
  assert.equal(bindings.get("mod+enter"), "submit");
  for (const chord of RESERVED) {
    assert.ok(bindings.has(chord), `default keymap omitted reserved chord ${chord}`);
  }
});


test("merging changes one leaf, preserves the rest, and adds new actions", async () => {
  const defaults = await loadKeymap(DEFAULT_KEYMAP_PATH);
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
  const defaults = await loadKeymap(DEFAULT_KEYMAP_PATH);
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
  const defaults = await loadKeymap(DEFAULT_KEYMAP_PATH);
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


test("explicit primary modifiers collide with the portable reserved chord", async () => {
  const defaults = await loadKeymap(DEFAULT_KEYMAP_PATH);
  for (const chord of ["ctrl+l", "meta+l", "mod+l", "MOD+L"]) {
    const { bindings, rejected } = mergeKeymap(
      defaults,
      parseKeymap(JSON.stringify({ [chord]: "open-links" })),
    );
    assert.equal(rejected.length, 1, `${chord} was not rejected`);
    assert.equal(
      rejected[0].reason,
      "collides with reserved chord mod+l: focus input must always remain reachable",
    );
    assert.equal(bindings.get("mod+l"), "focus-input");
    assert.equal(bindings.has(chord.toLowerCase()), chord.toLowerCase() === "mod+l");
    assert.equal(
      lookup(bindings, { key: "l", ctrlKey: true }, { platform: "linux" }),
      "focus-input",
    );
    assert.equal(
      lookup(bindings, { key: "l", metaKey: true }, { platform: "darwin" }),
      "focus-input",
    );
  }
});


test("duplicate chords name the chord and both actions in either file", () => {
  const cases = [
    '{"mod+shift+x":"first-default","shift+mod+x":"second-default"}',
    '{"ctrl+x":"first-user","ctrl+x":"second-user"}',
    '{"mod+k":"portable-control","ctrl+k":"explicit-control"}',
    '{"mod+k":"portable-meta","meta+k":"explicit-meta"}',
  ];
  for (const text of cases) {
    const [[firstChord, firstAction], [secondChord, secondAction]] = (() => {
      const matches = [...text.matchAll(/"([^"]+)":"([^"]+)"/g)];
      return matches.map((match) => [match[1], match[2]]);
    })();
    assert.throws(
      () => parseKeymap(text),
      (error) => {
        assert.ok(error instanceof KeymapError);
        assert.ok(error.message.includes(firstChord));
        assert.ok(error.message.includes(secondChord));
        assert.ok(error.message.includes(firstAction));
        assert.ok(error.message.includes(secondAction));
        return true;
      },
    );
  }
});


test("explicit non-primary chords do not acquire portable spellings", () => {
  const bindings = parseKeymap('{"ctrl+k":"control","alt+k":"alternate"}');
  assert.equal(bindings.size, 2);
  assert.equal(bindings.get("ctrl+k"), "control");
  assert.equal(bindings.get("alt+k"), "alternate");
});


test("canonical user chords replace defaults but spelling collisions do not", () => {
  const defaults = parseKeymap(JSON.stringify({
    "mod+shift+p": "palette",
    "mod+k": "clear-chat",
  }));
  const replacement = mergeKeymap(
    defaults,
    parseKeymap('{"shift+mod+p":"new-palette"}'),
  );
  assert.deepEqual(replacement.rejected, []);
  assert.equal(replacement.bindings.get("shift+mod+p"), "new-palette");

  const collision = mergeKeymap(
    defaults,
    parseKeymap('{"ctrl+k":"kill-line"}'),
  );
  assert.equal(collision.rejected.length, 1);
  assert.match(collision.rejected[0].reason, /mod\+k/);
  assert.equal(collision.bindings.get("mod+k"), "clear-chat");
  assert.equal(collision.bindings.has("ctrl+k"), false);
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
  assert.ok(source.includes("function chordSpellings(chord)"));
});


test("reserved chords are frozen", () => {
  assert.equal(Object.isFrozen(RESERVED), true);
  assert.throws(() => RESERVED.push("alt+escape"), TypeError);
});
