import test from "node:test";
import assert from "node:assert/strict";

import { generalRows, modelRows, SettingsError } from "../src/settings.js";

test("general rows sort by key, retain scope, and render complete values", () => {
  const longValue = "x".repeat(5000);
  const reply = { settings: { config: [
    { key: "z.list", value: ["first", "second"], scope: "project" },
    { key: "a.enabled", value: true, scope: "user" },
    { key: "b.disabled", value: false, scope: "session" },
    { key: "c.count", value: 1234, scope: "private" },
    { key: "d.long", value: longValue, scope: "project" },
  ] } };

  assert.deepEqual(generalRows(reply), [
    { key: "a.enabled", value: "on", scope: "user" },
    { key: "b.disabled", value: "off", scope: "session" },
    { key: "c.count", value: "1234", scope: "private" },
    { key: "d.long", value: longValue, scope: "project" },
    { key: "z.list", value: "first, second", scope: "project" },
  ]);
  assert.equal(reply.settings.config[0].key, "z.list");
  assert.deepEqual(generalRows({ settings: {} }), []);
});

test("model rows keep only provider name, environment variable, and presence", () => {
  const reply = { settings: { providers: [
    { name: "openai", env_var: "OPENAI_API_KEY", key_present: true },
    { name: "anthropic", env_var: "ANTHROPIC_API_KEY", key_present: false },
  ] } };

  assert.deepEqual(modelRows(reply), [
    { name: "anthropic", envVar: "ANTHROPIC_API_KEY", present: false },
    { name: "openai", envVar: "OPENAI_API_KEY", present: true },
  ]);
  assert.deepEqual(modelRows({ settings: {} }), []);
});

test("invalid replies fail while absent sections are empty", () => {
  for (const invalid of [null, [], "broken", 42]) {
    assert.throws(() => generalRows(invalid), SettingsError);
    assert.throws(() => modelRows(invalid), SettingsError);
  }
  assert.deepEqual(generalRows({}), []);
  assert.deepEqual(modelRows({}), []);
});
