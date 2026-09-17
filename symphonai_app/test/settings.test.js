import test from "node:test";
import assert from "node:assert/strict";

import {
  generalRows, inventoryRows, modelRows, rosterRows, serverRows, SettingsError,
} from "../src/settings.js";

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

test("server rows retain the full command and startup state", () => {
  const command = `python -m ${"example.".repeat(600)}server`;
  const reply = { settings: { mcp_servers: [
    { name: "one", command, started: true },
    { name: "two", command: "other", started: false },
  ] } };

  assert.deepEqual(serverRows(reply), [
    { name: "one", command, started: true },
    { name: "two", command: "other", started: false },
  ]);
  assert.deepEqual(serverRows({ settings: {} }), []);
});

test("rosters sort each extension kind and ignore unknown kinds", () => {
  const reply = { settings: {
    skills: ["zeta", "alpha"],
    plugins: ["west", "east"],
    agents: ["reviewer", "builder"],
  } };

  assert.deepEqual(rosterRows(reply, "skills"), ["alpha", "zeta"]);
  assert.deepEqual(rosterRows(reply, "plugins"), ["east", "west"]);
  assert.deepEqual(rosterRows(reply, "agents"), ["builder", "reviewer"]);
  assert.deepEqual(rosterRows(reply, "unknown"), []);
  assert.deepEqual(rosterRows({ settings: {} }, "skills"), []);
  assert.deepEqual(reply.settings.skills, ["zeta", "alpha"]);
});

test("inventory rows retain withheld details and explain an absent reason", () => {
  const reply = { settings: { withheld: [
    { scope: "project", directory: ".symphonai/skills", names: ["blocked"], reason: "repository not trusted" },
    { scope: "user", directory: "/users/example/plugins", names: [] },
  ] } };

  assert.deepEqual(inventoryRows(reply), [
    { scope: "project", directory: ".symphonai/skills", names: ["blocked"], reason: "repository not trusted" },
    { scope: "user", directory: "/users/example/plugins", names: [], reason: "This scope offered nothing." },
  ]);
  assert.deepEqual(inventoryRows({ settings: {} }), []);
});

test("invalid replies fail while absent sections are empty", () => {
  for (const invalid of [null, [], "broken", 42]) {
    assert.throws(() => generalRows(invalid), SettingsError);
    assert.throws(() => modelRows(invalid), SettingsError);
    assert.throws(() => serverRows(invalid), SettingsError);
    assert.throws(() => rosterRows(invalid, "skills"), SettingsError);
    assert.throws(() => inventoryRows(invalid), SettingsError);
  }
  assert.deepEqual(generalRows({}), []);
  assert.deepEqual(modelRows({}), []);
  assert.deepEqual(serverRows({}), []);
  assert.deepEqual(rosterRows({}, "plugins"), []);
  assert.deepEqual(inventoryRows({}), []);
});
