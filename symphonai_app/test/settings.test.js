import test from "node:test";
import assert from "node:assert/strict";

import {
  ceilingRows, generalRows, hookRows, inventoryRows, modelRows, rosterRows,
  serverRows, SettingsError, trustRows,
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
    skills: [{ name: "zeta", path: "/home/skills/zeta.md" }, { name: "alpha", path: ".symphonai/skills/alpha.md" }],
    plugins: [{ name: "west", path: "/home/plugins/west" }, { name: "east", path: ".symphonai/plugins/east" }],
    agents: [{ name: "reviewer", path: "" }, { name: "builder", path: ".symphonai/agents/builder.toml" }],
  } };

  assert.deepEqual(rosterRows(reply, "skills"), [
    { name: "alpha", path: ".symphonai/skills/alpha.md" },
    { name: "zeta", path: "/home/skills/zeta.md" },
  ]);
  assert.deepEqual(rosterRows(reply, "plugins"), [
    { name: "east", path: ".symphonai/plugins/east" },
    { name: "west", path: "/home/plugins/west" },
  ]);
  assert.deepEqual(rosterRows(reply, "agents"), [
    { name: "builder", path: ".symphonai/agents/builder.toml" },
    { name: "reviewer", path: "" },
  ]);
  assert.deepEqual(rosterRows(reply, "unknown"), []);
  assert.deepEqual(rosterRows({ settings: {} }, "skills"), []);
  assert.equal(reply.settings.skills[0].name, "zeta");
});

test("inventory rows retain withheld details and explain an absent reason", () => {
  const reply = { settings: { withheld: [
    { scope: "project", directory: ".symphonai/skills", names: ["blocked"], reason: "repository not trusted" },
    { scope: "user", directory: "/users/example/plugins", names: [] },
  ] } };

  assert.deepEqual(inventoryRows(reply), [
    { scope: "project", directory: ".symphonai/skills", names: ["blocked"], reason: "repository not trusted" },
    { scope: "user", directory: "/users/example/plugins", names: [], reason: "No reason recorded." },
  ]);
  assert.deepEqual(inventoryRows({ settings: {} }), []);
});

test("ceiling rows distinguish no grant from no limit and preserve list order", () => {
  const reply = { settings: { ceiling: {
    allowed_write_scope: [],
    fetch_allowlist: null,
    fetch_enabled: true,
    modes: ["plan", "auto"],
    shell_allowlist: [["git", "status"], ["ls"]],
  } } };
  assert.deepEqual(ceilingRows(reply), [
    { capability: "allowed_write_scope", value: "none" },
    { capability: "fetch_allowlist", value: "not set" },
    { capability: "fetch_enabled", value: "on" },
    { capability: "modes", value: "plan, auto" },
    { capability: "shell_allowlist", value: "git status, ls" },
    { capability: "shell_enabled", value: "not set" },
  ]);
  assert.equal(ceilingRows({ settings: { ceiling: { shell_enabled: false } } })[5].value, "off");
  for (const empty of [{ settings: { ceiling: {} } }, { settings: {} }]) {
    const rows = ceilingRows(empty);
    assert.equal(rows.length, 6);
    assert.ok(rows.every(({ value }) => value === "not set"));
  }
});

test("trust and hook rows sort host entries without changing their values", () => {
  const reply = { settings: {
    trust: [
      { root: "/z", allow: [] },
      { root: "/a", allow: ["agents", "skills"] },
    ],
    hooks: [
      { event: "turn", command: "z-hook" },
      { event: "prompt", command: "p-hook" },
      { event: "turn", command: "a-hook" },
    ],
  } };
  assert.deepEqual(trustRows(reply), [
    { root: "/a", allow: ["agents", "skills"] },
    { root: "/z", allow: [] },
  ]);
  assert.deepEqual(hookRows(reply), [
    { event: "prompt", command: "p-hook" },
    { event: "turn", command: "a-hook" },
    { event: "turn", command: "z-hook" },
  ]);
  assert.deepEqual(trustRows({ settings: {} }), []);
  assert.deepEqual(hookRows({ settings: {} }), []);
});

test("invalid replies fail while absent sections are empty", () => {
  for (const invalid of [null, [], "broken", 42]) {
    assert.throws(() => generalRows(invalid), SettingsError);
    assert.throws(() => modelRows(invalid), SettingsError);
    assert.throws(() => serverRows(invalid), SettingsError);
    assert.throws(() => rosterRows(invalid, "skills"), SettingsError);
    assert.throws(() => inventoryRows(invalid), SettingsError);
    assert.throws(() => ceilingRows(invalid), SettingsError);
    assert.throws(() => trustRows(invalid), SettingsError);
    assert.throws(() => hookRows(invalid), SettingsError);
  }
  assert.deepEqual(generalRows({}), []);
  assert.deepEqual(modelRows({}), []);
  assert.deepEqual(serverRows({}), []);
  assert.deepEqual(rosterRows({}, "plugins"), []);
  assert.deepEqual(inventoryRows({}), []);
});
