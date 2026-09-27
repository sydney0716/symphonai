import test from "node:test";
import assert from "node:assert/strict";

import {
  agentRows, ceilingRows, composeAgentText, generalRows, hookRows, inventoryRows,
  modelRows, parseAgentText, rosterRows, serverRows, SettingsError, trustRows,
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

test("agent rows include scope and withheld reasons", () => {
  const reply = { settings: {
    agents: [{ name: "project-agent", path: ".symphonai/agents/project-agent.toml" }, { name: "builtin", path: "" }],
    withheld: [
      { scope: "project", directory: ".symphonai/agents", names: ["blocked"], reason: "repository not trusted" },
      { scope: "project", directory: ".symphonai/skills", names: ["skill"], reason: "not an agent" },
    ],
  } };

  assert.deepEqual(agentRows(reply), [
    { name: "blocked", path: ".symphonai/agents/blocked.toml", scope: "project", reason: "repository not trusted", editable: false },
    { name: "builtin", path: "", scope: "user", reason: "", editable: false },
    { name: "project-agent", path: ".symphonai/agents/project-agent.toml", scope: "project", reason: "", editable: true },
  ]);
});

test("agent text parses and composes the editable fields", () => {
  const source = `prompt = "Review the change.\\nKeep it short."\n`
    + `tools = ["read_file", "grep"]\n`
    + `deny_tools = ["grep"]\n`
    + `deadline_seconds = 30\n`
    + `[model]\nprovider = "openai"\nmodel = "gpt-test"\neffort = "high"\n`
    + `\n[budget]\nmax_turns = 4\n`;
  const fields = parseAgentText(source);
  assert.deepEqual(fields, {
    prompt: "Review the change.\nKeep it short.",
    provider: "openai",
    model: "gpt-test",
    effort: "high",
    tools: ["read_file", "grep"],
    denyTools: ["grep"],
  });
  assert.equal(composeAgentText(fields, source),
    'prompt = "Review the change.\\nKeep it short."\n'
    + 'tools = ["read_file", "grep"]\n'
    + 'deny_tools = ["grep"]\n'
    + 'deadline_seconds = 30\n'
    + '[model]\nprovider = "openai"\nmodel = "gpt-test"\neffort = "high"\n'
    + '\n[budget]\nmax_turns = 4\n');
});

test("editing owned fields preserves all other lines exactly", () => {
  const source = [
    'prompt = "old prompt"',
    'tools = ["read_file"]',
    'deny_tools = ["grep"]',
    'deadline_seconds = 30',
    '',
    '[model]',
    'provider = "openai"',
    'model = "gpt-test"',
    'effort = "low"',
    '',
    '[budget]',
    'max_turns = 4',
    '',
    '[memory]',
    'enabled = true',
    '',
    '[policy]',
    'mode = "ask"',
    '# keep this comment',
    '',
  ].join("\n");
  const fields = parseAgentText(source);
  assert.equal(composeAgentText({ ...fields, prompt: "new prompt" }, source), source.replace('prompt = "old prompt"', 'prompt = "new prompt"'));
  assert.equal(composeAgentText({ ...fields, denyTools: [] }, source), source.replace('deny_tools = ["grep"]\n', ''));
});

test("adding a missing model field leaves other tables untouched", () => {
  const source = 'prompt = "review"\n\n[budget]\nmax_turns = 4\n\n[memory]\nenabled = true\n';
  const fields = parseAgentText(source);
  const output = composeAgentText({ ...fields, effort: "high" }, source);
  assert.equal(output, `${source}[model]\neffort = "high"\n`);
});

test("hashes inside strings stay part of the value while trailing comments are removed", () => {
  const source = 'prompt = "issue #42 matters"\ntools = ["read_file"] # keep\n';
  const fields = parseAgentText(source);
  assert.equal(fields.prompt, "issue #42 matters");
  assert.deepEqual(fields.tools, ["read_file"]);
  assert.equal(composeAgentText(fields, source), source);
});

test("parsing an existing definition does not leak its text into a new definition", () => {
  parseAgentText('tools = ["read_file"] # keep\n\n[budget]\nmax_turns = 4\n');
  assert.equal(composeAgentText({
    prompt: "p",
    provider: "openai",
    model: "m",
    tools: ["read_file"],
  }, ""), 'prompt = "p"\ntools = ["read_file"]\n[model]\nprovider = "openai"\nmodel = "m"\n');
});

test("new agent composition retains its minimal exact file", () => {
  const fields = parseAgentText("");
  assert.equal(composeAgentText(fields, ""), 'prompt = ""\n');
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
    modes: ["plan", "allow"],
    shell_allowlist: [["git", "status"], ["ls"]],
  } } };
  assert.deepEqual(ceilingRows(reply), [
    { capability: "allowed_write_scope", value: "none" },
    { capability: "fetch_allowlist", value: "not set" },
    { capability: "fetch_enabled", value: "on" },
    { capability: "modes", value: "plan, allow" },
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
