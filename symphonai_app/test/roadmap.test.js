import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import * as roadmapModule from "../src/roadmap.js";
import {
  followUpsFor,
  RoadmapError,
  itemsOf,
  parseRoadmap,
  phaseProgress,
  renderRoadmap,
  reportPathFor,
  specPaths,
} from "../src/roadmap.js";

test("the real roadmap exposes all fifteen phase identities", async () => {
  const path = process.env.SYMPHONAI_ROADMAP_PATH;
  assert.ok(path, "the roadmap path was not passed to the Node test");
  const roadmap = parseRoadmap(await readFile(path, "utf8"));
  assert.deepEqual(
    roadmap.phases.map(({ id, name, status }) => ({ id, name, status })),
    [
      { id: "00", name: "Canonical shapes and identity", status: "done" },
      { id: "01", name: "Cancellation and typed event channel", status: "done" },
      { id: "02", name: "Tool suite", status: "done" },
      { id: "13", name: "Test harness", status: "done" },
      { id: "03", name: "Harness ergonomics", status: "done" },
      { id: "14", name: "Budgets, cost, and failure handling", status: "done" },
      { id: "16", name: "Web tools", status: "done" },
      { id: "04", name: "Durable sessions", status: "done" },
      { id: "05", name: "Streaming", status: "done" },
      { id: "17", name: "Host process", status: "done" },
      { id: "07", name: "Agent control plane", status: "done" },
      { id: "08", name: "Agent definitions", status: "done" },
      { id: "10", name: "Extensibility", status: "done" },
      { id: "19", name: "Extensions reach the runtime", status: "done" },
      { id: "18", name: "App", status: "next" },
    ],
  );
});

test("both item forms normalize in one fixture", () => {
  const roadmap = parseRoadmap(JSON.stringify({
    goal: "mixed",
    phases: [{
      id: "18",
      name: "App",
      status: "next",
      items: [
        "plain",
        {
          title: "one binding",
          desc: "details",
          spec: "specs/18/18c.md",
          done: true,
        },
        {
          title: "several bindings",
          spec: ["specs/10/10b.md", "specs/10/10c.md"],
        },
      ],
    }],
  }));
  assert.deepEqual(roadmap.phases[0].items, [
    { title: "plain", desc: "", spec: [] },
    {
      title: "one binding",
      desc: "details",
      spec: ["specs/18/18c.md"],
      done: true,
    },
    {
      title: "several bindings",
      desc: "",
      spec: ["specs/10/10b.md", "specs/10/10c.md"],
    },
  ]);
});

test("specPaths accepts both binding forms and specPath is gone", () => {
  assert.deepEqual(
    specPaths({ title: "one", spec: "specs/18/18c.md" }),
    ["specs/18/18c.md"],
  );
  assert.deepEqual(
    specPaths({ title: "several", spec: ["specs/10/10b.md", "specs/10/10c.md"] }),
    ["specs/10/10b.md", "specs/10/10c.md"],
  );
  assert.deepEqual(specPaths({ title: "unbound" }), []);
  assert.ok(!Object.hasOwn(roadmapModule, "specPath"));
});

test("the real roadmap contains capabilities rather than follow-up findings", async () => {
  const roadmap = parseRoadmap(
    await readFile(process.env.SYMPHONAI_ROADMAP_PATH, "utf8"),
  );
  const phases = new Map(roadmap.phases.map((phase) => [phase.id, phase]));
  const allBindings = roadmap.phases.flatMap((phase) =>
    phase.items.flatMap((item) => specPaths(item))
  );
  assert.deepEqual(
    allBindings.filter((path) => /[0-9]+[A-Za-z]F[0-9]*-/.test(path)),
    [],
  );
  assert.deepEqual(
    ["10", "18", "19"].map((id) => phases.get(id).items.length),
    [6, 15, 6],
  );

  const originalPhase10Titles = [
    "MCP client - subprocess boundary, language-neutral",
    "Skills: progressively loaded procedures, not permanent system prompt, where the always-on cost is frontmatter only and is measured per skill",
    "Hooks on the event channel: prompt submit, pre/post tool, tool failure as its own event, permission request and permission denied as separate events, compaction, session, subagent",
    "Scoped config: user -> project -> private -> session, with provenance and a capability ceiling",
    "Plugins last, once agents/skills/hooks are independently useful",
  ];
  assert.deepEqual(
    phases.get("10").items.slice(0, 5).map((item) => item.title),
    originalPhase10Titles,
  );
  const hooks = phases.get("10").items[2];
  assert.deepEqual(hooks.spec, [
    "specs/10/10b-the-missing-events.md",
    "specs/10/10c-hooks.md",
  ]);

  const unresolved = [
    "Context and cost display (needs phase 14)",
    "Run tree, per-agent transcript, current tool, and waiting reason (needs phase 07)",
    "Create, edit, and inspect agents without editing files by hand (needs phase 08)",
    "Agent, skill, and hook config surfaces, and extension diagnostics (needs phase 10)",
  ];
  for (const title of unresolved) {
    const item = phases.get("18").items.find((candidate) => candidate.title === title);
    assert.ok(item, `missing unresolved item: ${title}`);
    assert.deepEqual(specPaths(item), []);
  }
});

test("follow-ups are inferred from a base spec in finding order", () => {
  const paths = [
    "specs/10/10cF2-a-filename-is-not-a-trust-boundary.md",
    "specs/18/18aF2-a-check-that-fails-on-a-finder-window.md",
    "specs/10/10c-hooks.md",
    "specs/10/10cF-hooks-that-can-actually-load.md",
    "specs/18/18aF-a-double-more-capable-than-the-real-thing.md",
    "specs/18/18a-a-client-for-the-boundary.md",
    "specs/10/10e-mcp-client.md",
  ];
  assert.deepEqual(followUpsFor("specs/10/10c-hooks.md", paths), [
    "specs/10/10cF-hooks-that-can-actually-load.md",
    "specs/10/10cF2-a-filename-is-not-a-trust-boundary.md",
  ]);
  assert.deepEqual(followUpsFor("specs/18/18a-a-client-for-the-boundary.md", paths), [
    "specs/18/18aF-a-double-more-capable-than-the-real-thing.md",
    "specs/18/18aF2-a-check-that-fails-on-a-finder-window.md",
  ]);
  assert.deepEqual(followUpsFor("specs/10/10e-mcp-client.md", paths), []);
  assert.deepEqual(
    followUpsFor("specs/10/10cF-hooks-that-can-actually-load.md", paths),
    [],
  );
});

test("report paths are derived without filesystem access", () => {
  const cases = [
    ["specs/10/10e-mcp-client.md", "specs/report/10/10e-mcp-client-report.md"],
    [
      "specs/99/99zF-follow-up-does-not-exist.md",
      "specs/report/99/99zF-follow-up-does-not-exist-report.md",
    ],
    ["specs/no-phase.md", null],
    ["docs/roadmap.json", null],
  ];
  assert.equal(cases.length, 4);
  for (const [path, expected] of cases) {
    assert.equal(reportPathFor(path), expected);
  }
});

test("phase progress counts markers unless phase status is authoritative", () => {
  const active = {
    id: "18",
    status: "next",
    items: [
      { title: "one", done: true },
      { title: "two", done: false },
      "three",
    ],
  };
  assert.deepEqual(phaseProgress(active), { done: 1, total: 3 });
  assert.deepEqual(
    phaseProgress({ ...active, status: "done" }),
    { done: 3, total: 3 },
  );
});

test("malformed roadmaps fail atomically and name the member", () => {
  const cases = [
    [JSON.stringify({ goal: "bad" }), /phases/],
    [
      JSON.stringify({ goal: "bad", phases: [{ name: "missing", status: "next", items: [] }] }),
      /phase 0.*id/,
    ],
    [
      JSON.stringify({ goal: "bad", phases: [{ id: "18", name: "bad", status: "next", items: {} }] }),
      /phase 18.*items/,
    ],
    [
      JSON.stringify({ goal: "bad", phases: [{ id: "18", name: "bad", status: "next", items: [3] }] }),
      /phase 18.*item 0/,
    ],
  ];
  assert.equal(cases.length, 4);
  for (const [json, pattern] of cases) {
    assert.throws(() => parseRoadmap(json), (error) => {
      assert.ok(error instanceof RoadmapError);
      assert.match(error.message, pattern);
      return true;
    });
  }
});

test("rendering fresh data carries no prior tree state", () => {
  const first = parseRoadmap(JSON.stringify({
    goal: "first-only-goal",
    phases: [{ id: "01", name: "first-only-phase", status: "next", items: ["first-only-item"] }],
  }));
  const second = parseRoadmap(JSON.stringify({
    goal: "second-only-goal",
    phases: [{ id: "02", name: "second-only-phase", status: "done", items: ["second-only-item"] }],
  }));
  const firstTree = renderRoadmap(first);
  const secondTree = renderRoadmap(second);
  assert.notDeepEqual(firstTree, secondTree);
  assert.ok(!JSON.stringify(secondTree).includes("first-only"));
  assert.equal(secondTree.phases[0].progress.done, 1);
});
