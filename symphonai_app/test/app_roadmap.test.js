import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { start } from "../src/app.js";
import { parseRoadmap, renderRoadmap } from "../src/roadmap.js";
import { layoutRoadmap } from "../src/roadmap_graph.js";
import { FakeDocument, fakeClient } from "./app.test.js";

async function realRoadmapText() {
  return readFile(new URL("../../docs/roadmap.json", import.meta.url), "utf8");
}

function phaseLine(phase) {
  return `${phase.id} · ${phase.name} — ${phase.progress.done}/${phase.progress.total}`;
}

async function assertRailMatches(roadmapText, expected) {
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient(roadmapText) });
  assert.deepEqual(
    document.getElementById("roadmap").children.map((line) => line.textContent),
    [...expected, "Open roadmap"],
  );
}

test("the real roadmap rail lists only unfinished phases in order", async () => {
  const roadmapText = await realRoadmapText();
  const roadmap = renderRoadmap(parseRoadmap(roadmapText));
  const expected = roadmap.phases
    .filter((phase) => phase.status !== "done")
    .map(phaseLine);

  await assertRailMatches(roadmapText, expected);
});

test("a fixture with its first phase unfinished lists it first", async () => {
  const roadmap = JSON.parse(await realRoadmapText());
  roadmap.phases[0].status = "in_progress";
  const roadmapText = JSON.stringify(roadmap);
  const rendered = renderRoadmap(parseRoadmap(roadmapText));

  await assertRailMatches(roadmapText, rendered.phases
    .filter((phase) => phase.status !== "done")
    .map(phaseLine));
});

test("the real roadmap and each phase task graph lay out without errors", async () => {
  const roadmap = renderRoadmap(parseRoadmap(await realRoadmapText()));
  const graph = layoutRoadmap(roadmap.phases);

  assert.equal(graph.error, undefined);
  const phases = graph.rows.flatMap((row) => row.flatMap((node) => node.group ?? [node]));
  assert.equal(phases.length, roadmap.phases.length);
  for (const phase of phases) {
    assert.equal(phase.taskLayout.error, undefined, `phase ${phase.id}`);
  }
});

test("an unknown phase dependency reports the phase that names it", async () => {
  const roadmap = JSON.parse(await realRoadmapText());
  const phase = roadmap.phases[0];
  phase.after = ["99"];

  assert.throws(
    () => parseRoadmap(JSON.stringify(roadmap)),
    (error) => error.message.includes(`phase ${phase.id}`) && error.message.includes("99"),
  );
});
