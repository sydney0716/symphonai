import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { start } from "../src/app.js";
import { parseRoadmap } from "../src/roadmap.js";
import { FakeDocument, fakeClient } from "./app.test.js";

async function realRoadmapText() {
  return readFile(new URL("../../docs/roadmap.json", import.meta.url), "utf8");
}

function walk(node) {
  return [node, ...node.children.flatMap(walk)];
}

test("the real roadmap renders its full graph in the rail without errors", async () => {
  const roadmapText = await realRoadmapText();
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient(roadmapText) });

  const rail = document.getElementById("roadmap");
  const graph = findClass(rail, "roadmap-overview");
  assert.ok(graph);
  assert.equal(findClass(rail, "roadmap-graph-error"), undefined);
  assert.ok(walk(graph).some((node) => node.className.startsWith("roadmap-phase-box")));
  assert.equal(findClass(rail, "roadmap-open"), undefined);
});

function findClass(root, className) {
  return walk(root).find((node) => node.className === className);
}

test("an unknown phase dependency reports the phase that names it", async () => {
  const roadmap = JSON.parse(await realRoadmapText());
  const phase = roadmap.phases[0];
  phase.after = ["99"];

  assert.throws(
    () => parseRoadmap(JSON.stringify(roadmap)),
    (error) => error.message.includes(`phase ${phase.id}`) && error.message.includes("99"),
  );
});
