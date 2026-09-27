import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { start } from "../src/app.js";
import { FakeDocument, fakeClient } from "./app.test.js";

async function assertRailOpensFirstUnfinished(roadmapText) {
  const roadmap = JSON.parse(roadmapText);
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient(roadmapText) });
  const phases = document.getElementById("roadmap").children;
  const current = roadmap.phases.findIndex((phase) => phase.status !== "done");

  assert.equal(phases.length, roadmap.phases.length);
  assert.deepEqual(
    phases.map((phase) => phase.open),
    roadmap.phases.map((_, index) => index === current),
  );
}

test("the real roadmap opens no phase when every phase is done", async () => {
  const roadmapText = await readFile(
    new URL("../../docs/roadmap.json", import.meta.url),
    "utf8",
  );
  await assertRailOpensFirstUnfinished(roadmapText);
});

test("a roadmap fixture opens only its first unfinished phase", async () => {
  const roadmapText = await readFile(
    new URL("../../docs/roadmap.json", import.meta.url),
    "utf8",
  );
  const roadmap = JSON.parse(roadmapText);
  roadmap.phases[0].status = "in_progress";

  await assertRailOpensFirstUnfinished(JSON.stringify(roadmap));
});
