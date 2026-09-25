import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { start } from "../src/app.js";
import { FakeDocument, fakeClient } from "./app.test.js";

test("the real roadmap opens the first unfinished phase and no other phase", async () => {
  const roadmapText = await readFile(
    new URL("../../docs/roadmap.json", import.meta.url),
    "utf8",
  );
  const roadmap = JSON.parse(roadmapText);
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient(roadmapText) });
  const phases = document.getElementById("roadmap").children;
  const open = phases.filter((phase) => phase.open);
  const current = roadmap.phases.findIndex((phase) => phase.status !== "done");

  assert.equal(phases.length, roadmap.phases.length);
  assert.equal(open.length, 1);
  assert.deepEqual(
    phases.map((phase) => phase.open),
    roadmap.phases.map((_, index) => index === current),
  );
});
