import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { start } from "../src/app.js";
import { FakeDocument, fakeClient } from "./app.test.js";

test("the real roadmap opens phase 18 and no other phase", async () => {
  const roadmapText = await readFile(
    new URL("../../docs/roadmap.json", import.meta.url),
    "utf8",
  );
  const document = new FakeDocument();
  await start({ global: {}, document, client: fakeClient(roadmapText) });
  const phases = document.getElementById("roadmap").children;
  const open = phases.filter((phase) => phase.open);
  const phase20 = phases.find((phase) => phase.children[0].textContent.startsWith("20 ·"));

  assert.equal(phases.length, JSON.parse(roadmapText).phases.length);
  assert.equal(open.length, 1);
  assert.match(open[0].children[0].textContent, /^18 · App —/);
  assert.equal(phase20.open, false);
});
