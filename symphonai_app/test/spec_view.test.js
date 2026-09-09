import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import * as specViewModule from "../src/spec_view.js";
import { createSpecView } from "../src/spec_view.js";

const SPEC_PATH = "specs/18/18d-the-spec-and-its-report.md";
const REPORT_PATH = "specs/report/18/18d-the-spec-and-its-report-report.md";
const ITEM = { title: "The spec and its report", spec: SPEC_PATH };

function clientWith(replies) {
  const calls = [];
  return {
    calls,
    async file(path) {
      calls.push(path);
      return replies.get(path) ?? { status: 404 };
    },
  };
}

test("open enumerates spec with report, spec alone, and missing spec", async () => {
  const cases = [
    {
      label: "spec and report",
      replies: new Map([
        [SPEC_PATH, { status: 200, path: SPEC_PATH, text: "spec text" }],
        [REPORT_PATH, { status: 200, path: REPORT_PATH, text: "report text" }],
      ]),
      expected: {
        spec: { path: SPEC_PATH, text: "spec text" },
        report: { path: REPORT_PATH, text: "report text" },
      },
    },
    {
      label: "missing report",
      replies: new Map([
        [SPEC_PATH, { status: 200, path: SPEC_PATH, text: "spec text" }],
      ]),
      expected: { spec: { path: SPEC_PATH, text: "spec text" }, report: null },
    },
    {
      label: "missing spec",
      replies: new Map(),
      expected: null,
    },
  ];
  assert.equal(cases.length, 3);
  for (const { label, replies, expected } of cases) {
    const client = clientWith(replies);
    const result = await createSpecView({ client }).open(ITEM);
    if (expected === null) {
      assert.equal(result.spec, null, label);
      assert.equal(result.error.name, "SpecViewError", label);
      assert.match(result.error.message, /18d-the-spec-and-its-report/, label);
      assert.deepEqual(client.calls, [SPEC_PATH], label);
    } else {
      assert.deepEqual(result, expected, label);
    }
  }
});

test("the spec view has no write surface", async () => {
  assert.deepEqual(Object.keys(specViewModule), ["createSpecView"]);
  const source = await readFile(
    new URL("../src/spec_view.js", import.meta.url),
    "utf8",
  );
  assert.ok(!/POST\s+\/file/i.test(source));
  for (const name of ["write", "save", "edit"]) {
    assert.ok(!Object.hasOwn(specViewModule, name));
  }
});

test("ask and review intents are request-free prompt data", () => {
  const client = {
    file() {
      throw new Error("intent performed a request");
    },
  };
  const view = createSpecView({ client });
  const ask = view.askIntent(ITEM, "What changed?");
  const review = view.reviewIntent(ITEM);
  assert.match(ask, /What changed\?/);
  assert.match(ask, /specs\/18\/18d-the-spec-and-its-report\.md/);
  assert.match(review, /specs\/18\/18d-the-spec-and-its-report\.md/);
  assert.match(
    review,
    /specs\/report\/18\/18d-the-spec-and-its-report-report\.md/,
  );
});
