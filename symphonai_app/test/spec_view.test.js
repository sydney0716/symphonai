import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import * as specViewModule from "../src/spec_view.js";
import { createSpecView } from "../src/spec_view.js";
import { ProtocolError } from "../src/protocol.js";

const SPEC_PATH = "specs/18/18d-the-spec-and-its-report.md";
const REPORT_PATH = "specs/report/18/18d-the-spec-and-its-report-report.md";
const ITEM = { title: "The spec and its report", spec: SPEC_PATH };

function fileFailure(status) {
  const error = new ProtocolError(`file failed with status ${status}`);
  error.status = status;
  return error;
}

function clientWith(replies, { forbidden = [] } = {}) {
  const calls = [];
  return {
    calls,
    async file(path) {
      calls.push(path);
      if (forbidden.includes(path)) {
        throw new Error(`forbidden eager fetch: ${path}`);
      }
      const reply = replies.get(path);
      if (reply instanceof Error) {
        throw reply;
      }
      if (reply === undefined) {
        throw fileFailure(404);
      }
      return reply;
    },
  };
}

test("open returns a spec and its primary report", async () => {
  const client = clientWith(
    new Map([
      [SPEC_PATH, { path: SPEC_PATH, text: "spec text" }],
      [REPORT_PATH, { path: REPORT_PATH, text: "report text" }],
    ]),
  );
  assert.deepEqual(await createSpecView({ client }).open(ITEM), {
    specs: [{ path: SPEC_PATH, text: "spec text" }],
    report: { path: REPORT_PATH, text: "report text" },
    followUps: [],
    error: null,
  });
});

test("the real hooks item opens every bound spec in order", async () => {
  const roadmap = JSON.parse(
    await readFile(new URL("../../docs/roadmap.json", import.meta.url), "utf8"),
  );
  const phase = roadmap.phases.find(({ id }) => id === "10");
  const item = phase.items.find(
    ({ spec }) => Array.isArray(spec) && spec.includes("specs/10/10c-hooks.md"),
  );
  assert.deepEqual(item.spec, [
    "specs/10/10b-the-missing-events.md",
    "specs/10/10c-hooks.md",
  ]);
  const replies = new Map(
    item.spec.map((path) => [path, { path, text: `text for ${path}` }]),
  );
  const client = clientWith(replies);

  const result = await createSpecView({ client }).open(item);

  assert.deepEqual(
    result.specs.map(({ path }) => path),
    item.spec,
  );
  assert.deepEqual(client.calls.slice(0, 2), item.spec);
});

test("a missing bound spec does not hide the other bound specs", async () => {
  const first = "specs/18/18a-first.md";
  const second = "specs/18/18b-second.md";
  const item = { title: "two specs", spec: [first, second] };
  const cases = [
    ["second missing", first, second],
    ["first missing", second, first],
  ];
  assert.equal(cases.length, 2);

  for (const [label, available, missing] of cases) {
    const client = clientWith(
      new Map([[available, { path: available, text: `${available} text` }]]),
    );
    const result = await createSpecView({ client }).open(item);
    assert.deepEqual(result.specs.map(({ path }) => path), [available], label);
    assert.deepEqual(client.calls.slice(0, 2), [first, second], label);
    assert.match(result.error, new RegExp(missing), label);
  }
});

test("only the first bound spec determines the report", async () => {
  const first = "specs/18/18a-first.md";
  const second = "specs/18/18b-second.md";
  const firstReport = "specs/report/18/18a-first-report.md";
  const secondReport = "specs/report/18/18b-second-report.md";
  const replies = new Map([
    [first, { path: first, text: "first" }],
    [second, { path: second, text: "second" }],
    [firstReport, { path: firstReport, text: "first report" }],
    [secondReport, { path: secondReport, text: "second report" }],
  ]);
  const client = clientWith(replies);

  const result = await createSpecView({ client }).open({
    title: "two specs",
    spec: [first, second],
  });

  assert.deepEqual(result.report, {
    path: firstReport,
    text: "first report",
  });
  assert.deepEqual(
    client.calls.filter((path) => path.includes("specs/report/")),
    [firstReport],
  );
});

test("a missing primary report is ordinary", async () => {
  const client = clientWith(
    new Map([[SPEC_PATH, { path: SPEC_PATH, text: "spec text" }]]),
  );
  const result = await createSpecView({ client }).open(ITEM);
  assert.equal(result.report, null);
  assert.equal(result.error, null);
});

test("follow-ups are listed in order without fetching their text", async () => {
  const primary = "specs/10/10c-hooks.md";
  const followUps = [
    "specs/10/10cF-hooks-that-can-actually-load.md",
    "specs/10/10cF2-a-filename-is-not-a-trust-boundary.md",
  ];
  const client = clientWith(
    new Map([[primary, { path: primary, text: "hooks" }]]),
    { forbidden: followUps },
  );

  const result = await createSpecView({ client }).open(
    { title: "hooks", spec: primary },
    { specPaths: [followUps[1], primary, followUps[0]] },
  );

  assert.deepEqual(
    result.followUps,
    followUps.map((path) => ({ path, text: null })),
  );
  assert.ok(followUps.every((path) => !client.calls.includes(path)));
});

test("open defaults to no follow-up inventory", async () => {
  const client = clientWith(
    new Map([[SPEC_PATH, { path: SPEC_PATH, text: "spec text" }]]),
  );
  const result = await createSpecView({ client }).open(ITEM);
  assert.deepEqual(result.followUps, []);
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
