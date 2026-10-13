import test from "node:test";
import assert from "node:assert/strict";

import { layoutPhase, layoutRoadmap } from "../src/roadmap_graph.js";

const spec = (id, after = undefined, fields = {}) => ({
  title: id,
  spec: `specs/39/${id}-task.md`,
  ...(after === undefined ? {} : { after }),
  ...fields,
});

test("lays out phase 43 as sorted dependency waves", () => {
  const result = layoutPhase([
    spec("a"), spec("b", ["a"]), spec("e", ["a"]), spec("g", ["a"]), spec("i", ["a"]),
    spec("c", ["b"]), spec("d", ["b"]), spec("h", ["e"]), spec("f", ["c", "d", "e"]), spec("j", ["i"]),
  ]);

  assert.deepEqual(result.rows.map((row) => row.map(({ id }) => id)), [
    ["a"], ["b", "e", "g", "i"], ["c", "d", "h", "j"], ["f"],
  ]);
  assert.equal("edges" in result, false);
  assert.equal(result.rows.flat().some(({ tags }) => tags.some((tag) => tag.startsWith("← "))), false);
  for (const node of result.rows.flat()) {
    assert.deepEqual(Object.keys(node).sort(), ["followUps", "id", "items", "state", "tags", "title"]);
  }
});

test("keeps five parallel tasks on separate lines", () => {
  const result = layoutPhase([spec("a"), ...["f", "b", "e", "c", "d"].map((id) => spec(id, ["a"]))]);

  assert.deepEqual(result.rows[1].map(({ id }) => id), ["b", "c", "d", "e", "f"]);
  assert.equal(result.rows[1].length, 5);
  assert.equal(result.rows.flat().some((node) => "group" in node), false);
});

test("keeps a task title longer than sixteen characters whole", () => {
  const title = "A title longer than sixteen characters";
  const result = layoutPhase([spec("a", undefined, { title })]);

  assert.equal(result.rows[0][0].title, title);
});

test("folds items for the same spec into one node without marking partial work done", () => {
  const first = { ...spec("18b"), title: "Turn state machine", done: true };
  const second = { ...spec("18b"), title: "Queue input while running", done: false };
  const result = layoutPhase([first, second]);
  const node = result.rows.flat().find(({ id }) => id === "18b");

  assert.equal(result.error, undefined);
  assert.equal(result.rows.flat().filter(({ id }) => id === "18b").length, 1);
  assert.equal(result.summary, "");
  assert.deepEqual(node.items, ["Turn state machine", "Queue input while running"]);
});

test("folded spec items union their dependencies", () => {
  const result = layoutPhase([
    spec("18a"),
    { ...spec("18b", ["18a"]), title: "Turn state machine" },
    { ...spec("18b", ["18c"]), title: "Queue input while running" },
    spec("18c"),
  ]);

  assert.deepEqual(result.rows.map((row) => row.map(({ id }) => id)), [["18a", "18c"], ["18b"]]);
});

test("same task ids from different specs still report a duplicate", () => {
  const result = layoutPhase([
    { title: "First", spec: "specs/18/18b-first.md" },
    { title: "Second", spec: "specs/18/18b-second.md" },
  ]);

  assert.match(result.error, /duplicate id 18b/);
});

test("tags cross-phase links and reports cycles and unknown links", () => {
  const cross = layoutPhase([spec("z", ["37eF"])], { titles: { "37eF": "follow-up" } });
  assert.deepEqual(cross.rows[0][0].tags, ["← 37eF"]);
  assert.match(layoutPhase([spec("x", ["y"]), spec("y", ["x"])]).error, /cycle.*x|cycle.*y/);
  assert.match(layoutPhase([spec("x", ["missing"])]).error, /missing/);
});

test("collapses leading done waves and keeps the remaining tasks", () => {
  const result = layoutPhase([
    spec("a", undefined, { done: true }),
    spec("b", ["a"], { done: true }),
    spec("c", ["b"]),
  ]);
  assert.equal(result.summary, "✓ a b");
  assert.deepEqual(result.rows.map((row) => row.map(({ id }) => id)), [["c"]]);
});

test("keeps every wave visible when all tasks are done", () => {
  const result = layoutPhase([
    spec("a", undefined, { done: true }),
    spec("b", ["a"], { done: true }),
    spec("c", ["b"], { done: true }),
  ]);
  assert.equal(result.summary, "");
  assert.deepEqual(result.rows.map((row) => row.map(({ id }) => id)), [["a"], ["b"], ["c"]]);
});

test("lays out phase splits and merges in the overview with simple edges", () => {
  const phases = [
    { id: "36", name: "Hand it anything", status: "done", progress: { done: 1, total: 1 }, items: [spec("36a")] },
    { id: "37", name: "Side by side", status: "next", progress: { done: 0, total: 1 }, items: [spec("37a")] },
    { id: "38", name: "Code it knows", status: "next", after: ["36"], progress: { done: 0, total: 1 }, items: [spec("38a")] },
    { id: "39", name: "Specs run", status: "next", after: ["37", "38"], progress: { done: 0, total: 1 }, items: [spec("39a")] },
  ];
  const result = layoutRoadmap(phases);

  assert.deepEqual(result.rows.map((row) => row.map(({ id }) => id)), [["36"], ["37", "38"], ["39"]]);
  assert.deepEqual(result.edges, [
    { from: "36", to: "37" }, { from: "36", to: "38" },
    { from: "37", to: "39" }, { from: "38", to: "39" },
  ]);
  const overview = layoutRoadmap(phases, { maxColumns: 1 });
  assert.deepEqual(overview.edges, [{ from: "36", to: "group-1" }, { from: "group-1", to: "39" }]);
});

test("stacks independent roadmap phases in a parallel group", () => {
  const phases = ["37", "38"].map((id) => ({
    id, name: `Phase ${id}`, status: "next", after: [], progress: { done: 0, total: 1 }, items: [spec(`${id}a`)],
  }));
  const result = layoutRoadmap(phases, { maxColumns: 1 });

  assert.equal(result.error, undefined);
  assert.equal(result.rows.length, 1);
  assert.equal(result.rows[0].length, 1);
  assert.equal(result.rows[0][0].title, "parallel");
  assert.deepEqual(result.rows[0][0].group.map(({ id, column }) => [id, column]), [["37", 0], ["38", 0]]);
  assert.equal(result.rows[0][0].group[0].taskLayout.error, undefined);
  assert.equal(result.rows[0][0].group[1].taskLayout.error, undefined);
});

test("defaults phase dependencies to a chain and reports unknown dependencies", () => {
  const phases = ["01", "02", "03", "04"].map((id) => ({
    id, name: id, status: "next", progress: { done: 0, total: 0 }, items: [],
  }));
  const result = layoutRoadmap(phases);
  assert.deepEqual(result.edges, [{ from: "01", to: "02" }, { from: "02", to: "03" }, { from: "03", to: "04" }]);
  assert.match(layoutRoadmap([phases[0], { ...phases[1], after: ["99"] }]).error, /phase 02.*99/);
});

test("phase task waves recognize cross-phase task references", () => {
  const phases = [
    { id: "37", name: "Prior", status: "done", progress: { done: 1, total: 1 }, items: [spec("37e")] },
    { id: "39", name: "Current", status: "next", progress: { done: 0, total: 1 }, items: [spec("39a", ["37eF"])] },
  ];
  const result = layoutRoadmap(phases);
  const current = result.rows.flat().find(({ id }) => id === "39");

  assert.equal(current.taskLayout.error, undefined);
  assert.deepEqual(current.taskLayout.rows[0][0].tags, ["← 37eF"]);
});

test("phase dependency fallbacks preserve known task titles", () => {
  const phases = [
    { id: "38", name: "Current", status: "next", progress: { done: 0, total: 2 }, items: [
      spec("39b", undefined, { title: "run a spec" }), spec("39c", ["39b"]),
    ] },
    { id: "39", name: "Next", status: "next", progress: { done: 0, total: 1 }, items: [spec("39d", ["39b"]) ] },
  ];
  const result = layoutRoadmap(phases);
  const current = result.rows.flat().find(({ id }) => id === "38");
  const next = result.rows.flat().find(({ id }) => id === "39");

  assert.equal(current.taskLayout.rows[0][0].title, "run a spec");
  assert.deepEqual(next.taskLayout.rows[0][0].tags, ["← 39b"]);
});
