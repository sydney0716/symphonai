import test from "node:test";
import assert from "node:assert/strict";

import { layoutPhase, layoutRoadmap } from "../src/roadmap_graph.js";

const spec = (id, after = undefined, fields = {}) => ({
  title: id,
  spec: `specs/39/${id}-task.md`,
  ...(after === undefined ? {} : { after }),
  ...fields,
});

test("lays out splits, follow-ups, and a merge in dependency rows", () => {
  const result = layoutPhase([
    spec("a"), spec("b", ["a"]), spec("c", ["a"]),
    spec("d", ["b"]), spec("e", ["c", "d"]),
  ], {
    followUps: { b: [{ id: "bF", title: "fix goal" }] },
  });
  assert.deepEqual(result.rows.map((row) => row.map((node) => node.id)), [["a"], ["b", "c"], ["d"], ["e"]]);
  assert.deepEqual(result.rows[1][0].followUps.map(({ id }) => id), ["bF"]);
  assert.deepEqual(result.edges, [
    { from: "a", to: "b" }, { from: "a", to: "c" },
    { from: "b", to: "d" }, { from: "c", to: "e" }, { from: "d", to: "e" },
  ]);
});

test("folds items for the same spec into one node without marking partial work done", () => {
  const first = { ...spec("18b"), title: "Turn state machine", done: true };
  const second = { ...spec("18b"), title: "Queue input while running", done: false };
  const result = layoutPhase([first, second]);
  const node = result.rows.flat().find(({ id }) => id === "18b");

  assert.equal(result.error, undefined);
  assert.equal(result.rows.flat().filter(({ id }) => id === "18b").length, 1);
  assert.equal(node.done, false);
  assert.deepEqual(node.items, ["Turn state machine", "Queue input while running"]);
});

test("folded spec items union their dependencies", () => {
  const result = layoutPhase([
    spec("18a"),
    { ...spec("18b", ["18a"]), title: "Turn state machine" },
    { ...spec("18b", ["18c"]), title: "Queue input while running" },
    spec("18c"),
  ]);

  assert.deepEqual(result.edges.filter(({ to }) => to === "18b"), [
    { from: "18a", to: "18b" },
    { from: "18c", to: "18b" },
  ]);
});

test("same task ids from different specs still report a duplicate", () => {
  const result = layoutPhase([
    { title: "First", spec: "specs/18/18b-first.md" },
    { title: "Second", spec: "specs/18/18b-second.md" },
  ]);

  assert.match(result.error, /duplicate id 18b/);
});

test("groups four parallel children and attaches their incoming edges to the group", () => {
  const result = layoutPhase([spec("a"), ...["d", "b", "c", "e"].map((id) => spec(id, ["a"]))]);
  assert.equal(result.rows[1].length, 1);
  assert.deepEqual(result.rows[1][0].group.map(({ id }) => id), ["b", "c", "d", "e"]);
  assert.deepEqual(result.edges, [{ from: "a", to: "group-1" }]);
});

test("tags known cross-phase links, and reports cycles and unknown links", () => {
  const cross = layoutPhase([spec("z", ["37eF"])], { titles: { "37eF": "follow-up" } });
  assert.deepEqual(cross.rows[0][0].tags, ["← 37eF"]);
  assert.deepEqual(cross.edges, []);
  assert.match(layoutPhase([spec("x", ["y"]), spec("y", ["x"])]).error, /cycle.*x|cycle.*y/);
  assert.match(layoutPhase([spec("x", ["missing"])]).error, /missing/);
});

test("tags a long edge blocked by a node in its source column", () => {
  const result = layoutPhase([spec("a"), spec("x", ["a"]), spec("d", ["a", "x"])]);
  assert.deepEqual(result.rows.map((row) => row.map((node) => node.id)), [["a"], ["x"], ["d"]]);
  assert.ok(result.rows[2][0].tags.includes("← a"));
  assert.ok(!result.edges.some((edge) => edge.from === "a" && edge.to === "d"));
});

test("collapses leading done rows and drops their outgoing edges", () => {
  const result = layoutPhase([
    spec("a", undefined, { done: true }),
    spec("b", ["a"], { done: true }),
    spec("c", ["b"]),
  ]);
  assert.equal(result.summary, "✓ a b");
  assert.equal(result.rows[0][0].id, "c");
  assert.equal(result.rows[0][0].row, 0);
  assert.deepEqual(result.edges, []);
});

test("keeps the full graph when every node is done", () => {
  const result = layoutPhase([
    spec("a", undefined, { done: true }),
    spec("b", ["a"], { done: true }),
    spec("c", ["b"], { done: true }),
  ]);
  assert.equal(result.summary, "");
  assert.deepEqual(result.rows.map((row) => row.map((node) => node.id)), [["a"], ["b"], ["c"]]);
});

test("lays out phase splits and merges with each phase task graph", () => {
  const phases = [
    { id: "36", name: "Hand it anything", status: "done", progress: { done: 1, total: 1 }, items: [spec("36a")] },
    { id: "37", name: "Side by side", status: "next", progress: { done: 0, total: 1 }, items: [spec("37a")] },
    { id: "38", name: "Code it knows", status: "next", after: ["36"], progress: { done: 0, total: 1 }, items: [spec("38a")] },
    { id: "39", name: "Specs run", status: "next", after: ["37", "38"], progress: { done: 0, total: 1 }, items: [spec("39a")] },
  ];
  const result = layoutRoadmap(phases);
  assert.deepEqual(result.rows.map((row) => row.map((node) => node.id)), [
    ["36"], ["37", "38"], ["39"],
  ]);
  assert.deepEqual(result.edges, [
    { from: "36", to: "37" },
    { from: "36", to: "38" },
    { from: "37", to: "39" },
    { from: "38", to: "39" },
  ]);
  assert.deepEqual(result.rows[1].map((node) => node.taskLayout.rows[0][0].id), ["37a", "38a"]);
});

test("defaults phases to an array-order chain and marks blocked phase edges", () => {
  const phases = ["01", "02", "03", "04"].map((id) => ({
    id, name: id, status: "next", progress: { done: 0, total: 0 }, items: [],
  }));
  const result = layoutRoadmap(phases);
  assert.deepEqual(result.edges, [
    { from: "01", to: "02" },
    { from: "02", to: "03" },
    { from: "03", to: "04" },
  ]);
  assert.match(layoutRoadmap([
    phases[0],
    { ...phases[1], after: ["99"] },
  ]).error, /phase 02.*99/);
});

test("phase task layouts recognize cross-phase task references", () => {
  const phases = [
    { id: "37", name: "Prior", status: "done", progress: { done: 1, total: 1 }, items: [spec("37e")] },
    { id: "39", name: "Current", status: "next", progress: { done: 0, total: 1 }, items: [spec("39a", ["37eF"])] },
  ];
  const result = layoutRoadmap(phases);
  const current = result.rows.flat().find(({ id }) => id === "39");

  assert.equal(current.taskLayout.error, undefined);
  assert.ok(current.taskLayout.rows[0][0].tags.includes("← 37eF"));
});
